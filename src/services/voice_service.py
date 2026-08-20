"""One turn of a spoken conversation, from microphone to loudspeaker.

The shape of the thing is a pipeline where every stage hands the next one work
before it has finished its own:

    audio ─► Saaras ─► transcript ─► chat model ─┬─► delta events (text)
                                                 └─► segmenter ─► Bulbul ─► PCM

Nothing here waits for a stage to complete when it could be working with what
that stage has already produced. The reply is read token by token so the
segmenter can cut the first clause after ~30 of them; the first clause goes to
the synthesiser while the model is still writing the second; and the audio for
segment N streams to the browser while segment N+1 is already being
synthesised. That look-ahead is what keeps the seam between two segments from
being audible — a fresh synthesis request costs ~300 ms of time-to-first-byte,
and 300 ms of silence in the middle of a sentence is a stutter.

Measured end to end (Sarvam, Tamil, this machine): first token ~435 ms, first
speakable segment ~670 ms, first audio byte ~1.1 s after the transcript.

State per connection is the conversation history and, at most, one running
turn. Barge-in cancels that turn: the reply stops mid-word, and what was
already spoken is what goes into the history, because the model must not
believe it said sentences the listener never heard.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

from src.core.config import Settings, get_settings
from src.schemas import voice as events
from src.voice import llm, stt, tts
from src.voice.http import ProviderError
from src.voice.languages import LANGUAGES, display, normalise
from src.voice.segment import Segmenter

if TYPE_CHECKING:  # the RAG stack is imported lazily — see `_retrieve`
    from src.schemas.ask import Timings

log = logging.getLogger("vec.voice")

# How many PCM chunks may sit buffered for one segment before the provider
# stream is made to wait. 64 × ~16 KB is about ten seconds of audio — enough
# that a synthesiser running ahead is never throttled by a slow socket, small
# enough that a stalled client cannot make the server hold a whole reply.
_CHUNK_BUFFER = 64

Emit = Callable[[events.Wire], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]


def _stage_detail(timings: "Timings") -> str | None:
    """The two slowest retrieval stages, named — "search 92 ms · embed 11 ms".

    A single "retrieval took 140 ms" says nothing actionable; which stage ate
    the budget is the whole point of having per-stage timings at all.
    """
    spent = [
        (name, ms)
        for name, ms in timings.model_dump().items()
        if name != "total" and isinstance(ms, (int, float)) and ms >= 1
    ]
    if not spent:
        return None

    spent.sort(key=lambda pair: pair[1], reverse=True)
    return " · ".join(f"{name} {ms:.0f} ms" for name, ms in spent[:2])


@dataclass(slots=True)
class _Synth:
    """A segment already being synthesised, with somewhere to put the bytes."""

    index: int
    text: str
    chunks: "asyncio.Queue[bytes | BaseException | None]"
    task: asyncio.Task


@dataclass(slots=True)
class _Turn:
    """Stopwatch and tally for one exchange."""

    id: str
    started: float
    stt: float | None = None
    first_token: float | None = None
    first_segment: float | None = None
    first_audio: float | None = None
    reply: float | None = None
    spoken: str = ""
    segments: int = 0
    language_code: str | None = None

    def mark(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    def timings(self) -> events.VoiceTimings:
        return events.VoiceTimings(
            stt=self.stt,
            first_token=self.first_token,
            first_segment=self.first_segment,
            first_audio=self.first_audio,
            reply=self.reply,
            total=self.mark(),
        )


@dataclass
class VoiceSession:
    """One conversation, one WebSocket.

    `emit` sends a JSON event; `send_audio` sends a binary frame. Splitting
    them is what lets audio stay raw on the wire instead of paying a third of
    its size in base64.
    """

    emit: Emit
    send_audio: SendAudio
    settings: Settings = field(default_factory=get_settings)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[llm.Message] = field(default_factory=list)
    _turn: asyncio.Task | None = field(default=None, init=False, repr=False)

    # ---- lifecycle ------------------------------------------------------

    def describe(self) -> events.Ready:
        """What this session can actually do, said up front."""
        target = self.settings.resolve_llm()
        speaks = self.settings.sarvam_ready or self.settings.openai_ready

        return events.Ready(
            session_id=self.session_id,
            providers=events.Providers(
                stt="sarvam" if self.settings.sarvam_ready else ("openai" if self.settings.openai_ready else None),
                llm=target.provider if target.ready else None,
                llm_model=target.model if target.ready else None,
                tts=("sarvam" if self.settings.sarvam_ready else "openai") if speaks else None,
                rag_enabled=self.settings.rag_enabled,
            ),
            sample_rate=self.settings.tts_sample_rate,
            languages=dict(LANGUAGES),
        )

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    def start(self, coro) -> None:
        """Run a turn, replacing whatever was running.

        Speaking while the assistant speaks is barge-in, not an error, so the
        old turn is cancelled rather than the new one refused.
        """
        self.cancel()
        self._turn = asyncio.create_task(coro)

    def cancel(self) -> None:
        if self.busy and self._turn is not None:
            self._turn.cancel()

    def reset(self) -> None:
        self.cancel()
        self.history.clear()

    async def aclose(self) -> None:
        self.cancel()
        if self._turn is not None:
            try:
                await self._turn
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — closing
                pass

    # ---- the running commentary -----------------------------------------

    async def _say(
        self,
        turn: _Turn | None,
        step: events.ActivityStep,
        state: events.ActivityState,
        label: str,
        *,
        detail: str | None = None,
        ms: float | None = None,
    ) -> None:
        """Announce a step of the pipeline as it happens.

        Every seam in a turn goes through here, so the log the client draws is
        the pipeline's actual shape rather than a guess made from the four
        coarse `status` stages. Cheap enough to put anywhere: one small JSON
        frame on a socket that is already carrying audio.
        """
        await self.emit(
            events.Activity(
                turn_id=turn.id if turn else None,
                step=step,
                state=state,
                label=label,
                detail=detail,
                ms=ms,
            )
        )

    # ---- turns ----------------------------------------------------------

    async def from_audio(self, audio: bytes, *, mime: str, language: str | None = None) -> None:
        """The ordinary path: someone spoke."""
        turn = _Turn(id=str(uuid.uuid4()), started=time.perf_counter())

        try:
            await self.emit(events.Status(stage="transcribing", turn_id=turn.id))
            await self._say(
                turn,
                "stt",
                "running",
                "Transcribing what you said",
                detail="sarvam saaras" if self.settings.sarvam_ready else "openai",
            )
            heard = await stt.transcribe(
                audio, mime=mime, settings=self.settings, language=language
            )
            turn.stt = turn.mark()
        except ProviderError as error:
            await self._fail(turn, error, stage="stt")
            return
        except asyncio.CancelledError:
            await self._canceled(turn)
            raise

        turn.language_code = heard.language_code or normalise(language)
        await self.emit(
            events.TranscriptEvent(
                turn_id=turn.id,
                text=heard.text,
                language_code=turn.language_code,
                language=display(turn.language_code),
                confidence=heard.confidence,
                provider=heard.provider,
                ms=turn.stt,
            )
        )
        await self._say(
            turn,
            "stt",
            "done",
            f"Heard you in {display(turn.language_code) or 'your language'}",
            detail=heard.provider,
            ms=turn.stt,
        )

        await self._answer(turn, heard.text)

    async def from_text(self, text: str, *, language: str | None = None) -> None:
        """The same turn, typed instead of spoken. Useful without a microphone."""
        turn = _Turn(id=str(uuid.uuid4()), started=time.perf_counter())
        turn.language_code = normalise(language)

        await self._say(turn, "stt", "skipped", "Typed — nothing to transcribe")
        await self.emit(
            events.TranscriptEvent(
                turn_id=turn.id,
                text=text,
                language_code=turn.language_code,
                language=display(turn.language_code),
                provider="typed",
                ms=0.0,
            )
        )
        await self._answer(turn, text)

    # ---- the answer half ------------------------------------------------

    async def _answer(self, turn: _Turn, transcript: str) -> None:
        question = transcript.strip()
        if len(question) < 2:
            await self.emit(
                events.VoiceError(
                    turn_id=turn.id,
                    stage="stt",
                    message="I didn't catch that — say it once more?",
                )
            )
            await self.emit(events.Status(stage="idle", turn_id=turn.id))
            return

        try:
            await self.emit(events.Status(stage="thinking", turn_id=turn.id))
            context = await self._retrieve(question, turn)

            messages = llm.build_messages(
                transcript=question,
                history=self.history,
                language_code=turn.language_code,
                context=context,
                max_turns=self.settings.llm_history_turns,
            )

            target = self.settings.resolve_llm()
            await self._say(
                turn,
                "llm",
                "start",
                "Sending the question to the model",
                detail=f"{target.provider} · {target.model}",
                ms=turn.mark(),
            )

            voice = tts.choose(turn.language_code, self.settings)
            await self._run(turn, messages, voice)

        except ProviderError as error:
            self._remember(turn, question)
            await self._fail(turn, error, stage="reply")
            return
        except asyncio.CancelledError:
            self._remember(turn, question)
            await self._canceled(turn)
            raise
        except Exception as error:  # a provider changed shape, a socket died…
            log.exception("turn %s failed", turn.id)
            self._remember(turn, question)
            await self._fail(turn, error, stage="reply")
            return

        self._remember(turn, question)
        await self._say(
            turn,
            "speech",
            "done",
            f"Spoke {turn.segments} {'part' if turn.segments == 1 else 'parts'}",
            ms=turn.mark(),
        )
        await self.emit(
            events.TurnEnd(
                turn_id=turn.id,
                reply=turn.spoken,
                language_code=turn.language_code,
                segments=turn.segments,
                timings=turn.timings(),
            )
        )
        await self._say(turn, "turn", "done", "Turn complete", ms=turn.mark())
        await self.emit(events.Status(stage="idle", turn_id=turn.id))

    async def _run(self, turn: _Turn, messages: list[llm.Message], voice: tts.Voice) -> None:
        """Generate, segment, synthesise and send — all at once.

        Two tasks with a queue between them. `produce` reads the model and
        starts a synthesis per segment; `consume` forwards finished audio in
        order. The queue's `maxsize` is the look-ahead: it blocks `produce`
        once that many segments are already waiting, so a long reply cannot
        open twenty synthesis requests at once.
        """
        # One job is always in the consumer's hands, so the queue holds
        # `lookahead - 1`. asyncio reads maxsize=0 as unbounded, which is why
        # the floor is 1 — two segments in flight is the least this can mean.
        jobs: asyncio.Queue[_Synth | None] = asyncio.Queue(
            maxsize=max(1, self.settings.speech_lookahead - 1)
        )
        started: list[_Synth] = []

        async def produce() -> None:
            index = 0
            try:
                async for segment in self._segments(turn, messages):
                    job = self._synthesise(segment, index, voice)
                    started.append(job)
                    index += 1
                    await jobs.put(job)
            finally:
                await jobs.put(None)

        async def consume() -> None:
            while True:
                job = await jobs.get()
                if job is None:
                    return
                await self._forward(turn, job, voice)

        producer = asyncio.create_task(produce())
        consumer = asyncio.create_task(consume())

        try:
            await asyncio.gather(producer, consumer)
        finally:
            # Cancellation and failure both land here. Anything still
            # synthesising is audio nobody will hear — stop paying for it.
            for task in (producer, consumer):
                task.cancel()
            for job in started:
                job.task.cancel()

    async def _segments(self, turn: _Turn, messages: list[llm.Message]) -> AsyncIterator[str]:
        """Reply tokens in, speakable segments out — emitting deltas on the way."""
        segmenter = Segmenter(
            first_chars=self.settings.speech_first_segment_chars,
            chars=self.settings.speech_segment_chars,
            max_chars=self.settings.speech_max_segment_chars,
        )

        async for piece in llm.stream_reply(messages, settings=self.settings):
            # Models like to open with a newline. It is invisible in a chat
            # window and not in a reply that gets rendered as it streams, so
            # the leading whitespace is dropped before anyone sees it.
            if not turn.spoken:
                piece = piece.lstrip()
                if not piece:
                    continue

            if turn.first_token is None:
                turn.first_token = turn.mark()
                await self._say(
                    turn,
                    "llm",
                    "running",
                    "Model is writing the reply",
                    ms=turn.first_token,
                )

            turn.spoken += piece
            await self.emit(events.Delta(turn_id=turn.id, text=piece))

            for segment in segmenter.feed(piece):
                if turn.first_segment is None:
                    turn.first_segment = turn.mark()
                yield segment

        turn.reply = turn.mark()
        words = len(turn.spoken.split())
        await self._say(
            turn,
            "llm",
            "done",
            "Reply written",
            detail=f"{words} {'word' if words == 1 else 'words'}",
            ms=turn.reply,
        )

        tail = segmenter.flush()
        if tail:
            if turn.first_segment is None:
                turn.first_segment = turn.mark()
            yield tail

    def _synthesise(self, text: str, index: int, voice: tts.Voice) -> _Synth:
        """Start turning one segment into sound. Returns before it is done."""
        chunks: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue(maxsize=_CHUNK_BUFFER)

        async def run() -> None:
            try:
                async for chunk in tts.stream_speech(text, voice, settings=self.settings):
                    await chunks.put(chunk)
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001 — handed to the consumer
                await chunks.put(error)
            finally:
                # The consumer is waiting on this queue; without a sentinel a
                # failed synthesis would hang the whole reply.
                await chunks.put(None)

        return _Synth(index=index, text=text, chunks=chunks, task=asyncio.create_task(run()))

    async def _forward(self, turn: _Turn, job: _Synth, voice: tts.Voice) -> None:
        """Send one segment's audio, framed so the client knows what it is."""
        if turn.segments == 0:
            await self.emit(events.Status(stage="speaking", turn_id=turn.id))

        await self._say(
            turn,
            "speech",
            "running",
            f"Speaking part {job.index + 1}",
            detail=f"{voice.provider} · {voice.voice}" if voice.voice else voice.provider,
            ms=turn.mark(),
        )

        await self.emit(
            events.SpeechStart(
                turn_id=turn.id,
                segment=job.index,
                text=job.text,
                provider=voice.provider,
                voice=voice.voice,
                language_code=voice.language_code,
                sample_rate=voice.sample_rate,
            )
        )

        started = time.perf_counter()
        total = 0

        while True:
            chunk = await job.chunks.get()
            if chunk is None:
                break
            if isinstance(chunk, BaseException):
                raise chunk

            if turn.first_audio is None:
                turn.first_audio = turn.mark()
            total += len(chunk)
            await self.send_audio(chunk)

        turn.segments += 1
        await self.emit(
            events.SpeechEnd(
                turn_id=turn.id,
                segment=job.index,
                bytes=total,
                ms=round((time.perf_counter() - started) * 1000, 1),
            )
        )

    # ---- retrieval (off) ------------------------------------------------

    async def _retrieve(self, question: str, turn: _Turn) -> str | None:
        """The RAG seam. Returns None while `RAG_ENABLED` is false.

        This is the whole switch: with retrieval on, the corpus passages become
        the context the reply is grounded in and the guardrails in
        `src/rag/guardrails.py` decide whether there is an answer at all. The
        pipeline underneath it is built and measured (docs/09-v1.md) — it is
        turned off here, not removed.
        """
        if not self.settings.rag_enabled:
            await self._say(
                turn,
                "retrieval",
                "skipped",
                "Retrieval is off — answering from the model",
                ms=turn.mark(),
            )
            return None

        import anyio

        from src.schemas.ask import AskRequest
        from src.services.ask_service import get_ask_service

        await self._say(
            turn, "retrieval", "running", "Searching the corpus", ms=turn.mark()
        )

        service = get_ask_service()
        answer = await anyio.to_thread.run_sync(
            service.ask,
            AskRequest(
                transcript=question,
                language_code=turn.language_code,
                request_id=turn.id,
            ),
        )

        # The stage timings the harness produced, said out loud. This is the
        # inside of the 200 ms budget — the one place a listener can see which
        # stage spent it.
        await self._say(
            turn,
            "retrieval",
            "done" if answer.status == "answered" else "skipped",
            {
                "answered": f"Found {len(answer.citations)} passages",
                "abstained": "No source covers this",
                "refused": "Question turned down at the gate",
            }[answer.status],
            detail=_stage_detail(answer.timings),
            ms=turn.mark(),
        )

        if answer.status != "answered" or not answer.citations:
            # An abstention is a real answer: say so rather than inventing one.
            return f"No source covers this. Tell the user: {answer.reason}"

        return "\n".join(f"- {citation.text}" for citation in answer.citations[:3])

    # ---- bookkeeping ----------------------------------------------------

    def _remember(self, turn: _Turn, question: str) -> None:
        """Record the exchange — including a half-spoken one.

        After a barge-in the model has to believe it said exactly what was
        heard, no more. Storing the full generated reply would have it
        referring back to sentences that were cut off mid-word.
        """
        self.history.append({"role": "user", "content": question})
        if turn.spoken.strip():
            self.history.append({"role": "assistant", "content": turn.spoken.strip()})

        keep = max(2, self.settings.llm_history_turns * 2)
        if len(self.history) > keep:
            del self.history[:-keep]

    async def _canceled(self, turn: _Turn) -> None:
        await self._say(
            turn, "turn", "skipped", "Stopped — you started talking", ms=turn.mark()
        )
        await self.emit(events.Canceled(turn_id=turn.id, spoken=turn.spoken or None))

    async def _fail(self, turn: _Turn, error: Exception, *, stage: str) -> None:
        provider = getattr(error, "provider", None) or None
        message = str(error) or "Something went wrong on my side — try that again."

        log.warning("turn %s failed at %s: %s", turn.id, stage, message)
        await self._say(
            turn,
            "stt" if stage == "stt" else "llm",
            "error",
            message,
            detail=provider,
            ms=turn.mark(),
        )
        await self.emit(
            events.VoiceError(
                turn_id=turn.id, stage=stage, message=message, provider=provider
            )
        )
        await self.emit(events.Status(stage="idle", turn_id=turn.id))


def get_voice_session(emit: Emit, send_audio: SendAudio) -> VoiceSession:
    return VoiceSession(emit=emit, send_audio=send_audio, settings=get_settings())
