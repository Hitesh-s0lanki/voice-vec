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

That history is also written down. The first sentence anyone says opens a row
in `conversations` and the client puts `/c/{id}` in the address bar; every
question and every reply — including the half-spoken one a barge-in leaves —
is appended to it. Reloading that URL hands the model its own past back
(`bind`), so a refresh continues the conversation instead of starting one.

Storage is never in the turn's way. Writes go onto a queue that a separate
task drains through a worker thread, so a slow Neon round trip cannot delay a
spoken word, and a dead one costs the listener nothing but the URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

import anyio

from src.chat.tools import ToolCallStore, get_tool_call_store
from src.integrations.agent import ToolAgent, get_agent
from src.chat.store import (
    ChatStore,
    Owner,
    get_chat_store,
    is_conversation_id,
    new_conversation_id,
    title_from,
)
from src.core.config import Settings, get_settings
from src.memory.store import MemoryStore, as_prompt, get_memory
from src.schemas import voice as events
from src.voice import llm, stt, tts
from src.voice.http import ProviderError
from src.voice.languages import LANGUAGES, display, normalise
from src.voice.segment import Segmenter

if TYPE_CHECKING:  # the RAG stack is imported lazily — see `_retrieve`
    from src.schemas.ask import AskResponse, Timings

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


def _retrieval_detail(answer: "AskResponse") -> str | None:
    """What the rung actually did, then where the time went.

    A cache hit and a four-call adaptive run both come back as "answered", and
    the difference between them is the entire reason the effort slider exists.
    Leading with it means the activity feed reports the *shape* of the work and
    not only its duration.
    """
    parts: list[str] = []
    if answer.cached:
        parts.append("from cache")
    elif answer.escalations:
        parts.append(" + ".join(answer.escalations[:3]))

    stages = _stage_detail(answer.timings)
    if stages:
        parts.append(stages)
    return " · ".join(parts) or None


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
    #: How many tools the agent ran this turn. Reported on `turn.end` so a
    #: reply that took four seconds has a visible reason.
    tools: int = 0
    language_code: str | None = None
    #: The effort rung the client asked for this turn, or None to fall back to
    #: `Settings.effort_default`. Per-turn, not per-session: the slider can
    #: move between two sentences of the same conversation.
    effort: int | None = None

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
    #: Whose conversation this is — an account when there is one, the browser's
    #: own `sess_…` when there is not. Empty means nothing gets written down.
    owner: Owner = field(default_factory=Owner)
    #: The row every turn is appended to. Null until the first real sentence.
    conversation_id: str | None = None
    chat: ChatStore = field(default_factory=get_chat_store)
    #: What the agent may run, and where the record of it goes. Both are
    #: no-ops for a session whose owner has linked nothing.
    agent: ToolAgent = field(default_factory=get_agent)
    tool_calls: ToolCallStore = field(default_factory=get_tool_call_store)
    #: What the agent remembers from *other* conversations. A no-op for a
    #: deployment with no Agent Memory service — see `src/memory/store.py`.
    memory: MemoryStore = field(default_factory=get_memory)
    _turn: asyncio.Task | None = field(default=None, init=False, repr=False)
    _writes: "asyncio.Queue[Callable[[], object]] | None" = field(
        default=None, init=False, repr=False
    )
    _writer: asyncio.Task | None = field(default=None, init=False, repr=False)

    # ---- lifecycle ------------------------------------------------------

    def describe(self) -> events.Ready:
        """What this session can actually do, said up front."""
        # Imported here rather than at module scope for the same reason the
        # rest of the RAG stack is: a checkout with retrieval off should not
        # pay for it at import time.
        from src.rag.cache import get_cache

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
                effort_max=self.settings.effort_max,
                cache=get_cache().describe(),
                memory=self.memory.describe(),
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
        """Start over. The old conversation stays in Postgres; this one is new.

        Unbinding rather than deleting is what makes `reset` cheap and
        recoverable: the next take opens a fresh row and the client is told the
        new id, while everything already said is still at its own `/c/…`.
        """
        self.cancel()
        self.history.clear()
        self.conversation_id = None

    async def aclose(self) -> None:
        self.cancel()
        if self._turn is not None:
            try:
                await self._turn
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — closing
                pass

        # The last reply is usually still on the queue here: the socket closes
        # the moment the tab does, and the turn that ended it finished
        # milliseconds ago. Give the writer a bounded chance to land it — but
        # bounded, because a hung database must not hold the connection open.
        if self._writes is not None and self._writer is not None:
            try:
                await asyncio.wait_for(self._writes.join(), timeout=5)
            except (TimeoutError, asyncio.TimeoutError):
                log.warning("dropped %d unwritten message(s)", self._writes.qsize())

        if self._writer is not None:
            self._writer.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer
            self._writer = None

    # ---- the conversation it is written into ----------------------------

    @property
    def whose(self) -> str | None:
        """The one id both stores key on — the account, or the browser's own.

        Postgres matches a conversation on *either* column; agent memory has a
        single `actorId`/`ownerId`, so it has to be told which one this is. The
        account wins when there is one, which means signing in carries a
        visitor's memories forward under their `sess_…` id rather than
        stranding them — the same widening the conversations predicate does.
        """
        return self.owner.user_id or self.owner.session_id or None

    @property
    def persists(self) -> bool:
        """Whether anything said here can be written down at all.

        Two ways it can be false, and both are ordinary rather than broken: a
        checkout with no DATABASE_URL, and a client that sent no identity. The
        voice loop is unchanged in either case — it just leaves no trace.
        """
        return self.chat.configured and self.owner.known

    async def bind(self, conversation_id: str | None) -> None:
        """Pick an existing conversation back up, history and all.

        This is what a reload of `/c/{id}` costs: one indexed read for the row
        (which also proves it belongs to whoever is asking) and one for the
        tail of its messages, both before the socket says `ready`. Without it a
        refresh would leave the model with no idea what it had just been
        talking about.

        Anything unexpected — a stranger's id, a database that is down — leaves
        the session unbound rather than failing the connection. The next take
        then opens a new conversation, which is a far better outcome for
        someone holding a microphone than a socket that refuses to open.
        """
        if not conversation_id or not self.persists or not is_conversation_id(conversation_id):
            return

        try:
            found = await anyio.to_thread.run_sync(
                partial(self.chat.get, conversation_id, self.owner)
            )
            if found is None:
                log.info("conversation %s is not available to this caller", conversation_id)
                return

            self.history = await anyio.to_thread.run_sync(
                partial(
                    self.chat.history,
                    found.id,
                    max_messages=self.settings.llm_history_turns * 2,
                )
            )
        except Exception as error:  # unreachable database, bad DSN, …
            log.warning("could not open conversation %s: %s", conversation_id, error)
            return

        self.conversation_id = found.id
        await self.emit(
            events.ConversationEvent(
                id=found.id, title=found.title, turns=found.turns, created=False
            )
        )

    async def _open(self, turn: _Turn, question: str) -> None:
        """Open a conversation on the first thing worth keeping.

        The id is minted here rather than read back from the insert, so the
        client has its URL a round trip earlier — the address bar changes while
        the model is still writing. The insert is queued *before* any message,
        and the queue is serial, so the foreign key is satisfied by
        construction even though nothing waited for it.
        """
        if self.conversation_id or not self.persists:
            return

        conversation_id = new_conversation_id()
        title = title_from(question)
        self.conversation_id = conversation_id

        self._enqueue(
            partial(
                self.chat.create,
                self.owner,
                conversation_id=conversation_id,
                title=title,
                language=turn.language_code,
            )
        )
        await self.emit(
            events.ConversationEvent(id=conversation_id, title=title, created=True)
        )

    def _enqueue(self, job: Callable[[], object]) -> None:
        """Hand a blocking write to the writer task. Never awaited by a turn."""
        if not self.persists:
            return

        if self._writes is None:
            self._writes = asyncio.Queue()
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._drain())

        self._writes.put_nowait(job)

    async def _drain(self) -> None:
        """Run queued writes one at a time, in the order they were made.

        Serial on purpose. The rows have an order — the conversation before its
        first message, the question before its answer — and a pool that runs
        them concurrently would be free to invert it.
        """
        assert self._writes is not None

        while True:
            job = await self._writes.get()
            try:
                await anyio.to_thread.run_sync(job)
            except Exception as error:  # noqa: BLE001 — storage must not surface
                log.warning("conversation write failed: %s", error)
            finally:
                self._writes.task_done()

    def _save(
        self,
        turn: _Turn,
        role: str,
        text: str,
        *,
        status: str | None = None,
        reason: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Append one message to the open conversation, eventually."""
        if not self.conversation_id or not text.strip():
            return

        self._enqueue(
            partial(
                self.chat.append,
                self.conversation_id,
                role=role,
                text=text.strip(),
                turn_id=turn.id,
                language_code=turn.language_code,
                status=status,
                reason=reason,
                latency_ms=latency_ms,
            )
        )
        self._mirror(role, text, status=status)

    #: Assistant turns worth extracting facts from. A barge-in is included —
    #: what was heard before the interruption is as true as a whole reply — and
    #: a provider failure is not, because the text on an errored turn is either
    #: empty or half a sentence and its only effect downstream would be to
    #: spend a 30 MB instance on noise.
    _EXTRACTABLE = frozenset({None, "answered", "interrupted"})

    def _mirror(self, role: str, text: str, *, status: str | None = None) -> None:
        """Put the same turn into session memory, for the service to distil.

        Rides the writer queue the Postgres append just went on, so the two
        stores are written in the same order by the same serial task and a slow
        memory service delays a spoken word exactly as much as a slow database
        does, which is not at all.

        The gate is `conversation_id`, already checked by the caller: if a turn
        was not worth a row in Postgres it is not worth an event here either,
        and that single rule is what keeps coughs, anonymous callers and
        five-second takes out of an instance this small.
        """
        whose = self.whose
        if not whose or not self.memory.configured or status not in self._EXTRACTABLE:
            return

        self._enqueue(
            partial(
                self.memory.remember,
                session_id=self.conversation_id,
                actor_id=whose,
                role=role,
                text=text,
            )
        )

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

    async def from_audio(
        self,
        audio: bytes,
        *,
        mime: str,
        language: str | None = None,
        effort: int | None = None,
    ) -> None:
        """The ordinary path: someone spoke."""
        turn = _Turn(id=str(uuid.uuid4()), started=time.perf_counter(), effort=effort)

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

    async def from_text(
        self, text: str, *, language: str | None = None, effort: int | None = None
    ) -> None:
        """The same turn, typed instead of spoken. Useful without a microphone."""
        turn = _Turn(id=str(uuid.uuid4()), started=time.perf_counter(), effort=effort)
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

        # After the length guard, not before: a take that caught a cough should
        # not leave a conversation behind for someone to find in the rail.
        await self._open(turn, question)
        self._save(turn, "user", question)

        try:
            await self.emit(events.Status(stage="thinking", turn_id=turn.id))
            # Concurrently, because they are independent and both are network
            # round trips: recall asks a service a region away what it knows
            # about this listener while retrieval searches the corpus for this
            # question. Run in sequence they would add; run together the slower
            # one is the whole cost, and recall is never the slower one.
            #
            # `_retrieve` raises into the handler below exactly as it did
            # before. `_recall` adds no new way for a turn to fail: its lookup
            # swallows everything, and the only thing left in it that can throw
            # is the same `_say` every other stage already calls.
            context, memories = await asyncio.gather(
                self._retrieve(question, turn),
                self._recall(question, turn),
            )

            messages = llm.build_messages(
                transcript=question,
                history=self.history,
                language_code=turn.language_code,
                context=context,
                memories=memories,
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

            # The agent's turn to act, before it has said anything. Runs
            # only for somebody who has linked a toolkit — see `_use_tools`.
            messages = await self._use_tools(turn, messages)

            voice = tts.choose(turn.language_code, self.settings)
            await self._run(turn, messages, voice)

        except ProviderError as error:
            self._remember(turn, question, status="error", reason=str(error) or None)
            await self._fail(turn, error, stage="reply")
            return
        except asyncio.CancelledError:
            self._remember(turn, question, status="interrupted")
            await self._canceled(turn)
            raise
        except Exception as error:  # a provider changed shape, a socket died…
            log.exception("turn %s failed", turn.id)
            self._remember(turn, question, status="error", reason=str(error) or None)
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
                tools=turn.tools,
                timings=turn.timings(),
            )
        )
        await self._say(turn, "turn", "done", "Turn complete", ms=turn.mark())
        await self.emit(events.Status(stage="idle", turn_id=turn.id))

    async def _use_tools(self, turn: _Turn, messages: list[llm.Message]) -> list[llm.Message]:
        """Let the model call the user's tools, then hand back what to say.

        The shape is decide → run → decide, bounded by `tool_max_rounds`.
        Each round is a *buffered* completion, because a tool call cannot be
        streamed into a synthesiser: the arguments arrive in fragments across
        many chunks and mean nothing until the last one.

        That buffering is real latency in front of the first spoken word, and
        it is why the first thing this does is leave. A session whose owner has
        linked nothing returns the messages it was handed, untouched, having
        made no network call and added no schema to the prompt — which is every
        session until somebody opens the Connectors panel.

        Returns the message list to stream from. On any failure that is the
        original list: an agent that cannot use its tools should still answer.
        """
        if not self.settings.tools_enabled or not self.owner.user_id:
            return messages

        tools = await anyio.to_thread.run_sync(self.agent.tools_for, self.owner.user_id)
        if not tools:
            return messages

        working = list(messages)
        ran = 0

        for round_number in range(self.settings.tool_max_rounds):
            try:
                completion = await llm.complete(
                    working, settings=self.settings, tools=tools
                )
            except ProviderError:
                raise
            except Exception as error:
                log.warning("tool round %d failed: %s", round_number, error)
                # Whatever already ran still counts: its results are in
                # `working` and throwing them away would run somebody's tools
                # and then answer as if they had not.
                return working if ran else messages

            if not completion.wants_tools:
                # Nothing more to run. Which list to hand on is the whole
                # correctness question here: with tools already run, `working`
                # carries their results and the spoken pass must see them —
                # returning `messages` would execute somebody's tools and then
                # answer from a prompt that never mentions what they returned.
                # With nothing run, `messages` is right, because appending this
                # assistant turn would have the spoken pass reply to itself.
                return working if ran else messages

            await self._say(
                turn,
                "tool",
                "start",
                f"Running {len(completion.tool_calls)} tool"
                f"{'' if len(completion.tool_calls) == 1 else 's'}",
                detail=", ".join(call.name for call in completion.tool_calls),
                ms=turn.mark(),
            )

            # The assistant message carrying the calls has to go back verbatim,
            # or the `tool` replies below have nothing to attach to.
            working.append(
                {
                    "role": "assistant",
                    "content": completion.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in completion.tool_calls
                    ],
                }
            )

            for call in completion.tool_calls:
                ran += 1
                result = await self._run_tool(turn, call)
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.for_model(),
                    }
                )

        return working

    async def _run_tool(self, turn: _Turn, call: "llm.ToolCall"):
        """One tool: run it, tell the client, write it down.

        Off the event loop, and with a ceiling on it. A tool that has not
        answered inside `tool_timeout_s` is costing the listener more than it
        is worth, so the turn continues and the model is told it timed out —
        which it can say out loud, unlike a silence.
        """
        assert self.owner.user_id is not None

        try:
            with anyio.fail_after(self.settings.tool_timeout_s):
                result = await anyio.to_thread.run_sync(
                    partial(
                        self.agent.execute,
                        self.owner.user_id,
                        call.name,
                        call.arguments,
                    )
                )
        except TimeoutError:
            from src.integrations.agent import ToolResult

            result = ToolResult(
                call.name,
                ok=False,
                error="timed out",
                ms=self.settings.tool_timeout_s * 1000,
            )

        turn.tools += 1
        await self._say(
            turn,
            "tool",
            "done" if result.ok else "error",
            call.name,
            detail=None if result.ok else (result.error or "failed"),
            ms=turn.mark(),
        )
        self._save_tool(turn, call, result)
        return result

    def _save_tool(self, turn: _Turn, call: "llm.ToolCall", result) -> None:
        """Record the call, eventually — on the same queue as the messages.

        The result is measured, not stored: see `src/chat/tools.py` for why an
        audit table should not become a copy of everything the agent has read.
        """
        if not self.tool_calls.configured:
            return

        rendered = result.for_model() if result.ok else ""

        self._enqueue(
            partial(
                self.tool_calls.record,
                slug=call.name,
                status="ok" if result.ok else "failed",
                conversation_id=self.conversation_id,
                turn_id=turn.id,
                user_id=self.owner.user_id,
                arguments=call.arguments,
                error=None if result.ok else result.error,
                result_bytes=len(rendered) if rendered else None,
                latency_ms=result.ms,
            )
        )

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

    async def _recall(self, question: str, turn: _Turn) -> str | None:
        """What the agent already knows about whoever is speaking.

        The half of memory that reads. Everything about it is deliberately
        weaker than retrieval: it is scoped to one owner, capped at a handful
        of lines, filtered by a similarity floor, bounded by its own timeout
        and allowed to fail silently — because a fact recalled wrongly is
        asserted about the listener in the model's opening sentence, where it
        is both the most damaging thing to get wrong and the least likely to be
        noticed by anything except a human hearing it.

        Returns `None` when there is nothing to say, never an empty section.
        """
        whose = self.whose
        if not whose or not self.memory.configured:
            return None

        found = await self.memory.recall(query=question, owner_id=whose)
        rendered = as_prompt(found)

        if rendered:
            await self._say(
                turn,
                "memory",
                "done",
                f"Recalled {len(found)} thing{'' if len(found) == 1 else 's'} from earlier",
                ms=turn.mark(),
            )
        return rendered

    async def _retrieve(self, question: str, turn: _Turn) -> str | None:
        """The RAG seam. Returns None while `RAG_ENABLED` is false.

        This is the whole switch: with retrieval on, the corpus passages become
        the context the reply is grounded in and the guardrails in
        `src/rag/guardrails.py` decide whether there is an answer at all.

        How hard it works is the turn's own effort level (docs/15-effort.md).
        Rung 0 hands back passages, rung 1 an extracted span, rungs 2 and up an
        answer the pipeline already synthesised and checked — and in every case
        what comes back here is *context for the spoken reply*, not the reply.
        The voice model still writes what is said, because it is the half that
        knows the language, the history and how a sentence sounds out loud.
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

        from src.rag import effort as rungs
        from src.schemas.ask import AskRequest
        from src.services.ask_service import get_ask_service

        level = self.settings.effort_level(turn.effort)
        await self._say(
            turn,
            "retrieval",
            "running",
            "Searching the corpus",
            detail=rungs.name(level),
            ms=turn.mark(),
        )

        service = get_ask_service()
        # Off the event loop, and with the account attached: a user who
        # connected Pinecone is searched against theirs, everybody else against
        # the deployment's store (docs/13-connectors.md).
        answer = await anyio.to_thread.run_sync(
            partial(service.ask, user_id=self.owner.user_id),
            AskRequest(
                transcript=question,
                language_code=turn.language_code,
                effort=level,
                request_id=turn.id,
            ),
        )

        # The stage timings the harness produced, said out loud. This is the
        # inside of the budget — the one place a listener can see which stage
        # spent it, and at which rung.
        await self._say(
            turn,
            "retrieval",
            "done" if answer.status == "answered" else "skipped",
            {
                "answered": f"Found {len(answer.citations)} passages",
                "abstained": "No source covers this",
                "refused": "Question turned down at the gate",
            }[answer.status],
            detail=_retrieval_detail(answer),
            ms=turn.mark(),
        )

        if "direct" in answer.flags:
            # Rung 4's router decided this was never a corpus question —
            # "hello", "say that again". Answer it as a conversation rather
            # than reading out an abstention about sources.
            return None

        if answer.status != "answered" or not answer.citations:
            # An abstention is a real answer: say so rather than inventing one.
            return f"No source covers this. Tell the user: {answer.reason}"

        if answer.method in {"synthesis", "cache"} and answer.answer:
            # The upper rungs already wrote a grounded answer and put it past
            # Gate 4. Handing the passages over again would invite the voice
            # model to write a second, unchecked one from the same material.
            return answer.answer

        return "\n".join(f"- {citation.text}" for citation in answer.citations[:3])

    # ---- bookkeeping ----------------------------------------------------

    def _remember(
        self,
        turn: _Turn,
        question: str,
        *,
        status: str = "answered",
        reason: str | None = None,
    ) -> None:
        """Record the exchange — including a half-spoken one.

        After a barge-in the model has to believe it said exactly what was
        heard, no more. Storing the full generated reply would have it
        referring back to sentences that were cut off mid-word. `turn.spoken`
        is what actually reached the speakers, so both the in-memory history
        and the row in Postgres get the same honest version.

        Called on every way out of a turn, which is why the status lives here:
        it is the one place that sees success, barge-in and failure alike.
        """
        self.history.append({"role": "user", "content": question})
        if turn.spoken.strip():
            self.history.append({"role": "assistant", "content": turn.spoken.strip()})

        self._save(
            turn,
            "assistant",
            turn.spoken,
            status=status,
            reason=reason,
            latency_ms=turn.mark(),
        )

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
