"""The voice surface: one WebSocket for conversation, two routes for parts.

The WebSocket exists because a turn is a stream in both directions and because
audio has to arrive as bytes. HTTP could carry the reply — chunked, or as SSE —
but not the interruption: barge-in means the client says "stop" *while* the
server is mid-send, and a request/response channel has nowhere to put that.

The two REST routes are the same pipeline with the conversation taken out, for
when you want to check one stage without holding a session open.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse

from src.core.config import Settings, get_settings
from src.schemas import voice as events
from src.schemas.voice import SpeakRequest, TranscribeResponse
from src.services.voice_service import VoiceSession
from src.voice import stt, tts
from src.voice.http import ProviderError
from src.voice.languages import display, normalise

log = logging.getLogger("vec.voice.ws")

router = APIRouter(prefix="/voice", tags=["voice"])

# A take is capped at 30 s by Saaras; this is the byte ceiling that goes with
# it, enforced while the audio arrives rather than after, so a client that
# ignores the limit cannot fill memory before we look.
_MAX_AUDIO = 8 * 1024 * 1024


SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/config", response_model=events.Ready, summary="Which providers are wired up")
def config(settings: SettingsDep) -> events.Ready:
    """What the client can expect before it opens a socket."""
    return VoiceSession(emit=_unused, send_audio=_unused, settings=settings).describe()


@router.websocket("/ws")
async def voice_ws(websocket: WebSocket) -> None:
    """A conversation.

    Client → server, JSON: `audio.start` `audio.end` `text` `cancel` `reset`
    `ping`; binary frames in between `audio.start` and `audio.end` are the
    recording. Server → client: the events in `src/schemas/voice.py`, with
    binary frames carrying PCM for the segment last announced.
    """
    await websocket.accept()
    settings = get_settings()

    # Starlette's send is not safe to call from two tasks at once, and here two
    # do: the turn streams events while the receive loop answers pings.
    lock = asyncio.Lock()
    closed = asyncio.Event()

    async def emit(event: events.Wire) -> None:
        if closed.is_set():
            return
        async with lock:
            try:
                await websocket.send_text(event.model_dump_json(by_alias=True))
            except (RuntimeError, WebSocketDisconnect):
                closed.set()

    async def send_audio(chunk: bytes) -> None:
        if closed.is_set():
            return
        async with lock:
            try:
                await websocket.send_bytes(chunk)
            except (RuntimeError, WebSocketDisconnect):
                closed.set()

    session = VoiceSession(emit=emit, send_audio=send_audio, settings=settings)
    await emit(session.describe())

    recording: bytearray | None = None
    mime = "audio/webm"
    language: str | None = None
    overflowed = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if recording is None:
                    continue  # audio outside a take — nothing asked for it
                if len(recording) + len(data) > _MAX_AUDIO:
                    overflowed = True
                    continue
                recording.extend(data)
                continue

            raw = message.get("text")
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await emit(events.VoiceError(message="That wasn't JSON."))
                continue

            kind = payload.get("type")

            if kind == "audio.start":
                # Talking over the assistant stops it. That is barge-in, and it
                # is the difference between a conversation and a walkie-talkie.
                session.cancel()
                recording = bytearray()
                overflowed = False
                mime = str(payload.get("mime") or "audio/webm")
                language = normalise(payload.get("language"))

            elif kind == "audio.end":
                take, recording = recording, None
                if overflowed:
                    await emit(
                        events.VoiceError(
                            stage="stt",
                            message="That take is too long — keep it under 30 seconds.",
                        )
                    )
                elif take:
                    session.start(session.from_audio(bytes(take), mime=mime, language=language))

            elif kind == "text":
                text = str(payload.get("text") or "").strip()
                if text:
                    session.start(
                        session.from_text(text, language=normalise(payload.get("language")))
                    )

            elif kind == "cancel":
                session.cancel()

            elif kind == "reset":
                session.reset()
                await emit(events.Status(stage="idle"))

            elif kind == "ping":
                await emit(events.Pong())

    except WebSocketDisconnect:
        pass
    except Exception:  # a client that sends nonsense should not page anyone
        log.exception("voice socket failed")
    finally:
        closed.set()
        await session.aclose()


@router.post("/transcribe", response_model=TranscribeResponse, summary="Audio in, text out")
async def transcribe(
    file: Annotated[UploadFile, File(description="A recording, 30 seconds or less")],
    language: Annotated[str | None, Form(description="Force a language; omit to detect")] = None,
) -> TranscribeResponse:
    """Saaras on its own, for checking the hearing half without a socket."""
    settings = get_settings()
    audio = await file.read()
    started = time.perf_counter()

    try:
        heard = await stt.transcribe(
            audio,
            mime=file.content_type or "audio/webm",
            settings=settings,
            language=language,
        )
    except ProviderError as error:
        raise HTTPException(status_code=error.status or 502, detail=str(error)) from error

    return TranscribeResponse(
        transcript=heard.text,
        language_code=heard.language_code,
        language=display(heard.language_code),
        confidence=heard.confidence,
        provider=heard.provider,
        ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.post("/speak", summary="Text in, streamed audio out")
async def speak(request: SpeakRequest) -> StreamingResponse:
    """Bulbul on its own. Streams while it synthesises — `curl -N` hears it."""
    settings = get_settings()
    voice = tts.choose(request.language_code, settings)

    if voice.provider == "none":
        raise HTTPException(status_code=503, detail="No speech provider is configured.")

    async def audio():
        if request.format == "wav":
            yield _wav_header(voice.sample_rate)
        try:
            async for chunk in tts.stream_speech(request.text, voice, settings=settings):
                yield chunk
        except ProviderError as error:
            # The status line is long gone by here; failing mid-stream can only
            # be reported by stopping, so it is logged where someone will see it.
            log.warning("speak failed: %s", error)

    media = "audio/wav" if request.format == "wav" else "audio/pcm"
    return StreamingResponse(
        audio(),
        media_type=media,
        headers={
            "x-voice-provider": voice.provider,
            "x-voice-speaker": voice.voice,
            "x-voice-sample-rate": str(voice.sample_rate),
            "cache-control": "no-store",
        },
    )


def _wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A RIFF header for audio whose length nobody knows yet.

    The size fields are written as the largest value that fits. That is what
    every streaming WAV encoder does: the real length is unknowable at the
    moment the header goes out, and players read until the socket closes.
    """
    block = channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block, block, bits)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


async def _unused(*_args, **_kwargs) -> None:
    """`/voice/config` builds a session only to ask it what it can do."""
