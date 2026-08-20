"""A segment of text, streamed back as sound.

Both providers are asked for raw little-endian 16-bit mono PCM rather than MP3
or WAV. That is the format a browser's Web Audio API can play the instant it
arrives: a compressed chunk has to be decoded as a *whole file* before a single
sample comes out of it, which puts the whole download in front of playback and
undoes the streaming. PCM has no header and no framing, so byte N is playable
without byte N+1 — the client converts and schedules it as it lands.

Bulbul speaks the eleven languages it speaks well; OpenAI covers the rest. The
sample rate travels with the audio because the two do not agree on one: OpenAI
emits 24 kHz and nothing else, Sarvam emits whatever it is asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from src.core.config import Settings
from src.voice.http import ProviderError, get_client, read_error
from src.voice.languages import normalise, speaks

log = logging.getLogger("vec.voice.tts")

SARVAM_TTS_STREAM = "https://api.sarvam.ai/text-to-speech/stream"

# OpenAI's `pcm` response format is fixed at 24 kHz, 16-bit, mono.
OPENAI_SAMPLE_RATE = 24_000

# Bulbul takes 3,500 characters per request; segments are a tenth of that at
# most, so this only ever catches a pathological one.
MAX_CHARS = 3_000


@dataclass(frozen=True, slots=True)
class Voice:
    """Who is speaking, and in what format — sent ahead of the audio."""

    provider: str
    voice: str
    language_code: str | None
    sample_rate: int
    fmt: str = "pcm_s16le"


def choose(language_code: str | None, settings: Settings) -> Voice:
    """Pick the synthesiser for a language before any text exists.

    Called once per turn rather than per segment: switching voices halfway
    through a reply would be heard.
    """
    language = normalise(language_code)

    if settings.sarvam_ready and speaks(language):
        return Voice(
            provider="sarvam",
            voice=settings.tts_speaker,
            language_code=language,
            sample_rate=settings.tts_sample_rate,
        )

    if settings.openai_ready:
        return Voice(
            provider="openai",
            voice=settings.openai_tts_voice,
            language_code=language,
            sample_rate=OPENAI_SAMPLE_RATE,
        )

    if settings.sarvam_ready:
        # Off Bulbul's list with no OpenAI key. Bulbul will read the script it
        # is given with an Indian-English voice — imperfect, and better than
        # silence.
        return Voice(
            provider="sarvam",
            voice=settings.tts_speaker,
            language_code="en-IN",
            sample_rate=settings.tts_sample_rate,
        )

    return Voice(provider="none", voice="", language_code=language, sample_rate=OPENAI_SAMPLE_RATE)


async def stream_speech(
    text: str,
    voice: Voice,
    *,
    settings: Settings,
) -> AsyncIterator[bytes]:
    """Yield PCM for one segment, in the format `voice` describes."""
    spoken = text.strip()[:MAX_CHARS]
    if not spoken:
        return

    if voice.provider == "sarvam":
        async for chunk in _sarvam(spoken, voice, settings):
            yield chunk
    elif voice.provider == "openai":
        async for chunk in _openai(spoken, voice, settings):
            yield chunk
    else:
        raise ProviderError(
            "No speech provider is configured — set SARVAM_API_KEY.",
            provider="none",
        )


async def _sarvam(text: str, voice: Voice, settings: Settings) -> AsyncIterator[bytes]:
    payload = {
        "text": text,
        "language_code": voice.language_code or "en-IN",
        "model": settings.tts_model,
        "speaker": voice.voice,
        "pace": settings.tts_pace,
        "output_audio_codec": "linear16",
        "speech_sample_rate": voice.sample_rate,
    }

    async with get_client().stream(
        "POST",
        SARVAM_TTS_STREAM,
        headers={"api-subscription-key": settings.sarvam_api_key},
        json=payload,
    ) as response:
        if response.status_code >= 400:
            raw = await response.aread()
            raise ProviderError(
                read_error(_loads(raw), f"Sarvam responded with {response.status_code}."),
                status=response.status_code,
                provider="sarvam",
            )

        async for chunk in _headerless(response.aiter_bytes()):
            yield chunk


async def _openai(text: str, voice: Voice, settings: Settings) -> AsyncIterator[bytes]:
    payload = {
        "model": settings.openai_tts_model,
        "input": text,
        "voice": voice.voice,
        "response_format": "pcm",
        "stream_format": "audio",
    }

    async with get_client().stream(
        "POST",
        f"{settings.openai_base_url}/audio/speech",
        headers={"authorization": f"Bearer {settings.openai_api_key}"},
        json=payload,
    ) as response:
        if response.status_code >= 400:
            raw = await response.aread()
            raise ProviderError(
                read_error(_loads(raw), f"OpenAI responded with {response.status_code}."),
                status=response.status_code,
                provider="openai",
            )

        async for chunk in _headerless(response.aiter_bytes()):
            yield chunk


async def _headerless(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Drop a RIFF header if one shows up.

    Both providers are asked for raw PCM and both deliver it, but a WAV header
    on the front is the one difference that would be silently catastrophic: 44
    bytes of ASCII played as samples is a click, and every sample after it is
    shifted by two bytes, which sounds like static rather than a voice.
    """
    first = True
    async for chunk in chunks:
        if first:
            first = False
            if chunk[:4] == b"RIFF":
                data = chunk.find(b"data")
                chunk = chunk[data + 8 :] if data != -1 else chunk[44:]
            if not chunk:
                continue
        yield chunk


def _loads(raw: bytes) -> object:
    import json

    try:
        return json.loads(raw)
    except Exception:
        return raw.decode("utf-8", "replace")
