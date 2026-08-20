"""Audio in, transcript and a language out.

Sarvam Saaras does the hearing (requirement 1 in problem-statement.md picks
Sarvam or ElevenLabs; this repo picked Sarvam). It is asked to detect the
language rather than being told one, because the speaker chooses the language
here — nobody sets a dropdown before talking, and the whole app turns on
getting that answer right.

If Sarvam fails and an OpenAI key is present, Whisper picks it up. That is the
only reason a French speaker gets anywhere: Saaras covers 22 Indic languages
plus English, and outside that list it has nothing to say.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.config import Settings
from src.voice.http import ProviderError, get_client, read_error
from src.voice.languages import normalise

log = logging.getLogger("vec.voice.stt")

SARVAM_STT = "https://api.sarvam.ai/speech-to-text"


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language_code: str | None
    confidence: float | None
    provider: str
    request_id: str | None = None


async def transcribe(
    audio: bytes,
    *,
    mime: str,
    settings: Settings,
    language: str | None = None,
) -> Transcript:
    """Whichever provider can hear it. Raises ProviderError when none can."""
    if not audio:
        raise ProviderError("No audio was received.")
    if len(audio) > settings.stt_max_bytes:
        raise ProviderError("That take is too long — keep it under 30 seconds.")

    if settings.sarvam_ready:
        try:
            return await _sarvam(audio, mime=mime, settings=settings, language=language)
        except ProviderError as error:
            if not settings.openai_ready:
                raise
            # Saaras returns 4xx for a language it does not cover. Whisper does,
            # so the turn is still answerable — just not by Sarvam.
            log.info("sarvam stt failed (%s) — falling back to openai", error)

    if settings.openai_ready:
        return await _openai(audio, mime=mime, settings=settings, language=language)

    raise ProviderError(
        "No speech-to-text provider is configured — set SARVAM_API_KEY.",
        provider="none",
    )


async def _sarvam(
    audio: bytes,
    *,
    mime: str,
    settings: Settings,
    language: str | None,
) -> Transcript:
    # Sarvam matches content types exactly: `audio/webm` passes, the
    # `audio/webm;codecs=opus` a browser records does not.
    content_type = (mime or "audio/webm").split(";")[0].strip()

    form = {
        "model": settings.stt_model,
        "language_code": normalise(language) or settings.stt_language,
    }
    # `mode` exists on saaras:v3 only — v4 rejects it.
    if settings.stt_model.startswith("saaras:v3"):
        form["mode"] = "transcribe"

    response = await get_client().post(
        SARVAM_STT,
        headers={"api-subscription-key": settings.sarvam_api_key},
        files={"file": (f"speech.{_extension(content_type)}", audio, content_type)},
        data=form,
    )

    body = _json(response)
    if response.status_code >= 400:
        raise ProviderError(
            read_error(body, f"Sarvam responded with {response.status_code}."),
            status=response.status_code,
            provider="sarvam",
        )

    text = str((body or {}).get("transcript") or "").strip()
    if not text:
        raise ProviderError("Nothing was picked up — try speaking a little closer.")

    return Transcript(
        text=text,
        language_code=normalise((body or {}).get("language_code")),
        confidence=(body or {}).get("language_probability"),
        provider="sarvam",
        request_id=(body or {}).get("request_id"),
    )


async def _openai(
    audio: bytes,
    *,
    mime: str,
    settings: Settings,
    language: str | None,
) -> Transcript:
    """Whisper, for everything Saaras does not cover."""
    content_type = (mime or "audio/webm").split(";")[0].strip()

    form: dict[str, str] = {"model": "whisper-1", "response_format": "verbose_json"}
    normalised = normalise(language)
    if normalised:
        form["language"] = normalised.split("-")[0]

    response = await get_client().post(
        f"{settings.openai_base_url}/audio/transcriptions",
        headers={"authorization": f"Bearer {settings.openai_api_key}"},
        files={"file": (f"speech.{_extension(content_type)}", audio, content_type)},
        data=form,
    )

    body = _json(response)
    if response.status_code >= 400:
        raise ProviderError(
            read_error(body, f"OpenAI responded with {response.status_code}."),
            status=response.status_code,
            provider="openai",
        )

    text = str((body or {}).get("text") or "").strip()
    if not text:
        raise ProviderError("Nothing was picked up — try speaking a little closer.")

    # Whisper reports a bare ISO code (`ta`) or an English name (`tamil`).
    return Transcript(
        text=text,
        language_code=normalise((body or {}).get("language")),
        confidence=None,
        provider="openai",
    )


def _extension(content_type: str) -> str:
    if "mp4" in content_type or "m4a" in content_type:
        return "m4a"
    if "ogg" in content_type:
        return "ogg"
    if "wav" in content_type:
        return "wav"
    if "mpeg" in content_type or "mp3" in content_type:
        return "mp3"
    return "webm"


def _json(response) -> dict | str | None:
    try:
        return response.json()
    except Exception:
        return response.text
