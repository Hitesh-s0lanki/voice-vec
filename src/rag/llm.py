"""The synchronous model client the upper rungs answer with.

`src/voice/llm.py` is the *voice* model: async, streamed token by token,
because the synthesiser is reading its output as it arrives. Nothing on this
path is spoken while it is written — a grader's verdict is worthless until it
is complete, and rung 2's synthesis is checked by Gate 4 before anyone hears
it — so streaming would buy nothing and cost an async/sync bridge in the middle
of a pipeline that FastAPI already runs in a worker thread.

So: one blocking POST, one reply, no event loop. The parameter-naming quirks
are the one thing shared with the voice client, and `adapt_payload` is imported
rather than reimplemented so a model that renames `max_tokens` is learned about
in one place.

**Every call here is a network round trip after the transcript arrived**, which
is precisely what rungs 0 and 1 are defined by not doing. Nothing in this module
is reachable below rung 2, and the latency it adds is reported rather than
folded into the 200 ms figure (docs/04-latency.md).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from src.core.config import LlmTarget, Settings
from src.voice.llm import adapt_payload

log = logging.getLogger("vec.rag.llm")

Message = dict[str, str]

# A model asked for JSON returns JSON, a fence around JSON, or a sentence and
# then JSON. All three are common and only the first parses, so the object is
# cut out rather than trusted to be the whole reply.
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LlmUnavailable(RuntimeError):
    """No key, no model, or the provider refused. Callers degrade; never 500."""


def ready(settings: Settings) -> bool:
    return settings.resolve_llm().ready


def complete(
    messages: list[Message],
    *,
    settings: Settings,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> str:
    """One completion, as text. Raises `LlmUnavailable` rather than returning junk."""
    target = settings.resolve_llm()
    if not target.ready:
        raise LlmUnavailable("no model is configured — set OPENAI_API_KEY or SARVAM_API_KEY")

    cap = "max_completion_tokens" if target.provider == "openai" else "max_tokens"
    payload: dict[str, Any] = {
        "model": target.model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        cap: max_tokens,
    }

    # One retry, and only for a parameter the model named. Anything else — a
    # bad key, a missing model, a rate limit — fails the same way twice and the
    # retry would only spend more of the rung's budget confirming it.
    for attempt in (0, 1):
        try:
            return _post(target, payload, timeout_s)
        except _Refused as refusal:
            adapted = adapt_payload(payload, refusal.message) if attempt == 0 else None
            if adapted is None:
                raise LlmUnavailable(refusal.message) from refusal
            payload, field, replacement = adapted
            log.info("%s rejected %r; retrying as %r", target.model, field, replacement)

    raise LlmUnavailable("the model refused every payload we could build")


def complete_json(
    messages: list[Message],
    *,
    settings: Settings,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any] | None:
    """A structured verdict, or None when the model did not produce one.

    None is a real answer here and every caller has to handle it: the graders
    on rungs 3 and 4 treat an unparseable verdict as "no useful judgement" and
    fall back to the deterministic guardrail, rather than defaulting to `yes`
    and letting a parse failure quietly approve an answer.

    `response_format` is *not* sent. OpenAI honours it, Sarvam's endpoint does
    not, and a request that is refused for carrying it costs the round trip the
    strictness was meant to protect. The prompt asks for bare JSON and this
    parses defensively, which works on both.
    """
    try:
        raw = complete(
            messages,
            settings=settings,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout_s=timeout_s,
        )
    except LlmUnavailable as error:
        log.info("grader unavailable: %s", error)
        return None

    match = _OBJECT.search(raw)
    if not match:
        log.debug("grader returned no JSON object: %r", raw[:200])
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.debug("grader returned malformed JSON: %r", raw[:200])
        return None

    return parsed if isinstance(parsed, dict) else None


class _Refused(Exception):
    """The provider answered with an error body worth reading."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _post(target: LlmTarget, payload: dict[str, Any], timeout_s: float) -> str:
    try:
        response = httpx.post(
            f"{target.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {target.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
    except httpx.HTTPError as error:
        raise LlmUnavailable(f"{target.provider}: {error}") from error

    if response.status_code >= 400:
        raise _Refused(_error_text(response))

    try:
        body = response.json()
        return str(body["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LlmUnavailable(f"{target.provider}: unreadable reply") from error


def _error_text(response: httpx.Response) -> str:
    """The provider's message, which is what `adapt_payload` reads."""
    try:
        body = response.json()
    except ValueError:
        return f"{response.status_code}: {response.text[:200]}"

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if body.get("message"):
            return str(body["message"])
    return f"{response.status_code}: {str(body)[:200]}"
