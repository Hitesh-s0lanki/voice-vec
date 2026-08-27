"""The reply, token by token.

Any OpenAI-compatible chat-completions endpoint does — OpenAI's own and
Sarvam's both are, which is what lets `resolve_llm()` pick between them without
this file knowing which one it got.

Streaming is not a nicety here. The synthesiser is fed from this stream, so the
first sound the listener hears is gated on the first *clause*, not the last
token: waiting for a finished reply would add its whole generation time to the
silence before speech starts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

from src.core.config import LlmTarget, Settings
from src.voice.http import ProviderError, get_client, read_error
from src.voice.languages import LANGUAGES, display

log = logging.getLogger("vec.voice.llm")

Message = dict[str, str]

# The chat-completions protocol is one thing; which of its parameters a given
# model still accepts is another. OpenAI's newer models renamed `max_tokens` to
# `max_completion_tokens` and several of them take only the default
# temperature, while Sarvam's endpoint speaks the older spelling. So the token
# cap is named per provider, and anything the model *still* refuses is dropped
# on one retry rather than hardcoded into a list of model quirks here.
_UNSUPPORTED = re.compile(
    r"[Uu]nsupported (?:parameter|value)s?: '([\w.]+)'"
    r"|[Uu]nrecognized request argument supplied: ([\w.]+)"
)
_INSTEAD = re.compile(r"[Uu]se '([\w.]+)' instead")
# Give these up and there is nothing left to ask for.
_ESSENTIAL = frozenset({"model", "messages", "stream"})

# What a model has already refused, and what to send instead — `None` meaning
# send nothing. Learning costs one wasted round trip, and this is what keeps it
# to one per process instead of one in front of every reply, which is exactly
# the latency this file exists to avoid.
_LEARNED: dict[str, dict[str, str | None]] = {}


class _Incompatible(Exception):
    """A parameter the model rejected, and the payload with it made good."""

    def __init__(self, payload: dict, field: str, replacement: str | None) -> None:
        super().__init__(field)
        self.payload = payload
        self.field = field
        self.replacement = replacement


def build_payload(messages: list[Message], *, settings: Settings, llm: LlmTarget) -> dict:
    """The request body, with the token cap spelled the way this provider wants."""
    cap = "max_completion_tokens" if llm.provider == "openai" else "max_tokens"
    payload = {
        "model": llm.model,
        "messages": messages,
        "stream": True,
        "temperature": settings.llm_temperature,
        cap: settings.llm_max_tokens,
    }
    for field, replacement in _LEARNED.get(_key(llm), {}).items():
        if field in payload:
            value = payload.pop(field)
            if replacement:
                payload[replacement] = value
    return payload


def adapt_payload(payload: dict, message: str) -> tuple[dict, str, str | None] | None:
    """Rework a payload around the parameter an error names, or None.

    None means the refusal was about something else — a bad key, a missing
    model — and retrying the same call would only fail the same way.
    """
    match = _UNSUPPORTED.search(message)
    if not match:
        return None

    field = match.group(1) or match.group(2)
    if field not in payload or field in _ESSENTIAL:
        return None

    fixed = dict(payload)
    value = fixed.pop(field)
    rename = _INSTEAD.search(message)
    # "Use 'max_completion_tokens' instead" is a rename; a bare "unsupported"
    # (temperature on a reasoning model) means the value has to go entirely.
    replacement = rename.group(1) if rename and rename.group(1) not in fixed else None
    if replacement:
        fixed[replacement] = value
    return fixed, field, replacement


def _key(llm: LlmTarget) -> str:
    return f"{llm.provider}:{llm.model}"


# Written for a synthesiser, not a screen. Every rule here exists because its
# absence is audible: markdown gets read out as punctuation, a six-sentence
# answer outlasts the listener's patience, and a model that quietly switches to
# English is the single most common way a multilingual voice app breaks.
_SYSTEM = """You are Vec, a voice assistant. Everything you write is spoken aloud immediately.

Language:
- Reply in {language}. Match the language the user actually spoke, every time, for the whole reply.
- If the user switches language mid-conversation, switch with them.
- Never explain which language you are using, and never translate your own answer.

Voice:
- Two or three sentences. Lead with the answer; drop the preamble.
- Plain spoken sentences. No markdown, no bullet points, no emoji, no headings, no code blocks.
- Write numbers, dates and units the way you would say them out loud.
- If you do not know, say so briefly and stop.
"""

_LANGUAGE_UNKNOWN = "the same language the user spoke"


def system_prompt(
    language_code: str | None,
    context: str | None = None,
    memories: str | None = None,
) -> str:
    """The instruction the model opens with, aimed at the detected language."""
    name = display(language_code)
    # A code we have no name for — a language outside Sarvam's list — is worse
    # than useless in the prompt: "reply in fr-FR" invites a model to answer
    # *about* the code. Fall back to the instruction that always holds.
    language = name if name and language_code in LANGUAGES else _LANGUAGE_UNKNOWN

    prompt = _SYSTEM.format(language=language)

    if memories:
        # Facts Redis Agent Memory extracted from earlier conversations with
        # this same listener (src/memory/store.py) — the only part of this
        # prompt that outlives a conversation.
        #
        # Three sentences of framing for what is usually one line of content,
        # because the failure mode is specific and expensive out loud: a model
        # handed bare facts recites them ("Since you're vegetarian…") when
        # nobody asked, which is how an assistant that remembers stops sounding
        # like one that listens. They are also *stale by construction* —
        # extracted from a conversation that has since ended — so the live
        # transcript has to be told, explicitly, that it wins.
        prompt += (
            "\nWhat you already know about this person, from earlier conversations:\n"
            f"{memories}\n"
            "Use this only when it changes the answer. Never recite it, never mention "
            "remembering, and never bring it up unprompted. If what they say now "
            "contradicts it, they are right and it is out of date.\n"
        )

    if context:
        # Only reachable with RAG_ENABLED=true. Retrieval is off in this build.
        prompt += (
            "\nAnswer from these sources, and say you do not have it rather than "
            "filling the gap yourself:\n"
            f"{context}\n"
        )

    return prompt


def build_messages(
    *,
    transcript: str,
    history: list[Message],
    language_code: str | None,
    context: str | None = None,
    memories: str | None = None,
    max_turns: int = 8,
) -> list[Message]:
    """System prompt, the recent past, then what was just said.

    The system prompt is rebuilt every turn rather than pinned once, because
    the detected language can change between turns and the language rule is the
    part that has to be right.
    """
    recent = history[-(max_turns * 2) :] if max_turns > 0 else []
    return [
        {"role": "system", "content": system_prompt(language_code, context, memories)},
        *recent,
        {"role": "user", "content": transcript},
    ]


async def stream_reply(
    messages: list[Message],
    *,
    settings: Settings,
    target: LlmTarget | None = None,
) -> AsyncIterator[str]:
    """Yield reply text as it is generated. Raises ProviderError on refusal."""
    llm = target or settings.resolve_llm()
    if not llm.ready:
        raise ProviderError(
            "No reply model is configured — set OPENAI_API_KEY or SARVAM_API_KEY.",
            provider="none",
        )

    payload = build_payload(messages, settings=settings, llm=llm)

    # Each retry gives up exactly one parameter, so the loop is bounded by the
    # number of optional ones and cannot spin. Nothing has been yielded when
    # `_Incompatible` is raised — the refusal arrives before the first token —
    # so the listener hears one reply, not a stutter.
    for _ in range(len(payload)):
        try:
            async for piece in _stream_once(payload, llm=llm):
                yield piece
            return
        except _Incompatible as retry:
            log.info(
                "%s rejected %r on %s; sending %s from here on",
                llm.provider,
                retry.field,
                llm.model,
                retry.replacement or "nothing in its place",
            )
            _LEARNED.setdefault(_key(llm), {})[retry.field] = retry.replacement
            payload = retry.payload


async def _stream_once(payload: dict, *, llm: LlmTarget) -> AsyncIterator[str]:
    """One attempt. Raises `_Incompatible` if the model refused a parameter."""
    client = get_client()
    async with client.stream(
        "POST",
        f"{llm.base_url}/chat/completions",
        headers={"authorization": f"Bearer {llm.api_key}"},
        json=payload,
    ) as response:
        if response.status_code >= 400:
            raw = await response.aread()
            message = read_error(
                _loads(raw), f"{llm.provider} responded with {response.status_code}."
            )
            adapted = adapt_payload(payload, message) if response.status_code == 400 else None
            if adapted is not None:
                raise _Incompatible(*adapted)
            raise ProviderError(
                message,
                status=response.status_code,
                provider=llm.provider,
            )

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    break
                continue

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                log.debug("undecodable stream line: %s", data[:120])
                continue

            for choice in chunk.get("choices") or []:
                # `reasoning_content` is deliberately ignored: it is thinking,
                # not speech, and reading it aloud would be nonsense.
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece


def _loads(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except Exception:
        return raw.decode("utf-8", "replace")
