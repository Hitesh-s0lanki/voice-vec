"""The reply, token by token.

Any OpenAI-compatible chat-completions endpoint does — OpenAI's own and
Sarvam's both are, which is what lets `resolve_llm()` pick between them without
this file knowing which one it got.

Streaming is not a nicety here. The synthesiser is fed from this stream, so the
first sound the listener hears is gated on the first *clause*, not the last
token: waiting for a finished reply would add its whole generation time to the
silence before speech starts.

`complete()` is the exception and exists for exactly one reason: a tool call
cannot be streamed into a synthesiser. Deciding *whether* to call a tool is a
whole-response question — the arguments arrive in fragments across many chunks
and mean nothing until the last one — so that pass is buffered, and only the
final answer, once the tools have run, goes through `stream_reply`. Nothing
pays for it unless the caller passes tools.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

from src.core.config import LlmTarget, Settings
from src.voice.http import ProviderError, get_client, read_error
from src.voice.languages import LANGUAGES, display

log = logging.getLogger("vec.voice.llm")

# Loose on purpose. A plain turn is {"role", "content"}, but a tool round trip
# adds an assistant message carrying `tool_calls` (a list) and tool results
# carrying `tool_call_id`. Typing this as dict[str, str] would have been a lie
# the moment tools arrived.
Message = dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool the model asked for, with its arguments already parsed.

    `arguments` arrives as a JSON *string* on the wire and is decoded here so
    no caller has to remember to. A model that emits malformed JSON — which
    happens — becomes an empty dict rather than an exception, because the tool
    refusing sensibly is a better turn than the whole reply failing.
    """

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class Completion:
    """A buffered reply: what it said, and what it wants run."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

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


def build_payload(
    messages: list[Message],
    *,
    settings: Settings,
    llm: LlmTarget,
    stream: bool = True,
    tools: list[dict] | None = None,
) -> dict:
    """The request body, with the token cap spelled the way this provider wants."""
    cap = "max_completion_tokens" if llm.provider == "openai" else "max_tokens"
    payload: dict = {
        "model": llm.model,
        "messages": messages,
        "stream": stream,
        "temperature": settings.llm_temperature,
        cap: settings.llm_max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
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
#
# The `Facts` block is newer and is here because the answering path changed.
# With the vector index off (docs/18-datasets.md), the only grounded source is
# a dataset queried through a tool — and a model handed a *description* of a
# dataset will happily answer from the description. "How many Hindi rows are
# there" gets "about twenty-five thousand" read straight off the card, spoken
# with total confidence, without a query ever running. Out loud there is
# nothing to distinguish that from a real answer: no citation, no row count, no
# SQL a listener could check. So the rule is stated as a rule.
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

Facts:
- Any number, count, total, comparison, or specific record about data you can reach
  must come from actually running a query. Never read one off a description, and
  never estimate one from memory.
- When results come back cut short, say so — "at least forty" and not "forty".
- When they describe part of a dataset rather than all of it, say that too.
- Never read a list of results aloud. Say how many there were and name one or two.
"""

_LANGUAGE_UNKNOWN = "the same language the user spoke"


def system_prompt(
    language_code: str | None,
    context: str | None = None,
    memories: str | None = None,
    stores: str | None = None,
    discovery: bool = False,
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

    if stores:
        # What this listener has connected, measured rather than assumed:
        # stores and tools from `src/connectors/profile.py`, and datasets from
        # `src/datasets/profile.py`. A string, already rendered, because this
        # module may not import either — `src.connectors.narrate` reaches
        # `src.rag.llm`, which imports this file, and the cycle would close on
        # the first profile written.
        #
        # Framed as *reach* rather than as knowledge, and that framing is doing
        # more work now than it used to. A model handed a list of corpora
        # starts answering from the list — "I have twelve books on habits" —
        # and a model handed a dataset card starts answering from the card,
        # which is worse: the card carries real numbers, so what it recites
        # sounds exactly like a measurement. The `Facts` rules above are the
        # other half of this; both are needed, because the card cannot be
        # withheld (it is how the model decides whether to query at all) and it
        # cannot be trusted to be read as a menu rather than as an answer.
        prompt += (
            "\nWhat you can reach for this person:\n"
            f"{stores}\n"
            "This is reach, not knowledge: it is what you can look up, act on or "
            "query, and it is never the answer itself. Anything countable or "
            "specific comes from running a query against it. If a question falls "
            "outside all of it, say you do not have it rather than answering from "
            "memory.\n"
        )
        if discovery:
            # The counts above say *that* something is there; this says how to
            # find out what. Written as the first step rather than as an
            # option, because the failure it prevents is silent: a model that
            # skips discovery answers a question about somebody's own data from
            # its own memory, fluently, with nothing to mark it as invented.
            prompt += (
                "You have not been told what any of it holds. Before answering "
                "anything that needs their data or an action, call "
                "`find_capability` with what you need in plain words; it names "
                "which one fits and the exact tool to call next. Never guess "
                "which source to use, and never answer from these counts.\n"
            )

    if context:
        # Only reachable for a listener who attached a vector store: the index
        # is not the whole answering path any more (docs/18-datasets.md).
        # Retrieval and datasets are additive rather than alternatives — a
        # connected index appends passages here while the dataset tool keeps
        # answering the countable half — so both can be in one prompt.
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
    stores: str | None = None,
    discovery: bool = False,
    max_turns: int = 8,
) -> list[Message]:
    """System prompt, the recent past, then what was just said.

    The system prompt is rebuilt every turn rather than pinned once, because
    the detected language can change between turns and the language rule is the
    part that has to be right.
    """
    recent = history[-(max_turns * 2) :] if max_turns > 0 else []
    return [
        {
            "role": "system",
            "content": system_prompt(language_code, context, memories, stores, discovery),
        },
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


def _tool_calls(message: dict) -> tuple[ToolCall, ...]:
    """The tool calls off a finished choice, parsed and filtered.

    A call with no name is dropped rather than passed on: it cannot be routed
    to anything, and letting it through would mean an execution layer deciding
    what an unnamed tool means.
    """
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue

        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            # Models do emit malformed argument JSON. An empty dict lets the
            # tool refuse on its own terms, which reads far better than the
            # whole turn dying on a parse error.
            log.info("tool %s sent arguments that would not parse", name)
            arguments = {}

        calls.append(
            ToolCall(
                id=str(raw.get("id") or ""),
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return tuple(calls)


async def complete(
    messages: list[Message],
    *,
    settings: Settings,
    tools: list[dict] | None = None,
    target: LlmTarget | None = None,
) -> Completion:
    """One buffered reply, so tool calls can be read whole.

    The same parameter-learning retry as `stream_reply`, for the same reason:
    the two share a provider and a model, so a rename learned here saves the
    streaming pass a wasted round trip and vice versa.

    A model that rejects `tools` outright degrades to answering without them —
    `adapt_payload` drops the parameter and the retry goes through — which is
    the right outcome for a provider that has no tool support at all.
    """
    llm = target or settings.resolve_llm()
    if not llm.ready:
        raise ProviderError(
            "No reply model is configured — set OPENAI_API_KEY or SARVAM_API_KEY.",
            provider="none",
        )

    payload = build_payload(messages, settings=settings, llm=llm, stream=False, tools=tools)

    for _ in range(len(payload)):
        try:
            return await _complete_once(payload, llm=llm)
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

    return Completion(content="")


async def _complete_once(payload: dict, *, llm: LlmTarget) -> Completion:
    """One attempt. Raises `_Incompatible` if the model refused a parameter."""
    client = get_client()
    response = await client.post(
        f"{llm.base_url}/chat/completions",
        headers={"authorization": f"Bearer {llm.api_key}"},
        json=payload,
    )

    if response.status_code >= 400:
        body = _loads(response.content)
        message = read_error(body, f"{llm.provider} responded with {response.status_code}.")
        adapted = adapt_payload(payload, message) if response.status_code == 400 else None
        if adapted is not None:
            raise _Incompatible(*adapted)
        raise ProviderError(message, status=response.status_code, provider=llm.provider)

    choices = (response.json() or {}).get("choices") or []
    if not choices:
        return Completion(content="")

    message = choices[0].get("message") or {}
    return Completion(
        content=str(message.get("content") or ""),
        tool_calls=_tool_calls(message),
    )
