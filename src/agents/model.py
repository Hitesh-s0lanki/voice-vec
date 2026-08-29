"""One chat model, built from the same `resolve_llm()` everything else reads.

LangChain's `ChatOpenAI` talks to any OpenAI-compatible `/chat/completions`
endpoint, which is exactly the contract this system already relied on: OpenAI's
own API and Sarvam's are both compatible, and `Settings.resolve_llm()` is the
one place that decides which of them answers. So the model object is built from
an `LlmTarget` and nothing here knows which provider it got.

Three decisions worth stating, because each is a place LangChain's defaults are
wrong for this system:

**`max_retries=0`.** LangChain retries a failed call twice by default, which
turns a 6-second grader timeout into an 18-second one on a rung whose whole
budget is ~5 s. Every caller here already degrades on failure — a stage that
returns `None` falls back to the deterministic guardrail — so a retry buys a
slower version of the same answer. Retrying is the harness's decision
(`src/rag/harness.py`), not the client's.

**One model per (target, budget).** `ChatOpenAI` is a client, and building one
per call would build an `httpx` connection pool per call. They are cached on the
values that actually change the request, so a rung that runs three graders
reuses one pool and one set of sockets.

**Parameter quirks are `langchain-openai`'s problem now.** The hand-rolled
client learns them from the provider's error body and retries (`adapt_payload`
in `src/voice/llm.py`, still used by the streaming voice path). `ChatOpenAI`
already knows the common ones — it drops `temperature` for models that refuse
it, and sends `max_completion_tokens` where that is the name — and anything it
does not know surfaces as a failed stage that degrades, which is the same place
an unknown quirk landed before.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI

from src.core.config import LlmTarget, Settings
# The system's single "there is no model" error, raised by the hand-rolled
# client and caught by the narrators. Agents raise the same one so a caller
# that already handles it does not learn a second name for the same thing.
from src.rag.llm import LlmUnavailable

# A model asked for JSON returns JSON, a fence around JSON, or a sentence and
# then JSON. All three are common — the third especially on providers without
# a JSON mode — and only the first parses, so the object is cut out rather than
# trusted to be the whole reply.
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def chat_model(
    settings: Settings,
    *,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ChatOpenAI:
    """The model this deployment answers with, on this stage's budget.

    Raises `LlmUnavailable` when nothing is configured, rather than returning a
    client that fails on first use: the callers all check `ready` first, and the
    ones that do not are better off failing where the reason is legible.
    """
    target = settings.resolve_llm()
    if not target.ready:
        raise LlmUnavailable("no model is configured — set OPENAI_API_KEY or SARVAM_API_KEY")
    return _build(target, max_tokens, temperature, timeout_s)


@lru_cache(maxsize=32)
def _build(
    target: LlmTarget, max_tokens: int, temperature: float, timeout_s: float
) -> ChatOpenAI:
    return ChatOpenAI(
        model=target.model,
        base_url=target.base_url,
        api_key=target.api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_s,
        max_retries=0,
    )


class TolerantJson(JsonOutputParser):
    """`JsonOutputParser`, but it finds the object in a reply that has prose too.

    The stock parser accepts a bare object or a fenced one and rejects anything
    else, which is right for a model with a JSON mode and wrong here: the
    graders run against whichever provider is configured, `response_format` is
    deliberately not sent (OpenAI honours it, Sarvam's endpoint refuses the
    request outright), and "Sure — {...}" is a reply this system has to be able
    to read.

    What it does *not* do is repair. An unparseable reply raises, the agent
    turns that into `None`, and the caller falls back to the deterministic
    guardrail — because a grader that guesses is worse than no grader.
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        # `parse_result`, not `parse`: inside a chain the parser is handed the
        # generation, and an override on `parse` alone is never reached — a
        # tolerance that only works when called by hand is not tolerance.
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException:
            text = result[0].text if result else ""
            match = _OBJECT.search(text or "")
            if not match:
                raise
            return super().parse_result([Generation(text=match.group(0))], partial=partial)
