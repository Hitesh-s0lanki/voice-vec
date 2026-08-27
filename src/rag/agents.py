"""The model-driven stages: synthesis, grading, rewriting, routing.

Everything in this file is a rung-2-and-up stage, and everything in it is a
network round trip. Two rules apply throughout, and they are what separate this
from the notebook implementations the design came from
(docs/agentic-rag/05-rag-architectures.md):

**A grader that cannot answer returns `None`, never a default.** An
unconfigured model, a timeout and an unparseable reply all produce `None`, and
every caller falls back to the deterministic guardrail it already had. The
tempting alternative — default to `relevant=True` on a parse failure — turns
every provider hiccup into an approval, silently, in the direction that emits
answers rather than withholding them.

**Ungrounded and off-target are different failures with different repairs.**
`grade_answer` returns both bits separately because an answer that is not
supported by the context needs regeneration from the same context, while an
answer that is supported but does not address the question needs a new query
and a new retrieval. Diagnosing them as one "bad answer" signal means half the
repairs attack the wrong problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from src.core.config import Settings
from src.rag import llm
from src.rag.store import Hit

log = logging.getLogger("vec.rag.agents")

#: What the synthesiser is told to say when the context does not answer the
#: question. A sentinel rather than a judgement call about the phrasing of a
#: refusal, so the caller can turn it into a real abstention with the reason
#: text the rest of the pipeline uses.
NO_ANSWER = "NO_ANSWER"


@dataclass(slots=True)
class Relevance:
    """Corrective RAG's grade over the whole retrieval, not one document.

    The distinction matters and is where the reference implementation goes
    wrong: grading document-by-document and firing the expensive repair path
    whenever *any* one fails means it fires on nearly every query, because a
    top-10 nearly always contains a weak result. The paper grades confidence
    over the retrieval as a whole into correct / ambiguous / incorrect, and
    reserves the escape hatch for the last.
    """

    keep: list[str] = field(default_factory=list)
    verdict: str = "ambiguous"

    @property
    def usable(self) -> bool:
        return self.verdict != "incorrect" and bool(self.keep)


@dataclass(slots=True)
class Verdict:
    """Adaptive RAG's three outcomes, as the two bits they actually are."""

    supported: bool
    useful: bool

    @property
    def ok(self) -> bool:
        return self.supported and self.useful


@dataclass(slots=True)
class Route:
    """Where a question should go, decided before anything is retrieved."""

    destination: str  # "vectorstore" | "direct" | "none"
    reason: str = ""


def context_block(hits: Sequence[Hit], settings: Settings, *, english: bool = False) -> str:
    """The retrieved passages, numbered so the model can cite them by id.

    Truncated per passage rather than in total: a single long passage would
    otherwise eat the whole context and silently drop the other four, which
    looks exactly like a retriever that only found one thing.
    """
    lines = []
    for hit in hits[: settings.synthesis_context_passages]:
        text = hit.rendering(english=english)[: settings.extract_max_chars]
        lines.append(f"[{hit.chunk_id}] {text}")
    return "\n\n".join(lines)


def synthesise(
    *,
    query: str,
    hits: Sequence[Hit],
    settings: Settings,
    english: bool = False,
) -> str | None:
    """One grounded answer over the retrieved passages, or None.

    None means the model said the context does not answer the question — which
    is a *correct* outcome and becomes an abstention upstream, not an error.
    Gate 4 still checks whatever comes back: a model instructed to stay inside
    the context is not the same thing as a model that did.
    """
    context = context_block(hits, settings, english=english)
    if not context.strip():
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You answer strictly from the passages given to you.\n"
                "Rules:\n"
                "- Use only what the passages say. Never add facts from your own knowledge.\n"
                "- Two or three sentences at most. Lead with the answer.\n"
                f"- If the passages do not answer the question, reply with exactly {NO_ANSWER}.\n"
                "- Plain prose. No markdown, no bullet points, no passage ids in the reply.\n"
                "- Answer in the same language the question was asked in."
            ),
        },
        {
            "role": "user",
            "content": f"Passages:\n{context}\n\nQuestion: {query}",
        },
    ]

    try:
        answer = llm.complete(
            messages,
            settings=settings,
            max_tokens=settings.synthesis_max_tokens,
            temperature=settings.synthesis_temperature,
            timeout_s=settings.ask_llm_timeout_s,
        )
    except llm.LlmUnavailable as error:
        log.info("synthesis unavailable: %s", error)
        return None

    if not answer or NO_ANSWER in answer.upper():
        return None
    return answer


def grade_relevance(
    *,
    query: str,
    hits: Sequence[Hit],
    settings: Settings,
    english: bool = False,
) -> Relevance | None:
    """Which retrieved passages actually bear on the question."""
    context = context_block(hits, settings, english=english)
    if not context.strip():
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You grade retrieved passages for relevance to a question.\n"
                "Return a bare JSON object and nothing else:\n"
                '{"keep": ["<id of each passage that helps answer the question>"], '
                '"verdict": "correct" | "ambiguous" | "incorrect"}\n'
                "verdict is about the retrieval as a whole: correct when it clearly "
                "contains the answer, incorrect when nothing here bears on the question, "
                "ambiguous otherwise. A passage on the same topic that does not answer "
                "the question does not belong in keep."
            ),
        },
        {"role": "user", "content": f"Question: {query}\n\nPassages:\n{context}"},
    ]

    parsed = llm.complete_json(
        messages,
        settings=settings,
        max_tokens=settings.grader_max_tokens,
        timeout_s=settings.grader_timeout_s,
    )
    if parsed is None:
        return None

    known = {hit.chunk_id for hit in hits}
    keep = [str(k) for k in (parsed.get("keep") or []) if str(k) in known]
    verdict = str(parsed.get("verdict") or "ambiguous").lower()
    if verdict not in {"correct", "ambiguous", "incorrect"}:
        verdict = "ambiguous"

    return Relevance(keep=keep, verdict=verdict)


def rewrite_query(*, query: str, settings: Settings) -> str | None:
    """A second attempt at the search key, for when the first retrieval failed.

    Rewriting as a **repair**, never as a default pre-processing step: a rewrite
    on the happy path is a round trip in front of every question, which is what
    puts query enhancement outside this system's budget entirely. Here it only
    runs after retrieval has already been graded as bad, so it is paid for by a
    query that was going to be abstained on anyway.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the question as a better search query for a passage index.\n"
                "Keep the user's language. Keep every proper noun, number and unit.\n"
                "Prefer the vocabulary a written article would use over the phrasing of speech.\n"
                "Return only the rewritten query, on one line, with no quotes or preamble."
            ),
        },
        {"role": "user", "content": query},
    ]

    try:
        rewritten = llm.complete(
            messages,
            settings=settings,
            max_tokens=120,
            temperature=0.0,
            timeout_s=settings.grader_timeout_s,
        )
    except llm.LlmUnavailable as error:
        log.info("rewrite unavailable: %s", error)
        return None

    rewritten = rewritten.strip().strip('"').strip()
    # A "rewrite" that came back identical, empty, or as a paragraph is not a
    # second attempt at anything — spending another retrieval on it just pays
    # the same latency for the same result.
    if not rewritten or rewritten.lower() == query.strip().lower() or len(rewritten) > 400:
        return None
    return rewritten


def grade_answer(
    *,
    query: str,
    answer: str,
    hits: Sequence[Hit],
    settings: Settings,
    english: bool = False,
) -> Verdict | None:
    """Is the answer supported by the context, and does it address the question?

    Two independent bits from one call. They could be two calls with tighter
    prompts, and the paper's formulation uses separate graders — but each is a
    full round trip on a rung that already makes several, and the two questions
    are answerable from exactly the same material.
    """
    context = context_block(hits, settings, english=english)
    messages = [
        {
            "role": "system",
            "content": (
                "You check an answer against the passages it was written from.\n"
                "Return a bare JSON object and nothing else:\n"
                '{"supported": true | false, "useful": true | false}\n'
                "supported: every claim in the answer is stated in the passages.\n"
                "useful: the answer actually addresses the question that was asked.\n"
                "These are independent. An answer can be faithful to the passages "
                "and still not answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nPassages:\n{context}\n\nAnswer: {answer}",
        },
    ]

    parsed = llm.complete_json(
        messages,
        settings=settings,
        max_tokens=settings.grader_max_tokens,
        timeout_s=settings.grader_timeout_s,
    )
    if parsed is None:
        return None

    return Verdict(
        supported=bool(parsed.get("supported")),
        useful=bool(parsed.get("useful")),
    )


def route(*, query: str, settings: Settings, corpus: str) -> Route | None:
    """Adaptive RAG's router: decide before retrieving, not after.

    Only two destinations exist here, and that is deliberate. The reference
    architecture routes between the vector store and a web search; this system
    has no web search on the answer path, so offering one would be a branch
    that cannot be taken. What it does have is a *lot* of retrievals that were
    never going to help — "hello", "say that again", "what did I just ask" —
    and skipping retrieval for those is where the routing actually pays.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You decide whether a question needs a document search.\n"
                "Return a bare JSON object and nothing else:\n"
                '{"destination": "vectorstore" | "direct", "reason": "<a few words>"}\n'
                "vectorstore: the question asks for a fact, definition, explanation or "
                "detail about the world — anything a document could answer.\n"
                "direct: greetings, thanks, small talk, questions about this conversation "
                "or about you, and requests to repeat or rephrase something already said.\n"
                "When in doubt, choose vectorstore. Retrieving unnecessarily costs a little "
                "time; skipping retrieval wrongly costs the answer."
            ),
        },
        {"role": "user", "content": f"Corpus: {corpus}\n\nQuestion: {query}"},
    ]

    parsed = llm.complete_json(
        messages,
        settings=settings,
        max_tokens=settings.grader_max_tokens,
        timeout_s=settings.grader_timeout_s,
    )
    if parsed is None:
        return None

    destination = str(parsed.get("destination") or "vectorstore").lower()
    if destination not in {"vectorstore", "direct"}:
        destination = "vectorstore"
    return Route(destination=destination, reason=str(parsed.get("reason") or "")[:120])
