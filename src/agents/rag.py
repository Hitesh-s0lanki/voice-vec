"""The ask ladder's model-driven stages, one agent each.

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
`AnswerGrader` returns both bits separately because an answer that is not
supported by the context needs regeneration from the same context, while an
answer that is supported but does not address the question needs a new query
and a new retrieval. Diagnosing them as one "bad answer" signal means half the
repairs attack the wrong problem.

**None of these is a tool loop, and none of them should become one.** Each is
one decision from one model call — `prompt | model | parser`, LangChain's
simplest composition. An agent loop here would be a stage that can spend an
unbounded number of round trips inside a rung whose budget is stated in
seconds, to answer a question that has no tool to call. The one real agent in
this package is `DatasetAgent`, which has something to call.

Five stages, five classes, one bundle. `RagAgents` exists so `AskService` holds
one collaborator rather than five nearly identical ones, and so the stages are
constructed once with the deployment's settings — prompt files read, clients
built — instead of at every call inside a request that already has a deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.agents.base import ModelAgent
from src.core.config import Settings
from src.rag.store import Hit

#: What the synthesiser is told to say when the context does not answer the
#: question. A sentinel rather than a judgement call about the phrasing of a
#: refusal, so the caller can turn it into a real abstention with the reason
#: text the rest of the pipeline uses. It is written out in
#: `src/prompts/synthesis.md`, and `tests/test_prompts.py` keeps the two the same.
NO_ANSWER = "NO_ANSWER"


@dataclass(slots=True)
class Relevance:
    """Corrective RAG's grade over the whole retrieval, not one document.

    The distinction matters and is where the reference implementation goes
    wrong: grading document-by-document and firing the expensive repair path
    whenever *any* one fails means it fires on nearly every query, because a
    top-10 nearly always contains a weak result. The paper grades confidence
    over the retrieval as a whole into correct / ambiguous / incorrect, and
    that is what the repair triggers on.
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

    A module function rather than a method: it is a formatter three of the
    agents below share, it makes no model call, and hanging it off a class
    would make it look like a decision.
    """
    lines = []
    for hit in hits[: settings.synthesis_context_passages]:
        text = hit.rendering(english=english)[: settings.extract_max_chars]
        lines.append(f"[{hit.chunk_id}] {text}")
    return "\n\n".join(lines)


class SynthesisAgent(ModelAgent):
    """Rung 2's writer: one grounded answer over the retrieved passages."""

    name = "synthesis"

    @property
    def _max_tokens(self) -> int:
        return self._settings.synthesis_max_tokens

    @property
    def _temperature(self) -> float:
        return self._settings.synthesis_temperature

    @property
    def _timeout_s(self) -> float:
        return self._settings.ask_llm_timeout_s

    def write(self, *, query: str, hits: Sequence[Hit], english: bool = False) -> str | None:
        """The answer, or None.

        None means the model said the context does not answer the question —
        which is a *correct* outcome and becomes an abstention upstream, not an
        error. Gate 4 still checks whatever comes back: a model instructed to
        stay inside the context is not the same thing as a model that did.
        """
        context = context_block(hits, self._settings, english=english)
        if not context.strip():
            return None

        answer = self._text(query=query, context=context)
        if not answer or NO_ANSWER in answer.upper():
            return None
        return answer


class RelevanceGrader(ModelAgent):
    """Corrective RAG's grader: which retrieved passages bear on the question."""

    name = "relevance-grader"

    def grade(
        self, *, query: str, hits: Sequence[Hit], english: bool = False
    ) -> Relevance | None:
        context = context_block(hits, self._settings, english=english)
        if not context.strip():
            return None

        parsed = self._json(query=query, context=context)
        if parsed is None:
            return None

        known = {hit.chunk_id for hit in hits}
        keep = [str(k) for k in (parsed.get("keep") or []) if str(k) in known]
        verdict = str(parsed.get("verdict") or "ambiguous").lower()
        if verdict not in {"correct", "ambiguous", "incorrect"}:
            verdict = "ambiguous"

        return Relevance(keep=keep, verdict=verdict)


class QueryRewriter(ModelAgent):
    """A second attempt at the search key, for when the first retrieval failed.

    Rewriting as a **repair**, never as a default pre-processing step: a rewrite
    on the happy path is a round trip in front of every question, which is what
    puts query enhancement outside this system's budget entirely. It only runs
    after retrieval has already been graded as bad, so it is paid for by a query
    that was going to be abstained on anyway.
    """

    name = "query-rewriter"

    #: A search key, not a paragraph. Anything longer is the model explaining
    #: itself, which is rejected below.
    @property
    def _max_tokens(self) -> int:
        return 120

    def rewrite(self, *, query: str) -> str | None:
        rewritten = self._text(query=query)
        if rewritten is None:
            return None

        rewritten = rewritten.strip().strip('"').strip()
        # A "rewrite" that came back identical, empty, or as a paragraph is not
        # a second attempt at anything — spending another retrieval on it just
        # pays the same latency for the same result.
        if not rewritten or rewritten.lower() == query.strip().lower() or len(rewritten) > 400:
            return None
        return rewritten


class AnswerGrader(ModelAgent):
    """Gate 4's model half: is the answer supported, and does it answer?"""

    name = "answer-grader"

    def grade(
        self, *, query: str, answer: str, hits: Sequence[Hit], english: bool = False
    ) -> Verdict | None:
        """Two independent bits from one call.

        They could be two calls with tighter prompts, and the paper's
        formulation uses separate graders — but each is a full round trip on a
        rung that already makes several, and the two questions are answerable
        from exactly the same material.
        """
        context = context_block(hits, self._settings, english=english)
        parsed = self._json(query=query, context=context, answer=answer)
        if parsed is None:
            return None

        return Verdict(
            supported=bool(parsed.get("supported")),
            useful=bool(parsed.get("useful")),
        )


class RouterAgent(ModelAgent):
    """Adaptive RAG's router: decide before retrieving, not after.

    Only two destinations exist here, and that is deliberate. The reference
    architecture routes between the vector store and a web search; this system
    has no web search on the answer path, so offering one would be a branch
    that cannot be taken. What it does have is a *lot* of retrievals that were
    never going to help — "hello", "say that again", "what did I just ask" —
    and skipping retrieval for those is where the routing actually pays.
    """

    name = "router"

    def route(self, *, query: str, corpus: str) -> Route | None:
        parsed = self._json(query=query, corpus=corpus)
        if parsed is None:
            return None

        destination = str(parsed.get("destination") or "vectorstore").lower()
        if destination not in {"vectorstore", "direct"}:
            destination = "vectorstore"
        return Route(destination=destination, reason=str(parsed.get("reason") or "")[:120])


class RagAgents:
    """The five stages, built once, held by whatever runs the ladder.

    `ready` is the whole set's answer to "is there a model at all": every stage
    here reads the same configuration, so one flag is honest and five would be
    the same flag five times. `AskService` reads it nowhere — the harness
    already treats an unavailable stage as an optional one that returned None —
    but a health endpoint asking "does this deployment synthesise or only
    extract?" is asking exactly this.
    """

    def __init__(self, settings: Settings) -> None:
        self.synthesis = SynthesisAgent(settings)
        self.relevance = RelevanceGrader(settings)
        self.rewriter = QueryRewriter(settings)
        self.verdict = AnswerGrader(settings)
        self.router = RouterAgent(settings)

    @property
    def ready(self) -> bool:
        return self.synthesis.ready

    def __repr__(self) -> str:
        return f"<RagAgents ready={self.ready}>"
