"""The gates — knowing when not to answer (docs/06-guardrails.md).

Three of the four gates run in v1. Gate 4 (entailment) only applies to
LLM-generated answers, which arrive with Tier 3.

    transcript ─► Gate 1 ─► retrieve ─► Gate 2 ─► extract ─► Gate 3 ─► user
                  input                 retrieval            grounding

Gate 2 is the one that does the real work, because it is the one the labelled
data scores: ~39% of MSMARCO-XI rows have no gold passage, so abstention
precision and recall are measurable rather than asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.core.config import Settings
from src.rag.chunk import normalise
from src.rag.languages import display_name, to_flores
from src.rag.store import Hit

Status = Literal["answered", "abstained", "refused"]

# Deliberately short. This is an information-retrieval system over a public web
# corpus; over-blocking ordinary queries is itself a failure, so the list only
# covers instruction-seeking for direct physical harm. The false-positive rate
# on the eval sample is a number we report, not a detail we assume away.
_UNSAFE = re.compile(
    r"\b(how (to|do i) (make|build|synthesi[sz]e) (a )?(bomb|explosive|nerve agent|meth)"
    r"|kill (myself|yourself)|child (porn|sexual))",
    re.IGNORECASE,
)

# Spoken prompt injection: it has to survive being said out loud and
# transcribed, which makes it a meaningfully harder attack than typed
# injection. Strip the imperative, keep the rest of the question, flag it.
_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions?"
    r"|disregard (your|all) (instructions?|rules?|prompts?)"
    r"|you are now\b|forget (your|all) (instructions?|rules?)"
    r"|system prompt|reveal your (prompt|instructions?))",
    re.IGNORECASE,
)


@dataclass(slots=True)
class InputVerdict:
    """Gate 1's decision, plus the sanitised query the pipeline should use."""

    status: Status
    query: str
    language: str | None
    reason: str | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.status != "answered"


@dataclass(slots=True)
class RetrievalVerdict:
    ok: bool
    confidence: float
    top_score: float
    margin: float
    reason: str | None = None


def gate_input(
    transcript: str,
    language_code: str | None,
    settings: Settings,
    indexed_languages: list[str],
) -> InputVerdict:
    """Cheap, lexical, no model. Runs before anything expensive."""
    query = normalise(transcript)
    flags: list[str] = []

    if len(query) < settings.min_transcript_chars:
        return InputVerdict(
            status="refused",
            query=query,
            language=None,
            reason="I didn't catch a question there.",
            flags=["empty"],
        )

    if _UNSAFE.search(query):
        return InputVerdict(
            status="refused",
            query=query,
            language=None,
            reason="I can't help with that.",
            flags=["unsafe"],
        )

    if _INJECTION.search(query):
        # Continue on the sanitised text rather than refusing: the underlying
        # question is usually legitimate, and the pipeline is extractive anyway.
        query = normalise(_INJECTION.sub(" ", query))
        flags.append("injection")

    language = to_flores(language_code)

    if language_code and language is None:
        return InputVerdict(
            status="abstained",
            query=query,
            language=None,
            reason=f"I don't have an index for {language_code} yet.",
            flags=[*flags, "unsupported-language"],
        )

    if language and indexed_languages and language not in indexed_languages:
        return InputVerdict(
            status="abstained",
            query=query,
            language=language,
            reason=(
                f"I heard {display_name(language)}, but my sources are only indexed in "
                + ", ".join(display_name(code) for code in indexed_languages)
                + "."
            ),
            flags=[*flags, "unsupported-language"],
        )

    return InputVerdict(status="answered", query=query, language=language, flags=flags)


def gate_retrieval(hits: list[Hit], settings: Settings) -> RetrievalVerdict:
    """Score floor + margin + minimum hit count.

    The margin test matters as much as the floor. Ten results all at 0.61 mean
    the corpus has nothing; one at 0.79 against a field of 0.55 usually means it
    does. An absolute floor alone cannot tell those apart — and the hard case
    here is exactly that, because an unanswerable query's own passages are in
    the index and are topically adjacent to it.
    """
    if not hits:
        return RetrievalVerdict(
            ok=False,
            confidence=0.0,
            top_score=0.0,
            margin=0.0,
            reason="I don't have anything on that in my sources.",
        )

    top = hits[0].score
    rest = [hit.score for hit in hits[1:5]]
    margin = top - (sum(rest) / len(rest)) if rest else settings.retrieval_margin
    above_floor = sum(1 for hit in hits if hit.score >= settings.retrieval_floor)

    span = max(1e-6, 1.0 - settings.retrieval_floor)
    score_term = min(1.0, max(0.0, (top - settings.retrieval_floor) / span))
    margin_term = min(1.0, max(0.0, margin / max(1e-6, settings.retrieval_margin)))
    confidence = round(0.65 * score_term + 0.35 * margin_term, 3)

    if top < settings.retrieval_floor:
        return RetrievalVerdict(
            ok=False,
            confidence=confidence,
            top_score=top,
            margin=margin,
            reason="I don't have that in my sources.",
        )

    if above_floor < settings.min_hits:
        return RetrievalVerdict(
            ok=False,
            confidence=confidence,
            top_score=top,
            margin=margin,
            reason="The evidence I found is too thin to answer from.",
        )

    if margin < settings.retrieval_margin:
        return RetrievalVerdict(
            ok=False,
            confidence=confidence,
            top_score=top,
            margin=margin,
            reason="Nothing in my sources stands out as the answer to that.",
        )

    return RetrievalVerdict(ok=True, confidence=confidence, top_score=top, margin=margin)


def gate_grounding(answer: str, context: str, citations: list) -> str | None:
    """Verify by construction: the span must be lifted from the context.

    Not a similarity score — a substring check. If it fails, the extractor has a
    bug and the system abstains rather than emitting text of unknown provenance.
    Every answer also carries a citation; no citation, no answer.
    """
    if not answer.strip():
        return "I couldn't pull a clear answer out of what I found."

    if normalise(answer) not in normalise(context):
        return "I couldn't verify that answer against my sources."

    if not citations:
        return "I found an answer but couldn't cite where it came from."

    return None
