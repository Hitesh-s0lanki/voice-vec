"""The gates — knowing when not to answer (docs/06-guardrails.md).

All four gates run. Gates 1–3 are on every path; Gate 4 replaces Gate 3 from
rung 2 up, where the answer is generated rather than lifted.

    transcript ─► Gate 1 ─► retrieve ─► Gate 2 ─► extract  ─► Gate 3 ─► user
                  input                 retrieval  synthesis   Gate 4

Gate 3 and Gate 4 check the same property by different means, because the
answer is a different kind of object. An extracted span is verified *by
construction* — it is a substring of the passage, and anything else is a bug in
the extractor. Generated text can be perfectly faithful without sharing a
single sequence of characters with its source, so Gate 4 asks the weaker
question it can actually answer: is every sentence of this answer close, in the
embedding space, to something that was in the context?

That is a real check and a weaker one, and the difference is why rung 1 is the
tier whose hallucination rate is structurally zero rather than merely low.

Gate 2 is the one that does the real work, because it is the one the labelled
data scores: ~39% of MSMARCO-XI rows have no gold passage, so abstention
precision and recall are measurable rather than asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.core.config import Settings
from src.rag.chunk import normalise, split_sentences
from src.rag.embed import Embedder
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
    #: The question is in a language the index does not hold. Not a refusal —
    #: a routing decision. See the language block in `gate_input`.
    cross_lingual: bool = False

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

    # ---- language: a routing decision, not a refusal ---------------------
    #
    # Asking in English against a Hindi index used to abstain here, and that
    # was the gate contradicting the rest of the design. `multilingual-e5` was
    # chosen over a monolingual embedder precisely because it puts a question
    # and its translation in the same region of the same space, and a chunk
    # that carries its English original beside the indexed translation can be
    # read back in either. Both halves of a cross-lingual answer are already
    # here; refusing on the strength of a language *label* threw them away.
    #
    # `indexed_languages` is empty for every caller in the app now: a connected
    # store does not declare what languages it holds, and this app stopped
    # holding a corpus it could ask. So the mismatch branch fires only on a
    # language code that does not resolve at all.
    #
    # So a mismatch turns the language filter off and answers from the English
    # rendering, and the question of whether retrieval was actually good enough
    # goes where it belongs — Gate 2, where it is a measured score against a
    # swept floor rather than a tag comparison.
    language = to_flores(language_code)
    cross_lingual = bool(
        language_code and (language is None or (indexed_languages and language not in indexed_languages))
    )

    return InputVerdict(
        status="answered",
        query=query,
        language=language,
        flags=[*flags, "cross-lingual"] if cross_lingual else flags,
        cross_lingual=cross_lingual,
    )


def gate_retrieval(
    hits: list[Hit],
    settings: Settings,
    *,
    floor: float | None = None,
    margin_floor: float | None = None,
) -> RetrievalVerdict:
    """Score floor + margin + minimum hit count.

    The margin test matters as much as the floor. Ten results all at 0.61 mean
    the corpus has nothing; one at 0.79 against a field of 0.55 usually means it
    does. An absolute floor alone cannot tell those apart — and the hard case
    here is exactly that, because an unanswerable query's own passages are in
    the index and are topically adjacent to it.

    Both thresholds can be overridden, and a cross-lingual question overrides
    both. The query and the passage are in different languages there, and e5
    puts a translation pair close but not as close as a paraphrase — so the
    scores compress, and so do the gaps between them. Measured over 200
    questions asked in both languages: mean top 0.8942 → 0.8469, mean margin
    0.0239 → 0.0179. Holding that to thresholds swept on Hindi-against-Hindi
    abstains on retrieval that was in fact right, and the margin is where most
    of the damage is (coverage 52% → 34% at the same setting).
    """
    floor = settings.retrieval_floor if floor is None else floor
    margin_floor = settings.retrieval_margin if margin_floor is None else margin_floor
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
    margin = top - (sum(rest) / len(rest)) if rest else margin_floor
    above_floor = sum(1 for hit in hits if hit.score >= floor)

    span = max(1e-6, 1.0 - floor)
    score_term = min(1.0, max(0.0, (top - floor) / span))
    margin_term = min(1.0, max(0.0, margin / max(1e-6, margin_floor)))
    confidence = round(0.65 * score_term + 0.35 * margin_term, 3)

    if top < floor:
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

    if margin < margin_floor:
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


def gate_generation(
    answer: str,
    contexts: list[str],
    citations: list,
    *,
    embedder: Embedder,
    settings: Settings,
) -> str | None:
    """Gate 4 — is a generated answer supported by what was retrieved?

    Every sentence of the answer is scored against every sentence of the
    context, and a sentence is supported when its best match clears
    `generation_support_floor`. The answer passes when enough of its sentences
    are supported (`generation_support_ratio`).

    Per sentence rather than whole-answer, because the failure this exists to
    catch is local: a model handed four good passages writes three faithful
    sentences and one fluent invention, and a similarity score over the whole
    paragraph averages that invention away. Per sentence, it is the one that
    fails.

    Local, so it costs milliseconds and not a round trip — the embedder is
    already loaded and the context is short. A model-based entailment check
    would be better and would mean asking an LLM to mark its own homework at
    the cost of another call on a rung that already made several.

    Returns None when the answer stands, or the user-facing reason to abstain.
    """
    if not answer.strip():
        return "I couldn't get a clear answer out of what I found."

    if not citations:
        return "I found an answer but couldn't cite where it came from."

    claims = [span.text for span in split_sentences(answer)] or [answer]
    evidence: list[str] = []
    for context in contexts:
        spans = split_sentences(context)
        if spans:
            evidence.extend(span.text for span in spans)
        elif context.strip():
            evidence.append(context)

    if not evidence:
        return "I couldn't verify that answer against my sources."

    # One batch, not two. Cheaper, and it removes a whole class of failure:
    # two calls can only be compared if they landed in the same space, and
    # nothing in the signature guarantees that.
    both = [*claims, *evidence]
    try:
        vectors = np.asarray(embedder.embed_passages(both, batch_size=len(both)))
    except Exception:
        # The gate could not run. Refusing to answer because the *check*
        # failed would trade a possible hallucination for a certain
        # abstention on every request while the embedder is unwell — so this
        # falls through to the weaker guarantee the pipeline still has (the
        # synthesis prompt and the citations) and says so in the trace, rather
        # than either passing silently or failing everything.
        return None

    # Both halves are L2-normalised, so the matrix product is cosine.
    claim_vectors, evidence_vectors = vectors[: len(claims)], vectors[len(claims) :]
    support = (claim_vectors @ evidence_vectors.T).max(axis=1)
    supported = int((support >= settings.generation_support_floor).sum())

    if supported / len(claims) < settings.generation_support_ratio:
        return "I couldn't verify that answer against my sources."

    return None
