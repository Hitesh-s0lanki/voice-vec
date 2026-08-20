"""Extractive answering — the reason Tier 1 can be fast *and* grounded.

MS MARCO answers are largely lifted from the selected passage, and that
property survives translation (docs/01-dataset.md). So the answer is a span cut
out of a retrieved chunk, never generated text: the hallucination rate at this
tier is structurally zero, not merely low.

The span is chosen in two passes because embedding is the expensive part:

  1. a lexical prefilter over every candidate sentence — microseconds
  2. an embedding rerank of the survivors only

Measured on this machine, one sentence costs ~4.5 ms to embed and the cost is
linear in sentence count — batching buys nothing. Embedding all ~30 sentences
of the top-3 passages would spend ~135 ms of a 200 ms budget, so the prefilter
is what makes the second pass affordable. When the deadline is already tight
the second pass is skipped and the lexical ranking stands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.config import Settings
from src.rag.chunk import Sentence, split_sentences, tokens
from src.rag.embed import Embedder
from src.rag.store import Hit

# Rough per-sentence embedding cost, used only to decide whether the rerank
# fits in the remaining budget. Measured, not guessed — see the module docstring.
EMBED_MS_PER_WINDOW = 5.0


@dataclass(slots=True)
class Extraction:
    answer: str
    chunk_id: str
    score: float
    method: str
    sentences: int
    hit: Hit


@dataclass(slots=True)
class _Window:
    hit: Hit
    text: str
    lexical: float
    sentences: int


def _lexical_score(query_tokens: set[str], sentence: str) -> float:
    """Share of the query's words present in the sentence, length-damped.

    Crude on purpose: it only has to keep the right sentence inside the top few,
    not rank them. The embedding pass does the ranking.
    """
    sentence_tokens = tokens(sentence)
    if not sentence_tokens or not query_tokens:
        return 0.0

    overlap = sum(1 for token in sentence_tokens if token in query_tokens)
    return overlap / (len(query_tokens) ** 0.5 * (1 + len(sentence_tokens) ** 0.25))


def _windows(hit: Hit, spans: list[Sentence], query_tokens: set[str], settings: Settings) -> list[_Window]:
    """Single sentences, plus a two-sentence variant for the strongest one.

    Slicing the parent text by offsets keeps the window verbatim, which is what
    lets Gate 3 be a substring check.
    """
    out: list[_Window] = []

    for index, span in enumerate(spans):
        lexical = _lexical_score(query_tokens, span.text)
        out.append(_Window(hit, span.text[: settings.extract_max_chars], lexical, 1))

        if settings.extract_max_sentences > 1 and index + 1 < len(spans):
            joined = hit.text[span.start : spans[index + 1].end]
            out.append(
                _Window(hit, joined[: settings.extract_max_chars], lexical, 2)
            )

    return out


def extract_span(
    *,
    query: str,
    query_vector: np.ndarray,
    hits: list[Hit],
    embedder: Embedder,
    settings: Settings,
    budget_ms: float,
) -> Extraction | None:
    """Pick the best answer span across the top retrieved chunks."""
    if not hits:
        return None

    query_tokens = set(tokens(query))
    candidates: list[_Window] = []

    for hit in hits[: settings.extract_passages]:
        spans = split_sentences(hit.text)
        if spans:
            candidates.extend(_windows(hit, spans, query_tokens, settings))

    if not candidates:
        return None

    # Retrieval score breaks ties, so sentences from a passage that matched well
    # still make the shortlist when no single sentence shares the query's words —
    # which is the normal case for a paraphrase.
    candidates.sort(key=lambda w: (w.lexical, w.hit.score), reverse=True)
    shortlist = candidates[: settings.extract_rerank]

    estimate_ms = len(shortlist) * EMBED_MS_PER_WINDOW
    if estimate_ms > budget_ms:
        best = shortlist[0]
        return Extraction(
            answer=best.text,
            chunk_id=best.hit.chunk_id,
            score=best.hit.score,
            method="lexical",
            sentences=best.sentences,
            hit=best.hit,
        )

    vectors = embedder.embed_passages([w.text for w in shortlist], batch_size=len(shortlist))
    # Both sides are L2-normalised, so the dot product is the cosine.
    scores = vectors @ query_vector
    winner = int(np.argmax(scores))
    best = shortlist[winner]

    return Extraction(
        answer=best.text,
        chunk_id=best.hit.chunk_id,
        score=float(scores[winner]),
        method="embedding",
        sentences=best.sentences,
        hit=best.hit,
    )
