"""Reordering what retrieval returned — rung 2's precision stage.

Retrieval is a funnel: the recall stage casts wide and cheap and must not lose
the right passage, the precision stage orders what survived. Raising `k` on a
single dense retriever improves the first and *degrades* the second, because
rank 18 is now in the context handed to the model.

Two things happen here, and they fix different problems.

**Scoring.** After hybrid fusion, the candidate list contains passages the
lexical channel found that the dense channel never scored — a name or a numeral
matched, and there is no cosine for them. Embedding every candidate gives the
whole list one comparable number, which fusion by rank deliberately cannot.

**Diversity (MMR).** The five chunking strategies of docs/03-chunking.md
overlap by construction, so the top five hits are frequently five renderings of
one passage. Handing those to a synthesiser is paying five slots for one fact
and crowding out the second fact the question needed. MMR trades a little
relevance for coverage:

    pick argmax  λ·sim(q, d) − (1−λ)·max sim(d, chosen)

**No cross-encoder here, and that is a measurement, not an oversight.** A
cross-encoder is the higher-quality reranker and the obvious thing to reach
for, but the ones small enough for this budget — the `ms-marco-MiniLM` family —
are English-only, and the index is Devanagari. The multilingual alternatives
(`bge-reranker-v2-m3` and relatives) are several hundred million parameters and
do not fit the latency this rung claims. So the bi-encoder already loaded does
the job, at a quality cost stated openly rather than a latency cost discovered
in production.
"""

from __future__ import annotations

import numpy as np

from src.core.config import Settings
from src.rag.embed import Embedder
from src.rag.store import Hit

# Passages are longer than the sentences `extract.py` embeds, so the per-item
# cost is higher: measured around 8 ms each on this machine. Only used to
# decide whether the stage fits in what is left of the rung's budget.
EMBED_MS_PER_PASSAGE = 8.0


def rerank(
    *,
    query_vector: np.ndarray,
    hits: list[Hit],
    embedder: Embedder,
    settings: Settings,
    budget_ms: float,
    english: bool = False,
) -> tuple[list[Hit], str]:
    """Rescore and diversify. Returns the kept hits and how they were chosen.

    The method string travels onto the response, because "reranked" and "fusion
    order, no budget to rerank" are different claims and reporting the second
    as the first is how a latency story stops being true.
    """
    if not hits:
        return [], "empty"

    candidates = hits[: settings.rerank_candidates]
    keep = max(1, settings.rerank_keep)

    estimate_ms = len(candidates) * EMBED_MS_PER_PASSAGE
    if estimate_ms > budget_ms:
        return candidates[:keep], "order"

    texts = [hit.rendering(english=english)[: settings.extract_max_chars] for hit in candidates]
    try:
        vectors = embedder.embed_passages(texts, batch_size=len(texts))
    except Exception:
        # A reranker that fails is a reranker that did not run. The fusion
        # order is still a real order, so the answer survives at lower quality
        # rather than being lost to a stage that is by definition optional.
        return candidates[:keep], "order"

    # Both sides are L2-normalised by the embedder, so a dot product is cosine.
    relevance = np.asarray(vectors) @ np.asarray(query_vector)
    chosen = _mmr(vectors, relevance, keep=keep, lam=settings.mmr_lambda)

    ranked = []
    for index in chosen:
        hit = candidates[index]
        # The rescored cosine replaces the channel-specific score, so
        # everything downstream — Gate 2's floor, the citation, the confidence
        # blend — keeps reading one comparable quantity.
        ranked.append(
            Hit(
                chunk_id=hit.chunk_id,
                strategy=hit.strategy,
                score=float(relevance[index]),
                text=hit.text,
                payload=hit.payload,
            )
        )
    return ranked, "embedding"


def _mmr(vectors: np.ndarray, relevance: np.ndarray, *, keep: int, lam: float) -> list[int]:
    """Maximal marginal relevance over pre-normalised vectors.

    `lam = 1` is pure relevance and reproduces a plain rerank exactly, which is
    the escape hatch if diversity ever measures worse than it reads.
    """
    matrix = np.asarray(vectors)
    remaining = list(range(len(matrix)))
    chosen: list[int] = []

    first = int(np.argmax(relevance))
    chosen.append(first)
    remaining.remove(first)

    while remaining and len(chosen) < keep:
        similarity = matrix[remaining] @ matrix[chosen].T
        redundancy = similarity.max(axis=1)
        scored = lam * relevance[remaining] - (1.0 - lam) * redundancy
        pick = remaining[int(np.argmax(scored))]
        chosen.append(pick)
        remaining.remove(pick)

    return chosen
