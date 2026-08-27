"""Combining two ranked lists into one — rung 2's hybrid channel.

Dense search scores cosine similarity; lexical search scores `ts_rank_cd`.
Those are different quantities on different scales, and averaging them is the
standard way to get a hybrid retriever that is worse than either half. The two
numbers do not even agree on what "high" means: cosine over e5 is compressed
into roughly 0.75–0.95, while `ts_rank_cd` is a long tail near zero.

**Reciprocal rank fusion** throws the scores away and fuses the *positions*,
which is the only thing the two lists genuinely share:

    score(d) = Σ  1 / (k + rank(d, list))

A document near the top of both lists beats one that is first in a single list,
and `k` decides how strongly. That property is the whole point here: dense
retrieval finds paraphrases and misses names and numerals, lexical does the
opposite, and a passage both channels like is the one worth answering from.

One consequence worth stating because it looks like a bug: **the fused `Hit`
keeps its dense score**, not its fusion score. Gate 2's floor and margin were
swept on cosine (`src/core/config.py`), so handing them an RRF score — which
lives around 0.03 — would abstain on everything. Fusion decides the *order*;
the guardrail still reads the cosine it was calibrated against.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.rag.store import Hit


def rrf(
    lists: Sequence[Sequence[Hit]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[Hit]:
    """Fuse ranked lists by reciprocal rank. Best first.

    A document is identified by `chunk_id`, and the `Hit` kept for it is the
    one from the **first** list that contained it — dense, by convention of the
    call site, so the surviving `score` is a cosine. Empty lists contribute
    nothing rather than shifting anyone's rank.
    """
    scores: dict[str, float] = {}
    kept: dict[str, Hit] = {}

    for ranked in lists:
        for position, hit in enumerate(ranked):
            key = hit.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            kept.setdefault(key, hit)

    ordered = sorted(kept.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    return ordered[:limit] if limit is not None else ordered


def dedupe(hits: Iterable[Hit]) -> list[Hit]:
    """First occurrence of each `chunk_id`, order preserved.

    Two chunking strategies over the same passage produce two rows with
    different keys and the same text; that is a chunking concern and not this
    one. This only removes the literal duplicate a re-retrieval creates when
    round two finds what round one already had.
    """
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        out.append(hit)
    return out
