"""The one thing a vector store has to do for retrieval: answer a search.

`VectorStore` in `src/rag/store.py` grew up as the only store there was, so it
does ingest, schema, index building, warming and counting as well as searching.
Three of those are meaningless for a Pinecone index somebody else populated —
this app does not create their index and will not build their indexes — and one
of them, search, is the only thing the answer path calls.

So the protocol here is deliberately narrow. It is what `AskService` needs and
nothing more, which is what makes a connected backend implementable in a
hundred lines instead of reimplementing an ingest pipeline nobody asked for.

    search(vector, strategies, limit, language) -> list[Hit]
    ready()                                     -> bool
    describe()                                  -> str
    capabilities()                              -> Capabilities

`capabilities()` is the one addition the effort ladder needed. Rung 2 fuses a
lexical channel into the dense one, and only the app's own Postgres has a
lexical channel — a hosted Pinecone or Astra index is nearest-neighbour and
nothing else. Asking the backend rather than switching on its slug is what lets
the ladder degrade honestly: a connected index runs rung 2 dense-only and the
response says `dense-only` in `flags`, instead of the pipeline either erroring
or silently pretending it reranked something it did not.

`Hit` is imported from `src/rag/store.py` rather than redefined. Two Hit types
that had to be kept identical would drift the first time either was touched,
and `rendering()` — the cross-lingual fallback that decides what an answer is
cut out of — is behaviour every backend must share, not copy.

**Latency.** The in-process pgvector store measured ~11 ms; a hosted index is a
network round trip inside the same 200 ms as everything else
(docs/04-latency.md). A connected backend is a real trade and the deadline is
what enforces it: the harness degrades rather than overrunning.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.rag.store import Capabilities, Hit, StoreUnavailable

# `Capabilities` and `Hit` are imported from the store rather than defined
# here for the same reason: two copies that had to be kept identical would
# drift the first time either was touched.
__all__ = ["Capabilities", "Hit", "StoreUnavailable", "VectorBackend"]


@runtime_checkable
class VectorBackend(Protocol):
    """What retrieval needs from wherever the vectors happen to be."""

    @property
    def name(self) -> str:
        """The connector slug this backend speaks for, for logs and /health."""
        ...

    def describe(self) -> str:
        """Where this points, with no credential in it. Safe to log."""
        ...

    def ready(self) -> bool:
        """Whether a search would work right now. Never raises."""
        ...

    def capabilities(self) -> Capabilities:
        """What this store can do beyond dense search. Never raises."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Turn a question into a vector *this* store can be searched with.

        The backend owns this, rather than the pipeline embedding once and
        handing the vector down, because a connected index was built by
        whatever model its owner used and a 384-dimensional query cannot be
        compared to a 768-dimensional index. No column mapping or distance
        metric reconciles that after the fact; the only honest answer is to
        embed with the model that built the index.

        Every width goes through the one embedder, which asks
        `text-embedding-3` for exactly this store's number of dimensions. So
        this is one call whatever the store is — reached through the object
        that knows what width to ask for.
        """
        ...

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        """Nearest neighbours, best first. Raises `StoreUnavailable`."""
        ...
