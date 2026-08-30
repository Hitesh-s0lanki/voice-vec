"""Embedding, through OpenAI. One door, one cache, one width.

There was a local ONNX model here — `intfloat/multilingual-e5-small`, 384
dimensions, ~4 ms a query and no network call at all. It is gone, and
docs/25-no-local-embedder.md is why: the session held ~700 MiB resident, which
is not a trade a 512 MiB deployment can make. What replaces it is
`text-embedding-3` over `src/rag/remote_embed.py`.

Everything that embeds goes through this object, so there is exactly one place
that holds the client, the width and the cache:

    embed_query(text)              the app's own width — the question, a need
    embed_query(text, dim=768)     a connected store's width, for its backend
    embed_passages(texts)          one call for the whole batch

**Two properties the callers already depend on**, both now this module's job to
keep rather than fastembed's. Vectors come back **L2-normalised**, because
every caller takes a dot product and calls it a cosine. And a batch comes back
**in the order it was given**, which `remote_embed` asserts rather than trusts.

**The e5 prefixes are gone with the model that needed them.** `query: ` and
`passage: ` were an e5 rule; prepending them to a `text-embedding-3` input is
not a no-op, it is six characters of unrelated content in every vector.

**What used to be free is now a round trip**, and that is the whole character
of this change. `count_tokens` went with the tokeniser: it had no callers left
once the ingest pipeline was removed, and the alternative — a network call to
count tokens — would have been absurd.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from src.core.config import Settings, get_settings
from src.rag.remote_embed import MAX_DIM, RemoteEmbedUnavailable, embed_texts, model_for

log = logging.getLogger("vec.rag.embed")


class Embedder:
    """`text-embedding-3` at this app's width, with a small query cache."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._settings.embed_dim

    @property
    def model_name(self) -> str:
        """The model a query at this app's own width actually reaches."""
        try:
            return model_for(self._settings.embed_dim)
        except RemoteEmbedUnavailable:
            return f"unsupported width {self._settings.embed_dim} (max {MAX_DIM})"

    @property
    def ready(self) -> bool:
        """Whether embedding can happen at all.

        A key, and a width something can produce. Both are configuration, so
        this answers without a network call — which is what lets `/health` and
        the capability index ask it on a path where nobody can wait.

        It is emphatically not a promise the key *works*. That is only knowable
        by spending a call, and the callers are already built to survive one
        failing: rerank falls back to fusion order, extraction to lexical, the
        grounding gate to not running, capability discovery to word overlap,
        and a connected store to an honest "my sources are unavailable".
        """
        if not self._settings.openai_api_key:
            return False
        try:
            model_for(self._settings.embed_dim)
        except RemoteEmbedUnavailable:
            return False
        return True

    # ---- the cache ------------------------------------------------------
    #
    # Only queries, never passages. A question repeats — across a turn's
    # discovery and its search, across the two turns somebody spends rephrasing
    # — and repeats are exactly what a round trip should not be paid for twice.
    # Retrieved passages do not repeat within a turn and caching them would
    # hold somebody's documents in memory for no benefit.

    def _cached(self, key: tuple[int, str]) -> np.ndarray | None:
        with self._lock:
            vector = self._cache.get(key)
            if vector is not None:
                self._cache.move_to_end(key)
            return vector

    def _remember(self, key: tuple[int, str], vector: np.ndarray) -> None:
        limit = max(0, self._settings.embed_cache_size)
        if not limit:
            return
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > limit:
                self._cache.popitem(last=False)

    # ---- embedding ------------------------------------------------------

    def embed_query(self, text: str, dim: int | None = None) -> np.ndarray:
        """One vector, at this app's width or at a store's.

        `dim` is what lets a backend reach its own width through the same
        client and the same cache instead of holding a second one. A connected
        index built at 768 cannot be searched with a vector of any other size,
        and no column mapping reconciles that after the fact.
        """
        width = dim or self._settings.embed_dim
        key = (width, text)

        cached = self._cached(key)
        if cached is not None:
            return cached

        vector = embed_texts([text], width, settings=self._settings)[0]
        self._remember(key, vector)
        return vector

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """One call for the whole batch.

        `batch_size` is accepted and ignored — it was the ONNX forward-pass
        batch, and over HTTP the shape that matters is one request rather than
        many. The signature stays because four callers pass it, and changing
        them to say nothing would be a bigger diff than honouring a parameter
        that no longer has anything to tune.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return embed_texts(texts, self._settings.embed_dim, settings=self._settings)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings())
