"""A backend whose `capabilities()` is a measurement instead of a guess.

Every `VectorBackend` has answered this question by hard-coding it. Pinecone
returns `filters=True` for every index on earth and its own docstring concedes
the point: *"`filters` is a hope rather than a guarantee."* The hope is that the
connected index carries a `strategy` field. When it does not, the metadata
predicate matches everything, the effort ladder records a filtered search it did
not get, and nothing anywhere fails — the recall is simply worse than the logs
claim.

`src/connectors/` now measures that, by sampling the store and reporting what
share of records actually carry each field. This wraps a backend so the ladder
reads the measurement.

**A wrapper rather than an argument threaded through three backends.** The
three of them differ in how they talk to a service and not at all in how they
should answer this, so putting the merge in one place keeps a fix from landing
for pgvector and missing Astra. It also means a fourth backend gets measured
capabilities for free, without its author knowing this file exists.

**The measurement only ever removes a claim.** `merge` takes the falsy answer
of the two, so a backend that says "no lexical channel" is believed and one
that says "yes, filters" is believed only if the sample agrees. Profiling can
be off, stale, or missing entirely, and a store then behaves exactly as it did
before this existed — which is the property that makes it safe to layer on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.connectors.profile import CapabilityFacts
from src.rag.backends.base import Capabilities, Hit, VectorBackend


def merge(declared: Capabilities, facts: CapabilityFacts | None) -> Capabilities:
    """Declared capabilities, narrowed by what the sample actually found.

    `and` in both directions on purpose. A backend that knows it cannot do
    something is right — it knows its own protocol — and a sample that did not
    find the field the backend hoped for is also right. The only combination
    that yields a capability is both saying yes.

    `None` means nothing was measured, and the answer is the backend's own.
    Absence of a measurement is not evidence of absence, and treating it that
    way would silently switch every connected store to dense-only the first
    time profiling was turned off.
    """
    if facts is None:
        return declared
    return Capabilities(
        lexical=declared.lexical and facts.lexical,
        filters=declared.filters and facts.filters,
        parallel_text=declared.parallel_text and facts.parallel_text,
    )


class ProfiledBackend:
    """Delegates everything, and answers `capabilities()` from the profile.

    Explicit delegation rather than `__getattr__`: `VectorBackend` is a
    `runtime_checkable` Protocol, so a wrapper that forwards attributes
    dynamically satisfies `isinstance` while quietly failing any call the
    Protocol grows later. Four methods written out cannot drift silently.
    """

    __slots__ = ("_inner", "_facts")

    def __init__(self, inner: VectorBackend, facts: CapabilityFacts | None) -> None:
        self._inner = inner
        self._facts = facts

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def inner(self) -> VectorBackend:
        """The wrapped backend, for the resolver's own bookkeeping (`close`)."""
        return self._inner

    def describe(self) -> str:
        return self._inner.describe()

    def ready(self) -> bool:
        return self._inner.ready()

    def capabilities(self) -> Capabilities:
        return merge(self._inner.capabilities(), self._facts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._inner.embed_query(text)

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        return self._inner.search(
            vector, strategies=strategies, limit=limit, language=language
        )

    def search_lexical(
        self,
        query: str,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        # Only reachable when `capabilities().lexical` is true, which now takes
        # the measurement into account — so a store whose `tsv` column the probe
        # did not find never gets here in the first place.
        return self._inner.search_lexical(
            query, strategies=strategies, limit=limit, language=language
        )

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()
