"""Which vector store answers for which user.

A user who has connected Pinecone, Astra or their own Postgres is searched
against that. Everybody else — signed out, or signed in with nothing connected
— gets `None`, and retrieval abstains. There is no deployment corpus behind
this app any more: what a question can be answered from is exactly what its
asker attached (docs/13-connectors.md).

Two rules that are easy to get wrong and expensive to get wrong:

**Return nothing, never somebody else's.** A user's backend is built from
credentials read under their own id, and when that fails the answer is `None`
— never another user's store. Guaranteed by construction here, because the
only per-user thing in this module is the credential lookup and it takes a
user id, and because there is no shared store left to reach for.

**Connecting two vector stores is not an error.** Somebody can have Pinecone
and pgvector attached at once; `PREFERENCE` decides, in a fixed order, so the
same user gets the same store on every request rather than whichever row came
back first.

**A backend that cannot answer is not a backend.** `verify` runs when the form
is submitted, and an index can be emptied, dropped or renamed long after that.
So a freshly built backend is probed once — `ready()` — and one that says no is
cached as a miss, which reads as nothing connected. Reconnecting re-seals the
credentials under a new blob, so the probe runs again the moment somebody fixes
what was wrong.

Backends are cached per user because building one is not free — a pgvector
backend opens a connection pool — and invalidated by the same trick the
Composio clients use: the cache remembers the sealed credential blob it was
built from, so rotating a key rebuilds rather than serving a revoked one.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Protocol

from src.connectors.profile_service import ProfileService, get_profile_service
from src.connectors.registry import vector_slugs
from src.connectors.service import ConnectorService, get_connector_service
from src.connectors.store import ConnectorStore, get_connector_store
from src.core.config import Settings, get_settings
from src.rag.backends.astra import AstraBackend
from src.rag.backends.base import VectorBackend
from src.rag.backends.pgvector import PgVectorBackend
from src.rag.backends.pinecone import PineconeBackend
from src.rag.backends.profiled import ProfiledBackend

log = logging.getLogger("vec.rag")

# Which one wins when somebody has attached more than one. Hosted stores first
# because a user who went to the trouble of connecting Pinecone meant it; the
# user's own Postgres last because it is also the most likely thing to have
# been connected "to look at" rather than to answer from.
PREFERENCE = ("pinecone", "astra", "pgvector")

_BUILDERS = {
    "pinecone": PineconeBackend,
    "astra": AstraBackend,
    "pgvector": PgVectorBackend,
}


class _Closeable(Protocol):
    def close(self) -> None: ...


class BackendResolver:
    def __init__(
        self,
        connectors: ConnectorService,
        store: ConnectorStore,
        settings: Settings,
        profiles: ProfileService | None = None,
    ) -> None:
        self._connectors = connectors
        self._store = store
        self._settings = settings
        # Optional, and read through `_facts` so it stays optional at runtime
        # too: a resolver built without one behaves exactly as it did before
        # profiling existed.
        self._profiles = profiles
        # user_id → (slug, sealed blob it was built from, backend or None)
        # A cached `None` is a backend that built but could not answer: the
        # probe is worth one round trip per connect, not one per question.
        self._cache: OrderedDict[
            str, tuple[str, str, VectorBackend | None]
        ] = OrderedDict()

    def for_user(self, user_id: str | None, *, prefer: str | None = None) -> VectorBackend | None:
        """The store that should answer this user's question, or None.

        Never raises. `None` is the honest answer for a caller with nothing
        attached, for one signed out, and for a connector that cannot be built
        or cannot answer — all four are "there is nowhere to look", and the
        ladder turns that into an abstention that says so rather than a 500.

        `prefer` names one of *this user's* connectors and is how a question
        reaches the right store when several are attached: capability discovery
        decides that the students question belongs to their pgvector, and this
        is where that decision is honoured (docs/23-capabilities.md). It is a
        preference and not an assertion — a slug this user has not connected
        falls back to the standing order rather than failing, because the name
        came from a model.
        """
        if not user_id:
            return None

        try:
            return self._connected(user_id, prefer)
        except Exception as error:
            log.warning("could not resolve a backend for %s: %s", user_id, error)
            return None

    def _connected(self, user_id: str, prefer: str | None = None) -> VectorBackend | None:
        rows = {
            row.connector: row
            for row in self._store.list(user_id)
            if row.connector in vector_slugs()
        }
        if not rows:
            self._evict(user_id)
            return None

        slug = prefer if prefer in rows else next((s for s in PREFERENCE if s in rows), None)
        if slug is None:
            return None
        row = rows[slug]

        cached = self._cache.get(user_id)
        if cached and cached[0] == slug and cached[1] == row.credentials:
            self._cache.move_to_end(user_id)
            return cached[2]

        credentials = self._connectors.credentials(user_id, slug)
        if not credentials:
            # Sealed under a rotated master key. Not connected, as far as
            # anything that needs to use it is concerned.
            self._evict(user_id)
            return None

        backend = _BUILDERS[slug](credentials)
        # Wrapped before the probe so that everything downstream — including
        # the `close()` on the failure path below — deals in one object. The
        # wrapper only overrides `capabilities()`; `ready()` still reaches the
        # real backend.
        backend = ProfiledBackend(backend, self._facts(user_id, slug))
        usable: VectorBackend | None = backend
        if not backend.ready():
            # Built fine, answers nothing: an index that was emptied, dropped,
            # or renamed since it was connected. Building is not the test that
            # matters — this is the failure `verify` cannot catch, because it
            # happens after the form was green. Treating it as nothing attached
            # is what turns it into "no sources are connected" rather than an
            # error from a layer the asker cannot place.
            log.warning("connected %s for %s is not searchable", slug, user_id)
            self._close(backend)
            usable = None

        self._evict(user_id)
        self._cache[user_id] = (slug, row.credentials, usable)
        self._cache.move_to_end(user_id)

        while len(self._cache) > self._settings.vector_backend_cache:
            _, (_, _, evicted) = self._cache.popitem(last=False)
            self._close(evicted)

        return usable

    def _facts(self, user_id: str, slug: str):
        """What the profile measured about this store, or None if nothing did.

        Never blocks and never raises. A missing profile is the common case on
        the first request after connecting — the probe is still running — and
        the right answer for that request is the backend's own declared
        capabilities, not a stall while somebody else's index is sampled.
        """
        if self._profiles is None:
            return None
        try:
            return self._profiles.facts(user_id, slug)
        except Exception as error:
            log.debug("could not read the %s profile for %s: %s", slug, user_id, error)
            return None

    def _evict(self, user_id: str) -> None:
        """Drop a cached backend, returning any connections it holds."""
        existing = self._cache.pop(user_id, None)
        if existing is not None:
            self._close(existing[2])

    @staticmethod
    def _close(backend: object) -> None:
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as error:  # a pool already gone, a dead socket
                log.debug("closing a backend failed: %s", error)


class FixedResolver:
    """Always this store, whoever is asking.

    For the scripts and tests that already hold the backend they want searched.
    `BackendResolver` needs the connector store and a database behind it, which
    an offline sweep over one known index has no business requiring — and a
    resolver that could return somebody's connected Pinecone mid-sweep would
    make the numbers unattributable anyway.
    """

    def __init__(self, backend: VectorBackend) -> None:
        self._backend = backend

    def for_user(
        self, user_id: str | None = None, *, prefer: str | None = None
    ) -> VectorBackend:
        """The one backend, whoever asks and whichever store they asked for.

        `prefer` is accepted and ignored on purpose. It is a preference
        everywhere (`BackendResolver.for_user`), so a resolver holding exactly
        one store honours it by having nothing else to offer — and taking the
        argument is what keeps the two resolvers substitutable.
        """
        return self._backend


@lru_cache
def get_resolver() -> BackendResolver:
    return BackendResolver(
        get_connector_service(),
        get_connector_store(),
        get_settings(),
        get_profile_service(),
    )
