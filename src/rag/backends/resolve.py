"""Which vector store answers for which user.

A user who has connected Pinecone, Astra or their own Postgres is searched
against that. Everybody else — signed out, or signed in with nothing connected
— is searched against the deployment's own store, which is the behaviour this
app had before connectors existed and is what keeps `/ask` working out of the
box.

Two rules that are easy to get wrong and expensive to get wrong:

**Fall back, never cross over.** A user's backend is built from credentials
read under their own id. When that fails the answer is the *deployment* store,
never another user's — which is guaranteed by construction here, because the
only per-user thing in this module is the credential lookup and it takes a
user id.

**Connecting two vector stores is not an error.** Somebody can have Pinecone
and pgvector attached at once; `PREFERENCE` decides, in a fixed order, so the
same user gets the same store on every request rather than whichever row came
back first.

**A backend that cannot answer is not a backend.** `verify` runs when the form
is submitted, and an index can be emptied, dropped or renamed long after that.
So a freshly built backend is probed once — `ready()` — and one that says no is
cached as a miss, which falls through to the deployment store. Reconnecting
re-seals the credentials under a new blob, so the probe runs again the moment
somebody fixes what was wrong.

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

from src.connectors.registry import vector_slugs
from src.connectors.service import ConnectorService, get_connector_service
from src.connectors.store import ConnectorStore, get_connector_store
from src.core.config import Settings, get_settings
from src.rag.backends.astra import AstraBackend
from src.rag.backends.base import VectorBackend
from src.rag.backends.pgvector import PgVectorBackend
from src.rag.backends.pinecone import PineconeBackend
from src.rag.store import VectorStore, get_store

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
    ) -> None:
        self._connectors = connectors
        self._store = store
        self._settings = settings
        # user_id → (slug, sealed blob it was built from, backend or None)
        # A cached `None` is a backend that built but could not answer: the
        # probe is worth one round trip per connect, not one per question.
        self._cache: OrderedDict[
            str, tuple[str, str, VectorBackend | None]
        ] = OrderedDict()

    def default(self) -> VectorStore:
        """The deployment's own store — what everyone got before connectors."""
        return get_store()

    def for_user(self, user_id: str | None) -> VectorBackend:
        """The store that should answer this user's question.

        Never raises. A connected backend that cannot be built — or that
        builds and then cannot answer — is logged and the deployment store is
        returned, because a broken connector should degrade retrieval rather
        than delete it.
        """
        if not user_id:
            return self.default()

        try:
            chosen = self._connected(user_id)
        except Exception as error:
            log.warning("could not resolve a backend for %s: %s", user_id, error)
            return self.default()

        return chosen or self.default()

    def _connected(self, user_id: str) -> VectorBackend | None:
        rows = {
            row.connector: row
            for row in self._store.list(user_id)
            if row.connector in vector_slugs()
        }
        if not rows:
            self._evict(user_id)
            return None

        slug = next((s for s in PREFERENCE if s in rows), None)
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
        usable: VectorBackend | None = backend
        if not backend.ready():
            # Built fine, answers nothing: an index that was emptied, dropped,
            # or renamed since it was connected. Building is not the test that
            # matters — this is the failure `verify` cannot catch, because it
            # happens after the form was green. Falling back here is what makes
            # it a degraded answer instead of an abstain on every question.
            log.warning(
                "connected %s for %s is not searchable — using the deployment store",
                slug,
                user_id,
            )
            self._close(backend)
            usable = None

        self._evict(user_id)
        self._cache[user_id] = (slug, row.credentials, usable)
        self._cache.move_to_end(user_id)

        while len(self._cache) > self._settings.vector_backend_cache:
            _, (_, _, evicted) = self._cache.popitem(last=False)
            self._close(evicted)

        return usable

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
    an offline evaluation over the deployment's own index has no business
    requiring — and a resolver that could return somebody's connected Pinecone
    mid-sweep would make the numbers unattributable anyway.
    """

    def __init__(self, backend: VectorBackend) -> None:
        self._backend = backend

    def default(self) -> VectorBackend:
        return self._backend

    def for_user(self, user_id: str | None = None) -> VectorBackend:
        return self._backend


@lru_cache
def get_resolver() -> BackendResolver:
    return BackendResolver(get_connector_service(), get_connector_store(), get_settings())
