"""Searching a Postgres the user connected.

`VectorStore` in `src/rag/store.py` is the query half of exactly this — the
`<=>` search, the lexical channel, the HNSW `ef_search` — and since the
deployment's own corpus went away it exists for no other caller. This is where
it is pointed at a database: a private pool against the user's DSN, and the
column map discovered when the connector was verified.

The pool is private and small on purpose. A user's Neon has its own connection
limit and this app is a guest in it — one or two connections is enough for a
rail panel and an occasional query, and taking eight would be rude in a way
that only shows up under load somebody else is having.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from src.core.config import get_settings
from src.core.db import Database
from src.rag.columns import ColumnMap
from src.rag.backends.base import Capabilities, Hit, StoreUnavailable
from src.rag.store import DEFAULT_TABLE, VectorStore

log = logging.getLogger("vec.rag.pgvector")


class PgVectorBackend:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        settings = get_settings()
        # `verify_pgvector` resolves the table and seals the name it settled
        # on, so this is blank only for an account attached before it did.
        table = (credentials.get("table") or "").strip() or DEFAULT_TABLE

        # A copy of the app's settings pointed somewhere else. Everything the
        # search depends on — ef_search, the statement timeout, the embedding
        # dimension — stays as configured; only the destination changes.
        self._settings = settings.model_copy(
            update={
                "database_url": credentials["dsn"],
                "pg_pool_min": 1,
                "pg_pool_max": 2,
            }
        )

        # The column map discovered when this connector was verified, sealed
        # beside the DSN under `col_*` keys. Absent — an account attached
        # before mapping existed — falls back to this app's own schema, which
        # is what those accounts were verified against anyway.
        columns = ColumnMap.from_mapping(
            {
                key[len("col_") :]: value
                for key, value in credentials.items()
                if key.startswith("col_")
            }
        )

        self._store = VectorStore(
            self._settings, Database(self._settings), columns, table
        )

        # Which model built this index, and how wide. Empty whenever the
        # index matches this app's own width, which is the common case.
        self._dim = int(credentials.get("dim") or settings.embed_dim)

    @property
    def name(self) -> str:
        return "pgvector"

    def describe(self) -> str:
        """Redacted by `Database.location` — the DSN carries a password."""
        return f"pgvector/{self._store.location}#{self._store.table}"

    def ready(self) -> bool:
        try:
            return self._store.ready()
        except Exception as error:
            log.debug("connected pgvector not ready: %s", error)
            return False

    def capabilities(self) -> Capabilities:
        """Whatever `VectorStore` can do with *this* column map.

        The caveat this used to carry — that a table with no `tsv` column would
        fail the lexical query — is gone. The map knows whether there is a
        tsvector column, `VectorStore.capabilities()` reads it off, and rung 2
        is told dense-only before it asks rather than discovering it from an
        error. No round trip: the map was built when the connector was verified.
        """
        return self._store.capabilities()

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        return self._store.search(
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
        return self._store.search_lexical(
            query, strategies=strategies, limit=limit, language=language
        )


    def embed_query(self, text: str) -> np.ndarray:
        """Embedded at *this* store's width, whatever that is.

        `text-embedding-3` can be asked for exactly the number of dimensions
        the store's own catalogue reported, so there is one embedder and one
        call — the branch that used to pick a local model when the widths
        happened to agree went with the local model itself
        (docs/25-no-local-embedder.md).
        """
        from src.rag.embed import get_embedder
        from src.rag.remote_embed import RemoteEmbedUnavailable

        try:
            return get_embedder().embed_query(text, dim=self._dim or None)
        except RemoteEmbedUnavailable as error:
            # `StoreUnavailable` is what the ladder already abstains on, so a
            # provider outage — or a missing key — degrades to "my sources are
            # unavailable" rather than a 500 from a layer nobody can place.
            raise StoreUnavailable(str(error)) from error

    def close(self) -> None:
        """Give the user's database its connections back."""
        self._store.close()
