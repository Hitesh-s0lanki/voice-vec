"""Searching a Postgres the user connected, with this app's own schema.

The odd one of the three, because the code already exists. `VectorStore` in
`src/rag/store.py` speaks exactly this — the same table, the same `<=>`, the
same HNSW `ef_search` — and the only thing that differs is which database it
points at and which pool it checks out of.

So this does not reimplement it. It builds a `VectorStore` over a private pool
against the user's DSN, and forwards. That is worth more than the lines it
saves: the search SQL, the strategy filter and the language filter stay in one
place, so a fix to the query cannot land for the deployment's store and miss
everybody's connected one.

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
from src.rag.backends.base import Capabilities, Hit, StoreUnavailable
from src.rag.store import VectorStore

log = logging.getLogger("vec.rag.pgvector")


class PgVectorBackend:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        settings = get_settings()
        table = (credentials.get("table") or "").strip() or settings.pg_table

        # A copy of the app's settings pointed somewhere else. Everything the
        # search depends on — ef_search, the statement timeout, the embedding
        # dimension — stays as configured; only the destination changes.
        self._settings = settings.model_copy(
            update={
                "database_url": credentials["dsn"],
                "pg_table": table,
                "pg_pool_min": 1,
                "pg_pool_max": 2,
            }
        )

        self._store = VectorStore(self._settings, Database(self._settings))

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
        """Whatever `VectorStore` can do, since that is what is running.

        One caveat this cannot check cheaply: a user who pointed us at their
        own Postgres may have built the table with an older migration and have
        no `tsv` column. The lexical query would then fail — which the ladder
        already handles, because rung 2 treats a failed lexical channel as an
        empty contribution and falls back to dense-only rather than losing the
        answer. Probing for the column on every request would cost a round trip
        to save a failure that is both rare and already survivable.
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

    def close(self) -> None:
        """Give the user's database its connections back."""
        self._store.close()
