"""Searching a Postgres + pgvector table — one row per chunk.

This is not a store this deployment owns. It used to be: there was a `chunks`
table behind `DATABASE_URL` that everybody was searched against, built by an
ingest script from one dataset. That is gone. Retrieval is a property of what a
user connected (docs/13-connectors.md), so the only thing left here is the
*query* half — and its single caller is `PgVectorBackend`, which builds one of
these over the DSN somebody attached.

What that leaves is deliberately narrow. There is no DDL, no upsert and no
`get_store()`: this app never writes to, and never owns, the table it reads.
The columns it reads are not assumed either — a `ColumnMap` discovered when the
connector was verified says which column plays which role, so a table holding
`id` and `chunk_text` is searchable without being a copy of anything.

Both channels live in `src/rag/columns.py` rather than as literals here, so a
fix to the SQL cannot land for one caller and miss another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from psycopg_pool import ConnectionPool

from src.core.config import Settings
from src.core.db import Database, DatabaseUnavailable
from src.rag.columns import DEFAULT as DEFAULT_COLUMNS
from src.rag.columns import ColumnMap, lexical_sql, search_sql, text_config

#: What `table` means when a connector was sealed before `verify_pgvector`
#: resolved one for itself. Every account attached since carries the name the
#: verify settled on, so this is a compatibility floor and not a default worth
#: configuring.
DEFAULT_TABLE = "chunks"


@dataclass(slots=True)
class Hit:
    chunk_id: str
    strategy: str
    score: float
    text: str
    payload: dict

    def rendering(self, *, english: bool) -> str:
        """The text an answer should be cut out of.

        An index whose chunks carry the English original beside the indexed
        translation can answer a cross-lingual question in the language it was
        asked in: the retrieval was cross-lingual, so the passage is right, but
        reading Hindi back to an English speaker is not an answer.

        Falls back to the indexed text whenever there is no English rendering,
        so this can never turn a hit into an empty one.
        """
        if english:
            original = str(self.payload.get("english") or "").strip()
            if original:
                return original
        return self.text


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What this store can do beyond returning nearest neighbours.

    Defaults are the floor, not the typical case: a backend that says nothing
    is assumed to do dense search only. That is the safe direction — claiming a
    channel that is not there fails at query time, whereas not claiming one
    that is only costs the recall it would have added.
    """

    #: A keyword channel that can be fused with the dense one (rung 2's hybrid).
    lexical: bool = False
    #: `strategies` and `language` actually narrow the result set. False means
    #: they are passed and ignored, which is what a user-populated index that
    #: never carried this app's metadata does.
    filters: bool = True
    #: Chunks carry the original English beside the indexed translation, so a
    #: cross-lingual question can be answered in the language it was asked in
    #: (`Hit.rendering`).
    parallel_text: bool = False


class StoreUnavailable(DatabaseUnavailable):
    """Postgres is unreachable, or the table is not there any more.

    A subclass rather than a sibling: the pool raises `DatabaseUnavailable`
    before this module ever sees a cursor, and callers catching the store's own
    error should not have to know which layer gave up.
    """


class VectorStore:
    def __init__(
        self,
        settings: Settings,
        db: Database | None = None,
        columns: ColumnMap | None = None,
        table: str = DEFAULT_TABLE,
    ) -> None:
        self._settings = settings
        # The pool for *these* settings. `PgVectorBackend` hands in one built
        # against the user's own DSN, small on purpose — this app is a guest in
        # their database, not the owner of it.
        self._db = db or Database(settings)
        # Which column plays which role (`src/rag/columns.py`), discovered at
        # verify time. The default map is what the connectors attached before
        # mapping existed were verified against.
        self._columns = columns or DEFAULT_COLUMNS
        self._table = table

    @property
    def columns(self) -> ColumnMap:
        return self._columns

    # ---- connection -----------------------------------------------------

    @property
    def location(self) -> str:
        return self._db.location

    @property
    def table(self) -> str:
        return self._table

    @property
    def pool(self) -> ConnectionPool:
        """This store's pool.

        Re-raised as `StoreUnavailable` because that is the exception
        `ask_service` retries and abstains on; a bare `DatabaseUnavailable`
        escaping that handler would surface as a 500 instead of an honest "my
        sources are unavailable".
        """
        try:
            return self._db.pool
        except StoreUnavailable:
            raise
        except DatabaseUnavailable as error:
            raise StoreUnavailable(str(error)) from error

    def close(self) -> None:
        self._db.close()

    # ---- query ----------------------------------------------------------

    def ready(self) -> bool:
        """Is there a table here with rows in it?

        `LIMIT 1` rather than `count(*) > 0`, which answers the same question
        and reads every row to do it. This is on the connect path for a store
        somebody else is hosting (`src/rag/backends/resolve.py`), and a probe
        whose cost grows with their corpus is a probe that times out on the
        indexes most worth connecting.
        """
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {self.table} LIMIT 1")
                return cur.fetchone() is not None
        except Exception:
            return False

    def capabilities(self) -> Capabilities:
        """Read off the column map, not asserted.

        The map is whatever was discovered on the connected Postgres, and a
        table with no `tsv` column reports no lexical channel rather than
        claiming one the ladder would then try to query. This used to be a
        constant, which is why `PgVectorBackend` forwarded a promise of three
        channels for every table it was pointed at.
        """
        return Capabilities(
            lexical=self._columns.lexical,
            filters=self._columns.filters,
            parallel_text=self._columns.parallel_text,
        )

    def search_lexical(
        self,
        query: str,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
        english: bool = False,
    ) -> list[Hit]:
        """Keyword search over the same rows, for rung 2's hybrid fusion.

        Empty query, or a query made entirely of stop words, returns nothing
        rather than raising — a lexical channel with no terms in it is a
        contribution of zero to the fusion, not a failed search.
        """
        terms = query.strip()
        if not terms:
            return []

        params = {
            "query": terms,
            "strategies": list(strategies),
            "language": language,
            "limit": limit,
        }
        if not self._columns.lexical:
            # No tsvector column to run against. Empty rather than raising: the
            # ladder treats a failed lexical channel as a zero contribution and
            # falls back to dense-only, and `capabilities()` already said so.
            return []

        sql = lexical_sql(
            self.table,
            self._columns,
            english=english,
            config=text_config(self._columns, english=english),
        )
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

        return [self._hit(row) for row in rows]

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
        english: bool = False,
    ) -> list[Hit]:
        """Dense ANN search over the requested strategies.

        `english` searches the parallel vector column when the map has one —
        the same passages, embedded from the English original rather than from
        a machine translation. That is native retrieval for an English question
        instead of the cross-lingual hop of docs/13a-cross-lingual.md.
        """
        sql = search_sql(self.table, self._columns, english=english)
        params = {
            "vector": np.asarray(vector, dtype=np.float32),
            "strategies": list(strategies),
            "language": language,
            "limit": limit,
        }
        try:
            # One statement, one round trip — which took some doing, and is
            # worth the six lines. Measured against Neon at 66 ms of round
            # trip, where the ANN query itself costs ~4 ms:
            #
            #   BEGIN + SET LOCAL + SELECT + COMMIT     273 ms   (what shipped)
            #   SELECT + COMMIT                         192 ms
            #   SELECT                                   66 ms
            #
            # `hnsw.ef_search` moved to `Database._configure`, so it is set
            # once per connection instead of needing a transaction to scope a
            # `SET LOCAL`. What was left was the implicit transaction: a read
            # opens one and the pool commits it on the way out, for a round
            # trip that protects nothing. `autocommit` is client-side state in
            # psycopg — flipping it emits no statement — and it is restored
            # before the connection goes back, because the pool does not reset
            # it and this pool may be shared with everything else that writes.
            with self.pool.connection() as conn:
                autocommit = conn.autocommit
                conn.autocommit = True
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                finally:
                    conn.autocommit = autocommit
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

        return [self._hit(row) for row in rows]

    @staticmethod
    def _hit(row: tuple) -> Hit:
        """One result row. Both queries select the same five columns."""
        chunk_key, strategy, text, meta, score = row
        return Hit(
            chunk_id=str(chunk_key),
            strategy=str(strategy),
            score=float(score),
            text=str(text or ""),
            payload=dict(meta or {}),
        )
