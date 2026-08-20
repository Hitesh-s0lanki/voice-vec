"""Postgres + pgvector — one table, one row per chunk.

Replaces the embedded Qdrant store. The move buys three things the vector
engine could not:

  * **One engine for all three retrieval channels.** Dense (`embedding`),
    lexical (`tsv`) and structured (`WHERE`/`COUNT`) run against the same rows,
    so a filtered search is one indexed query instead of a Python scan. The
    Qdrant path measured 92 ms of a 200 ms budget filtering 19,870 points by
    payload; a B-tree does it in microseconds.
  * **Concurrent writers.** The embedded store was a single-writer lock, so
    ingest, evaluation and the API could not run together. A server can.
  * **Somewhere to put the rest of the product** — documents, tenants, saved
    turns — next to the chunks rather than in a second system.

`strategy` keeps the five cuts of docs/03-chunking.md in one table rather than
five named vectors. v1 populates `S1` only.

**Dense-only, deliberately.** `tsv` and its GIN index are built at ingest so the
sparse channel is a query change and not a re-index, but `search()` does not
read them yet: switching store *and* adding a channel in one step would make any
movement in recall@5 unattributable. See docs/03-chunking.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from src.core.config import Settings, get_settings
from src.rag.chunk import Chunk


@dataclass(slots=True)
class Hit:
    chunk_id: str
    strategy: str
    score: float
    text: str
    payload: dict


class StoreUnavailable(RuntimeError):
    """Postgres is unreachable, or the table has not been ingested yet."""


# `text` and `english` are stored as columns rather than dug out of `meta` so
# the lexical and structured channels can index them. `meta` still carries the
# whole chunk payload verbatim — `origins` is the one field evaluation cannot
# lose (docs/07-evaluation.md), and round-tripping it as JSONB keeps
# `Hit.payload` identical to what the Qdrant path returned.
# `CREATE EXTENSION` lives in `_configure`, not here — see the note there.
_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_key   TEXT PRIMARY KEY,
    strategy    TEXT NOT NULL,
    language    TEXT NOT NULL,
    query_type  TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    english     TEXT,
    embedding   VECTOR({dim}) NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Built *after* the bulk load, not with the table. Maintaining an HNSW graph
# across 19,870 individual inserts costs far more than building it once over a
# populated table, and the ingest never queries what it is writing.
_INDEXES = """
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
    ON {table} USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS {table}_tsv_gin
    ON {table} USING gin (tsv);

CREATE INDEX IF NOT EXISTS {table}_strategy_language
    ON {table} (strategy, language);
"""

_UPSERT = """
INSERT INTO {table}
    (chunk_key, strategy, language, query_type, text, english, embedding, meta)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_key) DO UPDATE SET
    strategy   = EXCLUDED.strategy,
    language   = EXCLUDED.language,
    query_type = EXCLUDED.query_type,
    text       = EXCLUDED.text,
    english    = EXCLUDED.english,
    embedding  = EXCLUDED.embedding,
    meta       = EXCLUDED.meta
"""

# `1 - (embedding <=> q)` is cosine *similarity*, which is the scale Gate 2's
# floor and margin were swept on (src/core/config.py). `<=>` alone is distance
# and would invert every comparison in guardrails.py.
#
# The strategy/language predicate makes this a filtered ANN search: pgvector
# walks the HNSW graph and discards non-matching rows, so a very selective
# filter can return fewer than `limit`. Harmless while S1 is the only populated
# strategy and the index holds one language; revisit when either changes.
_SEARCH = """
SELECT chunk_key,
       strategy,
       text,
       meta,
       1 - (embedding <=> %(vector)s) AS score
FROM {table}
WHERE strategy = ANY(%(strategies)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
ORDER BY embedding <=> %(vector)s
LIMIT %(limit)s
"""


def _redact(dsn: str) -> str:
    """Neon puts the password in the DSN. Logs and /health must not carry it."""
    if "@" not in dsn:
        return dsn or "unset"
    scheme, _, rest = dsn.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: ConnectionPool | None = None

    # ---- connection -----------------------------------------------------

    @property
    def location(self) -> str:
        return _redact(self._settings.database_url)

    @property
    def table(self) -> str:
        return self._settings.pg_table

    @property
    def collection(self) -> str:
        """The table, under the name the index manifest and /health already use."""
        return self.table

    def _configure(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            # SET takes no bind parameters — `SET x = %s` is a syntax error at
            # the server, and because the pool swallows configure failures as
            # "error connecting" it surfaces as a pool timeout rather than as
            # the syntax error it is. Literal-interpolate, via sql.Literal so
            # the value is still escaped.
            cur.execute(
                sql.SQL("SET statement_timeout = {}").format(
                    sql.Literal(int(self._settings.pg_statement_timeout_ms))
                )
            )
            # The extension has to exist before `register_vector` can look the
            # type up in the catalogue — and on a fresh database nothing has
            # created it yet, because the pool builds its first connection
            # *before* `ensure_schema` is ever handed one. Doing it here rather
            # than in the DDL is what makes an empty Neon database work on the
            # first call instead of failing every checkout with "vector type
            # not found".
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            except psycopg.errors.InsufficientPrivilege:
                # A least-privilege role on a database where someone else
                # already installed it. `register_vector` below is the real
                # check, and it raises with a clearer message than this would.
                conn.rollback()

        # Per connection, not per pool: without it a numpy array binds as an
        # opaque blob and `<=>` fails to resolve.
        register_vector(conn)

        # Neon's pooled endpoint is PgBouncer in transaction mode, which does
        # not carry server-side prepared statements across a checkout. psycopg
        # auto-prepares from the fifth execution onward, so leaving this on
        # breaks the store only *after* it has looked healthy for a while.
        conn.prepare_threshold = None

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            dsn = self._settings.database_url
            if not dsn:
                raise StoreUnavailable(
                    "DATABASE_URL is unset — point it at the Neon instance"
                )
            try:
                self._pool = ConnectionPool(
                    conninfo=dsn,
                    min_size=self._settings.pg_pool_min,
                    max_size=self._settings.pg_pool_max,
                    configure=self._configure,
                    open=True,
                    timeout=self._settings.pg_connect_timeout_s,
                )
            except Exception as error:
                raise StoreUnavailable(str(error)) from error
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ---- index management (ingest) --------------------------------------

    def ensure_schema(self, dim: int, *, recreate: bool = False) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            if recreate:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(_DDL.format(table=self.table, dim=dim))
            conn.commit()

    def create_indexes(self) -> None:
        """Build the ANN and lexical indexes. Call once, after the bulk load."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_INDEXES.format(table=self.table))
            conn.commit()
            cur.execute(f"ANALYZE {self.table}")
            conn.commit()

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Idempotent by `chunk_key`, so a crashed ingest resumes by repeating.

        `executemany` runs in psycopg's pipeline mode, so a batch of 128 costs
        one round trip rather than 128. Over a hosted Postgres that is the
        difference between seconds and twenty minutes.
        """
        rows = [
            (
                chunk.chunk_id,
                chunk.strategy,
                chunk.language,
                chunk.query_type,
                chunk.text,
                chunk.english,
                vector,
                Jsonb(chunk.payload()),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.executemany(_UPSERT.format(table=self.table), rows)
                conn.commit()
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

    # ---- query ----------------------------------------------------------

    def warm(self) -> int:
        """Open the pool and confirm the table exists and holds rows."""
        try:
            return self.count()
        except StoreUnavailable:
            raise
        except psycopg.errors.UndefinedTable as error:
            raise StoreUnavailable(
                f"table {self.table!r} does not exist — run scripts/ingest.py"
            ) from error
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

    def count(self) -> int:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def ready(self) -> bool:
        try:
            return self.count() > 0
        except Exception:
            return False

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        """Dense ANN search over the requested strategies."""
        params = {
            "vector": np.asarray(vector, dtype=np.float32),
            "strategies": list(strategies),
            "language": language,
            "limit": limit,
        }
        try:
            with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                # SET LOCAL, not SET: the pooled Neon endpoint hands the
                # connection to another caller after the transaction, and a
                # session-scoped GUC would leak into their query.
                cur.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                        sql.Literal(int(max(self._settings.hnsw_ef_search, limit)))
                    )
                )
                cur.execute(_SEARCH.format(table=self.table), params)
                rows = cur.fetchall()
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

        return [
            Hit(
                chunk_id=str(chunk_key),
                strategy=str(strategy),
                score=float(score),
                text=str(text or ""),
                payload=dict(meta or {}),
            )
            for chunk_key, strategy, text, meta, score in rows
        ]


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(get_settings())
