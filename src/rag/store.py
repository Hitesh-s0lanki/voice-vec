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
from itertools import repeat
from typing import Sequence

import numpy as np
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from src.core.config import Settings, get_settings
from src.core.db import Database, DatabaseUnavailable, get_db
from src.rag.chunk import Chunk


@dataclass(slots=True)
class Hit:
    chunk_id: str
    strategy: str
    score: float
    text: str
    payload: dict

    def rendering(self, *, english: bool) -> str:
        """The text an answer should be cut out of.

        MSMARCO-XI ships every passage twice — the Indic translation, which is
        what got embedded, and the original English it was translated from
        (docs/01-dataset.md). `english=True` answers from the original, which
        is what a question asked in a language the index does not hold wants:
        the retrieval was cross-lingual, so the passage is right, but reading
        Hindi back to an English speaker is not an answer.

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
    #: (`Hit.rendering`). Only true for an index this app ingested.
    parallel_text: bool = False


class StoreUnavailable(DatabaseUnavailable):
    """Postgres is unreachable, or the table has not been ingested yet.

    A subclass rather than a sibling: the pool raises `DatabaseUnavailable`
    before this module ever sees a cursor, and callers catching the store's own
    error should not have to know which layer gave up.
    """


# `text` and `english` are stored as columns rather than dug out of `meta` so
# the lexical and structured channels can index them. `meta` still carries the
# whole chunk payload verbatim — `origins` is the one field evaluation cannot
# lose (docs/07-evaluation.md), and round-tripping it as JSONB keeps
# `Hit.payload` identical to what the Qdrant path returned.
# `CREATE EXTENSION` lives in `Database._configure` (src/core/db.py), not here.
_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_key    TEXT PRIMARY KEY,
    strategy     TEXT NOT NULL,
    language     TEXT NOT NULL,
    query_type   TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    english      TEXT,
    embedding    VECTOR({dim}) NOT NULL,
    -- The same passage, embedded from its English original. Nullable because
    -- a chunk strategy that merges passages may have no single one (S2–S5);
    -- `search()` falls back to the Indic column whenever it is missing.
    embedding_en VECTOR({dim}),
    meta         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    -- `simple` for the Indic side: Postgres has no Hindi stemmer, and asking
    -- for one silently gets you no stemming plus the wrong stop-word list.
    -- `english` for the other, where the stemmer is real and earns its keep.
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    tsv_en       TSVECTOR GENERATED ALWAYS AS
                 (to_tsvector('english', coalesce(english, ''))) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Everything the DDL above gained after the table already existed. `IF NOT
# EXISTS` on every line, so this is a no-op on a fresh table and the only path
# a populated one needs — nobody should have to re-ingest twenty thousand rows
# to pick up a column.
_MIGRATE = """
ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_en VECTOR({dim});
ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tsv_en TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(english, ''))) STORED;
"""

# Built *after* the bulk load, not with the table. Maintaining an HNSW graph
# across 19,870 individual inserts costs far more than building it once over a
# populated table, and the ingest never queries what it is writing.
_INDEXES = """
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
    ON {table} USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Partial: the English vector is null wherever a chunk has no single English
-- original, and an HNSW graph over nulls is wasted pages.
CREATE INDEX IF NOT EXISTS {table}_embedding_en_hnsw
    ON {table} USING hnsw (embedding_en vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding_en IS NOT NULL;

CREATE INDEX IF NOT EXISTS {table}_tsv_gin
    ON {table} USING gin (tsv);

CREATE INDEX IF NOT EXISTS {table}_tsv_en_gin
    ON {table} USING gin (tsv_en);

CREATE INDEX IF NOT EXISTS {table}_strategy_language
    ON {table} (strategy, language);
"""

_UPSERT = """
INSERT INTO {table}
    (chunk_key, strategy, language, query_type, text, english, embedding,
     embedding_en, meta)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_key) DO UPDATE SET
    strategy     = EXCLUDED.strategy,
    language     = EXCLUDED.language,
    query_type   = EXCLUDED.query_type,
    text         = EXCLUDED.text,
    english      = EXCLUDED.english,
    embedding    = EXCLUDED.embedding,
    -- COALESCE, not EXCLUDED: a re-ingest that skips the English pass must not
    -- wipe vectors an earlier one paid six minutes to compute.
    embedding_en = COALESCE(EXCLUDED.embedding_en, {table}.embedding_en),
    meta         = EXCLUDED.meta
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
       1 - ({embedding} <=> %(vector)s) AS score
FROM {table}
WHERE strategy = ANY(%(strategies)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
  AND {embedding} IS NOT NULL
ORDER BY {embedding} <=> %(vector)s
LIMIT %(limit)s
"""

# Which columns a question reads, by whether it is in English.
#
# Both channels move together: an English question searches the vectors made
# from the English original *and* the tsvector built from it, never one of each.
# The Indic side is indexed with the `simple` configuration because Postgres
# has no Hindi stemmer; `english` gets the real one, and its stemmer is most of
# why the lexical channel is worth having on that side.
_INDIC = ("embedding", "tsv", "simple")
_ENGLISH = ("embedding_en", "tsv_en", "english")


def columns(english: bool) -> tuple[str, str, str]:
    """`(vector column, tsvector column, text-search config)`."""
    return _ENGLISH if english else _INDIC

# The sparse channel, finally read. `tsv` and its GIN index have been built at
# ingest since the pgvector migration and deliberately left unqueried, so that
# a change in recall@5 could be attributed to the store move alone
# (see the module docstring). Rung 2 of the effort ladder is what turns it on.
#
# `websearch_to_tsquery` rather than `plainto_tsquery`: it never raises on
# punctuation, which matters when the query is whatever the speech recogniser
# heard. `ts_rank_cd` with normalisation 32 divides by the rank itself, giving
# a score in (0, 1) — but it is *not* on the cosine scale the guardrail floors
# were swept on, which is exactly why fusion is by rank and never by score.
_LEXICAL = """
SELECT chunk_key,
       strategy,
       text,
       meta,
       ts_rank_cd({tsv}, query, 32) AS score
FROM {table}, websearch_to_tsquery('{config}', %(query)s) AS query
WHERE {tsv} @@ query
  AND strategy = ANY(%(strategies)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
ORDER BY score DESC
LIMIT %(limit)s
"""


class VectorStore:
    def __init__(self, settings: Settings, db: Database | None = None) -> None:
        self._settings = settings
        # `get_store()` hands in the shared pool. A caller that builds a store
        # around its own Settings — a test, a script pointed at a second
        # database — gets a pool for *those* settings rather than silently
        # talking to whatever the environment said.
        self._db = db or Database(settings)

    # ---- connection -----------------------------------------------------

    @property
    def location(self) -> str:
        return self._db.location

    @property
    def table(self) -> str:
        return self._settings.pg_table

    @property
    def collection(self) -> str:
        """The table, under the name the index manifest and /health already use."""
        return self.table

    @property
    def pool(self) -> ConnectionPool:
        """The shared pool — conversations check out of this one too.

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

    # ---- index management (ingest) --------------------------------------

    def ensure_schema(self, dim: int, *, recreate: bool = False) -> None:
        """Create the table, and bring an existing one up to the current shape.

        `statement_timeout` is lifted for the same reason `create_indexes`
        lifts it: adding a *generated* column rewrites every row, which on a
        populated table takes longer than a ceiling sized for searches. Any DDL
        against this table wants the lift — the two are the only places that
        issue it.
        """
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")
            if recreate:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(_DDL.format(table=self.table, dim=dim))
            cur.execute(_MIGRATE.format(table=self.table, dim=dim))

    def create_indexes(self) -> None:
        """Build the ANN and lexical indexes. Call once, after the bulk load.

        `statement_timeout` is lifted for the duration. Every connection out of
        the pool carries `PG_STATEMENT_TIMEOUT_MS` — 5 s, which is deliberately
        far more than a search may take and far less than an HNSW graph over
        twenty thousand vectors needs. Without this the ingest embeds for six
        minutes and then loses the index build to a timeout, leaving a
        populated table that answers every query by sequential scan.

        `SET LOCAL` inside an explicit transaction, so the pooled connection
        goes back with its usual ceiling rather than an unbounded one.
        """
        with self.pool.connection() as conn:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = 0")
                cur.execute(_INDEXES.format(table=self.table))

            with conn.transaction(), conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = 0")
                cur.execute(f"ANALYZE {self.table}")

    def backfill_english(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        """Fill `embedding_en` on rows that already exist.

        The migration path for an index built before the English column did.
        Re-running the whole ingest would work and would spend six minutes
        recomputing Indic vectors that have not changed; this touches one
        column and leaves everything else — including `origins`, which
        evaluation cannot lose — exactly as it was.
        """
        rows = [(vector, key) for key, vector in zip(keys, vectors)]
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.executemany(
                    f"UPDATE {self.table} SET embedding_en = %s WHERE chunk_key = %s",
                    rows,
                )
                conn.commit()
        except Exception as error:
            raise StoreUnavailable(str(error)) from error

    def english_backlog(self) -> list[tuple[str, str]]:
        """`(chunk_key, english)` for every row still missing its vector."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT chunk_key, english FROM {self.table}
                    WHERE embedding_en IS NULL
                      AND btrim(coalesce(english, '')) <> ''
                    ORDER BY length(english)"""
            )
            return [(str(key), str(text)) for key, text in cur.fetchall()]

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        english_vectors: np.ndarray | None = None,
    ) -> None:
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
                english_vector,
                Jsonb(chunk.payload()),
            )
            for chunk, vector, english_vector in zip(
                chunks, vectors, english_vectors if english_vectors is not None else repeat(None)
            )
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
        """Everything, because this is the store the app ingests into.

        The lexical channel exists because `tsv` is a generated column with a
        GIN index on it, and the parallel English because the ingest writes it
        (docs/01-dataset.md). A *connected* pgvector speaks the same schema, so
        `PgVectorBackend` forwards to this rather than claiming less.
        """
        return Capabilities(lexical=True, filters=True, parallel_text=True)

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
        _, tsv, config = columns(english)
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    _LEXICAL.format(table=self.table, tsv=tsv, config=config), params
                )
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

        `english` searches `embedding_en` — the same passages, embedded from
        the English original MS MARCO wrote rather than from its machine
        translation. That is native retrieval for an English question instead
        of the cross-lingual hop of docs/13-cross-lingual.md, and the partial
        HNSW index behind it covers only the rows that have one.
        """
        embedding, _, _ = columns(english)
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
            # it and this pool is shared with everything else that writes.
            with self.pool.connection() as conn:
                autocommit = conn.autocommit
                conn.autocommit = True
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            _SEARCH.format(table=self.table, embedding=embedding), params
                        )
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


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(get_settings(), get_db())
