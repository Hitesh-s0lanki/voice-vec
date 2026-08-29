"""Reading a user's Postgres: the catalogue first, then a sample of the rows.

Postgres is the one connected store that will describe itself honestly before
being asked for data, and this probe leans on that hard. `pg_attribute` gives
every column and its type in one round trip, `pg_indexes` gives the distance
metric the index was actually built for, and `reltuples` gives a row count
without a scan. Three cheap catalogue reads answer most of what the profile
needs, and only what is left — what the text looks like, whether the metadata
columns are ever actually populated — costs a sample.

Two measurements here exist because of a specific failure seen in a real
connected database, and neither is something a schema check would catch:

  **A `NOT NULL` column can still be dead.** `book_chunks.page` was `1` on all
  2,366 rows. It satisfies every constraint, survives every migration check,
  and cannot narrow a search. `field_stats` reports it as constant and the card
  says so, because the alternative is an agent that keeps trying to cite a page.

  **Vectors are not always unit length.** A table of 768-dim vectors whose norms
  all sat at ~0.588 turned out to be 3072-dim Matryoshka embeddings truncated
  and never re-normalised. Cosine does not care and `<#>` inner product does, so
  the norm is measured and recorded rather than assumed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Mapping

from src.connectors.probes.base import (
    PROBE_TIMEOUT_S,
    SAMPLE_SIZE,
    embedding_match,
    excerpts_of,
    field_stats,
    length_stats,
    pick_text_field,
    scripts_of,
    text_of,
    unreachable,
)
from src.connectors.profile import EMBEDDING_MATCH_MIN, Observation, VectorShape

log = logging.getLogger("vec.connectors.probe")

_VECTOR_TYPE = re.compile(r"^(?:vector|halfvec)\((\d+)\)$")

#: `reltuples` is an estimate maintained by ANALYZE and it is wrong in the
#: direction that matters — it read 2116 on a table holding 2366 rows. Small
#: tables get an exact count because it costs nothing; large ones keep the
#: estimate, because an exact `count(*)` on somebody else's 10M-row table is a
#: seq scan this app has no business running.
EXACT_COUNT_CEILING = 200_000

#: Below this a random-ordered sample is affordable and unbiased. Above it,
#: `ORDER BY random()` sorts the whole table, so `TABLESAMPLE SYSTEM` reads a
#: few pages instead. Page-level sampling clusters — rows near each other on
#: disk arrive together — which is acceptable for "does this column exist" and
#: is why the profile reports the sample size it actually got.
RANDOM_SAMPLE_CEILING = 50_000

_COLUMNS = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %s
  AND n.nspname = ANY(current_schemas(false))
  AND a.attnum > 0 AND NOT a.attisdropped
"""

_ESTIMATE = """
SELECT c.reltuples::bigint
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %s AND n.nspname = ANY(current_schemas(false))
"""

_INDEXES = "SELECT indexdef FROM pg_indexes WHERE tablename = %s"

#: opclass → what a person calls that distance. Read off the index rather than
#: assumed, because a table indexed `vector_l2_ops` and searched with `<=>`
#: does not use its index at all — it seq-scans, silently, and only shows up as
#: latency nobody can explain.
_METRICS = {
    "vector_cosine_ops": "cosine",
    "vector_ip_ops": "inner product",
    "vector_l2_ops": "euclidean",
    "halfvec_cosine_ops": "cosine",
}


class PgVectorProbe:
    def __init__(
        self, credentials: Mapping[str, str], *, table: str, excerpts: bool = True, **_: Any
    ) -> None:
        self._dsn = credentials["dsn"].strip()
        self._table = table
        self._excerpts = excerpts

    @property
    def location(self) -> str:
        """Host and table. Redacted, and short enough to head a card.

        `redact` keeps the whole DSN minus the password, which is right for a
        log line and wrong here: this is the first thing on the card an agent
        reads, and forty characters of connection parameters crowd out the
        name of the corpus.
        """
        from src.core.db import redact

        host = redact(self._dsn).partition("@")[2].partition("/")[0] or "postgres"
        return f"pgvector/{host}#{self._table}"

    def observe(self) -> Observation:
        import psycopg
        from psycopg import sql

        started = time.perf_counter()
        try:
            with psycopg.connect(self._dsn, connect_timeout=int(PROBE_TIMEOUT_S)) as conn:
                # A profile may not outlive its usefulness holding somebody
                # else's connection. Anything slower than this is a store the
                # agent should be told about rather than wait for.
                #
                # Literal-interpolated, via `sql.Literal` so it is still
                # escaped: `SET x = %s` is a syntax error at the server, which
                # `src/core/db.py` documents having been caught by once
                # already. Here it would have surfaced as every connected
                # Postgres reporting itself unreachable.
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SET statement_timeout = {}").format(
                            sql.Literal(int(PROBE_TIMEOUT_S * 1000))
                        )
                    )
                return self._read(conn, started)
        except Exception as error:
            log.info("pgvector probe failed: %s", type(error).__name__)
            return unreachable(
                "pgvector", "vector", self.location, f"could not read the table: {error}"
            )

    def _read(self, conn: Any, started: float) -> Observation:
        from psycopg import sql

        notes: list[str] = []
        with conn.cursor() as cur:
            cur.execute(_COLUMNS, (self._table,))
            columns = {name: kind for name, kind in cur.fetchall()}

            if not columns:
                return unreachable(
                    "pgvector", "vector", self.location,
                    f"there is no table called “{self._table}” on the search path",
                )

            vector_column, dimensions = _vector_column(columns)
            metric, index = _index_shape(cur, self._table)
            records = _count(cur, self._table)

            ident = sql.Identifier(self._table)
            # The catalogue's answer to "which columns are vectors", passed
            # down so the sample can drop them. This connection has no
            # `register_vector` — it is a bare psycopg connect, not the app's
            # pool — so a pgvector column arrives as the *string* "[0.1,0.2,…]"
            # and every heuristic for "is this an array" reads it as text.
            vector_columns = {
                name for name, kind in columns.items() if _VECTOR_TYPE.match(kind.strip())
            }
            rows = _sample(cur, ident, records, vector_columns)
            normalised = (
                _normalised(cur, ident, vector_column) if vector_column else None
            )
            # One record, its text and its vector, read together — the only
            # way to find out whether this index and our query embedder are
            # the same space. See `embedding_match`.
            paired = _paired(cur, ident, vector_column, columns) if vector_column else None

        if vector_column is None:
            notes.append("no pgvector column — this table cannot answer a dense search")

        # `tsv` needs no special handling: it is a generated stored column, so
        # `SELECT *` returns it and `field_stats` measures it like anything
        # else. Injecting it as a marker — which this did at first — overwrote
        # the real value with a constant and put "tsv" on the card under
        # "carried but useless".
        text_field = pick_text_field(rows)
        texts = text_of(rows, text_field)
        if not text_field:
            notes.append("no readable text column — hits from here cannot be quoted")
        if normalised is False:
            notes.append(
                "vectors are not unit length — cosine is unaffected, inner product is not"
            )

        stats = field_stats(rows)
        for stat in stats:
            if stat.constant and stat.name != text_field:
                notes.append(f"{stat.name} is the same value on every sampled row")

        match = self._match(paired, dimensions)
        if match is not None and match < EMBEDDING_MATCH_MIN:
            notes.append(
                "these vectors were built by a different embedding model — a search "
                f"here returns unrelated records (self-similarity {match:.3f})"
            )

        return Observation(
            connector="pgvector",
            kind="vector",
            location=self.location,
            reachable=True,
            sampled=len(rows),
            vectors=VectorShape(
                dimensions=dimensions,
                metric=metric,
                index=index,
                normalised=normalised,
                records=records,
            ),
            fields=stats,
            text_field=text_field,
            text_chars=length_stats(texts),
            scripts=scripts_of(texts),
            excerpts=excerpts_of(rows, text_field, stats=stats, allowed=self._excerpts),
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes[:6]),
            embedding_match=match,
        )

    def _match(self, paired: "tuple[str, list[float]] | None", dimensions: int | None) -> float | None:
        """Compare one record against our own embedding of its own text.

        The embedder is chosen the same way `PgVectorBackend.embed_query`
        chooses it — locally when the widths agree, remotely at the store's
        width when they do not — because the question is not "can we embed
        this" but "would the vector we search with land near theirs".
        """
        if paired is None:
            return None

        text, vector = paired

        def embed(value: str):
            from src.core.config import get_settings
            from src.rag.embed import get_embedder
            from src.rag.remote_embed import embed_query as embed_remote

            settings = get_settings()
            if not dimensions or dimensions == settings.embed_dim:
                return get_embedder().embed_query(value)
            return embed_remote(value, dimensions, settings=settings)

        return embedding_match(vector, text, embed)


def _vector_column(columns: Mapping[str, str]) -> tuple[str | None, int | None]:
    """The first `vector(n)` column, and its width.

    First rather than "the one called embedding": a table this app did not
    build names it whatever it likes, and a probe that only recognises its own
    schema reports every other table as having no vectors at all.
    """
    for name, kind in columns.items():
        match = _VECTOR_TYPE.match(kind.strip())
        if match:
            return name, int(match.group(1))
    return None, None


def _index_shape(cur: Any, table: str) -> tuple[str, str]:
    try:
        cur.execute(_INDEXES, (table,))
        defs = [row[0] for row in cur.fetchall()]
    except Exception:
        return "", ""

    for definition in defs:
        for opclass, metric in _METRICS.items():
            if opclass in definition:
                kind = "hnsw" if "USING hnsw" in definition else (
                    "ivfflat" if "USING ivfflat" in definition else ""
                )
                return metric, kind
    # A vector column with no vector index. Searches work and seq-scan; worth
    # knowing, because it is the difference between 11 ms and several seconds.
    return "", "none"


def _count(cur: Any, table: str) -> int | None:
    from psycopg import sql

    try:
        cur.execute(_ESTIMATE, (table,))
        row = cur.fetchone()
        estimate = int(row[0]) if row and row[0] is not None else -1
    except Exception:
        estimate = -1

    if 0 <= estimate <= EXACT_COUNT_CEILING:
        try:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
            return int(cur.fetchone()[0])
        except Exception:
            pass
    return estimate if estimate >= 0 else None


def _sample(
    cur: Any, ident: Any, records: int | None, vector_columns: set[str]
) -> list[dict[str, Any]]:
    """Rows as dicts, without the vector column's hundreds of floats.

    `SELECT *` would pull the embeddings — 200 rows × 768 floats is megabytes
    over the wire to learn nothing, since the geometry was already read from
    the catalogue.
    """
    from psycopg import sql

    big = records is not None and records > RANDOM_SAMPLE_CEILING
    query = (
        sql.SQL("SELECT * FROM {} TABLESAMPLE SYSTEM (1) LIMIT {}").format(
            ident, sql.Literal(SAMPLE_SIZE)
        )
        if big
        else sql.SQL("SELECT * FROM {} ORDER BY random() LIMIT {}").format(
            ident, sql.Literal(SAMPLE_SIZE)
        )
    )

    try:
        cur.execute(query)
        names = [d.name for d in cur.description or []]
        rows = [dict(zip(names, row)) for row in cur.fetchall()]
    except Exception as error:
        log.debug("pgvector sample failed: %s", error)
        return []

    # TABLESAMPLE on a small or sparsely-packed table can return nothing at
    # all. An empty sample would be read as "no fields exist", which is a much
    # stronger claim than "we did not look hard enough".
    if not rows and big:
        try:
            cur.execute(sql.SQL("SELECT * FROM {} LIMIT {}").format(ident, sql.Literal(SAMPLE_SIZE)))
            names = [d.name for d in cur.description or []]
            rows = [dict(zip(names, row)) for row in cur.fetchall()]
        except Exception:
            return []

    return [_scrub(row, vector_columns) for row in rows]


def _scrub(row: Mapping[str, Any], vector_columns: set[str] = frozenset()) -> dict[str, Any]:
    """Drop the vectors, flatten JSONB metadata up into the record.

    A store that keeps its metadata in one `meta` JSONB column has fields the
    agent can filter on; leaving them nested would report one field called
    `meta` with 100% coverage and nothing about what is in it.
    """
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key in vector_columns:
            continue  # named by the catalogue as a vector, whatever it arrived as
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict, list)):
            continue  # anything else array-shaped
        if key in ("meta", "metadata", "payload") and isinstance(value, dict):
            for inner, nested in value.items():
                flat.setdefault(inner, nested)
            continue
        flat[key] = value
    return flat


def _normalised(cur: Any, ident: Any, column: str) -> bool | None:
    """Are the vectors unit length? Measured over a bounded sample.

    `-(v <#> v)` is the dot product of a vector with itself, so its square root
    is the L2 norm. Roundabout, but it needs no extension function beyond the
    operator pgvector already defines, and `l2_norm` is overloaded across
    `vector`/`halfvec` in a way that makes a bare call ambiguous.
    """
    from psycopg import sql

    try:
        cur.execute(
            sql.SQL(
                "SELECT min(n), max(n) FROM ("
                "SELECT sqrt((-1) * ({col} <#> {col})) AS n FROM {tbl} "
                "WHERE {col} IS NOT NULL LIMIT {n}) s"
            ).format(col=sql.Identifier(column), tbl=ident, n=sql.Literal(SAMPLE_SIZE)),
        )
        low, high = cur.fetchone()
    except Exception:
        return None

    if low is None or high is None:
        return None
    return 0.99 <= float(low) and float(high) <= 1.01


def _paired(
    cur: Any, ident: Any, column: str, columns: Mapping[str, str]
) -> tuple[str, list[float]] | None:
    """One row's text and its stored vector, read in the same statement.

    They have to come from the *same* record or the comparison means nothing —
    two different passages under the same model score somewhere around 0.6,
    which is neither a match nor a mismatch. One row is enough: the question is
    which space the vectors live in, and every row in a table lives in the same
    one.
    """
    from psycopg import sql

    text_column = next(
        (
            name
            for name, kind in columns.items()
            if kind.strip().lower() in {"text", "character varying", "varchar"}
        ),
        None,
    )
    if text_column is None:
        return None

    try:
        cur.execute(
            sql.SQL(
                "SELECT {txt}, {vec}::text FROM {tbl} "
                "WHERE {vec} IS NOT NULL AND {txt} IS NOT NULL LIMIT 1"
            ).format(txt=sql.Identifier(text_column), vec=sql.Identifier(column), tbl=ident)
        )
        row = cur.fetchone()
    except Exception as error:
        log.info("could not read a paired record: %s", type(error).__name__)
        return None

    if not row or not row[0] or not row[1]:
        return None

    try:
        return str(row[0]), json.loads(row[1])
    except (ValueError, TypeError):
        return None
