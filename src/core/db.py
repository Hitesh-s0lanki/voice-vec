"""One Postgres pool, shared by everything that stores anything.

The vector store opened this pool first, and its own docstring said why the
move off an embedded engine was worth it: *"somewhere to put the rest of the
product — documents, tenants, saved turns — next to the chunks."* Conversations
are the first of those, and they differ from chunks in one way that decides
where the pool lives: they are needed whether or not retrieval is on. A second
pool to the same database would double the connection count Neon charges
against the project for no gain, so the connection concern moved out of
`VectorStore` and into here, and both callers check out of the same pool.

Everything below is the connection lifecycle that used to live in
`src/rag/store.py`, unchanged in behaviour — `register_vector` still runs per
checkout, because a chat query and a vector query can be handed the same
connection.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool

from src.core.config import Settings, get_settings


class DatabaseUnavailable(RuntimeError):
    """Postgres is unreachable, or DATABASE_URL was never set."""


def redact(dsn: str) -> str:
    """Neon puts the password in the DSN. Logs and /health must not carry it."""
    if "@" not in dsn:
        return dsn or "unset"
    scheme, _, rest = dsn.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: ConnectionPool | None = None

    @property
    def configured(self) -> bool:
        """Whether there is a database to talk to at all.

        Callers that must degrade rather than fail — conversation storage is
        one — ask this instead of catching the exception, so "no DSN in this
        checkout" stays a quiet no-op and a *broken* DSN still raises.
        """
        return bool(self._settings.database_url)

    @property
    def location(self) -> str:
        return redact(self._settings.database_url)

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
            # HNSW search breadth, set once per connection rather than per
            # query. It used to be a `SET LOCAL` inside an explicit transaction
            # around every search, which is correct and cost three extra
            # network round trips — measured against Neon, that turned a 70 ms
            # search into a 273 ms one, and it was the single largest item in
            # the 200 ms budget of docs/04-latency.md. Session scope is safe
            # here in a way a per-request value would not be: every checkout
            # sets the same number, so there is nothing to leak between them.
            cur.execute(
                sql.SQL("SET hnsw.ef_search = {}").format(
                    sql.Literal(
                        int(
                            max(
                                self._settings.hnsw_ef_search,
                                self._settings.search_limit,
                            )
                        )
                    )
                )
            )
            # The extension has to exist before `register_vector` can look the
            # type up in the catalogue — and on a fresh database nothing has
            # created it yet, because the pool builds its first connection
            # *before* any schema call is ever handed one. Doing it here rather
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
                raise DatabaseUnavailable(
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
                raise DatabaseUnavailable(str(error)) from error
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


@lru_cache
def get_db() -> Database:
    return Database(get_settings())
