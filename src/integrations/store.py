"""Who connected what, in the same Postgres as everything else.

**Composio is the connector, and the user connects it.** There is no
project-wide Composio account behind this app: each signed-in person brings
their own API key. That key is *not* here — it lives in `connector_accounts`
with every other connector's credentials (`src/connectors/store.py`), sealed,
because there is nothing Composio-specific about storing a secret.

What is here is the Composio-specific part: the toolkits connected through
somebody's Composio project, which no other connector has an equivalent of.

Two tables:

    integration_auth_configs   (user_id, toolkit) → the auth config it connects
                               through. Keyed by user because an auth config id
                               only means anything inside the project that
                               created it: user A's `ac_…` is not a thing user
                               B's Composio has ever heard of.

    integration_connections    user_id → the connection opened under it.
                               One row per (user, toolkit); reconnecting a
                               toolkit updates the row rather than growing a
                               second one.

Composio remains the authority on whether a connection still works. So why
write the last two down at all? Because the question this app asks is not "is
this token valid" — Composio answers that — it is *"whose is it"*. Keeping the
mapping in our own database means the ownership check is a `WHERE user_id = %s`
against a row we wrote, not a filter argument we trusted a remote list call to
have honoured. It also means the panel can say "Gmail, connected" while
Composio is down, and that a connection opened but never finished still has a
row to reconcile against when the browser comes back from consent.

Status is a *cache* of Composio's, refreshed by `reconcile` on every list.
Nothing here decides that a connection is live — it only remembers what
Composio last said, so a dead upstream degrades to stale rather than to blank.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable, Sequence

from src.core.db import Database, get_db

log = logging.getLogger("vec.integrations")

AUTH_CONFIGS = "integration_auth_configs"
CONNECTIONS = "integration_connections"

# Composio's own vocabulary, kept verbatim rather than mapped to something of
# our own. A status this app has never heard of is still worth storing and
# showing: the alternative is silently rendering a REVOKED account as fine
# because it did not match an enum written before that state existed.
ACTIVE = "ACTIVE"
PENDING = ("INITIALIZING", "INITIATED")
DEAD = ("FAILED", "EXPIRED", "REVOKED", "INACTIVE")


def new_connection_id() -> str:
    return f"icn_{uuid.uuid4().hex}"


@dataclass(slots=True)
class Connection:
    id: str
    user_id: str
    toolkit: str
    auth_config_id: str
    connected_account_id: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    @property
    def live(self) -> bool:
        return self.status == ACTIVE

    @property
    def pending(self) -> bool:
        return self.status in PENDING


_SCHEMA = f"""
-- `integration_auth_configs` was keyed by toolkit alone when this app had one
-- Composio project behind it. It has a per-user key now, because an `ac_…`
-- means nothing outside the project that made it — and a row from the old
-- shape would point every user at somebody else's auth config. Every row is a
-- rebuildable cache of Composio's own state, so dropping is safe and is the
-- honest fix; the next connect recreates what it needs.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = '{AUTH_CONFIGS}'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = '{AUTH_CONFIGS}' AND column_name = 'user_id'
    ) THEN
        DROP TABLE {AUTH_CONFIGS};
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {AUTH_CONFIGS} (
    user_id             TEXT NOT NULL,
    toolkit             TEXT NOT NULL,
    auth_config_id      TEXT NOT NULL,
    composio_managed    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, toolkit)
);

CREATE TABLE IF NOT EXISTS {CONNECTIONS} (
    id                   TEXT PRIMARY KEY,
    -- A Clerk `sub`, and only ever one that arrived on a verified token.
    -- There is deliberately no session_id twin here: conversations may belong
    -- to a browser, but a Gmail account may not.
    user_id              TEXT NOT NULL,
    toolkit              TEXT NOT NULL,
    auth_config_id       TEXT NOT NULL,
    connected_account_id TEXT,
    status               TEXT NOT NULL DEFAULT 'INITIALIZING',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Reconnecting Slack replaces the row it already had. Without this a user
    -- who cancels consent twice accumulates dead rows the panel has to dedupe.
    CONSTRAINT {CONNECTIONS}_one_per_toolkit UNIQUE (user_id, toolkit)
);

-- Every read here is "this user's connections, newest first".
CREATE INDEX IF NOT EXISTS {CONNECTIONS}_user
    ON {CONNECTIONS} (user_id, updated_at DESC);

-- The callback comes back naming a Composio account id and nothing else, so
-- that lookup needs to be an index rather than a scan.
CREATE INDEX IF NOT EXISTS {CONNECTIONS}_account
    ON {CONNECTIONS} (connected_account_id) WHERE connected_account_id IS NOT NULL;
"""

_COLUMNS = (
    "id, user_id, toolkit, auth_config_id, connected_account_id, status, "
    "created_at, updated_at"
)


def _connection(row: Sequence[Any]) -> Connection:
    return Connection(
        id=row[0],
        user_id=row[1],
        toolkit=row[2],
        auth_config_id=row[3],
        connected_account_id=row[4],
        status=row[5],
        created_at=row[6],
        updated_at=row[7],
    )


class IntegrationStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    def ensure_schema(self) -> None:
        """Idempotent, and cheap enough to call on every write path.

        Not set on failure, deliberately — a database that comes up late
        should start working on the next click rather than stay broken because
        one boot missed it. Same reasoning as `ChatStore.ensure_schema`.
        """
        if self._ensured:
            return
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    # ---- Composio going away ---------------------------------------------

    def purge(self, user_id: str) -> int:
        """Forget everything that only made sense while Composio was connected.

        Called when the Composio *connector* is disconnected. Auth configs and
        connections name ids inside a Composio project this app can no longer
        reach, so leaving them would show a panel full of rows that cannot be
        acted on. Both go in one transaction.

        The credentials themselves are not this store's to delete — they live
        in `connector_accounts`, and the connector service removes them.
        """
        if not user_id:
            return 0

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(f"DELETE FROM {CONNECTIONS} WHERE user_id = %s", (user_id,))
            removed = cur.rowcount
            cur.execute(f"DELETE FROM {AUTH_CONFIGS} WHERE user_id = %s", (user_id,))

        return removed

    # ---- auth configs ---------------------------------------------------

    def auth_config(self, user_id: str, toolkit: str) -> tuple[str, bool] | None:
        """The auth config this user's toolkit connects through, if made.

        Per user, not project-wide: an auth config id only means something
        inside the Composio project that created it, and every user here has
        their own.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT auth_config_id, composio_managed FROM {AUTH_CONFIGS}
                WHERE user_id = %s AND toolkit = %s
                """,
                (user_id, toolkit),
            )
            row = cur.fetchone()

        return (row[0], bool(row[1])) if row else None

    def remember_auth_config(
        self, user_id: str, toolkit: str, auth_config_id: str, *, composio_managed: bool
    ) -> None:
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {AUTH_CONFIGS} (user_id, toolkit, auth_config_id, composio_managed)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, toolkit) DO UPDATE
                    SET auth_config_id   = EXCLUDED.auth_config_id,
                        composio_managed = EXCLUDED.composio_managed
                """,
                (user_id, toolkit, auth_config_id, composio_managed),
            )
            conn.commit()

    def forget_auth_config(self, user_id: str, toolkit: str) -> None:
        """Drop a cached auth config that Composio no longer recognises.

        Deleting an auth config in the dashboard leaves this table pointing at
        an id that 404s, and every subsequent connect would fail the same way
        forever. Clearing it turns the next attempt into a fresh create.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {AUTH_CONFIGS} WHERE user_id = %s AND toolkit = %s",
                (user_id, toolkit),
            )
            conn.commit()

    # ---- connections ----------------------------------------------------

    def open(
        self,
        *,
        user_id: str,
        toolkit: str,
        auth_config_id: str,
        connected_account_id: str | None,
        status: str,
    ) -> Connection:
        """Record a connection request, replacing any earlier one for this
        toolkit.

        Written *before* the browser is sent to consent, so a user who closes
        the tab on Composio's screen still leaves a row this app can reconcile
        instead of a connection nobody knows about.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {CONNECTIONS}
                    (id, user_id, toolkit, auth_config_id, connected_account_id, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, toolkit) DO UPDATE
                    SET auth_config_id       = EXCLUDED.auth_config_id,
                        connected_account_id = EXCLUDED.connected_account_id,
                        status               = EXCLUDED.status,
                        updated_at           = now()
                RETURNING {_COLUMNS}
                """,
                (
                    new_connection_id(),
                    user_id,
                    toolkit,
                    auth_config_id,
                    connected_account_id,
                    status,
                ),
            )
            row = cur.fetchone()
            conn.commit()

        return _connection(row)

    def list(self, user_id: str) -> list[Connection]:
        if not user_id:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_COLUMNS} FROM {CONNECTIONS}
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        return [_connection(row) for row in rows]

    def get(self, user_id: str, toolkit: str) -> Connection | None:
        """One toolkit's connection, scoped to its owner.

        `user_id` is in the predicate rather than checked afterwards for the
        same reason it is in the conversation store's: a check that lives in
        the SQL cannot be forgotten by a caller.
        """
        if not user_id or not toolkit:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {CONNECTIONS} WHERE user_id = %s AND toolkit = %s",
                (user_id, toolkit),
            )
            row = cur.fetchone()

        return _connection(row) if row else None

    def mark(
        self,
        *,
        user_id: str,
        toolkit: str,
        status: str,
        connected_account_id: str | None = None,
    ) -> Connection | None:
        """Write back what Composio last said about a connection.

        COALESCE on the account id because reconciliation sometimes learns the
        status before the id (a request polled while still INITIALIZING), and
        overwriting a known id with NULL would lose the only handle we have on
        it.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {CONNECTIONS}
                SET status               = %(status)s,
                    connected_account_id = COALESCE(%(account)s, connected_account_id),
                    updated_at           = now()
                WHERE user_id = %(user_id)s AND toolkit = %(toolkit)s
                RETURNING {_COLUMNS}
                """,
                {
                    "status": status,
                    "account": connected_account_id,
                    "user_id": user_id,
                    "toolkit": toolkit,
                },
            )
            row = cur.fetchone()
            conn.commit()

        return _connection(row) if row else None

    def reconcile(self, user_id: str, live: Iterable[tuple[str, str, str]]) -> None:
        """Bring this user's rows in line with what Composio reports.

        `live` is (toolkit, connected_account_id, status), straight off
        Composio's list. Toolkits it does not mention are marked REVOKED rather
        than deleted: a row that disappears takes the audit trail with it, and
        "you disconnected this" is worth being able to show.

        Scoped to one user per call. A bulk reconcile across everybody would be
        cheaper per row and would also mean one buggy query could rewrite
        somebody else's connections, which is not a trade worth taking here.
        """
        if not user_id:
            return

        rows = list(live)
        self.ensure_schema()

        with self._db.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            for toolkit, account_id, status in rows:
                cur.execute(
                    f"""
                    INSERT INTO {CONNECTIONS}
                        (id, user_id, toolkit, auth_config_id, connected_account_id, status)
                    VALUES (%(id)s, %(user_id)s, %(toolkit)s, '', %(account)s, %(status)s)
                    ON CONFLICT (user_id, toolkit) DO UPDATE
                        SET connected_account_id = EXCLUDED.connected_account_id,
                            status               = EXCLUDED.status,
                            updated_at           = now()
                    """,
                    {
                        "id": new_connection_id(),
                        "user_id": user_id,
                        "toolkit": toolkit,
                        "account": account_id,
                        "status": status,
                    },
                )

            # Anything we think is connected that Composio did not list has
            # been revoked out from under us — from the provider's own security
            # page, or by a delete in the Composio dashboard.
            known = [toolkit for toolkit, _, _ in rows]
            cur.execute(
                f"""
                UPDATE {CONNECTIONS} SET status = 'REVOKED', updated_at = now()
                WHERE user_id = %(user_id)s
                  AND status = ANY(%(alive)s)
                  AND NOT (toolkit = ANY(%(known)s))
                """,
                {
                    "user_id": user_id,
                    "alive": [ACTIVE, *PENDING],
                    "known": known,
                },
            )

    def delete(self, user_id: str, toolkit: str) -> Connection | None:
        """Forget a connection, returning what was there so the caller can
        also delete it at Composio.

        Ordering matters and is the caller's problem: this row is the only
        record of the Composio account id, so it is read out here and removed,
        and the service deletes upstream first.
        """
        if not user_id or not toolkit:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {CONNECTIONS} WHERE user_id = %s AND toolkit = %s RETURNING {_COLUMNS}",
                (user_id, toolkit),
            )
            row = cur.fetchone()
            conn.commit()

        return _connection(row) if row else None


@lru_cache
def get_integration_store() -> IntegrationStore:
    return IntegrationStore(get_db())
