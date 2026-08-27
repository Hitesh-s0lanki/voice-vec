"""One table: which connectors a user has attached, and their sealed credentials.

    connector_accounts   (user_id, connector) → credentials, sealed

`PRIMARY KEY (user_id, connector)` is the whole ownership model. One Pinecone
per person, one Composio per person, and reconnecting replaces rather than
accumulates — enforced by the key rather than by a rule some call site
remembers.

The `credentials` column holds Fernet ciphertext over a JSON object and nothing
here ever opens it. Opening is `crypto.Sealed`'s job one layer up, which means
no query, log line or exception repr from this module can contain a credential.

`hints` beside it is the readable half: the non-secret fields as typed, plus the
last four characters of the secret one. It exists so the panel can say "Pinecone
· vec-chunks · ····8fa2" without decrypting anything, and so an operator looking
at the table can tell which row is which without the master key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from src.core.db import Database, get_db

log = logging.getLogger("vec.connectors")

ACCOUNTS = "connector_accounts"


@dataclass(slots=True)
class Account:
    """One connector, attached to one user.

    `credentials` is ciphertext and stays that way through this layer.
    """

    user_id: str
    connector: str
    credentials: str
    hints: dict[str, str]
    created_at: datetime
    updated_at: datetime


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {ACCOUNTS} (
    -- A Clerk `sub`, from a verified token. There is deliberately no
    -- session_id twin: a conversation may belong to a browser, an API key
    -- may not.
    user_id      TEXT NOT NULL,
    -- A slug from src/connectors/registry.py.
    connector    TEXT NOT NULL,
    -- Fernet over a JSON object. Never credentials as typed.
    credentials  TEXT NOT NULL,
    -- The readable half: non-secret fields, plus the last four characters of
    -- the secret one. Enough to label a row without the master key.
    hints        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, connector)
);

-- Every read is "this user's connectors".
CREATE INDEX IF NOT EXISTS {ACCOUNTS}_user ON {ACCOUNTS} (user_id);

-- `composio_accounts` held exactly this, for exactly one connector, before
-- there was more than one. Nothing migrates across: the credential format
-- changed from a bare sealed string to a sealed JSON object, and inventing a
-- shape for rows we cannot read would be worse than one reconnect.
DROP TABLE IF EXISTS composio_accounts;
"""

_COLUMNS = "user_id, connector, credentials, hints, created_at, updated_at"


def _account(row: Sequence[Any]) -> Account:
    return Account(
        user_id=row[0],
        connector=row[1],
        credentials=row[2],
        hints=dict(row[3] or {}),
        created_at=row[4],
        updated_at=row[5],
    )


class ConnectorStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    def ensure_schema(self) -> None:
        """Idempotent, and cheap enough to call on every path that touches it.

        Not set on failure, deliberately: a database that comes up late should
        start working on the next click rather than stay broken because one
        boot missed it.
        """
        if self._ensured:
            return
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    def save(
        self, user_id: str, connector: str, *, credentials: str, hints: dict[str, str]
    ) -> Account:
        """Attach a connector, replacing whatever was there for it."""
        if not user_id or not connector:
            raise ValueError("a connector account needs a user_id and a connector")

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {ACCOUNTS} (user_id, connector, credentials, hints)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, connector) DO UPDATE
                    SET credentials = EXCLUDED.credentials,
                        hints       = EXCLUDED.hints,
                        updated_at  = now()
                RETURNING {_COLUMNS}
                """,
                (user_id, connector, credentials, Jsonb(hints)),
            )
            row = cur.fetchone()
            conn.commit()

        return _account(row)

    def get(self, user_id: str, connector: str) -> Account | None:
        if not user_id or not connector:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {ACCOUNTS} WHERE user_id = %s AND connector = %s",
                (user_id, connector),
            )
            row = cur.fetchone()

        return _account(row) if row else None

    def list(self, user_id: str) -> list[Account]:
        if not user_id:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {ACCOUNTS} WHERE user_id = %s ORDER BY connector",
                (user_id,),
            )
            rows = cur.fetchall()

        return [_account(row) for row in rows]

    def delete(self, user_id: str, connector: str) -> bool:
        if not user_id or not connector:
            return False

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {ACCOUNTS} WHERE user_id = %s AND connector = %s",
                (user_id, connector),
            )
            removed = cur.rowcount
            conn.commit()

        return bool(removed)


@lru_cache
def get_connector_store() -> ConnectorStore:
    return ConnectorStore(get_db())
