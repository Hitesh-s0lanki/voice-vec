"""One table: what this app has understood about each connected store.

    connector_profiles   (user_id, connector) → profile JSONB, card, fingerprint

The same key as `connector_accounts`, and a **foreign key to it, cascading**.
That is the whole lifecycle: a profile describes a specific connection, so
disconnecting deletes the understanding along with the credential, and no
cleanup path has to remember to. A profile that outlived its account would be
this app telling an agent about an index it can no longer reach.

`fingerprint` is what makes a stored profile trustworthy. It is a digest of the
sealed credential blob the profile was built from, so rotating a key, pointing
the connector at a different index, or reconnecting to a different table all
change it — and a profile whose fingerprint no longer matches the account is
treated as absent rather than served. `src/rag/backends/resolve.py` invalidates
its backend cache the same way; this is that trick, persisted.

**Nothing here holds a credential.** The digest is one-way and the blob it is
taken over is already ciphertext, so this table can be read by an operator
without the master key and still say nothing about anybody's Pinecone.

`card` is stored beside the JSON rather than rendered on read. It is what goes
into a system prompt on the voice path, and rendering it there would mean
parsing a JSONB blob and re-deriving a string inside a 200 ms budget, every
turn, to produce the same bytes every time.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from src.connectors.store import ACCOUNTS, ConnectorStore, get_connector_store
from src.core.db import Database, get_db

log = logging.getLogger("vec.connectors.profile")

PROFILES = "connector_profiles"


def fingerprint(sealed: str) -> str:
    """A stable digest of the sealed credential blob.

    Over the ciphertext, not the plaintext: this runs wherever a profile is
    written or checked, including places that have no business decrypting
    anything, and a digest of a *secret* is a thing worth not creating even
    when it cannot be reversed.
    """
    return hashlib.sha256((sealed or "").encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class ProfileRow:
    user_id: str
    connector: str
    status: str
    profile: dict[str, Any]
    card: str
    fingerprint: str
    error: str
    profiled_at: datetime | None
    updated_at: datetime

    def matches(self, sealed: str) -> bool:
        """Was this understanding built from the credentials in force now?"""
        return bool(self.fingerprint) and self.fingerprint == fingerprint(sealed)

    def stale(self, ttl: timedelta) -> bool:
        if self.profiled_at is None:
            return True
        return datetime.now(timezone.utc) - self.profiled_at > ttl


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {PROFILES} (
    user_id      TEXT NOT NULL,
    connector    TEXT NOT NULL,
    -- pending | ok | degraded | failed. `pending` is the state a row is born
    -- in: connecting writes the row and returns, and the probe fills it in.
    status       TEXT NOT NULL DEFAULT 'pending',
    -- The whole `Profile`, versioned. Read back through `Profile.from_json`,
    -- which returns None for a version this build cannot read — a profile is a
    -- cache of somebody else's database and re-probing beats migrating.
    profile      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    -- The agent-facing rendering, stored rather than derived on read: this is
    -- read on the voice path and rendering it there would redo the same work
    -- every turn.
    card         TEXT NOT NULL DEFAULT '',
    -- Digest of the sealed credential blob this was built from. A profile
    -- whose fingerprint no longer matches the account is not served.
    fingerprint  TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    profiled_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, connector),
    -- The lifecycle, enforced rather than remembered: disconnecting a
    -- connector deletes what this app understood about it.
    CONSTRAINT {PROFILES}_account
        FOREIGN KEY (user_id, connector) REFERENCES {ACCOUNTS} (user_id, connector)
        ON DELETE CASCADE
);

-- Every read is "this user's profiles", the same as the accounts beside them.
CREATE INDEX IF NOT EXISTS {PROFILES}_user ON {PROFILES} (user_id);
"""

_COLUMNS = (
    "user_id, connector, status, profile, card, fingerprint, error, profiled_at, updated_at"
)


def _row(row: Sequence[Any]) -> ProfileRow:
    return ProfileRow(
        user_id=row[0],
        connector=row[1],
        status=row[2],
        profile=dict(row[3] or {}),
        card=row[4] or "",
        fingerprint=row[5] or "",
        error=row[6] or "",
        profiled_at=row[7],
        updated_at=row[8],
    )


class ProfileStore:
    def __init__(self, db: Database | None = None, accounts: ConnectorStore | None = None) -> None:
        self._db = db or get_db()
        self._accounts = accounts or get_connector_store()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    def ensure_schema(self) -> None:
        """Accounts first — the foreign key needs the table it points at.

        Not memoised on failure, for the same reason `ConnectorStore` is not: a
        database that comes up late should start working on the next call
        rather than stay broken because one boot missed it.
        """
        if self._ensured:
            return
        self._accounts.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    def mark_pending(self, user_id: str, connector: str, *, sealed: str) -> None:
        """Claim the row before probing, so the panel can say "understanding…".

        Written with the *new* fingerprint straight away. A reconnect that
        changed the index must not leave the old profile readable for the
        seconds the probe takes: it describes a store these credentials no
        longer point at, and serving it is worse than serving nothing.
        """
        self._write(
            user_id,
            connector,
            status="pending",
            profile={},
            card="",
            fingerprint=fingerprint(sealed),
            error="",
            profiled_at=None,
        )

    def save(
        self,
        user_id: str,
        connector: str,
        *,
        status: str,
        profile: dict[str, Any],
        card: str,
        sealed: str,
        error: str = "",
        profiled_at: datetime | None = None,
    ) -> ProfileRow | None:
        return self._write(
            user_id,
            connector,
            status=status,
            profile=profile,
            card=card,
            fingerprint=fingerprint(sealed),
            error=error,
            profiled_at=profiled_at or datetime.now(timezone.utc),
        )

    def _write(
        self,
        user_id: str,
        connector: str,
        *,
        status: str,
        profile: dict[str, Any],
        card: str,
        fingerprint: str,
        error: str,
        profiled_at: datetime | None,
    ) -> ProfileRow | None:
        if not user_id or not connector:
            return None

        self.ensure_schema()
        try:
            with self._db.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {PROFILES}
                        (user_id, connector, status, profile, card, fingerprint, error, profiled_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, connector) DO UPDATE
                        SET status      = EXCLUDED.status,
                            profile     = EXCLUDED.profile,
                            card        = EXCLUDED.card,
                            fingerprint = EXCLUDED.fingerprint,
                            error       = EXCLUDED.error,
                            profiled_at = EXCLUDED.profiled_at,
                            updated_at  = now()
                    RETURNING {_COLUMNS}
                    """,
                    (
                        user_id,
                        connector,
                        status,
                        Jsonb(profile),
                        card,
                        fingerprint,
                        error[:500],
                        profiled_at,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        except Exception as error_:
            # A profile is an enhancement. Failing to store one may not break
            # the connector it describes, and the foreign key means a race
            # against a disconnect lands here rather than anywhere louder.
            log.warning("could not store the %s profile for %s: %s", connector, user_id, error_)
            return None

        return _row(row) if row else None

    def get(self, user_id: str, connector: str) -> ProfileRow | None:
        if not user_id or not connector:
            return None
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {PROFILES} WHERE user_id = %s AND connector = %s",
                (user_id, connector),
            )
            row = cur.fetchone()
        return _row(row) if row else None

    def list(self, user_id: str) -> list[ProfileRow]:
        if not user_id:
            return []
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {PROFILES} WHERE user_id = %s ORDER BY connector",
                (user_id,),
            )
            rows = cur.fetchall()
        return [_row(row) for row in rows]

    def delete(self, user_id: str, connector: str) -> bool:
        """Rarely needed — the cascade does this — but explicit for a re-probe."""
        if not user_id or not connector:
            return False
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {PROFILES} WHERE user_id = %s AND connector = %s",
                (user_id, connector),
            )
            removed = cur.rowcount
            conn.commit()
        return bool(removed)


@lru_cache
def get_profile_store() -> ProfileStore:
    return ProfileStore(get_db(), get_connector_store())
