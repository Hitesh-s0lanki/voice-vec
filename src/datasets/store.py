"""One table: the datasets a user has attached, and what was understood of each.

    agent_datasets   (user_id, dataset_id) → url, profile JSONB, cards, local file

`PRIMARY KEY (user_id, dataset_id)` is the ownership model and the multiplicity
answer in one line: a person may attach as many datasets as they like, each
under an id derived from its URL, and re-adding the same URL replaces rather
than accumulates. Two people who add the same public dataset get a row each and
a file each — no sharing, for the same reason `connector_accounts` has none.

**No foreign key, unlike `connector_profiles`.** A connector profile describes a
credential and must die with it; a dataset is a public URL and nothing here
depends on a row in another table. What it *does* depend on is a file on local
disk, which is why `local_path` is a stored column rather than a derived one:
deleting a row has to be able to delete the file, and a path recomputed from a
slug is a path that stops matching the moment the naming rule is edited.

**Two cards, stored rather than rendered on read.** `card` goes into a system
prompt on every turn and `schema_card` goes to the SQL writer. Both are pure
functions of `profile`, and both are re-derived from JSONB inside a 200 ms
budget if they are not written down — the same argument
`connector_profiles.card` makes, twice.

Nothing here holds a credential, because nothing in this feature has one: only
public URLs are accepted (`src/datasets/source.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from src.core.db import Database, get_db

log = logging.getLogger("vec.datasets")

DATASETS = "agent_datasets"


@dataclass(slots=True)
class DatasetRow:
    user_id: str
    dataset_id: str
    url: str
    kind: str
    location: str
    status: str
    profile: dict[str, Any]
    card: str
    schema_card: str
    local_path: str
    error: str
    rows: int
    bytes: int
    built_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def stale(self, ttl: timedelta) -> bool:
        if self.built_at is None:
            return True
        return datetime.now(timezone.utc) - self.built_at > ttl


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {DATASETS} (
    -- A Clerk `sub`, from a verified token. Same rule as connectors: a saved
    -- conversation may belong to a browser, an attached dataset may not.
    user_id      TEXT NOT NULL,
    -- Derived from the URL by src/datasets/source.py, so re-adding the same
    -- dataset replaces its row instead of making a second one.
    dataset_id   TEXT NOT NULL,
    url          TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT '',
    location     TEXT NOT NULL DEFAULT '',
    -- pending | ok | degraded | failed. `pending` is the state a row is born
    -- in: adding writes the row and returns, and the builder fills it in.
    status       TEXT NOT NULL DEFAULT 'pending',
    -- The whole `DatasetProfile`, versioned. Read back through
    -- `DatasetProfile.from_json`, which returns None for a version this build
    -- cannot read — re-measuring a local file beats migrating a stored shape.
    profile      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    -- Goes into a system prompt every turn: is this dataset worth asking?
    card         TEXT NOT NULL DEFAULT '',
    -- Goes only to the SQL writer, and only on turns that query.
    schema_card  TEXT NOT NULL DEFAULT '',
    -- The materialised DuckDB file. Stored, not derived: deleting a row must
    -- be able to delete the file it wrote.
    local_path   TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    rows         BIGINT NOT NULL DEFAULT 0,
    bytes        BIGINT NOT NULL DEFAULT 0,
    built_at     TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, dataset_id)
);

-- Every read is "this user's datasets".
CREATE INDEX IF NOT EXISTS {DATASETS}_user ON {DATASETS} (user_id);
"""

_COLUMNS = (
    "user_id, dataset_id, url, kind, location, status, profile, card, schema_card, "
    "local_path, error, rows, bytes, built_at, created_at, updated_at"
)


def _row(row: Sequence[Any]) -> DatasetRow:
    return DatasetRow(
        user_id=row[0],
        dataset_id=row[1],
        url=row[2],
        kind=row[3],
        location=row[4],
        status=row[5],
        profile=dict(row[6] or {}),
        card=row[7] or "",
        schema_card=row[8] or "",
        local_path=row[9] or "",
        error=row[10] or "",
        rows=int(row[11] or 0),
        bytes=int(row[12] or 0),
        built_at=row[13],
        created_at=row[14],
        updated_at=row[15],
    )


class DatasetStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    def ensure_schema(self) -> None:
        """Idempotent, and not latched on failure.

        Same reasoning as `ConnectorStore.ensure_schema`: a database that comes
        up late should start working on the next request rather than stay
        broken because one boot missed it.
        """
        if self._ensured:
            return
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    def claim(self, user_id: str, dataset_id: str, *, url: str, kind: str, location: str) -> DatasetRow:
        """Write the pending row, before anything is pulled.

        Adding returns immediately and the build runs behind it, so the row has
        to exist first — otherwise a panel that polls sees nothing between the
        click and the first byte, and a second click starts a second build.
        """
        if not user_id or not dataset_id:
            raise ValueError("a dataset needs a user_id and a dataset_id")

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {DATASETS} (user_id, dataset_id, url, kind, location, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (user_id, dataset_id) DO UPDATE
                    SET url        = EXCLUDED.url,
                        kind       = EXCLUDED.kind,
                        location   = EXCLUDED.location,
                        status     = 'pending',
                        error      = '',
                        updated_at = now()
                RETURNING {_COLUMNS}
                """,
                (user_id, dataset_id, url, kind, location),
            )
            row = cur.fetchone()
            conn.commit()
        return _row(row)

    def finish(
        self,
        user_id: str,
        dataset_id: str,
        *,
        status: str,
        profile: dict[str, Any],
        card: str,
        schema_card: str,
        local_path: str,
        rows: int,
        bytes_: int,
        error: str = "",
    ) -> DatasetRow | None:
        """The built result. Only touches a row that already exists.

        `UPDATE` rather than upsert, deliberately: a build whose row was deleted
        while it ran has had its dataset removed, and re-inserting it would
        resurrect something the user asked to be gone.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {DATASETS}
                   SET status = %s, profile = %s, card = %s, schema_card = %s,
                       local_path = %s, rows = %s, bytes = %s, error = %s,
                       built_at = now(), updated_at = now()
                 WHERE user_id = %s AND dataset_id = %s
                RETURNING {_COLUMNS}
                """,
                (
                    status,
                    Jsonb(profile),
                    card,
                    schema_card,
                    local_path,
                    int(rows),
                    int(bytes_),
                    error,
                    user_id,
                    dataset_id,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        return _row(row) if row else None

    def fail(self, user_id: str, dataset_id: str, error: str) -> None:
        """Record why a build failed, without losing the row.

        The row survives so the panel can show the reason and offer a rebuild.
        Deleting it on failure would leave somebody who pasted a URL with no
        trace of having done so and no message about why.
        """
        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {DATASETS}
                   SET status = 'failed', error = %s, updated_at = now()
                 WHERE user_id = %s AND dataset_id = %s
                """,
                (error[:500], user_id, dataset_id),
            )
            conn.commit()

    def get(self, user_id: str, dataset_id: str) -> DatasetRow | None:
        if not user_id or not dataset_id:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {DATASETS} WHERE user_id = %s AND dataset_id = %s",
                (user_id, dataset_id),
            )
            row = cur.fetchone()
        return _row(row) if row else None

    def list(self, user_id: str) -> list[DatasetRow]:
        if not user_id:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {DATASETS} WHERE user_id = %s ORDER BY created_at",
                (user_id,),
            )
            rows = cur.fetchall()
        return [_row(row) for row in rows]

    def delete(self, user_id: str, dataset_id: str) -> str | None:
        """Remove the row and hand back the file the caller must now unlink.

        Three-valued, and the third value is the point. `None` means there was
        no such row; `""` means there was one and it has no file yet; a path
        means there is a file to remove. Collapsing the last two into `""` made
        removing a *still-building* dataset — the ordinary way somebody cancels
        a URL they mistyped — report `removed: false` while the row was in fact
        gone, because a pending row's `local_path` is empty.

        The path comes back rather than the file being deleted here, because
        this module owns a table and nothing else. A store that also touched
        the filesystem would be the second place that knows where datasets
        live, and the two would disagree the first time either changed.
        """
        if not user_id or not dataset_id:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {DATASETS} WHERE user_id = %s AND dataset_id = %s "
                "RETURNING local_path",
                (user_id, dataset_id),
            )
            row = cur.fetchone()
            conn.commit()
        return (row[0] or "") if row else None


@lru_cache
def get_dataset_store() -> DatasetStore:
    return DatasetStore(get_db())
