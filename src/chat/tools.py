"""What the agent ran, written down beside the conversation that caused it.

A tool call is the one thing a spoken turn does that has an effect *outside*
this app. A message can be re-read; an email is sent. So it is recorded with
more care than the text around it, and for three different readers:

  the user      "what did it actually do?" — the panel shows the calls under
                the turn that made them, named and timed
  the operator  a tool that fails for everybody looks like the model being
                unhelpful until there is a table saying otherwise
  whoever asks  an agent with access to somebody's mailbox needs an audit
                trail that is not the model's own account of itself

    tool_calls ─── conversation_id → conversations.id  (ON DELETE CASCADE)
               └── turn_id, so a call sits under the exchange that caused it

**Arguments are stored; results are not stored whole.** The arguments are what
the agent *decided*, which is the interesting half and is small. A result can be
an entire inbox page and belongs to the provider, not here — so only its size,
its status and any error are kept. Storing tool output wholesale would turn this
table into an uncontrolled copy of everything the agent has ever read, which is
a worse thing to hold than the credential that reached it.

The same rule as the rest of chat storage applies: **writing must never be able
to break a turn.** Every call here runs off the voice path through the session's
queue, and a dead database costs the listener nothing.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Sequence

from psycopg.types.json import Jsonb

from src.chat.store import CONVERSATIONS
from src.core.db import Database, get_db

log = logging.getLogger("vec.chat.tools")

TOOL_CALLS = "tool_calls"

# Arguments do go in, so they are bounded. A model that emits a 100 KB argument
# blob is malfunctioning and should not be able to fill a table over it.
MAX_ARGUMENT_CHARS = 8_000


def new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex}"


@dataclass(slots=True)
class ToolCallRow:
    id: str
    conversation_id: str | None
    turn_id: str | None
    user_id: str | None
    toolkit: str | None
    slug: str
    arguments: dict
    status: str
    error: str | None
    result_bytes: int | None
    latency_ms: float | None
    created_at: datetime

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TOOL_CALLS} (
    id              TEXT PRIMARY KEY,
    -- Nullable: a turn can run a tool before its conversation row exists (no
    -- DATABASE_URL, or a take that never opened one). Losing the link is
    -- better than losing the record.
    conversation_id TEXT REFERENCES {CONVERSATIONS} (id) ON DELETE CASCADE,
    turn_id         TEXT,
    -- The Clerk `sub` the tool ran as. Denormalised on purpose: this is the
    -- audit question — "who did this run for" — and it must survive the
    -- conversation being deleted out from under it.
    user_id         TEXT,
    toolkit         TEXT,
    -- Composio's own, e.g. GMAIL_SEND_EMAIL.
    slug            TEXT NOT NULL,
    -- What the agent decided. The interesting half, and small.
    arguments       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    -- ok · failed · refused
    status          TEXT NOT NULL,
    error           TEXT,
    -- The size of what came back, not what came back. See the module docstring.
    result_bytes    INTEGER,
    latency_ms      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every read is "the calls in this conversation, in the order they ran".
CREATE INDEX IF NOT EXISTS {TOOL_CALLS}_thread
    ON {TOOL_CALLS} (conversation_id, created_at, id);

-- And the audit one: everything this account has ever had run for it.
CREATE INDEX IF NOT EXISTS {TOOL_CALLS}_user
    ON {TOOL_CALLS} (user_id, created_at DESC) WHERE user_id IS NOT NULL;
"""

_COLUMNS = (
    "id, conversation_id, turn_id, user_id, toolkit, slug, arguments, status, "
    "error, result_bytes, latency_ms, created_at"
)


def _row(row: Sequence[Any]) -> ToolCallRow:
    return ToolCallRow(
        id=row[0],
        conversation_id=row[1],
        turn_id=row[2],
        user_id=row[3],
        toolkit=row[4],
        slug=row[5],
        arguments=dict(row[6] or {}),
        status=row[7],
        error=row[8],
        result_bytes=int(row[9]) if row[9] is not None else None,
        latency_ms=float(row[10]) if row[10] is not None else None,
        created_at=row[11],
    )


def toolkit_of(slug: str) -> str:
    """`GMAIL_SEND_EMAIL` → `gmail`.

    Composio's convention, and stored as its own column so the panel can group
    by service without parsing a slug every time it renders a row.
    """
    head, _, _ = (slug or "").partition("_")
    return head.lower()


def trim(arguments: dict) -> dict:
    """Bound what goes into the column, without losing what it was.

    A single oversized argument is replaced by a marker rather than the whole
    call being dropped: knowing that `GMAIL_SEND_EMAIL` ran with a body too
    large to keep is most of the value of the record.
    """
    try:
        encoded = json.dumps(arguments, default=str)
    except Exception:
        return {"_unserialisable": True}

    if len(encoded) <= MAX_ARGUMENT_CHARS:
        return arguments

    trimmed: dict[str, Any] = {}
    for key, value in arguments.items():
        rendered = json.dumps(value, default=str)
        trimmed[key] = value if len(rendered) <= 512 else f"…{len(rendered)} chars"
    return trimmed


class ToolCallStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    def ensure_schema(self) -> None:
        """Idempotent, and not flagged on failure — a database that comes up
        late should start working on the next turn."""
        if self._ensured:
            return
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    def record(
        self,
        *,
        slug: str,
        status: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        user_id: str | None = None,
        arguments: dict | None = None,
        error: str | None = None,
        result_bytes: int | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallRow | None:
        """Write one call down. Returns None when there is no database."""
        if not slug:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {TOOL_CALLS}
                    (id, conversation_id, turn_id, user_id, toolkit, slug,
                     arguments, status, error, result_bytes, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_COLUMNS}
                """,
                (
                    new_tool_call_id(),
                    conversation_id,
                    turn_id,
                    user_id,
                    toolkit_of(slug),
                    slug,
                    Jsonb(trim(arguments or {})),
                    status,
                    (error or None) and str(error)[:500],
                    result_bytes,
                    latency_ms,
                ),
            )
            row = cur.fetchone()
            conn.commit()

        return _row(row) if row else None

    def for_conversation(self, conversation_id: str, *, limit: int = 200) -> list[ToolCallRow]:
        """The calls in one conversation, in the order they ran."""
        if not conversation_id:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_COLUMNS} FROM {TOOL_CALLS}
                WHERE conversation_id = %s
                ORDER BY created_at, id
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            rows = cur.fetchall()

        return [_row(row) for row in rows]

    def recent(self, user_id: str, *, limit: int = 50) -> list[ToolCallRow]:
        """The audit view: what has been run for this account, newest first."""
        if not user_id:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_COLUMNS} FROM {TOOL_CALLS}
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()

        return [_row(row) for row in rows]


@lru_cache
def get_tool_call_store() -> ToolCallStore:
    return ToolCallStore(get_db())
