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

**Arguments are stored whole; results are stored as a bounded preview.** The
arguments are what the agent *decided*, which is the interesting half and is
small. The result is what came *back*, and the thread is unreadable without it —
"it ran `GMAIL_FETCH_EMAILS`" says nothing about what the answer was built from.
So the first `MAX_RESULT_CHARS` of the rendered result are kept, alongside
`result_bytes`, which stays the size of the *whole* thing: the pair reads as
"here is what came back, and here is how much of it you are seeing".

The ceiling is the containment. A result can be an entire inbox page and belongs
to the provider, not here; keeping a preview rather than the payload is what
stops this table becoming an uncontrolled copy of everything the agent has ever
read. It is still somebody's mail — the same care as the credential that reached
it applies, and it is why a preview is read back only by the account that owns
the conversation.

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

log = logging.getLogger("vec.chat.tool_calls")

TOOL_CALLS = "tool_calls"

# Arguments do go in, so they are bounded. A model that emits a 100 KB argument
# blob is malfunctioning and should not be able to fill a table over it.
MAX_ARGUMENT_CHARS = 8_000

# And so is the result preview. Deliberately smaller than the ceiling the model
# itself sees (`src/agents/tool_agent.py`): this is the readable head of a
# result, not a second copy of the provider's page.
MAX_RESULT_CHARS = 4_000


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
    #: The head of what came back — see `trim_result`. None when nothing did.
    result: str | None
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
    -- The head of what came back, bounded. See the module docstring.
    result          TEXT,
    -- The size of the whole of it, so a preview can say what it is a preview of.
    result_bytes    INTEGER,
    latency_ms      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- `result` arrived after the table did, and `ensure_schema` only ever creates.
-- Adding it here rather than in a migration keeps the one rule this module has:
-- a database that comes up in any state starts working on the next turn.
ALTER TABLE {TOOL_CALLS} ADD COLUMN IF NOT EXISTS result TEXT;

-- Every read is "the calls in this conversation, in the order they ran".
CREATE INDEX IF NOT EXISTS {TOOL_CALLS}_thread
    ON {TOOL_CALLS} (conversation_id, created_at, id);

-- And the audit one: everything this account has ever had run for it.
CREATE INDEX IF NOT EXISTS {TOOL_CALLS}_user
    ON {TOOL_CALLS} (user_id, created_at DESC) WHERE user_id IS NOT NULL;
"""

_COLUMNS = (
    "id, conversation_id, turn_id, user_id, toolkit, slug, arguments, status, "
    "error, result, result_bytes, latency_ms, created_at"
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
        result=row[9],
        result_bytes=int(row[10]) if row[10] is not None else None,
        latency_ms=float(row[11]) if row[11] is not None else None,
        created_at=row[12],
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


def trim_result(rendered: str | None) -> str | None:
    """The head of a result, marked when there is more of it.

    Empty and whitespace-only results become None rather than a blank row: a
    tool that returned nothing is better read from its status than from an
    empty box in the thread.
    """
    text = (rendered or "").strip()
    if not text:
        return None

    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + "… (truncated)"


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
        id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        user_id: str | None = None,
        arguments: dict | None = None,
        error: str | None = None,
        result: str | None = None,
        result_bytes: int | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallRow | None:
        """Write one call down. Returns None when there is no database.

        `id` is a parameter rather than always minted here so the caller can
        mint it *before* the write and put the same one on the wire — the voice
        socket announces a finished call immediately and this row lands a
        moment later, and the two being the same call has to be something the
        client can see rather than infer from a slug and a timestamp.
        """
        if not slug:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {TOOL_CALLS}
                    (id, conversation_id, turn_id, user_id, toolkit, slug,
                     arguments, status, error, result, result_bytes, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_COLUMNS}
                """,
                (
                    id or new_tool_call_id(),
                    conversation_id,
                    turn_id,
                    user_id,
                    toolkit_of(slug),
                    slug,
                    Jsonb(trim(arguments or {})),
                    status,
                    (error or None) and str(error)[:500],
                    trim_result(result),
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
