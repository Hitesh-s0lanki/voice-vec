"""Conversations and their messages, in the same Postgres as the chunks.

A spoken turn used to live for exactly as long as the socket that carried it:
history was a list on `VoiceSession`, and the browser kept its own copy in
`localStorage`. Neither survives a refresh, and neither can be opened on a
second device. This is where a conversation actually lives.

Two tables and one rule about who owns what:

    conversations ─┬─ user_id     signed in
                   └─ session_id  not signed in — the browser's own id
    messages      ─── conversation_id → conversations.id  (ON DELETE CASCADE)

`user_id` is the column that will carry an account once there is one. Until
then the browser mints a `sess_…` id, holds it in `localStorage`, and sends it
on every request; the ownership predicate below accepts a row matching *either*
column, so signing in later can adopt a session's conversations by writing
`user_id` onto them without moving a single message.

**Storage must never be able to break a turn.** Every method here is called
from a worker thread through `anyio.to_thread`, and the voice session queues
its writes rather than awaiting them, so a slow or dead database costs the
listener nothing. `configured` is the switch for a checkout with no
DATABASE_URL at all: the voice loop still runs, the URL simply never gains a
`/c/…` and nothing is written down.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Sequence

from src.core.db import Database, DatabaseUnavailable, get_db

log = logging.getLogger("vec.chat")

# Fixed names, unlike the chunk table's `PG_TABLE`. That one is configurable
# because an experiment may want a second index beside the live one; there is
# no such reason to run two conversation tables, and a literal name keeps
# every statement below a constant rather than an interpolation.
CONVERSATIONS = "conversations"
MESSAGES = "messages"

ROLES = ("user", "assistant")

# What the client sees in the URL bar. Prefixed and dash-free because it is
# read aloud in support conversations and pasted into shells — `conv_9f3a…`
# survives both better than a bare UUID does.
def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def is_conversation_id(value: str | None) -> bool:
    """Cheap shape check, so a junk path segment never reaches the database."""
    if not value or not value.startswith("conv_"):
        return False
    body = value[5:]
    return len(body) == 32 and all(c in "0123456789abcdef" for c in body)


@dataclass(frozen=True, slots=True)
class Owner:
    """Who is asking. Exactly one of these is usually set."""

    user_id: str | None = None
    session_id: str | None = None

    @property
    def known(self) -> bool:
        return bool(self.user_id or self.session_id)


@dataclass(slots=True)
class Conversation:
    id: str
    user_id: str | None
    session_id: str | None
    title: str | None
    language: str | None
    turns: int
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class Message:
    id: str
    conversation_id: str
    turn_id: str | None
    role: str
    text: str
    language_code: str | None
    status: str | None
    reason: str | None
    latency_ms: float | None
    created_at: datetime


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CONVERSATIONS} (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    session_id  TEXT,
    title       TEXT,
    language    TEXT,
    turns       INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT {CONVERSATIONS}_owned CHECK (user_id IS NOT NULL OR session_id IS NOT NULL)
);

-- One index per owner column rather than one composite: a row carries a user
-- id or a session id, never usefully both, and a partial index skips the nulls.
CREATE INDEX IF NOT EXISTS {CONVERSATIONS}_user
    ON {CONVERSATIONS} (user_id, updated_at DESC) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS {CONVERSATIONS}_session
    ON {CONVERSATIONS} (session_id, updated_at DESC) WHERE session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {MESSAGES} (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES {CONVERSATIONS} (id) ON DELETE CASCADE,
    turn_id         TEXT,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text            TEXT NOT NULL,
    language_code   TEXT,
    -- answered · abstained · interrupted · error. Null on a user message.
    status          TEXT,
    reason          TEXT,
    latency_ms      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every read of a conversation is "its messages, oldest first". `id` breaks the
-- tie: two inserts inside the same millisecond are ordinary here, because a
-- question and its answer can land back to back.
CREATE INDEX IF NOT EXISTS {MESSAGES}_thread
    ON {MESSAGES} (conversation_id, created_at, id);
"""

_CONVERSATION_COLUMNS = "id, user_id, session_id, title, language, turns, created_at, updated_at"

# A row is yours if either identity matches. Signing in mid-session therefore
# widens what you can see rather than hiding what you just said.
_OWNED = """
  AND ((%(user_id)s::text IS NOT NULL AND user_id = %(user_id)s)
    OR (%(session_id)s::text IS NOT NULL AND session_id = %(session_id)s))
"""

_MESSAGE_COLUMNS = (
    "id, conversation_id, turn_id, role, text, language_code, status, reason, "
    "latency_ms, created_at"
)


def _conversation(row: Sequence[Any]) -> Conversation:
    return Conversation(
        id=row[0],
        user_id=row[1],
        session_id=row[2],
        title=row[3],
        language=row[4],
        turns=int(row[5]),
        created_at=row[6],
        updated_at=row[7],
    )


def _message(row: Sequence[Any]) -> Message:
    return Message(
        id=row[0],
        conversation_id=row[1],
        turn_id=row[2],
        role=row[3],
        text=row[4],
        language_code=row[5],
        status=row[6],
        reason=row[7],
        latency_ms=float(row[8]) if row[8] is not None else None,
        created_at=row[9],
    )


def title_from(text: str, limit: int = 72) -> str:
    """The first thing said, trimmed to fit a sidebar row.

    Cut on a word boundary when there is one near the end — a title ending
    mid-word reads as broken rather than as truncated. Scripts without spaces
    (Tamil is not one, but Japanese is) fall through to the hard cut.
    """
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean

    cut = clean[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip() + "…"


class ChatStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()
        self._ensured = False

    @property
    def configured(self) -> bool:
        return self._db.configured

    # ---- schema ---------------------------------------------------------

    def ensure_schema(self) -> None:
        """Idempotent, and cheap enough to call on every write path.

        The flag makes it free after the first success. It is not set on
        failure on purpose: a database that comes up late should start working
        on the next turn, not stay broken because one boot missed it.
        """
        if self._ensured:
            return
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA)
            conn.commit()
        self._ensured = True

    # ---- conversations --------------------------------------------------

    def create(
        self,
        owner: Owner,
        *,
        conversation_id: str | None = None,
        title: str | None = None,
        language: str | None = None,
    ) -> Conversation:
        """Open a conversation, optionally under an id the caller already told
        someone about.

        The voice session mints the id itself so it can put `/c/…` in the URL
        the moment the first sentence is transcribed, without waiting on a
        round trip to Neon. Passing it in here is what makes that safe.
        """
        if not owner.known:
            raise ValueError("a conversation needs a user_id or a session_id")

        self.ensure_schema()
        row_id = conversation_id or new_conversation_id()

        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {CONVERSATIONS} (id, user_id, session_id, title, language)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING {_CONVERSATION_COLUMNS}
                """,
                (row_id, owner.user_id, owner.session_id, title, language),
            )
            row = cur.fetchone()
            conn.commit()

        if row is None:  # the id already existed — hand back what is there
            existing = self.get(row_id, owner)
            if existing is None:
                raise DatabaseUnavailable(f"conversation {row_id} exists but is not yours")
            return existing

        return _conversation(row)

    def get(self, conversation_id: str, owner: Owner) -> Conversation | None:
        """None covers both "no such conversation" and "not yours".

        Deliberately the same answer: distinguishing them tells a stranger
        which ids exist.
        """
        if not is_conversation_id(conversation_id) or not owner.known:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_CONVERSATION_COLUMNS} FROM {CONVERSATIONS} WHERE id = %(id)s{_OWNED}",
                {"id": conversation_id, "user_id": owner.user_id, "session_id": owner.session_id},
            )
            row = cur.fetchone()

        return _conversation(row) if row else None

    def list(self, owner: Owner, *, limit: int = 30) -> list[Conversation]:
        """Newest activity first, and never the empty ones.

        A conversation row is written the moment someone speaks, so a take that
        failed at transcription leaves a row with no messages. Showing those
        would fill the panel with blanks nobody can open.
        """
        if not owner.known:
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLUMNS} FROM {CONVERSATIONS}
                WHERE turns > 0{_OWNED}
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                {
                    "user_id": owner.user_id,
                    "session_id": owner.session_id,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()

        return [_conversation(row) for row in rows]

    def rename(self, conversation_id: str, owner: Owner, title: str) -> Conversation | None:
        if not is_conversation_id(conversation_id) or not owner.known:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {CONVERSATIONS} SET title = %(title)s, updated_at = now()
                WHERE id = %(id)s{_OWNED}
                RETURNING {_CONVERSATION_COLUMNS}
                """,
                {
                    "id": conversation_id,
                    "title": title_from(title, 120),
                    "user_id": owner.user_id,
                    "session_id": owner.session_id,
                },
            )
            row = cur.fetchone()
            conn.commit()

        return _conversation(row) if row else None

    def delete(self, conversation_id: str, owner: Owner) -> bool:
        """Messages go with it — the foreign key cascades."""
        if not is_conversation_id(conversation_id) or not owner.known:
            return False

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {CONVERSATIONS} WHERE id = %(id)s{_OWNED}",
                {"id": conversation_id, "user_id": owner.user_id, "session_id": owner.session_id},
            )
            deleted = cur.rowcount
            conn.commit()

        return bool(deleted)

    def adopt(self, *, user_id: str, session_id: str) -> int:
        """Hand a browser's anonymous conversations to the account that just
        signed in, and report how many moved.

        This is the whole reason `user_id` and `session_id` are two columns
        rather than one owner string. Everything said before signing in is
        already filed under the browser; signing in writes an account onto
        those same rows, and not one message moves.

        `user_id IS NULL` is what keeps it safe to call twice, and keeps it
        from taking conversations off whoever owns them — a shared browser
        hands over what nobody has claimed, and nothing else.
        """
        if not user_id or not session_id:
            return 0

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {CONVERSATIONS} SET user_id = %(user_id)s
                WHERE session_id = %(session_id)s AND user_id IS NULL
                """,
                {"user_id": user_id, "session_id": session_id},
            )
            moved = cur.rowcount
            conn.commit()

        return int(moved or 0)

    # ---- messages -------------------------------------------------------

    def append(
        self,
        conversation_id: str,
        *,
        role: str,
        text: str,
        turn_id: str | None = None,
        language_code: str | None = None,
        status: str | None = None,
        reason: str | None = None,
        latency_ms: float | None = None,
    ) -> Message | None:
        """Add one message and move the conversation's clock in the same
        transaction.

        Both statements or neither: a conversation whose `updated_at` says
        "just now" but holds no new message sorts to the top of the panel for
        an exchange that never happened.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {role!r}")

        clean = text.strip()
        if not clean:
            return None

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {MESSAGES}
                    (id, conversation_id, turn_id, role, text, language_code,
                     status, reason, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_MESSAGE_COLUMNS}
                """,
                (
                    new_message_id(),
                    conversation_id,
                    turn_id,
                    role,
                    clean,
                    language_code,
                    status,
                    reason,
                    latency_ms,
                ),
            )
            row = cur.fetchone()

            # `turns` counts questions, not messages, so the panel can say
            # "4 takes" without dividing by two and guessing about the
            # half-turn a barge-in leaves behind. COALESCE on title fills it
            # from the first question and never overwrites a rename.
            cur.execute(
                f"""
                UPDATE {CONVERSATIONS}
                SET updated_at = now(),
                    turns      = turns + %(counts)s,
                    language   = COALESCE(language, %(language)s),
                    title      = COALESCE(title, %(title)s)
                WHERE id = %(id)s
                """,
                {
                    "id": conversation_id,
                    "counts": 1 if role == "user" else 0,
                    "language": language_code,
                    "title": title_from(clean) if role == "user" else None,
                },
            )

        return _message(row) if row else None

    def messages(self, conversation_id: str, *, limit: int = 200) -> list[Message]:
        """The thread, oldest first — the order it is read in.

        The limit takes the *newest* messages and then puts them back in order.
        Ordering ascending and limiting would cut the other end: a long
        conversation would come back as its own opening, with everything said
        since missing, which is the half nobody wants.
        """
        if not is_conversation_id(conversation_id):
            return []

        self.ensure_schema()
        with self._db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MESSAGE_COLUMNS} FROM (
                    SELECT {_MESSAGE_COLUMNS} FROM {MESSAGES}
                    WHERE conversation_id = %(id)s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %(limit)s
                ) AS recent
                ORDER BY created_at, id
                """,
                {"id": conversation_id, "limit": limit},
            )
            rows = cur.fetchall()

        return [_message(row) for row in rows]

    def history(self, conversation_id: str, *, max_messages: int) -> list[dict[str, str]]:
        """The tail of the thread, in the shape the chat model takes.

        This is what makes a reload continue a conversation rather than start
        one: the model is handed what it already said, read back out of
        Postgres instead of out of the socket that closed. The *tail*, because
        `llm.build_messages` trims to the same window anyway and there is no
        point paying Neon for turns that will be dropped.
        """
        if max_messages <= 0:
            return []

        rows = self.messages(conversation_id, limit=max_messages)
        return [{"role": row.role, "content": row.text} for row in rows]


@lru_cache
def get_chat_store() -> ChatStore:
    return ChatStore(get_db())
