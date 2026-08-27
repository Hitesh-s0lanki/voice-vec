"""Conversation logic: whose it is, and what comes back.

Thin, because the interesting decisions are one layer down — the ownership
predicate lives in the SQL, so no code path can read a conversation by
forgetting a check. What is here is the mapping from stored rows to the wire
and the one policy call this layer owns: a conversation nobody can claim (no
user, no session) is not an error, it is an empty list.
"""

from __future__ import annotations

import logging

from src.chat.store import ChatStore, Conversation, Message, Owner, get_chat_store
from src.chat.tools import ToolCallRow, ToolCallStore, get_tool_call_store
from src.core.db import DatabaseUnavailable
from src.schemas.chat import (
    ChatMessage,
    ToolCall,
    ConversationDetail,
    ConversationList,
    ConversationSummary,
)

log = logging.getLogger("vec.chat")


def summarise(row: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=row.id,
        title=row.title,
        language=row.language,
        turns=row.turns,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def to_wire(row: Message) -> ChatMessage:
    return ChatMessage(
        id=row.id,
        role=row.role,  # type: ignore[arg-type] — the column has a CHECK constraint
        text=row.text,
        language_code=row.language_code,
        turn_id=row.turn_id,
        status=row.status,
        reason=row.reason,
        latency_ms=row.latency_ms,
        created_at=row.created_at,
    )


def tool_to_wire(row: ToolCallRow) -> ToolCall:
    return ToolCall(
        id=row.id,
        turn_id=row.turn_id,
        toolkit=row.toolkit,
        slug=row.slug,
        arguments=row.arguments,
        status=row.status,
        ok=row.ok,
        error=row.error,
        result_bytes=row.result_bytes,
        latency_ms=row.latency_ms,
        created_at=row.created_at,
    )


class ConversationService:
    def __init__(self, store: ChatStore, tools: ToolCallStore | None = None) -> None:
        self._store = store
        self._tools = tools or get_tool_call_store()

    @property
    def enabled(self) -> bool:
        return self._store.configured

    def create(
        self, owner: Owner, *, title: str | None = None, language: str | None = None
    ) -> ConversationSummary:
        return summarise(self._store.create(owner, title=title, language=language))

    def list(self, owner: Owner, *, limit: int = 30) -> ConversationList:
        """A dead database empties the rail rather than breaking the page.

        The panel this feeds sits beside a working voice loop; a 500 there
        would take the orb down with it for something nobody asked to see.
        """
        try:
            rows = self._store.list(owner, limit=limit)
        except DatabaseUnavailable as error:
            log.warning("cannot list conversations: %s", error)
            return ConversationList()

        return ConversationList(conversations=[summarise(row) for row in rows])

    def read(self, conversation_id: str, owner: Owner) -> ConversationDetail | None:
        conversation = self._store.get(conversation_id, owner)
        if conversation is None:
            return None

        # A conversation that reads fine but cannot list its tool calls is
        # better than a 404. The thread is the thing being asked for; what the
        # agent ran alongside it is annotation.
        try:
            ran = self._tools.for_conversation(conversation_id)
        except Exception as error:
            log.warning("could not read tool calls for %s: %s", conversation_id, error)
            ran = []

        return ConversationDetail(
            conversation=summarise(conversation),
            messages=[to_wire(row) for row in self._store.messages(conversation_id)],
            tool_calls=[tool_to_wire(row) for row in ran],
        )

    def rename(
        self, conversation_id: str, owner: Owner, title: str
    ) -> ConversationSummary | None:
        row = self._store.rename(conversation_id, owner, title)
        return summarise(row) if row else None

    def delete(self, conversation_id: str, owner: Owner) -> bool:
        return self._store.delete(conversation_id, owner)

    def adopt(self, user_id: str, session_id: str) -> int:
        """Sign-in housekeeping, and never worth failing a sign-in over."""
        try:
            return self._store.adopt(user_id=user_id, session_id=session_id)
        except DatabaseUnavailable as error:
            log.warning("could not adopt conversations for %s: %s", user_id, error)
            return 0


def get_conversation_service() -> ConversationService:
    return ConversationService(get_chat_store(), get_tool_call_store())
