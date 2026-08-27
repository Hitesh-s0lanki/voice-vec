"""The conversation contract — what `/conversations` returns.

Two shapes and nothing clever: a summary for the list in the rail, and the
summary plus its messages for the thread on screen. The voice socket writes
these rows; this is only how they are read back.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from src.schemas.wire import Wire


class ConversationSummary(Wire):
    id: str = Field(description="conv_ followed by 32 hex characters")
    title: str | None = Field(default=None, description="The first question, trimmed")
    language: str | None = Field(default=None, description="Language of the first take")
    turns: int = Field(default=0, description="Questions asked, not messages stored")
    created_at: datetime
    updated_at: datetime


class ChatMessage(Wire):
    id: str
    role: Literal["user", "assistant"]
    text: str
    language_code: str | None = None
    turn_id: str | None = Field(
        default=None, description="The voice turn this belongs to — pairs the two rows"
    )
    status: str | None = Field(
        default=None,
        description="answered · abstained · interrupted · error. Null on a question.",
    )
    reason: str | None = None
    latency_ms: float | None = None
    created_at: datetime


class ToolCall(Wire):
    """One tool the agent ran, as the panel shows it.

    No result. The store keeps its *size* and not its content — an audit table
    should not become a copy of everything the agent has ever read — so there
    is nothing here that could carry somebody's inbox onto the wire.
    """

    id: str
    turn_id: str | None = None
    toolkit: str | None = Field(default=None, description="gmail, slack, …")
    slug: str = Field(description="Composio's own, e.g. GMAIL_SEND_EMAIL")
    arguments: dict = Field(default_factory=dict, description="What the agent decided")
    status: str = Field(description="ok · failed")
    ok: bool = False
    error: str | None = None
    result_bytes: int | None = None
    latency_ms: float | None = None
    created_at: datetime


class ConversationDetail(Wire):
    conversation: ConversationSummary
    messages: list[ChatMessage] = []
    #: What the agent ran, in the order it ran. Paired to a message by
    #: `turn_id`, the same way a question is paired to its answer.
    tool_calls: list[ToolCall] = []


class ConversationList(Wire):
    conversations: list[ConversationSummary] = []


class CreateConversation(Wire):
    """Optional — the voice socket opens one by itself on the first take."""

    title: str | None = None
    language: str | None = None


class RenameConversation(Wire):
    title: str = Field(min_length=1, max_length=200)


class AdoptConversations(Wire):
    """Claim what this browser said before anyone signed in."""

    session_id: str = Field(
        min_length=1, max_length=128, description="The browser's own sess_… id"
    )


class Adopted(Wire):
    moved: int = Field(description="Conversations now owned by the signed-in account")
