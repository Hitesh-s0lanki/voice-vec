"""The `/integrations` contract — what the rail panel renders.

Three shapes: a catalogue entry (a toolkit they could connect), a connection
(one they have), and the redirect that turns one into the other.

Connecting *Composio* is not here — that is `/connectors`, with the vector
stores. This is only what happens afterwards.

**No credential travels on this wire at all.** Composio holds the third-party
tokens and the Composio key itself lives behind the connector framework, so the
most sensitive thing here is the name of a service somebody uses.
"""

from datetime import datetime

from pydantic import Field

from src.schemas.wire import Wire


class Toolkit(Wire):
    """One connectable service, as Composio describes it."""

    slug: str = Field(description="Composio's id for the toolkit, e.g. `gmail`")
    name: str
    description: str | None = None
    logo: str | None = None
    categories: list[str] = []
    tools: int = Field(default=0, description="How many actions the toolkit exposes")
    # False when Composio has no OAuth app of its own for this service. Those
    # need an auth config created in the Composio dashboard before anyone can
    # connect, so the panel disables the row instead of offering a button that
    # is guaranteed to fail.
    connectable: bool = Field(
        default=True, description="Whether this can be connected without dashboard setup"
    )
    no_auth: bool = Field(default=False, description="Needs no account at all")


class ToolkitList(Wire):
    toolkits: list[Toolkit] = []
    # Composio pages its catalogue; this is the cursor for the next page, or
    # null at the end.
    next_cursor: str | None = None


class Tool(Wire):
    """One action the model can ask for, as the panel lists it."""

    slug: str = Field(description="Composio's own, e.g. GMAIL_SEND_EMAIL")
    name: str = Field(description="The slug with its toolkit prefix off, in words")
    description: str | None = Field(
        default=None, description="Composio's, trimmed to a line the panel can hold"
    )


class ToolkitTools(Wire):
    """What one linked service actually gives the agent."""

    toolkit: str
    tools: list[Tool] = []
    count: int = Field(default=0, description="How many, in case the list is capped")


class ToolInventory(Wire):
    """Everything this account can be handed on the next turn.

    Not the catalogue's tool counts — those say what a toolkit *has*. This is
    read back through the same call the voice turn makes, so it is bounded by
    `tool_schema_limit` and by which connections reconciled as ACTIVE, and it
    is the honest answer to "what can it do for me".
    """

    toolkits: list[ToolkitTools] = []
    total: int = 0
    limited: bool = Field(
        default=False,
        description="The per-turn schema limit was reached — there may be more",
    )


class Connection(Wire):
    """A service this account has connected."""

    toolkit: str
    name: str | None = Field(default=None, description="Display name, when known")
    logo: str | None = None
    status: str = Field(description="Composio's own: ACTIVE, INITIALIZING, FAILED, …")
    active: bool = Field(default=False, description="Ready to use")
    pending: bool = Field(default=False, description="Consent started, not finished")
    connected_at: datetime | None = None
    updated_at: datetime | None = None


class ConnectionList(Wire):
    connections: list[Connection] = []
    # False when the server cannot store keys at all. Distinct from "you have
    # not connected Composio", which `composio` below carries — those are
    # different problems and only one of them is the user's to fix.
    configured: bool = True


class ConnectRequest(Wire):
    """Start connecting a toolkit.

    Deliberately no `user_id`: it comes from the verified Clerk token and
    accepting one here would be an invitation to connect somebody else's
    account to your own consent screen.
    """

    toolkit: str = Field(min_length=1, max_length=64)


class ConnectStarted(Wire):
    toolkit: str
    redirect_url: str = Field(description="Send the browser here for consent")
    status: str


class Disconnected(Wire):
    toolkit: str
    removed: bool
