"""The `/connectors` contract — enough for the panel to render a form it has
never seen.

`ConnectorField` is the interesting shape. It is a *description of an input*,
sent to the browser so the panel can build the right form for a connector added
after that panel was written. Without it, every new connector means a React
component, and the form and the server-side validation drift the first time
either is edited.

**Credentials travel one way.** They go in on `ConnectCredentials` and nothing
sends one back. `hints` carries the non-secret fields as typed plus the last
four characters of the secret one, which is what lets the panel say
"Pinecone · vec-chunks · ····8fa2" without ever holding the key.
"""

from datetime import datetime

from pydantic import Field

from src.schemas.wire import Wire


class ConnectorField(Wire):
    """One input the panel should render."""

    name: str
    label: str
    # Renders as a password, is never echoed back, and only its last four
    # characters are kept readable. The single most important flag here.
    secret: bool = False
    required: bool = True
    placeholder: str = ""
    help: str = ""


class Connector(Wire):
    """One connector, described, with this user's state on it."""

    slug: str
    name: str
    # "tools" or "vector". The panel groups by it, and retrieval asks for it
    # rather than enumerating slugs.
    kind: str
    summary: str
    docs_url: str = ""
    fields: list[ConnectorField] = []

    connected: bool = False
    hints: dict[str, str] = {}
    connected_at: datetime | None = None
    updated_at: datetime | None = None
    # A credential is stored but no longer decrypts — the master key was
    # rotated. "Connected" would be a lie and "not connected" loses the fact
    # that there is a row to replace, so it is its own flag.
    stale: bool = False


class ConnectorList(Wire):
    connectors: list[Connector] = []
    # False when the server has no COMPOSIO_ENCRYPTION_KEY and so cannot hold a
    # credential at all. Different from "you have not connected anything", and
    # only one of the two is the signed-in user's to fix.
    configured: bool = True
    # Which connector is currently answering this user's retrieval, if any.
    # Null means the deployment's own store, which is the default and fine.
    vector_backend: str | None = None


class ConnectCredentials(Wire):
    """The one request on this API that carries secrets.

    Free-form because the fields differ per connector; the server takes only
    what that connector declared and drops the rest, so this being open does
    not mean arbitrary JSON reaches a credential blob.

    No `user_id`: whose account this becomes is decided by the verified Clerk
    token, and accepting one here would let anybody file a key under somebody
    else's name.
    """

    values: dict[str, str] = Field(
        default_factory=dict, description="Field name → value, per the connector's schema"
    )


class Disconnected(Wire):
    connector: str
    removed: bool
