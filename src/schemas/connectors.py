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
    # Which connector is currently answering this user's retrieval. Null means
    # *nothing* answers it — there is no deployment corpus behind this app, so
    # a user with no vector store attached has nowhere to be searched and /ask
    # says so (docs/13-connectors.md).
    vector_backend: str | None = None


class ProfileField(Wire):
    """One metadata key on the connected store, and how much of it carries it.

    `coverage` rather than a bare presence flag: a key on 3% of an index is not
    something a query may filter on, and the two are indistinguishable to
    anything that only records whether the key was ever seen.
    """

    name: str
    types: list[str] = []
    coverage: float = 0.0
    distinct: int | None = None
    examples: list[str] = []
    filterable: bool = False
    # One value across the whole sample — carried, and useless for narrowing.
    constant: bool = False


class ConnectorProfile(Wire):
    """What this app understood about a connected store, measured and written.

    Two halves that are deliberately not merged. `facts` is measured and is
    what the retrieval path acts on; `summary`, `topics`, `good_for` and
    `not_for` are written by a model over sampled text and only steer routing.
    A caller that confuses them turns a wrong sentence into a wrong query.
    """

    connector: str
    kind: str
    # pending | ok | degraded | failed
    status: str
    # Where it points, with no credential in it.
    location: str = ""
    reachable: bool = False

    title: str = ""
    summary: str = ""
    topics: list[str] = []
    good_for: list[str] = []
    not_for: list[str] = []

    records: int | None = None
    dimensions: int | None = None
    metric: str = ""
    index: str = ""
    sampled: int = 0
    text_field: str = ""
    scripts: list[str] = []
    fields: list[ProfileField] = []
    # Things a person should be told in their own words — a column that is the
    # same on every row, a namespace holding vectors nobody is searching.
    notes: list[str] = []

    # The measured capabilities. `filters` here is the answer every backend
    # used to hard-code as true.
    lexical: bool = False
    filters: bool = False
    parallel_text: bool = False
    searchable: bool = False

    # The agent-facing rendering: what goes into a system prompt verbatim.
    card: str = ""
    error: str = ""
    profiled_at: datetime | None = None


class Capabilities(Wire):
    """Everything the agent knows it can reach right now, in one call.

    `card` is the block that goes into a system prompt as-is. The structured
    profiles are beside it for a panel that wants to render the same thing with
    its own layout, never as a second source of truth.
    """

    card: str = ""
    profiles: list[ConnectorProfile] = []
    # Which store is answering retrieval. Null means nothing is: retrieval
    # abstains until this user connects one.
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
