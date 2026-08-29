"""HTTP surface for the toolkits connected through somebody's Composio.

Composio *itself* is connected elsewhere — it is one of the connectors under
`/connectors`, alongside the vector stores, because there is nothing
Composio-specific about verifying and sealing a credential. What is here is the
part that has no equivalent for the others: browsing Composio's catalogue and
walking a user through consent for one of its toolkits.

Every route answers **409** until that user has attached Composio, because
there is no server-side Composio account to fall back to.

Every route requires a signed-in caller, and "signed in" means a Clerk session
token whose signature checked out — never a `x-user-id` header and never the
browser's `sess_…`. That is stricter than `/conversations`, deliberately: a
conversation belongs to whoever is holding the browser, which is a guarantee
worth exactly what it sounds like, whereas an API key is a credential and a
connected account is standing permission to read somebody's mail. Anonymous is
a perfectly good state for the first and no state at all for these, so
`require_user` raises where `identify` shrugs.

`user_id` is never read from a path, a query or a body. It is the `sub` of a
verified token and nothing else, which is what makes it impossible to file a
key under somebody else's name or read their connections by typing their id.

Route order matters here and is not incidental: `/toolkits` and `/connect` are
declared before `/{toolkit}`, because FastAPI matches in order and either is a
perfectly good toolkit slug as far as a path parameter is concerned.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from src.core.clerk import Verifier, get_verifier
from src.schemas.integrations import (
    ConnectRequest,
    ConnectStarted,
    Connection,
    ConnectionList,
    Disconnected,
    ToolInventory,
    ToolkitList,
)
from src.services.integration_service import (
    IntegrationError,
    IntegrationService,
    get_integration_service,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])

ServiceDep = Annotated[IntegrationService, Depends(get_integration_service)]


def require_user(
    verifier: Annotated[Verifier, Depends(get_verifier)],
    authorization: Annotated[str | None, Header(description="Bearer <clerk token>")] = None,
) -> str:
    """The signed-in account, or 401.

    An expired token lands here the same way a forged one does. That is the
    right answer even though Clerk's session tokens live about a minute: the
    client refreshes and retries, and the alternative — serving *something* to
    an unverified caller — is the failure mode this whole module exists to
    avoid.
    """
    scheme, _, token = (authorization or "").partition(" ")
    user_id = verifier.user_id(token if scheme.lower() == "bearer" else None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to manage connected accounts.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


UserDep = Annotated[str, Depends(require_user)]


def _handle(error: IntegrationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))


@router.get("/toolkits", response_model=ToolkitList, summary="What can be connected")
def list_toolkits(
    service: ServiceDep,
    user_id: UserDep,
    search: Annotated[str, Query(max_length=64)] = "",
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 40,
) -> ToolkitList:
    """The catalogue, read with this user's own Composio key.

    409 before they have connected Composio, which is the panel's cue to show
    the connect form instead of an empty list.
    """
    try:
        return service.catalog(user_id, search=search, cursor=cursor, limit=limit)
    except IntegrationError as error:
        raise _handle(error) from error


@router.get(
    "/tools",
    response_model=ToolInventory,
    summary="What your linked services let the agent do",
)
def list_tools(service: ServiceDep, user_id: UserDep) -> ToolInventory:
    """The tools the model would be handed on the next turn, by toolkit.

    A static segment, so it resolves ahead of `/{toolkit}` below — `tools` is
    a perfectly good toolkit slug as far as a path parameter is concerned.

    Empty rather than 409 when Composio is not attached: this hangs under the
    Composio detail view, which is only reachable once it is, and a panel
    section is not worth an error page.
    """
    return service.tools(user_id)


@router.get("", response_model=ConnectionList, summary="Your connected accounts")
def list_connections(service: ServiceDep, user_id: UserDep) -> ConnectionList:
    return service.connections(user_id)


@router.post(
    "/connect",
    response_model=ConnectStarted,
    status_code=status.HTTP_201_CREATED,
    summary="Start connecting a toolkit",
)
def connect(body: ConnectRequest, service: ServiceDep, user_id: UserDep) -> ConnectStarted:
    """Returns the URL to send the browser to. It does not follow it.

    A redirect response would be the obvious thing and the wrong one: this is
    called by `fetch` from a panel, and a 30x would be followed by the fetch
    rather than by the window, landing Composio's consent HTML in a promise
    nobody can render.
    """
    try:
        return service.connect(user_id, body.toolkit)
    except IntegrationError as error:
        raise _handle(error) from error


@router.get(
    "/{toolkit}",
    response_model=Connection,
    summary="One connection, refreshed",
)
def connection_status(toolkit: str, service: ServiceDep, user_id: UserDep) -> Connection:
    """What the page polls after consent. 404 when it is not yours or not there."""
    try:
        found = service.status(user_id, toolkit)
    except IntegrationError as error:
        raise _handle(error) from error

    if found is None:
        raise HTTPException(status_code=404, detail="Not connected.")
    return found


@router.delete("/{toolkit}", response_model=Disconnected, summary="Disconnect a toolkit")
def disconnect(toolkit: str, service: ServiceDep, user_id: UserDep) -> Disconnected:
    try:
        removed = service.disconnect(user_id, toolkit)
    except IntegrationError as error:
        raise _handle(error) from error

    if not removed:
        raise HTTPException(status_code=404, detail="Not connected.")
    return Disconnected(toolkit=toolkit, removed=True)
