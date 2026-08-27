"""HTTP surface for connectors: the outside services a user attaches.

Composio for tools; Pinecone, Astra and pgvector for where their vectors live.
One shape for all of them, because the differences between them are data —
`src/connectors/registry.py` — rather than code.

Every route requires a signed-in caller, and "signed in" means a Clerk session
token whose signature checked out. Nothing else names a user: not a header, not
a query parameter, not a field in a body. `ConnectCredentials` carries values
and deliberately nothing else, so there is nowhere to file a key under somebody
else's name.

That is stricter than `/conversations` and it is the same reason as
`/integrations`: a saved conversation belongs to whoever holds the browser, but
an API key is a credential. Anonymous is a fine state for the first and no state
at all for these.

Credentials arrive in a **JSON body**, never a query parameter or a path
segment, because those end up in access logs, proxy logs and browser history.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from src.connectors.service import Attached, ConnectorService, get_connector_service
from src.connectors.spec import ConnectorError
from src.core.clerk import Verifier, get_verifier
from src.rag.backends.resolve import get_resolver
from src.schemas.connectors import (
    ConnectCredentials,
    Connector,
    ConnectorField,
    ConnectorList,
    Disconnected,
)
from src.services.integration_service import get_integration_service

log = logging.getLogger("vec.connectors")

router = APIRouter(prefix="/connectors", tags=["connectors"])

ServiceDep = Annotated[ConnectorService, Depends(get_connector_service)]


def require_user(
    verifier: Annotated[Verifier, Depends(get_verifier)],
    authorization: Annotated[str | None, Header(description="Bearer <clerk token>")] = None,
) -> str:
    """The signed-in account, or 401.

    An expired token lands here the same way a forged one does. That is right
    even though Clerk's session tokens live about a minute: the client
    refreshes and retries, and the alternative — serving something to an
    unverified caller — is the failure this module exists to avoid.
    """
    scheme, _, token = (authorization or "").partition(" ")
    user_id = verifier.user_id(token if scheme.lower() == "bearer" else None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to manage connectors.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


UserDep = Annotated[str, Depends(require_user)]


def to_wire(attached: Attached) -> Connector:
    spec = attached.spec
    return Connector(
        slug=spec.slug,
        name=spec.name,
        kind=spec.kind,
        summary=spec.summary,
        docs_url=spec.docs_url,
        fields=[
            ConnectorField(
                name=f.name,
                label=f.label,
                secret=f.secret,
                required=f.required,
                placeholder=f.placeholder,
                help=f.help,
            )
            for f in spec.fields
        ],
        connected=attached.connected,
        hints=attached.hints,
        connected_at=attached.connected_at,
        updated_at=attached.updated_at,
        stale=attached.stale,
    )


@router.get("", response_model=ConnectorList, summary="Every connector, and your state on it")
def list_connectors(service: ServiceDep, user_id: UserDep) -> ConnectorList:
    """The catalogue, not just what is attached.

    "What could I connect" is the question somebody with nothing connected is
    asking, and answering it from the same call means the panel renders its
    whole state without a second round trip.
    """
    attached = service.list(user_id)
    backend = get_resolver().for_user(user_id)

    return ConnectorList(
        connectors=[to_wire(row) for row in attached],
        configured=service.configured,
        # `None` when it is the deployment's own store. The panel says so
        # explicitly rather than leaving "which store answers me" a mystery.
        vector_backend=getattr(backend, "name", None) if backend is not None else None,
    )


@router.put(
    "/{slug}",
    response_model=Connector,
    summary="Connect a service, or replace its credentials",
)
def connect(
    slug: str, body: ConnectCredentials, service: ServiceDep, user_id: UserDep
) -> Connector:
    """PUT because it is idempotent in the way that matters.

    Sending credentials twice leaves one account, and sending *different* ones
    replaces the first rather than adding a second — which is the same promise
    `PRIMARY KEY (user_id, connector)` makes in the table.
    """
    try:
        return to_wire(service.connect(user_id, slug, body.values))
    except ConnectorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.delete("/{slug}", response_model=Disconnected, summary="Disconnect a service")
def disconnect(slug: str, service: ServiceDep, user_id: UserDep) -> Disconnected:
    """Forget the credentials, and whatever only made sense alongside them.

    For Composio that means its toolkit rows go too — they name ids inside a
    project this app can no longer reach. Nothing is revoked upstream: those
    connections live in the user's own account and stay theirs.
    """
    attached = service.get(user_id, slug)
    if attached is None:
        raise HTTPException(status_code=404, detail=f"There is no {slug!r} connector.")

    if attached.spec.slug == "composio":
        get_integration_service().purge(user_id)

    return Disconnected(connector=attached.spec.slug, removed=service.disconnect(user_id, slug))
