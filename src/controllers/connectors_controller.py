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

from src.connectors.profile import Profile
from src.connectors.profile_service import PROFILE_ORDER, ProfileService, get_profile_service
from src.connectors.registry import get_spec
from src.connectors.service import Attached, ConnectorService, get_connector_service
from src.connectors.spec import ConnectorError
from src.core.clerk import Verifier, get_verifier
from src.datasets.service import DatasetError, DatasetService, get_dataset_service
from src.rag.backends.resolve import get_resolver
from src.schemas.connectors import (
    Capabilities,
    ConnectCredentials,
    Connector,
    ConnectorField,
    ConnectorList,
    ConnectorProfile,
    Disconnected,
    ProfileField,
)
from src.services.integration_service import get_integration_service

log = logging.getLogger("vec.connectors")

router = APIRouter(prefix="/connectors", tags=["connectors"])

ServiceDep = Annotated[ConnectorService, Depends(get_connector_service)]
ProfilesDep = Annotated[ProfileService, Depends(get_profile_service)]
DatasetsDep = Annotated[DatasetService, Depends(get_dataset_service)]


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


def to_profile_wire(profile: Profile, card: str) -> ConnectorProfile:
    """A `Profile` flattened for the wire.

    Flattened rather than nested because the panel and the agent read different
    halves of it and neither wants to walk `observation.vectors.dimensions` to
    find a number. The nesting exists in the dataclass to keep "what was
    measured" separate from "what was written about it"; on the wire that
    separation is carried by the field names.
    """
    observation = profile.observation
    shape = observation.vectors

    return ConnectorProfile(
        connector=profile.connector,
        kind=profile.kind,
        status=profile.status,
        location=observation.location,
        reachable=observation.reachable,
        title=profile.understanding.title,
        summary=profile.understanding.summary,
        topics=list(profile.understanding.topics),
        good_for=list(profile.understanding.good_for),
        not_for=list(profile.understanding.not_for),
        records=shape.records if shape else None,
        dimensions=shape.dimensions if shape else None,
        metric=shape.metric if shape else "",
        index=shape.index if shape else "",
        sampled=observation.sampled,
        text_field=observation.text_field,
        scripts=list(observation.scripts),
        fields=[
            ProfileField(
                name=f.name,
                types=list(f.types),
                coverage=round(f.coverage, 3),
                distinct=f.distinct,
                examples=list(f.examples),
                filterable=f.filterable,
                constant=f.constant,
            )
            for f in observation.fields
        ],
        notes=list(observation.notes),
        lexical=profile.facts.lexical,
        filters=profile.facts.filters,
        parallel_text=profile.facts.parallel_text,
        searchable=profile.facts.searchable,
        card=card,
        error=profile.error,
        profiled_at=profile.profiled_at,
    )


@router.get(
    "/capabilities",
    response_model=Capabilities,
    summary="What the agent can reach right now",
)
def capabilities(profiles: ProfilesDep, user_id: UserDep) -> Capabilities:
    """The realtime read: one call, everything this user's agent can search or do.

    Declared **before** any `/{slug}` route, because `capabilities` would
    otherwise be captured as a connector slug by whichever matched first.

    Never blocks on a probe. A connector attached seconds ago has no profile
    yet and is simply absent from the response — the panel shows it as
    understanding, and the next poll has it. Waiting here would put somebody
    else's cold Pinecone in the way of a page load.
    """
    found: list[ConnectorProfile] = []
    for slug in PROFILE_ORDER:
        profile = profiles.get(user_id, slug)
        if profile is not None:
            found.append(to_profile_wire(profile, profiles.card(user_id, slug)))

    backend = get_resolver().for_user(user_id)
    return Capabilities(
        card=profiles.cards(user_id),
        profiles=found,
        vector_backend=getattr(backend, "name", None) if backend is not None else None,
    )


@router.get(
    "/{slug}/profile",
    response_model=ConnectorProfile,
    summary="What this app understood about one connected store",
)
def profile(
    slug: str, service: ServiceDep, profiles: ProfilesDep, user_id: UserDep
) -> ConnectorProfile:
    """404 and 202 are different questions, and only one of them is worth polling.

    "You have not connected Pinecone" will not change by asking again; "the
    probe is still running" will. Collapsing both into 202 — which this did at
    first — tells a panel to poll forever for a profile that can never exist.
    """
    spec = get_spec(slug)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"There is no {slug!r} connector.")

    attached = service.get(user_id, spec.slug)
    if attached is None or not attached.connected:
        raise HTTPException(
            status_code=404, detail=f"You have not connected {spec.name}."
        )

    understood = profiles.get(user_id, spec.slug)
    if understood is None:
        # Connected, and no understanding yet: the probe was scheduled by the
        # read that just missed. Worth asking again in a moment.
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Still reading that store. Try again in a moment.",
        )
    return to_profile_wire(understood, profiles.card(user_id, spec.slug))


@router.post(
    "/{slug}/profile",
    response_model=ConnectorProfile,
    summary="Read the connected store again, now",
)
def reprofile(slug: str, profiles: ProfilesDep, user_id: UserDep) -> ConnectorProfile:
    """Synchronous, unlike every other path into profiling.

    This is the one place somebody has explicitly asked to wait — they ingested
    into their index and want the agent to know about it without waiting out
    the TTL. Everywhere else the probe runs behind the request precisely so
    nobody waits on somebody else's database.
    """
    spec = get_spec(slug)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"There is no {slug!r} connector.")

    understood = profiles.refresh(user_id, spec.slug)
    if understood is None:
        raise HTTPException(
            status_code=404, detail=f"You have not connected {spec.name}."
        )
    return to_profile_wire(understood, profiles.card(user_id, spec.slug))


def _summarise_datasets(row: Connector, datasets: DatasetService, user_id: str) -> Connector:
    """Replace the dataset row's hints with what is *actually* attached.

    Every other connector is one credential, so its row describes itself. The
    dataset connector is a doorway to many — the same shape Composio has, where
    one credential opens onto a list of linked toolkits — so a row rendered
    from the last URL somebody typed would name one of several and read as
    though the others were gone.

    The list comes from `agent_datasets`, which is the real state: the
    connector row records that datasets are attached at all, and the table
    records which. Reading the table here means a dataset removed through
    `DELETE /datasets/{id}` disappears from this row too, with nothing to keep
    in step.
    """
    try:
        attached = datasets.list(user_id)
    except Exception as error:
        log.debug("could not summarise datasets: %s", error)
        return row

    names = [d.dataset_id for d in attached]
    building = sum(1 for d in attached if d.status == "pending")
    failed = sum(1 for d in attached if d.status == "failed")

    hints: dict[str, str] = {}
    if names:
        hints["datasets"] = ", ".join(names[:3]) + ("" if len(names) <= 3 else f" +{len(names) - 3}")
    if building:
        hints["building"] = f"{building} building"
    if failed:
        hints["failed"] = f"{failed} failed"

    return row.model_copy(update={"connected": bool(names), "hints": hints})


@router.get("", response_model=ConnectorList, summary="Every connector, and your state on it")
def list_connectors(service: ServiceDep, datasets: DatasetsDep, user_id: UserDep) -> ConnectorList:
    """The catalogue, not just what is attached.

    "What could I connect" is the question somebody with nothing connected is
    asking, and answering it from the same call means the panel renders its
    whole state without a second round trip.
    """
    attached = service.list(user_id)
    backend = get_resolver().for_user(user_id)

    rows = [to_wire(row) for row in attached]
    rows = [
        _summarise_datasets(row, datasets, user_id) if row.kind == "dataset" else row
        for row in rows
    ]

    return ConnectorList(
        connectors=rows,
        configured=service.configured,
        # `None` when nothing is attached, which is a state the panel has to
        # state outright: there is no fallback index behind it, so a question
        # asked now abstains rather than being answered from somewhere else.
        vector_backend=getattr(backend, "name", None) if backend is not None else None,
    )


@router.put(
    "/{slug}",
    response_model=Connector,
    summary="Connect a service, or replace its credentials",
)
def connect(
    slug: str,
    body: ConnectCredentials,
    service: ServiceDep,
    profiles: ProfilesDep,
    datasets: DatasetsDep,
    user_id: UserDep,
) -> Connector:
    """PUT because it is idempotent in the way that matters.

    Sending credentials twice leaves one account, and sending *different* ones
    replaces the first rather than adding a second — which is the same promise
    `PRIMARY KEY (user_id, connector)` makes in the table.

    **A dataset is the exception, and it is still idempotent.** `agent_datasets`
    is keyed by an id derived from the URL, so PUT-ing the same dataset twice
    leaves one — the promise above, held per dataset instead of per connector.
    PUT-ing a *different* URL adds it alongside, up to `dataset_max_per_user`,
    because a person has many datasets the way they have many Composio
    toolkits and only one Pinecone. The connector row records that datasets are
    attached; the table records which, and `_summarise_datasets` reads it back.
    """
    try:
        attached = service.connect(user_id, slug, body.values)
    except ConnectorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    # Understanding what was just attached happens *after* the response is
    # decided and off this request's thread. Probing here would put a sample of
    # somebody's index and a model call between them and a green form, to learn
    # something no part of connecting depends on.
    #
    # `forget` first: reconnecting may point at a different index entirely, and
    # the in-process card would otherwise describe the old one until its TTL
    # expired. The stored row is handled by the fingerprint, which changed with
    # the credentials.
    if attached.spec.kind == "dataset":
        # A dataset's "profile" is not a connector probe — there is no store to
        # sample, and `_PROBES` has no entry for it, so scheduling one here
        # would write a `failed` profile row and show a red panel above a
        # connector that works. Its equivalent is the build, which resolves,
        # pulls, measures and narrates on the same off-request worker
        # (docs/18-datasets.md).
        _attach_dataset(datasets, user_id, body.values.get("url", ""))
    else:
        profiles.forget(user_id, attached.spec.slug)
        profiles.schedule(user_id, attached.spec.slug)
    return to_wire(attached)


def _attach_dataset(datasets: DatasetService, user_id: str, url: str) -> None:
    """Start the build for a dataset that was just connected.

    Never raises. The credential — a public URL — is already verified and
    stored, so the connector *is* connected whatever happens next, and turning
    a build failure into a 500 here would leave a green row in the table behind
    a red form. A failure lands on the dataset row instead, where the panel can
    show the reason and offer a rebuild.
    """
    try:
        datasets.add(user_id, url)
    except DatasetError as error:
        log.warning("dataset connected but not built for %s: %s", user_id, error)
    except Exception as error:
        log.warning("could not start the dataset build for %s: %s", user_id, error)


@router.delete("/{slug}", response_model=Disconnected, summary="Disconnect a service")
def disconnect(
    slug: str,
    service: ServiceDep,
    profiles: ProfilesDep,
    datasets: DatasetsDep,
    user_id: UserDep,
) -> Disconnected:
    """Forget the credentials, and whatever only made sense alongside them.

    For Composio that means its toolkit rows go too — they name ids inside a
    project this app can no longer reach. Nothing is revoked upstream: those
    connections live in the user's own account and stay theirs.

    For a dataset it means the measured profile *and* the materialised DuckDB
    file. There is no foreign key doing this one: `agent_datasets` deliberately
    has none, because a dataset is a public URL rather than something that dies
    with a credential — and a cascade cannot unlink a file anyway. Leaving it
    would keep hundreds of megabytes on disk describing something the panel no
    longer shows.
    """
    attached = service.get(user_id, slug)
    if attached is None:
        raise HTTPException(status_code=404, detail=f"There is no {slug!r} connector.")

    if attached.spec.slug == "composio":
        get_integration_service().purge(user_id)

    if attached.spec.kind == "dataset":
        # Every one of them, not the one whose URL happens to be in `hints`.
        # Disconnecting the connector is "I am done with datasets"; leaving the
        # other four attached would keep them answerable through the voice tool
        # while the panel showed nothing connected. Removing one is
        # `DELETE /datasets/{id}`.
        _detach_datasets(datasets, user_id)

    # The stored profile goes with the account through the foreign key's
    # cascade. This is the in-process copy, which no cascade can reach.
    profiles.forget(user_id, attached.spec.slug)

    return Disconnected(connector=attached.spec.slug, removed=service.disconnect(user_id, slug))


def _detach_datasets(datasets: DatasetService, user_id: str) -> None:
    """Remove every dataset this user attached, and the files behind them.

    Per dataset rather than in one sweep, and each in its own `try`: one file
    that will not unlink — a stale handle, a permission — must not strand the
    other four as rows describing datasets the panel no longer shows.

    A failure here must not fail the disconnect either. Somebody asking to
    detach gets detached; a file left on disk is space to reclaim, not a reason
    to keep a connector they asked to remove.
    """
    try:
        rows = datasets.list(user_id)
    except Exception as error:
        log.warning("could not list datasets to detach for %s: %s", user_id, error)
        return

    for row in rows:
        try:
            datasets.remove(user_id, row.dataset_id)
        except Exception as error:
            log.warning("could not remove dataset %s: %s", row.dataset_id, error)
