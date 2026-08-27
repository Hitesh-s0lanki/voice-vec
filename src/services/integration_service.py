"""Composio, connected by the user, and the toolkits they connect through it.

**Composio is the connector.** There is no project-wide Composio account behind
this app. A signed-in person connects their own — `connect_composio` verifies
the key before storing it, sealed — and every call afterwards runs against that
person's Composio project. Two people using this app share no Composio state at
all: not a connected account, not an auth config, not a catalogue response.

That inverts where the trust sits, and three things follow from it.

**Nothing works before Composio is connected.** `_sdk` raises `NotConnected`
rather than falling back to anything, because there is nothing to fall back to.
The panel turns that into "connect Composio first", which is a different
sentence from "try again" and needs to be.

**Auth configs are per user.** An auth config *is* an OAuth application, and it
lives inside one Composio project. User A's `ac_…` is not a thing user B's
Composio has ever heard of, so `integration_auth_configs` is keyed by both and
`_auth_config` creates one in whichever project it is talking to.

**`link()`, not `initiate()`.** Composio retired `initiate()` for its own
managed OAuth apps — blocked for new organisations from 2026-05-08 and for
everybody from 2026-07-03, both in the past. Calling it on a managed auth
config now raises rather than returning a URL. `link()` is the replacement:
same arguments, same `redirect_url` back, routing the user through a consent
screen first. `initiate()` is still correct for a *custom* auth config, so
`_start` reads `is_composio_managed` and picks. Hard-coding either one breaks
half the deployments, and the failure is invisible without a real Composio.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Iterable
from urllib.parse import quote

from composio import Composio

from src.core.config import Settings, get_settings
from src.integrations.client import (
    ComposioClients,
    ComposioUnavailable,
    NotConnected,
    get_clients,
)
from src.integrations.store import (
    ACTIVE,
    PENDING,
    Connection as StoredConnection,
    IntegrationStore,
    get_integration_store,
)
from src.schemas.integrations import (
    ConnectStarted,
    Connection,
    ConnectionList,
    Toolkit,
    ToolkitList,
)

log = logging.getLogger("vec.integrations")



class IntegrationError(RuntimeError):
    """Something the caller can be told about without leaking internals."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _slug(value: str) -> str:
    """Composio's toolkit slugs are lowercase. Normalise before anything else.

    A slug reaches a URL path at Composio and a primary key here, so it is
    also length-capped and stripped of everything that is not a slug: this is
    the one string on these routes that comes from the client.
    """
    clean = (value or "").strip().lower()[:64]
    return "".join(c for c in clean if c.isalnum() or c in "-_")


def _meta(item: Any, field: str, default: Any = None) -> Any:
    """Read a field off a Composio response object without assuming its shape.

    The SDK's models are generated and gain fields between releases; several of
    these are optional and at least one (`meta`) is a nested model on the list
    response and absent on some items. Reading defensively here keeps a new
    Composio release from turning the catalogue into a 500.
    """
    meta = getattr(item, "meta", None)
    if meta is None:
        return default
    return getattr(meta, field, default) or default


class IntegrationService:
    def __init__(
        self,
        clients: ComposioClients,
        store: IntegrationStore,
        settings: Settings,
    ) -> None:
        self._clients = clients
        self._store = store
        self._settings = settings
        # (user_id, cursor, search) → (fetched_at, payload). Keyed by user
        # because it is fetched with that user's key against that user's
        # project — a shared cache here would serve one person's Composio to
        # another, which is the exact mistake this whole rewrite is about.
        self._catalog: dict[tuple[str, str, str], tuple[float, ToolkitList]] = {}

    @property
    def configured(self) -> bool:
        """Whether this deployment can hold a Composio key at all."""
        return self._clients.configured

    def _sdk(self, user_id: str) -> Composio:
        """This user's Composio, or an `IntegrationError` the panel can render.

        The three failures are deliberately different status codes: 409 is
        "connect Composio first" and is the user's to fix, 503 is the server
        missing its encryption key, 502 is Composio not answering.
        """
        try:
            return self._clients.for_user(user_id)
        except NotConnected as error:
            raise IntegrationError(str(error), status_code=409) from error
        except ComposioUnavailable as error:
            raise IntegrationError(str(error), status_code=503) from error

    def purge(self, user_id: str) -> int:
        """Drop the toolkit rows when the Composio connector goes away.

        Called by the connector service on disconnect. It does *not* revoke
        anything inside the user's Composio project: those connections are
        theirs and stay theirs after they stop pointing this app at their
        account. Deleting them would be this app reaching into somewhere it no
        longer has any business being.
        """
        self._clients.forget(user_id)
        self._forget_catalog(user_id)
        try:
            return self._store.purge(user_id)
        except Exception as error:
            log.warning("could not purge Composio rows for %s: %s", user_id, error)
            return 0

    # ---- catalogue ------------------------------------------------------

    def catalog(
        self, user_id: str, *, search: str = "", cursor: str | None = None, limit: int = 40
    ) -> ToolkitList:
        """What this user could connect, from their own Composio.

        Composio's own search and paging are used rather than fetching the
        whole catalogue and filtering here: it is several hundred toolkits, and
        the panel shows twenty at a time.
        """
        sdk = self._sdk(user_id)

        key = (user_id, cursor or "", search.strip().lower())
        cached = self._catalog.get(key)
        if cached and time.monotonic() - cached[0] < self._settings.composio_catalog_ttl_s:
            return cached[1]

        query: dict[str, Any] = {"limit": limit, "sort_by": "usage"}
        if search.strip():
            query["search"] = search.strip()[:64]
        if cursor:
            query["cursor"] = cursor

        try:
            # The raw client rather than `toolkits.get`, which drops the cursor
            # on the way back and so cannot page.
            response = sdk.client.toolkits.list(**query)
        except Exception as error:
            log.warning("could not read the catalogue for %s: %s", user_id, error)
            raise IntegrationError("Composio did not answer — try again in a moment.") from error

        payload = ToolkitList(
            toolkits=[self._toolkit(item) for item in (response.items or [])],
            next_cursor=getattr(response, "next_cursor", None),
        )
        self._catalog[key] = (time.monotonic(), payload)
        return payload

    def _toolkit(self, item: Any) -> Toolkit:
        managed = list(getattr(item, "composio_managed_auth_schemes", None) or [])
        no_auth = bool(getattr(item, "no_auth", False))

        categories = [
            getattr(category, "name", None) or getattr(category, "id", "")
            for category in (_meta(item, "categories", []) or [])
        ]

        return Toolkit(
            slug=getattr(item, "slug", ""),
            name=getattr(item, "name", "") or getattr(item, "slug", ""),
            description=_meta(item, "description"),
            logo=_meta(item, "logo"),
            categories=[c for c in categories if c],
            tools=int(_meta(item, "tools_count", 0) or 0),
            # Connectable means "this can open a consent screen unaided".
            # Without a Composio-managed scheme it needs an auth config the
            # user creates in their own dashboard first, so the row is shown
            # and disabled rather than hidden — hiding it makes the toolkit
            # look unsupported when it is one setting away.
            connectable=bool(managed) and not no_auth,
            no_auth=no_auth,
        )

    # ---- auth configs ---------------------------------------------------

    def _auth_config(self, sdk: Composio, user_id: str, toolkit: str) -> tuple[str, bool]:
        """The auth config this user's toolkit connects through.

        Looked up in three places in increasing order of cost: our own table,
        their Composio's list (a project that already had one from the
        dashboard), and finally a create in their project.
        """
        remembered = self._store.auth_config(user_id, toolkit)
        if remembered:
            auth_config_id, managed = remembered
            try:
                # Confirm it still exists — a config deleted in the dashboard
                # would otherwise 404 every connect attempt forever.
                config = sdk.auth_configs.get(auth_config_id)
                return auth_config_id, bool(getattr(config, "is_composio_managed", managed))
            except Exception:
                log.info("auth config %s for %s is gone — recreating", auth_config_id, toolkit)
                self._store.forget_auth_config(user_id, toolkit)

        try:
            existing = sdk.auth_configs.list(toolkit_slug=toolkit, limit=1)
            items = list(getattr(existing, "items", None) or [])
        except Exception as error:
            raise IntegrationError("Composio did not answer — try again in a moment.") from error

        if items:
            found = items[0]
            auth_config_id = getattr(found, "id", "")
            managed = bool(getattr(found, "is_composio_managed", True))
            if auth_config_id:
                self._store.remember_auth_config(
                    user_id, toolkit, auth_config_id, composio_managed=managed
                )
                return auth_config_id, managed

        try:
            created = sdk.auth_configs.create(
                toolkit=toolkit,
                options={"type": "use_composio_managed_auth"},
            )
        except Exception as error:
            log.warning("could not create an auth config for %s: %s", toolkit, error)
            raise IntegrationError(
                f"{toolkit} needs an auth config in your Composio dashboard before it "
                "can be connected.",
                status_code=409,
            ) from error

        # `create` answers with {auth_config, toolkit} rather than the config
        # itself, which is the one place this SDK's shapes are not uniform.
        config = getattr(created, "auth_config", created)
        auth_config_id = getattr(config, "id", "")
        if not auth_config_id:
            raise IntegrationError("Composio created an auth config without an id.")

        self._store.remember_auth_config(
            user_id, toolkit, auth_config_id, composio_managed=True
        )
        return auth_config_id, True

    # ---- connecting a toolkit -------------------------------------------

    def connect(self, user_id: str, toolkit: str) -> ConnectStarted:
        """Open a consent screen for this user, in their own Composio project.

        The row is written before the URL is handed back, so a user who closes
        the tab on the consent screen leaves something to reconcile rather than
        a connection nobody here knows about.
        """
        slug = _slug(toolkit)
        if not slug:
            raise IntegrationError("That is not a toolkit.", status_code=400)

        sdk = self._sdk(user_id)
        auth_config_id, managed = self._auth_config(sdk, user_id, slug)

        try:
            request = self._start(sdk, slug, user_id, auth_config_id, managed=managed)
        except Exception as error:
            log.warning("could not start a %s connection for %s: %s", slug, user_id, error)
            raise IntegrationError(f"Composio would not start a {slug} connection.") from error

        redirect_url = getattr(request, "redirect_url", None)
        if not redirect_url:
            # Every scheme this app offers is redirectable, so a request with
            # nowhere to send the browser is a state the panel cannot render.
            raise IntegrationError(f"{slug} did not return a consent URL.")

        status = str(getattr(request, "status", "INITIALIZING") or "INITIALIZING")
        self._store.open(
            user_id=user_id,
            toolkit=slug,
            auth_config_id=auth_config_id,
            connected_account_id=getattr(request, "id", None),
            status=status,
        )

        return ConnectStarted(toolkit=slug, redirect_url=redirect_url, status=status)

    def _callback(self, toolkit: str) -> str:
        """Where consent lands, carrying the toolkit it was for.

        Composio appends its own query string to whatever is handed over, so
        the slug goes on as a parameter: the landing page has to read the
        toolkit back out of `location.search` either way, and a parameter is
        the half of the URL it is guaranteed to survive in.

        Without it the page knows consent happened and not what for, which is
        the difference between "Slack is connected" and "something worked".
        """
        return f"{self._clients.callback_url}?toolkit={quote(toolkit, safe='')}"

    def _start(
        self, sdk: Composio, toolkit: str, user_id: str, auth_config_id: str, *, managed: bool
    ):
        """`link()` for Composio's own OAuth apps, `initiate()` for your own.

        Both take the same arguments and both come back with `redirect_url`.
        The split exists because `initiate()` now refuses managed OAuth configs
        outright — see this module's docstring — and `link()` is not the right
        call for a custom app, which needs no third-party consent interstitial.
        """
        accounts = sdk.connected_accounts
        callback = self._callback(toolkit)

        if managed:
            return accounts.link(user_id, auth_config_id, callback_url=callback)
        return accounts.initiate(user_id, auth_config_id, callback_url=callback)

    # ---- reading --------------------------------------------------------

    def connections(self, user_id: str) -> ConnectionList:
        """This user's connections, reconciled against their Composio.

        A dead upstream degrades to what was last written down rather than to
        an error: the panel showing "Gmail, connected" from a stale row is more
        useful than a panel showing nothing, and the next click reconciles.

        Not connected to Composio at all is not an error either — it is the
        resting state of a new account, and the panel renders it as the thing
        to do next.
        """
        if not self._clients.configured:
            return ConnectionList(configured=False)

        try:
            sdk = self._sdk(user_id)
        except IntegrationError as error:
            # 409 is "Composio is not connected", which is a resting state and
            # not a failure — the panel renders the connector list instead.
            if error.status_code == 409:
                return ConnectionList(configured=True)
            raise

        try:
            live = sdk.connected_accounts.list(user_ids=[user_id])
            self._store.reconcile(user_id, self._pairs(live))
        except Exception as error:
            log.warning("could not reconcile connections for %s: %s", user_id, error)

        rows = self._store.list(user_id)
        names = self._names(user_id, {row.toolkit for row in rows})
        return ConnectionList(
            connections=[self._connection(row, names) for row in rows],
            configured=True,
        )

    def _pairs(self, response: Any) -> Iterable[tuple[str, str, str]]:
        """(toolkit, account id, status) out of a Composio list response.

        `toolkit` arrives as a nested object on some responses and as a bare
        slug on others, so both are accepted. An item naming no toolkit is
        dropped rather than stored under an empty slug, which would collide
        with every other such item on the unique constraint.
        """
        for item in getattr(response, "items", None) or []:
            toolkit = getattr(item, "toolkit", None)
            slug = getattr(toolkit, "slug", None) if toolkit is not None else None
            slug = _slug(slug or (toolkit if isinstance(toolkit, str) else ""))
            account_id = getattr(item, "id", None)
            if not slug or not account_id:
                continue
            yield slug, account_id, str(getattr(item, "status", "") or "INITIALIZING")

    def _names(self, user_id: str, slugs: set[str]) -> dict[str, tuple[str, str | None]]:
        """Display name and logo per slug, best effort.

        Straight off this user's cached catalogue rather than a lookup per
        connection: this is decoration, and three extra round trips to render
        three rows in a popover is not a trade worth making. A slug the
        catalogue has not cached renders under its own name, which is readable
        enough.
        """
        found: dict[str, tuple[str, str | None]] = {}
        for (cached_user, _, _), (_, payload) in self._catalog.items():
            if cached_user != user_id:
                continue
            for toolkit in payload.toolkits:
                if toolkit.slug in slugs:
                    found[toolkit.slug] = (toolkit.name, toolkit.logo)
        return found

    def _connection(
        self, row: StoredConnection, names: dict[str, tuple[str, str | None]]
    ) -> Connection:
        name, logo = names.get(row.toolkit, (None, None))
        return Connection(
            toolkit=row.toolkit,
            name=name,
            logo=logo,
            status=row.status,
            active=row.status == ACTIVE,
            pending=row.status in PENDING,
            connected_at=row.created_at,
            updated_at=row.updated_at,
        )

    def status(self, user_id: str, toolkit: str) -> Connection | None:
        """One toolkit, refreshed. What the callback page polls.

        Scoped by toolkit at Composio *and* by user in the store, so a
        connection is only ever reported to the account it was opened under.
        """
        slug = _slug(toolkit)
        row = self._store.get(user_id, slug)
        if row is None:
            return None

        try:
            sdk = self._sdk(user_id)
            live = sdk.connected_accounts.list(user_ids=[user_id], toolkit_slugs=[slug])
            for found_slug, account_id, status in self._pairs(live):
                if found_slug == slug:
                    row = (
                        self._store.mark(
                            user_id=user_id,
                            toolkit=slug,
                            status=status,
                            connected_account_id=account_id,
                        )
                        or row
                    )
                    break
        except Exception as error:
            log.info("could not refresh %s for %s: %s", slug, user_id, error)

        return self._connection(row, self._names(user_id, {slug}))

    # ---- disconnecting a toolkit ----------------------------------------

    def disconnect(self, user_id: str, toolkit: str) -> bool:
        """Revoke in their Composio, then forget locally.

        Upstream first, and the local row is kept if it fails. The other order
        loses the account id — the only handle on the grant — and leaves
        somebody's mailbox connected to an app whose UI says it is not.
        """
        slug = _slug(toolkit)
        row = self._store.get(user_id, slug)
        if row is None:
            return False

        if row.connected_account_id:
            try:
                self._sdk(user_id).connected_accounts.delete(row.connected_account_id)
            except IntegrationError:
                raise
            except Exception as error:
                log.warning(
                    "could not delete connected account %s: %s", row.connected_account_id, error
                )
                raise IntegrationError(
                    f"Composio would not disconnect {slug} — nothing was changed."
                ) from error

        return self._store.delete(user_id, slug) is not None


@lru_cache
def get_integration_service() -> IntegrationService:
    """One per process, because it owns the catalogue cache.

    Built per request it would still work and would simply never hit that
    cache — every panel open paying a Composio round trip for a list that
    changes weekly, and `_names` finding nothing to label rows with.
    """
    return IntegrationService(get_clients(), get_integration_store(), get_settings())
