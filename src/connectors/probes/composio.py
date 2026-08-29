"""What the agent can *do*, which is the other half of "what are my capabilities".

Every other probe in this package answers "what is in the store". This one
answers a question with the same shape and completely different stakes: a
retrieval capability that is wrong costs a bad answer, and an *action*
capability that is wrong costs a sent email.

That asymmetry decides what gets recorded. The interesting number is not how
many tools Composio offers — it offers hundreds, and the agent can reach none
of them — but which toolkits **this user has actually authorised**, because a
connected account is the only thing that turns a listed tool into one that
executes. So `authorised` comes from `connected_accounts`, and `toolkits` from
the catalogue, and the card quotes the first.

Tool *schemas* are deliberately not stored here. `IntegrationAgent` already
fetches them per turn and caches them on a TTL — they are large, they change
under the user, and a second copy in a profile row would be a stale answer to a
question something else already answers correctly. The profile records the
shape of what is reachable; the agent still asks for the schemas at the moment
it needs them.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from src.connectors.probes.base import PROBE_TIMEOUT_S, unreachable
from src.connectors.profile import Observation, ToolShape

log = logging.getLogger("vec.connectors.probe")

#: The catalogue is long and the card has a line for it. Enough to say what
#: kind of project this is without turning a system prompt into a directory.
MAX_TOOLKITS = 40

#: Named action slugs, for the profile's JSON rather than the card. A person
#: debugging "why did it not send the email" wants to see whether
#: GMAIL_SEND_EMAIL was ever reachable.
MAX_ACTIONS = 60


class ComposioProbe:
    def __init__(self, credentials: Mapping[str, str], *, user_id: str, **_: Any) -> None:
        self._api_key = credentials["api_key"]
        self._user_id = user_id

    @property
    def location(self) -> str:
        return "composio"

    def observe(self) -> Observation:
        from src.integrations.mcp import is_gateway_key

        started = time.perf_counter()

        if is_gateway_key(self._api_key):
            return self._observe_gateway(started)

        from composio import Composio  # local: keeps the SDK off the import path

        try:
            sdk = Composio(api_key=self._api_key, timeout=int(PROBE_TIMEOUT_S))
        except Exception as error:
            log.info("composio probe failed to build a client: %s", type(error).__name__)
            return unreachable("composio", "tools", self.location, f"could not reach Composio: {error}")

        authorised = self._authorised(sdk)
        toolkits = self._toolkits(sdk)
        actions = self._actions(sdk, authorised)

        notes: list[str] = []
        if not authorised:
            # The single most common state, and the one worth being explicit
            # about: the credential is fine, the project is real, and the agent
            # can still do nothing until a toolkit is connected.
            notes.append("no toolkit is authorised yet — the agent cannot act on anything")

        return Observation(
            connector="composio",
            kind="tools",
            location=self.location,
            reachable=True,
            sampled=len(actions),
            tools=ToolShape(
                toolkits=tuple(toolkits[:MAX_TOOLKITS]),
                actions=tuple(actions[:MAX_ACTIONS]),
                authorised=tuple(authorised),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes),
        )

    def _observe_gateway(self, started: float) -> Observation:
        """The same question, asked of the MCP gateway.

        The shape of the answer is identical and the way it is reached is not.
        The gateway has no catalogue call and no "list every connected
        account" call, so both come out of one search: the toolkits it surfaces
        for a spread of common use cases, and then the live status of exactly
        those. That makes this a *sample* rather than a census — which is why
        `sampled` is the count of what was actually looked at, and why a
        toolkit connected in the dashboard but outside the sample simply does
        not appear rather than being reported as unauthorised.
        """
        from src.integrations.mcp import ComposioGateway, GatewayError

        gateway = ComposioGateway(self._api_key, timeout=PROBE_TIMEOUT_S)

        try:
            toolkits = [entry["slug"] for entry in gateway.search_toolkits("", limit=MAX_TOOLKITS)]
        except GatewayError as error:
            log.info("composio gateway probe failed: %s", error)
            return unreachable(
                "composio", "tools", self.location, f"could not reach Composio: {error}"
            )

        authorised: list[str] = []
        try:
            for slug, _account_id, status in gateway.connections(toolkits):
                if status in ("ACTIVE", "INITIALIZED") and slug not in authorised:
                    authorised.append(slug)
        except GatewayError as error:
            log.debug("gateway connection listing failed: %s", error)

        actions: list[str] = []
        if authorised:
            try:
                actions = sorted(
                    schema["function"]["name"]
                    for schema in gateway.tools_for(authorised, limit=MAX_ACTIONS)
                )
            except GatewayError as error:
                log.debug("gateway tool listing failed: %s", error)

        notes: list[str] = []
        if not authorised:
            notes.append("no toolkit is authorised yet — the agent cannot act on anything")

        return Observation(
            connector="composio",
            kind="tools",
            location=self.location,
            reachable=True,
            sampled=len(actions),
            tools=ToolShape(
                toolkits=tuple(sorted(toolkits)[:MAX_TOOLKITS]),
                actions=tuple(actions[:MAX_ACTIONS]),
                authorised=tuple(sorted(authorised)),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes),
        )

    def _authorised(self, sdk: Any) -> list[str]:
        """Toolkits with a live connected account for this user.

        Wrapped defensively and returning a list rather than raising, because
        this runs against an SDK whose response shapes are not this app's to
        pin. A probe that dies on an unexpected field would take the whole
        profile with it over a cosmetic change upstream.
        """
        try:
            live = sdk.connected_accounts.list(user_ids=[self._user_id])
        except Exception as error:
            log.debug("composio connected_accounts failed: %s", error)
            return []

        found: set[str] = set()
        for account in _items(live):
            slug = _slug(account)
            status = str(_get(account, "status") or "").upper()
            # ACTIVE is the only status that executes. INITIATED and FAILED are
            # accounts somebody started connecting and did not finish, and
            # reporting them as capabilities is how an agent confidently
            # announces it can read a mailbox it cannot open.
            if slug and status in ("", "ACTIVE", "INITIALIZED"):
                found.add(slug.lower())
        return sorted(found)

    def _toolkits(self, sdk: Any) -> list[str]:
        try:
            listed = sdk.client.toolkits.list(limit=MAX_TOOLKITS)
        except Exception as error:
            log.debug("composio toolkits.list failed: %s", error)
            return []
        return sorted({s.lower() for s in (_slug(t) for t in _items(listed)) if s})

    def _actions(self, sdk: Any, toolkits: list[str]) -> list[str]:
        """The named tools reachable through the authorised toolkits.

        Only asked for when something is authorised: with no connected account
        this returns the whole catalogue, which is a large, slow answer to a
        question whose true answer is "nothing".
        """
        if not toolkits:
            return []
        try:
            schemas = sdk.tools.get(user_id=self._user_id, toolkits=toolkits, limit=MAX_ACTIONS)
        except Exception as error:
            log.debug("composio tools.get failed: %s", error)
            return []

        names: list[str] = []
        for schema in _items(schemas):
            function = _get(schema, "function") or {}
            name = _get(function, "name") or _get(schema, "name") or _get(schema, "slug")
            if name:
                names.append(str(name))
        return sorted(set(names))


def _items(response: Any) -> list[Any]:
    """Whatever this SDK version wraps a list in.

    Composio has returned bare lists, `.items` and `.data` across versions, and
    a probe is exactly the wrong place to find out which by raising.
    """
    if response is None:
        return []
    for attribute in ("items", "data"):
        value = getattr(response, attribute, None)
        if isinstance(value, list):
            return value
    if isinstance(response, Mapping):
        for key in ("items", "data"):
            if isinstance(response.get(key), list):
                return response[key]
    return list(response) if isinstance(response, (list, tuple)) else []


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _slug(obj: Any) -> str:
    for key in ("toolkit_slug", "slug", "toolkit", "app_name", "appName", "name"):
        value = _get(obj, key)
        if isinstance(value, Mapping):
            value = value.get("slug") or value.get("name")
        if value:
            return str(value)
    return ""
