"""Composio's MCP gateway — the door a `ck_` key opens.

Composio ships two credentials for two products and they are not
interchangeable. A platform key (`ak_…`) authenticates the REST API the Python
SDK wraps: `toolkits.list`, `auth_configs`, `connected_accounts`,
`tools.execute`. A gateway key (`ck_…`) authenticates one JSON-RPC endpoint
exposing seven meta-tools and no REST surface at all — `backend.composio.dev`
answers 401 to it, and there is no base URL that answers otherwise.

So this is not a wrapper around the SDK. It is a second transport, and every
operation the app needs is re-expressed in terms the gateway actually has:

    toolkits.list             COMPOSIO_SEARCH_TOOLS       (semantic, not a listing)
    tools.get                 COMPOSIO_GET_TOOL_SCHEMAS
    tools.execute             COMPOSIO_MULTI_EXECUTE_TOOL
    connected_accounts.list   COMPOSIO_MANAGE_CONNECTIONS  action=list
    connected_accounts.link   COMPOSIO_MANAGE_CONNECTIONS  action=add
    connected_accounts.delete COMPOSIO_MANAGE_CONNECTIONS  action=remove
    auth_configs.*            — nothing, and nothing is needed

The last two lines are the interesting ones. `add` returns a consent URL
directly, which collapses the entire auth-config dance in `IntegrationService`
into one call: there is no `ac_…` to find, create or remember, because the
gateway owns it. And the first line is the one real loss — the gateway has no
"list every toolkit" call, only a semantic search returning a handful of tools
per query. The catalogue is search-shaped here, and `search_toolkits` reports
no cursor rather than faking one the panel would page into nothing.

**Sessions are the sharp edge.** The endpoint is stateful: `initialize` returns
an `mcp-session-id` that every later call must carry, and it expires server
side without warning. Calls therefore re-establish and retry exactly once —
once, because a genuinely bad key would otherwise loop, and on a bad key the
second failure is the honest one to report.

**Every response shape here is read defensively.** These are meta-tool payloads
rather than a versioned SDK's models, they are documented by example only, and
a panel that 500s because a field moved is a worse outcome than one that shows
an empty row. Same reasoning as `probes/composio.py`, for the same reason.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Mapping, Sequence

import httpx

# The store's vocabulary, because that is where these strings are compared.
# `connections` decides what is worth reporting and the store decides what a
# report means; keeping one list means they cannot drift apart.
from src.integrations.store import DEAD

log = logging.getLogger("vec.integrations.mcp")

#: The gateway. One endpoint for every toolkit, unlike the REST API's many.
GATEWAY_URL = "https://connect.composio.dev/mcp"

#: Composio's gateway keys carry this prefix and platform keys carry `ak_`.
#: The prefix is the only thing that distinguishes them before a network call,
#: and picking the wrong transport costs a 401 the user cannot act on.
GATEWAY_PREFIX = "ck_"

#: MCP revision this client speaks. Pinned rather than "latest" so a server
#: rev cannot change the response framing under a running deployment.
PROTOCOL_VERSION = "2024-11-05"

#: The meta-tools this module drives. `tools/list` returning these is what
#: `verify` treats as proof the key works.
SEARCH = "COMPOSIO_SEARCH_TOOLS"
SCHEMAS = "COMPOSIO_GET_TOOL_SCHEMAS"
EXECUTE = "COMPOSIO_MULTI_EXECUTE_TOOL"
CONNECTIONS = "COMPOSIO_MANAGE_CONNECTIONS"


def is_gateway_key(api_key: str) -> bool:
    """Whether this credential belongs to the gateway rather than the REST API.

    Called before any client is built. A `ck_` key handed to the SDK and an
    `ak_` key handed to the gateway both fail with an authentication error that
    tells the user to check a key which is, in fact, perfectly valid — so the
    fork happens here, on the one signal available without a round trip.
    """
    return (api_key or "").strip().startswith(GATEWAY_PREFIX)


class GatewayError(RuntimeError):
    """The gateway refused, or did not answer."""


class ComposioGateway:
    """One JSON-RPC session against the gateway, for one user's key.

    Not thread safe by accident: the session id is mutable shared state and
    these are cached per user, so two voice turns for the same person can land
    here at once. The lock is around session establishment rather than every
    call, because that is the only part that races.
    """

    def __init__(self, api_key: str, *, url: str = GATEWAY_URL, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._url = url
        self._timeout = timeout
        self._session_id: str | None = None
        self._lock = threading.Lock()

    # ---- transport ------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-consumer-api-key": self._api_key,
            "Content-Type": "application/json",
            # The endpoint answers in SSE framing, and says so only if asked.
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    def _post(self, body: dict) -> tuple[int, dict | None]:
        """One JSON-RPC round trip. Returns the status and the parsed envelope.

        The body comes back as `text/event-stream` — one `data:` line carrying
        the whole response rather than a stream of deltas — so this reads the
        last such line instead of assuming a bare JSON body. A notification
        answers 202 with no body at all, which is not an error.
        """
        try:
            response = httpx.post(
                self._url, headers=self._headers(), json=body, timeout=self._timeout
            )
        except Exception as error:
            raise GatewayError(f"Composio's gateway did not answer: {error}") from error

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        raw = response.text or ""
        parsed: dict | None = None
        for line in raw.splitlines():
            if line.startswith("data: "):
                try:
                    parsed = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
        if parsed is None and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None

        return response.status_code, parsed

    def _open_session(self) -> None:
        """`initialize` then `notifications/initialized`, per the MCP handshake.

        Skipping the notification leaves the server holding a half-open session
        that rejects the first real call, which reads exactly like a bad key.
        """
        self._session_id = None
        status, envelope = self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "voice-vec", "version": "1"},
                },
            }
        )
        if status == 401 or (envelope or {}).get("error"):
            raise GatewayError("Composio's gateway rejected that key.")
        if status >= 400 or not self._session_id:
            raise GatewayError(f"Composio's gateway would not open a session ({status}).")

        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _rpc(self, method: str, params: dict) -> dict:
        """A call, re-establishing the session once if it has gone stale.

        The retry is deliberately not a loop. A session that expired comes back
        on the second attempt; a key that is wrong fails identically forever,
        and retrying it would turn one clear 401 into a hang.
        """
        with self._lock:
            if not self._session_id:
                self._open_session()

        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        status, envelope = self._post(body)

        if status in (400, 404) or (envelope or {}).get("error"):
            with self._lock:
                self._open_session()
            status, envelope = self._post(body)

        if status >= 400:
            raise GatewayError(f"Composio's gateway answered {status}.")
        if envelope is None:
            raise GatewayError("Composio's gateway sent an empty response.")
        if envelope.get("error"):
            message = (envelope["error"] or {}).get("message") or "the gateway refused"
            raise GatewayError(str(message))

        return envelope.get("result") or {}

    def call(self, tool: str, arguments: dict) -> dict:
        """Run one meta-tool and unwrap Composio's own envelope.

        Two layers of failure sit on top of each other and they mean different
        things: MCP's `error` (the call never ran) is raised by `_rpc`, and
        Composio's `successful: false` inside the content (the call ran and the
        operation failed) is raised here. Collapsing them would report a dead
        endpoint and a missing Slack connection with the same sentence.
        """
        result = self._rpc("tools/call", {"name": tool, "arguments": arguments})

        payload: dict = {}
        for chunk in result.get("content") or []:
            if isinstance(chunk, Mapping) and chunk.get("type") == "text":
                try:
                    payload = json.loads(chunk.get("text") or "{}")
                except json.JSONDecodeError:
                    continue

        if not isinstance(payload, Mapping):
            return {}

        if payload.get("successful") is False and not payload.get("data"):
            raise GatewayError(str(payload.get("error") or f"{tool} failed"))

        return dict(payload)

    # ---- what the app actually needs ------------------------------------

    def tool_names(self) -> list[str]:
        """The meta-tools this key can reach. The cheapest authenticated call."""
        result = self._rpc("tools/list", {})
        return [
            str(tool.get("name"))
            for tool in (result.get("tools") or [])
            if isinstance(tool, Mapping) and tool.get("name")
        ]

    def verify(self) -> None:
        """Prove the key works, or raise saying it does not."""
        if not self.tool_names():
            raise GatewayError("Composio's gateway listed no tools for that key.")

    def connections(self, toolkits: Sequence[str] = ()) -> list[tuple[str, str, str]]:
        """(toolkit, account id, STATUS) for this key's connected accounts.

        Shaped to feed `IntegrationStore.reconcile` unchanged, which is what
        keeps the store, the panel and the voice path indifferent to which
        transport produced the rows. Statuses are upper-cased because the
        gateway says `active` where the REST API says `ACTIVE` and the store's
        `ACTIVE` comparison is exact.

        With no toolkits named there is nothing to ask about — the gateway
        takes a list of toolkits, not a wildcard — so the caller passes the
        slugs it wants the state of and gets back those it has an account for.

        A toolkit with no account is not reported at all. That is the whole
        difference between this and the REST list: `connected_accounts.list`
        answers with what exists, whereas this answers about whatever it was
        asked, connected or not. Passing the "not connected" half on would turn
        every toolkit the caller merely wondered about into a row.
        """
        wanted = [slug for slug in toolkits if slug]
        if not wanted:
            return []

        payload = self.call(
            CONNECTIONS, {"toolkits": [{"name": slug, "action": "list"} for slug in wanted]}
        )
        results = ((payload.get("data") or {}).get("results")) or {}
        if not isinstance(results, Mapping):
            return []

        found: list[tuple[str, str, str]] = []
        for slug, entry in results.items():
            if not isinstance(entry, Mapping):
                continue
            accounts = entry.get("accounts") or []
            if not accounts:
                # A toolkit the gateway knows and has no account for. Its
                # *toolkit-level* status is not news: the gateway answers
                # `initiated` for every toolkit it has never connected, so
                # relaying that would file a "waiting for consent" row against
                # a service nobody has touched — and the panel polls every
                # three seconds for as long as one of those exists. Only a
                # genuinely dead status says something an empty answer does
                # not, so only that is passed on.
                status = str(entry.get("status") or "").upper()
                if status in DEAD:
                    found.append((str(slug).lower(), "", status))
                continue
            for account in accounts:
                if not isinstance(account, Mapping):
                    continue
                account_id = str(account.get("id") or "")
                status = str(account.get("status") or entry.get("status") or "").upper()
                if account_id:
                    found.append((str(slug).lower(), account_id, status or "INITIALIZING"))
        return found

    def add_connection(self, toolkit: str) -> tuple[str, str]:
        """Start consent for a toolkit. Returns (consent URL, status).

        This is the whole of `_auth_config` plus `_start` on the REST path. The
        gateway owns the auth config, so there is no `ac_…` to look up, create,
        remember or repair — the reason the gateway branch of `connect()` is
        four lines where the SDK branch is forty.

        The URL's key is not documented and has moved between gateway builds,
        so it is searched for rather than indexed: a consent flow that breaks
        on a renamed field is the failure this whole module is meant to end.
        """
        payload = self.call(CONNECTIONS, {"toolkits": [{"name": toolkit, "action": "add"}]})
        data = payload.get("data") or {}

        url = _find_url(data)
        if not url:
            raise GatewayError(f"Composio's gateway did not return a consent URL for {toolkit}.")

        status = "INITIALIZING"
        results = data.get("results")
        if isinstance(results, Mapping):
            entry = results.get(toolkit) or results.get(toolkit.lower())
            if isinstance(entry, Mapping) and entry.get("status"):
                status = str(entry["status"]).upper()

        return url, status

    def remove_connection(self, toolkit: str, account_id: str) -> None:
        """Revoke one connected account in the user's own Composio."""
        self.call(
            CONNECTIONS,
            {"toolkits": [{"name": toolkit, "action": "remove", "account_id": account_id}]},
        )

    def search_toolkits(self, query: str, *, limit: int = 40) -> list[dict[str, Any]]:
        """The catalogue, as far as a semantic search can stand in for one.

        The gateway has no listing call, so a browse becomes a search and an
        empty search becomes a spread of common use cases — enough to populate
        a panel that would otherwise open blank. Each result names its toolkits
        and its tool slugs; the toolkits are what the panel renders.

        Returned as plain dicts because the two transports disagree about what
        is knowable: the REST catalogue carries a description, a logo, a
        category and a tool count, and the gateway carries a slug. Inventing
        the rest would put fabricated copy under real logos.
        """
        use_cases = [query.strip()] if query.strip() else list(_SEED_USE_CASES)
        payload = self.call(
            SEARCH,
            {
                "queries": [{"use_case": case[:1024]} for case in use_cases],
                "session": {"generate_id": True},
            },
        )

        slugs: dict[str, int] = {}
        for result in ((payload.get("data") or {}).get("results")) or []:
            if not isinstance(result, Mapping):
                continue
            for toolkit in result.get("toolkits") or []:
                slug = _toolkit_slug(toolkit)
                if slug:
                    slugs[slug] = slugs.get(slug, 0) + 1
            for tool_slug in list(result.get("primary_tool_slugs") or []):
                # `GMAIL_SEND_EMAIL` → `gmail`. The toolkits field is the
                # better source and this is the fallback for a result that
                # names tools without naming what they belong to.
                head = str(tool_slug).split("_", 1)[0].lower()
                if head and head not in slugs:
                    slugs[head] = 0

        ordered = sorted(slugs.items(), key=lambda pair: (-pair[1], pair[0]))
        return [{"slug": slug} for slug, _ in ordered[:limit]]

    def schemas(self, slugs: Sequence[str]) -> list[dict]:
        """OpenAI-format tool schemas, the shape `llm.build_payload` sends.

        Built here rather than by a provider object because the gateway returns
        Composio's own `{toolkit, tool_slug, description, input_schema}` and
        there is no SDK in this path to translate it.
        """
        wanted = [slug for slug in slugs if slug]
        if not wanted:
            return []

        payload = self.call(SCHEMAS, {"tool_slugs": list(wanted), "include": ["input_schema"]})
        described = ((payload.get("data") or {}).get("tool_schemas")) or {}
        if not isinstance(described, Mapping):
            return []

        built: list[dict] = []
        for slug, schema in described.items():
            if not isinstance(schema, Mapping):
                continue
            built.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(schema.get("tool_slug") or slug),
                        "description": str(schema.get("description") or "")[:1024],
                        "parameters": schema.get("input_schema")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return built

    def tools_for(self, toolkits: Sequence[str], *, limit: int = 40) -> list[dict]:
        """Schemas for everything reachable through the named toolkits.

        Two calls rather than one: the gateway indexes tools by use case and by
        slug, never by toolkit, so the slugs are discovered by searching for
        each toolkit by name and then described in a single batch.
        """
        wanted = [slug for slug in toolkits if slug]
        if not wanted:
            return []

        payload = self.call(
            SEARCH,
            {
                "queries": [{"use_case": f"use the {slug} app"} for slug in wanted],
                "session": {"generate_id": True},
            },
        )

        slugs: list[str] = []
        for result in ((payload.get("data") or {}).get("results")) or []:
            if not isinstance(result, Mapping):
                continue
            # Both lists. The search ranks a handful of tools as "primary" for
            # the phrasing it was given and the rest as related, and the split
            # is about that phrasing rather than about capability — taking only
            # the primaries makes the model's toolset swing between turns,
            # because the same query does not rank identically twice.
            found = list(result.get("primary_tool_slugs") or []) + list(
                result.get("related_tool_slugs") or []
            )
            for tool_slug in found:
                slug = str(tool_slug)
                # Only tools belonging to a toolkit the user actually linked.
                # The search is semantic and happily returns a Slack tool for
                # "use the gmail app", which the model must never be offered.
                if slug not in slugs and slug.split("_", 1)[0].lower() in wanted:
                    slugs.append(slug)

        return self.schemas(slugs[:limit])

    def execute(self, slug: str, arguments: dict) -> tuple[bool, Any, str | None]:
        """Run one tool. Returns (ok, data, error) and does not raise.

        The caller is `ToolAgent.execute`, which is mid-turn with somebody
        waiting, and every failure there has to become a `ToolResult` anyway.

        Composio reports per-tool failure inside a successful envelope — the
        multi-executor runs up to fifty tools and one of them failing is not
        the call failing — so the result array is what decides, not the status.
        """
        try:
            payload = self.call(
                EXECUTE,
                {"tools": [{"tool_slug": slug, "arguments": arguments}]},
            )
        except GatewayError as error:
            return False, None, str(error)

        results = ((payload.get("data") or {}).get("results")) or []
        entry = results[0] if results and isinstance(results[0], Mapping) else {}

        if entry.get("error"):
            return False, None, str(entry["error"])
        if payload.get("successful") is False:
            return False, None, str(payload.get("error") or "the tool failed")

        # Three envelopes deep: the MCP result, then Composio's multi-executor,
        # then the tool's own `{successful, data}`. The innermost one is the
        # one that says whether the *tool* worked — a Gmail call that comes
        # back `successful: false` rides inside a multi-execute that succeeded,
        # so unwrapping only the outer two reports a failed send as a sent one.
        inner: Any = None
        for key in ("response", "data", "result", "output"):
            if entry.get(key) is not None:
                inner = entry[key]
                break

        if isinstance(inner, Mapping) and ("successful" in inner or "data" in inner):
            if inner.get("successful") is False:
                return False, None, str(inner.get("error") or "the tool failed")
            return True, inner.get("data"), None

        return True, inner if inner is not None else (entry or None), None


#: Enough breadth that an unsearched panel opens with recognisable services
#: rather than nothing. Only used when the user has typed no query.
_SEED_USE_CASES = (
    "send an email",
    "post a message to a team chat channel",
    "create an issue in a code repository",
    "add a row to a spreadsheet",
    "create a calendar event",
    "save a note or document page",
)


def _toolkit_slug(value: Any) -> str:
    """A toolkit slug out of whatever the search wrapped it in."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Mapping):
        for key in ("slug", "name", "toolkit"):
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip().lower()
    return ""


def _find_url(value: Any, *, depth: int = 0) -> str:
    """The first consent URL anywhere in a payload.

    A search rather than a lookup because the field has been `redirect_url`,
    `auth_url` and `url` across gateway builds, at two different nesting
    depths. Bounded so a cyclic or pathological payload cannot spin.
    """
    if depth > 6:
        return ""
    if isinstance(value, str):
        return value if value.startswith("https://") else ""
    if isinstance(value, Mapping):
        for key in ("redirect_url", "auth_url", "connection_url", "url", "link"):
            found = value.get(key)
            if isinstance(found, str) and found.startswith("https://"):
                return found
        for nested in value.values():
            found = _find_url(nested, depth=depth + 1)
            if found:
                return found
        return ""
    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_url(nested, depth=depth + 1)
            if found:
                return found
    return ""
