"""What the agent can actually do on somebody's behalf, and doing it.

Connecting a toolkit is only half of it. This is the half that matters at the
microphone: turning a user's *active* Composio connections into tool schemas
the model can ask for, and running what it asks for against that user's own
Composio project.

Three rules this module exists to keep.

**No connections, no tools, no cost.** `tools_for` returns an empty list the
moment a user has nothing linked, and the turn loop skips the whole tool pass
when it does. Somebody who has never opened the Connectors panel pays nothing —
not a round trip, not a token of schema in their prompt. That matters because
the tool pass is buffered (`llm.complete`) and buffering is exactly what the
voice path spends its latency budget avoiding.

**Only what they linked.** The schemas are fetched per toolkit, from the
toolkits that reconciled as ACTIVE for that user. Handing the model Composio's
whole catalogue would be a prompt the size of a book and an invitation to call
something the user never authorised.

**Execution is scoped to the caller, always.** `execute` passes `user_id` to
Composio, so a tool runs against the connected account belonging to the person
who spoke — never an ambient one, because there is no ambient one to reach for.

Tool schemas are cached briefly. They change when somebody links a toolkit,
which the cache key covers by including the set of active toolkits, so a fresh
connection shows up on the next turn rather than after a timeout.
"""

from __future__ import annotations

import time
from functools import lru_cache

from src.agents.base import BaseAgent
from src.core.config import Settings, get_settings
from src.integrations.client import ComposioClients, NotConnected, get_clients
from src.integrations.mcp import ComposioGateway
from src.integrations.store import ACTIVE, IntegrationStore, get_integration_store
# The tool layer: what a call returns, and how a slug names its toolkit. This
# agent decides *which* tools exist and runs them; `src/tools/` owns what comes
# back, because the voice loop and the dataset tool read the same shape.
from src.tools.result import ToolResult, toolkit_of


class ToolAgent(BaseAgent):
    """The one agent here that runs no model of its own.

    It sits on the other side of the loop: the voice model decides *that* a
    tool should run, and this decides which tools that decision may choose from
    and runs the one it picked. `needs_model` is False for exactly that reason —
    a deployment with no model key has nothing for this to serve, but a
    deployment with no Composio credentials is the one where it can do nothing,
    which is what `ready` reports instead.
    """

    name = "tools"
    needs_model = False

    def __init__(
        self, clients: ComposioClients, store: IntegrationStore, settings: Settings
    ) -> None:
        super().__init__(settings)
        self._clients = clients
        self._store = store
        # user_id → (fetched_at, toolkits it was built for, schemas)
        self._cache: dict[str, tuple[float, frozenset[str], list[dict]]] = {}

    @property
    def ready(self) -> bool:
        """Composio configured, not a model key — see the class docstring."""
        return self._clients.configured

    def active_toolkits(self, user_id: str) -> frozenset[str]:
        """The toolkits this user has actually linked and that are live.

        Read from our own table rather than from Composio: it is the row we
        wrote, it is already scoped by user id, and it costs no round trip on
        a path where the listener is waiting.
        """
        if not user_id:
            return frozenset()

        return self._guard(
            f"reading linked toolkits for {user_id}",
            lambda: frozenset(
                row.toolkit for row in self._store.list(user_id) if row.status == ACTIVE
            ),
            default=frozenset(),
        )

    def tools_for(self, user_id: str, *, only: frozenset[str] | None = None) -> list[dict]:
        """OpenAI-format tool schemas for what this user has linked.

        Empty is the common case and the fast one. Every failure also lands
        here as empty: a turn that cannot list tools should be answered without
        them, not dropped.

        `only` narrows to the toolkits a caller actually wants — capability
        discovery names one, and fetching the schemas for just that one is the
        difference between a prompt carrying somebody's whole Composio account
        and one carrying the mailbox they asked about (`src/tools/kit.py`). A
        toolkit that is not linked cannot be asked for: the narrowing is an
        intersection with what reconciled as ACTIVE, never a way past it.
        """
        toolkits = self.active_toolkits(user_id)
        if only is not None:
            toolkits = frozenset(toolkits & only)
        if not toolkits:
            return []

        cached = self._cache.get(user_id)
        if (
            cached
            and cached[1] == toolkits
            and time.monotonic() - cached[0] < self._settings.tool_schema_ttl_s
        ):
            return cached[2]

        try:
            sdk = self._clients.for_user(user_id)
            if isinstance(sdk, ComposioGateway):
                # The gateway indexes tools by use case rather than by toolkit
                # and returns Composio's own schema shape, so it does the
                # translation to OpenAI format itself — there is no provider
                # object in this path to do it.
                schemas = sdk.tools_for(
                    sorted(toolkits), limit=self._settings.tool_schema_limit
                )
            else:
                # The SDK's default provider is OpenAI's, so these come back in
                # the exact shape `llm.build_payload` puts on the wire.
                schemas = list(
                    sdk.tools.get(
                        user_id=user_id,
                        toolkits=sorted(toolkits),
                        limit=self._settings.tool_schema_limit,
                    )
                )
        except NotConnected:
            return []
        except Exception as error:
            self.log.warning("could not fetch tool schemas for %s: %s", user_id, error)
            return []

        self._cache[user_id] = (time.monotonic(), toolkits, schemas)
        return schemas

    def inventory(self, user_id: str) -> dict[str, list[dict[str, str | None]]]:
        """The same tools, grouped by toolkit, for a panel to show.

        Read through `tools_for` rather than off Composio's catalogue on
        purpose. A catalogue count is what a toolkit *has*; this is what this
        account can actually be handed on the next turn — bounded by
        `tool_schema_limit`, and by which connections reconciled as ACTIVE. The
        panel is answering "what can it do for me", and those are different
        numbers whenever the limit bites.

        It also costs nothing extra while the panel is open: `tools_for` is the
        cache the voice path fills, keyed by the same user and toolkit set.
        """
        grouped: dict[str, list[dict[str, str | None]]] = {}

        for schema in self.tools_for(user_id):
            # OpenAI's shape nests the interesting half under `function`; the
            # gateway path builds the same shape by hand. Read either, because
            # a panel is not worth a KeyError.
            if not isinstance(schema, dict):
                continue
            body = schema.get("function")
            if not isinstance(body, dict):
                body = schema

            slug = str(body.get("name") or "").strip()
            if not slug:
                continue

            description = str(body.get("description") or "").strip()
            grouped.setdefault(toolkit_of(slug), []).append(
                {"slug": slug, "description": description or None}
            )

        for tools in grouped.values():
            tools.sort(key=lambda tool: tool["slug"] or "")
        return grouped

    def forget(self, user_id: str) -> None:
        self._cache.pop(user_id, None)

    def execute(self, user_id: str, slug: str, arguments: dict) -> ToolResult:
        """Run one tool as this user. Never raises.

        A tool that fails is a `ToolResult` with `ok=False`, because the turn
        has to continue either way — the model is mid-sentence and the listener
        is waiting. The exception would only travel as far as the caller's
        `except` and become this same object.
        """
        started = time.perf_counter()

        try:
            sdk = self._clients.for_user(user_id)
        except NotConnected as error:
            return ToolResult(slug, ok=False, error=str(error))

        if isinstance(sdk, ComposioGateway):
            # The gateway already reports per-tool failure without raising —
            # it runs up to fifty tools per call and one failing is not the
            # call failing — so it maps straight onto `ToolResult`.
            ok, data, error_text = sdk.execute(slug, arguments)
            ms = self._ms(started)
            if not ok:
                self.log.warning("tool %s failed for %s: %s", slug, user_id, error_text)
                return ToolResult(slug, ok=False, error=error_text or "the tool failed", ms=ms)
            return ToolResult(slug, ok=True, data=data, ms=ms)

        try:
            response = sdk.tools.execute(slug, arguments, user_id=user_id)
        except Exception as error:
            self.log.warning("tool %s failed for %s: %s", slug, user_id, error)
            return ToolResult(
                slug,
                ok=False,
                error=f"{type(error).__name__}",
                ms=self._ms(started),
            )

        ms = self._ms(started)

        # Composio reports failure in the body rather than by raising, so a
        # successful call is not the same thing as a successful tool.
        successful = getattr(response, "successful", None)
        data = getattr(response, "data", None)
        error = getattr(response, "error", None)

        if successful is False:
            return ToolResult(slug, ok=False, error=str(error or "the tool failed"), ms=ms)

        return ToolResult(slug, ok=True, data=data, ms=ms)


@lru_cache
def get_agent() -> ToolAgent:
    return ToolAgent(get_clients(), get_integration_store(), get_settings())
