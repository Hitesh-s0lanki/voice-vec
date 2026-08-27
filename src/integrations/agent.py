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

import logging
import time
from functools import lru_cache
from typing import Any

from src.core.config import Settings, get_settings
from src.integrations.client import ComposioClients, NotConnected, get_clients
from src.integrations.store import ACTIVE, IntegrationStore, get_integration_store

log = logging.getLogger("vec.integrations.agent")

# A tool result can be an entire inbox page. It goes back into the prompt, so it
# is capped: past a few thousand characters it stops adding anything a spoken
# answer can use and starts costing latency on every subsequent turn.
MAX_RESULT_CHARS = 4_000


class ToolResult:
    """What running one tool produced, in a shape both the model and the
    database can take."""

    __slots__ = ("slug", "ok", "data", "error", "ms")

    def __init__(
        self, slug: str, *, ok: bool, data: Any = None, error: str | None = None, ms: float = 0.0
    ) -> None:
        self.slug = slug
        self.ok = ok
        self.data = data
        self.error = error
        self.ms = ms

    def for_model(self) -> str:
        """The string that goes back as the `tool` message.

        A failure is reported rather than hidden. The model handling "that
        mailbox is not reachable" out loud is a better turn than it inventing
        an answer from a silence.
        """
        import json

        if not self.ok:
            return json.dumps({"error": self.error or "the tool failed"})

        try:
            rendered = json.dumps(self.data, default=str)
        except Exception:
            rendered = str(self.data)

        if len(rendered) > MAX_RESULT_CHARS:
            return rendered[:MAX_RESULT_CHARS] + "… (truncated)"
        return rendered


class ToolAgent:
    def __init__(
        self, clients: ComposioClients, store: IntegrationStore, settings: Settings
    ) -> None:
        self._clients = clients
        self._store = store
        self._settings = settings
        # user_id → (fetched_at, toolkits it was built for, schemas)
        self._cache: dict[str, tuple[float, frozenset[str], list[dict]]] = {}

    def active_toolkits(self, user_id: str) -> frozenset[str]:
        """The toolkits this user has actually linked and that are live.

        Read from our own table rather than from Composio: it is the row we
        wrote, it is already scoped by user id, and it costs no round trip on
        a path where the listener is waiting.
        """
        if not user_id:
            return frozenset()

        try:
            return frozenset(
                row.toolkit for row in self._store.list(user_id) if row.status == ACTIVE
            )
        except Exception as error:
            log.warning("could not read linked toolkits for %s: %s", user_id, error)
            return frozenset()

    def tools_for(self, user_id: str) -> list[dict]:
        """OpenAI-format tool schemas for everything this user has linked.

        Empty is the common case and the fast one. Every failure also lands
        here as empty: a turn that cannot list tools should be answered without
        them, not dropped.
        """
        toolkits = self.active_toolkits(user_id)
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
            # The SDK's default provider is OpenAI's, so these come back in the
            # exact shape `llm.build_payload` puts on the wire.
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
            log.warning("could not fetch tool schemas for %s: %s", user_id, error)
            return []

        self._cache[user_id] = (time.monotonic(), toolkits, schemas)
        return schemas

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

        try:
            response = sdk.tools.execute(slug, arguments, user_id=user_id)
        except Exception as error:
            log.warning("tool %s failed for %s: %s", slug, user_id, error)
            return ToolResult(
                slug,
                ok=False,
                error=f"{type(error).__name__}",
                ms=(time.perf_counter() - started) * 1000,
            )

        ms = (time.perf_counter() - started) * 1000

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
