"""Every tool one turn may call, and the order they become callable in.

A turn does not start with the user's whole account in its prompt. It starts
with one tool — `find_capability` — and what that returns *unlocks* the tools
for the thing it named:

    round 1   find_capability            "check my inbox"  → gmail
    round 2   GMAIL_FETCH_EMAILS         (unlocked by round 1)
    round 3   the spoken answer

The alternative, and what this replaced, was handing the model every schema for
every linked toolkit on every turn. That is a prompt that grows with what
somebody connects, paid for on turns that need none of it, and it is buffered —
the tool pass cannot be streamed, so it sits in front of the first spoken word
(docs/23-capabilities.md).

**Unlocking is per turn and forward-only.** A new `ToolKit` per turn, because
"which tools are live" is a fact about *this* question; nothing carries over,
so a turn cannot act on a capability discovered while answering something else.

**Discovery is the only thing that is free.** `find_capability` is offered
whenever the user has anything at all. Everything else — the stores, the
datasets, somebody's mail — appears only after a search has said it is
relevant, and a session that discovers nothing runs exactly one extra tool
schema over a session with no tools at all.

**Dispatch by name, in one place.** Three sources answer to different objects,
and every one of them returns `ToolResult`, so the caller — which is streaming
audio and writing audit rows — never learns which is which.
"""

from __future__ import annotations

import logging

from src.agents.tool_agent import ToolAgent
from src.capabilities.catalogue import DATASET, STORE
from src.core.config import Settings
from src.tools.capabilities import CapabilityTools
from src.tools.dataset import DatasetTools
from src.tools.result import ToolResult
from src.tools.store import StoreTools

log = logging.getLogger("vec.tools.kit")


class ToolKit:
    """One turn's tools: what is callable now, and who runs what is called."""

    __slots__ = (
        "_capabilities", "_stores", "_datasets", "_composio", "_settings",
        "_toolkits", "_kinds",
    )

    def __init__(
        self,
        *,
        capabilities: CapabilityTools,
        stores: StoreTools,
        datasets: DatasetTools,
        composio: ToolAgent,
        settings: Settings,
    ) -> None:
        self._capabilities = capabilities
        self._stores = stores
        self._datasets = datasets
        self._composio = composio
        self._settings = settings
        #: Composio toolkits a discovery has named this turn.
        self._toolkits: set[str] = set()
        #: `store` / `dataset`, once discovery has pointed at one.
        self._kinds: set[str] = set()

    # ---- what is callable right now --------------------------------------

    def schemas(self, user_id: str) -> list[dict]:
        """The tool list for the next round. Never raises.

        Rebuilt each round rather than computed once, because the point is that
        it changes: what round 1 discovers is what round 2 may call.

        **No discovery, no gate.** When `find_capability` is not on offer — the
        feature switched off, profiling off, or a store linked thirty seconds
        ago whose probe has not finished — every tool is offered the way it was
        before this existed. Gating behind a discovery that cannot run would
        take somebody's mailbox away *because* the description of it is
        missing, which is the one outcome worse than a prompt that is too long.
        """
        if not self._settings.tools_enabled or not user_id:
            return []

        discovery = self._safe("discovery schemas", self._capabilities.tools_for, user_id)
        if not discovery:
            return self._everything(user_id)

        tools = list(discovery)
        if STORE in self._kinds:
            tools += self._safe("store schemas", self._stores.tools_for, user_id)
        if DATASET in self._kinds:
            tools += self._safe("dataset schemas", self._datasets.tools_for, user_id)
        if self._toolkits:
            tools += self._safe(
                "toolkit schemas",
                lambda uid: self._composio.tools_for(uid, only=frozenset(self._toolkits)),
                user_id,
            )
        return tools

    def _everything(self, user_id: str) -> list[dict]:
        """The pre-discovery shape: both sources, whole, in one list.

        Kept as one list rather than two passes so a single turn can read
        somebody's mail *and* query their dataset; two loops would make those
        alternatives.
        """
        return self._safe(
            "toolkit schemas", self._composio.tools_for, user_id
        ) + self._safe("dataset schemas", self._datasets.tools_for, user_id)

    @property
    def discovered(self) -> bool:
        """Has anything been unlocked this turn? Read for the activity feed."""
        return bool(self._toolkits or self._kinds)

    # ---- running one call ------------------------------------------------

    def execute(self, user_id: str, name: str, arguments: dict) -> ToolResult:
        """Run one tool as this user, and unlock whatever it discovered.

        Never raises: every executor here already turns its failures into a
        `ToolResult`, and the one thing this must not do is end a turn that is
        mid-sentence.
        """
        if self._capabilities.owns(name):
            result = self._capabilities.execute(user_id, name, arguments)
            self._unlock(result)
            return result
        if self._stores.owns(name):
            return self._stores.execute(user_id, name, arguments)
        if self._datasets.owns(name):
            return self._datasets.execute(user_id, name, arguments)
        return self._composio.execute(user_id, name, arguments)

    def _unlock(self, result: ToolResult) -> None:
        """Turn a discovery into the tools that act on what it found."""
        self._toolkits |= self._capabilities.toolkits(result)
        if not result.ok or not isinstance(result.data, dict):
            return
        for found in result.data.get("found") or []:
            if isinstance(found, dict) and found.get("kind") in (STORE, DATASET):
                self._kinds.add(str(found["kind"]))

    # ---- and the one rule about failing ----------------------------------

    def _safe(self, what: str, call, user_id: str) -> list[dict]:
        """A source that cannot list its tools contributes none of them.

        Not the whole turn's tools: one unreachable table must not cost
        somebody the mailbox they linked, which is what a single `try` around
        the four of them would do.
        """
        try:
            return list(call(user_id))
        except Exception as error:
            log.warning("%s failed for %s: %s", what, user_id, error)
            return []
