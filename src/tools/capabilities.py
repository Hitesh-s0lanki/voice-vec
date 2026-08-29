"""The tool that answers "what can I use for this?" before anything is used.

This is the first call of the flow in docs/23-capabilities.md:

    question ──► find_capability(need) ──► "gmail — use GMAIL_FETCH_EMAILS"
                                      └──► "pgvector — search_store(...)"
                                      └──► "demo — query_dataset(...)"
             ──► the named tool ──► the answer

**It returns instructions, not data.** A match is a capability id, what it is
good for, and the exact call to make next. The agent still has to make that
call, which is the property that keeps a description from being mistaken for an
answer: a model handed a dataset card will happily read a number off it, and
the card carries real numbers.

**Nothing matched is a successful answer.** `ok=True` with an empty list and a
sentence saying so, because "nothing you have connected covers this" is a true
and useful thing for the agent to say out loud. Reporting it as a tool failure
would make the model retry a search that will keep succeeding at finding
nothing.

**It never enumerates everything.** The whole point of the indirection is that
the prompt does not grow with what somebody connects, so a search that matched
nothing does not fall back to "here is the list".
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from src.capabilities.catalogue import TOOLKIT, Capability
from src.capabilities.index import CapabilityIndex, Match, get_capability_index
from src.core.config import Settings, get_settings
from src.tools.result import ToolResult

log = logging.getLogger("vec.tools.capabilities")

#: The one discovery tool. Lower snake case, so it cannot collide with a
#: Composio slug.
FIND_TOOL = "find_capability"

_DESCRIPTION = (
    "Find out what you can use to answer a question or carry out a request for this "
    "person — their connected stores, datasets and authorised accounts. Call this "
    "FIRST, before answering, whenever the request needs data you were not given or "
    "an action outside this conversation: mail, calendars, documents, records, "
    "counts, or anything specific about their own data.\n\n"
    "It returns what can serve the request and the exact tool call to make next. "
    "It returns nothing when they have connected nothing that covers it — in which "
    "case say you do not have it rather than answering from memory."
)


class CapabilityTools:
    """`find_capability`, in the shape the voice loop's tool pass takes."""

    def __init__(self, index: CapabilityIndex, settings: Settings) -> None:
        self._index = index
        self._settings = settings

    def owns(self, name: str) -> bool:
        return name == FIND_TOOL

    def tools_for(self, user_id: str) -> list[dict]:
        """The schema, or nothing at all when there is nothing to discover.

        Same rule as every other tool source here: somebody who has connected
        nothing pays no schema and no round trip. The check is a profile read
        from memory, not a probe.
        """
        if not user_id or not self._settings.capabilities_enabled:
            return []
        if not self._index.capabilities(user_id):
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": FIND_TOOL,
                    "description": _DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "need": {
                                "type": "string",
                                "description": (
                                    "What you need, in plain words — 'read this person's "
                                    "recent email', 'student records', 'average marks by "
                                    "class'. Use the words of the request, not a tool name."
                                ),
                            }
                        },
                        "required": ["need"],
                    },
                },
            }
        ]

    def available(self, user_id: str) -> list[Capability]:
        """Everything this person has, for a caller that needs the count.

        The prompt says *that* there is something to discover; only this and
        the tool know what. Kept here rather than reaching into the index from
        the voice loop, so there is one door onto capabilities and it is this
        object.
        """
        return self._index.capabilities(user_id)

    def search(self, user_id: str, need: str) -> list[Match]:
        return self._index.search(
            user_id, need, limit=self._settings.capability_max_matches
        )

    def execute(self, user_id: str, name: str, arguments: dict) -> ToolResult:
        """Run one discovery. Never raises."""
        started = time.perf_counter()
        if not self.owns(name):
            return ToolResult(name, ok=False, error=f"{name} is not a capability tool")

        need = str((arguments or {}).get("need") or "").strip()
        if not need:
            return ToolResult(name, ok=False, error="A need is required.")

        try:
            matches = self.search(user_id, need)
        except Exception as error:  # `search` does not raise; this is the belt
            log.warning("capability search failed for %s: %s", user_id, error)
            return ToolResult(
                name, ok=False, error=type(error).__name__, ms=_since(started)
            )

        ms = _since(started)
        if not matches:
            return ToolResult(
                name,
                ok=True,
                data={
                    "found": [],
                    "advice": (
                        "Nothing this person has connected covers that. Say you do not "
                        "have it — do not answer from memory."
                    ),
                },
                ms=ms,
            )

        return ToolResult(
            name,
            ok=True,
            data={
                "found": [match.capability.as_dict() for match in matches],
                "advice": (
                    "Call the tool named in `use` for the one that fits. Anything "
                    "specific — a count, a record, a message — must come from that "
                    "call, never from this description."
                ),
            },
            ms=ms,
        )

    @staticmethod
    def toolkits(result: ToolResult) -> frozenset[str]:
        """Which toolkits a discovery surfaced, for the caller to unlock.

        The schemas for somebody's whole Composio account are a prompt the size
        of a book. Reading them off the result means a turn carries the tools
        for what was actually asked about and nothing else.
        """
        if not result.ok or not isinstance(result.data, dict):
            return frozenset()
        return frozenset(
            str(found.get("id"))
            for found in result.data.get("found") or []
            if isinstance(found, dict) and found.get("kind") == TOOLKIT
        )


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


@lru_cache
def get_capability_tools() -> CapabilityTools:
    return CapabilityTools(get_capability_index(), get_settings())
