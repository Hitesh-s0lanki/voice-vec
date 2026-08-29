"""Searching one *named* store, once discovery has decided which one.

`query_dataset` answers the countable half and this answers the readable half:
a question in words, against the vector store capability discovery pointed at.

    find_capability("student records") ──► pgvector
    search_store(store="pgvector", question="how many are enrolled?")

**The store is an argument, not an ambient fact.** `BackendResolver` picks one
store per user by a standing order, which is the right default and the wrong
answer for somebody with three attached: the Pinecone of product docs and the
Postgres of student records answer different questions, and only the question
knows which. So the slug rides in, and the resolver treats it as a preference —
a name this user has not connected falls back to the standing order rather than
failing, because the name came from a model.

**An abstention is a successful call.** "No source covers this" is the answer
the ladder is built to produce (docs/22-no-local-corpus.md), and reporting it as
a tool failure would have the model retry a search that will keep succeeding at
finding nothing. `ok=False` here means the *call* failed.

**The rung is the deployment's, not the model's.** How hard retrieval works is
the effort ladder's business (docs/15-effort.md), so the tool takes no dial: a
model that could ask for rung 4 on every turn would, and the listener pays for
it in seconds.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from src.capabilities.catalogue import STORE
from src.capabilities.index import CapabilityIndex, get_capability_index
from src.core.config import Settings, get_settings
from src.tools.result import ToolResult

log = logging.getLogger("vec.tools.store")

SEARCH_TOOL = "search_store"

MAX_CITATIONS = 3


class StoreTools:
    def __init__(self, index: CapabilityIndex, settings: Settings) -> None:
        self._index = index
        self._settings = settings

    def owns(self, name: str) -> bool:
        return name == SEARCH_TOOL

    def stores(self, user_id: str) -> list[str]:
        """The stores that can actually answer, as the schema's enum.

        A blocked one is deliberately absent: discovery still *names* it, so
        the agent can say why it cannot be used, but offering it here would let
        the model call a search that is guaranteed to come back with nothing.
        """
        return [
            c.id
            for c in self._index.capabilities(user_id)
            if c.kind == STORE and not c.blocked
        ]

    def tools_for(self, user_id: str) -> list[dict]:
        """The schema, with this user's own store ids as the enum.

        From what is actually connected, so a slug invented out of the
        conversation is refused by the provider rather than becoming a
        resolution that quietly falls back to a different store.
        """
        if not user_id:
            return []
        connected = self.stores(user_id)
        if not connected:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": SEARCH_TOOL,
                    "description": (
                        "Search one of this person's connected vector stores and get a "
                        "grounded answer with its sources. Use it for questions about "
                        "the documents or records a store holds — after "
                        "`find_capability` has named which store."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store": {
                                "type": "string",
                                "enum": connected,
                                "description": "Which store, as named by find_capability.",
                            },
                            "question": {
                                "type": "string",
                                "description": (
                                    "The question, in the asker's own words and language."
                                ),
                            },
                        },
                        "required": ["store", "question"],
                    },
                },
            }
        ]

    def execute(self, user_id: str, name: str, arguments: dict) -> ToolResult:
        """Run one search. Never raises."""
        started = time.perf_counter()
        if not self.owns(name):
            return ToolResult(name, ok=False, error=f"{name} is not a store tool")

        store = str((arguments or {}).get("store") or "").strip()
        question = str((arguments or {}).get("question") or "").strip()
        if not store or not question:
            return ToolResult(
                name, ok=False, error="A store and a question are both required."
            )

        # Imported here rather than at module scope: the ask pipeline pulls in
        # the agents, which import this package — and this is a tool call, so
        # the cost lands on a turn that is already retrieving.
        from src.schemas.ask import AskRequest
        from src.services.ask_service import get_ask_service

        try:
            answer = get_ask_service().ask(
                AskRequest(transcript=question), user_id=user_id, store=store
            )
        except Exception as error:
            log.warning("search of %s failed for %s: %s", store, user_id, error)
            return ToolResult(
                name, ok=False, error=type(error).__name__, ms=_since(started)
            )

        if answer.status != "answered":
            # No passages on an abstention, and this is the whole reason the
            # branch exists. `Citation` means "where the answer came from — or,
            # on an abstention, what was rejected", so passing them through
            # would hand the model the exact text the guardrails just refused,
            # under the heading `sources`. It would answer from them, fluently,
            # and the refusal would have cost a round trip and changed nothing.
            return ToolResult(
                name,
                ok=True,
                data={
                    "store": store,
                    "status": answer.status,
                    "reason": answer.reason or "nothing in that store covers it",
                    "sources": [],
                    "advice": (
                        "That store was searched and does not answer this. Say so — "
                        "do not fill the gap from memory."
                    ),
                },
                ms=_since(started),
            )

        return ToolResult(
            name,
            ok=True,
            data={
                "store": store,
                "status": answer.status,
                "answer": answer.answer or "",
                "sources": [
                    {"id": c.doc_id, "text": c.text[: self._settings.extract_max_chars]}
                    for c in (answer.citations or [])[:MAX_CITATIONS]
                ],
            },
            ms=_since(started),
        )


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


@lru_cache
def get_store_tools() -> StoreTools:
    return StoreTools(get_capability_index(), get_settings())
