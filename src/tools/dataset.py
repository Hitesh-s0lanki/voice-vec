"""How the voice loop gets at a dataset: one tool, described from what is attached.

The voice loop already runs a tool pass for Composio (`src/agents/tool_agent.py`),
and this plugs into it rather than beside it — same OpenAI schema shape, same
`ToolResult`, so `_use_tools` merges two lists and `_run_tool` dispatches on the
name. A second loop would mean a spoken turn could either act on somebody's
mailbox or query their dataset, never both in one answer.

Three rules, and the first two are the ones that keep a turn cheap.

**No datasets, no tool, no cost.** `tools_for` returns an empty list the moment
a user has nothing queryable attached, and the tool pass is skipped entirely
when both sources are empty. Somebody who has never added a dataset pays
nothing — not a round trip, not a token of schema. The tool pass is *buffered*
(`llm.complete`), and buffering is exactly what the voice path spends its
latency budget avoiding.

**The dataset ids are in the schema, as an enum.** Built from what this user
actually has, so the model picks from real ids instead of inventing one out of
the conversation — and a wrong pick becomes a schema violation the provider
rejects rather than a query against a dataset that does not exist.

**The SQL is written inside, not by the caller.** The outer model is answering
in speech and has never seen the column list; the schema card is thousands of
characters and belongs only on turns that query. So the tool takes a question
in English, and `DatasetAgent` — which is handed the measured schema — writes
the SQL. The query comes back in the result so the answer stays checkable.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from typing import TYPE_CHECKING

from src.core.config import Settings, get_settings
from src.datasets.service import DatasetService, get_dataset_service
from src.tools.result import ToolResult

if TYPE_CHECKING:  # pragma: no cover - the annotation only
    from src.agents.dataset_agent import DatasetAgent

log = logging.getLogger("vec.tools.dataset")

#: The one tool name. Checked by `_run_tool` to decide which agent runs a call,
#: so it must not collide with a Composio slug — those are upper snake case
#: (`GMAIL_SEND_EMAIL`), which this deliberately is not.
QUERY_TOOL = "query_dataset"


class DatasetTools:
    def __init__(
        self, datasets: DatasetService, agent: "DatasetAgent", settings: Settings
    ) -> None:
        self._datasets = datasets
        self._agent = agent
        self._settings = settings

    def owns(self, name: str) -> bool:
        return name == QUERY_TOOL

    def tools_for(self, user_id: str) -> list[dict]:
        """The schema for this user's datasets, or nothing at all.

        Nothing at all is the common case and the fast one. Every failure lands
        here as empty too: a turn that cannot list datasets should be answered
        without them, not dropped.
        """
        if not user_id or not self._settings.datasets_enabled:
            return []

        try:
            catalogue = self._datasets.catalogue(user_id)
        except Exception as error:
            log.warning("could not list datasets for %s: %s", user_id, error)
            return []

        if not catalogue:
            return []

        listing = "\n".join(f"- {dataset_id}: {title}" for dataset_id, title in catalogue)
        return [
            {
                "type": "function",
                "function": {
                    "name": QUERY_TOOL,
                    "description": (
                        "Answer a question by running SQL over one of this user's attached "
                        "datasets. Use it for counts, filters, aggregates, distributions and "
                        "lookups over structured data — not for open-ended questions about "
                        "documents, which retrieval answers.\n\n"
                        "Ask in plain English; the SQL is written for you against the real, "
                        "measured schema. The result carries the query that ran, the rows, and "
                        "whether they were cut short. Rows are the whole answer: report what "
                        "came back, say when it was truncated rather than treating a partial "
                        "result as a total, and treat counts as describing the loaded sample "
                        "when the result says so.\n\n"
                        f"Datasets available to this user:\n{listing}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset": {
                                "type": "string",
                                # From the catalogue, so a hallucinated id is
                                # refused by the provider rather than becoming
                                # a lookup that fails one round trip later.
                                "enum": [dataset_id for dataset_id, _ in catalogue],
                                "description": "Which dataset to query.",
                            },
                            "question": {
                                "type": "string",
                                "description": (
                                    "The question to answer, in plain English. Be specific about "
                                    "filters and about how many rows you want back."
                                ),
                            },
                        },
                        "required": ["dataset", "question"],
                    },
                },
            }
        ]

    def execute(self, user_id: str, name: str, arguments: dict) -> ToolResult:
        """Run one dataset query as this user. Never raises.

        Returns `ToolResult` rather than an `Answer` so the voice loop's
        existing plumbing — the client event, the audit row, the `tool`
        message — takes it without a second code path.
        """
        started = time.perf_counter()
        if not self.owns(name):
            return ToolResult(name, ok=False, error=f"{name} is not a dataset tool")

        dataset_id = str((arguments or {}).get("dataset") or "").strip()
        question = str((arguments or {}).get("question") or "").strip()
        if not dataset_id or not question:
            return ToolResult(
                name, ok=False, error="A dataset and a question are both required."
            )

        try:
            answer = self._agent.ask(user_id, dataset_id, question)
        except Exception as error:  # `ask` does not raise; this is the belt
            log.warning("dataset query failed for %s: %s", user_id, error)
            return ToolResult(
                name, ok=False, error=type(error).__name__, ms=_since(started)
            )

        ms = (time.perf_counter() - started) * 1000
        if not answer.ok:
            # The SQL goes back on failure too. A model told only "that failed"
            # will try the same question again; one that can see the query it
            # ran can say what it actually asked.
            return ToolResult(
                name,
                ok=False,
                error=f"{answer.error} (query: {answer.sql})" if answer.sql else answer.error,
                ms=ms,
            )

        assert answer.result is not None
        return ToolResult(
            name,
            ok=True,
            data={
                "dataset": answer.dataset_id,
                "sql": answer.sql,
                "result": answer.result.for_model(budget=self._settings.dataset_result_chars),
            },
            ms=ms,
        )


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


@lru_cache
def get_dataset_tools() -> DatasetTools:
    # Imported here rather than at module scope: `DatasetAgent` imports this
    # package's `sql` module for the tool it runs, and importing back into
    # `src.agents` from the top of this file would close that loop.
    from src.agents.dataset_agent import get_dataset_agent

    return DatasetTools(get_dataset_service(), get_dataset_agent(), get_settings())
