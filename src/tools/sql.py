"""The one tool `DatasetAgent`'s loop calls: guarded SQL against a sealed file.

This is the tool half of the dataset agent, kept apart from the loop that
drives it because the two answer different questions. The agent decides *when*
to query and when to stop; this decides *what a query is allowed to be* and
what the model is told when it is not.

    run_sql ──► guard.guard ──► Sandbox.run ──► rows
                    │                │
                    └── Rejected ────┴── DuckDB error ──► back to the model

**The guard is not relaxed for anybody.** It is what makes the connection safe
to hold open at all: read-only, no ATTACH, no file reads, one statement. The
model asking nicely is not an argument, and neither is a human — `DatasetAgent.run`
puts hand-written SQL through this same call.

**The error message is the repair.** DuckDB's binder says "Did you mean ...?",
and `guard.Rejected` is written to be read by a model — "this is a read-only
view of the dataset" produces a working second attempt where "invalid SQL"
produces the same query again. Both go back verbatim.

**What ran is recorded here, not reconstructed later.** `SqlRun` accumulates the
statement, the rows and the attempt count as the tool executes, so the caller
builds its answer from evidence rather than from the model's own account of
what it did — the one source in the loop that is not evidence.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from src.core.config import Settings
from src.datasets import sql as guard
from src.datasets.sandbox import Result, SandboxUnavailable
from src.datasets.service import DatasetService

#: The tool name the model calls. Lower snake case, so it cannot collide with a
#: Composio slug (`GMAIL_SEND_EMAIL`) if the two ever share a tool list.
RUN_SQL = "run_sql"

_DESCRIPTION = (
    "Run one read-only DuckDB SELECT against this dataset and return what came back. "
    "On failure it returns the database's own error; read it and call this once more "
    "with a corrected query."
)


class SqlRun:
    """One question's worth of tool calls, and what they produced.

    Constructed per question, because it *is* the record of that question. The
    tool handed to the agent closes over it, which is what makes "what SQL ran"
    a fact rather than a reading of the transcript.
    """

    def __init__(
        self, datasets: DatasetService, settings: Settings, dataset_id: str, path: str
    ) -> None:
        self._datasets = datasets
        self._settings = settings
        self.dataset_id = dataset_id
        self.path = path
        self.sql = ""
        self.error = ""
        self.result: Result | None = None
        self.attempts = 0

    @property
    def ok(self) -> bool:
        return self.result is not None

    def tool(self) -> StructuredTool:
        """The tool object, bound to this run."""
        return StructuredTool.from_function(
            func=self.attempt, name=RUN_SQL, description=_DESCRIPTION
        )

    def attempt(self, sql: str) -> str:
        """Guard it, run it, remember it — and tell the model what happened.

        What goes back is deliberately thin on success and verbose on failure.
        On success the loop stops here (the agent's `before_model` hook) and
        the caller renders the rows itself, so a row dump would be tokens
        nobody reads. On failure the message is the whole repair.
        """
        self.attempts += 1
        self.result = None

        try:
            statement = guard.guard(sql)
        except guard.Rejected as rejection:
            self.sql = guard.clean(sql)
            self.error = str(rejection)
            return f"rejected: {self.error}"

        self.sql = statement
        try:
            result = self._datasets.sandbox.run(
                self.path,
                statement,
                max_rows=self._settings.dataset_query_rows,
                timeout_s=self._settings.dataset_query_timeout_s,
            )
        except SandboxUnavailable as error:
            self.error = str(error)
            return f"failed: {self.error}"
        except Exception as error:
            self.error = reason(error)
            return f"failed: {self.error}"

        self.result = result
        self.error = ""
        rows = len(getattr(result, "rows", ()) or ())
        return f"ok: {rows} row(s) returned"


def reason(error: Exception) -> str:
    """DuckDB's own first line, which names the column or function at fault.

    Passed back to the model verbatim. DuckDB's binder errors carry a
    "Did you mean ...?" that fixes a misremembered column in one round, and
    replacing them with a generic message throws that away.
    """
    name = type(error).__name__
    if "Interrupt" in name:
        return "The query took too long and was stopped. Make it narrower."
    first = str(error).strip().splitlines()
    return (first[0] if first else name)[:300]
