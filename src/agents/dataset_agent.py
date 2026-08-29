"""The realtime half: a question in English, a row set out, and the SQL between.

This is the agent other agents call, and the only one here that runs a **tool
loop** rather than a single call. It never sees the network and never sees a
credential; it sees a schema card that was *measured* and a sealed connection
to a local file, and its whole job is to close the gap between those two.

    question ──► model ──► run_sql ──► guard ──► sandbox ──► rows
                   ▲                      │
                   └──── DuckDB's error ──┘   (once, and only once)

The loop is LangChain's (`create_agent`, which is a LangGraph state machine),
because that loop is exactly what this was hand-rolling: write, run, read the
error, correct. What is *not* LangChain's is where it stops, and both bounds
are here on purpose:

  ModelCallLimitMiddleware   `dataset_sql_repairs + 1` model calls, then end.
                             A second failure means the question cannot be
                             answered from these columns, and further attempts
                             are the model trying different wrong answers while
                             somebody waits.
  stop when answered         a `before_model` hook that jumps to `end` the
                             moment a query has run. Without it the graph pays
                             for one more completion to narrate rows the caller
                             is going to render itself — a full round trip, on
                             the voice path, for nothing.

**The schema card is the product.** Handing a model `DESCRIBE` output produces
SQL that is syntactically perfect and semantically wrong: `WHERE query_type =
'FACT'` against a column holding five values, none of them that one, returns
zero rows and reads exactly like an honest empty answer. What prevents it is
not a better prompt, it is `src/datasets/probe.py` having counted the column and
`profile.schema()` having written the five values down. Everything good about
the SQL here comes from that, not from this file. The prompt itself is
`src/prompts/dataset-sql.md`.

**Failure is an answer.** No model, a rejected statement, a query that will not
run — all come back as an `Answer` carrying the reason, never as an exception.
The caller is mid-turn with a listener waiting, and "I could not query that"
spoken out loud beats a dropped turn. It is also the honest output: a tool that
returns an empty row set on failure is a tool that teaches an agent to report
"no results" for "the database is down".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, before_model
from langchain_core.messages import AIMessage

from src.agents.base import ModelAgent
from src.core.config import Settings, get_settings
from src.datasets.sandbox import Result
from src.datasets.service import DatasetService, get_dataset_service
from src.tools.sql import SqlRun

#: SQL is short. This is a ceiling on a runaway CTE, not a budget to fill.
MAX_TOKENS = 500

#: Somebody is waiting through this, so it is well under the tool timeout the
#: voice loop applies on top of it.
TIMEOUT_S = 20.0


@dataclass(frozen=True, slots=True)
class Answer:
    """One question, and everything the caller needs to answer honestly.

    `sql` is here on success *and* on failure. A dataset answer that cannot be
    checked is a number somebody has to trust, and the query is the only thing
    that makes it checkable — the panel shows it, and a model that had to
    approximate the question can say what it actually asked.
    """

    dataset_id: str
    question: str = ""
    sql: str = ""
    result: Result | None = None
    error: str = ""
    attempts: int = 0
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.error


class DatasetAgent(ModelAgent):
    name = "dataset-sql"

    #: `src/prompts/dataset-sql.md` is a system message only — the user turn is the
    #: question, handed to the graph rather than rendered from a template.
    single_shot = False

    def __init__(self, datasets: DatasetService, settings: Settings) -> None:
        super().__init__(settings)
        self._datasets = datasets

    @property
    def _max_tokens(self) -> int:
        return MAX_TOKENS

    @property
    def _timeout_s(self) -> float:
        return TIMEOUT_S

    # ---- the two entry points -------------------------------------------

    def ask(self, user_id: str, dataset_id: str, question: str) -> Answer:
        """A question in English → SQL → rows. Never raises."""
        started = time.perf_counter()
        row = self._datasets.queryable(user_id, dataset_id)

        if row is None:
            return Answer(
                dataset_id,
                question=question,
                error=self._why_not(user_id, dataset_id),
                ms=self._ms(started),
            )
        if not self.ready:
            return Answer(
                dataset_id,
                question=question,
                error="No model is configured, so a question cannot be turned into SQL.",
                ms=self._ms(started),
            )

        run = self._run_for(row)
        graph = self._guard("building the agent", lambda: self._graph(run, row.schema_card))
        if graph is None:
            return self._answer(run, question, started, "The agent could not be built.")

        final = self._guard(
            "the SQL agent",
            lambda: graph.invoke({"messages": [{"role": "user", "content": question.strip()}]}),
        )
        if final is None:
            return self._answer(run, question, started, "The model did not answer.")

        if not run.attempts:
            # A provider that ignored the tool and wrote the SQL as prose. It
            # happens, and the query is right there — running it is a better
            # turn than reporting that the model held it the wrong way.
            self._salvage(run, final)

        return self._answer(run, question, started, "No query could be made to run.")

    def run(self, user_id: str, dataset_id: str, statement: str) -> Answer:
        """SQL somebody wrote themselves. Guarded identically. Never raises.

        The guard is not relaxed for a human. It is what makes the connection
        safe to hold open at all, and a path around it would be the one that is
        reachable from an HTTP request.
        """
        started = time.perf_counter()
        row = self._datasets.queryable(user_id, dataset_id)
        if row is None:
            return Answer(
                dataset_id,
                sql=statement,
                error=self._why_not(user_id, dataset_id),
                ms=self._ms(started),
            )

        run = self._run_for(row)
        run.attempt(statement)
        return self._answer(run, "", started, "The query did not run.")

    # ---- the tool, the graph, and the answer -----------------------------

    def _run_for(self, row) -> SqlRun:
        """The tool bound to one dataset, and the record of what it does."""
        return SqlRun(self._datasets, self._settings, row.dataset_id, row.local_path)

    def _answer(self, run: SqlRun, question: str, started: float, fallback: str) -> Answer:
        """Everything a run produced, as the one type the caller reads.

        `fallback` is only reached when the tool never recorded a failure of
        its own — the model never called it, or the graph never got that far —
        so the reason is the caller's to supply rather than the tool's.
        """
        return Answer(
            run.dataset_id,
            question=question,
            sql=run.sql,
            result=run.result,
            error="" if run.ok else (run.error or fallback),
            attempts=run.attempts,
            ms=self._ms(started),
        )

    def _graph(self, run: SqlRun, schema_card: str):
        """One agent, built per question.

        Per question rather than cached per dataset because the tool closes
        over `run` — the record of what this question actually executed, which
        is what the `Answer` is built from. Compiling the graph is local work
        measured in a millisecond or two, against a model call measured in
        hundreds.
        """

        @before_model(can_jump_to=["end"])
        def stop_when_answered(state, runtime):  # noqa: ANN001 - LangChain's signature
            if run.ok:
                return {"jump_to": "end"}
            return None

        return create_agent(
            self.model,
            tools=[run.tool()],
            system_prompt=self.prompt.system(schema=schema_card),
            middleware=[
                stop_when_answered,
                ModelCallLimitMiddleware(
                    thread_limit=self._settings.dataset_sql_repairs + 1,
                    exit_behavior="end",
                ),
            ],
        )

    def _salvage(self, run: SqlRun, final: dict) -> None:
        messages = final.get("messages") or []
        last = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None
        )
        if last is None:
            return
        text = last.content if isinstance(last.content, str) else str(last.content)
        if "select" in text.lower():
            run.attempt(text)

    def _why_not(self, user_id: str, dataset_id: str) -> str:
        """Why this dataset cannot answer, in a sentence somebody can act on.

        The four states are genuinely different actions — wait, look at the
        error, add it, or check the id — and one message for all of them sends
        people to the wrong one.
        """
        row = self._datasets.get(user_id, dataset_id)
        if row is None:
            attached = [d for d, _ in self._datasets.catalogue(user_id)]
            if attached:
                return f"No dataset called {dataset_id}. Attached: {', '.join(attached)}."
            return "No datasets are attached to this account."
        if row.status == "pending":
            return f"{dataset_id} is still being built. Try again shortly."
        if row.status == "failed":
            return f"{dataset_id} could not be built: {row.error or 'unknown error'}"
        return f"{dataset_id} is not queryable right now — it is being rebuilt."


@lru_cache
def get_dataset_agent() -> DatasetAgent:
    return DatasetAgent(get_dataset_service(), get_settings())
