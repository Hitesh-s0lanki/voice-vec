"""Every tool an agent can run, and the shape a tool call comes back in.

A **tool** here is something with an effect or an answer *outside* the model:
somebody's mailbox, somebody's dataset. `src/agents/` decides; this decides
nothing and does the thing.

They come in two shapes, because two different loops call them:

    the voice loop      OpenAI-format schemas merged into one tool pass, run
                        through `ToolAgent.execute` / `DatasetTools.execute`,
                        every call written to `tool_calls` for the panel to
                        show. `dataset.py` builds the `query_dataset` half.
    a LangChain agent   `StructuredTool` objects handed to `create_agent`.
                        `sql.py` builds `run_sql`, the one tool the dataset
                        agent's loop calls.

    result.py       ToolResult — what any call produced, for the model, the
                    database and the browser, with failures stated rather than
                    hidden
    capabilities.py find_capability — which of this person's connected things
                    bears on the request, and the call that uses it
    store.py        search_store — retrieval against the store discovery named
    dataset.py      query_dataset — ask a dataset in English; `DatasetAgent`
                    writes the SQL against the measured schema
    sql.py          run_sql — one guarded SELECT in the sealed DuckDB sandbox
    kit.py          which of the above is callable this round, and who runs
                    what is called

The Composio tools are not files here: they are somebody's linked accounts,
discovered per user at the moment of the turn, so the code that owns them is
`ToolAgent` (`src/agents/tool_agent.py`) and what it hands back is
`result.ToolResult`. What this package owns is everything *this* codebase
wrote a tool for.

They do not all arrive at once. `kit.py` offers `find_capability` and holds the
rest back until a discovery names them, so a turn carries the schemas for what
was asked about rather than for everything somebody has ever connected
(docs/23-capabilities.md).

One rule across all of it, the same one `src/agents/base.py` states: **a tool
that fails returns a failure, it does not raise.** The turn is mid-sentence and
somebody is listening; "that mailbox is not reachable" spoken out loud beats a
dropped turn, and a tool that returns an empty result on failure teaches the
model to report "no results" for "the database is down".
"""

from src.tools.capabilities import FIND_TOOL, CapabilityTools, get_capability_tools
from src.tools.dataset import QUERY_TOOL, DatasetTools, get_dataset_tools
from src.tools.kit import ToolKit
from src.tools.result import MAX_RESULT_CHARS, ToolResult, toolkit_of
from src.tools.sql import RUN_SQL, SqlRun, reason
from src.tools.store import SEARCH_TOOL, StoreTools, get_store_tools

__all__ = [
    "FIND_TOOL",
    "MAX_RESULT_CHARS",
    "QUERY_TOOL",
    "RUN_SQL",
    "SEARCH_TOOL",
    "CapabilityTools",
    "DatasetTools",
    "SqlRun",
    "StoreTools",
    "ToolKit",
    "ToolResult",
    "get_capability_tools",
    "get_dataset_tools",
    "get_store_tools",
    "reason",
    "toolkit_of",
]
