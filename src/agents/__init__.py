"""Every agent in this system, and the base class they share.

An agent here is a component that decides something with a model and is called
as a unit — handed a question and some measured context, returning a typed
answer that never arrives as an exception. They were previously spread across
the packages that happened to call them first (`src/rag/`, `src/datasets/`,
`src/integrations/`), which hid the fact that they all keep the same three
promises. `src/agents/base.py` is where those promises are written down.

    ToolAgent        what a user's linked Composio accounts can do, and doing it
    DatasetAgent     a question in English → guarded DuckDB SQL → rows
    RagAgents        the ask ladder's five stages: routing, relevance grading,
                     query rewriting, synthesis, answer grading

Every model call in here is LangChain: the single-shot stages are
`prompt | model | parser` chains, and `DatasetAgent` is a `create_agent` tool
loop. The prompts themselves live in `src/prompts/<name>.md` — one file per agent,
named for its `name` — and `src/agents/prompts.py` is the only thing that reads
them.

What is deliberately *not* here: the model clients (`src/rag/llm.py`,
`src/voice/llm.py`) — an agent is a decision, a client is a socket, and the
voice client streams tokens into a synthesiser that is already speaking; the
tools themselves (`src/tools/`) — what a tool returns, the `query_dataset`
schema the voice loop offers, and the guarded `run_sql` the dataset agent
calls: things agents run, not things that decide; and the profile narrators (`src/connectors/narrate.py`,
`src/datasets/narrate.py`), which are single model calls owned by the measuring
that produced their input, and whose own docstrings describe what they write as
being *for* an agent rather than by one.
"""

from src.agents.base import BaseAgent, ModelAgent
from src.agents.model import TolerantJson, chat_model
from src.agents.dataset_agent import Answer, DatasetAgent, get_dataset_agent
from src.agents.rag import (
    NO_ANSWER,
    AnswerGrader,
    RagAgents,
    Relevance,
    RelevanceGrader,
    Route,
    RouterAgent,
    QueryRewriter,
    SynthesisAgent,
    Verdict,
    context_block,
)
from src.agents.tool_agent import ToolAgent, get_agent

__all__ = [
    "NO_ANSWER",
    "Answer",
    "AnswerGrader",
    "BaseAgent",
    "DatasetAgent",
    "ModelAgent",
    "QueryRewriter",
    "RagAgents",
    "Relevance",
    "RelevanceGrader",
    "Route",
    "RouterAgent",
    "SynthesisAgent",
    "ToolAgent",
    "TolerantJson",
    "Verdict",
    "chat_model",
    "context_block",
    "get_agent",
    "get_dataset_agent",
]
