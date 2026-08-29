"""Test doubles shared by the agent tests.

`FakeToolModel` is the one that matters: `DatasetAgent` is a LangChain agent
now, so faking "the model said X" means faking a *tool call*, not a completion.
The scripts it takes are read the way the agent reads the loop —

    "SELECT ..."    the model calls `run_sql` with this statement
    "!some text"    the model replies in prose without calling anything

— which is exactly the two behaviours the agent has to handle: a provider that
calls the tool, and one that writes the SQL into the message and hopes.

`bind_tools` returns `self` because there is nothing to bind: the scripts
already say what the model will ask for. It has to exist all the same, since
`create_agent` binds tools to whatever model it is given.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolModel(BaseChatModel):
    scripts: list[str] = []
    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        script = self.scripts.pop(0) if self.scripts else "!nothing scripted"
        if script.startswith("!"):
            message = AIMessage(content=script[1:])
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_sql", "args": {"sql": script}, "id": f"call-{self.calls}"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"


class FakeTextModel(BaseChatModel):
    """A model that just says things — for the single-shot stages."""

    replies: list[str] = []
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    @property
    def _llm_type(self) -> str:
        return "fake-text-model"
