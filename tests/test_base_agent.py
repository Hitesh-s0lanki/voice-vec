"""The contract `src/agents/base.py` makes, tested where breaking it is silent.

Every agent in `src/agents/` runs inside something with a deadline — a spoken
turn, or a rung of the ask ladder — and each promise below is one whose breach
looks like something else entirely:

  - an agent that raises drops a turn, and the traceback names the provider
    rather than the stage that could not run;
  - an agent that reports `ready` from the wrong configuration makes a
    deployment with no Composio credentials look like one with no model key;
  - a grader that defaults instead of returning `None` turns a provider hiccup
    into an approval, in the direction that emits answers rather than
    withholding them.

The model is faked at `src.agents.base.chat_model`, which is the seam every
`ModelAgent` builds its chain through. Nothing here reaches a network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.agents.base as base
from src.agents.base import BaseAgent, ModelAgent
from src.agents.rag import RelevanceGrader, RouterAgent, SynthesisAgent
from src.agents.tool_agent import ToolAgent
from src.core.config import get_settings
from tests.fakes import FakeTextModel


def settings(**overrides):
    """Settings that say what this test means, not what the checkout has."""
    return get_settings().model_copy(update={"openai_api_key": "test-key", **overrides})


def keyless():
    return get_settings().model_copy(
        update={"openai_api_key": "", "sarvam_api_key": "", "llm_base_url": ""}
    )


class Hit(SimpleNamespace):
    """Enough of `src.rag.store.Hit` for `context_block`."""

    def rendering(self, *, english: bool = False) -> str:
        return self.text


def _hits() -> list[Hit]:
    return [Hit(chunk_id="c1", text="Sleep is consolidated by the first cycle.")]


def _model(monkeypatch, *replies: str) -> FakeTextModel:
    model = FakeTextModel(replies=list(replies))
    monkeypatch.setattr(base, "chat_model", lambda *a, **k: model)
    return model


class TestNeverRaises:
    def test_guard_turns_an_exception_into_the_default(self, caplog):
        """And logs it, because a swallowed failure nobody can see is how a
        degraded rung becomes a mystery."""

        class Boom(BaseAgent):
            name = "boom"

        agent = Boom(settings())
        with caplog.at_level("WARNING"):
            out = agent._guard("reading toolkits", lambda: 1 / 0, default=frozenset())

        assert out == frozenset()
        assert "ZeroDivisionError" in caplog.text

    def test_a_provider_that_throws_is_none_not_an_exception(self, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(base, "chat_model", explode)
        assert SynthesisAgent(settings()).write(query="why?", hits=_hits()) is None

    def test_no_model_configured_costs_nothing_at_all(self, monkeypatch):
        """Not a client, not a socket — the readiness check comes first."""
        built = []
        monkeypatch.setattr(base, "chat_model", lambda *a, **k: built.append(1))

        assert SynthesisAgent(keyless()).write(query="why?", hits=_hits()) is None
        assert built == []

    def test_a_verdict_that_will_not_parse_is_none_not_a_default(self, monkeypatch):
        """The whole point of the rule: `None` falls back to the deterministic
        guardrail, where a default would approve."""
        _model(monkeypatch, "sure, sounds relevant", "no idea really")

        assert RelevanceGrader(settings()).grade(query="why?", hits=_hits()) is None
        assert RouterAgent(settings()).route(query="hello", corpus="books") is None


class TestParsingIsToleratedButNeverRepaired:
    def test_json_after_prose_is_still_read(self, monkeypatch):
        """Providers without a JSON mode do this constantly, and
        `response_format` is deliberately not sent."""
        _model(monkeypatch, 'Sure! {"destination": "direct", "reason": "greeting"}')

        route = RouterAgent(settings()).route(query="hello", corpus="books")
        assert route is not None and route.destination == "direct"

    def test_an_unknown_destination_falls_back_to_retrieving(self, monkeypatch):
        """Retrieving unnecessarily costs a little time; skipping retrieval
        wrongly costs the answer."""
        _model(monkeypatch, '{"destination": "websearch", "reason": "?"}')

        route = RouterAgent(settings()).route(query="who?", corpus="books")
        assert route is not None and route.destination == "vectorstore"

    def test_a_grader_cannot_keep_a_passage_that_was_not_retrieved(self, monkeypatch):
        _model(monkeypatch, '{"keep": ["c1", "invented"], "verdict": "correct"}')

        grade = RelevanceGrader(settings()).grade(query="why?", hits=_hits())
        assert grade is not None and grade.keep == ["c1"]

    def test_the_no_answer_sentinel_becomes_an_abstention(self, monkeypatch):
        _model(monkeypatch, "NO_ANSWER")
        assert SynthesisAgent(settings()).write(query="why?", hits=_hits()) is None


class TestReadyReadsTheRightConfiguration:
    def test_a_model_agent_follows_the_model_key(self):
        assert SynthesisAgent(settings()).ready is True
        assert SynthesisAgent(keyless()).ready is False

    def test_the_tool_agent_follows_composio_instead(self):
        """It runs no model of its own, so a missing model key says nothing
        about whether it can do anything."""
        agent = ToolAgent(
            SimpleNamespace(configured=True),
            SimpleNamespace(list=lambda user_id: []),
            keyless(),
        )
        assert agent.needs_model is False
        assert agent.ready is True

    def test_a_tool_agent_without_credentials_is_not_ready(self):
        agent = ToolAgent(
            SimpleNamespace(configured=False),
            SimpleNamespace(list=lambda user_id: []),
            settings(),
        )
        assert agent.ready is False


class TestTheHierarchyItself:
    def test_every_agent_inherits_the_base(self):
        for agent in (SynthesisAgent, RelevanceGrader, RouterAgent, ToolAgent):
            assert issubclass(agent, BaseAgent)

    def test_only_the_model_driven_ones_get_the_model_helpers(self):
        """`ToolAgent` is an agent by contract, not by making model calls."""
        assert issubclass(SynthesisAgent, ModelAgent)
        assert not issubclass(ToolAgent, ModelAgent)

    def test_names_are_unique_because_they_name_the_logger_and_the_prompt(self):
        from src.agents.dataset_agent import DatasetAgent
        from src.agents.rag import AnswerGrader, QueryRewriter

        agents = [
            SynthesisAgent, RelevanceGrader, QueryRewriter, AnswerGrader,
            RouterAgent, ToolAgent, DatasetAgent,
        ]
        names = [a.name for a in agents]
        assert len(set(names)) == len(names)

    def test_an_agent_that_does_not_name_itself_is_refused(self):
        """At class creation, not at the first log line it writes."""
        with pytest.raises(TypeError, match="must set `name`"):

            class Nameless(ModelAgent):
                pass

    def test_each_stage_runs_on_its_own_budget(self):
        """A grader's tokens are not a synthesiser's, and a call site that could
        pass its own would eventually pass a different one."""
        config = settings()
        assert SynthesisAgent(config)._max_tokens == config.synthesis_max_tokens
        assert RelevanceGrader(config)._max_tokens == config.grader_max_tokens
        assert SynthesisAgent(config)._timeout_s == config.ask_llm_timeout_s
        assert RouterAgent(config)._timeout_s == config.grader_timeout_s
