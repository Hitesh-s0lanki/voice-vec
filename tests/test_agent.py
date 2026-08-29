"""Unit tests for the agent's tool calling — the parts that fail silently.

Every case here is one where the turn still *sounds* fine while being wrong,
which is what makes them worth pinning:

  - results discarded after the tools ran, so the agent acts on somebody's
    mailbox and then answers from a prompt that never mentions what came back;
  - the tool pass running for a user who has linked nothing, which buys a
    buffered round trip in front of the first spoken word for no reason;
  - a failed tool swallowed instead of reported, so the model invents an answer
    from a silence;
  - arguments stored unbounded, or a result stored whole rather than as the
    bounded preview the thread reads back.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import src.voice.llm as llm
from src.chat.store import Owner
from src.chat.tool_calls import (
    MAX_ARGUMENT_CHARS,
    MAX_RESULT_CHARS as STORED_RESULT_CHARS,
    toolkit_of,
    trim,
    trim_result,
)
from src.tools.result import MAX_RESULT_CHARS, ToolResult
from src.services.voice_service import VoiceSession

TOOLS = [{"type": "function", "function": {"name": "GMAIL_FETCH", "parameters": {}}}]
BASE: list[dict] = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "any mail?"},
]


async def _emit(event):
    pass


async def _send(chunk):
    pass


def _turn():
    return SimpleNamespace(id="t1", tools=0, mark=lambda: 1.0, language_code="en")


def _session(tools, execute):
    session = VoiceSession(emit=_emit, send_audio=_send, owner=Owner(user_id="u1"))
    session.agent = SimpleNamespace(tools_for=lambda user_id: tools, execute=execute)
    session.tool_calls = SimpleNamespace(configured=False)
    return session


def _scripted(monkeypatch, *completions):
    """Answer each `llm.complete` with the next scripted completion."""
    calls = {"n": 0}

    async def fake(messages, *, settings, tools=None, target=None):
        index = calls["n"]
        calls["n"] += 1
        result = completions[min(index, len(completions) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm, "complete", fake)
    return calls


def _run(session, turn=None):
    return asyncio.run(session._use_tools(turn or _turn(), BASE))


class TestToolResultsReachTheAnswer:
    """The bug this suite exists for.

    Running a tool and then generating the spoken reply from the *original*
    messages means the agent sent the email and then answered as though it had
    not. Nothing about the turn sounds broken.
    """

    def test_results_are_carried_into_the_spoken_pass(self, monkeypatch):
        _scripted(
            monkeypatch,
            llm.Completion("", (llm.ToolCall("c1", "GMAIL_FETCH", {"n": 3}),)),
            llm.Completion("You have three."),
        )
        out = _run(_session(TOOLS, lambda u, s, a: ToolResult(s, ok=True, data={"unread": 3})))

        assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
        assert any(m["role"] == "tool" for m in out)

    def test_the_assistant_turn_carries_the_calls_verbatim(self, monkeypatch):
        """Without it the `tool` replies have nothing to attach to."""
        _scripted(
            monkeypatch,
            llm.Completion("", (llm.ToolCall("c1", "GMAIL_FETCH", {"n": 3}),)),
            llm.Completion("Done."),
        )
        out = _run(_session(TOOLS, lambda u, s, a: ToolResult(s, ok=True, data={})))

        assistant = next(m for m in out if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "c1"
        assert assistant["tool_calls"][0]["function"]["name"] == "GMAIL_FETCH"

        tool = next(m for m in out if m["role"] == "tool")
        assert tool["tool_call_id"] == "c1"

    def test_a_model_failure_mid_loop_keeps_what_already_ran(self, monkeypatch):
        _scripted(
            monkeypatch,
            llm.Completion("", (llm.ToolCall("c1", "GMAIL_FETCH", {}),)),
            RuntimeError("provider died"),
        )
        out = _run(_session(TOOLS, lambda u, s, a: ToolResult(s, ok=True, data={"unread": 1})))

        assert any(m["role"] == "tool" for m in out)


class TestNothingLinkedCostsNothing:
    """The latency promise. A buffered pass sits in front of the first spoken
    word, so a user who has linked nothing must not pay for one."""

    def test_no_model_call_at_all(self, monkeypatch):
        calls = _scripted(monkeypatch, llm.Completion("x"))
        out = _run(_session([], lambda u, s, a: None))

        assert calls["n"] == 0
        assert out == BASE

    def test_an_anonymous_session_never_calls_tools(self, monkeypatch):
        calls = _scripted(monkeypatch, llm.Completion("x"))
        session = VoiceSession(emit=_emit, send_audio=_send, owner=Owner(session_id="sess_x"))
        session.agent = SimpleNamespace(tools_for=lambda user_id: TOOLS, execute=None)

        assert asyncio.run(session._use_tools(_turn(), BASE)) == BASE
        assert calls["n"] == 0

    def test_the_switch_turns_it_off_without_disconnecting_anyone(self, monkeypatch):
        calls = _scripted(monkeypatch, llm.Completion("x"))
        session = _session(TOOLS, lambda u, s, a: None)
        session.settings = session.settings.model_copy(update={"tools_enabled": False})

        assert asyncio.run(session._use_tools(_turn(), BASE)) == BASE
        assert calls["n"] == 0

    def test_the_model_answering_without_tools_leaves_messages_alone(self, monkeypatch):
        """Appending that assistant turn would have the spoken pass reply to itself."""
        _scripted(monkeypatch, llm.Completion("Sure."))
        assert _run(_session(TOOLS, lambda u, s, a: None)) == BASE


class TestFailuresAreReported:
    def test_a_failing_tool_is_told_to_the_model(self, monkeypatch):
        """So it can say so out loud, rather than inventing from a silence."""
        _scripted(
            monkeypatch,
            llm.Completion("", (llm.ToolCall("c1", "GMAIL_FETCH", {}),)),
            llm.Completion("I couldn't reach it."),
        )
        out = _run(_session(TOOLS, lambda u, s, a: ToolResult(s, ok=False, error="401")))

        tool = next(m for m in out if m["role"] == "tool")
        assert "401" in tool["content"]
        assert "error" in tool["content"]

    def test_a_tool_that_raises_becomes_a_result_not_an_exception(self):
        """The turn is mid-sentence; it has to continue either way."""
        from src.agents.tool_agent import ToolAgent

        agent = ToolAgent(
            SimpleNamespace(
                for_user=lambda user_id: SimpleNamespace(
                    tools=SimpleNamespace(
                        execute=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
                    )
                )
            ),
            SimpleNamespace(list=lambda user_id: []),
            SimpleNamespace(tool_schema_ttl_s=300, tool_schema_limit=40),
        )
        result = agent.execute("u1", "GMAIL_FETCH", {})

        assert not result.ok
        assert result.error == "RuntimeError"

    def test_composio_reporting_failure_in_the_body_is_not_success(self):
        """A 200 with successful=False is a failed tool, not a working one."""
        from src.agents.tool_agent import ToolAgent

        agent = ToolAgent(
            SimpleNamespace(
                for_user=lambda user_id: SimpleNamespace(
                    tools=SimpleNamespace(
                        execute=lambda *a, **k: SimpleNamespace(
                            successful=False, data=None, error="no such mailbox"
                        )
                    )
                )
            ),
            SimpleNamespace(list=lambda user_id: []),
            SimpleNamespace(tool_schema_ttl_s=300, tool_schema_limit=40),
        )
        result = agent.execute("u1", "GMAIL_FETCH", {})

        assert not result.ok
        assert "no such mailbox" in (result.error or "")

    def test_an_oversized_result_is_truncated_before_it_reenters_the_prompt(self):
        result = ToolResult("X", ok=True, data={"body": "x" * (MAX_RESULT_CHARS * 2)})
        rendered = result.for_model()

        assert len(rendered) <= MAX_RESULT_CHARS + 32
        assert rendered.endswith("(truncated)")


class TestWhatIsWrittenDown:
    def test_the_toolkit_is_derived_from_the_slug(self):
        assert toolkit_of("GMAIL_SEND_EMAIL") == "gmail"
        assert toolkit_of("SLACK_POST_MESSAGE") == "slack"

    @pytest.mark.parametrize("value", ["", None, "NOUNDERSCORE"])
    def test_a_slug_with_no_prefix_does_not_explode(self, value):
        assert isinstance(toolkit_of(value), str)

    def test_small_arguments_are_stored_as_given(self):
        assert trim({"to": "a@b.c"}) == {"to": "a@b.c"}

    def test_an_oversized_argument_is_replaced_not_dropped(self):
        """Knowing GMAIL_SEND_EMAIL ran is most of the record's value."""
        trimmed = trim({"body": "x" * (MAX_ARGUMENT_CHARS + 10), "to": "a@b.c"})

        assert trimmed["to"] == "a@b.c"
        assert "chars" in trimmed["body"]
        assert "xxxx" not in trimmed["body"]

    def test_unserialisable_arguments_do_not_break_the_write(self):
        assert trim({"f": object()}) != {}

    def test_a_small_result_is_stored_as_it_came_back(self):
        assert trim_result('{"unread": 3}') == '{"unread": 3}'

    def test_an_oversized_result_is_cut_and_marked(self):
        """The ceiling is the containment — a preview, not the inbox page."""
        stored = trim_result("x" * (STORED_RESULT_CHARS * 3))

        assert stored is not None
        assert len(stored) <= STORED_RESULT_CHARS + 32
        assert stored.endswith("(truncated)")

    @pytest.mark.parametrize("value", [None, "", "   \n "])
    def test_nothing_coming_back_stores_nothing(self, value):
        """An empty box in the thread says less than the status already does."""
        assert trim_result(value) is None

    def test_the_wire_carries_the_preview_and_the_whole_size(self):
        """Both, or a truncated result reads as the entire result."""
        from src.schemas.chat import ToolCall

        assert "result" in ToolCall.model_fields
        assert "result_bytes" in ToolCall.model_fields
