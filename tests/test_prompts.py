"""The prompt files, and the three ways one can be wrong without looking wrong.

A prompt lives in `src/prompts/<name>.md` now, which is better for reading and
editing and worse in exactly one way: nothing type-checks a markdown file. So
the failures it can have are pinned here.

  - **a variable written with two braces** substitutes HTML-escaped text, so a
    passage containing `<` or `&` reaches the model altered. Nothing raises,
    nothing logs, the grade is just slightly worse;
  - **a sentinel that drifts from the constant** — `NO_ANSWER` is compared in
    Python and instructed in markdown, and if the two stop matching the
    synthesiser's refusals silently become answers;
  - **a file that does not exist** for an agent that expects one, which must
    fail at startup rather than at somebody's turn.
"""

from __future__ import annotations

import pytest

from src.agents import prompts
from src.agents.dataset_agent import DatasetAgent
from src.agents.rag import NO_ANSWER, AnswerGrader, QueryRewriter, RelevanceGrader, RouterAgent, SynthesisAgent

STAGES = [SynthesisAgent, RelevanceGrader, QueryRewriter, AnswerGrader, RouterAgent]


class TestEveryAgentHasOne:
    @pytest.mark.parametrize("agent", [*STAGES, DatasetAgent])
    def test_the_file_exists_and_has_a_system_section(self, agent):
        assert prompts.load(agent.name).system_template.strip()

    @pytest.mark.parametrize("agent", STAGES)
    def test_a_single_shot_stage_also_has_a_human_section(self, agent):
        """It is where the question goes. Without it the model is handed rules
        and no material."""
        assert prompts.load(agent.name).chat.input_variables

    def test_the_tool_loop_agent_needs_no_human_section(self):
        """`DatasetAgent` supplies its own user turn — the question, straight
        into the graph."""
        prompt = prompts.load(DatasetAgent.name)
        assert not prompt.human_template
        assert "{{{schema}}}" not in prompt.system(schema="hinval(id BIGINT)")


class TestTheFormat:
    def test_notes_above_the_first_heading_are_not_sent(self):
        """The `#` title and the reasoning under it are for whoever edits the
        file. A prompt directory nobody can annotate ends up annotated in the
        prompt."""
        text = (prompts.PROMPTS / "router.md").read_text()
        assert "adaptive RAG" in text
        assert "adaptive RAG" not in prompts.load("router").system_template

    def test_two_braced_variables_are_refused(self, tmp_path, monkeypatch):
        """They HTML-escape the value, which changes the text the model grades
        and says nothing about it."""
        monkeypatch.setattr(prompts, "PROMPTS", tmp_path)
        prompts.load.cache_clear()
        (tmp_path / "bad.md").write_text("## System\n\nGrade {{query}} please\n")

        with pytest.raises(prompts.PromptError, match="three"):
            prompts.load("bad")
        prompts.load.cache_clear()

    def test_json_literals_are_not_mistaken_for_variables(self, tmp_path, monkeypatch):
        """Three of these files instruct a bare JSON object. Mustache is used
        precisely so those need no escaping."""
        monkeypatch.setattr(prompts, "PROMPTS", tmp_path)
        prompts.load.cache_clear()
        (tmp_path / "ok.md").write_text('## System\n\nReturn {"supported": true}\n')

        assert '{"supported": true}' in prompts.load("ok").system_template
        prompts.load.cache_clear()

    def test_a_missing_file_raises_rather_than_degrading(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompts, "PROMPTS", tmp_path)
        prompts.load.cache_clear()

        with pytest.raises(prompts.PromptError):
            prompts.load("nothing-here")
        prompts.load.cache_clear()


class TestTheSentinelDoesNotDrift:
    def test_the_synthesiser_is_told_the_constant_the_code_compares(self):
        assert NO_ANSWER in prompts.load(SynthesisAgent.name).system_template

    def test_the_context_and_question_both_reach_the_synthesiser(self):
        rendered = prompts.load(SynthesisAgent.name).chat.format_messages(
            context="[c1] a & b", query="why <this>?"
        )
        human = rendered[1].content
        # Raw, not escaped — the whole reason for triple braces.
        assert "[c1] a & b" in human and "why <this>?" in human
