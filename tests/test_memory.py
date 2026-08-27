"""Unit tests for the agent's memory — the parts that fail silently.

The service is not tested here. A mock of Redis Agent Memory only ever proves
the mock works, and the interesting half of that product runs in a background
worker we do not call. What *is* tested is everything this repository decides on
its own, which is all of the behaviour that a 30 MB instance and a live
microphone depend on:

  * that an unconfigured store is inert rather than broken;
  * that nothing reaching the wire is bigger than the budget allows;
  * that a recall failure produces no memories rather than no answer;
  * that a prompt with nothing to remember says nothing.
"""

import asyncio

import pytest

from src.core.config import Settings
from src.memory.store import MemoryStore, Recollection, as_prompt
from src.voice import llm


def configured(**overrides) -> Settings:
    """A Settings that looks like a wired-up deployment, without being one."""
    return Settings(
        _env_file=None,
        agent_memory_endpoint="https://memory.example.invalid",
        agent_memory_store_id="store-test",
        agent_memory_api_key="key-test",
        **overrides,
    )


class TestAvailability:
    def test_unset_is_not_configured(self):
        store = MemoryStore(Settings(_env_file=None))
        assert store.configured is False
        assert store.describe() == "unset"

    def test_all_three_values_are_required(self):
        settings = Settings(
            _env_file=None,
            agent_memory_endpoint="https://memory.example.invalid",
            agent_memory_store_id="store-test",
        )
        assert MemoryStore(settings).configured is False

    def test_configured_when_all_three_are_set(self):
        store = MemoryStore(configured())
        assert store.configured is True
        assert store.describe() == "on"

    def test_the_off_switch_beats_a_full_configuration(self):
        store = MemoryStore(configured(agent_memory_enabled=False))
        assert store.configured is False
        assert store.describe() == "off"


class TestWritesAreInert:
    """A write with nothing to write must not reach the client at all.

    Each of these would otherwise cost a round trip and an entry in a database
    the whole product is sharing.
    """

    def test_an_unconfigured_store_never_builds_a_client(self):
        store = MemoryStore(Settings(_env_file=None))
        store.remember(session_id="conv_1", actor_id="u1", role="user", text="hello")
        assert store._client is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"role": "tool"},          # not a conversation event
            {"text": "   "},           # a take that caught a cough
            {"session_id": ""},        # no conversation was ever opened
            {"actor_id": ""},          # a caller with no identity
        ],
    )
    def test_nothing_worth_storing_is_stored(self, kwargs):
        store = MemoryStore(configured())
        base = dict(session_id="conv_1", actor_id="u1", role="user", text="hello")
        store.remember(**{**base, **kwargs})
        assert store._client is None


class TestTheByteBudget:
    def test_a_long_turn_is_trimmed_before_it_reaches_the_wire(self):
        """The instance is what this protects, so it is checked at the call.

        Trimming after the request has been built would be a different and
        useless guarantee: the cost being avoided is what the service *stores*
        for the life of the session, not what the request weighs.
        """
        sent: dict = {}

        class Spy:
            def add_session_event(self, **kwargs):
                sent.update(kwargs)

        store = MemoryStore(configured(agent_memory_max_chars=50))
        store._client = Spy()
        store.remember(session_id="conv_1", actor_id="u1", role="user", text="ब" * 5_000)

        text = sent["content"][0].text
        assert len(text) <= 51          # the trim, plus the ellipsis that marks it
        assert text.endswith("…")

    def test_a_short_turn_is_sent_whole(self):
        sent: dict = {}

        class Spy:
            def add_session_event(self, **kwargs):
                sent.update(kwargs)

        store = MemoryStore(configured())
        store._client = Spy()
        store.remember(session_id="conv_1", actor_id="u1", role="assistant", text="  ठीक है।  ")

        assert sent["content"][0].text == "ठीक है।"
        assert sent["role"] == "ASSISTANT"


class TestRecallNeverBreaksATurn:
    def test_a_dead_service_returns_no_memories_not_an_error(self):
        class Dead:
            async def search_long_term_memory_async(self, **kwargs):
                raise RuntimeError("connection refused")

        store = MemoryStore(configured())
        store._client = Dead()
        assert asyncio.run(store.recall(query="what do I eat", owner_id="u1")) == []

    def test_an_unconfigured_store_asks_nothing(self):
        store = MemoryStore(Settings(_env_file=None))
        assert asyncio.run(store.recall(query="anything", owner_id="u1")) == []

    def test_a_caller_with_no_identity_asks_nothing(self):
        store = MemoryStore(configured())
        assert asyncio.run(store.recall(query="anything", owner_id="")) == []

    def test_the_owner_and_the_floor_are_both_on_the_request(self):
        """Scope and threshold are the two things that make recall safe.

        Losing either is silent: an unscoped search returns another person's
        memories, and an unfloored one always returns *something*.
        """
        seen: dict = {}

        class Spy:
            async def search_long_term_memory_async(self, *, request):
                seen.update(request)
                return type("R", (), {"items": []})()

        store = MemoryStore(configured())
        store._client = Spy()
        asyncio.run(store.recall(query="food", owner_id="user_42"))

        assert seen["filter_"] == {"owner_id": {"eq": "user_42"}}
        assert seen["similarity_threshold"] == 0.62
        assert seen["limit"] == 3

    def test_blank_memories_are_dropped(self):
        class Spy:
            async def search_long_term_memory_async(self, **kwargs):
                items = [
                    type("M", (), {"text": "User is vegetarian", "memory_type": "semantic"})(),
                    type("M", (), {"text": "   ", "memory_type": "semantic"})(),
                ]
                return type("R", (), {"items": items})()

        store = MemoryStore(configured())
        store._client = Spy()
        found = asyncio.run(store.recall(query="food", owner_id="u1"))
        assert found == [Recollection("User is vegetarian", "semantic")]


class TestThePrompt:
    def test_nothing_remembered_adds_no_section(self):
        assert as_prompt([]) is None
        assert as_prompt([Recollection("   ")]) is None

    def test_a_prompt_with_no_memories_is_the_prompt_it_always_was(self):
        assert llm.system_prompt("ta-IN") == llm.system_prompt("ta-IN", None, None)

    def test_recalled_facts_carry_their_own_instructions(self):
        """The facts are the smaller half of what this section adds.

        A model handed bare facts recites them, which out loud is the whole
        difference between an assistant that remembers and one that listens —
        so the framing is asserted here rather than left to review.
        """
        prompt = llm.system_prompt("ta-IN", None, as_prompt([Recollection("User is vegetarian")]))
        assert "User is vegetarian" in prompt
        assert "Never recite it" in prompt
        assert "out of date" in prompt

    def test_memories_come_before_the_sources(self):
        prompt = llm.system_prompt("ta-IN", "a passage", as_prompt([Recollection("A fact")]))
        assert prompt.index("A fact") < prompt.index("a passage")

    def test_build_messages_passes_them_through(self):
        messages = llm.build_messages(
            transcript="what should I eat",
            history=[],
            language_code="ta-IN",
            memories="- User is vegetarian",
        )
        assert "User is vegetarian" in messages[0]["content"]
        assert messages[-1]["content"] == "what should I eat"
