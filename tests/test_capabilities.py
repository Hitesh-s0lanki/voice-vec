"""The discovery flow: ask → find_capability → the tool it named → answer.

The shape this pins is the one in docs/23-capabilities.md, and each test is a
way it can silently stop being that shape:

  - the mailbox schemas arriving on round 1, which is the prompt this replaced;
  - discovery matching the *nearest* capability rather than a relevant one, so
    "check my inbox" queries somebody's student database;
  - a profile that has not been written yet taking the mailbox away with it,
    because the gate is on the description rather than on the tool;
  - the counts line growing back into a list of names, which is a menu a model
    will answer from.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.capabilities.catalogue import DATASET, STORE, TOOLKIT, Capability
from src.capabilities.index import CapabilityIndex
from src.core.config import get_settings
from src.tools.capabilities import FIND_TOOL, CapabilityTools
from src.tools.kit import ToolKit
from src.tools.result import ToolResult


def _capabilities() -> list[Capability]:
    return [
        Capability(
            id="gmail",
            kind=TOOLKIT,
            title="gmail",
            summary="Actions on this person's gmail account.",
            good_for=("gmail fetch emails", "gmail send email"),
            tool="find_capability",
            use="call the GMAIL_* tool the answer names",
        ),
        Capability(
            id="pgvector",
            kind=STORE,
            title="Student records",
            summary="Enrolment records and reports for a school.",
            good_for=("student enrolment", "attendance"),
            tool="search_store",
            use='search_store(store="pgvector", question="…")',
        ),
        Capability(
            id="marks",
            kind=DATASET,
            title="Term marks",
            summary="One row per student per subject, with marks.",
            tool="query_dataset",
            use='query_dataset(dataset="marks", question="…")',
        ),
    ]


class FakeCatalogue:
    def __init__(self, found):
        self._found = found

    def for_user(self, user_id):
        return list(self._found) if user_id else []


class FakeEmbedder:
    """No embedder in a unit test — the index falls back to word overlap."""

    ready = False

    def embed_query(self, text):  # pragma: no cover - never reached
        raise AssertionError("should not embed")

    def embed_passages(self, texts):  # pragma: no cover - never reached
        raise AssertionError("should not embed")


def _tools(found=None, **overrides) -> CapabilityTools:
    settings = get_settings().model_copy(update=overrides)
    index = CapabilityIndex(
        FakeCatalogue(_capabilities() if found is None else found), FakeEmbedder(), settings
    )
    return CapabilityTools(index, settings)


class TestDiscovery:
    def test_the_inbox_question_finds_the_mailbox(self):
        result = _tools().execute("u1", FIND_TOOL, {"need": "fetch emails from gmail"})
        found = result.data["found"]

        assert result.ok
        assert found[0]["id"] == "gmail"
        assert "GMAIL" in found[0]["use"]

    def test_a_question_about_records_finds_the_store_not_the_mailbox(self):
        result = _tools().execute("u1", FIND_TOOL, {"need": "student enrolment records"})

        ids = [f["id"] for f in result.data["found"]]
        assert ids[0] == "pgvector"
        assert "gmail" not in ids

    def test_nothing_relevant_is_a_successful_answer_with_advice(self):
        """Not a failure: "nothing you connected covers this" is a true thing
        to say out loud, and a failure would have the model retry it."""
        result = _tools().execute("u1", FIND_TOOL, {"need": "the weather in Oslo"})

        assert result.ok
        assert result.data["found"] == []
        assert "do not have it" in result.data["advice"]

    def test_a_user_with_nothing_connected_is_offered_no_discovery_tool(self):
        assert _tools(found=[]).tools_for("u1") == []

    def test_the_tool_is_off_when_the_feature_is(self):
        assert _tools(capabilities_enabled=False).tools_for("u1") == []


class TestTheThresholdIsLiftNotScore:
    """The bug: a store whose card said *finding book passages* did not match
    "summary of the book The Laws of Human Nature".

    It ranked first and was cut by an absolute floor. What separates a question
    a capability covers from one it does not is **lift** — how far the best
    card sits above the mean of this person's own cards — and not the raw
    cosine, which moves with the cards.

    The magnitudes below are `text-embedding-3`'s, which spreads cards much
    further apart than e5 did (docs/25-no-local-embedder.md). The *shape* is
    what is pinned and it is unchanged by that swap: the irrelevant question
    here scores **higher** in absolute terms than the relevant one and still
    matches nothing, which no absolute floor can express.
    """

    def _index(self, scores, **overrides):
        import numpy as np

        found = [
            Capability(id=f"c{i}", kind=STORE, title=f"c{i}", summary="x")
            for i in range(len(scores))
        ]

        class Embedder:
            """Vectors chosen so the cosines are exactly `scores`."""

            def embed_passages(self, texts):
                return np.array(
                    [[s, float(np.sqrt(max(0.0, 1 - s * s)))] for s in scores],
                    dtype="float32",
                )

            def embed_query(self, text):
                return np.array([1.0, 0.0], dtype="float32")

        settings = get_settings().model_copy(update=overrides)
        return CapabilityIndex(FakeCatalogue(found), Embedder(), settings)

    def test_the_book_question_matches_the_book_store(self):
        """0.30 against a 0.12 baseline — a lift of 0.18, clear of the floor."""
        found = self._index([0.30, 0.10, 0.05, 0.03]).search("u1", "the laws of human nature")

        assert [m.capability.id for m in found] == ["c0"]

    def test_small_talk_scoring_higher_still_matches_nothing(self):
        """0.35 — a *higher* top score than the book question's 0.30 — and a
        lift of 0.038, because every card scores about the same.

        This is the whole argument in one pair: any absolute floor that admits
        the test above also admits this one."""
        found = self._index([0.35, 0.32, 0.30, 0.28]).search("u1", "how are you feeling today")

        assert found == []

    def test_a_lone_capability_is_returned_rather_than_thresholded(self):
        """With one card the mean is the card and lift is always zero. The
        agent gets it, with its good_for, and decides."""
        found = self._index([0.74]).search("u1", "anything at all")

        assert [m.capability.id for m in found] == ["c0"]


class TestUnlocking:
    def _kit(self, capabilities, composio_tools=(), settings=None):
        settings = settings or get_settings()
        return ToolKit(
            capabilities=capabilities,
            stores=SimpleNamespace(
                owns=lambda n: n == "search_store",
                tools_for=lambda u: [{"function": {"name": "search_store"}}],
                execute=lambda *a: ToolResult("search_store", ok=True),
            ),
            datasets=SimpleNamespace(
                owns=lambda n: n == "query_dataset",
                tools_for=lambda u: [{"function": {"name": "query_dataset"}}],
                execute=lambda *a: ToolResult("query_dataset", ok=True),
            ),
            composio=SimpleNamespace(
                tools_for=lambda u, only=None: [
                    t for t in composio_tools if only is None or t["toolkit"] in only
                ],
                execute=lambda *a: ToolResult("GMAIL_FETCH_EMAILS", ok=True),
            ),
            settings=settings,
        )

    def _names(self, schemas):
        return [s.get("function", {}).get("name") or s.get("name") for s in schemas]

    def test_round_one_offers_discovery_and_nothing_else(self):
        """The prompt this replaced carried every schema for every linked
        toolkit, on every turn, buffered in front of the first spoken word."""
        kit = self._kit(_tools(), composio_tools=[{"toolkit": "gmail", "name": "GMAIL_FETCH"}])

        assert self._names(kit.schemas("u1")) == [FIND_TOOL]

    def test_discovering_the_mailbox_unlocks_its_tools_next_round(self):
        capabilities = _tools()
        kit = self._kit(
            capabilities,
            composio_tools=[
                {"toolkit": "gmail", "function": {"name": "GMAIL_FETCH"}},
                {"toolkit": "slack", "function": {"name": "SLACK_POST"}},
            ],
        )
        kit.execute("u1", FIND_TOOL, {"need": "fetch emails from gmail"})

        names = self._names(kit.schemas("u1"))
        assert "GMAIL_FETCH" in names
        # Only what was asked about: the rest of the account stays out of the
        # prompt, which is the whole saving.
        assert "SLACK_POST" not in names

    def test_discovering_a_store_unlocks_search_store(self):
        kit = self._kit(_tools())
        kit.execute("u1", FIND_TOOL, {"need": "student enrolment records"})

        assert "search_store" in self._names(kit.schemas("u1"))

    def test_discovering_a_dataset_unlocks_query_dataset(self):
        kit = self._kit(_tools())
        kit.execute("u1", FIND_TOOL, {"need": "term marks per subject"})

        assert "query_dataset" in self._names(kit.schemas("u1"))

    def test_no_discovery_means_no_gate(self):
        """A store connected thirty seconds ago has no profile yet. Gating on a
        description that does not exist would take away the mailbox that
        does."""
        kit = self._kit(
            _tools(found=[]),
            composio_tools=[{"toolkit": "gmail", "function": {"name": "GMAIL_FETCH"}}],
        )

        names = self._names(kit.schemas("u1"))
        assert "GMAIL_FETCH" in names and "query_dataset" in names
        assert FIND_TOOL not in names


class TestAStoreBuiltByAnotherModel:
    """Width is not identity, and assuming it was is a silent failure.

    A 768-dim index built by somebody else's model accepts a 768-dim query
    vector and returns its nearest neighbours; every one of them is noise.
    Measured on a real connected store: the cosine between a record's own
    vector and this app's embedding of that record's own text was 0.003, top
    retrieval similarity was 0.086 where a real match is ~0.85, and the ladder
    abstained on every question while the store read as connected and
    searchable.
    """

    def _profile(self, match):
        from src.connectors.profile import (
            CapabilityFacts,
            Observation,
            Profile,
            VectorShape,
            render_card,
        )

        observation = Observation(
            connector="pgvector",
            kind="vector",
            location="pgvector/host#book_chunks",
            reachable=True,
            sampled=200,
            vectors=VectorShape(dimensions=384, metric="cosine", records=2366),
            embedding_match=match,
        )
        facts = CapabilityFacts.derive(observation, embed_dim=384)
        return (
            Profile(
                connector="pgvector", kind="vector", status="ok",
                observation=observation, facts=facts,
            ),
            facts,
            render_card,
        )

    def test_a_mismatch_is_measured_and_takes_searchable_away(self):
        _, facts, _ = self._profile(0.0032)

        assert facts.compatible is False
        assert facts.searchable is False

    def test_a_match_leaves_it_searchable(self):
        _, facts, _ = self._profile(0.97)

        assert facts.compatible is True
        assert facts.searchable is True

    def test_untested_takes_nothing_away(self):
        """Absence of a measurement is not evidence of absence — the same rule
        `ProfiledBackend.merge` applies to every other capability."""
        _, facts, _ = self._profile(None)

        assert facts.compatible is None
        assert facts.searchable is True

    def test_the_card_says_what_is_wrong_and_it_is_actionable(self):
        profile, _, render_card = self._profile(0.0032)

        card = render_card(profile)
        assert "different embedding model" in card

    def test_discovery_still_finds_it_and_says_why_it_cannot_be_used(self):
        """Dropping it silently is how somebody who connected a book store gets
        told, with total confidence, that they have nothing about books."""
        blocked = Capability(
            id="pgvector", kind=STORE, title="Book Chunks",
            summary="Book passage chunks.", good_for=("book passages",),
            blocked="connected, but its vectors were built by a different embedding model",
        )
        result = _tools(found=[blocked]).execute("u1", FIND_TOOL, {"need": "book passages"})
        found = result.data["found"][0]

        assert result.ok
        assert "unavailable" in found and "use" not in found
        assert "different embedding model" in found["unavailable"]

    def test_a_blocked_store_is_not_offered_to_search_store(self):
        """Discovery names it so the agent can explain it; the search tool must
        not accept it, because that search cannot come back with anything."""
        from src.tools.store import StoreTools

        blocked = Capability(id="pgvector", kind=STORE, title="Book Chunks", blocked="…")
        usable = Capability(id="pinecone", kind=STORE, title="Snippets")
        index = _tools(found=[blocked, usable])._index

        assert StoreTools(index, get_settings()).stores("u1") == ["pinecone"]


class TestTheCountsLine:
    def test_it_counts_and_does_not_name(self):
        """A name is a hint, and a model given hints routes on them instead of
        looking."""
        from src.services.voice_service import _reach

        line = _reach(_capabilities())

        assert line == "1 connected store, 1 dataset, 1 connected account."
        for name in ("gmail", "pgvector", "marks", "Student records"):
            assert name not in line

    def test_the_prompt_tells_the_model_to_go_and_look(self):
        from src.voice.llm import system_prompt

        prompt = system_prompt("en", stores="1 dataset.", discovery=True)

        assert "find_capability" in prompt
        assert "never answer from these counts" in prompt.lower()

    def test_cards_carry_no_discovery_instruction(self):
        """The fallback shape is the old one: descriptions, and nothing to
        follow them up with."""
        from src.voice.llm import system_prompt

        prompt = system_prompt("en", stores="Some card text.", discovery=False)

        assert "find_capability" not in prompt
