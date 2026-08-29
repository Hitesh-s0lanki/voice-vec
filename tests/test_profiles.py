"""Unit tests for connector profiling — the layer whose failures are all silent.

Every claim this package makes is one nothing else checks. A store that reports
a filter it cannot honour still answers; it just answers from an unnarrowed
search and calls it narrowed. A profile served for credentials that have since
been rotated describes an index nobody can reach, confidently. A card that
survives a JSON round trip with a field quietly dropped tells an agent it may
not filter on something it may.

So what is tested here is the decisions this app makes on its own:

  - **coverage, not presence**, because they are indistinguishable to anything
    that only records whether a key was ever seen, and the difference is
    whether a filter drops 0% or 97% of the store;
  - **the merge direction**, because a measurement that could *add* a
    capability would let a sampling artefact turn on a channel that is not
    there;
  - **fingerprint invalidation**, because a stale profile is the one failure in
    this layer with no visible symptom;
  - **the card**, because it goes into a system prompt verbatim and an agent
    cannot sanity-check a sentence about a store it cannot see.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.connectors.narrate import _list, _text, _tools
from src.connectors.probes.base import (
    dominant_facet,
    excerpts_of,
    field_stats,
    length_stats,
    pick_text_field,
    scripts_of,
    unreachable,
)
from src.connectors.probes.pgvector import _scrub, _vector_column
from src.connectors.profile import (
    EXCERPT_CHARS,
    FILTERABLE_COVERAGE,
    ToolShape,
    STATUS_FAILED,
    CapabilityFacts,
    FieldStat,
    Observation,
    Profile,
    Understanding,
    VectorShape,
)
from src.connectors.registry import (
    _name_candidates,
    discover_columns,
    _resolve_table,
    metric_of,
    reconcile_dimension,
)
from src.rag.remote_embed import model_for
from src.connectors.spec import ConnectorError
from src.connectors.profile_service import _usable
from src.connectors.profile_store import ProfileRow, fingerprint
from src.rag.backends.profiled import ProfiledBackend, merge
from src.rag.columns import ColumnMap
from src.rag.store import Capabilities


def records(n: int, **fields) -> list[dict]:
    """`n` records, each carrying every field given as `name=share`.

    `share` is the fraction of records that should have it, so a test can say
    "strategy on 2% of the index" without hand-writing 200 dicts.
    """
    out = []
    for index in range(n):
        record = {"text": f"passage number {index} with enough words to count"}
        for name, share in fields.items():
            if index < round(n * share):
                record[name] = f"{name}-{index % 3}"
        out.append(record)
    return out


class TestCoverage:
    """The measurement the whole package exists to make.

    `filters=True` has been hard-coded in every backend since connectors
    shipped, on the hope that a connected index carries a `strategy` field.
    These are the cases that separate the hope from the fact.
    """

    def test_a_field_on_every_record_is_filterable(self):
        stats = field_stats(records(200, strategy=1.0))
        strategy = next(s for s in stats if s.name == "strategy")

        assert strategy.coverage == pytest.approx(1.0)
        assert strategy.filterable

    def test_a_field_on_two_percent_of_records_is_not(self):
        """The failure this catches: a predicate that drops 98% of the store.

        Presence-based detection calls this a filter, the search returns four
        results out of a possible two hundred, and nothing reports an error —
        the ladder records a successful filtered retrieval.
        """
        stats = field_stats(records(200, strategy=0.02))
        strategy = next(s for s in stats if s.name == "strategy")

        assert strategy.coverage == pytest.approx(0.02)
        assert not strategy.filterable

    def test_the_threshold_is_inclusive(self):
        stats = field_stats(records(100, strategy=FILTERABLE_COVERAGE))
        assert next(s for s in stats if s.name == "strategy").filterable

    def test_null_and_empty_do_not_count_as_carried(self):
        """A key present with no value cannot narrow anything.

        Filtering on a column that is null on half the rows drops half the
        store just as surely as filtering on one that is absent.
        """
        rows = [{"text": "x" * 100, "language": None} for _ in range(50)]
        rows += [{"text": "y" * 100, "language": ""} for _ in range(50)]

        assert not any(s.name == "language" for s in field_stats(rows))

    def test_one_value_everywhere_is_constant(self):
        """`book_chunks.page` was 1 on all 2,366 rows — NOT NULL and useless."""
        rows = [{"text": "x" * 100, "page": 1} for _ in range(200)]
        page = next(s for s in field_stats(rows) if s.name == "page")

        assert page.filterable  # it is on every record…
        assert page.constant  # …and cannot narrow a thing

    def test_the_vector_is_not_reported_as_a_field(self):
        rows = [{"text": "x" * 100, "$vector": [0.1] * 768} for _ in range(10)]
        assert not any(s.name == "$vector" for s in field_stats(rows))

    def test_free_text_fields_are_never_quoted_back(self):
        """Values are counted well past the point they stop being quoted.

        The count is what tells a facet from a primary key; the quoting cap is
        what keeps a sample of the user's data out of the card. They are
        different limits because they answer different questions.
        """
        rows = [{"text": "x" * 100, "note": f"unique note {i}"} for i in range(200)]
        note = next(s for s in field_stats(rows) if s.name == "note")

        assert note.distinct == 200
        assert note.examples == ()

    def test_a_field_unique_per_record_is_an_identifier_not_a_facet(self):
        """`id` is on 100% of records, never null, and useless to filter on.

        Every other test for a good filter passes it. Only cardinality against
        the record count says it identifies rows rather than grouping them.
        """
        rows = [{"text": "x" * 100, "id": i, "book_id": f"b{i % 12}"} for i in range(200)]
        stats = {s.name: s for s in field_stats(rows)}

        assert stats["id"].filterable and stats["id"].unique
        assert stats["book_id"].filterable and not stats["book_id"].unique

    def test_a_single_record_is_not_evidence_of_uniqueness(self):
        """With one record every field is trivially unique and nothing is known."""
        stats = {s.name: s for s in field_stats([{"text": "x" * 100, "tag": "a"}])}
        assert not stats["tag"].unique


class TestDerivedCapabilities:
    """What the measurement implies, which is the only part retrieval acts on."""

    def _observed(self, fields, *, dimensions=384, count=100) -> Observation:
        return Observation(
            connector="pinecone",
            kind="vector",
            reachable=True,
            sampled=200,
            vectors=VectorShape(dimensions=dimensions, records=count),
            fields=tuple(
                FieldStat(name=name, coverage=coverage) for name, coverage in fields.items()
            ),
        )

    def test_filters_follow_the_strategy_field(self):
        assert CapabilityFacts.derive(
            self._observed({"strategy": 1.0}), embed_dim=384
        ).filters
        assert not CapabilityFacts.derive(
            self._observed({"strategy": 0.02}), embed_dim=384
        ).filters
        assert not CapabilityFacts.derive(self._observed({}), embed_dim=384).filters

    def test_a_dimension_mismatch_is_not_searchable(self):
        """768-dim vectors and a 384-dim embedder cannot be compared at all.

        This is the connected-store failure with the worst symptom: every
        search raises, per question, forever, and the connector panel stays
        green because the credential is fine.
        """
        facts = CapabilityFacts.derive(
            self._observed({"strategy": 1.0}, dimensions=768), embed_dim=384
        )
        assert not facts.searchable

    def test_an_empty_index_is_not_searchable(self):
        assert not CapabilityFacts.derive(
            self._observed({}, count=0), embed_dim=384
        ).searchable

    def test_an_uncountable_store_is_not_assumed_empty(self):
        """None and 0 are different answers and only one of them means empty."""
        observation = self._observed({"strategy": 1.0})
        observation = Observation(
            connector="pinecone",
            kind="vector",
            reachable=True,
            vectors=VectorShape(dimensions=384, records=None),
            fields=observation.fields,
        )
        assert CapabilityFacts.derive(observation, embed_dim=384).searchable

    def test_an_unreachable_store_is_not_searchable(self):
        facts = CapabilityFacts.derive(
            unreachable("astra", "vector", "astra/x.y", "gone"), embed_dim=384
        )
        assert not facts.searchable
        assert not facts.filters

    def test_defaults_are_the_pessimistic_answer(self):
        """Claiming a channel that is absent fails at query time; the reverse
        only costs the recall it would have added."""
        facts = CapabilityFacts()
        assert not (facts.lexical or facts.filters or facts.parallel_text or facts.searchable)


class TestMerge:
    """A measurement may remove a claim and may never add one."""

    DECLARED = Capabilities(lexical=True, filters=True, parallel_text=True)

    def test_no_measurement_leaves_the_backend_alone(self):
        """Turning profiling off must not silently downgrade every store."""
        assert merge(self.DECLARED, None) == self.DECLARED

    def test_the_sample_can_withdraw_a_claim(self):
        merged = merge(self.DECLARED, CapabilityFacts(lexical=False, filters=True, parallel_text=True))
        assert not merged.lexical
        assert merged.filters

    def test_the_sample_cannot_invent_one(self):
        """A backend that knows its protocol has no lexical channel is right.

        Pinecone cannot do keyword search on a dense index whatever a sampled
        `tsv`-shaped field suggests, so a measurement must not switch it on.
        """
        declared = Capabilities(lexical=False, filters=False, parallel_text=False)
        merged = merge(declared, CapabilityFacts(lexical=True, filters=True, parallel_text=True))

        assert not (merged.lexical or merged.filters or merged.parallel_text)


class FakeBackend:
    name = "pinecone"

    def __init__(self) -> None:
        self.closed = False
        self.searched: list[tuple] = []

    def describe(self) -> str:
        return "pinecone/idx"

    def ready(self) -> bool:
        return True

    def capabilities(self) -> Capabilities:
        return Capabilities(lexical=False, filters=True, parallel_text=False)

    def search(self, vector, *, strategies, limit, language=None):
        self.searched.append((tuple(strategies), limit, language))
        return []

    def close(self) -> None:
        self.closed = True


class TestProfiledBackend:
    def test_it_narrows_the_backends_own_answer(self):
        wrapped = ProfiledBackend(FakeBackend(), CapabilityFacts(filters=False))
        assert not wrapped.capabilities().filters

    def test_it_forwards_everything_else(self):
        inner = FakeBackend()
        wrapped = ProfiledBackend(inner, None)

        wrapped.search([0.0], strategies=["S1"], limit=5, language="hin_Deva")
        wrapped.close()

        assert inner.searched == [(("S1",), 5, "hin_Deva")]
        assert inner.closed
        assert wrapped.name == "pinecone" and wrapped.describe() == "pinecone/idx"

    def test_it_still_satisfies_the_protocol(self):
        """`VectorBackend` is runtime_checkable, so a wrapper that forwards by
        `__getattr__` would pass isinstance and fail any method added later."""
        from src.rag.backends.base import VectorBackend

        assert isinstance(ProfiledBackend(FakeBackend(), None), VectorBackend)


class TestTextDetection:
    def test_the_conventional_key_wins(self):
        assert pick_text_field([{"text": "a", "body": "b" * 500}]) == "text"

    def test_it_falls_back_to_the_longest_string(self):
        """A store nobody built for this app names its text whatever it likes."""
        rows = [{"id": "x", "passage_body": "word " * 60, "tag": "short"} for _ in range(5)]
        assert pick_text_field(rows) == "passage_body"

    def test_short_string_fields_are_ids_not_documents(self):
        assert pick_text_field([{"id": "abc", "tag": "x"} for _ in range(5)]) == ""

    def test_length_stats_describe_the_chunker(self):
        assert length_stats(["a" * 400, "a" * 3200, "a" * 1800]) == (400, 1800, 3200)

    def test_no_text_is_none_not_zero(self):
        assert length_stats([]) is None


class TestScripts:
    def test_it_names_the_writing_system(self):
        assert scripts_of(["the quick brown fox jumps"]) == ("Latin",)
        assert scripts_of(["नमस्ते दुनिया कैसे हो आप"]) == ("Devanagari",)

    def test_a_mixed_corpus_reports_both(self):
        assert set(scripts_of(["hello world " * 20, "नमस्ते दुनिया " * 20])) == {
            "Latin",
            "Devanagari",
        }

    def test_one_stray_name_does_not_make_a_second_language(self):
        """A transliterated word in an English corpus is noise, not a script."""
        assert scripts_of(["english text everywhere " * 40 + "नम"]) == ("Latin",)


class TestPgVectorReading:
    def test_it_finds_a_vector_column_by_type_not_by_name(self):
        columns = {"id": "integer", "chunk_text": "text", "vec": "vector(768)"}
        assert _vector_column(columns) == ("vec", 768)

    def test_a_table_with_no_vectors_says_so(self):
        assert _vector_column({"id": "integer", "body": "text"}) == (None, None)

    def test_jsonb_metadata_is_flattened_into_the_record(self):
        """Nested metadata would report one field called `meta` at 100% coverage
        and nothing about what an agent could actually filter on."""
        flat = _scrub({"id": 1, "text": "x", "meta": {"strategy": "S1", "language": "hin"}})

        assert flat["strategy"] == "S1"
        assert flat["language"] == "hin"
        assert "meta" not in flat


class TestCard:
    """What an agent is handed. It cannot check any of this against the store."""

    def _profile(self, **kwargs) -> Profile:
        observation = Observation(
            connector="pgvector",
            kind="vector",
            location="pgvector/db#book_chunks",
            reachable=True,
            sampled=200,
            vectors=VectorShape(dimensions=384, metric="cosine", records=2366),
            fields=(
                FieldStat(name="book_id", types=("string",), coverage=1.0, distinct=12, carried=200),
                FieldStat(name="page", types=("int",), coverage=1.0, distinct=1,
                          carried=200, examples=("1",)),
                FieldStat(name="strategy", types=("string",), coverage=0.01, carried=2),
                # A timestamp is on every record and is not a facet.
                FieldStat(name="created_at", types=("datetime",), coverage=1.0, carried=200),
            ),
            text_field="chunk_text",
        )
        return Profile(
            connector="pgvector",
            kind="vector",
            status=kwargs.pop("status", "ok"),
            observation=observation,
            understanding=Understanding(
                title="Self-help library",
                summary="Twelve popular self-help and business books.",
                good_for=("habit formation", "negotiation"),
                not_for=("anything after 2018",),
            ),
            facts=kwargs.pop("facts", CapabilityFacts(searchable=True)),
            **kwargs,
        )

    def test_it_leads_with_what_and_how_big(self):
        head = self._profile().card().splitlines()[0]
        assert "Self-help library" in head and "2.4k records" in head

    def test_only_usable_fields_are_offered_as_filters(self):
        card = self._profile().card()
        assert "Filter on: book_id" in card
        assert "strategy" not in card.split("Filter on:")[1].split("\n")[0]

    def test_identifiers_and_the_text_column_are_not_offered_as_filters(self):
        """What a live run against a real foreign schema first produced:
        "Filter on: book_id, chunk_text, id" — two of which identify a row."""
        observation = self._profile().observation
        card = Profile(
            connector="pgvector",
            kind="vector",
            status="ok",
            observation=Observation(
                connector="pgvector",
                kind="vector",
                location=observation.location,
                reachable=True,
                sampled=200,
                vectors=observation.vectors,
                fields=(
                    FieldStat(name="book_id", types=("string",), coverage=1.0,
                              distinct=11, carried=200),
                    FieldStat(name="id", types=("int",), coverage=1.0, distinct=200, carried=200),
                    FieldStat(name="chunk_text", types=("string",), coverage=1.0,
                              distinct=200, carried=200),
                ),
                text_field="chunk_text",
            ),
            facts=CapabilityFacts(searchable=True),
        ).card()

        offered = card.split("Filter on:")[1].split("\n")[0]
        assert "book_id" in offered
        assert "id," not in offered and "chunk_text" not in offered

    def test_a_timestamp_is_not_offered_as_a_filter(self):
        """`created_at`, `origins` and `sourceQueryIds` were all offered on a
        real store. None of them is something a query can group by."""
        offered = self._profile().card().split("Filter on:")[1].split("\n")[0]
        assert "created_at" not in offered

    def test_a_constant_field_is_named_as_useless(self):
        """Otherwise an agent keeps trying to cite a page number that is always 1."""
        assert "Carried but useless" in self._profile().card()
        assert "page" in self._profile().card()

    def test_an_unsearchable_store_is_told_not_to_be_used(self):
        card = self._profile(facts=CapabilityFacts(searchable=False)).card()
        assert "do not route questions here" in card

    def test_a_failed_profile_says_why_and_stops(self):
        card = self._profile(status=STATUS_FAILED, error="the index did not answer").card()
        assert "Unavailable: the index did not answer" in card
        assert "Filter on" not in card

    def test_truncation_drops_whole_lines(self):
        """A card cut mid-sentence reads as a claim that stops, and the half it
        keeps is the half the model is about to act on."""
        card = self._profile().card(budget=90)
        assert not card.endswith("…")
        assert all(line.strip() for line in card.splitlines())

    def test_the_budget_is_honoured(self):
        assert len(self._profile().card(budget=120)) <= 120


class TestRoundTrip:
    def test_a_profile_survives_postgres(self):
        original = TestCard()._profile()
        restored = Profile.from_json(original.to_json())

        assert restored is not None
        assert restored.card() == original.card()
        assert restored.facts == original.facts
        assert restored.observation.fields == original.observation.fields

    def test_an_older_version_is_re_probed_not_migrated(self):
        """A profile is a cache of somebody else's database. Recomputing it
        beats inventing the fields a migration would have to fill in."""
        blob = TestCard()._profile().to_json()
        blob["version"] = 0

        assert Profile.from_json(blob) is None

    def test_a_corrupt_blob_is_absent_rather_than_fatal(self):
        assert Profile.from_json({"version": 1, "profiled_at": "not a date"}) is None
        assert Profile.from_json({}) is None


class TestFingerprint:
    """Why a stale profile is the one failure here with no visible symptom."""

    def _row(self, sealed: str) -> ProfileRow:
        return ProfileRow(
            user_id="user_1",
            connector="pinecone",
            status="ok",
            profile={},
            card="",
            fingerprint=fingerprint(sealed),
            error="",
            profiled_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def test_it_matches_the_credentials_it_was_built_from(self):
        assert self._row("sealed-blob-a").matches("sealed-blob-a")

    def test_rotating_a_key_invalidates_the_understanding(self):
        """Reconnecting may point at a different index entirely, and a profile
        describing the old one is confidently wrong about everything."""
        assert not self._row("sealed-blob-a").matches("sealed-blob-b")

    def test_an_unfingerprinted_row_never_matches(self):
        row = self._row("x")
        row.fingerprint = ""
        assert not row.matches("x")

    def test_it_is_taken_over_ciphertext(self):
        """A digest of a secret is a thing worth not creating, even one-way."""
        assert fingerprint("sealed") != "sealed"
        assert len(fingerprint("sealed")) == 32

    def test_staleness_is_measured_from_the_probe_not_the_write(self):
        row = self._row("x")
        row.profiled_at = datetime.now(timezone.utc) - timedelta(hours=48)

        assert row.stale(timedelta(hours=24))
        assert not row.stale(timedelta(hours=72))

    def test_a_row_that_never_probed_is_stale(self):
        row = self._row("x")
        row.profiled_at = None
        assert row.stale(timedelta(hours=24))


class TestNarration:
    def test_a_tools_connector_needs_no_model(self):
        """"The agent can send mail through Gmail" is the authorised list,
        rephrased — not an inference worth a round trip to get wrong."""
        observed = Observation(
            connector="composio",
            kind="tools",
            reachable=True,
            tools=ToolShape(authorised=("gmail", "slack")),
        )
        written = _tools(observed)

        assert written is not None
        assert "gmail" in written.summary and "slack" in written.summary

    def test_nothing_authorised_is_stated_plainly(self):
        written = _tools(
            Observation(connector="composio", kind="tools", reachable=True, tools=ToolShape())
        )
        assert written is not None
        assert "no toolkit" in written.summary.lower()

    def test_model_output_is_clamped_not_trusted(self):
        assert len(_text("x" * 900, 60)) == 60
        assert _list(["a", "b", "c", "d", "e", "f", "g"], 3) == ("a", "b", "c")

    def test_a_string_is_not_a_list_of_characters(self):
        assert _list("topics", 3) == ()

    def test_empty_understanding_is_detectable(self):
        assert Understanding().empty
        assert not Understanding(summary="something").empty


class TestExcerpts:
    """Which passages the model gets to read, which decides what it can say."""

    def _corpus(self) -> list[dict]:
        """One long document and four short ones — the shape that breaks a
        uniform sample, and the shape a chunked book library actually has."""
        filler = " and then several more words to clear the minimum length"
        rows = [
            {"book_id": "big", "text": f"a passage about habits, number {i}{filler}"}
            for i in range(180)
        ]
        for name in ("negotiation", "power", "longevity", "running"):
            rows += [
                {"book_id": name, "text": f"a passage about {name}, number {i}{filler}"}
                for i in range(5)
            ]
        return rows

    def test_it_spreads_across_documents_not_rows(self):
        """A uniform sample of this corpus is 90% one book, and the summary
        written from it describes that book instead of the library."""
        rows = self._corpus()
        picked = excerpts_of(rows, "text", stats=field_stats(rows))

        sources = {e.split("about ")[1].split(",")[0] for e in picked}
        assert len(sources) >= 4, f"only reached {sources}"

    def test_it_still_works_with_no_document_boundary(self):
        rows = [
            {"text": f"passage number {i} of a flat corpus with enough words to be kept"}
            for i in range(50)
        ]
        assert len(excerpts_of(rows, "text", stats=field_stats(rows))) > 0

    def test_the_facet_is_found_by_shape_not_by_name(self):
        """Nothing is looked up as `book_id`. A document boundary is whatever
        is on every record, has few values, and is not unique per row."""
        rows = self._corpus()
        assert dominant_facet(field_stats(rows), "text") == "book_id"

    def test_an_identifier_is_not_a_document_boundary(self):
        rows = [
            {"text": f"passage {i} with enough words in it to count as a document", "id": i}
            for i in range(50)
        ]
        assert dominant_facet(field_stats(rows), "text") == ""

    def test_disabling_excerpts_yields_none_at_all(self):
        """`PROFILE_EXCERPTS=false` must send no passage anywhere."""
        rows = self._corpus()
        assert excerpts_of(rows, "text", stats=field_stats(rows), allowed=False) == ()

    def test_excerpts_are_capped(self):
        rows = self._corpus()
        for excerpt in excerpts_of(rows, "text", stats=field_stats(rows)):
            assert len(excerpt) <= EXCERPT_CHARS


class TestClamping:
    def test_a_list_item_is_cut_at_a_word_boundary(self):
        """"matching passages by meaning rather than exact w" — what the live
        model actually returned through the old character-index clamp."""
        clamped = _text("matching passages by meaning rather than exact wording", 48)

        assert not clamped.endswith("w")
        assert clamped == "matching passages by meaning rather than exact"

    def test_one_very_long_word_still_clamps(self):
        assert len(_text("x" * 200, 48)) == 48

    def test_a_short_value_is_untouched(self):
        assert _text("negotiation", 48) == "negotiation"

    def test_a_clamped_phrase_does_not_end_on_a_conjunction(self):
        """"searching for entrepreneurship and" — what the live model returned.
        It reads as a phrase whose second half was lost, because it was."""
        assert _text("searching for entrepreneurship and finance", 34) == (
            "searching for entrepreneurship"
        )
        assert _text("lookups that require page variation since pages differ", 46) == (
            "lookups that require page variation"
        )

    def test_a_value_short_enough_to_keep_is_never_edited(self):
        """The trim exists to repair a cut. A complete phrase the model wrote
        is left alone even when it happens to end on one of these words."""
        assert _text("everything that is true of a page and", 48) == (
            "everything that is true of a page and"
        )

    def test_trimming_never_empties_a_single_word(self):
        assert _text("and", 48) == "and"


class TestWhatReachesThePrompt:
    """`cards()` feeds a prompt whose first line is "spoken aloud immediately"."""

    def _profile(self, *, kind="vector", searchable=True, authorised=()) -> Profile:
        return Profile(
            connector="pgvector" if kind == "vector" else "composio",
            kind=kind,
            status="ok" if searchable else "failed",
            observation=Observation(
                connector="pgvector", kind=kind, location="pgvector/h#t", reachable=searchable,
                vectors=VectorShape(dimensions=384, records=10) if kind == "vector" else None,
                tools=ToolShape(authorised=authorised) if kind == "tools" else None,
            ),
            facts=CapabilityFacts(searchable=searchable),
            error="" if searchable else 'there is no table called "chunks" on the search path',
        )

    def test_a_usable_store_is_offered(self):
        assert _usable(self._profile())

    def test_a_broken_store_is_not(self):
        """Its card carries an operator's error message. The panel reports that;
        a prompt that is read out loud does not need it."""
        assert not _usable(self._profile(searchable=False))

    def test_a_composio_with_nothing_authorised_is_not(self):
        """It can list hundreds of tools and execute none of them."""
        assert not _usable(self._profile(kind="tools"))
        assert _usable(self._profile(kind="tools", authorised=("gmail",)))

    def test_the_single_card_still_reports_the_failure(self):
        """`card()` is what the panel reads, and it must say what went wrong."""
        assert "Unavailable" in self._profile(searchable=False).card()


class TestProseIsNotAFacet:
    """The rule that stopped `english` flickering in and out of the filter list."""

    def test_a_label_column_is_a_facet(self):
        rows = [{"text": "x" * 200, "query_type": "DESCRIPTION"} for _ in range(50)]
        assert next(s for s in field_stats(rows) if s.name == "query_type").categorical

    def test_a_prose_column_is_not(self):
        """A parallel-translation column is a string, and a sample holding a
        repeated passage makes it look enumerable. Length settles it."""
        passage = "a long English rendering of the passage " * 4
        rows = [{"text": "x" * 200, "english": passage} for _ in range(50)]
        assert not next(s for s in field_stats(rows) if s.name == "english").categorical

    def test_a_timestamp_is_not(self):
        from datetime import datetime

        rows = [{"text": "x" * 200, "created_at": datetime.now()} for _ in range(50)]
        assert not next(s for s in field_stats(rows) if s.name == "created_at").categorical

    def test_a_list_is_not(self):
        rows = [{"text": "x" * 200, "origins": [1, 2, 3]} for _ in range(50)]
        assert not next(s for s in field_stats(rows) if s.name == "origins").categorical

    def test_mean_length_survives_the_round_trip(self):
        rows = [{"text": "x" * 200, "tag": "short"} for _ in range(20)]
        stat = next(s for s in field_stats(rows) if s.name == "tag")
        assert stat.mean_chars == 5


class TestTableDiscovery:
    """`table` is optional, so it arrives blank — which is the ordinary case.

    Defaulting a blank to this app's own `chunks` told people their database
    was missing a table they had never heard of. Their database was fine; the
    question was never asked.
    """

    def test_one_vector_table_is_not_a_guess(self):
        """It is the only table a vector search could run against. Asking
        someone to type a name the server can already see is a form that
        exists to be got wrong."""
        assert _resolve_table(["book_chunks"]) == "book_chunks"

    def test_several_are_named_so_the_user_can_pick(self):
        with pytest.raises(ConnectorError) as caught:
            _resolve_table(["chunks", "docs", "notes"])

        message = str(caught.value)
        assert "3 tables" in message
        assert all(name in message for name in ("chunks", "docs", "notes"))

    def test_none_is_a_statement_about_their_database(self):
        """Not about our default. The old message named `chunks`, which is
        ours, and sent people looking for a table nobody had told them to make."""
        with pytest.raises(ConnectorError) as caught:
            _resolve_table([])

        message = str(caught.value)
        assert "no table with a vector column" in message
        assert "chunks" not in message

    def test_a_long_list_is_truncated_with_a_count(self):
        message = _name_candidates([f"t{i}" for i in range(20)])
        assert "and 14 more" in message

    def test_verify_hands_back_what_it_resolved(self):
        """A blank field is stored as the name it resolved to, so nothing
        downstream has to guess again."""
        from src.connectors.registry import SPECS

        spec = next(s for s in SPECS if s.slug == "pgvector")
        assert spec.fields[1].name == "table" and not spec.fields[1].required


class TestDimension:
    """The width is read, never asked.

    This class used to test a model name typed on the form. That question was
    unanswerable by the person being asked — the reasonable reply to "stores
    768-dimensional vectors" is `768`, which is exactly what happened, and
    fastembed spent 39 seconds trying to download a repository by that name.
    The width is in the catalogue, `text-embedding-3` takes a `dimensions`
    parameter, so there is nothing left to get wrong.
    """

    def test_matching_width_needs_nothing(self):
        assert reconcile_dimension("“t”", 384, 384) == {"dim": "384"}

    def test_a_different_width_is_recorded_not_rejected(self):
        """768, 1536 and 3072 are all reachable — they are what connected
        stores are actually built at."""
        for width in (768, 1024, 1536, 3072):
            assert reconcile_dimension("“t”", width, 384) == {"dim": str(width)}

    def test_a_width_nothing_can_embed_is_refused(self):
        with pytest.raises(ConnectorError) as caught:
            reconcile_dimension("“t”", 4096, 384)

        message = str(caught.value)
        assert "4096" in message and "3072" in message

    def test_an_unknown_store_width_is_not_treated_as_a_mismatch(self):
        """A store that will not say how wide it is gets the benefit of doubt
        rather than an error about a number nobody measured."""
        assert reconcile_dimension("“t”", None, 384) == {"dim": "384"}

    def test_the_model_is_chosen_by_width_not_by_the_user(self):
        assert model_for(768) == "text-embedding-3-small"
        assert model_for(1536) == "text-embedding-3-small"
        assert model_for(3072) == "text-embedding-3-large"

    def test_the_connector_form_asks_nothing_about_models(self):
        """The field that caused this is gone, not merely validated."""
        from src.connectors.registry import SPECS

        for spec in SPECS:
            assert "model" not in [f.name for f in spec.fields], spec.slug


class TestMetric:
    """Querying a `vector_l2_ops` index with `<=>` does not use the index."""

    def test_it_is_read_off_the_opclass(self):
        for opclass, expected in [
            ("vector_cosine_ops", "cosine"),
            ("vector_ip_ops", "inner_product"),
            ("vector_l2_ops", "l2"),
            ("halfvec_cosine_ops", "cosine"),
        ]:
            defs = {"embedding": f"CREATE INDEX x ON t USING hnsw (embedding {opclass})"}
            assert metric_of(defs, "embedding") == expected

    def test_no_index_falls_back_to_cosine(self):
        """Not because it is known right — because it is what an unindexed
        table is scanned with, and the scale everything was calibrated on."""
        assert metric_of({}, "embedding") == "cosine"

    def test_an_index_on_another_column_does_not_count(self):
        defs = {"other": "CREATE INDEX x ON t USING hnsw (other vector_l2_ops)"}
        assert metric_of(defs, "embedding") == "cosine"


class TestRealWorldSchemas:
    """Every pgvector table this app is likely to be pointed at.

    Not hypothetical shapes — these are what LangChain, LlamaIndex, Supabase's
    quickstart, pgai and a hand-rolled Django model actually create. The bug
    this class exists for was found by writing it out: name heuristics alone
    picked LangChain's `id` varchar as the document column, so every answer
    would have been quoted from a UUID, from the single most widely used
    pgvector integration there is.
    """

    DOC = (
        "A reasonably long passage of prose that a retrieval system would "
        "actually quote back to somebody asking a question about it."
    )

    SCHEMAS = {
        "langchain": (
            {"id": "character varying", "collection_id": "uuid",
             "embedding": "vector(1536)", "document": "character varying",
             "cmetadata": "jsonb"},
            [{"id": "3f2a1e5c-aaaa-bbbb", "document": DOC}],
            {"text": "document", "id": "id", "meta": "cmetadata"},
        ),
        "langchain-legacy": (
            {"uuid": "uuid", "collection_id": "uuid", "embedding": "vector(1536)",
             "document": "character varying", "cmetadata": "json",
             "custom_id": "character varying"},
            [{"document": DOC, "custom_id": "doc-17"}],
            {"text": "document", "id": "uuid", "meta": "cmetadata"},
        ),
        "llamaindex": (
            {"id": "bigint", "text": "character varying", "metadata_": "json",
             "node_id": "character varying", "embedding": "vector(1536)"},
            [{"text": DOC, "node_id": "9c1f"}],
            {"text": "text", "id": "id", "meta": "metadata_"},
        ),
        "supabase": (
            {"id": "bigint", "content": "text", "metadata": "jsonb",
             "embedding": "vector(1536)"},
            [{"content": DOC}],
            {"text": "content", "id": "id", "meta": "metadata"},
        ),
        "pgai": (
            {"embedding_uuid": "uuid", "id": "integer", "chunk_seq": "integer",
             "chunk": "text", "embedding": "vector(768)"},
            [{"chunk": DOC}],
            {"text": "chunk", "id": "id"},
        ),
        "django": (
            {"id": "bigint", "title": "character varying", "body": "text",
             "embedding": "vector(384)"},
            [{"title": "Short title", "body": DOC}],
            {"text": "body", "id": "id"},
        ),
        "book_chunks": (
            {"id": "integer", "book_id": "text", "chunk_text": "text",
             "page": "integer", "embedding": "vector(768)"},
            [{"book_id": "69b9", "chunk_text": DOC}],
            {"text": "chunk_text", "id": "id"},
        ),
    }

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_it_maps_the_right_columns(self, name):
        columns, rows, expected = self.SCHEMAS[name]
        mapped = discover_columns(columns, rows=rows)

        for role, column in expected.items():
            assert mapped.get(role) == column, f"{name}: {role}"

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_all_of_them_are_searchable(self, name):
        columns, rows, _ = self.SCHEMAS[name]
        assert ColumnMap.from_mapping(discover_columns(columns, rows=rows)).searchable

    def test_none_of_them_claims_a_capability_it_lacks(self):
        """`strategy`, `tsv` and the English pair are this app's own schema —
        an ordinary connected table carries none of them, and must not be
        credited with the channels they would have bought."""
        for name, (columns, rows, _) in self.SCHEMAS.items():
            cmap = ColumnMap.from_mapping(discover_columns(columns, rows=rows))
            assert not (cmap.filters or cmap.lexical or cmap.parallel_text), name

    def test_the_longest_prose_column_wins_over_a_shorter_one(self):
        """A headline and an article are both `text` and both prose. The one
        worth quoting is the long one, and only the data says which."""
        mapped = discover_columns(
            {"id": "bigint", "headline": "text", "article": "text",
             "embedding": "vector(384)"},
            rows=[{"headline": "Short headline here", "article": self.DOC}],
        )
        assert mapped["text"] == "article"
        assert "headline" in mapped.get("payload", "")

    def test_names_alone_still_work_when_the_table_is_empty(self):
        """An empty table cannot be sampled. The conventional names are the
        fallback, which is where this started and is still right most of the
        time — just not for LangChain."""
        mapped = discover_columns(
            {"id": "bigint", "content": "text", "embedding": "vector(384)"}, rows=[]
        )
        assert mapped["text"] == "content"

    def test_an_id_shaped_column_is_never_the_document(self):
        """The LangChain bug, pinned: `id` is a varchar, comes first in
        catalogue order, and would win any 'first text column' fallback."""
        mapped = discover_columns(
            {"id": "character varying", "document": "character varying",
             "embedding": "vector(1536)"},
            rows=[],
        )
        assert mapped["text"] == "document"
