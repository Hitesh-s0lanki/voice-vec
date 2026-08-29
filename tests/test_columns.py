"""The column map, and the one guarantee everything else rests on.

`VectorStore`'s queries used to name this app's columns as literals. Making
them data is what lets somebody connect a pgvector table holding `id` and
`chunk_text` instead of being told their database is not a copy of ours — but
it also puts a string builder in the middle of the measured hot path.

So the first test in this file is the one that matters: **the default map must
generate the query it replaced, character for character.** That query is the
measured hot path and a refactor that quietly changed it would be a latency
regression nobody could attribute. The constants it is compared against are
pasted here rather than imported — which is what let them outlive the literals
in `store.py`, and is why this still pins the shape now that those are gone.

The rest is the rule that makes an unfamiliar schema safe: an absent column is
a *lost capability*, never a substituted one. A dropped predicate is honest; a
defaulted one is a narrowing the caller believes it applied.
"""

from __future__ import annotations

import pytest

from src.rag.columns import (
    DEFAULT,
    METRICS,
    ColumnMap,
    lexical_sql,
    operator,
    safe,
    score_expression,
    search_sql,
    text_config,
)

# What `store.py` shipped before the map existed. Pasted, not imported.
_SEARCH_WAS = """
SELECT chunk_key,
       strategy,
       text,
       meta,
       1 - ({embedding} <=> %(vector)s) AS score
FROM {table}
WHERE strategy = ANY(%(strategies)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
  AND {embedding} IS NOT NULL
ORDER BY {embedding} <=> %(vector)s
LIMIT %(limit)s
"""

_LEXICAL_WAS = """
SELECT chunk_key,
       strategy,
       text,
       meta,
       ts_rank_cd({tsv}, query, 32) AS score
FROM {table}, websearch_to_tsquery('{config}', %(query)s) AS query
WHERE {tsv} @@ query
  AND strategy = ANY(%(strategies)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
ORDER BY score DESC
LIMIT %(limit)s
"""


class TestTheDeploymentStoreIsUnchanged:
    """Character for character, with exactly one deliberate difference.

    The table name is quoted — `FROM "chunks"` rather than `FROM chunks`.
    Semantically identical for a lower-case name, and required for every other
    kind: Postgres folds an unquoted identifier, so a table created as
    `"MyDocs"` cannot be found by writing `MyDocs`, and a schema-qualified name
    cannot be written at all. Called out here rather than hidden, so the pin
    still means what it says.
    """

    def test_dense_search(self):
        assert search_sql("chunks", DEFAULT) == _SEARCH_WAS.format(
            table='"chunks"', embedding="embedding"
        )

    def test_dense_search_english(self):
        assert search_sql("chunks", DEFAULT, english=True) == _SEARCH_WAS.format(
            table='"chunks"', embedding="embedding_en"
        )

    def test_lexical(self):
        assert lexical_sql("chunks", DEFAULT, config="simple") == _LEXICAL_WAS.format(
            table='"chunks"', tsv="tsv", config="simple"
        )

    def test_lexical_english(self):
        got = lexical_sql("chunks", DEFAULT, english=True, config="english")
        assert got == _LEXICAL_WAS.format(table='"chunks"', tsv="tsv_en", config="english")

    def test_a_qualified_or_mixed_case_name_survives(self):
        assert 'FROM "rag"."My Docs"' in search_sql("rag.My Docs", DEFAULT)

    def test_a_bare_map_is_this_apps_schema(self):
        assert DEFAULT.lexical and DEFAULT.filters and DEFAULT.parallel_text


class TestAnUnfamiliarSchema:
    THEIRS = ColumnMap(
        id="id",
        text="chunk_text",
        embedding="embedding",
        meta="",
        strategy="",
        language="",
        english="",
        embedding_en="",
        tsv="",
        tsv_en="",
        payload=("book_id",),
    )

    def test_the_predicate_is_dropped_not_defaulted(self):
        """A defaulted `strategy = ANY(…)` against a column that is not there
        fails; one defaulted to match everything is worse — the ladder records
        a narrowing it never got."""
        sql = search_sql("book_chunks", self.THEIRS)

        assert "strategy = ANY" not in sql
        assert "language" not in sql
        assert "WHERE embedding IS NOT NULL" in sql

    def test_absent_roles_are_filled_honestly_not_invented(self):
        sql = search_sql("book_chunks", self.THEIRS)

        assert "''::text" in sql  # strategy, with no strategy to report
        assert "chunk_text" in sql

    def test_a_hit_can_still_name_its_source(self):
        """No `meta` column, but a `book_id` worth citing. Without this a hit
        from a connected store is an anonymous passage."""
        assert "jsonb_build_object('book_id', book_id)" in search_sql("t", self.THEIRS)

    def test_no_meta_and_nothing_to_cite_is_an_empty_payload(self):
        bare = ColumnMap(id="id", text="body", embedding="v", meta="", strategy="",
                         language="", english="", embedding_en="", tsv="", tsv_en="")
        assert "'{}'::jsonb" in search_sql("t", bare)

    def test_an_english_question_falls_back_to_the_only_vector(self):
        """The cross-lingual hop, rather than native English retrieval. A store
        with no `embedding_en` has one vector and that is what gets searched."""
        assert "embedding <=>" in search_sql("t", self.THEIRS, english=True)

    def test_and_stems_with_simple_because_there_is_no_english_column(self):
        assert text_config(self.THEIRS, english=True) == "simple"
        assert text_config(DEFAULT, english=True) == "english"


class TestCapabilitiesComeFromTheMap:
    def test_no_tsvector_is_no_lexical_channel(self):
        assert not ColumnMap(tsv="", tsv_en="").lexical

    def test_no_strategy_is_no_filtering(self):
        assert not ColumnMap(strategy="").filters

    def test_searchable_needs_something_to_read_back(self):
        """A store that can be searched and not quoted has no source for an
        answer cut out of it."""
        assert not ColumnMap(embedding="v", text="").searchable
        assert not ColumnMap(embedding="", text="body").searchable
        assert ColumnMap(embedding="v", text="body").searchable


class TestRoundTrip:
    def test_a_map_survives_the_credential_blob(self):
        original = TestAnUnfamiliarSchema.THEIRS
        assert ColumnMap.from_mapping(original.to_mapping()) == original

    def test_an_unmentioned_role_is_absent_not_defaulted(self):
        """The bug this caught in review: the dataclass defaults are *our*
        column names, so a partial map inherited `strategy` and generated a
        query against a column that was not there."""
        restored = ColumnMap.from_mapping({"text": "body", "embedding": "v"})

        assert restored.strategy == "" and restored.tsv == "" and restored.meta == ""
        assert not (restored.filters or restored.lexical)

    def test_an_empty_map_means_an_account_from_before_mapping(self):
        """Those were verified against this app's schema, so that is what they are."""
        assert ColumnMap.from_mapping({}) == DEFAULT
        assert ColumnMap.from_mapping(None) == DEFAULT


class TestIdentifierSafety:
    """Column names reach a query as text — Postgres takes no parameter there."""

    @pytest.mark.parametrize(
        "name", ["chunk_text", "_private", "col$1", "a" * 63]
    )
    def test_ordinary_names_pass(self, name):
        assert safe(name) == name

    @pytest.mark.parametrize(
        "name",
        ['"; DROP TABLE chunks; --', "has space", "1leading", "quote'd", "a" * 64, ""],
    )
    def test_anything_else_is_dropped_rather_than_escaped(self, name):
        """Dropped, not rejected: the same answer as "this store has no such
        column", so an exotic name costs one capability instead of the connection."""
        assert safe(name) == ""

    def test_an_injected_name_cannot_reach_the_sql(self):
        hostile = ColumnMap.from_mapping(
            {"text": "body", "embedding": "v", "strategy": "x; DROP TABLE chunks"}
        )
        sql = search_sql("t", hostile)

        assert "DROP" not in sql
        assert not hostile.filters


class TestDistanceMetric:
    """The operator has to match the opclass, and the score has to be a
    similarity. Both are silent when wrong — one costs the index, the other
    inverts every guardrail comparison."""

    def test_cosine_is_the_default_and_unchanged(self):
        sql = search_sql("t", ColumnMap())
        assert "1 - (embedding <=> %(vector)s)" in sql
        assert "ORDER BY embedding <=> %(vector)s" in sql

    def test_inner_product_negates_because_pgvector_returns_it_negative(self):
        sql = search_sql("t", ColumnMap(metric="inner_product"))
        assert "(-1) * (embedding <#> %(vector)s)" in sql
        assert "ORDER BY embedding <#> %(vector)s" in sql

    def test_l2_is_bounded_into_a_similarity(self):
        sql = search_sql("t", ColumnMap(metric="l2"))
        assert "1 / (1 + (embedding <-> %(vector)s))" in sql
        assert "ORDER BY embedding <-> %(vector)s" in sql

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_every_metric_orders_by_the_operator_it_scores_with(self, metric):
        """Scoring with one operator and ordering by another returns the right
        rows in the wrong order, or the wrong rows entirely."""
        sql = search_sql("t", ColumnMap(metric=metric))

        assert f"{score_expression('embedding', metric)} AS score" in sql
        assert f"ORDER BY embedding {operator(metric)} %(vector)s" in sql

    def test_an_unknown_metric_falls_back_to_cosine(self):
        assert search_sql("t", ColumnMap(metric="nonsense")) == search_sql("t", ColumnMap())

    def test_metric_survives_the_round_trip(self):
        original = ColumnMap(metric="l2")
        assert ColumnMap.from_mapping(original.to_mapping()).metric == "l2"

    def test_an_injected_metric_cannot_reach_the_sql(self):
        """`metric` is a keyword from a fixed set, not a column name."""
        hostile = ColumnMap.from_mapping(
            {"text": "t", "embedding": "v", "metric": "x; DROP TABLE chunks"}
        )
        assert hostile.metric == "cosine"
        assert "DROP" not in search_sql("t", hostile)
