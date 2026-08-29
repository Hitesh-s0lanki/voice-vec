"""Unit tests for attached datasets — the layer where a wrong answer looks right.

Almost nothing this package does fails loudly. That is the whole reason these
tests exist, and it is what each group below is aimed at:

  - **the SQL guard**, because a statement that gets through is executed. It is
    tested against the forms that *parse as* SELECT and are not — `PRAGMA` is
    reported as a SELECT by DuckDB's own parser, which is exactly the assumption
    an allow-list makes and gets wrong;
  - **the seal**, because the sandbox is the only thing standing between a
    model-written string and the filesystem, and every claim made about it in
    `sandbox.py` is asserted here rather than believed;
  - **the schema card**, because it is the entire reason generated SQL is
    correct. A card that loses `query_type`'s five values still reads fine and
    produces `WHERE query_type = 'FACT'` — valid SQL, zero rows, and a result
    indistinguishable from an honest empty one;
  - **omitted columns**, because a column that was too expensive to pull and is
    not *named* as missing is a column a model writes SQL against;
  - **sampling**, because "25,000 rows" and "the first 25,000 of 97,941" differ
    only in whether an aggregate is a fact.

The sandbox tests build a real DuckDB file. They are the ones worth keeping
slow: the seal is a claim about a C++ engine's behaviour, and a mock of it would
assert only that the mock was written to agree.
"""

from __future__ import annotations

import duckdb
import pytest

from src.agents.dataset_agent import DatasetAgent
from src.datasets.materialise import _top_level
from src.datasets.profile import (
    FILTERABLE_COVERAGE,
    ColumnStat,
    DatasetProfile,
    Observation,
    Omitted,
    TableStat,
    Understanding,
)
from src.datasets.sandbox import Sandbox, SandboxUnavailable
from src.datasets.source import SourceError, _ident, _slug, _suffix, _tables, resolve
from src.datasets.sql import Rejected, clean, guard
from src.tools.dataset import QUERY_TOOL, DatasetTools
from src.tools.sql import reason
from tests.fakes import FakeToolModel


# ---- fixtures ------------------------------------------------------------


@pytest.fixture
def dataset_file(tmp_path):
    """A small, real DuckDB database — the sandbox's claims are about this."""
    path = tmp_path / "sample.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE hinval AS "
        "SELECT i AS query_id, "
        "       ['DESCRIPTION', 'NUMERIC', 'ENTITY'][(i % 3) + 1] AS query_type, "
        "       'question number ' || i AS query "
        "FROM range(1, 501) t(i)"
    )
    con.close()
    return str(path)


def column(name: str, **kwargs) -> ColumnStat:
    return ColumnStat(name=name, **kwargs)


def profile_of(*tables: TableStat, understanding: Understanding | None = None) -> DatasetProfile:
    return DatasetProfile(
        dataset_id="demo",
        observation=Observation(source="hf", location="org/name", tables=tables),
        understanding=understanding,
    )


# ---- the guard -----------------------------------------------------------


class TestSqlGuard:
    """What may run. Everything here is a statement somebody's model will write."""

    @pytest.mark.parametrize(
        "statement",
        [
            "SELECT 1",
            "select 1",
            "SELECT a, b FROM t WHERE x ILIKE '%q%' LIMIT 10",
            "WITH a AS (SELECT 1) SELECT * FROM a",
            "FROM t SELECT count(*)",
            "VALUES (1)",
            "(SELECT 1)",
            "/* a comment */ SELECT 2",
        ],
    )
    def test_allows_reads(self, statement):
        assert guard(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a = 1",
            "DELETE FROM t",
            "CREATE TABLE x (a INT)",
            "DROP TABLE t",
            "ATTACH '/tmp/other.db'",
            "COPY (SELECT 1) TO '/tmp/out.csv'",
            "INSTALL httpfs",
            "SET memory_limit = '99GB'",
            "EXPLAIN SELECT 1",
        ],
    )
    def test_refuses_everything_else(self, statement):
        with pytest.raises(Rejected):
            guard(statement)

    def test_refuses_a_second_statement(self):
        """The classic. A trailing statement rides in on a legal first one."""
        with pytest.raises(Rejected, match="one SELECT"):
            guard("SELECT 1; DROP TABLE t")

    @pytest.mark.parametrize(
        "statement",
        ["PRAGMA database_list", "pragma show_tables", "-- fetch rows\nPRAGMA version"],
    )
    def test_refuses_pragma_despite_it_parsing_as_select(self, statement):
        """The reason the parser check is not the only check.

        DuckDB rewrites `PRAGMA` into a table function, so `extract_statements`
        reports it as a SELECT. An allow-list that trusts the statement type —
        which is the obvious implementation — lets every one of these through.
        """
        assert duckdb.extract_statements(statement)[0].type.name == "SELECT"
        with pytest.raises(Rejected):
            guard(statement)

    def test_a_semicolon_inside_a_string_is_not_a_statement(self):
        """A regex loses this one. The parser does not."""
        assert guard("SELECT * FROM t WHERE x = ';DROP TABLE t;'")

    def test_a_comment_is_not_a_statement(self):
        assert guard("select 1 -- ; drop table t")

    def test_strips_the_fence_a_model_adds(self):
        assert clean("```sql\nSELECT 1\n```") == "SELECT 1"
        assert guard("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_empty_is_refused_not_run(self):
        for empty in ("", "   ", "```sql\n\n```"):
            with pytest.raises(Rejected, match="No SQL"):
                guard(empty)

    def test_the_refusal_says_what_to_do(self):
        """The message goes back to the model verbatim, so it has to be usable.

        "Invalid SQL" produces the same query again; naming the constraint
        produces a working second attempt.
        """
        with pytest.raises(Rejected, match="read-only"):
            guard("DELETE FROM t")


# ---- the seal ------------------------------------------------------------


class TestSandbox:
    """Every claim `sandbox.py` makes, asserted against the real engine."""

    def test_reads_tables_without_qualifying_them(self, dataset_file):
        """The schema card shows bare names, so bare names have to resolve.

        This is why the file is opened as the database rather than attached
        into an in-memory one: `USE` does not survive into `connection.cursor()`.
        """
        sandbox = Sandbox()
        result = sandbox.run(dataset_file, "SELECT count(*) FROM hinval")
        assert result.rows == ((500,),)
        sandbox.close()

    @pytest.mark.parametrize(
        "attack",
        [
            "SELECT * FROM read_csv('/etc/hosts')",
            "SELECT * FROM read_text('/etc/hosts')",
            "SELECT * FROM glob('/etc/*')",
            "SELECT * FROM 'https://example.com/x.parquet'",
            "SELECT * FROM read_parquet('https://example.com/x.parquet')",
        ],
    )
    def test_the_filesystem_and_network_are_gone(self, dataset_file, attack):
        """All of these are honest SELECTs. The guard passes them; the seal does not.

        Which is the argument for having two layers: neither one is sufficient
        and the interesting failure needs both to be wrong at once. `guard` is
        called first here to prove that — every one of these gets past it.

        Two mechanisms do the blocking, and both are asserted rather than one
        being assumed. `disabled_filesystems` refuses anything that reaches for
        a path or a URL directly; a reader that would need httpfs is refused
        earlier still, because `autoload_known_extensions` is off and `INSTALL`
        cannot run. Either is sufficient; the test accepts both so that turning
        one off shows up here rather than in production.
        """
        guard(attack)
        sandbox = Sandbox()
        with pytest.raises(Exception) as caught:
            sandbox.run(dataset_file, attack)
        blocked = f"{type(caught.value).__name__} {caught.value}"
        assert "Permission" in blocked or "requires the extension" in blocked
        sandbox.close()

    def test_the_seal_cannot_be_lifted(self, dataset_file):
        sandbox = Sandbox()
        connection = sandbox.connection(dataset_file)
        with pytest.raises(Exception):
            connection.execute("SET disabled_filesystems = ''")
        sandbox.close()

    def test_writes_are_refused_even_though_the_guard_never_sees_them(self, dataset_file):
        """Defence in depth, stated as a test: the file is opened read-only.

        `DELETE` never reaches the sandbox past `sql.guard`, so this asserts the
        second lock rather than the first — the one that still holds if the
        guard is ever bypassed.
        """
        sandbox = Sandbox()
        connection = sandbox.connection(dataset_file)
        for statement in ("DELETE FROM hinval", "CREATE TABLE z (a INT)", "UPDATE hinval SET query = 'x'"):
            with pytest.raises(Exception):
                connection.execute(statement)
        sandbox.close()

    def test_truncation_is_measured_not_inferred(self, dataset_file):
        """A full page and a cut-off page are different answers.

        One more row than the cap is fetched, so "there were more" is known
        rather than guessed from having filled the page exactly.
        """
        sandbox = Sandbox()
        cut = sandbox.run(dataset_file, "SELECT query_id FROM hinval", max_rows=10)
        assert cut.count == 10 and cut.truncated

        exact = sandbox.run(dataset_file, "SELECT query_id FROM hinval LIMIT 10", max_rows=10)
        assert exact.count == 10 and not exact.truncated
        sandbox.close()

    def test_a_truncated_result_says_so_to_the_model(self, dataset_file):
        sandbox = Sandbox()
        rendered = sandbox.run(dataset_file, "SELECT query_id FROM hinval", max_rows=5).for_model()
        assert '"truncated": true' in rendered
        assert "rather than reporting this as the total" in rendered
        sandbox.close()

    def test_a_missing_file_is_a_message_not_a_crash(self, tmp_path):
        sandbox = Sandbox()
        with pytest.raises(SandboxUnavailable):
            sandbox.run(str(tmp_path / "never-built.duckdb"), "SELECT 1")

    def test_a_runaway_query_is_interrupted_and_the_connection_survives(self, dataset_file):
        """The timeout has to stop the engine, not abandon a thread.

        And the connection must still answer afterwards, or one bad query costs
        every later one on that dataset.
        """
        sandbox = Sandbox()
        with pytest.raises(Exception) as caught:
            sandbox.run(
                dataset_file,
                "SELECT count(*) FROM hinval a, hinval b, hinval c, hinval d",
                timeout_s=1.0,
            )
        assert "Interrupt" in type(caught.value).__name__
        assert sandbox.run(dataset_file, "SELECT count(*) FROM hinval").rows == ((500,),)
        sandbox.close()

    def test_rebuilding_drops_the_open_handle(self, dataset_file):
        sandbox = Sandbox()
        sandbox.run(dataset_file, "SELECT 1")
        sandbox.forget(dataset_file)
        assert sandbox.run(dataset_file, "SELECT count(*) FROM hinval").rows == ((500,),)
        sandbox.close()


# ---- what the SQL writer is handed ---------------------------------------


class TestSchemaCard:
    """The card is the reason generated SQL is right. These are its load-bearing lines."""

    def test_enumerated_values_are_listed(self):
        """The single highest-value line in the card.

        Without it a model writes `WHERE query_type = 'FACT'` — a value that is
        not in the column — and gets an empty result that reads like an answer.
        """
        card = profile_of(
            TableStat(
                name="hinval",
                rows=100,
                columns=(column("query_type", type="VARCHAR", coverage=1.0, distinct=3,
                                values=("DESCRIPTION", "ENTITY", "NUMERIC")),),
            )
        ).schema()
        assert "one of DESCRIPTION, ENTITY, NUMERIC" in card

    def test_a_constant_column_is_named_as_one(self):
        card = profile_of(
            TableStat(
                name="t",
                rows=100,
                columns=(column("source_lang", type="VARCHAR", coverage=1.0, distinct=1,
                                values=("eng_Latn",)),),
            )
        ).schema()
        assert "always eng_Latn" in card

    def test_partial_coverage_is_stated(self):
        card = profile_of(
            TableStat(name="t", rows=100, columns=(column("note", type="VARCHAR", coverage=0.2),))
        ).schema()
        assert "20% non-null" in card

    def test_a_wide_column_is_flagged_as_expensive(self):
        """The lever that decides whether a query returns in milliseconds."""
        card = profile_of(
            TableStat(name="t", rows=100, columns=(column("passages", type="VARCHAR", coverage=1.0,
                                                          avg_bytes=5000),))
        ).schema()
        assert "expensive" in card

    def test_an_omitted_column_is_named_as_unavailable(self):
        """Not silence. A column nobody mentioned is one a model writes SQL against."""
        card = profile_of(
            TableStat(
                name="t",
                rows=100,
                columns=(column("query", type="VARCHAR", coverage=1.0),),
                omitted=(Omitted(name="passages", megabytes=432.0),),
            )
        ).schema()
        assert "NOT AVAILABLE: passages" in card
        assert "432 MB" in card

    def test_sampling_is_stated_in_both_cards(self):
        """"25,000 rows" and "the first 25,000 of 97,941" differ in whether a
        COUNT is a fact."""
        profile = profile_of(TableStat(name="t", rows=25_000, total=97_941))
        assert "aggregates describe the sample" in profile.schema()
        assert "counts and averages are over the sample only" in profile.card()

    def test_a_whole_file_is_not_reported_as_sampled(self):
        """`total == rows` means everything is here, and saying otherwise
        understates an answer that is exact."""
        profile = profile_of(TableStat(name="t", rows=500, total=500))
        assert not profile.observation.sampled
        assert "Sampled" not in profile.card()

    def test_truncation_lands_on_a_line_boundary(self):
        """Cutting mid-line yields a column definition that reads complete and is not."""
        wide = TableStat(
            name="t",
            rows=1,
            columns=tuple(column(f"c{i}", type="VARCHAR", coverage=1.0) for i in range(200)),
        )
        card = profile_of(wide).schema(budget=400)
        assert len(card) <= 400
        assert card.endswith("-- (truncated)")


class TestRoutingCard:
    def test_names_the_dataset_and_its_shape(self):
        card = profile_of(
            TableStat(name="a", rows=10), TableStat(name="b", rows=15),
            understanding=Understanding(title="Indic search queries", summary="One row per query.",
                                        good_for=("counting by language",), not_for=("passage text",)),
        ).card()
        assert "Indic search queries" in card
        assert "2 tables (a, b); 25 rows queryable." in card
        assert "Good for: counting by language" in card
        assert "Not for: passage text" in card

    def test_omitted_columns_are_deduplicated_across_tables(self):
        """Fourteen splits of one corpus share a schema. Listing `passages`
        fourteen times spends the whole card saying one thing."""
        omitted = (Omitted(name="passages", megabytes=400.0),)
        card = profile_of(*(TableStat(name=f"t{i}", rows=10, omitted=omitted) for i in range(14))).card()
        assert card.count("passages") == 1


class TestRoundTrip:
    def test_a_profile_survives_storage(self):
        """It goes to Postgres as JSONB and comes back to build a prompt.

        A field quietly dropped here is a card that stops mentioning a column's
        values, which reads fine and produces empty result sets.
        """
        original = profile_of(
            TableStat(
                name="t",
                rows=100,
                total=900,
                columns=(column("k", type="VARCHAR", coverage=0.9, distinct=2, values=("a", "b"),
                                avg_bytes=12),),
                omitted=(Omitted(name="big", megabytes=99.0),),
            ),
            understanding=Understanding(title="T", topics=("x",)),
        )
        restored = DatasetProfile.from_json(original.to_json())
        assert restored is not None
        assert restored.schema() == original.schema()
        assert restored.card() == original.card()

    def test_an_unreadable_version_is_rebuilt_not_guessed(self):
        """The file can always be re-measured; inventing a missing field cannot
        be undone."""
        assert DatasetProfile.from_json({"version": 999}) is None
        assert DatasetProfile.from_json({}) is None
        assert DatasetProfile.from_json(None) is None


class TestFilterability:
    def test_a_sparse_column_is_not_filterable(self):
        """Same rule and same number as the connector profile: a predicate over
        a column present on 3% of rows drops the other 97% and reads as a
        narrowing."""
        assert not column("x", coverage=0.03).filterable
        assert column("x", coverage=FILTERABLE_COVERAGE).filterable

    def test_a_nested_type_is_recognised(self):
        assert column("p", type="STRUCT(a VARCHAR[])").nested
        assert column("l", type="BIGINT[]").nested
        assert not column("s", type="VARCHAR").nested


# ---- resolving a URL -----------------------------------------------------


class TestSource:
    @pytest.mark.parametrize(
        "url",
        ["not a url", "ftp://host/file.parquet", "https://example.com/data.xlsx",
         "https://huggingface.co/models/org/name", ""],
    )
    def test_refuses_what_it_cannot_read(self, url):
        with pytest.raises(SourceError):
            resolve(url)

    def test_a_direct_parquet_url_needs_no_network(self):
        source = resolve("https://example.com/data/sales.parquet")
        assert source.kind == "file"
        assert source.tables[0].name == "sales"
        assert source.tables[0].format == "parquet"

    def test_compression_is_not_the_format(self):
        assert _suffix("a/b/data.parquet.gz") == ".parquet"
        assert _suffix("data.csv.zst") == ".csv"

    def test_table_names_are_legal_to_write_unquoted(self):
        """They appear in SQL a model writes from the card, so a name needing
        quotes is a name every query has to quote."""
        assert _ident("hin-val 2024") == "hin_val_2024"
        assert _ident("2024") == "t_2024"
        assert _ident("!!!") == "data"

    def test_same_stem_in_two_directories_is_named_by_directory(self):
        """`train/x` and `test/x` become `train_x` and `test_x`, not `x` and `x_1`.

        The directory is the split or the config name — the one piece of
        information that says which is which — and a numeric suffix throws it
        away at exactly the moment a model needs it to choose between them.
        """
        tables = _tables([("train/x.parquet", "u1", "parquet"), ("test/x.parquet", "u2", "parquet")])
        assert [t.name for t in tables] == ["train_x", "test_x"]

    def test_the_slug_is_stable_for_a_repo(self):
        assert _slug("ai4bharat/MSMARCO-XI") == "ai4bharat-msmarco-xi"


class TestMeasurementIsReal:
    """The measurements, taken off a real file rather than constructed.

    Every other test in `TestSchemaCard` builds a `ColumnStat` by hand and
    checks how it renders. That is worth doing and it missed the bug that
    mattered: `avg_bytes` was 0 for every column of every dataset, because the
    width query used `octet_length`, which DuckDB accepts for BLOB and BIT and
    not for VARCHAR. `_scalar` swallowed the binder error as "this type will
    not support that aggregate", the card simply never called anything
    expensive, and a dataset with no expensive columns looks exactly the same.

    So these measure. A hand-built fixture cannot catch a query that does not run.
    """

    def test_a_wide_column_is_actually_measured_as_wide(self, tmp_path):
        import duckdb

        from src.datasets.materialise import Build, Built
        from src.datasets.probe import observe
        from src.datasets.source import Source, Table

        path = tmp_path / "wide.duckdb"
        con = duckdb.connect(str(path))
        con.execute(
            "CREATE TABLE t AS SELECT i AS n, repeat('x', 3000) AS body FROM range(1, 51) s(i)"
        )
        con.close()

        source = Source(slug="w", kind="file", location="w", url="w",
                        tables=(Table(name="t", url="w", format="parquet"),))
        observed = observe(source, Build(path=str(path), tables=[Built("t", 50, 50, "t.parquet")]))

        body = observed.tables[0].column("body")
        narrow = observed.tables[0].column("n")
        assert body.avg_bytes >= 3000, "the width query did not run"
        assert body.wide and not narrow.wide
        assert "expensive" in DatasetProfile(dataset_id="w", observation=observed).schema()

    def test_a_nested_column_is_still_measured(self, tmp_path):
        """Casting to VARCHAR is what puts a struct and a string on one scale."""
        import duckdb

        from src.datasets.materialise import Build, Built
        from src.datasets.probe import observe
        from src.datasets.source import Source, Table

        path = tmp_path / "nested.duckdb"
        con = duckdb.connect(str(path))
        con.execute(
            "CREATE TABLE t AS SELECT {'a': repeat('y', 2000), 'b': i} AS s FROM range(1, 21) r(i)"
        )
        con.close()
        source = Source(slug="n", kind="file", location="n", url="n",
                        tables=(Table(name="t", url="n", format="parquet"),))
        observed = observe(source, Build(path=str(path), tables=[Built("t", 20, 20, "t.parquet")]))
        assert observed.tables[0].column("s").avg_bytes >= 2000


class TestCappedIsSampled:
    """Hitting the row cap is being sampled, even when the total is unknowable.

    A parquet footer gives a denominator. A CSV gives nothing, so `total` stays
    `None` — and a table that stopped dead on 25,000 rows of a 785,000-row CSV
    reported `sampled=False` and put "25,000 rows queryable" in the card with no
    caveat at all. That is the exact dishonesty the rest of this module exists
    to prevent, in the one case where nothing else would hint the number is a
    floor.
    """

    def test_capped_with_no_total_still_counts_as_sampled(self):
        table = TableStat(name="t", rows=25_000, total=None, capped=True)
        assert table.sampled

    def test_the_card_says_so_without_inventing_a_denominator(self):
        profile = profile_of(TableStat(name="t", rows=25_000, total=None, capped=True))
        card = profile.card()
        assert "Sampled" in card
        assert "would not cheaply say how many" in card
        assert "of 25,000" not in card

    def test_a_known_total_still_reads_as_a_fraction(self):
        profile = profile_of(TableStat(name="t", rows=25_000, total=97_941, capped=True))
        assert "the first 25,000 rows of 97,941" in profile.card()

    def test_a_file_that_ended_on_its_own_is_not_sampled(self):
        """Under the cap means the whole file is here, and saying otherwise
        understates an answer that is exact."""
        assert not TableStat(name="t", rows=891, total=891, capped=False).sampled
        assert not TableStat(name="t", rows=891, total=None, capped=False).sampled

    def test_capped_survives_storage(self):
        original = profile_of(TableStat(name="t", rows=100, total=None, capped=True))
        restored = DatasetProfile.from_json(original.to_json())
        assert restored is not None and restored.observation.tables[0].capped
        assert restored.card() == original.card()


class TestTableNaming:
    """The names a model has to type. Measured against what Hugging Face ships."""

    def test_shard_coordinates_are_stripped(self):
        """`train-00000-of-00001.parquet` is how every auto-converted HF dataset
        is stored, and `SELECT ... FROM train_00000_of_00001` is a name a model
        gets wrong."""
        names = [t.name for t in _tables([
            ("plain_text/train-00000-of-00001.parquet", "u", "parquet"),
            ("plain_text/test-00000-of-00001.parquet", "u", "parquet"),
        ])]
        assert names == ["train", "test"]

    def test_a_content_hash_goes_too(self):
        names = [t.name for t in _tables(
            [("data/train-00000-of-00001-a09b74b3ef9c3b56.parquet", "u", "parquet")]
        )]
        assert names == ["train"]

    def test_colliding_splits_are_named_by_their_config(self):
        """gsm8k ships main/test and socratic/test. A numeric suffix would give
        `test` and `test_1`, which says nothing about which is which."""
        names = [t.name for t in _tables([
            ("main/test-00000-of-00001.parquet", "u", "parquet"),
            ("socratic/test-00000-of-00001.parquet", "u", "parquet"),
        ])]
        assert names == ["main_test", "socratic_test"]

    def test_a_true_collision_still_falls_back_to_a_suffix(self):
        """Same stem *and* same directory — the one case the directory cannot
        disambiguate, so the numeric suffix survives as a last resort."""
        names = [t.name for t in _tables([
            ("d/x.parquet", "u", "parquet"), ("d/x.csv", "u", "csv"),
        ])]
        assert names == ["d_x", "d_x_1"]


class TestRemoval:
    """Removing a dataset reports on the row, not on the file behind it.

    `DatasetStore.delete` is three-valued for one reason: `None` for no such
    row, `""` for a row with no file yet, a path for a row with one. Folding
    the middle case into `""` made cancelling a *still-building* dataset — the
    ordinary way somebody undoes a mistyped URL — answer `removed: false` while
    the row was in fact gone, sending them to look for something already
    deleted.
    """

    def test_the_three_values_are_distinct(self):
        from src.datasets.store import DatasetStore

        # Documented as a contract rather than exercised against Postgres here:
        # the distinction lives in the return type, and a test that only ever
        # sees a built dataset cannot tell `""` from `None`.
        assert DatasetStore.delete.__doc__ and "Three-valued" in DatasetStore.delete.__doc__

    def test_a_pending_dataset_reports_as_removed(self, tmp_path):
        class Store:
            """A row that exists and has no file yet."""

            def delete(self, user_id, dataset_id):
                return "" if dataset_id == "pending-one" else None

        from src.core.config import get_settings
        from src.datasets.service import DatasetService

        service = DatasetService.__new__(DatasetService)
        service._store = Store()
        service._settings = get_settings()
        service._sandbox = Sandbox()
        service._cache, service._lock = {}, __import__("threading").Lock()

        assert service.remove("u", "pending-one") is True
        assert service.remove("u", "never-existed") is False
        service._sandbox.close()


class TestConnectorRoute:
    """Attaching a dataset through the Connectors panel.

    The join between two tables with no foreign key between them: a row in
    `connector_accounts` keyed by the slug `dataset`, and a row in
    `agent_datasets` keyed by an id derived from the URL. Disconnecting has to
    find the second from the first, and the only thing it has to go on is the
    URL in `hints`.
    """

    def test_the_id_is_derivable_from_the_url_without_the_network(self):
        """Because it runs inside a DELETE. Re-resolving to learn the id would
        put an HTTP call to Hugging Face inside disconnecting."""
        from src.datasets.source import slug_for

        assert slug_for("https://huggingface.co/datasets/ai4bharat/MSMARCO-XI") == "ai4bharat-msmarco-xi"
        assert (
            slug_for("https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/tree/main/validation")
            == "ai4bharat-msmarco-xi-validation"
        )

    @pytest.mark.parametrize(
        "written",
        [
            "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI",
            "hf://datasets/ai4bharat/MSMARCO-XI",
            "ai4bharat/MSMARCO-XI",
        ],
    )
    def test_every_way_of_writing_one_dataset_gives_one_id(self, written):
        """Otherwise pasting the shorthand attaches a second copy of a dataset
        that is already there, and the quota fills with duplicates."""
        from src.datasets.source import slug_for

        assert slug_for(written) == "ai4bharat-msmarco-xi"

    def test_resolve_and_slug_for_cannot_drift(self):
        """They are one derivation because they must agree: `resolve` names the
        row that is written and `slug_for` names the row that is deleted."""
        from src.datasets.source import resolve, slug_for

        url = "https://example.com/data/sales.parquet"
        assert slug_for(url) == resolve(url).slug

    def test_a_bad_url_fails_the_form_rather_than_the_build(self):
        from src.connectors.spec import ConnectorError
        from src.connectors.registry import verify_dataset

        with pytest.raises(ConnectorError):
            verify_dataset({"url": "https://example.com/data.xlsx"})

    def test_the_url_is_public_and_the_connector_holds_no_secret(self):
        """A dataset URL is an address, not a credential. Marking it secret
        would hide the only field that says which dataset is attached."""
        from src.connectors.registry import get_spec

        spec = get_spec("dataset")
        assert spec.secret_fields == ()
        assert "url" in spec.public_fields
        assert spec.hint_source == "url"


class TestColumnPlanning:
    def test_a_nested_leaf_is_charged_to_its_top_level_column(self):
        """Parquet reports one chunk per leaf. Charging them separately makes a
        453 MB struct look like four cheap columns, and the plan keeps it."""
        assert _top_level("passages, Translated_passages, list, element") == "passages"
        assert _top_level(["passages", "English_passages"]) == "passages"
        assert _top_level("query_id") == "query_id"


# ---- the agent and the tool ----------------------------------------------


class FakeService:
    """Just the surface `DatasetAgent` uses."""

    def __init__(self, path: str, schema: str = "hinval(query_id BIGINT, query_type VARCHAR)"):
        self.sandbox = Sandbox()
        self._row = type("Row", (), {
            "dataset_id": "demo", "local_path": path, "schema_card": schema,
            "status": "ok", "error": "", "card": "Dataset `demo` — Demo", "location": "demo",
        })()

    def queryable(self, user_id, dataset_id):
        return self._row if dataset_id == "demo" else None

    def get(self, user_id, dataset_id):
        return self._row if dataset_id == "demo" else None

    def catalogue(self, user_id):
        return [("demo", "Dataset `demo` — Demo")]

    def list(self, user_id):
        return [self._row]


class TestQueryAgent:
    def _agent(self, monkeypatch, dataset_file, written):
        """An agent whose model calls `run_sql` with `written`, one per turn.

        Two things are faked, and neither of them is the loop: the model, and
        the fact that a model is configured at all. The loop itself — call,
        read the error, correct, stop — is LangChain's and is what these tests
        are actually exercising.

        Through `monkeypatch`, not by assignment: `src.agents.base` holds the
        module-level `chat_model`, so a bare assignment would replace it for
        every other test in the session too.
        """
        import src.agents.base as base
        from src.core.config import get_settings

        service = FakeService(dataset_file)
        model = FakeToolModel(scripts=list(written))
        monkeypatch.setattr(base, "chat_model", lambda *a, **k: model)
        # A key, so `ready` is about this test rather than about whether the
        # checkout running it happens to have a .env.
        settings = get_settings().model_copy(update={"openai_api_key": "test-key"})
        return DatasetAgent(service, settings), service

    def test_a_good_question_returns_rows_and_the_query(self, monkeypatch, dataset_file):
        agent, service = self._agent(monkeypatch, dataset_file, ["SELECT count(*) FROM hinval"])
        answer = agent.ask("u1", "demo", "how many rows?")
        assert answer.ok
        assert answer.result.rows == ((500,),)
        assert answer.sql == "SELECT count(*) FROM hinval"
        assert answer.attempts == 1
        service.sandbox.close()

    def test_a_bad_column_is_repaired_once(self, monkeypatch, dataset_file):
        """DuckDB's binder says "Did you mean ...?". Handing that back fixes a
        misremembered column in one round — replacing it with a generic message
        throws that away."""
        agent, service = self._agent(
            monkeypatch, dataset_file,
            ["SELECT nonexistent FROM hinval", "SELECT query_type FROM hinval LIMIT 1"],
        )
        answer = agent.ask("u1", "demo", "what types are there?")
        assert answer.ok and answer.attempts == 2
        service.sandbox.close()

    def test_repairs_are_bounded(self, monkeypatch, dataset_file):
        """A second failure means the question cannot be answered from these
        columns, and further attempts spend somebody's turn on it."""
        agent, service = self._agent(
            monkeypatch, dataset_file, ["SELECT nope FROM hinval", "SELECT still_nope FROM hinval"],
        )
        answer = agent.ask("u1", "demo", "?")
        assert not answer.ok
        assert answer.sql and answer.error
        service.sandbox.close()

    def test_a_write_from_the_model_is_refused_not_run(self, monkeypatch, dataset_file):
        agent, service = self._agent(
            monkeypatch, dataset_file, ["DELETE FROM hinval", "DELETE FROM hinval"]
        )
        answer = agent.ask("u1", "demo", "delete everything")
        assert not answer.ok
        assert "read-only" in answer.error
        service.sandbox.close()

    def test_a_model_that_ignores_the_tool_still_gets_its_query_run(
        self, monkeypatch, dataset_file
    ):
        """Some providers write the SQL into the message instead of calling the
        tool. The query is right there; running it beats reporting that the
        model held it the wrong way."""
        agent, service = self._agent(monkeypatch, dataset_file, ["!SELECT count(*) FROM hinval"])
        answer = agent.ask("u1", "demo", "how many rows?")
        assert answer.ok and answer.attempts == 1
        service.sandbox.close()

    def test_a_reply_with_no_query_in_it_is_a_failure_not_an_empty_result(
        self, monkeypatch, dataset_file
    ):
        agent, service = self._agent(monkeypatch, dataset_file, ["!I cannot answer that."])
        answer = agent.ask("u1", "demo", "?")
        assert not answer.ok and answer.attempts == 0 and answer.error
        service.sandbox.close()

    def test_the_loop_stops_as_soon_as_a_query_runs(self, monkeypatch, dataset_file):
        """No summarising round trip after the rows come back. The caller
        renders them; a completion narrating them is latency nobody hears."""
        agent, service = self._agent(
            monkeypatch, dataset_file,
            ["SELECT nonexistent FROM hinval", "SELECT count(*) FROM hinval", "SELECT 1"],
        )
        import src.agents.base as base

        answer = agent.ask("u1", "demo", "how many rows?")
        assert answer.ok and answer.attempts == 2
        assert base.chat_model().calls == 2
        service.sandbox.close()

    def test_no_model_is_reported_rather_than_guessed_at(self, dataset_file):
        from src.core.config import get_settings

        settings = get_settings().model_copy(
            update={"openai_api_key": "", "sarvam_api_key": "", "llm_base_url": ""}
        )
        service = FakeService(dataset_file)
        answer = DatasetAgent(service, settings).ask("u1", "demo", "how many rows?")
        assert not answer.ok and "No model is configured" in answer.error
        service.sandbox.close()

    def test_an_unknown_dataset_names_the_ones_that_exist(self):
        """Four states need four different actions. One message for all of them
        sends people to the wrong one."""
        from src.core.config import get_settings

        agent = DatasetAgent(FakeService("/nonexistent"), get_settings())
        answer = agent.ask("u1", "missing", "anything")
        assert not answer.ok
        assert "Attached: demo" in answer.error

    def test_an_interrupt_is_explained_as_one(self):
        assert "too long" in reason(duckdb.InterruptException("interrupted"))


class TestTools:
    def _tools(self, service):
        from src.core.config import get_settings

        return DatasetTools(service, None, get_settings())

    def test_no_datasets_means_no_tool_and_no_cost(self, dataset_file):
        """A user who has never added one pays nothing — not a round trip, not a
        token of schema. The tool pass is buffered, and buffering is what the
        voice path spends its budget avoiding."""
        service = FakeService(dataset_file)
        service.catalogue = lambda user_id: []
        assert self._tools(service).tools_for("u1") == []
        assert self._tools(service).tools_for("") == []
        service.sandbox.close()

    def test_the_ids_are_an_enum_of_what_exists(self, dataset_file):
        """So a hallucinated id is refused by the provider rather than becoming a
        lookup that fails a round trip later."""
        service = FakeService(dataset_file)
        schema = self._tools(service).tools_for("u1")[0]
        assert schema["function"]["name"] == QUERY_TOOL
        assert schema["function"]["parameters"]["properties"]["dataset"]["enum"] == ["demo"]
        service.sandbox.close()

    def test_it_only_claims_its_own_tool(self, dataset_file):
        """`_run_tool` dispatches on this. Claiming a Composio slug would run
        somebody's mail through a SQL agent."""
        service = FakeService(dataset_file)
        tools = self._tools(service)
        assert tools.owns(QUERY_TOOL)
        assert not tools.owns("GMAIL_SEND_EMAIL")
        service.sandbox.close()

    def test_missing_arguments_fail_without_touching_the_database(self, dataset_file):
        service = FakeService(dataset_file)
        result = self._tools(service).execute("u1", QUERY_TOOL, {"dataset": "demo"})
        assert not result.ok and "required" in result.error
        service.sandbox.close()
