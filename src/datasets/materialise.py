"""Pulling a bounded prefix of a remote dataset onto local disk.

This is the unsealed half. It loads httpfs, reaches the network and writes a
file — everything `sandbox.py` exists to forbid — which is exactly why it runs
in a worker thread off every request path, over URLs resolved by `source.py`,
and never over SQL anyone else wrote.

**A prefix, not a sample.** `LIMIT n` on a parquet read stops after the first
row groups, which is why adding a 55 GB dataset takes seconds. `USING SAMPLE`
would be a uniform sample and would read the whole file to get it, over the
network, for every dataset anybody adds. So the honest description of what is
local is *the first n rows*, and it is honest in three places: `TableStat.total`
beside `TableStat.rows`, a line in the card, and a line in the schema block the
SQL writer reads. A prefix is unbiased for "what does a row look like" and
biased for anything ordered — which for a dataset laid out by query id means
aggregates are fine and "earliest record" is not.

**Counting the source is conditional, because it is not always cheap.** Parquet
carries row counts in its footer, so `count(*)` over a remote parquet is a
metadata read. A CSV has no footer and counting means downloading it. So the
total is read only when the pull actually hit the cap — under it, the local
count *is* the total, measured rather than requested.

**Wide columns are planned around, not discovered.** `LIMIT n` only stops early
if the file has row groups to stop at, and this dataset's parquet has exactly
one row group of 97,941 rows — so `SELECT *` reads every column chunk in full
and a 3,000-row pull measured 102 seconds. The same read projected to the
narrow columns measured 0.8 s. The difference is one column: `passages` is
453 MB of a 470 MB file.

Parquet says so in its footer, for 1.8 s, before a byte of data is fetched. So
the pull is planned from `parquet_metadata` — per-column compressed size scaled
by how much of the file the row-group layout forces us to read — and a column
whose projected download exceeds the budget is left out and *named* in the
profile. An agent that knows `passages` is unavailable says so; one that was
never told writes SQL against it.

**The file is replaced, never edited.** A rebuild writes a new database beside
the old one and moves it into place, so a query running against the previous
build finishes against a consistent file instead of reading half of each.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from src.datasets.profile import Omitted
from src.datasets.source import Source, Table

log = logging.getLogger("vec.datasets.materialise")

#: How DuckDB is asked to read each format. `_auto` for the two text formats
#: because the alternative is asking a user to describe their own columns, and
#: getting a type wrong here is a column that arrives as VARCHAR and cannot be
#: compared against a number in generated SQL.
_READERS = {
    "parquet": "read_parquet({url})",
    "csv": "read_csv_auto({url}, sample_size=20000)",
    "json": "read_json_auto({url})",
}


class BuildError(RuntimeError):
    """The pull failed, phrased for whoever added the dataset."""


@dataclass(slots=True)
class Built:
    """One table, as it now exists locally."""

    name: str
    rows: int
    total: int | None
    path: str
    error: str = ""
    #: The pull stopped on the row cap rather than on the end of the file.
    #: Known without asking the source anything, which is what makes the
    #: caveat survive a format that will not be counted.
    capped: bool = False
    #: Columns in the source that the budget excluded. Carried through to the
    #: profile, because a column nobody was told about is a column a model
    #: will write SQL against.
    omitted: tuple[Omitted, ...] = ()


@dataclass(slots=True)
class Build:
    path: str
    tables: list[Built] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    bytes: int = 0
    ms: float = 0.0

    @property
    def ok(self) -> list[Built]:
        return [t for t in self.tables if not t.error]


def build(
    source: Source,
    destination: str | Path,
    *,
    sample_rows: int = 25_000,
    column_budget_bytes: int = 32 * 1024 * 1024,
    table_timeout_s: float = 180.0,
    http_timeout_s: float = 60.0,
) -> Build:
    """Materialise `source` into a DuckDB file at `destination`.

    One failed table does not fail the dataset. A repo of fourteen splits where
    one file is corrupt is still thirteen answerable tables, and refusing all of
    them because of the fourteenth is the kind of strictness that reads as
    breakage. The failure is recorded on the table and surfaces as a note.
    """
    if source.empty:
        raise BuildError("Nothing readable was found at that URL.")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the live file and moved over it, so a sandbox reading the
    # previous build never sees a half-written one.
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)

    started = time.perf_counter()
    report = Build(path=str(target))
    connection = duckdb.connect(str(staging))

    try:
        _prepare(connection, http_timeout_s=http_timeout_s)
        for table in source.tables:
            report.tables.append(
                _one(
                    connection,
                    table,
                    sample_rows=sample_rows,
                    column_budget_bytes=column_budget_bytes,
                    timeout_s=table_timeout_s,
                )
            )
    finally:
        connection.close()

    if not report.ok:
        staging.unlink(missing_ok=True)
        first = next((t.error for t in report.tables if t.error), "no readable tables")
        raise BuildError(f"Nothing could be loaded: {first}")

    failed = [t for t in report.tables if t.error]
    if failed:
        report.notes.append(
            f"{len(failed)} of {len(report.tables)} files could not be loaded: "
            + "; ".join(f"{t.name} ({t.error})" for t in failed[:3])
        )
    if source.truncated:
        report.notes.append(
            f"{source.truncated} further file(s) in the source were not loaded — "
            "this dataset covers only the tables listed above."
        )

    os.replace(staging, target)
    report.bytes = target.stat().st_size
    report.ms = (time.perf_counter() - started) * 1000
    return report


def _prepare(connection: duckdb.DuckDBPyConnection, *, http_timeout_s: float) -> None:
    try:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
    except Exception as error:
        # A deployment with no outbound network to DuckDB's extension
        # repository can still read a local path, so this is a note rather
        # than a failure — the per-table error will say the rest.
        log.info("httpfs unavailable, remote datasets will fail: %s", error)

    for statement in (
        f"SET http_timeout = {int(http_timeout_s * 1000)}",
        "SET http_retries = 2",
        # Parquet over HTTP is range requests; without keep-alive each row
        # group costs a fresh TLS handshake.
        "SET http_keep_alive = true",
    ):
        try:
            connection.execute(statement)
        except Exception as error:
            log.debug("could not apply %r: %s", statement, error)


def _one(
    connection: duckdb.DuckDBPyConnection,
    table: Table,
    *,
    sample_rows: int,
    column_budget_bytes: int,
    timeout_s: float,
) -> Built:
    reader = _READERS.get(table.format)
    if reader is None:
        return Built(table.name, 0, None, table.path, error=f"unsupported format {table.format}")

    source_expr = reader.format(url=_literal(table.url))
    name = _ident(table.name)

    # Only parquet carries a footer to plan from. CSV and JSON are read whole
    # because there is nothing cheap to read first, and they are small enough
    # that the http timeout is the right ceiling rather than a byte budget.
    projection, omitted = "*", ()
    if table.format == "parquet":
        projection, omitted = _plan(
            connection,
            _literal(table.url),
            sample_rows=sample_rows,
            budget_bytes=column_budget_bytes,
            timeout_s=timeout_s,
        )

    try:
        _run(
            connection,
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT {projection} FROM {source_expr} LIMIT {int(sample_rows)}",
            timeout_s=timeout_s,
        )
        rows = int(connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
    except Exception as error:
        return Built(table.name, 0, None, table.path, error=_reason(error))

    # Under the cap, what is local is the whole file and no round trip is
    # needed to know it. At the cap, the source is asked — but only when the
    # answer is a footer read rather than a download.
    total: int | None = rows
    if rows >= sample_rows:
        total = _total(connection, source_expr, table.format, timeout_s=timeout_s)

    return Built(table.name, rows, total, table.path, capped=rows >= sample_rows, omitted=omitted)


def _plan(
    connection: duckdb.DuckDBPyConnection,
    url: str,
    *,
    sample_rows: int,
    budget_bytes: int,
    timeout_s: float,
) -> tuple[str, tuple[Omitted, ...]]:
    """Which columns to pull, decided from the footer before any data moves.

    The scaling factor is the part that is easy to get wrong. A column's cost
    is not `size × sample_rows / total_rows` — parquet is read a row group at a
    time, so asking for 3,000 rows of a file with one 97,941-row group costs
    the *whole* column. So the fraction is over row groups, not rows, which is
    what makes this file's plan differ from an identically-sized file written
    with 1,000 groups.

    Falls back to every column on any failure. A footer that will not parse is
    a reason to read normally, not a reason to refuse — the http timeout is
    still the ceiling.
    """
    try:
        meta = _run(
            connection,
            f"SELECT num_rows, num_row_groups FROM parquet_file_metadata({url})",
            timeout_s=timeout_s,
        )
        total_rows = int(meta[0][0] or 0)
        groups = max(1, int(meta[0][1] or 1))
    except Exception as error:
        log.debug("no parquet footer to plan from: %s", error)
        return "*", ()

    if not total_rows:
        return "*", ()

    rows_per_group = max(1, total_rows // groups)
    groups_needed = min(groups, max(1, -(-int(sample_rows) // rows_per_group)))
    fraction = groups_needed / groups

    try:
        chunks = _run(
            connection,
            f"SELECT path_in_schema, total_compressed_size FROM parquet_metadata({url})",
            timeout_s=timeout_s,
        )
    except Exception as error:
        log.debug("no column sizes to plan from: %s", error)
        return "*", ()

    sizes: dict[str, int] = {}
    for path, size in chunks:
        # A nested column reports one chunk per leaf, under a path whose first
        # element is the top-level column. Cost is charged to that column,
        # because the projection can only include or exclude the whole thing.
        top = _top_level(path)
        if top:
            sizes[top] = sizes.get(top, 0) + int(size or 0)

    if not sizes:
        return "*", ()

    keep: list[str] = []
    dropped: list[Omitted] = []
    for column, size in sizes.items():
        projected = size * fraction
        if projected <= budget_bytes:
            keep.append(column)
        else:
            dropped.append(Omitted(name=column, megabytes=projected / (1024 * 1024)))

    if not keep:
        # Every column is over budget on its own. Reading nothing is not a
        # dataset, so the cheapest one comes back and the rest are announced.
        cheapest = min(sizes, key=lambda c: sizes[c])
        keep = [cheapest]
        dropped = [o for o in dropped if o.name != cheapest]

    if not dropped:
        return "*", ()

    log.info(
        "skipping %d wide column(s): %s",
        len(dropped),
        ", ".join(f"{o.name} ({o.megabytes:.0f} MB)" for o in dropped),
    )
    return ", ".join(_ident(c) for c in keep), tuple(dropped)


def _top_level(path: object) -> str:
    """The outermost column name from a `path_in_schema` entry.

    DuckDB has reported this as both a list and a comma-joined string across
    versions, and the difference is invisible until a nested column is charged
    to a column called `passages, English_passages, list, element`.
    """
    if isinstance(path, (list, tuple)):
        return str(path[0]) if path else ""
    text = str(path or "")
    return text.split(",")[0].strip()


def _total(
    connection: duckdb.DuckDBPyConnection,
    source_expr: str,
    fmt: str,
    *,
    timeout_s: float,
) -> int | None:
    """How many rows the source holds, when asking is a metadata read.

    None rather than a guess. A `None` total renders as "at least n rows" and a
    wrong one renders as a number, and the second is the failure this whole
    module is careful about.
    """
    if fmt != "parquet":
        return None
    try:
        row = _run(connection, f"SELECT count(*) FROM {source_expr}", timeout_s=timeout_s)
        return int(row[0][0]) if row else None
    except Exception as error:
        log.debug("could not count the source: %s", error)
        return None


def _run(connection: duckdb.DuckDBPyConnection, sql: str, *, timeout_s: float):
    """Execute with a ceiling, interrupting the engine rather than the thread.

    Same reasoning as `sandbox.run`: a network read that has stalled must stop
    inside DuckDB, or the worker returns and the query keeps pulling bytes
    against a dataset nobody is waiting for any more.
    """
    cursor = connection.cursor()
    timer = threading.Timer(timeout_s, _interrupt, args=(cursor,))
    try:
        timer.start()
        return cursor.execute(sql).fetchall()
    finally:
        timer.cancel()
        cursor.close()


def _interrupt(cursor: duckdb.DuckDBPyConnection) -> None:
    try:
        cursor.interrupt()
    except Exception:
        pass


def _reason(error: Exception) -> str:
    """The first line of a DuckDB error, which is the part that names the cause."""
    text = str(error).strip().splitlines()
    first = text[0] if text else type(error).__name__
    if "InterruptException" in type(error).__name__ or "interrupt" in first.lower():
        return "timed out"
    return first[:180]


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
