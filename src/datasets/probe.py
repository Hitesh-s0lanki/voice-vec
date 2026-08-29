"""Measuring the materialised file: the numbers that decide what SQL is correct.

A schema is not knowledge. `query_type VARCHAR` tells a model the column
exists and lets it write `WHERE query_type = 'FACT'` — valid SQL, correct
syntax, five values in the column, none of them that one, zero rows back, and a
zero-row result is indistinguishable from an honest empty answer. What closes
that gap is not a better prompt, it is the measurement: *this column holds
DESCRIPTION, NUMERIC, ENTITY, PERSON, LOCATION*.

Three measurements, and each removes a distinct class of wrong query:

    coverage    what share of rows have a value. A filter on a column that is
                20% populated silently drops the other 80% and reads as a
                narrowing. Same number, same reason as `connectors/profile.py`.
    distinct    few enough to enumerate, and the values go in the schema card.
                Too many and it is free text — listing it would put somebody's
                data in a prompt, and `~4,812 distinct` is what a model needs
                anyway.
    avg_bytes   the one nobody writes down. On MSMARCO-XI `passages` is ~5 KB a
                row against ~50 B for `query`, and it is the difference between
                a query that returns in milliseconds and one that does not.

**Every column is measured independently, and a column that will not measure is
skipped rather than failing the table.** Aggregates over a nested type are not
uniformly supported and never will be — a `STRUCT` of three lists is legal, is
in this dataset, and `approx_count_distinct` over it is not something to bet a
whole profile on. This runs against a local file, so the cost of one query per
column is microseconds and the robustness is free.
"""

from __future__ import annotations

import logging

import duckdb

from src.datasets.materialise import Build, Built
from src.datasets.profile import (
    EXAMPLE_CHARS,
    MAX_DISTINCT,
    MAX_EXAMPLES,
    ColumnStat,
    Observation,
    TableStat,
    excerpt_row,
    now,
)
from src.datasets.source import Source

log = logging.getLogger("vec.datasets.probe")

#: Types whose ends are worth stating. A date range and a numeric range are
#: both things a model gets wrong by assuming — "the data covers 2024" when it
#: stops in March is a confident wrong answer.
_ORDERED = ("INT", "BIGINT", "HUGEINT", "SMALLINT", "TINYINT", "DECIMAL", "DOUBLE",
            "FLOAT", "REAL", "DATE", "TIMESTAMP", "TIME", "UBIGINT", "UINTEGER")

#: How wide a row of this column is, in bytes.
#:
#: `strlen`, not `octet_length`. This is the one measurement that decides
#: whether a query is fast, and it was silently returning nothing: DuckDB's
#: `octet_length` takes BLOB and BIT only, so `octet_length(CAST(x AS VARCHAR))`
#: raised a binder error on *every* column, `_scalar` swallowed it as "this
#: type will not support that", and `avg_bytes` was 0 everywhere. Nothing
#: looked wrong — the schema card simply never said a column was expensive,
#: which is indistinguishable from a dataset that has no expensive columns.
#:
#: The cast to VARCHAR stays, so a struct and a string are measured on the same
#: scale: the question is what a row of this column costs a result set, not how
#: it is stored.
_WIDTH = "SELECT avg(strlen(CAST({col} AS VARCHAR))) FROM {src}"

#: How many whole rows go to the model that names the dataset. Small, and only
#: from narrow columns: the point is to show what the data is *about*, and a
#: 5 KB passage crowds out the other twelve fields without adding a topic.
MAX_EXCERPTS = 4
EXCERPT_COLUMN_BYTES = 400


def observe(source: Source, built: Build) -> Observation:
    """The materialised file, measured. Never raises — a table that will not
    measure lands as a note.

    A degraded observation is still a usable one: a dataset whose fourth table
    is unreadable can answer everything about the other three, and the note is
    what stops it claiming otherwise.
    """
    connection = duckdb.connect(built.path, read_only=True)
    tables: list[TableStat] = []
    notes = list(built.notes)

    try:
        for entry in built.ok:
            try:
                tables.append(_table(connection, entry))
            except Exception as error:
                log.warning("could not measure %s: %s", entry.name, error)
                notes.append(f"{entry.name} could not be measured and has no column detail.")
                tables.append(
                    TableStat(
                        name=entry.name,
                        rows=entry.rows,
                        total=entry.total,
                        omitted=entry.omitted,
                        capped=entry.capped,
                        path=entry.path,
                    )
                )

        excerpts = _excerpts(connection, tables)
    finally:
        connection.close()

    return Observation(
        source=source.kind,
        location=source.location,
        url=source.url,
        tables=tuple(tables),
        truncated=source.truncated,
        excerpts=excerpts,
        notes=tuple(notes),
        measured_at=now(),
    )


def _table(connection: duckdb.DuckDBPyConnection, entry: Built) -> TableStat:
    described = connection.execute(f"DESCRIBE {_ident(entry.name)}").fetchall()
    columns = [
        _column(connection, entry.name, str(row[0]), str(row[1]), entry.rows) for row in described
    ]

    return TableStat(
        name=entry.name,
        rows=entry.rows,
        total=entry.total,
        columns=tuple(columns),
        omitted=entry.omitted,
        capped=entry.capped,
        # Derived rather than read off the file: `duckdb_tables()` reports the
        # whole database's blocks, and what a query planner and a prompt both
        # care about is the width of a row, not the size of the file.
        bytes=sum(c.avg_bytes for c in columns) * entry.rows,
        path=entry.path,
    )


def _column(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    name: str,
    type_name: str,
    rows: int,
) -> ColumnStat:
    quoted = _ident(name)
    source = _ident(table)
    upper = type_name.upper()
    nested = upper.startswith(("STRUCT", "MAP", "UNION")) or upper.endswith("[]")

    present = _scalar(connection, f"SELECT count({quoted}) FROM {source}") or 0
    coverage = (present / rows) if rows else 0.0
    avg_bytes = int(_scalar(connection, _WIDTH.format(col=quoted, src=source)) or 0)

    distinct: int | None = None
    values: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    minimum = maximum = ""

    if not nested:
        distinct = _scalar(connection, f"SELECT approx_count_distinct({quoted}) FROM {source}")
        distinct = int(distinct) if distinct is not None else None

        if distinct is not None and distinct <= MAX_DISTINCT:
            # Exact, not approximate: these values go into a prompt as the set
            # a WHERE may match, and `approx_count_distinct` is allowed to be
            # off by one — which would silently drop a real value from the list.
            found = connection.execute(
                f"SELECT DISTINCT {quoted} FROM {source} WHERE {quoted} IS NOT NULL "
                f"ORDER BY 1 LIMIT {MAX_DISTINCT + 1}"
            ).fetchall()
            if len(found) <= MAX_DISTINCT:
                values = tuple(_short(row[0]) for row in found)
                distinct = len(found)

        if not values and any(marker in upper for marker in _ORDERED):
            bounds = connection.execute(
                f"SELECT min({quoted}), max({quoted}) FROM {source}"
            ).fetchone()
            minimum, maximum = _short(bounds[0]), _short(bounds[1])

    if not values and not minimum:
        # DISTINCT because a constant nested column — `meta` on this dataset is
        # the same decode config on every row — otherwise spends the card's
        # budget printing one value several times.
        found = connection.execute(
            f"SELECT DISTINCT {quoted} FROM {source} WHERE {quoted} IS NOT NULL "
            f"LIMIT {MAX_EXAMPLES}"
        ).fetchall()
        examples = tuple(dict.fromkeys(_short(row[0]) for row in found))

    return ColumnStat(
        name=name,
        type=type_name,
        coverage=coverage,
        distinct=distinct,
        values=values,
        examples=examples,
        avg_bytes=avg_bytes,
        minimum=minimum,
        maximum=maximum,
    )


def _excerpts(connection: duckdb.DuckDBPyConnection, tables: list[TableStat]) -> tuple[str, ...]:
    """A few whole rows, narrow columns only, for the model that names this.

    From the first table alone. Rows from four different splits of the same
    corpus describe the same subject four times, and the budget is better spent
    on four rows of one than one row of four.
    """
    table = next((t for t in tables if t.columns and t.rows), None)
    if table is None:
        return ()

    narrow = [c for c in table.columns if not c.nested and c.avg_bytes <= EXCERPT_COLUMN_BYTES]
    if not narrow:
        return ()

    projection = ", ".join(_ident(c.name) for c in narrow[:10])
    try:
        rows = connection.execute(
            f"SELECT {projection} FROM {_ident(table.name)} LIMIT {MAX_EXCERPTS}"
        ).fetchall()
    except Exception as error:
        log.debug("could not read excerpts: %s", error)
        return ()

    names = [c.name for c in narrow[:10]]
    return tuple(excerpt_row(list(zip(names, row))) for row in rows)


def _scalar(connection: duckdb.DuckDBPyConnection, sql: str):
    """One aggregate, or None when the type will not support it.

    None rather than zero. A column whose distinct count could not be measured
    and one that holds a single value are different facts, and only the second
    is worth telling a model.
    """
    try:
        row = connection.execute(sql).fetchone()
    except Exception as error:
        log.debug("skipped an aggregate: %s", str(error).splitlines()[0][:120])
        return None
    return row[0] if row else None


def _short(value: object) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text[:EXAMPLE_CHARS]


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
