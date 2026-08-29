"""What this app has measured about a dataset, and the two cards it renders.

The layering is `src/connectors/profile.py`'s, deliberately, because the
failure it was built against is the same one: a connector that says
`filters=True` for every index on earth is a store described by hope, and a
dataset described by its column *names* is a table described by hope. A model
handed `SELECT ... WHERE query_type = 'FACT'` has written valid SQL against a
column whose five values it never saw, and gets back zero rows that read
exactly like an honest empty result.

    Observation     what DuckDB measured off the materialised sample. Counts,
                    types, coverage, cardinality, byte width. No model in it.
    Understanding   what one model call made of a handful of rows. Prose and
                    routing hints. Allowed to be wrong.
    DatasetProfile  both, versioned, plus the sampling facts that decide how an
                    answer must be phrased.

Only `Observation` is trusted for anything that decides SQL. `Understanding`
steers which dataset gets asked, and nothing else.

**Two cards, because two readers.** `card()` goes in a system prompt on every
turn and answers "is this dataset worth asking" in about 700 characters.
`schema()` is handed only to the SQL writer, costs nothing on turns that never
query, and carries the thing that actually makes generated SQL correct — the
enumerable values, the coverage, and which columns are expensive to touch.

**The sample is stated everywhere it is served.** A prefix of 25,000 rows out
of 97,941 gives exact answers to `WHERE` and approximate ones to `COUNT`, and
the difference is invisible in a result set. Both cards say so, `sampled` is on
the wire, and `docs/18-datasets.md` says why it is a prefix and not a sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

#: Schema version of the stored JSON. A profile written by an older build is
#: re-measured rather than migrated: it describes a local file this app wrote
#: and can write again, so recomputing beats inventing missing fields.
VERSION = 1

#: Below this share of non-null rows a column is not something to filter on.
#: Same number and same reason as the connector profile: a predicate over a
#: column present on 3% of rows drops the other 97% and reads as a narrowing.
FILTERABLE_COVERAGE = 0.80

#: Past this many distinct values a column is free text, and listing its values
#: would put a sample of somebody's data in a system prompt.
MAX_DISTINCT = 25

#: A column wider than this is the one that decides whether a query is fast.
#: On MSMARCO-XI `passages` is ~5 KB a row against ~50 B for `query`, and
#: touching it is the difference between milliseconds and minutes.
WIDE_COLUMN_BYTES = 512

#: What the schema card may quote back per column. Small: enough to show the
#: shape of a value, not enough to be a copy of the column.
EXAMPLE_CHARS = 64
MAX_EXAMPLES = 4

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"  # measured, but something was unreadable
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"


# ---- the measurement -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnStat:
    """One column, measured rather than declared.

    `coverage` and `distinct` are the two that change what SQL gets written.
    A type says `VARCHAR`; only the measurement says it holds five values and
    which five, which is the difference between a `WHERE` that matches and one
    that silently matches nothing.
    """

    name: str
    type: str = ""
    coverage: float = 0.0
    distinct: int | None = None
    values: tuple[str, ...] = ()      # every value, when there are few enough
    examples: tuple[str, ...] = ()    # a few, when there are not
    avg_bytes: int = 0
    minimum: str = ""
    maximum: str = ""

    @property
    def filterable(self) -> bool:
        return self.coverage >= FILTERABLE_COVERAGE

    @property
    def constant(self) -> bool:
        """One value across the sample — carried, and worth nothing to a filter."""
        return self.distinct == 1 and self.coverage > 0

    @property
    def enumerable(self) -> bool:
        return bool(self.values)

    @property
    def wide(self) -> bool:
        return self.avg_bytes >= WIDE_COLUMN_BYTES

    @property
    def nested(self) -> bool:
        head = self.type.upper()
        return head.startswith("STRUCT") or head.startswith("MAP") or head.endswith("[]")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "coverage": round(self.coverage, 4),
            "distinct": self.distinct,
            "values": list(self.values),
            "examples": list(self.examples),
            "avg_bytes": self.avg_bytes,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "ColumnStat":
        return cls(
            name=str(blob.get("name") or ""),
            type=str(blob.get("type") or ""),
            coverage=float(blob.get("coverage") or 0.0),
            distinct=blob.get("distinct"),
            values=tuple(str(v) for v in (blob.get("values") or [])),
            examples=tuple(str(v) for v in (blob.get("examples") or [])),
            avg_bytes=int(blob.get("avg_bytes") or 0),
            minimum=str(blob.get("minimum") or ""),
            maximum=str(blob.get("maximum") or ""),
        )


@dataclass(frozen=True, slots=True)
class Omitted:
    """A column the source has and this app did not pull, and what it cost.

    Recorded rather than dropped, and it is the difference between an agent
    that knows a column is out of reach and one that invents a reason its
    query returned nothing. On MSMARCO-XI `passages` is 453 MB of a 470 MB
    file — 96% of the download for one column — so omitting it turns a
    102-second pull into a sub-second one. The schema card names it, so a
    model asked about passage text says it cannot see them rather than
    writing SQL against a column that is not there.
    """

    name: str
    megabytes: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "megabytes": round(self.megabytes, 2)}

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Omitted":
        return cls(name=str(blob.get("name") or ""), megabytes=float(blob.get("megabytes") or 0.0))


@dataclass(frozen=True, slots=True)
class TableStat:
    """One materialised table: how much of it is here, and out of how much.

    `rows` and `total` being separate is the whole honesty story. `rows` is
    what a query can actually see; `total` is what the dataset holds, when the
    source would say cheaply. Collapsing them into one number is how a
    `COUNT(*)` over a prefix gets reported as the size of a corpus.
    """

    name: str
    rows: int = 0
    total: int | None = None
    columns: tuple[ColumnStat, ...] = ()
    #: Columns present in the source and deliberately not materialised. See
    #: `Omitted` — announced everywhere the schema is, never silently absent.
    omitted: tuple[Omitted, ...] = ()
    bytes: int = 0
    path: str = ""
    #: The pull stopped because it hit `dataset_sample_rows`, not because the
    #: file ended. Separate from `total` because they answer different halves
    #: of the same question and only one of them is always knowable.
    capped: bool = False

    @property
    def sampled(self) -> bool:
        """Is what is here less than what is there?

        `capped` first, and that ordering is the bug this property was rewritten
        for. A CSV has no footer, so `total` stays `None` — and a table that
        stopped dead on 25,000 of a 785,000-row file reported `sampled=False`,
        which put "25,000 rows queryable" in the card with no caveat at all.
        Unknown-total is exactly the case where the caveat matters most: there
        is nothing else anywhere that would hint the number is a floor.
        """
        if self.capped:
            return True
        return self.total is not None and self.rows < self.total

    def column(self, name: str) -> ColumnStat | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "total": self.total,
            "bytes": self.bytes,
            "path": self.path,
            "capped": self.capped,
            "columns": [c.to_json() for c in self.columns],
            "omitted": [o.to_json() for o in self.omitted],
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "TableStat":
        return cls(
            name=str(blob.get("name") or ""),
            rows=int(blob.get("rows") or 0),
            total=blob.get("total"),
            bytes=int(blob.get("bytes") or 0),
            path=str(blob.get("path") or ""),
            capped=bool(blob.get("capped")),
            columns=tuple(ColumnStat.from_json(c) for c in (blob.get("columns") or [])),
            omitted=tuple(Omitted.from_json(o) for o in (blob.get("omitted") or [])),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """What was measured. Everything here was counted or is None."""

    source: str = ""          # "hf" | "file"
    location: str = ""
    url: str = ""
    tables: tuple[TableStat, ...] = ()
    #: Files the source held beyond the table cap. Stated, never swallowed.
    truncated: int = 0
    #: A few whole rows, rendered, for the model that writes the summary.
    excerpts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    measured_at: str = ""

    @property
    def rows(self) -> int:
        return sum(t.rows for t in self.tables)

    @property
    def bytes(self) -> int:
        return sum(t.bytes for t in self.tables)

    @property
    def sampled(self) -> bool:
        return any(t.sampled for t in self.tables)

    @property
    def sized(self) -> str:
        """How much of the source is here, phrased for whichever half is known.

        A parquet footer gives the denominator; a CSV gives nothing. "the first
        25,000 rows of 97,941" and "the first 25,000 rows of an unknown larger
        number" are both honest, and inventing the second number to match the
        shape of the first is the failure this whole module is careful about.
        """
        known = [t for t in self.tables if t.total is not None]
        if known and len(known) == len(self.tables):
            return f"the first {self.rows:,} rows of {sum(t.total or 0 for t in self.tables):,}"
        return (
            f"the first {self.rows:,} rows — the source holds more and would not "
            "cheaply say how many"
        )

    @property
    def omitted_names(self) -> tuple[str, ...]:
        """Every column left behind, deduplicated across tables.

        Deduplicated because the fourteen splits of a translated corpus share
        one schema, and listing `passages` fourteen times spends the card's
        whole budget saying one thing.
        """
        seen: dict[str, None] = {}
        for table in self.tables:
            for omitted in table.omitted:
                seen.setdefault(omitted.name, None)
        return tuple(seen)

    def table(self, name: str) -> TableStat | None:
        lowered = name.lower()
        return next((t for t in self.tables if t.name.lower() == lowered), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "location": self.location,
            "url": self.url,
            "truncated": self.truncated,
            "excerpts": list(self.excerpts),
            "notes": list(self.notes),
            "measured_at": self.measured_at,
            "tables": [t.to_json() for t in self.tables],
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Observation":
        return cls(
            source=str(blob.get("source") or ""),
            location=str(blob.get("location") or ""),
            url=str(blob.get("url") or ""),
            truncated=int(blob.get("truncated") or 0),
            excerpts=tuple(str(v) for v in (blob.get("excerpts") or [])),
            notes=tuple(str(v) for v in (blob.get("notes") or [])),
            measured_at=str(blob.get("measured_at") or ""),
            tables=tuple(TableStat.from_json(t) for t in (blob.get("tables") or [])),
        )


# ---- what a model made of it ---------------------------------------------


@dataclass(frozen=True, slots=True)
class Understanding:
    """The prose half. Steers which dataset is asked; decides nothing."""

    title: str = ""
    summary: str = ""
    topics: tuple[str, ...] = ()
    good_for: tuple[str, ...] = ()
    not_for: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.title or self.summary or self.topics)

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "topics": list(self.topics),
            "good_for": list(self.good_for),
            "not_for": list(self.not_for),
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Understanding":
        return cls(
            title=str(blob.get("title") or ""),
            summary=str(blob.get("summary") or ""),
            topics=tuple(str(v) for v in (blob.get("topics") or [])),
            good_for=tuple(str(v) for v in (blob.get("good_for") or [])),
            not_for=tuple(str(v) for v in (blob.get("not_for") or [])),
        )


# ---- both, and how they read ---------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    dataset_id: str = ""
    observation: Observation = field(default_factory=Observation)
    understanding: Understanding | None = None
    version: int = VERSION

    # ---- the routing card, for every turn's system prompt ----------------

    def card(self, *, budget: int = 700) -> str:
        """Is this dataset worth asking? In a few lines, cheap enough to always send.

        Deliberately not a schema. A turn that never queries anything should
        not pay for a column list, and a model deciding *whether* to query does
        not need one — it needs the subject, the size, and the honest note that
        the numbers are over a prefix.
        """
        observation = self.observation
        understanding = self.understanding
        title = (understanding.title if understanding else "") or observation.location or self.dataset_id

        # Ordered by what must survive `_fit`, which trims from the end. The
        # caveats come before the routing hints deliberately: a card cut short
        # after "good for counting by language" and before "these are the first
        # 75,000 of 293,823 rows" is a card that produces confident wrong
        # totals, which is worse than one that produces a missed routing
        # opportunity. Hints are the nice half; the caveats are the honest one.
        lines = [f"Dataset `{self.dataset_id}` — {title}"]
        if understanding and understanding.summary:
            lines.append(understanding.summary)

        shape = f"{len(observation.tables)} table{'' if len(observation.tables) == 1 else 's'}"
        names = ", ".join(t.name for t in observation.tables[:6])
        if names:
            shape += f" ({names}{', …' if len(observation.tables) > 6 else ''})"
        lines.append(f"{shape}; {observation.rows:,} rows queryable.")

        if observation.sampled:
            lines.append(
                f"Sampled: {observation.sized}. "
                "Filters and examples are exact; counts and averages are over the sample only."
            )
        if observation.truncated:
            lines.append(f"{observation.truncated} further file(s) in the source were not loaded.")
        if (missing := observation.omitted_names):
            lines.append(
                "Columns present in the source but not loaded, and so not queryable: "
                + ", ".join(missing)
                + "."
            )

        # Three apiece. The model is asked for up to five, and the tail of that
        # list is reliably the weakest — a card that spends 250 characters on
        # two more marginal hints crowds out the caveats above it.
        if understanding and understanding.good_for:
            lines.append("Good for: " + "; ".join(understanding.good_for[:3]))
        if understanding and understanding.not_for:
            lines.append("Not for: " + "; ".join(understanding.not_for[:3]))

        return _fit("\n".join(lines), budget)

    # ---- the schema card, for the SQL writer only ------------------------

    def schema(self, *, budget: int = 6000) -> str:
        """Everything needed to write correct SQL, and nothing needed to decide to.

        Three things beyond the column list, each of which is a class of wrong
        query this prevents:

          enumerated values   stops a `WHERE` against a value that is not there
          coverage            stops a filter on a column that is mostly null
          byte width          stops a `SELECT *` over a 5 KB column when two
                              50-byte ones were the question
        """
        observation = self.observation
        blocks: list[str] = []

        if observation.sampled:
            blocks.append(
                f"-- These tables hold {observation.sized}. "
                "Row-level answers are exact; aggregates describe the sample, not the dataset."
            )

        for table in observation.tables:
            head = f"{table.name}  -- {table.rows:,} rows"
            if table.sampled and table.total:
                head += f" of {table.total:,}"
            if table.path:
                head += f", from {table.path}"
            blocks.append(head)
            for column in table.columns:
                blocks.append("  " + _column_line(column))
            for omitted in table.omitted:
                # Stated inside the table block rather than as a footnote,
                # because the moment it is read separately from the column list
                # it stops preventing the query it exists to prevent.
                blocks.append(
                    f"  -- NOT AVAILABLE: {omitted.name} exists in the source but was not "
                    f"loaded ({omitted.megabytes:,.0f} MB). It cannot be selected or filtered on."
                )
            blocks.append("")

        for note in observation.notes:
            blocks.append(f"-- {note}")

        return _fit("\n".join(blocks).strip(), budget)

    # ---- storage ---------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "observation": self.observation.to_json(),
            "understanding": self.understanding.to_json() if self.understanding else None,
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any] | None) -> "DatasetProfile | None":
        """None for a blob this build cannot read — the caller re-measures.

        Re-measuring is cheap and correct; migrating a shape whose meaning
        changed is neither.
        """
        if not blob or int(blob.get("version") or 0) != VERSION:
            return None
        understanding = blob.get("understanding")
        return cls(
            dataset_id=str(blob.get("dataset_id") or ""),
            observation=Observation.from_json(blob.get("observation") or {}),
            understanding=Understanding.from_json(understanding) if understanding else None,
            version=VERSION,
        )


def _column_line(column: ColumnStat) -> str:
    """One column as the SQL writer needs to read it."""
    parts = [f"{column.name} {column.type or 'UNKNOWN'}"]
    notes: list[str] = []

    if column.coverage < 0.999:
        notes.append(f"{column.coverage:.0%} non-null")
    if column.constant:
        notes.append(f"always {column.values[0] if column.values else 'one value'}")
    elif column.enumerable:
        notes.append("one of " + ", ".join(column.values))
    elif column.minimum and column.maximum:
        notes.append(f"{column.minimum} … {column.maximum}")
    elif column.examples:
        notes.append("e.g. " + " | ".join(column.examples[:2]))

    if column.distinct is not None and not column.enumerable and not column.constant:
        notes.append(f"~{column.distinct:,} distinct")
    if column.wide:
        notes.append(f"~{column.avg_bytes:,} bytes/row — expensive, select only when asked")

    return parts[0] + ("  -- " + "; ".join(notes) if notes else "")


def _fit(text: str, budget: int) -> str:
    """Trim on a line boundary, and say that it was trimmed.

    Cutting mid-line in a schema block produces a column definition that reads
    complete and is not, which is exactly the kind of quiet wrongness the rest
    of this module exists to avoid.
    """
    if budget <= 0 or len(text) <= budget:
        return text

    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > budget - 24:
            break
        kept.append(line)
        used += len(line) + 1
    kept.append("-- (truncated)")
    return "\n".join(kept)


def excerpt_row(values: Sequence[tuple[str, Any]], *, limit: int = 400) -> str:
    """One sampled row, rendered for the model that names the dataset."""
    parts: list[str] = []
    for name, value in values:
        text = " ".join(str(value if value is not None else "").split())
        if not text:
            continue
        parts.append(f"{name}={text[:120]}")
    return "; ".join(parts)[:limit]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
