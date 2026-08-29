"""Which column is which, so the search SQL is not tied to one schema.

`VectorStore` grew up as the only store there was, so its queries named this
app's own columns — `chunk_key`, `strategy`, `text`, `meta` — as literals. That
was right while the only database it ran against was one this app had ingested
itself. It stopped being right the moment a user could connect their own
Postgres, because it made "connect your own database" mean "connect a copy of
ours": a perfectly good pgvector table holding `id`, `chunk_text` and `book_id`
was rejected at the form for not being a schema its owner had never heard of.
That ingest is gone now and every search runs against somebody's own table, so
the map is not a compatibility layer any more — it is the only thing that says
what the columns are called.

So the column names become data. A `ColumnMap` says which column plays each
role; the default reproduces this app's schema exactly, and
`src/connectors/registry.py` discovers one for a connected store at verify time.

**The default must generate the query it replaced, character for character.**
That is what `tests/test_columns.py` pins. A store this app ingests into is the
hot path and the measured one, and a refactor that quietly changed its SQL
would be a latency regression nobody could attribute.

**An absent column is a missing capability, never a substituted one.** No
`strategy` means the strategy predicate is *dropped*, not defaulted to
something that matches everything — the difference is that a dropped predicate
is honest and a default one is a filter the caller thinks it applied.
`Capabilities` is derived from the same map, so the ladder is told before it
asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Mapping

#: A column name this module is willing to interpolate. Names arrive from
#: `pg_attribute` rather than from a form, but they reach a query as text
#: rather than as a bound parameter — Postgres takes no parameter in that
#: position — so they are validated instead of trusted. A column that cannot
#: satisfy this is simply not mapped, which costs the capability it carried and
#: nothing else.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")


def quote_table(table: str) -> str:
    """`schema.table`, each part quoted, so a mixed-case name survives.

    Postgres folds an unquoted identifier to lower case, so a table created as
    `"MyDocs"` cannot be found by writing `MyDocs` in a query — and every
    connected store is somebody else's naming convention. Quoting also makes
    the qualified form work, which is how a table outside the search path is
    reached at all.

    Embedded quotes are doubled rather than rejected: this is the one
    identifier that comes from a form field, and it has already been checked
    against the catalogue by the time it gets here.
    """
    parts = [part for part in (table or "").split(".") if part]
    return ".".join('"' + part.replace('"', '""') + '"' for part in parts) or '""'


def safe(name: str) -> str:
    """The name if it can be interpolated, otherwise empty — never an exception.

    Empty is the same answer as "this store does not have that column", and the
    two collapse deliberately: both mean the capability is unavailable, and a
    store with an exotic column name should lose one channel rather than fail
    to connect.
    """
    name = (name or "").strip()
    return name if _IDENT.match(name) else ""


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Which column fills each role. Empty string means the store has none.

    Defaults are this app's own schema, so `ColumnMap()` is the deployment
    store and every existing call site keeps its behaviour without passing
    anything.
    """

    id: str = "chunk_key"
    text: str = "text"
    embedding: str = "embedding"
    meta: str = "meta"
    strategy: str = "strategy"
    language: str = "language"
    #: The cross-lingual pair — the English original and its own vector and
    #: tsvector. Only an index this app ingested has them.
    english: str = "english"
    embedding_en: str = "embedding_en"
    tsv: str = "tsv"
    tsv_en: str = "tsv_en"
    #: Extra columns worth carrying into `Hit.payload` so a hit can be cited —
    #: the document id of a chunked corpus, typically. Only used when the store
    #: has no `meta` column of its own.
    payload: tuple[str, ...] = ()
    #: Which distance the index was built for, read off its opclass. This is
    #: not cosmetic: querying a `vector_l2_ops` index with `<=>` does not use
    #: the index at all. It sequential-scans, silently, and shows up only as
    #: latency nobody can explain.
    metric: str = "cosine"

    @classmethod
    def from_mapping(cls, values: Mapping[str, str] | None) -> "ColumnMap":
        """Build from the flat string map a connector stores.

        **A role the map does not mention is absent, not defaulted.** The
        dataclass defaults are the historical column names, which is right
        for a bare `ColumnMap()` and exactly wrong here: a
        connected table with no `strategy` column would inherit the name
        `strategy` and generate a query against a column that is not there,
        which is the failure the whole map exists to prevent.

        An empty mapping is the one case that does fall back to the defaults:
        it means an account attached before mapping existed, and those were
        verified against this app's schema, so that is what they are.

        Unknown keys are ignored and unmappable names dropped, so a map written
        by an older version, or naming a column since renamed, degrades to
        fewer capabilities rather than to a broken query.
        """
        if not values:
            return cls()

        known = [
            name for name in cls.__dataclass_fields__ if name not in ("payload", "metric")
        ]
        fields = {name: "" for name in known}
        fields.update(
            {key: safe(str(val)) for key, val in values.items() if key in fields}
        )
        extra = tuple(
            cleaned
            for raw in str(values.get("payload") or "").split(",")
            if (cleaned := safe(raw))
        )
        # `metric` is a keyword from a fixed set, not a column name, so it is
        # validated against that set rather than as an identifier.
        metric = str(values.get("metric") or "cosine")
        return cls(
            **fields,
            payload=extra,
            metric=metric if metric in METRICS else "cosine",
        )

    def to_mapping(self) -> dict[str, str]:
        """Flat strings, for the credential blob a connector seals."""
        out = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("payload", "metric") and getattr(self, name)
        }
        if self.payload:
            out["payload"] = ",".join(self.payload)
        out["metric"] = self.metric
        return out

    # ---- what the map implies -------------------------------------------

    @property
    def lexical(self) -> bool:
        """A keyword channel needs a tsvector column to run against."""
        return bool(self.tsv)

    @property
    def filters(self) -> bool:
        """`strategies` and `language` narrow only if there is a column to narrow on.

        False means the predicate is dropped rather than passed and ignored —
        which is the honest reading of "this index never carried this app's
        chunking vocabulary".
        """
        return bool(self.strategy)

    @property
    def parallel_text(self) -> bool:
        return bool(self.english)

    @property
    def searchable(self) -> bool:
        """The floor: something to compare against and something to read back."""
        return bool(self.embedding and self.text)


DEFAULT = ColumnMap()

#: opclass suffix → (operator, how to turn its distance into a similarity).
#:
#: The similarity scale matters as much as the operator. Gate 2's floor and
#: margin were swept on *cosine* similarity (src/core/config.py), so a store
#: indexed for another distance is scored onto the same 0..1 orientation —
#: higher is nearer — and the profile notes that the floors were not calibrated
#: for it. Getting the direction wrong would invert every comparison in
#: guardrails.py, which is a silent failure rather than a loud one.
METRICS: dict[str, tuple[str, str]] = {
    # `<=>` is cosine distance in [0, 2].
    "cosine": ("<=>", "1 - ({column} <=> %(vector)s)"),
    # `<#>` returns the *negative* inner product, so negating it is the score.
    "inner_product": ("<#>", "(-1) * ({column} <#> %(vector)s)"),
    # `<->` is Euclidean distance in [0, inf). Monotone decreasing, bounded
    # into (0, 1] — an ordering that is right and a magnitude that is not on
    # the cosine scale.
    "l2": ("<->", "1 / (1 + ({column} <-> %(vector)s))"),
}


def operator(metric: str) -> str:
    return METRICS.get(metric, METRICS["cosine"])[0]


def score_expression(column: str, metric: str) -> str:
    return METRICS.get(metric, METRICS["cosine"])[1].format(column=column)


# ---- query construction --------------------------------------------------


def _vector_column(cmap: ColumnMap, *, english: bool) -> str:
    """Which vector to search. Falls back when the English pair is absent.

    An English question against a store with no `embedding_en` is answered from
    the one vector it has, which is the cross-lingual hop of
    docs/13a-cross-lingual.md rather than native English retrieval. Degrading
    here rather than at the call site keeps the fallback in one place.
    """
    if english and cmap.embedding_en:
        return cmap.embedding_en
    return cmap.embedding


def _tsv_column(cmap: ColumnMap, *, english: bool) -> str:
    if english and cmap.tsv_en:
        return cmap.tsv_en
    return cmap.tsv


def _select(cmap: ColumnMap) -> str:
    """The four columns both queries return, with absent ones filled honestly.

    An index with no strategy label yields `''`, not a made-up one, and a hit
    with no strategy is still a hit — `Hit.strategy` is used for reporting and
    grouping, never as a key.
    """
    strategy = cmap.strategy or "''::text"
    identifier = cmap.id or "''::text"

    if cmap.meta:
        meta = cmap.meta
    elif cmap.payload:
        # No `meta` column, but columns worth citing. `jsonb_build_object`
        # turns them into the same payload shape the rest of the pipeline
        # already reads, so a connected store's hits can name their source.
        pairs = ", ".join(f"'{name}', {name}" for name in cmap.payload)
        meta = f"jsonb_build_object({pairs})"
    else:
        meta = "'{}'::jsonb"

    return (
        f"SELECT {identifier},\n"
        f"       {strategy},\n"
        f"       {cmap.text},\n"
        f"       {meta},\n"
    )


def _predicates(cmap: ColumnMap) -> str:
    """The strategy and language filters, present only when they can narrow."""
    lines = []
    if cmap.strategy:
        lines.append("WHERE strategy = ANY(%(strategies)s)")
    if cmap.language:
        clause = "AND" if lines else "WHERE"
        lines.append(f"  {clause} (%(language)s::text IS NULL OR language = %(language)s)")
    return "\n".join(lines)


def search_sql(table: str, cmap: ColumnMap = DEFAULT, *, english: bool = False) -> str:
    """Dense ANN search, with the operator the index was actually built for.

    Two things that are easy to get backwards and silent when wrong:

    **The operator has to match the opclass.** `<=>` against a `vector_l2_ops`
    index does not use the index — it sequential-scans, and the only symptom is
    latency.

    **The score has to be a similarity, not a distance.** Higher is nearer, on
    the orientation `guardrails.py` compares against. Using the raw distance
    would invert every one of those comparisons without raising.
    """
    embedding = _vector_column(cmap, english=english)
    op = operator(cmap.metric)
    predicate = _predicates(cmap)
    filtered = (
        f"{predicate}\n  AND {embedding} IS NOT NULL"
        if predicate
        else f"WHERE {embedding} IS NOT NULL"
    )

    return (
        f"\n{_select(cmap)}"
        f"       {score_expression(embedding, cmap.metric)} AS score\n"
        f"FROM {quote_table(table)}\n"
        f"{filtered}\n"
        f"ORDER BY {embedding} {op} %(vector)s\n"
        f"LIMIT %(limit)s\n"
    )


def lexical_sql(
    table: str, cmap: ColumnMap = DEFAULT, *, english: bool = False, config: str = "simple"
) -> str:
    """Keyword search, for rung 2's hybrid fusion.

    `websearch_to_tsquery` rather than `plainto_tsquery`: it never raises on
    punctuation, which matters when the query is whatever the speech recogniser
    heard. `ts_rank_cd` with normalisation 32 gives a score in (0, 1) — but not
    on the cosine scale, which is exactly why fusion is by rank and never by
    score.
    """
    tsv = _tsv_column(cmap, english=english)
    predicate = _predicates(cmap)
    where = (
        f"WHERE {tsv} @@ query\n" + predicate.replace("WHERE", "  AND", 1) + "\n"
        if predicate
        else f"WHERE {tsv} @@ query\n"
    )

    return (
        f"\n{_select(cmap)}"
        f"       ts_rank_cd({tsv}, query, 32) AS score\n"
        f"FROM {quote_table(table)}, websearch_to_tsquery('{config}', %(query)s) AS query\n"
        f"{where}"
        f"ORDER BY score DESC\n"
        f"LIMIT %(limit)s\n"
    )


def text_config(cmap: ColumnMap, *, english: bool) -> str:
    """Which Postgres text-search configuration the lexical query stems with.

    `english` gets the real stemmer and is most of why the lexical channel is
    worth having on that side; the Indic side uses `simple` because Postgres
    has no Hindi stemmer. A connected store with no English pair gets `simple`
    whatever was asked, since there is no English column to stem.
    """
    return "english" if (english and cmap.tsv_en) else "simple"
