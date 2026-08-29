"""What a connected store actually holds, as a shape the agent can read.

Connecting a store proves a credential works. It does not tell the agent one
useful thing about what is *in* there, and that gap is where connected
retrieval quietly fails: `PineconeBackend.capabilities()` has always returned
`filters=True` for every Pinecone index on earth, and its own docstring calls
that "a hope rather than a guarantee". The hope is that the index carries a
`strategy` field. When it does not, the filter matches everything, the ladder
believes it narrowed the search, and nobody finds out.

So this module is the shape of a *measurement* instead. Three layers, and the
separation is the point:

    Observation     what a probe read off the store. Numbers and field names.
                    No opinions, no prose, no model in the loop.
    Understanding   what one LLM call made of that. Prose, topics, and the two
                    lists an agent actually routes on: good_for / not_for.
    Profile         both, plus the capability facts derived from the first,
                    stamped with the credential fingerprint it was built from.

Only `Observation` is trusted for control flow. `Understanding` is written by a
model over sampled text and is allowed to be wrong — it steers, it does not
gate. Anything that decides whether a query is filtered or fused reads the
measurement, never the prose.

**Excerpts are the user's own data.** A profile carries a few short passages so
the card can say what the corpus reads like, and they live in this app's
Postgres rather than the user's. They are capped hard (`MAX_EXCERPTS` ×
`EXCERPT_CHARS`), never logged, and a deployment that would rather hold none
sets `profile_excerpts = False`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

#: Schema version of the JSON blob. A profile written by an older version is
#: re-probed rather than migrated: it is a cache of somebody else's database,
#: so recomputing it is cheap and inventing missing fields is not.
VERSION = 1

#: A metadata field found on fewer than this share of sampled records is not
#: something a query may filter on. This is the whole reason the module exists:
#: `strategy` present on 3% of an index is not a filter, it is a trap — the
#: predicate would drop the other 97% and the ladder would call it a narrowing.
FILTERABLE_COVERAGE = 0.80

#: Cosine between a record's stored vector and this app's own embedding of
#: that record's own text. Identical text under the same model scores ~1.0
#: (a prefix convention like e5's "passage: " still leaves it above 0.9) and
#: a different model scores ~0. Measured on one real store built elsewhere:
#: 0.0032. So the threshold is nowhere near either, and it is a test of
#: *which space the vectors live in*, not of quality.
EMBEDDING_MATCH_MIN = 0.5

#: What the card is allowed to quote back. Small on purpose: enough for a model
#: to recognise the register of the corpus, not enough to be a copy of it —
#: and spread across documents rather than drawn uniformly, so the count is
#: about how many *sources* are represented rather than how many passages.
MAX_EXCERPTS = 5
EXCERPT_CHARS = 240

#: Longer than this and a string field is prose, not a label. Set below
#: `pick_text_field`'s 80-character floor for a text candidate, so nothing can
#: be both the document and a facet of it.
MAX_FACET_CHARS = 64

#: Distinct values are only *quoted* while they are few enough to enumerate.
#: Past this a field is free text, and listing its values would put a sample of
#: the user's data in the card.
MAX_DISTINCT = 12

#: How many distinct values are *counted* before a field is written off as
#: unbounded. Higher than `MAX_DISTINCT` because counting and quoting answer
#: different questions: a `language` field with 40 values is a perfectly good
#: filter that should not be quoted, and telling it apart from a primary key
#: needs the count.
COUNT_DISTINCT_CAP = 512

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"  # probed, but the store answered incompletely
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"


# ---- the measurement -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldStat:
    """One metadata key, and how much of the sample actually carries it.

    `coverage` is the field that matters. Presence is what a naive probe
    records — "the index has a `strategy` key" — and presence is what makes
    `filters=True` a lie, because one record out of two hundred having it is
    indistinguishable from all of them having it if you only ask whether you
    ever saw it.
    """

    name: str
    types: tuple[str, ...] = ()
    coverage: float = 0.0
    distinct: int | None = None
    examples: tuple[str, ...] = ()
    #: How many records carried it, so `distinct` can be read against something.
    carried: int = 0
    #: Mean length of its string values. A facet's values are labels; prose is
    #: not a facet however few distinct values a sample happens to see.
    mean_chars: int = 0

    @property
    def filterable(self) -> bool:
        return self.coverage >= FILTERABLE_COVERAGE

    @property
    def categorical(self) -> bool:
        """Is this the kind of value a query can group by?

        A timestamp, a float, a list and a nested object all pass every other
        test — present on every record, never null — and none of them is a
        facet. `created_at`, `origins` and `sourceQueryIds` were all offered as
        filters on a real store before this existed.

        So is prose. A parallel-translation column is a string, and a sample
        that happens to contain a repeated passage makes it look enumerable —
        `english` was offered as a filter on one run and not the next, which is
        the worst of both. Length settles it: a facet's values are labels.
        """
        if not self.types or not set(self.types) <= {"string", "int", "bool"}:
            return False
        return self.mean_chars <= MAX_FACET_CHARS

    @property
    def unique(self) -> bool:
        """A different value on every record — an identifier, not a facet.

        `id` and the text column both satisfy every other test for a good
        filter: present on 100% of records, well-typed, never null. Offering
        them to an agent as things to filter on is noise at best, and at worst
        it spends a turn constructing a predicate that can only ever match the
        one row it already had.
        """
        return (
            self.distinct is not None and self.carried > 1 and self.distinct >= self.carried
        )

    @property
    def constant(self) -> bool:
        """One value across the whole sample — carried, and worth nothing.

        `book_chunks.page` was `1` on every row of a 2,366-row table: a column
        that looks like metadata, satisfies a `NOT NULL`, and cannot narrow
        anything. Worth saying out loud, because the alternative is an agent
        that keeps trying to cite a page number.
        """
        return self.distinct == 1 and self.coverage > 0


@dataclass(frozen=True, slots=True)
class VectorShape:
    """The geometry of the index, which decides whether a search is even legal."""

    dimensions: int | None = None
    metric: str = ""
    index: str = ""
    #: Unit length? Cosine does not care, inner product does. A store of
    #: truncated Matryoshka vectors is not normalised and nothing says so.
    normalised: bool | None = None
    #: None when the store will not count cheaply. Not zero — the difference
    #: between "empty" and "did not ask" is the difference between an abstain
    #: and a bug.
    records: int | None = None


@dataclass(frozen=True, slots=True)
class ToolShape:
    """The other kind of capability: what the agent can *do*, not search."""

    toolkits: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    authorised: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Observation:
    """What the probe read. Everything here was measured or is None."""

    connector: str
    kind: str
    #: `describe()` — where this points, with no credential in it.
    location: str = ""
    reachable: bool = False
    sampled: int = 0
    vectors: VectorShape | None = None
    tools: ToolShape | None = None
    fields: tuple[FieldStat, ...] = ()
    #: Which field held the readable text. Empty means the probe found none,
    #: which is itself a finding: an index whose payload has no text can be
    #: searched but cannot be quoted, so an answer cut from it has no source.
    text_field: str = ""
    text_chars: tuple[int, int, int] | None = None  # min, mean, max
    #: Writing systems seen in the sample. Script, not language — Hindi and
    #: Marathi are both Devanagari and this does not pretend to tell them apart.
    scripts: tuple[str, ...] = ()
    excerpts: tuple[str, ...] = ()
    latency_ms: float = 0.0
    #: Things a person should be told, in their words. "page is 1 on every row."
    notes: tuple[str, ...] = ()
    #: Cosine between one record's stored vector and this app's embedding of
    #: that same record's text. `None` means the probe could not test it —
    #: which is not evidence of anything, so nothing is taken away for it.
    embedding_match: float | None = None

    def stat(self, name: str) -> FieldStat | None:
        return next((f for f in self.fields if f.name == name), None)

    def has(self, name: str) -> bool:
        """Is this field present *often enough to be used*, not merely seen."""
        stat = self.stat(name)
        return bool(stat and stat.filterable)


# ---- what the measurement implies ---------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityFacts:
    """`Capabilities`, derived rather than declared.

    Deliberately plain booleans and not `src.rag.store.Capabilities`: rag reads
    connectors and never the reverse, and a dataclass imported the wrong way up
    is how that rule gets broken quietly. `src/rag/backends/profiled.py` maps
    these onto the real thing.

    Every field defaults to the pessimistic answer, for the reason the original
    `Capabilities` docstring gives: claiming a channel that is not there fails
    at query time, while not claiming one that is only costs the recall it
    would have added.
    """

    lexical: bool = False
    filters: bool = False
    parallel_text: bool = False
    #: The store can be searched at all — reachable, non-empty, right width,
    #: and built in the same embedding space this app can produce queries in.
    searchable: bool = False
    #: Whether the vectors and our query embedder are the same space. `None`
    #: means untested; only a measured mismatch takes `searchable` away.
    compatible: bool | None = None

    @classmethod
    def derive(cls, observation: Observation, *, embed_dim: int) -> "CapabilityFacts":
        """Read the facts off the measurement. No service-specific special cases.

        `filters` is the one worth reading twice. It is true only when the
        sample says `strategy` is *actually there*, on nearly every record —
        which is what makes a filtered search a narrowing rather than a
        coincidence. Every backend has claimed it unconditionally until now.
        """
        shape = observation.vectors
        dims = shape.dimensions if shape else None
        records = shape.records if shape else None

        # Width is not identity, and assuming it was is the failure this field
        # exists for: a 768-dim index built by somebody else's model accepts a
        # 768-dim query vector, returns its nearest neighbours, and every one
        # of them is noise — top similarity 0.086 where a real match is 0.85.
        # Nothing errors. The ladder abstains on every question, and the store
        # reads as connected, searchable and simply unhelpful.
        compatible = (
            None
            if observation.embedding_match is None
            else observation.embedding_match >= EMBEDDING_MATCH_MIN
        )

        return cls(
            lexical=observation.has("tsv") or observation.has("_lexical"),
            filters=observation.has("strategy"),
            parallel_text=observation.has("english"),
            compatible=compatible,
            searchable=(
                observation.reachable
                and (dims is None or dims == embed_dim)
                and (records is None or records > 0)
                # Untested stays searchable. Absence of a measurement is not
                # evidence of absence — the same rule `ProfiledBackend.merge`
                # applies to every other capability.
                and compatible is not False
            ),
        )


@dataclass(frozen=True, slots=True)
class Understanding:
    """The written half — one LLM call over the sample.

    Steering, never gating. `good_for` and `not_for` are what a router reads to
    decide whether a question belongs to this store at all, and being wrong
    about them costs a wasted retrieval; being wrong about `CapabilityFacts`
    costs a silently empty result set. That asymmetry is why they are separate
    types and why only one of them is derived from prose.
    """

    title: str = ""
    summary: str = ""
    topics: tuple[str, ...] = ()
    good_for: tuple[str, ...] = ()
    not_for: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.summary or self.topics)


@dataclass(frozen=True, slots=True)
class Profile:
    """One connected store, understood. This is what gets stored and served."""

    connector: str
    kind: str
    status: str
    observation: Observation
    understanding: Understanding = field(default_factory=Understanding)
    facts: CapabilityFacts = field(default_factory=CapabilityFacts)
    error: str = ""
    profiled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = VERSION

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def card(self, *, budget: int = 700) -> str:
        """The agent-facing rendering: what this store is, in a prompt's worth of text.

        Written for a system prompt on a latency path, so it is plain lines
        rather than JSON — a model reads `Filter on: book_id` correctly and
        spends tokens parsing `{"fields":[{"name":"book_id",...}]}` to reach the
        same place. `budget` is characters, and the sections are ordered so that
        truncation drops the least decision-relevant part last.
        """
        return render_card(self, budget=budget)

    def to_json(self) -> dict[str, Any]:
        return _encode(self)

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Profile | None":
        """None for a blob this version cannot read — the caller re-probes.

        A profile is a cache of somebody else's database. Migrating one is
        strictly more work than recomputing it, and a half-migrated profile
        makes claims about an index nobody measured.
        """
        try:
            if int(blob.get("version") or 0) != VERSION:
                return None
            return _decode(blob)
        except (TypeError, ValueError, KeyError):
            return None

    def with_understanding(self, understanding: Understanding) -> "Profile":
        return replace(self, understanding=understanding)


# ---- the card ------------------------------------------------------------


def _count(value: int | None) -> str:
    if value is None:
        return "size unknown"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M records"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k records"
    return f"{value} records"


def _line_head(profile: Profile) -> str:
    """Name, then size. Branching on `kind`, never on which sub-block is set.

    Keying off `vectors is None` reads a vector store the probe could not open
    as a tools connector, and the card then opens with "0 authorised toolkits"
    for somebody's Postgres. The sub-blocks say what was *measured*; only
    `kind` says what the thing is.
    """
    observation = profile.observation
    title = profile.understanding.title or observation.location or profile.connector

    if profile.kind == "tools":
        tools = observation.tools
        n = len(tools.authorised) if tools else 0
        return f"{title} — {n} authorised toolkit{'s' if n != 1 else ''}"

    shape = observation.vectors
    if shape is None:
        # Reachable stores always have a shape; this is the unreadable one.
        return f"{title} — could not be read"

    bits = [_count(shape.records)]
    if shape.dimensions:
        bits.append(f"{shape.dimensions}-dim")
    if shape.metric:
        bits.append(shape.metric)
    return f"{title} — {', '.join(bits)}"


def render_card(profile: Profile, *, budget: int = 700) -> str:
    """Ordered most to least decision-relevant, because the tail is what truncates."""
    observation = profile.observation
    understanding = profile.understanding
    lines: list[str] = [_line_head(profile)]

    if profile.status == STATUS_FAILED:
        lines.append(f"Unavailable: {profile.error or 'the store did not answer'}.")
        return "\n".join(lines)

    if understanding.summary:
        lines.append(understanding.summary)

    if understanding.good_for:
        lines.append(f"Good for: {', '.join(understanding.good_for)}.")
    if understanding.not_for:
        lines.append(f"Not for: {', '.join(understanding.not_for)}.")

    if observation.tools is not None:
        if observation.tools.authorised:
            lines.append(f"Can act on: {', '.join(observation.tools.authorised)}.")
        return _fit(lines, budget)

    # The mechanical half — what a query is allowed to do. Last, because a
    # model that loses it still searches correctly; it just does not filter.
    usable = [
        f.name
        for f in observation.fields
        if f.filterable
        and f.categorical
        and not f.constant
        and not f.unique
        and f.name != observation.text_field
    ]
    if usable:
        lines.append(f"Filter on: {', '.join(sorted(usable)[:8])}.")

    dead = [f.name for f in observation.fields if f.constant]
    if dead:
        lines.append(f"Carried but useless (one value everywhere): {', '.join(sorted(dead)[:6])}.")

    channels = "dense + keyword" if profile.facts.lexical else "dense only"
    lines.append(f"Search: {channels}.")

    if profile.facts.compatible is False:
        # Said in terms of what somebody can do about it, because there is
        # exactly one thing: the index has to be rebuilt with an embedding
        # model this app can also produce queries with, or the app pointed at
        # the one that built it.
        lines.append(
            "Not searchable: these vectors were built by a different embedding model, "
            "so a search here returns unrelated records."
        )
    elif not profile.facts.searchable:
        lines.append("Not searchable right now — do not route questions here.")

    return _fit(lines, budget)


def _fit(lines: Sequence[str], budget: int) -> str:
    """Drop whole lines from the tail rather than cutting one mid-sentence.

    A card truncated mid-word reads to a model as a claim that stops, and the
    half it would keep is the half it was about to act on.
    """
    kept: list[str] = []
    spent = 0
    for line in lines:
        if spent + len(line) + 1 > budget and kept:
            break
        kept.append(line)
        spent += len(line) + 1
    return "\n".join(kept)


# ---- JSON round trip -----------------------------------------------------


def _encode(profile: Profile) -> dict[str, Any]:
    observation = profile.observation
    return {
        "version": profile.version,
        "connector": profile.connector,
        "kind": profile.kind,
        "status": profile.status,
        "error": profile.error,
        "profiled_at": profile.profiled_at.isoformat(),
        "facts": {
            "lexical": profile.facts.lexical,
            "filters": profile.facts.filters,
            "parallel_text": profile.facts.parallel_text,
            "searchable": profile.facts.searchable,
            "compatible": profile.facts.compatible,
        },
        "understanding": {
            "title": profile.understanding.title,
            "summary": profile.understanding.summary,
            "topics": list(profile.understanding.topics),
            "good_for": list(profile.understanding.good_for),
            "not_for": list(profile.understanding.not_for),
            "languages": list(profile.understanding.languages),
        },
        "observation": {
            "location": observation.location,
            "reachable": observation.reachable,
            "sampled": observation.sampled,
            "text_field": observation.text_field,
            "text_chars": list(observation.text_chars) if observation.text_chars else None,
            "scripts": list(observation.scripts),
            "excerpts": list(observation.excerpts),
            "latency_ms": observation.latency_ms,
            "notes": list(observation.notes),
            "embedding_match": observation.embedding_match,
            "vectors": (
                {
                    "dimensions": observation.vectors.dimensions,
                    "metric": observation.vectors.metric,
                    "index": observation.vectors.index,
                    "normalised": observation.vectors.normalised,
                    "records": observation.vectors.records,
                }
                if observation.vectors
                else None
            ),
            "tools": (
                {
                    "toolkits": list(observation.tools.toolkits),
                    "actions": list(observation.tools.actions),
                    "authorised": list(observation.tools.authorised),
                }
                if observation.tools
                else None
            ),
            "fields": [
                {
                    "name": f.name,
                    "types": list(f.types),
                    "coverage": f.coverage,
                    "distinct": f.distinct,
                    "examples": list(f.examples),
                    "carried": f.carried,
                    "mean_chars": f.mean_chars,
                }
                for f in observation.fields
            ],
        },
    }


def _decode(blob: Mapping[str, Any]) -> Profile:
    raw = dict(blob.get("observation") or {})
    vectors = raw.get("vectors")
    tools = raw.get("tools")
    chars = raw.get("text_chars")
    facts = dict(blob.get("facts") or {})
    written = dict(blob.get("understanding") or {})

    observation = Observation(
        connector=str(blob.get("connector") or ""),
        kind=str(blob.get("kind") or ""),
        location=str(raw.get("location") or ""),
        reachable=bool(raw.get("reachable")),
        sampled=int(raw.get("sampled") or 0),
        vectors=(
            VectorShape(
                dimensions=vectors.get("dimensions"),
                metric=str(vectors.get("metric") or ""),
                index=str(vectors.get("index") or ""),
                normalised=vectors.get("normalised"),
                records=vectors.get("records"),
            )
            if vectors
            else None
        ),
        tools=(
            ToolShape(
                toolkits=tuple(tools.get("toolkits") or ()),
                actions=tuple(tools.get("actions") or ()),
                authorised=tuple(tools.get("authorised") or ()),
            )
            if tools
            else None
        ),
        fields=tuple(
            FieldStat(
                name=str(f.get("name") or ""),
                types=tuple(f.get("types") or ()),
                coverage=float(f.get("coverage") or 0.0),
                distinct=f.get("distinct"),
                examples=tuple(f.get("examples") or ()),
                carried=int(f.get("carried") or 0),
                mean_chars=int(f.get("mean_chars") or 0),
            )
            for f in (raw.get("fields") or [])
        ),
        text_field=str(raw.get("text_field") or ""),
        text_chars=tuple(chars) if chars else None,  # type: ignore[arg-type]
        scripts=tuple(raw.get("scripts") or ()),
        excerpts=tuple(raw.get("excerpts") or ()),
        latency_ms=float(raw.get("latency_ms") or 0.0),
        notes=tuple(raw.get("notes") or ()),
        embedding_match=raw.get("embedding_match"),
    )

    return Profile(
        connector=observation.connector,
        kind=observation.kind,
        status=str(blob.get("status") or STATUS_PENDING),
        observation=observation,
        understanding=Understanding(
            title=str(written.get("title") or ""),
            summary=str(written.get("summary") or ""),
            topics=tuple(written.get("topics") or ()),
            good_for=tuple(written.get("good_for") or ()),
            not_for=tuple(written.get("not_for") or ()),
            languages=tuple(written.get("languages") or ()),
        ),
        facts=CapabilityFacts(
            lexical=bool(facts.get("lexical")),
            filters=bool(facts.get("filters")),
            parallel_text=bool(facts.get("parallel_text")),
            searchable=bool(facts.get("searchable")),
            compatible=facts.get("compatible"),
        ),
        error=str(blob.get("error") or ""),
        profiled_at=datetime.fromisoformat(str(blob["profiled_at"])),
        version=int(blob.get("version") or 0),
    )


def dumps(profile: Profile) -> str:
    return json.dumps(profile.to_json(), separators=(",", ":"))
