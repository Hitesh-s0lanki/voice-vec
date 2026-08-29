"""Reading someone else's store without being a nuisance in it.

Everything a probe does is bounded, read-only and cheap enough to run while a
person watches a spinner. That is not politeness for its own sake — a probe is
a guest in a database this app does not own, whose connection limit, query
budget and bill all belong to somebody else. `SELECT *` on a 10M-row table is
one careless line away and would be indistinguishable, from the user's side,
from this app breaking their production Postgres.

So three rules hold in every probe:

  **Sample, never scan.** `SAMPLE_SIZE` records, whatever the store holds. The
  statistics that matter — does this field exist, on what share of records, how
  long is the text — converge long before 200 records and do not get truer at
  200,000.

  **Estimate before counting.** An exact `count(*)` is a seq scan. Where the
  store has an estimate, it is used, and an exact count is only taken when the
  estimate says it is small.

  **Never raise past the caller.** A probe that fails returns an unreachable
  `Observation` carrying the reason. Profiling is a nice-to-have layered on a
  connector that already verified; it may not turn a working connector into a
  broken one.

The shared analysis lives here rather than in each probe because it is where
the one number that matters is computed. `field_stats` measures *coverage* —
the share of sampled records carrying a key — and coverage is what separates a
filter from a trap. Four probes computing it four ways would drift, and the
first drift would be a store that claims a filter it cannot honour.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.connectors.profile import (
    COUNT_DISTINCT_CAP,
    EXCERPT_CHARS,
    MAX_DISTINCT,
    MAX_EXCERPTS,
    FieldStat,
    Observation,
)

log = logging.getLogger("vec.connectors.probe")

#: Enough for coverage to converge, small enough to be free. A field on 80% of
#: records is distinguishable from one on 100% at n=200 well past any precision
#: the threshold needs.
SAMPLE_SIZE = 200

#: A probe runs off the request path, but not without end: a store that has not
#: answered in this long is a store the agent should be told is unreachable.
PROBE_TIMEOUT_S = 15.0

#: Which payload key holds the readable text, in the order worth trying. The
#: first two are this app's own schema and the shape its ingest writes; the
#: rest are what other people's pipelines conventionally call it.
TEXT_KEYS = ("text", "chunk_text", "content", "page_content", "body", "passage")

#: Keys that are never metadata worth reporting — the vector itself, and the
#: store's own bookkeeping. Reporting `$vector` as a field with 100% coverage
#: is true and useless.
IGNORED_KEYS = frozenset({"$vector", "$similarity", "vector", "values", "embedding"})

_WHITESPACE = re.compile(r"\s+")


class Probe(Protocol):
    """One connector's reader. Built from credentials, run once, thrown away."""

    def observe(self) -> Observation:
        """Measure the store. Never raises; an unreachable store is a result."""
        ...


# ---- analysis shared by every probe --------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, Mapping):
        return "object"
    return "null" if value is None else type(value).__name__


def _short(value: Any, limit: int = 40) -> str:
    text = _WHITESPACE.sub(" ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def field_stats(records: Sequence[Mapping[str, Any]]) -> tuple[FieldStat, ...]:
    """Coverage, types and cardinality per key, over the sample.

    A key present with a `None` value does not count as covered. That is the
    difference between "this record has no language" and "this store does not
    record language", and only the second is a reason to stop filtering — but
    for the purpose the caller has, filtering on a key that is null half the
    time drops half the store, so both are treated as absent here.
    """
    if not records:
        return ()

    total = len(records)
    seen: dict[str, int] = defaultdict(int)
    types: dict[str, Counter] = defaultdict(Counter)
    lengths: dict[str, int] = defaultdict(int)
    values: dict[str, set[str]] = defaultdict(set)
    overflowed: set[str] = set()

    for record in records:
        for key, value in (record or {}).items():
            if key in IGNORED_KEYS or value is None or value == "":
                continue
            seen[key] += 1
            types[key][_kind(value)] += 1
            if isinstance(value, str):
                lengths[key] += len(value)
            # Counted to `COUNT_DISTINCT_CAP`, quoted only to `MAX_DISTINCT`.
            # The count is what separates a facet from a primary key — `id` and
            # the text column are on 100% of records and are useless to filter
            # on, and nothing but cardinality says so. Past the cap the bucket
            # is dropped rather than grown, so a free-text field never becomes
            # a copy of the corpus in a stats dictionary.
            if key not in overflowed and isinstance(value, (str, int, float, bool)):
                bucket = values[key]
                bucket.add(str(value)[:64])
                if len(bucket) > COUNT_DISTINCT_CAP:
                    overflowed.add(key)
                    bucket.clear()

    stats = []
    for key, count in seen.items():
        distinct = None if key in overflowed else len(values[key]) or None
        examples = (
            tuple(sorted(_short(v) for v in values[key])[:3])
            if key not in overflowed and distinct and distinct <= MAX_DISTINCT
            else ()
        )
        stats.append(
            FieldStat(
                name=key,
                types=tuple(kind for kind, _ in types[key].most_common()),
                coverage=count / total,
                distinct=distinct,
                examples=examples,
                carried=count,
                mean_chars=round(lengths[key] / count) if count else 0,
            )
        )

    return tuple(sorted(stats, key=lambda s: (-s.coverage, s.name)))


def pick_text_field(records: Sequence[Mapping[str, Any]]) -> str:
    """Which key holds the prose, by convention first and by length second.

    Falling back to "the longest string field" is what makes this work on a
    store nobody built for this app. It is also why the answer is reported in
    the profile rather than assumed: an agent quoting the wrong field produces
    a citation that is technically from the corpus and reads as nonsense.
    """
    if not records:
        return ""

    keys = {key for record in records for key in (record or {})}
    for candidate in TEXT_KEYS:
        if candidate in keys:
            return candidate

    widest, best = "", 0.0
    for key in keys - IGNORED_KEYS:
        lengths = [
            len(record[key])
            for record in records
            if isinstance(record.get(key), str)
        ]
        if not lengths:
            continue
        mean = sum(lengths) / len(lengths)
        # A short string field is an id or a label, not the document.
        if mean > best and mean >= 80:
            widest, best = key, mean
    return widest


def text_of(records: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    if not key:
        return []
    return [
        _WHITESPACE.sub(" ", record[key]).strip()
        for record in records
        if isinstance(record.get(key), str) and record[key].strip()
    ]


def length_stats(texts: Sequence[str]) -> tuple[int, int, int] | None:
    """min, mean, max characters — the chunker's fingerprint.

    Worth carrying because it is how an agent knows what a hit will cost it: a
    store of 3,000-character chunks fills a context window in four hits, and a
    store of 200-character ones needs forty to say anything.
    """
    if not texts:
        return None
    lengths = [len(t) for t in texts]
    return (min(lengths), round(sum(lengths) / len(lengths)), max(lengths))


#: Unicode block prefixes → the writing system a reader would name. Script, not
#: language: Hindi and Marathi are both Devanagari and no amount of counting
#: code points tells them apart. Claiming the language would be a guess an
#: agent then routes on, so the profile says what was actually observed.
_SCRIPTS = {
    "LATIN": "Latin",
    "DEVANAGARI": "Devanagari",
    "BENGALI": "Bengali",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Gurmukhi",
    "ORIYA": "Odia",
    "ARABIC": "Arabic",
    "CYRILLIC": "Cyrillic",
    "HIRAGANA": "Japanese",
    "KATAKANA": "Japanese",
    "CJK": "Han",
    "HANGUL": "Korean",
    "GREEK": "Greek",
    "HEBREW": "Hebrew",
    "THAI": "Thai",
}

#: A script under this share of letters is noise — one transliterated name in
#: an English corpus should not make the card claim two writing systems.
_SCRIPT_FLOOR = 0.05


def scripts_of(texts: Iterable[str], *, cap: int = 20_000) -> tuple[str, ...]:
    """Which writing systems the sample is actually in, by share of letters."""
    counts: Counter = Counter()
    budget = cap

    for text in texts:
        for char in text[:budget]:
            if not char.isalpha():
                continue
            try:
                name = unicodedata.name(char)
            except ValueError:
                continue
            for prefix, label in _SCRIPTS.items():
                if name.startswith(prefix):
                    counts[label] += 1
                    break
        budget -= len(text)
        if budget <= 0:
            break

    total = sum(counts.values())
    if not total:
        return ()
    return tuple(
        label
        for label, count in counts.most_common()
        if count / total >= _SCRIPT_FLOOR
    )


#: A field with more distinct values than this is not a document boundary — it
#: is a label, an id, or free text, and stratifying over it would just be a
#: more expensive uniform sample.
MAX_FACET_VALUES = 64


def dominant_facet(stats: Sequence[FieldStat], text_field: str) -> str:
    """The field that most looks like "which document is this passage from".

    Lowest cardinality above one, among fields that are on nearly every record
    and are not the text itself. On a store of chunked books that is `book_id`;
    on one of scraped pages it is the domain or the source url.

    Not a schema assumption — nothing is looked up by name. It is the shape a
    document boundary has: present everywhere, few values, not unique per row.
    """
    candidates = [
        stat
        for stat in stats
        if stat.filterable
        and stat.categorical
        and not stat.constant
        and not stat.unique
        and stat.name != text_field
        and stat.distinct is not None
        and 1 < stat.distinct <= MAX_FACET_VALUES
    ]
    return min(candidates, key=lambda s: s.distinct or 0).name if candidates else ""


def excerpts_of(
    records: Sequence[Mapping[str, Any]],
    text_key: str,
    *,
    stats: Sequence[FieldStat] = (),
    allowed: bool = True,
) -> tuple[str, ...]:
    """A few short passages, spread across *documents* rather than across rows.

    A uniform sample of a chunked corpus is dominated by whichever document is
    longest. Quoting five passages from it produced a summary that called a
    twelve-book library "a small corpus of book text chunks" mentioning
    "philanthropy, political punishment, and football" — three excerpts, three
    books, and no sense of the shape of the whole thing.

    So when the sample reveals a document boundary, one excerpt is taken per
    document. Same cost to the model, same amount of the user's data, and the
    difference is whether the agent knows what it is holding.

    Also skipped: the head of each document. The first chunk of an ingested
    book is a copyright page, and a card quoting those describes the publisher.
    """
    if not allowed or not records or not text_key:
        return ()

    usable = [
        record
        for record in records
        if isinstance(record.get(text_key), str) and len(record[text_key].strip()) > 40
    ]
    if not usable:
        return ()

    facet = dominant_facet(stats, text_key)
    if facet:
        # One per document, in whatever order the sample gave them — the sample
        # is already random, so taking the first of each group is not the first
        # chunk of each document.
        by_document: dict[str, str] = {}
        for record in usable:
            key = str(record.get(facet, ""))
            by_document.setdefault(key, record[text_key])
            if len(by_document) >= MAX_EXCERPTS:
                break
        picked = list(by_document.values())
    else:
        step = max(1, len(usable) // MAX_EXCERPTS)
        picked = [record[text_key] for record in usable[::step]][:MAX_EXCERPTS]

    return tuple(
        _WHITESPACE.sub(" ", text)[:EXCERPT_CHARS].strip() for text in picked
    )


def unreachable(connector: str, kind: str, location: str, reason: str) -> Observation:
    """The result of a probe that could not read the store.

    A result and not an exception: the connector still works, the credential is
    still good, and the agent needs to be told "do not route here" rather than
    have the whole profile fail to exist.
    """
    return Observation(
        connector=connector,
        kind=kind,
        location=location,
        reachable=False,
        notes=(reason,) if reason else (),
    )


def embedding_match(stored: Sequence[float] | None, text: str, embed) -> float | None:
    """Cosine between a record's own vector and our embedding of its own text.

    The one test that says whether a connected index and this app's query
    embedder live in the same space — and it is a test, not an assumption.
    Every backend has assumed it from the *width*: a 768-dim index accepts a
    768-dim query vector, returns its nearest neighbours, and if the two models
    differ every one of those neighbours is noise. Nothing raises. The
    guardrail abstains on every question and the store reads as connected,
    searchable, and simply unhelpful.

    Identical text under the same model scores ~1.0 — a prefix convention like
    e5's "passage: " still leaves it above 0.9 — and a different model scores
    ~0. Measured on one real store built elsewhere: **0.0032**.

    `None` for every reason it could not be measured: no vector, no text, no
    embedder, a provider that refused. Untested is not evidence of a mismatch,
    and `CapabilityFacts.derive` takes nothing away for it.
    """
    if not text or not text.strip() or stored is None:
        return None

    try:
        import numpy as np

        theirs = np.asarray(stored, dtype="float32")
        ours = np.asarray(embed(text), dtype="float32")
        if theirs.size == 0 or theirs.size != ours.size:
            return None
        norms = float(np.linalg.norm(theirs)) * float(np.linalg.norm(ours))
        if norms == 0.0:
            return None
        return float(theirs @ ours / norms)
    except Exception as error:
        log.info("could not compare embeddings: %s", type(error).__name__)
        return None
