"""Chunking — shared by the offline ingest and the query-time extractor.

Sharing it is deliberate: chunking logic that differs between ingest and query
time produces quietly wrong retrieval that no test catches
(docs/03-chunking.md).

v1 indexes S1 only. The other four strategies are named here because the
payload and the collection are already shaped for them.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum


class ChunkStrategy(StrEnum):
    """The five cuts from docs/03-chunking.md. v1 ingests PASSAGE only."""

    PASSAGE = "S1"
    SENTENCE_WINDOW = "S2"
    CLUSTER_RECUT = "S3"
    SEMANTIC = "S4"
    BILINGUAL = "S5"


ALL_STRATEGIES: tuple[str, ...] = tuple(s.value for s in ChunkStrategy)

_WHITESPACE = re.compile(r"\s+")

# Devanagari ends sentences with danda `।` and double danda `॥`; Urdu uses the
# Arabic full stop `۔`. Splitting on the Latin period alone yields exactly one
# giant "sentence" per Hindi passage — a bug that reads as bad recall.
_SENTENCE_END = re.compile(r"(?<=[।॥۔?!])\s+|(?<=[.:;])\s+(?=[^\s])")

# Punctuation stripped before hashing, so two encodings of the same passage
# collapse to one chunk.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """NFC + collapse whitespace.

    Devanagari has multiple valid encodings for the same grapheme. Without NFC
    at ingest *and* at query time, dedup misses duplicates and lexical matching
    silently fails.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def content_key(text: str) -> str:
    """Stable hash of the *meaning-bearing* characters, for dedup."""
    stripped = _PUNCTUATION.sub("", normalise(text).casefold())
    stripped = _WHITESPACE.sub(" ", stripped).strip()
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:16]


def tokens(text: str) -> list[str]:
    """Word tokens, casefolded. Used for the lexical prefilter, never for embedding."""
    return _TOKEN.findall(normalise(text).casefold())


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence plus its span in the parent text.

    The span is what makes Gate 3 a substring check rather than a similarity
    score: the extractor slices the original text instead of re-joining pieces.
    """

    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    """Split on Indic and Latin terminators, keeping offsets into `text`."""
    if not text:
        return []

    spans: list[Sentence] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        piece = text[cursor : match.start()].strip()
        if piece:
            start = text.index(piece, cursor)
            spans.append(Sentence(piece, start, start + len(piece)))
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        start = text.index(tail, cursor)
        spans.append(Sentence(tail, start, start + len(tail)))

    return spans or [Sentence(text.strip(), 0, len(text))]


@dataclass(slots=True)
class Origin:
    """Where a chunk came from — plural, because dedup merges duplicates.

    docs/07-evaluation.md: a retrieved chunk counts as correct when its
    `passage_idx` intersects the gold indices *for the query under evaluation*,
    so the whole list has to survive ingest.
    """

    query_id: int
    passage_idx: int
    is_selected: int


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    strategy: str
    text: str
    english: str | None
    language: str
    query_type: str
    origins: list[Origin] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "chunkId": self.chunk_id,
            "strategy": self.strategy,
            "text": self.text,
            "english": self.english,
            "language": self.language,
            "queryType": self.query_type,
            "sourceQueryIds": sorted({o.query_id for o in self.origins}),
            "origins": [
                {
                    "queryId": o.query_id,
                    "passageIdx": o.passage_idx,
                    "isSelected": o.is_selected,
                }
                for o in self.origins
            ],
        }


@dataclass(slots=True)
class Row:
    """One flattened MSMARCO-XI record (docs/01-dataset.md)."""

    query_id: int
    query_type: str
    query: str
    answer: str
    language: str
    indic_passages: list[str]
    english_passages: list[str]
    is_selected: list[int]

    @property
    def answerable(self) -> bool:
        """`sum(is_selected) == 0` is the authoritative unanswerable label."""
        return sum(self.is_selected) > 0


def passage_atomic(row: Row) -> list[Chunk]:
    """S1 — each translated passage indexed verbatim, one vector per passage.

    The control the other four strategies are measured against, and not a
    throwaway: for DESCRIPTION queries (72% of the corpus) the passage boundary
    is often already the right answer boundary.
    """
    chunks: list[Chunk] = []

    for idx, raw in enumerate(row.indic_passages):
        text = normalise(raw)
        if not text:
            continue

        english = row.english_passages[idx] if idx < len(row.english_passages) else None
        selected = row.is_selected[idx] if idx < len(row.is_selected) else 0

        chunks.append(
            Chunk(
                chunk_id=f"{ChunkStrategy.PASSAGE.value}:{content_key(text)}",
                strategy=ChunkStrategy.PASSAGE.value,
                text=text,
                english=normalise(english) if english else None,
                language=row.language,
                query_type=row.query_type,
                origins=[Origin(row.query_id, idx, int(selected))],
            )
        )

    return chunks


def merge_duplicates(chunks: list[Chunk]) -> list[Chunk]:
    """Collapse chunks sharing a content hash, keeping every origin.

    Passages repeat across `query_id`s because MS MARCO retrieved the same
    passage for related queries. Doing this *before* embedding is the cheapest
    speedup in the whole ingest.
    """
    merged: dict[str, Chunk] = {}

    for chunk in chunks:
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            merged[chunk.chunk_id] = chunk
            continue
        existing.origins.extend(chunk.origins)

    return list(merged.values())
