"""Text normalisation and sentence splitting, for the query-time path.

This file used to be the other half of an ingest as well — the `Chunk`, `Row`
and `passage_atomic` that cut one dataset into the rows behind `DATABASE_URL`.
That ingest is gone; retrieval reads whatever store its asker connected, and
this app writes to none of them.

What is left is what the *query* side needs and what the chunking of an
already-built index has to agree with: `normalise`, which decides whether two
spellings of a passage are the same text, and `split_sentences`, which decides
where a lifted answer may start and stop (`src/rag/extract.py`,
`src/rag/guardrails.py`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")

# Devanagari ends sentences with danda `।` and double danda `॥`; Urdu uses the
# Arabic full stop `۔`. Splitting on the Latin period alone yields exactly one
# giant "sentence" per Hindi passage — a bug that reads as bad recall.
_SENTENCE_END = re.compile(r"(?<=[।॥۔?!])\s+|(?<=[.:;])\s+(?=[^\s])")

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """NFC + collapse whitespace.

    Devanagari has multiple valid encodings for the same grapheme. Without NFC
    at query time, a question and the passage answering it can be the same text
    and compare unequal, and lexical matching silently fails.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


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
