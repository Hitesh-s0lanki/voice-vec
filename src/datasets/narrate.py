"""One model call that turns a measurement into something an agent can route on.

`Observation` knows a table holds 25,000 rows with a `query_type` column
carrying five values and a `query` column of ~50 bytes. That is everything
except the one thing a router needs: **what is it about**. No amount of
counting columns produces "machine-translated Indic search queries with their
English originals", and an agent that cannot say that cannot decide whether a
question belongs to this dataset or to the vector store beside it.

Fenced the same three ways as `src/connectors/narrate.py`, for the same
reasons:

  **It is handed the measurement, not asked to make one.** Row counts, column
  names, coverage and enumerated values are already known and go in as facts.
  The model reads a few rows and names the subject.

  **Its output never touches SQL.** `Understanding` decides which dataset gets
  asked. The schema card — measured, every line of it — decides what is written
  against it. A hallucinated topic costs a wasted query. A hallucinated column
  would cost a wrong answer, which is why the model is nowhere near that half.

  **Failure is empty, never a default.** No key, a timeout, unparseable JSON —
  all return `None`, and the card renders from the numbers alone. A generic
  summary on failure would put a sentence nobody measured in front of every
  routing decision.

**The excerpts leave for a model provider.** For a public dataset that is a
smaller disclosure than the connector case — the rows are already public at the
URL somebody pasted — but it is the same setting, `dataset_narrate`, because a
deployment that has turned the connector one off did not mean "except here".
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.core.config import Settings
from src.datasets.profile import Observation, Understanding
from src.rag import llm

log = logging.getLogger("vec.datasets.narrate")

#: Small. Six short fields and two sentences; room for more is room for a
#: paragraph the card will truncate.
MAX_TOKENS = 400

#: Off the request path, so this is about not holding a worker rather than
#: about somebody waiting.
TIMEOUT_S = 20.0

_SYSTEM = """You describe a dataset so another agent can decide whether to query it in SQL.

You are given measurements that are already known to be true, and a few sample rows.
Describe only what those support. Never guess at the size, at columns you were not
shown, or at what is in a column that was not loaded.

Reply with bare JSON and nothing else:
{"title": "...", "summary": "...", "topics": ["..."], "good_for": ["..."], "not_for": ["..."]}

title      Three or four words naming the dataset. Not the file, not the host.
summary    One sentence, at most two. What this data is, and what one row represents.
topics     Three to six subjects actually present in the rows.
good_for   Three to five questions this data can answer in SQL. Be concrete and
           name the columns that answer them.
not_for    Two to four questions it cannot answer — grounded in what the
           measurements say is absent, constant, unloaded, or only sampled.
           Never invent a limitation you were not shown evidence for.

Use an empty list rather than filling one out with something you are unsure of."""


def describe(observation: Observation, *, settings: Settings) -> Understanding | None:
    """Name the dataset, or return None and let the card speak in numbers."""
    if not getattr(settings, "dataset_narrate", True):
        return None
    if not llm.ready(settings):
        log.info("no model configured — the dataset profile keeps its measurement only")
        return None
    if not observation.tables:
        return None

    parsed = llm.complete_json(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt(observation)},
        ],
        settings=settings,
        max_tokens=MAX_TOKENS,
        timeout_s=TIMEOUT_S,
    )
    if parsed is None:
        return None

    understanding = Understanding(
        title=_text(parsed.get("title"), 60),
        summary=_text(parsed.get("summary"), 220),
        topics=_list(parsed.get("topics"), 6),
        good_for=_list(parsed.get("good_for"), 5),
        not_for=_list(parsed.get("not_for"), 4),
    )
    return None if understanding.empty else understanding


def prompt(observation: Observation) -> str:
    """The measurement as facts, then a few rows.

    The omitted columns and the sampling are stated here as well as in the
    card, because `not_for` is the field they most usefully constrain: a model
    told that `passages` was not loaded writes "cannot answer questions about
    passage text", which is a genuinely useful limitation and one that nothing
    else in the pipeline would produce.
    """
    lines = [f"Dataset: {observation.location}"]

    for table in observation.tables[:6]:
        head = f"\nTable {table.name}: {table.rows:,} rows"
        if table.sampled and table.total:
            head += f" (the first {table.rows:,} of {table.total:,})"
        lines.append(head)

        for column in table.columns[:20]:
            note = f"  {column.name} ({column.type})"
            if column.constant and column.values:
                note += f" — always {column.values[0]}"
            elif column.values:
                note += " — one of " + ", ".join(column.values)
            elif column.coverage < 0.999:
                note += f" — {column.coverage:.0%} non-null"
            lines.append(note)

        for omitted in table.omitted:
            lines.append(f"  {omitted.name} — present in the source but NOT loaded, not queryable")

    if len(observation.tables) > 6:
        lines.append(f"\n… and {len(observation.tables) - 6} more tables with the same shape.")
    if observation.truncated:
        lines.append(f"{observation.truncated} further files in the source were not loaded.")
    for note in observation.notes:
        lines.append(f"Note: {note}")

    if observation.excerpts:
        lines.append(f"\nSample rows ({len(observation.excerpts)}):")
        for index, excerpt in enumerate(observation.excerpts, 1):
            lines.append(f"{index}. {excerpt}")

    return "\n".join(lines)


def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].strip()


def _list(value: Any, limit: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    cleaned = [_text(item, 90) for item in value]
    return tuple(item for item in cleaned if item)[:limit]
