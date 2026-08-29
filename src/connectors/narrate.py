"""One model call that turns a measurement into something an agent can route on.

`Observation` knows a store holds 2,366 records of ~3,150 characters in Latin
script with a `book_id` field on all of them. That is everything except the one
thing a router needs: **what is it about**. No amount of counting fields
produces "twelve popular self-help and business books", and an agent that
cannot say that cannot decide whether a question belongs here.

So this is the one place a model is allowed near the profile, and it is fenced
in three ways:

  **It is handed the measurement, not asked to make one.** Counts, dimensions,
  field coverage and scripts are already known and go into the prompt as facts.
  The model is only asked to read the excerpts and name the subject.

  **Its output never gates anything.** `Understanding` steers routing;
  `CapabilityFacts` decides whether a filter is applied. A hallucinated topic
  costs a wasted retrieval. A hallucinated capability costs a silently empty
  result set, which is why the model is nowhere near that half.

  **Failure is empty, never a default.** No key, a timeout, unparseable JSON —
  all return `None`, the profile keeps its measurement, and the card renders
  from numbers alone. The tempting alternative, a generic summary on failure,
  puts a sentence nobody measured in front of every routing decision.

**The excerpts are the user's own data and they leave for a model provider.**
That is a real disclosure, not a footnote: a deployment that cannot make it
sets `profile_narrate = False` and keeps the measurement-only card, which
degrades the routing hints and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.connectors.profile import Observation, Understanding
from src.core.config import Settings
from src.rag import llm

log = logging.getLogger("vec.connectors.narrate")

#: Small. The output is six short fields and a summary of two sentences; room
#: for more is room for a model to write a paragraph the card will truncate.
MAX_TOKENS = 400

#: This runs off the request path, so the ceiling is about not holding a worker
#: rather than about a listener waiting.
TIMEOUT_S = 20.0

_SYSTEM = """You describe a document store so another agent can decide whether to search it.

You are given measurements that are already known to be true, and a few excerpts.
Describe only what the excerpts and measurements support. Never guess at the size,
the coverage, or fields you were not shown.

Reply with bare JSON and nothing else:
{"title": "...", "summary": "...", "topics": ["..."], "good_for": ["..."], "not_for": ["..."], "languages": ["..."]}

title      Three or four words naming the corpus. Not the service, not the table.
summary    At most two sentences. What this collection is, and what it is made of.
topics     Three to six subjects actually present in the excerpts. One or two words each.
good_for   Three to five kinds of question this store can answer well. At most six words each.
not_for    Two to four kinds of question it cannot answer — at most six words each, grounded in what is
           missing or bounded, such as a subject outside the excerpts, a language
           the store is not in, or a field the measurements say is absent or
           constant. Never invent a limitation you were not shown evidence for.
languages  The languages you can actually read in the excerpts.

Use an empty list rather than filling one out with something you are unsure of."""


def describe(observation: Observation, *, settings: Settings) -> Understanding | None:
    """Name the corpus, or return None and let the card speak in numbers."""
    if not getattr(settings, "profile_narrate", True):
        return None
    if not llm.ready(settings):
        log.info("no model configured — profile keeps its measurement only")
        return None
    if observation.kind == "tools":
        return _tools(observation)
    if not observation.excerpts:
        # Nothing to read. A store with no readable text can still be searched,
        # but nothing here could say what it is about without inventing it.
        return None

    parsed = llm.complete_json(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _prompt(observation)},
        ],
        settings=settings,
        max_tokens=MAX_TOKENS,
        timeout_s=TIMEOUT_S,
    )
    if parsed is None:
        return None

    understanding = Understanding(
        title=_text(parsed.get("title"), 60),
        summary=_text(parsed.get("summary"), 320),
        topics=_list(parsed.get("topics"), 6),
        good_for=_list(parsed.get("good_for"), 5, chars=72),
        not_for=_list(parsed.get("not_for"), 4, chars=72),
        languages=_list(parsed.get("languages"), 4),
    )
    return None if understanding.empty else understanding


def _prompt(observation: Observation) -> str:
    """The measurement as facts, then the excerpts.

    Field coverage is quoted as a percentage rather than a raw count because
    the model does occasionally reason about it in `not_for` — "no page
    numbers" is a genuinely useful limitation and it is only visible as a field
    that is absent or constant.
    """
    shape = observation.vectors
    lines = [f"Store: {observation.location}"]

    if shape:
        if shape.records is not None:
            lines.append(f"Records: {shape.records:,}")
        if shape.dimensions:
            lines.append(f"Vectors: {shape.dimensions}-dimensional, {shape.metric or 'unknown'} distance")
    if observation.text_chars:
        low, mean, high = observation.text_chars
        lines.append(f"Passage length: {low}–{high} characters, {mean} typical")
    if observation.scripts:
        lines.append(f"Writing systems seen: {', '.join(observation.scripts)}")

    present = [
        f"{f.name} ({f.coverage:.0%}"
        + (f", always {f.examples[0]}" if f.constant and f.examples else "")
        + ")"
        for f in observation.fields[:12]
        if f.name != observation.text_field
    ]
    if present:
        lines.append(f"Metadata fields: {', '.join(present)}")
    for note in observation.notes:
        lines.append(f"Note: {note}")

    lines.append(f"\nExcerpts ({len(observation.excerpts)} of {observation.sampled} sampled records):")
    for index, excerpt in enumerate(observation.excerpts, 1):
        lines.append(f"{index}. {excerpt}")

    return "\n".join(lines)


def _tools(observation: Observation) -> Understanding | None:
    """A tools connector is described from its own facts, with no model call.

    "The agent can send mail through Gmail" is not an inference — it is the
    authorised-toolkit list, rephrased. Spending a round trip and a provider's
    judgement on a rename would only add a way for it to be wrong.
    """
    tools = observation.tools
    if tools is None:
        return None

    authorised = list(tools.authorised)
    if not authorised:
        return Understanding(
            title="Composio (nothing authorised)",
            summary="Composio is connected but no toolkit has been authorised, so no tool can run yet.",
            not_for=("anything requiring an outside service",),
        )

    names = ", ".join(authorised[:6])
    return Understanding(
        title="Composio tools",
        summary=f"Actions can run against {names} through this user's own Composio project.",
        topics=tuple(authorised[:6]),
        good_for=tuple(f"acting in {name}" for name in authorised[:4]),
        not_for=("services that are not authorised here",),
    )


def _text(value: Any, limit: int) -> str:
    """Clamped at a word boundary, not at a character index.

    A model's list item cut mid-word — "matching passages by meaning rather
    than exact w" — reads to the next model as a phrase that means something
    slightly different from the one that was written. The card already drops
    whole lines rather than cutting one; this is the same rule one level down.
    """
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text

    clipped = text[:limit]
    spaced = clipped.rsplit(" ", 1)[0]
    # Only honour the boundary if it leaves most of the value; a single very
    # long word would otherwise clamp to nothing.
    kept = (spaced if len(spaced) >= limit * 0.6 else clipped).strip().rstrip(",;:")

    # And do not end on a word that promises a continuation.
    words = kept.split()
    while len(words) > 1 and words[-1].lower().strip(",;:") in _DANGLING:
        words.pop()
    return " ".join(words)


def _list(value: Any, limit: int, *, chars: int = 48) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    cleaned = [_text(item, chars) for item in value]
    return tuple(item for item in cleaned if item)[:limit]


#: Words a clamped phrase must not end on. A list item cut to "searching for
#: entrepreneurship and" is worse than the same item one word shorter: it reads
#: as a phrase whose second half was lost, which is exactly what happened.
_DANGLING = frozenset(
    {"and", "or", "the", "a", "an", "of", "for", "with", "to", "in", "on",
     "that", "since", "about", "by", "from", "as", "at", "into"}
)
