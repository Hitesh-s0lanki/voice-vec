"""The rungs, and what each one is allowed to do (docs/15-effort.md).

One place that knows the ladder, because three others need to agree about it:
the pipeline dispatches on it, the response reports it, and the panel in
`frontend/src/lib/effort.ts` renders it. A rung added here and forgotten in one
of those is a slider position that silently does something else.

**The level is a ceiling, not a floor.** Asking for rung 4 does not mean four
LLM calls happen; it means up to four may. A question the cache answers is
answered from the cache at rung 4 exactly as it would be at rung 1, and
`AskResponse.tier` reports the rung that actually produced the answer while
`mode` reports the one that was asked for. Anything else would make the slider
a way to spend money rather than a way to buy quality.
"""

from __future__ import annotations

from typing import Final

#: Search only. No LLM anywhere on the path, at any point, for any reason.
LOOKUP: Final = 0
#: Extractive answering plus the answer cache. The 200 ms tier.
GROUNDED: Final = 1
#: Hybrid retrieval, passage rerank, and one LLM synthesis over what survived.
DEEP: Final = 2
#: Corrective: grade the retrieval, rewrite the query, retrieve once more.
CORRECTIVE: Final = 3
#: Adaptive: route before retrieving, then a capped repair loop after.
ADAPTIVE: Final = 4

NAMES: Final[tuple[str, ...]] = (
    "lookup",
    "grounded",
    "deep",
    "corrective",
    "adaptive",
)

#: Above this rung the pipeline makes network calls and cannot hold the 200 ms
#: budget of requirement 3. Used to decide what to *report*, not what to run —
#: the rung's own deadline (`Settings.deadline_for`) is what the harness
#: enforces.
OFFLINE_MAX: Final = GROUNDED


def name(level: int) -> str:
    """The rung's name, for the response and the metrics. Never raises."""
    if 0 <= level < len(NAMES):
        return NAMES[level]
    return f"level-{level}"


def uses_llm(level: int) -> bool:
    return level >= DEEP


def uses_hybrid(level: int) -> bool:
    """Whether to ask for a lexical channel. The backend decides if it has one."""
    return level >= DEEP


def uses_rerank(level: int) -> bool:
    return level >= DEEP


def uses_grading(level: int) -> bool:
    """Corrective RAG's relevance grader, and everything built on it."""
    return level >= CORRECTIVE


def uses_routing(level: int) -> bool:
    """Adaptive RAG's pre-retrieval router."""
    return level >= ADAPTIVE
