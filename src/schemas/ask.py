"""The /ask contract (docs/02-architecture.md).

Field names go over the wire in camelCase because the client is TypeScript;
Python keeps snake_case. `timings` is not decoration — it is the raw material
for requirement 4, and the harness fills it structurally.
"""

from typing import Literal

from pydantic import Field

from src.schemas.wire import Wire


class AskRequest(Wire):
    transcript: str = Field(description="What Sarvam heard")
    language_code: str | None = Field(
        default=None, description="Sarvam's detected language, e.g. hi-IN"
    )
    effort: int = Field(
        default=1,
        ge=0,
        le=4,
        description=(
            "EffortPanel index — the *ceiling* on escalation, not a floor. "
            "0 lookup, 1 grounded, 2 deep, 3 corrective, 4 adaptive "
            "(src/rag/effort.py). A question the cache answers costs nothing "
            "at any level; `tier` reports the rung that actually answered."
        ),
    )
    request_id: str | None = Field(default=None, description="Client-supplied trace id")


class Citation(Wire):
    """Where an answer came from — or, on an abstention, what was rejected."""

    doc_id: str
    strategy: str
    score: float
    text: str
    source_query_ids: list[int] = []
    is_gold: bool = Field(
        default=False,
        description="Labelled `is_selected` on at least one origin — eval signal, not shown to users",
    )


class Timings(Wire):
    """Per-stage milliseconds, in pipeline order. null means it did not run.

    Most stages are null on most requests, and that is the point: the shape of
    this block is a readout of which rung ran. A cache hit is `guard_in`,
    `embed`, `cache`, `total` and nothing else.
    """

    guard_in: float | None = None
    embed: float | None = None
    cache: float | None = None
    route: float | None = None
    search: float | None = None
    rerank: float | None = None
    extract: float | None = None
    generate: float | None = None
    grade: float | None = None
    rewrite: float | None = None
    guard_out: float | None = None
    total: float


class AskResponse(Wire):
    status: Literal["answered", "abstained", "refused"] = Field(
        description=(
            "`abstained` is a success, not an error: the pipeline ran and the "
            "corpus could not support an answer. `refused` means Gate 1 "
            "rejected the input."
        )
    )
    answer: str | None
    citations: list[Citation]
    confidence: float
    tier: int = Field(description="Which tier actually produced this")
    reason: str | None = Field(description="Why it abstained or refused — user-facing")
    timings: Timings
    request_id: str
    language: str | None = Field(default=None, description="FLORES code the index was filtered on")
    flags: list[str] = Field(default=[], description="Gate 1 findings, e.g. `injection`")
    method: str | None = Field(
        default=None,
        description=(
            "How the answer was produced: extractive `embedding`/`lexical`, "
            "`synthesis`, or `cache` when it was served from Redis"
        ),
    )
    mode: str = Field(
        default="grounded",
        description="The rung that was asked for — lookup, grounded, deep, corrective, adaptive",
    )
    cached: bool = Field(
        default=False, description="Served from the answer cache rather than computed"
    )
    backend: str | None = Field(
        default=None,
        description="Which vector store answered — the deployment's, or one the user connected",
    )
    escalations: list[str] = Field(
        default=[],
        description=(
            "What the pipeline did beyond the straight line: `hybrid`, `rerank`, "
            "`rewrite`, `regenerate`, `dense-only`. The audit trail for a rung "
            "that reports a tier different from the one asked for."
        ),
    )
    budget_ms: int = Field(
        default=200,
        description=(
            "The deadline this rung is measured against. Rungs 0-1 hold the "
            "200 ms of requirement 3; the upper rungs make network calls and "
            "are reported against their own budget rather than that one."
        ),
    )
    within_budget: bool = Field(description="Did the pipeline finish inside `budget_ms`")
