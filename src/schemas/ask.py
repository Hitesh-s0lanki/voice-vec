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
        le=3,
        description=(
            "EffortPanel index — the ceiling on escalation. v1 only implements "
            "Tier 1, so this is recorded but does not change the path yet."
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
    """Per-stage milliseconds. null means the stage did not run."""

    guard_in: float | None = None
    embed: float | None = None
    search: float | None = None
    rerank: float | None = None
    extract: float | None = None
    generate: float | None = None
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
        default=None, description="How the span was chosen: embedding or lexical"
    )
    within_budget: bool = Field(description="Did the pipeline finish inside the 200 ms SLO")
