"""The stage runner (docs/05-harness.md).

Everything between transcript and answer is a stage. The runner times each one,
catches each one, and decides whether to retry, fall back, skip or abstain — so
timing and error handling are *structural* rather than something a stage author
remembers to add. That is what makes the `timings` block trustworthy: it cannot
drift out of sync with the code, because the runner produces it.

The runner enforces the **deadline**, not just per-stage timeouts. A chain of
near-misses is how a per-stage budget quietly becomes a blown total.

One honest limitation: these stages are synchronous CPU work, and Python cannot
interrupt that from the outside. Per-stage hard timeouts therefore arrive with
the first network stage (Tier 3 synthesis); until then the deadline is enforced
*between* stages and optional stages are skipped when it has passed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Stage names double as the keys of AskResponse.timings.
STAGE_NAMES = (
    "guard_in",
    "embed",
    "search",
    "rerank",
    "extract",
    "generate",
    "guard_out",
)


class Abstain(Exception):
    """The pipeline ran fine; the corpus cannot support an answer."""

    def __init__(self, reason: str, *, confidence: float = 0.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.confidence = confidence


class Refuse(Exception):
    """The input was rejected at Gate 1."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class Ctx:
    request_id: str
    language: str | None
    effort: int
    deadline_ms: int
    started_at: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def deadline_at(self) -> float:
        return self.started_at + self.deadline_ms / 1000

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def remaining_ms(self) -> float:
        return max(0.0, self.deadline_ms - self.elapsed_ms())

    def overrun(self) -> bool:
        return self.elapsed_ms() > self.deadline_ms


class Harness:
    """Runs stages against one Ctx."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx

    def stage(
        self,
        name: str,
        run: Callable[[], T],
        *,
        retries: int = 0,
        backoff_ms: float = 0.0,
        retry_on: tuple[type[BaseException], ...] = (),
        fallback: Callable[[BaseException], T] | None = None,
        optional: bool = False,
        skip_value: T | None = None,
    ) -> T | None:
        """Run one stage: timed, retried where that is correct, always traced.

        Retries are only correct for network-boundary, idempotent work. A local
        deterministic stage that fails will fail identically on the retry, and
        the retry is paid for out of the deadline — so `retries` defaults to 0.
        """
        if optional and self.ctx.overrun():
            self.ctx.timings[name] = 0.0
            self.ctx.trace.append({"stage": name, "skipped": "deadline"})
            return skip_value

        attempt = 0
        started = time.perf_counter()

        while True:
            try:
                result = run()
                self.ctx.timings[name] = round((time.perf_counter() - started) * 1000, 3)
                if attempt:
                    self.ctx.trace.append({"stage": name, "retries": attempt})
                return result
            except (Abstain, Refuse):
                self.ctx.timings[name] = round((time.perf_counter() - started) * 1000, 3)
                raise
            except BaseException as error:  # noqa: BLE001 — recorded, then handled
                retryable = retries > attempt and isinstance(error, retry_on or ())
                if retryable:
                    attempt += 1
                    self.ctx.trace.append(
                        {"stage": name, "retrying": type(error).__name__, "attempt": attempt}
                    )
                    if backoff_ms:
                        time.sleep(backoff_ms / 1000)
                    continue

                self.ctx.timings[name] = round((time.perf_counter() - started) * 1000, 3)
                self.ctx.trace.append({"stage": name, "error": f"{type(error).__name__}: {error}"})

                if fallback is not None:
                    return fallback(error)
                raise

    def finish(self) -> dict[str, float | None]:
        """Stamp the total. A stage that never ran reports null, not zero."""
        timings: dict[str, float | None] = {
            name: self.ctx.timings.get(name) for name in STAGE_NAMES
        }
        timings["total"] = round(self.ctx.elapsed_ms(), 3)
        return timings
