"""Live latency percentiles, from the timings the harness already produces.

Requirement 4 asks for P50/P70/P100 over a reasonable number of queries. The
harness times every stage of every request anyway, so the only thing missing is
somewhere to put them: a ring buffer, aggregated on read.

P100 is a single sample and its expected value grows with N, so N is reported
alongside every percentile and P95/P99 sit next to it as the stable tail
statistics (docs/04-latency.md).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Any, Iterable

from src.core.config import Settings, get_settings
from src.rag.harness import STAGE_NAMES
from src.schemas.ask import AskResponse

PERCENTILES = (50, 70, 95, 99, 100)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation — every figure is a real sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil(p / 100 * len(ordered))))
    return round(ordered[rank - 1], 3)


class MetricsService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: deque[dict[str, Any]] = deque(maxlen=settings.metrics_buffer)
        self._lock = threading.Lock()

    def record(self, response: AskResponse, trace: list[dict[str, Any]]) -> None:
        entry = {
            "at": time.time(),
            "status": response.status,
            "tier": response.tier,
            # The rung asked for, beside the rung that answered. Aggregating on
            # `tier` alone hides the whole point of the ladder: a `mode: deep`
            # request answered at `tier: 1` is a synthesis that fell back, and
            # a run of them is the signal that the model key is wrong.
            "mode": response.mode,
            "cached": response.cached,
            "escalations": response.escalations,
            "backend": response.backend,
            "confidence": response.confidence,
            "budget_ms": response.budget_ms,
            "within_budget": response.within_budget,
            "timings": response.timings.model_dump(),
            "trace": trace,
        }
        with self._lock:
            self._records.append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)

        totals = [r["timings"]["total"] for r in records]

        return {
            "n": len(records),
            "buffer": self._settings.metrics_buffer,
            "deadline_ms": self._settings.deadline_ms,
            "within_budget": sum(1 for r in records if r["within_budget"]),
            "by_status": _counts(r["status"] for r in records),
            "by_mode": _counts(r.get("mode", "grounded") for r in records),
            # Cache hits are the one number that makes the upper rungs
            # affordable, and it is invisible in the latency percentiles
            # because a hit *is* the fast path being fast.
            "cached": sum(1 for r in records if r.get("cached")),
            "modes": {
                mode: _distribution(
                    [r["timings"]["total"] for r in records if r.get("mode") == mode]
                )
                for mode in sorted({r.get("mode", "grounded") for r in records})
            },
            "stages": {
                stage: _distribution(
                    [
                        r["timings"][stage]
                        for r in records
                        if r["timings"].get(stage) is not None
                    ]
                )
                for stage in (*STAGE_NAMES, "total")
            },
            "total": _distribution(totals),
            # No "index" here any more. Every record already carries the
            # `backend` that answered it, which is the honest attribution now
            # that two requests in the same buffer can have been served by two
            # different people's stores.
            "by_backend": _counts(r.get("backend") or "none" for r in records),
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)[-limit:]


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        **{f"p{p}": percentile(values, p) for p in PERCENTILES},
        "mean": round(sum(values) / len(values), 3) if values else 0.0,
    }


@lru_cache
def get_metrics_service() -> MetricsService:
    return MetricsService(get_settings())
