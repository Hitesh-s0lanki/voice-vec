"""HTTP surface for the latency ring buffer."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from src.services.metrics_service import MetricsService, get_metrics_service

router = APIRouter(prefix="/metrics", tags=["metrics"])

MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]


@router.get("", summary="Live P50/P70/P95/P99/P100 per stage")
def read_metrics(service: MetricsServiceDep) -> dict[str, Any]:
    """Percentiles over the requests still in the buffer, with N and the index.

    A latency number without its N and its index is not a result, so both ride
    along in the response.
    """
    return service.snapshot()


@router.get("/recent", summary="The last few request traces")
def read_recent(
    service: MetricsServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict[str, Any]]:
    """Answers "why did it refuse *that* query?" without re-running it."""
    return service.recent(limit)
