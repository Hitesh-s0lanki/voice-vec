"""HTTP surface for health checks."""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.schemas.health import HealthResponse
from src.services.health_service import HealthService, get_health_service

router = APIRouter(prefix="/health", tags=["health"])

HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]


@router.get("", response_model=HealthResponse, summary="Liveness and readiness")
def get_health(service: HealthServiceDep) -> HealthResponse:
    return service.check()
