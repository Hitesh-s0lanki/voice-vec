"""Response models for the health endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="ok when every checked dependency is reachable",
    )
    service: str
    version: str
    environment: str
    uptime_seconds: float = Field(description="Seconds since the process started")
    embedder: str = Field(description="Embedding model, or the reason it isn't loaded")
    embedder_ready: bool
