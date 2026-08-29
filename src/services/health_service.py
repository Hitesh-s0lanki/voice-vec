"""Health logic. Controllers stay thin; the checking lives here."""

import time

from src.core.config import Settings, get_settings
from src.rag.embed import Embedder, get_embedder
from src.schemas.health import HealthResponse

_STARTED_AT = time.monotonic()


class HealthService:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self._settings = settings
        self._embedder = embedder

    def check(self) -> HealthResponse:
        """Report process health and the dependencies this build actually has.

        There is no index to count. A question is answered from the vector
        store its asker connected (docs/13-connectors.md), which is per-user,
        credentialed, and hosted by somebody else — so a process-wide endpoint
        cannot speak for it, and a load balancer must not be told this server
        is unhealthy because one account attached an empty Pinecone. Whether a
        given user has somewhere to search is answered by `/connectors`, under
        their own identity.

        Degraded rather than raising: a health endpoint that 500s tells a load
        balancer nothing.
        """
        # Both halves, because this build needs both: a model to write the
        # reply and, for anyone who attached a store, an embedder to search it
        # with. There is no longer a switch that makes one of them optional.
        ready = self._embedder.ready and self._settings.resolve_llm().ready

        return HealthResponse(
            status="ok" if ready else "degraded",
            service=self._settings.app_name,
            version=self._settings.version,
            environment=self._settings.environment,
            uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
            embedder=self._embedder.model_name,
            embedder_ready=self._embedder.ready,
        )


def get_health_service() -> HealthService:
    return HealthService(get_settings(), get_embedder())
