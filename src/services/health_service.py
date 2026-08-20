"""Health logic. Controllers stay thin; the checking lives here."""

import time

from src.core.config import Settings, get_settings
from src.rag.embed import Embedder, get_embedder
from src.rag.manifest import read_manifest
from src.rag.store import VectorStore, get_store
from src.schemas.health import HealthResponse

_STARTED_AT = time.monotonic()


class HealthService:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def check(self) -> HealthResponse:
        """Report process health, plus the two dependencies /ask needs.

        Degraded rather than raising: a health endpoint that 500s tells a load
        balancer nothing, and "the index is empty" is exactly the state you
        want to see reported before you go looking for it in the logs.
        """
        # Counting means connecting, and with retrieval off that would open
        # the Qdrant path this build never reads — and lock it against ingest.
        if self._settings.rag_enabled:
            try:
                chunks = self._store.count()
            except Exception:
                chunks = 0
            ready = self._embedder.ready and chunks > 0
        else:
            chunks = 0
            ready = self._settings.resolve_llm().ready

        return HealthResponse(
            status="ok" if ready else "degraded",
            service=self._settings.app_name,
            version=self._settings.version,
            environment=self._settings.environment,
            uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
            embedder=self._embedder.model_name,
            embedder_ready=self._embedder.ready,
            index=self._store.location if self._settings.rag_enabled else "disabled",
            chunks=chunks,
            manifest=read_manifest() if self._settings.rag_enabled else None,
        )


def get_health_service() -> HealthService:
    return HealthService(get_settings(), get_embedder(), get_store())
