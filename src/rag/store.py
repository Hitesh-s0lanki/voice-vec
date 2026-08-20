"""Qdrant — one collection, one named vector per chunking strategy.

Named vectors are how all five strategies from docs/03-chunking.md live side by
side and stay queryable independently. v1 populates `S1` only; the other four
slots are declared now so adding a strategy is an ingest run, not a re-index.

Two modes:
  * embedded  — `QdrantClient(path=...)`, no Docker. Single writer: the ingest
    script and the API cannot hold the same path at once.
  * server    — `QDRANT_URL=http://localhost:6333`, both can run together.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from src.core.config import Settings, get_settings
from src.rag.chunk import ALL_STRATEGIES, Chunk

# Deterministic point ids: re-running ingest overwrites rather than duplicates,
# which is a better resume story than a checkpoint file at this corpus size.
_NAMESPACE = uuid.UUID("6f2b1a3c-9d4e-4b7a-8c1f-0d5e2a7b3c94")


@dataclass(slots=True)
class Hit:
    chunk_id: str
    strategy: str
    score: float
    text: str
    payload: dict


class StoreUnavailable(RuntimeError):
    """Qdrant is unreachable, or the collection has not been ingested yet."""


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: QdrantClient | None = None
        # QdrantLocal is sqlite-backed and not thread-safe; FastAPI runs sync
        # handlers in a threadpool, so calls into it are serialised here.
        self._lock = threading.Lock()
        # A *separate* lock for building the client. Guarding both with one lock
        # deadlocks the first caller that takes `_lock` and then touches the
        # lazily-built `client` inside it — Lock is not reentrant.
        self._connect_lock = threading.Lock()

    @property
    def embedded(self) -> bool:
        return not self._settings.qdrant_url

    @property
    def location(self) -> str:
        return self._settings.qdrant_url or f"embedded:{self._settings.qdrant_path}"

    @property
    def collection(self) -> str:
        return self._settings.qdrant_collection

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._connect_lock:
                if self._client is None:
                    self._client = self._connect()
        return self._client

    def _connect(self) -> QdrantClient:
        if self._settings.qdrant_url:
            return QdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key or None,
            )
        return QdrantClient(path=self._settings.qdrant_path)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- index management (ingest) --------------------------------------

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)

        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                strategy: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
                for strategy in ALL_STRATEGIES
            },
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, chunk.chunk_id)),
                vector={chunk.strategy: vector.tolist()},
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client = self.client
        with self._lock:
            client.upsert(self.collection, points=points, wait=True)

    # ---- query ----------------------------------------------------------

    def warm(self) -> int:
        """Open the connection and confirm the collection has points."""
        if not self.client.collection_exists(self.collection):
            raise StoreUnavailable(
                f"collection {self.collection!r} does not exist — run scripts/ingest.py"
            )
        return self.count()

    def count(self) -> int:
        client = self.client
        with self._lock:
            return client.count(self.collection, exact=True).count

    def ready(self) -> bool:
        try:
            return self.count() > 0
        except Exception:
            return False

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        """Dense search, one query per strategy, merged by score.

        v1 is dense-only. Sparse vectors and RRF fusion are the Phase B half of
        requirement 2 (docs/03-chunking.md) and slot in here.
        """
        query = vector.tolist()
        query_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language),
                    )
                ]
            )
            if language
            else None
        )

        hits: dict[str, Hit] = {}
        client = self.client
        with self._lock:
            for strategy in strategies:
                try:
                    points = client.query_points(
                        self.collection,
                        query=query,
                        using=strategy,
                        limit=limit,
                        query_filter=query_filter,
                        with_payload=True,
                    ).points
                except Exception as error:  # connection, missing collection, …
                    raise StoreUnavailable(str(error)) from error

                for point in points:
                    payload = point.payload or {}
                    chunk_id = str(payload.get("chunkId", point.id))
                    existing = hits.get(chunk_id)
                    if existing is not None and existing.score >= point.score:
                        continue
                    hits[chunk_id] = Hit(
                        chunk_id=chunk_id,
                        strategy=str(payload.get("strategy", strategy)),
                        score=float(point.score),
                        text=str(payload.get("text", "")),
                        payload=payload,
                    )

        return sorted(hits.values(), key=lambda h: h.score, reverse=True)[:limit]


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(get_settings())
