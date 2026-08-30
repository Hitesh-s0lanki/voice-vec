"""Searching a DataStax Astra collection the user connected.

Astra's Data API is one POST per command against
`{endpoint}/api/json/v1/{keyspace}/{collection}`, with the command as the
single top-level key of the body. A vector search is `find` with the query
vector in `sort`.

The trap, and the reason `_check` exists: **the Data API answers 200 with an
`errors` array**. A bad token, a missing collection and a malformed filter all
arrive as a successful HTTP response, so `raise_for_status` alone would read
every one of them as an empty result set — a retrieval that silently returns
nothing is worse than one that fails.

Nothing here writes. A connected Astra is one the user populated.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import httpx
import numpy as np

from src.connectors.registry import astra_url
from src.rag.backends.base import Capabilities, Hit, StoreUnavailable

log = logging.getLogger("vec.rag.astra")

QUERY_TIMEOUT_S = 2.0


class AstraBackend:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        self._token = credentials["token"]
        self._endpoint = credentials["endpoint"].rstrip("/")
        self._keyspace = credentials["keyspace"]
        self._collection = credentials["collection"]
        self._dim = int(credentials.get("dim") or 0)

    @property
    def name(self) -> str:
        return "astra"

    def describe(self) -> str:
        return f"astra/{self._keyspace}.{self._collection}"


    def embed_query(self, text: str) -> np.ndarray:
        """Embedded at *this* store's width, whatever that is.

        `text-embedding-3` can be asked for exactly the number of dimensions
        the store's own catalogue reported, so there is one embedder and one
        call — the branch that used to pick a local model when the widths
        happened to agree went with the local model itself
        (docs/25-no-local-embedder.md).
        """
        from src.rag.embed import get_embedder
        from src.rag.remote_embed import RemoteEmbedUnavailable

        try:
            return get_embedder().embed_query(text, dim=self._dim or None)
        except RemoteEmbedUnavailable as error:
            # `StoreUnavailable` is what the ladder already abstains on, so a
            # provider outage — or a missing key — degrades to "my sources are
            # unavailable" rather than a 500 from a layer nobody can place.
            raise StoreUnavailable(str(error)) from error

    def _post(self, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            response = httpx.post(
                astra_url(self._endpoint, self._keyspace, self._collection),
                headers={"Token": self._token, "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
        except Exception as error:
            raise StoreUnavailable(f"astra: {error}") from error

        return self._check(response.json() or {})

    def _check(self, body: dict[str, Any]) -> dict[str, Any]:
        """A 200 from the Data API is not the same thing as success."""
        errors = body.get("errors")
        if errors:
            message = str(errors[0].get("message", "")).strip() or "unknown error"
            raise StoreUnavailable(f"astra: {message}")
        return body

    def capabilities(self) -> Capabilities:
        """Dense only — see the note in the Pinecone twin.

        Astra's newer collections can carry a `$lexical` field and do hybrid
        server-side, but only when the collection was created with it; a
        collection this app did not create almost certainly was not, and asking
        for a field that does not exist is one of the errors the Data API
        returns inside a 200.
        """
        return Capabilities(lexical=False, filters=True, parallel_text=False)

    def ready(self) -> bool:
        try:
            self._post({"countDocuments": {}}, timeout=QUERY_TIMEOUT_S)
            return True
        except Exception as error:
            log.debug("astra not ready: %s", error)
            return False

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        find: dict[str, Any] = {
            "sort": {"$vector": [float(v) for v in np.asarray(vector, dtype=np.float32).ravel()]},
            "options": {"limit": int(limit), "includeSimilarity": True},
        }

        # Same reasoning as Pinecone's filter: one filtered query rather than
        # one per strategy. A collection without these fields matches
        # everything, which is right for a user who ingested their own way.
        conditions = []
        if strategies:
            conditions.append({"strategy": {"$in": list(strategies)}})
        if language:
            conditions.append({"language": language})
        if conditions:
            find["filter"] = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        body = self._post({"find": find}, timeout=QUERY_TIMEOUT_S)
        documents = ((body.get("data") or {}).get("documents")) or []
        return [self._hit(doc) for doc in documents]

    def _hit(self, doc: Mapping[str, Any]) -> Hit:
        """One document, read defensively — see the note in the Pinecone twin.

        `$vector` is dropped from the payload: it is hundreds of floats that
        nothing downstream reads, and carrying it into `Hit.payload` would put
        it in every log line and every rendered answer's context.
        """
        payload = {k: v for k, v in doc.items() if k != "$vector"}
        return Hit(
            chunk_id=str(doc.get("_id") or ""),
            strategy=str(doc.get("strategy") or ""),
            score=float(doc.get("$similarity") or 0.0),
            text=str(doc.get("text") or ""),
            payload=payload,
        )
