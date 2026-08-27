"""Searching a Pinecone index the user connected.

Two planes, which is the one thing about Pinecone that has to be got right.
`api.pinecone.io` is the control plane, where an index is described; queries go
to a **host of that index's own**, which the control plane hands out. Pinecone's
own guidance is to look the host up once and cache it, and that is what
`_host()` does — otherwise every search pays two round trips inside a 200 ms
budget that can barely afford one.

Nothing here writes. This app does not create anybody's index and does not
ingest into it: a connected Pinecone is one the user populated, and the schema
it has to satisfy is the metadata read in `_hit` below.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import httpx
import numpy as np

from src.connectors.registry import (
    PINECONE_API_VERSION,
    pinecone_host,
)
from src.rag.backends.base import Capabilities, Hit, StoreUnavailable

log = logging.getLogger("vec.rag.pinecone")

# The answer path's budget is 200 ms end to end. A hosted index that has not
# answered in two seconds has already lost the request; failing fast lets the
# harness degrade instead of holding a worker.
QUERY_TIMEOUT_S = 2.0


class PineconeBackend:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        self._api_key = credentials["api_key"]
        self._index = credentials["index"]
        self._namespace = (credentials.get("namespace") or "").strip()
        self._cached_host: str | None = None

    @property
    def name(self) -> str:
        return "pinecone"

    def describe(self) -> str:
        where = f"pinecone/{self._index}"
        return f"{where}#{self._namespace}" if self._namespace else where

    def _host(self) -> str:
        if self._cached_host is None:
            self._cached_host = pinecone_host(self._api_key, self._index)
        return self._cached_host

    def capabilities(self) -> Capabilities:
        """Dense only, and `filters` is a hope rather than a guarantee.

        Pinecone does have sparse vectors, but only in an index created as
        sparse or dense-sparse, and this app did not create this index and
        cannot change its type. Claiming a lexical channel here would mean rung
        2 asking for one that returns nothing on most connected indexes.

        `filters` stays true because the metadata predicate is always sent and
        always honoured — on an index whose metadata lacks `strategy` it simply
        matches everything, which is the right answer for someone who ingested
        their own documents without this app's chunking vocabulary.

        `parallel_text` is false: the English original beside each Indic
        passage is something this app's ingest writes, so a cross-lingual
        question against a connected index is answered from whatever text that
        index holds.
        """
        return Capabilities(lexical=False, filters=True, parallel_text=False)

    def ready(self) -> bool:
        try:
            self._host()
            return True
        except Exception as error:
            log.debug("pinecone not ready: %s", error)
            return False

    def search(
        self,
        vector: np.ndarray,
        *,
        strategies: Sequence[str],
        limit: int,
        language: str | None = None,
    ) -> list[Hit]:
        body: dict[str, Any] = {
            "vector": [float(v) for v in np.asarray(vector, dtype=np.float32).ravel()],
            "topK": int(limit),
            "includeMetadata": True,
            "includeValues": False,
        }
        if self._namespace:
            body["namespace"] = self._namespace

        # Strategy and language are metadata filters rather than separate
        # queries: Pinecone charges per query and one filtered call is both
        # cheaper and lower latency than one call per strategy. An index whose
        # metadata has neither field simply matches everything, which is the
        # right behaviour for a user who ingested their own documents without
        # this app's chunking vocabulary.
        conditions = []
        if strategies:
            conditions.append({"strategy": {"$in": list(strategies)}})
        if language:
            conditions.append({"language": {"$eq": language}})
        if conditions:
            body["filter"] = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        try:
            response = httpx.post(
                f"https://{self._host()}/query",
                headers={
                    "Api-Key": self._api_key,
                    "X-Pinecone-Api-Version": PINECONE_API_VERSION,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=QUERY_TIMEOUT_S,
            )
            response.raise_for_status()
        except Exception as error:
            # A host that has moved is the one failure worth retrying, and it
            # only shows up as a connection error. Drop the cache so the next
            # search looks it up again rather than failing forever.
            self._cached_host = None
            raise StoreUnavailable(f"pinecone: {error}") from error

        matches = (response.json() or {}).get("matches") or []
        return [self._hit(match) for match in matches]

    def _hit(self, match: Mapping[str, Any]) -> Hit:
        """One match, read defensively.

        Everything except the id and score is somebody else's metadata, written
        by whatever populated their index. Missing fields become empty rather
        than raising: a hit with no strategy label is still a hit, and a
        KeyError here would take out the whole answer.
        """
        meta = dict(match.get("metadata") or {})
        return Hit(
            chunk_id=str(match.get("id") or ""),
            strategy=str(meta.get("strategy") or ""),
            score=float(match.get("score") or 0.0),
            text=str(meta.get("text") or ""),
            payload=meta,
        )
