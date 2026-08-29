"""Reading an Astra collection, which — unlike Pinecone — will page through itself.

Astra's Data API has a real `find`, so the sample here is a genuine scan of the
first N documents rather than a union of neighbourhoods. That is a better
sample in one way and a worse one in another: it is unbiased with respect to
the embedding space and completely biased with respect to insertion order. A
collection loaded one corpus at a time has its first 200 documents all from the
first corpus, so the probe pages a little wider than it needs and says what it
did.

The trap this shares with `src/rag/backends/astra.py` is worth restating
because it is the single easiest way to get a wrong profile: **the Data API
answers 200 with an `errors` array**. A bad token, a missing collection and an
unsupported option all arrive as successful HTTP responses. A probe that
trusted `raise_for_status` would read every one of them as "an empty
collection" and write a profile saying the user's store holds no documents —
which is not a failure the user would ever think to question.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import httpx

from src.connectors.probes.base import (
    PROBE_TIMEOUT_S,
    SAMPLE_SIZE,
    excerpts_of,
    field_stats,
    length_stats,
    pick_text_field,
    scripts_of,
    text_of,
    unreachable,
)
from src.connectors.profile import Observation, VectorShape

log = logging.getLogger("vec.connectors.probe")

#: `countDocuments` refuses without a ceiling and errors when the collection
#: exceeds it. Asking for a large one and reading the error as "bigger than
#: this" is how an exact count and an honest lower bound come from one call.
COUNT_CEILING = 1_000_000

#: The Data API caps a page well below the sample size, so the probe follows
#: `nextPageState`. Bounded: a collection with a million documents must not
#: turn a profile into a full scan of somebody else's store.
MAX_PAGES = 10


class AstraProbe:
    def __init__(self, credentials: Mapping[str, str], *, excerpts: bool = True, **_: Any) -> None:
        self._token = credentials["token"]
        self._endpoint = credentials["endpoint"].rstrip("/")
        self._keyspace = credentials["keyspace"]
        self._collection = credentials["collection"]
        self._excerpts = excerpts

    @property
    def location(self) -> str:
        return f"astra/{self._keyspace}.{self._collection}"

    def _post(self, body: dict[str, Any], *, collection: bool = True) -> dict[str, Any]:
        from src.connectors.registry import astra_url

        response = httpx.post(
            astra_url(self._endpoint, self._keyspace, self._collection if collection else ""),
            headers={"Token": self._token, "Content-Type": "application/json"},
            json=body,
            timeout=PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
        parsed = response.json() or {}

        errors = parsed.get("errors")
        if errors:
            message = str(errors[0].get("message", "")).strip() or "unknown error"
            raise RuntimeError(message)
        return parsed

    def observe(self) -> Observation:
        started = time.perf_counter()
        notes: list[str] = []

        try:
            documents = self._sample()
        except Exception as error:
            log.info("astra probe failed: %s", type(error).__name__)
            return unreachable(
                "astra", "vector", self.location, f"the collection did not answer: {error}"
            )

        dimensions, metric = self._shape()
        records, exact = self._count()
        if not exact and records is not None:
            notes.append(f"more than {records:,} documents — the count is a floor")

        text_field = pick_text_field(documents)
        texts = text_of(documents, text_field)
        stats = field_stats(documents)
        if not text_field:
            notes.append("no text field found — hits from here cannot be quoted")
        if documents:
            notes.append("sampled in insertion order, not uniformly")

        # `$vectorize` means Astra embeds server-side from text the user sends.
        # An index like that has its own embedding model, which is almost
        # certainly not this app's — worth flagging before a search returns
        # neighbours of a vector computed by something else entirely.
        if any("$vectorize" in doc for doc in documents):
            notes.append("collection uses Astra's server-side $vectorize embeddings")

        return Observation(
            connector="astra",
            kind="vector",
            location=self.location,
            reachable=True,
            sampled=len(documents),
            vectors=VectorShape(
                dimensions=dimensions,
                metric=metric,
                index="astra",
                normalised=None,
                records=records,
            ),
            fields=stats,
            text_field=text_field,
            text_chars=length_stats(texts),
            scripts=scripts_of(texts),
            excerpts=excerpts_of(documents, text_field, stats=stats, allowed=self._excerpts),
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes[:6]),
        )

    def _sample(self) -> list[dict[str, Any]]:
        """Documents, paged, with the vector stripped before it reaches analysis.

        `$vector` is hundreds of floats per document. Projecting it away at the
        server means the probe does not pull megabytes to learn nothing — the
        dimension already came from the collection's own options.
        """
        documents: list[dict[str, Any]] = []
        state: str | None = None

        for _ in range(MAX_PAGES):
            options: dict[str, Any] = {"limit": SAMPLE_SIZE - len(documents)}
            if state:
                options["pageState"] = state

            body = self._post(
                {"find": {"projection": {"$vector": 0}, "options": options}}
            )
            data = body.get("data") or {}
            page = data.get("documents") or []
            documents.extend({k: v for k, v in doc.items() if k != "$vector"} for doc in page)

            state = data.get("nextPageState")
            if not state or len(documents) >= SAMPLE_SIZE:
                break

        return documents[:SAMPLE_SIZE]

    def _shape(self) -> tuple[int | None, str]:
        """Dimension and metric from the collection's own definition.

        A keyspace-level command, so it works even when the sample came back
        empty — an empty collection still has a geometry, and "0 documents at
        1536 dimensions" is a much more useful thing to tell an agent than
        "0 documents, shape unknown".
        """
        try:
            body = self._post({"findCollections": {"options": {"explain": True}}}, collection=False)
        except Exception as error:
            log.debug("astra findCollections failed: %s", error)
            return None, ""

        for entry in (body.get("status") or {}).get("collections") or []:
            if not isinstance(entry, Mapping) or entry.get("name") != self._collection:
                continue
            vector = ((entry.get("options") or {}).get("vector")) or {}
            dimension = vector.get("dimension")
            return (int(dimension) if dimension else None), str(vector.get("metric") or "")
        return None, ""

    def _count(self) -> tuple[int | None, bool]:
        """(count, exact). A collection over the ceiling reports the ceiling as a floor."""
        try:
            body = self._post({"countDocuments": {"upperBound": COUNT_CEILING}})
        except Exception as error:
            log.debug("astra countDocuments failed: %s", error)
            return None, False

        status = body.get("status") or {}
        count = status.get("count")
        if count is None:
            return None, False
        return int(count), not bool(status.get("moreData"))
