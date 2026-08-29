"""Reading a Pinecone index that has no way to be read sequentially.

Pinecone is a nearest-neighbour service and not a database: there is no
`SELECT *`, no cursor, and — outside serverless — no way to list ids. The only
universally available way to get records back out is to *search for them*,
which makes an unbiased sample impossible by construction.

So this probe does the honest approximation: **query from several random
directions and merge**. One random unit vector returns the 200 records nearest
one arbitrary point in the embedding space, which is a cluster, not a sample —
if that cluster happens to be one document, the field coverage it reports is
that document's and not the index's. Three widely separated directions cannot
fix that, but they turn "one neighbourhood" into "three neighbourhoods", which
is the difference between a coverage figure that is usually right and one that
is right only when the index is homogeneous.

The directions are **seeded**, so re-profiling an unchanged index returns the
same sample and the profile does not flap between runs for no reason.

`describe_index_stats` is the part that is exact — count, dimension, namespaces
— and it is one call, so the numbers on the card are measured even though the
metadata coverage beside them is sampled. Where a figure is an estimate, the
profile says which.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import httpx
import numpy as np

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

#: How many directions to search from. Three is where the cost stops buying
#: coverage: a fourth query is another billed read for a marginal chance of
#: reaching a region the first three missed.
DIRECTIONS = 3

#: Fixed, so an unchanged index profiles identically twice running. A profile
#: that reports different field coverage on every refresh is one nobody trusts,
#: and the difference would be sampling noise rather than anything that changed.
SEED = 20260401


class PineconeProbe:
    def __init__(self, credentials: Mapping[str, str], *, excerpts: bool = True, **_: Any) -> None:
        self._api_key = credentials["api_key"]
        self._index = credentials["index"]
        self._namespace = (credentials.get("namespace") or "").strip()
        self._excerpts = excerpts

    @property
    def location(self) -> str:
        where = f"pinecone/{self._index}"
        return f"{where}#{self._namespace}" if self._namespace else where

    def _headers(self) -> dict[str, str]:
        from src.connectors.registry import PINECONE_API_VERSION

        return {
            "Api-Key": self._api_key,
            "X-Pinecone-Api-Version": PINECONE_API_VERSION,
            "Content-Type": "application/json",
        }

    def observe(self) -> Observation:
        from src.connectors.registry import pinecone_host

        started = time.perf_counter()
        try:
            host = pinecone_host(self._api_key, self._index)
            described = self._describe()
            stats = self._stats(host)
        except Exception as error:
            log.info("pinecone probe failed: %s", type(error).__name__)
            return unreachable(
                "pinecone", "vector", self.location, f"the index did not answer: {error}"
            )

        dimensions = described.get("dimension") or stats.get("dimension")
        records = self._records(stats)
        notes: list[str] = []

        records_seen = self._sample(host, int(dimensions or 0))
        if not records_seen and records:
            notes.append("the index answered but returned no records to sample")

        metadata = [dict(r.get("metadata") or {}) for r in records_seen]
        text_field = pick_text_field(metadata)
        texts = text_of(metadata, text_field)

        if not text_field:
            notes.append("no text in the metadata — hits from here cannot be quoted")
        namespaces = [n for n in (stats.get("namespaces") or {}) if n]
        if namespaces and not self._namespace:
            notes.append(
                f"searching the default namespace; {len(namespaces)} others hold vectors"
            )
        stats = field_stats(metadata)

        # The sample is a union of neighbourhoods, not a random draw. Anything
        # reading coverage off this profile should know that.
        if records_seen:
            notes.append(
                f"metadata sampled from {DIRECTIONS} nearest-neighbour probes, not uniformly"
            )

        return Observation(
            connector="pinecone",
            kind="vector",
            location=self.location,
            reachable=True,
            sampled=len(records_seen),
            vectors=VectorShape(
                dimensions=int(dimensions) if dimensions else None,
                metric=str(described.get("metric") or ""),
                index=_index_kind(described),
                normalised=None,  # Pinecone never returns values here
                records=records,
            ),
            fields=stats,
            text_field=text_field,
            text_chars=length_stats(texts),
            scripts=scripts_of(texts),
            excerpts=excerpts_of(metadata, text_field, stats=stats, allowed=self._excerpts),
            latency_ms=(time.perf_counter() - started) * 1000,
            notes=tuple(notes[:6]),
        )

    def _describe(self) -> dict[str, Any]:
        from src.connectors.registry import PINECONE_CONTROL

        response = httpx.get(
            f"{PINECONE_CONTROL}/indexes/{self._index}",
            headers=self._headers(),
            timeout=PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json() or {}

    def _stats(self, host: str) -> dict[str, Any]:
        response = httpx.post(
            f"https://{host}/describe_index_stats",
            headers=self._headers(),
            json={},
            timeout=PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json() or {}

    def _records(self, stats: Mapping[str, Any]) -> int | None:
        """The count for the namespace actually being searched, not the index total.

        A user with vectors in five namespaces and a connector pointed at one of
        them is searching that one. Reporting the index total would tell the
        agent it has 500,000 records to draw on when it has 900.
        """
        namespaces = stats.get("namespaces") or {}
        if self._namespace:
            bucket = namespaces.get(self._namespace) or {}
            return int(bucket.get("vectorCount", 0))
        total = stats.get("totalVectorCount")
        return int(total) if total is not None else None

    def _sample(self, host: str, dimensions: int) -> list[dict[str, Any]]:
        if dimensions <= 0:
            return []

        rng = np.random.default_rng(SEED)
        merged: dict[str, dict[str, Any]] = {}
        per_direction = max(1, SAMPLE_SIZE // DIRECTIONS)

        for _ in range(DIRECTIONS):
            vector = rng.normal(size=dimensions)
            vector /= np.linalg.norm(vector) or 1.0

            body: dict[str, Any] = {
                "vector": [float(v) for v in vector],
                "topK": per_direction,
                "includeMetadata": True,
                "includeValues": False,
            }
            if self._namespace:
                body["namespace"] = self._namespace

            try:
                response = httpx.post(
                    f"https://{host}/query",
                    headers=self._headers(),
                    json=body,
                    timeout=PROBE_TIMEOUT_S,
                )
                response.raise_for_status()
            except Exception as error:
                # One direction failing is not the probe failing. Two of three
                # neighbourhoods still describes the index better than nothing,
                # and the sample size on the profile says how much was read.
                log.debug("pinecone probe direction failed: %s", error)
                continue

            for match in (response.json() or {}).get("matches") or []:
                identifier = str(match.get("id") or "")
                if identifier:
                    merged.setdefault(identifier, match)

        return list(merged.values())


def _index_kind(described: Mapping[str, Any]) -> str:
    spec = described.get("spec") or {}
    if "serverless" in spec:
        return "serverless"
    if "pod" in spec:
        return "pod"
    return ""
