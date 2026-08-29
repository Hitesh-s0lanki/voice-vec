"""The answer cache — cache-augmented generation, in Redis (docs/15-effort.md).

A question that has already been answered should not be searched, reranked and
synthesised again. From rung 1 up, every request looks here first: on a hit the
whole pipeline collapses to one local embedding and one Redis round trip, which
is the difference between 200 ms and 15 ms — and at rung 3 or 4, between six
LLM calls and none.

**Two layouts, because Redis is two different products.**

    semantic   KNN over past query vectors. Needs the query engine
               (RediSearch — Redis 8, Redis Stack, Redis Cloud).
    exact      SHA-256 of the normalised query. The fallback, on a plain Redis.

Exactly one of them is written, never both. The semantic layer is the one that
earns its keep — *"What is a corporation?"* and *"Explain what a corporation
is"* are different strings and neighbouring vectors, so the second is served
from the first one's answer — and it *subsumes* the exact layer, because the
same text embeds to the same vector and scores a cosine of 1.0 against itself.
Writing both would double the storage to save a few milliseconds on the repeat
of a question the KNN was going to find anyway, and storage is the binding
constraint on a small instance (see the sizing note below).

So the engine's absence is detected once at connect and the layout falls back
to exact-only, with a line in the log. An honest downgrade — the cache still
works, it just stops catching paraphrases.

**Sizing, measured rather than estimated.** One cached answer is a 384-dim
float32 vector (1.5 KB) plus the JSON payload — the answer and up to three
citations carrying their passage text, which for Devanagari is three UTF-8
bytes per character. Against a managed Redis 8, a hundred entries with a
3.7 KB payload measured **6.09 KB each** by `MEMORY USAGE`, so a 30 MB instance
holds on the order of five thousand before the index's own overhead. That is
where `cache_max_entries` comes from, and why an unusually large payload is
skipped rather than cached.

The real backstop is Redis itself: every key written here carries a TTL, so an
instance with `volatile-lru` — which is the managed default — evicts the least
recently used answer when it fills rather than refusing the write. That is
exactly the behaviour a cache wants, and it is why running out of room degrades
the hit rate instead of the pipeline.

**Everything is namespaced under `cache_prefix`**, keys and index alike, because
a Redis instance is a place other things also live.

**Two rules that make this safe rather than merely fast.**

*Only successes are cached.* An abstention is a statement about the corpus at
one moment; cache it and a write that fills the gap is invisible for a day.
A refusal is a statement about the input and is cheap to recompute anyway.

*The threshold is high and the scope is narrow.* A cache that answers a
question nobody asked is a correctness bug wearing a performance win's
clothes — and it is invisible in every metric except the answers themselves.
So the similarity floor is 0.97 rather than the 0.45-ish figure that circulates
in write-ups (which is a raw squared-L2 distance over un-normalised vectors and
does not transfer to cosine at all), and the scope key below separates every
axis that can change what the right answer is.

**It never raises.** Every failure — no Redis, wrong password, a timeout, a
malformed entry — is a miss. A cache that can take the answer path down with it
is worse than no cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from src.core.config import Settings, get_settings

log = logging.getLogger("vec.rag.cache")

try:  # the cache is optional, and so is its client
    import redis as _redis
except ImportError:  # pragma: no cover - exercised by a checkout without the extra
    _redis = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Scope:
    """Everything that can change what the right answer is.

    Not decoration — each field here is a real cache-poisoning bug if dropped:

    ``user``      whose vector store answered. Two users on two Pinecone
                  indexes ask the same question and must not share a row.
    ``backend``   *which* store, redacted. Reconnecting somewhere else has to
                  invalidate, and it does, because this string changes.
    ``mode``      the rung that produced it. Rung 0 returns passages and rung 2
                  returns synthesis; they are not interchangeable answers.
    ``language``  what it was answered in, and whether it was answered from the
                  parallel English. Same question, different spoken reply.
    """

    user: str
    backend: str
    mode: str
    language: str
    english: bool

    def key(self) -> str:
        """A hex digest, which is also what makes it safe as a RediSearch TAG.

        TAG values are tokenised and several punctuation characters have to be
        escaped in a query. A digest has none of them, so the escaping question
        never arises rather than being answered correctly once and forgotten.
        """
        raw = f"{self.user}|{self.backend}|{self.mode}|{self.language}|{int(self.english)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class Hit:
    """A cached answer and how it was found."""

    payload: dict[str, Any]
    similarity: float
    #: "exact" or "semantic" — reported on the response so a cached answer is
    #: never mistaken for a fresh one in the metrics.
    how: str


class AnswerCache:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._semantic: bool | None = None
        self._connected = False

    # ---- connection ------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self._settings.redis_url) and self._settings.cache_enabled and _redis is not None

    @property
    def index_name(self) -> str:
        return f"{self._settings.cache_prefix}:idx"

    @property
    def semantic(self) -> bool:
        """Whether paraphrases are caught. False until a connection says so."""
        return bool(self._semantic)

    def describe(self) -> str:
        if not self._settings.cache_enabled:
            return "off"
        if _redis is None:
            return "unavailable (redis client not installed)"
        if not self._settings.redis_url:
            return "unset"
        if not self._connected:
            return "not connected yet"
        return "semantic" if self._semantic else "exact-only"

    def _connect(self) -> Any | None:
        if self._client is not None or not self.configured:
            return self._client

        try:
            self._client = _redis.from_url(
                self._settings.redis_url,
                # Two different budgets, and collapsing them into one is a bug
                # that only shows up against a *remote* Redis: the per-operation
                # ceiling is tight because a cache lookup sits on the answer
                # path, but opening a TLS connection across a region costs
                # ~100 ms and would never complete inside it. One timeout for
                # both means the cache silently never connects.
                socket_timeout=self._settings.cache_timeout_s,
                socket_connect_timeout=self._settings.cache_connect_timeout_s,
                retry_on_timeout=False,
                decode_responses=False,
            )
            self._client.ping()
            self._connected = True
        except Exception as error:
            log.warning("answer cache unavailable: %s", error)
            self._client = None
            return None

        self._semantic = self._ensure_index(self._client)
        log.info("answer cache connected: %s", self.describe())
        return self._client

    def _ensure_index(self, client: Any) -> bool:
        """Create the vector index, or report that this Redis has no engine.

        `FT.CREATE` on a plain Redis fails with "unknown command", which is the
        detection: there is no capability flag to read, and `MODULE LIST` is
        blocked on several managed offerings. So the creation attempt *is* the
        probe, and it runs once per process.
        """
        if not self._settings.cache_semantic:
            return False

        prefix = f"{self._settings.cache_prefix}:e:"
        try:
            client.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", prefix,
                "SCHEMA",
                "scope", "TAG",
                "vector", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self._settings.embed_dim),
                "DISTANCE_METRIC", "COSINE",
            )
            return True
        except Exception as error:
            message = str(error).lower()
            if "index already exists" in message:
                return True
            log.info("answer cache is exact-only (no query engine): %s", error)
            return False

    def warm(self) -> bool:
        """Connect and settle the layout, at boot rather than mid-answer.

        Returns whether there is a working cache. Never raises — an unreachable
        Redis at boot is the same thing as an unreachable Redis at question
        time, which is a miss.
        """
        return self._connect() is not None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # a socket already gone
                pass
            self._client = None
            self._connected = False

    # ---- read ------------------------------------------------------------

    def get(self, query: str, vector: np.ndarray, scope: Scope) -> Hit | None:
        """The cached answer for this question, or None. Never raises."""
        client = self._connect()
        if client is None:
            return None

        scope_key = scope.key()

        if not self._semantic:
            try:
                raw = client.get(self._exact_key(scope_key, query))
                return Hit(payload=json.loads(raw), similarity=1.0, how="exact") if raw else None
            except Exception as error:
                log.debug("cache exact lookup failed: %s", error)
                return None

        try:
            return self._nearest(client, vector, scope_key)
        except Exception as error:
            log.debug("cache semantic lookup failed: %s", error)
            return None

    def _fits(self, vector: np.ndarray) -> bool:
        """Is this vector the width the index was created at?

        The RediSearch index is declared `DIM = embed_dim` once, at creation.
        A user whose connected store was built by a 768-dimensional model gets
        768-dimensional query vectors, and there is no version of this index
        that holds both — so the semantic half is skipped for them and the
        exact half, which keys on the query text, keeps working.

        Silently, and correctly: `Scope` already separates their entries from
        everybody else's, so nothing is mixed. What they lose is near-miss
        matching, which is a smaller loss than a cache that raises on every
        write. Rebuilding the index per width would mean one index per model
        per deployment, for a saving that is already the smaller half.
        """
        return int(np.asarray(vector).size) == self._settings.embed_dim

    def _nearest(self, client: Any, vector: np.ndarray, scope_key: str) -> Hit | None:
        if not self._fits(vector):
            return None

        reply = client.execute_command(
            "FT.SEARCH", self.index_name,
            f"(@scope:{{{scope_key}}})=>[KNN 1 @vector $v AS dist]",
            "PARAMS", "2", "v", _to_bytes(vector),
            "RETURN", "2", "payload", "dist",
            # No SORTBY: `KNN 1` returns one document and already returns the
            # nearest, so sorting is redundant — and sorting on a computed KNN
            # alias is the part of this query older query-engine builds are
            # most likely to reject.
            "DIALECT", "2",
        )
        fields = _first_document(reply)
        if fields is None:
            return None

        payload, distance = fields
        # RediSearch reports cosine *distance*: 1 - similarity. Reading it as a
        # similarity would serve the least similar entry in the scope, every
        # time, and still look like a working cache.
        similarity = 1.0 - distance
        if similarity < self._settings.cache_similarity:
            return None

        return Hit(payload=payload, similarity=round(similarity, 4), how="semantic")

    # ---- write -----------------------------------------------------------

    def put(
        self,
        query: str,
        vector: np.ndarray,
        scope: Scope,
        payload: dict[str, Any],
    ) -> None:
        """Remember an answered question. Best effort; failures are ignored.

        Only ever called with a successful answer — the abstention rule lives
        at the call site, where the status is, rather than being re-derived
        here from the shape of the payload.
        """
        client = self._connect()
        if client is None:
            return

        scope_key = scope.key()
        ttl = self._settings.cache_ttl_s
        body = json.dumps(payload, ensure_ascii=False)

        size = len(body.encode("utf-8"))
        if size > self._settings.cache_max_entry_bytes:
            # One pathological passage should not take a measurable share of a
            # small instance. Skipping it costs that question its cache hit;
            # storing it costs every *other* question a little of the room they
            # would have been cached in.
            log.debug("cache skipped a %d byte payload", size)
            return

        try:
            pipe = client.pipeline(transaction=False)

            # Exact-only when there is no query engine, and equally when this
            # vector is not the width the index was declared at — writing a
            # 768-dimensional vector into a 384-dimensional index is rejected
            # per entry, which would turn every write for that user into a
            # logged failure for a cache that could still serve them on the
            # exact path.
            if not self._semantic or not self._fits(vector):
                pipe.setex(self._exact_key(scope_key, query), ttl, body)
                pipe.execute()
                return

            entry = f"{self._settings.cache_prefix}:e:{scope_key}:{uuid.uuid4().hex}"
            pipe.hset(
                entry,
                mapping={
                    "scope": scope_key,
                    "vector": _to_bytes(vector),
                    "payload": body,
                },
            )
            pipe.expire(entry, ttl)
            # The roster is what makes `cache_max_entries` enforceable: the TTL
            # bounds how *old* the index gets, this bounds how *large*. Without
            # it a busy scope grows until the KNN scan is slower than the search
            # it replaced. Redis' own `volatile-lru` eviction is the backstop
            # under memory pressure; this is the bound under none.
            roster = f"{self._settings.cache_prefix}:z:{scope_key}"
            pipe.zadd(roster, {entry: time.time()})
            pipe.expire(roster, ttl)
            pipe.execute()
        except Exception as error:
            log.debug("cache write failed: %s", error)
            return

        self._trim(client, scope_key)

    def _trim(self, client: Any, scope_key: str) -> None:
        """Drop the oldest entries once a scope is over its ceiling."""
        roster = f"{self._settings.cache_prefix}:z:{scope_key}"
        try:
            excess = client.zcard(roster) - self._settings.cache_max_entries
            if excess <= 0:
                return
            stale = client.zpopmin(roster, excess)
            keys = [name for name, _score in stale]
            if keys:
                client.delete(*keys)
        except Exception as error:
            log.debug("cache trim failed: %s", error)

    # ---- housekeeping ----------------------------------------------------

    def invalidate(self, scope: Scope) -> int:
        """Forget everything in one scope. Returns how many keys went.

        `SCAN`, not `KEYS` — this can run against a shared Redis, and `KEYS` on
        a large keyspace blocks the server for everyone using it.
        """
        client = self._connect()
        if client is None:
            return 0

        scope_key = scope.key()
        removed = 0
        try:
            for pattern in (
                f"{self._settings.cache_prefix}:x:{scope_key}:*",
                f"{self._settings.cache_prefix}:e:{scope_key}:*",
                f"{self._settings.cache_prefix}:z:{scope_key}",
            ):
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor=cursor, match=pattern, count=200)
                    if keys:
                        removed += client.delete(*keys)
                    if cursor == 0:
                        break
        except Exception as error:
            log.debug("cache invalidation failed: %s", error)
        return removed

    def _exact_key(self, scope_key: str, query: str) -> str:
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return f"{self._settings.cache_prefix}:x:{scope_key}:{digest}"


def _to_bytes(vector: np.ndarray) -> bytes:
    """Little-endian float32, which is the only layout RediSearch reads."""
    return np.asarray(vector, dtype=np.float32).ravel().tobytes()


def _first_document(reply: Any) -> tuple[dict[str, Any], float] | None:
    """Pull `(payload, distance)` out of an FT.SEARCH reply, in either shape.

    There are two, and which one arrives is decided by the client's protocol
    negotiation rather than by anything this module asks for. redis-py 8 speaks
    RESP3 to a server that supports it, and RESP3 returns a map; RESP2 returns
    a flat array. Handling only one of them is a bug that no unit test with a
    hand-written fake will ever catch, because the fake returns whichever shape
    its author had in mind:

        RESP3   {"total_results": 1,
                 "results": [{"id": …, "extra_attributes": {"payload": …, "dist": …}}]}
        RESP2   [1, "<key>", ["payload", "…", "dist", "…"]]

    Field values are bytes or str depending on `decode_responses`, so
    everything is coerced rather than assumed. A shape this does not recognise
    is a miss, not an exception.
    """
    fields = _resp3_fields(reply) if isinstance(reply, dict) else _resp2_fields(reply)
    if fields is None:
        return None

    payload = fields.get("payload")
    distance = fields.get("dist")
    if payload is None or distance is None:
        return None

    try:
        return json.loads(_as_text(payload)), float(_as_text(distance))
    except (ValueError, TypeError):
        return None


def _resp3_fields(reply: dict) -> dict[str, Any] | None:
    results = _get(reply, "results")
    if not isinstance(results, (list, tuple)) or not results:
        return None

    attributes = _get(results[0], "extra_attributes") if isinstance(results[0], dict) else None
    if not isinstance(attributes, dict):
        return None
    return {_as_text(k): v for k, v in attributes.items()}


def _resp2_fields(reply: Any) -> dict[str, Any] | None:
    if not isinstance(reply, (list, tuple)) or len(reply) < 3:
        return None
    if _as_int(reply[0]) < 1:
        return None

    raw = reply[2]
    if not isinstance(raw, (list, tuple)):
        return None
    return {_as_text(raw[i]): raw[i + 1] for i in range(0, len(raw) - 1, 2)}


def _get(mapping: dict, key: str) -> Any:
    """A key that may have arrived as bytes or as str."""
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode())


def _as_text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def _as_int(value: Any) -> int:
    try:
        return int(_as_text(value))
    except (ValueError, TypeError):
        return 0


@lru_cache
def get_cache() -> AnswerCache:
    return AnswerCache(get_settings())
