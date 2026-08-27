"""What the agent remembers after the socket closes — Redis Agent Memory.

Two stores already hold a conversation, and neither of them is memory:

    history on VoiceSession   the last few turns, for as long as the socket lives
    Postgres                  every turn, verbatim, so /c/{id} can be reloaded

Both are *transcripts*. Reload a conversation and the model gets its own words
back; open a **new** one and it has never met you. Nothing carries across the
boundary, because nothing is ever distilled — a transcript is not a fact, and
"the user is vegetarian" is not something you find by replaying eight turns of
Tamil into a context window.

Redis Agent Memory is the layer that does the distilling
(https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/). Two
tiers, and only one of them is ours to manage:

    session memory     the turns, mirrored as they happen. Short TTL. Ours.
    long-term memory   facts the service extracts from those turns, in the
                       background, on its own cadence. Searchable across every
                       conversation the same owner ever had. Not ours.

So this module writes turns *in* and reads facts *out*, and the interesting
part happens in between, in a worker we never call.

**Postgres stays the source of truth.** Nothing here replaces `src/chat/store.py`.
The rail, the titles, the `/c/{id}` URL and the history a reload hands back all
come from Postgres exactly as before; a session that expires here loses nothing
a listener can see. What the agent gains is one paragraph in its system prompt
that Postgres could never have produced.

**One identity, two stores.** The agent-memory session id *is* the Postgres
conversation id, and the actor id is the same `user_id`-or-`sess_…` owner the
conversations table keys on. That is not a convenience — it is what makes a
turn traceable across both stores with one id, and it means a conversation that
was never opened (a cough, a caller with no identity) is never mirrored either.

**30 MB is the whole budget, and the answer cache lives in it too.** Redis
Cloud's Agent Memory service attaches to a Redis database rather than
provisioning its own, and on the free tier that is the same 30 MB instance
`src/rag/cache.py` is already budgeting against — see docs/16-memory.md for the
split. Three things here follow from that and are not incidental:

  * only a persisted turn is mirrored, so nothing transient is ever stored;
  * every event is trimmed to `agent_memory_max_chars` before it is sent, so
    one pathological transcript cannot take a measurable share of the instance;
  * session TTL is set *short* in the service configuration (hours, not days),
    because session memory is a staging area for extraction here, not a store.
    The facts are what we keep; the turns are already in Postgres.

**Nothing in this file may break a turn.** Writes go on the same queue the
Postgres writes do and are never awaited by the turn that made them. The one
call on the answer path — `recall` — is bounded by `agent_memory_timeout_ms`
and returns an empty list for every failure there is: unconfigured, unreachable,
slow, malformed. An agent that cannot remember answers anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Sequence

from src.core.config import Settings, get_settings

log = logging.getLogger("vec.memory")

# Imported the way `redis` is in the cache: a checkout that has not installed
# the extra should degrade to "no memory", not fail at import and take the
# voice loop down with it.
try:  # pragma: no cover - exercised by its own absence
    from redis_agent_memory import AgentMemory as _Sdk
    from redis_agent_memory import models as _models
except Exception:  # noqa: BLE001
    _Sdk = None  # type: ignore[assignment]
    _models = None  # type: ignore[assignment]

#: Roles the service understands, keyed by the role names used everywhere else
#: in this codebase. Anything not in here is not mirrored rather than guessed
#: at — a tool trace is not a conversation event.
_ROLES = {"user": "USER", "assistant": "ASSISTANT"}


@dataclass(frozen=True, slots=True)
class Recollection:
    """One fact the service extracted, on its way into a prompt."""

    text: str
    memory_type: str | None = None


class MemoryStore:
    """The agent's memory, or a well-behaved absence of one.

    Every method is safe to call on an unconfigured store; they return
    immediately and do nothing, which is what lets the call sites read as
    straight-line code rather than as a chain of feature checks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = None

    # ---- availability ---------------------------------------------------

    @property
    def configured(self) -> bool:
        s = self._settings
        return bool(
            _Sdk is not None
            and s.agent_memory_enabled
            and s.agent_memory_endpoint
            and s.agent_memory_store_id
            and s.agent_memory_api_key
        )

    def describe(self) -> str:
        """What the client should be told, in the same vocabulary the cache uses.

        Four states rather than a boolean, for the same reason: a deployment
        that believes it has memory and does not will read the difference as a
        model regression rather than as a missing environment variable.
        """
        if not self._settings.agent_memory_enabled:
            return "off"
        if _Sdk is None:
            return "unavailable (redis-agent-memory not installed)"
        if not (
            self._settings.agent_memory_endpoint
            and self._settings.agent_memory_store_id
            and self._settings.agent_memory_api_key
        ):
            return "unset"
        return "on"

    def _sdk(self) -> Any | None:
        """The client, built once and kept for the life of the process.

        Construction opens no socket — it only builds the two httpx clients the
        generated SDK holds — so doing it lazily costs a branch rather than a
        round trip, and doing it once avoids a fresh TLS handshake per turn
        against a service that is a region away.
        """
        if self._client is not None or not self.configured:
            return self._client

        s = self._settings
        try:
            self._client = _Sdk(
                s.agent_memory_endpoint.rstrip("/"),
                store_id=s.agent_memory_store_id,
                api_key=s.agent_memory_api_key,
                timeout_ms=s.agent_memory_timeout_ms,
            )
        except Exception as error:  # noqa: BLE001 — a bad endpoint is not fatal
            log.warning("agent memory is configured but unusable: %s", error)
            self._client = None
        return self._client

    # ---- writing --------------------------------------------------------

    def remember(
        self,
        *,
        session_id: str,
        actor_id: str,
        role: str,
        text: str,
        created_at: datetime | None = None,
    ) -> None:
        """Mirror one turn into session memory. Blocking, and never on the turn's path.

        Deliberately synchronous: this rides the same writer queue as the
        Postgres append (`VoiceSession._enqueue`), which runs its jobs one at a
        time in a worker thread. That queue already guarantees the ordering
        session memory wants — the question before the answer — and reusing it
        means a slow or dead memory service costs a listener exactly what a
        slow or dead database does, which is nothing.

        Everything is swallowed. A turn that was heard, answered and written to
        Postgres is not a failure because a fact about it went unextracted.
        """
        mapped = _ROLES.get(role)
        body = (text or "").strip()
        if not mapped or not body or not session_id or not actor_id:
            return

        client = self._sdk()
        if client is None:
            return

        # Trimmed *before* the wire, not after. The point is the instance, not
        # the request: an untrimmed 40 KB monologue costs 40 KB of a 30 MB
        # database for the hours its session lives, and contributes no more to
        # extraction than its first two thousand characters do.
        limit = self._settings.agent_memory_max_chars
        if limit > 0 and len(body) > limit:
            body = body[:limit].rstrip() + "…"

        try:
            client.add_session_event(
                session_id=session_id,
                actor_id=actor_id,
                role=_models.MessageRole(mapped),
                content=[_models.Text(text=body)],
                created_at=created_at or datetime.now(timezone.utc),
                namespace=self._settings.agent_memory_namespace or None,
            )
        except Exception as error:  # noqa: BLE001 — memory must not surface
            log.warning("agent memory write failed: %s", error)

    # ---- reading --------------------------------------------------------

    async def recall(
        self,
        *,
        query: str,
        owner_id: str,
        limit: int | None = None,
    ) -> list[Recollection]:
        """Facts about this owner that bear on this question. Bounded, and never raises.

        Scoped by `owner_id` and nothing else, which is the whole point: the
        useful memories are the ones from *other* conversations, so filtering by
        session here would return only what the prompt already contains.

        `similarity_threshold` is what keeps this from being a liability. An
        unfiltered nearest-neighbour search always returns something, and a
        prompt that opens with an irrelevant fact stated as true about the
        listener is worse than one that opens with nothing — it is confidently
        wrong in the first sentence, and it is invisible in every metric except
        the answers themselves. Same reasoning as the cache's 0.97 floor, at a
        looser setting because a recalled fact informs an answer rather than
        replacing it.
        """
        body = (query or "").strip()
        if not body or not owner_id or not self.configured:
            return []

        client = self._sdk()
        if client is None:
            return []

        try:
            found = await client.search_long_term_memory_async(
                request={
                    "text": body,
                    "filter_": {"owner_id": {"eq": owner_id}},
                    "similarity_threshold": self._settings.agent_memory_similarity,
                    "limit": limit or self._settings.agent_memory_limit,
                },
            )
        except Exception as error:  # noqa: BLE001 — a lookup is not a turn
            log.warning("agent memory recall failed: %s", error)
            return []

        out: list[Recollection] = []
        for item in getattr(found, "items", None) or []:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                out.append(Recollection(text=text, memory_type=getattr(item, "memory_type", None)))
        return out


def as_prompt(memories: Sequence[Recollection]) -> str | None:
    """Render recalled facts for a system prompt, or nothing at all.

    Returns `None` rather than an empty section, so `build_messages` never adds
    a heading with nothing under it — an empty "what you know about them"
    block reads to a model as *there is nothing to know*, which is a stronger
    and less true claim than silence.
    """
    lines = [stripped for m in memories if (stripped := m.text.strip())]
    if not lines:
        return None
    return "\n".join(f"- {line}" for line in lines)


@lru_cache(maxsize=1)
def get_memory() -> MemoryStore:
    return MemoryStore()
