"""The agent that builds the understanding, and the door agents read it through.

Two responsibilities that look like one and have opposite constraints:

    refresh()   slow. Talks to somebody else's store and then to a model.
                Seconds. Runs off every request path there is.
    card()      fast. Read on the voice path, inside a turn a listener is
                waiting through. Microseconds, from memory, or it is not usable.

They meet at a Postgres row, which is what makes "the agent understands its
capabilities in realtime" work at all: the expensive half runs once when a
connector is attached, and every turn after that reads a string.

    connect ──► verify ──► seal ──► store ──► schedule()
                                                 │  (worker thread, off the request)
                                                 ▼
                                probe ──► narrate ──► derive facts
                                                 │
                                    connector_profiles (user_id, connector)
                                                 │
             ┌───────────────────────────────────┼──────────────────────┐
             ▼                                   ▼                      ▼
      card() → system prompt          facts() → Capabilities      GET /profile

**Profiling may never break connecting.** Every failure — an unreachable store,
no model, a dead Postgres — lands as a stored `failed` profile or as no profile
at all, and the connector stays exactly as usable as it was. The user got a
green form because their credential works, and it does.

**A profile is only served for the credentials in force.** Every read checks
the stored fingerprint against the account's current sealed blob. Rotating a
key, renaming an index or pointing at another table changes the blob, and a
profile that no longer matches is treated as missing — never as approximately
right. Describing an index the credentials no longer reach is the one failure
mode of this whole layer that would be invisible.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from functools import lru_cache
from typing import Any, Mapping

from src.connectors.narrate import describe
from src.connectors.probes.astra import AstraProbe
from src.connectors.probes.composio import ComposioProbe
from src.connectors.probes.pgvector import PgVectorProbe
from src.connectors.probes.pinecone import PineconeProbe
from src.connectors.profile import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OK,
    CapabilityFacts,
    Profile,
    Understanding,
)
from src.connectors.profile_store import ProfileStore, get_profile_store
from src.connectors.registry import get_spec, vector_slugs
from src.connectors.service import ConnectorService, get_connector_service
from src.connectors.store import ConnectorStore, get_connector_store
from src.core.config import Settings, get_settings
from src.rag.store import DEFAULT_TABLE

log = logging.getLogger("vec.connectors.profile")

#: The order an agent should read its stores in: retrieval first, in the same
#: preference order `BackendResolver` picks between them, then tools. Derived
#: from the registry rather than listed, so a fifth vector connector appears
#: here without this file being edited.
PROFILE_ORDER: tuple[str, ...] = vector_slugs() + ("composio",)

#: Backends, for the round-trip check only. Imported here rather than at the
#: top of the module because `src.rag.backends` imports this one — the resolver
#: reads profiles — and a module-level import would close that loop.
def _backend(slug: str):
    from src.rag.backends.astra import AstraBackend
    from src.rag.backends.pgvector import PgVectorBackend
    from src.rag.backends.pinecone import PineconeBackend

    return {"pgvector": PgVectorBackend, "pinecone": PineconeBackend, "astra": AstraBackend}[slug]


class _LazyBackends(dict):
    def __missing__(self, slug):
        return _backend(slug)


_BUILDERS_FOR_PROBE = _LazyBackends()

_PROBES: dict[str, Any] = {
    "pgvector": PgVectorProbe,
    "pinecone": PineconeProbe,
    "astra": AstraProbe,
    "composio": ComposioProbe,
}


class ProfileService:
    def __init__(
        self,
        connectors: ConnectorService,
        accounts: ConnectorStore,
        profiles: ProfileStore,
        settings: Settings,
    ) -> None:
        self._connectors = connectors
        self._accounts = accounts
        self._profiles = profiles
        self._settings = settings
        # Small and bounded. Profiling is not the product, and a burst of
        # reconnects must not be able to open a hundred connections into a
        # hundred different people's databases at once.
        self._pool = ThreadPoolExecutor(
            max_workers=settings.profile_workers, thread_name_prefix="profile"
        )
        # (user_id, slug) → (fingerprint, monotonic, Profile). The realtime
        # read. Bounded by eviction rather than by TTL alone, because the cost
        # here is memory per signed-in user and there is no upper bound on those.
        self._cache: dict[tuple[str, str], tuple[str, float, Profile]] = {}
        self._lock = threading.Lock()
        self._running: set[tuple[str, str]] = set()

    # ---- realtime reads --------------------------------------------------

    def card(self, user_id: str | None, slug: str) -> str:
        """One store, as a paragraph an agent can act on. Never raises, never blocks.

        Empty string for every reason there is — not signed in, not connected,
        never profiled, profiled under credentials since rotated. The caller
        puts it in a prompt, and an absent card has to mean "say nothing about
        this store" rather than "say something wrong".
        """
        profile = self.get(user_id, slug)
        if profile is None:
            return ""
        return profile.card(budget=self._settings.profile_card_chars)

    def cards(self, user_id: str | None) -> str:
        """What this user's agent can actually reach, for its system prompt.

        Ordered by kind so the retrieval stores read together and the tools
        connector reads last — an agent scanning this is answering two
        different questions ("where do I look" and "what can I do") and mixing
        them makes both harder.

        **Only usable stores.** A connector that cannot be read contributes
        nothing an agent can act on, and its card carries an operator's error
        message — "there is no table called chunks on the search path" — into a
        prompt whose first line is *everything you write is spoken aloud
        immediately*. The panel is where a broken connector should be reported;
        `card()` still returns it in full for exactly that. Retrieval abstains
        on a store it cannot read either way, so the agent loses nothing by not
        being told about one.
        """
        if not user_id:
            return ""

        blocks: list[str] = []
        for slug in PROFILE_ORDER:
            profile = self.get(user_id, slug)
            if profile is None or not _usable(profile):
                continue
            blocks.append(profile.card(budget=self._settings.profile_card_chars))
        return "\n\n".join(blocks)

    def facts(self, user_id: str | None, slug: str) -> CapabilityFacts | None:
        """The measured capabilities, for the retrieval path. None means "not measured".

        None and a default-constructed `CapabilityFacts` are different answers
        and the caller must not confuse them: the first means fall back to the
        backend's own static guess, the second is a measurement saying this
        store can do nothing.
        """
        profile = self.get(user_id, slug)
        return profile.facts if profile else None

    def get(self, user_id: str | None, slug: str) -> Profile | None:
        """The stored understanding, if it is still about the right store.

        Reads memory first, then Postgres. A row that is stale is served
        anyway and a refresh is scheduled behind it: a slightly old description
        of a corpus is worth far more to the turn in flight than a blank one,
        and the corpus itself rarely changes character between refreshes.
        """
        spec = get_spec(slug)
        if not user_id or spec is None or not self._profiles.configured:
            return None

        account = self._account(user_id, spec.slug)
        if account is None:
            return None

        cached = self._cache.get((user_id, spec.slug))
        if (
            cached
            and cached[0] == account.credentials
            and time.monotonic() - cached[1] < self._settings.profile_cache_ttl_s
        ):
            return cached[2]

        try:
            row = self._profiles.get(user_id, spec.slug)
        except Exception as error:
            log.debug("could not read the %s profile: %s", spec.slug, error)
            return None

        if row is None or not row.matches(account.credentials):
            # No profile, or one built from credentials that have since been
            # replaced. Either way there is nothing honest to serve, and a
            # probe is worth starting.
            self.schedule(user_id, spec.slug)
            return None

        profile = Profile.from_json(row.profile)
        if profile is None:
            self.schedule(user_id, spec.slug)
            return None

        if row.stale(timedelta(hours=self._settings.profile_ttl_hours)):
            self.schedule(user_id, spec.slug)

        self._remember(user_id, spec.slug, account.credentials, profile)
        return profile

    # ---- building --------------------------------------------------------

    def schedule(self, user_id: str, slug: str) -> None:
        """Profile this connector soon, off whatever path asked for it.

        De-duplicated: a user who reloads the panel five times while a probe is
        running gets one probe. Without this, "refresh on read when stale"
        turns every poll into another connection into their database.
        """
        spec = get_spec(slug)
        if spec is None or not user_id or not self._settings.profile_enabled:
            return

        key = (user_id, spec.slug)
        with self._lock:
            if key in self._running:
                return
            self._running.add(key)

        try:
            self._pool.submit(self._run, user_id, spec.slug)
        except RuntimeError:
            # The pool is shut down — the process is going away. Not an error
            # worth raising into whoever happened to be connecting.
            with self._lock:
                self._running.discard(key)

    def _run(self, user_id: str, slug: str) -> None:
        try:
            self.refresh(user_id, slug)
        except Exception as error:  # a worker thread may not die loudly
            log.warning("profiling %s for %s failed: %s", slug, user_id, error)
        finally:
            with self._lock:
                self._running.discard((user_id, slug))

    def refresh(self, user_id: str, slug: str) -> Profile | None:
        """Probe, narrate, derive, store. The slow half, in full.

        Synchronous and safe to call directly — a test, a script, or the
        explicit re-profile endpoint all want to wait for the answer.
        """
        spec = get_spec(slug)
        if spec is None:
            return None

        account = self._account(user_id, spec.slug)
        if account is None:
            return None

        credentials = self._connectors.credentials(user_id, spec.slug)
        if not credentials:
            # Sealed under a rotated master key. Nothing to probe with, and the
            # panel already shows this connector as stale.
            return None

        self._profiles.mark_pending(user_id, spec.slug, sealed=account.credentials)
        profile = self._build(spec.slug, spec.kind, credentials, user_id=user_id)

        self._profiles.save(
            user_id,
            spec.slug,
            status=profile.status,
            profile=profile.to_json(),
            card=profile.card(budget=self._settings.profile_card_chars),
            sealed=account.credentials,
            error=profile.error,
            profiled_at=profile.profiled_at,
        )
        self._remember(user_id, spec.slug, account.credentials, profile)
        return profile

    def _build(
        self, slug: str, kind: str, credentials: Mapping[str, str], *, user_id: str
    ) -> Profile:
        """One probe, one optional model call, and the facts read off the result."""
        probe_class = _PROBES.get(slug)
        if probe_class is None:
            return Profile(
                connector=slug,
                kind=kind,
                status=STATUS_FAILED,
                observation=_missing(slug, kind),
                error=f"no probe knows how to read {slug}",
            )

        probe = probe_class(
            credentials,
            user_id=user_id,
            table=(credentials.get("table") or "").strip() or DEFAULT_TABLE,
            excerpts=self._settings.profile_excerpts,
        )
        observation = probe.observe()

        if not observation.reachable:
            reason = observation.notes[0] if observation.notes else "the store did not answer"
            return Profile(
                connector=slug,
                kind=kind,
                status=STATUS_FAILED,
                observation=observation,
                facts=CapabilityFacts(),
                error=reason,
            )

        # A width match is not a model match. `reconcile_dimension` can only
        # check that the named model is 768-dimensional, not that it is *the*
        # 768-dimensional model that built this index — and the wrong one gives
        # a search that runs, returns rows, and means nothing. This is the only
        # check that can tell them apart, and it is why it lives here rather
        # than in `verify`: it costs a model load.
        # The width this store is actually queried at. A connected index of
        # another width is embedded remotely at that width, so comparing it to
        # this app's own would call a perfectly searchable store broken.
        dim = int(credentials.get("dim") or 0) or self._settings.embed_dim

        round_trip: bool | None = None
        if kind == "vector":
            # Every connected vector store, not only those of another width.
            # Width was never what was being checked: a 384-dimensional index
            # built by some *other* 384-dimensional model is just as unsearchable
            # and just as silent about it. This is the only check that looks.
            observation, round_trip = _check_round_trip(observation, credentials, slug)

        understanding = _narrate(observation, self._settings)
        facts = CapabilityFacts.derive(observation, embed_dim=dim)

        if round_trip is False:
            # A store whose own passages do not retrieve themselves answers
            # every question with plausible nonsense, which is worse than
            # answering none. Gating on a heuristic is the right asymmetry
            # here: a false negative costs an abstention the user can act on,
            # a false positive is confident garbage nobody can detect.
            facts = replace(facts, searchable=False)

        # Reachable but not usable: the right shape of answer to the wrong
        # question. The agent needs to be told rather than discover it per
        # question.
        status = STATUS_OK if (facts.searchable or kind == "tools") else STATUS_DEGRADED
        error = "" if status == STATUS_OK else _why_unsearchable(observation, dim)

        return Profile(
            connector=slug,
            kind=kind,
            status=status,
            observation=observation,
            understanding=understanding or Understanding(),
            facts=facts,
            error=error,
        )

    # ---- plumbing --------------------------------------------------------

    def _account(self, user_id: str, slug: str):
        try:
            return self._accounts.get(user_id, slug)
        except Exception as error:
            log.debug("could not read the %s account: %s", slug, error)
            return None

    def _remember(self, user_id: str, slug: str, sealed: str, profile: Profile) -> None:
        with self._lock:
            self._cache[(user_id, slug)] = (sealed, time.monotonic(), profile)
            while len(self._cache) > self._settings.profile_cache_size:
                self._cache.pop(next(iter(self._cache)))

    def forget(self, user_id: str, slug: str | None = None) -> None:
        """Drop cached profiles — on disconnect, and on reconnect.

        The row itself goes with the account through the foreign key's cascade;
        this is the in-process copy, which no cascade can reach.
        """
        with self._lock:
            for key in [k for k in self._cache if k[0] == user_id and (slug is None or k[1] == slug)]:
                self._cache.pop(key, None)


#: A passage embedded by the model that indexed it retrieves itself at very
#: high similarity. The bar is loose because the excerpt is a whitespace-
#: normalised prefix of the chunk rather than the chunk, and because a wrong
#: model does not land near this — it lands in the 0.2–0.5 band that any two
#: unrelated English texts share.
ROUND_TRIP_FLOOR = 0.75


def _check_round_trip(
    observation, credentials: Mapping[str, str], slug: str
) -> tuple[Any, bool | None]:
    """Embed a passage from the store and ask the store for it back.

    The one test that distinguishes "a 768-dimensional model" from "the model
    that built this index". If the named model is right, a passage lifted out
    of the index retrieves itself at the top; if it is wrong, the query lands
    somewhere unrelated and the search returns plausible nonsense — which is
    worse than an error, because nothing reports it.

    A note *and* a gate. The note says what was measured; the gate is in
    `_build`, because a store whose own passages do not retrieve themselves
    answers every question with plausible nonsense — and a false negative there
    only costs an abstention that names the store.
    """
    if not observation.excerpts:
        return observation, None

    try:
        backend = _BUILDERS_FOR_PROBE[slug](credentials)
    except Exception as error:
        log.debug("round trip: could not build a backend: %s", error)
        return observation, None

    try:
        passage = observation.excerpts[0]
        hits = backend.search(backend.embed_query(passage), strategies=[], limit=1)
    except Exception as error:
        log.info("round trip failed for %s: %s", slug, error)
        return (
            replace(
                observation,
                notes=observation.notes + ("could not confirm the embedding model",),
            ),
            None,
        )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    if not hits:
        return observation, None

    top = hits[0].score
    if top >= ROUND_TRIP_FLOOR:
        return observation, True

    return (
        replace(
            observation,
            notes=observation.notes
            + (
                f"a passage from this store retrieves itself at only {top:.2f} — "
                "it was not built with the embeddings this app queries it with",
            ),
        ),
        False,
    )


def _usable(profile: Profile) -> bool:
    """Is there anything here an agent could actually reach for?

    A vector store has to be searchable. A tools connector has to have at least
    one authorised toolkit — a Composio project with nothing connected can list
    hundreds of tools and run none of them.
    """
    if profile.kind == "tools":
        tools = profile.observation.tools
        return bool(tools and tools.authorised)
    return profile.facts.searchable


def _narrate(observation, settings: Settings) -> Understanding | None:
    """The model call, fenced so it can never take the profile down with it."""
    try:
        return describe(observation, settings=settings)
    except Exception as error:
        log.info("could not narrate the %s profile: %s", observation.connector, error)
        return None


def _why_unsearchable(observation, dim: int) -> str:
    """Say which of the reasons it was, because they have different fixes."""
    shape = observation.vectors
    if shape is None:
        return "this store holds no vectors"
    if shape.dimensions and shape.dimensions != dim:
        return (
            f"stores {shape.dimensions}-dimensional vectors and queries are embedded "
            f"at {dim} — they cannot be searched against each other"
        )
    if shape.records == 0:
        return "the index is empty"
    for note in observation.notes:
        if "not built with the embeddings" in note:
            return (
                "this index was not built with the embeddings this app queries it "
                "with — its own passages do not retrieve themselves"
            )
    return "the store answered but cannot be searched"


def _missing(slug: str, kind: str):
    from src.connectors.profile import Observation

    return Observation(connector=slug, kind=kind, reachable=False)


@lru_cache
def get_profile_service() -> ProfileService:
    return ProfileService(
        get_connector_service(),
        get_connector_store(),
        get_profile_store(),
        get_settings(),
    )
