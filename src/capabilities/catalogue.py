"""Everything one person has connected, as things an agent can be handed.

A connector, a dataset and an authorised toolkit are three different objects in
three different tables, and to an agent deciding what to do next they are one
kind of thing: *something I could use, and the call that uses it*. This turns
all three into `Capability`.

    connector profile ──┐
    dataset profile ────┼──► Capability(id, kind, what it is good for, how to use it)
    authorised toolkit ─┘

**One capability per toolkit, not per connector.** Composio is a single row in
`integrations`, but "gmail" and "slack" answer completely different questions
and the agent unlocks them separately — a discovery that returned "you have
Composio" would tell it nothing it could act on.

**`use` is the next call, in words.** Discovery that names a capability without
naming the tool that uses it makes the model guess, and the guesses are the
expensive kind: querying a dataset by retrieval, or searching a store by SQL.

Nothing here reaches a network. Every field comes from a profile that was
already measured and written down (docs/17-understanding.md); a store connected
seconds ago simply has no capability yet, and the agent is told what is ready
rather than made to wait for what is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from src.connectors.profile import Profile
from src.connectors.profile_service import PROFILE_ORDER, ProfileService, get_profile_service
from src.core.config import Settings, get_settings
from src.datasets.service import DatasetService, get_dataset_service

log = logging.getLogger("vec.capabilities.catalogue")

#: What kind of thing this is, and therefore what it is *for*. The agent reads
#: these to tell "somewhere to look" from "something to do".
STORE = "store"
DATASET = "dataset"
TOOLKIT = "toolkit"


@dataclass(frozen=True, slots=True)
class Capability:
    """One thing this user has, and the one call that uses it."""

    id: str
    kind: str
    title: str
    summary: str = ""
    good_for: tuple[str, ...] = ()
    not_for: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    card: str = ""
    #: The tool to call next, named exactly as the model must call it.
    tool: str = ""
    #: The arguments that tool needs, in the words the model will read.
    use: str = ""
    #: Why this cannot be used, in a sentence somebody can act on. Empty means
    #: it can. A blocked capability is still *found* — see `_stores`.
    blocked: str = ""

    def text(self) -> str:
        """What the index embeds: what this is about, and nothing else.

        Three things are deliberately *not* in here, and each one measurably
        hurt when it was:

        **The tool name and its arguments.** The same strings on every
        capability of a kind, so including them pulls each embedding towards
        its kind rather than towards its subject.

        **The rendered card.** It repeats the title and summary and then adds
        the mechanical tail — "2.4k records, 768-dim, cosine", "Filter on:
        book_id", "Search: dense only" — which every card carries in almost the
        same words. Embedding it dragged every capability towards the same
        point: the gap between a real match and an irrelevant question fell to
        0.008, and "summary of the book The Laws of Human Nature" stopped
        matching a store whose card literally says *finding book passages*.
        Dropping it roughly doubled the separation.

        **Counts and dimensions.** Same reason, and they answer no question
        anybody asks in words.
        """
        return "\n".join(
            part
            for part in (
                self.title,
                self.summary,
                " ".join(self.topics),
                " ".join(self.good_for),
            )
            if part
        )

    def as_dict(self) -> dict[str, object]:
        """What discovery hands back to the model."""
        found: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "what": self.summary or self.title,
        }
        if self.blocked:
            # No `use`, because there is no call to make. The reason takes its
            # place so the agent can say what is wrong instead of reporting
            # that it has nothing — the difference between "you have no book
            # store" and "your book store cannot be searched", which are
            # different problems with different fixes.
            found["unavailable"] = self.blocked
        else:
            found["use"] = self.use
        if self.good_for:
            found["good_for"] = list(self.good_for)
        if self.not_for:
            found["not_for"] = list(self.not_for)
        return found


class Catalogue:
    """The user's capabilities, gathered from the profiles that measured them."""

    def __init__(
        self, profiles: ProfileService, datasets: DatasetService, settings: Settings
    ) -> None:
        self._profiles = profiles
        self._datasets = datasets
        self._settings = settings

    def for_user(self, user_id: str | None) -> list[Capability]:
        """Never raises, never blocks. An empty list means "nothing is ready"."""
        if not user_id:
            return []

        found: list[Capability] = []
        found.extend(self._stores(user_id))
        found.extend(self._datasets_for(user_id))
        return found

    # ---- the three sources ----------------------------------------------

    def _stores(self, user_id: str) -> list[Capability]:
        found: list[Capability] = []
        for slug in PROFILE_ORDER:
            try:
                profile = self._profiles.get(user_id, slug)
            except Exception as error:
                log.debug("could not read the %s profile for %s: %s", slug, user_id, error)
                continue
            if profile is None:
                continue
            if profile.kind == "tools":
                found.extend(self._toolkits(profile))
            elif profile.observation.reachable:
                # Reachable but not searchable is still a capability worth
                # returning — carrying its reason. Dropping it silently is how
                # somebody who connected a book store gets told, with total
                # confidence, that they have nothing about books.
                found.append(self._store(profile))
        return found

    def _store(self, profile: Profile) -> Capability:
        understanding = profile.understanding
        return Capability(
            blocked=_why_blocked(profile),
            id=profile.connector,
            kind=STORE,
            title=understanding.title or profile.connector,
            summary=understanding.summary,
            good_for=understanding.good_for,
            not_for=understanding.not_for,
            topics=understanding.topics,
            card=profile.card(budget=self._settings.profile_card_chars),
            tool="search_store",
            use=f'search_store(store="{profile.connector}", question="…")',
        )

    def _toolkits(self, profile: Profile) -> list[Capability]:
        """One per *authorised* toolkit — what this account may actually act on.

        Read off the profile rather than Composio's catalogue: a toolkit the
        person has not authorised is a tool the agent cannot run, and offering
        it is how a turn ends in a consent screen nobody is looking at.
        """
        shape = profile.observation.tools
        if shape is None:
            return []

        return [
            Capability(
                id=toolkit,
                kind=TOOLKIT,
                title=toolkit,
                summary=f"Actions on this person's {toolkit} account.",
                good_for=self._actions(shape.actions, toolkit),
                card=profile.card(budget=self._settings.profile_card_chars),
                tool="find_capability",
                use=(
                    f"ask for {toolkit} again once its actions are listed, then call the "
                    f"{toolkit.upper()}_* tool the answer names"
                ),
            )
            for toolkit in shape.authorised
        ]

    @staticmethod
    def _actions(actions: tuple[str, ...], toolkit: str) -> tuple[str, ...]:
        """The slugs belonging to one toolkit, as the searchable half.

        `GMAIL_FETCH_EMAILS` reads as "gmail fetch emails" to an embedder, which
        is exactly the vocabulary somebody asking to check their inbox uses.
        """
        head = f"{toolkit.upper()}_"
        return tuple(
            slug.replace("_", " ").lower() for slug in actions if slug.upper().startswith(head)
        )[:12]

    def _datasets_for(self, user_id: str) -> list[Capability]:
        try:
            rows = self._datasets.list(user_id)
        except Exception as error:
            log.debug("could not list datasets for %s: %s", user_id, error)
            return []

        found = []
        for row in rows:
            if getattr(row, "status", "") != "ok":
                continue
            card = getattr(row, "card", "") or ""
            found.append(
                Capability(
                    id=row.dataset_id,
                    kind=DATASET,
                    title=card.split("—")[-1].strip() if "—" in card else row.dataset_id,
                    summary=getattr(row, "summary", "") or card,
                    card=card,
                    tool="query_dataset",
                    use=f'query_dataset(dataset="{row.dataset_id}", question="…")',
                )
            )
        return found


@lru_cache
def get_catalogue() -> Catalogue:
    return Catalogue(get_profile_service(), get_dataset_service(), get_settings())


def _why_blocked(profile: Profile) -> str:
    """Why this store cannot answer, in the words the agent will say out loud.

    Only the reasons that are *measured*. A store that is merely slow, or that
    nobody has profiled yet, is not blocked — it is untested, and untested is
    not a finding.
    """
    facts = profile.facts
    if facts.compatible is False:
        return (
            "connected, but its vectors were built by a different embedding model, "
            "so it cannot be searched from here — it would return unrelated records"
        )
    if not facts.searchable:
        return "connected, but not searchable right now"
    return ""
