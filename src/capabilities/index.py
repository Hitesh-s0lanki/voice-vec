"""Which of this person's capabilities bears on what was just asked.

The same question retrieval answers, asked of a much smaller corpus: instead of
passages, the documents are the *cards* — what each connected store, dataset and
toolkit was measured to be about. So it is answered the same way, with the local
embedder that is already loaded and already warm.

    "check my inbox"          → gmail          (toolkit)
    "how many students?"      → pgvector       (store, "student records")
    "average marks by class"  → the dataset    (countable, so SQL)

**Local, in-memory, and small.** A person has a handful of capabilities, not a
corpus of them, so the index is a matrix in a dict and the search is one dot
product. No network, nothing persisted: the cards it embeds already live in
Postgres, and rebuilding from them costs milliseconds.

**Keyed by what it indexed.** The cache key is a fingerprint of the capability
set — ids and titles — so connecting a store, authorising a toolkit or a profile
being rewritten invalidates it by construction rather than by a TTL that has to
be guessed at.

**A miss is empty, never a guess** — but the test for a miss is *lift*, not
score. e5 cosines over short cards sit in a narrow band and the band moves with
the cards, so an absolute floor cannot separate anything. Measured over one real
account's four capabilities:

    summary of the book The Laws of Human Nature   0.797   lift 0.027   ✓
    how are you feeling today                      0.794   lift 0.010   ✗

The two scores are the same to within 0.003 and the two questions are not
remotely alike. What does separate them is how far the best card sits above the
*mean of this person's own cards*: 0.027 against 0.010, and across a dozen
queries the relevant ones lift 0.027–0.120 while the irrelevant ones lift
0.010–0.019. So the threshold is on lift, and it sits in that gap.

**One capability is a special case, and it returns.** With a single card the
mean is the card, so lift is always zero and no threshold means anything.
Handing it back — with its `good_for` and `not_for` — and letting the agent
decide is the honest move; withholding the only thing somebody connected
because the arithmetic degenerates is not.

**No embedder, still useful.** The fallback is token overlap over the same
text. It is worse, and it is what runs when the ONNX model is not configured;
returning nothing there would make the whole flow depend on an optional
component.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.capabilities.catalogue import Capability, Catalogue, get_catalogue
from src.core.config import Settings, get_settings
from src.rag.embed import Embedder, get_embedder

log = logging.getLogger("vec.capabilities.index")

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Match:
    capability: Capability
    score: float


class CapabilityIndex:
    """Embeds a user's cards once, then answers "what covers this?" in memory."""

    def __init__(
        self, catalogue: Catalogue, embedder: Embedder, settings: Settings
    ) -> None:
        self._catalogue = catalogue
        self._embedder = embedder
        self._settings = settings
        # user_id → (fingerprint, capabilities, matrix or None)
        self._cache: dict[str, tuple[str, list[Capability], np.ndarray | None]] = {}

    def capabilities(self, user_id: str | None) -> list[Capability]:
        return self._catalogue.for_user(user_id)

    def search(self, user_id: str | None, need: str, *, limit: int = 3) -> list[Match]:
        """The capabilities that bear on `need`, best first. Never raises."""
        if not user_id or not need.strip():
            return []

        found = self.capabilities(user_id)
        if not found:
            return []

        matrix = self._matrix(user_id, found)
        scores = self._cosine(need, matrix) if matrix is not None else None

        if scores is None:
            # The fallback is on its own scale — a share of the question's own
            # words — so it keeps its own floor. One threshold across both
            # would make this path either mute or indiscriminate.
            ranked = sorted(zip(found, self._overlap(need, found)), key=_score, reverse=True)
            floor = self._settings.capability_min_overlap
            return [Match(c, float(s)) for c, s in ranked[:limit] if s >= floor]

        if len(found) == 1:
            # No baseline to lift above. See the module docstring.
            return [Match(found[0], float(scores[0]))]

        baseline = sum(scores) / len(scores)
        lift = self._settings.capability_min_lift
        ranked = sorted(zip(found, scores), key=_score, reverse=True)
        return [
            Match(c, float(s)) for c, s in ranked[:limit] if float(s) - baseline >= lift
        ]

    # ---- the two ways of scoring ----------------------------------------

    def _cosine(self, need: str, matrix: np.ndarray) -> list[float] | None:
        """One dot product against the embedded cards, or None to fall back."""
        try:
            vector = self._embedder.embed_query(need)
        except Exception as error:
            log.debug("could not embed the need: %s", error)
            return None
        return list(matrix @ _unit(vector))

    @staticmethod
    def _overlap(need: str, found: list[Capability]) -> list[float]:
        """Share of the question's words that appear in the card.

        Scaled to sit on roughly the same scale as a cosine so one threshold
        reads sensibly for both — an exact score either way is not the point,
        the ordering is.
        """
        words = {w.lower() for w in _WORD.findall(need)}
        if not words:
            return [0.0] * len(found)
        scores = []
        for capability in found:
            text = {w.lower() for w in _WORD.findall(capability.text())}
            scores.append(len(words & text) / len(words))
        return scores

    def _matrix(self, user_id: str, found: list[Capability]) -> np.ndarray | None:
        """The embedded cards, rebuilt whenever the capability set changes."""
        fingerprint = "|".join(f"{c.kind}:{c.id}:{len(c.text())}" for c in found)
        cached = self._cache.get(user_id)
        if cached and cached[0] == fingerprint:
            return cached[2]

        # No `ready` check: the embedder warms itself on first use and the app
        # warms it at boot, so gating on "already warm" would send the first
        # discovery of the process down the fallback path — the one that cannot
        # tell an inbox from a student record.
        matrix: np.ndarray | None = None
        try:
            vectors = self._embedder.embed_passages([c.text() for c in found])
            matrix = np.vstack([_unit(v) for v in vectors])
        except Exception as error:
            log.warning("could not embed capabilities for %s: %s", user_id, error)
            matrix = None

        self._cache[user_id] = (fingerprint, found, matrix)
        return matrix

    def forget(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._cache.clear()
        else:
            self._cache.pop(user_id, None)


def _score(pair: tuple[Capability, float]) -> float:
    return float(pair[1])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


@lru_cache
def get_capability_index() -> CapabilityIndex:
    return CapabilityIndex(get_catalogue(), get_embedder(), get_settings())
