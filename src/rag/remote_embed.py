"""Embedding a query for a store this app did not build, at that store's width.

This app embeds locally with `multilingual-e5-small` at 384 dimensions, which
is the right answer for a connected index that happens to be 384 wide. It is
the wrong answer for every other one — built by somebody else at whatever width
they chose, 768, 1536, 3072 — because a 384-dimensional query cannot be
compared to any of them.

The first attempt at this asked the user to name the model and loaded it from
HuggingFace. That was wrong twice over. It put a question on the form that the
form could not help anyone answer — the reply to "stores 768-dimensional
vectors" is, reasonably, to type `768`, which fastembed then spent 39 seconds
trying to download as a repository. And it could not answer the common case at
all: the widths most connected stores actually use, 1536 and 3072, have no
locally loadable model.

**OpenAI's `text-embedding-3` models take a `dimensions` parameter.** So the
width read from the store's own catalogue *is* the answer. Nothing is asked,
nothing is downloaded, and every width up to 3072 is reachable:

    width ≤ 1536   text-embedding-3-small, truncated to it
    width ≤ 3072   text-embedding-3-large, truncated to it
    wider          not something this can embed for

**This is a network call on the answer path**, which docs/04-latency.md
otherwise forbids. It is confined to connected stores, which are already a
round trip away — a connected Pinecone or Astra is a network hop by
construction — and it is skipped entirely whenever the local embedder is
already the right width. The deadline in the harness is what keeps it honest
when it is slow.

**Width is not identity.** Asking for 768 dimensions gets a vector that can be
compared to a 768-dimensional index; it does not make it *the* model that built
one. A store embedded with Gemini or bge at 768 will return neighbours that are
arithmetically valid and semantically meaningless. That is what the profiler's
round-trip check is for, and it is why this module is not the last word on
whether a connected store can be searched.
"""

from __future__ import annotations

import logging
import threading

import httpx
import numpy as np

from src.core.config import Settings

log = logging.getLogger("vec.rag.remote_embed")

#: The two models and the widths each can be truncated to. Ordered, so the
#: cheaper one is chosen whenever it can produce the width asked for.
MODELS: tuple[tuple[str, int], ...] = (
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072),
)

MAX_DIM = MODELS[-1][1]

#: A connected store is already a round trip; this is a second one. Short
#: enough that a slow provider degrades the answer rather than holding the
#: worker past the point anybody is waiting.
TIMEOUT_S = 8.0


_client: httpx.Client | None = None
_lock = threading.Lock()


def get_client() -> httpx.Client:
    """One pooled client, so a query does not pay a TLS handshake to embed.

    Worth being precise about what this buys, because measuring it from
    ap-southeast-1 does not show it: observed latency ranged 0.4–2.5 s per call
    either way, dominated by provider variance far larger than a handshake.
    Pooling removes a cost that is real and small; it does not make this fast.

    **This is the expensive part of a connected store of another width**, and
    it is why docs/04-latency.md's zero-network-calls rule survives only for
    the deployment index. The harness deadline is what keeps it honest.
    """
    global _client
    with _lock:
        if _client is None:
            _client = httpx.Client(
                timeout=TIMEOUT_S,
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return _client


class RemoteEmbedUnavailable(RuntimeError):
    """No key, an unreachable provider, or a width nothing can produce."""


def model_for(dim: int) -> str:
    """The cheapest model that can be truncated to this width."""
    for name, native in MODELS:
        if 0 < dim <= native:
            return name
    raise RemoteEmbedUnavailable(
        f"{dim}-dimensional vectors are wider than anything this can embed "
        f"({MAX_DIM} is the maximum)."
    )


def supported(dim: int) -> bool:
    return 0 < dim <= MAX_DIM


def embed_query(text: str, dim: int, *, settings: Settings) -> np.ndarray:
    """One embedding, at exactly `dim` dimensions.

    Raises rather than returning a wrong-width vector: a caller that got 1536
    floats where it asked for 768 would hand them to Postgres and get an error
    from a layer that cannot explain it.
    """
    if not settings.openai_api_key:
        raise RemoteEmbedUnavailable(
            "This store needs a query embedded at its own width, which needs "
            "OPENAI_API_KEY."
        )

    model = model_for(dim)
    try:
        response = get_client().post(
            f"{settings.openai_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": model, "input": text, "dimensions": dim},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise RemoteEmbedUnavailable(
            f"{model} answered {error.response.status_code}."
        ) from error
    except Exception as error:
        raise RemoteEmbedUnavailable(f"could not reach {model}: {error}") from error

    try:
        values = (response.json()["data"])[0]["embedding"]
    except (KeyError, IndexError, TypeError) as error:
        raise RemoteEmbedUnavailable(f"{model} returned no embedding.") from error

    vector = np.asarray(values, dtype=np.float32)
    if vector.size != dim:
        raise RemoteEmbedUnavailable(
            f"{model} returned {vector.size} dimensions, not {dim}."
        )

    # `text-embedding-3` returns unit-length vectors at the native width, and
    # OpenAI's guidance is to renormalise after truncation. The API does this
    # when `dimensions` is passed, so this is a cheap assertion of a promise
    # rather than a correction — but cosine on a non-unit vector is still
    # correct, and inner product on one is not, so it costs nothing to be sure.
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm and abs(norm - 1.0) > 1e-3 else vector
