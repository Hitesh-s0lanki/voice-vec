"""Embedding through OpenAI. Every vector this app compares comes from here.

`text-embedding-3` takes a `dimensions` parameter, so one transport answers two
different questions with the same call:

    the app's own width   ── src/rag/embed.py, for text this app already holds:
                             the question, the reranked passages, the sentences
                             extraction chooses between, the capability cards
    a store's own width   ── the backends, for a connected index built by
                             somebody else at 768, 1536 or 3072

There used to be a local ONNX model for the first of those and this module was
only the second. That model is gone (docs/25-no-local-embedder.md): it held
~700 MiB resident to save a round trip, which is a trade that stops making
sense the moment the deployment has 512 MiB to live in.

    width <= 1536   text-embedding-3-small, truncated to it
    width <= 3072   text-embedding-3-large, truncated to it
    wider           not something this can embed for

**Width is not identity.** Asking for 768 dimensions gets a vector that can be
compared to a 768-dimensional index; it does not make it *the* model that built
one. A store embedded with Gemini or bge at 768 will return neighbours that are
arithmetically valid and semantically meaningless. That is what the profiler's
round-trip check is for, and it is why this module is not the last word on
whether a connected store can be searched.

**Every call here is on the answer path**, which docs/04-latency.md used to
forbid outright. What keeps that honest now is not avoidance but the budget:
the optional stages price a call before making it and skip when it will not
fit, and the harness deadline bounds the rest.
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


def embed_texts(texts: list[str], dim: int, *, settings: Settings) -> np.ndarray:
    """`len(texts)` embeddings, at exactly `dim` dimensions, in one call.

    One request for the whole batch rather than one per text. That is not a
    micro-optimisation here: a round trip is the entire cost of this module, so
    a loop over twenty passages would be twenty times the latency of the same
    twenty sent together. `/embeddings` takes an array and returns them in
    order, and the order is asserted below rather than trusted.

    Raises rather than returning wrong-width or short results: a caller that
    got 1536 floats where it asked for 768, or nineteen vectors for twenty
    passages, would hand them to Postgres or to a matrix product and get an
    error from a layer that cannot explain it.
    """
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    if not settings.openai_api_key:
        raise RemoteEmbedUnavailable(
            "Embedding needs OPENAI_API_KEY. There is no local model any more "
            "(docs/25-no-local-embedder.md)."
        )

    model = model_for(dim)
    try:
        response = get_client().post(
            f"{settings.openai_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": model, "input": texts, "dimensions": dim},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise RemoteEmbedUnavailable(
            f"{model} answered {error.response.status_code}."
        ) from error
    except Exception as error:
        raise RemoteEmbedUnavailable(f"could not reach {model}: {error}") from error

    try:
        data = response.json()["data"]
    except (KeyError, TypeError) as error:
        raise RemoteEmbedUnavailable(f"{model} returned no embeddings.") from error

    if len(data) != len(texts):
        raise RemoteEmbedUnavailable(
            f"{model} returned {len(data)} embeddings for {len(texts)} inputs."
        )

    # Sorted by `index`, not taken in arrival order. The API documents that it
    # may return them out of order, and a silently permuted batch is the worst
    # kind of wrong here: every vector is valid, every one belongs to the wrong
    # passage, and nothing downstream can tell.
    try:
        ordered = sorted(data, key=lambda row: row["index"])
        vectors = np.asarray([row["embedding"] for row in ordered], dtype=np.float32)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RemoteEmbedUnavailable(f"{model} returned no embedding.") from error

    if vectors.shape != (len(texts), dim):
        raise RemoteEmbedUnavailable(
            f"{model} returned {vectors.shape} vectors, not {(len(texts), dim)}."
        )

    # `text-embedding-3` returns unit-length vectors at the native width, and
    # OpenAI's guidance is to renormalise after truncation. The API does this
    # when `dimensions` is passed, so this is a cheap assertion of a promise
    # rather than a correction — but cosine on a non-unit vector is still
    # correct, and inner product on one is not, so it costs nothing to be sure.
    #
    # Every caller in this app takes a dot product and calls it a cosine, so
    # this is load-bearing for all of them, not only the connected stores.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def embed_query(text: str, dim: int, *, settings: Settings) -> np.ndarray:
    """One embedding, at exactly `dim` dimensions."""
    return embed_texts([text], dim, settings=settings)[0]
