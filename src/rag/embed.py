"""Local ONNX embedding. No network call sits on the answer path.

`intfloat/multilingual-e5-small` — 384 dims, ~118M params, genuinely
multilingual including Devanagari. fastembed does not ship it as a built-in, so
it is registered as a custom model pointing at the ONNX export in the same HF
repo (docs/02-architecture.md).

e5 has one rule that fails silently when broken: queries are prefixed
`query: `, passages `passage: `. Both prefixes are applied here and nowhere
else, so no caller can forget one.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from src.core.config import Settings, get_settings

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_registered: set[str] = set()
_registry_lock = threading.Lock()


def _register(settings: Settings) -> None:
    """Teach fastembed about the e5 ONNX export. Idempotent."""
    with _registry_lock:
        if settings.embed_model in _registered:
            return

        known = {m["model"] for m in TextEmbedding.list_supported_models()}
        if settings.embed_model not in known:
            TextEmbedding.add_custom_model(
                model=settings.embed_model,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=settings.embed_model),
                dim=settings.embed_dim,
                model_file=settings.embed_model_file,
            )

        _registered.add(settings.embed_model)


class Embedder:
    """Wraps one ONNX session. Load once at boot, never per request."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._settings.embed_dim

    @property
    def model_name(self) -> str:
        return self._settings.embed_model

    @property
    def ready(self) -> bool:
        return self._model is not None

    def warm(self) -> float:
        """Load the session and run one forward pass. Returns seconds spent.

        Called from the app lifespan so the first real request does not pay
        model load — otherwise P100 measures our startup, not our pipeline.
        """
        started = time.perf_counter()
        with self._lock:
            if self._model is None:
                _register(self._settings)
                self._model = TextEmbedding(
                    model_name=self._settings.embed_model,
                    cache_dir=self._settings.embed_cache_dir,
                    threads=self._settings.embed_threads,
                )
                list(self._model.embed([QUERY_PREFIX + "warm"]))
        return time.perf_counter() - started

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self._model is None:
            self.warm()
        assert self._model is not None
        vectors = list(self._model.embed(texts, batch_size=batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text], batch_size=1)[0]

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._embed([PASSAGE_PREFIX + t for t in texts], batch_size=batch_size)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Real token lengths, padding excluded.

        Chunk sizes are set in tokens, never characters: Indic scripts cost more
        tokens than equivalent English, and truncation at the model's limit is
        silent (docs/03-chunking.md).

        `len(encoding.ids)` is the wrong measure here — the tokeniser pads every
        encoding in a batch out to the longest member, so counting ids reports
        the batch maximum for every text and makes a corpus of short passages
        look like it is entirely at the cap. The attention mask is the content.
        """
        if self._model is None:
            self.warm()

        tokenizer = getattr(getattr(self._model, "model", None), "tokenizer", None)
        if tokenizer is None:
            return []

        encoded = tokenizer.encode_batch([PASSAGE_PREFIX + t for t in texts])
        return [sum(e.attention_mask) for e in encoded]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings())
