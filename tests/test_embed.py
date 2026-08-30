"""Embedding over the network, and the things that go silently wrong there.

The local ONNX model is gone (docs/25-no-local-embedder.md) and every vector
now comes back from `text-embedding-3`. That swap moves four guarantees from a
library into this codebase, and each one fails quietly rather than loudly:

  - a batch coming back in a different order than it went out, so every vector
    is valid and every one belongs to the wrong passage;
  - a vector that is not unit length, which every caller here dot-products and
    calls a cosine;
  - a batch silently short, so the claim/evidence split in the grounding gate
    slices the wrong rows;
  - a query embedded twice in one turn, which is a round trip nobody sees on a
    latency graph but everybody hears.

Nothing here touches the network: the transport is stubbed at `get_client`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import Settings
from src.rag import remote_embed
from src.rag.embed import Embedder
from src.rag.remote_embed import (
    MAX_DIM,
    RemoteEmbedUnavailable,
    embed_query,
    embed_texts,
    model_for,
)


def _settings(**overrides) -> Settings:
    base = {"openai_api_key": "sk-test", "embed_dim": 8, "embed_cache_size": 4}
    base.update(overrides)
    return Settings(**base)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeClient:
    """Records every request, and answers with unit vectors of the right width."""

    def __init__(self, payload=None) -> None:
        self.calls: list[dict] = []
        self._payload = payload

    def post(self, url, *, headers, json):  # noqa: A002 - httpx's own name
        self.calls.append(json)
        if self._payload is not None:
            return FakeResponse(self._payload)
        dim = json["dimensions"]
        rows = []
        for index, _ in enumerate(json["input"]):
            vector = np.zeros(dim, dtype="float64")
            vector[index % dim] = 1.0
            rows.append({"index": index, "embedding": vector.tolist()})
        return FakeResponse({"data": rows})


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(remote_embed, "get_client", lambda: fake)
    return fake


class TestWidthRouting:
    def test_the_cheaper_model_serves_every_width_it_can(self):
        assert model_for(1) == "text-embedding-3-small"
        assert model_for(768) == "text-embedding-3-small"
        assert model_for(1536) == "text-embedding-3-small"

    def test_the_larger_model_only_above_the_smaller_ones_native_width(self):
        assert model_for(1537) == "text-embedding-3-large"
        assert model_for(MAX_DIM) == "text-embedding-3-large"

    def test_a_width_nothing_can_produce_is_refused_by_name(self):
        with pytest.raises(RemoteEmbedUnavailable) as caught:
            model_for(MAX_DIM + 1)
        assert str(MAX_DIM) in str(caught.value)


class TestOneCallPerBatch:
    def test_twenty_passages_are_one_request(self, client):
        vectors = embed_texts([f"passage {i}" for i in range(20)], 8, settings=_settings())
        assert len(client.calls) == 1, "a loop here would be twenty round trips"
        assert vectors.shape == (20, 8)

    def test_an_empty_batch_costs_no_request_at_all(self, client):
        vectors = embed_texts([], 8, settings=_settings())
        assert client.calls == []
        assert vectors.shape == (0, 8)

    def test_the_requested_width_is_what_is_asked_for(self, client):
        embed_texts(["x"], 768, settings=_settings())
        assert client.calls[0]["dimensions"] == 768
        assert client.calls[0]["model"] == "text-embedding-3-small"


class TestTheQuietFailures:
    def test_a_batch_returned_out_of_order_is_put_back_in_order(self, monkeypatch):
        """The one that no downstream assertion would ever catch.

        Every vector is well-formed and the shape is right; only the pairing of
        vector to passage is wrong. The API documents that `index` is the
        authority, so it is sorted on rather than trusted to arrive sorted.
        """
        shuffled = {
            "data": [
                {"index": 2, "embedding": [0.0, 0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            ]
        }
        monkeypatch.setattr(remote_embed, "get_client", lambda: FakeClient(shuffled))
        vectors = embed_texts(["first", "second", "third"], 3, settings=_settings())
        assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
        assert np.allclose(vectors[1], [0.0, 1.0, 0.0])
        assert np.allclose(vectors[2], [0.0, 0.0, 1.0])

    def test_vectors_come_back_unit_length(self, monkeypatch):
        """Every caller takes a dot product and calls it a cosine."""
        payload = {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}
        monkeypatch.setattr(remote_embed, "get_client", lambda: FakeClient(payload))
        vector = embed_texts(["x"], 2, settings=_settings())[0]
        assert np.isclose(np.linalg.norm(vector), 1.0)
        assert np.allclose(vector, [0.6, 0.8])

    def test_a_short_batch_raises_rather_than_being_sliced_wrong(self, monkeypatch):
        payload = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        monkeypatch.setattr(remote_embed, "get_client", lambda: FakeClient(payload))
        with pytest.raises(RemoteEmbedUnavailable) as caught:
            embed_texts(["one", "two"], 2, settings=_settings())
        assert "1 embeddings for 2 inputs" in str(caught.value)

    def test_the_wrong_width_back_raises_rather_than_reaching_postgres(self, monkeypatch):
        payload = {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}
        monkeypatch.setattr(remote_embed, "get_client", lambda: FakeClient(payload))
        with pytest.raises(RemoteEmbedUnavailable):
            embed_texts(["x"], 2, settings=_settings())

    def test_no_key_says_so_and_never_reaches_the_network(self, monkeypatch):
        def explode():
            raise AssertionError("should not have opened a client")

        monkeypatch.setattr(remote_embed, "get_client", explode)
        with pytest.raises(RemoteEmbedUnavailable) as caught:
            embed_texts(["x"], 8, settings=_settings(openai_api_key=""))
        assert "OPENAI_API_KEY" in str(caught.value)

    def test_a_provider_that_is_down_is_unavailable_not_a_crash(self, monkeypatch):
        class Dead:
            def post(self, *args, **kwargs):
                raise RuntimeError("connection reset")

        monkeypatch.setattr(remote_embed, "get_client", lambda: Dead())
        with pytest.raises(RemoteEmbedUnavailable) as caught:
            embed_query("x", 8, settings=_settings())
        assert "could not reach" in str(caught.value)


class TestTheQueryCache:
    def test_the_same_question_is_embedded_once(self, client):
        embedder = Embedder(_settings())
        first = embedder.embed_query("how many students are enrolled?")
        second = embedder.embed_query("how many students are enrolled?")
        assert len(client.calls) == 1
        assert np.array_equal(first, second)

    def test_the_same_text_at_two_widths_is_two_entries(self, client):
        """A store's width and the app's are different spaces, not one."""
        embedder = Embedder(_settings())
        embedder.embed_query("same words", dim=8)
        embedder.embed_query("same words", dim=16)
        assert [call["dimensions"] for call in client.calls] == [8, 16]

    def test_the_cache_is_bounded(self, client):
        embedder = Embedder(_settings(embed_cache_size=2))
        for text in ("a", "b", "c"):
            embedder.embed_query(text)
        embedder.embed_query("a")  # evicted by "c", so this is a fourth call
        assert len(client.calls) == 4

    def test_a_cache_size_of_zero_disables_it(self, client):
        embedder = Embedder(_settings(embed_cache_size=0))
        embedder.embed_query("x")
        embedder.embed_query("x")
        assert len(client.calls) == 2

    def test_passages_are_not_cached(self, client):
        """They do not repeat, and holding them would keep somebody's documents."""
        embedder = Embedder(_settings())
        embedder.embed_passages(["one", "two"])
        embedder.embed_passages(["one", "two"])
        assert len(client.calls) == 2


class TestReadiness:
    def test_no_key_is_not_ready(self):
        assert Embedder(_settings(openai_api_key="")).ready is False

    def test_a_key_and_a_producible_width_is_ready(self):
        assert Embedder(_settings(embed_dim=1536)).ready is True

    def test_a_width_nothing_can_produce_is_not_ready(self):
        """Configuration this app cannot honour must not read as healthy."""
        embedder = Embedder(_settings(embed_dim=MAX_DIM + 1))
        assert embedder.ready is False
        assert "unsupported width" in embedder.model_name

    def test_readiness_costs_no_network_call(self, monkeypatch):
        def explode():
            raise AssertionError("readiness must not spend a call")

        monkeypatch.setattr(remote_embed, "get_client", explode)
        assert Embedder(_settings()).ready is True

    def test_an_empty_batch_is_the_app_width(self, client):
        embedder = Embedder(_settings(embed_dim=1536))
        assert embedder.embed_passages([]).shape == (0, 1536)
        assert client.calls == []


class TestABackendAsksForItsOwnWidth:
    """A connected index cannot be searched with a vector of another size.

    The branch that used to pick a local model when the widths happened to
    agree is gone; what is left has to route the store's own width through
    every time, and turn a provider failure into the thing the ladder already
    abstains on rather than a 500.
    """

    def _backend(self, dim, monkeypatch, client, **settings_overrides):
        """A backend whose embedder is built here, not from the environment.

        `get_embedder` is a module-level singleton over the real `Settings`,
        which reads `.env`. Left alone, these tests pass on a laptop that has an
        OPENAI_API_KEY and fail in CI, which is the wrong way round for a test
        about routing a width.
        """
        from src.rag.backends.pinecone import PineconeBackend

        settings = _settings(**settings_overrides)
        monkeypatch.setattr(remote_embed, "get_client", lambda: client)
        monkeypatch.setattr("src.rag.embed.get_embedder", lambda: Embedder(settings))
        return PineconeBackend({"api_key": "k", "index": "i", "dim": str(dim)}), settings

    def test_the_stores_width_is_what_is_requested(self, monkeypatch):
        client = FakeClient()
        backend, _ = self._backend(768, monkeypatch, client)
        backend.embed_query("a question")
        assert client.calls[0]["dimensions"] == 768

    def test_a_store_with_no_recorded_width_falls_back_to_the_apps(self, monkeypatch):
        client = FakeClient()
        backend, settings = self._backend(0, monkeypatch, client)
        backend.embed_query("a question")
        assert client.calls[0]["dimensions"] == settings.embed_dim

    def test_a_missing_key_reads_as_the_store_being_unavailable(self, monkeypatch):
        from src.rag.backends.base import StoreUnavailable

        backend, _ = self._backend(
            768, monkeypatch, FakeClient(), openai_api_key=""
        )
        with pytest.raises(StoreUnavailable) as caught:
            backend.embed_query("a question")
        assert "OPENAI_API_KEY" in str(caught.value)
