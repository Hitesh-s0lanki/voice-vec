"""One HTTP client for the whole voice path.

Every provider call here is a streaming one, and a fresh client per call would
pay a TLS handshake in front of each — roughly 100 ms on top of a budget where
the whole point is the first sound arriving quickly. So the connection pool is
opened once and closed with the app (`src/main.py`).
"""

from __future__ import annotations

import httpx

# Connecting should fail fast; reading must not. `read` is the gap *between*
# chunks of a stream, not the length of the stream: a synthesiser sending audio
# steadily for a minute never trips it, and one that has died trips it in five
# seconds.
_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            headers={"user-agent": "voice-vec/0.1"},
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


class ProviderError(RuntimeError):
    """A provider refused or failed. Carries something a user can read."""

    def __init__(self, message: str, *, status: int | None = None, provider: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.provider = provider


def read_error(payload: object, fallback: str) -> str:
    """Dig the human-readable half out of an error body.

    Sarvam nests it under `error.message`, OpenAI under `error.message` too but
    sometimes `message`; a plain string body happens as well.
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        if isinstance(payload.get("message"), str):
            return payload["message"]
        if isinstance(payload.get("detail"), str):
            return payload["detail"]
    if isinstance(payload, str) and payload.strip():
        return payload.strip()[:300]
    return fallback
