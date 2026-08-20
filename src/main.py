"""Application entrypoint.

Run with:  uv run python -m src.main
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import api_router
from src.core.config import Settings, get_settings
from src.rag.embed import Embedder, get_embedder
from src.rag.store import VectorStore, get_store
from src.voice.http import close_client

log = logging.getLogger("vec.startup")


def _touch(embedder: Embedder, store: VectorStore, settings: Settings) -> None:
    """One tiny embed + search, to keep both paths hot."""
    vector = embedder.embed_query("warm")
    store.search(vector, strategies=settings.search_strategies, limit=1)


async def keepalive(embedder: Embedder, store: VectorStore, settings: Settings) -> None:
    """Keep the ONNX session and the vector matrix warm between questions.

    Warming once at boot only helps the first request. Measured here, 30 s of
    silence costs the next request ~60 ms across embed and search — enough to
    push an answered query past the 200 ms budget — and interactive voice use is
    nothing *but* idle gaps. Failures are logged and the loop continues: a
    keepalive that dies takes the latency with it, silently.
    """
    while True:
        await asyncio.sleep(settings.keepalive_seconds)
        try:
            await anyio.to_thread.run_sync(_touch, embedder, store, settings)
        except Exception as error:  # index missing, store closed, …
            log.debug("keepalive skipped: %s", error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the embedder and open the index *before* the first request.

    The ONNX session takes seconds to load. Paying that on request one means
    P100 measures our startup rather than our pipeline (docs/04-latency.md), so
    it happens here, where the server is not yet accepting traffic.

    None of it runs with retrieval off. The voice loop calls no local model, so
    warming a 465 MB ONNX session it will not use costs seconds of boot and
    holds the Qdrant path open against `scripts/ingest.py` for nothing.
    """
    settings = get_settings()
    store: VectorStore | None = None
    warm_loop: asyncio.Task | None = None

    llm = settings.resolve_llm()
    log.info(
        "voice: stt=%s llm=%s/%s tts=%s",
        "sarvam" if settings.sarvam_ready else "openai" if settings.openai_ready else "none",
        llm.provider if llm.ready else "none",
        llm.model if llm.ready else "-",
        "sarvam" if settings.sarvam_ready else "openai" if settings.openai_ready else "none",
    )
    if not llm.ready:
        log.warning("no reply model — set SARVAM_API_KEY or OPENAI_API_KEY in .env")

    if settings.rag_enabled:
        embedder = get_embedder()
        store = get_store()

        seconds = await anyio.to_thread.run_sync(embedder.warm)
        log.info("embedder %s warm in %.2fs", embedder.model_name, seconds)

        try:
            chunks = await anyio.to_thread.run_sync(store.warm)
            log.info("index %s ready at %s — %d chunks", store.collection, store.location, chunks)

            if settings.keepalive_seconds > 0:
                warm_loop = asyncio.create_task(keepalive(embedder, store, settings))
                log.info("keepalive every %ds", settings.keepalive_seconds)
        except Exception as error:  # index not built yet — /health says so, /ask abstains
            log.warning("index unavailable (%s) — run scripts/ingest.py", error)
    else:
        log.info("retrieval disabled (RAG_ENABLED=false) — /ask abstains, voice answers directly")

    yield

    if warm_loop is not None:
        warm_loop.cancel()
        with suppress(asyncio.CancelledError):
            await warm_loop

    await close_client()

    if store is not None:
        store.close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    main()
