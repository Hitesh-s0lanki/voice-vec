"""Application entrypoint.

Run with:  uv run python -m src.main
"""

import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import api_router
from src.chat.store import ChatStore, get_chat_store
from src.core.config import get_settings
from src.core.db import get_db
from src.datasets.service import get_dataset_service
from src.rag.cache import get_cache
from src.rag.embed import get_embedder
from src.voice.http import close_client

log = logging.getLogger("vec.startup")

# What `/docs` and `/openapi.json` say this server is. The title and version
# come from settings because a deployment may override them; these two do not
# change per deployment, so they live beside the app rather than in the
# environment.
SUMMARY = "Speech in, speech out, in the language it was spoken."

DESCRIPTION = """\
Vec hears a question in any of the twenty-two languages Sarvam's Saaras
transcribes, answers it, and reads the answer back with Bulbul in the same
language — every stage streaming into the next, and interruptible mid-word.

This server holds **no corpus of its own**. A question is answered from the
vector store its asker connected — Pinecone, Astra, or their own Postgres —
and an asker who has connected nothing is answered by the model and the tools
they *did* connect.

`WS /voice/ws` is the spoken loop; `POST /ask` is the same pipeline with the
microphone taken off both ends. `GET /metrics` reports what each stage
actually cost.
"""



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open what a request should not have to open, and say what is wired up.

    There is no model to warm here any more. The ONNX session this used to load
    before accepting traffic is gone (docs/25-no-local-embedder.md) — embedding
    is a call to `text-embedding-3` now, so there is nothing to hold warm and
    nothing whose load time would otherwise land in the first request. Boot is
    correspondingly close to instant.

    What is left is the two things that are expensive once and cheap forever:
    the conversation schema, and the answer cache's connection. Both degrade
    rather than fail — a database that is missing or unreachable means "this
    build does not remember anything", not a boot failure.
    """
    settings = get_settings()
    chat: ChatStore = get_chat_store()

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

    if chat.configured:
        try:
            await anyio.to_thread.run_sync(chat.ensure_schema)
            log.info("conversations ready at %s", get_db().location)
        except Exception as error:  # a wrong DSN, a database still booting…
            log.warning("conversations unavailable (%s) — turns will not be saved", error)
    else:
        log.info("DATABASE_URL unset — conversations are not saved")

    # Configuration, not a probe: whether the key *works* costs a call, and the
    # callers are all built to survive one failing. Logged because "no
    # retrieval, no reranking, no grounding gate and word-overlap capability
    # discovery" is a very different deployment from the one somebody thinks
    # they booted, and nothing else would say so.
    embedder = get_embedder()
    if embedder.ready:
        log.info("embedder %s at %d dims", embedder.model_name, embedder.dim)
    else:
        log.warning(
            "no embedder — set OPENAI_API_KEY, or nothing can be searched, "
            "reranked or grounded (docs/25-no-local-embedder.md)"
        )

    # Open the answer cache here rather than on the first question. Two
    # reasons, and the second is the one that matters: connecting costs a
    # round trip and a possible index creation, which would land inside
    # somebody's first answer; and until it has connected, the cache cannot
    # say whether it got the semantic layout or fell back to exact-only —
    # so `/voice/config` would report "not connected yet" to a client that
    # asked precisely so it could tell the user.
    cache = get_cache()
    if cache.configured:
        await anyio.to_thread.run_sync(cache.warm)
        log.info("answer cache %s", cache.describe())
    else:
        log.info("answer cache off (REDIS_URL unset) — every question runs the full path")

    yield

    get_cache().close()

    # Sealed DuckDB handles onto materialised datasets, plus the build pool.
    # Closed before the process goes away so a rebuild scheduled seconds ago
    # does not write into a file the next boot is opening.
    get_dataset_service().close()

    await close_client()

    # Conversations, connectors, profiles and datasets all check out of this
    # one. Nothing searched through it — a connected store brings its own.
    get_db().close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        summary=SUMMARY,
        description=DESCRIPTION,
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
