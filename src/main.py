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
from src.chat.store import ChatStore, get_chat_store
from src.core.config import Settings, get_settings
from src.core.db import get_db
from src.datasets.service import get_dataset_service
from src.rag.cache import get_cache
from src.rag.embed import Embedder, get_embedder
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



async def keepalive(embedder: Embedder, settings: Settings) -> None:
    """Keep the ONNX session warm between questions.

    Warming once at boot only helps the first request. Measured here, 30 s of
    silence costs the next request ~30 ms on the embed alone — enough to eat a
    sixth of the 200 ms budget — and interactive voice use is nothing *but*
    idle gaps. The search half of this loop is gone with the deployment store:
    what gets searched now is a pool against somebody else's database, and
    holding *that* warm on a timer is not this process's business.

    Failures are logged and the loop continues: a keepalive that dies takes the
    latency with it, silently.
    """
    while True:
        await asyncio.sleep(settings.keepalive_seconds)
        try:
            await anyio.to_thread.run_sync(embedder.embed_query, "warm")
        except Exception as error:  # a session torn down, a model evicted, …
            log.debug("keepalive skipped: %s", error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the embedder *before* the first request.

    The ONNX session takes seconds to load. Paying that on request one means
    P100 measures our startup rather than our pipeline (docs/04-latency.md), so
    it happens here, where the server is not yet accepting traffic.

    There is no index to open beside it. This deployment holds no corpus of its
    own — a question is answered from the vector store its asker connected — so
    what used to be an index warm and a count is now the embedder alone.

    It runs on every boot. There is no longer a switch that says whether this
    process retrieves — whether a *question* can be retrieved for is a property
    of what its asker connected, and boot cannot know that for anyone. So the
    session is loaded here rather than inside whichever turn happens to be the
    first one asked by somebody with a store attached.

    Conversation storage sits beside it, and a database that is missing or
    unreachable degrades to "this build does not remember anything" rather than
    to a boot failure.
    """
    settings = get_settings()
    chat: ChatStore = get_chat_store()
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

    if chat.configured:
        try:
            await anyio.to_thread.run_sync(chat.ensure_schema)
            log.info("conversations ready at %s", get_db().location)
        except Exception as error:  # a wrong DSN, a database still booting…
            log.warning("conversations unavailable (%s) — turns will not be saved", error)
    else:
        log.info("DATABASE_URL unset — conversations are not saved")

    embedder = get_embedder()

    seconds = await anyio.to_thread.run_sync(embedder.warm)
    log.info("embedder %s warm in %.2fs", embedder.model_name, seconds)

    if settings.keepalive_seconds > 0:
        warm_loop = asyncio.create_task(keepalive(embedder, settings))
        log.info("keepalive every %ds", settings.keepalive_seconds)

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

    if warm_loop is not None:
        warm_loop.cancel()
        with suppress(asyncio.CancelledError):
            await warm_loop

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
