# voice-vec

Speak in Hindi, Tamil, Kannada — any of twenty-two languages — and be answered **out loud,
in the language you spoke**, starting about a second after you stop talking.

[Sarvam AI](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview) hears
(Saaras) and speaks (Bulbul); the reply is written by OpenAI when its key is present and by
Sarvam when it is not. Every stage streams into the next — the answer is being read aloud
while it is still being written — and talking over it stops it mid-word.
[`docs/11-voice.md`](docs/11-voice.md) is how, and what it measures.

> **Retrieval is switched off** (`RAG_ENABLED=false`). The RAG pipeline in
> [`docs/`](docs/) is built and measured; the spoken turn answers conversationally for now.
> Turning it back on is one environment variable — the seam is `VoiceSession._retrieve()`.

## Layout

| Directory | What's in it |
| --- | --- |
| [`frontend/`](frontend/) | the Next.js app — UI, mic capture, socket, streaming playback |
| [`src/`](src/) | the FastAPI backend — the voice loop and the RAG pipeline, uv-managed (Python 3.13) |
| [`scripts/`](scripts/) | ingest and evaluation, run offline |
| [`docs/`](docs/) | design docs; [`11-voice.md`](docs/11-voice.md) is what runs today, [`09-v1.md`](docs/09-v1.md) is the retrieval half |
| [`tests/`](tests/) | unit tests for the parts that fail silently |

## Setup

Backend — one key, no index, no Docker:

```bash
cp .env.example .env              # paste SARVAM_API_KEY from dashboard.sarvam.ai
uv run python -m src.main         # http://127.0.0.1:8001/health
```

With `RAG_ENABLED=false` there is nothing to build first: boot skips the ONNX session and
the vector index entirely, so the server is up in under a second. Add `OPENAI_API_KEY` to
have OpenAI write the replies and cover the languages Bulbul does not speak.

Frontend:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev                       # http://localhost:3002
```

The browser opens the FastAPI socket directly — a Next route handler speaks
request/response, and a spoken turn is a two-way stream with an interrupt. No key is
exposed by that: Sarvam and OpenAI are only ever called from the Python side.

<details>
<summary>Turning retrieval back on</summary>

Retrieval needs a Postgres with `pgvector`. Neon, or locally:

```bash
docker run -d --name vec-pg -p 5432:5432 -e POSTGRES_PASSWORD=vec \
  -e POSTGRES_USER=vec -e POSTGRES_DB=vec pgvector/pgvector:pg17
```

```bash
uv run python -m scripts.migrate                # check the DSN, report the round trip
uv run python -m scripts.ingest --rows 2000 --recreate
uv run python -m scripts.suggestions --n 12     # with the API running
# then set RAG_ENABLED=true and restart
```

The first ingest downloads ~9 MB of MSMARCO-XI rows and the embedding model, then embeds
~17k passages — about 12 minutes, all of it local CPU. Run `migrate` first: it costs a
second and catches a wrong DSN or a too-distant region before the embedding does.

Postgres takes concurrent writers, so ingest and the API can run together. The HNSW and
GIN indexes are built once at the end of ingest rather than maintained per insert.

</details>

## How it works

Tap the orb — or press space — and talk. It closes the take itself when you stop, sends
what it heard to Saaras, writes a reply in your language, and starts speaking before the
reply is finished. Tap again while it talks and it stops mid-word.

```
mic ─► Saaras ─► transcript ─► chat model ─┬─► text on screen
                                           └─► segmenter ─► Bulbul ─► PCM ─► speakers
```

Measured, Hindi, from the end of the take: transcript 841 ms, first token 1,072 ms, first
audio out **1,879 ms** — with the reply still being written 300 ms after the speakers
start. Full table in [`docs/11-voice.md`](docs/11-voice.md).

| Piece | File |
| --- | --- |
| Mic, socket, playback, barge-in | [`frontend/src/hooks/use-voice-session.ts`](frontend/src/hooks/use-voice-session.ts) |
| Streaming PCM player | [`frontend/src/lib/pcm-player.ts`](frontend/src/lib/pcm-player.ts) |
| The turn, end to end | [`src/services/voice_service.py`](src/services/voice_service.py) |
| Hearing, replying, speaking | [`src/voice/`](src/voice/) |
| Where a reply gets cut for speech | [`src/voice/segment.py`](src/voice/segment.py) |
| The wire contract | [`src/schemas/voice.py`](src/schemas/voice.py) ↔ [`voice-protocol.ts`](frontend/src/lib/voice-protocol.ts) |
| The retrieval pipeline (off) | [`src/services/ask_service.py`](src/services/ask_service.py), [`src/rag/`](src/rag/) |
| Index build | [`scripts/ingest.py`](scripts/ingest.py) |
| Measured numbers | [`scripts/evaluate.py`](scripts/evaluate.py) → `reports/results.json` |

`src/` is layered: [`controllers/`](src/controllers/) own the HTTP surface,
[`services/`](src/services/) own the orchestration, [`rag/`](src/rag/) owns the pipeline
stages, [`schemas/`](src/schemas/) own the wire models, and [`api/router.py`](src/api/router.py)
mounts every controller.

| Endpoint | What it does |
| --- | --- |
| `WS /voice/ws` | a spoken conversation — audio in, text and PCM out, interruptible |
| `GET /voice/config` | which providers are wired up, and every language it hears |
| `POST /voice/transcribe` | audio in, text out — Saaras on its own |
| `POST /voice/speak` | text in, streamed audio out (`curl -N` hears it) |
| `GET /health` | process, embedder, index size, and what the last ingest built |
| `POST /ask` | transcript in, grounded answer or honest abstention out (abstains while retrieval is off) |
| `GET /suggestions` | questions this index demonstrably answers |
| `GET /metrics` | live P50/P70/P95/P99/P100 per stage, with N |
| `GET /metrics/recent` | the last few request traces — why did it refuse *that* query? |

### Constraints worth knowing

- Sarvam's REST endpoint accepts **30s of audio per request**; recording auto-stops at 29s.
- Sarvam matches content types **exactly**. `MediaRecorder` reports
  `audio/webm;codecs=opus`, which is rejected — the codec parameter is stripped before
  upload.
- Bulbul speaks **11** of the 22 languages Saaras hears. The other 11 are transcribed and
  answered correctly and then synthesised by OpenAI; without `OPENAI_API_KEY` they are read
  by an Indian-English voice, which is imperfect and better than silence.
- Speakers are model-specific: `bulbul:v3` rejects `bulbul:v2`'s names (`anushka` and
  friends) with a 400 that lists the valid ones.
- Recording requires a secure context: `localhost` works, any other host needs HTTPS.
- Audio playback needs a user gesture — the first tap on the orb is what unlocks it.
- `abstained` is a **success**, not an error. It renders as a real reply.
