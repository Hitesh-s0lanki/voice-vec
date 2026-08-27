# voice-vec

Speak in Hindi, Tamil, Kannada — any of twenty-two languages — and be answered **out loud,
in the language you spoke**, starting about a second after you stop talking.

[Sarvam AI](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview) hears
(Saaras) and speaks (Bulbul); the reply is written by OpenAI when its key is present and by
Sarvam when it is not. Every stage streams into the next — the answer is being read aloud
while it is still being written — and talking over it stops it mid-word.
[`docs/11-voice.md`](docs/11-voice.md) is how, and what it measures.

> **Retrieval is on** (`RAG_ENABLED=true`), over 19,870 Hindi passages from MSMARCO-XI —
> and it answers questions asked in *any* language, not just Hindi: the embedder is
> cross-lingual and every chunk carries its English original.
> [`docs/13-cross-lingual.md`](docs/13-cross-lingual.md) measures what that costs.
> One caveat before reading the latency numbers below: the 200 ms budget in
> [`docs/04-latency.md`](docs/04-latency.md) was measured with the index in-process. Over a
> Neon instance 66 ms of round trip away, a full answered query now measures **221 ms** —
> close, and not inside. An abstention, which stops at the search, is 87 ms.

## Layout

| Directory | What's in it |
| --- | --- |
| [`frontend/`](frontend/) | the Next.js app — UI, mic capture, socket, streaming playback |
| [`src/`](src/) | the FastAPI backend — the voice loop, saved conversations, and the RAG pipeline, uv-managed (Python 3.13) |
| [`scripts/`](scripts/) | ingest and evaluation, run offline |
| [`docs/`](docs/) | design docs; [`11-voice.md`](docs/11-voice.md) is the spoken loop, [`09-v1.md`](docs/09-v1.md) the retrieval half, [`15-effort.md`](docs/15-effort.md) the five rungs the effort slider picks between, [`16-memory.md`](docs/16-memory.md) what the agent remembers |
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

`DATABASE_URL` is optional and separate from that switch — it is what saves conversations,
whether or not retrieval is on. Set it and `uv run python -m scripts.migrate` creates the
two tables in a second. Leave it unset and the voice loop runs exactly as before; the URL
just never gains a `/c/…` and nothing is written down.

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
<summary>Rebuilding the index</summary>

Retrieval needs a Postgres with `pgvector`. Neon, or locally:

```bash
docker run -d --name vec-pg -p 5432:5432 -e POSTGRES_PASSWORD=vec \
  -e POSTGRES_USER=vec -e POSTGRES_DB=vec pgvector/pgvector:pg17
```

```bash
uv run python -m scripts.migrate                # check the DSN, report the round trip
uv run python -m scripts.ingest --rows 2000 --recreate
uv run python -m scripts.suggestions --n 12     # with the API running
uv run python -m scripts.crosslingual --n 200   # re-sweep the cross-lingual thresholds
# then set RAG_ENABLED=true and restart
```

Both of Gate 2's threshold pairs are properties of *this* index, so a re-ingest invalidates
them. `scripts/evaluate.py` re-sweeps the same-language pair, `scripts/crosslingual.py` the
cross-lingual one.

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

The first thing you say opens a conversation: the address bar becomes `/c/conv_…` while the
reply is still being written, and every turn after it is saved there. Reload that URL and
the model picks up where it left off — it is handed its own history back out of Postgres
before the socket is ready. The History panel lists the rest of them.

Signed out, they belong to the browser. Sign in with Clerk and they follow the account
instead — including everything said before, claimed in one statement at sign-in — so the
same conversations are there on the next device. Identity is always a **verified Clerk
token**, on the socket as well as over HTTP; there is no user id header to type.
[`docs/12-conversations.md`](docs/12-conversations.md) is how, and what is stored for a turn
you talked over.

A reloaded conversation gets its own words back; a **new** one used to start from nothing.
Now it does not. Every turn is also mirrored into [Redis Agent
Memory](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/), which distils
durable facts out of them in the background — *the user is vegetarian*, *they prefer being
answered in Tamil* — and the next conversation opens having searched them. Postgres stays the
source of truth for what was **said**; this is what was **learned**. It is scoped to one
person, capped at three facts, floored on similarity, run beside retrieval rather than before
it, and told in the prompt never to recite itself — because a remembered fact is asserted in
the model's opening sentence, which is the worst place to be wrong. Unconfigured, the agent
simply forgets between conversations and nothing else changes.
[`docs/16-memory.md`](docs/16-memory.md) is how, including how it and the answer cache share
one 30 MB instance without evicting each other.

Signing in also unlocks the Connectors panel, where **you attach services with your own
credentials** — none are baked into this server. Composio links Gmail, Slack or Notion into
*your* Composio project; Pinecone, Astra or your own Postgres becomes where *your* questions
get searched, falling back to this deployment's index when you have connected nothing.
Once a toolkit is linked, a spoken turn can actually **run** it — the agent decides, calls,
and then answers from what came back, and every call is written down beside the turn that
caused it. Credentials are encrypted before they reach Postgres and only ever come back as
their last four characters. Adding a connector is a backend change: each one declares its own fields
and the panel builds the form from that. Those routes are the one part of the app with no
anonymous path through them — a saved conversation belongs to whoever holds the browser, but
a credential does not. [`docs/13-connectors.md`](docs/13-connectors.md) is how.

The **Effort** slider in the rail picks how the question gets answered, and each position is
a different retrieval architecture rather than a different amount of the same one: *Lookup*
searches and shows the passages with no model involved at all, *Grounded* lifts a sentence out
of one and checks it came from there, *Deep* fuses keyword and vector search then has a model
write the answer up, *Corrective* grades what it found and searches again with a better query,
*Adaptive* decides where to look before looking and repairs its own answer afterwards. The
level is a **ceiling, not a floor** — a repeat question is answered from Redis at any of them,
and a rung whose model is unreachable falls back to the one below and says so rather than
failing. Only the first two make no network call after the transcript arrives, which is what
the 200 ms is a claim about; the rest are reported against their own budget instead of hiding
behind it. [`docs/15-effort.md`](docs/15-effort.md) is the whole ladder.

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
| Conversations, saved and reloadable | [`src/chat/store.py`](src/chat/store.py), [`docs/12-conversations.md`](docs/12-conversations.md) |
| Connectors, attached per signed-in user | [`src/connectors/`](src/connectors/), [`docs/13-connectors.md`](docs/13-connectors.md) |
| The agent running a user's tools | [`src/integrations/agent.py`](src/integrations/agent.py), [`src/chat/tools.py`](src/chat/tools.py) |
| Where a user's vectors are searched | [`src/rag/backends/`](src/rag/backends/) |
| The retrieval pipeline | [`src/services/ask_service.py`](src/services/ask_service.py), [`src/rag/`](src/rag/) |
| The effort ladder, rung by rung | [`src/rag/effort.py`](src/rag/effort.py), [`docs/15-effort.md`](docs/15-effort.md) |
| The answer cache, in Redis | [`src/rag/cache.py`](src/rag/cache.py) |
| What the agent remembers between conversations | [`src/memory/store.py`](src/memory/store.py), [`docs/16-memory.md`](docs/16-memory.md) |
| Asking in a language the index does not hold | [`docs/13-cross-lingual.md`](docs/13-cross-lingual.md), [`scripts/crosslingual.py`](scripts/crosslingual.py) |
| Index build | [`scripts/ingest.py`](scripts/ingest.py) |
| Measured numbers | [`scripts/evaluate.py`](scripts/evaluate.py) → `reports/results.json` |

`src/` is layered: [`controllers/`](src/controllers/) own the HTTP surface,
[`services/`](src/services/) own the orchestration, [`rag/`](src/rag/) owns the pipeline
stages, [`schemas/`](src/schemas/) own the wire models, and [`api/router.py`](src/api/router.py)
mounts every controller.

| Endpoint | What it does |
| --- | --- |
| `WS /voice/ws` | a spoken conversation — audio in, text and PCM out, interruptible |
| `GET /conversations` | your saved conversations, newest first |
| `POST /conversations/adopt` | sign-in: claim what this browser said anonymously |
| `GET /conversations/{id}` | one thread — every question and reply in it |
| `DELETE /conversations/{id}` | it and its messages |
| `GET /voice/config` | which providers are wired up, and every language it hears |
| `POST /voice/transcribe` | audio in, text out — Saaras on its own |
| `POST /voice/speak` | text in, streamed audio out (`curl -N` hears it) |
| `GET /health` | process, embedder, index size, and what the last ingest built |
| `POST /ask` | transcript in, grounded answer or honest abstention out — in any language |
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
