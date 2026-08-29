<div align="center">

# Vec

### Speak, and be answered — out loud, in the language you spoke.

Say it in Hindi, Tamil, Kannada — any of twenty-two languages — and hear the answer
back about a second after you stop talking.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](src/main.py)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)](frontend/package.json)
[![Sarvam AI](https://img.shields.io/badge/Sarvam-Saaras%20%C2%B7%20Bulbul-FF6B35)](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview)
[![pgvector](https://img.shields.io/badge/pgvector%20%C2%B7%20Pinecone%20%C2%B7%20Astra-4169E1?logo=postgresql&logoColor=white)](docs/13-connectors.md)

<img src="images/home.png" alt="Vec's home screen: a white room, the orb at its centre, the activity feed top-right, and three suggested questions in Hindi, Tamil and English" width="900">

</div>

---

[Sarvam AI](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview) hears
(Saaras) and speaks (Bulbul); the reply is written by OpenAI when its key is present and by
Sarvam when it is not. Every stage streams into the next — the answer is being read aloud
while it is still being written — and talking over it stops it mid-word.
[`docs/11-voice.md`](docs/11-voice.md) is how, and what it measures.

> **Retrieval belongs to whoever asks.** This app holds no corpus: a question is answered
> from the vector store its asker connected — Pinecone, Astra, or their own Postgres — and a
> listener who has attached nothing is answered by the model and the tools they *did*
> connect. There is no switch to turn it on ([`docs/13-connectors.md`](docs/13-connectors.md)).
> It answers questions asked in *any* language, whatever the store was indexed in: the
> embedder is cross-lingual, and where a store is wider than 384 dimensions the query is
> embedded at that store's own width.
> [`docs/13a-cross-lingual.md`](docs/13a-cross-lingual.md) measures what that costs.

> [!NOTE]
> One caveat before the latency numbers below. The 200 ms budget in
> [`docs/04-latency.md`](docs/04-latency.md) was measured against an in-process index. Over
> a Neon instance 66 ms of round trip away, a full answered query measures **221 ms** —
> close, and not inside. An abstention, which stops at the search, is 87 ms.

## Contents

- [Quickstart](#quickstart) — two keys, no index, no Docker
- [The spoken loop](#the-spoken-loop) — what happens between the tap and the sound
- [Conversations, and what carries between them](#conversations-and-what-carries-between-them)
- [Connectors: your store, your tools](#connectors-your-store-your-tools)
- [The effort ladder](#the-effort-ladder) — five retrieval architectures on one slider
- [API surface](#api-surface)
- [Repo layout](#repo-layout)
- [Constraints worth knowing](#constraints-worth-knowing)
- [Design docs](#design-docs)

## Quickstart

### Backend

One key, no index, no Docker:

```bash
cp .env.example .env              # paste SARVAM_API_KEY from dashboard.sarvam.ai
uv run python -m src.main         # http://127.0.0.1:8001/health
```

Without uv, the same set installs from pip: `pip install -r requirements.txt`
(`requirements-dev.txt` adds the test deps). Both are pinned exports of `uv.lock` — the lock
stays the source of truth, so bump a version in `pyproject.toml`, `uv lock`, then re-export;
the header of each file carries the exact command.

Boot loads the ONNX embedding session before it accepts traffic, which is a few seconds and
the reason the first question is not the one that pays for it. Add `OPENAI_API_KEY` to have
OpenAI write the replies and cover the languages Bulbul does not speak.

There is no index to build. This app holds no corpus of its own: a question is answered from
the vector store its asker connected, and a user with nothing attached is told so rather
than answered from somewhere else. See [`docs/13-connectors.md`](docs/13-connectors.md).

`DATABASE_URL` is what this app stores its *own* data in — conversations, connected accounts
and their profiles, datasets — and is never searched. Set it and
`uv run python -m scripts.migrate` creates every table in a second. Leave it unset and the
voice loop runs exactly as before; the URL just never gains a `/c/…` and nothing is written
down.

### Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev                       # http://localhost:3002
```

The browser opens the FastAPI socket directly — a Next route handler speaks
request/response, and a spoken turn is a two-way stream with an interrupt. No key is exposed
by that: Sarvam and OpenAI are only ever called from the Python side.

<details>
<summary><b>Connecting a vector store</b></summary>

<br>

Retrieval reads whatever the signed-in user attached in the connectors panel — a Pinecone
index, an Astra collection, or a Postgres with `pgvector`. Nothing is ingested here and no
schema is imposed: `verify` finds the table, reads its columns, and the search SQL is built
from what it found ([`src/rag/columns.py`](src/rag/columns.py)), so a table holding `id` and
`chunk_text` is searchable without being a copy of anything.

A local Postgres to attach, if you want one:

```bash
docker run -d --name vec-pg -p 5432:5432 -e POSTGRES_PASSWORD=vec \
  -e POSTGRES_USER=vec -e POSTGRES_DB=vec pgvector/pgvector:pg17
```

```bash
uv run python -m scripts.migrate    # this app's own tables; check the DSN and round trip
# then attach it as a pgvector connector and it is searched from the next question
```

Gate 2's threshold pairs in `.env` were swept against one corpus and are not a property of
whatever you connect. If recall looks thin on a connected store, `RETRIEVAL_FLOOR` and
`RETRIEVAL_MARGIN` are the first dials to re-check
([`docs/06-guardrails.md`](docs/06-guardrails.md)).

</details>

## The spoken loop

Tap the orb — or press space — and talk. It closes the take itself when you stop, sends what
it heard to Saaras, writes a reply in your language, and starts speaking before the reply is
finished. Tap again while it talks and it stops mid-word.

```
mic ─► Saaras ─► transcript ─► chat model ─┬─► text on screen
                                           └─► segmenter ─► Bulbul ─► PCM ─► speakers
```

Measured in Hindi, from the end of the take — with the reply still being written 300 ms
after the speakers start:

| Stage | From end of take |
| --- | --- |
| Transcript back from Saaras | 841 ms |
| First token of the reply | 1,072 ms |
| **First audio out of the speakers** | **1,879 ms** |

Full table in [`docs/11-voice.md`](docs/11-voice.md).

| Piece | File |
| --- | --- |
| Mic, socket, playback, barge-in | [`frontend/src/hooks/use-voice-session.ts`](frontend/src/hooks/use-voice-session.ts) |
| Streaming PCM player | [`frontend/src/lib/pcm-player.ts`](frontend/src/lib/pcm-player.ts) |
| The turn, end to end | [`src/services/voice_service.py`](src/services/voice_service.py) |
| Hearing, replying, speaking | [`src/voice/`](src/voice/) |
| Where a reply gets cut for speech | [`src/voice/segment.py`](src/voice/segment.py) |
| The wire contract | [`src/schemas/voice.py`](src/schemas/voice.py) ↔ [`voice-protocol.ts`](frontend/src/lib/voice-protocol.ts) |

## Conversations, and what carries between them

The first thing you say opens a conversation: the address bar becomes `/c/conv_…` while the
reply is still being written, and every turn after it is saved there. Reload that URL and the
model picks up where it left off — it is handed its own history back out of Postgres before
the socket is ready. The History panel lists the rest of them.

Signed out, they belong to the browser. Sign in with Clerk and they follow the account
instead — including everything said before, claimed in one statement at sign-in — so the same
conversations are there on the next device. Identity is always a **verified Clerk token**, on
the socket as well as over HTTP; there is no user id header to type.
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

| Piece | File |
| --- | --- |
| Conversations, saved and reloadable | [`src/chat/store.py`](src/chat/store.py), [`docs/12-conversations.md`](docs/12-conversations.md) |
| What the agent remembers between conversations | [`src/memory/store.py`](src/memory/store.py), [`docs/16-memory.md`](docs/16-memory.md) |
| The answer cache, in Redis | [`src/rag/cache.py`](src/rag/cache.py) |

## Connectors: your store, your tools

Signing in unlocks the Connectors panel, where **you attach services with your own
credentials** — none are baked into this server. Composio links Gmail, Slack or Notion into
*your* Composio project; Pinecone, Astra or your own Postgres becomes where *your* questions
get searched.

Once a toolkit is linked, a spoken turn can actually **run** it — the agent decides, calls,
and then answers from what came back, and every call is written down beside the turn that
caused it. Credentials are encrypted before they reach Postgres and only ever come back as
their last four characters. Adding a connector is a backend change: each one declares its own
fields and the panel builds the form from that. Those routes are the one part of the app with
no anonymous path through them — a saved conversation belongs to whoever holds the browser,
but a credential does not. [`docs/13-connectors.md`](docs/13-connectors.md) is how.

| Piece | File |
| --- | --- |
| Connectors, attached per signed-in user | [`src/connectors/`](src/connectors/), [`docs/13-connectors.md`](docs/13-connectors.md) |
| The agent running a user's tools | [`src/agents/tool_agent.py`](src/agents/tool_agent.py), [`src/chat/tool_calls.py`](src/chat/tool_calls.py) |
| Every agent, and the contract under them | [`src/agents/`](src/agents/), [`docs/21-agents.md`](docs/21-agents.md) |
| What each agent is told, in markdown | [`src/prompts/`](src/prompts/) |
| The tools an agent can run | [`src/tools/`](src/tools/) |
| How the agent finds what it can reach | [`src/capabilities/`](src/capabilities/), [`docs/23-capabilities.md`](docs/23-capabilities.md) |
| Where a user's vectors are searched | [`src/rag/backends/`](src/rag/backends/) |

## The effort ladder

The **Effort** slider in the rail picks how the question gets answered, and each position is
a different retrieval architecture rather than a different amount of the same one:

| Rung | What it runs | Network after the transcript |
| --- | --- | --- |
| **Lookup** | searches and shows the passages, no model involved at all | none |
| **Grounded** | lifts a sentence out of one and checks it came from there | none |
| **Deep** | fuses keyword and vector search, then a model writes the answer up | yes |
| **Corrective** | grades what it found and searches again with a better query | yes |
| **Adaptive** | decides where to look before looking, repairs its own answer after | yes |

The level is a **ceiling, not a floor** — a repeat question is answered from Redis at any of
them, and a rung whose model is unreachable falls back to the one below and says so rather
than failing. Only the first two make no network call after the transcript arrives, which is
what the 200 ms is a claim about; the rest are reported against their own budget instead of
hiding behind it. [`docs/15-effort.md`](docs/15-effort.md) is the whole ladder.

| Piece | File |
| --- | --- |
| The retrieval pipeline | [`src/services/ask_service.py`](src/services/ask_service.py), [`src/rag/`](src/rag/) |
| The effort ladder, rung by rung | [`src/rag/effort.py`](src/rag/effort.py), [`docs/15-effort.md`](docs/15-effort.md) |
| Asking in a language the index does not hold | [`docs/13a-cross-lingual.md`](docs/13a-cross-lingual.md) |

## API surface

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
| `GET /health` | process and embedder. What a *user* can search is `GET /connectors` |
| `POST /ask` | transcript in, grounded answer or honest abstention out — in any language |
| `GET /metrics` | live P50/P70/P95/P99/P100 per stage, with N |
| `GET /metrics/recent` | the last few request traces — why did it refuse *that* query? |

Interactive docs are at `/docs` once the server is up.

## Repo layout

| Directory | What's in it |
| --- | --- |
| [`frontend/`](frontend/) | the Next.js app — UI, mic capture, socket, streaming playback |
| [`src/`](src/) | the FastAPI backend — the voice loop, saved conversations, and the RAG pipeline, uv-managed (Python 3.13) |
| [`scripts/`](scripts/) | `migrate.py` — create this app's tables and check the DSN |
| [`docs/`](docs/) | design docs; [`11-voice.md`](docs/11-voice.md) is the spoken loop, [`09-v1.md`](docs/09-v1.md) the retrieval half, [`15-effort.md`](docs/15-effort.md) the five rungs the effort slider picks between, [`16-memory.md`](docs/16-memory.md) what the agent remembers |
| [`tests/`](tests/) | unit tests for the parts that fail silently |

`src/` is layered: [`controllers/`](src/controllers/) own the HTTP surface,
[`services/`](src/services/) own the orchestration, [`agents/`](src/agents/) own the
model-driven decisions with their prompts in [`prompts/`](src/prompts/) and what they run in
[`tools/`](src/tools/), [`rag/`](src/rag/) owns the pipeline stages,
[`schemas/`](src/schemas/) own the wire models, and [`api/router.py`](src/api/router.py)
mounts every controller.

## Constraints worth knowing

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

## Design docs

[`docs/README.md`](docs/README.md) is the index and the reading order. The ones worth opening
first:

| Doc | What it settles |
| --- | --- |
| [`11-voice.md`](docs/11-voice.md) | the spoken loop — streaming, languages, barge-in, and every measured number |
| [`13-connectors.md`](docs/13-connectors.md) | what a user attaches, and why nothing is baked in |
| [`15-effort.md`](docs/15-effort.md) | the slider as five architectures, and the answer cache under all of them |
| [`16-memory.md`](docs/16-memory.md) | what carries between conversations, and what deliberately does not |
| [`22-no-local-corpus.md`](docs/22-no-local-corpus.md) | why "nothing connected" is an answer rather than an error |
| [`04-latency.md`](docs/04-latency.md) | the 200 ms budget, where it holds, and where it does not |
