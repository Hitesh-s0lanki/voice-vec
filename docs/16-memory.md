# 16 — Memory

*What the agent knows about you before you say anything.*

Two stores already hold a conversation, and neither of them is memory. `VoiceSession.history`
is the last few turns, alive for exactly as long as the socket. Postgres is every turn,
verbatim, so `/c/{id}` can be reloaded ([12-conversations.md](12-conversations.md)). Both are
*transcripts*, and a transcript is not a fact: reload a conversation and the model gets its
own words back; open a **new** one and it has never met you.

That gap is not closed by keeping more history. "The user is vegetarian" is not something you
find by replaying eight turns of Tamil into a context window — it has to be *extracted*, once,
and then found again by meaning rather than by recency.

[Redis Agent Memory](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/) is
the managed service that does that. Two tiers, and only one of them is ours to manage:

```
                       ours                       not ours
  a spoken turn ─► session memory ─► (background worker) ─► long-term memory
                   short TTL, ordered                        facts, embedded,
                   events                                    searchable forever
                                                                    │
  the next question ──────────── KNN, scoped to this owner ─────────┘
                                          │
                                          └─► one paragraph of the system prompt
```

We write turns *in* and read facts *out*. The interesting part happens in between, in a worker
we never call, on a cadence we only configure.

## What is where

| | Holds | Lives in | Source of truth for |
| --- | --- | --- | --- |
| `history` | last ~8 turns | the process | the reply being written now |
| `conversations` / `messages` | every turn, verbatim | Postgres | the rail, titles, `/c/{id}` |
| session memory | turns, mirrored | Redis Agent Memory | nothing — a staging area |
| long-term memory | extracted facts | Redis Agent Memory | what the agent *knows* |

**Postgres stays the source of truth.** Nothing here replaces [`src/chat/store.py`](../src/chat/store.py).
Session memory is a staging area for extraction, not a store: a session that expires loses
nothing a listener can see, because the turns are already in Postgres. What is new is the
bottom row — and it is the only row that survives a conversation ending.

## One identity, two stores

The agent-memory session id **is** the Postgres conversation id, and the actor id is the same
`user_id`-or-`sess_…` owner the conversations table keys on ([`whose`](../src/services/voice_service.py)).

```
conv_9f3a…  ─┬─ conversations.id       (Postgres)
             └─ sessionId              (agent memory)

user_2xY… / sess_1a2b…  ─┬─ conversations.user_id / session_id
                         └─ actorId, and so ownerId
```

That is not a convenience. It makes one id trace a turn across both stores, and — because
`_save` returns before `_mirror` when there is no conversation open — it means **a turn not
worth a row in Postgres is never worth an event in Redis either**. Coughs, callers with no
identity and takes below the length guard cost the instance nothing.

Signing in *widens* what is remembered rather than resetting it, for the same reason the
conversations predicate matches on either column: the account wins when there is one, so a
visitor's `sess_…` memories are carried forward rather than stranded.

## The 30 MB problem

Redis Cloud's Agent Memory service **attaches to a database rather than provisioning one**.
On the free tier that is the same 30 MB instance the answer cache is already budgeting
against ([15-effort.md](15-effort.md), [`src/rag/cache.py`](../src/rag/cache.py)) — and the
database's `volatile-lru` cannot tell the two apart. Both put TTLs on their keys, so an
unbounded cache does not simply fill up: **it evicts the agent's memories**, and the only
symptom is an agent that gradually stops remembering.

Five things follow, and none of them are incidental:

| Decision | Setting | Why this and not more |
| --- | --- | --- |
| The cache gives back a third | `CACHE_MAX_ENTRIES=900` | ~5.5 MB per scope at the measured 6.09 KB/entry, down from ~9 MB |
| Session TTL is hours, not days | console: **1–2 h** | the turns are already in Postgres; session memory only has to outlive the extraction cadence |
| Only persisted turns are mirrored | `_mirror`'s gate | nothing transient ever reaches the instance |
| Every event is trimmed first | `AGENT_MEMORY_MAX_CHARS=2000` | an untrimmed monologue costs its full length for the life of its session and adds nothing to extraction |
| Errored turns are not mirrored | `_EXTRACTABLE` | half a sentence from a provider failure is noise the instance would pay to store |

A barge-in **is** mirrored. What was heard before the interruption is as true as a whole
reply, and it is the same honest version Postgres gets.

## Recall, and why it is the dangerous half

Reading is where this can hurt. A recalled fact is asserted about the listener in the model's
**opening sentence** — the most damaging place to be wrong, and the least likely to be caught
by anything except a human ear. So recall is deliberately weaker than retrieval at every
point:

- **scoped to one owner** and nothing else. Not to the session: the useful memories are the
  ones from *other* conversations, so a session filter would return only what the prompt
  already contains.
- **floored at 0.62** cosine. An unfiltered nearest-neighbour search always returns
  *something*. Looser than the cache's 0.97 because a recalled fact informs an answer rather
  than replacing one — the same reasoning, a different consequence for being wrong.
- **capped at 3**. These arrive ahead of the retrieved context and the question itself, and a
  paragraph of half-relevant biography is a distraction the model pays attention to.
- **bounded at 600 ms**, and run *concurrently with retrieval* rather than before it, so this
  is what recall can add to a turn, not what it costs.
- **allowed to fail silently.** Unconfigured, unreachable, slow, malformed — all four produce
  an empty list. An agent that cannot remember still answers.

### The framing is most of the section

The facts are the smaller half of what reaches the prompt:

> What you already know about this person, from earlier conversations:
> - User is vegetarian
>
> Use this only when it changes the answer. Never recite it, never mention remembering, and
> never bring it up unprompted. If what they say now contradicts it, they are right and it is
> out of date.

Three sentences of instruction for one line of content, because the failure mode is specific
and expensive out loud. A model handed bare facts *recites* them — "Since you're
vegetarian…" — when nobody asked, which is exactly how an assistant that remembers stops
sounding like one that listens. And the facts are **stale by construction**: extracted from a
conversation that has since ended, so the live transcript has to be told, explicitly, that it
wins.

When there is nothing to recall the section is absent entirely rather than empty. A heading
with nothing under it reads to a model as *there is nothing to know*, which is a stronger and
less true claim than silence.

## Setting it up

The service is created in the console, not in code — three values come back and go in `.env`.

1. Redis Cloud console → **Agent Memory** → **Create custom service**.
2. Select the database `REDIS_URL` already points at, and its `default` user.
3. **Memory configuration** — the settings that decide whether 30 MB is enough:

   | Setting | Value | Why |
   | --- | --- | --- |
   | Short-term TTL | **2 hours** | not the 1-day default; session memory is a staging area here |
   | Long-term TTL | 365 days | the facts are the point, and they are small |
   | Extraction cadence | 300 s | the default; a conversation is over long before a fact is needed |
   | Automatic summarization | on, after **12**, keep **4** | bounds a long session's bytes without losing the recent turns |

4. **Memory types & extraction** — optional. A `voice_preference` type with fields for
   language, pace and how the listener likes to be addressed is the one worth defining here,
   because it is the domain-specific thing this product knows and generic extraction will not
   reach for.
5. Copy **Endpoint** and **Store ID** from the Configuration tab; the API key is shown
   **exactly once**, at creation.

```sh
AGENT_MEMORY_ENDPOINT=https://<service>.agent-memory.redis.io
AGENT_MEMORY_STORE_ID=<store id>
AGENT_MEMORY_API_KEY=<service key>
```

Any of the three unset and the agent keeps nothing between conversations. Everything else is
unchanged — the same contract `REDIS_URL` has, and `/voice/ws` reports which one you got in
`providers.memory` (`on` / `unset` / `off` / `unavailable (…)`).

## What is not wired

`/ask` — the stateless HTTP path in [`src/services/ask_service.py`](../src/services/ask_service.py) —
has no owner and no conversation, so it neither mirrors nor recalls. Giving it memory means
giving it an identity first, which is a decision about the API's shape rather than about
memory.

## Files

| Path | What it does |
| --- | --- |
| [`src/memory/store.py`](../src/memory/store.py) | the whole client: `configured`, `remember`, `recall`, `as_prompt` |
| [`src/services/voice_service.py`](../src/services/voice_service.py) | `whose`, `_mirror` on the write queue, `_recall` beside retrieval |
| [`src/voice/llm.py`](../src/voice/llm.py) | the prompt section and its instructions |
| [`src/core/config.py`](../src/core/config.py) | the seven `AGENT_MEMORY_*` settings and the cache's reduced budget |
| [`tests/test_memory.py`](../tests/test_memory.py) | the byte budget, the recall floor, and every way this is allowed to fail |
