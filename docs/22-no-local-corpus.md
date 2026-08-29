# 22 — No local corpus

This app used to hold an index of its own: a `chunks` table behind `DATABASE_URL`, built by
`scripts/ingest.py` from 2,000 rows of MSMARCO-XI, and searched by everybody who had not
connected anything. It is gone. **What a question can be answered from is exactly what its
asker attached** — a Pinecone index, an Astra collection, or their own Postgres
([13-connectors.md](13-connectors.md)) — and a user with nothing attached is told so.

## Why a fallback corpus was the wrong shape

Connectors shipped as an *override*: search the user's store if they have one, otherwise the
deployment's. That reads as generous and is three problems.

**It answered the wrong question with confidence.** The fallback held 19,870 Hindi passages
of 2016 web text. Someone who connected their company's documentation and hit a credential
error got a fluent, cited, entirely irrelevant answer drawn from MS MARCO — and nothing
anywhere reported a failure, because falling back *was* the designed behaviour. The
degradation ladder had no rung for "I could not reach your store"; it had a rung that
silently changed the subject.

**It made every deployment-wide number a lie.** `/health` reported a chunk count. `/metrics`
reported the ingest manifest beside the latency percentiles. Both described one index while
requests in the same buffer were being served from several different people's stores. The
honest replacement for a chunk count is per-user, so it moved to `/connectors`, and
`/metrics` now aggregates `by_backend` instead.

**It kept an ingest nobody could use.** The pipeline wrote to the app's own Postgres, so it
built exactly one corpus for everyone. Per-user ingest was never going to be built on top of
it — the point of connectors is that people bring stores they already have.

## What "nothing connected" does now

`BackendResolver.for_user` returns `None` rather than a store, for all four of: signed out,
nothing attached, credentials unreadable, and connected-but-not-searchable. It still never
raises. `AskService._resolve_backend` turns `None` into an abstention:

```
I don't have a source to search yet — connect a vector store and I can answer from it.
```

An abstention and not a 500, because it is a true and actionable statement — the connectors
panel is one tap away — where an error reads as the app being broken. It is tagged
`no-backend` in `escalations`, so `/metrics` can tell *"nobody connected anything"* apart
from *"the corpus did not cover it"*; those need different fixes.

A store that builds but cannot answer takes the same path. `verify` runs once, when the form
is submitted, and an index can be emptied, dropped or renamed long afterwards — so `ready()`
probes a freshly built backend, and one that says no is cached as a miss and reads as
nothing attached.

## What came out with it

| Gone | Why |
| --- | --- |
| `scripts/ingest.py`, `evaluate.py`, `crosslingual.py`, `demo.py` | they built and measured the one corpus |
| `scripts/suggestions.py`, `GET /suggestions`, the whole UI path | a deployment-wide list of openers has no single subject when every user has a different store; nothing in the frontend consumed it |
| `src/rag/manifest.py`, `data/index-manifest.json` | described what the last ingest built |
| The ingest half of `src/rag/chunk.py` | `Chunk`, `Row`, `Origin`, `passage_atomic`, `merge_duplicates`, `content_key` |
| `VectorStore.ensure_schema` / `create_indexes` / `upsert` / `backfill_english` / `english_backlog` / `warm` / `count`, and the DDL behind them | this app writes to no store it reads |
| `get_store()`, `BackendResolver.default()` | there is no store to be the default |
| `PG_TABLE` | the table comes from the connector, resolved by `verify_pgvector` |
| The index half of the boot keepalive | what gets searched is a pool against somebody else's database; holding *that* warm on a timer is not this process's business |

`DATABASE_URL` stays, and is never searched. It holds what this app owns: conversations,
tool calls, connected accounts, their profiles, and datasets. `scripts/migrate.py` creates
those and checks the round trip; it has no `--recreate`, `--indexes` or `--english` any more
because there is no table of ours to recreate or index.

`VectorStore` stays too, stripped to its query half. Its one caller is `PgVectorBackend`,
which builds one over the user's DSN with a private two-connection pool and the `ColumnMap`
discovered when the connector was verified. Keeping it is what stops the search SQL, the
strategy filter and the language filter existing twice.

## Two things the manifest was quietly deciding

Removing it changed behaviour in two places that were not obviously about ingest.

**Gate 1's language check.** `gate_input` took the indexed languages and flagged a question
as cross-lingual when its language was not among them. That list came off the manifest. A
connected store does not declare what languages it holds, so the app now passes an empty
list and the branch fires only on a language code that does not resolve to a FLORES tag at
all. The question it was standing in for — *was the retrieval actually good enough* — is
settled at Gate 2 on a measured score, which is where
[13a-cross-lingual.md](13a-cross-lingual.md) already argued it belonged.

**The language filter on search.** `_retrieve` filtered by language whenever the manifest
showed more than one. It now never does. Guessing at the language tags in somebody else's
metadata is not a neutral cost: pgvector discards non-matching rows *while* walking the HNSW
graph, so a predicate on a field the store does not carry the way we assume returns fewer
candidates rather than none — which shows up as poor recall and not as an error.

## The router still needs to know what it is routing to

Rung 4 asks a model whether a question is worth retrieving for at all, and it needs a
description of the corpus to answer. A vague one is a routing bug: told only *"a vector
index"*, the router decides an ordinary factual question is something search could not help
with and skips retrieval entirely. That description used to be built from the manifest.

It now comes from the profile — the store was already sampled at connect time and a
paragraph written about what is in it ([17-understanding.md](17-understanding.md)), which is
strictly better because it is about the store actually being searched. It falls back to the
backend's own one-line `describe()`, and never blocks: a store connected seconds ago has no
profile yet, and the right answer for that request is a weaker hint rather than a stall
while somebody's index is sampled.

## What is not re-measured

`RETRIEVAL_FLOOR`, `RETRIEVAL_MARGIN` and their cross-lingual pair were swept against the
labelled abstention set of the corpus that is now gone ([09-v1.md](09-v1.md)). They are
still the defaults and they are still the right *kind* of number — a cosine floor and a
margin on an L2-normalised e5 space — but they are not a property of whatever you connect.
If recall looks thin on a connected store, they are the first dials to re-check.
