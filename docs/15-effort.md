# 15 — The effort ladder

## What the slider actually is

[`EffortPanel`](../frontend/src/components/panels/effort-panel.tsx) has been in the app since
before there was a pipeline behind it, and until now it moved a number that nothing read. It
is now the control that chooses a **retrieval architecture** — not an amount of the same one.

| Rung | Label | What runs | Model calls | Budget |
| --- | --- | --- | --- | --- |
| 0 | Lookup | Gate 1 → embed → search → Gate 2. The passages, as they are. | 0 | ~60 ms |
| 1 | Grounded | + answer cache, extractive span, Gate 3 | 0 | **< 200 ms** |
| 2 | Deep | + hybrid retrieval, rerank, synthesis, Gate 4 | 1 | ~1 s |
| 3 | Corrective | + relevance grading, query rewrite, one re-retrieval | 3–6 | ~5 s |
| 4 | Adaptive | + routing before retrieval, capped repair loop | 4–8 | ~10 s |

The taxonomy is the one in [agentic-rag/05-rag-architectures.md](agentic-rag/05-rag-architectures.md);
the mapping onto this system's latency budget is the one sketched in
[agentic-rag/07-findings.md](agentic-rag/07-findings.md#what-to-port-into-vec).

## Three rules the whole design rests on

### 1. The level is a ceiling, not a floor

Asking for Adaptive does not mean four model calls happen. It means up to four **may**. A
question the answer cache already holds is answered from Redis at rung 4 exactly as it would
be at rung 1, and a rung whose stages are unavailable falls through to the rung below rather
than failing.

So every response carries two numbers:

```
mode   the rung that was asked for
tier   the rung that actually produced this answer
```

A `mode: deep` answer with `tier: 1` is a synthesis that fell back to the extractive path —
usually because no model key is configured. A run of them is a configuration problem
reporting itself, and it would be invisible behind a single number. `escalations` carries the
rest of the audit trail: `hybrid`, `rerank`, `graded`, `rewrite`, `regenerate`, `dense-only`,
`fallback-extractive`, `cache-semantic`.

### 2. Rungs 0 and 1 make no network call after the transcript arrives

That is the entire basis of requirement 3's 200 ms, and it is why the ladder is split exactly
where it is. Everything from rung 2 up is at least one round trip and cannot meet that
number — so it is not measured against it. Each rung carries its own budget
(`EFFORT_DEADLINE_MS`), and `withinBudget` compares against `budgetMs`, which is reported on
every answer.

Trading latency for quality on an explicit request is a defensible engineering decision.
Quietly reporting only the fast tier's latency is not.

### 3. Every loop has a counter, and they share it

`MAX_REPAIRS` bounds repairs across the **whole request**, not per branch. A query rewrite at
Gate 2 and a regeneration at Gate 4 spend from the same budget.

Per-branch counters are how a self-correction loop ends up unbounded while every individual
branch still looks capped — which is precisely the defect in the `adaptive-rag` notebook this
rung is modelled on, where `not supported` is wired `generate → generate` with no cap and
only fails to spin because an undefined name crashes it first
([agentic-rag/07-findings.md](agentic-rag/07-findings.md#adaptive-loop)).

## Rung by rung

### 0 — Lookup

Search, and show what came back. Gate 2 has already decided that something in the index is
close enough to be worth showing, so the top passage *is* the answer and the rest are the
citations.

Gate 3 is skipped rather than run: it checks that an answer is a substring of its source, and
here the answer is the source. There is nothing to hallucinate because nothing was written.

When the corpus has nothing, this is where "Nothing in your sources matches that" comes from —
and it is the same floor-and-margin test as every other rung, swept against the ~39% of
MSMARCO-XI rows labelled unanswerable ([06-guardrails.md](06-guardrails.md)).

### 1 — Grounded

The tier the 200 ms claim refers to, unchanged from v1, plus the cache in front of it.

**The answer cache (CAG).** A repeat question is answered from Redis at embedding cost alone.
Two layouts, because Redis is two different products:

```
semantic   KNN over past query vectors. Needs the query engine
           (Redis 8, Redis Stack, Redis Cloud).
exact      SHA-256 of the normalised query. The fallback, on a plain Redis.
```

**Exactly one is written, never both.** The semantic layout subsumes the exact one — the same
text embeds to the same vector and scores a cosine of 1.0 against itself — so writing both
doubles the storage to save a few milliseconds on a repeat the KNN was going to find anyway.
On a small instance, storage is the binding constraint.

The semantic layout is the one that earns its keep: *"what is a corporation"* and *"define a
corporation"* are different strings and neighbouring vectors. It is not available everywhere,
so its absence is detected once at connect (the `FT.CREATE` attempt **is** the probe; there is
no capability flag to read and `MODULE LIST` is blocked on several managed offerings) and the
layout falls back to exact-only with a line in the log. `providers.cache` on the socket
handshake reports which one you got, because a deployment that believes it has semantic caching
and does not will read its hit rate as a tuning problem.

Two things that only show up against a **remote** Redis, both found by running one:

- **Connecting and querying are different budgets.** A cache lookup sits on the answer path, so
  the per-operation ceiling is tight (`CACHE_TIMEOUT_S`, 150 ms — a warm round trip to a managed
  instance one region away measured ~6 ms). Opening the connection measured ~93 ms, and a single
  timeout covering both means the cache quietly never connects at all.
- **The reply shape depends on protocol negotiation, not on anything we ask for.** redis-py 8
  speaks RESP3 to a server that supports it, and RESP3 returns `FT.SEARCH` results as a map
  where RESP2 returns a flat array. Parsing only one shape is a bug that no hand-written test
  fake will ever catch, because the fake returns whichever shape its author had in mind. Both
  are parsed, and both are tested.

Four rules that make it safe rather than merely fast:

- **Only successes are cached.** An abstention is a statement about the corpus at one moment;
  cache it and a re-ingest that fills the gap stays invisible for a day.
- **The threshold is cosine, and it is high.** `CACHE_SIMILARITY=0.97` over L2-normalised e5
  vectors — the same scale as `RETRIEVAL_FLOOR`. The figure that circulates in write-ups
  (~0.45) is a raw squared-L2 distance over un-normalised MiniLM vectors and does not transfer
  between embedding models at all. A loose threshold answers a question nobody asked, and the
  answer still reads perfectly well: a correctness bug that presents as a performance win.

  Measured against the real index: *"what is a corporation"* scores 1.0 against itself,
  *"define a corporation"* scores **0.985** and is served, *"explain what a corporation is"*
  falls below 0.97 and is **not**. Unrelated questions — sourdough, the capital of Peru — miss
  by a wide margin. That is the threshold doing what it was set to do, erring towards a second
  retrieval rather than towards a wrong answer.
- **The scope separates every axis that can change the right answer** — user, backend, rung,
  language, and whether it was answered from the parallel English. Each of those is a
  cache-poisoning bug if dropped.
- **It never raises.** No Redis, wrong password, timeout, malformed entry: all misses. A cache
  that can take the answer path down with it is worse than no cache.

**Sizing, on a small instance.** One entry with a realistic Devanagari payload measured
**6.09 KB** by `MEMORY USAGE` — a 1.5 KB vector plus ~3.7 KB of JSON — so a 30 MB instance
holds roughly five thousand answers before index overhead. `CACHE_MAX_ENTRIES` bounds one
scope; `CACHE_MAX_ENTRY_BYTES` stops one pathological passage taking a measurable share of the
whole database. The real backstop is Redis itself: every key written carries a TTL, so an
instance with `volatile-lru` evicts the least recently used answer when it fills rather than
refusing the write — running out of room degrades the hit rate, not the pipeline. Everything
lives under `CACHE_PREFIX`, keys and index alike, because a Redis instance is a place other
things also live.

This is *semantic response caching*, and it is labelled as such. Real CAG — preload the corpus
into the context window, reuse the precomputed KV cache, never retrieve — is a decoder-level
technique at a different layer, and conflating the two is a documented defect in the source
this idea came from.

### 2 — Deep

**Hybrid.** The `tsv` column and its GIN index have been built at ingest since the pgvector
migration and deliberately left unqueried, so that any movement in recall@5 could be
attributed to the store move alone ([03-chunking.md](03-chunking.md)). This rung is what turns
them on.

Fusion is by **reciprocal rank**, never by score. Cosine and `ts_rank_cd` are different
quantities on different scales, and averaging them is the standard way to build a hybrid
retriever that is worse than either half.

One consequence that looks like a bug and is not: the fused `Hit` keeps its **dense** score,
and Gate 2 reads a separately-maintained cosine-ordered list rather than the fused one. The
floor and margin were swept on cosine over a cosine-ranked list; handing the gate an RRF score
(~0.03) would abstain on everything, and handing it the fused *order* makes the margin test
compare the fusion winner against a higher-scoring neighbour and go negative.

**Rerank.** Embedding rescore plus MMR, over the fused candidates. Two different problems:
after fusion the list contains passages the lexical channel found and the dense channel never
scored, and embedding them all gives one comparable number; and the five chunking strategies
overlap by construction, so the top five are frequently five renderings of one passage.

**No cross-encoder, and that is a measurement rather than an oversight.** The rerankers small
enough for this budget — the `ms-marco-MiniLM` family — are English-only, and the index is
Devanagari. The multilingual alternatives are several hundred million parameters and do not
fit the latency this rung claims. The bi-encoder already loaded does the job, at a quality cost
stated openly rather than a latency cost discovered in production.

**Synthesis, and Gate 4.** One structured call over the retrieved passages, instructed to
answer strictly from them and to say `NO_ANSWER` when they do not cover the question — which
becomes a real abstention rather than an error.

Gate 3 cannot check generated text: an extracted span is verified by construction, and
generated prose can be perfectly faithful without sharing a character sequence with its source.
So Gate 4 asks the weaker question it can actually answer — every **sentence** of the answer is
embedded and scored against the sentences of the context, and enough of them must clear
`GENERATION_SUPPORT_FLOOR`. Per sentence, because the failure this exists to catch is local: a
model handed four good passages writes three faithful sentences and one fluent invention, and
a similarity score over the whole paragraph averages that invention away.

Local, so it costs milliseconds rather than a round trip. And if the gate itself fails, it
passes rather than abstains — refusing because the *check* broke trades a possible
hallucination for a certain abstention on every request while the embedder is unwell.

### 3 — Corrective

An LLM grader over the retrieval, and a query rewrite when it fails.

The trigger is **aggregate**, not per-document: `correct` / `ambiguous` / `incorrect` over the
whole retrieval, with the repair reserved for the last and for a retrieval too thin to work
with (`GRADER_RELEVANT_MIN`). Grading one document at a time and firing whenever any single one
fails runs the expensive path on nearly every query — a top-10 almost always contains a weak
result, and that is what a top-10 is for.

Query rewriting appears **only** here, as a repair. On the happy path it would be a round trip
in front of every question, which is what puts query enhancement outside this system's budget
entirely ([agentic-rag/04-query-enhancement.md](agentic-rag/04-query-enhancement.md)). After a
retrieval that has already been graded bad, it is paid for by a query that was going to be
abstained on anyway.

Both rounds' results are kept and fused. A rewrite is a second opinion about the search key,
not a verdict that the first retrieval was worthless, and a rewrite that drifts off-topic must
not turn a passable first attempt into an abstention.

### 4 — Adaptive

**Routing before retrieval.** The one stage that can save the entire pipeline rather than
improve it: "hello", "say that again", "what did I just ask you" are questions no corpus search
can help with, and retrieving for them spends the budget to produce an abstention that was
knowable up front. A `direct` route sets the `direct` flag, which the voice loop reads as "answer
this conversationally" rather than reading out an abstention about sources.

The router gets a real description of the corpus, built from the ingest manifest. A vague one is
a routing bug: told only "a passage index", the router decides perfectly ordinary factual
questions are things a search could not help with. And a router that cannot answer routes to the
vector store — retrieving unnecessarily costs a little time, skipping retrieval wrongly costs the
answer.

**Three outcomes, two repairs.** The generation verdict separates two failures because they need
different fixes:

```
not supported  context was fine, the writing was not  → regenerate, same context
not useful     writing was fine, the context was not  → rewrite the query, retrieve again
useful         done
```

Diagnosing them as one "bad answer" signal means half the repairs attack the wrong problem. The
usefulness half is only *asked for* at this rung, because rungs 2 and 3 have no repair for
"grounded but off-target" and the question would cost a round trip to learn something nothing
downstream could act on.

## Connected stores

A user who has connected Pinecone, Astra or their own Postgres is searched against that
([13-connectors.md](13-connectors.md)). The ladder has to work there too, and the three
backends are not equally capable — so `VectorBackend.capabilities()` is what the pipeline asks,
rather than switching on a slug:

| | lexical channel | filters | parallel English |
| --- | --- | --- | --- |
| pgvector (deployment's, or connected) | ✅ `tsv` + GIN | ✅ | ✅ |
| Pinecone | ❌ | ✅ metadata | ❌ |
| Astra | ❌ | ✅ | ❌ |

Pinecone does have sparse vectors and Astra does have a `$lexical` field — but only in an index
or collection created for it, and this app creates neither. Claiming the channel would mean rung
2 asking for one that returns nothing on most connected indexes.

So rung 2 against a hosted index runs **dense-only** and puts `dense-only` in `escalations`. The
rung still runs; it just runs with one channel, and says so. Same for a connected Postgres built
by an older migration with no `tsv` column: the lexical query raises, the failure is traced, and
the result is a dense-only retrieval rather than no retrieval.

## Wiring

The value is a property of the **turn**, not the connection — moving the slider has to take
effect on the next question, and a socket query parameter would need a reconnect to change. So
it rides on the client events that start a question:

```
{ "type": "audio.start", "mime": "audio/webm", "effort": 2 }
{ "type": "text", "text": "...", "effort": 2 }
```

Read at `audio.start` and applied at `audio.end`, because that is when the take becomes a
question. `Settings.effort_level` clamps whatever arrives — the value comes off an open
WebSocket — and `EFFORT_MAX` caps what a deployment will run regardless of what the slider says.

## Configuration

```
EFFORT_MAX=4                              # what this deployment will run
EFFORT_DEFAULT=1
EFFORT_DEADLINE_MS=[200,200,2500,9000,16000]

REDIS_URL=                                # unset → no cache, full pipeline every time
CACHE_TTL_S=86400
CACHE_SIMILARITY=0.97                     # cosine; sweep before lowering
CACHE_TIMEOUT_S=0.15

GENERATION_SUPPORT_FLOOR=0.62             # Gate 4
MAX_REPAIRS=1                             # spans the whole request
```

## What has not been measured

Stated plainly, because the rest of these docs report numbers and this one does not yet:

- **Every latency figure in the table above is a design target**, not a measurement. The rungs
  have been exercised end to end against a real provider and against fakes; they have not been
  swept over the labelled evaluation set.
- **`CACHE_SIMILARITY=0.97` is a conservative guess**, chosen to fail closed. It needs the same
  treatment `RETRIEVAL_FLOOR` got in [09-v1.md](09-v1.md) — a sweep against the labelled set,
  reporting false-hit rate against hit rate.
- **`GENERATION_SUPPORT_FLOOR=0.62` likewise.** It should be swept against gold answers, where a
  hallucination rate is measurable rather than asserted.
- **No rung above 1 has a recall or coverage number yet.** The rule that should govern this:
  *a rung that does not beat the rung below on quality does not ship.* `scripts/evaluate.py`
  aggregates by `mode` now, which is what makes that rule checkable rather than aspirational.
