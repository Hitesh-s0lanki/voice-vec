# 08 — Build plan

Ordered so that the highest-risk unknowns are resolved first and every phase ends with
something demonstrable.

## Phase 0 — De-risk the two things that can kill the design

Do this **before** writing pipeline code. Both are cheap; both invalidate large parts of
[02-architecture.md](02-architecture.md) if they come back wrong.

**0a — Measure query embedding latency.** Load `multilingual-e5-small` via
`@huggingface/transformers` on `onnxruntime-node`, embed 100 Hindi queries batch-of-1, report
P50 and P100. If it lands above ~80 ms, the 200 ms budget is in trouble and we change model
or quantise now rather than in Phase C.

**0b — Measure Indic token inflation.** Tokenise 200 Hindi passages and 200 English ones.
Get the real tokens-per-passage ratio. Every chunk size in
[03-chunking.md](03-chunking.md) depends on this number, and the 512-token truncation limit
is silent when exceeded.

Also: `docker run -p 6333:6333 qdrant/qdrant`, confirm loopback round-trip time.

**Exit criteria:** three measured numbers written into [04-latency.md](04-latency.md),
replacing the estimates.

## Phase A — Thin end-to-end slice, 2,000 rows

Get a spoken question to a grounded answer. Quality is irrelevant here; the loop existing is
the point.

- `scripts/ingest.ts` — DuckDB over `hinval.parquet`, `LIMIT 2000`, **S1 only**, dedup,
  embed, upsert to Qdrant
- `frontend/src/instrumentation.ts` — `register()` warms embedder + Qdrant client before traffic
- `frontend/src/lib/rag/{embed,store,extract}.ts`
- `frontend/src/app/api/ask/route.ts` — no harness yet, straight-line, but emit `timings` from the
  start
- Wire [`Turn.reply`](../frontend/src/lib/conversation.tsx) so the answer renders in the UI

**Exit criteria:** speak a Hindi question, get a grounded answer with a citation, see
per-stage timings in the response.

## Phase B — The two requirements that carry the most weight

Now that the loop exists, build the parts being graded. These are independent and can go in
either order.

**B1 — Chunking (requirement 2).** S2–S5, Qdrant named vectors, sparse vectors, RRF fusion,
intent routing, post-fusion result dedup. Re-ingest at 25,000 rows.

**B2 — Guardrails (requirement 6).** Four gates, `status: answered | abstained | refused`,
abstention rendered as a real turn rather than an error.

**B3 — Harness (requirement 5).** Refactor the straight-line route into typed stages with
the deadline-aware runner, retries, circuit breaker, degradation matrix.

**Exit criteria:** all five strategies queryable; the system abstains on a labelled
unanswerable query; killing Qdrant produces a clean abstention rather than a 500.

## Phase C — Measure everything

The phase that produces the submission.

- `scripts/evaluate.ts` — E1–E4, seeded, writing `reports/results.json` plus markdown tables
- **Build the provenance scorer first and unit-test it** against hand-checked rows
  ([07-evaluation.md](07-evaluation.md)) — everything downstream is wrong if it is wrong
- Sweep `FLOOR` / `MARGIN`, pick and justify the operating point
- Full ingest of `hinval` (~98k rows) once Phase B numbers look sane
- Fill in the comparison matrix and both latency tables

**Exit criteria:** every table in these docs has real numbers, and every number is
reproducible from `results.json`.

## Phase D — Stretch, only if C is fully done

Ranked by submission value per hour:

1. **`/api/metrics` + live latency panel.** Live P50/P70/P100 during the demo beats a static
   table. Cheap — the ring buffer already exists.
2. **Tier 3 LLM synthesis** with tool calls and Gate 4. Completes the harness story.
3. **Second language** (`tamval` or `benval`) → the cross-lingual `query_id` demo.
4. Cross-encoder rerank for Tier 2, only if it measures under ~30 ms.

## Sequencing rules

**Do not scale the corpus before the pipeline is correct.** A 980k-chunk re-ingest after
finding a chunking bug costs hours. Phase A at 2k, Phase B at 25k, full only in Phase C.

**Emit `timings` from the very first route handler.** Retrofitting instrumentation is how
latency work gets deferred until there is no time to fix what it finds.

**Share `chunk.ts` between ingest and runtime.** Divergence there produces silently wrong
retrieval that no test catches.

**Keep an ingest checkpoint file.** Ingest will crash. Restarting from zero is avoidable.

## Dependencies to add

| Package | For |
| --- | --- |
| `@huggingface/transformers` + `onnxruntime-node` | local embedding |
| `@qdrant/js-client-rest` | vector DB |
| `zod` | structured I/O validation |
| `duckdb` / `@duckdb/node-api` (or a Python ingest script) | reading parquet |
| `tsx` | running `scripts/*.ts` |

Node-only, all of them — `onnxruntime-node` cannot run on the edge runtime, which is
deprecated in Next 16 anyway.

## Two Next 16 gotchas

- **`middleware.ts` is deprecated, renamed `proxy.ts`.** Same functionality, different file
  and export name. Codemod: `npx @next/codemod@canary middleware-to-proxy .`
- **`instrumentation.ts#register()` runs once per server instance and must complete before
  the server accepts requests.** This is the correct and only good place to warm the ONNX
  session. Skip it and P100 measures cold-start, not the pipeline.

## Decisions still open

From [02-architecture.md](02-architecture.md), needed before Phase B:

1. Ship Tier 3 at all? *Recommendation: yes, default to Tier 1, report both.*
2. One language or two? *Recommendation: Hindi only until Phase C is green.*
3. Cross-encoder rerank? *Recommendation: defer to Phase D, gate on a measurement.*

## What to submit

- These docs, with the estimate tables replaced by measured ones
- `reports/results.json` and the generated markdown tables
- A README section naming the scoping decisions — validation split, Hindi, N rows — and the
  limitations list from [07-evaluation.md](07-evaluation.md)
- The six-query demo script from [06-guardrails.md](06-guardrails.md)

Scoping choices stated up front read as engineering judgment. The same choices discovered by
a judge read as gaps.
