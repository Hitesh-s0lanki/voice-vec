# Vec — voice-enabled RAG

Design docs for the HH Goa 2026 Task 2 submission.

> **Both halves run now.** [11-voice.md](11-voice.md) is the spoken loop — speak in any of
> 22 languages, be answered out loud in the same one, every stage streaming into the next.
> Retrieval is on behind it over 19,870 Hindi passages, and answers questions asked in any
> language ([13-cross-lingual.md](13-cross-lingual.md)).
>
> Two caveats documents 01–10 do not yet reflect: the index moved from embedded Qdrant to
> Postgres + pgvector, and with it a round trip away the **200 ms budget in
> [04-latency.md](04-latency.md) no longer holds** — search alone costs ~400 ms. That
> budget was measured in-process and remains the target, not a claim about today.

## The 60-second version

A user speaks a question in an Indic language. Sarvam transcribes it. We embed the
transcript **locally**, search a Qdrant index built from MSMARCO-XI passages under five
different chunking strategies, and answer from the retrieved context — or refuse to.

The whole query-time path targets **under 200 ms**, which is only achievable because the
default answer path contains **zero network calls after the transcript arrives**. Every
architectural decision in these docs falls out of that one constraint.

The thing that makes this submission defensible is not the pipeline — everyone will build
roughly the same pipeline. It is that MSMARCO-XI ships **labelled ground truth** for both
retrieval (`is_selected`) and abstention (`"No Answer Present."`), so every claim we make is
a measured number on thousands of queries rather than a demo that worked once.

## Requirement → document map

| # | Requirement | Where it is answered |
| --- | --- | --- |
| 1 | Speech-to-text (Sarvam or ElevenLabs) | [02-architecture.md](02-architecture.md), [11-voice.md](11-voice.md) — built, Sarvam `saaras:v3` |
| 2 | Chunking strategy, "vast" | [03-chunking.md](03-chunking.md) — five strategies, indexed side by side |
| 3 | Under 200 ms | [04-latency.md](04-latency.md) — budget, boundary definition, what we cut to make it |
| 4 | P50 / P70 / P100 analytics | [04-latency.md](04-latency.md) + [07-evaluation.md](07-evaluation.md) |
| 5 | Harness the model | [05-harness.md](05-harness.md) — typed stages, retries, escalation, structured I/O |
| 6 | Guardrail the model | [06-guardrails.md](06-guardrails.md) — four gates, with a labelled abstention set |

## Reading order

1. **[01-dataset.md](01-dataset.md)** — what MSMARCO-XI actually contains, and the scoping decision
2. **[02-architecture.md](02-architecture.md)** — end-to-end pipeline and the escalation ladder
3. **[03-chunking.md](03-chunking.md)** — requirement 2, the heaviest lift
4. **[04-latency.md](04-latency.md)** — the 200 ms budget and how we measure it
5. **[05-harness.md](05-harness.md)** — orchestration
6. **[06-guardrails.md](06-guardrails.md)** — knowing when not to answer
7. **[07-evaluation.md](07-evaluation.md)** — the numbers we will report
8. **[08-build-plan.md](08-build-plan.md)** — phased implementation
9. **[09-v1.md](09-v1.md)** — what actually shipped, and what it measured
10. **[10-request-flow.md](10-request-flow.md)** — the wired path, mic to rendered answer
11. **[11-voice.md](11-voice.md)** — the spoken loop that ships today: streaming, languages, barge-in
12. **[12-conversations.md](12-conversations.md)** — where a turn goes after it is heard: the two tables, who owns one, and `/c/{id}`
13. **[13-cross-lingual.md](13-cross-lingual.md)** — asking in a language the index does not hold: what it costs, measured, and the two thresholds it needs
13. **[13-connectors.md](13-connectors.md)** — the services a user attaches with their own credentials: Composio for tools, Pinecone/Astra/pgvector for where their questions get answered
14. **[14-glass.md](14-glass.md)** — the interface's one surface vocabulary: the ambient room glass needs to be visible in, the nine surfaces, and the two hover directions
15. **[15-effort.md](15-effort.md)** — the effort slider as five retrieval architectures: what each rung runs, the Redis answer cache, and why the level is a ceiling rather than a floor
16. **[16-memory.md](16-memory.md)** — what the agent knows before you speak: Redis Agent Memory, why Postgres stays the source of truth, and how two features share one 30 MB instance

## Background research

**[agentic-rag/](agentic-rag/)** — a technique analysis of the
[`Hitesh-s0lanki/agentic-rag`](https://github.com/Hitesh-s0lanki/agentic-rag) repo: every
chunking, retrieval, and RAG architecture it implements, plus the ones it doesn't. Not part
of the submission. It is the menu the escalation ladder in
[02-architecture.md](02-architecture.md) picks from —
[agentic-rag/07-findings.md](agentic-rag/07-findings.md#what-to-port-into-vec) maps specific
techniques onto specific effort levels and separates the index-time ones (free) from the
query-time ones (billed against the 200 ms).

## Status

**v1 is built and measured — [09-v1.md](09-v1.md).** Phase A of the build plan, in Python
rather than the TypeScript these docs assume: the FastAPI service in [`src/`](../src/) owns
the pipeline, and the Next app proxies to it. That is the one design decision in this
directory that the implementation overrides, and 09-v1.md says why.

Built and working:

- The spoken loop — mic, Saaras, a streamed reply, Bulbul, and barge-in
  ([11-voice.md](11-voice.md)): [`src/voice/`](../src/voice/),
  [`src/services/voice_service.py`](../src/services/voice_service.py),
  [`frontend/src/hooks/use-voice-session.ts`](../frontend/src/hooks/use-voice-session.ts)
- UI shell, orb, panels — [`frontend/src/components/voice-app.tsx`](../frontend/src/components/voice-app.tsx)
- Conversations saved to Postgres, each at its own `/c/{id}`, reloadable and resumable
  ([12-conversations.md](12-conversations.md)): [`src/chat/store.py`](../src/chat/store.py),
  [`src/controllers/conversations_controller.py`](../src/controllers/conversations_controller.py)
- Connectors attached per signed-in user with their own credentials, encrypted at rest —
  Composio for tools, Pinecone/Astra/pgvector for retrieval
  ([13-connectors.md](13-connectors.md)): [`src/connectors/`](../src/connectors/),
  [`src/rag/backends/`](../src/rag/backends/)
- S1 index over 2,000 `hinval` rows, local ONNX embedding, Postgres + pgvector, extractive
  answers, Gates 1–3, per-stage timings, live percentiles at `GET /metrics`
- Cross-lingual retrieval — a question in any language against the Hindi index, answered
  from the English original each chunk carries, with its own measured thresholds
  ([13-cross-lingual.md](13-cross-lingual.md))

Not built yet: S2–S5 and hybrid retrieval (requirement 2's heavy half), Tier 2 rerank,
Tier 3 LLM synthesis with Gate 4, and the full-corpus ingest.

## A note on numbers in these docs

Every latency figure in 01–08 is a **budget or an estimate**, labelled as such. The measured
ones live in [09-v1.md](09-v1.md) and in `reports/results.json`. The dataset statistics in
[01-dataset.md](01-dataset.md) *are* measured — they come from reading the actual parquet
files — and say so explicitly. Do not copy a budget number into the submission as if it
were a result.
