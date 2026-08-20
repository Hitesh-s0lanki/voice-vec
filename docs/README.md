# Vec — voice-enabled RAG

Design docs for the HH Goa 2026 Task 2 submission.

> **What runs today is the voice loop, not the retrieval one.**
> [11-voice.md](11-voice.md) — speak in any of 22 languages, be answered out loud in the
> same one, with every stage streaming into the next. Retrieval is built, measured
> ([09-v1.md](09-v1.md)) and switched off behind `RAG_ENABLED`; documents 01–10 describe it
> and remain accurate for the day it comes back on.

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
- UI shell, orb, panels, persisted turns — [`frontend/src/components/voice-app.tsx`](../frontend/src/components/voice-app.tsx)
- S1 index over 2,000 `hinval` rows, local ONNX embedding, Qdrant, extractive answers,
  Gates 1–3, per-stage timings, live percentiles at `GET /metrics` — **currently switched
  off**, see the note at the top

Not built yet: S2–S5 and hybrid retrieval (requirement 2's heavy half), Tier 2 rerank,
Tier 3 LLM synthesis with Gate 4, and the full-corpus ingest.

## A note on numbers in these docs

Every latency figure in 01–08 is a **budget or an estimate**, labelled as such. The measured
ones live in [09-v1.md](09-v1.md) and in `reports/results.json`. The dataset statistics in
[01-dataset.md](01-dataset.md) *are* measured — they come from reading the actual parquet
files — and say so explicitly. Do not copy a budget number into the submission as if it
were a result.
