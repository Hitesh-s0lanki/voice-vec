# Vec — voice-enabled RAG

Design docs for the HH Goa 2026 Task 2 submission.

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
| 1 | Speech-to-text (Sarvam or ElevenLabs) | [02-architecture.md](02-architecture.md) — already built, Sarvam `saaras:v3` |
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

## Status

Built and working:

- Mic capture with live analyser — [`src/hooks/use-voice-capture.ts`](../src/hooks/use-voice-capture.ts)
- Sarvam STT proxy — [`src/app/api/transcribe/route.ts`](../src/app/api/transcribe/route.ts)
- UI shell, orb, panels, persisted turns — [`src/components/voice-app.tsx`](../src/components/voice-app.tsx)

Not built yet — everything downstream of the transcript. [`Turn.reply`](../src/lib/conversation.tsx)
is the empty slot the RAG answer lands in, and the four levels in
[`EffortPanel`](../src/components/panels/effort-panel.tsx) are already the right shape for
the escalation ladder in [02-architecture.md](02-architecture.md).

## A note on numbers in these docs

Every latency figure here is a **budget or an estimate**, labelled as such. Nothing in this
directory has been measured yet. The dataset statistics in
[01-dataset.md](01-dataset.md) *are* measured — they come from reading the actual parquet
files — and say so explicitly. Do not copy a budget number into the submission as if it
were a result.
