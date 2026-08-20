# 05 — Harness

> Requirement 5: *"Your model/pipeline should be run inside a proper harness — structured
> orchestration around the model (tool calls, retries, structured input/output handling,
> error recovery) rather than a single raw prompt-in, text-out call."*

## The shape

Everything between transcript and answer is a **typed stage**. A stage declares its input
type, output type, timeout, retry policy and fallback. The runner executes stages in
sequence, times each one, catches each one, and decides whether to retry, fall back,
escalate, or abstain.

Timing and error handling are therefore **structural**, not something a stage author
remembers to add. That is what makes the `timings` block in
[04-latency.md](04-latency.md) trustworthy: it cannot drift out of sync with the code,
because the runner produces it.

```ts
// src/lib/rag/harness.ts

type Stage<In, Out> = {
  name: keyof AskResponse["timings"];
  run: (input: In, ctx: Ctx) => Promise<Out>;
  timeoutMs: number;
  retries?: { attempts: number; backoffMs: number; retryOn: (e: unknown) => boolean };
  fallback?: (input: In, ctx: Ctx, error: unknown) => Promise<Out>;
  /** Abstain instead of throwing when this stage cannot produce a usable result. */
  abstainOn?: (out: Out) => string | null;
};

type Ctx = {
  requestId: string;
  languageCode: string | null;
  effort: 0 | 1 | 2 | 3;
  deadlineAt: number;              // hrtime ms — the 200 ms budget, absolute
  timings: Partial<AskResponse["timings"]>;
  trace: TraceEvent[];
};
```

The runner enforces the **deadline, not just per-stage timeouts**. A stage that would start
after `deadlineAt` is skipped and the pipeline degrades to whatever it has. Per-stage
timeouts alone let a chain of near-misses blow the total budget.

## Retry policy — and where retries are wrong

Retries are only correct for **network-boundary, idempotent** stages. Retrying a CPU-bound
local stage burns the deadline for a deterministic failure that will fail identically.

| Stage | Retry? | Why |
| --- | --- | --- |
| Sarvam STT | 2 attempts, 200 ms backoff, on 5xx/network only | network, idempotent, outside the SLO window |
| Query embed | **no** | local and deterministic — a failure is a bug, not a blip |
| Qdrant search | 1 attempt, 20 ms backoff, connection errors only | loopback; a retry that costs 20 ms is affordable |
| Extractive span | **no** | pure function |
| LLM synthesis (Tier 3) | 2 attempts, on 429/5xx/timeout, jittered | network |
| Schema repair (Tier 3) | 1 attempt | see below |

Never retry a 4xx that is not 429. Retrying a 401 twice is 400 ms spent confirming the API
key is still wrong.

## Structured I/O

`AskRequest` in, `AskResponse` out, both validated at the boundary — see
[02-architecture.md](02-architecture.md) for the shapes. Validate with Zod (or hand-written
guards matching the existing style in
[`frontend/src/lib/conversation.tsx`](../frontend/src/lib/conversation.tsx), which already does careful
`reviveTurns` narrowing — worth staying consistent with).

For Tier 3 the LLM is **never** asked for free text. It returns a schema:

```ts
const SynthesisResult = z.object({
  answer: z.string(),
  usedChunkIds: z.array(z.string()).min(1),   // forces attribution
  grounded: z.boolean(),                       // self-report, verified downstream
  abstainReason: z.string().nullable(),
});
```

`usedChunkIds` is the load-bearing field. It forces the model to name which retrieved chunks
it used, which we then verify independently in Gate 4
([06-guardrails.md](06-guardrails.md)). A model that cites a chunk id we did not retrieve is
caught mechanically — no judge model required.

**Schema repair loop**: on a validation failure, one repair attempt that feeds the validation
error back. If it fails twice, abstain. Do not fall back to parsing free text — that
discards the only structural guarantee we have.

## Tool calls

At Tier 3 the model gets tools rather than a pre-stuffed prompt, so it can decide it needs
more evidence:

| Tool | Purpose |
| --- | --- |
| `search_corpus(query, strategy?, k?)` | re-query the index with a reformulated query |
| `expand_chunk(chunkId, window)` | pull neighbouring chunks — recovers S2/S3 context |
| `list_strategies()` | let the model pick a different chunking view of the corpus |

Tool loops cost round trips, so they are **capped at 2 iterations** and gated behind Tier 3.
This is the honest position: a tool-calling agent loop and a 200 ms budget are incompatible,
so tool use lives in the tier that has explicitly traded latency for quality.

Tiers 1–2 use the same `search_corpus` function directly as a typed call — same code path,
no model in the loop.

## Error recovery — the degradation matrix

Every dependency has a defined behaviour when it fails. Nothing 500s to the user if a
degraded answer is available.

| Failure | Behaviour | User sees |
| --- | --- | --- |
| Sarvam 401 / key missing | fail fast, no retry | "Speech service isn't configured." |
| Sarvam timeout / 5xx | retry ×2, then fail | "Couldn't hear that — try again." |
| Sarvam returns empty transcript | do not enter pipeline | existing 422 copy |
| Embedder not warm | await warm-up once, log a cold-start event | slower first answer |
| Qdrant down | **circuit breaker opens**, abstain | "My sources are unavailable right now." |
| Qdrant slow (> deadline) | skip remaining stages, abstain | "That took too long to look up." |
| Zero results above floor | abstain — Gate 2 | "I don't have that in my sources." |
| LLM 429 / down (Tier 3) | **fall back to Tier 1 extractive answer** | a shorter but grounded answer |
| LLM schema invalid ×2 | abstain | "I couldn't answer that reliably." |
| Any unhandled throw | catch at the route, return `status: "refused"` | generic message, `requestId` for logs |

The Qdrant circuit breaker matters: once open, subsequent requests abstain in ~1 ms instead
of each burning a full connection timeout. Half-open probe every 5 s.

The LLM→Tier 1 fallback is the nicest property in the table — because the extractive path
needs no network, an LLM outage degrades answer quality without taking the product down.

## Escalation

The ladder from [02-architecture.md](02-architecture.md), expressed as policy:

```ts
function nextTier(current: Tier, confidence: number, ctx: Ctx): Tier | "abstain" {
  if (confidence >= HIGH) return current;                     // good enough, stop
  if (current >= ctx.effort) return "abstain";                // user capped effort here
  if (Date.now() > ctx.deadlineAt && ctx.effort <= 1) return "abstain";  // out of budget
  return (current + 1) as Tier;                               // escalate
}
```

Three ways to stop: confident enough, out of user-permitted effort, or out of time. **All
three terminate in a real answer or an honest abstention — never in an error and never in a
guess.**

`ctx.effort` comes straight from the existing
[`EffortPanel`](../frontend/src/components/panels/effort-panel.tsx) slider, so the user's stated
patience is a first-class input to the control flow rather than a decorative setting.

## Observability

Every request emits one structured trace: `requestId`, tier reached, per-stage timings,
retry counts, circuit-breaker state, guardrail verdicts, final status. Appended to a ring
buffer, aggregated at `GET /api/metrics`.

Next 16's `instrumentation.ts` also exports **`onRequestError`**, which fires when the server
captures an error — the right place to record unhandled failures with their `requestId`
rather than scattering try/catch reporting through the route.

Two things this makes cheap: the live latency panel in the demo, and answering "why did it
refuse *that* query?" in seconds instead of by re-running it.
