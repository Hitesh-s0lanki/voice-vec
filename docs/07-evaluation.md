# 07 — Evaluation

Every claim in the submission traces back to something in this document. Nothing gets
asserted that isn't measured here.

## The four evaluations

| # | Question | Ground truth | Metric |
| --- | --- | --- | --- |
| E1 | Does retrieval find the right passage? | `is_selected` | recall@1/5/10, MRR@10 |
| E2 | Does the system know when not to answer? | `sum(is_selected) == 0` | precision / recall / F1 |
| E3 | Are the answers any good? | `Answer` | embedding similarity, token F1, LLM judge |
| E4 | Is it fast? | wall-clock | P50/P70/P95/P99/P100 |

E1 and E2 need no labelling and no judge model. That is the whole reason this dataset is
worth more than it first appears.

## The provenance problem — read this before writing the eval

`is_selected` labels a **passage**. We index **chunks**, and one passage becomes 1–N chunks
across five strategies ([03-chunking.md](03-chunking.md)). A naive `retrieved_id == gold_id`
comparison therefore scores 0 for every strategy except S1, which would look like a
catastrophic result and actually be a broken harness.

So every chunk must carry its provenance from ingest:

```ts
type ChunkPayload = {
  chunkId: string;
  strategy: ChunkStrategy;      // S1…S5
  sourceQueryId: number;        // the row it came from
  sourcePassageIdx: number[];   // WHICH passages — plural: S3 spans several
  language: string;
  text: string;
};
```

A retrieved chunk counts as **correct** if `sourcePassageIdx` intersects the gold passage
indices for that query. `sourcePassageIdx` is an array because S3 chunks deliberately span
passage boundaries — that is the point of the strategy, and the scorer has to know it.

Dedup complicates this further: after content-hash dedup, one chunk maps to several
`(query_id, passage_idx)` origins. Keep the full list; match against the entry for the query
under evaluation.

Get this wrong and every number downstream is wrong. Build and unit-test the scorer against
a handful of hand-checked rows **before** running it over 500.

## E1 — Retrieval

**Eval set.** Rows where `sum(is_selected) > 0` (~61%). Sample 500, stratified by
`query_type`.

**Query.** `query` (the translated one). This is the realistic input — it is what the user
speaks. Also run `Eng_Query` as a secondary condition to isolate how much retrieval quality
the translation costs us; that comparison is free and interesting.

**Metrics.**

```
recall@k = |{q : ∃ gold chunk in top-k}| / |Q|
MRR@10   = (1/|Q|) Σ 1 / rank of first gold chunk    (0 if none in top 10)
```

**Report per strategy and per query_type** — that is the comparison matrix in
[03-chunking.md](03-chunking.md), and the per-type split is where the argument lives.
Aggregate numbers hide the whole point, which is that different cuts win on different query
kinds.

### Two honesty notes

**The gold passage is guaranteed to be in the corpus.** We built the corpus from these rows,
so for answerable queries the right answer is always present. A real open-domain system has
no such guarantee. Recall here measures *ranking against ~250k–980k distractors*, which is
meaningful, but it is not open-domain recall. Say so.

**The queries retrieved these passages.** MS MARCO passages were selected by an earlier
retrieval system responding to these exact queries, so there is a lexical bias toward the
query in every candidate. This is inherent to MS MARCO and affects all strategies equally,
so strategy comparisons remain valid — absolute numbers should not be read as
open-web performance.

## E2 — Abstention

The headline result.

**Eval set.** All 500 sampled queries, ~39% of which are labelled unanswerable. Do **not**
rebalance to 50/50 — the natural ratio is the realistic workload, and precision on a
rebalanced set is not comparable to production.

**Labels.** `sum(is_selected) == 0` → *should abstain*. Treat this as authoritative rather
than the `"No Answer Present."` string; the two disagreed by one row in the 2,000-row sample
([01-dataset.md](01-dataset.md)) and the structural signal is the more reliable of the two.

**Confusion matrix.**

| | should answer | should abstain |
| --- | --- | --- |
| **answered** | ✅ correct answer | ❌ **hallucination risk** — answered something unanswerable |
| **abstained** | ❌ over-refusal — lost coverage | ✅ correct abstention |

```
abstention precision = correct abstentions / all abstentions
abstention recall    = correct abstentions / all should-abstain
answer coverage      = answered / all should-answer
```

The top-right cell is the one requirement 6 is really about. Report it as a raw count, not
only as a rate.

### Why this benchmark is hard, not trivial

The unanswerable rows still contributed their ~10 passages to the index. So for an
unanswerable query, the corpus contains passages that were retrieved *for that query* and are
topically adjacent — they simply don't answer it. The system cannot abstain by finding
nothing; it has to abstain despite finding plausible-looking, on-topic content.

That is exactly the situation where RAG systems hallucinate in production, and it is worth
stating explicitly in the writeup — it is the difference between "we tested abstention" and
"we tested abstention on the hard case."

**Deliverable:** sweep `FLOOR` and `MARGIN` from [06-guardrails.md](06-guardrails.md), plot
the precision/recall curve, mark the chosen operating point, justify it.

## E3 — Answer quality

Only on rows where the system answered and gold says it should have.

- **Embedding similarity** to gold `Answer`. Primary metric.
- **Token F1** (SQuAD-style, whitespace-tokenised after NFC normalisation). Secondary and
  noisy — see below.
- **LLM judge**, offline only, on a 100-row subsample: *is the answer supported by the cited
  context, and does it address the question?* Never on the latency path.

**Exact match is not usable here.** The translations are noisy — the mangled buffet phrase in
[01-dataset.md](01-dataset.md) is representative — and our answers are extracted from
*translated passages* while gold is a *translated answer*, produced by the same MT model but
from a different source sentence. Exact match would penalise correct extractions for
disagreeing with MT noise. Report token F1 with that caveat attached, and lead with embedding
similarity.

## E4 — Latency

Method, warm-up, environment reporting and the P100 caveat are all in
[04-latency.md](04-latency.md). Same 500-query sample, so latency and quality numbers
describe the same workload — including abstentions, which are faster than answers and would
otherwise flatter the median if excluded.

## Harness

`scripts/evaluate.ts`, one command, all four evaluations:

```
scripts/evaluate.ts --split validation --lang hi --n 500 --seed 42 --out reports/
```

- **Seeded sampling**, so runs are comparable and the reported sample is reproducible.
- **Writes both** a machine-readable `results.json` and the markdown tables for the writeup.
  Hand-copying numbers into a doc is how transcription errors get shipped.
- **Runs against the live `/api/ask`**, not against internal functions — otherwise the eval
  measures a code path the demo does not use.
- **Records the config** it ran against: chunk strategies enabled, floors, model, index size,
  commit SHA.

## Reporting checklist

Before the numbers go in the submission:

- [ ] N stated on every percentile and every rate
- [ ] Environment stated: machine, cores, RAM, Node version, Qdrant local/remote
- [ ] Index stated: chunk count per strategy, vector count, index size on disk
- [ ] Warm-up discarded and said to be discarded
- [ ] Both latency tables present — SLO window **and** full wall-clock including STT
- [ ] Tier 3 reported separately, with its 200 ms miss acknowledged
- [ ] Per-strategy and per-query_type breakdowns, not just aggregates
- [ ] Abstention confusion matrix as raw counts
- [ ] Gate 1 false-positive rate on legitimate queries
- [ ] Known limitations section: gold-in-corpus, MS MARCO lexical bias, MT noise, single
      language, validation-split-only
- [ ] Every number in the writeup traceable to `reports/results.json`

That last checkbox is the one that matters. A judge who spot-checks one number and finds it
reproducible will believe the rest.
