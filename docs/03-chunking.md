# 03 — Chunking

> Requirement 2: *"Chunking strategy should be vast — don't submit a single naive fixed-size
> chunking approach. We want to see real thought put into how the dataset is split, indexed,
> and retrieved."*

This is the requirement with the most room to win or lose on, and it has a trap in it.

## The trap

**MS MARCO passages are already chunked.** Each one is a ~50–100 word span that some earlier
retrieval system pulled for a query. If we ingest `Translated_passages[]` one row per
passage and call it done, we have performed *zero* chunking work while appearing to have a
populated index. That is precisely the "single naive approach" the brief rejects — it just
doesn't look naive, because the naivety was inherited from the dataset.

Real chunking work here means deliberately **re-cutting** the corpus along several different
axes and proving which cut retrieves best for which kind of query.

## What the dataset gives us to cut along

Three structural facts, all from [01-dataset.md](01-dataset.md):

1. **`query_id` clusters passages.** The ~10 passages under one row were retrieved for the
   same query, so they are topically coherent. They form a natural pseudo-document, which
   means larger-than-passage chunks are available to us even though the source is
   passage-shaped.
2. **Every chunk has an aligned English twin.** `English_passages[i]` and
   `Translated_passages[i]` are the same content. We can index both against one chunk id.
3. **Every chunk carries free metadata** — `query_type`, `target_lang`, `is_selected`,
   `query_id` — which becomes payload filters and routing signals.

## Five strategies

All five are indexed into one Qdrant collection using **named vectors**, so a single query
can hit one strategy, several, or all of them and fuse. Every chunk carries
`strategy: ChunkStrategy` in its payload.

### S1 — Passage-atomic *(the control)*

Each `Translated_passages[i]` indexed verbatim, one vector per passage.

This is the baseline we are measuring the others against, and it is not a throwaway — for
`DESCRIPTION` queries, which are 72% of the corpus, the passage boundary is often already
the right answer boundary. Reporting that a fancier strategy *lost* to this on some slice is
a stronger result than pretending it didn't.

### S2 — Sentence-window

Split each passage into sentences. Index each sentence individually, but attach the
neighbouring sentence on each side as context that is returned but **not embedded**.

- Embed: the sentence alone → precise matching
- Return: sentence ±1 → enough context to answer from

This is the strategy that should win on `NUMERIC` (20% of the corpus) and `ENTITY` queries,
where the answer is one clause and passage-level embedding dilutes it across 80 irrelevant
words.

Indic sentence splitting is not `.split(".")`. Devanagari uses **danda `।`** and double
danda `॥`; Urdu uses **Arabic full stop `۔`** and Arabic comma `،`. Splitting on the Latin
period alone silently produces one giant "sentence" per Hindi passage — a bug that looks
like bad recall, not like a bug.

### S3 — Cluster-and-recut

Concatenate all ~10 passages under a `query_id` into one pseudo-document, then re-split at a
larger target size with overlap.

- Target ~256 tokens, ~15% overlap
- Boundaries snap to sentence edges, never mid-sentence
- Overlap is stored once and referenced, not duplicated, to keep the index from bloating

This recovers context that per-passage indexing destroys: when the answer spans two adjacent
passages, S1 and S2 can only ever return half of it. It is also the only strategy that
produces chunks larger than the source granularity.

Deduplicate before concatenating — the same passage recurs across `query_id`s.

### S4 — Semantic (similarity-trough) splitting

Split at topic shifts rather than at a fixed size. Embed each sentence, walk the sequence,
and cut where cosine similarity between consecutive sentences drops below a threshold
(percentile-based, not absolute — absolute thresholds do not transfer across languages).

Expensive offline, free at query time. Worth it because MS MARCO passages are frequently
*not* coherent — passage 4 in the sample row is a dictionary entry with two unrelated
example sentences glued together. Semantic splitting separates them; fixed-size splitting
cements the mess.

Optional refinement — **late chunking**: embed the full cluster document once, then mean-pool
over each chunk's token span. Each chunk vector then carries whole-document context. Only
viable where the cluster fits in the model's 512-token window, which after Indic tokenisation
often it will not. Treat as a stretch goal, not a dependency.

### S5 — Bilingual, metadata-prefixed

The interesting one. For each chunk:

- **Named vector `indic`** — embedding of the Hindi/Tamil/… text
- **Named vector `english`** — embedding of the aligned `English_passages[i]` span
- Prefix a compact metadata header before embedding:
  `[hi | DESCRIPTION] <chunk text>`

Two payoffs. First, cross-lingual retrieval **without a query-time translation call** — a
Hindi query can match the English vector directly through the shared multilingual embedding
space, which is the only way to get cross-lingual benefit inside the 200 ms budget (see
[02-architecture.md](02-architecture.md)). Second, the metadata prefix nudges the embedding
toward the right region of space for typed queries, at the cost of a few tokens.

Ablate the prefix. Metadata prefixing sometimes helps and sometimes adds noise; we should
report which it did rather than assume.

## Overlap, dedup, and the things that quietly break

**Overlap.** Only S3 overlaps (~15%). S2's window is *returned* context, not *embedded*
context, so it is not overlap in the index sense and does not inflate vector count. Track
these separately or the index-size accounting will not add up.

**Dedup.** Passages repeat across `query_id`s because MS MARCO retrieved the same passage
for related queries. Hash normalised chunk text (NFC-normalise, collapse whitespace, strip
punctuation) and keep one copy with a list of source `query_id`s. Do this **before**
embedding — it is the cheapest speedup available in the whole ingest. Measure and record the
dedup rate; it is a good number to report.

**Unicode normalisation.** Devanagari has multiple valid encodings for the same grapheme
(precomposed vs. combining nukta). Without `String.prototype.normalize("NFC")` at ingest
*and* at query time, dedup misses duplicates and exact-match lexical search silently fails.

**Token budgets are not character budgets.** Indic scripts tokenise far worse than English
in multilingual sentencepiece vocabularies — commonly 1.5–2.5× more tokens for equivalent
content. A "100-word passage" that is 130 tokens in English may be 300+ in Hindi and get
truncated at the model's 512-token limit without any error being raised. **Set all chunk
sizes in tokens, measured with the actual tokeniser, never in characters.** Log the p99
token length per strategy during ingest and confirm nothing is being silently cut.

**The e5 prefix rule.** `query: ` on queries, `passage: ` on chunks. Applied inconsistently
this degrades recall substantially with no error and no obvious symptom.

## Retrieval: hybrid and fused

Chunking only pays off if retrieval exploits it.

**Dense + sparse in one query.** Qdrant carries a sparse vector (BM25-style term weights)
alongside each dense vector. Dense handles paraphrase; sparse handles exact tokens — model
numbers, product names, years — which is where `NUMERIC` and `ENTITY` queries live. Fuse
with Reciprocal Rank Fusion, which needs no score calibration between the two:

```
RRF(d) = Σ_r  1 / (k + rank_r(d))      k = 60
```

**Strategy routing.** `query_type` is not available at query time — it is a dataset label,
not an input. So we cheaply infer intent from the transcript (numeral presence, question
word, length) and route:

| Inferred intent | Strategies queried | Rationale |
| --- | --- | --- |
| numeric / entity | S2 + S5, sparse-weighted | short precise answers, exact tokens matter |
| descriptive | S1 + S3 | needs surrounding context |
| ambiguous / long | S3 + S4 + S1, RRF-fused | cast wide, let fusion sort it |

Routing costs microseconds and is pure classical code — no model call, no latency.

**Then deduplicate results.** Because the same underlying text is indexed under five
strategies, the top-10 will contain the same content repeatedly. Collapse by source chunk id
*after* fusion, keeping the highest-scoring representative. Skipping this hands the answer
stage five copies of one passage and makes the context look richer than it is.

## Proving it — the comparison matrix

This table is the deliverable for requirement 2. It cannot be predicted, only measured, and
it is what separates "we thought about chunking" from "we tested chunking":

|  | recall@1 | recall@5 | MRR@10 | chunks | index MB | mean latency |
| --- | --- | --- | --- | --- | --- | --- |
| S1 passage-atomic | | | | | | |
| S2 sentence-window | | | | | | |
| S3 cluster-recut | | | | | | |
| S4 semantic | | | | | | |
| S5 bilingual | | | | | | |
| **RRF fusion of all** | | | | | | |

Broken out by `query_type`, since the whole argument is that different cuts win on different
query kinds. Ground truth is `is_selected` — no labelling needed. Method in
[07-evaluation.md](07-evaluation.md).

Expect S3 or the fusion row to win overall and S2 to win on `NUMERIC`. If the results
disagree, report the results.

## Ingest pipeline

`scripts/ingest.ts`, run offline, resumable:

```
parquet (DuckDB, streamed)
  → flatten rows to (query_id, passage_idx, indic, english, is_selected, query_type)
  → NFC normalise
  → dedup by content hash                        ← log the dedup rate
  → fan out into S1…S5                           ← shares src/lib/rag/chunk.ts with runtime
  → tokenise + length check                      ← log p99 tokens/chunk, assert < 512
  → embed in batches (multilingual-e5-small ONNX)
  → upsert to Qdrant with named vectors + payload
```

Notes that matter in practice:

- **Batch the embedding.** Per-chunk calls will be 10–50× slower than batches of 64–256.
- **Checkpoint by `query_id` range.** Ingest will crash at some point; restarting from zero
  after 40 minutes is avoidable.
- **`chunk.ts` is shared with the runtime**, not duplicated. Chunking logic that differs
  between ingest and query time is a class of bug that produces quietly wrong retrieval and
  is nearly impossible to spot from the outside.
- Embedding ~250k chunks (Phase B) is plausibly 10–30 minutes on Apple Silicon with
  batching; ~980k (Phase C) proportionally more. **These are estimates — measure Phase A and
  extrapolate before committing to Phase C.**
