# 13 — Asking in a language the index does not hold

*Why "I heard English, but my sources are only indexed in Hindi" was the wrong answer,
and what replaced it.*

The index holds one language. Hindi, 19,870 chunks from `hinval-2000.parquet`
([01-dataset.md](01-dataset.md)). Ask it something in English and it used to say:

> I heard English, but my sources are only indexed in Hindi.

That sentence came out of Gate 1, before a single vector was compared. It was wrong twice
over, and the second reason is the interesting one.

## It was hiding an empty table

> **Since superseded.** There is no deployment index and no manifest any more — retrieval
> reads whatever vector store its asker connected ([13-connectors.md](13-connectors.md)), so
> `gate_input` is called with an empty language list and the mismatch branch fires only on a
> language code that does not resolve at all. The finding below is why the gate stopped
> refusing on a language label, and that part still holds.

The gate read `languages` out of `data/index-manifest.json`. That manifest said
`"collection": "vec-chunks"` and was written on 19 August against the **embedded Qdrant
store**, one day before the store moved to Postgres. Nothing had ever been ingested into
Neon: `SELECT count(*) FROM chunks` returned **0**.

So the language refusal was the only thing anyone ever saw, and it was standing in front of
a table with nothing in it. A Hindi question would have got as far as the database and
abstained too, just with different words. Both are fixed by the same ingest — but the
manifest is why it looked like a language problem.

> The lesson worth keeping: the manifest is a *claim* about the index, written by the
> ingest. It is not the index. `/health` reports `chunks` from `count(*)` for exactly this
> reason; that number was the one telling the truth all along.

## It was refusing something the design supports

The embedder is `multilingual-e5-small`, chosen over a monolingual model precisely because
it puts a question and its translation in the same region of the same 384-dim space. And
every chunk carries `english` — the original MS MARCO passage the Hindi was translated from
([01-dataset.md](01-dataset.md)). Both halves of a cross-lingual answer were already in the
table:

```
"what is a corporation"  ──► e5 embeds the English question
                         ──► HNSW over Hindi passages          (shared space)
                         ──► the right chunk comes back
                         ──► answer cut from chunk.english     (original English)
```

Refusing on the strength of a language *label* threw both away. So Gate 1 no longer
refuses; it **routes**. A mismatch turns the language filter off, flags the request
`cross-lingual`, and answers from the English rendering. Whether the retrieval was actually
good enough is Gate 2's job, where it is a measured score rather than a tag comparison.

The same applies to a language code that does not map to a FLORES tag at all — French, say.
e5 covers far more languages than this dataset is tagged with, and with one language indexed
there is no filter to get wrong.

## What it costs, measured

`scripts/crosslingual.py` asks the same question twice — MSMARCO-XI carries `Eng_Query`
beside the Hindi `query` for every row, which is unusually good ground truth for this. 200
questions, seed 42, against the live index:

| | top score | margin | recall@5 |
| --- | --- | --- | --- |
| Hindi → Hindi index | 0.8942 | 0.0239 | **0.6967** |
| English → Hindi index | 0.8469 | 0.0179 | **0.6475** |

**Cross-lingual retrieval costs about five points of recall@5.** That is the whole finding.
The capability the gate was refusing works, and works nearly as well as the native path.

What does not survive the crossing is the *scale*. Both the scores and the gaps between them
compress, so thresholds swept on Hindi-against-Hindi are a stricter test in English than
they were in Hindi — and they abstain on retrieval that was right.

### The floor

| floor | coverage | abstention recall |
| --- | --- | --- |
| 0.70 – 0.78 | 34.43% | 75.64% |
| 0.80 | 33.61% | 75.64% |
| 0.845 *(Hindi default)* | 25.41% | 79.49% |
| 0.86 | 13.11% | 85.90% |

`RETRIEVAL_FLOOR_CROSS_LINGUAL = 0.78` — the highest floor that costs nothing. Coverage is
flat below it, which means the floor is inert and the margin decides. That is the same
regime `RETRIEVAL_FLOOR = 0.845` was picked in for Hindi ([09-v1.md](09-v1.md)).

### The margin, which matters more

| margin | coverage (hi / en) | abstention recall (hi / en) |
| --- | --- | --- |
| 0.010 | 77.05% / 73.77% | 33.33% / 30.77% |
| 0.015 | 58.20% / **53.28%** | 47.44% / 52.56% |
| 0.020 | **52.46%** / 34.43% | 62.82% / 75.64% |
| 0.030 | 29.51% / 13.11% | 88.46% / 93.59% |

At the same setting, the margin test lets 52% of answerable Hindi questions through and only
34% of English ones. `RETRIEVAL_MARGIN_CROSS_LINGUAL = 0.015` is chosen to land English on
the *same operating point* Hindi already sits at — 53.28% against 52.46% — rather than to
express a new preference about how often the system should answer.

The Hindi thresholds are untouched. They are the swept, reported ones behind E2 in
[09-v1.md](09-v1.md), and moving them would invalidate a measured number to make a demo
look better.

## The thing that is still off, and is not cross-lingual

`कॉर्पोरेशन क्या है?` — the dataset's own first row — **abstains in Hindi** while its
English twin now answers. Its trace:

```json
{"gate": "retrieval", "top": 0.874, "margin": 0.0088, "floor": 0.845, "ok": false}
```

Top score well clear of the floor; the margin kills it. The four best hits are *all*
definitions of a corporation, within 0.013 of each other, so the margin reads "several
passages are equally good" as "nothing stands out".

That is the margin test working exactly as specified, on a corpus that breaks its
assumption: MS MARCO gives every query ten passages and several of them usually answer it.
The guardrail was tuned for best abstention F1 on a validation split that is 39%
unanswerable — it is *supposed* to prefer silence. Whether that is the right trade for a
voice demo is a product decision with a measured price, and the table above is the price.
Nothing here changes it.

## What this cost the latency budget, and what paid it back

Turning retrieval on put the 200 ms of [04-latency.md](04-latency.md) under a load it was
never measured against: that budget assumed a ~11 ms in-process search, and the index now
lives on Neon, 66 ms of round trip away.

The first measurement of the moved store was worse than the distance explains — 273 ms for
a search whose ANN query costs about 4 ms. It was not the database. Every search wrapped
itself in an explicit transaction so it could `SET LOCAL hnsw.ef_search`, and then the pool
committed that transaction on the way out:

| | round trips | measured |
| --- | --- | --- |
| `BEGIN` + `SET LOCAL` + `SELECT` + `COMMIT` | 4 | 273 ms |
| `SELECT` + `COMMIT` | 2 | 192 ms |
| `SELECT` | 1 | **66 ms** |

Three of the four round trips were bookkeeping around a read. `hnsw.ef_search` moved to
`Database._configure`, where it is set once per connection, and the implicit transaction is
skipped by flipping `autocommit` for the duration of the query — client-side state in
psycopg, so it emits no statement, and restored before the connection goes back because
this pool is shared with everything that writes.

End to end, warm, against the live index:

| | before | after |
| --- | --- | --- |
| answered, English, cross-lingual | ~500 ms | **221 ms** |
| abstained, Hindi (stops at the search) | ~420 ms | **87 ms** |

Still outside 200 ms for an answered query, and honestly so: 66 ms of it is the distance to
`ap-southeast-1` and 82 ms is the extraction rerank. `scripts/migrate.py` warns about the
first at ingest time for exactly this reason. The budget is met again by moving the database
into the region the API runs in, not by anything left in the code.

## Reproducing

```bash
uv run python -m scripts.crosslingual --n 200     # → reports/crosslingual.json
```

Both thresholds are properties of **this index**, not of the model. Re-run after any
re-ingest, for the same reason the Hindi floor has to be re-swept.
