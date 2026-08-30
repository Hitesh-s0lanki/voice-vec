# 25 — Removing the local embedder

*The ONNX session cost ~700 MiB to save a round trip. On a 512 MiB instance that is not a
trade, it is a crash loop.*

## What happened

The first deploy of the image from [24-deploy.md](24-deploy.md) died on Render before it ever
bound a port:

```
==> uvicorn on 0.0.0.0:10000 (1 worker(s))
INFO:     Waiting for application startup.
==> Out of memory (used over 512Mi)
==> Instance restarted
```

It stops at `Waiting for application startup.` because that is the line before the lifespan
called `embedder.warm()`. Measured on the boot path:

| Stage | Resident |
| --- | --- |
| Bare interpreter | 19 MiB |
| `src.main` imported — FastAPI, langchain, langgraph, psycopg, numpy | 214 MiB |
| **ONNX session loaded** — `intfloat/multilingual-e5-small` | **922 MiB** |

Steady state was 922 MiB too, unchanged after twenty queries — so it was not a load spike
that a bigger `start-period` could ride out. The model file is 448 MB of fp32 weights and
onnxruntime's arena adds ~260 MiB on top.

**Baking the model into the image had nothing to do with it.** That decision changed where
the weights came from, not how much RAM they occupy; downloading them at boot would have
OOMed on the same line, just slower.

The quantised export was measured too — `onnx/model_qint8_avx512_vnni.onnx`, which
`EMBED_MODEL_FILE` could already have selected — and it peaked at **732 MiB**. Not enough.
There was no configuration that fitted.

## What replaced it

`text-embedding-3`, through [`src/rag/remote_embed.py`](../src/rag/remote_embed.py), which
already existed. It was written for connected stores that are not 384 wide
([13-connectors.md](13-connectors.md)) and it is now the only embedder there is.

```
before                                   after
──────                                   ─────
e5-small, 384d, in-process               text-embedding-3-small, 1536d, over HTTPS
  the question                             the question
  reranked passages                        reranked passages
  extraction shortlist                     extraction shortlist
  grounding claims + evidence              grounding claims + evidence
  capability cards                         capability cards
text-embedding-3 for other widths        …and the connected stores, unchanged
```

[`src/rag/embed.py`](../src/rag/embed.py) keeps the interface the five callers already used —
`embed_query`, `embed_passages`, `dim`, `model_name`, `ready` — so nothing outside it changed
shape. What it lost is `warm()`, `count_tokens()` (no callers since the ingest pipeline went)
and the e5 `query: ` / `passage: ` prefixes, which were an e5 rule and would be six
characters of noise in a `text-embedding-3` input.

### Measured result

| | Before | After |
| --- | --- | --- |
| Resident at startup | 922 MiB | **184 MiB** |
| Image size | ~1.5 GB | **120 MB** |
| Boot to first response | ~7 s | **1.6 s** |
| Runtime packages | — | **15 fewer** (onnxruntime, huggingface-hub, tokenizers, pillow, protobuf, …) |
| One query embed | 3.9 ms | **0.4 – 2.9 s** |

Verified in a container with `--memory=512m --memory-swap=512m`: boots in 1.6 s, serves
`/health` as `ok`, sits at **182 MiB / 512 MiB**, and the image healthcheck reports healthy.

## What it costs, honestly

**The 200 ms budget does not survive this, and the code now says so.** Up to five embedding
points exist in a turn — capability discovery, the store query, rerank, extraction, the
grounding gate — and each is a round trip where each was a few milliseconds.

The budget machinery already existed for exactly this and it is what keeps the change honest.
`rerank` and `extract` price the stage before running it and skip when it will not fit; the
grounding gate returns "did not run" rather than failing. What changed is *how* they price it:

```python
# before — the ONNX forward pass was linear in the batch
estimate_ms = len(candidates) * EMBED_MS_PER_PASSAGE   # 8 ms each

# after — one HTTP request for the whole batch, so count is irrelevant
if settings.embed_call_ms > budget_ms:                 # 600 ms, once
```

`embed_call_ms` defaults to 600, which is deliberately larger than the whole 200 ms deadline.
Inside a spoken turn the optional stages therefore decline to run and report it, rather than
overrunning silently. The consequence is real and worth stating plainly: **on the voice path
this build reranks less and extracts lexically more often than the one before it.**

**A new failure mode.** Embedding can now fail — no key, a 401, a provider outage. Every
caller was already built for an embedder that can raise, and each degrades rather than
breaking the turn:

| Caller | Without an embedder |
| --- | --- |
| `rerank` | fusion order, reported as `order` rather than `reranked` |
| `extract` | the lexical winner |
| `guardrails` | the gate does not run, and says so in the trace |
| `capabilities` | word-overlap discovery |
| a connected store | `StoreUnavailable` → "my sources are unavailable" |

`Embedder.ready` answers from configuration — a key, and a width something can produce — and
never spends a call. It is not a promise the key works; a bogus key boots clean, reports
`embedder_ready: true`, and fails on first use with `text-embedding-3-small answered 401`.
That is the honest split: readiness is knowable for free, correctness is not.

## The thresholds had to move, and one of them was silently broken

This is the part that would not have shown up in a test suite. Two thresholds were swept
against **e5's** score distribution, and `text-embedding-3` spreads scores far wider.

### `capability_min_lift`: 0.023 → 0.10

e5 put every card in a band about 0.003 wide, and 0.023 was the gap in it
([23-capabilities.md](23-capabilities.md)). On the new embedder that floor admitted
*everything* — all five of the probe's deliberately unanswerable queries matched a capability:

```
BEFORE (lift 0.023)                         AFTER (lift 0.10)
what is the weather in Oslo → titanic  ✗    → nothing  ✓
who won the world cup       → titanic  ✗    → nothing  ✓
tell me a joke              → pgvector ✗    → nothing  ✓
what time is it             → gmail    ✗    → nothing  ✓
how are you feeling today   → gmail    ✗    → nothing  ✓
```

Re-measured over 21 queries against a four-capability account:

```
relevant     0.075 – 0.485   (nine of eleven above 0.149)
irrelevant   0.025 – 0.089
```

0.10 sits in that gap with room on both sides: **14/14 probe queries route correctly**. It
costs one — *"summary of the book The Laws of Human Nature"* lifts 0.075, because the card
says *book passages* and the question names a title the card has never seen. That is a card
that could be better rather than a threshold that should be lower; dropping to 0.075 to catch
it also admits *"sing me a song"* at 0.089.

### `generation_support_floor`: 0.62 → 0.50

Measured over three grounded and three ungrounded claim/evidence pairs:

```
grounded     0.611 – 0.856
ungrounded   0.199 – 0.405
```

0.62 was e5's number and sat *above* the lowest genuinely grounded claim, so it refused an
answer its own evidence supported. 0.50 is the middle of the measured gap.

**Both are probes, not sweeps.** Twenty-one hand-written queries and six claim pairs are
enough to show the old numbers are wrong here and the new ones are not; they are not enough
to call the new ones optimal. The corpus those sweeps would need was removed in
[22-no-local-corpus.md](22-no-local-corpus.md).

### What did *not* need re-measuring

`retrieval_floor` and `retrieval_margin` score a **connected store's** own similarity numbers,
which come back from somebody else's index. Those were already not a property of anything
this app embeds, and [13a-cross-lingual.md](13a-cross-lingual.md) already said they are the
first dials to re-check on a store you connect. `capability_min_overlap` is a share of the
question's words, not a cosine, so no embedder change touches it.

## One thing that quietly got better

A store that reports 384 dimensions used to be searched with e5, on the assumption that e5
built it. Now every width — the app's own included — goes through one call to one model, so
the "is this the same vector space" question is asked in exactly one place and answered the
same way for every store. It does not make width identity: a 1536-wide index built with
something other than `text-embedding-3` still returns neighbours that are arithmetically
valid and semantically meaningless, which is what the profiler's round-trip check exists for
([17-understanding.md](17-understanding.md)).

## If the memory comes back

Nothing here is irreversible, and the shape to restore is `Embedder` — one class, one
interface, five callers that do not care. What would have to come back with it is
`EMBED_MODEL`, the fastembed dependency, ~700 MiB of instance, and both thresholds above.
