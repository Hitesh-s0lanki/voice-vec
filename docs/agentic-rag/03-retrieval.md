# 03 — Retrieval

Source modules:
[`hybrid-search-strategies/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/hybrid-search-strategies) ·
[`multi-model-openai/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/multi-model-openai) ·
[`graphdb/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/graphdb)

Retrieval is a **funnel**, not a step. The standard shape:

```
query → [enhance] → [recall: cast wide, cheap]  → [precision: rerank, expensive] → [diversify] → context
                     BM25 + dense, k≈20-50         cross-encoder, k≈5              MMR
```

Each stage has a different job. Recall stages must not lose the right answer — nothing
downstream can recover a passage that was never fetched. Precision stages must order what
survived. Confusing the two is the most common design error: raising `k` on a single dense
retriever improves recall and *degrades* the context you hand the LLM, because rank 18 is
now in the prompt.

---

## Dense retrieval

The baseline, and what "vector search" usually means. Embed the query, find nearest
neighbours, return them.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

Used in essentially every notebook in the repo. Strengths and weaknesses are the same fact
seen twice: it matches on **meaning**, so it finds "MacBook M3 for developers" for "best
laptop for coding" — and it misses exact tokens, so it can fail on a product SKU, an error
code, a person's name, or a rare acronym that never appeared in the embedding model's
training data.

## Sparse retrieval — BM25

```python
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3
```

Lexical scoring over term frequency and inverse document frequency, with length
normalisation. No model, no embeddings, no GPU — it is a data structure. Exactly complements
dense: it nails the exact token and has no idea about synonyms.

Never treat BM25 as the legacy option. On rare terms, identifiers, and out-of-domain
vocabulary it beats dense retrieval outright, and it costs nothing to run alongside.

## Hybrid {#hybrid}

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weight=[0.7, 0.3]           # ← BUG: the field is `weights`
)
```

`EnsembleRetriever` fuses ranked lists with **weighted Reciprocal Rank Fusion**:

```
score(d) = Σ_i  weight_i / (c + rank_i(d))        c = 60 by default
```

RRF fuses **ranks, not scores**, which is why it works at all — a FAISS L2 distance and a
BM25 score are not on comparable scales and cannot be averaged, but their rank orderings can
be combined.

**The repo's call has a real bug.** The field is `weights` (plural). Passing `weight=` leaves
`weights` unset, and the validator then defaults it to equal weighting — `[0.5, 0.5]`. The
intended 70/30 dense-lean is silently not applied. Verified against
`langchain_classic/retrievers/ensemble.py`:

```python
weights = values.get("weights")
if not weights:
    values["weights"] = [1 / n_retrievers] * n_retrievers
```

Nothing errors and nothing warns, so the notebook looks like it works. Detail in
[07-findings.md](07-findings.md#ensemble-weights).

## MMR — Maximal Marginal Relevance {#mmr}

```python
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
# elsewhere: search_kwargs={"k": 4, "lambda_mult": 0.7}
```

Solves the **redundancy** problem. Pure top-*k* similarity, run over an overlapping-chunk
index, happily returns five near-copies of the same passage — five slots of context spent on
one fact. MMR selects iteratively:

```
MMR = argmax [ λ · sim(d, query) − (1 − λ) · max sim(d, already_selected) ]
```

Each pick is rewarded for matching the query and penalised for resembling what's already
chosen. `lambda_mult=1.0` is pure relevance (plain similarity); `0.0` is pure diversity;
`0.5` is LangChain's default and `0.7` (the repo's explicit setting) leans relevance.

MMR is the natural partner to chunk overlap and to multi-query expansion, both of which
manufacture duplicates by design.

## Re-ranking

Two-stage retrieval: fetch wide and cheap, then re-score with something accurate and slow.

```python
retriever_openai = vectorstore.as_retriever(search_kwargs={"k": 8})   # wide recall
# → LLM scores all 8 and returns a ranked index list
```

The repo implements the **LLM-as-reranker** variant: format the 8 candidates into a prompt,
ask for `2,1,3,0`-style ordering, parse, reorder.

Why re-ranking works: a bi-encoder embeds query and document *independently*, so it can
never model term-level interaction between them. A **cross-encoder** reads the pair jointly
and scores it — far more accurate, and far too slow to run over a whole corpus. Hence the
funnel: bi-encoder for recall over millions, cross-encoder for precision over dozens.

| Reranker | Latency | Notes |
| --- | --- | --- |
| Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | ~10–50 ms local | Best quality/cost ratio. **Not in this repo.** |
| Managed API (Cohere Rerank, Voyage) | 100–300 ms | One network hop |
| LLM reranker (the repo's) | 500 ms – 2 s | Most flexible, slowest, non-deterministic |

The repo picked the most expensive option and skipped the standard one. Its implementation
also has a parsing bug — the prompt shows a 0-based example (`2,1,3,0`) while the documents
are numbered 1-based and parsed with `int(x) - 1`, so a literal `0` silently becomes index
−1, which Python reads as the *last* document. See
[07-findings.md](07-findings.md#rerank-indexing).

## Graph retrieval

```python
graph = Neo4jGraph(url=..., username=..., password=...)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
chain.invoke({"query": "Who was the director of the movie Casino"})
```

A different retrieval paradigm entirely: **generate a query language, execute it, answer from
the rows.** The LLM sees the graph schema (`graph.refresh_schema()`), writes Cypher, Neo4j
executes it, the LLM narrates the result.

What this buys that vectors cannot: **multi-hop and aggregate questions.** "How many artists
are there?" is a `COUNT` — no passage in any corpus contains that number, so no amount of
semantic search will find it. "Which directors worked with actors who were in Casino?" is a
two-hop traversal. Vector search retrieves *passages that mention things*; graph search
retrieves *answers computed over relationships*.

The repo also shows the two techniques that make text-to-Cypher usable in practice:

- **Schema pruning** — `exclude_types=["Genre"]` shrinks the schema in the prompt, cutting
  tokens and the model's opportunity to hallucinate a label.
- **Few-shot Cypher examples** in `promptstatergies.ipynb`, which is the single highest-value
  intervention for query-language generation accuracy.

The unaddressed risk: a generated query is arbitrary code against your database. There is no
validation, no read-only enforcement, no timeout, no row cap. `GraphCypherQAChain` carries an
explicit upstream warning about this.

> ⚠️ Both graph notebooks hardcode live Neo4j Aura credentials.
> [07-findings.md](07-findings.md#security).

## Multimodal retrieval {#multimodal}

```python
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# text and images embedded into ONE 512-dim space, both L2-normalised
features = features / features.norm(dim=-1, keepdim=True)
```

[`multi_modal_openai.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/multi-model-openai/src/multi_modal_openai.ipynb)
is the most technically interesting notebook in the repo. It walks pages with PyMuPDF,
extracts text *and* embedded images, and puts both into a **single unified FAISS index** via
CLIP — so one text query retrieves text chunks and images ranked against each other in the
same space.

The two-model split is the design worth stealing:

- **CLIP for retrieval** — cheap, joint space, but a 77-token text limit and weak at fine
  detail.
- **GPT-4V for generation** — images re-attached as base64 `image_url` parts alongside the
  text excerpts, so the answering model actually *sees* the figures.

CLIP is the retriever precisely because it is the only cheap model with text and images in
one space; it is not the generator because it cannot reason about what it sees.

Two limits the notebook doesn't flag: CLIP truncates text at 77 tokens, so 500-character
chunks are being **silently clipped** during embedding; and CLIP is trained on
caption-length text, so it is a poor encoder for prose paragraphs. A production build embeds
text with a text model and images with CLIP, in separate indexes, and fuses the results.

---

## Not in this repo

| Technique | What it solves |
| --- | --- |
| **Cross-encoder reranking** | The standard, fast reranker. Its absence is the biggest gap here. |
| **Multi-query retrieval** | Generate N paraphrases, retrieve for each, union. Covers phrasing variance the single query misses. |
| **Parent-document retrieval** | Search small, return big. See [01-chunking.md](01-chunking.md#not-in-this-repo). |
| **Self-query retrieval** | LLM extracts metadata filters from natural language ("papers after 2023" → `year > 2023`) and combines them with semantic search. |
| **Contextual compression** | Trim retrieved chunks to only the query-relevant sentences before they hit the prompt. |
| **ColBERT / late interaction** | Per-token embeddings with MaxSim scoring — cross-encoder-ish quality at bi-encoder-ish speed. |
| **Time-weighted retrieval** | Decay by recency for changing corpora. |
| **Metadata pre-filtering at scale** | Shown for FAISS, never discussed as a design tool. |

---

## Choosing

| Symptom | Fix |
| --- | --- |
| Misses exact IDs, codes, names | Add BM25, fuse with RRF |
| Misses paraphrases | Dense retrieval; multi-query expansion |
| Right passage retrieved but ranked 8th | Cross-encoder reranker |
| Top-*k* is five copies of one passage | MMR, `lambda_mult` 0.5–0.7 |
| Chunks too small to answer from | Parent-document or sentence-window |
| "How many…" / multi-hop relational | Graph retrieval |
| Answer is in a figure or chart | Multimodal index + vision generator |
| Query has implicit filters | Self-query retriever |

**The default stack for a new system:** BM25 + dense, fused with RRF, `k≈20`, then a local
cross-encoder down to `k≈5`, then MMR if the corpus has overlapping chunks. The repo has
three of those four pieces and is missing the cross-encoder.

Next: [04-query-enhancement.md](04-query-enhancement.md).
