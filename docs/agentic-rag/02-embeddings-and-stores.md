# 02 — Embeddings and vector stores

Source modules:
[`vector-embeddings/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/vector-embeddings) ·
[`vector-store-database/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/vector-store-database)

## The one idea

An embedding maps text to a point in ℝⁿ such that semantically similar text lands nearby.
Everything else — the index, the database, the distance metric — is machinery for finding
near points quickly. The repo's `vector-embeddings/README.md` makes the distinction that
most write-ups blur:

> Semantic search **uses** cosine similarity, but cosine similarity alone does **not** make a
> search semantic.

Cosine similarity is a ruler. The embedding model is what makes the thing being measured
meaningful. Swap in a bad model and cosine will still return confident numbers.

## Distance metrics

`vector-representation.ipynb` implements cosine from scratch:

```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

| Metric | Range | Notes |
| --- | --- | --- |
| Cosine | −1 … 1 | Angle only, magnitude-invariant. The default for text. |
| Dot product | unbounded | Cosine × magnitudes. **Identical to cosine when vectors are unit-normalised** — which is why normalised indexes use it: it's cheaper. |
| Euclidean (L2) | 0 … ∞ | Lower is better. Monotonically related to cosine *only* for normalised vectors: `d² = 2(1 − cos)`. |

That last row is the trap. FAISS `IndexFlatL2` returns **squared** L2 distance, and LangChain
passes it through unchanged as the "score." A threshold like `0.45` therefore means neither
"cosine 0.45" nor "distance 0.45" — and if the vectors aren't normalised it means nothing
transferable at all. The repo trips on this in `cache-augmented-rag`; see
[07-findings.md](07-findings.md#cache-threshold).

Text embeddings are near-uniformly positive-correlated, so real-world cosine scores cluster
in roughly 0.3–0.9. Treating 0.5 as "half relevant" is wrong — thresholds must be calibrated
per model against actual data.

## Embedding models

The repo uses two families and documents five open-source options:

| Model | Dims | Repo's note |
| --- | --- | --- |
| `all-MiniLM-L6-v2` | 384 | Fast, good quality — general purpose, real-time |
| `all-MiniLM-L12-v2` | 384 | Slightly better, bit slower |
| `all-mpnet-base-v2` | 768 | Best quality, slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Tuned for Q&A / asymmetric search |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ languages |
| `text-embedding-3-small` (OpenAI) | 1536, truncatable to 1024 | Used in the Pinecone notebook |
| CLIP `ViT-B/32` | 512 | Joint text+image space — [03-retrieval.md](03-retrieval.md#multimodal) |

Four things the table doesn't say and should:

**Local vs API is the latency decision, not the quality decision.** MiniLM runs in single-digit
milliseconds on CPU. `OpenAIEmbeddings` is a network round trip — 100–300 ms, and it is on
the **query path**, once per query, every query. For Vec's 200 ms budget that single call
would consume most of the budget on its own. This is the entire reason
[../02-architecture.md](../02-architecture.md) embeds locally.

**Symmetric vs asymmetric matters.** Most `all-*` models are trained for
sentence-to-sentence similarity. RAG is asymmetric: a short question against a long passage.
`multi-qa-*` and `e5`/`bge` families (which use `query:` / `passage:` prefixes) are trained
for that shape and typically retrieve better on the same corpus. The repo lists
`multi-qa-MiniLM-L6-cos-v1` and then never uses it.

**Dimensionality is a storage and speed decision.** 384 dims × 4 bytes = 1.5 KB per vector.
One million passages ≈ 1.5 GB at full precision, ≈ 380 MB under int8 scalar quantisation.
Doubling to 768 doubles both the memory and the per-comparison cost for a usually modest
recall gain.

**Never mix models across an index.** Vectors from different models are not comparable, even
at identical dimensionality. Re-embedding the whole corpus is the only way to change models.

## Vector stores

Five demonstrated, with meaningfully different operational profiles.

| Store | Type | Persistence | Where it fits |
| --- | --- | --- | --- |
| `InMemoryVectorStore` | Library | None | Tests and demos only |
| **FAISS** | Library, in-process | `save_local` / `load_local` to disk | Default here — fastest, no server, no network hop |
| **Chroma** | Embedded server | `persist_directory` + SQLite | Local dev with collections and metadata filters |
| **Pinecone** | Managed cloud | Managed | Scale and zero ops; every query is a network call |
| **AstraDB** | Managed cloud (Cassandra) | Managed | Same, for teams already on Cassandra |

The configurations the repo actually uses:

```python
# FAISS — in-process, persisted to disk
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("data/faiss_index")

# Chroma — named, persisted collection
Chroma.from_documents(docs, embedding,
    persist_directory="./chroma_db", collection_name="rag_collection")

# Pinecone — serverless, cosine, 1024-dim
pc.create_index(name=..., dimension=1024, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```

FAISS is the correct default for anything latency-sensitive: it is a library, not a service,
so a search is a function call rather than an RPC. That property is what makes a sub-200 ms
budget achievable at all.

### What the repo doesn't cover

- **Index type.** Every FAISS index here is `IndexFlat*` — exhaustive brute-force search,
  exact but O(n) per query. At ~1M vectors that stops being instant. The production answer
  is an approximate index (HNSW, IVF-PQ) which trades a few points of recall for orders of
  magnitude of speed. Chroma and Pinecone use HNSW internally; the repo never surfaces the
  `M` / `ef_construction` / `ef_search` knobs.
- **Quantisation.** Scalar (int8) or product quantisation cuts memory ~4× with a small
  recall cost. Not mentioned.
- **Hybrid-native stores.** Qdrant, Weaviate, and Elasticsearch can store sparse and dense
  vectors in one index and fuse server-side. The repo's hybrid search is instead assembled
  client-side from FAISS + BM25 — see [03-retrieval.md](03-retrieval.md#hybrid).
- **Metadata filter cost.** `similarity_search(..., filter={...})` is shown for FAISS, but
  not the fact that pre-filtering and post-filtering behave very differently: post-filtering
  a top-*k* can return fewer than *k* results, or none.

## Practical checklist

1. **Pick the model before the store.** The model sets dimensionality, the distance metric
   that makes sense, and the recall ceiling. The store is swappable; the model is not.
2. **Normalise, then use dot product.** It makes distances interpretable and comparisons
   cheaper, and turns any L2 threshold into a real cosine threshold.
3. **Calibrate thresholds against your own data.** A number that works for MiniLM will not
   work for OpenAI embeddings.
4. **Budget memory before indexing.** `n_vectors × dims × 4 bytes`, times the number of
   chunking strategies you index in parallel.
5. **Local model + in-process index if latency is the constraint.** Every managed store adds
   a network round trip to every query.

Next: [03-retrieval.md](03-retrieval.md).
