# 07 — Findings

Concrete defects found while reading
[`Hitesh-s0lanki/agentic-rag`](https://github.com/Hitesh-s0lanki/agentic-rag), and what is
worth porting into Vec.

Everything here was verified against source — either the repo's own files or the upstream
LangChain source it calls. Nothing was executed; the notebooks need API keys. Where a claim
depends on runtime behaviour I say what it was verified against.

---

## Security {#security}

### Live Neo4j credentials committed to a public repo

`graphdb/experiments.ipynb` and `graphdb/promptstatergies.ipynb` both hardcode a Neo4j Aura
connection in plain text — URI, username, and password — as the first code cell:

```python
NEO4J_URI      = "neo4j+s://427c9a4f.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "rQJMGc6on…"          # full value is in the committed notebooks
```

The two files carry slightly different strings for both URI and password, so at least one
may be stale or mistyped. That does not reduce the exposure — the repo is public, and the
values are in the committed source and in every clone.

**These are the only two files in the repo that hardcode credentials.** Every other module
correctly uses `os.getenv` with `python-dotenv`, and no `.env` file is committed.

What to do, in order:

1. **Rotate the Neo4j password now.** Assume it is compromised; public-repo credentials are
   scraped within minutes by automated crawlers.
2. Check the Aura instance for unexpected queries or data changes.
3. Move both notebooks to `os.getenv`, matching the rest of the repo.
4. **Purge from history.** Deleting the lines in a new commit is not enough — the values
   remain in every prior object and in existing clones. Use `git filter-repo` or the BFG,
   then force-push. Given the repo is a learning archive with a linear history, the simpler
   option is worth considering: rotate, fix forward, and accept that the old value is burned.

Rotation is the step that actually matters. History purging without rotation is theatre.

---

## Correctness bugs

### 1. Hybrid retriever weights are silently ignored {#ensemble-weights}

`hybrid-search-strategies/src/dense_sparse.ipynb`:

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weight=[0.7, 0.3]        # ← field is `weights`
)
```

The field is `weights`. Verified in `langchain_classic/retrievers/ensemble.py`:

```python
weights: list[float]

@model_validator(mode="before")
def _set_weights(cls, values):
    weights = values.get("weights")
    if not weights:
        values["weights"] = [1 / n_retrievers] * n_retrievers
```

`BaseRetriever.model_config` sets only `arbitrary_types_allowed=True` — no
`extra="forbid"` — so under Pydantic v2's default (`extra="ignore"`) the stray `weight=`
kwarg is dropped without error or warning. `weights` is then unset, and the validator
substitutes equal weighting.

**Effect:** the notebook runs, returns plausible results, and applies `[0.5, 0.5]` instead of
the intended `[0.7, 0.3]`. A silent config bug of exactly the kind that survives code review.

**Fix:** `weights=[0.7, 0.3]`.

### 2. `NameError` on the adaptive-RAG failure path {#adaptive-loop}

`adaptive-rag/src/adaptive_rag.ipynb`, in `grade_generation_v_documents_and_question`:

```python
else:
    pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"
```

`pprint` is never imported anywhere in the notebook — every other branch in the same function
uses bare `print`. Any generation the hallucination grader rejects raises `NameError` instead
of returning `"not supported"`.

This hides a second problem. The edge map routes `"not supported"` back to `generate`:

```python
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "not supported": "generate",     # ← self-loop, no iteration cap
    "useful": END,
    "not useful": "transform_query",
})
```

There is no attempt counter on this cycle. Fixing the `NameError` alone converts a crash into
an unbounded `generate → generate` loop, bounded only by LangGraph's default recursion limit.

**Fix:** use `print`, *and* add an attempt counter to `GraphState` with a cap in the edge
condition — the pattern `autonomus-rag` already uses (`attempts >= 2`).

### 3. Re-ranker index parsing is off by one {#rerank-indexing}

`hybrid-search-strategies/src/re_ranking.ipynb`. Documents are numbered **1-based** in the
prompt:

```python
doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
```

The prompt's own example is **0-based**:

```
Output format: comma-separated document indices (e.g., 2,1,3,0,...)
```

And parsing subtracts one:

```python
indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
```

If the model follows the example and emits `0`, it becomes `-1`. The guard `0 <= i` catches
that particular value — but only by **dropping the document entirely**, so a reranked list
silently loses an item. If the model follows the numbering instead, everything works. The
model is being given contradictory instructions and the outcome depends on which one it
obeys.

**Fix:** make the example 1-based (`e.g., 3,1,2,4`) to match the numbering.

### 4. Conflicting edges in the iterative-retrieval graph

`autonomus-rag/src/iterative_retrieval.ipynb`:

```python
builder.add_edge("answer", "reflect")
...
builder.add_edge("answer", END)          # ← also from "answer"
```

Two unconditional edges leave `answer`: one into the reflection cycle, one straight to `END`.
LangGraph treats multiple outgoing edges as a fan-out, so both fire — the reflection loop
runs *and* the graph is simultaneously told it is finished. The `refine → retrieve → answer`
cycle then re-triggers the `END` edge on every pass.

The `add_edge("answer", END)` line is almost certainly leftover scaffolding from before the
reflection loop was added.

**Fix:** delete `builder.add_edge("answer", END)`. Termination is already handled by the
conditional edge on `reflect`.

### 5. Double retrieval in the e2e project {#e2e-double-retrieval}

`e2e-project/src/node/reactnode.py`. The graph is `retriever → responder`:

```python
def retrieve_docs(self, state):
    docs = self.retriever.invoke(state.question)      # retrieval #1
    return RAGState(question=state.question, retrieved_docs=docs)

def generate_answer(self, state):
    result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
    #        ↑ ReAct agent with its own retriever tool → retrieval #2
    return RAGState(..., retrieved_docs=state.retrieved_docs, answer=answer)
```

`generate_answer` never reads `state.retrieved_docs`. It hands the raw question to a ReAct
agent that retrieves again through its own tool. The first retrieval's results are carried
through the state, returned to the caller, displayed as "sources" in the Streamlit UI — and
have **no causal relationship to the answer**.

That last part is the real problem: the UI shows sources that did not produce the answer.
This is a citation-integrity bug, not just wasted work.

**Fix:** pick one. Either drop the `retriever` node and let the agent own retrieval (then
surface the agent's actual tool calls as sources), or drop the agent and generate from
`state.retrieved_docs`.

---

## Mislabelled techniques

### CAG is implemented as semantic response caching {#cag-mislabel}

`cache-augmented-rag/src/cache_augment_generation.ipynb` opens with a correct definition:

> CAG is a retrieval-free approach … preloads relevant documents into the LLM's extended
> context window, precomputes the model's key-value (KV) cache, and reuses this during
> inference

The code implements neither half of that. There is (a) a Python dict keyed on the exact query
string, and (b) a FAISS index over previous *questions*, returning the stored answer when a
new question is close enough. That is **semantic response caching** — a different technique,
at a different layer, solving a different problem.

| | Real CAG | What's implemented |
| --- | --- | --- |
| Caches | KV tensors for the corpus | Question → answer pairs |
| Eliminates | Retrieval, for every query | Retrieval **and** generation, for repeat queries |
| Works when | Corpus fits the context window | Queries repeat |
| Cold-start | Effective immediately | Empty — no benefit until traffic accumulates |

Both are worth knowing. The notebook teaches one and names it the other.

### Cache threshold is not a similarity score {#cache-threshold}

```python
CACHE_DISTANCE_THRESHOLD = 0.45
hits = QA_CACHE.similarity_search_with_score(q, k=CACHE_TOP_K)
if dist <= CACHE_DISTANCE_THRESHOLD: ...
```

`QA_CACHE` is built on `faiss.IndexFlatL2`, which returns **squared L2 distance**, and
LangChain passes that through unchanged as `score`. Meanwhile `HuggingFaceEmbeddings` does
not normalise by default, so the vectors are not unit-length either.

So `0.45` is a squared L2 distance over un-normalised MiniLM vectors. It is not a cosine
similarity, it has no intuitive interpretation, and it will not transfer if the embedding
model changes. It was presumably found by trial.

The failure mode is asymmetric and user-visible: too loose, and a user gets a confident
answer to a question they did not ask, served from cache with no retrieval to correct it.
That reads as a caching win in the logs and a hallucination to the user.

**Fix:** normalise the embeddings (`encode_kwargs={"normalize_embeddings": True}`), use
`IndexFlatIP`, and set the threshold as a real cosine value — then calibrate it against
labelled question pairs rather than by eye.

---

## Structural gaps

| Gap | Detail |
| --- | --- |
| **`multi-agent-rag/` is empty** | `requirements.txt` and `command.md` only. The top-level README advertises "collaborative agent systems" and "specialized agent roles." |
| **No evaluation anywhere** | No recall@k, no MRR, no faithfulness scoring, no LLM judge, no golden set. Every technique is demonstrated on one query and declared to work. This is the single biggest gap — with 16 modules of alternatives and no way to compare them, technique selection is guesswork. |
| **No latency measurement** | Only the CAG notebook times anything, with `time.time()` around one call. |
| **No cross-encoder reranker** | The standard, fast reranker is absent; the slowest option (LLM) is the only one shown. |
| **No parent-document retrieval** | The standard fix for the chunk-size trade-off. |
| **Sequential fan-out** | `answer_synthesis.ipynb` runs four independent retrievals in series — latency is the sum, should be the max. |
| **Free-text parsing for control flow** | Decomposition and reflection split LLM prose on newlines and substring-match `"yes"`, while the same repo uses `with_structured_output` correctly in `adaptive-rag`. |
| **No timeouts, retries, or tracing** | No node-level timeout; one hung Tavily call hangs the graph. No LangSmith or callbacks, which makes multi-judge graphs effectively undebuggable. |
| **`binary_score: str`** | Should be `Literal["yes", "no"]`. As typed, `"Yes"` passes validation and fails every `== "yes"` comparison. |
| **CLIP truncation unflagged** | 500-character chunks are embedded through CLIP's 77-token text encoder and silently truncated. |
| **Empty module READMEs** | 8 of the 11 module READMEs are 0 bytes. |
| **`requirements copy.txt`** | Stray duplicate in `adaptive-rag/`. |

---

## What to port into Vec {#what-to-port-into-vec}

Vec's constraint is the inverse of this repo's: the default answer path allows **zero network
calls after the transcript arrives**, targeting under 200 ms end to end
([../04-latency.md](../04-latency.md)). Almost everything above costs at least one LLM round
trip. So the question is not "which technique is best" but "which fit in the budget, and
which belong on the escalation ladder."

### Free — index-time, costs nothing at query time

| Technique | Why |
| --- | --- |
| **BM25 alongside dense** | Pure data structure, no model. Covers the exact-token failures dense retrieval has — names, numerals, codes. Fuse with RRF. Fits the budget outright. |
| **Unicode NFKC normalisation** | Microseconds. Fixes ligature and composition mismatches between STT output and indexed passages. Both notebooks' `ﬁ → fi` hack, done properly. |
| **Materialised relationships** | The SQL-JOIN-as-document idea from `database_parsing.ipynb`. Anything you want retrievable must exist as text at index time. |
| **Parent-document retrieval** | Embed small, return big. Directly relevant to comparing chunking strategies in [../03-chunking.md](../03-chunking.md). |
| **Typed metadata at chunk time** | Enables pre-filtering by `query_type`, language, or `query_id` at zero model cost. |

### Cheap — tens of milliseconds, plausibly inside the budget

| Technique | Cost | Note |
| --- | --- | --- |
| **MMR** | Negligible, runs over the fetched candidates | Worth it if the chunking strategies produce overlapping passages, which they will. |
| **Cross-encoder reranking** | ~10–50 ms local for k≈20 | **The highest-value item on this list.** Local `ms-marco-MiniLM`-class model, no network. Measure it against the budget before committing — it may be the single best accuracy-per-millisecond trade available. |
| **Semantic answer cache** | One local embedding + one search | The idea from `cache-augmented-rag`, fixed per [above](#cache-threshold): normalised vectors, `IndexFlatIP`, calibrated cosine threshold. Turns repeat questions into sub-50 ms answers. |

### Escalation ladder — one round trip or more, needs explicit opt-in

The four levels in `EffortPanel` are already the right shape. A defensible mapping:

| Level | Technique | Source module |
| --- | --- | --- |
| **1 — Fast** | Local embed → hybrid search → extractive answer. No LLM. | — |
| **2 — Better** | Query rewriting as a **repair only**, triggered when retrieval scores below threshold | `corrective-rag` |
| **3 — Thorough** | HyDE or expansion; relevance + grounding graders | `query-enhancement`, `adaptive-rag` |
| **4 — Exhaustive** | Decomposition, iterative retrieval, web fallback | `autonomus-rag`, `corrective-rag` |

Two structural lessons transfer regardless of level:

**Grade-and-branch is the guardrail primitive.** The typed binary graders in `adaptive-rag`
— relevance, grounding, usefulness — are exactly the mechanism
[../06-guardrails.md](../06-guardrails.md) needs for "knowing when not to answer." Vec has an
advantage this repo lacks: MSMARCO-XI's `is_selected` and `"No Answer Present."` labels mean
those graders can be **measured** against ground truth rather than trusted
([../07-evaluation.md](../07-evaluation.md)). That turns a demo pattern into a reported
number.

**Separate "ungrounded" from "off-target."** `adaptive-rag`'s three-way generation verdict
(`not supported` / `not useful` / `useful`) drives three different repairs. Collapsing them
into one "bad answer" signal means half the retries attack the wrong failure.

### What not to port

- **Anything on the query path that calls an API.** Query expansion, HyDE, LLM reranking,
  routing — each is a network round trip before retrieval starts, and any one of them
  exceeds the whole budget.
- **Translate-then-retrieve.** Already ruled out for the same reason in
  [../01-dataset.md](../01-dataset.md); the parallel English is an index-time and eval-time
  asset.
- **LLM reranking.** Use a local cross-encoder instead — same job, 20–100× faster.
- **Unbounded reflection loops.** If a reflection cycle ships at all, it ships with a counter.
