# 05 — Types of RAG

The taxonomy. What each variant is, what problem it exists to solve, what it costs, and
which ones this repo actually builds.

## The generational frame

Most RAG variants are one answer to a single question: **who decides what happens next?**

| Generation | Control | Shape |
| --- | --- | --- |
| **Naive** | Nobody — fixed | `retrieve → generate` |
| **Advanced** | Fixed, but with pre/post stages | `enhance → retrieve → rerank → generate` |
| **Modular** | Configured routing | Swappable components, branching by config |
| **Agentic** | **The model** | Cycles, tool choice, self-assessment, retries |

The jump that matters is the last one. In naive/advanced/modular RAG the control flow is
written by you and fixed at build time. In agentic RAG the model decides — whether to
retrieve, what to retrieve, whether the result is good enough, and whether to try again.
That is the actual definition of "agentic," and it is why every agentic variant below has a
**cycle** in its graph while the earlier ones are straight lines.

---

## 1. Naive RAG

```
query → embed → search → stuff into prompt → generate
```

**In this repo:**
[`rag-implementation/src/basic_arag.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/rag-implementation/src/basic_arag.ipynb)
and `e2e-project/src/graph_builder/` — two LangGraph nodes, `retriever → responder → END`.

Every failure mode of RAG is visible here: retrieves even when retrieval is pointless,
retrieves the wrong thing without noticing, answers from bad context anyway, and never
abstains. It is also fast, cheap, predictable, and correct for a large share of real
questions. Do not skip building it — it is the baseline every other variant must beat, and
without it you cannot tell whether your reranker earned its latency.

## 2. Advanced RAG

Naive plus fixed pre- and post-retrieval stages: query enhancement in front, reranking and
MMR behind.

**In this repo:** `hybrid-search-strategies/` and `query-enhancement/` — hybrid retrieval,
MMR, LLM reranking, expansion, decomposition, HyDE. All demonstrated individually; never
composed into one pipeline.

Still a straight line. Nothing decides anything; every query pays for every stage.

## 3. Modular RAG

Interchangeable components with configured routing — different indexes per domain, different
retrievers per query type, swappable generators. The organising idea is that retrieval and
generation are *modules*, not a fixed chain.

**In this repo:** partially. `e2e-project/` separates config, ingestion, vector store,
nodes, and graph builder into a proper package. But it exposes one path, so it is modular in
its code structure rather than in its behaviour.

## 4. Agentic RAG

The model drives. Retrieval becomes a **tool** the model may call zero, one, or many times,
with arguments it chooses.

**In this repo:**
[`rag-implementation/src/reAct_rag_tool.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/rag-implementation/src/reAct_rag_tool.ipynb)

```python
tools = [wiki_tool, arxiv_tool, internal_tool_1, internal_tool_2]
react_node = create_react_agent(llm, tools)
```

Four retrieval tools — Wikipedia, ArXiv, and two separate internal corpora — and the agent
picks. Ask *"What do our internal research notes say about transformer variants, and what does
ArXiv suggest recently?"* and it calls two different tools, unprompted, because the tool
**descriptions** told it what each one is for.

Tool descriptions are the actual programming surface here. `"Search internal research notes
for experimental results and agent designs"` is not documentation — it is the routing logic.
A vague description is a routing bug.

Mechanics in [06-agentic-patterns.md](06-agentic-patterns.md).

---

## Variants by what they fix

The four above are a progression. The rest are orthogonal — each targets one specific failure.

### 5. Corrective RAG (CRAG) — *fixes: retrieval returned garbage*

**In this repo:**
[`corrective-rag/src/corrective_rag.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/corrective-rag/src/corrective_rag.ipynb)

```
retrieve → grade each doc → any irrelevant? → rewrite query → web search → generate
                          ↘ all relevant  → generate
```

An LLM grader scores every retrieved document `yes`/`no` for relevance. If **any** document
fails, the query is rewritten for web search, Tavily results are appended, and generation
proceeds on the combined set.

The insight: *the corpus may simply not contain the answer*, and no amount of better
retrieval fixes that. The escape hatch is an external source.

The repo's trigger is aggressive — one bad document out of four flips `web_search = "Yes"`,
even though three good ones remain. That fires the expensive path constantly. The paper's
formulation grades **confidence over the whole retrieval** into correct / ambiguous /
incorrect, with web search reserved for the last. A threshold (say, fewer than 2 relevant
docs) would be closer to intent.

### 6. Adaptive RAG — *fixes: not every query needs the same pipeline* {#adaptive}

**In this repo:**
[`adaptive-rag/src/adaptive_rag.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/adaptive-rag/src/adaptive_rag.ipynb)
— the most complete architecture in the repo.

```
        ┌─ router ─→ web_search ────────────────→ generate ─┐
START ──┤                                                    ├→ grade generation
        └─ router ─→ retrieve → grade docs → generate ──────┘        │
                          ↑                                          │
                          └──── transform_query ←── not useful ──────┤
                                                                     │
                              generate ←──── not supported ──────────┘
                                                    useful → END
```

Five nodes, four LLM judges, three cycles:

| Component | Decides |
| --- | --- |
| **Router** | vectorstore vs web search — *before* any retrieval |
| **Retrieval grader** | is each document relevant? |
| **Hallucination grader** | is the generation grounded in the documents? |
| **Answer grader** | does the generation actually address the question? |
| **Rewriter** | produce a better query and retry |

The three outcomes of the generation check drive different repairs, and that separation is
the good idea here:

- `not supported` → the answer is ungrounded → **regenerate** (same context, try again)
- `not useful` → grounded but off-target → **rewrite the query and re-retrieve**
- `useful` → done

Distinguishing "hallucinated" from "irrelevant" is what lets it pick the right fix.
Diagnosing them as one failure — which most implementations do — means half the retries
attack the wrong problem.

Every grader uses `with_structured_output` with a typed Pydantic model. That is the correct
pattern and it is used consistently throughout this notebook.

Cost: 4–8 LLM calls per query, unbounded. **`not supported` is wired `generate → generate`
with no iteration cap.** It never actually loops today only because that branch hits an
undefined `pprint` and crashes first. See [07-findings.md](07-findings.md#adaptive-loop).

### 7. Self-RAG — *fixes: the model can't tell when it's wrong*

Self-reflection as a first-class stage: generate, critique your own output, revise.

**In this repo:**
[`autonomus-rag/src/self_reflection.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/autonomus-rag/src/self_reflection.ipynb)
and `iterative_retrieval.ipynb`.

```python
builder.add_conditional_edges(
    "reflector",
    lambda s: "done" if not s.revised or s.attempts >= 2 else "retriever"
)
```

Note `attempts >= 2` — a real iteration cap, which `adaptive-rag` lacks. Self-critique loops
**must** be bounded; the model has no reliable sense of when it is converging.

The repo's version prompts for free text and greedily matches `"reflection: yes"` in the
lowercased response. Any other phrasing reads as failure and burns another iteration. The
canonical Self-RAG paper uses trained **reflection tokens** (`ISREL`, `ISSUP`, `ISUSE`);
`with_structured_output` is the practical substitute and the same repo already knows how.

### 8. Iterative / multi-hop RAG — *fixes: one retrieval isn't enough*

Retrieve, answer, notice the answer is incomplete, refine the query, retrieve again — using
what you learned from round *n* to search better in round *n+1*.

**In this repo:** `iterative_retrieval.ipynb`, capped at 2 attempts.

```
retrieve → answer → reflect → (insufficient) → refine query → retrieve → …
```

The distinction from decomposition: decomposition splits the question **up front** into
independent parts; iterative retrieval discovers what to ask next **from what it just
found**. Genuinely sequential dependencies — "who succeeded the person who founded X?" —
need the iterative form.

### 9. Cache-Augmented Generation (CAG) — *fixes: paying full price for repeat questions*

**In this repo:**
[`cache-augmented-rag/src/cache_augment_generation.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/cache-augmented-rag/src/cache_augment_generation.ipynb)

Two implementations:

```python
# 1. Exact-match dict cache — keyed by literal query string
if Model_Cache.get(query): return Model_Cache[query]

# 2. Semantic answer cache — FAISS over past questions
hits = QA_CACHE.similarity_search_with_score(q, k=3)
if dist <= 0.45: return best_doc.metadata["answer"]
```

The semantic version is the useful one: `"What is LangGraph?"` and `"Explain about
LangGraph"` are different strings but nearby vectors, so the second is served from the first
one's answer at embedding cost only — no retrieval, no generation.

**The module's own definition contradicts its code.** Its markdown describes CAG correctly
as preloading documents into an extended context window and reusing the precomputed **KV
cache** — a decoder-level technique that eliminates retrieval entirely for small corpora.
What is implemented is **semantic response caching**, a different technique at a different
layer. Both are legitimate; they are not the same thing, and the notebook labels one as the
other. See [07-findings.md](07-findings.md#cag-mislabel).

The `0.45` threshold is also uncalibrated — it is a raw (squared) FAISS L2 distance over
un-normalised MiniLM vectors, not a cosine score, and it does not transfer across embedding
models. Set it too loose and users get answers to questions they didn't ask, which is a
correctness bug that presents as a caching win.

Real CAG — preload the whole corpus into context, precompute KV, never retrieve — is
genuinely attractive when the corpus fits in a context window. It is not what this notebook
does.

### 10. Graph RAG — *fixes: relationships and aggregates are unretrievable*

**In this repo:** `graphdb/` — Neo4j + `GraphCypherQAChain`. Covered in
[03-retrieval.md](03-retrieval.md#graph-retrieval).

The variant the repo does *not* show is the more powerful one: **LLM-extracted knowledge
graphs** (Microsoft's GraphRAG). Build a graph from unstructured text by extracting entities
and relations, cluster it into communities, summarise each community, and answer global
questions ("what are the main themes?") from the summaries. That class of question is
unanswerable by chunk retrieval — no single chunk contains a corpus-wide theme.

### 11. Multimodal RAG — *fixes: the answer is in a figure*

**In this repo:** `multi-model-openai/` — CLIP unified index + GPT-4V generation. Covered in
[03-retrieval.md](03-retrieval.md#multimodal).

### 12. Multi-agent RAG — *fixes: one agent's context can't hold the whole job*

Specialised agents — researcher, retriever, critic, writer — coordinated by a supervisor.
Justified when subtasks need genuinely different tools or prompts, or when parallel breadth
matters more than a single coherent context.

**In this repo: not implemented.** `multi-agent-rag/` contains `requirements.txt` and
`command.md` and no source, despite the top-level README advertising "collaborative agent
systems" and "specialized agent roles."

The nearest thing present is
[`answer_synthesis.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/autonomus-rag/src/answer_synthesis.ipynb),
which fans out across four sources (internal docs, YouTube transcript, Wikipedia, ArXiv) and
synthesises one answer. But its nodes run **sequentially** —
`text → yt → wiki → arxiv → synthesize` — as a chain of `add_edge` calls. These four
retrievals are fully independent and should run concurrently; as written, total latency is
the sum rather than the max. LangGraph supports parallel fan-out by adding multiple edges
from one node.

---

## Comparison

| Variant | LLM calls/query | Latency | Bounded? | Fixes |
| --- | --- | --- | --- | --- |
| Naive | 1 | ~1 s | ✅ | — |
| Advanced (rerank) | 1–2 | 1–3 s | ✅ | Ranking quality |
| Agentic (ReAct) | 2–10 | 3–20 s | recursion limit | Tool/source selection |
| Corrective | 3–6 | 3–10 s | ✅ | Bad or missing context |
| Adaptive | 4–8 | 4–15 s | ❌ in this repo | Query-appropriate routing |
| Self-RAG | 3–6 | 3–10 s | ✅ (2 attempts) | Ungrounded answers |
| Iterative | 4–10 | 5–20 s | ✅ (2 attempts) | Multi-hop |
| CAG (cache hit) | **0** | <100 ms | ✅ | Repeat cost |
| Graph | 2 | 1–3 s | ✅ | Relations, aggregates |
| Multimodal | 1 (vision) | 2–5 s | ✅ | Figures, images |
| Multi-agent | 10–50 | 10–60 s | varies | Breadth, specialisation |

Latency figures are rough orders of magnitude for typical API models — the repo measures
nothing, so treat these as design intuitions, not results.

## Choosing

Escalate only when a measured failure demands it:

1. **Naive** first. Measure recall@k and answer quality. Many corpora need nothing more.
2. Retrieval quality bad → **advanced** (hybrid + rerank). Index-time and cheap.
3. Corpus sometimes lacks the answer → **corrective** with a web fallback.
4. Query types genuinely differ → **adaptive** routing.
5. Hallucinations persist → **self-RAG** grounding checks, capped.
6. Questions are multi-hop → **iterative** or decomposition.
7. Repeat traffic is high → **semantic caching** in front of everything.
8. Relational/aggregate questions → **graph**.
9. Answers live in figures → **multimodal**.

The ordering is deliberate: each step buys accuracy with latency and cost. In a
latency-bound system like Vec, most of this ladder is unaffordable by default and belongs
behind an explicit effort control — which is what
[07-findings.md](07-findings.md#what-to-port-into-vec) maps out.

Next: [06-agentic-patterns.md](06-agentic-patterns.md).
