# 04 — Query enhancement

Source module:
[`query-enhancement/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/query-enhancement) ·
plus the rewriters in `adaptive-rag/` and `corrective-rag/`

Everything so far assumed the user's query is a good search key. It usually isn't. The user
writes a question; the corpus contains answers. Those two texts are written in different
registers, at different lengths, using different vocabulary — the **semantic gap**. Query
enhancement is the family of techniques that rewrites the query into something closer to what
the corpus actually looks like.

Every technique here costs at least one LLM call **on the query path**. That is the tax, and
it is what puts all of them outside Vec's default 200 ms budget — see
[../04-latency.md](../04-latency.md).

---

## Query expansion

```python
"""Expand the following query to improve document retrieval by adding relevant
synonyms, technical terms, and useful context.

Original query: "{query}"
Expanded query:"""
```

Widen the query with synonyms and domain vocabulary. `"Langchain memory"` becomes a
paragraph mentioning conversation buffers, state persistence, `ConversationBufferMemory`, and
so on — which overlaps far more corpus vocabulary than two words ever could.

Helps most with **short queries**, which is exactly the voice case: spoken questions are
terse and lack the technical terms the documents use.

Risk: expansion drifts. Add enough loosely-related terms and the embedding migrates away from
the user's actual intent, pulling in confidently-wrong neighbours.

## Query decomposition

```python
"""Decompose the following complex question into 2 to 4 smaller sub-questions
for better document retrieval."""
```

Split a multi-part question into atomic ones, retrieve for each, answer each, combine.

```
"How does LangChain use memory and agents compared to CrewAI?"
   → "What memory does LangChain provide?"
   → "How does LangChain implement agents?"
   → "How does CrewAI handle memory and agents?"
```

The necessity here is structural, not stylistic. A compound question embeds to a point
*between* its topics — often near nothing at all, since the average of "LangChain memory" and
"CrewAI agents" may sit in empty space. Each sub-question embeds to a dense, well-populated
region. This is also what makes **multi-hop** answering possible, and it parallelises: the
sub-questions are independent.

The repo implements it twice — as an LCEL chain in `query_decomposition.ipynb`, and as a
LangGraph planner node in
[`autonomus-rag/src/query_planning_decomposition.ipynb`](https://github.com/Hitesh-s0lanki/agentic-rag/blob/main/autonomus-rag/src/query_planning_decomposition.ipynb).
Both parse sub-questions by splitting the LLM's free text on newlines and stripping bullet
characters:

```python
sub_questions = [q.strip("-•1234567890. ").strip() for q in text.split("\n") if q.strip()]
```

That is fragile. Any preamble ("Sure! Here are three sub-questions:") becomes a
sub-question and gets retrieved against. `with_structured_output` — which the same repo uses
correctly for its graders in `adaptive-rag` — is the fix, and it is one line.

## HyDE — Hypothetical Document Embeddings

The most counter-intuitive technique here, and the sharpest.

```python
template = """Imagine you are an expert writing a detailed explanation on the topic:
'{query}'. Create a hypothetical answer for the topic"""

matched_doc = base_retriever.invoke(get_hyde_doc(query))   # search with the ANSWER
```

Ask the LLM to **hallucinate an answer**, then embed that answer and search with it —
discarding the query entirely.

The reason it works: you are searching a corpus of *answers* with an *answer*, instead of
with a question. `"When was Steve Jobs fired from Apple?"` and a Wikipedia paragraph about
the 1985 power struggle share almost no vocabulary. A fabricated paragraph about Jobs, Apple,
1985, and John Sculley shares nearly all of it. Answer-to-answer is a symmetric comparison;
question-to-answer is asymmetric and harder.

**The hallucination is not a bug.** Factual errors in the hypothetical document are largely
irrelevant — the vector only needs to land in the right *neighbourhood*. Retrieval then
returns real passages, and the real passages are what the answer is generated from. The
fake document is a search key, never context.

The repo shows both forms:

| Form | Mechanism |
| --- | --- |
| Manual | Generate hypothetical text per query, embed it, search. Query-time cost only. |
| `HypotheticalDocumentEmbedder` | Wraps the embedding function itself; the vector store is *built* through it. |

The second is worth a warning the notebook doesn't give. Passing a HyDE embedder to
`Chroma.from_documents(...)` means every **document** is also run through the hypothetical
generator at index time — an LLM call per chunk, embedding a hallucinated paraphrase of your
own corpus rather than the corpus. That is almost certainly not intended. HyDE belongs on the
query side; `prompt_key="web_search"` is a query-shaped prompt.

Where HyDE earns its cost: short queries, jargon mismatch between users and corpus, and
cross-lingual retrieval — you can generate the hypothetical document *in the corpus
language*, which sidesteps a translation call.

## Query rewriting

```python
"""You are a question re-writer that converts an input question to a better version
that is optimized for vectorstore retrieval. Look at the input and try to reason
about the underlying semantic intent / meaning."""
```

Rewrite for the target retriever. Used as a **repair** step rather than a first move — in
both `adaptive-rag` and `corrective-rag` the rewriter fires only after the relevance grader
rejects everything retrieved, then loops back:

```
retrieve → grade → (all rejected) → transform_query → retrieve again
```

The two modules differ in one instructive detail. `adaptive-rag` optimises the rewrite
**for the vectorstore** and retries retrieval; `corrective-rag` optimises **for web search**
and routes to Tavily. Same node, different target, different downstream — the prompt encodes
which retriever the query is being tuned for.

## Query routing

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...)

structured_llm_router = llm.with_structured_output(RouteQuery)
```

Not a rewrite — a **classification**. Decide *where* the query should go before spending
anything on retrieval. The router prompt describes what the vectorstore contains ("agents,
prompt engineering, adversarial attacks") and everything outside that goes to web search.

This is the entry node of Adaptive RAG ([05-rag-architectures.md](05-rag-architectures.md#adaptive))
and the one place in the repo where `with_structured_output` is used properly — a typed
Pydantic model, a constrained `Literal`, no string parsing. Note the contrast with the
newline-splitting in decomposition above: the same repo does it right in one place and wrong
in another.

Routing generalises well beyond two destinations: per-domain indexes, SQL vs vector, cheap
model vs expensive model, or straight to an answer with no retrieval at all when the question
is conversational.

---

## Not in this repo

**Step-back prompting.** Generate a more *general* question first ("What is the physics of
projectile motion?" before "What's the trajectory of a ball thrown at 30° at 20 m/s?"),
retrieve for both. Grounds specifics in principles. The inverse of decomposition.

**Multi-query retrieval.** N paraphrases of the same question, retrieve for each, union and
dedupe. Cheaper than decomposition (one call, N outputs) and directly targets phrasing
variance. LangChain ships `MultiQueryRetriever`.

**RAG-Fusion.** Multi-query plus RRF over the per-query result lists, rather than a plain
union. Passages that rank well across several phrasings float to the top.

**Query classification / intent detection.** Route by *type* — factual, comparative,
summarisation, conversational — and pick a different pipeline per type. `query_type` in
Vec's MSMARCO-XI data (`DESCRIPTION` / `NUMERIC` / `ENTITY` / …) is exactly this label,
already provided ([../01-dataset.md](../01-dataset.md)).

**HyDE with multiple hypotheses.** Generate k hypothetical documents and average their
embeddings, reducing variance from any one hallucination.

---

## Cost, and why it matters for Vec

| Technique | LLM calls on query path | Added latency (typical) |
| --- | --- | --- |
| Routing | 1 (small model, short output) | 200–500 ms |
| Expansion | 1 | 300–800 ms |
| HyDE | 1 (long output — a whole paragraph) | 800 ms – 2 s |
| Decomposition | 1 + N retrievals + N answers | 2–10 s |
| Rewriting (repair) | 1, only on failure | 300–800 ms, conditionally |

None of these fit a 200 ms budget. Every one of them is a **network round trip before
retrieval even begins**, which is the specific thing
[../02-architecture.md](../02-architecture.md) removes from Vec's default path.

That makes them escalation-ladder material, not default-path material:

- **Level 1 (fast):** no enhancement. Embed the transcript locally, search, answer.
- **Level 2:** query rewriting, but only as a *repair* after retrieval scores low — so the
  fast path stays fast and only failures pay.
- **Level 3:** HyDE or expansion, for queries the fast path abstains on.
- **Level 4:** decomposition and multi-hop, for genuinely compound questions.

One free option worth separating out: **normalisation is not enhancement.** Lowercasing,
punctuation stripping, and Unicode NFKC take microseconds, need no model, and fix a
meaningful slice of the mismatch — particularly for STT output, which arrives with
inconsistent casing and punctuation. That belongs on the default path.

Next: [05-rag-architectures.md](05-rag-architectures.md).
