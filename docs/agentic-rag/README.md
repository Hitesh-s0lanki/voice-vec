# Agentic RAG — technique analysis

An analysis of [`Hitesh-s0lanki/agentic-rag`](https://github.com/Hitesh-s0lanki/agentic-rag):
what it actually implements, how each technique works, and where each one sits in the
wider RAG landscape.

Read at commit `HEAD` of `main`, 2026-08-17. Every claim about the repo below comes from
reading the notebook source, not the module READMEs — the READMEs oversell in several
places and those gaps are recorded in [07-findings.md](07-findings.md).

## What this repo is

Sixteen self-contained learning modules, ~43 notebooks plus one packaged Streamlit app.
The stack is uniform across all of them:

| Layer | Choice |
| --- | --- |
| Framework | LangChain + LangGraph |
| LLM | OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `o4-mini`) and Groq (`gemma2-9b-it`) |
| Embeddings | `OpenAIEmbeddings` and `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS by default; Chroma, Pinecone, AstraDB, `InMemoryVectorStore` demonstrated |
| Web search | Tavily |
| Graph store | Neo4j Aura + `GraphCypherQAChain` |

It is a **breadth** repo. It covers a lot of the RAG surface, at demo depth: no evaluation
harness, no benchmark numbers, no latency measurement, no test suite. Judge it as a
technique catalogue and it is a good one. Judge it as a system and the gaps in
[07-findings.md](07-findings.md) matter.

## Document map

| # | Document | Covers |
| --- | --- | --- |
| 01 | [Chunking](01-chunking.md) | Fixed, recursive, token, semantic, document-aware, structured; the 5 the repo has and the 6 it doesn't |
| 02 | [Embeddings & vector stores](02-embeddings-and-stores.md) | Model choice, dimensionality, distance metrics, FAISS/Chroma/Pinecone/Astra trade-offs |
| 03 | [Retrieval](03-retrieval.md) | Dense, sparse/BM25, hybrid RRF, MMR, re-ranking, multi-vector, graph, multimodal |
| 04 | [Query enhancement](04-query-enhancement.md) | Expansion, decomposition, HyDE, rewriting, routing, step-back |
| 05 | [RAG architectures](05-rag-architectures.md) | The taxonomy — naive → advanced → modular → agentic, plus corrective, adaptive, self, cache, graph, multimodal |
| 06 | [Agentic patterns](06-agentic-patterns.md) | ReAct, tool-calling, LangGraph state machines, reflection loops, planning, multi-agent |
| 07 | [Findings](07-findings.md) | Bugs, mislabelled techniques, security issues, and what is worth porting into Vec |

## Coverage at a glance

What the repo implements, by module:

| Module | Techniques |
| --- | --- |
| `data-ingestion/` | 7 loaders (text, dir, PDF ×2, DOCX ×2, CSV ×2, Excel ×2, JSON/JSONL, SQL); character / recursive / token / semantic splitting |
| `vector-embeddings/` | Cosine similarity from scratch, HF vs OpenAI embeddings, model comparison table |
| `vector-store-database/` | FAISS (+ persistence, metadata filter), Chroma (persisted collections), Pinecone (serverless), AstraDB, `InMemoryVectorStore` |
| `hybrid-search-strategies/` | Dense + BM25 ensemble, MMR, LLM re-ranking, semantic chunking |
| `query-enhancement/` | Query expansion, query decomposition, HyDE (manual + `HypotheticalDocumentEmbedder`) |
| `rag-implementation/` | Linear LangGraph RAG, ReAct agent, ReAct with multiple retriever tools |
| `autonomus-rag/` | Query planning, chain-of-thought planning, iterative retrieval, self-reflection, multi-source synthesis |
| `adaptive-rag/` | Router + 3 graders + rewriter + web-search fallback, full LangGraph cycle |
| `corrective-rag/` | Relevance grading → rewrite → web search → generate |
| `cache-augmented-rag/` | Exact-match response cache; semantic answer cache over FAISS |
| `multi-model-openai/` | CLIP unified text+image embedding, GPT-4V answer synthesis |
| `graphdb/` | Neo4j ingestion, `GraphCypherQAChain`, few-shot Cypher prompting, schema pruning |
| `langgraph/` | State schemas, tool binding, ReAct, streaming, Pydantic state |
| `e2e-project/` | Packaged app: cached ingestion, FAISS, ReAct node, Streamlit UI |
| `multi-agent-rag/` | **Empty** — `requirements.txt` and `command.md` only, no source |

## Two things to fix before anything else

**1. Live credentials are committed to a public repo.** `graphdb/experiments.ipynb` and
`graphdb/promptstatergies.ipynb` both hardcode a Neo4j Aura URI, username, and password in
plain text. Rotate the password and purge it from history — deleting the lines in a new
commit is not enough, the values stay in the object store. Detail in
[07-findings.md](07-findings.md#security).

**2. `multi-agent-rag/` is advertised but empty.** The top-level README describes
"collaborative agent systems / specialized agent roles." The directory contains no code.

## How this relates to Vec

Vec is a **latency-bound, single-language, retrieval-only** system — the whole answer path
targets 200 ms with zero network calls after the transcript arrives
([../04-latency.md](../04-latency.md)). Most of what this repo demonstrates is
LLM-in-the-loop and costs one or more round trips per stage, which puts it firmly outside
that budget.

That does not make it useless to Vec — it makes it the **escalation ladder**. The four
effort levels in `EffortPanel` are the right shape for exactly these techniques, ordered by
cost. [07-findings.md](07-findings.md#what-to-port-into-vec) maps specific techniques onto
specific levels, and flags which ones are index-time (free at query time) versus
query-time (billed against the 200 ms).

## A note on scope

The user ask was "all the types of chunking, retrieval, type of RAG" — so these documents
cover the **full landscape**, not only what the repo contains. Every section is explicitly
split:

> **In this repo** — implemented, with a file reference.
> **Not in this repo** — part of the standard toolkit, described for completeness.

Nothing here is benchmarked. The repo ships no evaluation harness, so any comparative claim
about which technique performs better is either sourced from the literature and labelled as
such, or stated as a trade-off rather than a result.
