# 06 — Agentic patterns

Source modules:
[`langgraph/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/langgraph) ·
[`rag-implementation/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/rag-implementation) ·
[`autonomus-rag/`](https://github.com/Hitesh-s0lanki/agentic-rag/tree/main/autonomus-rag)

[05-rag-architectures.md](05-rag-architectures.md) covered *which* architectures exist. This
covers the machinery they are built from — the orchestration layer that turns a chain into
an agent.

## State is the whole design

Every LangGraph app in this repo is a state machine. Nodes are pure functions
`state → state`; edges decide what runs next; the state schema is the contract between them.
Getting the schema right is most of the design work.

Three schema styles appear, and the difference is not cosmetic:

```python
# 1. TypedDict + reducer — for message-passing agents
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. TypedDict, plain — for simple pipelines
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# 3. Pydantic BaseModel — for validated, defaulted state
class IterativeRAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    verified: bool = False
    attempts: int = 0
```

`Annotated[..., add_messages]` is the important one. It declares a **reducer**: returning
messages from a node *appends* rather than overwrites. Without it, each node clobbers the
conversation and the agent loses its own history. Reducers are how LangGraph expresses
accumulation, and `add_messages` additionally handles message deduplication by ID.

Pydantic state buys validation and defaults, which is why the `autonomus-rag` notebooks use
it for anything carrying a counter. Note the update idiom difference:

```python
# Pydantic — copy with changes, everything else preserved
return state.model_copy(update={"retrieved_docs": docs})

# vs. the pattern used in query_planning_decomposition.ipynb
return RAGState(question=state.question, sub_questions=sub_questions)   # drops other fields
```

The second reconstructs the whole object and **silently drops any field not re-listed**.
`model_copy` is the safer idiom; the repo uses both, sometimes in the same module.

## Pattern 1 — ReAct

Reason → Act → Observe, looping until the model stops calling tools.

```python
react_node = create_react_agent(llm, tools)
```

Under the hood this is a two-node cycle: the model node emits either a final answer or tool
calls; the tool node executes them and appends results; control returns to the model. The
loop ends when the model produces a message with no tool calls.

What makes it work is **tool descriptions**:

```python
Tool(name="InternalResearchNotes",
     description="Search internal research notes for experimental results and agent designs.")
```

That string is the routing logic. The model chooses among tools by reading their
descriptions — so a vague description is a routing bug, and two overlapping descriptions
produce non-deterministic tool selection. Treat descriptions as the agent's API contract,
not as comments.

`reAct_rag_tool.ipynb` gives the agent four retrieval tools (Wikipedia, ArXiv, and two
distinct internal corpora) and lets it choose. The multi-tool query in that notebook triggers
two different tool calls without being told to — that is the pattern working as intended.

**Retrieval-as-tool vs retrieval-as-node** is the fundamental fork:

| | Node | Tool |
| --- | --- | --- |
| Runs | Always | Only if the model decides |
| Query | The user's | The model's, rewritten |
| Count | Exactly one | Zero to many |
| Latency | Predictable | Unbounded |
| Debugging | Trivial | Requires tracing |

Naive/advanced RAG uses nodes. Agentic RAG uses tools. `e2e-project/src/node/reactnode.py`
is a hybrid worth noting — it runs a retriever node *and* hands the same retriever to a ReAct
agent inside the generation node, so documents are fetched twice and the node-fetched ones
are never actually used in the prompt. Recorded in
[07-findings.md](07-findings.md#e2e-double-retrieval).

## Pattern 2 — Router / conditional entry

```python
workflow.add_conditional_edges(START, route_question, {
    "web_search": "web_search",
    "vectorstore": "retrieve",
})
```

A classifier at the entry point picks the pipeline. Cheap, bounded, and the single
highest-value agentic pattern for a cost-sensitive system: one small call that can save the
entire expensive path.

Pair it with `with_structured_output` and a `Literal` type so the destination is guaranteed
to be a valid node name — as `adaptive-rag` does.

## Pattern 3 — Grade-and-branch

```python
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant, 'yes' or 'no'")

retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)
```

An LLM judge whose output is a **control signal**, not prose. The repo uses four:
retrieval relevance, hallucination grounding, answer usefulness, and generation quality.

Two things this gets right and one it doesn't:

- ✅ **Binary, not scalar.** LLMs are unreliable at "rate this 1–10" and reliable at
  yes/no. Binary judgments are also directly usable as edges.
- ✅ **Structured output.** No parsing, no regex, retried by the SDK on mismatch.
- ⚠️ **`binary_score: str`.** Typing it `Literal["yes", "no"]` would constrain the model to
  exactly two values; as `str`, `"Yes"` or `"yes."` both slip through and every comparison
  site (`grade == "yes"`) is a latent bug.

The grader is also the **guardrail primitive**. A grounding check that can reject an answer
is the mechanism behind knowing when *not* to answer — directly relevant to
[../06-guardrails.md](../06-guardrails.md).

## Pattern 4 — Reflection loop

```python
builder.add_conditional_edges(
    "reflect",
    lambda s: END if s.verified or s.attempts >= 2 else "refine"
)
```

Self-critique with a **bounded** retry. The `attempts >= 2` disjunct is not optional — a
model asked "is this good enough?" has no reliable convergence property and will happily
loop indefinitely. Every reflection cycle needs:

1. a hard iteration cap in the edge condition,
2. a counter incremented inside a node (not in the condition — conditions can be re-evaluated),
3. a defined terminal behaviour when the cap is hit — return the best attempt, or abstain.

`adaptive-rag` satisfies none of these on its `not supported` branch, which is wired
`generate → generate` with no cap. In practice it never actually spins, because that branch
calls an undefined `pprint` and raises `NameError` first — a bug masking a design flaw. Fix
the `NameError` without adding a cap and you convert a crash into an infinite loop.

## Pattern 5 — Plan-and-execute

```python
def plan_query(state):     # LLM decomposes into sub-questions
def retrieve_for_each(s):  # one retrieval per sub-question
def generate_final(s):     # synthesise across all results
```

Plan up front, execute the plan, synthesise. Linear — `planner → retriever → responder` —
with no feedback loop, which makes it predictable and cheap relative to iterative retrieval.
The trade-off: a bad plan is never revised.

`chain_of_thoughts.ipynb` is the same graph with the prompt changed from "sub-questions" to
"reasoning steps." Worth noting that the *structure* is identical — the difference is
entirely prompt-level, which is a useful reminder about how much of "agent design" is really
prompt design.

## Pattern 6 — Multi-source fan-out

`answer_synthesis.ipynb` queries four sources — internal docs, a YouTube transcript,
Wikipedia, ArXiv — and synthesises one answer with source-labelled context blocks:

```python
context += "\n\n[Internal Docs]\n" + ...
context += "\n\n[Wikipedia]\n"     + ...
```

Labelling each block by provenance is good practice: it lets the model weigh sources and
makes citation possible.

But the graph is **sequential**:

```python
builder.add_edge("retrieve_text", "retrieve_yt")
builder.add_edge("retrieve_yt",   "retrieve_wiki")
builder.add_edge("retrieve_wiki", "retrieve_arxiv")
```

These four retrievals are independent — nothing in `retrieve_wiki` depends on
`retrieve_yt`'s output. Total latency is the **sum** where it should be the **max**. In
LangGraph, adding four edges *from the same source node* fans out concurrently and the
convergent node waits for all of them. This is a one-line-per-edge fix for a ~4× latency
win, and it is the clearest missed optimisation in the repo.

## Pattern 7 — Checkpointing

```python
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
app.invoke({...}, {"configurable": {"thread_id": "demo-user-1"}})
```

State persisted per thread. Used only in `cache-augmented-rag`. This is what makes
multi-turn conversation, resumption after failure, and human-in-the-loop interrupts possible.
`MemorySaver` is in-process and lost on restart; the durable backends (SQLite, Postgres) are
not shown.

---

## Not in this repo

| Pattern | What it adds |
| --- | --- |
| **Supervisor / hierarchical agents** | A coordinator delegating to specialists. Advertised in `multi-agent-rag/`, absent. |
| **Human-in-the-loop** | LangGraph `interrupt()` to pause for approval before consequential actions. |
| **Streaming to the user** | `langgraph/src/streaming.ipynb` covers streaming mechanics, but no RAG graph streams its answer. For a voice UI this is the difference between 4 s of silence and 400 ms to first word. |
| **Parallel fan-out** | See pattern 6 — supported by the framework, used nowhere. |
| **Durable checkpointing** | SQLite/Postgres savers for real persistence. |
| **Tracing / observability** | No LangSmith, no callbacks, no token accounting. Multi-node agents are close to undebuggable without it. |
| **Timeouts and retries** | No node-level timeout anywhere. One hung Tavily call hangs the whole graph. |

---

## Rules that hold across all of them

1. **Cap every cycle.** A loop without an iteration bound is an outage waiting for the right
   input.
2. **Type every control signal.** `with_structured_output` + `Literal` for anything an edge
   branches on. Never parse prose to decide control flow.
3. **Tool descriptions are code.** They are how the model routes; write them like an API.
4. **Fan out what's independent.** Sequential edges between unrelated retrievals are pure
   latency.
5. **State updates should be copies, not reconstructions.** `model_copy(update=...)`, so
   adding a field later doesn't silently drop it in five places.
6. **Instrument before you compose.** A 5-node graph with 4 LLM judges has no debuggable
   failure mode without tracing.

Next: [07-findings.md](07-findings.md).
