# 21 — One package for the agents, one contract under them

## What moved

The agents were spread across the packages that happened to call them first. They are now
one package, and the thing they had in common is now written down instead of repeated.

| Was | Is |
| --- | --- |
| `src/integrations/agent.py` | [`src/agents/tool_agent.py`](../src/agents/tool_agent.py) |
| `src/datasets/agent.py` | [`src/agents/dataset_agent.py`](../src/agents/dataset_agent.py) |
| `src/rag/agents.py` (five module functions) | [`src/agents/rag.py`](../src/agents/rag.py) (five classes + `RagAgents`) |
| — | [`src/agents/base.py`](../src/agents/base.py) — `BaseAgent`, `ModelAgent` |
| — | [`src/agents/model.py`](../src/agents/model.py) — the LangChain client, and the parser |
| — | [`src/agents/prompts.py`](../src/agents/prompts.py) + [`src/prompts/`](../src/prompts/) — one markdown file per agent |

The model calls are **LangChain** now. The single-shot stages are `prompt | model | parser`
chains; `DatasetAgent` is a `create_agent` tool loop. The prompts moved out of triple-quoted
strings into `src/prompts/<name>.md`, because a prompt is the part of an agent most likely to be
edited by somebody who is not editing Python that day, and the diff of a prompt change should
be a diff of the prompt.

What did **not** change: the budgets, the fallbacks, the return types, and every rule about
what happens when a stage cannot run. The one behaviour that did is `DatasetAgent`'s loop,
which is described below.

## What counts as an agent here

A component that **decides something with a model** and is called as a unit: handed a
question and some measured context, returning a typed answer.

| Agent | `name` | What it decides |
| --- | --- | --- |
| `ToolAgent` | `tools` | which of a user's linked Composio tools the model may pick from, and runs the one it picked |
| `DatasetAgent` | `dataset-sql` | the DuckDB SQL that answers an English question about an attached dataset |
| `SynthesisAgent` | `synthesis` | rung 2's grounded answer over the retrieved passages |
| `RelevanceGrader` | `relevance-grader` | whether the retrieval as a whole bears on the question |
| `QueryRewriter` | `query-rewriter` | a second search key, only after retrieval was graded bad |
| `AnswerGrader` | `answer-grader` | supported / useful, as two independent bits |
| `RouterAgent` | `router` | whether the question needs a document search at all |

`ToolAgent` is the one that decides nothing itself — the voice loop's model decides *that*
a tool should run. It lives here because the surface it presents is the model's, and
because it keeps the same three promises.

## What LangChain is used for, and what it is not

| Used for | Not used for |
| --- | --- |
| `prompt \| model \| parser` for the five single-shot stages | the streaming voice reply ([`src/voice/llm.py`](../src/voice/llm.py)) — token-level control of a socket a listener is already hearing |
| `create_agent` for the dataset SQL tool loop | the voice loop's own tool pass ([`voice_service.py`](../src/services/voice_service.py)) — it streams, barges in, and writes an audit row per call |
| `ChatOpenAI` against whatever `resolve_llm()` picked — OpenAI or Sarvam, both compatible | retrieval, reranking, guardrails: measured, deterministic, and not model calls |
| `ChatPromptTemplate` (mustache) over the files in `src/prompts/` | retries — `max_retries=0`, because escalation is [`harness.py`](../src/rag/harness.py)'s decision |

Two of LangChain's defaults are wrong for this system and are overridden in
[`src/agents/model.py`](../src/agents/model.py): it retries a failed call twice, which turns a
6-second grader timeout into an 18-second one on a rung whose budget is ~5 s; and its
`JsonOutputParser` rejects a reply with prose in front of the JSON, which is a reply this
system has to be able to read because `response_format` is deliberately not sent (OpenAI
honours it, Sarvam refuses the request carrying it). `TolerantJson` cuts the object out —
and still raises rather than repairing, so an unreadable verdict stays `None`.

## The three promises `BaseAgent` exists to hold

**Never raise.** Every agent runs inside a turn somebody is listening through, or inside a
rung with a deadline. A raised exception there is a dropped turn; a typed failure is an
answer that can be spoken ("I could not query that") or degraded to the rung below. `_guard`
is the one place that conversion is written, and it logs at warning — a swallowed exception
nobody can see is how a degraded stage becomes a mystery.

**Say when you cannot run.** `ready` answers "would calling this do anything, as configured"
so a caller can skip the round trip rather than pay it to learn the same thing. It reads
configuration, never the network. `ToolAgent` overrides it, because for that one
"configured" means Composio credentials, not a model key — `needs_model = False` is the flag
that says so.

**A grader that cannot answer returns `None`, never a default.** Inherited from
`llm.complete_json` and enforced by every caller in [`ask_service.py`](../src/services/ask_service.py):
an unconfigured model, a timeout and an unparseable reply all produce `None`, and the
pipeline falls back to the deterministic guardrail it already had. Defaulting to
`relevant=True` on a parse failure would turn every provider hiccup into an approval,
silently, in the direction that emits answers rather than withholding them.

## The hierarchy

```
BaseAgent          settings, a logger named for the agent, `ready`, `_guard`, `_ms`
├── ModelAgent     + src/prompts/<name>.md, a cached ChatOpenAI, `_text` / `_json`
│   ├── SynthesisAgent, RelevanceGrader, QueryRewriter, AnswerGrader, RouterAgent
│   └── DatasetAgent      (single_shot = False — it drives its own loop)
└── ToolAgent      needs_model = False
```

`ModelAgent.__init__` loads the prompt file, so an agent whose markdown is missing or
malformed cannot be constructed — the failure lands at startup, where a broken checkout
belongs, instead of degrading a turn hours later. That is the one failure in this package
treated differently from a provider having a bad minute.

Budgets are properties on the class, not arguments at the call site: a grader gets the
grader's tokens and the grader's timeout on every call it will ever make, and a call site
that *could* pass its own would eventually pass a different one.

`__init_subclass__` refuses a subclass that does not set `name`. It is not decoration: the
name is the logger *and* the prompt filename, and an agent logging its degradations under an
inherited name is the one line an operator has to go on when a stage quietly stops running.

## The one real agent loop

`DatasetAgent` is a `create_agent` graph with one tool, `run_sql`, which guards the statement
and runs it in the sealed DuckDB sandbox. The model writes a query, sees DuckDB's own error
if it fails, and corrects it. Two bounds are this package's, not LangChain's:

```python
ModelCallLimitMiddleware(thread_limit=dataset_sql_repairs + 1, exit_behavior="end")

@before_model(can_jump_to=["end"])
def stop_when_answered(state, runtime):
    if run.result is not None:
        return {"jump_to": "end"}
```

The first is the repair budget that was always there — a second failure means the question
cannot be answered from these columns. The second is latency: without it the graph pays for
one more completion to narrate rows the caller renders itself, a full round trip on the voice
path for nothing.

The tool closes over a `_Run` that records what actually executed, so `Answer` is built from
evidence rather than from the model's account of what it did. And when a provider ignores the
tool and writes the SQL into the message instead — which happens — the query is right there,
so it is run rather than reported as a failure to hold the tool correctly.

## The tools, and where they live

`src/agents/` decides; [`src/tools/`](../src/tools/) does the thing. A tool here is something
with an effect or an answer outside the model — somebody's mailbox, somebody's dataset — and
they come in two shapes because two different loops call them:

| File | Tool | Called by |
| --- | --- | --- |
| [`result.py`](../src/tools/result.py) | — `ToolResult`, what any call produced | everything below |
| [`dataset.py`](../src/tools/dataset.py) | `query_dataset` — ask a dataset in English | the voice loop's tool pass, as an OpenAI schema |
| [`sql.py`](../src/tools/sql.py) | `run_sql` — one guarded SELECT in the sandbox | `DatasetAgent`'s LangChain loop, as a `StructuredTool` |

The two dataset entries are not a duplication and the split is the point: `query_dataset`
takes a *question*, because the voice model is answering out loud and has never seen the
column list; `run_sql` takes a *statement*, because the dataset agent has been handed the
measured schema card. One is the outer surface, the other is the inner one, and the agent is
what sits between them.

The Composio tools have no file here: they are somebody's linked accounts, discovered per
user at the moment of the turn, so `ToolAgent` owns them and hands back the same
`ToolResult`. `src/tools/` owns the tools this codebase *wrote*.

`SqlRun` lives in `sql.py` rather than in the agent because it is the record of what
executed — statement, rows, attempt count, accumulated as the tool runs. The `Answer` is
built from that, not from the model's account of what it did.

## The prompt files

One per agent, named for its `name`: `src/prompts/router.md` is what `RouterAgent` says. The
format and its two traps are documented in [`src/prompts/README.md`](../src/prompts/README.md); the
one worth repeating here is that variables are **triple-braced** (`{{{query}}}`), because
mustache's two-brace form HTML-escapes what it substitutes and would send a passage
containing `<` or `&` to the model altered, silently. `src/agents/prompts.py` refuses a file
that does it, and `tests/test_prompts.py` pins that along with the `NO_ANSWER` sentinel, which
is instructed in markdown and compared in Python.

## What is deliberately *not* in here

- **The model clients** ([`src/rag/llm.py`](../src/rag/llm.py), [`src/voice/llm.py`](../src/voice/llm.py)) — an agent is a decision, a client is a socket.
- **The tools** ([`src/tools/`](../src/tools/)) — things agents *run*, not things that decide. See the section below.
- **The profile narrators** ([`src/connectors/narrate.py`](../src/connectors/narrate.py), [`src/datasets/narrate.py`](../src/datasets/narrate.py)) — single model calls owned by the measuring that produced their input. Their own docstrings describe what they write as being *for* an agent to route on, rather than by one.

## How the ladder holds them

`AskService` builds `RagAgents(settings)` once, in its constructor, and the five stages are
constructed with the deployment's settings rather than handed them again inside a request
that already has a deadline. A call site now names the stage and its arguments and nothing
else:

```python
lambda: self._agents.router.route(query=run.query, corpus=self._corpus_hint())
lambda: self._agents.relevance.grade(query=run.query, hits=run.hits, english=run.english)
```

The harness still wraps each of those in `stage(..., optional=True)`, so an agent that
returns `None` is a stage that did not run — which is the same thing it was before.
