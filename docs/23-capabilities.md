# 23 — Capabilities are a tool, not a prompt

The agent no longer opens every turn holding a description of everything you have connected.
It opens holding one tool, and goes and looks.

```
user question
     │
     ▼
  agent ──► find_capability("check my inbox")
     │            │
     │            ▼
     │      semantic search over what this person has connected
     │      (the measured cards — docs/17-understanding.md)
     │            │
     │            ▼
     │      gmail — "use the GMAIL_* tool the answer names"
     │
     ├──► GMAIL_FETCH_EMAILS(...)     ← unlocked by the discovery above
     │
     ▼
  spoken answer, grounded in what came back
```

The same path, for the other two kinds:

| Asked | Discovery returns | Then the agent calls |
| --- | --- | --- |
| "check my inbox" | `gmail` (toolkit) | `GMAIL_FETCH_EMAILS` |
| "how many students are enrolled?" | `pgvector` — *student records* (store) | `search_store(store="pgvector", …)` |
| "average marks by class" | `marks` (dataset) | `query_dataset(dataset="marks", …)` |

## What this replaced, and why

Every connected store, dataset and toolkit used to be described in the system prompt of every
turn: the card from [17-understanding.md](17-understanding.md) for each store, and the OpenAI
schema for every action in every linked toolkit. Three problems, and the third is the one
that mattered.

**It grew with the account, not with the question.** Somebody with five toolkits linked paid
for five toolkits' worth of schemas on a turn asking what time it is. The tool pass is
*buffered* — a tool call cannot be streamed into a synthesiser — so that cost sits directly
in front of the first spoken word.

**It was a menu the model answered from.** A card carries real numbers, so a model handed one
recites it: "you have about twenty-five thousand Hindi rows" spoken with total confidence,
with no query behind it and nothing out loud to distinguish it from a measurement. The prompt
had grown two paragraphs of counter-instruction about exactly this.

**It could not choose.** With a Pinecone of product docs *and* a Postgres of student records
attached, `BackendResolver` picked one by a standing order and the question had no say. A
question about students reached whichever store came first in `PREFERENCE`.

## The pieces

| File | What it does |
| --- | --- |
| [`src/capabilities/catalogue.py`](../src/capabilities/catalogue.py) | turns profiles, datasets and *authorised* toolkits into `Capability` — what it is good for, and the call that uses it |
| [`src/capabilities/index.py`](../src/capabilities/index.py) | embeds the cards with the local embedder and answers "what covers this?" — the same search retrieval does, over a handful of documents instead of a corpus |
| [`src/tools/capabilities.py`](../src/tools/capabilities.py) | `find_capability`, the tool |
| [`src/tools/store.py`](../src/tools/store.py) | `search_store(store, question)` — retrieval against the store discovery named |
| [`src/tools/kit.py`](../src/tools/kit.py) | which tools are callable *this round*, and what a discovery unlocks |

## Four rules the design rests on

### 1. Discovery returns instructions, never data

A match is an id, what it is good for, and the exact call to make next — never a count, never
a row, never an excerpt. The agent still has to make that call, and that is the property that
stops a description being mistaken for an answer.

### 2. Nothing is unlocked until it is relevant

Round 1 offers exactly one schema. `search_store`, `query_dataset` and a toolkit's actions
appear only after a discovery has named them, and only the toolkit that was named:
`ToolAgent.tools_for(user_id, only={"gmail"})` fetches gmail's schemas and leaves the rest of
the account out of the prompt.

Unlocking is per turn and forward-only — a new `ToolKit` per question — so a turn cannot act
on something discovered while answering something else.

### 3. No discovery, no gate

Capability discovery is off, or profiling is, or the store was connected thirty seconds ago
and its probe has not finished: the kit then offers every tool the way it did before any of
this existed. Gating a mailbox behind a description of it, and then not having the
description, is the one outcome worse than a prompt that is too long.

The prompt degrades the same way: with discovery available it carries a counts line —
`2 connected stores, 1 dataset, 1 connected account.` — and with discovery unavailable it
carries the old cards, because a counts line means nothing to a model with no tool to follow
it up with.

### 4. The test is lift, not score

`capability_min_score` was an absolute cosine floor, and it was wrong. Measured on a real
account with four capabilities:

```
query                                          top     lift    matched
summary of the book The Laws of Human Nature   0.797   0.027   pgvector  ✓
how are you feeling today                      0.794   0.010   —         ✗
```

The two scores are the same to within 0.003. An absolute floor either rejects the first — it
did, which is how a store whose card says *finding book passages* failed to match a question
about a book — or accepts the second. e5 over short cards lives in a narrow band, and the band
moves with the cards, so there is no number to put between them.

**Lift** — how far the best card sits above the mean of this person's own cards — separates
cleanly. Across a dozen queries on that account:

```
relevant    0.027 – 0.120
irrelevant  0.010 – 0.019
```

`capability_min_lift` is **0.023**, in the gap. After the change, all fourteen probe queries
route correctly:

```
summary of the book The Laws of Human Nature  → pgvector 0.801
book passages about nutrition                 → pgvector 0.842
check my inbox                                → gmail    0.832
open a github issue                           → github   0.853
finance event descriptions                    → pinecone 0.860
movie synopsis                                → pinecone 0.850
how many passengers survived the titanic      → titanic  0.873
what is the weather in Oslo / who won the world cup / tell me a joke /
what time is it / how are you feeling today   → nothing
```

**One capability is a special case, and it returns.** With a single card the mean *is* the
card, lift is always zero, and no threshold means anything. It is handed back with its
`good_for` and `not_for` for the agent to judge — withholding the only thing somebody
connected because the arithmetic degenerates is not honesty, it is a bug.

An empty result is `ok=True`. It is an answer, not a failure — reporting it as a failure would
have the model retry a search that will keep succeeding at finding nothing.

The word-overlap fallback — what runs when no embedder is available — keeps its own floor,
`capability_min_overlap`, because it is a share of the question's own words rather than a
cosine.

### 5. What gets embedded is the subject, not the card

The rendered card is *not* what the index embeds. It repeats the title and summary and then
adds the mechanical tail — `2.4k records, 768-dim, cosine`, `Filter on: book_id`,
`Search: dense only` — which every card carries in nearly the same words, and which dragged
every capability towards the same point. Embedding title + summary + topics + `good_for`
instead roughly doubled the separation: the book question's lift went from 0.008 over the
runner-up to a clean match, and "check my inbox" from 0.014 lift to 0.029.

## The counts line

What is left in the prompt is the one thing the model cannot discover on its own: that there
is something to discover.

```
What you can reach for this person:
2 connected stores, 1 dataset, 1 connected account.
This is reach, not knowledge …
You have not been told what any of it holds. Before answering anything that needs
their data or an action, call `find_capability` …
```

Counts and not names, deliberately. A name is a hint, and a model given hints routes on them:
"you have a students dataset" is enough for it to answer a question about students without
ever querying one.

## What it costs

One extra buffered round trip on turns that need a tool — discovery, then the tool. In
exchange, turns that need no tool carry one schema instead of the whole account, and the
turns that do carry only what they asked about.

The trade is deliberate and it is the opposite of the one the prompt block made: that one
paid on *every* turn to save a round trip on a few.
