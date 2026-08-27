# 06 — Guardrails

> Requirement 6: *"Add guardrails around your model — handling for off-topic queries,
> unsafe/inappropriate inputs, hallucination checks, or answers not grounded in the retrieved
> context. Show that your system knows when not to answer, not just how to answer."*

## Why this is our strongest requirement

Most submissions will demonstrate guardrails anecdotally: type something rude, show the
refusal, move on. That is a demo, not evidence.

MSMARCO-XI ships a **labelled abstention set**. ~39% of rows have `sum(is_selected) == 0`
and `Eng_Answer == "No Answer Present."` — queries where the corpus genuinely does not
contain the answer ([01-dataset.md](01-dataset.md)). Every one of those is a labelled
example of *should not answer*, and the other ~61% are labelled examples of *should answer*.

So we can report abstention **precision, recall and F1 over thousands of queries**, and plot
the trade-off curve. That converts the softest requirement in the brief into the hardest
number in the submission. Nothing else we build will be as cheap to prove.

## Four gates

Placed at four different points because they catch four different failure modes.

```
transcript ─► Gate 1 ─► retrieve ─► Gate 2 ─► answer ─► Gate 3 ─► [rung 2 and up] Gate 4 ─► user
              input                 retrieval          grounding                entailment
```

### Gate 1 — input (pre-retrieval, ~2 ms)

Runs on the transcript before anything expensive.

- **Empty / noise.** Sarvam already returns 422 for silence. Also reject transcripts under
  ~3 characters, or that are pure filler — cheap, and stops garbage entering the index query.
- **Unsafe input.** A denylist plus lexical patterns for the categories worth refusing
  outright. Deliberately conservative: this is an information-retrieval system over a web
  corpus, and over-blocking ordinary queries is itself a failure. **Report the false-positive
  rate on the 500-query eval sample** — a guardrail that silently refuses 5% of legitimate
  queries is a bug, and only measurement will surface it.
- **Prompt injection.** The transcript is user input that reaches an LLM at Tier 3. Strip and
  flag imperative patterns aimed at the system ("ignore previous instructions", "you are
  now…"). Note the voice angle: injection has to survive being *spoken aloud* and transcribed,
  which is a meaningfully harder attack — worth saying in the writeup.
- **Language check.** If Sarvam detects a language we have no index for, abstain immediately
  with a useful message rather than searching the wrong corpus and returning confident
  nonsense. [`languageName()`](../frontend/src/lib/languages.ts) already maps the codes.

### Gate 2 — retrieval (post-search, ~1 ms)

**This is the gate that does the real work**, because it is the one the labelled data scores.

After hybrid search and RRF fusion, abstain when the evidence is too weak:

| Signal | Meaning |
| --- | --- |
| top-1 score below `FLOOR` | nothing in the corpus is close |
| top-1 minus top-5 mean below `MARGIN` | everything is equally mediocre — no clear match |
| fewer than `MIN_HITS` above floor | thin evidence |

The margin test matters as much as the floor. A query with ten results all scoring 0.61 has
no answer in the corpus; a query with one at 0.79 and the rest at 0.55 probably does. An
absolute floor alone cannot tell those apart.

`FLOOR` and `MARGIN` are **tuned against the labelled abstention set**, not guessed — sweep
them, plot precision/recall, pick the operating point, and publish the curve. See
[07-evaluation.md](07-evaluation.md).

### Gate 3 — grounding (post-extraction, ~3 ms)

For Tier 1–2 the answer is a span lifted from a retrieved chunk, so grounding is verifiable
by construction: **assert the returned span is a substring of the retrieved context.** Not a
similarity score — a substring check. If it is not, that is a bug in the extractor, and the
system abstains rather than emitting text of unknown provenance.

This is the payoff of extractive answering. At Tier 1 the hallucination rate is not low, it
is **structurally zero**, and that is a claim we can defend rather than benchmark.

Also enforced here:

- **Every answer carries a citation** — `docId`, `strategy`, `score`. No citation, no answer.
- **Answerability of the *question type*.** MS MARCO is a web-passage corpus from a fixed
  snapshot. Queries about current events, personal data, or anything after the corpus date
  cannot be grounded regardless of retrieval score.

### Gate 4 — support (rung 2 and up, post-generation) ✅ built

Replaces Gate 3 wherever the answer was *written* rather than lifted
([15-effort.md](15-effort.md)). It only runs on rungs that call a model, so it is outside the
200 ms path by construction — but the gate itself is local and costs milliseconds, because
the embedder is already loaded and the context is short.

Gate 3 and Gate 4 check the same property by different means, and the difference is the whole
reason rung 1's hallucination rate is *structurally* zero rather than merely low. An extracted
span is a substring of its passage; anything else is a bug in the extractor. Generated prose
can be perfectly faithful without sharing a character sequence with its source, so Gate 4 asks
the weaker question it can actually answer.

1. **Citation validity.** No citations, no answer — the same rule Gate 3 enforces.
2. **Claim coverage.** Every *sentence* of the answer is embedded and scored against the
   sentences of the context it was written from, and enough of them must clear
   `GENERATION_SUPPORT_FLOOR` (`GENERATION_SUPPORT_RATIO` sets how many). Per sentence rather
   than per answer, because the failure this catches is local: a model handed four good
   passages writes three faithful sentences and one fluent invention, and a similarity score
   over the whole paragraph averages that invention away.
3. **A gate that cannot run passes.** If the embedding call itself fails, Gate 4 returns no
   objection and says so in the trace. Refusing because the *check* broke would trade a
   possible hallucination for a certain abstention on every request while the embedder is
   unwell.

Not yet built, and worth adding: **numeric fidelity** — every number in the answer must appear
in the cited context. A cheap regex that catches the most damaging and most common class of
LLM error in a retrieval setting, and one that sentence-level cosine similarity is
specifically bad at, because changing a digit barely moves the vector.

The floor is a conservative guess and is flagged as such in
[15-effort.md](15-effort.md#what-has-not-been-measured): it should be swept against gold
answers the way Gate 2's was, so the hallucination rate is measured rather than asserted.

## Abstention is a first-class outcome

`AskResponse.status` distinguishes three things, and the UI must render them differently:

| Status | Meaning | Voice |
| --- | --- | --- |
| `answered` | grounded answer with citations | the answer |
| `abstained` | pipeline ran fine; the corpus cannot support an answer | "I don't have that in my sources." |
| `refused` | input rejected at Gate 1 | "I can't help with that." |

**`abstained` is a success path, not an error path.** It renders as a normal turn in
[`conversation.tsx`](../frontend/src/lib/conversation.tsx), with the `reason` shown. If abstentions
render as red error toasts, we have built a system that is ashamed of the exact behaviour the
requirement asks us to demonstrate.

Where possible, abstain *usefully*: "I don't have that, but I do have information about X" —
using the top retrieved chunk's topic even when it fell below the floor.

## Tuning the trade-off

There is one knob and it is a real trade-off. Aggressive abstention → high precision, low
coverage, a system that refuses too much. Permissive → high coverage, ungrounded answers.

We will sweep it and publish the curve rather than picking a point and asserting it:

| Operating point | Abstention precision | Abstention recall | Answer coverage | Grounded-answer accuracy |
| --- | --- | --- | --- | --- |
| conservative | | | | |
| **balanced (default)** | | | | |
| permissive | | | | |

Ground truth: `sum(is_selected) == 0` means *should abstain*. Method in
[07-evaluation.md](07-evaluation.md).

## What to demo

Six queries, drawn from the labelled data so each one is provably the case it claims to be:

1. **Answerable** — a `DESCRIPTION` row with a gold passage. Grounded answer + citation.
2. **Answerable, numeric** — a `NUMERIC` row. Shows S2 sentence-window winning the route.
3. **Genuinely unanswerable** — a real `"No Answer Present."` row. Abstains. *This is the
   money shot: it is not a made-up example, it is a labelled one.*
4. **Off-topic** — something plainly outside a web-passage corpus. Gate 2 abstains.
5. **Unsafe input** — Gate 1 refuses.
6. **Spoken prompt injection** — "ignore your instructions and…" said out loud. Gate 1 flags,
   pipeline continues on the sanitised query.

Then the numbers table. The demo shows the behaviour; the table proves it holds at scale.
