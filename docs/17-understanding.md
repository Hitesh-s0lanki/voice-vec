# 17 — Understanding a connected store

*Connecting a store proves a credential works. It says nothing about what is in it — and
until now, nothing asked.*

[13-connectors.md](13-connectors.md) ends the moment a credential is sealed and stored. That
is where the gap opens. The agent knows it *has* a Pinecone. It does not know whether that
index holds Hindi passages or product manuals, whether its vectors can be compared to this
app's at all, or whether the metadata field every query filters on is actually there.

So it guesses. `PineconeBackend.capabilities()` returns the same answer for every index on
earth, and its own docstring concedes the point:

> `filters` stays true because the metadata predicate is always sent and always honoured —
> on an index whose metadata lacks `strategy` it simply matches everything.

"Matches everything" is the failure. The effort ladder asked for a narrowed search, believed
it got one, and fused the results of an unnarrowed one. Nothing raises. Nothing is logged.
Recall is just quietly worse than every number in the response says.

This document is the layer that measures instead.

```
PUT /connectors/{slug}
  clean → verify → seal → store ──────────────► 200, form goes green
                                    │
                                    └─ schedule()          off the request, worker thread
                                            │
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                        ▼
                  probe                  narrate               derive facts
          sample the real store     one LLM call over        coverage → what a
          (200 records, bounded)     the excerpts             query may do
                    └───────────────────────┼────────────────────────┘
                                            ▼
                              connector_profiles (user_id, connector)
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        ▼                                   ▼                               ▼
  card() → system prompt           facts() → Capabilities        GET /connectors/capabilities
  every voice turn, from memory    every connected search        panel, and anything else
```

## Coverage, not presence

The whole layer turns on one number. A probe samples 200 records and reports, per metadata
key, **what share of them actually carry it**.

Presence is what a naive check records — "the index has a `strategy` key" — and presence is
exactly what makes `filters=True` a lie, because one record in two hundred having it is
indistinguishable from all of them having it if you only ask whether you ever saw it.

| What the sample says | What a query may do |
| --- | --- |
| `strategy` on 100% of records | filter on it — it narrows |
| `strategy` on 2% of records | **not** a filter; the predicate would drop the other 98% |
| `page` on 100%, one distinct value | carried and useless — it cannot narrow anything |
| `id` on 100%, distinct on every record | an identifier, not a facet |

That last two rows are not hypothetical. A real connected database had `page NOT NULL` set
to `1` on all 2,366 of its rows: a column that satisfies every constraint, survives every
schema check, and can never narrow a search. And the first card this layer ever rendered
offered `Filter on: book_id, chunk_text, id` — two of which identify a single row.

Both are now named on the card, because the alternative is an agent that keeps trying to
cite a page number that is always 1.

## Three layers, and only one of them decides anything

| Layer | Made by | Allowed to |
| --- | --- | --- |
| `Observation` | a probe, from the store itself | be trusted — it was measured |
| `Understanding` | one LLM call over sampled text | steer routing, and be wrong |
| `CapabilityFacts` | derived from `Observation` alone | gate whether a filter is applied |

The split is the safety property. A hallucinated topic costs a wasted retrieval. A
hallucinated *capability* costs a silently empty result set — so the model is nowhere near
the half that decides anything. `good_for` and `not_for` help a router pick a store;
`filters` and `lexical` decide what the query is allowed to contain.

## The measurement may only remove a claim

[`profiled.py`](../src/rag/backends/profiled.py) wraps a backend and merges the two answers
with `and` in both directions:

```python
lexical = declared.lexical and facts.lexical
```

A backend that knows its own protocol has no keyword channel is right — no sample should
switch one on. A backend that hoped for a metadata field the sample did not find is wrong,
and the sample wins. Only both saying yes yields a capability.

**No measurement means no change.** `facts is None` returns the backend's declared answer
untouched, so profiling can be off, stale, or still running and every store behaves exactly
as it did before this existed. That is what makes it safe to layer onto a working system.

## Being a guest in somebody else's database

A probe runs against a Postgres, a Pinecone or an Astra this app does not own, whose
connection limit, query budget and bill all belong to someone else. Three rules hold in all
four probes:

- **Sample, never scan.** 200 records, whatever the store holds. Coverage converges long
  before that and does not get truer at 200,000.
- **Estimate before counting.** `reltuples` first; an exact `count(*)` only when the estimate
  says the table is small. On a big table `count(*)` is a seq scan this app has no business
  running. (`reltuples` read 2116 on a table holding 2366 rows — an estimate, and reported as
  measured only when it is one.)
- **Never raise past the caller.** An unreachable store is an `Observation` with a reason on
  it, not an exception. Profiling is layered on a connector that already verified; it may not
  turn a working connector into a broken one.

Each store fights this differently:

| | How records come back | What that costs the sample |
| --- | --- | --- |
| **pgvector** | catalogue + `TABLESAMPLE` / `ORDER BY random()` | genuinely random; the honest one |
| **Pinecone** | there is no scan — query from 3 seeded random directions and merge | a union of neighbourhoods, not a sample. Said so on the card. |
| **Astra** | real `find`, paged | unbiased in embedding space, fully biased by insertion order |
| **Composio** | authorised toolkits, not the catalogue | a listed tool the user never connected is not a capability |

Pinecone's is the compromise worth knowing about. One random unit vector returns the 200
records nearest one arbitrary point, which is a cluster; three widely separated directions
turn "one neighbourhood" into "three", which is the difference between coverage that is
usually right and coverage that is right only when the index is homogeneous. The directions
are **seeded**, so re-profiling an unchanged index does not make the numbers flap.

## Two failures with no symptom

**A profile that outlived its credentials.** Rotating a key, renaming an index or pointing at
a different table all produce a store the old profile describes confidently and incorrectly.
Every stored profile carries a digest of the sealed credential blob it was built from
([`fingerprint`](../src/connectors/profile_store.py)), every read checks it, and a mismatch
is treated as *missing* rather than approximately right. This is the same trick
`BackendResolver` uses for its backend cache, persisted.

**A profile that outlived its account.** `connector_profiles` has a foreign key to
`connector_accounts` with `ON DELETE CASCADE`, so disconnecting deletes the understanding
along with the credential and no cleanup path has to remember to.

## What it costs a turn

Nothing that a listener can hear, which is the only reason the card can be in the system
prompt at all.

| | When | Cost |
| --- | --- | --- |
| probe + narrate | once, on connect; then on a 24 h TTL | seconds, on a worker thread |
| `card()` | every voice turn | a dict lookup behind a 60 s TTL |
| `facts()` | every connected search | the same lookup |

A stale profile is **served while its refresh runs behind it**. A slightly old description of
a corpus is worth far more to the turn in flight than a blank one, and corpora rarely change
character between refreshes.

## Excerpts are spread across documents, not across rows

A uniform sample of a chunked corpus is dominated by whichever document is longest. The
first live run of this layer sampled 200 rows of a twelve-book library, quoted three
passages, and produced:

> A small corpus of book text chunks… on topics including philanthropy, political
> punishment, and football commentary.

Three excerpts, three books, and no sense of the shape of the thing. So the probe finds the
**dominant facet** — the field that is on nearly every record, has few distinct values, and
is not unique per row — and takes one excerpt per value of it. Nothing is looked up by name;
that is simply the shape a document boundary has. On this store it is `book_id`, on a store
of scraped pages it would be the domain. Same cost to the model, same amount of the user's
data, and the summary becomes:

> A small vector index of English-language book passages… on topics like film production,
> culture change, travel, mining finance, and competition.

## The user's data leaves, and that is a setting

The profile carries at most five passages of at most 240 characters, so the card can say
what the corpus reads like — and they are sent to a model provider to be summarised, and
stored in *this app's* Postgres rather than the user's.

That is a disclosure, not a footnote. `PROFILE_EXCERPTS=false` keeps the profile to
measurements — counts, fields, coverage, scripts — which still drives every capability
decision and costs only the routing hints. `PROFILE_NARRATE=false` skips the model entirely.
The card then reads in numbers, and nothing that gates a query changes.

## What it produced, on a real store

Run against a Neon Postgres holding somebody else's `book_chunks` table — a schema this app
did not build and does not recognise:

```
Mixed English Book Chunks — 2.4k records, 768-dim, cosine
A small vector index of English-language book passages. The excerpts show narrative prose
and nonfiction text on topics like film production, culture change, travel, mining finance,
and competition.
Good for: finding related book passages, topic-based passage search, English prose retrieval.
Not for: non-English text, page-level document lookup, metadata-rich bibliographic queries.
Filter on: book_id.
Carried but useless (one value everywhere): page.
Search: dense only.
Not searchable right now — do not route questions here.
```

Measured in ~2.0 s of probe and ~2.9 s of narration, once, off the request path.

The middle three lines are written; every other line is measured. The last one is the one
that matters: the table stores 768-dimensional
vectors and this app embeds at 384, so **not one query against it could ever have worked** —
and before this layer, the connector panel showed it green and every question failed
individually, forever.

## The measurement that width was standing in for

`filters=True` was a hope about a *field*. There was a second hope, about the vectors
themselves, and it was worse: **that an index of the right width is an index in the same
space**. It is not. A 768-dim store built by another model accepts a 768-dim query vector,
returns its nearest neighbours, and every one of them is unrelated — nothing raises, the
guardrail abstains on every question, and the store reads as connected, searchable and
merely unhelpful.

So the probe now measures it, the same way it measures everything else. It reads one record's
text *and* that record's stored vector, embeds the text with the embedder this app would
search with, and takes the cosine:

```
identical text, same model        ~1.0   (e5's "passage: " prefix still leaves it >0.9)
a different model                 ~0.0
one real connected store          0.0032
```

`Observation.embedding_match` carries the number, `CapabilityFacts.compatible` carries the
verdict against `EMBEDDING_MATCH_MIN = 0.5`, and a measured mismatch takes `searchable` away.
Untested — no vector, no text, no embedder — stays `None` and takes nothing away, because
absence of a measurement is not evidence of absence.

What the user sees is a card that says the one thing they can act on:

```
Not searchable: these vectors were built by a different embedding model,
so a search here returns unrelated records.
```

Capability discovery still *finds* the store and reports it as `unavailable` with that reason
(docs/23-capabilities.md). Dropping it silently is how somebody who connected a store of book
chunks gets told, with total confidence, that they have nothing about books.
