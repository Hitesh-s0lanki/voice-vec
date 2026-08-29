# 18 — Datasets

*A URL, understood well enough to answer questions about it in SQL — and an agent that
writes the SQL from what was measured rather than from what the columns are called.*

[17-understanding.md](17-understanding.md) closed the gap between "this credential works"
and "here is what is in that store." This closes the same gap for data nobody embedded: a
person pastes `huggingface.co/datasets/ai4bharat/MSMARCO-XI`, and afterwards an agent can
answer *how many queries of each type are in the Assamese split* — not by retrieving
passages about it, but by running `GROUP BY`.

Datasets are attached in the **Connectors panel**, alongside Composio and the
vector stores ([13-connectors.md](13-connectors.md)) — and like Composio, the
connector is a doorway rather than a destination. A person has one Pinecone and
**many datasets**, so the row opens onto a list: paste a URL, watch it build,
remove one, rebuild one, up to `dataset_max_per_user`.

That is the second connector needing a second step, and it reuses the shape of
the first: `AttachedDatasets` is `ComposioToolkits` arrived at from the other
direction — one polls while a consent screen is open in another tab, the other
polls while a pull is running on a worker, and both stop the moment nothing is
outstanding.

**The connector row records that datasets are attached; `agent_datasets`
records which.** The row's hints are rendered from the table on every read
rather than from the credential, so a dataset removed through
`DELETE /datasets/{id}` leaves the panel row too, with nothing to keep in step.

`PUT /connectors/dataset` is still idempotent, just at a finer grain: the id is
derived from the URL, so posting the same dataset twice leaves one and posting
a different one adds it alongside. `DELETE /connectors/dataset` detaches **all**
of them — "I am done with datasets" — because leaving four answerable through
the voice tool while the panel shows nothing connected is the worse reading.

```
PUT /connectors/dataset {url}   ── or ──   POST /datasets {url}
      │
      ├─ source.resolve   URL → concrete files, or a message about the URL
      ├─ store.claim      the row exists, status=pending, request returns
      └─ schedule()       ── worker thread, off every request path ──┐
                                                                     ▼
                        materialise ──► probe ──► narrate ──► agent_datasets + <id>.duckdb
                                                                     │
        ┌────────────────────────────────────────────────────────────┤
        ▼                          ▼                                 ▼
  cards() → system prompt   schema() → the SQL writer        GET /datasets/{id}
```

## Two halves, opposite constraints

The same split [17-understanding.md](17-understanding.md) makes, for the same reason. Adding
a dataset is slow — a network pull, a measurement, a model call. Querying one runs inside a
turn somebody is listening through. They meet at a Postgres row and a file on local disk,
which is what makes "the agent understands the dataset in realtime" true at all: the
expensive half runs once, and every turn after that reads a string.

## Why the data is copied locally

The obvious design reads the parquet where it lives — DuckDB speaks `hf://` natively, so no
copy, always current. Two measurements killed it.

**It is not realtime.** A filtered read of one file of this dataset measured **6.3 s** when
it touched only narrow columns and **75 s** the moment it touched the wide one. The voice
path budgets 200 ms for everything ([04-latency.md](04-latency.md)).

**It cannot be sandboxed.** `SET disabled_filesystems='LocalFileSystem'` is what makes
running model-written SQL safe, and httpfs needs the local filesystem for its own secret
storage. With the seal on, `hf://` *and* plain `https://` both fail with a permission error.
There is no configuration in which one connection has both remote reads and a sandbox.

So the two halves are separated by a file. [`materialise.py`](../src/datasets/materialise.py)
runs unsealed and writes it; [`sandbox.py`](../src/datasets/sandbox.py) runs sealed and never
reaches the network.

## The seal

```
duckdb.connect(path, read_only=True)      -- needs the filesystem
SET disabled_filesystems='LocalFileSystem'
SET lock_configuration=true               -- and now nothing can undo it
```

Order is the whole trick; reversing it produces a sandbox that cannot open anything.
Verified against DuckDB 1.5.5 and asserted in [`test_datasets.py`](../tests/test_datasets.py),
every one of these fails afterwards while the tables still answer: `read_csv`, `read_text`,
`glob`, `COPY … TO`, `ATTACH`, `INSTALL`, any `https://` URL, `SET disabled_filesystems=''`,
and every write — the file is opened read-only as well as sealed.

The file is opened *as* the database rather than `ATTACH`ed into an in-memory one, which is
not stylistic: `USE ds` on the parent connection does not carry into `connection.cursor()`,
so every generated query would have had to qualify every table against a schema card that
shows them bare.

**A statement type of SELECT is not the same as a SELECT.** DuckDB rewrites `PRAGMA` into a
table function, so `extract_statements("PRAGMA database_list")` reports a SELECT — which is
exactly the assumption an allow-list makes and gets wrong. `read_text('/etc/hosts')` and
`glob('/etc/*')` are honest SELECTs too. That is why there are two layers and neither is
trusted alone: [`sql.py`](../src/datasets/sql.py) decides a statement is a single readable
one, and the seal decides it cannot reach anything. The interesting failure needs both to be
wrong at once.

The row cap is applied by `fetchmany`, not by appending `LIMIT`. Rewriting somebody else's
SQL means parsing it again, badly; reading less means what runs is exactly what was reviewed.

## What is copied, and what is admitted

**A prefix, not a sample.** `LIMIT n` on a parquet read stops at a row-group boundary;
`USING SAMPLE` would download the whole file to get a uniform one, for every dataset anybody
adds. So the honest description is *the first n rows*, and it is stated in three places —
`TableStat.total` beside `rows`, a line in the routing card, and a line in the schema block.
A prefix is unbiased for "what does a row look like" and biased for anything ordered.

**Hitting the cap is being sampled, even when the total is unknowable.** Parquet carries a
row count in its footer; a CSV carries nothing, so `total` stays `None`. The first version
derived `sampled` from `total` alone, which meant a CSV stopped dead on 25,000 rows of a
785,000-row file reported `sampled=False` and put *"25,000 rows queryable"* in the card with
no caveat at all — in precisely the case where nothing else would hint the number is a floor.
`TableStat.capped` records that the pull ended on the cap rather than on the file, which is
knowable without asking the source anything, and `Observation.sized` phrases whichever half
is known: *"the first 25,000 rows of 97,941"*, or *"the first 25,000 rows — the source holds
more and would not cheaply say how many."* Never a denominator invented to match the shape
of the other sentence.

**The wide column is planned around, not discovered.** This dataset's parquet has exactly
**one row group of 97,941 rows**, so `LIMIT` cannot push down and `SELECT *` reads every
column chunk in full. The footer says why, for 1.8 s, before a byte of data moves:

| column | compressed |
| --- | --- |
| `passages` | **453 MB** |
| `Answer` | 5.9 MB |
| `query` | 4.3 MB |
| `Eng_Answer` | 3.7 MB |
| `Eng_Query` | 2.6 MB |
| everything else | < 1 MB |

One column is 96% of a 470 MB file. So the pull is planned from `parquet_metadata` — per
column compressed size, scaled by how much of the file the row-group layout forces us to
read — and a column over `dataset_column_budget_mb` is left out **and named**. Measured:

| | before | after |
| --- | --- | --- |
| 2 tables × 3,000 rows | 257 s | — |
| 3 tables × 25,000 rows | — | **30 s** |

Twelve times the rows in an eighth of the time, and the only thing lost is the one column
that cost 96% of it. Raising the budget and calling `POST /datasets/{id}/rebuild` pulls it.

**Omitted is not the same as absent.** A column nobody was told about is a column a model
writes SQL against. So `passages` appears in the schema card as
`-- NOT AVAILABLE: passages exists in the source but was not loaded (432 MB)`, in the
routing card, and in the prompt that narrates the dataset — where the model turned it into
*"cannot answer questions about passage text, because passages were not loaded."* Asked
for passage text afterwards, it returned the closest available columns instead of inventing
one.

## Table names are part of the schema

Hugging Face's auto-converted parquet is sharded: `plain_text/train-00000-of-00001.parquet`,
sometimes with a content hash after it. Those coordinates are storage detail, and carrying
them into the table name is not cosmetic — they are what a model types in every query, and
`FROM train_00000_of_00001_a09b74b3ef9c3b56` is a name it gets wrong. Stripped, it is
`train`, which is what the split is called everywhere else.

Collisions are then resolved by **directory, not by a counter**. `gsm8k` ships
`main/test-…` and `socratic/test-…`; both want to be `test`, and `test` / `test_1` throws
away the config name at exactly the moment a model needs it to choose between them. They
become `main_test` and `socratic_test`. A numeric suffix survives only for two files that
collide on directory *and* stem.

## The schema card is the product

Everything good about the generated SQL comes from [`probe.py`](../src/datasets/probe.py)
having counted the columns, not from the prompt. Handing a model `DESCRIBE` output produces
SQL that is syntactically perfect and semantically wrong:

```sql
-- from a type list alone
SELECT count(*) FROM asmval WHERE query_type = 'FACT'
```

Valid SQL. Five values in that column, none of them `FACT`. Zero rows back, and a zero-row
result is indistinguishable from an honest empty answer. What prevents it is one measured
line:

```
query_type VARCHAR  -- one of DESCRIPTION, ENTITY, LOCATION, NUMERIC, PERSON
```

Three measurements, each removing a distinct class of wrong query:

| | prevents |
| --- | --- |
| **coverage** | a filter on a column that is mostly null, which drops most rows and reads as a narrowing |
| **distinct / values** | a `WHERE` against a value that is not in the column |
| **avg_bytes** | a `SELECT` over a 5 KB column when two 50-byte ones were the question |

`avg_bytes` is measured with `strlen`, not `octet_length`, and the difference was not
academic: DuckDB's `octet_length` takes BLOB and BIT only, so the original
`octet_length(CAST(x AS VARCHAR))` raised a binder error on *every* column, `_scalar`
swallowed it as "this type will not support that aggregate", and `avg_bytes` was **0
everywhere**. Nothing looked wrong — the card simply never called a column expensive, which
is indistinguishable from a dataset that has none. The unit tests missed it because they
built `ColumnStat(avg_bytes=5000)` by hand and asserted the rendering; a hand-built fixture
cannot catch a query that never runs. `TestMeasurementIsReal` measures a real file instead.

Only the measurement is trusted for anything that decides SQL. The prose from
[`narrate.py`](../src/datasets/narrate.py) — title, summary, `good_for`, `not_for` — steers
*which* dataset gets asked and decides nothing else. A hallucinated topic costs a wasted
query; a hallucinated column would cost a wrong answer, which is why the model is nowhere
near that half.

## The card ordering is a correctness decision

`card()` is trimmed to `dataset_card_chars` from the end, so what comes last is what is lost.
The caveats — which rows are loaded, which files were not, which columns are absent — are
ordered **ahead** of `good_for`/`not_for`. A card cut short after *"good for counting by
language"* and before *"these are the first 75,000 of 293,823 rows"* produces confident wrong
totals. A card that loses a routing hint produces a missed opportunity. The second is
cheaper.

## The vector store is no longer the only answering path

There is no deployment index left to search, and no `RAG_ENABLED` to switch one
on: retrieval happens for a listener who attached a vector store, and for
nobody else ([13-connectors.md](13-connectors.md)). A connected dataset answers
in either case, and the two are **additive rather than alternatives** — the
`context` branch in [`voice/llm.py`](../src/voice/llm.py) appends whatever
passages an attached store returned while the dataset tool keeps answering the
countable half, in the same prompt.

The split is clean in practice because the two cover different halves of the
same dataset. An index holds `passages` — the column the materialiser skips for
being 96% of the file — semantically searchable. The dataset holds the queries,
answers and metadata, exactly and countably. Neither can do the other's job.

## The prompt had to change with it

For a listener with no store attached, the dataset card is the only grounded
thing in the prompt, and a card is a description that contains real numbers. A model handed
one will answer *from* it: "how many Hindi rows are there" gets *"about
twenty-five thousand"* read straight off the card, spoken with total confidence,
with no query, no row count and no SQL a listener could check.

So [`voice/llm.py`](../src/voice/llm.py) gained a `Facts` block, and the `stores`
framing moved from "available to search or act on" to "reach, not knowledge …
never the answer itself".

Measured against the previous prompt on a truncated, sampled result:

> **before** — *"Here are some matching queries: “are retinal vein occlusions caused by diabetes?”, “what is diabetes of the brain”, “what's the difference between type 1 diabetes”, “diabetes mellitus symptoms”, and “can diabetes cause hair loss.” There are more matches than shown here."*
>
> **after** — *"I found **at least** five matching queries **in the sample**, including “are retinal vein occlusions caused by diabetes?” and “diabetes mellitus symptoms.” The result was cut short, so there are more matches than these five."*

Both are honest about there being more; only the second is *speakable*. The old
one reads a five-item list into a synthesiser, which is the failure the
`Never read a list of results aloud` rule exists for.

Worth recording what did **not** change: tool-use discipline was already fine.
Asked *"how many rows are in the Assamese split?"*, both prompts called the tool
rather than reciting the card. The `Facts` rules earn their place on how a result
is *reported*, not on whether one is fetched.

## How other agents reach it

One tool, `query_dataset`, in the same OpenAI schema shape as Composio's, merged into the
list `_use_tools` already builds ([13-connectors.md](13-connectors.md)). A second loop would
have made "read my mail" and "query my dataset" alternatives within one turn rather than
both available.

- **No datasets, no tool, no cost.** `tools_for` returns empty and the tool pass is skipped.
  Somebody who has never added one pays nothing — the pass is buffered (`llm.complete`), and
  buffering is what the voice path spends its latency budget avoiding.
- **The ids are an `enum` built from what this user has**, so a hallucinated id is refused by
  the provider rather than becoming a lookup that fails a round trip later.
- **The SQL is written inside.** The outer model is answering in speech and has never seen
  the column list; the schema card is thousands of characters and belongs only on turns that
  query. The tool takes English, `DatasetAgent` writes the SQL, and the query comes back in
  the result so the answer stays checkable.

**One repair, not a loop.** A first failure is usually a half-remembered column, and DuckDB's
binder says *"Did you mean …?"* — handing that back fixes it in one round. A second failure
means the question cannot be answered from these columns, and further attempts are the model
trying different wrong answers while somebody waits.

**Failure is an answer.** No model, a rejected statement, a query that will not run — all
come back as an `Answer` carrying the reason and the SQL, never as an exception. A tool that
returned an empty row set on failure would teach an agent to report "no results" for "the
database is down."

## Measured, end to end

Against `MSMARCO-XI/validation`, 3 splits × 25,000 rows, `gpt-5.4-mini` writing the SQL:

| question | attempts | latency | result |
| --- | --- | --- | --- |
| how many queries of each type in the Assamese split | 1 | 1.3 s | `GROUP BY query_type` — 4 rows |
| 3 English queries mentioning diabetes with Bengali translations | 1 | 2.4 s | `ILIKE '%diabetes%'` — 3 rows |
| the full passage text for `query_id` 128 | 1 | 1.5 s | returned the available columns; did **not** invent `passages` |

Query execution itself is **~5–7 ms** — local, sealed, and the reason the pull is worth
paying for once.

## Multiplicity, and what belongs to whom

`PRIMARY KEY (user_id, dataset_id)` where `dataset_id` is derived from the URL: a person may
attach up to `dataset_max_per_user`, and re-adding the same URL replaces rather than
accumulates. Files are keyed by user as well as dataset — two people who add the same public
dataset get a file each. Sharing one would be a cache that leaks whichever sample size and
column budget were in force for whoever added it first, and would tie one person's rebuild to
another person's query.

Every route requires a verified Clerk session, the same rule as
[13-connectors.md](13-connectors.md) but for a weaker reason: a dataset holds no credential,
and only public URLs are accepted. It is that this deployment downloads and stores the file,
and anonymous callers attaching those is an open invitation to fill the disk.

**A profile is only served for a file that exists.** The row and the file are separate state
and a redeploy onto a fresh disk breaks the second without touching the first. `queryable()`
checks the file, not the status, and a missing one schedules a rebuild from the URL still in
the row — so it heals rather than needing somebody to notice.

## Settings

| | default | |
| --- | --- | --- |
| `dataset_sample_rows` | 25,000 | rows per table, a prefix |
| `dataset_column_budget_mb` | 32 | the largest dial in the feature — see the table above |
| `dataset_max_tables` | 12 | files per source; the remainder is announced, never dropped silently |
| `dataset_max_per_user` | 10 | each is a file on disk and a block in a system prompt |
| `dataset_card_chars` | 1,000 | paid every turn |
| `dataset_schema_chars` | 6,000 | paid only on turns that query |
| `dataset_query_rows` | 200 | a prompt budget more than a memory one |
| `dataset_query_timeout_s` | 10 | only ever catches an accidental cross join |
| `dataset_sql_repairs` | 1 | see above |
