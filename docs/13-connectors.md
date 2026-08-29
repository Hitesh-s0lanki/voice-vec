# 13 — Connectors

*The outside services a user attaches to their account — and, for a vector store, where their questions get answered.*

There is no shared Composio account behind this app, and no shared Pinecone. Each signed-in
person attaches **their own**, and everything done on their behalf runs inside their own
project. Two people share no connector state at all.

| Connector | Kind | What it does once attached |
| --- | --- | --- |
| Composio | `tools` | Link Gmail, Slack, Notion — through *their* Composio project |
| Pinecone | `vector` | Their questions are searched against their index |
| Astra DB | `vector` | Same, against a collection in their Astra database |
| Postgres + pgvector | `vector` | Same, against any pgvector table in their own Postgres |

```
panel ─► PUT /connectors/{slug} {values}
              │
              ├─ spec.clean   drop undeclared keys, insist on required ones
              ├─ spec.verify  one real authenticated call — nothing stored yet
              ├─ seal         Fernet over the whole credential set
              └─ store        connector_accounts (user_id, connector)
```

## Adding a connector is a backend change

A `ConnectorSpec` ([`spec.py`](../src/connectors/spec.py)) carries enough about a service —
name, kind, fields, which fields are secret, how to verify — that the panel can render a
correct form for a connector it has never heard of. The field list goes over the wire and
the React form is built from it.

The alternative, a component per service, means every connector touches two languages, and
means the form and the server-side validation drift the first time either is edited.

So a new connector is [`registry.py`](../src/connectors/registry.py) plus — for a vector
store — a backend under [`src/rag/backends/`](../src/rag/backends/).

**No frontend change for a new connector; one line for a new *kind*.** That
distinction was found rather than designed. The panel renders a section per kind
from a `GROUPS` array and filters rows into it, so `dataset` arriving as a third
kind was a connector the server offered and the panel silently dropped — not an
error, not an empty section, just absent. A fifth *vector* connector still costs
no frontend edit; a third kind costs `ConnectorKind` in
[`connectors.ts`](../frontend/src/lib/connectors.ts) and an entry in `GROUPS`,
and the two are commented to be edited together.

`secret` is the flag that matters most. It decides three separate things: the input renders
as a password, the value is never sent back to the browser, and only its last four
characters are kept readable. A field that carries a credential and is not marked secret is
stored in the clear in `hints` and echoed to the browser, and nothing else looks wrong —
which is why [`test_connectors.py`](../tests/test_connectors.py) pins the exact secret set
per connector rather than inferring it. (`keyspace` contains "key" and is a namespace name.)

## Verify before storing

Every `verify` makes one cheap, **real, authenticated** call — not a ping, not a format
check. The failure worth catching is "well-formed and wrong", which no regex finds. Doing it
before anything is written means a credential that cannot answer is never encrypted and
kept, and the user gets one clear message instead of every later action failing opaquely.

Two details that would otherwise bite:

- **Pinecone** describes the index rather than just authenticating, so a wrong key (401) and
  a wrong index name (404) are told apart. Sending someone to re-check a key when the index
  name has a typo costs an afternoon.
- **Astra's Data API answers 200 with an `errors` array.** A bad token is a successful HTTP
  response, so `raise_for_status` alone reads it as success. Both the verify and the search
  path check the body.

## Credentials at rest

Sealed with Fernet before they reach Postgres ([`crypto.py`](../src/connectors/crypto.py)),
under a master key that lives in the environment and never in the database — the two have to
be stolen separately. `cryptography` is already a dependency via `pyjwt[crypto]`, so this
costs no new package.

Sealed as **one blob per connector**, not field by field: a Pinecone key without its index
name is not usable, so there is nothing gained by separating them, and one blob means one
place where plaintext exists.

`hints` beside it is the readable half — non-secret fields as typed, plus the secret's last
four characters — so the panel can say `vec-chunks · docs · ····8fa2` without decrypting
anything, and an operator can tell rows apart without the master key.

**Rotating `COMPOSIO_ENCRYPTION_KEY` is not a migration.** Every credential stops decrypting,
`open_map` returns `None` rather than raising, the panel shows `stale`, and each user
reconnects. (The variable is still named for Composio because it was already deployed under
that name; it seals every connector.)

## Which store answers a question

[`resolve.py`](../src/rag/backends/resolve.py) decides, per request:

```
signed out            ─► nothing
nothing connected     ─► nothing
Pinecone connected    ─► theirs
several connected     ─► PREFERENCE order, so it is the same one every request
connected but empty   ─► nothing
anything goes wrong   ─► nothing
```

`nothing` is `None`, and it is a real answer rather than an error. **There is no deployment
corpus behind this app.** There used to be — a `chunks` table built by an ingest script from
one dataset, which everybody fell back to — and it is gone. What a question can be answered
from is exactly what its asker attached.

So the two rules restate as one: **return nothing, never somebody else's.** The only
per-user thing in that module is a credential lookup that takes a user id, and there is no
shared store left to reach for, so a failure cannot resolve to another account's index.
`for_user` still never raises: `/ask` turns `None` into *"I don't have a source to search
yet — connect a vector store and I can answer from it"*, which is a true and actionable
answer where a 500 would read as the app being broken.

**Built is not the same as searchable.** `verify` runs once, when the form is submitted, and
an index can be emptied, dropped or renamed long after that. So a freshly built backend is
probed with `ready()` before it is used, and one that answers no is treated as nothing
attached — which is what turns it into *"connect a vector store"* rather than an error from
a layer the asker cannot place. The probe is `SELECT 1 … LIMIT 1`, not `count(*)`, because
it runs against somebody else's database and a cost that grows with their corpus is a cost
that times out on the indexes most worth connecting.

Backends are cached per user and keyed by the sealed credential blob, so rotating a key
rebuilds rather than serving a revoked one. A failed probe is cached the same way — one
round trip per connect, not one per question — and reconnecting re-seals the credentials
under a new blob, so the probe runs again the moment somebody fixes what was wrong. The cache is bounded and eviction closes the
backend, which matters for pgvector: it holds a connection pool against somebody else's
database, sized to 2 because this app is a guest in it.

## What the vector backends do and don't do

`VectorStore` in [`store.py`](../src/rag/store.py) used to do ingest, schema, index
building, warming and counting as well as searching. All of those were meaningless for an
index somebody else populated, and one — search — is all the answer path ever called. They
are gone with the deployment corpus, and what is left is the query half, whose only caller
is `PgVectorBackend`. [`base.py`](../src/rag/backends/base.py) is narrow for the same
reason: `search`, `ready`, `describe`.

**Nothing is ingested here at all.** This app reads connected stores and writes to none of
them; `DATABASE_URL` holds conversations, credentials, profiles and datasets, and is never
searched. A user who connects Pinecone is searching an index *they* populated, and the
metadata it has to carry is what each backend's `_hit` reads.

For pgvector the schema is not a convention, it is the search: `PgVectorBackend` forwards
to `VectorStore`, and the column names come from a `ColumnMap` discovered at verify time
([`columns.py`](../src/rag/columns.py)) rather than being assumed — so a table holding `id`
and `chunk_text` is searchable, and a table missing `tsv` loses the lexical channel instead
of being rejected. `verify_pgvector` reads the catalogue and says what is
missing while the form is still open, rather than letting a Postgres full of somebody's own
`book_chunks` connect green and abstain on everything. `tsv` and `tsv_en` are deliberately
*not* required: a table built by an older migration has none, and rung 2 already degrades a
missing lexical channel to dense-only.

`Hit` is imported from `store.py` rather than redefined, because `rendering()` — the
cross-lingual fallback deciding what an answer is cut from — is behaviour every backend must
share, not copy.

**Latency.** In-process pgvector measured ~11 ms; a hosted index is a network round trip
inside the same 200 ms as everything else ([04-latency.md](04-latency.md)). Query timeouts
are 2 s so the harness degrades rather than holding a worker.

## Only when signed in

Every `/connectors` route requires a **verified Clerk session token**; the account is the
`sub` of a signature that checked out. Nothing else names a user — not a header, not a query
parameter, not a body field. Credentials arrive in a **JSON body**, never a query parameter
or path segment, because those end up in access logs and browser history.

That is stricter than [12-conversations.md](12-conversations.md), deliberately: a saved
conversation belongs to whoever holds the browser, but a credential does not.

`/ask` is the exception and stays open — it answered anonymously before connectors existed
and still does. Identity there is optional and only decides *which store answers*.

## The agent actually calling tools

Linking a toolkit is not the point on its own — the point is that a spoken turn can *use*
it. [`tool_agent.py`](../src/agents/tool_agent.py) turns a user's live connections into tool
schemas, and [`_use_tools`](../src/services/voice_service.py) runs the loop:

```
decide ─► run ─► decide ─► speak
   │        │       │
   │        │       └─ no more calls: stream the answer
   │        └─ execute as that user, record the call
   └─ llm.complete(), buffered — see below
```

**The decide pass is buffered, and that costs latency.** A tool call cannot be streamed into
a synthesiser: the arguments arrive in fragments across many chunks and mean nothing until
the last one. So `llm.complete()` exists alongside `stream_reply()`, and only the final
answer — once the tools have run — is streamed.

Which is why the very first thing `_use_tools` does is leave. **A user who has linked nothing
pays nothing**: no round trip, no schema in their prompt, no change to the numbers in
[11-voice.md](11-voice.md). That is every session until somebody opens the Connectors panel.

**Only what they linked.** Schemas are fetched for the toolkits that reconciled as `ACTIVE`
for that user, read from our own table so it costs no round trip on a path where somebody is
waiting. Handing the model Composio's whole catalogue would be a prompt the size of a book
and an invitation to call something nobody authorised.

**Execution is always scoped to the speaker.** `execute` passes `user_id` to Composio, so a
tool runs against the connected account belonging to the person who spoke. There is no
ambient account to reach for by mistake.

Three failure rules, each of which keeps a turn alive:

- A tool that raises becomes a result with `ok=False`, never an exception. The turn is
  mid-sentence and the listener is waiting.
- A failure is *told to the model* rather than hidden, so it can say "I couldn't reach your
  mailbox" instead of inventing an answer from a silence.
- Composio reports failure in the response body, not by raising — a 200 with
  `successful: false` is a failed tool, and treating it as success is the quiet bug.

The subtlest one has its own test. When the model stops asking for tools, the loop must hand
on the messages **including the tool results** — returning the original list would run
somebody's tools and then answer from a prompt that never mentions what came back. Nothing
about that turn sounds wrong, which is exactly why
[`test_agent.py`](../tests/test_agent.py) pins it.

Bounded by `tool_max_rounds` (2 — enough for "search, then send"; more is usually a loop)
and `tool_timeout_s`. `TOOLS_ENABLED=false` turns the whole thing off without disconnecting
anybody.

## What gets written down

A tool call is the one thing a spoken turn does that has an effect *outside* this app. A
message can be re-read; an email is sent. So [`tool_calls`](../src/chat/tool_calls.py) records
every one, for three readers: the user (the thread shows what ran, under the turn that
caused it), the operator (a tool failing for everybody looks like the model being unhelpful
until a table says otherwise), and anyone auditing an agent with access to a mailbox.

| Stored whole | Stored bounded | Not stored |
| --- | --- | --- |
| slug, toolkit, status, error, latency, `user_id`, `turn_id` | arguments, and the **head** of the result plus `result_bytes` | the rest of a result past `MAX_RESULT_CHARS` |

**Both halves of the call, and the second one capped.** The arguments are what the agent
*decided*; the result is what came *back*, and without it the thread cannot be read —
"it ran `GMAIL_FETCH_EMAILS`" says nothing about what the answer was built from. So the
first 4,000 characters go down beside `result_bytes`, which stays the size of the *whole*
thing: the card reads "first 4013 of 6.0k chars" rather than implying it has all of it.

The ceiling is the containment. A result can be an entire inbox page and belongs to the
provider; a preview is what keeps this table from becoming an uncontrolled copy of
everything the agent has ever read. It is still somebody's mail — read back only by the
account that owns the conversation, and worth the same care as the credential that
reached it.

Arguments are bounded too — an oversized one is replaced by a marker rather than the call
being dropped, because knowing `GMAIL_SEND_EMAIL` ran is most of the record's value.
`user_id` is denormalised so the audit survives the conversation being deleted, and writes
go through the same queue as the messages, so a dead database still cannot break a turn.

## Composio's second step

Composio is the one connector that is not finished when its credentials verify: its project
is a doorway to further consent screens. That flow lives in
[`src/integrations/`](../src/integrations/) and is documented by its own module docstrings —
the auth configs keyed `(user_id, toolkit)`, the reconcile-and-revoke rules, and why it calls
`link()` rather than the `initiate()` Composio retired for managed OAuth on 2026-07-03.

Disconnecting Composio drops its toolkit rows, because they name ids inside a project this
app can no longer reach. It revokes **nothing** upstream: those connections live in the
user's own account and stay theirs.

## Their schema, not ours

`verify` used to require the connected table to carry this app's exact columns —
`chunk_key`, `strategy`, `text`, `meta`, `language`, `embedding_en` — because
[`store.py`](../src/rag/store.py)'s queries named them as literals. That made "connect your
own Postgres" mean "connect a copy of ours": a working pgvector table holding `id` and
`chunk_text` was turned away at the form for lacking columns its owner had never heard of.

[`columns.py`](../src/rag/columns.py) makes the names data. A `ColumnMap` says which column
plays each role, `verify` discovers one from the catalogue, and it is sealed beside the DSN
under `col_*` keys. The bar is now **readability, not sameness**:

| Role | Required? | Absent means |
| --- | --- | --- |
| `embedding` | yes | nothing to search |
| `text` | yes | nothing to quote an answer out of |
| `strategy` | no | the predicate is *dropped* — `filters` is false |
| `tsv` | no | no keyword channel — `lexical` is false |
| `english`, `embedding_en` | no | no native English retrieval |
| `meta` | no | up to six scalar columns are carried instead, so a hit can cite its source |

**An absent column is a lost capability, never a substituted one.** A dropped predicate is
honest; one defaulted to match everything is a narrowing the ladder believes it applied.
`Capabilities` is read off the map, so rung 2 is told dense-only before it asks.

The default map reproduces the schema this app used to ingest into, and generates the query
it replaced *character for character* — pinned by
[`test_columns.py`](../tests/test_columns.py), because that query is the measured hot path
and a silent SQL change in it is a latency regression nobody could attribute. Those literals
are gone from `store.py`; the constants in the test are pasted rather than imported, which
is exactly why they outlived them.

## Every case a connected store can present

Enumerated once, rather than discovered one connection at a time.

| What differs | How it is handled |
| --- | --- |
| Table field left blank | discovered — one vector table is used, several are named so you can pick |
| No table with vectors | said as a fact about *their* database, not about our default |
| Column names | `ColumnMap`, discovered from the catalogue at verify time |
| No `strategy` / `language` | predicate **dropped**; `filters` false |
| No `tsv` | `lexical` false — rung 2 told before it asks |
| No English pair | `parallel_text` false; an English question takes the cross-lingual hop |
| No `meta` | up to six scalar columns carried, so a hit can cite its source |
| No text column | rejected — searchable but unquotable is not usable |
| `halfvec` instead of `vector` | accepted; same search, half the index |
| Distance is l2 or inner product | operator **and** score expression read off the index opclass |
| No ANN index at all | works; the profile notes the sequential scan |
| **Width differs** | query embedded remotely at exactly that width — nothing asked |
| Store built by a different model | caught by a round trip; the store is marked unroutable |
| Semantic cache width | semantic half skipped, exact half kept |
| Empty index | `searchable` false, with the reason |
| Table in another schema | `schema.table` accepted; the picker lists qualified names |
| Mixed-case or spaced table name | quoted in every generated query |

## Which tooling's tables actually work

Pinned as regression tests in [`test_profiles.py`](../tests/test_profiles.py), because these
are what people connect:

| Built by | Text | Id | Metadata |
| --- | --- | --- | --- |
| LangChain (`langchain_postgres`) | `document` | `id` | `cmetadata` |
| LangChain (community, legacy) | `document` | `uuid` | `cmetadata` |
| LlamaIndex `PGVectorStore` | `text` | `id` | `metadata_` |
| Supabase quickstart | `content` | `id` | `metadata` |
| pgai vectorizer | `chunk` | `id` | — (`chunk_seq` carried) |
| Django / hand-rolled | `body` | `id` | — (`title` carried) |
| This app | `text` | `chunk_key` | `meta` |

**Names are a hint; the data is the answer.** Writing that table out is what exposed the bug:
in LangChain's schema `id`, `collection_id` and `document` are all `character varying`, and a
name-only heuristic picked `id` — so every answer from the single most widely used pgvector
integration would have been quoted from a UUID. Verify now reads five rows and takes the
column whose values are actually prose. An empty table falls back to the names, which is
still right most of the time.

Two other cases are worth their own paragraph.

**Width.** A 768-dimensional index and a 384-dimensional query are not comparable, and
nothing after the fact reconciles them. OpenAI's `text-embedding-3` models take a
`dimensions` parameter, so the width read from the store's own catalogue *is* the answer:
[`remote_embed.py`](../src/rag/remote_embed.py) asks for exactly that many dimensions and
the form asks nothing.

The first attempt took a model name on the form instead. That was wrong twice. The question
was unanswerable by the person being asked — the reasonable reply to "stores 768-dimensional
vectors" is `768`, which is what happened, and fastembed spent 39 seconds trying to download
a HuggingFace repository by that name. And it could not serve the common case anyway: 1536
and 3072, which is what most connected stores are actually built at, have no locally
loadable model at all.

The cost is honest and it is real: **a network call on the answer path**, which
[04-latency.md](04-latency.md)'s zero-network-calls rule otherwise forbids. It is confined to
connected stores of another width — a connected Pinecone or Astra is already a round trip —
and measured from `ap-southeast-1` it ran 0.4–2.5 s, dominated by provider variance. The
harness deadline is what keeps it honest.

**A store built by a different model** is the case that looks handled and is not. Matching
the width makes the arithmetic legal, not the answer meaningful — a store embedded with
Gemini or bge at 768 returns neighbours of a query vector that has nothing to do with it. The
profiler catches it with a round trip on **every** connected vector store, whatever its
width: lift a passage out of the store, embed it the way queries are embedded, ask the store
for it back. Built the same way, it retrieves itself near 1.0. Measured on a real one that
was not:

```
a passage from this store retrieves itself at only 0.06 —
it was not built with the embeddings this app queries it with
```

That gates routing, not just the card. A false negative costs an abstention that names the
store; a false positive is confident garbage nobody can detect.

## Connecting is not understanding

A verified credential proves the store answers. It says nothing about what is in it, and
every backend has hard-coded its answer to "what can this store do" since connectors
shipped — `filters=True` for every Pinecone index on earth, on the hope that it carries a
`strategy` field.

[17-understanding.md](17-understanding.md) is the layer that measures instead: a probe
samples the connected store, a model names what it holds, and the result is stored so the
agent reads a paragraph rather than a guess. Adding a fifth connector means adding a probe
beside its backend — [`src/connectors/probes/`](../src/connectors/probes/) — or accepting
that its capabilities stay declared rather than measured, which is the same behaviour this
app had before.

## Configuration

```bash
# Generate with:
#   uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
COMPOSIO_ENCRYPTION_KEY=
FRONTEND_URL=http://localhost:3002   # where Composio's consent lands
CLERK_PUBLISHABLE_KEY=pk_test_…      # a prerequisite, not an extra
```

There are deliberately **no service API keys here**. Users bring their own.

Without `CLERK_PUBLISHABLE_KEY` nobody is ever signed in, so every connector route 401s.
Without `COMPOSIO_ENCRYPTION_KEY` the panel says the server cannot store credentials — a
different sentence from "you have not connected anything", and only one of the two is the
user's to fix.

`uv run python -m scripts.migrate` creates every table.

```bash
PROFILE_ENABLED=true       # probe a store when it is connected
PROFILE_EXCERPTS=true      # may passages of the user's data reach the summarising model
PROFILE_NARRATE=true       # write a summary at all, or keep the profile to measurements
PROFILE_TTL_HOURS=24       # how long an understanding is trusted before a refresh
```
