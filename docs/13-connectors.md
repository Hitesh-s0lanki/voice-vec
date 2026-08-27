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
| Postgres + pgvector | `vector` | Same, against their own Postgres |

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

So a fifth connector is [`registry.py`](../src/connectors/registry.py) plus — for a vector
store — a backend under [`src/rag/backends/`](../src/rag/backends/). No frontend change.

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
signed out            ─► the deployment's own store
nothing connected     ─► the deployment's own store
Pinecone connected    ─► theirs
several connected     ─► PREFERENCE order, so it is the same one every request
connected but empty   ─► the deployment's own store
anything goes wrong   ─► the deployment's own store
```

Two rules that are cheap to state and expensive to get wrong. **Fall back, never cross
over**: the only per-user thing in that module is a credential lookup that takes a user id,
so a failure resolves to the *deployment* store and never to somebody else's. And **a broken
connector degrades retrieval rather than deleting it** — `for_user` never raises.

**Built is not the same as searchable.** `verify` runs once, when the form is submitted, and
an index can be emptied, dropped or renamed long after that. So a freshly built backend is
probed with `ready()` before it is used, and one that answers no falls through to the
deployment store — which is the difference between a degraded answer and *"my sources are
unavailable"* on every question. The probe is `SELECT 1 … LIMIT 1`, not `count(*)`, because
it runs against somebody else's database and a cost that grows with their corpus is a cost
that times out on the indexes most worth connecting.

Backends are cached per user and keyed by the sealed credential blob, so rotating a key
rebuilds rather than serving a revoked one. A failed probe is cached the same way — one
round trip per connect, not one per question — and reconnecting re-seals the credentials
under a new blob, so the probe runs again the moment somebody fixes what was wrong. The cache is bounded and eviction closes the
backend, which matters for pgvector: it holds a connection pool against somebody else's
database, sized to 2 because this app is a guest in it.

## What the vector backends do and don't do

`VectorStore` in [`store.py`](../src/rag/store.py) does ingest, schema, index building,
warming and counting as well as searching. Three of those are meaningless for an index
somebody else populated, and one — search — is all the answer path calls. So
[`base.py`](../src/rag/backends/base.py) is deliberately narrow: `search`, `ready`,
`describe`. That is what makes a backend a hundred lines instead of a second ingest pipeline.

**Ingest is still deployment-side.** [`scripts/ingest.py`](../scripts/ingest.py) writes to
the app's own Postgres. A user who connects Pinecone is searching an index *they* populated,
and the schema it has to satisfy is the metadata read in each backend's `_hit`. Per-user
ingest is not built.

For pgvector that schema is not a convention, it is the search: `PgVectorBackend` forwards
to `VectorStore`, so the table it points at must be the one `ingest.py` builds —
`chunk_key`, `strategy`, `text`, `meta`, `language` and `embedding` / `embedding_en` at the
app's embedding dimension. `verify_pgvector` reads the catalogue and says which of those are
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
it. [`agent.py`](../src/integrations/agent.py) turns a user's live connections into tool
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
message can be re-read; an email is sent. So [`tool_calls`](../src/chat/tools.py) records
every one, for three readers: the user (the thread shows what ran, under the turn that
caused it), the operator (a tool failing for everybody looks like the model being unhelpful
until a table says otherwise), and anyone auditing an agent with access to a mailbox.

| Stored | Not stored |
| --- | --- |
| slug, toolkit, arguments, status, error, latency, `user_id`, `turn_id` | **the result** — only its size |

**Arguments in, results not.** The arguments are what the agent *decided*, which is the
interesting half and is small. A result can be an entire inbox page and belongs to the
provider: storing it would turn an audit trail into an uncontrolled copy of everything the
agent has ever read, which is a worse thing to hold than the credential that reached it.
`ToolCall` on the wire has no field that could carry one, and a test asserts that.

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
