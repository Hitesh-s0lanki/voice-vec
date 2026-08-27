# 12 — Conversations

*Where a spoken turn goes after it has been heard.*

Before this, a conversation lived exactly as long as the socket carrying it. History was a
list on `VoiceSession`, the browser kept its own copy in `localStorage`, and a refresh was
the end of both: the model forgot what it had just said, and the panels forgot what you had
just asked. Now the turn is written to Postgres as it happens, the URL names the
conversation it belongs to, and reloading that URL hands the model its own past back.

```
first sentence ─► conv_9f3a… minted ─┬─► the client puts /c/conv_9f3a… in the address bar
                                     └─► INSERT conversations, then every message
```

## The two tables

They live in the same Neon database as the chunks, on the same pool. The vector store's own
docstring is why: *"somewhere to put the rest of the product — documents, tenants, saved
turns — next to the chunks rather than in a second system."* Conversations differ from
chunks in one way that mattered, though — they are needed whether or not retrieval is on —
so the connection pool moved out of [`src/rag/store.py`](../src/rag/store.py) and into
[`src/core/db.py`](../src/core/db.py), and both check out of it.

| Table | Holds |
| --- | --- |
| `conversations` | one row per conversation: owner, title, language, take count, timestamps |
| `messages` | one row per question and per reply, `ON DELETE CASCADE` to its conversation |

`turns` counts **questions, not messages**, so the rail can say "4 takes" without dividing
by two and guessing about the half-turn a barge-in leaves behind. `title` is filled from the
first question by `COALESCE` on insert, which means a rename is never overwritten by the
next thing said.

## Who owns one

```
conversations ─┬─ user_id     the Clerk account, from a verified token
               └─ session_id  the browser's own id, before anyone signs in
```

A visitor with no account is known by an id the browser mints for itself — `sess_` and a
dash-free UUID, kept in `localStorage`
([`frontend/src/lib/identity.ts`](../frontend/src/lib/identity.ts)) — sent as
`x-session-id` over HTTP and `?session=` on the socket. A signed-in visitor is known by
their Clerk user id. The ownership predicate matches a row on **either** column:

```sql
AND ((%(user_id)s::text IS NOT NULL AND user_id = %(user_id)s)
  OR (%(session_id)s::text IS NOT NULL AND session_id = %(session_id)s))
```

…which is why signing in *widens* what you can see rather than hiding what you just said.
The predicate lives in the SQL rather than in the service on purpose: no code path can read
a conversation by forgetting to check.

A conversation that is gone, and one that was never yours, return the same 404. Telling
those two apart tells a stranger which ids exist.

> The session id is **not a credential**. Anyone holding it can read the conversations
> opened under it — the same guarantee as "anyone holding this browser", no more. It is
> why an account is never established from one.

### The account half is a token, never an id

There is no `x-user-id` header and no `?user=` parameter, and there should never be one.
This server is reachable from the browser directly — that is the whole point of the voice
socket — so a user id sent as a header or a query parameter is a value anyone can type, and
nothing downstream could tell the difference between it and the truth.

Identity is a **Clerk session token**, verified in
[`src/core/clerk.py`](../src/core/clerk.py) against Clerk's own signing key, and the account
is the `sub` claim of a signature that checked out. Two transports, one rule:

| Transport | Carries the token as | Where it comes from |
| --- | --- | --- |
| `/api/conversations/*` → FastAPI | `Authorization: Bearer …` | the route handler's `await auth()` + `getToken()`, server-side |
| `WS /voice/ws` | `?token=` | `useAuth().getToken()`, fetched per connection |

The Next route handlers never forward an identity the browser asked them to send; they mint
a token from the session Clerk has already verified. The socket has to use the query string
because a browser cannot set headers on a WebSocket handshake — acceptable because Clerk's
session tokens live about a minute and a fresh one is fetched per connection, which is also
why signing in reopens the socket rather than waiting for the next one.

The signing key is found from `CLERK_PUBLISHABLE_KEY` alone: the instance host is base64'd
into it, the JWKS lives at a well-known path under that host, and the fetched keys are
cached for an hour. Measured here: 249 ms for the first token of a process, 0.2 ms
thereafter — and off the event loop, so it is not 249 ms every other socket waits through.
`CLERK_JWT_KEY` pins the PEM instead, for a deployment that would rather make no outbound
call at all. Set neither and verification is off: every caller is anonymous, and the voice
loop is identical.

**A token that does not verify makes the caller anonymous — it does not fail the request.**
Expired, forged, from another instance, absent: all the same answer, and all still served.
Anything else would put a broken microphone in front of someone whose token lapsed while
they were reading.

### Signing in claims what you already said

```sql
UPDATE conversations SET user_id = %(user_id)s
WHERE session_id = %(session_id)s AND user_id IS NULL
```

One statement, at `POST /conversations/adopt`, called once per account per browser by
[`use-adoption.ts`](../frontend/src/hooks/use-adoption.ts). Which account is claiming is
**not** in the body and cannot be — it comes from the verified token — so the worst a forged
call achieves is handing someone their own conversations again.

`user_id IS NULL` is what makes it both idempotent and safe on a shared browser: it hands
over what nobody has claimed, and nothing else. Verified end to end — two conversations held
by a browser, adopted on sign-in, then visible from the token alone with no session id
anywhere near the request; a second call moved 0; a different account moved 0.

## The id, and the URL

`conv_` followed by 32 hex characters. Prefixed and dash-free because it is read aloud in
support conversations and pasted into shells, both of which `conv_9f3a…` survives better
than a bare UUID. `is_conversation_id` is the same cheap shape check on both sides, so a
junk path segment never reaches the database.

The interesting part is *when* the URL changes. The conversation is opened on the first
transcript — after the "is this longer than two characters" guard, so a take that caught a
cough leaves nothing behind — and the id is **minted in Python rather than read back from
the insert**, so the client has its address a round trip earlier. The insert is queued
before any message, and the queue is serial, so the foreign key is satisfied by
construction even though nothing waited for it.

The client then calls `window.history.replaceState`, **not** `router.replace`. A real
navigation would remount the page, close the socket, and cut off the reply that is being
spoken at that exact moment. Next.js syncs `replaceState` into `usePathname`, so the rest
of the app sees the new id without anything unmounting.

## Storage is never in the turn's way

Every write goes onto an `asyncio.Queue` that a separate task drains through a worker
thread. Three properties fall out of that:

- **A slow database costs the listener nothing.** The round trip to `ap-southeast-1`
  measures ~77 ms; no spoken word waits on it.
- **Order is guaranteed.** The rows have one — the conversation before its first message,
  the question before its answer — and a pool running them concurrently would be free to
  invert it.
- **A dead database is a no-op.** Failures are logged and dropped. `persists` is false for a
  checkout with no `DATABASE_URL` and for a client that sent no identity; in both cases the
  voice loop runs exactly as before and simply leaves no trace.

`aclose` gives the queue a bounded five seconds to land the last reply — the socket usually
closes milliseconds after the turn that ended it — and then stops waiting, because a hung
database must not hold a connection open.

## What is stored for a turn that did not work

`_remember` is called on every way out of a turn, which is why the status lives there: it is
the one place that sees success, barge-in and failure alike.

| `status` | Means |
| --- | --- |
| `answered` | it finished |
| `interrupted` | you talked over it — and what is stored is **only what reached the speakers** |
| `error` | a provider failed mid-reply; `reason` carries what it said |
| `abstained` / `refused` | the `/ask` outcomes, for when retrieval is answering |

The barge-in case is the one worth reading twice. `turn.spoken` is what was actually heard,
not what was generated, so neither the model's history nor the stored row can refer back to
a sentence that was cut off mid-word.

## Picking a conversation back up

`bind` runs before the socket says `ready`: one indexed read for the row (which also proves
it belongs to whoever is asking) and one for the **tail** of its messages, seeded into
`VoiceSession.history`. Verified end to end — a fresh socket, told a word in one connection,
still knew it in the next.

The tail, not the head: `LIMIT` on an ascending order would return a long conversation's
opening and drop everything said since, which is the half nobody wants. The store selects
the newest rows and puts them back in order.

An id that is junk, deleted, or someone else's does not fail the connection — the session
stays unbound, the thread loads empty, and the next thing said opens a new conversation.
That is a far better answer to a stale bookmark than a socket that refuses to open in front
of someone holding a microphone.

## The surface

| Endpoint | What it does |
| --- | --- |
| `GET /conversations` | yours, newest first, empty ones hidden |
| `POST /conversations` | open one up front — the socket does this itself on the first take |
| `GET /conversations/{id}` | the conversation and its messages |
| `PATCH /conversations/{id}` | rename |
| `DELETE /conversations/{id}` | it and its messages, by cascade |
| `POST /conversations/adopt` | claim this browser's conversations for the signed-in account |
| `WS /voice/ws?session=&token=&conversation=` | the spoken turn, written into that conversation |

The browser reaches all of it through Next route handlers under `/api/conversations`, which
are same-origin (no CORS preflight in front of a panel that opens on a click) and are where
the backend's address stays a server-side detail.

In the UI: **History** now lists conversations and is how you get to a different one — a
real navigation, because a different conversation needs a different socket.
**Conversations** shows the turns inside the one on screen, read back out of Postgres. That
is what makes the answer never being printed on the stage tenable: it was *heard*, and this
is where it can be re-read tomorrow, on another device.

## What is not here yet

- **Organisations.** Clerk has them; a conversation belongs to a person. A third column and
  the same predicate is the shape of it.
- **Anything to stop one account filling the table.** No quota, no retention window.
- **Streaming a turn into the panel as it is spoken.** The optimistic copy appends on
  `turn.end`, so the thread on screen is one turn behind the database for about a second.
