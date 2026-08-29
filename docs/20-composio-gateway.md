# Composio's two keys, and the second transport

## The bug this started as

A `ck_…` key pasted into the Composio connector was rejected with:

> Composio rejected that API key. Check it in your Composio dashboard.

The key was fine. The message was not, and neither was the assumption under it.

## Composio ships two credentials for two products

They are not interchangeable, and nothing about their shape says so.

| Key | Endpoint | Header | What it opens |
|---|---|---|---|
| `ak_…` | `backend.composio.dev/api/v3` | `x-api-key` | The REST API the Python SDK wraps |
| `ck_…` | `connect.composio.dev/mcp` | `x-consumer-api-key` | One JSON-RPC endpoint, seven meta-tools |

Measured, not assumed — the same `ck_` key, against both:

```
200  connect.composio.dev/mcp          → initialize OK, 7 tools
401  backend.composio.dev/api/v3       → {"code":801,"slug":"APIKey_InvalidAPIKey"}
```

There is no base URL that serves REST to a `ck_` key. `connect.composio.dev/api/v3`
answers `307` to a dashboard login page. So a gateway key cannot be made to work
through the SDK by configuration — it needs a different transport, which is what
[`src/integrations/mcp.py`](../src/integrations/mcp.py) is.

## What maps onto what

The gateway exposes meta-tools rather than resources, so every operation is
re-expressed:

| SDK (`ak_`) | Gateway (`ck_`) |
|---|---|
| `toolkits.list` | `COMPOSIO_SEARCH_TOOLS` — semantic, **not** a listing |
| `tools.get` | `COMPOSIO_GET_TOOL_SCHEMAS` |
| `tools.execute` | `COMPOSIO_MULTI_EXECUTE_TOOL` |
| `connected_accounts.list` | `COMPOSIO_MANAGE_CONNECTIONS` `action=list` |
| `connected_accounts.link` | `COMPOSIO_MANAGE_CONNECTIONS` `action=add` |
| `connected_accounts.delete` | `COMPOSIO_MANAGE_CONNECTIONS` `action=remove` |
| `auth_configs.*` | — nothing, and nothing is needed |

The last two rows are the interesting ones.

**`add` returns a consent URL directly.** The gateway owns its auth config, so
the gateway branch of `connect()` is four lines where the SDK branch is forty:
there is no `ac_…` to find, create, remember or repair.

**There is no catalogue call.** Only a semantic search returning a handful of
tools per query. This is the one real capability loss — see below.

## Where the two paths differ, honestly

Three differences are visible to a user and are not worked around, because
working around them would mean inventing data:

1. **The catalogue is search-shaped.** A gateway user's toolkit grid comes from
   a semantic search — no cursor, no paging, and no description, logo or tool
   count, because the gateway supplies a slug and nothing else. An unsearched
   panel opens on a spread of seed use cases so it is not blank. The row's
   subtitle falls back to "Ready to link" rather than "0 tools", which would be
   a fact about the transport rather than about the toolkit.
2. **Reconciliation samples rather than censuses.** The gateway answers about
   toolkits it is *asked* about; there is no "list everything". It is asked
   about what is already on file **and** what the catalogue search surfaces, so
   a toolkit connected directly in the Composio dashboard does show up — as
   long as the search names it. One outside that spread stays invisible until
   somebody searches for it. The REST path has no such blind spot.

   Asking about a toolkit is not the same as having one: the gateway answers
   `initiated` for every toolkit it has never connected, so an answer carrying
   no *account* is dropped unless there is already a row for it. Without that,
   widening the question files a "waiting for consent" row against every
   service in the catalogue — and the panel polls every three seconds for as
   long as one of those exists.

   The cost is the other half: a consent genuinely in flight has no account
   either, so the gateway cannot see it and cannot testify that it is dead.
   `IntegrationStore.reconcile` takes `sees_pending=False` on this path and
   leaves pending rows alone, revoking only what it can see was ACTIVE.
3. **The capability probe samples rather than censuses.** Same reason. It
   searches for common toolkits, then asks about exactly those — so `sampled`
   is what was looked at, not what exists.

## The bug inside the fix

Composio nests **three** envelopes on an execution, and only the innermost one
says whether the *tool* worked:

```json
{"data": {"results": [                        // multi-executor: ran the batch
  {"tool_slug": "GMAIL_SEND_EMAIL",
   "response": {"successful": false,          // ← the actual verdict
                "error": "scope insufficient"}}]},
 "successful": true}                          // ← the batch succeeded
```

Unwrapped one level too shallow, a refused send is reported to the model as a
sent email. The multi-executor runs up to fifty tools and one of them failing is
not the call failing, so the outer `successful` cannot be trusted for this.
`test_a_tool_that_failed_inside_a_successful_call_is_a_failure` pins it.

## Cost

The gateway is slower than REST, and it is worth knowing where:

- tool discovery for one toolkit: ~1.5 s
- a capability probe: **~25 s** (three sequential semantic searches)

Nothing awaits the probe — `ProfileService` submits it to a background pool and
never calls `.result()` — so it is off the request path and off the voice path
entirely. Tool execution (~1.5–3.5 s) is the number that reaches a spoken turn,
and it is comparable to REST.

Semantic search is also non-deterministic: `primary_tool_slugs` alone swung
between 2 and 10 tools for the same query across runs. Taking
`related_tool_slugs` as well makes it stable at 17 — which matters, because a
model whose toolset changes between turns is a model that forgets it can do
something it did a minute ago.

## Files

- [`src/integrations/mcp.py`](../src/integrations/mcp.py) — the transport (new)
- [`src/integrations/client.py`](../src/integrations/client.py) — `build()` picks a transport on the key prefix
- [`src/agents/tool_agent.py`](../src/agents/tool_agent.py) — tool discovery and execution branch
- [`src/services/integration_service.py`](../src/services/integration_service.py) — catalogue, connect, reconcile, disconnect branch
- [`src/connectors/registry.py`](../src/connectors/registry.py) — `verify_composio` grades a key against the right product
- [`src/connectors/probes/composio.py`](../src/connectors/probes/composio.py) — the capability probe
- [`tests/test_mcp.py`](../tests/test_mcp.py) — 41 offline tests over the translation layer
