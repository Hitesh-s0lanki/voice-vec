"""The connectors this app knows about, and how to prove each one works.

Four today: one for tools and three for vectors. Everything the panel renders
comes from here, so adding a fifth is this file plus — for a vector store — a
backend under `src/rag/backends/`.

**Every `verify` makes one cheap, read-only call.** Not a ping and not a
credential-format check: an actual authenticated request against the actual
service, because the failure this catches is "the key is well-formed and
wrong", which no amount of regex finds. They are called before anything is
stored, so a credential that cannot answer is never encrypted and kept.

They are also all bounded by a short timeout. This runs inside a request that
somebody is watching a spinner for, and a vector store that has gone away
should fail the form in seconds rather than hold a worker until the client
gives up.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping

import httpx

from src.connectors.spec import ConnectorError, ConnectorSpec, Field

log = logging.getLogger("vec.connectors")

# Short on purpose — see the module docstring. Long enough for a cold serverless
# index to answer, short enough that a dead host does not hold the request.
VERIFY_TIMEOUT_S = 10.0

# Pinecone pins behaviour to a dated API version. Sending it explicitly means a
# future default cannot change the response shape underneath this code.
PINECONE_API_VERSION = "2025-10"
PINECONE_CONTROL = "https://api.pinecone.io"


def _http_error(service: str, error: Exception) -> ConnectorError:
    """Turn an httpx failure into something the person who typed it can act on."""
    if isinstance(error, httpx.HTTPStatusError):
        code = error.response.status_code
        if code in (401, 403):
            return ConnectorError(f"{service} rejected those credentials.")
        if code == 404:
            return ConnectorError(f"{service} could not find that — check the name.")
        return ConnectorError(f"{service} answered {code}.", status_code=502)
    return ConnectorError(f"{service} did not answer.", status_code=502)


# ---- Composio ------------------------------------------------------------


def verify_composio(credentials: Mapping[str, str]) -> None:
    """List one toolkit. The cheapest authenticated call the SDK offers."""
    from composio import Composio  # local: keeps the SDK off the import path

    try:
        sdk = Composio(api_key=credentials["api_key"], timeout=int(VERIFY_TIMEOUT_S))
        sdk.client.toolkits.list(limit=1)
    except Exception as error:
        log.info("composio verify failed: %s", type(error).__name__)
        raise ConnectorError(
            "Composio rejected that API key. Check it in your Composio dashboard."
        ) from error


# ---- Pinecone ------------------------------------------------------------


def pinecone_host(api_key: str, index: str) -> str:
    """The index's own DNS host, which every data-plane call needs.

    Pinecone splits the control plane (`api.pinecone.io`, where indexes are
    described) from the data plane (a host per index, where queries go). The
    host is stable and Pinecone's own guidance is to look it up once and cache
    it, which is what the backend does — this is the lookup.
    """
    try:
        response = httpx.get(
            f"{PINECONE_CONTROL}/indexes/{index}",
            headers={
                "Api-Key": api_key,
                "X-Pinecone-Api-Version": PINECONE_API_VERSION,
            },
            timeout=VERIFY_TIMEOUT_S,
        )
        response.raise_for_status()
    except Exception as error:
        raise _http_error("Pinecone", error) from error

    host = (response.json() or {}).get("host")
    if not host:
        raise ConnectorError(f"Pinecone described index {index!r} without a host.", 502)
    return str(host)


def verify_pinecone(credentials: Mapping[str, str]) -> None:
    """Describing the index proves the key *and* that the index exists.

    Both matter and they fail differently: a wrong key is 401 and a wrong index
    name is 404, and telling someone to re-check their key when the index name
    has a typo is the kind of error message that costs an afternoon.
    """
    pinecone_host(credentials["api_key"], credentials["index"])


# ---- Astra DB ------------------------------------------------------------


def astra_url(endpoint: str, keyspace: str, collection: str = "") -> str:
    """Astra's Data API path. `endpoint` is the database's own https origin."""
    base = endpoint.rstrip("/")
    path = f"{base}/api/json/v1/{keyspace}"
    return f"{path}/{collection}" if collection else path


def verify_astra(credentials: Mapping[str, str]) -> None:
    """Ask the keyspace what collections it has.

    A keyspace-level command, so it verifies the token, the endpoint and the
    keyspace in one call without needing the collection to exist yet.
    """
    endpoint = credentials["endpoint"].strip()
    if not endpoint.startswith("https://"):
        raise ConnectorError("The Astra endpoint should start with https://.")

    try:
        response = httpx.post(
            astra_url(endpoint, credentials["keyspace"]),
            headers={
                "Token": credentials["token"],
                "Content-Type": "application/json",
            },
            json={"findCollections": {}},
            timeout=VERIFY_TIMEOUT_S,
        )
        response.raise_for_status()
    except Exception as error:
        raise _http_error("Astra DB", error) from error

    # The Data API answers 200 with an `errors` array rather than a status
    # code, so a bad token looks like success to `raise_for_status`.
    body = response.json() or {}
    if body.get("errors"):
        message = str(body["errors"][0].get("message", "")).strip()
        raise ConnectorError(f"Astra DB rejected that: {message or 'unknown error'}.")


# ---- pgvector ------------------------------------------------------------


# What a dense search reads out of the table (`src/rag/store.py`). `tsv` and
# `tsv_en` are deliberately not here: a table built by an older migration has
# no tsvector column, and rung 2 already degrades that to dense-only rather
# than losing the answer. A missing *vector* column has no such fallback.
PGVECTOR_COLUMNS = (
    "chunk_key",
    "strategy",
    "text",
    "meta",
    "language",
    "embedding",
    "embedding_en",
)

# `format_type` renders a pgvector column as `vector(384)`.
_VECTOR_DIM = re.compile(r"^vector\((\d+)\)$")

# Every column of the table, in one round trip. `current_schemas(false)` is the
# search path the app's own queries resolve against, so a table this cannot see
# is a table the search would not find either.
_COLUMNS_SQL = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %s
  AND n.nspname = ANY(current_schemas(false))
  AND a.attnum > 0
  AND NOT a.attisdropped
"""


def verify_pgvector(credentials: Mapping[str, str]) -> None:
    """Connect, confirm pgvector is available, and read the table's shape.

    Reachability alone is not enough to be worth storing: a Postgres without
    pgvector accepts the connection and then fails every search, which would
    turn a clear error here into a mystery later.

    Neither is the extension. This connector searches *this app's schema* — the
    same `chunks` table `scripts/ingest.py` writes — and a database that has
    pgvector but a table of somebody else's shape fails exactly the same way,
    one abstain at a time, with the panel still showing the connector as green.
    Somebody who points this at a Postgres holding their own `book_chunks` is
    told so here, while the form is still open and the fix is still cheap.
    """
    import psycopg

    from src.core.config import get_settings

    settings = get_settings()
    table = (credentials.get("table") or "").strip() or settings.pg_table

    dsn = credentials["dsn"].strip()
    if not dsn.startswith(("postgres://", "postgresql://")):
        raise ConnectorError("That does not look like a Postgres connection string.")

    try:
        with psycopg.connect(dsn, connect_timeout=int(VERIFY_TIMEOUT_S)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
                )
                if cur.fetchone() is None:
                    raise ConnectorError(
                        "That Postgres does not have the pgvector extension available."
                    )
                # Parameterised against the catalogue rather than interpolated
                # into a query: the table name is somebody's form input.
                cur.execute(_COLUMNS_SQL, (table,))
                found = {name: kind for name, kind in cur.fetchall()}
    except ConnectorError:
        raise
    except Exception as error:
        log.info("pgvector verify failed: %s", type(error).__name__)
        raise ConnectorError("Could not connect to that Postgres.", status_code=502) from error

    _check_pgvector_table(table, found, settings.embed_dim)


def _check_pgvector_table(table: str, found: Mapping[str, str], dim: int) -> None:
    """Is this table one the search can actually read?

    Split out from the connection so the shape rules are testable without a
    Postgres, and so a failure here cannot be mistaken for a network one.
    """
    if not found:
        raise ConnectorError(
            f"That Postgres has no table called “{table}”. Ingest into it with "
            "scripts/ingest.py, or name the table you want searched."
        )

    missing = [column for column in PGVECTOR_COLUMNS if column not in found]
    if missing:
        raise ConnectorError(
            f"“{table}” is not this app's schema — it has no "
            f"{', '.join(missing)}. Build it with scripts/ingest.py pointed at "
            "this database."
        )

    match = _VECTOR_DIM.match(found["embedding"].strip())
    if not match:
        raise ConnectorError(
            f"“{table}”.embedding is {found['embedding']}, not a pgvector column."
        )
    if int(match.group(1)) != dim:
        raise ConnectorError(
            f"“{table}” stores {match.group(1)}-dimensional vectors and this app "
            f"embeds at {dim}. They cannot be searched against each other."
        )


# ---- the registry --------------------------------------------------------

SPECS: tuple[ConnectorSpec, ...] = (
    ConnectorSpec(
        slug="composio",
        name="Composio",
        kind="tools",
        summary="Your tools, through your own Composio project.",
        docs_url="https://app.composio.dev/developers",
        fields=(
            Field(
                name="api_key",
                label="API key",
                secret=True,
                placeholder="ak_…",
                help="From app.composio.dev → Developers.",
            ),
        ),
        verify=verify_composio,
    ),
    ConnectorSpec(
        slug="pinecone",
        name="Pinecone",
        kind="vector",
        summary="Search your own Pinecone index.",
        docs_url="https://app.pinecone.io",
        fields=(
            Field(name="api_key", label="API key", secret=True, placeholder="pcsk_…"),
            Field(
                name="index",
                label="Index name",
                placeholder="vec-chunks",
                help="The index host is looked up from this and cached.",
            ),
            Field(
                name="namespace",
                label="Namespace",
                required=False,
                help="Optional. Leave empty for the default namespace.",
            ),
        ),
        verify=verify_pinecone,
    ),
    ConnectorSpec(
        slug="astra",
        name="Astra DB",
        kind="vector",
        summary="Search your own DataStax Astra collection.",
        docs_url="https://astra.datastax.com",
        aliases=("datastax", "astradb"),
        fields=(
            Field(
                name="token",
                label="Application token",
                secret=True,
                placeholder="AstraCS:…",
            ),
            Field(
                name="endpoint",
                label="API endpoint",
                placeholder="https://<id>-<region>.apps.astra.datastax.com",
            ),
            Field(name="keyspace", label="Keyspace", placeholder="default_keyspace"),
            Field(name="collection", label="Collection", placeholder="chunks"),
        ),
        verify=verify_astra,
    ),
    ConnectorSpec(
        slug="pgvector",
        name="Postgres + pgvector",
        kind="vector",
        summary="Search your own Postgres, this app's schema.",
        docs_url="https://github.com/pgvector/pgvector",
        aliases=("postgres",),
        fields=(
            Field(
                name="dsn",
                label="Connection string",
                secret=True,
                placeholder="postgresql://user:password@host:5432/db",
                help="Use the pooled host on Neon. The password makes this a secret.",
            ),
            Field(
                name="table",
                label="Table",
                required=False,
                placeholder="chunks",
                help="Optional. Defaults to the app's own table name.",
            ),
        ),
        verify=verify_pgvector,
    ),
)

_BY_SLUG: dict[str, ConnectorSpec] = {}
for _spec in SPECS:
    _BY_SLUG[_spec.slug] = _spec
    for _alias in _spec.aliases:
        _BY_SLUG[_alias] = _spec


def get_spec(slug: str) -> ConnectorSpec | None:
    """Look a connector up by slug, or by one of the names people actually type."""
    return _BY_SLUG.get((slug or "").strip().lower())


def vector_slugs() -> tuple[str, ...]:
    """Which connectors can back retrieval, asked rather than hard-coded."""
    return tuple(spec.slug for spec in SPECS if spec.kind == "vector")
