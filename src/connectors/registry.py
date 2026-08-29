"""The connectors this app knows about, and how to prove each one works.

Five today: one for tools, three for vectors, and one for a dataset.
Everything the panel renders comes from here, so adding a sixth is this file
plus — for a vector store — a backend under `src/rag/backends/`.

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
from typing import Any, Mapping, Sequence

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


def reconcile_dimension(label: str, store_dim: int | None, app_dim: int) -> dict[str, str]:
    """Can this app query an index of that width, and how?

    Same width as the deployment: the local embedder is right, and nothing on
    the answer path changes.

    Different width: the query is embedded remotely at exactly that width
    (`src/rag/remote_embed.py`), because OpenAI's `text-embedding-3` models take
    a `dimensions` parameter. So the width read from the catalogue is the whole
    answer and there is nothing to ask.

    This used to take a model name from the form. That was a question the form
    could not help anyone answer — the reasonable reply to "stores
    768-dimensional vectors" is to type `768`, which is what happened, and
    fastembed then spent 39 seconds trying to download a repository called 768.
    Reading the width and using it is strictly more information and strictly
    less to get wrong.
    """
    from src.rag.remote_embed import MAX_DIM, supported

    if not store_dim or store_dim == app_dim:
        return {"dim": str(store_dim or app_dim)}

    if not supported(store_dim):
        raise ConnectorError(
            f"{label} stores {store_dim}-dimensional vectors, which is wider "
            f"than anything this app can embed a query at ({MAX_DIM} is the "
            "maximum)."
        )

    return {"dim": str(store_dim)}


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
    """Prove the key works — against whichever Composio it belongs to.

    Composio issues two credentials for two products and neither one works
    where the other does. `ak_…` opens the REST API the Python SDK wraps;
    `ck_…` opens only the MCP gateway, and the REST API answers 401 to it with
    a message about an invalid key — which sends people to rotate a credential
    that was never wrong. The prefix picks the transport, so that mistake is
    no longer reachable, and each branch names the product it failed against.
    """
    from src.integrations.mcp import GatewayError, ComposioGateway, is_gateway_key

    api_key = credentials["api_key"]

    if is_gateway_key(api_key):
        try:
            ComposioGateway(api_key, timeout=VERIFY_TIMEOUT_S).verify()
        except GatewayError as error:
            log.info("composio gateway verify failed: %s", error)
            raise ConnectorError(
                "Composio's MCP gateway rejected that key. Check it at "
                "dashboard.composio.dev."
            ) from error
        return

    from composio import Composio  # local: keeps the SDK off the import path

    try:
        sdk = Composio(api_key=api_key, timeout=int(VERIFY_TIMEOUT_S))
        sdk.client.toolkits.list(limit=1)
    except Exception as error:
        log.info("composio verify failed: %s", type(error).__name__)
        raise ConnectorError(
            "Composio rejected that API key. An MCP gateway key starts with "
            "“ck_”; a platform API key starts with “ak_” and comes from "
            "app.composio.dev → Developers."
        ) from error


# ---- Pinecone ------------------------------------------------------------


def pinecone_host(api_key: str, index: str) -> str:
    """The index's own DNS host, which every data-plane call needs.

    Pinecone splits the control plane (`api.pinecone.io`, where indexes are
    described) from the data plane (a host per index, where queries go). The
    host is stable and Pinecone's own guidance is to look it up once and cache
    it, which is what the backend does — this is the lookup.
    """
    return str(_describe_pinecone(api_key, index)["host"])


def verify_pinecone(credentials: Mapping[str, str]) -> Mapping[str, str] | None:
    """Describing the index proves the key, that it exists, *and* how wide it is.

    All three matter and they fail differently: a wrong key is 401, a wrong
    index name is 404, and an index built by another embedding model is a 200
    that then fails every query. Telling someone to re-check their key when the
    index name has a typo is the kind of error that costs an afternoon; letting
    a width mismatch through is the kind that costs a week, because nothing
    reports it.
    """
    from src.core.config import get_settings

    described = _describe_pinecone(credentials["api_key"], credentials["index"])
    settings = get_settings()
    dimension = described.get("dimension")

    return reconcile_dimension(
        f"“{credentials['index']}”",
        int(dimension) if dimension else None,
        settings.embed_dim,
    )


def _describe_pinecone(api_key: str, index: str) -> dict:
    """The control plane's description of the index — host, dimension, metric."""
    try:
        response = httpx.get(
            f"{PINECONE_CONTROL}/indexes/{index}",
            headers={"Api-Key": api_key, "X-Pinecone-Api-Version": PINECONE_API_VERSION},
            timeout=VERIFY_TIMEOUT_S,
        )
        response.raise_for_status()
    except Exception as error:
        raise _http_error("Pinecone", error) from error

    described = response.json() or {}
    if not described.get("host"):
        raise ConnectorError(f"Pinecone described index {index!r} without a host.", 502)
    return described


# ---- Astra DB ------------------------------------------------------------


def astra_url(endpoint: str, keyspace: str, collection: str = "") -> str:
    """Astra's Data API path. `endpoint` is the database's own https origin."""
    base = endpoint.rstrip("/")
    path = f"{base}/api/json/v1/{keyspace}"
    return f"{path}/{collection}" if collection else path


def verify_astra(credentials: Mapping[str, str]) -> Mapping[str, str] | None:
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
            json={"findCollections": {"options": {"explain": True}}},
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

    # The collection's own width, when it declares one. Same reconciliation as
    # the other two: an index built by another model is a 200 that then fails
    # every query, and this is the only place it can still be said cheaply.
    from src.core.config import get_settings

    wanted = credentials.get("collection", "")
    dimension = None
    for entry in (body.get("status") or {}).get("collections") or []:
        if isinstance(entry, Mapping) and entry.get("name") == wanted:
            dimension = ((entry.get("options") or {}).get("vector") or {}).get("dimension")

    return reconcile_dimension(
        f"“{wanted}”",
        int(dimension) if dimension else None,
        get_settings().embed_dim,
    )


# ---- Dataset -------------------------------------------------------------


def verify_dataset(credentials: Mapping[str, str]) -> Mapping[str, str] | None:
    """Resolve the URL into the actual files behind it.

    This is a real read, not a format check, and it is the same call the build
    will make: `resolve` asks Hugging Face for the repository's file list and
    fails if the dataset does not exist, is gated, or holds nothing readable.
    Which means the errors worth showing somebody — a typo, a private repo, a
    URL that points at a model rather than a dataset — land on the form while
    they are still looking at it, instead of thirty seconds into a download.

    Everything after this is slow and belongs on a worker: the pull, the
    measurement and the model call all happen after the form goes green
    (docs/18-datasets.md).

    Returns what resolving found, so the panel can say `ai4bharat/MSMARCO-XI ·
    12 files` without re-asking, and so nothing downstream has to resolve the
    URL a second time to learn the id it was stored under.
    """
    from src.core.config import get_settings
    from src.datasets.source import SourceError, resolve

    settings = get_settings()
    if not settings.datasets_enabled:
        raise ConnectorError("Datasets are not enabled on this deployment.", status_code=503)

    try:
        source = resolve(
            credentials["url"],
            max_tables=settings.dataset_max_tables,
            timeout_s=VERIFY_TIMEOUT_S,
        )
    except SourceError as error:
        # `SourceError` is already written for the person who typed the URL —
        # "No dataset called x/y on Hugging Face", "that is gated or private".
        # Rephrasing it here would lose the specific half.
        raise ConnectorError(str(error)) from error

    return {
        "dataset_id": source.slug,
        "location": source.location,
        "files": str(len(source.tables)),
    }


# ---- pgvector ------------------------------------------------------------


# What a dense search *needs*: something to compare against, and something to
# read back. Everything else this app's own schema carries — `strategy`,
# `language`, `meta`, the English pair, the tsvectors — is a capability, and a
# table without one loses that channel rather than being rejected.
#
# It used to be the full seven, which made "connect your own Postgres" mean
# "connect a copy of ours": a working pgvector table holding `id` and
# `chunk_text` was turned away at the form for not having columns its owner had
# never heard of. `src/rag/columns.py` is what made the difference — the search
# SQL now takes the column names rather than hardcoding them.
PGVECTOR_REQUIRED = ("embedding", "text")

# The names other people's pipelines give the readable text, in the order worth
# trying. `document` is LangChain's, `chunk` is pgai's, `page_content` is
# LangChain's Python-side name — between them these cover the tooling most
# connected tables were built by.
_TEXT_NAMES = (
    "text", "chunk_text", "content", "contents", "document", "page_content",
    "chunk", "body", "passage",
)
_ID_NAMES = ("chunk_key", "id", "_id", "uuid", "key", "doc_id", "node_id", "custom_id")
# `cmetadata` is LangChain's, `metadata_` is LlamaIndex's — both trail the
# convention by one character, which is exactly enough to miss.
_META_NAMES = ("meta", "metadata", "cmetadata", "metadata_", "payload")

# A sample small enough to be free, large enough to tell prose from an id. Five
# rows settle "which column holds the document" definitively, and guessing it
# from names alone got LangChain's own schema wrong — it picked the `id`
# varchar, so every answer would have been cut from a UUID.
_TEXT_SAMPLE = 5

#: A column whose values average shorter than this is a label or a key, not the
#: document. Matches the floor `pick_text_field` uses when the profiler samples
#: properly later.
MIN_TEXT_CHARS = 80

# Postgres types that can hold prose. `character varying` is what `varchar`
# renders as through `format_type`.
_TEXT_TYPES = ("text", "character varying", "character", "citext")

# How many extra columns are carried into `Hit.payload` when the table has no
# `meta` of its own. Enough for a hit to name its source — the `book_id` of a
# chunked library — without turning every result into a row dump.
MAX_PAYLOAD_COLUMNS = 6

# The primary key, for the identifier a hit is returned under.
_PRIMARY_KEY = """
SELECT a.attname
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
WHERE c.relname = %(table)s
  AND ((%(schema)s::text IS NULL AND n.nspname = ANY(current_schemas(false)))
       OR n.nspname = %(schema)s)
  AND i.indisprimary
ORDER BY a.attnum
"""

# `format_type` renders a pgvector column as `vector(384)`. `halfvec` is the
# same thing at half the storage and is searched identically, so a table that
# used it to halve its index size is not a table this app should refuse.
_VECTOR_DIM = re.compile(r"^(?:vector|halfvec)\((\d+)\)$")

# opclass → the distance it was built for. Read rather than assumed: querying a
# `vector_l2_ops` index with `<=>` does not use the index at all — it
# sequential-scans, and the only symptom is latency nobody can explain.
_OPCLASS_METRIC = {
    "cosine_ops": "cosine",
    "ip_ops": "inner_product",
    "l2_ops": "l2",
}

# Which index, if any, covers the vector column, and for which distance. The
# opclass name is read out of `pg_get_indexdef` because that is the form a
# person recognises when it appears in an error.
_VECTOR_INDEXES = """
SELECT a.attname, pg_get_indexdef(i.indexrelid)
FROM pg_index i
JOIN pg_class c  ON c.oid = i.indrelid
JOIN pg_class ic ON ic.oid = i.indexrelid
JOIN pg_am am    ON am.oid = ic.relam
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
WHERE c.relname = %(table)s
  AND ((%(schema)s::text IS NULL AND n.nspname = ANY(current_schemas(false)))
       OR n.nspname = %(schema)s)
  AND am.amname IN ('hnsw', 'ivfflat')
"""

# Every table on the search path that has a vector column, for the case the
# form leaves `table` blank — which is almost every case, because the field is
# optional and nobody types a value they were not asked for.
#
# Defaulting a blank to this app's own `chunks` was the original behaviour and
# it is wrong in the ordinary case: `chunks` is an *internal* name, so a person
# connecting their own Postgres is told their database is missing a table they
# have never heard of. Their database is fine; the question was never asked.
_VECTOR_TABLES = """
SELECT CASE WHEN n.nspname = 'public' THEN c.relname
            ELSE n.nspname || '.' || c.relname END AS name,
       count(*) AS vectors
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = ANY(current_schemas(false))
  AND c.relkind = 'r'
  AND a.attnum > 0 AND NOT a.attisdropped
  AND (format_type(a.atttypid, a.atttypmod) LIKE 'vector(%'
       OR format_type(a.atttypid, a.atttypmod) LIKE 'halfvec(%')
GROUP BY n.nspname, c.relname
ORDER BY 1
"""

# Every column of the table, in one round trip. `current_schemas(false)` is the
# search path the app's own queries resolve against, so a table this cannot see
# is a table the search would not find either.
_COLUMNS_SQL = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %(table)s
  AND ((%(schema)s::text IS NULL AND n.nspname = ANY(current_schemas(false)))
       OR n.nspname = %(schema)s)
  AND a.attnum > 0
  AND NOT a.attisdropped
"""


def verify_pgvector(credentials: Mapping[str, str]) -> Mapping[str, str] | None:
    """Connect, confirm pgvector is there, find the table, and read its shape.

    Reachability alone is not enough to be worth storing: a Postgres without
    pgvector accepts the connection and then fails every search, which would
    turn a clear error here into a mystery later.

    **Finding the table is part of verifying, not something to make the user
    do.** `table` is optional, so it arrives blank almost every time, and the
    old behaviour then guessed this app's own `chunks` — an internal name the
    person connecting has never seen. They were told their database was missing
    a table they never made. Now a blank field means *look*: one vector table
    is used, several are named so they can pick, and none is a real answer
    about their database rather than about our default.

    Returns the table it settled on, so a blank field is stored as the name it
    resolved to and nothing downstream has to guess again.
    """
    import psycopg

    from src.core.config import get_settings

    settings = get_settings()
    asked = (credentials.get("table") or "").strip()

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

                cur.execute(_VECTOR_TABLES)
                candidates = [name for name, _ in cur.fetchall()]
                table = asked or _resolve_table(candidates)

                # Parameterised against the catalogue rather than interpolated
                # into a query: the table name is somebody's form input.
                # `schema.table` is accepted, because a table outside the
                # search path cannot be named any other way — and plenty of
                # people keep their embeddings in a schema of their own.
                schema, _, bare = table.rpartition(".")
                where = {"table": bare, "schema": schema or None}

                cur.execute(_COLUMNS_SQL, where)
                found = {name: kind for name, kind in cur.fetchall()}

                cur.execute(_PRIMARY_KEY, where)
                keys = [name for (name,) in cur.fetchall()]

                cur.execute(_VECTOR_INDEXES, where)
                index_defs = {column: definition for column, definition in cur.fetchall()}

                sample = _sample_rows(cur, table, found)
    except ConnectorError:
        raise
    except Exception as error:
        log.info("pgvector verify failed: %s", type(error).__name__)
        raise ConnectorError("Could not connect to that Postgres.", status_code=502) from error

    if asked and not found and candidates:
        # They named a table and it is not the one holding vectors. Say which
        # ones are, rather than repeating that theirs does not exist.
        raise ConnectorError(
            f"That Postgres has no table called “{asked}”. "
            f"{_name_candidates(candidates)}"
        )

    mapped = discover_columns(found, keys[0] if len(keys) == 1 else "", sample)
    if mapped.get("embedding"):
        mapped["metric"] = metric_of(index_defs, mapped["embedding"])
    store_dim = _check_pgvector_table(table, found, settings.embed_dim, mapped)
    resolved = reconcile_dimension(f"“{table}”", store_dim, settings.embed_dim)

    # The map is sealed with the DSN, so the backend builds its SQL from what
    # was actually found here rather than rediscovering it per request — and a
    # reconnect re-runs discovery, which is what makes a renamed column a
    # reconnect rather than a mystery.
    return {
        "table": table,
        **resolved,
        **{f"col_{role}": name for role, name in mapped.items()},
    }


def _sample_rows(cur: Any, table: str, found: Mapping[str, str]) -> list[dict[str, Any]]:
    """A few rows of the text-ish columns only. Never the vectors.

    Selecting the named columns rather than `*` keeps hundreds of floats per
    row off the wire, and keeps this to the one question it is asked: which of
    these strings is the document.
    """
    from psycopg import sql

    wanted = [
        name for name, kind in found.items() if kind.strip() in _TEXT_TYPES
    ]
    if not wanted:
        return []

    try:
        cur.execute(
            sql.SQL("SELECT {} FROM {} LIMIT {}").format(
                sql.SQL(", ").join(sql.Identifier(n) for n in wanted),
                sql.SQL(".").join(sql.Identifier(p) for p in table.split(".") if p),
                sql.Literal(_TEXT_SAMPLE),
            )
        )
        return [dict(zip(wanted, row)) for row in cur.fetchall()]
    except Exception as error:
        # An empty or unreadable table still connects; it just falls back to
        # the name heuristic, which is where this started.
        log.debug("could not sample %s for text detection: %s", table, error)
        return []


def _resolve_table(candidates: Sequence[str]) -> str:
    """Which table a blank field means, or an error that says how to answer.

    One candidate is not a guess — it is the only table in that database a
    vector search could run against, and asking someone to type a name the
    server can already see is a form that exists to be got wrong.
    """
    if not candidates:
        raise ConnectorError(
            "That Postgres has pgvector but no table with a vector column. "
            "Connect a database that already holds embeddings."
        )
    if len(candidates) == 1:
        return candidates[0]
    raise ConnectorError(
        f"That Postgres has {len(candidates)} tables with vectors. "
        f"{_name_candidates(candidates)}"
    )


def _name_candidates(candidates: Sequence[str]) -> str:
    shown = ", ".join(f"“{name}”" for name in candidates[:6])
    more = "" if len(candidates) <= 6 else f", and {len(candidates) - 6} more"
    return f"Name the one you want searched: {shown}{more}."


def metric_of(index_defs: Mapping[str, str], embedding: str) -> str:
    """Which distance the embedding column is indexed for.

    Cosine when there is no ANN index at all — not because that is known to be
    right, but because it is what an unindexed table is scanned with and it is
    the scale everything downstream was calibrated on. The profiler reports the
    missing index separately, since the consequence there is latency rather
    than a wrong ordering.
    """
    definition = index_defs.get(embedding, "")
    for suffix, metric in _OPCLASS_METRIC.items():
        if suffix in definition:
            return metric
    return "cosine"


def widest_text_column(rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> str:
    """Which candidate actually holds prose, measured over a handful of rows.

    Names are a hint and data is the answer. `id`, `document` and
    `collection_id` are all `character varying` in LangChain's table, and only
    one of them is the document — no amount of reading the catalogue tells them
    apart, and picking wrong means every answer is quoted from a UUID.
    """
    best, longest = "", 0.0
    for name in candidates:
        lengths = [len(row[name]) for row in rows if isinstance(row.get(name), str)]
        if not lengths:
            continue
        mean = sum(lengths) / len(lengths)
        if mean > longest and mean >= MIN_TEXT_CHARS:
            best, longest = name, mean
    return best


def discover_columns(
    found: Mapping[str, str],
    primary_key: str = "",
    rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Work out which column plays which role, from the catalogue alone.

    Structural roles only — what a query needs to be *written*. Nothing here
    looks at data, because this runs while somebody watches a spinner and the
    question "which column holds the text" is answered well enough by name and
    type. The profiler samples rows later and reports what it found, including
    where this guessed differently.

    Every role is optional except the two in `PGVECTOR_REQUIRED`. An absent one
    is a lost capability, and `ColumnMap` turns that into a `Capabilities` the
    ladder is told about before it asks.
    """
    vectors = [name for name, kind in found.items() if _VECTOR_DIM.match(kind.strip())]
    texts = [name for name, kind in found.items() if kind.strip() in _TEXT_TYPES]
    jsonb = [name for name, kind in found.items() if kind.strip() in ("jsonb", "json")]
    tsvectors = [name for name, kind in found.items() if kind.strip() == "tsvector"]

    # The English pair, if this table has one: a second vector and a second
    # tsvector, both named for it. Only an index this app ingested does.
    english_vec = next((n for n in vectors if n.endswith("_en")), "")
    english_tsv = next((n for n in tsvectors if n.endswith("_en")), "")

    embedding = _prefer(vectors, ("embedding",), exclude={english_vec})
    identifier = _prefer(list(found), _ID_NAMES) or primary_key

    # The identifier is settled first so it can be excluded from the text
    # candidates. Without that, a table whose id is a `varchar` — LangChain's
    # is — offers it as prose and, being first in catalogue order, wins.
    candidates = [n for n in texts if n not in (identifier, "english")]
    text = _prefer(candidates, _TEXT_NAMES)
    if not text or (rows and text not in _TEXT_NAMES[:3]):
        # Either no conventional name, or one whose claim is worth checking
        # against the data. Measured beats guessed whenever there is a sample.
        measured = widest_text_column(rows, candidates)
        text = measured or text

    mapped = {
        "embedding": embedding,
        "embedding_en": english_vec,
        "text": text,
        "id": identifier,
        "meta": _prefer(jsonb, _META_NAMES),
        "strategy": "strategy" if "strategy" in found else "",
        "language": "language" if "language" in found else "",
        "english": "english" if "english" in texts else "",
        "tsv": next((n for n in tsvectors if not n.endswith("_en")), ""),
        "tsv_en": english_tsv,
    }

    # Whatever is left that a hit could be cited by. Only consulted when the
    # table has no `meta` of its own, so this costs nothing on this app's own
    # schema and gives a foreign one a `book_id` to name its source with.
    if not mapped["meta"]:
        taken = {v for v in mapped.values() if v}
        spare = [
            name
            for name, kind in found.items()
            if name not in taken
            and kind.strip() in _TEXT_TYPES + ("integer", "bigint", "smallint", "boolean")
        ]
        if spare:
            mapped["payload"] = ",".join(sorted(spare)[:MAX_PAYLOAD_COLUMNS])

    return {key: value for key, value in mapped.items() if value}


def _prefer(
    available: Sequence[str], names: Sequence[str], exclude: set[str] | None = None
) -> str:
    """The first conventional name present, else the first thing available.

    Falling through to "whatever there is" is what makes this work on a table
    nobody built for this app. It is also why the profile reports the choice
    rather than assuming it: an agent quoting the wrong column produces a
    citation that is technically from the corpus and reads as nonsense.
    """
    pool = [name for name in available if name and name not in (exclude or set())]
    for candidate in names:
        if candidate in pool:
            return candidate
    return pool[0] if pool else ""


def _check_pgvector_table(
    table: str, found: Mapping[str, str], dim: int, mapped: Mapping[str, str]
) -> int:
    """Is this table one the search can actually read?

    Split out from the connection so the shape rules are testable without a
    Postgres, and so a failure here cannot be mistaken for a network one.

    Two questions now, where there used to be one. The schema no longer has to
    *match* — it has to be *readable*, which is a much lower bar and the right
    one, since `src/rag/columns.py` builds the query from whatever is there.
    The dimension is the one thing no mapping can paper over.
    """
    if not found:
        raise ConnectorError(f"That Postgres has no table called “{table}”.")

    missing = [role for role in PGVECTOR_REQUIRED if not mapped.get(role)]
    if missing:
        what = {
            "embedding": "a vector column to search",
            "text": "a text column to read answers out of",
        }
        raise ConnectorError(
            f"“{table}” has no {' and no '.join(what[role] for role in missing)}."
        )

    match = _VECTOR_DIM.match(found[mapped["embedding"]].strip())
    if not match:
        raise ConnectorError(
            f"“{table}”.{mapped['embedding']} is {found[mapped['embedding']]}, "
            "not a pgvector column."
        )
    # The width itself is reconciled by the caller, which has the model field.
    return int(match.group(1))


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
                placeholder="ak_… or ck_…",
                help=(
                    "A platform key from app.composio.dev → Developers, or an "
                    "MCP gateway key (ck_…) from dashboard.composio.dev."
                ),
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
        summary="Search any pgvector table in your Postgres.",
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
                placeholder="found automatically",
                help="Leave blank and the table holding your vectors is found for you.",
            ),
        ),
        verify=verify_pgvector,
    ),
    ConnectorSpec(
        slug="dataset",
        name="Dataset",
        kind="dataset",
        summary="Answer questions about a dataset in SQL.",
        docs_url="https://huggingface.co/datasets",
        aliases=("datasets", "huggingface", "hf"),
        fields=(
            Field(
                name="url",
                label="Dataset URL",
                # Not secret. It is a public address, and marking it secret
                # would hide from the panel the one thing that identifies which
                # dataset is attached.
                secret=False,
                placeholder="https://huggingface.co/datasets/owner/name",
                help=(
                    "A Hugging Face dataset, or a direct link to a .parquet or .csv file. "
                    "Point at a subfolder to attach just those splits. Public data only."
                ),
            ),
        ),
        verify=verify_dataset,
        # No secret field to take four characters from, so the panel labels the
        # row with the URL itself — which is also the only useful label.
        hint_field="url",
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
