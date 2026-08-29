"""Create this app's tables, and prove the database is reachable.

    uv run python -m scripts.migrate

There is no chunk table here and no corpus to build. This database holds what
the app itself owns — conversations, tool calls, connected accounts and what
was profiled about them, datasets — while retrieval reads whatever vector store
its asker connected (docs/13-connectors.md), which this app never writes to.

Every `ensure_schema` below is idempotent and the stores call their own on
first use, so this is not a prerequisite. It exists because the failures worth
catching early — a wrong DSN, a region far enough away to blow the latency
budget — are cheap to find here and expensive to find from a spoken turn.
"""

from __future__ import annotations

import argparse
import time

from src.chat.store import CONVERSATIONS, MESSAGES, get_chat_store
from src.chat.tools import TOOL_CALLS, get_tool_call_store
from src.core.db import DatabaseUnavailable, get_db
from src.connectors.profile_store import PROFILES, get_profile_store
from src.connectors.store import ACCOUNTS, get_connector_store
from src.datasets.store import DATASETS, get_dataset_store
from src.integrations.store import AUTH_CONFIGS, CONNECTIONS, get_integration_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    db = get_db()

    print(f"connecting to {db.location}")
    started = time.perf_counter()
    try:
        with db.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            connect_ms = (time.perf_counter() - started) * 1000

            # The number that matters is the *warm* one. Connecting also pays
            # TLS, `CREATE EXTENSION` and the type lookup, none of which a
            # search repeats, so reporting that as the round trip would
            # overstate the link by an order of magnitude.
            samples = []
            for _ in range(5):
                tick = time.perf_counter()
                cur.execute("SELECT 1")
                cur.fetchone()
                samples.append((time.perf_counter() - tick) * 1000)
    except DatabaseUnavailable as error:
        raise SystemExit(f"cannot reach the database: {error}") from error

    rtt_ms = sorted(samples)[len(samples) // 2]
    print(f"  {str(row[0]).split(' on ')[0] if row else 'connected'}")
    print(f"  connect (once, at boot): {connect_ms:.0f} ms")
    print(f"  warm round trip: {rtt_ms:.1f} ms (median of {len(samples)})")

    # Every saved turn and every credential read pays this. Against
    # extraction's measured 78 ms p50 it is what decides whether the 200 ms
    # deadline in docs/04-latency.md still holds.
    if rtt_ms > 40:
        print("  warning: too far to hold the 200 ms budget — move the database closer")
    elif rtt_ms > 15:
        print("  note: workable, but the latency headroom is thinner than in-process")

    # Conversations — the saved turns, needed whether or not retrieval is on.
    get_chat_store().ensure_schema()
    print(f"schema ready: {CONVERSATIONS}, {MESSAGES}")

    # What the agent ran, beside the conversation that caused it. A tool call
    # is the one thing a spoken turn does that has an effect outside this app,
    # so it gets its own audit trail (src/chat/tools.py).
    get_tool_call_store().ensure_schema()
    print(f"schema ready: {TOOL_CALLS}")

    # Connectors — the outside services a user attaches to their account, with
    # their credentials encrypted. One of them *is* retrieval: a connected
    # vector store is the only thing that answers that user's questions
    # (docs/13-connectors.md).
    get_connector_store().ensure_schema()
    print(f"schema ready: {ACCOUNTS}")

    # What this app understood about each of those connected stores
    # (docs/17-understanding.md). Ordered after the accounts because it carries
    # a foreign key into them — `ensure_schema` does that itself, so the order
    # here is for the printed output rather than for correctness.
    get_profile_store().ensure_schema()
    print(f"schema ready: {PROFILES}")

    # The toolkits connected through somebody's Composio. Composio-specific,
    # so it lives apart from the generic credential table above.
    get_integration_store().ensure_schema()
    print(f"schema ready: {AUTH_CONFIGS}, {CONNECTIONS}")

    # Datasets a user has pointed at a URL, and what was measured of each. The
    # profile lives here; the rows it describes live in a DuckDB file on local
    # disk (docs/18-datasets.md). Same reasoning again: needed whether or not
    # retrieval is on, since this answers in SQL rather than by embedding.
    get_dataset_store().ensure_schema()
    print(f"schema ready: {DATASETS}")

    db.close()


if __name__ == "__main__":
    main()
