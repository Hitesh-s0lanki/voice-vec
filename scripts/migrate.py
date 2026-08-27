"""Apply the store schema, and prove the database is reachable before ingest.

    uv run python -m scripts.migrate

Ingest calls `ensure_schema` itself, so this is not a prerequisite — it exists
because the failures worth catching early (wrong DSN, `vector` extension not
available, a region far enough away to blow the latency budget) are all cheap to
find here and expensive to find thirteen minutes into an embedding run.

`--recreate` drops the table. `--indexes` builds HNSW and GIN, which ingest
already does at the end of a run; use it only after loading rows some other way.
"""

from __future__ import annotations

import argparse
import time

from src.core.config import get_settings
from src.connectors.store import ACCOUNTS, get_connector_store
from src.rag.store import StoreUnavailable, get_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="drop the table first")
    parser.add_argument("--indexes", action="store_true", help="also build the indexes")
    parser.add_argument(
        "--english",
        action="store_true",
        help="embed the English original of every row missing embedding_en",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = get_store()

    print(f"connecting to {store.location}")
    started = time.perf_counter()
    try:
        with store.pool.connection() as conn, conn.cursor() as cur:
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
    except StoreUnavailable as error:
        raise SystemExit(f"cannot reach the database: {error}") from error

    rtt_ms = sorted(samples)[len(samples) // 2]
    print(f"  {str(row[0]).split(' on ')[0] if row else 'connected'}")
    print(f"  connect (once, at boot): {connect_ms:.0f} ms")
    print(f"  warm round trip: {rtt_ms:.1f} ms (median of {len(samples)})")

    # Every search pays this twice — SET LOCAL, then the query. Against
    # extraction's measured 78 ms p50 it is what decides whether the 200 ms
    # deadline in docs/04-latency.md still holds after the move off in-process.
    if rtt_ms > 40:
        print("  warning: too far to hold the 200 ms budget — move the database closer")
    elif rtt_ms > 15:
        print("  note: workable, but the latency headroom is thinner than in-process")

    store.ensure_schema(settings.embed_dim, recreate=args.recreate)
    print(f"schema ready: {store.table} (vector({settings.embed_dim}))")

    # Connectors — the outside services a user attaches to their account, with
    # their credentials encrypted. Same reasoning again: needed whether or not
    # retrieval is on, and in fact one of them *is* retrieval, since a
    # connected vector store is what answers that user's questions
    # (docs/13-connectors.md).
    get_connector_store().ensure_schema()
    print(f"schema ready: {ACCOUNTS}")

    if args.english:
        # The migration for an index built before the English column existed.
        # Only the rows that need it, so it is resumable: interrupt it and the
        # next run picks up exactly where it stopped, because "needs a vector"
        # is a question the table answers.
        from src.rag.embed import get_embedder

        embedder = get_embedder()
        backlog = store.english_backlog()
        print(f"embedding {len(backlog)} English originals")

        if backlog:
            embedder.warm()
            started = time.perf_counter()
            batch_size = 128

            for offset in range(0, len(backlog), batch_size):
                batch = backlog[offset : offset + batch_size]
                vectors = embedder.embed_passages(
                    [text for _, text in batch], batch_size=len(batch)
                )
                store.backfill_english([key for key, _ in batch], vectors)

                done = offset + len(batch)
                rate = done / (time.perf_counter() - started)
                print(
                    f"  {done}/{len(backlog)}  {rate:.0f}/s  "
                    f"eta {(len(backlog) - done) / rate / 60:.1f} min   ",
                    end="\r",
                    flush=True,
                )
            print(f"\n  done in {(time.perf_counter() - started) / 60:.1f} min")
            # The partial HNSW index only covers non-null rows, so it has to be
            # rebuilt over what just stopped being null.
            args.indexes = True

    if args.indexes:
        started = time.perf_counter()
        store.create_indexes()
        print(f"indexes built in {time.perf_counter() - started:.1f}s")

    print(f"rows: {store.count()}")
    store.close()


if __name__ == "__main__":
    main()
