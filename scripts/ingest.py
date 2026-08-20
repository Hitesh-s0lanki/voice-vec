"""Build the index: parquet → chunks → embeddings → Postgres.

    uv run python -m scripts.ingest --rows 2000

Offline by default once the rows are cached locally. The remote parquet is
461 MB and DuckDB range-reads whole row groups over HTTPS, so the first pull
costs ~2 minutes even for 2,000 rows — cache it and move on.

v1 ingests S1 (passage-atomic) only. Phase B adds S2–S5 and re-runs this with
`--strategies S1,S2,...`; point ids are derived from the chunk hash, so a
re-run overwrites rather than duplicating and a crashed run resumes by
repeating itself.

Writes to Postgres (`DATABASE_URL`), which unlike the embedded store it
replaced takes concurrent writers — the API can keep serving while this runs.
The HNSW and GIN indexes are built at the end, once, rather than maintained
across every insert.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.rag.chunk import ChunkStrategy, Row, merge_duplicates, passage_atomic
from src.rag.embed import get_embedder
from src.rag.manifest import IndexManifest
from src.rag.store import get_store

HF_URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/"
    "validation/hinval.parquet"
)
DEFAULT_CACHE = Path("data/hinval-2000.parquet")

# multilingual-e5 truncates at 512 tokens without raising, so the run reports how
# many chunks sit at the cap rather than assuming none do. (docs/03-chunking.md
# expected 1.5–2.5× Indic token inflation; measured here it is 1.21×, so the cap
# bites far less than planned for — 0.3% of chunks.)
TOKEN_LIMIT = 512

BUILDERS = {ChunkStrategy.PASSAGE.value: passage_atomic}


def read_rows(source: str, rows: int) -> list[Row]:
    """Flatten the parquet into Row objects, remote or local.

    The cached file is already flat; the HF file keeps the three passage lists
    inside a `passages` struct (docs/01-dataset.md).
    """
    con = duckdb.connect()
    if source.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs;")

    columns = {
        name for (name, *_rest) in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{source}') LIMIT 1"
        ).fetchall()
    }
    nested = "passages" in columns

    passages = (
        """passages.Translated_passages AS indic_passages,
           passages.English_passages    AS english_passages,
           passages.is_selected         AS is_selected"""
        if nested
        else "indic_passages, english_passages, is_selected"
    )

    records = con.execute(
        f"""
        SELECT query_id, query_type, query, Answer, target_lang, {passages}
        FROM read_parquet('{source}')
        LIMIT {rows}
        """
    ).fetchall()

    return [
        Row(
            query_id=int(r[0]),
            query_type=str(r[1]),
            query=str(r[2] or ""),
            answer=str(r[3] or ""),
            language=str(r[4] or "hin_Deva"),
            indic_passages=list(r[5] or []),
            english_passages=list(r[6] or []),
            is_selected=[int(v) for v in (r[7] or [])],
        )
        for r in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000, help="MSMARCO-XI rows to ingest")
    parser.add_argument(
        "--source",
        default=None,
        help=f"parquet path or URL (default: {DEFAULT_CACHE} if present, else HF)",
    )
    parser.add_argument("--strategies", default=ChunkStrategy.PASSAGE.value)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true", help="drop the table first")
    args = parser.parse_args()

    source = args.source or (str(DEFAULT_CACHE) if DEFAULT_CACHE.exists() else HF_URL)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in BUILDERS]
    if unknown:
        raise SystemExit(f"v1 only builds {sorted(BUILDERS)} — asked for {unknown}")

    started = time.perf_counter()

    print(f"reading {args.rows} rows from {source}")
    rows = read_rows(source, args.rows)
    passages = sum(len(row.indic_passages) for row in rows)
    print(f"  {len(rows)} rows, {passages} passages")

    chunks = [c for row in rows for s in strategies for c in BUILDERS[s](row)]
    before = len(chunks)
    chunks = merge_duplicates(chunks)
    dedup_rate = 1 - len(chunks) / before if before else 0.0
    print(f"  {before} chunks → {len(chunks)} after dedup ({dedup_rate:.1%} duplicates)")

    embedder = get_embedder()
    print(f"warming {embedder.model_name}")
    embedder.warm()

    lengths = embedder.count_tokens([c.text for c in chunks])
    token_p99 = at_cap = None
    if lengths:
        ordered = sorted(lengths)
        token_p50 = ordered[len(ordered) // 2]
        token_p99 = ordered[int(0.99 * (len(ordered) - 1))]

        # The tokeniser truncates at the cap before returning, so a count can
        # never come back *over* it — "over 512: 0" would be the check being
        # blind, not the corpus being safe. Chunks sitting exactly at the cap
        # are the ones losing content, and they lose it silently.
        at_cap = sum(1 for n in lengths if n >= TOKEN_LIMIT)
        share = at_cap / len(lengths)
        print(f"  tokens/chunk: p50 {token_p50}, p99 {token_p99}")
        print(f"  at the {TOKEN_LIMIT}-token cap: {at_cap} ({share:.1%}) — truncated, silently")

        # Requirement 0b from the build plan: the real Indic token inflation
        # ratio, measured on aligned pairs rather than assumed. Pairs at the cap
        # are excluded because truncation flattens the ratio towards 1.
        english = [c.english or "" for c in chunks]
        english_lengths = embedder.count_tokens(english)
        pairs = [
            (hi, en)
            for hi, en, text in zip(lengths, english_lengths, english)
            if text and hi < TOKEN_LIMIT and en < TOKEN_LIMIT
        ]
        if pairs:
            ratio = sum(hi / en for hi, en in pairs) / len(pairs)
            print(f"  Indic token inflation: {ratio:.2f}× English, over {len(pairs)} aligned pairs")

        # ONNX pads every batch to its longest member, so a batch of mixed
        # lengths charges each short chunk the price of the longest one.
        # Sorting by length first is the single biggest ingest speedup here and
        # changes nothing about the vectors.
        chunks = [chunk for _, chunk in sorted(zip(lengths, chunks), key=lambda pair: pair[0])]

    store = get_store()
    store.ensure_schema(embedder.dim, recreate=args.recreate)
    print(f"upserting into {store.collection} at {store.location}")

    embedding_started = time.perf_counter()
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        vectors = embedder.embed_passages([c.text for c in batch], batch_size=len(batch))
        store.upsert(batch, vectors)

        done = start + len(batch)
        spent = time.perf_counter() - embedding_started
        rate = done / spent
        eta = (len(chunks) - done) / rate
        print(
            f"  {done}/{len(chunks)}  {rate:.0f} chunks/s  eta {eta / 60:.1f} min   ",
            end="\r",
            flush=True,
        )

    # After the rows land, never alongside them: an HNSW graph maintained
    # across every insert costs far more than one built over a populated table,
    # and nothing queries the index until ingest is done.
    print("\nbuilding indexes (hnsw, gin)")
    index_started = time.perf_counter()
    store.create_indexes()
    print(f"  built in {time.perf_counter() - index_started:.1f}s")

    elapsed = time.perf_counter() - started
    total = store.count()
    print(f"indexed {total} points in {elapsed:.1f}s")

    IndexManifest(
        collection=store.collection,
        model=embedder.model_name,
        dim=embedder.dim,
        strategies=strategies,
        languages=sorted({row.language for row in rows}),
        rows=len(rows),
        passages=passages,
        chunks=total,
        dedup_rate=round(dedup_rate, 4),
        token_p99=token_p99,
        elapsed_seconds=round(elapsed, 1),
        source=source,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Index properties only. Gate 2's floor and margin are query-time knobs
        # tuned per deployment — recording them here would freeze a stale copy
        # of a setting the running service can change without a re-ingest.
        extras={
            "answerable_rows": sum(1 for row in rows if row.answerable),
            "unanswerable_rows": sum(1 for row in rows if not row.answerable),
            "at_token_cap": at_cap,
        },
    ).write()

    store.close()
    print("wrote data/index-manifest.json")


if __name__ == "__main__":
    main()
