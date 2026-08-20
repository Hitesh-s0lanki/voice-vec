"""Find questions this index can actually answer, and cache them for the UI.

    uv run python -m scripts.suggestions --n 12

A corpus of 2,000 MSMARCO-XI rows answers ~2,000 specific 2016 web questions and
nothing else. Without a hint, the first thing anyone asks is something it has
never heard of, gets an abstention, and concludes the system is broken — when it
is in fact working exactly as designed.

So the app shows real in-corpus questions to try. They are verified here rather
than assumed: each candidate is run through the live pipeline and kept only if
it comes back `answered`, because a suggestion that abstains is worse than no
suggestion at all.

Talks to the running API over HTTP on purpose — embedded Qdrant is single-writer,
so this cannot open the store while the service holds it.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

from scripts.ingest import DEFAULT_CACHE

OUTPUT = Path("data/suggestions.json")


def ask(api: str, transcript: str) -> dict | None:
    body = json.dumps(
        {"transcript": transcript, "languageCode": "hi-IN", "effort": 1}
    ).encode()
    request = urllib.request.Request(
        f"{api}/ask", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12, help="suggestions to collect")
    parser.add_argument("--tries", type=int, default=120, help="candidates to test")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--source", default=str(DEFAULT_CACHE))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if ask(args.api, "warm") is None:
        raise SystemExit(f"no API at {args.api} — start it with `uv run python -m src.main`")

    # Only rows with a gold passage: if the labels say the corpus cannot answer
    # it, it has no business being offered as a suggestion.
    rows = duckdb.connect().execute(
        f"""
        SELECT query, query_type FROM read_parquet('{args.source}')
        WHERE list_sum(is_selected) > 0 AND length(query) BETWEEN 12 AND 70
        """
    ).fetchall()

    random.Random(args.seed).shuffle(rows)
    print(f"testing up to {args.tries} of {len(rows)} answerable candidates")

    found: list[dict] = []
    for index, (query, query_type) in enumerate(rows[: args.tries]):
        response = ask(args.api, query)
        if response and response["status"] == "answered":
            found.append(
                {
                    "query": query,
                    "queryType": query_type,
                    "confidence": response["confidence"],
                }
            )
            print(f"  ✓ {query}")
        if len(found) >= args.n:
            break

    tested = index + 1
    print(f"\n{len(found)} answered out of {tested} tested ({len(found) / tested:.0%})")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {"suggestions": found, "tested": tested, "answered": len(found)},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
