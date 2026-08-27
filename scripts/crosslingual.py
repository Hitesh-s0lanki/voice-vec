"""Measure what a cross-lingual question actually scores, and set the floor.

    uv run python -m scripts.crosslingual --n 200

Gate 1 no longer refuses a question asked in a language the index does not
hold: `multilingual-e5` puts a question and its translation in the same region
of the same space, so an English question can retrieve a Hindi passage, and
every chunk carries the original English to answer from. What that leaves open
is Gate 2's floor. `RETRIEVAL_FLOOR` was swept on Hindi questions against Hindi
passages (docs/09-v1.md); a cross-lingual pair scores lower on the same index,
and holding it to that floor abstains on retrieval that was right.

So this measures the other distribution. MSMARCO-XI is unusually good ground
truth for it: every row carries `Eng_Query` — the original English question —
beside the Hindi `query` it was translated into, and `is_selected` marks which
passage answers it. The same query, in two languages, against one index.

What comes out is a table of floors against what each keeps and each costs,
and the number that goes in `RETRIEVAL_FLOOR_CROSS_LINGUAL`. Re-run it after
any re-ingest, for the same reason the Hindi floor has to be re-swept: both are
properties of the index, not of the model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb

from src.core.config import get_settings
from src.rag.embed import get_embedder
from src.rag.store import get_store

SOURCE = Path("data/hinval-2000.parquet")
REPORT = Path("reports/crosslingual.json")


def sample(n: int, seed: int) -> list[dict]:
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT query_id, Eng_Query, query, Answer, is_selected FROM '{SOURCE}'"
    ).fetchall()

    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    return [
        {
            "query_id": int(r[0]),
            "english": (r[1] or "").lstrip(". ").strip(),
            "hindi": (r[2] or "").strip(),
            # The abstention label docs/07-evaluation.md uses: no passage was
            # selected, so there is no answer to find and a low score is right.
            "answerable": bool(sum(r[4] or [])),
        }
        for r in picked
        if (r[1] or "").strip() and (r[2] or "").strip()
    ]


# The three ways the same question can reach the same passages.
ARMS = {
    # the native path, and the baseline everything else is judged against
    "hindi": ("hindi", False),
    # the cross-lingual hop: an English question against the Hindi vectors
    "english": ("english", False),
    # native again, on the English original the Hindi was translated from
    "english_native": ("english", True),
}


def probe(rows: list[dict]) -> list[dict]:
    """Ask each question three ways against the same index."""
    settings, embedder, store = get_settings(), get_embedder(), get_store()
    out: list[dict] = []

    for index, row in enumerate(rows, 1):
        measured = {"query_id": row["query_id"], "answerable": row["answerable"]}

        for arm, (language, english) in ARMS.items():
            hits = store.search(
                embedder.embed_query(row[language]),
                strategies=settings.search_strategies,
                limit=settings.search_limit,
                english=english,
            )
            rest = [h.score for h in hits[1:5]]

            gold_rank = None
            for rank, hit in enumerate(hits, 1):
                origins = hit.payload.get("origins") or []
                if any(
                    o.get("queryId") == row["query_id"] and o.get("isSelected")
                    for o in origins
                ):
                    gold_rank = rank
                    break

            measured[arm] = {
                "top": round(hits[0].score, 4) if hits else 0.0,
                "margin": round(hits[0].score - sum(rest) / len(rest), 4) if rest else 0.0,
                "gold_rank": gold_rank,
            }

        out.append(measured)
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}")

    return out


def sweep(
    probed: list[dict], margin_floor: float, language: str = "english"
) -> list[dict]:
    """What each candidate floor keeps, and what it costs."""
    answerable = [r for r in probed if r["answerable"]]
    unanswerable = [r for r in probed if not r["answerable"]]

    table = []
    for floor in [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.845, 0.86]:

        def passes(row: dict) -> bool:
            side = row[language]
            return side["top"] >= floor and side["margin"] >= margin_floor

        # Answered *and* the gold passage was actually retrieved — a floor that
        # lets through confident nonsense is not doing its job.
        right = sum(1 for r in answerable if passes(r) and (r[language]["gold_rank"] or 99) <= 5)
        answered = sum(1 for r in answerable if passes(r))
        leaked = sum(1 for r in unanswerable if passes(r))

        table.append(
            {
                "floor": floor,
                "answered": answered,
                "answered_with_gold_top5": right,
                "coverage": round(answered / max(1, len(answerable)), 4),
                "unanswerable_leaked": leaked,
                "abstention_recall": round(
                    1 - leaked / max(1, len(unanswerable)), 4
                ),
            }
        )

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="rows to probe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    rows = sample(args.n, args.seed)
    print(f"probing {len(rows)} questions, in Hindi and in English")
    probed = probe(rows)

    def mean(language: str, field: str) -> float:
        values = [r[language][field] for r in probed if r["answerable"]]
        return round(sum(values) / max(1, len(values)), 4)

    def recall_at5(language: str) -> float:
        answerable = [r for r in probed if r["answerable"]]
        found = sum(1 for r in answerable if (r[language]["gold_rank"] or 99) <= 5)
        return round(found / max(1, len(answerable)), 4)

    print("\n  arm                        top score   margin   recall@5")
    for arm in ARMS:
        print(
            f"  {arm:24}  {mean(arm, 'top'):9.4f} {mean(arm, 'margin'):8.4f}"
            f"   {recall_at5(arm):8.4f}"
        )

    # The margin, swept the same way. It is the test that actually decides
    # here: MS MARCO gives a query ten passages and several of them usually
    # answer it, so "nothing stands out" fires on a field of *equally right*
    # passages rather than on an absent answer. Both languages, because that
    # failure has nothing to do with which language was spoken.
    print("\n  margin   coverage (hi / en)   abstention recall (hi / en)")
    for margin in [0.005, 0.01, 0.015, 0.02, 0.03]:
        cells = []
        for language in ("hindi", "english"):
            row = next(
                line
                for line in sweep(probed, margin, language)
                if line["floor"] == (0.845 if language == "hindi" else 0.78)
            )
            cells.append(row)
        print(
            f"  {margin:.3f}   {cells[0]['coverage']:8.2%} / {cells[1]['coverage']:.2%}"
            f"        {cells[0]['abstention_recall']:8.2%} / {cells[1]['abstention_recall']:.2%}"
        )

    table = sweep(probed, settings.retrieval_margin)
    print("\n  floor   coverage   abstention recall   gold@5 kept")
    for line in table:
        print(
            f"  {line['floor']:.3f}   {line['coverage']:8.2%}   {line['abstention_recall']:15.2%}"
            f"   {line['answered_with_gold_top5']:>11}"
        )
    print(f"\n  current RETRIEVAL_FLOOR_CROSS_LINGUAL = {settings.retrieval_floor_cross_lingual}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "n": len(probed),
                "seed": args.seed,
                "margin": settings.retrieval_margin,
                "means": {
                    language: {
                        "top": mean(language, "top"),
                        "margin": mean(language, "margin"),
                        "recall@5": recall_at5(language),
                    }
                    for language in ARMS
                },
                "sweep": table,
                "margin_sweep": {
                    str(margin): {
                        language: next(
                            line
                            for line in sweep(probed, margin, language)
                            if line["floor"] == (0.845 if language == "hindi" else 0.78)
                        )
                        for language in ("hindi", "english")
                    }
                    for margin in [0.005, 0.01, 0.015, 0.02, 0.03]
                },
            },
            indent=2,
        )
    )
    print(f"  written to {REPORT}")
    store = get_store()
    store.close()


if __name__ == "__main__":
    main()
