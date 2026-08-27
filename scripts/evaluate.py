"""Measure the v1 pipeline against the labels the dataset already ships.

    uv run python -m scripts.evaluate --n 300 --seed 42

Three of the four evaluations in docs/07-evaluation.md, on the slice v1 builds:

  E1  retrieval — recall@1/5/10 and MRR@10, ground truth `is_selected`
  E2  abstention — precision/recall/F1, ground truth `sum(is_selected) == 0`,
      swept over Gate 2's floor and margin so the operating point is chosen
      rather than guessed
  E4  latency — P50/P70/P95/P99/P100 of the transcript→answer window

E3 (answer quality against gold `Answer`) needs an embedding-similarity scorer
and is left for Phase C.

Two things this run is *not*. It calls the service in-process rather than over
HTTP, so it excludes serialisation and the network hop — the Phase C harness
must hit the live endpoint instead. And the gold passage is guaranteed to be in
the corpus, because the corpus was built from these rows, so recall here
measures ranking against distractors, not open-domain retrieval.

The provenance rule that makes E1 meaningful: a hit counts as gold when one of
its origins matches *this* query_id with `isSelected == 1`. Dedup merges
duplicate passages across rows, so the whole origin list has to be checked, not
just the first entry.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from scripts.ingest import DEFAULT_CACHE, HF_URL, read_rows
from src.core.config import get_settings
from src.rag.chunk import normalise
from src.rag.embed import get_embedder
from src.rag.backends.resolve import FixedResolver
from src.rag.cache import AnswerCache
from src.rag.store import get_store
from src.schemas.ask import AskRequest
from src.services.ask_service import AskService
from src.services.metrics_service import MetricsService, percentile

FLOOR_GRID = [round(0.78 + 0.005 * i, 3) for i in range(19)]  # 0.780 … 0.870
MARGIN_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]


def evaluate(rows, n: int, seed: int) -> dict:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()
    embedder.warm()

    sample = random.Random(seed).sample(rows, min(n, len(rows)))

    # The cache is switched off for the sweep, deliberately. The sample is
    # drawn with replacement across grid points and the same query is asked
    # more than once; a cache would serve the second ask from the first and
    # every latency figure after it would be a measurement of Redis.
    service = AskService(
        settings,
        embedder,
        FixedResolver(store),
        MetricsService(settings),
        AnswerCache(settings.model_copy(update={"cache_enabled": False})),
    )

    measurements: list[dict] = []

    for row in sample:
        query = normalise(row.query)
        if not query:
            continue

        vector = embedder.embed_query(query)
        # No language filter, for the same reason the service skips it: the
        # index holds one language, so the filter cannot change the results —
        # and on a filtered ANN search it still costs candidates.
        hits = store.search(vector, strategies=settings.search_strategies, limit=10)

        gold_rank = None
        for rank, hit in enumerate(hits, start=1):
            origins = hit.payload.get("origins") or []
            if any(
                o.get("queryId") == row.query_id and o.get("isSelected")
                for o in origins
            ):
                gold_rank = rank
                break

        scores = [hit.score for hit in hits]
        top = scores[0] if scores else 0.0
        rest = scores[1:5]
        margin = top - (sum(rest) / len(rest)) if rest else 0.0

        # The full pipeline, for the latency window and the realised status.
        started = time.perf_counter()
        response = service.ask(
            AskRequest(transcript=row.query, language_code="hi-IN", effort=1)
        )
        wall_ms = (time.perf_counter() - started) * 1000

        measurements.append(
            {
                "query_id": row.query_id,
                "query_type": row.query_type,
                "answerable": row.answerable,
                "gold_rank": gold_rank,
                "top": top,
                "margin": margin,
                "status": response.status,
                "server_ms": response.timings.total,
                "wall_ms": round(wall_ms, 3),
                "timings": response.timings.model_dump(),
            }
        )

    return {
        "n": len(measurements),
        "seed": seed,
        "settings": {
            "floor": settings.retrieval_floor,
            "margin": settings.retrieval_margin,
            "strategies": settings.search_strategies,
            "model": settings.embed_model,
            "deadline_ms": settings.deadline_ms,
        },
        "e1_retrieval": e1(measurements),
        "e2_abstention": e2(measurements),
        "e4_latency": e4(measurements),
        "measurements": measurements,
    }


def e1(rows: list[dict]) -> dict:
    """Recall@k and MRR@10 over the answerable rows only."""
    answerable = [r for r in rows if r["answerable"]]
    if not answerable:
        return {"n": 0}

    def recall_at(k: int) -> float:
        found = sum(1 for r in answerable if r["gold_rank"] and r["gold_rank"] <= k)
        return round(found / len(answerable), 4)

    mrr = sum(1 / r["gold_rank"] for r in answerable if r["gold_rank"]) / len(answerable)

    by_type: dict[str, dict] = {}
    for r in answerable:
        bucket = by_type.setdefault(r["query_type"], {"n": 0, "hit@5": 0})
        bucket["n"] += 1
        if r["gold_rank"] and r["gold_rank"] <= 5:
            bucket["hit@5"] += 1
    for bucket in by_type.values():
        bucket["recall@5"] = round(bucket["hit@5"] / bucket["n"], 4)

    return {
        "n": len(answerable),
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "mrr@10": round(mrr, 4),
        "by_query_type": by_type,
    }


def _confusion(rows: list[dict], floor: float, margin_floor: float) -> dict:
    """What Gate 2 would have decided at this operating point."""
    correct_abstentions = over_refusals = missed_abstentions = answered_correctly = 0

    for r in rows:
        abstains = r["top"] < floor or r["margin"] < margin_floor
        if r["answerable"]:
            if abstains:
                over_refusals += 1
            else:
                answered_correctly += 1
        else:
            if abstains:
                correct_abstentions += 1
            else:
                missed_abstentions += 1

    abstentions = correct_abstentions + over_refusals
    should_abstain = correct_abstentions + missed_abstentions
    should_answer = answered_correctly + over_refusals

    precision = correct_abstentions / abstentions if abstentions else 0.0
    recall = correct_abstentions / should_abstain if should_abstain else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "floor": floor,
        "margin": margin_floor,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": round(answered_correctly / should_answer, 4) if should_answer else 0.0,
        # The cell requirement 6 is really about: answered something the corpus
        # cannot support. Reported as a raw count, not only as a rate.
        "answered_unanswerable": missed_abstentions,
        "correct_abstentions": correct_abstentions,
        "over_refusals": over_refusals,
    }


def e2(rows: list[dict]) -> dict:
    sweep = [
        _confusion(rows, floor, margin)
        for floor in FLOOR_GRID
        for margin in MARGIN_GRID
    ]
    best = max(sweep, key=lambda p: p["f1"])

    return {
        "n": len(rows),
        "unanswerable": sum(1 for r in rows if not r["answerable"]),
        "realised": _counts(r["status"] for r in rows),
        "best_by_f1": best,
        "sweep": sweep,
    }


def e4(rows: list[dict]) -> dict:
    server = [r["server_ms"] for r in rows]

    # Split by outcome. An abstention stops at Gate 2 and never pays for
    # extraction, so a corpus that abstains often has a median that describes
    # the cheap path rather than the pipeline. Both are reported; neither is
    # excluded, because abstentions are real workload.
    by_status = {
        status: {
            "n": len(times),
            **{f"p{p}": percentile(times, p) for p in (50, 70, 95, 99, 100)},
        }
        for status in ("answered", "abstained", "refused")
        if (times := [r["server_ms"] for r in rows if r["status"] == status])
    }

    return {
        "n": len(server),
        **{f"p{p}": percentile(server, p) for p in (50, 70, 95, 99, 100)},
        "mean": round(sum(server) / len(server), 3) if server else 0.0,
        "within_budget": sum(1 for r in rows if r["server_ms"] <= 200),
        "by_status": by_status,
        "stages": {
            stage: {
                f"p{p}": percentile(
                    [r["timings"][stage] for r in rows if r["timings"].get(stage)], p
                )
                for p in (50, 100)
            }
            for stage in ("guard_in", "embed", "search", "extract", "guard_out")
        },
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="queries to evaluate")
    parser.add_argument("--rows", type=int, default=2000, help="rows to sample from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default=None)
    parser.add_argument("--out", default="reports/results.json")
    args = parser.parse_args()

    source = args.source or (str(DEFAULT_CACHE) if DEFAULT_CACHE.exists() else HF_URL)
    rows = read_rows(source, args.rows)

    results = evaluate(rows, args.n, args.seed)

    e1_ = results["e1_retrieval"]
    e2_ = results["e2_abstention"]
    e4_ = results["e4_latency"]
    best = e2_["best_by_f1"]

    print(f"\nN={results['n']}  seed={results['seed']}  strategies={results['settings']['strategies']}")
    print("\nE1 retrieval (answerable rows only)")
    print(f"  n={e1_['n']}  recall@1={e1_['recall@1']}  recall@5={e1_['recall@5']}"
          f"  recall@10={e1_['recall@10']}  MRR@10={e1_['mrr@10']}")
    for query_type, bucket in sorted(e1_["by_query_type"].items()):
        print(f"    {query_type:<12} n={bucket['n']:<4} recall@5={bucket['recall@5']}")

    print("\nE2 abstention")
    print(f"  unanswerable {e2_['unanswerable']}/{e2_['n']}  realised={e2_['realised']}")
    print(f"  best F1 at floor={best['floor']} margin={best['margin']}: "
          f"precision={best['precision']} recall={best['recall']} f1={best['f1']} "
          f"coverage={best['coverage']} answered-unanswerable={best['answered_unanswerable']}")
    print(f"  current config: {_confusion(results['measurements'], results['settings']['floor'], results['settings']['margin'])}")

    print("\nE4 latency (transcript → answer, in-process, ms)")
    print(f"  p50={e4_['p50']}  p70={e4_['p70']}  p95={e4_['p95']}  p99={e4_['p99']}  p100={e4_['p100']}")
    print(f"  within 200 ms: {e4_['within_budget']}/{e4_['n']}")
    for status, values in e4_["by_status"].items():
        print(f"    {status:<10} n={values['n']:<4} p50={values['p50']:<8} p95={values['p95']:<8} p100={values['p100']}")
    for stage, values in e4_["stages"].items():
        print(f"    {stage:<10} p50={values['p50']:<8} p100={values['p100']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    get_store().close()


if __name__ == "__main__":
    main()
