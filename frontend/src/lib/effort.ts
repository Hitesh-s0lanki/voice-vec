"use client";

import { usePersistentState } from "@/hooks/use-persistent-state";

/**
 * How hard the agent should work before it answers. Held per-device, read by
 * the panel that sets it and by the voice session that spends it.
 *
 * Each rung is a different retrieval architecture, not a different amount of
 * the same one — the ladder in `src/rag/effort.py`, which this mirrors. Two
 * things about it are worth knowing before touching this file:
 *
 * **The level is a ceiling, not a floor.** Asking for Adaptive does not mean
 * four model calls happen; it means up to four may. A question the answer
 * cache already holds costs one embedding at any rung, and the response says
 * which rung actually answered.
 *
 * **The first two rungs make no network call after the transcript arrives.**
 * That is the whole 200 ms claim of requirement 3, and it is why `cost` exists
 * at all: a slider whose positions all look equally free is one nobody can
 * make an informed choice with.
 *
 * `hint` is written to fit **one line** in a `w-72` rail panel — about forty
 * characters. `detail` is the fuller sentence, which rides on the row's
 * `title` so nothing is lost to that budget.
 */
export const EFFORT_LEVELS = [
  {
    label: "Lookup",
    hint: "The passages themselves. No model.",
    detail:
      "Searches your sources and shows what came back, verbatim. Nothing is written, so there is nothing to invent.",
    cost: "~60 ms",
  },
  {
    label: "Grounded",
    hint: "A sentence lifted from a passage.",
    detail:
      "Picks the best sentence out of the best passage and verifies it was lifted from there, not paraphrased.",
    cost: "<200 ms",
  },
  {
    label: "Deep",
    hint: "Reranked, then written up by a model.",
    detail:
      "Fuses keyword and vector search, reranks what survives, and has a model write the answer from those passages only.",
    cost: "~1 s · 1 call",
  },
  {
    label: "Corrective",
    hint: "Grades what it found, then retries.",
    detail:
      "Judges whether the retrieval actually bears on your question, and if it does not, rewrites the question and searches again.",
    cost: "~5 s · 4 calls",
  },
  {
    label: "Adaptive",
    hint: "Routes first, then checks its answer.",
    detail:
      "Decides whether searching will help before it searches, then checks its own answer and repairs whichever half was wrong.",
    cost: "~10 s · 8 calls",
  },
] as const;

/**
 * Below this the pipeline makes no network call after the transcript, which is
 * what the 200 ms budget is a claim about. Above it, latency is reported
 * against that rung's own budget instead — see `budgetMs` on the response.
 */
export const OFFLINE_MAX_LEVEL = 1;

export const EFFORT_STORAGE_KEY = "vec-effort";

export const DEFAULT_EFFORT = 1;

export function reviveEffort(raw: unknown): number | null {
  if (typeof raw !== "number" || !Number.isInteger(raw)) return null;
  if (raw < 0 || raw >= EFFORT_LEVELS.length) return null;
  return raw;
}

export function useEffort() {
  return usePersistentState<number>(
    EFFORT_STORAGE_KEY,
    DEFAULT_EFFORT,
    reviveEffort,
  );
}
