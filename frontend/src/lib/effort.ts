"use client";

import { usePersistentState } from "@/hooks/use-persistent-state";

/**
 * How hard the agent should think before it answers. Held per-device, and read
 * by both the panel that sets it and the capture flow that spends it.
 *
 * The levels are the escalation ladder from docs/02-architecture.md. Level 0
 * skips retrieval entirely; the backend implements Tier 1 today, so 1–3 all
 * take the same path and the slider caps escalation rather than adding to it.
 */
export const EFFORT_LEVELS = [
  { label: "Instant", hint: "Transcribe and stop. No retrieval pass." },
  { label: "Balanced", hint: "Search your sources and answer from them." },
  { label: "Deep", hint: "Rerank before answering. Not wired up yet." },
  { label: "Max", hint: "Longest reasoning budget. Not wired up yet." },
] as const;

/** Below this the app transcribes and stops — no call to /api/ask. */
export const RETRIEVAL_FLOOR_LEVEL = 1;

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
