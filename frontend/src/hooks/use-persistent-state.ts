"use client";

import { useCallback, useSyncExternalStore } from "react";

type Listener = () => void;

const listeners = new Set<Listener>();

function subscribe(listener: Listener) {
  listeners.add(listener);
  // another tab writing the same key should move this one too
  window.addEventListener("storage", listener);

  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

function emit() {
  for (const listener of listeners) listener();
}

/**
 * `getSnapshot` runs on every render and must hand back the *same* reference
 * when nothing changed, or React re-renders forever. Parsed values are keyed
 * to the raw string they came from, so a re-read only re-parses on a real write.
 */
const snapshots = new Map<string, { raw: string | null; value: unknown }>();

function readRaw(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    // blocked (private mode, third-party context) — behave as if unset
    return null;
  }
}

/**
 * State backed by `localStorage`, read through React's external-store channel
 * so the server render and the hydrating client render agree: both start at
 * `fallback`, and the stored value swaps in once hydration is done.
 *
 * `revive` guards whatever is actually in storage — a hand-edited key, or a
 * value written by an older build. Return `null` to reject it.
 *
 * `fallback` and `revive` must be stable references (module constants).
 */
export function usePersistentState<T>(
  key: string,
  fallback: T,
  revive: (raw: unknown) => T | null,
) {
  const getSnapshot = useCallback(() => {
    const raw = readRaw(key);

    const cached = snapshots.get(key);
    if (cached && cached.raw === raw) return cached.value as T;

    let value = fallback;
    if (raw !== null) {
      try {
        const revived = revive(JSON.parse(raw));
        if (revived !== null) value = revived;
      } catch {
        // malformed JSON — fall back rather than throwing mid-render
      }
    }

    snapshots.set(key, { raw, value });
    return value;
  }, [fallback, key, revive]);

  const getServerSnapshot = useCallback(() => fallback, [fallback]);

  const value = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const setValue = useCallback(
    (update: T | ((previous: T) => T)) => {
      const next =
        typeof update === "function"
          ? (update as (previous: T) => T)(getSnapshot())
          : update;

      try {
        localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // quota or private mode — the value still holds for this session
      }

      // seeded from the write, so the snapshot matches even if storage refused
      snapshots.set(key, { raw: readRaw(key), value: next });
      emit();
    },
    [getSnapshot, key],
  );

  return [value, setValue] as const;
}
