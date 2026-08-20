"use client";

import { usePersistentState } from "@/hooks/use-persistent-state";

/**
 * Hands-free: reopen the microphone as soon as the assistant stops talking.
 *
 * Off by default, because it is a real trade. On, the exchange runs like a
 * phone call and nobody touches anything; off, every turn starts with a tap,
 * which is what you want with the speakers up and a room full of people.
 */
export const HANDS_FREE_STORAGE_KEY = "vec-hands-free";

function reviveHandsFree(raw: unknown): boolean | null {
  return typeof raw === "boolean" ? raw : null;
}

export function useHandsFree() {
  return usePersistentState<boolean>(HANDS_FREE_STORAGE_KEY, false, reviveHandsFree);
}
