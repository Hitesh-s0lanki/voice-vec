"use client";

import { useCallback, useSyncExternalStore } from "react";

/**
 * A media query as React state.
 *
 * `useSyncExternalStore` rather than an effect, because the server snapshot
 * has to be an explicit answer: there is no viewport during prerender, so
 * every query reads false there and the markup React sends is the wide one.
 * The subscription corrects it on the client before paint, and since the only
 * thing that hangs off these queries is *which wrapper* a closed panel gets,
 * the correction is invisible — nothing on screen is keyed to it at rest.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    [query],
  );

  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/**
 * Phone or tablet: anything below Tailwind's `lg`.
 *
 * That is the line the whole layout turns on. Above it there are two free
 * corners for the transcript and the activity log and a third for the rail;
 * below it there is one column, the rail sits centred under it, and the
 * panels open in the middle of the screen instead of off an anchor.
 */
export function useCompactLayout(): boolean {
  return useMediaQuery("(max-width: 1023.98px)");
}
