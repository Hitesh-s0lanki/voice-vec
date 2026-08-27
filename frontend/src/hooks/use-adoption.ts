"use client";

import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";

import { adoptConversations } from "@/lib/conversations";
import { browserSessionId } from "@/lib/identity";

/** One key per account, so a shared browser adopts once for each person. */
const STORAGE_PREFIX = "vec-adopted:";

/**
 * Hand this browser's anonymous conversations to whoever just signed in.
 *
 * Everything said before signing in is filed under the browser's `sess_…`.
 * Signing in writes the account onto those same rows — one `UPDATE`, no
 * message moved — and from then on they are reachable from any device that
 * account signs in on.
 *
 * Runs once per account per browser id, remembered locally so the round trip
 * is not repeated on every mount. Repeating it would be harmless anyway: the
 * backend only claims rows nobody owns yet, so a second call moves nothing
 * and a shared browser can never take a conversation off someone else.
 *
 * A failure here is silent and stays silent. The conversations are still
 * there, still owned by the browser, still listed — the account simply has
 * not claimed them, and the next sign-in tries again.
 */
export function useAdoption() {
  const { isSignedIn, userId } = useAuth();

  useEffect(() => {
    if (!isSignedIn || !userId) return;

    const session = browserSessionId();
    if (!session) return;

    const key = `${STORAGE_PREFIX}${userId}`;
    try {
      if (localStorage.getItem(key) === session) return;
    } catch {
      // storage blocked — adopt every mount rather than never
    }

    void adoptConversations().then(() => {
      try {
        localStorage.setItem(key, session);
      } catch {
        // nothing to remember it with; the repeat is a no-op server-side
      }
    });
  }, [isSignedIn, userId]);
}
