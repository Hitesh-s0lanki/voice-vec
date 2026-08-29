import type { Metadata } from "next";
import { Suspense } from "react";

import { ConsentResult } from "@/components/consent-result";

export const metadata: Metadata = {
  title: "Connecting",
  description:
    "Where Composio sends the browser back after you approve a toolkit for your own account.",
  // A consent landing page is a redirect target, never a destination.
  robots: { index: false, follow: false },
};

/**
 * Where Composio sends the browser back after consent.
 *
 * The redirect lands here carrying `?toolkit=<slug>` — put on by
 * `IntegrationService._callback`, because Composio's own query string says
 * whether *something* worked and not what it was for.
 *
 * The outcome is not read out of that query string either way. A URL bar is
 * something anyone can type, and "connected" is exactly the claim worth
 * checking: the page asks our own backend, which asks Composio, and reports
 * what came back. So a hand-typed `?toolkit=gmail` shows whatever the account
 * actually has, which for most visitors is "not connected".
 */
export default function IntegrationPage() {
  return (
    <main className="stage flex min-h-dvh flex-1 flex-col items-center justify-center gap-4 px-4 py-24 sm:px-6">
      {/*
        `useSearchParams` opts the subtree into client-side rendering, and
        without a boundary that would opt the whole route out of static
        rendering with a build-time error. The fallback is the same bead the
        page rests on, so the swap is invisible.
      */}
      <Suspense fallback={<span aria-hidden className="bead size-5 rounded-full" />}>
        <ConsentResult />
      </Suspense>
    </main>
  );
}
