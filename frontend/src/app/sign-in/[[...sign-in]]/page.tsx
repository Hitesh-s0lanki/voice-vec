import type { Metadata } from "next";

import { SignIn } from "@clerk/nextjs";

export const metadata: Metadata = {
  title: "Sign in",
  description:
    "Sign in to keep your conversations, memory and connectors across devices.",
  // Clerk's hosted flows are the canonical ones; these are the same screens at
  // our own address and would only compete with them in an index.
  robots: { index: false, follow: false },
};


export default function SignInPage() {
  return (
    // `.stage` is what gives Clerk's card something to refract — it is a
    // backdrop root, so its own `backdrop-filter` samples the washes painted
    // here. On a bare `<div>` the card blurs the page background and reads as
    // flat white. The glass itself comes from the `.cl-*` block in globals.css.
    <main className="stage flex min-h-dvh flex-1 items-center justify-center px-4 py-20 sm:px-6">
      <SignIn />
    </main>
  );
}
