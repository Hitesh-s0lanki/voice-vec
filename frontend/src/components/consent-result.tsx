"use client";

import { useAuth } from "@clerk/nextjs";
import { AlertCircle, Check, Loader2, Lock } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { connectionStatus, type Connection } from "@/lib/integrations";

/**
 * What happened on the way back from a consent screen.
 *
 * Composio marks a connection ACTIVE when the provider's callback reaches it,
 * which is a server-to-server hop that can land after the browser redirect it
 * raced. So arriving here and reading INITIALIZING is the normal case, not the
 * failure one — this polls for a few seconds before it will call anything
 * pending a problem.
 *
 * Nothing here trusts the URL. `?toolkit=` names which connection to ask
 * about; the answer comes from the backend, scoped to the verified account.
 */

/** Long enough to cover the provider→Composio hop, short enough to not hang. */
const ATTEMPTS = 12;
const EVERY_MS = 1_500;

type Phase = "checking" | "connected" | "pending" | "failed" | "missing";

export function ConsentResult() {
  const { isSignedIn, isLoaded } = useAuth();
  const toolkit = useSearchParams().get("toolkit") ?? "";

  const [phase, setPhase] = useState<Phase>("checking");
  const [connection, setConnection] = useState<Connection | null>(null);

  useEffect(() => {
    if (!isLoaded || !isSignedIn || !toolkit) return;

    const controller = new AbortController();
    let attempts = 0;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      const found = await connectionStatus(toolkit, controller.signal);
      if (controller.signal.aborted) return;

      setConnection(found);

      if (found?.active) return setPhase("connected");
      if (!found) return setPhase("missing");

      if (!found.pending) return setPhase("failed");

      if (++attempts >= ATTEMPTS) return setPhase("pending");
      timer = setTimeout(poll, EVERY_MS);
    };

    void poll();

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [isLoaded, isSignedIn, toolkit]);

  if (!isLoaded) return <Bead />;

  if (!isSignedIn) {
    return (
      <Result
        icon={<Lock aria-hidden className="size-4 text-ink-muted" />}
        title="Sign in to see this"
        body="Connected accounts belong to an account. Sign in and open the Integration panel."
      />
    );
  }

  if (!toolkit) {
    return (
      <Result
        icon={<span aria-hidden className="bead size-5 rounded-full" />}
        title="Integration"
        body="Connect Vec to the services it should be able to reach. The rail's Integration panel is where they live."
      />
    );
  }

  const name = connection?.name ?? toolkit;

  if (phase === "checking") {
    return (
      <Result
        icon={<Loader2 aria-hidden className="size-4 animate-spin text-ink-muted" />}
        title={`Connecting ${name}…`}
        body="Waiting for the provider to confirm. This usually takes a second or two."
      />
    );
  }

  if (phase === "connected") {
    return (
      <Result
        icon={<Check aria-hidden className="size-4 text-ink" />}
        title={`${name} is connected`}
        body="Vec can reach it from your account now. Manage it any time from the Integration panel."
      />
    );
  }

  if (phase === "pending") {
    return (
      <Result
        icon={<Loader2 aria-hidden className="size-4 animate-spin text-ink-muted" />}
        title={`${name} is still connecting`}
        body="The provider hasn't confirmed yet. It may finish on its own — the Integration panel will show it when it does."
      />
    );
  }

  if (phase === "missing") {
    return (
      <Result
        icon={<AlertCircle aria-hidden className="size-4 text-ink-muted" />}
        title={`${name} isn't connected`}
        body="Nothing was connected under this account. Try again from the Integration panel."
      />
    );
  }

  return (
    <Result
      icon={<AlertCircle aria-hidden className="size-4 text-destructive" />}
      title={`${name} didn't connect`}
      body={
        connection
          ? `The provider reported ${connection.status.toLowerCase().replace(/_/g, " ")}. Try connecting it again.`
          : "Try connecting it again from the Integration panel."
      }
    />
  );
}

function Bead() {
  return <span aria-hidden className="bead size-5 rounded-full" />;
}

function Result({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    // A card rather than bare copy on the stage: this is the one screen a
    // provider redirects into, and a surface with an edge is what says the
    // journey ended somewhere rather than dropping the reader on a blank page.
    <div className="glass rise flex w-full max-w-md flex-col items-center gap-3 rounded-2xl px-8 py-9">
      <span className="glass-tile grid size-9 place-items-center rounded-xl">
        {icon}
      </span>
      <h1 className="text-2xl font-medium tracking-[-0.02em] text-ink">{title}</h1>
      <p className="max-w-sm text-center text-[0.9rem] leading-relaxed text-ink-muted">
        {body}
      </p>
      <Button asChild size="sm" className="mt-1">
        <Link href="/">Back to Vec</Link>
      </Button>
    </div>
  );
}
