"use client";

import { AlertCircle, Blocks, Check, Loader2, Plug, Search, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { PanelChip, PanelEmpty } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  connectToolkit,
  disconnectToolkit,
  IntegrationError,
  listConnections,
  listToolkits,
  type Connection,
  type Toolkit,
} from "@/lib/integrations";
import { cn } from "@/lib/utils";

/**
 * What somebody does *after* attaching Composio: link Gmail, Slack, Notion.
 *
 * Composio is the only connector with a second step — the vector stores are
 * done once their credentials verify, but a Composio project is a doorway to
 * further consent screens. So this lives apart from the connector list rather
 * than inside it, and only ever renders once Composio is attached.
 */
export function ComposioToolkits() {
  const [connections, setConnections] = useState<Connection[] | null>(null);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [reloads, setReloads] = useState(0);
  const reload = useCallback(() => setReloads((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();

    void listConnections(controller.signal).then((body) => {
      if (controller.signal.aborted) return;
      setConnections(body.connections);
    });

    return () => controller.abort();
  }, [reloads]);

  const pending = useMemo(
    () => (connections ?? []).some((connection) => connection.pending),
    [connections],
  );

  // A connection sitting in INITIALIZING has its consent screen open in another
  // tab. Poll while that is true and stop the moment it is not.
  useEffect(() => {
    if (!pending) return;
    const timer = setInterval(reload, 3_000);
    return () => clearInterval(timer);
  }, [pending, reload]);

  const term = query.trim();
  const [browsing, setBrowsing] = useState(false);
  const catalogue = term.length > 0 || browsing;
  const [results, setResults] = useState<{ term: string; toolkits: Toolkit[] } | null>(null);

  useEffect(() => {
    if (!catalogue) return;

    const controller = new AbortController();
    const timer = setTimeout(() => {
      void listToolkits({ search: term, limit: 24, signal: controller.signal }).then(
        (body) => {
          if (controller.signal.aborted) return;
          setResults({ term, toolkits: body.toolkits });
        },
      );
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [term, catalogue]);

  // Tagged with the term it answered, so a slow "no" landing after a fast
  // "notion" is simply not the current results rather than a race to guard.
  const shown = results?.term === term ? results.toolkits : null;

  const connected = useMemo(
    () => new Set((connections ?? []).filter((c) => c.active).map((c) => c.toolkit)),
    [connections],
  );

  const link = async (toolkit: string) => {
    setBusy(toolkit);
    setError(null);
    try {
      const started = await connectToolkit(toolkit);
      // Same tab. A popup is the nicer flow and is also the one browsers block
      // when the opener is a click three promises deep, which this is.
      window.location.href = started.redirectUrl;
    } catch (raised) {
      setError(
        raised instanceof IntegrationError ? raised.message : "Could not start that.",
      );
      setBusy(null);
    }
  };

  const unlink = async (toolkit: string) => {
    setBusy(toolkit);
    setError(null);
    try {
      await disconnectToolkit(toolkit);
      reload();
    } catch (raised) {
      setError(
        raised instanceof IntegrationError ? raised.message : "Could not disconnect that.",
      );
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      {/* the <label> is the field here, which is why `.glass-field` lights
          on `focus-within` rather than on `focus` */}
      <label className="glass-field flex items-center gap-2 rounded-lg px-2.5 py-1.5">
        <Search aria-hidden className="size-3.5 shrink-0 text-ink-muted" />
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search services to link"
          className="min-w-0 flex-1 bg-transparent text-[0.75rem] text-ink outline-none placeholder:text-ink-muted"
        />
        {catalogue && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setBrowsing(false);
            }}
            aria-label="Back to your linked services"
            className="glass-row rounded-full p-0.5 text-ink-muted hover:text-ink"
          >
            <X aria-hidden className="size-3.5" />
          </button>
        )}
      </label>

      {error && (
        <p
          role="alert"
          className="glass-alert flex items-start gap-1.5 rounded-lg px-2.5 py-1.5 text-[0.7rem] leading-snug text-destructive"
        >
          <AlertCircle aria-hidden className="mt-px size-3 shrink-0" />
          {error}
        </p>
      )}

      <ScrollArea className="max-h-56">
        <div className="pr-2">
          {catalogue ? (
            <Catalogue results={shown} connected={connected} busy={busy} onLink={link} />
          ) : (
            <Linked
              connections={connections}
              busy={busy}
              onUnlink={unlink}
              onBrowse={() => setBrowsing(true)}
            />
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function Waiting({ children }: { children: string }) {
  return (
    <p className="flex items-center justify-center gap-2 px-3 py-6 text-[0.75rem] text-ink-muted">
      <Loader2 aria-hidden className="size-3.5 animate-spin" />
      {children}
    </p>
  );
}

function Linked({
  connections,
  busy,
  onUnlink,
  onBrowse,
}: {
  connections: Connection[] | null;
  busy: string | null;
  onUnlink: (toolkit: string) => void;
  onBrowse: () => void;
}) {
  if (connections === null) return <Waiting>Loading linked services…</Waiting>;

  if (connections.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 px-6 py-5 text-center">
        <Blocks aria-hidden className="size-4 text-ink-muted" />
        <p className="max-w-52 text-[0.75rem] leading-relaxed text-ink-muted">
          Nothing linked through Composio yet.
        </p>
        <Button type="button" variant="ghost" size="xs" onClick={onBrowse}>
          Browse services
        </Button>
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-0.5">
      {connections.map((connection) => (
        <li
          key={connection.toolkit}
          className="glass-row group/row flex items-center gap-3 rounded-lg px-2 py-2"
        >
          <Logo src={connection.logo} />
          <div className="flex min-w-0 flex-col">
            <p className="truncate text-[0.8rem] font-medium text-ink">
              {connection.name ?? connection.toolkit}
            </p>
            <p className="truncate text-[0.7rem] text-ink-muted">
              <Status connection={connection} />
            </p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="xs"
            disabled={busy === connection.toolkit}
            onClick={() => onUnlink(connection.toolkit)}
            className="glass-row-danger ml-auto shrink-0 rounded-full text-ink-muted opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100"
          >
            {busy === connection.toolkit ? (
              <Loader2 aria-hidden className="animate-spin" />
            ) : (
              "Unlink"
            )}
          </Button>
        </li>
      ))}
    </ul>
  );
}

/**
 * Composio's status verbatim where it is not one of the two everyday ones. A
 * REVOKED account read as "not connected" would be indistinguishable from one
 * that was never connected, and those need different next actions.
 */
function Status({ connection }: { connection: Connection }) {
  if (connection.active) return <>Linked</>;
  if (connection.pending) return <>Waiting for consent…</>;
  return <>{connection.status.toLowerCase().replace(/_/g, " ")}</>;
}

function Catalogue({
  results,
  connected,
  busy,
  onLink,
}: {
  results: Toolkit[] | null;
  connected: Set<string>;
  busy: string | null;
  onLink: (toolkit: string) => void;
}) {
  if (results === null) return <Waiting>Searching…</Waiting>;
  if (results.length === 0) {
    return <PanelEmpty icon={Search}>No service matches that.</PanelEmpty>;
  }

  return (
    <ul className="flex flex-col gap-0.5">
      {results.map((toolkit) => {
        const already = connected.has(toolkit.slug);
        const working = busy === toolkit.slug;

        return (
          <li
            key={toolkit.slug}
            className="glass-row flex items-center gap-3 rounded-lg px-2 py-2"
          >
            <Logo src={toolkit.logo} />
            <div className="flex min-w-0 flex-col">
              <p className="truncate text-[0.8rem] font-medium text-ink">{toolkit.name}</p>
              <p className="truncate text-[0.7rem] text-ink-muted">
                {toolkit.noAuth
                  ? "No account needed"
                  : /* Composio has no OAuth app of its own for this one — it
                       needs an auth config in the user's own dashboard first. */
                    !toolkit.connectable
                    ? "Needs setup in your Composio dashboard"
                    : (toolkit.description ?? `${toolkit.tools} tools`)}
              </p>
            </div>
            <div className="ml-auto shrink-0">
              {already ? (
                <PanelChip className="flex items-center gap-1">
                  <Check aria-hidden className="size-2.5" />
                  Linked
                </PanelChip>
              ) : (
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  disabled={working || !toolkit.connectable || toolkit.noAuth}
                  onClick={() => onLink(toolkit.slug)}
                  className="rounded-full text-ink-muted hover:text-ink"
                >
                  {working ? (
                    <Loader2 aria-hidden className="animate-spin" />
                  ) : (
                    <Plug aria-hidden />
                  )}
                  {working ? "Opening…" : "Link"}
                </Button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Composio serves the logos. A remote image that 404s would leave a broken
 * glyph in a 28px box, so a failure falls back to the same placeholder an
 * entry with no logo gets.
 *
 * Deliberately a plain `<img>`: `next/image` would need every Composio CDN
 * host in `next.config.ts`, and the list is however many services they have.
 */
function Logo({ src }: { src: string | null }) {
  const [broken, setBroken] = useState(false);

  return (
    <span
      className={cn(
        "glass-tile grid size-7 shrink-0 place-items-center overflow-hidden rounded-lg",
        "text-ink-muted",
      )}
    >
      {src && !broken ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt=""
          width={28}
          height={28}
          loading="lazy"
          onError={() => setBroken(true)}
          className="size-full object-contain p-1"
        />
      ) : (
        <Blocks aria-hidden className="size-3.5" />
      )}
    </span>
  );
}
