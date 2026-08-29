"use client";

import {
  AlertCircle,
  Blocks,
  Check,
  ChevronDown,
  Loader2,
  Plug,
  Search,
  Wrench,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { PanelChip, PanelEmpty } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  connectToolkit,
  disconnectToolkit,
  IntegrationError,
  listConnections,
  listTools,
  listToolkits,
  type Connection,
  type Toolkit,
  type ToolkitTools,
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

  /**
   * What those connections actually let the agent do.
   *
   * A second call rather than a field on the first: it is a round trip to
   * Composio for the tool schemas, and the linked list should paint without
   * waiting on it. It arrives late and fills the counts in.
   */
  const [tools, setTools] = useState<ToolkitTools[] | null>(null);
  const [capped, setCapped] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    void listTools(controller.signal).then((body) => {
      if (controller.signal.aborted) return;
      setTools(body.toolkits);
      setCapped(body.limited);
    });

    return () => controller.abort();
  }, [reloads]);

  const byToolkit = useMemo(
    () => new Map((tools ?? []).map((kit) => [kit.toolkit, kit])),
    [tools],
  );

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

      {/* Taller than it was: a linked service now opens onto its tool list,
          and 17 rows behind a 224px window is a scroll bar with a keyhole in
          front of it. The panel's own popover caps the total. */}
      <ScrollArea className="max-h-72">
        <div className="pr-2">
          {catalogue ? (
            <Catalogue results={shown} connected={connected} busy={busy} onLink={link} />
          ) : (
            <Linked
              connections={connections}
              tools={byToolkit}
              loadingTools={tools === null}
              capped={capped}
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
  tools,
  loadingTools,
  capped,
  busy,
  onUnlink,
  onBrowse,
}: {
  connections: Connection[] | null;
  tools: Map<string, ToolkitTools>;
  loadingTools: boolean;
  capped: boolean;
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

  const total = [...tools.values()].reduce((sum, kit) => sum + kit.count, 0);

  return (
    <div className="flex flex-col gap-1">
      <ul className="flex flex-col gap-0.5">
        {connections.map((connection) => (
          <li key={connection.toolkit}>
            <LinkedService
              connection={connection}
              tools={tools.get(connection.toolkit) ?? null}
              loadingTools={loadingTools}
              busy={busy === connection.toolkit}
              onUnlink={() => onUnlink(connection.toolkit)}
            />
          </li>
        ))}
      </ul>

      {/*
        The total, and whether it is the whole total. `tool_schema_limit` caps
        what any one turn is handed, so a user with six services linked can be
        told the truth — "these are the 40 it can reach for" — rather than
        being left to wonder why the seventh never gets used.
      */}
      {total > 0 && (
        <p className="px-2 pt-0.5 text-[0.66rem] text-ink-muted">
          {total} {total === 1 ? "tool" : "tools"} available to the agent
          {capped && " — the per-turn limit, so there may be more behind it"}
        </p>
      )}
    </div>
  );
}

/**
 * One linked service, and what it actually lets the agent do.
 *
 * The count is the point of this row. "Gmail · Linked" says a consent screen
 * was completed; "Gmail · 14 tools" says what completing it bought, and
 * opening the row names them — which is the only place in the app you can
 * check what the model is allowed to reach for before it reaches.
 *
 * Read through the agent's own `tools_for`, so it is bounded the same way the
 * turn is. A service that is linked but contributes nothing shows zero rather
 * than nothing, because that is a real and confusing state worth naming.
 */
function LinkedService({
  connection,
  tools,
  loadingTools,
  busy,
  onUnlink,
}: {
  connection: Connection;
  tools: ToolkitTools | null;
  loadingTools: boolean;
  busy: boolean;
  onUnlink: () => void;
}) {
  const count = tools?.count ?? 0;
  const listable = connection.active && count > 0;

  const summary = (
    <>
      <Logo src={connection.logo} />
      <div className="flex min-w-0 flex-col">
        <p className="truncate text-[0.8rem] font-medium text-ink">
          {connection.name ?? connection.toolkit}
        </p>
        <p className="flex items-center gap-1.5 truncate text-[0.7rem] text-ink-muted">
          <Status connection={connection} />
          {connection.active && (
            <>
              <span aria-hidden>·</span>
              <span className="tabular-nums">
                {loadingTools && tools === null
                  ? "counting tools…"
                  : `${count} ${count === 1 ? "tool" : "tools"}`}
              </span>
            </>
          )}
        </p>
      </div>

      <Button
        type="button"
        variant="ghost"
        size="xs"
        disabled={busy}
        /*
          Inside a <summary>, so the click has to be stopped from toggling the
          row open on its way past. `preventDefault` is what does that — the
          disclosure is the summary's default action, not a listener.
        */
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onUnlink();
        }}
        className="glass-row-danger ml-auto shrink-0 rounded-full text-ink-muted opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100"
      >
        {busy ? <Loader2 aria-hidden className="animate-spin" /> : "Unlink"}
      </Button>
    </>
  );

  // Nothing to open. A service still waiting on consent, or one whose tools
  // could not be read, gets the same row without a disclosure rather than an
  // arrow that opens onto an empty box.
  if (!listable) {
    return (
      <div className="glass-row group/row flex items-center gap-3 rounded-lg px-2 py-2">
        {summary}
      </div>
    );
  }

  return (
    <details className="glass-row group/row group rounded-lg">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-lg px-2 py-2 [&::-webkit-details-marker]:hidden">
        {summary}
        <ChevronDown
          aria-hidden
          className="size-3 shrink-0 text-ink-muted transition-transform group-open:rotate-180"
        />
      </summary>

      <ul className="flex flex-col gap-0.5 px-2 pb-2">
        {tools?.tools.map((tool) => (
          <li key={tool.slug} className="flex items-start gap-2 py-1">
            <Wrench aria-hidden className="mt-1 size-3 shrink-0 text-ink-muted" />
            <div className="min-w-0">
              {/* The label reads, the slug identifies — and the slug is what
                  a conversation's tool call shows, so both are here. */}
              <p className="text-[0.74rem] leading-snug text-ink-soft">
                {tool.name}
                <span className="ml-1.5 font-mono text-[0.62rem] text-ink-muted">
                  {tool.slug}
                </span>
              </p>
              {tool.description && (
                <p className="text-[0.68rem] leading-snug text-ink-muted">
                  {tool.description}
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>
    </details>
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
                    : /* A gateway key reaches Composio through its MCP
                         endpoint, which names toolkits and counts nothing —
                         so "0 tools" here would be a fact about the
                         transport, not about the toolkit. */
                      (toolkit.description ??
                        (toolkit.tools > 0 ? `${toolkit.tools} tools` : "Ready to link"))}
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
