"use client";

import { SignInButton, useAuth } from "@clerk/nextjs";
import {
  AlertCircle,
  ArrowLeft,
  Blocks,
  Check,
  Database,
  ExternalLink,
  Loader2,
  Lock,
  Plug,
  Table2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AttachedDatasets } from "@/components/panels/attached-datasets";
import { ComposioToolkits } from "@/components/panels/composio-toolkits";
import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  connect,
  ConnectorError,
  describeHints,
  disconnect,
  listConnectors,
  type Connector,
  type ConnectorKind,
} from "@/lib/connectors";

/**
 * The services attached to this account.
 *
 * Composio for tools; Pinecone, Astra and a user's own Postgres for where their
 * vectors live. One list, because they are the same thing from the outside: an
 * outside service reached with a credential the user handed over.
 *
 * Nothing here knows what a Pinecone key looks like. The backend registry
 * describes each connector's fields and this renders a form from that, so a
 * fifth connector appears here — correctly labelled, correctly masked — without
 * this file changing.
 *
 * Signing in is the gate. The rail's other panels serve anybody holding the
 * browser and this one does not, because a credential is not something a
 * browser should own. The backend enforces it from a verified token; this is
 * the version a person can see.
 */
export function IntegrationPanel() {
  const { isSignedIn, isLoaded } = useAuth();

  if (!isLoaded) {
    return (
      <>
        <Heading />
        <PanelRule />
        <Waiting>Checking your session…</Waiting>
      </>
    );
  }

  if (!isSignedIn) return <SignedOut />;
  return <Connectors />;
}

function Heading({ children }: { children?: React.ReactNode }) {
  return (
    <PanelHeading title="Connectors" hint="Services attached to your account.">
      {children}
    </PanelHeading>
  );
}

function Waiting({ children }: { children: string }) {
  return (
    <p className="flex items-center justify-center gap-2 px-3 py-7 text-[0.75rem] text-ink-muted">
      <Loader2 aria-hidden className="size-3.5 animate-spin" />
      {children}
    </p>
  );
}

function Alert({ children }: { children: string }) {
  return (
    <p
      role="alert"
      className="glass-alert flex items-start gap-1.5 rounded-lg px-2.5 py-1.5 text-[0.7rem] leading-snug text-destructive"
    >
      <AlertCircle aria-hidden className="mt-px size-3 shrink-0" />
      {children}
    </p>
  );
}

function SignedOut() {
  return (
    <>
      <Heading />
      <PanelRule />

      <div className="flex flex-col items-center gap-3 px-6 py-6 text-center">
        <span className="glass-tile grid size-8 place-items-center rounded-xl text-ink-muted">
          <Lock aria-hidden className="size-3.5" />
        </span>
        <p className="max-w-56 text-[0.75rem] leading-relaxed text-ink-muted">
          Connectors belong to an account, not to a browser. Sign in to attach
          Composio or your own vector store.
        </p>
        <SignInButton mode="modal">
          <Button size="sm">Sign in</Button>
        </SignInButton>
      </div>
    </>
  );
}

// One section per kind, in the order somebody scanning this reads them: what
// it can *do*, what it can *search*, what it can *count*. A kind absent from
// this list is a connector the server offers and this panel drops on the floor,
// which is why `ConnectorKind` and this array are edited together.
const GROUPS: { kind: ConnectorKind; label: string; hint: string }[] = [
  { kind: "tools", label: "Tools", hint: "Services Vec can act through" },
  { kind: "dataset", label: "Datasets", hint: "Data Vec can query in SQL" },
  { kind: "vector", label: "Vector store", hint: "Where your embeddings live" },
];

function Connectors() {
  const [connectors, setConnectors] = useState<Connector[] | null>(null);
  const [configured, setConfigured] = useState(true);
  const [backend, setBackend] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const [reloads, setReloads] = useState(0);
  const reload = useCallback(() => setReloads((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();

    void listConnectors(controller.signal).then((body) => {
      if (controller.signal.aborted) return;
      setConnectors(body.connectors);
      setConfigured(body.configured);
      setBackend(body.vectorBackend);
    });

    return () => controller.abort();
  }, [reloads]);

  const selected = useMemo(
    () => (connectors ?? []).find((c) => c.slug === open) ?? null,
    [connectors, open],
  );

  if (connectors === null) {
    return (
      <>
        <Heading />
        <PanelRule />
        <Waiting>Loading…</Waiting>
      </>
    );
  }

  // The server has no COMPOSIO_ENCRYPTION_KEY, so it cannot hold a credential
  // at all. Not the user's to fix, so it is said differently.
  if (!configured) {
    return (
      <>
        <Heading />
        <PanelRule />
        <div className="px-2 py-5">
          <Alert>
            This server can&apos;t store credentials yet. Set
            COMPOSIO_ENCRYPTION_KEY in the backend&apos;s .env.
          </Alert>
        </div>
      </>
    );
  }

  if (selected) {
    return (
      <Detail
        connector={selected}
        onBack={() => setOpen(null)}
        onChanged={() => {
          reload();
        }}
      />
    );
  }

  return (
    <>
      <Heading>
        {/* Which store actually answers this account's questions. Worth stating
            rather than leaving to be inferred from which row is green. */}
        <PanelChip
          className="shrink-0"
          title={
            backend
              ? "Your questions are searched against this"
              : "Connect a vector store and your questions get searched against it"
          }
        >
          {backend ?? "no store connected"}
        </PanelChip>
      </Heading>

      <PanelRule />

      <ScrollArea className="max-h-80">
        <div className="flex flex-col gap-2 pr-2">
          {GROUPS.map(({ kind, label, hint }) => {
            const rows = connectors.filter((c) => c.kind === kind);
            if (rows.length === 0) return null;

            return (
              <section key={kind} className="flex flex-col gap-0.5">
                <p className="px-2 pt-1 text-[0.66rem] font-medium tracking-wide text-ink-muted">
                  {label}
                  <span className="ml-1.5 font-normal opacity-70">{hint}</span>
                </p>

                {rows.map((connector) => (
                  <Row
                    key={connector.slug}
                    connector={connector}
                    onOpen={() => setOpen(connector.slug)}
                  />
                ))}
              </section>
            );
          })}
        </div>
      </ScrollArea>
    </>
  );
}

function Row({ connector, onOpen }: { connector: Connector; onOpen: () => void }) {
  const hints = describeHints(connector);
  const status = connector.stale
    ? "Stored credentials can't be read — reconnect"
    : connector.connected
      ? hints || "Connected"
      : connector.summary;

  return (
    <button
      type="button"
      onClick={onOpen}
      className="glass-row flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left focus-visible:outline-none"
    >
      <span className="glass-tile grid size-7 shrink-0 place-items-center rounded-lg text-ink-muted">
        {connector.kind === "vector" ? (
          <Database aria-hidden className="size-3.5" />
        ) : connector.kind === "dataset" ? (
          <Table2 aria-hidden className="size-3.5" />
        ) : (
          <Blocks aria-hidden className="size-3.5" />
        )}
      </span>

      {/*
        The status chip sits on the *name* line, not beside the whole block.
        Measured: with the chip in the row's third column the description had
        174px to live in and the longest one needed 407px, so more than half a
        sentence was being chopped mid-word. Above the description instead, it
        costs the name — which is short and never truncates — a slice it can
        afford, and hands the description the column's full width.
      */}
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[0.8rem] font-medium text-ink">
            {connector.name}
          </span>
          <span className="shrink-0">
            {connector.stale ? (
              <PanelChip className="text-destructive">Reconnect</PanelChip>
            ) : connector.connected ? (
              <PanelChip className="flex items-center gap-1">
                <Check aria-hidden className="size-2.5" />
                Connected
              </PanelChip>
            ) : (
              <PanelChip>Connect</PanelChip>
            )}
          </span>
        </span>

        {/* `title` so the whole string is still reachable on hover, whatever a
            future connector's summary turns out to be. */}
        <span className="truncate text-[0.7rem] text-ink-muted" title={status}>
          {status}
        </span>
      </span>
    </button>
  );
}

function Detail({
  connector,
  onBack,
  onChanged,
}: {
  connector: Connector;
  onBack: () => void;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      await disconnect(connector.slug);
      onChanged();
      onBack();
    } catch (raised) {
      setError(
        raised instanceof ConnectorError ? raised.message : "Could not disconnect that.",
      );
      setBusy(false);
    }
  };

  return (
    <>
      <PanelHeading title={connector.name} hint={connector.summary}>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          onClick={onBack}
          className="shrink-0 rounded-full text-ink-muted hover:text-ink"
        >
          <ArrowLeft aria-hidden />
          <span className="sr-only">Back to connectors</span>
        </Button>
      </PanelHeading>

      <PanelRule />

      {error && <Alert>{error}</Alert>}

      {connector.connected && !connector.stale ? (
        <div className="flex flex-col gap-2">
          {/*
            This one wraps rather than truncates, which is the opposite of the
            row above and for the opposite reason. In the list, the summary is
            a description and the first few words carry it. Here it is
            *identifying* information — which index, which keyspace, which key
            — and "chunks · default_keyspace · https://a1b2c3-ap-south-" tells
            you nothing about whether you attached the right thing. Measured
            against a four-field Astra connection it needed 526px of a 193px
            line, so nearly two thirds of the identity was being thrown away.

            `wrap-anywhere` rather than `break-words` because the longest part
            is an endpoint URL with no spaces in it, which `break-words` leaves
            overflowing rather than breaking.
          */}
          <div className="glass-tile flex items-start justify-between gap-2 rounded-lg px-2.5 py-2">
            <span className="min-w-0">
              <span className="block text-[0.7rem] text-ink-muted">Connected as</span>
              <span className="block wrap-anywhere text-[0.78rem] leading-snug text-ink">
                {describeHints(connector) || connector.name}
              </span>
            </span>
            <Button
              type="button"
              variant="ghost"
              size="xs"
              disabled={busy}
              onClick={remove}
              className="glass-row-danger shrink-0 rounded-full text-ink-muted"
            >
              {busy ? <Loader2 aria-hidden className="animate-spin" /> : "Disconnect"}
            </Button>
          </div>

          {/* Composio is the one connector with a second step: its credentials
              open a doorway to further consent screens. The vector stores are
              finished the moment they verify. */}
          {connector.slug === "composio" && <ComposioToolkits />}
          {connector.slug === "dataset" && <AttachedDatasets />}
        </div>
      ) : (
        <CredentialForm connector={connector} onDone={onChanged} />
      )}
    </>
  );
}

/**
 * The form, built from what the backend said this connector needs.
 *
 * Values live in one piece of state keyed by field name and are cleared the
 * moment a connect succeeds. Secrets render as password inputs, are not offered
 * to autofill, and are never what comes back — the response carries hints.
 */
function CredentialForm({
  connector,
  onDone,
}: {
  connector: Connector;
  onDone: () => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = connector.fields
    .filter((field) => field.required)
    .every((field) => (values[field.name] ?? "").trim().length > 0);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready || busy) return;

    setBusy(true);
    setError(null);
    try {
      await connect(connector.slug, values);
      setValues({});
      onDone();
    } catch (raised) {
      setError(
        raised instanceof ConnectorError ? raised.message : "Could not connect that.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-2 px-1 py-1">
      {connector.stale && (
        <p className="text-[0.72rem] leading-relaxed text-ink-muted">
          The stored credentials can no longer be read. Enter them again to
          reconnect.
        </p>
      )}

      {connector.fields.map((field) => (
        <label key={field.name} className="flex flex-col gap-1">
          <span className="text-[0.7rem] text-ink-muted">
            {field.label}
            {!field.required && <span className="ml-1 opacity-70">optional</span>}
          </span>
          <input
            type={field.secret ? "password" : "text"}
            value={values[field.name] ?? ""}
            onChange={(event) =>
              setValues((current) => ({ ...current, [field.name]: event.target.value }))
            }
            placeholder={field.placeholder}
            autoComplete="off"
            spellCheck={false}
            // `.glass-field` owns the ground, the border and both the hover
            // and focus states — the recessed half of the glass system.
            className="glass-field rounded-lg px-2.5 py-1.5 text-[0.75rem] text-ink placeholder:text-ink-muted"
          />
          {field.help && (
            <span className="text-[0.66rem] leading-snug text-ink-muted opacity-80">
              {field.help}
            </span>
          )}
        </label>
      ))}

      {error && <Alert>{error}</Alert>}

      <div className="flex items-center justify-between gap-2 pt-0.5">
        {connector.docsUrl ? (
          <a
            href={connector.docsUrl}
            target="_blank"
            rel="noreferrer noopener"
            className="glass-row inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.7rem] text-ink-muted hover:text-ink"
          >
            Get credentials
            <ExternalLink aria-hidden className="size-2.5" />
          </a>
        ) : (
          <span />
        )}
        <Button type="submit" size="xs" disabled={!ready || busy}>
          {busy ? <Loader2 aria-hidden className="animate-spin" /> : <Plug aria-hidden />}
          {busy ? "Checking…" : "Connect"}
        </Button>
      </div>
    </form>
  );
}
