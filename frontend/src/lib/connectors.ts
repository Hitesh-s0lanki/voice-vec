/**
 * The `/connectors` contract, mirrored from `src/schemas/connectors.py`.
 *
 * The point of `ConnectorField` is that the panel renders a form from data
 * rather than from a component per service. A connector added to the backend
 * registry shows up here with its own inputs, correctly labelled and correctly
 * masked, without this file or the panel changing.
 *
 * Credentials travel one way. `connect()` sends them; nothing returns one.
 * `hints` carries the non-secret fields as typed plus the last four characters
 * of the secret one, which is what lets a row read "vec-chunks · ····8fa2".
 */

export type ConnectorField = {
  name: string;
  label: string;
  /** Renders as a password and is never echoed back. */
  secret: boolean;
  required: boolean;
  placeholder: string;
  help: string;
};

/** "tools" is Composio; "vector" is where a user's embeddings live. */
export type ConnectorKind = "tools" | "vector";

export type Connector = {
  slug: string;
  name: string;
  kind: ConnectorKind;
  summary: string;
  docsUrl: string;
  fields: ConnectorField[];

  connected: boolean;
  hints: Record<string, string>;
  connectedAt: string | null;
  updatedAt: string | null;
  /** Stored but no longer decrypting — the master key was rotated. */
  stale: boolean;
};

export type ConnectorList = {
  connectors: Connector[];
  /** False when the server cannot hold a credential at all. */
  configured: boolean;
  /** Which store answers this user's retrieval; null = the deployment's own. */
  vectorBackend: string | null;
};

export class ConnectorError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ConnectorError";
    this.status = status;
  }

  get signedOut(): boolean {
    return this.status === 401;
  }
}

async function fail(response: Response): Promise<never> {
  let detail = "Something went wrong.";
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // no JSON body — keep the generic message
  }
  throw new ConnectorError(detail, response.status);
}

/**
 * Every connector and this account's state on each.
 *
 * Empty-but-unconfigured on failure rather than thrown: this feeds a panel
 * that opens on a click, and it should read as unavailable rather than take
 * the page down.
 */
export async function listConnectors(signal?: AbortSignal): Promise<ConnectorList> {
  try {
    const response = await fetch("/api/connectors", { signal, cache: "no-store" });
    if (!response.ok) return { connectors: [], configured: false, vectorBackend: null };

    return (await response.json()) as ConnectorList;
  } catch {
    return { connectors: [], configured: false, vectorBackend: null };
  }
}

/** Attach a service. Throws — somebody filled in a form and pressed a button. */
export async function connect(
  slug: string,
  values: Record<string, string>,
): Promise<Connector> {
  const response = await fetch(`/api/connectors/${encodeURIComponent(slug)}`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ values }),
  });

  if (!response.ok) return fail(response);
  return (await response.json()) as Connector;
}

export async function disconnect(slug: string): Promise<void> {
  const response = await fetch(`/api/connectors/${encodeURIComponent(slug)}`, {
    method: "DELETE",
  });
  if (!response.ok) await fail(response);
}

/** What a row shows under its name: the readable half of what was stored. */
export function describeHints(connector: Connector): string {
  const parts = Object.entries(connector.hints)
    .filter(([, value]) => value)
    .map(([key, value]) => (key.endsWith("_hint") ? `····${value}` : value));

  return parts.join(" · ");
}
