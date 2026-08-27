/**
 * The `/integrations` contract, mirrored from `src/schemas/integrations.py`,
 * and the fetches that use it.
 *
 * Everything goes through the Next route handlers under `/api/integrations`
 * for the reasons `conversations.ts` gives — same-origin, and the backend's
 * address stays server-side — plus one this file has and that one does not:
 * the Clerk token is minted in the route handler from the verified session, so
 * it never has to exist in browser JavaScript.
 *
 * Unlike the conversation fetches, these do not all swallow their failures.
 * Listing does: a panel that cannot reach the backend should read as empty
 * rather than take the page down. Connecting and disconnecting do not: those
 * are things the user pressed a button for, and silently doing nothing is the
 * worst available answer. They throw `IntegrationError`, which carries the
 * status the panel needs to tell "sign in again" from "set this up in the
 * Composio dashboard".
 */

/**
 * The connector: whether this user has connected their own Composio account.
 *
 * `keyHint` is the last four characters of their API key and is the only thing
 * about it that ever leaves the server. There is no field here that could
 * carry the key itself, which is the point.
 */
export type ComposioAccount = {
  connected: boolean;
  keyHint: string | null;
  connectedAt: string | null;
  updatedAt: string | null;
  /** False when the *server* cannot store keys — not the user's to fix. */
  configured: boolean;
  /** A key is stored but no longer decrypts. Reconnecting is the fix. */
  stale: boolean;
};

export type Toolkit = {
  /** Composio's id for the toolkit, e.g. `gmail`. */
  slug: string;
  name: string;
  description: string | null;
  logo: string | null;
  categories: string[];
  tools: number;
  /** False when it needs an auth config created in the Composio dashboard. */
  connectable: boolean;
  noAuth: boolean;
};

export type ToolkitList = {
  toolkits: Toolkit[];
  nextCursor: string | null;
};

export type Connection = {
  toolkit: string;
  name: string | null;
  logo: string | null;
  /** Composio's own vocabulary: ACTIVE, INITIALIZING, FAILED, REVOKED, … */
  status: string;
  active: boolean;
  pending: boolean;
  connectedAt: string | null;
  updatedAt: string | null;
};

export type ConnectionList = {
  connections: Connection[];
  /** False when the server has no COMPOSIO_ENCRYPTION_KEY — a different empty. */
  configured: boolean;
  /** The connector itself, so the panel renders its whole state from one call. */
  composio: ComposioAccount;
};

export type ConnectStarted = {
  toolkit: string;
  redirectUrl: string;
  status: string;
};

export class IntegrationError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "IntegrationError";
    this.status = status;
  }

  /** The token expired, or nobody is signed in. Worth saying differently. */
  get signedOut(): boolean {
    return this.status === 401;
  }

  /** Composio itself is not connected yet — the panel shows the form, not an error. */
  get needsComposio(): boolean {
    return this.status === 409;
  }
}

/** FastAPI puts the message in `detail`; a dead proxy puts it there too. */
async function fail(response: Response): Promise<never> {
  let detail = "Something went wrong.";
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // no JSON body — keep the generic message
  }
  throw new IntegrationError(detail, response.status);
}

/**
 * What could be connected. Empty rather than thrown on failure — this feeds a
 * browse list, and an error banner over a catalogue nobody asked to see yet is
 * noise.
 */
export async function listToolkits(
  options: { search?: string; cursor?: string; limit?: number; signal?: AbortSignal } = {},
): Promise<ToolkitList> {
  const query = new URLSearchParams();
  if (options.search) query.set("search", options.search);
  if (options.cursor) query.set("cursor", options.cursor);
  if (options.limit) query.set("limit", String(options.limit));

  try {
    const response = await fetch(`/api/integrations/toolkits?${query}`, {
      signal: options.signal,
      cache: "no-store",
    });
    if (!response.ok) return { toolkits: [], nextCursor: null };

    return (await response.json()) as ToolkitList;
  } catch {
    return { toolkits: [], nextCursor: null };
  }
}

/**
 * This account's connections.
 *
 * `configured: true` with an empty list is "nothing connected yet";
 * `configured: false` is "this deployment has no Composio key". The panel
 * says different things, so the failure path reports the second rather than
 * inventing the first.
 */
export async function listConnections(signal?: AbortSignal): Promise<ConnectionList> {
  try {
    const response = await fetch("/api/integrations", { signal, cache: "no-store" });
    if (!response.ok) return unreachable();

    return (await response.json()) as ConnectionList;
  } catch {
    return unreachable();
  }
}

/**
 * What the panel shows when it cannot reach the backend at all.
 *
 * `configured: false` rather than "nothing connected": the two render
 * differently, and inventing an empty list in front of a dead server would
 * invite somebody to reconnect a Composio account that is already fine.
 */
function unreachable(): ConnectionList {
  return {
    connections: [],
    configured: false,
    composio: {
      connected: false,
      keyHint: null,
      connectedAt: null,
      updatedAt: null,
      configured: false,
      stale: false,
    },
  };
}

/** Store this user's Composio key. Throws — they pressed a button. */
export async function connectComposio(apiKey: string): Promise<ComposioAccount> {
  const response = await fetch("/api/integrations/composio", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ apiKey }),
  });

  if (!response.ok) return fail(response);
  return (await response.json()) as ComposioAccount;
}

/** Forget the key, and the local rows that only made sense with it. */
export async function disconnectComposio(): Promise<void> {
  const response = await fetch("/api/integrations/composio", { method: "DELETE" });
  if (!response.ok) await fail(response);
}

/** Start consent. Throws — the user pressed a button and deserves an answer. */
export async function connectToolkit(toolkit: string): Promise<ConnectStarted> {
  const response = await fetch("/api/integrations/connect", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ toolkit }),
  });

  if (!response.ok) return fail(response);
  return (await response.json()) as ConnectStarted;
}

/** One connection, refreshed against Composio. `null` when there isn't one. */
export async function connectionStatus(
  toolkit: string,
  signal?: AbortSignal,
): Promise<Connection | null> {
  try {
    const response = await fetch(`/api/integrations/${encodeURIComponent(toolkit)}`, {
      signal,
      cache: "no-store",
    });
    if (!response.ok) return null;

    return (await response.json()) as Connection;
  } catch {
    return null;
  }
}

export async function disconnectToolkit(toolkit: string): Promise<void> {
  const response = await fetch(`/api/integrations/${encodeURIComponent(toolkit)}`, {
    method: "DELETE",
  });
  if (!response.ok) await fail(response);
}
