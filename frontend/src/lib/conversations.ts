/**
 * The `/conversations` contract, mirrored from `src/schemas/chat.py`, and the
 * fetches that use it.
 *
 * Everything goes through the Next route handlers under `/api/conversations`
 * rather than straight at FastAPI: they are same-origin (no CORS preflight in
 * front of a panel that opens on a click) and they are where the backend's
 * address stays a server-side detail.
 *
 * Nothing here throws on a dead backend. A rail panel that cannot list your
 * conversations should show that it is empty, not take the page down with it.
 */

import { browserSessionId } from "@/lib/identity";

export type ConversationSummary = {
  id: string;
  title: string | null;
  language: string | null;
  /** Questions asked, not messages stored. */
  turns: number;
  createdAt: string;
  updatedAt: string;
};

/**
 * How a stored turn ended. The first three mirror `/ask`; the last two are
 * outcomes only a spoken turn has — talked over, and failed mid-reply.
 */
export type MessageStatus =
  | "answered"
  | "abstained"
  | "refused"
  | "interrupted"
  | "error";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  languageCode: string | null;
  /** The voice turn this belongs to — what pairs a question with its answer. */
  turnId: string | null;
  status: MessageStatus | null;
  reason: string | null;
  latencyMs: number | null;
  createdAt: string;
};

/**
 * One tool the agent ran, as the thread shows it.
 *
 * There is no `result` field and that is deliberate, not an omission: the
 * backend stores a result's *size* and never its content, so an audit trail
 * cannot become a copy of everything the agent has read. What is here is what
 * the agent decided — the tool and its arguments — plus how it went.
 */
export type ToolCall = {
  id: string;
  /** Pairs a call to the exchange that caused it, like a message's turnId. */
  turnId: string | null;
  toolkit: string | null;
  /** Composio's own, e.g. GMAIL_SEND_EMAIL. */
  slug: string;
  arguments: Record<string, unknown>;
  status: string;
  ok: boolean;
  error: string | null;
  resultBytes: number | null;
  latencyMs: number | null;
  createdAt: string;
};

export type ConversationDetail = {
  conversation: ConversationSummary;
  messages: ChatMessage[];
  toolCalls: ToolCall[];
};

/** `conv_` and 32 hex characters — the same check the backend runs. */
export function isConversationId(value: string | null | undefined): value is string {
  return typeof value === "string" && /^conv_[0-9a-f]{32}$/.test(value);
}

/** The conversation a path is showing, if it is showing one. */
export function conversationIdFrom(pathname: string | null): string | null {
  const id = pathname?.startsWith("/c/") ? pathname.slice(3).split("/")[0] : null;
  return isConversationId(id) ? id : null;
}

function headers(): HeadersInit {
  return {
    "content-type": "application/json",
    "x-session-id": browserSessionId(),
  };
}

export async function listConversations(
  signal?: AbortSignal,
): Promise<ConversationSummary[]> {
  try {
    const response = await fetch("/api/conversations", {
      headers: headers(),
      signal,
      cache: "no-store",
    });
    if (!response.ok) return [];

    const body = (await response.json()) as { conversations?: ConversationSummary[] };
    return body.conversations ?? [];
  } catch {
    return [];
  }
}

/** `null` means gone, not yours, or unreachable — all three read the same. */
export async function readConversation(
  id: string,
  signal?: AbortSignal,
): Promise<ConversationDetail | null> {
  if (!isConversationId(id)) return null;

  try {
    const response = await fetch(`/api/conversations/${id}`, {
      headers: headers(),
      signal,
      cache: "no-store",
    });
    if (!response.ok) return null;

    return (await response.json()) as ConversationDetail;
  } catch {
    return null;
  }
}

/**
 * Hand everything said before signing in to the account that just did.
 *
 * Called once per account per browser. The account is not a parameter — the
 * route handler takes it from the verified Clerk session — so all this has to
 * say is which browser is asking.
 */
export async function adoptConversations(): Promise<number> {
  const session = browserSessionId();
  if (!session) return 0;

  try {
    const response = await fetch("/api/conversations/adopt", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ sessionId: session }),
    });
    if (!response.ok) return 0;

    const body = (await response.json()) as { moved?: number };
    return body.moved ?? 0;
  } catch {
    return 0;
  }
}

export async function deleteConversation(id: string): Promise<boolean> {
  if (!isConversationId(id)) return false;

  try {
    const response = await fetch(`/api/conversations/${id}`, {
      method: "DELETE",
      headers: headers(),
    });
    return response.ok;
  } catch {
    return false;
  }
}
