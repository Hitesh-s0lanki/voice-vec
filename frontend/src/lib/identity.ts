"use client";

/**
 * Who this browser is, before anybody signs in.
 *
 * Every conversation belongs to someone. Until there are accounts, that
 * someone is the browser itself: an id minted once, kept in `localStorage`,
 * and sent on every request — as `x-session-id` over HTTP and as `?session=`
 * on the voice socket. The backend stores it in `conversations.session_id`,
 * beside the `user_id` column an account will fill in later, and matches a row
 * on either. Signing in can therefore adopt what was said before it without
 * moving a message.
 *
 * It is not a credential and is not treated as one. Anyone holding the id can
 * read the conversations opened under it, which is the same guarantee as
 * "anyone holding this browser" — no more, and worth remembering before
 * anything sensitive is spoken into it.
 */

const STORAGE_KEY = "vec-session";

/** Matches the backend's `conv_…`: a prefix, then a dash-free UUID. */
const SHAPE = /^sess_[0-9a-f]{32}$/;

let cached: string | null = null;

function mint(): string {
  // `randomUUID` needs a secure context — and so does `getUserMedia`, so a
  // build without it has no microphone either. `Math.random` would be a
  // collision risk across every browser this ever runs in; better to go
  // without an id than to hand two people the same one.
  if (typeof crypto?.randomUUID !== "function") return "";
  return `sess_${crypto.randomUUID().replace(/-/g, "")}`;
}

/**
 * The id for this browser, created on first use.
 *
 * Returns `""` when there is nowhere to keep it (server render, private mode
 * with storage blocked, an insecure origin). That is a real state and not an
 * error: the voice loop runs exactly as before, and nothing is written down.
 */
export function browserSessionId(): string {
  if (cached !== null) return cached;
  if (typeof window === "undefined") return "";

  let stored: string | null = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    // storage blocked — fall through to a per-tab id
  }

  if (stored && SHAPE.test(stored)) {
    cached = stored;
    return cached;
  }

  const fresh = mint();
  try {
    if (fresh) localStorage.setItem(STORAGE_KEY, fresh);
  } catch {
    // blocked again: the id lives for this page's lifetime and no longer,
    // so the conversation is still saved — it just cannot be found tomorrow.
  }

  cached = fresh;
  return cached;
}
