/**
 * Shared plumbing for the integration route handlers.
 *
 * The same same-origin door as `../conversations/backend.ts`, with one rule
 * tightened: there is no anonymous path through here. A conversation belongs
 * to whoever holds the browser, so its routes forward a `sess_…` and let the
 * backend serve a signed-out visitor. A connected account is standing
 * permission to read somebody's mail, so this file refuses before it makes a
 * request — no token, no call.
 *
 * That check is a courtesy, not the enforcement. FastAPI's `require_user`
 * verifies the signature and 401s on its own; short-circuiting here only
 * spares a round trip and gives the panel a shape it can render. Never treat
 * the presence of a token here as proof of anything — this process does not
 * check the signature, and Clerk's middleware giving us one only means the
 * cookie parsed.
 */

import { auth } from "@clerk/nextjs/server";

/** The FastAPI backend in `src/` — `uv run python -m src.main`. */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/**
 * Longer than the conversation routes' 6 s, because these calls are not two
 * indexed reads — most of them are a round trip to Composio with Neon behind
 * it, and creating an auth config on a first-ever connect is slower again.
 * Still short enough that a wedged upstream fails the panel rather than
 * hanging it.
 */
const TIMEOUT_MS = 20_000;

/** 401 as the backend would phrase it, without asking the backend. */
function unauthorized(): Response {
  return Response.json(
    { detail: "Sign in to manage connected accounts." },
    { status: 401 },
  );
}

/**
 * Call the backend as the signed-in user, or 401 without calling it.
 *
 * The status is passed straight through. 401 means the token expired mid-panel
 * and the client should retry, 409 means the toolkit needs setting up in the
 * Composio dashboard, and 503 means this deployment has no Composio key —
 * three different things the panel says differently.
 */
export async function forward(
  path: string,
  init: { method?: string; body?: string } = {},
): Promise<Response> {
  const { getToken } = await auth();
  const token = await getToken();
  if (!token) return unauthorized();

  let response: Response;

  try {
    response = await fetch(`${BACKEND}${path}`, {
      method: init.method ?? "GET",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${token}`,
      },
      body: init.body,
      signal: AbortSignal.timeout(TIMEOUT_MS),
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { detail: "The integration service isn't running — start the FastAPI backend." },
      { status: 502 },
    );
  }

  if (response.status === 204) return new Response(null, { status: 204 });

  const raw = await response.text();
  return new Response(raw || null, {
    status: response.status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}
