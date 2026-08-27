/**
 * Shared plumbing for the conversation route handlers.
 *
 * Two things reach FastAPI, and only one of them comes from the browser:
 *
 *   Authorization  a Clerk session token, minted here from the *verified*
 *                  session — never a user id the client asked us to pass on
 *   x-session-id   the browser's own `sess_…`, forwarded as sent
 *
 * That split is the point of this file. A `x-user-id` header would be a value
 * anyone could type into curl, and the backend would have no way to tell the
 * difference; a token carries a signature Clerk has to have produced. The
 * browser id is not verified and does not need to be — it grants exactly what
 * holding that browser already grants.
 *
 * Everything else about the caller is dropped. These routes are a same-origin
 * door in front of FastAPI, not a general proxy.
 */

import { auth } from "@clerk/nextjs/server";

/** The FastAPI backend in `src/` — `uv run python -m src.main`. */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/**
 * Two indexed reads against Neon, from a panel that opens on a click. Long
 * enough to cover a pool that has to build its first connection (~1.5 s to
 * ap-southeast-1), short enough that a dead backend empties the panel rather
 * than hanging it.
 */
const TIMEOUT_MS = 6_000;

/** Who is asking, as the backend's `identify` dependency expects it. */
async function identity(request: Request): Promise<HeadersInit> {
  const headers: Record<string, string> = { "content-type": "application/json" };

  const browser = request.headers.get("x-session-id");
  if (browser) headers["x-session-id"] = browser;

  // Signed out, this is simply null and the caller stays anonymous — which is
  // a first-class state here, not a failure. `auth()` works because the Clerk
  // middleware matcher in proxy.ts covers /api.
  const { getToken } = await auth();
  const token = await getToken();
  if (token) headers.authorization = `Bearer ${token}`;

  return headers;
}

/**
 * Call the backend and hand its answer straight back.
 *
 * The status code is passed through rather than flattened: 404 is how the
 * client learns a conversation is gone or was never theirs, and 204 is how it
 * learns a delete worked. Both carry no body, which `Response` is happy with
 * and `NextResponse.json` is not.
 */
export async function forward(
  request: Request,
  path: string,
  init: { method?: string; body?: string } = {},
): Promise<Response> {
  let response: Response;

  try {
    response = await fetch(`${BACKEND}${path}`, {
      method: init.method ?? "GET",
      headers: await identity(request),
      body: init.body,
      signal: AbortSignal.timeout(TIMEOUT_MS),
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { error: "The conversation service isn't running — start the FastAPI backend." },
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
