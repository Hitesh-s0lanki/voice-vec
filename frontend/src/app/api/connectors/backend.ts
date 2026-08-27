/**
 * Shared plumbing for the connector route handlers.
 *
 * The same same-origin door as the conversation routes, with one rule
 * tightened: there is no anonymous path through here. A conversation belongs to
 * whoever holds the browser; a connector credential does not.
 *
 * That check is a courtesy, not the enforcement — FastAPI verifies the
 * signature and 401s on its own. Short-circuiting here only spares a round trip
 * and gives the panel a shape it can render. Never treat the presence of a
 * token here as proof of anything: this process does not check the signature.
 *
 * Credentials pass through this file on their way in. They are never logged,
 * never put in a URL, and nothing sends one back out.
 */

import { auth } from "@clerk/nextjs/server";

/** The FastAPI backend in `src/` — `uv run python -m src.main`. */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/**
 * Long, because connecting runs a live verification against somebody else's
 * service — a cold serverless index, a Postgres on the other side of the
 * world. Still short enough that a wedged upstream fails the form rather than
 * hanging it.
 */
const TIMEOUT_MS = 20_000;

export async function forward(
  path: string,
  init: { method?: string; body?: string } = {},
): Promise<Response> {
  const { getToken } = await auth();
  const token = await getToken();
  if (!token) {
    return Response.json({ detail: "Sign in to manage connectors." }, { status: 401 });
  }

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
      { detail: "The connector service isn't running — start the FastAPI backend." },
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
