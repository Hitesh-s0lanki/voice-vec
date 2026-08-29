/**
 * Shared plumbing for the dataset route handlers.
 *
 * The same same-origin door as `../connectors/backend.ts`, and refusing on the
 * same rule: no token, no call. A dataset is not a credential, but attaching
 * one spends this deployment's disk and its model calls against a per-account
 * limit, so there is no anonymous path through here either.
 *
 * That check is a courtesy, not the enforcement — `require_user` in
 * `datasets_controller.py` verifies the signature and 401s on its own.
 * Short-circuiting here only spares a round trip and gives the panel a shape
 * it can render. Never treat the presence of a token here as proof of
 * anything: this process does not check the signature.
 */

import { auth } from "@clerk/nextjs/server";

/** The FastAPI backend in `src/` — `uv run python -m src.main`. */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/**
 * Long, for the same reason as the connector routes: `POST /datasets` resolves
 * the URL against Hugging Face before it answers, so a cold or slow upstream
 * is inside this window. Only the *resolve* is — the pull, the measurement and
 * the model call all run on a worker and the row comes back `pending`, so this
 * is never the length of a build.
 */
const TIMEOUT_MS = 20_000;

export async function forward(
  path: string,
  init: { method?: string; body?: string } = {},
): Promise<Response> {
  const { getToken } = await auth();
  const token = await getToken();
  if (!token) {
    return Response.json({ detail: "Sign in to manage datasets." }, { status: 401 });
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
      { detail: "The dataset service isn't running — start the FastAPI backend." },
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
