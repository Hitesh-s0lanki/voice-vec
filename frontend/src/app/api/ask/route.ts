import { NextResponse } from "next/server";

/** The FastAPI backend in `src/` — `uv run python -m src.main`. */
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/**
 * The pipeline answers inside 200 ms (measured P100: 159 ms), so this is not a
 * latency budget — it is the point at which the backend is dead rather than
 * slow. Generous enough to cover a cold start still warming the ONNX session.
 */
const TIMEOUT_MS = 5_000;

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Malformed request." }, { status: 400 });
  }

  let response: Response;
  try {
    response = await fetch(`${BACKEND}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch {
    return NextResponse.json(
      { error: "The answer service isn't running — start the FastAPI backend." },
      { status: 502 },
    );
  }

  const raw = await response.text();
  let parsed: unknown = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;
  } catch {
    // Backend returned something that isn't JSON — surface it as upstream.
  }

  if (!response.ok) {
    const detail =
      (parsed as { detail?: string; error?: string } | null)?.detail ??
      (parsed as { error?: string } | null)?.error ??
      raw.slice(0, 200);

    return NextResponse.json(
      { error: detail || `The answer service responded with ${response.status}.` },
      { status: 502 },
    );
  }

  return NextResponse.json(parsed);
}
