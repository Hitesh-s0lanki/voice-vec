import { NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8001";

/** Cheap read off a cached file — a short timeout is plenty. */
const TIMEOUT_MS = 3_000;

export async function GET(request: Request) {
  const limit = new URL(request.url).searchParams.get("limit") ?? "4";

  try {
    const response = await fetch(`${BACKEND}/suggestions?limit=${limit}`, {
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (!response.ok) {
      // Nothing to suggest is not an error — the UI just hides the row.
      return NextResponse.json({ suggestions: [], corpus: null });
    }

    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({ suggestions: [], corpus: null });
  }
}
