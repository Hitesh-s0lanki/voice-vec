import { forward } from "@/app/api/conversations/backend";

/** Your conversations, newest first. Empty rather than broken when it fails. */
export async function GET(request: Request) {
  const limit = new URL(request.url).searchParams.get("limit") ?? "30";
  return forward(request, `/conversations?limit=${encodeURIComponent(limit)}`);
}

/**
 * Open one up front.
 *
 * The voice socket opens its own on the first take, so nothing in this app
 * calls this today — it is here for a client that wants a URL before it has
 * anything to say.
 */
export async function POST(request: Request) {
  const body = await request.text();
  return forward(request, "/conversations", { method: "POST", body: body || "{}" });
}
