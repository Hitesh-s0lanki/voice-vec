import { forward } from "@/app/api/conversations/backend";

/**
 * Claim this browser's anonymous conversations for the account that just
 * signed in.
 *
 * The body carries only the browser's `sess_…`. Which account is claiming it
 * is not in the body and cannot be: it comes from the verified session that
 * `forward` turns into a bearer token, so the worst a forged call can do is
 * hand someone their *own* conversations again.
 */
export async function POST(request: Request) {
  const body = await request.text();
  return forward(request, "/conversations/adopt", { method: "POST", body: body || "{}" });
}
