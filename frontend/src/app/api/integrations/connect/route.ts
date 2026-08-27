import { forward } from "@/app/api/integrations/backend";

/**
 * Start connecting a toolkit.
 *
 * Answers with a URL rather than a redirect. The caller is `fetch`, and a 30x
 * would be followed by the fetch instead of by the window — landing Composio's
 * consent page in a promise nobody can render. The panel does the navigating.
 *
 * The body is forwarded as sent and carries only a toolkit slug. Whose account
 * it connects is decided by the token, on the Python side, from a signature —
 * never by anything in here.
 */
export async function POST(request: Request) {
  const body = await request.text();
  return forward("/integrations/connect", { method: "POST", body: body || "{}" });
}
