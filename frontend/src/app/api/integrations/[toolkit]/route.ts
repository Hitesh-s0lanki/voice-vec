import { forward } from "@/app/api/integrations/backend";

/**
 * One connection: its current status, or its removal.
 *
 * `toolkit` is encoded rather than trusted — it arrives from the URL bar on
 * the way back from consent, and this is the last thing between it and a
 * backend path. The Python side normalises it again before it reaches a
 * primary key; neither check is redundant, because neither one runs in the
 * other's process.
 */
export async function GET(
  _request: Request,
  ctx: RouteContext<"/api/integrations/[toolkit]">,
) {
  const { toolkit } = await ctx.params;
  return forward(`/integrations/${encodeURIComponent(toolkit)}`);
}

export async function DELETE(
  _request: Request,
  ctx: RouteContext<"/api/integrations/[toolkit]">,
) {
  const { toolkit } = await ctx.params;
  return forward(`/integrations/${encodeURIComponent(toolkit)}`, { method: "DELETE" });
}
