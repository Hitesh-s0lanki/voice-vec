import { forward } from "@/app/api/datasets/backend";

/**
 * Detach one, and unlink the file behind it. Idempotent.
 *
 * `id` is encoded rather than trusted; the Python side looks it up against
 * this account's rows, so somebody else's dataset id is a 404 and never a path.
 */
export async function DELETE(
  _request: Request,
  ctx: RouteContext<"/api/datasets/[id]">,
) {
  const { id } = await ctx.params;
  return forward(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
}
