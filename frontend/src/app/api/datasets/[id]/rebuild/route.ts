import { forward } from "@/app/api/datasets/backend";

/**
 * Pull and measure it again — how an already-attached dataset picks up a
 * changed column budget without being removed and re-added.
 *
 * 202 like the attach, and for the same reason: the row goes back to `pending`
 * and the work runs on a worker behind it.
 */
export async function POST(
  _request: Request,
  ctx: RouteContext<"/api/datasets/[id]/rebuild">,
) {
  const { id } = await ctx.params;
  return forward(`/datasets/${encodeURIComponent(id)}/rebuild`, {
    method: "POST",
    body: "{}",
  });
}
