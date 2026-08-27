import { forward } from "@/app/api/conversations/backend";

/**
 * One thread, with its messages.
 *
 * `id` is encoded rather than trusted: it arrives from the URL bar, and the
 * only thing standing between it and a backend path is this call.
 */
export async function GET(request: Request, ctx: RouteContext<"/api/conversations/[id]">) {
  const { id } = await ctx.params;
  return forward(request, `/conversations/${encodeURIComponent(id)}`);
}

export async function PATCH(request: Request, ctx: RouteContext<"/api/conversations/[id]">) {
  const { id } = await ctx.params;
  const body = await request.text();

  return forward(request, `/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body,
  });
}

export async function DELETE(request: Request, ctx: RouteContext<"/api/conversations/[id]">) {
  const { id } = await ctx.params;
  return forward(request, `/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
}
