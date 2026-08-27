import { forward } from "@/app/api/connectors/backend";

/**
 * Connect a service, or disconnect it.
 *
 * The PUT body carries credentials — the one thing that crosses this boundary
 * and matters. It stays in the body the whole way: never a query parameter,
 * never a path segment, because those are what end up in access logs, proxy
 * logs and browser history. Nothing here logs, and the response carries hints
 * rather than secrets.
 *
 * `slug` is encoded rather than trusted; the Python side resolves it against
 * the registry, so an unknown one is a 404 and never a path.
 */
export async function PUT(request: Request, ctx: RouteContext<"/api/connectors/[slug]">) {
  const { slug } = await ctx.params;
  const body = await request.text();

  return forward(`/connectors/${encodeURIComponent(slug)}`, {
    method: "PUT",
    body: body || "{}",
  });
}

export async function DELETE(
  _request: Request,
  ctx: RouteContext<"/api/connectors/[slug]">,
) {
  const { slug } = await ctx.params;
  return forward(`/connectors/${encodeURIComponent(slug)}`, { method: "DELETE" });
}
