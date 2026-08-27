import { forward } from "@/app/api/integrations/backend";

/**
 * The catalogue of connectable services.
 *
 * A static segment, so it resolves ahead of `[toolkit]` next door and a
 * toolkit that happened to be called "toolkits" could not shadow it.
 *
 * Only three parameters are carried over, each re-encoded rather than passed
 * through: the query string arrives from a search box, and this is the last
 * place before it becomes a backend URL.
 */
export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;

  const query = new URLSearchParams();
  const search = params.get("search");
  const cursor = params.get("cursor");
  const limit = params.get("limit");

  if (search) query.set("search", search);
  if (cursor) query.set("cursor", cursor);
  if (limit) query.set("limit", limit);

  const suffix = query.size > 0 ? `?${query}` : "";
  return forward(`/integrations/toolkits${suffix}`);
}
