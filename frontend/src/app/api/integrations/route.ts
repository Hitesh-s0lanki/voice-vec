import { forward } from "@/app/api/integrations/backend";

/** Your connected accounts. Requires a signed-in caller — see `backend.ts`. */
export async function GET() {
  return forward("/integrations");
}
