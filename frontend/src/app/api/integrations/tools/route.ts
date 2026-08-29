import { forward } from "@/app/api/integrations/backend";

/**
 * What this account's linked services let the agent do.
 *
 * A static segment like `/toolkits` next door, and for the same reason: it has
 * to resolve ahead of `[toolkit]`, or a service that happened to be called
 * "tools" would shadow it.
 */
export async function GET() {
  return forward("/integrations/tools");
}
