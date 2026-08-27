import { forward } from "@/app/api/connectors/backend";

/** Every connector, with this account's state on each. Signed-in only. */
export async function GET() {
  return forward("/connectors");
}
