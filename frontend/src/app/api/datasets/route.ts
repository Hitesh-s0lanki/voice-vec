import { forward } from "@/app/api/datasets/backend";

/** Everything this account has attached, with the per-account limit. */
export async function GET() {
  return forward("/datasets");
}

/**
 * Attach one by URL.
 *
 * Answers 202, not 201, and the status is passed through untouched: the row
 * exists but the dataset answers nothing yet, and `status: "pending"` is what
 * the panel polls on. The failures worth showing — a typo, a gated repo, a URL
 * that is not a dataset — come back as a 400 whose `detail` is written for
 * whoever typed the URL, so it is forwarded rather than replaced.
 */
export async function POST(request: Request) {
  const body = await request.text();
  return forward("/datasets", { method: "POST", body: body || "{}" });
}
