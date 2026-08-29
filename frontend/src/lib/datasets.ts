/**
 * The `/datasets` contract, mirrored from `src/schemas/datasets.py`.
 *
 * Datasets are attached through the Connectors panel, but they are not *a*
 * connector the way Pinecone is — one connector row opens onto many of them,
 * the same shape Composio has with its toolkits. So the list lives here rather
 * than in `connectors.ts`: the connector row says datasets are attached, and
 * this says which.
 *
 * `status` is the field the panel is built around. Attaching returns
 * immediately with `pending` and the pull, the measurement and the model call
 * all happen on a worker, so the list is polled while anything is building.
 */

export type Dataset = {
  datasetId: string;
  url: string;
  kind: string;
  location: string;
  /** pending | ok | degraded | failed */
  status: string;
  /** What it is, in a few words — the first line of the agent's card. */
  title: string;
  card: string;
  rows: number;
  bytes: number;
  tables: number;
  /** Only ever set on `failed`, and always set when it is. */
  error: string;
  builtAt: string | null;
  createdAt: string | null;
};

export type DatasetList = {
  datasets: Dataset[];
  /** How many this account may attach in total. */
  limit: number;
  enabled: boolean;
};

export class DatasetError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "DatasetError";
    this.status = status;
  }
}

async function fail(response: Response): Promise<never> {
  let detail = "Something went wrong.";
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // no JSON body — keep the generic message
  }
  throw new DatasetError(detail, response.status);
}

/**
 * Everything attached, newest state first.
 *
 * Empty-and-disabled on failure rather than thrown, the same as
 * `listConnectors`: this feeds a panel that opens on a click and should read
 * as unavailable rather than take the panel down.
 */
export async function listDatasets(signal?: AbortSignal): Promise<DatasetList> {
  try {
    const response = await fetch("/api/datasets", { signal, cache: "no-store" });
    if (!response.ok) return { datasets: [], limit: 0, enabled: false };

    return (await response.json()) as DatasetList;
  } catch {
    return { datasets: [], limit: 0, enabled: false };
  }
}

/**
 * Attach one. Throws — somebody pasted a URL and pressed a button, and the
 * errors worth showing them are all raised here: a typo, a private repo, a URL
 * that is not a dataset.
 *
 * Returns with `status: "pending"`. The build runs behind it.
 */
export async function addDataset(url: string): Promise<Dataset> {
  const response = await fetch("/api/datasets", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) return fail(response);
  return (await response.json()) as Dataset;
}

/** Detach one, and unlink the file behind it. Idempotent. */
export async function removeDataset(datasetId: string): Promise<void> {
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}`, {
    method: "DELETE",
  });
  if (!response.ok) await fail(response);
}

/** Re-pull and re-measure — how a dataset picks up a changed column budget. */
export async function rebuildDataset(datasetId: string): Promise<void> {
  const response = await fetch(
    `/api/datasets/${encodeURIComponent(datasetId)}/rebuild`,
    { method: "POST" },
  );
  if (!response.ok) await fail(response);
}

/** "75,000 rows · 3 tables · 40 MB", skipping whatever is not known yet. */
export function describeDataset(dataset: Dataset): string {
  const parts: string[] = [];
  if (dataset.rows) parts.push(`${dataset.rows.toLocaleString()} rows`);
  if (dataset.tables) parts.push(`${dataset.tables} table${dataset.tables === 1 ? "" : "s"}`);
  if (dataset.bytes) parts.push(`${Math.max(1, Math.round(dataset.bytes / 1_000_000))} MB`);
  return parts.join(" · ");
}
