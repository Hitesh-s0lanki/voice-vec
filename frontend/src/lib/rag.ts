/**
 * The `/ask` contract, mirrored from the FastAPI side (`src/schemas/ask.py`).
 * Kept in one place so the panels, the card and the proxy route agree on it.
 */

export type AskStatus = "answered" | "abstained" | "refused";

export type Citation = {
  docId: string;
  strategy: string;
  score: number;
  text: string;
  sourceQueryIds: number[];
  isGold: boolean;
};

/**
 * Per-stage milliseconds, in pipeline order. `null` means the stage did not
 * run — which is most of them on most requests, and is the point: the shape of
 * this block is a readout of which rung ran. A cache hit is `guardIn`,
 * `embed`, `cache` and nothing else.
 */
export type Timings = {
  guardIn: number | null;
  embed: number | null;
  cache: number | null;
  route: number | null;
  search: number | null;
  rerank: number | null;
  extract: number | null;
  generate: number | null;
  grade: number | null;
  rewrite: number | null;
  guardOut: number | null;
  total: number;
};

export type AskResponse = {
  status: AskStatus;
  answer: string | null;
  citations: Citation[];
  confidence: number;
  tier: number;
  reason: string | null;
  timings: Timings;
  requestId: string;
  language: string | null;
  flags: string[];
  /** `embedding` / `lexical` (extractive), `passage`, `synthesis`, or `cache`. */
  method: string | null;
  /** The rung that was *asked for*. `tier` is the one that answered. */
  mode: string;
  cached: boolean;
  /** Which vector store answered — the deployment's, or one the user connected. */
  backend: string | null;
  /** What the pipeline did beyond the straight line: `hybrid`, `rewrite`, … */
  escalations: string[];
  /**
   * The deadline this rung is measured against. Rungs 0–1 hold requirement 3's
   * 200 ms; the upper rungs make network calls and are reported against their
   * own budget rather than against one they were never going to meet.
   */
  budgetMs: number;
  withinBudget: boolean;
};

export type AskRequest = {
  transcript: string;
  languageCode: string | null;
  /** The EffortPanel index — a ceiling on escalation, not a floor. */
  effort: number;
  requestId: string;
};

/**
 * `abstained` is a success, not an error — the pipeline ran and the corpus
 * couldn't support an answer. Only a transport failure throws.
 */
export async function askVec(
  input: AskRequest,
  signal?: AbortSignal,
): Promise<AskResponse> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });

  const body = (await response.json()) as AskResponse & { error?: string };

  if (!response.ok) {
    throw new Error(body.error ?? "The answer service didn't respond.");
  }

  return body;
}

export type Suggestion = {
  query: string;
  queryType: string;
  confidence: number;
};

/**
 * Questions this index demonstrably answers, verified by
 * `scripts/suggestions.py` rather than assumed.
 *
 * The corpus is ~2,000 specific MS MARCO questions, so an unprompted question
 * is far more likely to miss than hit — and a correct abstention reads as a
 * broken system when you have no way to know what is in there. Never throws:
 * no suggestions just means the row stays hidden.
 */
export async function fetchSuggestions(
  limit = 4,
): Promise<{ suggestions: Suggestion[]; corpus: string | null }> {
  try {
    const response = await fetch(`/api/suggestions?limit=${limit}`);
    if (!response.ok) return { suggestions: [], corpus: null };
    return await response.json();
  } catch {
    return { suggestions: [], corpus: null };
  }
}
