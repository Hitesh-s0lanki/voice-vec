"use client";

import { useState } from "react";
import { BookOpen, CircleSlash, ShieldX, TriangleAlert } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { PanelChip } from "@/components/panels/panel";
import { EFFORT_LEVELS } from "@/lib/effort";
import type { AskResponse, Citation } from "@/lib/rag";
import { cn } from "@/lib/utils";

/** What the answer stage is doing, from the capture flow's point of view. */
export type AnswerState =
  | { phase: "pending" }
  | { phase: "done"; response: AskResponse }
  | { phase: "error"; message: string };

const HEADLINE: Record<AskResponse["status"], { label: string; icon: LucideIcon }> = {
  answered: { label: "Answer", icon: BookOpen },
  abstained: { label: "No grounded answer", icon: CircleSlash },
  refused: { label: "Declined", icon: ShieldX },
};

/**
 * The reply, under the transcript that produced it.
 *
 * Not mounted by the spoken loop, which prints no reply at all — it shows the
 * transcript in `components/transcript-card.tsx` and says the answer out
 * loud. This is for the read paths, where the citations are the point and are
 * worth reading — see docs/11-voice.md.
 *
 * An abstention is rendered as a real turn rather than an error state: the
 * pipeline ran, the corpus couldn't support an answer, and saying so is the
 * behaviour requirement 6 asks us to demonstrate. Only a transport failure
 * gets the error treatment.
 */
export function AnswerCard({ state }: { state: AnswerState }) {
  if (state.phase === "pending") return <Searching />;

  if (state.phase === "error") {
    return (
      <div
        role="alert"
        className="glass glass-danger rise flex w-full items-center gap-3 rounded-2xl px-5 py-4"
      >
        <TriangleAlert aria-hidden className="size-4 shrink-0 text-ink" />
        <p className="text-[0.82rem] leading-relaxed text-ink-soft">
          {state.message}
        </p>
      </div>
    );
  }

  const { response } = state;
  const { label, icon: Icon } = HEADLINE[response.status];
  const answered = response.status === "answered";

  return (
    <section
      aria-label="Answer"
      className="glass rise flex w-full flex-col gap-3 rounded-2xl px-5 py-4"
    >
      <header className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-[0.72rem] font-medium tracking-wide text-ink-soft">
          <Icon aria-hidden className="size-3.5 text-ink-muted" />
          {label}
        </span>

        <span className="flex items-center gap-1.5">
          {/* The rung asked for, and — only when they differ — the rung that
              actually answered. A `deep` request answered at tier 1 is a
              synthesis that fell back to the extractive path, and hiding that
              behind one number is how a degraded answer reads as a fast one. */}
          <PanelChip className="capitalize">{response.mode}</PanelChip>
          {response.cached ? (
            <PanelChip>cached</PanelChip>
          ) : (
            tierName(response.tier) !== response.mode && (
              <PanelChip className="capitalize">
                via {tierName(response.tier)}
              </PanelChip>
            )
          )}
          <PanelChip
            className={cn(
              "tabular-nums",
              response.withinBudget ? "text-ink-soft" : "text-ink",
            )}
            /* Requirement 3 is a per-request claim, so it is shown per request. */
          >
            {Math.round(response.timings.total)} ms
          </PanelChip>
        </span>
      </header>

      <p
        className={cn(
          "text-[0.98rem] leading-relaxed",
          answered ? "type-quote text-ink" : "text-ink-soft",
        )}
      >
        {answered ? response.answer : response.reason}
      </p>

      {answered && (
        <Meter confidence={response.confidence} method={response.method} />
      )}

      {response.citations.length > 0 && (
        <Sources citations={response.citations} answered={answered} />
      )}

      {response.flags.includes("injection") && (
        <p className="text-[0.7rem] leading-relaxed text-ink-muted">
          Heard an instruction aimed at the system — answered from the rest of
          the question.
        </p>
      )}
    </section>
  );
}

/** The rung's name, lowercased to match `response.mode`. */
function tierName(tier: number): string {
  return (EFFORT_LEVELS[tier]?.label ?? `level ${tier}`).toLowerCase();
}

function Searching() {
  return (
    <div className="glass fade flex w-full flex-col gap-2.5 rounded-2xl px-5 py-4">
      <span className="shimmer h-2.5 w-2/5 rounded-full" />
      <span className="shimmer h-2.5 w-full rounded-full" />
      <span className="shimmer h-2.5 w-3/4 rounded-full" />
    </div>
  );
}

/** Confidence as a bar, because a bare 0.62 means nothing to a reader. */
function Meter({
  confidence,
  method,
}: {
  confidence: number;
  method: string | null;
}) {
  return (
    <div className="flex items-center gap-2.5 text-[0.68rem] text-ink-muted">
      <span className="tabular-nums">{Math.round(confidence * 100)}%</span>
      <span
        aria-hidden
        className="glass-track h-1.5 flex-1 overflow-hidden rounded-full"
      >
        <span
          className="block h-full rounded-full bg-ink/35"
          style={{ width: `${Math.max(3, Math.min(100, confidence * 100))}%` }}
        />
      </span>
      {method && <span>{method === "lexical" ? "lexical span" : "embedded span"}</span>}
    </div>
  );
}

/**
 * The passages behind the answer — or, on an abstention, the ones that were
 * found and rejected. Showing the near misses is what makes an abstention
 * inspectable instead of a shrug.
 */
function Sources({
  citations,
  answered,
}: {
  citations: Citation[];
  answered: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        className="glass-row self-start rounded-md px-1.5 py-0.5 text-[0.7rem] font-medium tracking-wide text-ink-muted hover:text-ink-soft"
        aria-expanded={open}
      >
        {open ? "Hide" : "Show"} {answered ? "source" : "closest match"}
        {citations.length === 1 ? "" : "es"} ({citations.length})
      </button>

      {open && (
        <ul className="flex flex-col gap-2">
          {citations.map((citation) => (
            <li
              key={citation.docId}
              className="glass-tile flex flex-col gap-1 rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-1.5 text-[0.66rem] text-ink-muted">
                <PanelChip>{citation.strategy}</PanelChip>
                <span className="tabular-nums">{citation.score.toFixed(3)}</span>
                {citation.isGold && <PanelChip>labelled gold</PanelChip>}
              </div>
              <p className="line-clamp-3 text-[0.75rem] leading-relaxed text-ink-soft">
                {citation.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
