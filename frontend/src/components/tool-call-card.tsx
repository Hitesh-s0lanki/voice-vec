"use client";

import { useId, useState } from "react";
import { Check, ChevronDown, CircleSlash, TriangleAlert, Wrench } from "lucide-react";

import { PanelChip, PanelRule } from "@/components/panels/panel";
import type { ToolCall, VoiceStatus } from "@/hooks/use-voice-session";
import { cn } from "@/lib/utils";

/**
 * What the agent reached for, above the transcript.
 *
 * A turn that calls tools is a turn that goes quiet for several seconds, and
 * the orb can only say "thinking" through all of it. The activity log does
 * name the step, but it folds a whole round into one line — so "three ran, the
 * second one failed" arrives as a single row saying whichever tool reported
 * last. This is that round unfolded: one row per call, its own duration, and
 * the provider's error where there is one.
 *
 * It collapses because it is a *detail* of an answer that is spoken, not the
 * answer: open while it is happening, shut once the reply starts, without
 * either state moving the orb. The summary line survives collapsing, which is
 * the part worth keeping at a glance — how many ran, how long they took, and
 * whether any of them failed.
 */
export function ToolCallCard({
  calls,
  status,
}: {
  calls: ToolCall[];
  status: VoiceStatus;
}) {
  const [open, setOpen] = useState(true);
  const listId = useId();

  /*
   * Mid-take the last turn's tools are stale — the same reason the transcript
   * blanks. The card goes rather than emptying: a card headed "Tool calls"
   * with nothing under it reads as a tool that is about to run.
   */
  const stale =
    status === "listening" ||
    status === "connecting" ||
    status === "transcribing";
  if (stale || calls.length === 0) return null;

  const running = calls.filter((call) => call.state === "running").length;
  const failed = calls.filter((call) => call.state === "error").length;
  // Durations only — a call that never reported has no number to add.
  const spent = calls.reduce((total, call) => total + (call.ms ?? 0), 0);

  const summary = running
    ? `Running ${running} of ${calls.length}`
    : `${calls.length} ${calls.length === 1 ? "tool" : "tools"}${spent ? ` · ${elapsed(spent)}` : ""}`;

  return (
    <section
      aria-label="Tool calls"
      className="glass card-edge rise flex w-full flex-col gap-2 rounded-2xl p-2"
    >
      <button
        type="button"
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
        aria-controls={listId}
        // `.glass-row` reads `aria-expanded` for its held state, so the header
        // stays lit while the list under it is open.
        className="glass-row flex items-center gap-2 rounded-xl px-2 py-1.5 text-left"
      >
        {running ? (
          <span
            aria-hidden
            className="pulse-dot size-1.5 shrink-0 rounded-full bg-ink"
          />
        ) : (
          <Wrench aria-hidden className="size-3.5 shrink-0 text-ink-muted" />
        )}

        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-[0.82rem] font-medium tracking-[-0.01em] text-ink">
            Tool calls
          </span>
          <span className="truncate text-[0.72rem] leading-snug text-ink-muted">
            {summary}
          </span>
        </span>

        {failed > 0 && (
          <PanelChip title="The model was told, and can say so out loud">
            {failed} failed
          </PanelChip>
        )}

        <ChevronDown
          aria-hidden
          className={cn(
            "size-3.5 shrink-0 text-ink-muted transition-transform duration-200 motion-reduce:transition-none",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <>
          <PanelRule />
          <ul
            id={listId}
            className="flex max-h-[20dvh] min-h-0 flex-col gap-0.5 overflow-y-auto scrollbar-thin lg:max-h-[24dvh]"
          >
            {calls.map((call) => (
              <CallRow key={call.key} call={call} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

/** One call: how it went, what it was, and how long it took. */
function CallRow({ call }: { call: ToolCall }) {
  const running = call.state === "running";
  const Icon =
    call.state === "error"
      ? TriangleAlert
      : call.state === "skipped"
        ? CircleSlash
        : Check;

  return (
    <li
      data-active={running}
      className="glass-row flex items-start gap-2 rounded-lg px-2 py-1.5"
    >
      {running ? (
        <span
          aria-hidden
          className="pulse-dot mt-1.5 size-1.5 shrink-0 rounded-full bg-ink"
        />
      ) : (
        <Icon
          aria-hidden
          className={cn(
            "mt-0.5 size-3 shrink-0",
            call.state === "error" ? "text-ink-soft" : "text-ink-muted",
          )}
        />
      )}

      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span
          // The slug is the name the connector actually exposes — worth
          // keeping reachable, not worth reading at a glance.
          title={call.name}
          className={cn(
            "truncate text-[0.75rem] leading-relaxed",
            running ? "text-ink" : "text-ink-soft",
          )}
        >
          {readable(call.name)}
        </span>
        {call.detail && (
          <span className="text-[0.68rem] leading-snug text-ink-muted">
            {call.detail}
          </span>
        )}
      </span>

      {call.ms !== null && (
        <span
          title="How long this call took"
          className="mt-0.5 shrink-0 tabular-nums text-[0.66rem] text-ink-muted"
        >
          {elapsed(call.ms)}
        </span>
      )}
    </li>
  );
}

/**
 * `GMAIL_FETCH_EMAILS` → `Gmail fetch emails`.
 *
 * Enough to read, and nothing invented: the words are the slug's own, just
 * unshouted. The row keeps the slug in its tooltip for anyone matching this
 * against a connector's tool list.
 */
function readable(slug: string): string {
  const words = slug.replace(/[_-]+/g, " ").trim().toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Milliseconds at the precision worth reading at a glance. */
function elapsed(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
