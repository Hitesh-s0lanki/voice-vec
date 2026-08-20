"use client";

import {
  AudioLines,
  Check,
  CircleSlash,
  Database,
  Radio,
  Sparkles,
  TriangleAlert,
  Volume2,
  Wrench,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import type { ActivityLine, Exchange, VoiceStatus } from "@/hooks/use-voice-session";
import { languageName } from "@/lib/languages";
import { cn } from "@/lib/utils";

/** Steps the card holds. Older ones are still in the hook's scrollback. */
const STEPS = 5;

/** Exchanges kept underneath. The Conversations panel keeps the rest. */
const EXCHANGES = 2;

/** Before the socket has said anything, the client's own state has to. */
const WAITING: Partial<Record<VoiceStatus, string>> = {
  connecting: "Reaching the voice service…",
  listening: "Listening…",
  transcribing: "Uploading the take…",
};

/**
 * One icon per step of the pipeline. Unknown steps get the plain dot rather
 * than nothing — the backend is allowed to add a step without this file
 * knowing about it, which is the point of `label` being written server-side.
 */
const ICONS: Record<string, LucideIcon> = {
  stt: AudioLines,
  retrieval: Database,
  tool: Wrench,
  llm: Sparkles,
  speech: Volume2,
  turn: Check,
};

/**
 * Top-right corner: what the backend is doing, then what was said.
 *
 * The orb has four states and cannot say more than "thinking" — which covers
 * retrieval, a tool call and the model itself, three things with very
 * different reasons for being slow. This card is where that distinction
 * lives: every step the server announces, in the order it started, newest
 * first, with the elapsed time it reported. Underneath it, the exchange those
 * steps produced, so the log and the conversation read as one thing.
 */
export function ActivityFeed({
  status,
  activity,
  exchanges,
}: {
  status: VoiceStatus;
  activity: ActivityLine[];
  exchanges: Exchange[];
}) {
  const live = status !== "idle" && status !== "error";

  // Steps arrive in pipeline order and each one keeps the slot it started in,
  // so reversing puts the newest at the top without any row ever moving.
  const steps = activity.slice(-STEPS).reverse();
  const recent = exchanges.slice(-EXCHANGES).reverse();
  const waiting = live ? WAITING[status] : undefined;

  return (
    <aside
      aria-label="Activity"
      /*
        Hidden below `md`: the step log plus two exchanges is a tall card, and
        on a phone it grows straight down over the orb. `max-h` keeps it off
        the transcript card in the opposite corner on a short laptop.
      */
      className="glass fixed top-5 right-5 z-40 hidden max-h-[calc(100dvh-2.5rem)] w-80 flex-col gap-2 overflow-hidden rounded-2xl p-2 md:flex"
    >
      <PanelHeading
        title="Activity"
        hint={
          live
            ? "What the backend is doing"
            : exchanges.length
              ? `${exchanges.length} ${exchanges.length === 1 ? "turn" : "turns"} this session`
              : "Nothing captured yet"
        }
      />

      <PanelRule />

      {steps.length === 0 && !waiting && recent.length === 0 ? (
        <p className="px-2 pt-0.5 pb-1.5 text-[0.72rem] leading-relaxed text-ink-muted">
          Tap the orb and every step the backend takes shows up here as it
          happens.
        </p>
      ) : (
        <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
          {(steps.length > 0 || waiting) && (
            <ul aria-live="polite" className="flex flex-col gap-0.5">
              {/*
                The client's own state, shown only until the server's first
                step for this turn lands. Recording and uploading happen
                entirely in the browser, so nothing on the socket describes
                them — and a log that goes blank while the microphone is open
                reads as broken.
              */}
              {waiting && steps[0]?.state !== "running" && (
                <li className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[0.75rem] text-ink-soft">
                  <Radio aria-hidden className="size-3 shrink-0 text-ink-muted" />
                  {waiting}
                </li>
              )}

              {steps.map((step) => (
                <StepRow key={step.key} step={step} />
              ))}
            </ul>
          )}

          {recent.length > 0 && (
            <>
              <PanelRule className="mx-2" />
              <ul className="flex flex-col gap-1.5">
                {recent.map((exchange) => (
                  <ExchangeRow key={exchange.id} exchange={exchange} />
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </aside>
  );
}

/** One step of the pipeline: a dot for its state, the server's own sentence. */
function StepRow({ step }: { step: ActivityLine }) {
  const running = step.state === "start" || step.state === "running";
  const Icon =
    step.state === "error"
      ? TriangleAlert
      : step.state === "skipped"
        ? CircleSlash
        : (ICONS[step.step] ?? Check);

  return (
    <li
      className={cn(
        "flex items-center gap-2 rounded-lg px-2 py-1.5",
        running && "bg-surface-2",
      )}
    >
      {running ? (
        <span
          aria-hidden
          className="pulse-dot size-1.5 shrink-0 rounded-full bg-ink"
        />
      ) : (
        <Icon aria-hidden className="size-3 shrink-0 text-ink-muted" />
      )}

      <span
        className={cn(
          "min-w-0 flex-1 truncate text-[0.75rem] leading-relaxed",
          running ? "text-ink" : "text-ink-soft",
        )}
        title={step.detail ? `${step.label} — ${step.detail}` : step.label}
      >
        {step.label}
        {step.detail && (
          <span className="text-ink-muted"> · {step.detail}</span>
        )}
      </span>

      {step.ms !== null && (
        <span className="shrink-0 tabular-nums text-[0.66rem] text-ink-muted">
          {elapsed(step.ms)}
        </span>
      )}
    </li>
  );
}

/**
 * What was actually said. Both halves — the earlier card showed the question
 * and labelled the reply without ever printing it, which reads as a bug the
 * moment an answer is longer than its label.
 */
function ExchangeRow({ exchange }: { exchange: Exchange }) {
  const language = languageName(exchange.languageCode);

  return (
    <li className="flex flex-col gap-1 rounded-lg px-2 py-1.5 transition-colors hover:bg-surface-2">
      <div className="flex items-start justify-between gap-2">
        <p className="line-clamp-2 min-w-0 text-[0.75rem] leading-relaxed text-ink">
          {exchange.question}
        </p>
        {language && <PanelChip className="mt-px shrink-0">{language}</PanelChip>}
      </div>

      <p className="flex gap-1.5 text-[0.72rem] leading-relaxed text-ink-muted">
        <span aria-hidden className="shrink-0 text-ink-soft">
          &rsaquo;
        </span>
        <span className="line-clamp-3 min-w-0">
          {exchange.reply ||
            (exchange.done ? "No answer came back." : "Writing the reply…")}
          {exchange.interrupted && (
            <span className="text-ink-muted"> (interrupted)</span>
          )}
        </span>
      </p>
    </li>
  );
}

/** Server milliseconds, at the precision worth reading at a glance. */
function elapsed(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
