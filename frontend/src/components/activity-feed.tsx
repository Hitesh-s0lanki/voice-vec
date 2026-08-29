"use client";

import {
  AudioLines,
  Brain,
  Check,
  CircleSlash,
  Database,
  Radio,
  Sparkles,
  TriangleAlert,
  Volume2,
  Waves,
  Wrench,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { PanelHeading, PanelRule } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import type { ActivityLine, Exchange, VoiceStatus } from "@/hooks/use-voice-session";
import { cn } from "@/lib/utils";

/** Steps the card holds. Older ones are still in the hook's scrollback. */
const STEPS = 5;

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
  memory: Brain,
  retrieval: Database,
  tool: Wrench,
  llm: Sparkles,
  speech: Volume2,
  turn: Check,
};

type FeedProps = {
  status: VoiceStatus;
  activity: ActivityLine[];
  exchanges: Exchange[];
};

/**
 * What the backend is doing.
 *
 * The orb has four states and cannot say more than "thinking" — which covers
 * retrieval, a tool call and the model itself, three things with very
 * different reasons for being slow. This is where that distinction lives:
 * every step the server announces, in the order it started, newest first,
 * with the elapsed time it reported. What was actually said belongs to the
 * transcript and Conversations panels, not here.
 *
 * From `lg` up it is a card pinned to the top-right corner, out of the flow
 * so the step log can grow without touching the orb. Below that there is no
 * spare corner — the log would grow straight down over the orb — so it moves
 * behind a button in the same corner and opens as a drawer off the right
 * edge. Same contents either way; only one of the two is ever mounted.
 */
export function ActivityFeed(props: FeedProps) {
  return (
    <>
      <aside
        aria-label="Activity"
        // `max-h` keeps it off the transcript card in the opposite corner on
        // a short laptop.
        className="glass fixed top-5 right-5 z-40 hidden max-h-[calc(100dvh-2.5rem)] w-80 flex-col gap-2 overflow-hidden rounded-2xl p-2 lg:flex"
      >
        <FeedBody {...props} />
      </aside>

      <ActivityDrawer {...props} />
    </>
  );
}

/**
 * The small-screen entry point: one glass button where the card would have
 * been, carrying the live dot so a running pipeline is still visible from the
 * stage without the log being open.
 */
function ActivityDrawer(props: FeedProps) {
  const live = props.status !== "idle" && props.status !== "error";

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-lg"
          className="glass fixed top-4 right-4 z-40 rounded-xl text-ink-muted sm:top-5 sm:right-5 lg:hidden"
        >
          {live ? (
            <span
              aria-hidden
              className="pulse-dot size-2 rounded-full bg-ink"
            />
          ) : (
            <Waves aria-hidden className="size-[1.15rem]" />
          )}
          <span className="sr-only">Activity</span>
        </Button>
      </SheetTrigger>

      <SheetContent
        side="right"
        title="Activity"
        // The drawer is as tall as the screen, so unlike the corner card the
        // list has real room — it gets the scroll rather than the panel.
        className="gap-2 p-2 pt-3"
      >
        <FeedBody {...props} />
      </SheetContent>
    </Sheet>
  );
}

/** Heading, rule and the step list — the part both presentations share. */
function FeedBody({ status, activity, exchanges }: FeedProps) {
  const live = status !== "idle" && status !== "error";

  // Steps arrive in pipeline order and each one keeps the slot it started in,
  // so reversing puts the newest at the top without any row ever moving.
  const steps = activity.slice(-STEPS).reverse();
  const waiting = live ? WAITING[status] : undefined;

  return (
    <>
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

      {steps.length === 0 && !waiting ? (
        <p className="px-2 pt-0.5 pb-1.5 text-[0.72rem] leading-relaxed text-ink-muted">
          Tap the orb and every step the backend takes shows up here as it
          happens.
        </p>
      ) : (
        <ul
          aria-live="polite"
          className="flex min-h-0 flex-col gap-0.5 overflow-y-auto scrollbar-thin"
        >
          {/*
            The client's own state, shown only until the server's first step
            for this turn lands. Recording and uploading happen entirely in the
            browser, so nothing on the socket describes them — and a log that
            goes blank while the microphone is open reads as broken.
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
    </>
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
    // `data-active` is the hook `.glass-row` watches for a held selection —
    // the same one the effort levels use. Hovering a row still lights it,
    // which is worth having: the labels truncate and carry a title tooltip.
    <li data-active={running} className="glass-row flex items-center gap-2 rounded-lg px-2 py-1.5">
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

/** Server milliseconds, at the precision worth reading at a glance. */
function elapsed(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
