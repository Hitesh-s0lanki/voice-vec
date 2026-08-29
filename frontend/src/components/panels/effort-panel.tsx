"use client";

import { Database, Sparkles } from "lucide-react";

import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import { Slider } from "@/components/ui/slider";
import { EFFORT_LEVELS, OFFLINE_MAX_LEVEL, useEffort } from "@/lib/effort";
import { cn } from "@/lib/utils";

/**
 * How the question gets answered. Five rungs, each a different retrieval
 * architecture — the levels and the persisted state live in `@/lib/effort`,
 * because the voice session reads the same value when it starts a take.
 *
 * A range, because the rungs are ordered and the ordering is the point: one
 * notch right is strictly more work, more latency and more model calls than
 * the notch left of it, and a filled track says that at a glance in a way
 * five equal rows never did. Arrow keys move it, which is the same
 * increase/decrease the eye is being shown.
 *
 * The reason it used to be a list is real, and is why only *one* rung is
 * named here: five labels laid across a `w-72` panel need 291px of a 272px
 * content box, so "Adaptive" was clipped mid-word. Naming the selected rung
 * alone fits, and buys back the description a tick label could never carry.
 *
 * Three things are shown, in the order the choice is actually made in — what
 * this rung *is*, where it sits on the ladder, and what it costs you. The
 * last of those is the whole reason the control exists, so it is a sentence
 * and not just a number: the jump from rung 1 to rung 2 is the jump from no
 * model call to one, and that is a different kind of answer, not a slower one.
 */
export function EffortPanel() {
  const [effort, setEffort] = useEffort();

  const last = EFFORT_LEVELS.length - 1;
  const { label, hint, detail, cost } = EFFORT_LEVELS[effort];
  const offline = effort <= OFFLINE_MAX_LEVEL;

  return (
    <>
      <PanelHeading title="Effort" hint="How the question gets answered.">
        <PanelChip title="Where this rung sits on the ladder">
          {effort + 1}/{EFFORT_LEVELS.length}
        </PanelChip>
      </PanelHeading>

      <PanelRule />

      <div className="flex flex-col gap-3 px-1 pt-1">
        {/*
          The selected rung, given a surface of its own so the panel reads as
          a control with a read-out rather than three stacked paragraphs. Two
          lines whatever the rung, so moving the thumb never resizes the panel
          under the thumb that is moving it — `hint` is written to one line in
          `@/lib/effort`, and `detail` rides the `title` so nothing is lost.
        */}
        <div
          title={detail}
          className="glass-tile flex flex-col gap-1 rounded-xl px-2.5 py-2"
        >
          <span className="flex items-baseline justify-between gap-2">
            <span className="text-[0.82rem] font-medium tracking-[-0.01em] text-ink">
              {label}
            </span>
            <span className="shrink-0 text-[0.66rem] tabular-nums tracking-wide text-ink-muted">
              {cost}
            </span>
          </span>

          <span className="truncate text-[0.7rem] leading-snug text-ink-muted">
            {hint}
          </span>
        </div>

        <div className="flex flex-col gap-1.5 px-0.5">
          <div className="relative">
            <Slider
              aria-label="Agent effort"
              value={[effort]}
              valueText={label}
              onValueChange={([next]) => setEffort(next)}
              min={0}
              max={last}
              step={1}
            />

            {/*
              The five stops, drawn over the track rather than by it — Radix
              offers no ticks. Each sits where the thumb would: its centre
              travels `track width − thumb width`, inset half a thumb at each
              end, which is what the half-thumb correction in `left` is. The
              selected one is left out because the thumb is already parked on
              it, and the ones behind it are painted light because the filled
              range is ink underneath them.
            */}
            <span aria-hidden className="pointer-events-none absolute inset-0">
              {EFFORT_LEVELS.map((level, index) => {
                const fraction = index / last;
                if (index === effort) return null;

                return (
                  <span
                    key={level.label}
                    style={{
                      left: `calc(${fraction * 100}% + ${(0.5 - fraction) * 0.875}rem)`,
                    }}
                    className={cn(
                      "absolute top-1/2 size-1 -translate-x-1/2 -translate-y-1/2 rounded-full",
                      index < effort ? "bg-shell/60" : "bg-line-strong",
                    )}
                  />
                );
              })}
            </span>
          </div>

          {/* The direction, named. Without these the track is a bare rail and
              which end is "more" is something you have to drag to find out. */}
          <div className="flex items-baseline justify-between text-[0.66rem] tracking-wide text-ink-muted">
            <span>{EFFORT_LEVELS[0].label}</span>
            <span>{EFFORT_LEVELS[last].label}</span>
          </div>
        </div>

        {/*
          The line the ladder actually turns on — `OFFLINE_MAX_LEVEL` is where
          a model enters the loop, and with it the difference between a budget
          measured in milliseconds and one measured in seconds. The two icons
          are the ones the activity feed already uses for retrieval and for a
          model call, so the same two things mean the same two things here.
        */}
        <p
          className="flex items-start gap-1.5 rounded-lg px-0.5 text-[0.7rem] leading-snug text-ink-muted"
          aria-live="polite"
        >
          {offline ? (
            <Database aria-hidden className="mt-px size-3 shrink-0" />
          ) : (
            <Sparkles aria-hidden className="mt-px size-3 shrink-0" />
          )}
          {offline
            ? "No model call — the 200 ms budget holds."
            : "A model writes it, so latency is seconds."}
        </p>
      </div>
    </>
  );
}
