"use client";

import { PanelHeading, PanelRule } from "@/components/panels/panel";
import { EFFORT_LEVELS, useEffort } from "@/lib/effort";
import { cn } from "@/lib/utils";

/**
 * How the question gets answered. Five rungs, each a different retrieval
 * architecture — the levels and the persisted state live in `@/lib/effort`,
 * because the voice session reads the same value when it starts a take.
 *
 * A vertical list rather than a slider with labels underneath, for a reason
 * that is measurable rather than aesthetic: five labels laid across a `w-72`
 * panel need 291px of a 272px content box, so the last one ("Adaptive") was
 * clipped mid-word. Stacking them also buys each rung a real one-line
 * description, which a tick label can never carry — and the ordering the
 * slider was there to express is just as visible top-to-bottom.
 */
export function EffortPanel() {
  const [effort, setEffort] = useEffort();

  return (
    <>
      <PanelHeading title="Effort" hint="How the question gets answered." />

      <PanelRule />

      <div role="radiogroup" aria-label="Agent effort" className="flex flex-col gap-0.5">
        {EFFORT_LEVELS.map(({ label, hint, detail, cost }, index) => (
          <button
            key={label}
            type="button"
            role="radio"
            aria-checked={index === effort}
            title={detail}
            onClick={() => setEffort(index)}
            // `data-active` is the hook `.glass-row` watches for a held
            // selection — see the note above it in globals.css.
            data-active={index === effort}
            className="glass-row flex flex-col gap-0.5 rounded-lg px-2 py-1.5 text-left"
          >
            <span className="flex items-baseline justify-between gap-2">
              <span
                className={cn(
                  "text-[0.78rem] tracking-[-0.01em]",
                  index === effort ? "font-medium text-ink" : "text-ink-soft",
                )}
              >
                {label}
              </span>
              <span className="shrink-0 text-[0.66rem] tabular-nums tracking-wide text-ink-muted">
                {cost}
              </span>
            </span>

            {/* One line, always. The strings in `@/lib/effort` are written to
                that budget; truncating is the backstop, not the plan. */}
            <span className="truncate text-[0.7rem] leading-snug text-ink-muted">
              {hint}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}
