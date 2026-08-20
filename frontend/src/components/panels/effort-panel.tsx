"use client";

import { useCallback } from "react";

import { PanelHeading, PanelRule } from "@/components/panels/panel";
import { Slider } from "@/components/ui/slider";
import { EFFORT_LEVELS, useEffort } from "@/lib/effort";
import { cn } from "@/lib/utils";

/**
 * How hard the agent should think before it answers. The levels themselves and
 * the persisted state live in `@/lib/effort`, because the capture flow reads
 * the same value when it decides whether to run retrieval.
 */
export function EffortPanel() {
  const [effort, setEffort] = useEffort();

  const onValueChange = useCallback(
    ([next]: number[]) => setEffort(next),
    [setEffort],
  );

  const level = EFFORT_LEVELS[effort];

  return (
    <>
      <PanelHeading title="Effort" hint={level.hint} />

      <PanelRule />

      <div className="flex flex-col gap-3 px-2 pt-1 pb-1">
        <Slider
          aria-label="Agent effort"
          min={0}
          max={EFFORT_LEVELS.length - 1}
          step={1}
          value={[effort]}
          onValueChange={onValueChange}
        />

        <div className="flex justify-between">
          {EFFORT_LEVELS.map(({ label }, index) => (
            <button
              key={label}
              type="button"
              onClick={() => setEffort(index)}
              className={cn(
                "rounded-md px-1 py-0.5 text-[0.68rem] tracking-wide transition-colors",
                index === effort
                  ? "font-medium text-ink"
                  : "text-ink-muted hover:text-ink-soft",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
