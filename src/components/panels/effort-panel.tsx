"use client";

import { useCallback } from "react";

import { PanelHeading, PanelRule } from "@/components/panels/panel";
import { Slider } from "@/components/ui/slider";
import { usePersistentState } from "@/hooks/use-persistent-state";
import { cn } from "@/lib/utils";

/**
 * How hard the agent should think before it answers. Held here and persisted
 * per-device — nothing consumes it yet, since Vec stops at the transcript.
 */
export const EFFORT_LEVELS = [
  { label: "Instant", hint: "Transcribe and stop. No reasoning pass." },
  { label: "Balanced", hint: "A quick reply on top of what you said." },
  { label: "Deep", hint: "The agent thinks it through before answering." },
  { label: "Max", hint: "Longest reasoning budget. Slowest to come back." },
] as const;

const STORAGE_KEY = "vec-effort";
const DEFAULT_LEVEL = 1;

function reviveEffort(raw: unknown): number | null {
  if (typeof raw !== "number" || !Number.isInteger(raw)) return null;
  if (raw < 0 || raw >= EFFORT_LEVELS.length) return null;
  return raw;
}

export function EffortPanel() {
  const [effort, setEffort] = usePersistentState<number>(
    STORAGE_KEY,
    DEFAULT_LEVEL,
    reviveEffort,
  );

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
