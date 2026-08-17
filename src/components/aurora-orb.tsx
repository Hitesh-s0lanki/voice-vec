import type { CSSProperties } from "react";

import type { VoiceState } from "@/lib/types";

/** Circumference of the countdown ring (r = 48 in a 100×100 viewBox). */
const RING = 2 * Math.PI * 48;

/** Heights the seven core bars rest at when nothing is being captured. */
const RESTING = [0.22, 0.38, 0.6, 0.86, 0.6, 0.38, 0.22];

/** Shortest a bar ever draws, so silence still reads as seven pills. */
const FLOOR = 0.16;

/** Distinct levels behind the mirror: four, for a seven-bar arch. */
const GROUPS = Math.ceil(RESTING.length / 2);

/**
 * Fold the analyser's bands down to the seven bars in the core.
 *
 * Speech energy piles into the low end, so a straight left-to-right fold
 * leaves the right-hand bars dead. Average four groups and mirror them
 * instead — the loudest band sits in the middle and the pulse reads as one
 * voice rather than a spectrum.
 */
function coreBars(bands: number[]): number[] {
  const size = Math.max(1, Math.floor(bands.length / GROUPS));

  // low to high, so the lowest group — the loudest — lands dead centre
  const levels = Array.from({ length: GROUPS }, (_, group) => {
    let sum = 0;
    for (let i = 0; i < size; i++) sum += bands[group * size + i] ?? 0;
    return sum / size;
  });

  return [...levels.slice(1).reverse(), ...levels].map(
    (value) => FLOOR + Math.min(1, value) * (1 - FLOOR),
  );
}

type AuroraOrbProps = {
  state: VoiceState;
  /** Live input level, 0–1. Drives the orb's scale while listening. */
  level?: number;
  /** Live band levels, 0–1 each. Drive the bar heights while listening. */
  bands?: number[];
  /** Share of the recording window still left, 0–1. Draws the countdown ring. */
  progress?: number;
  disabled?: boolean;
  onToggle: () => void;
};

/**
 * The centrepiece: one tappable circle, unfilled, with seven level bars on it.
 * Everything but the countdown ring is CSS — see `.orb-*` in globals.css.
 */
export function AuroraOrb({
  state,
  level = 0,
  bands = [],
  progress = 1,
  disabled = false,
  onToggle,
}: AuroraOrbProps) {
  const listening = state === "listening";
  const heights = listening && bands.length ? coreBars(bands) : RESTING;

  return (
    <div
      data-state={state}
      style={{ "--orb-live": listening ? level : 0 } as CSSProperties}
      className="orb relative grid aspect-square w-[min(68vw,19rem)] place-items-center"
    >
      {/* echo rings, only while capturing */}
      {listening && (
        <div className="pointer-events-none absolute inset-0">
          <span className="orb-echo absolute inset-0 rounded-full" />
          <span className="orb-echo absolute inset-0 rounded-full" />
          <span className="orb-echo absolute inset-0 rounded-full" />
        </div>
      )}

      {/* the 30s ceiling, drawn instead of counted out in words */}
      {listening && (
        <svg
          aria-hidden
          viewBox="0 0 100 100"
          className="orb-timer fade pointer-events-none absolute inset-0 -rotate-90"
        >
          <circle
            cx="50"
            cy="50"
            r="48"
            fill="none"
            stroke="var(--line)"
            strokeWidth="0.5"
          />
          <circle
            cx="50"
            cy="50"
            r="48"
            fill="none"
            stroke="var(--tone-strong)"
            strokeWidth="0.9"
            strokeLinecap="round"
            strokeDasharray={RING}
            strokeDashoffset={RING * (1 - Math.max(0, Math.min(1, progress)))}
          />
        </svg>
      )}

      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-pressed={listening}
        aria-label={listening ? "Stop and transcribe" : "Start listening"}
        className="group absolute inset-[7%] grid cursor-pointer place-items-center rounded-full border border-line shadow-sm outline-none transition-transform duration-300 hover:scale-[1.03] focus-visible:ring-2 focus-visible:ring-ink/30 focus-visible:ring-offset-4 focus-visible:ring-offset-shell active:scale-[0.97] disabled:cursor-wait disabled:hover:scale-100"
      >
        <span
          aria-hidden
          className="relative flex h-[62%] w-[78%] items-center justify-center gap-[5%] transition-transform duration-300 group-hover:scale-105 group-disabled:opacity-40"
        >
          {heights.map((height, i) => (
            <span
              key={i}
              className="orb-bar w-[9%] rounded-full"
              style={{ "--i": i, height: `${height * 100}%` } as CSSProperties}
            />
          ))}
        </span>
      </button>
    </div>
  );
}
