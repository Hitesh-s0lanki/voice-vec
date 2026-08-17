import type { CSSProperties } from "react";

type WaveformProps = {
  /** Live band levels, 0–1 each, straight off the analyser. */
  bars: number[];
};

export function Waveform({ bars }: WaveformProps) {
  return (
    <div aria-hidden className="flex h-10 items-center justify-center gap-1.25">
      {bars.map((value, i) => (
        <span
          key={i}
          className="wave-bar w-0.75 rounded-full"
          style={
            {
              "--i": i,
              transform: `scaleY(${Math.max(0.06, value)})`,
            } as CSSProperties
          }
        />
      ))}
    </div>
  );
}
