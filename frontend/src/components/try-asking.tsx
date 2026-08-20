"use client";

import { CornerDownLeft } from "lucide-react";

/**
 * Openers, in four scripts.
 *
 * Someone landing on a wordless orb has no way to know it will answer in
 * Tamil, or that it answers at all. These are here to be *read* rather than
 * tapped — the point they make is the language range, and tapping one is just
 * the shortest way to prove it without a microphone.
 */
const STARTERS: { text: string; language: string; label: string }[] = [
  { text: "नमस्ते, आज आप कैसे हैं?", language: "hi-IN", label: "Hindi" },
  { text: "தமிழ்நாட்டின் தலைநகரம் எது?", language: "ta-IN", label: "Tamil" },
  { text: "ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ಹೇಗಿದೆ?", language: "kn-IN", label: "Kannada" },
  { text: "Tell me something surprising about the sea.", language: "en-IN", label: "English" },
];

export function TryAsking({
  onPick,
  disabled,
}: {
  onPick: (text: string, language: string) => void;
  disabled?: boolean;
}) {
  return (
    <section
      aria-label="Ways to start"
      className="fade flex w-full flex-col items-center gap-2.5"
    >
      <p className="text-[0.72rem] text-ink-muted">
        Speak in any language — or tap one to hear it answer
      </p>

      <ul className="flex flex-wrap justify-center gap-1.5">
        {STARTERS.map((starter) => (
          <li key={starter.text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(starter.text, starter.language)}
              lang={starter.language}
              title={starter.label}
              className="glass glass-hover flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[0.78rem] text-ink-soft disabled:opacity-40"
            >
              {starter.text}
              <CornerDownLeft aria-hidden className="size-3 shrink-0 text-ink-muted" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
