"use client";

import { CornerDownLeft } from "lucide-react";

/**
 * Openers, in three scripts.
 *
 * Someone landing on a wordless orb has no way to know it answers at all, let
 * alone that it answers in the language it was spoken to. These are here to be
 * *read* rather than tapped — tapping one is just the shortest way to prove it
 * without a microphone.
 *
 * Three, not the twenty-two Vec listens in: an opener in a script the reader
 * cannot read proves nothing to them, it is a line of shapes above a button.
 * Hindi and English carry the demonstration for almost everyone who lands
 * here; the third is what stops the pair reading as the whole list. Tamil
 * holds that slot on reach — it is the largest of the remaining scripts — and
 * swapping it for another Indic opener is a one-line change.
 */
const STARTERS: { text: string; language: string; label: string }[] = [
  { text: "नमस्ते, आज आप कैसे हैं?", language: "hi-IN", label: "Hindi" },
  { text: "தமிழ்நாட்டின் தலைநகரம் எது?", language: "ta-IN", label: "Tamil" },
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

      {/*
        A pill is a single line by construction — the shape stops reading as
        one the moment its text wraps. On a narrow screen the English opener is
        wider than the column, so the text truncates inside the pill and the
        `title` carries the language it is written in either way.
      */}
      <ul className="flex w-full flex-wrap justify-center gap-1.5">
        {STARTERS.map((starter) => (
          <li key={starter.text} className="max-w-full">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(starter.text, starter.language)}
              lang={starter.language}
              title={starter.label}
              className="glass glass-hover flex max-w-full items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.74rem] text-ink-soft disabled:pointer-events-none disabled:opacity-40 sm:px-3 sm:py-1.5 sm:text-[0.78rem]"
            >
              <span className="truncate">{starter.text}</span>
              <CornerDownLeft aria-hidden className="size-3 shrink-0 text-ink-muted" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
