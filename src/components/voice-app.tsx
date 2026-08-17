"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy, RotateCcw, TriangleAlert } from "lucide-react";

import { AuroraOrb } from "@/components/aurora-orb";
import { Waveform } from "@/components/waveform";
import { Button } from "@/components/ui/button";
import {
  MAX_RECORDING_MS,
  useVoiceCapture,
  type Transcript,
} from "@/hooks/use-voice-capture";
import { useConversation } from "@/lib/conversation";
import { languageName } from "@/lib/languages";
import type { VoiceState } from "@/lib/types";

export function VoiceApp() {
  const {
    status,
    bars,
    level,
    transcript,
    error,
    remaining,
    start,
    stop,
    reset,
  } = useVoiceCapture();

  const { record } = useConversation();
  const recordedRef = useRef<Transcript | null>(null);

  // Log each take exactly once. The hook hands back a fresh object per upload,
  // so comparing identity survives both re-renders and StrictMode's double run.
  useEffect(() => {
    if (!transcript || recordedRef.current === transcript) return;
    recordedRef.current = transcript;
    record(transcript);
  }, [record, transcript]);

  const listening = status === "listening";
  const thinking = status === "thinking";
  const settled = !listening && !thinking;
  const orbState: VoiceState = thinking
    ? "thinking"
    : listening
      ? "listening"
      : "idle";

  const toggle = () => {
    if (thinking) return;
    if (listening) {
      stop();
      return;
    }
    if (transcript || error) reset();
    void start();
  };

  return (
    {/*
      Three rows, the outer two identical: whatever the pill above and the
      transcript below are doing, they get the same slice of the screen, so
      the orb sits on the true centre line instead of being nudged up by the
      taller half. `minmax` keeps both halves' space reserved at rest, which
      is what stops the orb jumping the moment a transcript lands.
    */}
    <div className="relative grid min-h-dvh grid-rows-[minmax(12rem,1fr)_auto_minmax(12rem,1fr)] justify-items-center gap-9 px-6 py-12">
      <StatusPill
        listening={listening}
        thinking={thinking}
        remaining={remaining}
      />

      <AuroraOrb
        state={orbState}
        level={level}
        bands={bars}
        progress={remaining / MAX_RECORDING_MS}
        disabled={thinking}
        onToggle={toggle}
      />

      {/* the page still needs a heading; the screen no longer needs to show it */}
      <h1 className="sr-only">Vec</h1>

      <div className="relative flex w-full max-w-xl flex-col items-center gap-4">
        {/*
          Always mounted, empty at rest. A live region has to be in the DOM
          before its text changes or the change goes unannounced — and with
          no siblings at rest, the empty node costs no layout.
        */}
        <p
          aria-live="polite"
          className="type-display text-[1.5rem] font-medium text-ink empty:hidden"
        >
          {listening
            ? "Listening"
            : thinking
              ? "Transcribing"
              : error
                ? "Didn't catch that"
                : transcript
                  ? "Heard you"
                  : null}
        </p>

        {/* wrapped only to keep the entrance — Waveform takes no className */}
        {listening && (
          <div className="fade">
            <Waveform bars={bars} />
          </div>
        )}

        {thinking && <Thinking />}

        {settled && transcript && (
          <Result
            text={transcript.text}
            languageCode={transcript.languageCode}
            onAgain={toggle}
          />
        )}

        {settled && error && <ErrorCard message={error} onRetry={toggle} />}
      </div>
    </div>
  );
}

/**
 * Floating state read-out above the orb, so the orb itself stays wordless.
 * Nothing shows at rest — an idle app has no status worth a badge — and it
 * hugs the bottom of its row, so it reads as attached to the orb rather than
 * stranded at the top of the screen.
 */
function StatusPill({
  listening,
  thinking,
  remaining,
}: {
  listening: boolean;
  thinking: boolean;
  remaining: number;
}) {
  const seconds = Math.ceil(remaining / 1000);

  return (
    <div className="flex h-8 items-center">
      {(listening || thinking) && (
        <span className="glass fade flex h-8 items-center gap-2 rounded-full px-3.5 text-[0.78rem] font-medium tracking-wide text-ink-muted">
          <span
            aria-hidden
            className={
              listening
                ? "pulse-dot size-1.5 rounded-full bg-ink"
                : "size-1.5 animate-pulse rounded-full bg-ink-soft"
            }
          />
          {listening ? (
            <span className="tabular-nums">
              Recording · {String(Math.floor(seconds / 60)).padStart(1, "0")}:
              {String(seconds % 60).padStart(2, "0")} left
            </span>
          ) : (
            <span>Sending to Sarvam Saaras</span>
          )}
        </span>
      )}
    </div>
  );
}

/** Placeholder lines that stand in for the transcript being written. */
function Thinking() {
  return (
    <div className="fade flex w-full max-w-md flex-col items-center gap-4">
      <div className="flex w-full flex-col items-center gap-2.5">
        <span className="shimmer h-2.5 w-full rounded-full" />
        <span className="shimmer h-2.5 w-11/12 rounded-full" />
        <span className="shimmer h-2.5 w-2/3 rounded-full" />
      </div>
      <span aria-hidden className="dots flex items-center gap-1.5">
        <i />
        <i />
        <i />
      </span>
    </div>
  );
}

function Result({
  text,
  languageCode,
  onAgain,
}: {
  text: string;
  languageCode: string | null;
  onAgain: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const language = languageName(languageCode);
  const words = text.trim().split(/\s+/).filter(Boolean).length;

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(timer);
  }, [copied]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      // clipboard blocked — leave the button in its resting state
    }
  };

  return (
    <figure className="glass card-edge rise relative w-full overflow-hidden rounded-2xl px-6 py-5">
      <blockquote className="type-quote text-center text-[1.2rem] leading-relaxed text-balance text-ink">
        {text}
      </blockquote>

      <figcaption className="mt-5 flex flex-wrap items-center justify-center gap-2 text-[0.75rem] text-ink-muted">
        {language && (
          <span className="rounded-full border border-line px-2.5 py-1 font-medium tracking-wide text-ink-soft">
            {language}
          </span>
        )}
        <span className="tabular-nums">
          {words} {words === 1 ? "word" : "words"}
        </span>

        <span aria-hidden className="mx-1 h-3.5 w-px bg-line" />

        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={copy}
          className="glass-hover rounded-full text-ink-muted"
        >
          {copied ? (
            <Check aria-hidden className="text-ink" />
          ) : (
            <Copy aria-hidden />
          )}
          {copied ? "Copied" : "Copy"}
        </Button>

        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onAgain}
          className="glass-hover rounded-full text-ink-muted"
        >
          <RotateCcw aria-hidden />
          Again
        </Button>
      </figcaption>
    </figure>
  );
}

function ErrorCard({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div
      role="alert"
      className="glass glass-danger rise flex w-full max-w-md flex-col items-center gap-3 rounded-2xl px-6 py-5 text-center"
    >
      <span className="grid size-9 place-items-center rounded-full bg-ink/8 text-ink">
        <TriangleAlert aria-hidden className="size-4.5" />
      </span>
      <p className="text-[0.9rem] leading-relaxed text-ink-soft">{message}</p>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onRetry}
        className="glass-hover rounded-full text-ink-muted"
      >
        <RotateCcw aria-hidden />
        Try again
      </Button>
    </div>
  );
}
