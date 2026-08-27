"use client";

import { useCallback, useEffect } from "react";
import { RotateCcw, TriangleAlert } from "lucide-react";

import { ActivityFeed } from "@/components/activity-feed";
import { AuroraOrb } from "@/components/aurora-orb";
import { TranscriptCard } from "@/components/transcript-card";
import { TryAsking } from "@/components/try-asking";
import { Waveform } from "@/components/waveform";
import { Button } from "@/components/ui/button";
import {
  MAX_RECORDING_MS,
  useVoiceSession,
  type Exchange,
} from "@/hooks/use-voice-session";
import { useConversation } from "@/lib/conversation";
import { cn } from "@/lib/utils";
import { useHandsFree } from "@/lib/voice-settings";
import type { VoiceState } from "@/lib/types";

/**
 * The whole app: speak, and be answered out loud in the language you spoke.
 *
 * The stage shows one thing at a time, and never the answer's words. That is
 * deliberate — a spoken conversation is heard, not read. The orb says what is
 * happening, the corner cards hold the transcript and the last few takes, and
 * the rail panels keep the full history for anyone who wants to read it back.
 */
export function VoiceApp({ conversationId }: { conversationId?: string }) {
  const [handsFree, setHandsFree] = useHandsFree();
  const { record, answer, adopt } = useConversation();

  // Log each finished exchange for the Conversations panel. The backend has
  // already filed the same turn under the same id; this is the optimistic
  // copy, so the thread on screen never waits on a round trip to Neon.
  const remember = useCallback(
    (exchange: Exchange) => {
      const id = record(
        { text: exchange.question, languageCode: exchange.languageCode },
        exchange.id,
      );
      if (!id) return;

      answer(id, {
        reply: exchange.reply || null,
        replyStatus: exchange.interrupted
          ? "interrupted"
          : exchange.reply
            ? "answered"
            : "abstained",
        replyReason: null,
        replyMs: exchange.timings?.total ?? null,
      });
    },
    [answer, record],
  );

  /**
   * The server opened (or reopened) the conversation this take belongs to.
   *
   * `adopt` writes `/c/{id}` into the address bar without navigating, so the
   * socket carrying the reply survives getting an address — a real navigation
   * here would cut the answer off mid-word.
   */
  const remembered = useCallback((id: string) => adopt(id), [adopt]);

  const {
    status,
    listening,
    speaking,
    working,
    bars,
    level,
    error,
    remaining,
    providers,
    activity,
    exchanges,
    current,
    start,
    stop,
    cancel,
    ask,
  } = useVoiceSession({
    handsFree,
    onExchange: remember,
    conversationId,
    onConversation: remembered,
  });

  const orbState: VoiceState = speaking
    ? "speaking"
    : working
      ? "thinking"
      : listening
        ? "listening"
        : "idle";

  /** One control for the whole loop: talk, stop talking, or talk over it. */
  const toggle = useCallback(() => {
    if (listening) {
      stop();
      return;
    }
    // Tapping mid-answer is barge-in — `start` stops the playback itself.
    void start();
  }, [listening, start, stop]);

  // Space is the natural key for push-to-talk, and this screen has nothing
  // else competing for it.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, button, [contenteditable]")) return;

      event.preventDefault();
      toggle();
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  /*
   * Three rows, the outer two identical: whatever the pill above and the
   * stack below are doing, they get the same slice of the screen, so the orb
   * sits on the true centre line instead of being nudged up by the taller
   * half. `minmax` keeps both halves' space reserved at rest, which is what
   * stops the orb shifting the moment a take starts.
   *
   * Everything that grows unpredictably — the transcript, the activity log —
   * is `fixed` to a corner from `md` up and costs this grid no height at all.
   * What is left in the centre column is bounded copy, so the reserved halves
   * can be small and the whole stage fits without page scroll on a short
   * laptop. That is the real guarantee that the orb never moves: nothing in
   * the flow can push it, and the page it sits on has nowhere to scroll to.
   *
   * Below `md` the transcript comes back into the column and needs room, so
   * the halves stop being equal — the bottom one is floored at the card's
   * height and the orb rides a little above centre. Unequal, but still fixed:
   * both floors are reserved whether or not anything is in them.
   */
  return (
    <div className="relative grid h-dvh min-h-152 grid-rows-[minmax(3rem,1fr)_auto_minmax(15rem,1fr)] justify-items-center gap-9 px-6 py-12 max-md:pb-32 md:grid-rows-[minmax(7.5rem,1fr)_auto_minmax(7.5rem,1fr)]">
      {/* fixed to its corner, so it costs the centred stack no height */}
      <ActivityFeed status={status} activity={activity} exchanges={exchanges} />

      <StatusPill
        status={status}
        remaining={remaining}
        handsFree={handsFree}
        onHandsFree={() => setHandsFree(!handsFree)}
      />

      <AuroraOrb
        state={orbState}
        level={level}
        bands={bars}
        progress={remaining / MAX_RECORDING_MS}
        onToggle={toggle}
      />

      {/* the page still needs a heading; the screen no longer needs to show it */}
      <h1 className="sr-only">Vec</h1>

      <div className="relative flex min-h-0 w-full max-w-xl flex-col items-center gap-4 overflow-x-hidden overflow-y-auto scrollbar-thin">
        {/*
          Always mounted, empty at rest. A live region has to be in the DOM
          before its text changes or the change goes unannounced — and with no
          siblings at rest, the empty node costs no layout.
        */}
        <p
          aria-live="polite"
          className="type-display text-[1.5rem] font-medium text-ink empty:hidden"
        >
          {listening
            ? "Listening"
            : status === "transcribing"
              ? "Transcribing"
              : status === "thinking"
                ? "Thinking"
                : speaking
                  ? "Answering"
                  : error
                    ? "Didn't catch that"
                    : null}
        </p>

        {/* wrapped only to keep the entrance — Waveform takes no className */}
        {listening && (
          <div className="fade">
            <Waveform bars={bars} />
          </div>
        )}

        {/*
          Rendered here, but `md:fixed` — from `md` up it lifts out of the flow
          into the bottom-left corner and this slot collapses to nothing. Only
          on a phone, where there is no spare corner, does it stay in the
          column and take height. Either way it holds the transcript alone: the
          answer is audio, and the words it spoke stay in the Conversations
          panel rather than racing the voice down the screen.
        */}
        <TranscriptCard
          exchange={current}
          status={status}
          speaking={speaking}
          onStop={cancel}
        />

        {/* below `md` the openers step aside once there is a take to read */}
        {!listening && !working && !speaking && !error && (
          <div className={cn("w-full", current && "max-md:hidden")}>
            <TryAsking onPick={(text, language) => void ask(text, language)} disabled={working} />
          </div>
        )}

        {error && <ErrorCard message={error} onRetry={toggle} />}

        {providers && !providers.llm && (
          <p className="text-center text-[0.72rem] leading-relaxed text-ink-muted">
            No reply model is configured — add SARVAM_API_KEY or OPENAI_API_KEY
            to the backend&rsquo;s .env and restart it.
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * Floating state read-out above the orb, so the orb itself stays wordless.
 * At rest it carries the one control worth having up here: whether the mic
 * reopens by itself after each answer.
 */
function StatusPill({
  status,
  remaining,
  handsFree,
  onHandsFree,
}: {
  status: string;
  remaining: number;
  handsFree: boolean;
  onHandsFree: () => void;
}) {
  const seconds = Math.ceil(remaining / 1000);
  const live = status !== "idle" && status !== "error";

  const label: Record<string, string> = {
    connecting: "Connecting",
    transcribing: "Sending to Sarvam Saaras",
    thinking: "Writing the reply",
    speaking: "Speaking",
  };

  return (
    <div className="flex h-8 items-center self-end">
      {live ? (
        <span className="glass fade flex h-8 items-center gap-2 rounded-full px-3.5 text-[0.78rem] font-medium tracking-wide text-ink-muted">
          <span
            aria-hidden
            className={
              status === "listening"
                ? "pulse-dot size-1.5 rounded-full bg-ink"
                : "size-1.5 animate-pulse rounded-full bg-ink-soft"
            }
          />
          {status === "listening" ? (
            <span className="tabular-nums">
              Recording · {String(Math.floor(seconds / 60)).padStart(1, "0")}:
              {String(seconds % 60).padStart(2, "0")} left
            </span>
          ) : (
            <span>{label[status] ?? "Working"}</span>
          )}
        </span>
      ) : (
        <button
          type="button"
          onClick={onHandsFree}
          aria-pressed={handsFree}
          /* `.glass-hover` reads the `aria-pressed` below — hands-free being
             on is a state the pill is left in, not just one it passes through
             under the cursor. */
          className="glass glass-hover fade flex h-8 items-center gap-2 rounded-full px-3.5 text-[0.78rem] font-medium tracking-wide text-ink-muted"
          title="Reopen the microphone as soon as the answer ends"
        >
          <span
            aria-hidden
            className={
              handsFree
                ? "size-1.5 rounded-full bg-ink"
                : "size-1.5 rounded-full border border-line"
            }
          />
          Hands-free {handsFree ? "on" : "off"}
        </button>
      )}
    </div>
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
      <span className="glass-tile grid size-9 place-items-center rounded-full text-ink">
        <TriangleAlert aria-hidden className="size-4.5" />
      </span>
      <p className="text-[0.9rem] leading-relaxed text-ink-soft">{message}</p>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onRetry}
        className="rounded-full text-ink-muted"
      >
        <RotateCcw aria-hidden />
        Try again
      </Button>
    </div>
  );
}
