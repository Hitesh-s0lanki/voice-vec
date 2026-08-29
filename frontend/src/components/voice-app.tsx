"use client";

import { useCallback, useEffect } from "react";
import { RotateCcw, TriangleAlert } from "lucide-react";

import { ActivityFeed } from "@/components/activity-feed";
import { AuroraOrb } from "@/components/aurora-orb";
import { ToolCallCard } from "@/components/tool-call-card";
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
  const { record, answer, attach, adopt } = useConversation();

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
    currentTools,
    exchanges,
    current,
    start,
    stop,
    cancel,
    ask,
  } = useVoiceSession({
    onExchange: remember,
    conversationId,
    onConversation: remembered,
    // Straight through: a finished call is already in the shape the thread
    // stores, and it is held for its turn until `remember` files one above.
    onTool: attach,
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
   * is `fixed` to a corner from `lg` up and costs this grid no height at all.
   * What is left in the centre column is bounded copy, so the reserved halves
   * can be small and the whole stage fits without page scroll on a short
   * laptop. That is the real guarantee that the orb never moves: nothing in
   * the flow can push it, and the page it sits on has nowhere to scroll to.
   *
   * Below `lg` there are no free corners — a tablet is not wide enough to
   * hold a 320px card either side of the orb without one of them landing on
   * it — so the transcript comes back into the column and the activity log
   * moves behind a drawer. The column needs room for it, so the halves stop
   * being equal: the bottom one is floored at the card's height and the orb
   * rides a little above centre. Unequal, but still fixed — both floors are
   * reserved whether or not anything is in them.
   *
   * Below `lg` the bottom row is `minmax(0, 1fr)` — a *weight*, never a fixed
   * floor. A floor there is what put the openers on top of the footer line on
   * a 667px phone: 13rem reserved for the bottom row, out of the 189px left
   * once the orb and the padding had taken theirs, and the grid simply
   * overflowed its own bottom padding into the footer's band. With no floor
   * the row takes what is left and the column inside it scrolls, so the stage
   * cannot outgrow its box at any height. The orb still cannot move: both
   * outer rows are `fr`, so their heights come from the viewport rather than
   * from whatever the column happens to be holding. The 0.55 weight is what
   * keeps the orb a little above centre, where the row below needs the room.
   *
   * The bottom padding is what the stage owes the fixed furniture below it:
   * below `lg` the centred rail with the footer credit line above it (120px),
   * from `lg` neither — the rail is back in its corner and the line no longer
   * reaches the middle (48px).
   *
   * Padding is `pt`/`pb` rather than `py` on purpose. `py-12` at `sm` and
   * `pb-32` at `max-lg` are the same property from two variants, and the
   * cascade — not the intent — picks the winner: `sm:py-12` was quietly
   * taking the bottom padding on every tablet back to 48px, leaving the rail
   * sitting over the stage's own content box.
   */
  return (
    <div className="relative grid h-dvh min-h-136 grid-rows-[minmax(2rem,0.55fr)_auto_minmax(0,1fr)] justify-items-center gap-6 px-4 pt-8 pb-30 sm:gap-9 sm:px-6 sm:pt-12 short:min-h-0 short:gap-4 short:pt-6 short:pb-28 lg:min-h-152 lg:grid-rows-[minmax(7.5rem,1fr)_auto_minmax(7.5rem,1fr)] lg:pb-12">
      {/* out of flow either way — a fixed corner card from `lg`, a fixed
          drawer trigger below it — so it costs the centred stack no height */}
      <ActivityFeed status={status} activity={activity} exchanges={exchanges} />

      <StatusPill status={status} remaining={remaining} />

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
          The bottom-left stack: what the agent ran, then what it heard.

          Rendered here, but `lg:fixed` — from `lg` up it lifts out of the flow
          into the bottom-left corner and this slot collapses to nothing. On a
          phone or tablet, where there is no spare corner, it stays in the
          column and takes height.

          Anchored by its `bottom` edge, so the transcript keeps the floor and
          the tool card grows upward into empty space above it — no card in the
          stack can move the orb, whichever of them is open. Order matters for
          the same reason it does in a stack trace: the tools ran *before* the
          answer, and both sit under the question they came from.

          What is not here is the reply's text. The answer is audio, and the
          words it spoke stay in the Conversations panel rather than racing the
          voice down the screen.
        */}
        <div className="flex w-full flex-col gap-2 empty:hidden lg:fixed lg:bottom-5 lg:left-5 lg:z-40 lg:w-80">
          <ToolCallCard calls={currentTools} status={status} />
          <TranscriptCard
            exchange={current}
            status={status}
            speaking={speaking}
            onStop={cancel}
          />
        </div>

        {/* below `lg` the openers step aside once there is a take to read */}
        {!listening && !working && !speaking && !error && (
          <div className={cn("w-full short:hidden", current && "max-lg:hidden")}>
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
 *
 * Three states, and no control: at rest it says how to begin, while recording
 * it counts the take down, and for everything between the take and the last
 * word of the answer it says `Streaming` — because that is one continuous
 * thing to a listener, not the three stages it is on the wire.
 */
function StatusPill({
  status,
  remaining,
}: {
  status: string;
  remaining: number;
}) {
  const seconds = Math.ceil(remaining / 1000);
  const listening = status === "listening";
  const resting = status === "idle" || status === "error";

  return (
    <div className="flex h-8 items-center self-end">
      <span className="glass fade flex h-8 items-center gap-2 rounded-full px-3.5 text-[0.78rem] font-medium tracking-wide text-ink-muted">
        <span
          aria-hidden
          className={
            listening
              ? "pulse-dot size-1.5 rounded-full bg-ink"
              : resting
                ? "size-1.5 rounded-full border border-line"
                : "size-1.5 animate-pulse rounded-full bg-ink-soft"
          }
        />
        {listening ? (
          <span className="tabular-nums">
            Listening · {String(Math.floor(seconds / 60)).padStart(1, "0")}:
            {String(seconds % 60).padStart(2, "0")} left
          </span>
        ) : (
          <span>{resting ? "Tap to speak" : "Streaming"}</span>
        )}
      </span>
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
