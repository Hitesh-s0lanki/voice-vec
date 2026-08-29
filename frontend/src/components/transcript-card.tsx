"use client";

import { useEffect, useState } from "react";
import { Check, Copy, Radio, Square } from "lucide-react";

import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import type { Exchange, VoiceStatus } from "@/hooks/use-voice-session";

/** What the heading says while a take is still on its way to being text. */
const HINT: Partial<Record<VoiceStatus, string>> = {
  connecting: "Reaching the voice service",
  listening: "Recording this take",
  transcribing: "Sending to Sarvam Saaras",
};

/**
 * Bottom-left: what Vec heard. Never what it said.
 *
 * The transcript is pinned to a corner rather than stacked under the orb for
 * two reasons. A corner card grows *against* its corner, so a long take can
 * never push the orb off the centre line the way a centred column could. And
 * the reply is deliberately absent — it is spoken, and printing it next to the
 * voice only invites reading ahead of it. The Conversations panel keeps the
 * written record for anyone who wants to go back over one.
 */
export function TranscriptCard({
  exchange,
  status,
  speaking,
  onStop,
}: {
  exchange: Exchange | null;
  status: VoiceStatus;
  speaking: boolean;
  onStop: () => void;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(timer);
  }, [copied]);

  const listening = status === "listening";
  // Mid-take the previous take's words are stale — the live state stands in.
  const stale =
    listening || status === "connecting" || status === "transcribing";
  const heard = stale ? "" : (exchange?.question ?? "");
  // Still waiting on words: the take is on its way to Saaras, or this is a
  // typed turn that went straight to thinking without one.
  const pending = !listening && !heard;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(heard);
      setCopied(true);
    } catch {
      // clipboard blocked — leave the button where it was
    }
  };

  /*
   * Nothing said yet, so there is no card — not an empty one explaining that
   * it is empty. It appears when a take starts and stays afterwards holding
   * the last thing heard, which is the only state worth a card at all.
   *
   * It costs the layout nothing to come and go: from `lg` up the stack it
   * sits in is `fixed`, and below that the row it lands in has its height
   * reserved either way.
   */
  const live = status !== "idle" && status !== "error";
  if (!heard && !live) return null;

  return (
    <aside
      aria-label="Transcript"
      /*
       * The last card in the bottom-left stack, and the one anchored to the
       * floor: from `lg` up that stack is `fixed` to the corner, so anything
       * above this — the tool calls — grows upward into empty space while the
       * transcript itself never moves. Below `lg` the stack is back in the
       * centre column and this behaves like any other card in it. The corner
       * geometry belongs to the stack, not here; see `voice-app.tsx`.
       */
      className="glass card-edge rise relative flex w-full flex-col gap-2 rounded-2xl p-2"
    >
      <PanelHeading title="Transcript">
        {heard && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={copy}
            // no `.glass-hover`: that one is for a surface floating over the
            // stage. This button sits *on* glass, where the row treatment the
            // ghost variant already carries is the one that reads.
            className="-mt-0.5 rounded-full text-ink-muted"
          >
            {copied ? (
              <Check aria-hidden className="text-ink" />
            ) : (
              <Copy aria-hidden />
            )}
            {copied ? "Copied" : "Copy"}
          </Button>
        )}
      </PanelHeading>

      <PanelRule />

      {/*
        The card mounts when the take starts, so this region is in the DOM
        holding "Listening…" well before the transcript replaces it — which is
        the change that has to be announced, and the one that will be.
      */}
      <div
        aria-live="polite"
        className="max-h-[30dvh] overflow-y-auto px-2 pt-0.5 pb-1.5 scrollbar-thin lg:max-h-[38dvh]"
      >
        {listening ? (
          <p className="flex items-center gap-2 text-[0.78rem] leading-relaxed text-ink-soft">
            <span
              aria-hidden
              className="pulse-dot size-1.5 shrink-0 rounded-full bg-ink"
            />
            Listening&hellip;
          </p>
        ) : pending ? (
          <div className="flex flex-col gap-2.5 py-1.5">
            <span className="shimmer h-2.5 w-full rounded-full" />
            <span className="shimmer h-2.5 w-3/5 rounded-full" />
          </div>
        ) : (
          <p
            lang={exchange?.languageCode ?? undefined}
            className="type-quote text-[0.95rem] leading-relaxed text-ink"
          >
            {heard}
          </p>
        )}
      </div>

      {heard &&
        (exchange?.language || exchange?.timings || exchange?.interrupted) && (
          <div className="flex flex-wrap items-center gap-1.5 px-2 pb-0.5 text-[0.68rem] text-ink-muted">
            {exchange.language && <PanelChip>{exchange.language}</PanelChip>}
            {exchange.timings?.firstAudio != null && (
              <span
                className="tabular-nums"
                title="Silence before the first sound"
              >
                spoke in {(exchange.timings.firstAudio / 1000).toFixed(1)}s
              </span>
            )}
            {exchange.interrupted && <PanelChip>Interrupted</PanelChip>}
          </div>
        )}

      {/* the reply has no text on screen, so its one control has to live here */}
      {speaking && (
        <>
          <PanelRule />
          <div className="flex items-center justify-between gap-3 px-1 pb-0.5">
            <span className="flex items-center gap-2 text-[0.72rem] font-medium tracking-wide text-ink-soft">
              <Radio
                aria-hidden
                className="size-3.5 animate-pulse text-ink-muted"
              />
              Speaking the reply
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onStop}
              className="rounded-full text-ink-muted"
            >
              <Square aria-hidden />
              Stop
            </Button>
          </div>
        </>
      )}
    </aside>
  );
}
