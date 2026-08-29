"use client";

import { useEffect, useRef } from "react";
import { Check, ChevronDown, MessagesSquare, Wrench, X } from "lucide-react";

import {
  PanelChip,
  PanelEmpty,
  PanelHeading,
  PanelRule,
} from "@/components/panels/panel";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useConversation, type Turn } from "@/lib/conversation";
import type { ToolCall } from "@/lib/conversations";
import { languageName } from "@/lib/languages";
import { relativeTime } from "@/lib/time";

/**
 * Radix lays the viewport's content out as a `display: table`, which sizes to
 * its widest child — and a tool call's result is a `pre` that does not wrap.
 * Left alone, one long JSON line stretches the whole thread to its width and
 * every bubble in it goes along. Forcing that wrapper to `block` puts the
 * panel's own width back in charge, so the overflow lands where it belongs:
 * inside the one box holding the JSON.
 */
const BLOCK_VIEWPORT = "[&_[data-slot=scroll-area-viewport]>div]:!block";

/**
 * The threaded read: what was heard, and what came back. Turns run oldest to
 * newest and the view opens pinned to the bottom, the way a chat log reads.
 *
 * On `/c/{id}` this is the stored thread, read back out of Postgres — which is
 * the whole point of the answer never being printed on the stage. It was
 * heard; this is where it can be re-read, tomorrow, on another device.
 */
export function ConversationsPanel() {
  const { turns, loading } = useConversation();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // ScrollArea keeps the scrolling element one level down, behind the
    // custom scrollbar — reach it by slot rather than threading a second ref.
    const viewport = scrollRef.current?.querySelector<HTMLElement>(
      '[data-slot="scroll-area-viewport"]',
    );
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [turns.length]);

  return (
    <>
      <PanelHeading
        title="Conversations"
        hint="Each take, paired with the agent's reply."
      />

      <PanelRule />

      {loading ? (
        <p className="px-3 py-7 text-center text-[0.75rem] text-ink-muted">Loading…</p>
      ) : turns.length === 0 ? (
        <PanelEmpty icon={MessagesSquare}>
          No exchanges yet. Speak once and the thread starts here.
        </PanelEmpty>
      ) : (
        /*
          Tied to the window rather than fixed, and kept a clear `7rem` inside
          the popover's own ceiling — the difference covers the heading, the
          rule and the padding, so the panel stops growing before it can be
          clipped by the box around it. Not `flex-1`: the popover is sized by
          its content, and a `basis-0` scroll container in an auto-height
          column collapses to nothing.
        */
        <ScrollArea
          ref={scrollRef}
          className={`max-h-[min(32rem,calc(100dvh-14rem))] ${BLOCK_VIEWPORT}`}
        >
          <div className="flex flex-col gap-4 py-1 pr-2">
            {turns.map((turn) => {
              const language = languageName(turn.languageCode);

              return (
                <article key={turn.id} className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 px-1 text-[0.66rem] text-ink-muted">
                    <span className="tabular-nums">
                      {relativeTime(turn.at)}
                    </span>
                    {language && <PanelChip>{language}</PanelChip>}
                  </div>

                  <Bubble speaker="You">{turn.text}</Bubble>

                  {/* Between the question and the answer, because that is
                      when it happened — the agent acted, then spoke. */}
                  <Tools calls={turn.tools} />

                  <Reply turn={turn} />
                </article>
              );
            })}
          </div>
        </ScrollArea>
      )}
    </>
  );
}

function Bubble({
  speaker,
  agent = false,
  children,
}: {
  speaker: string;
  agent?: boolean;
  children: string;
}) {
  return (
    <div
      /*
        The two sides of the conversation are the two sides of the glass
        system: the agent gets the light surface everything else in the panel
        is made of, and the speaker gets the ink one. `.glass-ink` rather than
        `.glass-dark` because a long thread renders one of these per turn, and
        a `backdrop-filter` on each would be a compositor layer per turn.
      */
      className={
        agent
          ? "glass-tile self-start rounded-xl rounded-bl-sm px-3 py-2"
          : "glass-ink max-w-[88%] self-end rounded-xl rounded-br-sm px-3 py-2"
      }
    >
      <p
        className={
          agent
            ? "text-[0.66rem] font-medium tracking-wide text-ink-muted"
            : "text-[0.66rem] font-medium tracking-wide text-canvas/60"
        }
      >
        {speaker}
      </p>
      <p
        className={
          agent
            ? "mt-1 text-[0.8rem] leading-relaxed text-ink-soft"
            : "mt-1 text-[0.8rem] leading-relaxed text-canvas"
        }
      >
        {children}
      </p>
    </div>
  );
}

/**
 * What the agent actually ran, under the turn that caused it — openable, so
 * the call can be read the way an API call is read: what went in, what came
 * back, how long it took.
 *
 * The one part of a conversation with an effect outside this app: a message
 * can be re-read, an email is sent. So it is shown rather than left to be
 * inferred from a reply that mentions it — and shown even when it failed,
 * because "it tried and could not" is the thing worth knowing.
 *
 * Closed by default. A thread is a conversation first; the arguments and the
 * returned page are the audit read, and they are one click away rather than
 * between every question and its answer.
 */
function Tools({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;

  return (
    <ul className="flex w-full flex-col gap-1 self-start">
      {calls.map((call) => (
        <li key={call.id}>
          <Tool call={call} />
        </li>
      ))}
    </ul>
  );
}

function Tool({ call }: { call: ToolCall }) {
  return (
    <details className="glass-dashed group rounded-lg">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-2.5 py-1.5 [&::-webkit-details-marker]:hidden">
        <Wrench aria-hidden className="size-3 shrink-0 text-ink-muted" />
        <span className="font-mono text-[0.66rem] text-ink-soft">{call.slug}</span>

        {call.latencyMs !== null && (
          <span className="shrink-0 text-[0.62rem] whitespace-nowrap tabular-nums text-ink-muted">
            {Math.round(call.latencyMs)} ms
          </span>
        )}

        {/* The failure reason is the provider's own string and can be long;
            it truncates here and stays whole in the title and in the box
            below, rather than folding the row onto two lines. */}
        <span
          className="ml-auto flex min-w-0 items-center gap-1 text-[0.62rem] text-ink-muted"
          title={call.error ?? undefined}
        >
          {call.ok ? (
            <Check aria-hidden className="size-2.5 shrink-0" />
          ) : (
            <X aria-hidden className="size-2.5 shrink-0 text-destructive" />
          )}
          <span className="truncate">
            {call.ok ? "ran" : (call.error ?? "failed")}
          </span>
        </span>

        <ChevronDown
          aria-hidden
          className="size-3 shrink-0 text-ink-muted transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="flex flex-col gap-2 px-2.5 pb-2.5 pt-0.5">
        <Field label="Input" empty="Called with no arguments.">
          {pretty(call.arguments)}
        </Field>

        {call.ok ? (
          <Field
            label="Output"
            note={size(call.resultBytes, call.result)}
            empty="The tool returned nothing."
          >
            {readable(call.result)}
          </Field>
        ) : (
          <Field label="Error" empty="It failed without saying why.">
            {call.error}
          </Field>
        )}
      </div>
    </details>
  );
}

/**
 * One half of a call. Recessed rather than raised — this is content that came
 * from somewhere else, and `.glass-field` is the surface that goes *in*.
 *
 * Wraps rather than scrolling sideways, and caps its height. A stored result
 * is one long JSON line — 24,000px of it, measured — and in a 300px rail a
 * horizontal scrollbar is a worse way to read that than a wrapped block with a
 * vertical one. Indentation survives, because `pre-wrap` keeps the newlines.
 */
function Field({
  label,
  note,
  empty,
  children,
}: {
  label: string;
  note?: string | null;
  empty?: string;
  children: string | null;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <p className="text-[0.62rem] font-medium tracking-wide text-ink-muted uppercase">
          {label}
        </p>
        {note && (
          <span className="text-[0.6rem] tabular-nums text-ink-muted">{note}</span>
        )}
      </div>

      {children ? (
        <pre className="glass-field max-h-64 overflow-y-auto rounded-md px-2 py-1.5 font-mono text-[0.64rem] leading-relaxed break-words whitespace-pre-wrap text-ink-soft">
          {children}
        </pre>
      ) : (
        <p className="text-[0.66rem] text-ink-muted">{empty ?? "Nothing."}</p>
      )}
    </div>
  );
}

/** Arguments as the agent sent them. `{}` is a real answer — it called with none. */
function pretty(value: Record<string, unknown>): string | null {
  if (!value || Object.keys(value).length === 0) return null;

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/**
 * The stored preview, indented if it parses.
 *
 * It usually will — tool results are rendered as JSON before they are stored —
 * but a truncated one has "… (truncated)" hung off the end and cannot parse.
 * That case falls through to the raw text, which is the point of keeping the
 * marker: a cut result still reads as a cut result rather than as an error.
 */
function readable(result: string | null): string | null {
  const text = result?.trim();
  if (!text) return null;

  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/**
 * How much of the result is on screen. `resultBytes` is the size of the whole
 * of it, so a preview shorter than that says so — otherwise a truncated inbox
 * page reads as the entire inbox.
 */
function size(bytes: number | null, result: string | null): string | null {
  if (bytes === null || bytes <= 0) return null;

  const whole = bytes < 1024 ? `${bytes} chars` : `${(bytes / 1024).toFixed(1)}k chars`;
  const shown = result?.length ?? 0;
  return shown > 0 && shown < bytes ? `first ${shown} of ${whole}` : whole;
}

/** What a turn that produced no words says instead. */
const NO_ANSWER: Record<string, string> = {
  refused: "Declined that one.",
  abstained: "Nothing in my sources covers that.",
  interrupted: "Stopped before it said anything.",
  error: "Something went wrong answering that one.",
};

/**
 * The agent's side of a turn. Not every outcome is text: an abstention or a
 * refusal is a real reply with no answer in it, so it gets a bubble of its own
 * rather than being left to look like a loading state.
 *
 * A turn you talked over is the odd one — it has words *and* an ending worth
 * marking, because what is stored is only what actually reached the speakers.
 */
function Reply({ turn }: { turn: Turn }) {
  if (turn.reply) {
    return (
      <div className="flex flex-col items-start gap-1">
        <Bubble speaker="Vec" agent>
          {turn.reply}
        </Bubble>
        {turn.replyStatus === "interrupted" && (
          <p className="px-1 text-[0.66rem] text-ink-muted">
            Stopped here — you started talking.
          </p>
        )}
      </div>
    );
  }

  if (turn.replyStatus) {
    return (
      <div className="glass-dashed self-start rounded-xl rounded-bl-sm px-3 py-2">
        <p className="text-[0.72rem] leading-relaxed text-ink-muted">
          {turn.replyReason ?? NO_ANSWER[turn.replyStatus] ?? "No answer to that one."}
        </p>
      </div>
    );
  }

  return (
    <div className="glass-dashed self-start rounded-xl rounded-bl-sm px-3 py-2">
      <p className="text-[0.72rem] leading-relaxed text-ink-muted">
        No reply — this take was transcribed only.
      </p>
    </div>
  );
}
