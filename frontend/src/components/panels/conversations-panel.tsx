"use client";

import { useEffect, useRef } from "react";
import { Check, MessagesSquare, Wrench, X } from "lucide-react";

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
        <ScrollArea ref={scrollRef} className="max-h-80">
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
 * What the agent actually ran, under the turn that caused it.
 *
 * The one part of a conversation with an effect outside this app: a message
 * can be re-read, an email is sent. So it is shown rather than left to be
 * inferred from a reply that mentions it — and shown even when it failed,
 * because "it tried and could not" is the thing worth knowing.
 *
 * No results. The backend keeps a result's size and never its content, so
 * there is nothing here that could put somebody's inbox on screen.
 */
function Tools({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;

  return (
    <ul className="flex flex-col gap-1 self-start">
      {calls.map((call) => (
        <li
          key={call.id}
          className="glass-dashed flex items-center gap-2 rounded-lg px-2.5 py-1.5"
        >
          <Wrench aria-hidden className="size-3 shrink-0 text-ink-muted" />
          <span className="font-mono text-[0.66rem] text-ink-soft">{call.slug}</span>

          {call.latencyMs !== null && (
            <span className="text-[0.62rem] tabular-nums text-ink-muted">
              {Math.round(call.latencyMs)} ms
            </span>
          )}

          <span
            className="ml-auto flex shrink-0 items-center gap-1 text-[0.62rem] text-ink-muted"
            title={call.error ?? undefined}
          >
            {call.ok ? (
              <Check aria-hidden className="size-2.5" />
            ) : (
              <X aria-hidden className="size-2.5 text-destructive" />
            )}
            {call.ok ? "ran" : (call.error ?? "failed")}
          </span>
        </li>
      ))}
    </ul>
  );
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
