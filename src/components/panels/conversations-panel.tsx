"use client";

import { useEffect, useRef } from "react";
import { MessagesSquare } from "lucide-react";

import {
  PanelChip,
  PanelEmpty,
  PanelHeading,
  PanelRule,
} from "@/components/panels/panel";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useConversation } from "@/lib/conversation";
import { languageName } from "@/lib/languages";
import { relativeTime } from "@/lib/time";

/**
 * The threaded read: what was heard, and what came back. Turns run oldest to
 * newest and the view opens pinned to the bottom, the way a chat log reads.
 */
export function ConversationsPanel() {
  const { turns } = useConversation();
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

      {turns.length === 0 ? (
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

                  {turn.reply ? (
                    <Bubble speaker="Vec" agent>
                      {turn.reply}
                    </Bubble>
                  ) : (
                    <PendingReply />
                  )}
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
      className={
        agent
          ? "self-start rounded-xl rounded-bl-sm border border-line bg-surface-2 px-3 py-2"
          : "max-w-[88%] self-end rounded-xl rounded-br-sm bg-ink px-3 py-2"
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
 * Nothing generates replies yet, so the agent's side of every turn is empty.
 * Say so rather than leaving a gap that reads as a loading state.
 */
function PendingReply() {
  return (
    <div className="self-start rounded-xl rounded-bl-sm border border-dashed border-line px-3 py-2">
      <p className="text-[0.72rem] leading-relaxed text-ink-muted">
        No agent reply — Vec only transcribes for now.
      </p>
    </div>
  );
}
