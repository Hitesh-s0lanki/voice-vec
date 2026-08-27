"use client";

import { MessagesSquare, Plus, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
  PanelChip,
  PanelEmpty,
  PanelHeading,
  PanelRule,
} from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useConversation } from "@/lib/conversation";
import {
  deleteConversation,
  listConversations,
  type ConversationSummary,
} from "@/lib/conversations";
import { languageName } from "@/lib/languages";
import { relativeTime } from "@/lib/time";

/**
 * Every conversation this browser has had, newest first.
 *
 * The list comes from Postgres rather than from this device, so it survives a
 * reload and a closed tab. It is fetched when the panel opens — which is what
 * mounting means for a popover — so there is nothing to keep in sync while it
 * is shut.
 *
 * Conversations shows the turns *inside* the one on screen; this is how you
 * get to a different one.
 */
export function HistoryPanel() {
  const router = useRouter();
  const { conversationId, reset } = useConversation();

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const abort = new AbortController();

    void listConversations(abort.signal).then((rows) => {
      if (abort.signal.aborted) return;
      setConversations(rows);
      setLoading(false);
    });

    return () => abort.abort();
  }, []);

  const open = useCallback(
    (id: string) => {
      if (id === conversationId) return;
      // A real navigation, unlike `adopt`: a different conversation needs a
      // different socket, because the history behind it is different.
      router.push(`/c/${id}`);
    },
    [conversationId, router],
  );

  const remove = useCallback(
    async (id: string) => {
      // Dropped from the list first — the round trip to Neon is ~80 ms and a
      // row that lingers after a click reads as a failed delete.
      setConversations((previous) => previous.filter((row) => row.id !== id));
      const gone = await deleteConversation(id);

      if (!gone) {
        setConversations(await listConversations());
        return;
      }
      // Standing in a conversation that no longer exists — leave for a new one.
      if (id === conversationId) reset();
    },
    [conversationId, reset],
  );

  return (
    <>
      <PanelHeading
        title="History"
        hint={
          loading
            ? undefined
            : conversations.length
              ? `${conversations.length} ${conversations.length === 1 ? "conversation" : "conversations"}`
              : undefined
        }
      >
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={reset}
          disabled={conversationId === null}
          className="-mt-0.5 rounded-full px-2 text-ink-muted hover:text-ink"
        >
          <Plus aria-hidden />
          New
        </Button>
      </PanelHeading>

      <PanelRule />

      {loading ? (
        <p className="px-3 py-7 text-center text-[0.75rem] text-ink-muted">Loading…</p>
      ) : conversations.length === 0 ? (
        <PanelEmpty icon={MessagesSquare}>
          Nothing saved yet. Speak once and the conversation shows up here.
        </PanelEmpty>
      ) : (
        <ScrollArea className="max-h-72">
          <ul className="flex flex-col gap-0.5 pr-2">
            {conversations.map((conversation) => {
              const language = languageName(conversation.language);
              const current = conversation.id === conversationId;

              return (
                <li key={conversation.id} className="group/row relative">
                  <button
                    type="button"
                    onClick={() => open(conversation.id)}
                    aria-current={current ? "page" : undefined}
                    /*
                      `.glass-row` reads `aria-current` for the open
                      conversation, so the row it is standing in holds the
                      selected ground on its own — no second class, and no way
                      for the two to disagree.
                    */
                    className="glass-row flex w-full flex-col gap-1.5 rounded-lg px-2 py-2 pr-8 text-left"
                  >
                    <div className="flex items-center gap-2 text-[0.68rem] text-ink-muted">
                      <span className="tabular-nums">
                        {relativeTime(Date.parse(conversation.updatedAt))}
                      </span>
                      {language && <PanelChip>{language}</PanelChip>}
                      <span className="tabular-nums">
                        {conversation.turns} {conversation.turns === 1 ? "take" : "takes"}
                      </span>
                    </div>
                    <p className="line-clamp-2 text-[0.8rem] leading-relaxed text-ink-soft">
                      {conversation.title ?? "Untitled"}
                    </p>
                  </button>

                  {/* Kept out of the row's own button — a delete nested inside
                      a navigation is one mis-click from opening what it just
                      removed. */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => void remove(conversation.id)}
                    className="glass-row-danger absolute top-1.5 right-1 rounded-full text-ink-muted opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100"
                  >
                    <Trash2 aria-hidden />
                    <span className="sr-only">Delete this conversation</span>
                  </Button>
                </li>
              );
            })}
          </ul>
        </ScrollArea>
      )}
    </>
  );
}
