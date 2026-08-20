"use client";

import { Clock3, Trash2 } from "lucide-react";

import {
  PanelChip,
  PanelEmpty,
  PanelHeading,
  PanelRule,
} from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useConversation } from "@/lib/conversation";
import { languageName } from "@/lib/languages";
import { relativeTime } from "@/lib/time";

/**
 * The flat log: every take, newest first. Conversations shows the same turns
 * threaded with their replies — this one is just "what have I said".
 */
export function HistoryPanel() {
  const { turns, clear } = useConversation();
  const recent = [...turns].reverse();

  return (
    <>
      <PanelHeading
        title="History"
        hint={
          turns.length
            ? `${turns.length} ${turns.length === 1 ? "take" : "takes"} on this device`
            : undefined
        }
      >
        {turns.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={clear}
            className="-mt-0.5 rounded-full text-ink-muted hover:text-ink"
          >
            <Trash2 aria-hidden />
            Clear
          </Button>
        )}
      </PanelHeading>

      <PanelRule />

      {recent.length === 0 ? (
        <PanelEmpty icon={Clock3}>
          Nothing captured yet. Tap the orb and your takes collect here.
        </PanelEmpty>
      ) : (
        <ScrollArea className="max-h-72">
          <ul className="flex flex-col gap-0.5 pr-2">
            {recent.map((turn) => {
              const language = languageName(turn.languageCode);

              return (
                <li
                  key={turn.id}
                  className="flex flex-col gap-1.5 rounded-lg px-2 py-2 transition-colors hover:bg-surface-2"
                >
                  <div className="flex items-center gap-2 text-[0.68rem] text-ink-muted">
                    <span className="tabular-nums">
                      {relativeTime(turn.at)}
                    </span>
                    {language && <PanelChip>{language}</PanelChip>}
                  </div>
                  <p className="line-clamp-2 text-[0.8rem] leading-relaxed text-ink-soft">
                    {turn.text}
                  </p>
                </li>
              );
            })}
          </ul>
        </ScrollArea>
      )}
    </>
  );
}
