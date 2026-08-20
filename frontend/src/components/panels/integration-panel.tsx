"use client";

import { Check, Copy, Globe, Hash, NotebookPen } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";

import {
  PanelChip,
  PanelHeading,
  PanelRule,
} from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { useConversation } from "@/lib/conversation";

type Destination = {
  id: string;
  name: string;
  detail: string;
  icon: LucideIcon;
};

/** Nothing here is wired to a service yet — the rows are the shape, not the wiring. */
const destinations: Destination[] = [
  {
    id: "slack",
    name: "Slack",
    detail: "Post each transcript to a channel",
    icon: Hash,
  },
  {
    id: "notion",
    name: "Notion",
    detail: "Append takes to a page",
    icon: NotebookPen,
  },
  {
    id: "webhook",
    name: "Webhook",
    detail: "POST the raw JSON anywhere",
    icon: Globe,
  },
];

export function IntegrationPanel() {
  const { turns } = useConversation();
  const latest = turns.at(-1);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(timer);
  }, [copied]);

  const copyLatest = async () => {
    if (!latest) return;
    try {
      await navigator.clipboard.writeText(latest.text);
      setCopied(true);
    } catch {
      // clipboard blocked — leave the button in its resting state
    }
  };

  return (
    <>
      <PanelHeading
        title="Integration"
        hint="Where transcripts go once Vec has them."
      />

      <PanelRule />

      <ul className="flex flex-col gap-0.5">
        {destinations.map(({ id, name, detail, icon: Icon }) => (
          <li
            key={id}
            className="flex items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-surface-2"
          >
            <span className="grid size-7 shrink-0 place-items-center rounded-lg border border-line text-ink-muted">
              <Icon aria-hidden className="size-3.5" />
            </span>
            <div className="flex min-w-0 flex-col">
              <p className="text-[0.8rem] font-medium text-ink">{name}</p>
              <p className="truncate text-[0.7rem] text-ink-muted">{detail}</p>
            </div>
            <PanelChip className="ml-auto shrink-0">Soon</PanelChip>
          </li>
        ))}
      </ul>

      <PanelRule />

      {/* the one destination that does work today */}
      <div className="flex items-center justify-between gap-3 px-2 pb-0.5">
        <p className="text-[0.72rem] text-ink-muted">
          {latest ? "Copy the latest take" : "Nothing captured to send yet"}
        </p>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          disabled={!latest}
          onClick={copyLatest}
          className="rounded-full text-ink-muted hover:text-ink"
        >
          {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
    </>
  );
}
