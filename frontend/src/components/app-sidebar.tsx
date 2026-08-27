"use client";

import { Show, UserButton } from "@clerk/nextjs";
import { Blocks, Gauge, History, MessagesSquare, Settings } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { ConversationsPanel } from "@/components/panels/conversations-panel";
import { EffortPanel } from "@/components/panels/effort-panel";
import { HistoryPanel } from "@/components/panels/history-panel";
import { IntegrationPanel } from "@/components/panels/integration-panel";
import { SettingsPanel } from "@/components/panels/settings-panel";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type RailItem = {
  id: string;
  label: string;
  icon: LucideIcon;
  panel: ReactNode;
  /** panels carry different amounts of content — size each to what it holds */
  width: string;
};

const items: RailItem[] = [
  {
    id: "history",
    label: "History",
    icon: History,
    panel: <HistoryPanel />,
    width: "w-80",
  },
  {
    id: "conversations",
    label: "Conversations",
    icon: MessagesSquare,
    panel: <ConversationsPanel />,
    width: "w-88",
  },
  {
    id: "connectors",
    label: "Connectors",
    icon: Blocks,
    panel: <IntegrationPanel />,
    width: "w-80",
  },
  {
    id: "effort",
    label: "Effort",
    icon: Gauge,
    panel: <EffortPanel />,
    width: "w-72",
  },
  {
    id: "settings",
    label: "Settings",
    icon: Settings,
    panel: <SettingsPanel />,
    width: "w-72",
  },
];

/**
 * The rail. Every entry opens a panel in place — nothing here navigates, so
 * whatever you came to check never costs you the orb you were talking to.
 */
export function AppSidebar() {
  return (
    // `fixed` doubles as the containing block the panel anchors below measure against
    <div className="glass fixed right-5 bottom-5 z-50 flex items-center gap-1.5 rounded-2xl p-1.5">
      {/* not a <nav> any more — none of these change the page */}
      <ul aria-label="Panels" className="flex flex-row items-center gap-1.5">
        {items.map(({ id, label, icon: Icon, panel, width }) => (
          <li key={id}>
            <Popover>
              <Tooltip>
                {/*
                  PopoverTrigger has to sit outside TooltipTrigger: both write
                  `data-state` onto the same button, and the outer one's value
                  is the one that survives the merge. The open styling below
                  reads it, so the popover has to be the one that owns it.
                */}
                <PopoverTrigger asChild>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-lg"
                      /*
                        No hover or open utilities here: the ghost variant
                        carries `.glass-row`, which already reads the
                        `data-state` Radix writes onto this button. Adding
                        `data-[state=open]:bg-*` would be dead weight — the
                        glass classes are unlayered and outrank utilities.
                      */
                      className="relative rounded-xl text-ink-muted hover:text-ink"
                    >
                      <Icon className="size-[1.15rem]" aria-hidden />
                      <span className="sr-only">{label}</span>
                      {/* underline that marks which panel is up */}
                      <span
                        aria-hidden
                        className="absolute -bottom-1.5 left-1/2 h-0.75 w-4 origin-center -translate-x-1/2 scale-x-0 rounded-full bg-ink transition-transform duration-200 group-data-[state=open]/button:scale-x-100"
                      />
                    </Button>
                  </TooltipTrigger>
                </PopoverTrigger>
                <TooltipContent side="top" sideOffset={10}>
                  {label}
                </TooltipContent>
              </Tooltip>

              {/*
                Panels line up with the rail's right edge, not with the button
                that opened them — otherwise `align="end"` hangs the leftmost
                panels out to the left of the rail. `-inset-px` walks the anchor
                back out over the rail's 1px border so the two glass edges meet
                exactly (the <li> is static, so this resolves against the fixed
                rail above).

                It has to sit *after* the trigger. Both register as the popper
                anchor through a ref callback and the last one to attach wins;
                the trigger only stops registering itself once it re-renders
                having seen a custom anchor, and the callback ignores its own
                `null` on the way out. Anchor first and the popover stays
                pinned to a detached button, which measures 0×0 — every panel
                lands in the top-left corner of the viewport.
              */}
              <PopoverAnchor asChild>
                <span
                  aria-hidden
                  className="pointer-events-none absolute -inset-px"
                />
              </PopoverAnchor>

              <PopoverContent
                side="top"
                align="end"
                sideOffset={12}
                collisionPadding={20}
                aria-label={label}
                className={`max-h-[min(28rem,calc(100dvh-7rem))] gap-2 overflow-hidden rounded-2xl p-2 text-ink ${width}`}
              >
                {panel}
              </PopoverContent>
            </Popover>
          </li>
        ))}
      </ul>

      {/*
        The account, kept out of the <ul> on purpose: it is the one thing in
        the rail that opens something other than a panel, and the list is
        labelled for panels. The rule marks that break rather than letting the
        avatar read as a sixth icon.

        Sized to the same 36px box as the icon buttons so the row keeps one
        baseline — the avatar itself is smaller, the way an avatar should be.
      */}
      <Show when="signed-in">
        <span aria-hidden className="h-6 w-px shrink-0 bg-line" />
        <div className="glass-row flex size-9 shrink-0 items-center justify-center rounded-xl">
          <UserButton
            appearance={{ elements: { avatarBox: "size-7" } }}
            userProfileMode="modal"
          />
        </div>
      </Show>
    </div>
  );
}
