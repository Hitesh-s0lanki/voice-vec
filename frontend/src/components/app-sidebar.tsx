"use client";

import { Show, UserButton } from "@clerk/nextjs";
import { Blocks, Gauge, History, MessagesSquare, Plus, Settings } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

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
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useCompactLayout } from "@/hooks/use-media-query";
import { useConversation } from "@/lib/conversation";

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
    // The widest of them, and the only one that has to hold a tool call's
    // arguments and its returned page as wrapped JSON. At `w-88` a stored
    // result was a 300px column of two-word lines.
    width: "w-[30rem]",
  },
  {
    id: "connectors",
    label: "Connectors",
    icon: Blocks,
    panel: <IntegrationPanel />,
    // The widest of them, and the one with the most to say: three groups of
    // services, and under Composio a tool list where every row carries a name,
    // a Composio slug and a line of description. Measured at `w-80` those
    // descriptions had 174px and were being cut mid-word.
    width: "w-[26rem]",
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
 *
 * From `lg` up it parks in the bottom-right corner, out of the way of the
 * transcript and activity cards in the other two. Below `lg` those cards are
 * gone from the corners and the single column owns the middle of the screen,
 * so the rail centres itself under it — a thumb reaches the middle of the
 * bottom edge far more easily than it reaches a corner.
 */
export function AppSidebar() {
  // Which wrapper the panels get: an anchored popover, or a centred sheet.
  const compact = useCompactLayout();

  return (
    // `fixed` doubles as the containing block the panel anchors below measure
    // against. The bottom inset clears the home indicator on a notched phone
    // and falls back to the same 20px everything else on this screen uses.
    <div className="glass fixed bottom-[max(1.25rem,env(safe-area-inset-bottom))] left-1/2 z-50 flex max-w-[calc(100vw-1.5rem)] -translate-x-1/2 items-center gap-1.5 rounded-2xl p-1.5 lg:left-auto lg:right-5 lg:translate-x-0">
      {/*
        New chat, first and outside the list: it is the one control here that
        *does* change the page, so it does not belong in a list labelled for
        panels — the same reason the avatar sits outside it on the other end.
        Leading, because it is the thing you reach for without reading.
      */}
      <NewChat />
      <span aria-hidden className="h-6 w-px shrink-0 bg-line" />

      {/* not a <nav> any more — none of these change the page */}
      <ul aria-label="Panels" className="flex flex-row items-center gap-1.5">
        {items.map((item) => (
          <li key={item.id}>
            {compact ? <SheetItem {...item} /> : <PopoverItem {...item} />}
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

/**
 * Leave this conversation for a new one.
 *
 * `reset` clears the turns on screen and puts the address bar back on `/`; the
 * conversation being left is already in Postgres and is one tap away in
 * History. Enabled even on a blank page — "start over" is a thing people press
 * mid-take, and a disabled button there reads as broken rather than as
 * unnecessary.
 */
function NewChat() {
  const { reset } = useConversation();

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-lg"
          onClick={reset}
          className="shrink-0 rounded-xl text-ink-muted hover:text-ink"
        >
          <Plus className="size-[1.15rem]" aria-hidden />
          <span className="sr-only">New chat</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={10}>
        New chat
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * The button itself, shared by both wrappers.
 *
 * Whatever opens it — Popover or Dialog — writes `data-state` onto this
 * button, which is what the ghost variant's `.glass-row` and the underline
 * below both read. Neither wrapper needs its own styling as a result.
 */
function RailButton({
  icon: Icon,
  label,
  ...props
}: ComponentProps<typeof Button> & { icon: LucideIcon; label: string }) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-lg"
      /*
        No hover or open utilities here: the ghost variant carries
        `.glass-row`, which already reads the `data-state` Radix writes onto
        this button. Adding `data-[state=open]:bg-*` would be dead weight —
        the glass classes are unlayered and outrank utilities.
      */
      className="relative rounded-xl text-ink-muted hover:text-ink"
      {...props}
    >
      <Icon className="size-[1.15rem]" aria-hidden />
      <span className="sr-only">{label}</span>
      {/* underline that marks which panel is up */}
      <span
        aria-hidden
        className="absolute -bottom-1.5 left-1/2 h-0.75 w-4 origin-center -translate-x-1/2 scale-x-0 rounded-full bg-ink transition-transform duration-200 group-data-[state=open]/button:scale-x-100"
      />
    </Button>
  );
}

/** Wide screens: the panel hangs off the rail it was opened from. */
function PopoverItem({ label, icon, panel, width }: RailItem) {
  return (
    <Popover>
      <Tooltip>
        {/*
          PopoverTrigger has to sit outside TooltipTrigger: both write
          `data-state` onto the same button, and the outer one's value is the
          one that survives the merge. The open styling reads it, so the
          popover has to be the one that owns it.
        */}
        <PopoverTrigger asChild>
          <TooltipTrigger asChild>
            <RailButton icon={icon} label={label} />
          </TooltipTrigger>
        </PopoverTrigger>
        <TooltipContent side="top" sideOffset={10}>
          {label}
        </TooltipContent>
      </Tooltip>

      {/*
        Panels line up with the rail's right edge, not with the button that
        opened them — otherwise `align="end"` hangs the leftmost panels out to
        the left of the rail. `-inset-px` walks the anchor back out over the
        rail's 1px border so the two glass edges meet exactly (the <li> is
        static, so this resolves against the fixed rail above).

        It has to sit *after* the trigger. Both register as the popper anchor
        through a ref callback and the last one to attach wins; the trigger
        only stops registering itself once it re-renders having seen a custom
        anchor, and the callback ignores its own `null` on the way out. Anchor
        first and the popover stays pinned to a detached button, which measures
        0×0 — every panel lands in the top-left corner of the viewport.
      */}
      <PopoverAnchor asChild>
        <span aria-hidden className="pointer-events-none absolute -inset-px" />
      </PopoverAnchor>

      <PopoverContent
        side="top"
        align="end"
        sideOffset={12}
        collisionPadding={20}
        aria-label={label}
        /*
          The ceiling, not the height — a panel is sized by its content and
          only the tall ones ever reach this. It is generous because the two
          that do reach it are threads and lists worth scrolling; `100dvh` is
          what keeps it off the top of a short window, and the panels that
          scroll inside subtract further so their own box never gets clipped.
        */
        className={`max-h-[min(38rem,calc(100dvh-7rem))] gap-2 overflow-hidden rounded-2xl p-2 text-ink ${width}`}
      >
        {panel}
      </PopoverContent>
    </Popover>
  );
}

/**
 * Phone and tablet: the panel takes the centre of the screen.
 *
 * Anchoring is pointless here — the widest panel (`w-88`, 352px) is wider
 * than a 360px phone has room for beside its margins, so an anchored popover
 * would be shoved into the middle by collision detection anyway, minus the
 * scrim and the swipe-free dismissal a dialog brings. This makes that the
 * intent rather than the fallback, and caps the width at the viewport.
 */
function SheetItem({ label, icon, panel }: RailItem) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <RailButton icon={icon} label={label} />
      </SheetTrigger>

      <SheetContent side="center" title={label}>
        {panel}
      </SheetContent>
    </Sheet>
  );
}
