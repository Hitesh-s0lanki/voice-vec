"use client"

import * as React from "react"
import { X } from "lucide-react"
import { Dialog as DialogPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

/**
 * The small-screen counterpart to the rail popovers.
 *
 * A popover is anchored to the thing that opened it, which is exactly what
 * you want when the rail is parked in a corner of a wide screen. Below `lg`
 * the rail sits at the bottom *centre* and the panels are as wide as the
 * viewport allows, so there is no corner left to anchor to — the panel simply
 * takes the middle of the screen (`side="center"`) or slides in from the edge
 * as a drawer (`side="right"`). Same glass, same contents, different geometry.
 */
function Sheet(props: React.ComponentProps<typeof DialogPrimitive.Root>) {
  return <DialogPrimitive.Root data-slot="sheet" {...props} />
}

function SheetTrigger(
  props: React.ComponentProps<typeof DialogPrimitive.Trigger>,
) {
  return <DialogPrimitive.Trigger data-slot="sheet-trigger" {...props} />
}

function SheetClose(props: React.ComponentProps<typeof DialogPrimitive.Close>) {
  return <DialogPrimitive.Close data-slot="sheet-close" {...props} />
}

function SheetOverlay({
  className,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Overlay>) {
  return (
    <DialogPrimitive.Overlay
      data-slot="sheet-overlay"
      className={cn(
        // Just enough scrim to lift the panel off the stage. The blur matters
        // more than the tint here: the room behind is near-white, and a wash
        // that dark enough to separate the two would fight the glass.
        "fixed inset-0 z-50 bg-ink/15 backdrop-blur-[2px] data-closed:animate-out data-closed:fade-out-0 data-open:animate-in data-open:fade-in-0",
        className,
      )}
      {...props}
    />
  )
}

const sides = {
  center:
    "fixed top-1/2 left-1/2 z-50 max-h-[calc(100dvh-3rem)] w-[calc(100vw-2rem)] max-w-sm -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-2xl data-closed:zoom-out-95 data-open:zoom-in-95 data-closed:slide-out-to-bottom-2 data-open:slide-in-from-bottom-2",
  right:
    "fixed inset-y-0 right-0 z-50 h-dvh w-[min(22rem,calc(100vw-2.5rem))] overflow-hidden rounded-l-2xl data-closed:slide-out-to-right data-open:slide-in-from-right",
  bottom:
    "fixed inset-x-0 bottom-0 z-50 max-h-[85dvh] overflow-hidden rounded-t-2xl data-closed:slide-out-to-bottom data-open:slide-in-from-bottom",
} as const

function SheetContent({
  className,
  children,
  side = "center",
  title,
  description,
  showClose = true,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & {
  side?: keyof typeof sides
  /** Required by Radix for the dialog's accessible name; rendered off-screen. */
  title: string
  description?: string
  showClose?: boolean
}) {
  return (
    <DialogPrimitive.Portal>
      <SheetOverlay />
      <DialogPrimitive.Content
        data-slot="sheet-content"
        data-side={side}
        className={cn(
          "glass-panel flex flex-col gap-2 p-2 text-ink ring-0 outline-hidden duration-150 data-closed:animate-out data-closed:fade-out-0 data-open:animate-in data-open:fade-in-0",
          sides[side],
          // Panels bring their own heading, and some of them park an action in
          // its trailing slot. Reserving the corner here is what keeps the
          // close button from landing on top of one.
          showClose && "**:data-[slot=panel-heading]:pr-9",
          className,
        )}
        {...props}
      >
        <DialogPrimitive.Title className="sr-only">
          {title}
        </DialogPrimitive.Title>
        {description ? (
          <DialogPrimitive.Description className="sr-only">
            {description}
          </DialogPrimitive.Description>
        ) : (
          /* Radix warns about a missing description unless it is opted out of
             explicitly — these panels carry their own hint line instead. */
          <DialogPrimitive.Description asChild>
            <span hidden />
          </DialogPrimitive.Description>
        )}

        {children}

        {showClose && (
          <DialogPrimitive.Close
            className="glass-row absolute top-2.5 right-2.5 grid size-7 place-items-center rounded-lg text-ink-muted outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
            aria-label="Close"
          >
            <X aria-hidden className="size-3.5" />
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export { Sheet, SheetClose, SheetContent, SheetOverlay, SheetTrigger }
