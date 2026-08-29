# 19 — Responsive

*One breakpoint, three pieces of furniture, and the orb that must not move.*

The stage was built for a wide screen and made to survive a narrow one. The orb sits on the
viewport's centre line; the transcript, the activity log and the rail sit in the three free
corners around it, all `fixed`, all costing the centred stack no height. That arrangement
needs about a thousand pixels of width to hold. Below that, two of the three corners stop
being free — a 320px card pinned to the left of a 768px screen lands on an orb that is
304px wide and centred — and the layout has to become something else.

## The line is `lg`, not `md`

Everything turns on one breakpoint, and it is width-based: **`lg` (1024px)**. Above it, the
corner layout. Below it — phone *and* tablet — a single centred column.

| | below `lg` | `lg` and up |
| --- | --- | --- |
| Rail | centred on the bottom edge | bottom-right corner |
| Rail panels | centred sheet, viewport-capped | popover anchored to the rail |
| Activity log | drawer off the right edge, behind a button | card in the top-right corner |
| Transcript | in the column, under the orb | card in the bottom-left corner |
| Openers | in the column | in the column |

It used to be `md` (768px), which put a tablet on the desktop side of the line. It is not:
at 768px the centre column (`max-w-xl`, 576px) already reaches within 96px of both edges,
so both corner cards land on it. `lg` is the first width where the three corners are
genuinely free.

## Why the panels centre rather than anchor

The widest rail panel is `w-88` — 352px. A 360px phone cannot hold it beside its margins,
so an anchored popover would be shoved into the middle by Radix's collision detection
anyway. Below `lg` the rail is centred on the bottom edge and there is no corner left to
anchor to either.

So the panels open as a **centred dialog** ([sheet.tsx](frontend/src/components/ui/sheet.tsx)),
which makes that the intent rather than the fallback and brings the two things a shoved
popover would not: a scrim, and a close affordance that does not depend on knowing you can
tap outside. The surface is the same `.glass-panel` and the contents are the same
components — only the geometry differs.

Which wrapper a panel gets is decided in JavaScript, not CSS, by
[`useCompactLayout`](frontend/src/hooks/use-media-query.ts). Rendering both and hiding one
with a utility would mount every panel twice, and these panels fetch on mount. The hook is
a `useSyncExternalStore` whose server snapshot is `false` — there is no viewport during
prerender, so the markup React sends is the wide one and the client corrects it before
paint. Nothing on screen is keyed to the value at rest, so the correction is invisible.

## Activity moves behind a drawer

On a phone the step log is the one piece of furniture that *cannot* stay where it is: it
grows downward as the pipeline runs, and there is nothing below the top-right corner except
the orb. Below `lg` it collapses to a single glass button in that corner which opens a
drawer off the right edge, and the button carries the live pulse dot so a running pipeline
is still visible from the stage with the drawer shut.

The card and the button are mutually exclusive by CSS (`hidden lg:flex` against `lg:hidden`),
so only one is ever in the accessibility tree, and Radix only mounts the drawer's contents
while it is open — there is never a second `aria-live` region.

## The bottom band, and the two bugs in it

The stage's bottom padding is what it owes the fixed furniture below it, and getting it
wrong is what put the opener pills on top of the footer credit line on a 667px phone.

**The floor that could not fit.** The bottom grid row was `minmax(min(13rem,30dvh), 1fr)`.
On a 375×667 screen, once the padding, the gaps and the orb had taken theirs, 189px were
left — and the row's floor asked for 200. A grid row whose floor exceeds the space
available does not shrink; the grid overflows its own container, straight down through the
bottom padding and into the footer's band.

The fix is to stop stating a floor at all below `lg`: `minmax(0, 1fr)`. The row is a
*weight* now, so it takes what is left and the column inside it scrolls when that is not
enough. The stage cannot outgrow its box at any height. The orb still cannot move — both
outer rows are `fr`, so their heights come from the viewport, never from what the column is
holding; the `0.55` weight on the top row is what keeps the orb a little above centre,
where the row below needs the room.

**The padding that lost a cascade fight.** `sm:py-12` and `max-lg:pb-32` set the same
property from two variants, and which one won was decided by the emitted order rather than
by intent — `sm:py-12` was quietly taking every tablet's bottom padding back to 48px, so
the rail sat over the stage's own content box. Splitting the shorthand into explicit `pt-*`
and `pb-*` removes the conflict: `pb-30` below `lg` (the centred rail with the footer line
above it), `lg:pb-12` from `lg` (neither — the rail is back in its corner).

**Four openers did not fit; three do.** After both fixes above the fourth opener pill was
still clipping mid-word on a 667px screen, which reads as broken rather than as scrollable.
The openers dropped to Hindi, Tamil and English — a content decision made for reach rather
than for layout, since an opener in a script the reader cannot read proves nothing to them
— and that returned about 35px, enough for the credit line to keep its place. So the bottom
padding is one value below `lg` (`pb-30`, 120px: the rail plus the line above it), and on a
375×667 screen the last opener now clears the line by 37px.

## The one height breakpoint

A phone held sideways is 852×393. Every width-based breakpoint calls that a small laptop,
and the stage it lays out is 544px tall in a 393px viewport.

`@custom-variant short (@media (max-height: 34rem))` in
[globals.css](frontend/src/app/globals.css) is the answer. Under it the stage drops its
minimum height, halves its padding, and gives up the language openers — which are an
onboarding nicety, not a control. That, plus the floorless bottom row above, is what gets
the whole stage inside a sideways phone with no page scroll at all.

The orb is capped the same way: `w-[min(62vw, 42dvh, 19rem)]`. The `dvh` term is the one
that matters on a short screen — without it the orb is sized off width alone and a
landscape phone gets a 528px circle in a 393px viewport. The `vw` term is what leaves the
openers room on a small phone; it only binds below about 490px, so tablets and desktops
still get the full 19rem.

## The invariant

None of the above is allowed to move the orb. The stage is a three-row grid whose outer two
rows are sized in `fr` — their heights come from the viewport, not from what they hold — so
the pill above and the column below get their share whether or not anything is in them. The
column itself is `min-h-0 overflow-y-auto`, so a long transcript scrolls inside it rather
than pushing anything. Above `lg`, where there is height to spare, those rows keep real
`7.5rem` floors and the two corner cards are `fixed` besides.

Measured against the running dev server across ten viewports from 360×640 to 1920×1080,
including landscape phone and both iPad orientations: no horizontal scroll anywhere, no
overlap between any two pieces of furniture, no opener clipped mid-pill, and **the orb's
`y` does not change by a single pixel** when a ten-line transcript lands in the column
beneath it.

One note on how that is checked, because the first pass missed the footer collision
entirely: the overlap test has to compare *painted* boxes, not flow boxes. The column
clips, so a pill's `getBoundingClientRect()` can sit well below what the eye sees. Every
box measured inside a scroll container is intersected with that container first.
