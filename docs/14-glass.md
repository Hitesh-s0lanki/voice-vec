# 14 — Glass

*One surface vocabulary, and the room it needs to be visible in.*

Before this, "glass" was three classes — `.glass`, `.glass-panel`, `.glass-hover` — used on
about eight elements. Everything else was flat: `bg-surface-2` for a hover, `border
border-line` for a box, shadcn's grey ramp for the primitives. Two surfaces sitting next to
each other were drawn by two different systems, and the seam showed.

Now every surface in the app is one of nine classes, all defined in one block of
[globals.css](frontend/src/app/globals.css), and every interactive element gets its hover
ground from one of two.

## The room comes first

Glass over flat white is white. The stage was `--stage: #ffffff` with nothing in it, so a
translucent card had nothing to refract and its edge had nothing to catch. **`--ambient` is
what makes the rest of the file mean anything**: three wide, very low-saturation washes,
painted by both `body` and `.stage`.

```
--wash-cool  rgb(106 124 166 / 0.10)   top-left
--wash-warm  rgb(178 146 122 / 0.075)  top-right
--wash-mint  rgb(118 154 146 / 0.065)  bottom-centre
```

**This is the dial.** Turn the three alphas up and every surface reads more like glass; take
them to zero and the whole system collapses back into white-on-white. The palette is still
the "plain white room" the theme was written around — these are near-greys with a whisper
of hue, not a colour scheme.

The grain layered on top of them is not decoration. Washes that wide and that shallow are
the exact shape that bands into visible rings on an 8-bit display; the noise dithers them.

Both copies use `background-attachment: fixed`, which is what keeps the two stacks aligned
to the same viewport origin so the boundary between `body` and `.stage` is invisible.

### Why `.stage` paints its own copy

`.stage` has `isolation: isolate`, which makes it a **backdrop root**. A `backdrop-filter`
anywhere inside it samples only what is painted *within* the stage — so if the stage were
transparent and let the body's ambient show through, every glass surface on the main screen
would blur an empty backdrop and render as nothing. Hence the duplicate declaration.

This is also why the sign-in and sign-up pages became `<main className="stage">` rather than
a bare `<div>`: Clerk's card has its own blur and needs a backdrop root with something in it.

## The nine surfaces

| class | what it is | blur? |
|---|---|---|
| `.glass` | floating card, pill, rail — sits over the ambient | ✅ |
| `.glass-panel` | popover full of list text — near-opaque, heavily blurred | ✅ |
| `.glass-dark` | the ink side: tooltips | ✅ |
| `.glass-lens` | the orb's face | ❌ |
| `.glass-ink` | `.glass-dark` without the blur, for many-at-once | ❌ |
| `.glass-tile` | icon square, logo box, citation, a small pane | ❌ |
| `.glass-chip` | state pill — language, tier, connection status | ❌ |
| `.glass-field` | an input, pressed *into* the surface rather than onto it | ❌ |
| `.glass-track` | a groove — slider rail, confidence meter | ❌ |

Plus three modifiers: `.glass-danger` (border only), `.glass-alert` (destructive banner),
`.glass-dashed` (an outline meaning *nothing here*), and `.glass-solid` (the one opaque
surface left — a primary button, which is what the glass sits *against*).

Each is built from the same four ingredients: a translucent white **tint**, a 1px inset
**sheen** along the top edge, an **edge** (border plus a second inset ring just inside it,
so it reads as a thickness rather than a drawn line), and an off-axis **facet** gradient
across the face — the reason two glass cards side by side don't look like one flat sheet.

### Only three of them blur

`backdrop-filter` costs a compositor layer per element. A row inside a panel is already
sitting on blurred pixels, so blurring them again buys nothing and reads as mud. Only
surfaces that actually *float* over the room get it.

`.glass-lens` is the pointed exception. The orb is ~300px across and its scale is driven off
the analyser at 110ms for the whole of every take — a real blur would be re-rasterised every
animation frame. Its convex read comes from shadows instead: a bright band along the top
edge, a soft occlusion along the bottom.

## Hover: one gesture, two directions

There are exactly two hover classes, and which one applies is decided by **what is
underneath**, not by what kind of element it is.

```
.glass-hover   floating over the ambient  →  brightens towards opaque white, lifts 1px
.glass-row     inside a near-white panel  →  darkens by an ink tint, edge fades in, no lift
```

A panel is already 82% white. There is nothing brighter for a row to move towards, so the
tint goes the other way — an ink wash plus a hairline that fades in is the only pair of
changes that reads at all on that ground. And it does not lift: a row that translates inside
a scrolling list looks like the list moved.

Both classes also answer to the states a control can be *left* in:

```css
[aria-current]  [aria-pressed="true"]  [aria-expanded="true"]
[data-state="open"]  [data-active="true"]
```

One step past hover, so a hovered row and the open one stay distinct. This is why the rail
buttons, the history rows and the running activity step all get their selected ground for
free — they already carried the right ARIA, and now something reads it. **Never add a second `bg-*` utility for a selected state**; it will be
overridden (see below) and the two will drift.

### Where it enters

Most call sites never name a hover class. [`ui/button.tsx`](frontend/src/components/ui/button.tsx)
carries them per variant:

```
default      glass-solid + bg-primary
outline      glass + glass-hover
secondary    glass + glass-hover
ghost        glass-row              ← the rail, every panel action, every row control
destructive  glass-alert + glass-row-danger
```

`ghost` is the high-traffic one. Every icon button in the rail and every action in a panel is
a ghost `Button`, so they all inherited both the hover ground and the open/pressed/current
state without a single call site opting in.

## The cascade rule that governs all of it

This block is **unlayered**, and Tailwind's utilities live in `@layer utilities`. Unlayered
CSS always beats layered CSS, whatever the specificity. So:

> A `bg-*`, `border-*` or `hover:bg-*` utility alongside a glass class is dead weight.
> The glass class wins.

That cuts both ways and is worth internalising before editing any of this:

- It is why `.glass-chip` can replace `border border-line` at the call site rather than
  fighting it.
- It is why a held state has to be a rule in globals.css and *cannot* be an
  `aria-pressed:border-line-strong` utility — `.glass` sets `border` outright and would win
  regardless of the variant.
- It is why none of these classes may set `position`. A `position` here would silently beat
  the `fixed` on the rail and the activity feed.

## shadcn and Clerk

The shadcn ramp is re-pointed at the glass tokens rather than at flat greys, so every
primitive built on it turns to glass in one place instead of in each component file:

```
--card --secondary  →  var(--glass)
--popover           →  var(--glass-dense)
--muted --accent    →  var(--hover-tint)
--border --input    →  var(--line)
```

`--background` and the `*-foreground` pairs stay opaque — they are text colours and a solid
ground of last resort.

Clerk's components read the same variables through `@clerk/ui/themes/shadcn.css`, so they
arrive translucent on their own. What that theme cannot supply is the blur, which a `.cl-*`
block at the end of the glass section hands them. It runs after Clerk's stylesheet (imported
at the top of globals.css) and so wins at equal specificity.

## Fallback

`@supports not (backdrop-filter: ...)` pushes `.glass` to 90% and `.glass-panel` to 97%
white. Translucent-without-blur over the ambient washes is unreadable at the normal tints,
so the borders carry the structure instead.

`prefers-reduced-motion` drops every transition in the system and, specifically, the 1px
lift — the one hover change that actually moves something.
