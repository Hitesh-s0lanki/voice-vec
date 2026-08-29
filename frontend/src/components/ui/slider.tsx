"use client"

import * as React from "react"
import { Slider as SliderPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  valueText,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root> & {
  /**
   * What the current value is *called*. A slider whose stops are named
   * things — effort rungs, sizes — should say the name, not the index:
   * "Grounded", not "1 of 5".
   */
  valueText?: string
}) {
  const _values = React.useMemo(
    () =>
      Array.isArray(value)
        ? value
        : Array.isArray(defaultValue)
          ? defaultValue
          : [min, max],
    [value, defaultValue, min, max]
  )

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:min-h-40 data-vertical:w-auto data-vertical:flex-col",
        className
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="glass-track relative grow overflow-hidden rounded-full data-horizontal:h-1.5 data-horizontal:w-full data-vertical:h-full data-vertical:w-1.5"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className="absolute bg-ink select-none data-horizontal:h-full data-vertical:w-full"
        />
      </SliderPrimitive.Track>
      {Array.from({ length: _values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          aria-valuetext={valueText}
          // `.glass-tile` is the surface; the ring is the hover/focus
          // affordance and stays a Tailwind utility because it is a
          // box-shadow the tile does not set.
          className="glass-tile relative block size-3.5 shrink-0 rounded-full ring-ink/20 transition-[box-shadow,transform] select-none after:absolute after:-inset-2 hover:ring-3 focus-visible:ring-3 focus-visible:outline-hidden active:ring-3 active:scale-95 disabled:pointer-events-none disabled:opacity-50"
        />
      ))}
    </SliderPrimitive.Root>
  )
}

export { Slider }
