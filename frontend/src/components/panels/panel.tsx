import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** Title row every rail panel opens with. `children` is the trailing action. */
export function PanelHeading({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children?: ReactNode;
}) {
  return (
    // `data-slot` so a sheet can reserve room for its close button without
    // every panel having to know it is in one.
    <div
      data-slot="panel-heading"
      className="flex items-start justify-between gap-3 px-1 pt-0.5"
    >
      <div className="flex flex-col gap-0.5">
        <p className="text-[0.82rem] font-medium tracking-[-0.01em] text-ink">
          {title}
        </p>
        {hint && (
          <p className="text-[0.72rem] leading-snug text-ink-muted">{hint}</p>
        )}
      </div>
      {children}
    </div>
  );
}

export function PanelEmpty({
  icon: Icon,
  children,
}: {
  icon: LucideIcon;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-7 text-center">
      <Icon aria-hidden className="size-4 text-ink-muted" />
      <p className="max-w-52 text-[0.75rem] leading-relaxed text-ink-muted">
        {children}
      </p>
    </div>
  );
}

/** Hairline between a panel's heading and its body. */
export function PanelRule({ className }: { className?: string }) {
  return <span aria-hidden className={cn("h-px bg-line", className)} />;
}

/** Small state chip — language, connection status, effort level. */
export function PanelChip({
  children,
  className,
  title,
}: {
  children: ReactNode;
  className?: string;
  /** Native tooltip, for a chip whose meaning is not obvious from its text. */
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        // `.glass-chip` brings its own border — a `border-line` utility here
        // would be overridden by it anyway, since the glass classes are
        // unlayered and land after the utilities.
        "glass-chip rounded-full px-2 py-0.5 text-[0.66rem] font-medium tracking-wide text-ink-muted",
        className,
      )}
    >
      {children}
    </span>
  );
}
