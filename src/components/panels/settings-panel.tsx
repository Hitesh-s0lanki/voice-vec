"use client";

import { PanelHeading, PanelRule } from "@/components/panels/panel";
import { MAX_RECORDING_MS } from "@/hooks/use-voice-capture";

/**
 * Deliberately thin. Nothing here is adjustable yet, so every line is shown as
 * the read-out of a constant rather than dressed up as a control that does
 * nothing.
 */
export function SettingsPanel() {
  return (
    <>
      <PanelHeading title="Settings" hint="More lands here as Vec grows." />

      <PanelRule />

      <dl className="flex flex-col gap-0.5">
        <Row label="Take length">
          <span className="text-[0.75rem] tabular-nums text-ink-soft">
            {Math.round(MAX_RECORDING_MS / 1000)}s
          </span>
        </Row>
        <Row label="Transcription">
          <span className="text-[0.75rem] text-ink-soft">Sarvam Saaras</span>
        </Row>
        <Row label="Language">
          <span className="text-[0.75rem] text-ink-soft">Auto-detected</span>
        </Row>
      </dl>
    </>
  );
}

function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-9 items-center justify-between gap-3 rounded-lg px-2 py-1 transition-colors hover:bg-surface-2">
      <dt className="text-[0.78rem] text-ink-muted">{label}</dt>
      <dd className="flex items-center">{children}</dd>
    </div>
  );
}
