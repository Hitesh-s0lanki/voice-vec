"use client";

import { useEffect, useState } from "react";

import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import { MAX_RECORDING_MS } from "@/hooks/use-voice-session";
import { useHandsFree } from "@/lib/voice-settings";
import { voiceHttpUrl, type Providers } from "@/lib/voice-protocol";
import { cn } from "@/lib/utils";

/**
 * One switch, and a read-out of what is actually answering.
 *
 * The providers are asked for rather than hard-coded, because which one speaks
 * depends on which key the backend was started with — and "why is it replying
 * in an English voice" is exactly the question this panel should answer.
 */
export function SettingsPanel() {
  const [handsFree, setHandsFree] = useHandsFree();
  const [providers, setProviders] = useState<Providers | null>(null);

  useEffect(() => {
    let live = true;
    fetch(voiceHttpUrl("/voice/config"))
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (live && body) setProviders(body.providers as Providers);
      })
      .catch(() => {
        // backend down — the rows fall back to "—" rather than lying
      });
    return () => {
      live = false;
    };
  }, []);

  return (
    <>
      <PanelHeading title="Settings" hint="What is listening, and what is answering." />

      <PanelRule />

      <dl className="flex flex-col gap-0.5">
        <Row label="Hands-free">
          <button
            type="button"
            role="switch"
            aria-checked={handsFree}
            onClick={() => setHandsFree(!handsFree)}
            title="Reopen the microphone as soon as the answer ends"
            className={cn(
              "relative h-5 w-9 rounded-full border border-line transition-colors",
              handsFree ? "bg-ink/75" : "bg-surface-2",
            )}
          >
            <span
              aria-hidden
              className={cn(
                "absolute top-0.5 size-3.5 rounded-full bg-shell shadow-sm transition-all",
                handsFree ? "left-4.5" : "left-0.5",
              )}
            />
          </button>
        </Row>

        <Row label="Take length">
          <span className="text-[0.75rem] tabular-nums text-ink-soft">
            {Math.round(MAX_RECORDING_MS / 1000)}s
          </span>
        </Row>

        <Row label="Hearing">
          <span className="text-[0.75rem] text-ink-soft">
            {providers?.stt === "sarvam"
              ? "Sarvam Saaras"
              : providers?.stt === "openai"
                ? "OpenAI Whisper"
                : "—"}
          </span>
        </Row>

        <Row label="Answering">
          <span className="text-[0.75rem] text-ink-soft" title={providers?.llmModel ?? ""}>
            {providers?.llmModel ?? "—"}
          </span>
        </Row>

        <Row label="Speaking">
          <span className="text-[0.75rem] text-ink-soft">
            {providers?.tts === "sarvam"
              ? "Sarvam Bulbul"
              : providers?.tts === "openai"
                ? "OpenAI"
                : "—"}
          </span>
        </Row>

        <Row label="Language">
          <span className="text-[0.75rem] text-ink-soft">Auto-detected</span>
        </Row>

        <Row label="Retrieval">
          <PanelChip>{providers?.ragEnabled ? "on" : "off"}</PanelChip>
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
