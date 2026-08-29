"use client";

import { useEffect, useState } from "react";

import { PanelChip, PanelHeading, PanelRule } from "@/components/panels/panel";
import { MAX_RECORDING_MS } from "@/hooks/use-voice-session";
import { voiceHttpUrl, type Providers } from "@/lib/voice-protocol";

/**
 * A read-out of what is actually listening, answering and speaking.
 *
 * The providers are asked for rather than hard-coded, because which one speaks
 * depends on which key the backend was started with — and "why is it replying
 * in an English voice" is exactly the question this panel should answer.
 */
export function SettingsPanel() {
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

        {/*
          Not a boolean, because the interesting difference is between the two
          working states: `semantic` catches a paraphrase of a question already
          answered, `exact-only` catches only the same words again. A
          deployment that believes it has the first and has the second will
          read its hit rate as a tuning problem rather than a missing module.
        */}
        <Row label="Answer cache">
          <PanelChip title="Repeat questions are answered from Redis instead of re-running the pipeline">
            {providers?.cache ?? "—"}
          </PanelChip>
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
    <div className="glass-row flex min-h-9 items-center justify-between gap-3 rounded-lg px-2 py-1">
      <dt className="text-[0.78rem] text-ink-muted">{label}</dt>
      <dd className="flex items-center">{children}</dd>
    </div>
  );
}
