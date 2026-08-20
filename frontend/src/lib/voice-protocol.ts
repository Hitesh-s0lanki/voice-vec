/**
 * The voice socket's contract, mirrored from `src/schemas/voice.py`.
 *
 * Two channels share the socket. JSON frames are the events below; binary
 * frames are PCM for whichever segment the last `speech.start` announced —
 * which is why the format lives on that event and not on the audio.
 */

export type VoiceStage = "idle" | "transcribing" | "thinking" | "speaking";

export type Providers = {
  stt: string | null;
  llm: string | null;
  llmModel: string | null;
  tts: string | null;
  ragEnabled: boolean;
};

export type VoiceTimings = {
  stt: number | null;
  firstToken: number | null;
  firstSegment: number | null;
  /** Silence before the first sound. The number that decides how this feels. */
  firstAudio: number | null;
  reply: number | null;
  total: number;
};

/**
 * The steps a turn is made of. `status` says which of four coarse stages the
 * turn is in — this says what the backend is actually doing inside one, and
 * stays open at the end so a new step (a tool, a query) needs no client
 * release. Anything unrecognised still renders; it just gets the plain dot.
 */
export type ActivityStep =
  | "stt"
  | "retrieval"
  | "tool"
  | "llm"
  | "speech"
  | "turn"
  | (string & {});

export type ActivityState = "start" | "running" | "done" | "skipped" | "error";

export type ServerEvent =
  | {
      type: "ready";
      sessionId: string;
      providers: Providers;
      sampleRate: number;
      languages: Record<string, string>;
    }
  | { type: "status"; stage: VoiceStage; turnId: string | null }
  | {
      type: "activity";
      turnId: string | null;
      step: ActivityStep;
      state: ActivityState;
      /** Written server-side, rendered as-is: "Model is writing the reply". */
      label: string;
      /** Who or what — provider, model, a count. */
      detail: string | null;
      /** Elapsed at this point in the turn. */
      ms: number | null;
    }
  | {
      type: "transcript";
      turnId: string;
      text: string;
      languageCode: string | null;
      language: string | null;
      confidence: number | null;
      provider: string | null;
      ms: number | null;
    }
  | { type: "delta"; turnId: string; text: string }
  | {
      type: "speech.start";
      turnId: string;
      segment: number;
      text: string;
      provider: string;
      voice: string;
      languageCode: string | null;
      sampleRate: number;
      format: "pcm_s16le";
    }
  | { type: "speech.end"; turnId: string; segment: number; bytes: number; ms: number }
  | {
      type: "turn.end";
      turnId: string;
      reply: string;
      languageCode: string | null;
      segments: number;
      timings: VoiceTimings;
    }
  | { type: "canceled"; turnId: string | null; spoken: string | null }
  | {
      type: "error";
      message: string;
      turnId: string | null;
      stage: string | null;
      provider: string | null;
    }
  | { type: "pong" };

export type ClientEvent =
  | { type: "audio.start"; mime: string; language?: string | null }
  | { type: "audio.end" }
  | { type: "text"; text: string; language?: string | null }
  | { type: "cancel" }
  | { type: "reset" }
  | { type: "ping" };

/**
 * Where the FastAPI socket lives.
 *
 * The browser talks to the backend directly rather than through a Next route
 * handler: route handlers speak request/response, and this is a long-lived
 * two-way stream. No key is exposed by doing so — Sarvam and OpenAI are only
 * ever called from the Python side.
 */
export function voiceSocketUrl(): string {
  const configured = process.env.NEXT_PUBLIC_VOICE_WS_URL;
  if (configured) return configured;

  if (typeof window === "undefined") return "";

  // Same host, backend port. Right for `next dev` running beside
  // `python -m src.main`; set NEXT_PUBLIC_VOICE_WS_URL for anything else.
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.hostname}:8001/voice/ws`;
}

/** The same backend over plain HTTP — `/voice/config`, `/voice/speak`. */
export function voiceHttpUrl(path: string): string {
  const socket = voiceSocketUrl();
  if (!socket) return path;

  const base = socket.replace(/^ws/, "http").replace(/\/voice\/ws$/, "");
  return `${base}${path}`;
}
