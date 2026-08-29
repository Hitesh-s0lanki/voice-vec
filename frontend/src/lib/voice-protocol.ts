/**
 * The voice socket's contract, mirrored from `src/schemas/voice.py`.
 *
 * Two channels share the socket. JSON frames are the events below; binary
 * frames are PCM for whichever segment the last `speech.start` announced —
 * which is why the format lives on that event and not on the audio.
 */

import type { ToolCall } from "@/lib/conversations";

export type VoiceStage = "idle" | "transcribing" | "thinking" | "speaking";

export type Providers = {
  stt: string | null;
  llm: string | null;
  llmModel: string | null;
  tts: string | null;
  /** Highest rung of the effort ladder this deployment will run. */
  effortMax: number;
  /**
   * The answer cache as it actually resolved — `semantic`, `exact-only`,
   * `unset`, `off`. Not a boolean: the gap between the first two is whether
   * paraphrases are caught, and a deployment that believes it has semantic
   * caching and does not will read its hit rate as a tuning problem.
   */
  cache: string;
  /**
   * Agent memory as it actually resolved — `on`, `unset`, `off`,
   * `unavailable (…)`. Same vocabulary and the same reason as `cache`: an
   * agent that has stopped remembering across conversations reads as a model
   * regression until someone checks whether the service is wired up.
   */
  memory: string;
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
  | "memory"
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
      /**
       * Which conversation this socket is writing into. Arrives once on
       * connect when the socket was opened against an existing `conv_…`, and
       * once with `created` the first time something worth keeping is said —
       * that second one is the cue to put `/c/{id}` in the address bar.
       */
      type: "conversation";
      id: string;
      title: string | null;
      turns: number;
      created: boolean;
    }
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
  | ({
      /**
       * One finished tool call, whole — arguments in, result out.
       *
       * `activity` says *that* a tool ran, which is all a step-shaped frame
       * can carry. This is the call itself, and it is deliberately the same
       * `ToolCall` the stored thread returns rather than a shape of its own:
       * the panel draws a turn being spoken and a turn read back out of
       * Postgres with one component, and two near-identical types would let
       * those two drift apart a field at a time.
       *
       * `id` is the id the row will be written under, so a call announced now
       * and the same call fetched after a reload are recognisably one call.
       */
      type: "tool";
    } & ToolCall)
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
  /**
   * `effort` rides on the events that *start a question*, not on the socket
   * URL, because it is a property of the turn rather than of the connection:
   * moving the slider has to take effect on the next question, and a query
   * parameter would need a reconnect to change. Omitted, the server uses its
   * configured default. See `src/rag/effort.py` for what each rung runs.
   */
  | {
      type: "audio.start";
      mime: string;
      language?: string | null;
      effort?: number;
    }
  | { type: "audio.end" }
  | { type: "text"; text: string; language?: string | null; effort?: number }
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
/** Where the socket lives, before anything is asked of it. */
function socketBase(): string {
  const configured = process.env.NEXT_PUBLIC_VOICE_WS_URL;
  if (configured) return configured;

  if (typeof window === "undefined") return "";

  // Same host, backend port. Right for `next dev` running beside
  // `python -m src.main`; set NEXT_PUBLIC_VOICE_WS_URL for anything else.
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.hostname}:8001/voice/ws`;
}

/**
 * The socket URL, carrying who is asking and what they are continuing.
 *
 * On the query string rather than in a first frame because the server binds
 * before it says `ready`: the model needs its own history back *before* the
 * first take, not a round trip after it. A browser cannot set headers on a
 * WebSocket handshake either, so it is also the only place `token` can ride —
 * acceptable because a Clerk session token lives about a minute and a fresh
 * one is fetched per connection.
 *
 * There is no `user` here on purpose. The account comes from the token's
 * verified `sub`, server-side; a user id on a query string is a value anyone
 * can type.
 */
export function voiceSocketUrl(
  who: { session?: string; token?: string; conversation?: string } = {},
): string {
  const base = socketBase();
  if (!base) return "";

  const url = new URL(base);
  if (who.session) url.searchParams.set("session", who.session);
  if (who.token) url.searchParams.set("token", who.token);
  if (who.conversation) url.searchParams.set("conversation", who.conversation);

  return url.toString();
}

/** The same backend over plain HTTP — `/voice/config`, `/voice/speak`. */
export function voiceHttpUrl(path: string): string {
  // The bare base, not `voiceSocketUrl()` — the query string that call adds
  // would end up in the middle of the path this builds.
  const socket = socketBase();
  if (!socket) return path;

  const base = socket.replace(/^ws/, "http").replace(/\/voice\/ws$/, "");
  return `${base}${path}`;
}
