"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { ToolCall as StoredToolCall } from "@/lib/conversations";
import { useEffort } from "@/lib/effort";
import { browserSessionId } from "@/lib/identity";
import { PcmPlayer } from "@/lib/pcm-player";
import {
  voiceSocketUrl,
  type ActivityState,
  type ActivityStep,
  type ClientEvent,
  type Providers,
  type ServerEvent,
  type VoiceTimings,
} from "@/lib/voice-protocol";

/** Saaras takes 30 s of audio per turn — stop just short of it. */
export const MAX_RECORDING_MS = 29_000;

export const BAR_COUNT = 14;

const IDLE_BARS = Array<number>(BAR_COUNT).fill(0.06);

/**
 * How long the recorder holds audio before handing it over. At 250 ms the
 * upload happens *while* you speak, so `audio.end` is the last quarter-second
 * of a take rather than the whole file.
 */
const SLICE_MS = 250;

/** Below this the meter counts a frame as silence. */
const SILENCE_LEVEL = 0.055;

/** Silence long enough to mean "I'm done talking". */
const SILENCE_MS = 1_100;

/** No point auto-stopping before there is anything to send. */
const MIN_SPEECH_MS = 600;

export type VoiceStatus =
  | "idle"
  | "connecting"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "error";

/**
 * One step of the backend's work, as the socket reported it.
 *
 * A step is a *row*, not an event: `stt` starting and `stt` finishing are the
 * same line changing state, because a log that prints both is twice as long
 * and no more informative. `key` is what decides sameness.
 */
export type ActivityLine = {
  key: string;
  turnId: string | null;
  step: ActivityStep;
  state: ActivityState;
  label: string;
  detail: string | null;
  ms: number | null;
  at: number;
};

/** Rows kept in memory. The card shows a handful; this is the scrollback. */
const MAX_ACTIVITY = 24;

/** How a tool the agent reached for ended up. */
export type ToolCallState = "running" | "ok" | "error" | "skipped";

/**
 * One tool the agent ran, as its own row.
 *
 * The activity log folds every `tool` frame of a turn into a single line —
 * right for a pipeline step, wrong for the tools themselves, where "three ran,
 * the second one failed" is the whole story and one row can only tell a third
 * of it. So the same frames are folded a second time here, keyed per call
 * instead of per step.
 *
 * The socket does not carry a call id, so the pairing is by name: a round
 * opens with `state: "start"` listing the names the model chose, and each call
 * then reports `done` or `error` under its own name, in the order the server
 * ran them. Matching the *first still-running* row with that name is what
 * keeps the same tool called twice in one round as two rows.
 */
export type ToolCall = {
  key: string;
  turnId: string | null;
  /** The raw slug — `GMAIL_FETCH_EMAILS`, `query_dataset`. */
  name: string;
  state: ToolCallState;
  /** Why it failed, when it did. */
  detail: string | null;
  /**
   * How long this call took, not how far into the turn it finished.
   *
   * Every `ms` on the wire is elapsed-since-the-turn-started, so the duration
   * is the gap between two of them. Tools in a round run one after another,
   * which makes the previous call's finish the next one's start.
   */
  ms: number | null;
  /** Elapsed at the moment this call began. Bookkeeping for the line above. */
  startedMs: number | null;
  at: number;
};

/** Tool rows kept in memory. A turn that runs more than this is pathological. */
const MAX_TOOL_CALLS = 32;

export type Exchange = {
  id: string;
  at: number;
  question: string;
  languageCode: string | null;
  language: string | null;
  /** Grows token by token while the reply is being written. */
  reply: string;
  done: boolean;
  interrupted: boolean;
  timings: VoiceTimings | null;
};

type Options = {
  /** Called once per completed take, so a caller can log the turn. */
  onExchange?: (exchange: Exchange) => void;
  /**
   * The conversation to continue, from `/c/{id}`. Omitted on a fresh page —
   * the server opens one on the first take and announces it below.
   *
   * Changing it opens a new socket, which is right: it is a different
   * conversation, with different history behind it.
   */
  conversationId?: string;
  /**
   * The server bound or opened a conversation for this socket. `created` is
   * the first take of a new one — the moment the address bar should change.
   */
  onConversation?: (id: string, created: boolean) => void;
  /**
   * A tool finished, with both halves of it — what the agent sent and what
   * came back. Fired once per call, as it lands rather than at the end of the
   * turn, and it carries the id the backend is writing the row under, so the
   * same call arriving again in a stored thread is recognisable.
   *
   * Separate from the `ToolCall` rows this hook keeps for the stage card:
   * those are folded from `activity` and exist from the moment a call starts,
   * which is what lets the card say "running". This one only exists once
   * there is a result to report.
   */
  onTool?: (call: StoredToolCall) => void;
};

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;

  return [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ].find((type) => MediaRecorder.isTypeSupported(type));
}

/**
 * One spoken conversation.
 *
 * Owns three moving parts and the seams between them: the microphone, the
 * socket to FastAPI, and the player that turns the PCM coming back into sound.
 * The interesting states are the overlaps — audio uploads while you are still
 * talking, the reply is read aloud while it is still being written, and
 * pressing the orb mid-answer stops the speech instead of queueing behind it.
 */
export function useVoiceSession({
  onExchange,
  conversationId,
  onConversation,
  onTool,
}: Options = {}) {
  // The socket is opened against FastAPI directly, so signing in has to be
  // proved to it rather than assumed: `getToken` mints a short-lived Clerk
  // token per connection, and `userId` in the dependencies below is what
  // reopens the socket when someone signs in or out mid-session — otherwise
  // the next conversation would be filed under the browser, not the account.
  const { getToken, userId } = useAuth();
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [bars, setBars] = useState<number[]>(IDLE_BARS);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(MAX_RECORDING_MS);
  const [providers, setProviders] = useState<Providers | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [activity, setActivity] = useState<ActivityLine[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [connected, setConnected] = useState(false);

  // React runs a state updater during render, so anything with a side effect
  // — the `onExchange` callback — cannot live inside one. The list is kept in
  // a ref as well, updated synchronously, and every change goes through
  // `update` below: state for rendering, ref for reading back immediately.
  const exchangesRef = useRef<Exchange[]>([]);
  /**
   * The turn that was interrupted. The server takes a moment to notice a
   * cancel, so a `speech.start` for that turn can still arrive afterwards —
   * and opening the player on it would restart the answer that was just cut
   * off. Anything carrying this id is ignored.
   */
  const interruptedRef = useRef<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frameRef = useRef<number | null>(null);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const armingRef = useRef(false);
  /** Monotonic, so a row's key survives the list being trimmed under it. */
  const toolSeqRef = useRef(0);
  /** A connection whose token is still being fetched — see `connect`. */
  const openingRef = useRef(false);
  const abandonedRef = useRef(false);
  const attemptsRef = useRef(0);
  const livingRef = useRef(true);

  // Read inside callbacks that must not be rebuilt when these change.
  const tokenRef = useRef(getToken);
  useEffect(() => {
    tokenRef.current = getToken;
  }, [getToken]);

  // The rung to ask for, read at the moment a take starts. Through a ref for
  // the same reason as everything else here: moving the slider mid-take must
  // not rebuild `start` and tear the recorder down under it.
  const [effort] = useEffort();
  const effortRef = useRef(effort);
  useEffect(() => {
    effortRef.current = effort;
  }, [effort]);

  const exchangeRef = useRef(onExchange);
  const conversationRef = useRef(onConversation);
  const toolRef = useRef(onTool);
  const statusRef = useRef<VoiceStatus>("idle");
  useEffect(() => {
    exchangeRef.current = onExchange;
  }, [onExchange]);
  useEffect(() => {
    conversationRef.current = onConversation;
  }, [onConversation]);
  useEffect(() => {
    toolRef.current = onTool;
  }, [onTool]);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // `connect` schedules its own retry, and a callback cannot name itself.
  const reconnectRef = useRef<() => void>(() => {});

  const update = useCallback((change: (previous: Exchange[]) => Exchange[]) => {
    const next = change(exchangesRef.current);
    exchangesRef.current = next;
    setExchanges(next);
    return next;
  }, []);

  /**
   * Fold one `activity` frame into the log.
   *
   * Same turn, same step → the row already there is updated in place, so a
   * step keeps the position it took when it started. Steps arrive in pipeline
   * order, which makes "last in the array" the newest step of the turn and
   * lets the card render newest-first without any row ever moving.
   */
  const note = useCallback((line: ActivityLine) => {
    setActivity((previous) => {
      const at = previous.findIndex((row) => row.key === line.key);
      if (at === -1) return [...previous, line].slice(-MAX_ACTIVITY);

      const next = [...previous];
      next[at] = line;
      return next;
    });
  }, []);

  /**
   * Fold one `tool` frame into the per-call list.
   *
   * `start` opens a round: one row per name the model chose, all pending.
   * Everything else closes one of them. A close with no matching pending row
   * still lands — a client that missed the round's opening frame should show
   * the tool that ran, not nothing.
   */
  const noteTool = useCallback(
    (event: Extract<ServerEvent, { type: "activity" }>) => {
      setToolCalls((previous) => {
        if (event.state === "start") {
          const names = (event.detail ?? "")
            .split(",")
            .map((name) => name.trim())
            .filter(Boolean);
          if (!names.length) return previous;

          const opened = names.map<ToolCall>((name) => ({
            key: `${event.turnId ?? "-"}|${toolSeqRef.current++}`,
            turnId: event.turnId,
            name,
            state: "running",
            detail: null,
            ms: null,
            startedMs: event.ms,
            at: Date.now(),
          }));
          return [...previous, ...opened].slice(-MAX_TOOL_CALLS);
        }

        const state: ToolCallState =
          event.state === "error"
            ? "error"
            : event.state === "done"
              ? "ok"
              : "skipped";

        const next = [...previous];
        const at = next.findIndex(
          (call) =>
            call.turnId === event.turnId &&
            call.name === event.label &&
            call.state === "running",
        );

        if (at === -1) {
          next.push({
            key: `${event.turnId ?? "-"}|${toolSeqRef.current++}`,
            turnId: event.turnId,
            name: event.label,
            state,
            detail: event.detail,
            ms: null,
            startedMs: null,
            at: Date.now(),
          });
          return next.slice(-MAX_TOOL_CALLS);
        }

        const opened = next[at];
        next[at] = {
          ...opened,
          state,
          detail: event.detail,
          ms:
            event.ms !== null && opened.startedMs !== null
              ? Math.max(0, event.ms - opened.startedMs)
              : null,
        };

        // Tools in a round are awaited one at a time, so this call finishing
        // is the next one starting — and the only clock the next row will get.
        const after = next.findIndex(
          (call, index) =>
            index > at && call.turnId === event.turnId && call.state === "running",
        );
        if (after !== -1 && event.ms !== null)
          next[after] = { ...next[after], startedMs: event.ms };

        return next;
      });
    },
    [],
  );

  /**
   * Stop the pulse on anything still running for a turn.
   *
   * A step only reports "done" when it finishes, and an interrupted one never
   * does — barge-in cuts the reply mid-segment. Without this the log keeps a
   * row pulsing forever over a turn that ended two questions ago, which is
   * worse than saying nothing. `skipped` is the honest state for it: the step
   * stopped, it did not complete.
   */
  const settle = useCallback((turnId: string | null) => {
    if (!turnId) return;

    setActivity((previous) =>
      previous.map((row) =>
        row.turnId === turnId &&
        (row.state === "running" || row.state === "start")
          ? { ...row, state: "skipped" as const }
          : row,
      ),
    );

    // Same reasoning for the tool rows: a call that was still running when the
    // turn ended never reports, and a row left pulsing over a finished turn
    // claims work is happening that stopped a question ago.
    setToolCalls((previous) =>
      previous.map((call) =>
        call.turnId === turnId && call.state === "running"
          ? { ...call, state: "skipped" as const }
          : call,
      ),
    );
  }, []);

  const send = useCallback((event: ClientEvent) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN)
      socket.send(JSON.stringify(event));
  }, []);

  // ---- playback ---------------------------------------------------------

  const player = useCallback(() => {
    if (!playerRef.current) {
      playerRef.current = new PcmPlayer(
        (state) => {
          // The assistant stops being "speaking" when the last block has been
          // played, not when the last byte arrived — the tail is still audible.
          if (state === "idle") {
            setStatus((current) => (current === "speaking" ? "idle" : current));
          }
        },
        (peak, next) => {
          setLevel(peak);
          setBars(next);
        },
        BAR_COUNT,
      );
    }
    return playerRef.current;
  }, []);

  // ---- the socket -------------------------------------------------------

  const connect = useCallback(() => {
    if (openingRef.current) return;
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN)
      return;

    // Minting the token is asynchronous, which opens a window where no socket
    // exists yet and a second caller — the retry timer, StrictMode's second
    // mount — would happily open one too. This flag closes it.
    openingRef.current = true;

    // Identity goes on the URL, so the server can bind the conversation before
    // it says `ready`. An empty session id (no storage, insecure origin) is
    // sent as nothing at all, and the turn simply is not written down.
    const open = (token: string | null) => {
      const url = voiceSocketUrl({
        session: browserSessionId(),
        conversation: conversationId,
        token: token ?? undefined,
      });
      if (!url) return;

      const socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => {
        attemptsRef.current = 0;
        setConnected(true);
        setError(null);
      };

      socket.onclose = () => {
        // A socket that has already been replaced — by StrictMode's double
        // mount, or by a reconnect that beat this event — must not schedule
        // another. Without this the retries multiply and each new socket is a
        // *separate conversation* on the server.
        if (socketRef.current !== socket) return;

        setConnected(false);
        socketRef.current = null;
        if (!livingRef.current) return;

        // Backing off matters: a backend that is down stays down for a while,
        // and a tight retry loop would spend the whole time reconnecting.
        const attempt = Math.min(attemptsRef.current++, 5);
        retryRef.current = setTimeout(
          () => reconnectRef.current(),
          Math.min(8_000, 400 * 2 ** attempt),
        );
      };

      socket.onerror = () => {
        // `onclose` follows and handles the retry; only the message is ours.
        if (statusRef.current !== "idle") {
          setError("Lost the connection to the voice service.");
          setStatus("error");
        }
      };

      socket.onmessage = (message) => {
        if (message.data instanceof ArrayBuffer) {
          player().push(message.data);
          return;
        }

        let event: ServerEvent;
        try {
          event = JSON.parse(message.data as string) as ServerEvent;
        } catch {
          return;
        }

        switch (event.type) {
          case "ready":
            setProviders(event.providers);
            break;

          case "conversation":
            conversationRef.current?.(event.id, event.created);
            break;

          case "status":
            if (event.stage === "transcribing") setStatus("transcribing");
            if (event.stage === "thinking") setStatus("thinking");
            if (event.stage === "speaking") setStatus("speaking");
            break;

          case "activity":
            // Two folds of the same frame: one row per pipeline step for the
            // log, one row per call for the tool card.
            if (event.step === "tool") noteTool(event);
            note({
              key: `${event.turnId ?? "-"}|${event.step}`,
              turnId: event.turnId,
              step: event.step,
              state: event.state,
              label: event.label,
              detail: event.detail,
              ms: event.ms,
              at: Date.now(),
            });
            break;

          case "tool":
            // The whole call, for the thread. The row the stage card draws was
            // already opened by the `activity` frame above and settled by the
            // one that follows it — this is the same call said in full, and
            // the panel is the only thing with room to show it.
            toolRef.current?.(event);
            break;

          case "transcript":
            update((previous) => [
              ...previous,
              {
                id: event.turnId,
                at: Date.now(),
                question: event.text,
                languageCode: event.languageCode,
                language: event.language,
                reply: "",
                done: false,
                interrupted: false,
                timings: null,
              },
            ]);
            break;

          case "delta":
            update((previous) =>
              previous.map((exchange) =>
                exchange.id === event.turnId
                  ? { ...exchange, reply: exchange.reply + event.text }
                  : exchange,
              ),
            );
            break;

          case "speech.start":
            if (event.turnId === interruptedRef.current) break;
            player().open(event.sampleRate);
            break;

          case "speech.end":
            player().close();
            break;

          case "turn.end": {
            const next = update((previous) =>
              previous.map((exchange) =>
                exchange.id === event.turnId
                  ? {
                      ...exchange,
                      reply: event.reply.trim(),
                      done: true,
                      timings: event.timings,
                    }
                  : exchange,
              ),
            );

            settle(event.turnId);

            const finished = next.find(
              (exchange) => exchange.id === event.turnId,
            );
            if (finished) exchangeRef.current?.(finished);
            break;
          }

          case "canceled": {
            settle(event.turnId);
            const next = update((previous) =>
              previous.map((exchange) =>
                exchange.id === event.turnId
                  ? { ...exchange, done: true, interrupted: true }
                  : exchange,
              ),
            );
            setStatus((current) => (current === "speaking" ? current : "idle"));

            // A talked-over turn is still a turn, and the backend has already
            // filed both halves of it. Reporting it here is what keeps the
            // panels showing the same thread the database holds.
            const stopped = next.find(
              (exchange) => exchange.id === event.turnId,
            );
            if (stopped?.question) exchangeRef.current?.(stopped);
            break;
          }

          case "error":
            setError(event.message);
            setStatus("error");
            break;
        }
      };
    };

    void (async () => {
      // Signed out, this is null and the turn is filed under the browser.
      // A failure here is the same outcome: never a reason not to connect.
      let token: string | null = null;
      try {
        token = (await tokenRef.current?.()) ?? null;
      } catch {
        token = null;
      }

      openingRef.current = false;
      if (!livingRef.current) return;
      if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN)
        return;

      open(token);
    })();
    // `userId` is not read in here, and that is the point: it is in the list
    // so that signing in or out rebuilds `connect` and the effect below
    // reopens the socket. Without it the server would keep the identity it was
    // handed at the last handshake, and the next conversation someone started
    // after signing in would still be filed under the browser.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, note, noteTool, player, settle, update, userId]);

  useEffect(() => {
    reconnectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    livingRef.current = true;
    connect();

    return () => {
      livingRef.current = false;
      if (retryRef.current) clearTimeout(retryRef.current);
      socketRef.current?.close();
      socketRef.current = null;
      void playerRef.current?.dispose();
      playerRef.current = null;
    };
  }, [connect]);

  // ---- the microphone ---------------------------------------------------

  const teardown = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;

    if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    stopTimerRef.current = null;

    if (tickTimerRef.current) clearInterval(tickTimerRef.current);
    tickTimerRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    void contextRef.current?.close().catch(() => {});
    contextRef.current = null;
    analyserRef.current = null;

    setBars(IDLE_BARS);
    setLevel(0);
  }, []);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.stop();
  }, []);

  /**
   * Meter the input, and decide when the speaker has stopped.
   *
   * A voice app that waits for a second tap is a walkie-talkie. Watching the
   * level for a pause and closing the take there is what makes it feel like
   * being listened to — with a floor on speech length, so a cough or the room
   * tone at the very start cannot end a take before it holds anything.
   */
  const meter = useCallback((onSilence: () => void) => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const spectrum = new Uint8Array(
      new ArrayBuffer(analyser.frequencyBinCount),
    );
    const usable = Math.floor(spectrum.length * 0.55);
    const band = Math.max(1, Math.floor(usable / BAR_COUNT));

    const startedAt = performance.now();
    let quietSince: number | null = null;
    let spoke = false;

    const tick = () => {
      analyser.getByteFrequencyData(spectrum);

      let peak = 0;
      const next = Array.from({ length: BAR_COUNT }, (_, i) => {
        let sum = 0;
        for (let j = 0; j < band; j++) sum += spectrum[i * band + j] ?? 0;

        const shaped = Math.min(1, Math.pow(sum / band / 255, 0.62) * 1.5);
        peak = Math.max(peak, shaped);
        return Math.max(0.06, shaped);
      });

      setBars(next);
      setLevel(peak);

      const now = performance.now();
      if (peak > SILENCE_LEVEL) {
        spoke = true;
        quietSince = null;
      } else if (spoke && now - startedAt > MIN_SPEECH_MS) {
        quietSince ??= now;
        if (now - quietSince > SILENCE_MS) {
          onSilence();
          return;
        }
      }

      frameRef.current = requestAnimationFrame(tick);
    };

    tick();
  }, []);

  const start = useCallback(async () => {
    if (armingRef.current || recorderRef.current?.state === "recording") return;
    armingRef.current = true;

    setError(null);
    abandonedRef.current = false;

    // Anything the assistant is still saying is over the moment someone else
    // starts talking. Stop the sound first, then tell the server to stop
    // generating — in that order, so the silence is immediate.
    if (statusRef.current !== "idle" && statusRef.current !== "error") {
      interruptedRef.current = exchangesRef.current.at(-1)?.id ?? null;
      settle(interruptedRef.current);
      playerRef.current?.stop();
      send({ type: "cancel" });
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      armingRef.current = false;
      setError("This browser can't record audio.");
      setStatus("error");
      return;
    }

    // Both need a user gesture behind them, and this call is inside one.
    await player().unlock();
    connect();

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Without cancellation the microphone hears the reply and the next
        // take is the assistant talking to itself.
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch {
      armingRef.current = false;
      setError("Microphone access was blocked — allow it and try again.");
      setStatus("error");
      return;
    }

    streamRef.current = stream;

    const context = new AudioContext();
    contextRef.current = context;
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.72;
    context.createMediaStreamSource(stream).connect(analyser);
    analyserRef.current = analyser;

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined,
    );
    recorderRef.current = recorder;

    const type = (recorder.mimeType || mimeType || "audio/webm")
      .split(";")[0]
      .trim();
    send({ type: "audio.start", mime: type, effort: effortRef.current });

    recorder.ondataavailable = async (event) => {
      if (event.data.size === 0 || abandonedRef.current) return;
      const socket = socketRef.current;
      if (socket?.readyState !== WebSocket.OPEN) return;
      // Sent as it is recorded, so the last slice is all that is left to
      // upload when the take ends.
      socket.send(await event.data.arrayBuffer());
    };

    recorder.onstop = () => {
      teardown();
      if (abandonedRef.current) {
        send({ type: "cancel" });
        setStatus("idle");
        return;
      }
      send({ type: "audio.end" });
      setStatus("transcribing");
    };

    recorder.start(SLICE_MS);
    armingRef.current = false;
    setStatus("listening");
    setRemaining(MAX_RECORDING_MS);
    meter(stop);

    const startedAt = performance.now();
    tickTimerRef.current = setInterval(() => {
      setRemaining(
        Math.max(0, MAX_RECORDING_MS - (performance.now() - startedAt)),
      );
    }, 250);

    stopTimerRef.current = setTimeout(stop, MAX_RECORDING_MS);
  }, [connect, meter, player, send, settle, stop, teardown]);

  /** Stop recording and throw the take away. */
  const cancel = useCallback(() => {
    abandonedRef.current = true;
    interruptedRef.current = exchangesRef.current.at(-1)?.id ?? null;
    settle(interruptedRef.current);
    stop();
    teardown();
    playerRef.current?.stop();
    send({ type: "cancel" });
    setStatus("idle");
  }, [send, settle, stop, teardown]);

  /** Ask without speaking — the same turn, typed. */
  const ask = useCallback(
    async (text: string, language?: string | null) => {
      if (!text.trim()) return;
      interruptedRef.current = exchangesRef.current.at(-1)?.id ?? null;
      settle(interruptedRef.current);
      playerRef.current?.stop();
      await player().unlock();
      connect();
      send({ type: "text", text, language, effort: effortRef.current });
      setStatus("thinking");
    },
    [connect, player, send, settle],
  );

  /** Forget the conversation, on both sides. */
  const reset = useCallback(() => {
    playerRef.current?.stop();
    send({ type: "reset" });
    update(() => []);
    setActivity([]);
    setToolCalls([]);
    setError(null);
    setStatus("idle");
  }, [send, update]);

  useEffect(() => teardown, [teardown]);

  const listening = status === "listening";
  const speaking = status === "speaking";
  const working = status === "transcribing" || status === "thinking";

  const current = exchanges[exchanges.length - 1] ?? null;

  /**
   * The tools this turn ran. The list behind it is scrollback for a session;
   * what a card in the corner can usefully hold is one turn's worth, and the
   * turn on screen is the one the transcript beside it is showing.
   */
  const currentTools = useMemo(
    () => toolCalls.filter((call) => call.turnId === current?.id),
    [toolCalls, current?.id],
  );

  return {
    status,
    listening,
    speaking,
    working,
    bars,
    level,
    error,
    remaining,
    providers,
    connected,
    exchanges,
    activity,
    toolCalls,
    currentTools,
    current,
    start,
    stop,
    cancel,
    ask,
    reset,
  };
}
