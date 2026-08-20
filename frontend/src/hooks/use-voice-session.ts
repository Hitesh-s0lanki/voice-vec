"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
  /** Keep the microphone going after the assistant finishes speaking. */
  handsFree?: boolean;
  /** Called once per completed take, so a caller can log the turn. */
  onExchange?: (exchange: Exchange) => void;
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
export function useVoiceSession({ handsFree = false, onExchange }: Options = {}) {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [bars, setBars] = useState<number[]>(IDLE_BARS);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(MAX_RECORDING_MS);
  const [providers, setProviders] = useState<Providers | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [activity, setActivity] = useState<ActivityLine[]>([]);
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
  const abandonedRef = useRef(false);
  const attemptsRef = useRef(0);
  const livingRef = useRef(true);

  // Read inside callbacks that must not be rebuilt when these change.
  const handsFreeRef = useRef(handsFree);
  const exchangeRef = useRef(onExchange);
  const statusRef = useRef<VoiceStatus>("idle");
  useEffect(() => {
    handsFreeRef.current = handsFree;
  }, [handsFree]);
  useEffect(() => {
    exchangeRef.current = onExchange;
  }, [onExchange]);
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
        row.turnId === turnId && (row.state === "running" || row.state === "start")
          ? { ...row, state: "skipped" as const }
          : row,
      ),
    );
  }, []);

  const send = useCallback((event: ClientEvent) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(event));
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
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return;

    const url = voiceSocketUrl();
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

        case "status":
          if (event.stage === "transcribing") setStatus("transcribing");
          if (event.stage === "thinking") setStatus("thinking");
          if (event.stage === "speaking") setStatus("speaking");
          break;

        case "activity":
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

          const finished = next.find((exchange) => exchange.id === event.turnId);
          if (finished) exchangeRef.current?.(finished);
          break;
        }

        case "canceled":
          settle(event.turnId);
          update((previous) =>
            previous.map((exchange) =>
              exchange.id === event.turnId
                ? { ...exchange, done: true, interrupted: true }
                : exchange,
            ),
          );
          setStatus((current) => (current === "speaking" ? current : "idle"));
          break;

        case "error":
          setError(event.message);
          setStatus("error");
          break;
      }
    };
  }, [note, player, settle, update]);

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
  const meter = useCallback(
    (onSilence: () => void) => {
      const analyser = analyserRef.current;
      if (!analyser) return;

      const spectrum = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
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
    },
    [],
  );

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
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
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
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;

    const type = (recorder.mimeType || mimeType || "audio/webm").split(";")[0].trim();
    send({ type: "audio.start", mime: type });

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
      setRemaining(Math.max(0, MAX_RECORDING_MS - (performance.now() - startedAt)));
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
      send({ type: "text", text, language });
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
    setError(null);
    setStatus("idle");
  }, [send, update]);

  // Hands-free: the moment the reply finishes playing, open the microphone
  // again. Without it every turn costs a tap, which is not a conversation.
  useEffect(() => {
    if (!handsFree || status !== "idle") return;
    if (exchanges.length === 0) return;

    const last = exchanges[exchanges.length - 1];
    if (!last.done) return;

    const timer = setTimeout(() => void start(), 350);
    return () => clearTimeout(timer);
  }, [exchanges, handsFree, start, status]);

  useEffect(() => teardown, [teardown]);

  const listening = status === "listening";
  const speaking = status === "speaking";
  const working = status === "transcribing" || status === "thinking";

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
    current: exchanges[exchanges.length - 1] ?? null,
    start,
    stop,
    cancel,
    ask,
    reset,
  };
}
