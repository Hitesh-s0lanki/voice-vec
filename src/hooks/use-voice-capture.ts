"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Sarvam's REST endpoint accepts up to 30s of audio — stop just short of it. */
export const MAX_RECORDING_MS = 29_000;

export const BAR_COUNT = 14;

const IDLE_BARS = Array<number>(BAR_COUNT).fill(0.06);

export type CaptureStatus = "idle" | "listening" | "thinking" | "error";

export type Transcript = {
  text: string;
  languageCode: string | null;
};

/** Pick a container the browser can record and Sarvam can decode. */
function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;

  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

function extensionFor(mimeType: string) {
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

/**
 * MediaRecorder reports `audio/webm;codecs=opus`, but Sarvam matches content
 * types exactly and only allows the bare `audio/webm`. Drop the parameters.
 */
function baseMimeType(mimeType: string) {
  return mimeType.split(";")[0].trim();
}

export function useVoiceCapture() {
  const [status, setStatus] = useState<CaptureStatus>("idle");
  const [bars, setBars] = useState<number[]>(IDLE_BARS);
  const [level, setLevel] = useState(0);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(MAX_RECORDING_MS);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const frameRef = useRef<number | null>(null);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abandonedRef = useRef(false);
  const armingRef = useRef(false);

  /** Release the mic, the meter loop and both timers. Safe to call twice. */
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

  useEffect(() => teardown, [teardown]);

  const upload = useCallback(async (blob: Blob, mimeType: string) => {
    setStatus("thinking");

    const form = new FormData();
    form.append("file", blob, `speech.${extensionFor(mimeType)}`);

    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        body: form,
      });

      const data = (await response.json()) as {
        transcript?: string;
        languageCode?: string | null;
        error?: string;
      };

      if (!response.ok) {
        setError(data.error ?? "Transcription failed.");
        setStatus("error");
        return;
      }

      setTranscript({
        text: data.transcript ?? "",
        languageCode: data.languageCode ?? null,
      });
      setStatus("idle");
    } catch {
      setError("Could not reach the server.");
      setStatus("error");
    }
  }, []);

  /** Sample the spectrum into BAR_COUNT bands, once per frame. */
  const meter = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const spectrum = new Uint8Array(analyser.frequencyBinCount);
    // voice energy sits low in the spectrum — ignore the sparse top end
    const usable = Math.floor(spectrum.length * 0.55);
    const band = Math.max(1, Math.floor(usable / BAR_COUNT));

    const tick = () => {
      analyser.getByteFrequencyData(spectrum);

      let peak = 0;
      const next = Array.from({ length: BAR_COUNT }, (_, i) => {
        let sum = 0;
        for (let j = 0; j < band; j++) sum += spectrum[i * band + j] ?? 0;

        const avg = sum / band / 255;
        // gamma curve so quiet speech still moves the bars
        const shaped = Math.min(1, Math.pow(avg, 0.62) * 1.5);
        peak = Math.max(peak, shaped);
        return Math.max(0.06, shaped);
      });

      setBars(next);
      setLevel(peak);
      frameRef.current = requestAnimationFrame(tick);
    };

    tick();
  }, []);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.stop();
  }, []);

  const start = useCallback(async () => {
    // getUserMedia can take a moment — ignore taps that land before it resolves
    if (armingRef.current || recorderRef.current?.state === "recording") return;
    armingRef.current = true;

    setError(null);
    setTranscript(null);
    abandonedRef.current = false;

    if (!navigator.mediaDevices?.getUserMedia) {
      armingRef.current = false;
      setError("This browser can't record audio.");
      setStatus("error");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
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
    chunksRef.current = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const type = baseMimeType(recorder.mimeType || mimeType || "audio/webm");
      const blob = new Blob(chunksRef.current, { type });
      chunksRef.current = [];
      teardown();

      if (abandonedRef.current || blob.size === 0) {
        setStatus("idle");
        return;
      }

      void upload(blob, type);
    };

    recorder.start();
    armingRef.current = false;
    setStatus("listening");
    setRemaining(MAX_RECORDING_MS);
    meter();

    const startedAt = performance.now();
    tickTimerRef.current = setInterval(() => {
      setRemaining(Math.max(0, MAX_RECORDING_MS - (performance.now() - startedAt)));
    }, 250);

    stopTimerRef.current = setTimeout(stop, MAX_RECORDING_MS);
  }, [meter, stop, teardown, upload]);

  /** Stop recording and throw the audio away. */
  const cancel = useCallback(() => {
    abandonedRef.current = true;
    stop();
    teardown();
    setStatus("idle");
  }, [stop, teardown]);

  const reset = useCallback(() => {
    setTranscript(null);
    setError(null);
    setStatus("idle");
  }, []);

  return {
    status,
    bars,
    level,
    transcript,
    error,
    remaining,
    start,
    stop,
    cancel,
    reset,
  };
}
