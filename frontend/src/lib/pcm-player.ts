"use client";

/**
 * Plays 16-bit PCM as it arrives, without waiting for the end of it.
 *
 * The naïve version of this — collect the reply's audio, then
 * `decodeAudioData` — cannot start until the last byte lands, which throws
 * away everything the streaming pipeline bought. Raw PCM needs no decoding at
 * all, so each chunk can be scheduled the moment it arrives.
 *
 * Scheduling is the whole trick. Web Audio plays a buffer at a time you name,
 * on a clock that keeps running whether or not you fed it anything, so the
 * player tracks the exact instant the audio queued so far runs out and starts
 * the next block there. Sample-accurate, which is what keeps a reply built
 * from a dozen separately-synthesised segments sounding like one voice.
 *
 * Two details that are audible when missed:
 *  - a chunk can end mid-sample, splitting a 16-bit frame across two network
 *    frames. The odd byte is carried into the next chunk; dropping it shifts
 *    every following sample by one byte and turns speech into static.
 *  - the first block is scheduled a beat into the future (`LEAD`). Scheduling
 *    it at `currentTime` means any hiccup lands in the past and is skipped.
 */

/** How far ahead of the clock the first block goes. Absorbs a slow chunk. */
const LEAD = 0.12;

/** Blocks shorter than this are merged before being scheduled. */
const MIN_BLOCK_SECONDS = 0.05;

export type PlayerState = "idle" | "playing";

export class PcmPlayer {
  private context: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private spectrum: Uint8Array<ArrayBuffer> | null = null;

  private sampleRate = 24_000;
  /** Where the audio queued so far runs out, on the context clock. */
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private carry: Uint8Array | null = null;
  /**
   * Dropped after a barge-in. Audio for the interrupted segment is still in
   * flight when the listener starts talking, and scheduling it would put a
   * fragment of the abandoned answer on top of the new one. Playback resumes
   * only when a fresh `speech.start` opens the next segment.
   */
  private accepting = true;
  private pending: Int16Array[] = [];
  private pendingFrames = 0;
  private frame: number | null = null;

  constructor(
    private readonly onState?: (state: PlayerState) => void,
    private readonly onLevel?: (level: number, bands: number[]) => void,
    private readonly bandCount = 14,
  ) {}

  /**
   * Build (or resume) the audio context. Must be called from a user gesture —
   * browsers start every context suspended until one happens.
   */
  async unlock(): Promise<void> {
    if (!this.context) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return;

      const context = new Ctor();
      const gain = context.createGain();
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.72;

      gain.connect(analyser);
      analyser.connect(context.destination);

      this.context = context;
      this.gain = gain;
      this.analyser = analyser;
      this.spectrum = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
    }

    if (this.context.state === "suspended") await this.context.resume();
  }

  /** A new segment is starting. Its sample rate travels with it. */
  open(sampleRate: number) {
    this.flushPending();
    this.accepting = true;
    this.sampleRate = sampleRate || this.sampleRate;
    this.carry = null;
  }

  /** Hand over a network frame of PCM. */
  push(bytes: ArrayBuffer) {
    const context = this.context;
    if (!context || !this.accepting) return;

    let raw = new Uint8Array(bytes);

    if (this.carry?.length) {
      const joined = new Uint8Array(this.carry.length + raw.length);
      joined.set(this.carry, 0);
      joined.set(raw, this.carry.length);
      raw = joined;
      this.carry = null;
    }

    // A trailing odd byte is half a sample — hold it for the next chunk.
    const usable = raw.length - (raw.length % 2);
    if (usable < raw.length) this.carry = raw.slice(usable);
    if (usable === 0) return;

    // `raw` may not be 2-byte aligned in its underlying buffer, so copy rather
    // than viewing: an unaligned Int16Array constructor throws.
    const samples = new Int16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + usable));

    this.pending.push(samples);
    this.pendingFrames += samples.length;

    if (this.pendingFrames / this.sampleRate >= MIN_BLOCK_SECONDS) this.flushPending();
  }

  /** The segment is complete — schedule whatever is left of it. */
  close() {
    this.flushPending();
  }

  /** Barge-in. Everything scheduled stops now, not at the end of the buffer. */
  stop() {
    this.accepting = false;
    this.pending = [];
    this.pendingFrames = 0;
    this.carry = null;

    for (const source of this.sources) {
      try {
        source.onended = null;
        source.stop();
      } catch {
        // already finished — nothing to stop
      }
    }
    this.sources.clear();
    this.nextTime = 0;
    this.settle();
  }

  /** Tear the context down for good — the hook does this on unmount. */
  async dispose(): Promise<void> {
    this.stop();
    await this.context?.close().catch(() => {});
    this.context = null;
  }

  get playing(): boolean {
    return this.sources.size > 0;
  }

  /** Seconds of audio still queued ahead of the clock. */
  get buffered(): number {
    if (!this.context) return 0;
    return Math.max(0, this.nextTime - this.context.currentTime);
  }

  // ---- internals --------------------------------------------------------

  private flushPending() {
    const context = this.context;
    const gain = this.gain;
    if (!context || !gain || this.pendingFrames === 0) return;

    const frames = this.pendingFrames;
    const buffer = context.createBuffer(1, frames, this.sampleRate);
    const channel = buffer.getChannelData(0);

    let offset = 0;
    for (const block of this.pending) {
      for (let i = 0; i < block.length; i++) {
        // Int16 → Float32. 32768 on the negative side, 32767 on the positive:
        // dividing both by the same number clips the loudest sample.
        const sample = block[i];
        channel[offset + i] = sample < 0 ? sample / 32768 : sample / 32767;
      }
      offset += block.length;
    }

    this.pending = [];
    this.pendingFrames = 0;

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(gain);

    // Behind the clock means the gap is real and already lost — restart from
    // now rather than scheduling into the past, which plays nothing at all.
    const start = Math.max(this.nextTime, context.currentTime + LEAD);
    source.start(start);
    this.nextTime = start + buffer.duration;

    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      this.settle();
    };

    this.onState?.("playing");
    this.meter();
  }

  /** Report idle once the last scheduled block has actually finished. */
  private settle() {
    if (this.sources.size > 0) return;

    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;

    this.onLevel?.(0, Array<number>(this.bandCount).fill(0.06));
    this.onState?.("idle");
  }

  /** Drive the orb from what is being played, the way the mic drives it. */
  private meter() {
    if (this.frame !== null || !this.analyser || !this.spectrum || !this.onLevel) return;

    const analyser = this.analyser;
    const spectrum = this.spectrum;
    const usable = Math.floor(spectrum.length * 0.55);
    const band = Math.max(1, Math.floor(usable / this.bandCount));

    const tick = () => {
      analyser.getByteFrequencyData(spectrum);

      let peak = 0;
      const bands = Array.from({ length: this.bandCount }, (_, i) => {
        let sum = 0;
        for (let j = 0; j < band; j++) sum += spectrum[i * band + j] ?? 0;
        const shaped = Math.min(1, Math.pow(sum / band / 255, 0.62) * 1.5);
        peak = Math.max(peak, shaped);
        return Math.max(0.06, shaped);
      });

      this.onLevel?.(peak, bands);
      this.frame = this.sources.size > 0 ? requestAnimationFrame(tick) : null;
      if (this.frame === null) this.settle();
    };

    this.frame = requestAnimationFrame(tick);
  }
}
