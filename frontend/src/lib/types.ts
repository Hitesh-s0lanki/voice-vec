/** Visual states of the orb — it listens, thinks, then speaks back. */
export type VoiceState = "idle" | "listening" | "thinking" | "speaking";

/** What was heard, and in what language. Null means nobody could tell. */
export type Transcript = {
  text: string;
  languageCode: string | null;
};
