"use client";

import { createContext, useCallback, useContext, useMemo } from "react";
import type { ReactNode } from "react";

import { usePersistentState } from "@/hooks/use-persistent-state";
import type { Transcript } from "@/hooks/use-voice-capture";

/**
 * One exchange. The History and Conversations panels read the same list — one
 * as a flat log of takes, the other as the back-and-forth it belongs to.
 */
export type Turn = {
  id: string;
  /** epoch ms */
  at: number;
  text: string;
  languageCode: string | null;
  /**
   * What the agent said back. Always `null` today: the app transcribes and
   * stops there, so nothing writes this yet. The Conversations panel renders
   * the empty slot rather than pretending the reply arrived.
   */
  reply: string | null;
};

const STORAGE_KEY = "vec-turns";

/** The panels only ever show the recent tail — keep storage from growing. */
const MAX_TURNS = 50;

/** Stable empty reference: the store hands this back on every unstored read. */
const NO_TURNS: Turn[] = [];

function isTurn(value: unknown): value is Turn {
  if (typeof value !== "object" || value === null) return false;
  const turn = value as Record<string, unknown>;

  return (
    typeof turn.id === "string" &&
    typeof turn.at === "number" &&
    typeof turn.text === "string"
  );
}

function reviveTurns(raw: unknown): Turn[] | null {
  if (!Array.isArray(raw)) return null;

  return raw.filter(isTurn).map((turn) => ({
    id: turn.id,
    at: turn.at,
    text: turn.text,
    languageCode: turn.languageCode ?? null,
    reply: turn.reply ?? null,
  }));
}

type ConversationValue = {
  turns: Turn[];
  /** Append a finished transcript. Called once per take, by the capture UI. */
  record: (transcript: Transcript) => void;
  clear: () => void;
};

const ConversationContext = createContext<ConversationValue | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [turns, setTurns] = usePersistentState<Turn[]>(
    STORAGE_KEY,
    NO_TURNS,
    reviveTurns,
  );

  const record = useCallback(
    (transcript: Transcript) => {
      const text = transcript.text.trim();
      if (!text) return;

      setTurns((previous) =>
        [
          ...previous,
          {
            id: crypto.randomUUID(),
            at: Date.now(),
            text,
            languageCode: transcript.languageCode,
            reply: null,
          },
        ].slice(-MAX_TURNS),
      );
    },
    [setTurns],
  );

  const clear = useCallback(() => setTurns(NO_TURNS), [setTurns]);

  const value = useMemo(
    () => ({ turns, record, clear }),
    [turns, record, clear],
  );

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversation() {
  const value = useContext(ConversationContext);
  if (!value) {
    throw new Error("useConversation must be used inside a ConversationProvider");
  }
  return value;
}
