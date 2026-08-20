"use client";

import { createContext, useCallback, useContext, useMemo } from "react";
import type { ReactNode } from "react";

import { usePersistentState } from "@/hooks/use-persistent-state";
import type { AskStatus } from "@/lib/rag";
import type { Transcript } from "@/lib/types";

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
   * What the agent said back. Null until the reply lands — and it stays null
   * when the pipeline abstained, because there is no answer to show.
   * `replyStatus` is what distinguishes "still waiting" from "declined".
   */
  reply: string | null;
  /**
   * How the pipeline ended. `abstained` is a success — the corpus couldn't
   * support an answer — so it renders as a real turn, not an error.
   */
  replyStatus: AskStatus | null;
  /** Why it abstained or refused, in the agent's own words. */
  replyReason: string | null;
  /** Server-side pipeline milliseconds, for the latency read-out. */
  replyMs: number | null;
};

const STORAGE_KEY = "vec-turns";

/** The panels only ever show the recent tail — keep storage from growing. */
const MAX_TURNS = 50;

/** Stable empty reference: the store hands this back on every unstored read. */
const NO_TURNS: Turn[] = [];

const REPLY_STATUSES: AskStatus[] = ["answered", "abstained", "refused"];

function isTurn(value: unknown): value is Turn {
  if (typeof value !== "object" || value === null) return false;
  const turn = value as Record<string, unknown>;

  return (
    typeof turn.id === "string" &&
    typeof turn.at === "number" &&
    typeof turn.text === "string"
  );
}

function reviveStatus(raw: unknown): AskStatus | null {
  return REPLY_STATUSES.includes(raw as AskStatus) ? (raw as AskStatus) : null;
}

function reviveTurns(raw: unknown): Turn[] | null {
  if (!Array.isArray(raw)) return null;

  return raw.filter(isTurn).map((turn) => ({
    id: turn.id,
    at: turn.at,
    text: turn.text,
    languageCode: turn.languageCode ?? null,
    reply: turn.reply ?? null,
    replyStatus: reviveStatus(turn.replyStatus),
    replyReason: turn.replyReason ?? null,
    replyMs: typeof turn.replyMs === "number" ? turn.replyMs : null,
  }));
}

/** Everything the answer stage writes back onto a turn. */
export type Reply = Pick<Turn, "reply" | "replyStatus" | "replyReason" | "replyMs">;

type ConversationValue = {
  turns: Turn[];
  /**
   * Append a finished transcript, returning the new turn's id so the caller
   * can attach the answer to it once the pipeline comes back.
   */
  record: (transcript: Transcript) => string | null;
  /** Fill in the reply slot on an already-recorded turn. */
  answer: (id: string, reply: Reply) => void;
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
      if (!text) return null;

      const id = crypto.randomUUID();

      setTurns((previous) =>
        [
          ...previous,
          {
            id,
            at: Date.now(),
            text,
            languageCode: transcript.languageCode,
            reply: null,
            replyStatus: null,
            replyReason: null,
            replyMs: null,
          },
        ].slice(-MAX_TURNS),
      );

      return id;
    },
    [setTurns],
  );

  const answer = useCallback(
    (id: string, reply: Reply) => {
      setTurns((previous) =>
        previous.map((turn) => (turn.id === id ? { ...turn, ...reply } : turn)),
      );
    },
    [setTurns],
  );

  const clear = useCallback(() => setTurns(NO_TURNS), [setTurns]);

  const value = useMemo(
    () => ({ turns, record, answer, clear }),
    [turns, record, answer, clear],
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
