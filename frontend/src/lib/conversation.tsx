"use client";

import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";

import { useAdoption } from "@/hooks/use-adoption";
import {
  conversationIdFrom,
  isConversationId,
  readConversation,
  type ChatMessage,
  type MessageStatus,
  type ToolCall,
} from "@/lib/conversations";
import type { Transcript } from "@/lib/types";

/**
 * The conversation on screen.
 *
 * It used to be a list in `localStorage`, which meant a refresh was the end of
 * it and a second device knew nothing. Now the rows live in Postgres and this
 * holds two things at once: what the server has stored for `/c/{id}`, loaded
 * when the URL names one, and what is being said right now, appended live as
 * the socket reports it. The two never fight, because whichever of them the
 * provider is already holding for a conversation wins — see `held` below.
 *
 * Nothing here writes. The voice socket is what saves a turn, as it happens;
 * this is the read side plus an optimistic copy of the turn in flight.
 */
export type Turn = {
  /** The voice turn id — the same one the backend filed the messages under. */
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
   * How the turn ended. `abstained` is a success — the corpus couldn't
   * support an answer — and `interrupted` means you talked over it, so both
   * render as real turns rather than errors.
   */
  replyStatus: MessageStatus | null;
  /** Why it abstained or refused, in the agent's own words. */
  replyReason: string | null;
  /** Server-side pipeline milliseconds, for the latency read-out. */
  replyMs: number | null;
  /**
   * What the agent ran during this turn, in the order it ran.
   *
   * Empty for every turn that called nothing, which is most of them — a turn
   * only has these once the speaker has linked a toolkit through Composio.
   */
  tools: ToolCall[];
};

/** Stable empty reference, so an unloaded conversation re-renders nothing. */
const NO_TURNS: Turn[] = [];

/** The same, for a turn that ran nothing — which is most of them. */
const NO_TOOLS: ToolCall[] = [];

/**
 * Add a call to a list, ignoring one that is already there.
 *
 * Idempotent by id, and that is what the backend minting the id before it
 * writes the row buys: the call the socket announced mid-turn and the call a
 * later fetch returns are the same entry rather than the same tool twice.
 */
function including(calls: ToolCall[], call: ToolCall): ToolCall[] {
  return calls.some((existing) => existing.id === call.id) ? calls : [...calls, call];
}

/**
 * Fold a stored thread back into turns.
 *
 * Messages come out of Postgres flat and in order; a turn is a question and
 * the answer that followed it. `turnId` is what pairs them — both rows carry
 * the id of the voice turn they belong to — and the fallbacks matter more than
 * they look: a take that failed at transcription leaves a question with no
 * answer, and a socket that died mid-reply can leave an answer with no
 * question. Both have to survive the fold or the panel silently loses them.
 */
function thread(messages: ChatMessage[], toolCalls: ToolCall[] = []): Turn[] {
  const turns: Turn[] = [];

  for (const message of messages) {
    if (message.role === "user") {
      turns.push({
        id: message.turnId ?? message.id,
        at: Date.parse(message.createdAt),
        text: message.text,
        languageCode: message.languageCode,
        reply: null,
        replyStatus: null,
        replyReason: null,
        replyMs: null,
        tools: [],
      });
      continue;
    }

    const open = turns.at(-1);
    const belongs =
      open !== undefined &&
      open.reply === null &&
      open.replyStatus === null &&
      (message.turnId === null || message.turnId === open.id);

    if (belongs) {
      open.reply = message.text;
      open.replyStatus = message.status;
      open.replyReason = message.reason;
      open.replyMs = message.latencyMs;
      continue;
    }

    // An answer with nothing to attach to. Rare, and better shown orphaned
    // than dropped — an empty question reads as "we lost the take".
    turns.push({
      id: message.turnId ?? message.id,
      at: Date.parse(message.createdAt),
      text: "",
      languageCode: message.languageCode,
      reply: message.text,
      replyStatus: message.status,
      replyReason: message.reason,
      replyMs: message.latencyMs,
      tools: [],
    });
  }

  // Tool calls arrive as their own flat list and are attached by `turnId`,
  // the same key that pairs a question to its answer. A call whose turn is not
  // on screen is dropped rather than orphaned: unlike a stray message it says
  // nothing on its own, and the turn it belongs to may simply be past the
  // message limit.
  if (toolCalls.length > 0) {
    const byTurn = new Map(turns.map((turn) => [turn.id, turn]));
    for (const call of toolCalls) {
      const turn = call.turnId ? byTurn.get(call.turnId) : undefined;
      if (turn) turn.tools.push(call);
    }
  }

  return turns;
}

/** Everything the answer stage writes back onto a turn. */
export type Reply = Pick<Turn, "reply" | "replyStatus" | "replyReason" | "replyMs">;

type ConversationValue = {
  /** The conversation the URL is showing, or null on a page with none yet. */
  conversationId: string | null;
  turns: Turn[];
  /** A stored thread is on its way. Only ever true right after a reload. */
  loading: boolean;
  /**
   * Append a finished transcript, returning the turn's id so the caller can
   * attach the answer to it once the reply comes back.
   */
  record: (transcript: Transcript, id?: string) => string | null;
  /** Fill in the reply slot on an already-recorded turn. */
  answer: (id: string, reply: Reply) => void;
  /**
   * File one finished tool call under the turn that ran it.
   *
   * Held aside when that turn is not on screen yet, which is the normal case
   * rather than the edge: tools run in the middle of a turn and the turn is
   * recorded at the end of it, so a call almost always arrives first.
   */
  attach: (call: ToolCall) => void;
  /**
   * The socket just opened a conversation for what is being said. Puts
   * `/c/{id}` in the address bar without a navigation — a route change here
   * would tear down the socket mid-sentence.
   */
  adopt: (id: string) => void;
  /** Leave this conversation for a new one. The old one stays where it is. */
  reset: () => void;
};

const ConversationContext = createContext<ConversationValue | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const conversationId = conversationIdFrom(pathname);

  // Signing in claims what this browser said before it did. This is the
  // component that owns "which conversations are mine", so it is where the
  // answer changing gets noticed.
  useAdoption();

  /**
   * The turns on screen, and which conversation they belong to.
   *
   * One piece of state rather than two, because "these turns" and "of that
   * conversation" have to change together: a list that outlives the id it was
   * loaded for would show the last conversation's thread under this one's URL
   * for exactly one render, which is all it takes to be seen.
   *
   * `id: null` is the real conversation on `/` — the one being spoken now,
   * which has no address until the server gives it one.
   */
  const [state, setState] = useState<{
    id: string | null;
    turns: Turn[];
    /**
     * Tool calls waiting for the turn that ran them, keyed by turn id.
     *
     * The socket announces a call the moment it finishes, and the turn itself
     * is recorded when the reply lands — so for a turn that ran anything, the
     * calls are here first and `record` collects them on the way past. They
     * live in state rather than a ref because dropping one is the bug being
     * fixed and a ref would let a render pass go by without noticing it.
     */
    pending: Record<string, ToolCall[]>;
  }>({
    id: null,
    turns: NO_TURNS,
    pending: {},
  });

  /**
   * The conversation whose turns we already hold, fetch or no fetch.
   *
   * This is what keeps a live conversation from being reloaded out from under
   * itself: `adopt` sets it before the URL changes, so the effect below sees
   * its own id arrive and does nothing. Every other way the id can change — a
   * reload, a click in the rail — leaves it stale, and the thread is fetched.
   */
  const held = useRef<string | null>(null);

  const mine = state.id === conversationId;
  const turns = mine ? state.turns : NO_TURNS;

  useEffect(() => {
    if (!conversationId || held.current === conversationId) return;

    const abort = new AbortController();

    void readConversation(conversationId, abort.signal).then((detail) => {
      if (abort.signal.aborted) return;

      held.current = conversationId;
      // A conversation that is gone, or was never this browser's, comes back
      // null. Showing it empty is the honest answer — and the next thing said
      // opens a new one, so the page is never stuck.
      //
      // `pending` starts over with it: this only runs for a conversation the
      // provider was not already holding, so anything buffered belongs to the
      // one being left, and the stored rows are the authority for this one.
      setState({
        id: conversationId,
        turns: detail ? thread(detail.messages, detail.toolCalls) : NO_TURNS,
        pending: {},
      });
    });

    return () => abort.abort();
  }, [conversationId]);

  const record = useCallback(
    (transcript: Transcript, id?: string) => {
      const text = transcript.text.trim();
      if (!text) return null;

      const turnId = id ?? crypto.randomUUID();
      const at = Date.now();

      setState((previous) => {
        // Whatever the socket already said this turn ran. Taken out of the
        // buffer as the turn is filed, so the calls stop being held for a turn
        // that now exists — the two halves of what the panel shows under one
        // exchange arrive from opposite ends of it and meet here.
        const { [turnId]: ran = NO_TOOLS, ...waiting } = previous.pending;

        const fresh: Turn = {
          id: turnId,
          at,
          text,
          languageCode: transcript.languageCode,
          reply: null,
          replyStatus: null,
          replyReason: null,
          replyMs: null,
          tools: ran,
        };

        // Recording into a different conversation than the list holds starts
        // the list over — that is what leaving one for another means. The
        // buffer goes with it: what is left in it belongs to turns of the
        // conversation being left, which are no longer on screen to hang it on.
        if (previous.id !== conversationId) {
          return { id: conversationId, turns: [fresh], pending: {} };
        }

        // Idempotent by id. A turn reported twice — a barge-in that the server
        // then also closed — must not become two rows in the panel.
        if (previous.turns.some((turn) => turn.id === turnId)) {
          return {
            id: previous.id,
            pending: waiting,
            turns: previous.turns.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    text,
                    languageCode: transcript.languageCode,
                    // Merged rather than replaced: the first report may already
                    // have collected calls this one would drop on the floor.
                    tools: ran.reduce(including, turn.tools),
                  }
                : turn,
            ),
          };
        }

        return { id: previous.id, turns: [...previous.turns, fresh], pending: waiting };
      });

      return turnId;
    },
    [conversationId],
  );

  const answer = useCallback((id: string, reply: Reply) => {
    setState((previous) => ({
      ...previous,
      turns: previous.turns.map((turn) =>
        turn.id === id ? { ...turn, ...reply } : turn,
      ),
    }));
  }, []);

  /**
   * One tool call, filed under its turn.
   *
   * The turn is usually not here yet — a call finishes in the middle of an
   * exchange and the exchange is recorded at the end of it — so the miss is
   * the normal path and buffering it is the point. A call with no turn id is
   * dropped: on its own it says nothing the thread can place.
   */
  const attach = useCallback((call: ToolCall) => {
    const turnId = call.turnId;
    if (!turnId) return;

    setState((previous) => {
      if (!previous.turns.some((turn) => turn.id === turnId)) {
        return {
          ...previous,
          pending: {
            ...previous.pending,
            [turnId]: including(previous.pending[turnId] ?? NO_TOOLS, call),
          },
        };
      }

      return {
        ...previous,
        turns: previous.turns.map((turn) =>
          turn.id === turnId ? { ...turn, tools: including(turn.tools, call) } : turn,
        ),
      };
    });
  }, []);

  const adopt = useCallback((id: string) => {
    if (!isConversationId(id)) return;

    // Claimed before the URL moves, so the effect above recognises it as ours
    // and the turns already on screen are not fetched back over themselves.
    held.current = id;
    setState((previous) => (previous.id === id ? previous : { ...previous, id }));

    const next = `/c/${id}`;
    if (window.location.pathname === next) return;

    // `replaceState` and not `router.replace`: this is the same conversation
    // it always was, only now it has an address. A real navigation would
    // remount the page, close the socket, and cut off the reply being spoken.
    window.history.replaceState(null, "", next);
  }, []);

  const reset = useCallback(() => {
    held.current = null;
    setState({ id: null, turns: NO_TURNS, pending: {} });
    router.push("/");
  }, [router]);

  const value = useMemo(
    () => ({
      conversationId,
      turns,
      // Only ever true for the moment after a reload, while the stored thread
      // is in flight. On `/` there is nothing to load and nothing to wait for.
      loading: conversationId !== null && !mine,
      record,
      answer,
      attach,
      adopt,
      reset,
    }),
    [conversationId, turns, mine, record, answer, attach, adopt, reset],
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
