# 11 — The voice loop

*Speak in any language; be answered out loud in the same one.*

This is what the app does now. The retrieval pipeline in 01–10 runs for a listener who has
**attached a vector store** and for nobody else: there is no deployment corpus and no switch
that turns retrieval on, so a spoken turn is grounded when there is somewhere to ground it
in and answers conversationally when there is not — [where that is
decided](#where-retrieval-plugs-in) is one method.

## The shape

```
mic ─► Saaras ─► transcript ─► chat model ─┬─► delta events  (text on screen)
                                           └─► segmenter ─► Bulbul ─► PCM ─► speakers
```

Four provider calls, three of them streaming, and no stage waits for the one before it to
*finish* — only for it to produce something usable:

| Stage | Waits for | Not for |
| --- | --- | --- |
| upload | the next 250 ms of audio | the end of the take |
| segmenter | the first clause | the finished reply |
| synthesiser | the first clause | the second one |
| player | the first PCM chunk | the segment |

That is the whole design. A version of this that awaited each stage would be four latencies
in series with silence over all of them.

## Why it is not one request

The obvious build — record, POST, get an MP3 back, play it — fails on three counts, and
each one is audible:

1. **A compressed file has to arrive whole before it decodes.** `decodeAudioData` cannot
   start on half an MP3, so the download sits in front of playback and the streaming above
   buys nothing. Raw PCM has no header and no framing: byte *N* is playable without byte
   *N+1*, which is why both providers here are asked for `linear16`/`pcm` and the browser
   schedules blocks as they land.
2. **The reply cannot be spoken until it is written.** Generation is ~1.3 s for three
   sentences; the first *clause* exists at ~600 ms. Splitting the token stream into
   speakable segments is worth ~700 ms of silence, every turn.
3. **Interruption has nowhere to live.** Talking over an answer means saying "stop" while
   the server is mid-send, and a request/response channel has no channel for that. Hence a
   WebSocket, and hence [barge-in](#barge-in) working at all.

## Measured

On this machine, against Sarvam, in Hindi and Tamil. Milliseconds from the top of the turn:

| | typed turn | spoken turn |
| --- | --- | --- |
| transcript (Saaras) | — | 841 |
| first token | 286 | 1,072 |
| first speakable segment | 616 | 1,534 |
| **first audio out** | **950** | **1,879** |
| reply fully written | 1,261 | 2,251 |

The number that decides how this feels is *first audio out* — everything before it is a
listener waiting in silence. Note the shape of it: the reply is still being written 300 ms
*after* the speakers start, and the listener never knows.

Two constants set the floor. Bulbul's time-to-first-byte is ~330 ms, and the first segment
cannot be cut before the model has written a clause. Nothing else in the path is
significant: the segmenter is string arithmetic, and the browser schedules PCM in ~50 ms
blocks.

## Language

Nobody picks a language before speaking, so nothing in the UI asks. Saaras is called with
`language_code=unknown` and its answer routes the rest of the turn:

| | covers | used for |
| --- | --- | --- |
| Sarvam Saaras | 22 Indic languages + English | hearing, always |
| Sarvam Bulbul | 11 (bn gu hi kn ml mr od pa ta te en) | speaking, when the language is one of them |
| OpenAI Whisper | ~100 | hearing, only when Saaras fails |
| OpenAI TTS | ~100 | speaking, everything Bulbul does not cover |

The detected code is also what the system prompt names — "reply in Tamil", not "reply in
`ta-IN`" — because a model given a code will occasionally answer *about* the code. Codes
outside the table fall back to "the same language the user spoke", which is the instruction
that always holds. The prompt is rebuilt every turn rather than pinned once, so switching
language mid-conversation switches the reply with it.

The reply model is whichever key is present: OpenAI if `OPENAI_API_KEY` is set, else
Sarvam's own chat model, which speaks the same chat-completions protocol. A checkout with
nothing but `SARVAM_API_KEY` holds a full conversation.

## The wire

One socket, two channels. JSON frames carry everything readable; binary frames carry PCM
for whichever segment the last `speech.start` announced — which is why the format lives on
that event and not on the audio, and why nothing pays a third of its size in base64.

```
→ audio.start {mime}      ← status   {stage: transcribing}
→ <binary …>              ← activity {step: stt, state: running}   … many
→ audio.end               ← transcript {text, languageCode}
                          ← status   {stage: thinking}
                          ← activity {step: tool, state: done}    … many
                          ← tool     {slug, arguments, result}    … many
                          ← delta   {text}                    … many
                          ← speech.start {segment, sampleRate}
                          ← <binary …>                        … many
                          ← speech.end   {segment}
                          ← turn.end {reply, timings}
```

`→ text` asks the same question typed, `→ cancel` interrupts, `→ reset` forgets the
conversation. Contract: [`src/schemas/voice.py`](../src/schemas/voice.py) and its mirror
[`frontend/src/lib/voice-protocol.ts`](../frontend/src/lib/voice-protocol.ts).

## Saying what it is doing

`status` has four values and drives the orb. That is the right size for an orb and the
wrong size for a listener wondering why an answer is taking three seconds — "thinking"
covers retrieval, the model, and anything between them, three things with very different
reasons for being slow.

So every seam in a turn also emits an `activity` frame: `{step, state, label, detail, ms}`,
where `step` is one of `stt · retrieval · tool · llm · speech · turn`, `state` runs
`start → running → done` (or `skipped`, or `error`), and `label` is the sentence the client
renders — "Searching the corpus", "Model is writing the reply", "Speaking part 2".

The sentence is written server-side on purpose. It is what lets a step that does not exist
yet — a query against a warehouse, a rung of the ladder — appear in the log the day the
backend starts emitting it, with no matching client release. The client keys rows by
`turn + step`, so a step is one line changing state rather than two lines saying the same
thing.

Cost is a few small JSON frames on a socket that is already carrying PCM. The pipeline is
untouched: no `activity` frame is ever awaited before the work it describes.

## The tool card

`turn + step` is the right key for a pipeline step and the wrong one for tools. A round
that runs three of them emits four `tool` frames — one opening the round, one per call —
and folding those into a single row leaves the log saying whichever tool reported last.
"Three ran, the second one failed" is the whole story of a slow turn, and one row can tell
a third of it.

So the same frames are folded a second time, per call, into the card above the transcript
in the bottom-left stack. There is no call id on the wire, so the pairing is by name: the
round opens with `state: "start"` and `detail` listing the names the model chose, and each
call then reports `done` or `error` under its own name, in the order the server ran them.
Matching the *first still-running* row with that name keeps one tool called twice in a
round as two rows.

Durations come out of the same absence. Every `ms` on the wire is elapsed-since-the-turn-
started, not a duration, and tools in a round are awaited one at a time — so a call's own
time is the gap between its frame and the one before it, and each finish is also the next
call's start. A call still running when the turn ends is marked `skipped` rather than left
pulsing, for the same reason the log settles its rows.

The card collapses to its summary line — how many ran, how long they took, whether any
failed — because it is a detail of an answer that is spoken, not the answer. It is
anchored *above* the transcript, which keeps the floor: the stack grows upward into empty
space, so neither opening the card nor shutting it can move the orb.

## The call itself, for the thread

The stage card is drawn from `activity`, and that is the ceiling of what a step-shaped
frame can say: a name, a state, a duration. The **Conversations** panel wants the other
thing — what the agent sent and what came back — because "it ran `GMAIL_FETCH_EMAILS`"
says nothing about what the answer was built from, and an email being sent is the one part
of a spoken turn with an effect outside the app.

That does not fit in a `label` and a `detail`, so a finished call also goes out as its own
`tool` frame carrying the whole of it: `{id, turnId, toolkit, slug, arguments, status, ok,
error, result, resultBytes, latencyMs}`. Three things about it are deliberate:

- **It is the stored shape.** Field for field, it is the `ToolCall` that
  `GET /conversations/{id}` returns ([`src/schemas/chat.py`](../src/schemas/chat.py)), and
  the panel renders a call heard live and a call read back out of Postgres with one
  component. Two near-identical shapes would drift a field at a time.
- **It carries the id the row will be written under.** The id is minted before the write
  and put on the wire, so a call already on screen and the same call arriving in a fetched
  thread are one entry rather than the same tool listed twice.
- **It is built from the bounded values** — the trimmed arguments and the marked result
  preview the column will hold, not the raw ones. The frame and the row say the same thing
  about the same call, and a provider's inbox page cannot be pushed down the socket whole.

It is sent whether or not there is a database. A deployment with no `DATABASE_URL` still
shows what it just ran; what it loses is reading it back tomorrow.

Two frames for one call is not duplication — they are folded for two readers with two
different needs. The stage card exists *from the moment a call starts*, which is what lets
it say "running"; the `tool` frame only exists once there is a result to report.

## Barge-in

Talking over the assistant stops it. Three things have to happen in the right order, and
the order is the whole feature:

1. the browser kills the scheduled audio **first**, so the silence is immediate;
2. it sends `cancel`, and the server cancels the turn — mid-token, mid-synthesis;
3. audio still in flight for that turn is dropped rather than played, because the socket
   round trip is slower than the listener's next breath.

What was actually *spoken* is what goes into the history — not what was generated. A model
that believes it said three sentences the listener never heard will refer back to them.

## Silence

A take ends when the speaker stops talking: the analyser watches the input level and closes
the recording after ~1.1 s of quiet, with a floor on speech length so a cough cannot end a
take before it holds anything. Measured against a 2.9 s question: 3.2 s of listening, then
it closes itself. Waiting for a second tap is a walkie-talkie, not a conversation.

The microphone does not reopen by itself once the answer ends. Every turn starts with a tap
— on the orb or on the space bar — and that is the whole trade: with speakers up and a room
full of people, a mic that re-arms itself is picking up the room, not the speaker.

The pill above the orb says which of the three things is happening: **Tap to speak** at rest,
**Listening** with the take counting down, and **Streaming** for everything from the upload to
the last word spoken — one continuous thing to a listener, whatever the wire is doing.

## Where retrieval plugs in

`VoiceSession._retrieve()` in
[`src/services/voice_service.py`](../src/services/voice_service.py). It asks the resolver
whether this listener has a vector store attached; if they do, it calls the same
`AskService` measured in [09-v1.md](09-v1.md), and the retrieved passages become the context
the reply is grounded in — an abstention included, which the prompt turns into "tell the
user you do not have this" rather than an invented answer.

If they do not, it returns `None` and the turn is answered by the model and by whatever
tools they *have* connected. That is a per-listener decision and not a deployment one: the
resolver is the same object `AskService` uses, so the two cannot disagree about whether
there is anywhere to look, and a listener who attaches Pinecone gets grounded turns on the
next question without anything being restarted.

One thing to hold onto about the numbers: the 200 ms budget in
[04-latency.md](04-latency.md) applies to the retrieval call only — a spoken reply is a
network round trip to a language model and was never going to fit. Boot pays for the ONNX
session on every start, because it cannot know who is about to sign in.
