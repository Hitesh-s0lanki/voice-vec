# 11 — The voice loop

*Speak in any language; be answered out loud in the same one.*

This is what the app does now. The retrieval pipeline in 01–10 is built, measured and
**switched off** (`RAG_ENABLED=false`): the spoken turn answers conversationally, and
[the seam where retrieval comes back](#where-retrieval-plugs-back-in) is one method.

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
yet — a tool call, a query against a warehouse — appear in the log the day the backend
starts emitting it, with no matching client release. `tool` is in the set for that reason
and nothing emits it yet. The client keys rows by `turn + step`, so a step is one line
changing state rather than two lines saying the same thing.

Cost is a few small JSON frames on a socket that is already carrying PCM. The pipeline is
untouched: no `activity` frame is ever awaited before the work it describes.

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

Hands-free (off by default) goes one step further and reopens the microphone as soon as the
answer finishes. It is off by default because it is a real trade — with speakers up and a
room full of people, the tap is the feature.

## Where retrieval plugs back in

`VoiceSession._retrieve()` in
[`src/services/voice_service.py`](../src/services/voice_service.py). With `RAG_ENABLED=true`
it calls the same `AskService` measured in [09-v1.md](09-v1.md), and the retrieved passages
become the context the reply is grounded in — an abstention included, which the prompt
turns into "tell the user you do not have this" rather than an invented answer. Everything
downstream of that call already exists; the switch is one environment variable.

Two things change when it goes on, and both are wanted: the 200 ms budget in
[04-latency.md](04-latency.md) applies to the retrieval call only — a spoken reply is a
network round trip to a language model and was never going to fit — and boot pays for the
ONNX session and the index again, which is why the lifespan skips both while it is off.
