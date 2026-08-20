"""The voice contract.

A turn is a stream of events, not a response, so the shape that matters is the
event union below rather than a single body. Two channels share one WebSocket:
JSON frames carry everything you can read, binary frames carry PCM. A binary
frame is always audio for the segment whose `speech.start` most recently
arrived — which is why that event carries the format instead of each chunk.

    → audio.start {mime}     ← status  {stage: transcribing}
    → <binary …>             ← activity {step: stt, state: start}
    → audio.end              ← transcript {text, languageCode}
                             ← status  {stage: thinking}
                             ← activity {step: llm, state: start}  … many
                             ← delta   {text}          … many
                             ← speech.start {segment, sampleRate, …}
                             ← <binary …>               … many
                             ← speech.end   {segment}
                             ← turn.end {reply, timings}

`status` says which of four stages the turn is in; `activity` says what the
backend is doing inside one. They are separate because the stage drives the
orb and the activity line drives a log — one has to stay a closed set of four,
the other has to stay open enough to add a step to.
"""

from typing import Literal

from pydantic import Field

from src.schemas.wire import Wire


class VoiceTimings(Wire):
    """Milliseconds from the top of the turn. null means it never happened.

    `first_audio` is the number that decides whether this feels like a
    conversation: everything before it is the listener waiting in silence.
    """

    stt: float | None = None
    first_token: float | None = None
    first_segment: float | None = None
    first_audio: float | None = None
    reply: float | None = None
    total: float


# ---- server → client ----------------------------------------------------


class Providers(Wire):
    """Who is actually wired up, so the client can say so before it fails."""

    stt: str | None
    llm: str | None
    llm_model: str | None
    tts: str | None
    rag_enabled: bool


class Ready(Wire):
    type: Literal["ready"] = "ready"
    session_id: str
    providers: Providers
    sample_rate: int
    languages: dict[str, str] = Field(description="Language code → English name")


class Status(Wire):
    type: Literal["status"] = "status"
    stage: Literal["idle", "transcribing", "thinking", "speaking"]
    turn_id: str | None = None


ActivityStep = Literal["stt", "retrieval", "tool", "llm", "speech", "turn"]

ActivityState = Literal["start", "running", "done", "skipped", "error"]


class Activity(Wire):
    """One line of what the backend is doing, while it is doing it.

    Deliberately not the payload — the *step*. "Searching the corpus", not the
    passages; "the model is writing", not the tokens. Those already have their
    own events, and a listener staring at an orb has no way to tell a slow
    retrieval from a slow model without this.

    `step` and `state` are for the client's icons and dots, `label` is the
    sentence it renders. Writing the sentence here rather than in the browser
    is what lets a new step — a tool call, a SQL query — show up in the log
    without shipping a matching client. `tool` is in the set for exactly that
    reason; nothing emits it until there is a tool to emit it for.
    """

    type: Literal["activity"] = "activity"
    turn_id: str | None = None
    step: ActivityStep
    state: ActivityState = "start"
    label: str = Field(description="Short, human, present tense: 'Searching the corpus'")
    detail: str | None = Field(default=None, description="Who or what — provider, model, count")
    ms: float | None = Field(default=None, description="Elapsed at this point in the turn")


class TranscriptEvent(Wire):
    type: Literal["transcript"] = "transcript"
    turn_id: str
    text: str
    language_code: str | None = None
    language: str | None = Field(default=None, description="English name of the language")
    confidence: float | None = None
    provider: str | None = None
    ms: float | None = None


class Delta(Wire):
    """A piece of the reply as it is written. Text only — audio is binary."""

    type: Literal["delta"] = "delta"
    turn_id: str
    text: str


class SpeechStart(Wire):
    type: Literal["speech.start"] = "speech.start"
    turn_id: str
    segment: int
    text: str
    provider: str
    voice: str
    language_code: str | None = None
    sample_rate: int
    format: Literal["pcm_s16le"] = "pcm_s16le"


class SpeechEnd(Wire):
    type: Literal["speech.end"] = "speech.end"
    turn_id: str
    segment: int
    bytes: int
    ms: float


class TurnEnd(Wire):
    type: Literal["turn.end"] = "turn.end"
    turn_id: str
    reply: str
    language_code: str | None = None
    segments: int
    timings: VoiceTimings


class Canceled(Wire):
    """The turn was stopped on purpose — someone started talking over it."""

    type: Literal["canceled"] = "canceled"
    turn_id: str | None = None
    spoken: str | None = None


class VoiceError(Wire):
    type: Literal["error"] = "error"
    message: str
    turn_id: str | None = None
    stage: str | None = None
    provider: str | None = None


class Pong(Wire):
    type: Literal["pong"] = "pong"


# ---- REST (the same pipeline, one stage at a time) ----------------------


class SpeakRequest(Wire):
    """`POST /voice/speak` — text in, audio out. For testing a voice without
    holding a whole conversation."""

    text: str = Field(min_length=1, max_length=3000)
    language_code: str | None = None
    format: Literal["wav", "pcm"] = "wav"


class TranscribeResponse(Wire):
    transcript: str
    language_code: str | None = None
    language: str | None = None
    confidence: float | None = None
    provider: str | None = None
    ms: float
