"""Application settings, read once at import and reused everywhere."""

from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "voice-vec"
    version: str = "0.1.0"
    environment: str = "development"

    host: str = "127.0.0.1"
    # 8000 is already taken on this machine by another server; override via PORT.
    port: int = 8001

    # The Next.js app in frontend/ runs on 3002 (see frontend/package.json).
    cors_origins: list[str] = ["http://localhost:3002"]

    # ---- Voice pipeline -------------------------------------------------
    # Speech in, speech out. Every stage streams: the reply model is read token
    # by token, split into speakable segments, and each segment is synthesised
    # while the next is still being written (docs/11-voice.md).
    sarvam_api_key: str = ""
    openai_api_key: str = ""

    # The reply model. Anything speaking the OpenAI chat-completions protocol
    # works; both providers here do. Left empty, the base URL and model are
    # resolved from whichever key is present — OpenAI first, then Sarvam — so
    # the app runs with a Sarvam key alone and upgrades the moment
    # OPENAI_API_KEY appears. `resolve_llm()` is the one place that decides.
    llm_base_url: str = ""
    llm_model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.4-mini"
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    sarvam_model: str = "sarvam-105b-conversations"
    llm_temperature: float = 0.6
    # A spoken answer that runs long is worse than one that stops early: nobody
    # listens to four paragraphs read aloud. The prompt asks for brevity; this
    # is the hard stop behind it.
    llm_max_tokens: int = 400
    # How many past exchanges ride along as context. Each turn is two messages.
    llm_history_turns: int = 8

    # ---- Speech to text -------------------------------------------------
    # saaras:v3 + mode=transcribe is what the app already shipped with;
    # saaras:v4 covers 22 Indic languages plus global English but takes no
    # `mode`. `unknown` lets Sarvam detect the language — which is the whole
    # point here, since the speaker picks the language, not the UI.
    stt_model: str = "saaras:v3"
    stt_language: str = "unknown"
    # Sarvam's REST endpoint caps a request at 30 s of audio.
    stt_max_bytes: int = 6 * 1024 * 1024

    # ---- Text to speech -------------------------------------------------
    # Sarvam speaks the 11 languages bulbul covers; anything else (French,
    # Japanese, …) falls through to OpenAI, which has no Indic edge but does
    # have the coverage. Both are asked for raw little-endian 16-bit PCM at one
    # sample rate, so the browser has exactly one decode path.
    tts_model: str = "bulbul:v3"
    tts_speaker: str = "ritu"
    tts_pace: float = 1.0
    tts_sample_rate: int = 24_000
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"

    # Segment sizing, in characters. The first segment is deliberately short —
    # it is the one the listener waits on — and later ones run longer, because
    # a whole clause carries better prosody than a fragment.
    speech_first_segment_chars: int = 90
    speech_segment_chars: int = 220
    # Never let a run-on sentence hold the audio hostage: flush at a comma, or
    # failing that mid-clause, once the buffer passes this.
    speech_max_segment_chars: int = 320

    # How many segments may be at the synthesiser at once. 2 hides the next
    # segment's ~300 ms time-to-first-byte under the current one's playback;
    # more only buys buffer nobody hears.
    speech_lookahead: int = 2

    # ---- Retrieval (off) ------------------------------------------------
    # The RAG pipeline below is built and tested but switched off: the voice
    # loop answers conversationally for now. Turn this on and /ask comes back
    # and each spoken turn gets grounded in the corpus first — see
    # `VoiceService._retrieve`. Everything downstream of it already exists.
    rag_enabled: bool = False

    # ---- Embedding ------------------------------------------------------
    # multilingual-e5-small: 384 dims, ONNX, ~3 ms per query on CPU. The ONNX
    # export lives in the same HF repo as the torch weights (`onnx/model.onnx`).
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384
    embed_model_file: str = "onnx/model.onnx"
    # None lets onnxruntime pick. Pin it if batch-of-1 latency looks noisy.
    embed_threads: int | None = None
    embed_cache_dir: str = "data/models"

    # ---- Vector store ---------------------------------------------------
    # Empty URL means embedded mode: Qdrant runs in-process over `qdrant_path`.
    # That path is a single-writer lock — ingest and the API cannot hold it at
    # the same time. Point QDRANT_URL at a server to run both together.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "vec-chunks"

    # ---- Retrieval ------------------------------------------------------
    search_limit: int = 10
    # Which chunking strategies to query. v1 ingests S1 only (see docs/03-chunking.md).
    search_strategies: list[str] = ["S1"]

    # ---- Guardrails (docs/06-guardrails.md) -----------------------------
    # Gate 2, swept against the labelled abstention set by `scripts/evaluate.py`
    # over N=300 and set to the balanced operating point (docs/09-v1.md):
    #
    #   margin  abstention recall  answer coverage
    #   0.010   0.34                0.75      permissive
    #   0.020   0.64                0.52      balanced  ← default
    #   0.030   0.90                0.29      conservative
    #
    # The floor is nearly inert over 0.78–0.86 — e5 cosine scores are compressed
    # high, so almost nothing falls below it and the margin does the work. Both
    # are index-size dependent: re-sweep after any re-ingest.
    retrieval_floor: float = 0.845
    retrieval_margin: float = 0.02
    min_hits: int = 1
    min_transcript_chars: int = 3

    # ---- Extraction -----------------------------------------------------
    # Sentences are embedded at query time, and embedding is linear in sentence
    # count (~4.5 ms each). Both caps exist to keep that inside the budget.
    extract_passages: int = 3
    extract_rerank: int = 6
    extract_max_sentences: int = 2
    extract_max_chars: int = 480

    # ---- Latency --------------------------------------------------------
    # Requirement 3: transcript in, answer out, under 200 ms. The harness skips
    # optional stages rather than overrunning this.
    deadline_ms: int = 200
    metrics_buffer: int = 500

    # Warming the embedder once at boot is not enough. Measured on this machine:
    # after 30 s of no traffic the next request pays ~+30 ms on embed and ~+30 ms
    # on search, and a cold *answered* request measured 200.3 ms — over budget.
    # Back-to-back it is 78 ms. The ONNX arena and Qdrant's vector matrix go cold
    # when nothing touches them, and interactive voice use is all cold requests:
    # one question, then minutes of silence.
    #
    # So a tiny embed + search runs on this interval to keep both hot. ~12 ms of
    # work per tick — a 0.06% duty cycle at 20 s. Set 0 to disable and measure
    # the difference.
    keepalive_seconds: int = 20

    # ---- Resolved providers ---------------------------------------------

    @property
    def sarvam_ready(self) -> bool:
        return bool(self.sarvam_api_key)

    @property
    def openai_ready(self) -> bool:
        return bool(self.openai_api_key)

    def resolve_llm(self) -> "LlmTarget":
        """Which chat model answers, and where it lives.

        An explicit LLM_MODEL / LLM_BASE_URL wins. Otherwise OpenAI is
        preferred when its key is set and Sarvam stands in when it is not, so a
        checkout holding only SARVAM_API_KEY still holds a conversation.
        """
        if self.openai_ready:
            base, model, key = self.openai_base_url, self.openai_model, self.openai_api_key
        elif self.sarvam_ready:
            base, model, key = self.sarvam_base_url, self.sarvam_model, self.sarvam_api_key
        else:
            base, model, key = self.openai_base_url, self.openai_model, ""

        base_url = (self.llm_base_url or base).rstrip("/")
        provider = "sarvam" if "sarvam.ai" in base_url else "openai"
        if self.llm_base_url:
            key = self.sarvam_api_key if provider == "sarvam" else self.openai_api_key

        return LlmTarget(
            base_url=base_url,
            model=self.llm_model or model,
            api_key=key,
            provider=provider,
        )


@dataclass(frozen=True, slots=True)
class LlmTarget:
    """Where the reply comes from, once the keys have had their say."""

    base_url: str
    model: str
    api_key: str
    provider: str

    @property
    def ready(self) -> bool:
        return bool(self.api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
