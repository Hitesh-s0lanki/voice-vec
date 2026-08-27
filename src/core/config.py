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

    # ---- Clerk (src/core/clerk.py) --------------------------------------
    # Who a caller is, when they are signed in. The browser opens the voice
    # socket against this server directly, so identity arrives as a Clerk
    # session token and is verified here — an id in a header or a query
    # parameter is something anyone can type.
    #
    # The publishable key is enough: the instance host is base64'd into it and
    # the signing keys are fetched from there once an hour. CLERK_JWT_KEY is
    # the PEM from the dashboard, for a deployment that would rather not make
    # an outbound call at handshake time; set either, or neither, in which case
    # every caller is anonymous and conversations belong to the browser.
    clerk_publishable_key: str = ""
    clerk_jwt_key: str = ""
    clerk_jwks_ttl_s: int = 3600

    # ---- Composio (src/integrations/) -----------------------------------
    # Composio is the connector, and *the user* connects it. There is no
    # project-wide API key here on purpose: each signed-in account brings its
    # own, so a toolkit connected through this app lands in that person's
    # Composio project and not in one belonging to whoever deployed the server.
    #
    # That makes the key a real secret at rest, held per `user_id`, so it is
    # encrypted before it reaches Postgres (src/integrations/crypto.py). This
    # is the master key that encrypts them — 32 url-safe base64 bytes, exactly
    # what `Fernet.generate_key()` prints:
    #
    #     uv run python -c "from cryptography.fernet import Fernet; \
    #         print(Fernet.generate_key().decode())"
    #
    # Unset, nobody can connect Composio at all and the panel says so. Losing
    # it or rotating it does not corrupt anything — every stored key simply
    # stops decrypting, and each user reconnects. Never commit it: it is the
    # one value in this file that is worth as much as the database.
    composio_encryption_key: str = ""

    # Where Composio sends the browser back once consent is given or refused.
    # It is the *frontend* origin, not this server's — the page that reports
    # the outcome is a Next route, and Composio appends its own query string
    # to whatever is handed over.
    frontend_url: str = "http://localhost:3002"
    composio_callback_path: str = "/integration"

    # How long a user's toolkit catalogue is held before it is asked for again.
    # It changes about as often as Composio ships an integration, and the panel
    # that renders it opens on a click. Cached per user, because it is fetched
    # with that user's key.
    composio_catalog_ttl_s: int = 900
    composio_timeout_s: float = 15.0
    # How many users' SDK clients to keep built at once. Small: a client is
    # cheap to rebuild and this is a rail panel, not the voice path.
    composio_client_cache: int = 64

    # How many users' vector backends to keep built. Smaller than the Composio
    # cache because these are more expensive to hold: a connected pgvector
    # backend keeps a connection pool open against somebody else's database,
    # and evicting one hands those connections back.
    vector_backend_cache: int = 16

    # ---- Tool calling (src/integrations/agent.py) ------------------------
    # A spoken turn can run the tools a user has linked through Composio. The
    # whole feature is skipped for anybody who has linked nothing, so these
    # only bite once somebody has.
    #
    # False turns it off everywhere without disconnecting anyone.
    tools_enabled: bool = True
    # How long tool schemas are held. Keyed by the set of linked toolkits too,
    # so linking one shows up on the next turn rather than after this elapses.
    tool_schema_ttl_s: int = 300
    # Schemas ride in every prompt of the turn, so the cap is a token budget as
    # much as a latency one.
    tool_schema_limit: int = 40
    # How many decide-run-decide rounds before the model has to answer. Two is
    # enough for "search, then send"; more is usually a loop.
    tool_max_rounds: int = 2
    # A tool that has not answered by here is costing the listener more than it
    # is worth. The turn continues and says so.
    tool_timeout_s: float = 20.0

    @property
    def composio_ready(self) -> bool:
        """Whether *anyone* can connect Composio on this deployment.

        Not "is Composio configured" — there is no server-side Composio account
        any more. This asks only whether there is somewhere safe to put a user's
        key once they hand one over.
        """
        return bool(self.composio_encryption_key)

    @property
    def composio_callback_url(self) -> str:
        return f"{self.frontend_url.rstrip('/')}{self.composio_callback_path}"

    # ---- Embedding ------------------------------------------------------
    # multilingual-e5-small: 384 dims, ONNX, ~3 ms per query on CPU. The ONNX
    # export lives in the same HF repo as the torch weights (`onnx/model.onnx`).
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384
    embed_model_file: str = "onnx/model.onnx"
    # None lets onnxruntime pick. Pin it if batch-of-1 latency looks noisy.
    embed_threads: int | None = None
    embed_cache_dir: str = "data/models"

    # ---- Vector store (Postgres + pgvector) -----------------------------
    # Neon, or any Postgres with the `vector` extension available. Use the
    # *pooled* endpoint (the host carrying `-pooler`) — the store disables
    # prepared statements for it, and a direct endpoint exhausts its connection
    # limit under the API's pool.
    #
    # Region matters to the SLO, not just to comfort: search measured ~11 ms
    # in-process, and every millisecond of round trip is spent inside the same
    # 200 ms as extraction's 78 ms (docs/04-latency.md). Keep the database in
    # the region the API runs in.
    database_url: str = ""
    pg_table: str = "chunks"
    pg_pool_min: int = 1
    pg_pool_max: int = 8
    pg_connect_timeout_s: float = 10.0
    # A query that outlives the 200 ms deadline is already lost; the ceiling is
    # generous because ingest shares the pool and legitimately runs longer.
    pg_statement_timeout_ms: int = 5000
    # HNSW search breadth. Below `search_limit` the index cannot return a full
    # page, so the store raises it to match. Higher trades latency for recall
    # and is the first dial to turn if recall@5 drops after the migration.
    hnsw_ef_search: int = 64

    # ---- Retrieval ------------------------------------------------------
    search_limit: int = 10
    # Which chunking strategies to query. v1 ingests S1 only (see docs/03-chunking.md).
    search_strategies: list[str] = ["S1"]

    # ---- The effort ladder (docs/15-effort.md) ---------------------------
    # One rung per RAG architecture, and the level the caller asks for is a
    # *ceiling* rather than a floor: a question the cache answers costs nothing
    # even at rung 4. `AskResponse.tier` reports which rung actually answered.
    #
    #   0 lookup      search only, no LLM anywhere on the path
    #   1 grounded    + extractive span + the semantic answer cache
    #   2 deep        + hybrid, rerank, LLM synthesis over the retrieved set
    #   3 corrective  + relevance grading, query rewrite, one re-retrieval
    #   4 adaptive    + routing before retrieval, capped repair loop
    effort_max: int = 4
    effort_default: int = 1

    # Each rung gets its own budget. Requirement 3's 200 ms is a claim about
    # rungs 0–1, which are the ones with no network call after the transcript;
    # a rung that makes LLM calls cannot meet it and says so rather than being
    # measured against a deadline it was never going to hold. The harness skips
    # optional stages against *this* number, so a shared 200 ms would make rung
    # 3 skip the grading it exists to do.
    effort_deadline_ms: list[int] = [200, 200, 2500, 9000, 16000]

    # ---- Hybrid + rerank (rung 2 and up) ---------------------------------
    # The lexical channel only runs on a backend that has one — pgvector does
    # (`tsv` and its GIN index are built at ingest), a hosted Pinecone or Astra
    # index does not. `VectorBackend.capabilities()` is what decides, so a
    # connected store degrades to dense-only instead of erroring.
    lexical_limit: int = 20
    # Reciprocal rank fusion's damping. 60 is the value from the original TREC
    # write-up and the one every implementation since has used; it matters
    # little as long as it is well above the list lengths being fused.
    rrf_k: int = 60
    # How many fused candidates the passage rerank scores, and how many survive.
    rerank_candidates: int = 20
    rerank_keep: int = 5
    # MMR's relevance/diversity trade. 1.0 is a plain rerank with no diversity
    # term at all, which is the escape hatch if diversity ever measures worse
    # than it reads — the five chunking strategies overlap by construction, so
    # it should not, but that is a claim to check rather than assume.
    mmr_lambda: float = 0.7

    # ---- Synthesis and graders (rung 2 and up) ---------------------------
    # These are the network calls the ladder buys with latency. Separate from
    # `llm_*` above, which is the *voice* model: this one answers in text, is
    # not streamed, and wants a much colder temperature than a conversation.
    synthesis_max_tokens: int = 320
    synthesis_temperature: float = 0.1
    synthesis_context_passages: int = 5
    ask_llm_timeout_s: float = 20.0
    grader_timeout_s: float = 10.0
    grader_max_tokens: int = 200

    # Gate 4. A generated answer cannot be checked by substring the way an
    # extracted span can, so every sentence of it is embedded and scored
    # against the sentences of the context it was given. Below this, the answer
    # is not supported by what was retrieved and the pipeline abstains.
    # Local — the embedder is already loaded — so the gate costs milliseconds,
    # not a round trip. Sweep it with scripts/evaluate.py after any re-ingest.
    generation_support_floor: float = 0.62
    # How many of the answer's sentences may fall below that floor and still
    # pass. One unsupported clause in a four-sentence answer is a hedge; half
    # of them is a hallucination.
    generation_support_ratio: float = 0.7

    # ---- Repairs (rungs 3 and 4) -----------------------------------------
    # Every self-correction loop ships with a counter. The reference
    # implementation this ladder is drawn from wires `generate → generate` with
    # no cap and only fails to spin because a different bug crashes it first
    # (docs/agentic-rag/07-findings.md).
    max_repairs: int = 1
    # Corrective RAG's trigger. The paper grades confidence over the *whole*
    # retrieval; grading one document at a time and firing on any single
    # failure runs the expensive path on nearly every query.
    grader_relevant_min: int = 2

    # ---- Answer cache (Redis) --------------------------------------------
    # Cache-augmented generation, rung 1 and up. A repeat question is answered
    # from Redis at embedding cost alone — no search, no synthesis.
    #
    #   REDIS_URL=redis://127.0.0.1:6379/0        local
    #   REDIS_URL=rediss://default:pw@host:6379   managed, TLS
    #
    # Unset, the cache is simply off and every question runs the full path.
    # Nothing else changes, which is what makes this safe to leave empty.
    redis_url: str = ""
    cache_enabled: bool = True
    cache_prefix: str = "vec:cache"
    cache_ttl_s: int = 86_400
    # Redis sits on the answer path, so it gets a budget rather than a default
    # socket timeout: a cache that has not answered in 150 ms has already cost
    # more than the search it was meant to save. Measured against a managed
    # instance one region away, a warm round trip is ~6 ms.
    cache_timeout_s: float = 0.15
    # *Opening* the connection is a different budget, and collapsing the two is
    # a bug that only appears against a remote Redis: a TLS handshake across a
    # region measured ~93 ms and can easily exceed the per-operation ceiling
    # above, in which case the cache silently never connects at all.
    cache_connect_timeout_s: float = 3.0
    # The semantic half needs RediSearch (Redis Stack, or Redis 8's bundled
    # query engine). Plain Redis keeps the exact-match half and says so once in
    # the log — an honest downgrade rather than a silent one.
    cache_semantic: bool = True
    # How close a past question has to be to answer this one. Cosine over
    # L2-normalised e5 vectors, so it is on the same scale as everything else
    # here — unlike the raw squared-L2 threshold in the notebook this idea came
    # from, which does not transfer between embedding models.
    #
    # Deliberately high. A loose cache threshold is a correctness bug that
    # presents as a performance win: it answers a question nobody asked, and
    # the answer looks perfectly reasonable. Sweep it before lowering it.
    cache_similarity: float = 0.97
    # How many cached answers to keep per scope before the oldest are dropped.
    # Measured against a managed Redis 8 with `MEMORY USAGE`: one entry with a
    # realistic Devanagari payload costs 6.09 KB, so a 30 MB instance holds
    # roughly five thousand before index overhead.
    #
    # Lowered from 1_500 when agent memory moved into the same instance
    # (docs/16-memory.md). Redis Cloud's Agent Memory service attaches to a
    # database rather than provisioning one, so on the free tier the cache and
    # the agent's memory share these 30 MB — and the database's `volatile-lru`
    # cannot tell them apart, which means an unbounded cache does not simply
    # fill up, it evicts the agent's memories. 900 entries is ~5.5 MB per
    # scope. Raise it with the instance, not with the hit rate.
    cache_max_entries: int = 900
    # A single payload larger than this is not cached at all. Guards against one
    # pathological passage taking a measurable share of a small instance.
    cache_max_entry_bytes: int = 16_384

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
    # The same floor, for a question asked in a language the index does not
    # hold. Cross-lingual cosines sit lower — the query and the passage are in
    # different languages, and e5 places a translation pair close but not as
    # close as a paraphrase — so the swept thresholds above abstain on
    # retrieval that was right. Measured rather than guessed, by
    # `scripts/crosslingual.py` over 200 questions asked twice, once in each
    # language (docs/13-cross-lingual.md):
    #
    #                top score  margin  recall@5
    #   hindi           0.8942  0.0239    0.6967
    #   english         0.8469  0.0179    0.6475
    #
    # 0.78 is the highest floor that costs nothing: coverage is flat from 0.70
    # to 0.78 and falls from 0.80 up, so under it the margin decides — the same
    # regime the Hindi floor was picked in. Re-measure after any re-ingest.
    retrieval_floor_cross_lingual: float = 0.78
    retrieval_margin: float = 0.02
    # The margin matters far more than the floor here, because the gaps
    # compress along with the scores. Picked to land on the *same operating
    # point* as Hindi rather than on a new preference — the coverage the margin
    # above was chosen for, measured on the same index:
    #
    #   margin   coverage (hi / en)   abstention recall (hi / en)
    #   0.010     77.05% / 73.77%          33.33% / 30.77%
    #   0.015     58.20% / 53.28%          47.44% / 52.56%  ← en matches hi@0.02
    #   0.020     52.46% / 34.43%          62.82% / 75.64%  ← hi default
    retrieval_margin_cross_lingual: float = 0.015
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
    # Back-to-back it is 78 ms. The ONNX arena goes cold when nothing touches it,
    # and a pooled connection idles out too; interactive voice use is all cold
    # requests:
    # one question, then minutes of silence.
    #
    # So a tiny embed + search runs on this interval to keep both hot. ~12 ms of
    # work per tick — a 0.06% duty cycle at 20 s. Set 0 to disable and measure
    # the difference.
    keepalive_seconds: int = 20

    # ---- Resolved providers ---------------------------------------------

    @property
    def redis_ready(self) -> bool:
        return bool(self.redis_url) and self.cache_enabled

    def effort_level(self, requested: int | None) -> int:
        """The rung this request may climb to, clamped to what exists."""
        if requested is None:
            requested = self.effort_default
        return max(0, min(int(requested), self.effort_max))

    def deadline_for(self, effort: int) -> int:
        """The budget that rung is measured against.

        Falls back to `deadline_ms` for a rung with no entry, so adding a rung
        without extending the list degrades to the strict budget rather than to
        an unbounded one.
        """
        table = self.effort_deadline_ms
        if 0 <= effort < len(table):
            return int(table[effort])
        return self.deadline_ms

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
