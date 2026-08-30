"""The RAG pipeline: transcript in, grounded answer or honest abstention out.

One entry point, five rungs (`src/rag/effort.py`, docs/15-effort.md). The rung
is a **ceiling**: it says how far the pipeline *may* escalate, never how far it
must. A question the answer cache already holds is answered from Redis at rung
4 exactly as it would be at rung 1, and `AskResponse.tier` reports the rung that
actually produced the answer against `mode`, which reports the one that was
asked for.

    0 lookup      guard → embed → search → guard          no LLM, ~60 ms
    1 grounded    + cache, extractive span, Gate 3        no LLM, < 200 ms
    2 deep        + hybrid, rerank, synthesis, Gate 4     1 call
    3 corrective  + relevance grading, rewrite, retry     3-6 calls
    4 adaptive    + routing, capped repair loop           4-8 calls

The straight line through the middle is the same at every rung, which is what
keeps this one function rather than five pipelines that drift: retrieval,
Gate 2 and the citation shape are shared, and the rung decides which optional
stages hang off them.

**Every rung degrades rather than failing.** A connected Pinecone has no lexical
channel, so rung 2 runs dense-only and says `dense-only` in `escalations`. No
model key, so synthesis is unavailable and rung 2 falls back to the extractive
answer of rung 1 and reports `tier: 1`. Redis is down, so the cache is a miss.
None of those are errors at the user, and none of them are silent.
"""

from __future__ import annotations

import uuid

import numpy as np

from src.agents.rag import RagAgents, Verdict
from src.core.config import Settings, get_settings
from src.rag import effort, fuse, guardrails
from src.rag.backends.base import Capabilities, VectorBackend
from src.connectors.profile_service import ProfileService, get_profile_service
from src.rag.backends.resolve import BackendResolver, get_resolver
from src.rag.cache import AnswerCache, Scope, get_cache
from src.rag.embed import Embedder, get_embedder
from src.rag.extract import Extraction, extract_span
from src.rag.harness import Abstain, Ctx, Harness, Refuse
from src.rag.rerank import rerank
from src.rag.store import Hit, StoreUnavailable
from src.schemas.ask import AskRequest, AskResponse, Citation, Timings
from src.services.metrics_service import MetricsService, get_metrics_service

MAX_CITATIONS = 3


class AskService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        resolver: BackendResolver,
        metrics: MetricsService,
        cache: AnswerCache,
        profiles: ProfileService | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._resolver = resolver
        self._metrics = metrics
        self._cache = cache
        # What was measured about the store answering this question, which is
        # where the router's description of the corpus comes from now that
        # there is no deployment corpus to describe (docs/17-understanding.md).
        # Optional so a service built without one still routes, on the
        # backend's own one-line `describe()`.
        self._profiles = profiles
        # The optional stages, built once. They hold the same settings this
        # service does, so a call site names the stage and its arguments and
        # nothing else — which is what makes the ladder readable as a ladder.
        self._agents = RagAgents(settings)

    # ---- the pipeline ----------------------------------------------------

    def ask(
        self,
        request: AskRequest,
        *,
        user_id: str | None = None,
        store: str | None = None,
    ) -> AskResponse:
        settings = self._settings
        level = settings.effort_level(request.effort)
        ctx = Ctx(
            request_id=request.request_id or str(uuid.uuid4()),
            language=None,
            effort=level,
            # Each rung against its own budget. A shared 200 ms would make the
            # harness skip every optional stage on rungs 2-4 — including the
            # graders those rungs exist to run.
            deadline_ms=settings.deadline_for(level),
        )
        harness = Harness(ctx)
        run = _Run(
            request=request, level=level, ctx=ctx, harness=harness, settings=settings
        )
        # Named by capability discovery when several stores are attached, and
        # a preference rather than an assertion — see `BackendResolver.for_user`.
        run.prefer = store

        try:
            self._gate_input(run)
            # Before embedding, not after: a connected index built by another
            # model needs its *own* embedder, and only the resolved backend
            # knows which that is. Resolving here rather than inside
            # `_from_cache` also makes the ordering explicit instead of a side
            # effect of the cache lookup.
            self._resolve_backend(run, user_id)
            self._embed(run)

            cached = self._from_cache(run, user_id)
            if cached is not None:
                return cached

            if effort.uses_routing(level):
                self._route(run, user_id)

            self._resolve_backend(run, user_id)
            self._retrieve(run)
            self._gate_retrieval(run)

            if effort.uses_rerank(level):
                self._rerank(run)

            if effort.uses_grading(level):
                self._grade_retrieval(run)

            response = self._answer(run)
            self._to_cache(run, user_id, response)
            return response

        except Refuse as refusal:
            return self._respond(
                run,
                status="refused",
                answer=None,
                citations=[],
                confidence=0.0,
                reason=refusal.reason,
                tier=0,
            )
        except Abstain as abstention:
            return self._respond(
                run,
                status="abstained",
                answer=None,
                # The near misses: what was found and rejected. `status` already
                # says nothing was answered, and seeing them is what makes an
                # abstention inspectable rather than a shrug.
                citations=self._citations(None, run),
                confidence=abstention.confidence or run.confidence,
                reason=abstention.reason,
            )
        except Exception as error:  # degradation matrix: never 500 at the user
            ctx.trace.append({"unhandled": f"{type(error).__name__}: {error}"})
            return self._respond(
                run,
                status="refused",
                answer=None,
                citations=[],
                confidence=0.0,
                reason="Something went wrong on my side — try that again.",
                tier=0,
            )

    # ---- stages ----------------------------------------------------------

    def _gate_input(self, run: _Run) -> None:
        verdict = run.harness.stage(
            "guard_in",
            lambda: guardrails.gate_input(
                run.request.transcript,
                run.request.language_code,
                self._settings,
                # Empty, and not an oversight. The languages a corpus held used
                # to be read off this deployment's ingest manifest; a connected
                # store is somebody else's and does not declare them. So the
                # gate stops claiming to know, and the cross-lingual question
                # is settled where it is actually measurable — Gate 2, on the
                # score (`_score_retrieval`).
                [],
            ),
        )
        assert verdict is not None
        run.verdict = verdict
        run.ctx.language = verdict.language

        if verdict.status == "refused":
            raise Refuse(verdict.reason or "I can't help with that.")
        if verdict.status == "abstained":
            raise Abstain(verdict.reason or "I don't have that in my sources.")

        # Asked in a language whose code did not resolve. Gate 1 routes rather
        # than refuses (see the language block there): search unfiltered, score
        # against a floor set for cross-lingual cosines, and answer from the
        # English original whenever a chunk carries one.
        run.english = verdict.cross_lingual

    def _embed(self, run: _Run) -> None:
        """Embed with whatever model can search the store that will answer.

        Through the backend rather than through `self._embedder` directly,
        because the backend is what knows its own width — and asking for the
        wrong one is the difference between a search and a dimension error on
        every question.
        """
        # `getattr` rather than a direct call: `VectorBackend` is a structural
        # Protocol, so a backend written before this method existed — a test
        # double, an out-of-tree implementation — is still a valid backend. It
        # gets this app's own width, which is the answer it had before.
        embed = getattr(run.store, "embed_query", None) if run.store else None
        if embed is None:
            embed = self._embedder.embed_query

        vector = run.harness.stage("embed", lambda: embed(run.query))
        assert vector is not None
        run.query_vector = vector

    def _route(self, run: _Run, user_id: str | None) -> None:
        """Adaptive RAG's pre-retrieval router.

        The one stage that can save the *entire* pipeline rather than improve
        it: "hello" and "what did I just ask you" are questions no corpus
        search can help with, and retrieving for them spends the budget to
        produce an abstention that was knowable up front.

        A router that cannot answer routes to the vector store. That is the
        conservative direction — retrieving unnecessarily costs latency,
        skipping retrieval wrongly costs the answer.
        """
        decision = run.harness.stage(
            "route",
            lambda: self._agents.router.route(
                query=run.query,
                corpus=self._corpus_hint(run, user_id),
            ),
            optional=True,
        )
        if decision is None:
            run.ctx.trace.append({"route": "unavailable", "to": "vectorstore"})
            return

        run.ctx.trace.append({"route": decision.destination, "why": decision.reason})
        if decision.destination == "direct":
            # Not an abstention about the corpus — a statement that the corpus
            # was never the right place to look. The `direct` flag is what the
            # voice loop reads to answer conversationally instead of reading
            # out "no source covers this".
            run.escalations.append("routed-direct")
            run.direct = True
            raise Abstain("That isn't something my sources cover.")

    def _resolve_backend(self, run: _Run, user_id: str | None) -> None:
        """Whose vectors these are — and whether there are any.

        A user who connected Pinecone, Astra or their own Postgres is searched
        against that, and nobody else is searched at all: this deployment holds
        no corpus of its own (docs/13-connectors.md). Resolved per call rather
        than held on the service, because the service is a singleton and the
        answer differs per caller.

        No backend is an abstention and not an error. "Nothing is connected"
        is a true, actionable answer — the connectors panel is one tap away —
        whereas a 500 from here would read as the app being broken.
        """
        if run.store is not None:
            return  # the cache stage already resolved it; the answer is per-user

        resolved = self._resolver.for_user(user_id, prefer=run.prefer)
        if resolved is None:
            run.escalate("no-backend")
            raise Abstain(
                "I don't have a source to search yet — connect a vector store "
                "and I can answer from it."
            )
        run.store = resolved
        try:
            run.caps = resolved.capabilities()
        except Exception:
            # A backend that cannot describe itself is treated as the least
            # capable one, which is the same thing every rung already handles.
            run.caps = Capabilities()
        run.backend = _describe(resolved)

    def _retrieve(self, run: _Run) -> None:
        settings = self._settings
        store = run.store
        assert store is not None and run.query_vector is not None

        # Never filtered by language. This used to filter whenever the ingest
        # manifest showed more than one language in the index; a connected
        # store has no manifest and this app has no business guessing at the
        # language tags in somebody else's metadata. Guessing wrong is not a
        # neutral cost either — on a filtered ANN search pgvector discards
        # non-matching rows *while* walking the HNSW graph, so a predicate on a
        # field the store does not carry the way we assume returns fewer
        # candidates rather than none, which reads as poor recall.
        language = None

        def search() -> list[Hit]:
            dense = store.search(
                run.query_vector,
                strategies=settings.search_strategies,
                limit=settings.search_limit,
                language=language,
            )
            # Gate 2's floor and margin were swept on cosine over a
            # cosine-ordered list, and fusion below reorders by rank. Keeping
            # the dense ranking separately is what lets the gate keep reading
            # the quantity it was calibrated against while the answer is drawn
            # from the better-ordered fused list. Without this the margin test
            # compares the fusion winner against a *higher*-scoring neighbour
            # and goes negative, abstaining on good retrieval.
            run.remember_dense(dense)

            if not effort.uses_hybrid(run.level):
                return dense

            if not run.caps.lexical:
                # A hosted index is nearest-neighbour and nothing else. Rung 2
                # still runs — it just runs dense-only, and says so rather than
                # reporting a hybrid retrieval it did not perform.
                run.escalate("dense-only")
                return dense

            lexical = self._lexical(run, store, language)
            if not lexical:
                return dense

            run.escalate("hybrid")
            # Dense first, so the `Hit` that survives fusion carries a cosine:
            # Gate 2's floor and margin were swept on that scale and an RRF
            # score would abstain on everything (see src/rag/fuse.py).
            return fuse.rrf(
                [dense, lexical],
                k=settings.rrf_k,
                limit=max(settings.search_limit, settings.rerank_candidates),
            )

        def unavailable(error: BaseException) -> list[Hit]:
            raise Abstain("My sources are unavailable right now.") from error

        # `search` fills `run.dense` as a side effect: Gate 2 has to read a
        # list that is still ordered by cosine, and the fused list is not one.
        run.hits = (
            run.harness.stage(
                "search",
                search,
                # Loopback or in-process: one cheap retry is affordable, and a
                # second would only confirm the same deterministic failure.
                retries=1,
                backoff_ms=20,
                retry_on=(StoreUnavailable,),
                fallback=unavailable,
            )
            or []
        )

    def _lexical(self, run: _Run, store: VectorBackend, language: str | None) -> list[Hit]:
        """The keyword half of the hybrid, or nothing.

        A failure here is never fatal. The channel is an *addition* to dense
        retrieval — a connected Postgres built by an older migration has no
        `tsv` column and this raises, and the right answer is a dense-only
        result rather than no result at all.
        """
        search_lexical = getattr(store, "search_lexical", None)
        if not callable(search_lexical):
            run.escalate("dense-only")
            return []

        try:
            return search_lexical(
                run.query,
                strategies=self._settings.search_strategies,
                limit=self._settings.lexical_limit,
                language=language,
            )
        except Exception as error:
            run.ctx.trace.append({"lexical": f"{type(error).__name__}: {error}"})
            run.escalate("dense-only")
            return []

    def _gate_retrieval(self, run: _Run) -> None:
        """Gate 2, and the one place the corrective rung can intervene."""
        verdict = self._score_retrieval(run)

        if not verdict.ok and effort.uses_grading(run.level) and run.repairs_left:
            # Corrective RAG: the query, not the corpus, may be the problem.
            # This only runs on retrieval that has *already* been graded bad,
            # so a rewrite is paid for by a query that was going to be
            # abstained on anyway — which is what keeps query rewriting off the
            # happy path where it would be a round trip in front of everything.
            if self._repair(run):
                verdict = self._score_retrieval(run)

        run.confidence = verdict.confidence
        if not verdict.ok:
            raise Abstain(
                verdict.reason or "I don't have that in my sources.",
                confidence=verdict.confidence,
            )

    def _score_retrieval(self, run: _Run) -> guardrails.RetrievalVerdict:
        settings = self._settings
        floor = settings.retrieval_floor_cross_lingual if run.english else None
        margin = settings.retrieval_margin_cross_lingual if run.english else None

        verdict = guardrails.gate_retrieval(run.dense, settings, floor=floor, margin_floor=margin)
        run.ctx.trace.append(
            {
                "gate": "retrieval",
                "top": round(verdict.top_score, 4),
                "margin": round(verdict.margin, 4),
                "floor": floor if floor is not None else settings.retrieval_floor,
                "marginFloor": margin if margin is not None else settings.retrieval_margin,
                "crossLingual": run.english,
                "ok": verdict.ok,
            }
        )
        return verdict

    def _repair(self, run: _Run) -> bool:
        """Rewrite the query and retrieve once more. Returns whether it ran.

        Counted against `max_repairs`, and the counter spans *every* repair on
        the request — a rewrite here and a regeneration later share one budget.
        Per-branch counters are how a self-correction loop ends up unbounded
        while every individual branch looks capped.
        """
        rewritten = run.harness.stage(
            "rewrite",
            lambda: self._agents.rewriter.rewrite(query=run.query),
            optional=True,
        )
        if not rewritten:
            return False

        run.spend_repair()
        run.escalate("rewrite")
        run.ctx.trace.append({"rewrite": rewritten})

        previous = run.hits
        run.rewritten = rewritten
        self._embed(run)
        self._retrieve(run)
        # Keep what round one found. The rewrite is a second opinion about the
        # search key, not a verdict that the first retrieval was worthless, and
        # fusing the two rounds is strictly better than replacing one with the
        # other when the rewrite drifts off-topic.
        run.hits = fuse.dedupe([*run.hits, *previous])
        return True

    def _rerank(self, run: _Run) -> None:
        assert run.query_vector is not None
        result = run.harness.stage(
            "rerank",
            lambda: rerank(
                query_vector=run.query_vector,
                hits=run.hits,
                embedder=self._embedder,
                settings=self._settings,
                budget_ms=run.ctx.remaining_ms(),
                english=run.english,
            ),
            optional=True,
        )
        if result is None:
            return

        ranked, method = result
        run.ctx.trace.append({"rerank": method, "kept": len(ranked)})
        if ranked and method == "embedding":
            run.escalate("rerank")
            run.hits = ranked

    def _grade_retrieval(self, run: _Run) -> None:
        """Corrective RAG's relevance grader, over the retrieval as a whole.

        The trigger is aggregate, not per-document. Firing the expensive path
        because one result in ten was weak fires it on nearly every query — a
        top-10 almost always contains a weak result, and that is what a top-10
        is for.
        """
        grade = run.harness.stage(
            "grade",
            lambda: self._agents.relevance.grade(
                query=run.query,
                hits=run.hits,
                english=run.english,
            ),
            optional=True,
        )
        if grade is None:
            # No grader, so the deterministic gate stands on its own — which is
            # exactly what rungs 0-2 run on. Falling back to "everything is
            # relevant" would be the same behaviour; falling back loudly is not.
            run.ctx.trace.append({"grade": "unavailable"})
            return

        run.ctx.trace.append({"grade": grade.verdict, "kept": len(grade.keep)})

        thin = len(grade.keep) < self._settings.grader_relevant_min
        if (grade.verdict == "incorrect" or thin) and run.repairs_left:
            if self._repair(run):
                return  # the fresh retrieval stands ungraded; one grade per request

        if grade.keep:
            kept = [hit for hit in run.hits if hit.chunk_id in set(grade.keep)]
            if kept:
                run.escalate("graded")
                run.hits = kept

    # ---- answering -------------------------------------------------------

    def _answer(self, run: _Run) -> AskResponse:
        if run.level == effort.LOOKUP:
            return self._answer_lookup(run)
        if run.level == effort.GROUNDED:
            return self._answer_extractive(run)
        return self._answer_synthesised(run)

    def _answer_lookup(self, run: _Run) -> AskResponse:
        """Rung 0: the passages, as they are. No model, no span selection.

        The whole rung. Gate 2 has already decided that something in the index
        is close enough to be worth showing, so the top passage is the answer
        and the rest are the citations. There is nothing to hallucinate because
        nothing was written — which is also why Gate 3 is skipped rather than
        run: it checks that an answer is a substring of its source, and here
        the answer *is* the source.
        """
        top = run.hits[0]
        citations = self._citations(None, run)
        return self._respond(
            run,
            status="answered",
            answer=top.rendering(english=run.english)[: self._settings.extract_max_chars],
            citations=citations,
            confidence=run.confidence,
            reason=None,
            method="passage",
            tier=effort.LOOKUP,
        )

    def _answer_extractive(self, run: _Run) -> AskResponse:
        """Rung 1: a span lifted verbatim, checked by construction (Gate 3)."""
        extraction = run.harness.stage(
            "extract",
            lambda: extract_span(
                query=run.query,
                query_vector=run.query_vector,
                hits=run.hits,
                embedder=self._embedder,
                settings=self._settings,
                # What is left of the budget decides whether the embedding
                # rerank inside extraction runs at all.
                budget_ms=run.ctx.remaining_ms(),
                english=run.english,
            ),
        )

        if extraction is None:
            raise Abstain(
                "I couldn't pull a clear answer out of what I found.",
                confidence=run.confidence,
            )

        citations = self._citations(extraction, run)
        grounding = run.harness.stage(
            "guard_out",
            # Against the rendering the span was actually cut from, not the
            # indexed text — checking an English span against a Devanagari
            # passage would fail every time and abstain on a good answer.
            lambda: guardrails.gate_grounding(extraction.answer, extraction.source, citations),
        )
        if grounding:
            raise Abstain(grounding, confidence=run.confidence)

        return self._respond(
            run,
            status="answered",
            answer=extraction.answer,
            citations=citations,
            confidence=self._confidence(run, extraction.score),
            reason=None,
            method=extraction.method,
            tier=effort.GROUNDED,
        )

    def _answer_synthesised(self, run: _Run) -> AskResponse:
        """Rungs 2-4: one grounded synthesis, checked by Gate 4.

        Two failures are distinguished here because they need different
        repairs, which is the good idea in adaptive RAG and the one most
        implementations collapse:

            not supported  → the context was fine, the writing was not
                             → regenerate from the same context
            not useful     → the writing was fine, the context was not
                             → rewrite the query and retrieve again

        Below rung 4 neither repair runs and the answer falls back to the
        extractive rung, which is a real answer rather than a failure.
        """
        for attempt in range(2):
            answer = run.harness.stage(
                "generate",
                lambda: self._agents.synthesis.write(
                    query=run.query,
                    hits=run.hits,
                    english=run.english,
                ),
            )

            if answer is None:
                # Either no model is configured or the model said the context
                # does not answer the question. The first is a degradation and
                # the second is a correct abstention — and rung 1 can tell them
                # apart by simply trying, at no network cost.
                run.ctx.trace.append({"synthesis": "unavailable-or-refused", "attempt": attempt})
                return self._fallback_extractive(run)

            citations = self._citations(None, run)
            contexts = [h.rendering(english=run.english) for h in run.hits[:MAX_CITATIONS]]

            # Bound as defaults rather than closed over. The harness calls this
            # within the same iteration, so late binding is harmless today —
            # but a stage that ever deferred would silently gate the wrong
            # attempt's answer, and that failure would be invisible.
            reason = run.harness.stage(
                "guard_out",
                lambda written=answer, ctx=contexts, cited=citations: (
                    guardrails.gate_generation(
                        written,
                        ctx,
                        cited,
                        embedder=self._embedder,
                        settings=self._settings,
                    )
                ),
            )

            verdict = self._grade_answer(run, answer)
            supported = reason is None and (verdict is None or verdict.supported)
            useful = verdict is None or verdict.useful

            if supported and useful:
                run.escalate("synthesis")
                return self._respond(
                    run,
                    status="answered",
                    answer=answer,
                    citations=citations,
                    confidence=self._confidence(run, run.hits[0].score),
                    reason=None,
                    method="synthesis",
                    tier=run.level,
                )

            run.ctx.trace.append(
                {"gate": "generation", "supported": supported, "useful": useful, "why": reason}
            )

            if not run.repairs_left or attempt:
                break

            if not useful and supported:
                # Grounded but off-target: the context is the problem.
                if self._repair(run):
                    continue
                break

            # Ungrounded: same context, one more attempt at writing from it.
            run.spend_repair()
            run.escalate("regenerate")

        return self._fallback_extractive(run)

    def _grade_answer(self, run: _Run, answer: str) -> Verdict | None:
        """The usefulness half of Gate 4, and only where it can be acted on.

        Rungs 2 and 3 have no repair for "grounded but off-target" — rung 3
        spends its repair budget on retrieval — so asking would cost a round
        trip to learn something nothing downstream can use.
        """
        if run.level < effort.ADAPTIVE:
            return None
        return run.harness.stage(
            "grade",
            lambda: self._agents.verdict.grade(
                query=run.query,
                answer=answer,
                hits=run.hits,
                english=run.english,
            ),
            optional=True,
        )

    def _fallback_extractive(self, run: _Run) -> AskResponse:
        """Synthesis did not produce a usable answer. Drop to the rung below.

        A rung that cannot do its own job should return the best answer the
        system *can* produce, not nothing. The response says `tier: 1` while
        `mode` still says what was asked for, so the degradation is visible in
        the metrics rather than showing up as a mysteriously good latency.
        """
        run.escalate("fallback-extractive")
        return self._answer_extractive(run)

    # ---- the cache -------------------------------------------------------

    def _scope(self, run: _Run, user_id: str | None) -> Scope:
        return Scope(
            user=user_id or "anonymous",
            backend=run.backend or "none",
            mode=effort.name(run.level),
            language=run.ctx.language or "unknown",
            english=run.english,
        )

    def _from_cache(self, run: _Run, user_id: str | None) -> AskResponse | None:
        """Rung 1 and up. Rung 0 is already cheaper than a cache round trip.

        The backend is already resolved — `ask` does it before embedding, since
        a connected index built by another model needs its own embedder — and
        it is part of the scope, so this only reads it.
        """
        if run.level < effort.GROUNDED or not self._cache.configured:
            return None

        self._resolve_backend(run, user_id)  # idempotent; a no-op once resolved
        scope = self._scope(run, user_id)

        hit = run.harness.stage(
            "cache",
            lambda: self._cache.get(run.query, run.query_vector, scope),
            optional=True,
        )
        if hit is None:
            return None

        payload = hit.payload
        run.ctx.trace.append({"cache": hit.how, "similarity": hit.similarity})
        run.escalate(f"cache-{hit.how}")

        return self._respond(
            run,
            status="answered",
            answer=payload.get("answer"),
            citations=[Citation(**c) for c in payload.get("citations", [])],
            confidence=float(payload.get("confidence") or 0.0),
            reason=None,
            method="cache",
            tier=int(payload.get("tier") or run.level),
            cached=True,
        )

    def _to_cache(self, run: _Run, user_id: str | None, response: AskResponse) -> None:
        """Only successes, and only what a future request could reuse.

        An abstention is a statement about the corpus at one moment — cache it
        and a re-ingest that fills the gap stays invisible for a day. A refusal
        is a statement about the input and costs microseconds to recompute.
        """
        if response.status != "answered" or run.level < effort.GROUNDED:
            return
        if not self._cache.configured or run.query_vector is None:
            return
        if "fallback-extractive" in response.escalations:
            # A rung that fell back produced the best answer available *at that
            # moment*, not the answer this rung gives. Caching it under the
            # rung's own key means a missing model key for one minute serves
            # degraded answers for the whole TTL, and the metrics show a
            # healthy cache hit rate the entire time.
            run.ctx.trace.append({"cache": "skipped", "why": "degraded"})
            return

        self._cache.put(
            run.query,
            run.query_vector,
            self._scope(run, user_id),
            {
                "answer": response.answer,
                "citations": [c.model_dump() for c in response.citations],
                "confidence": response.confidence,
                "tier": response.tier,
                "method": response.method,
            },
        )

    # ---- helpers ---------------------------------------------------------

    def _corpus_hint(self, run: _Run, user_id: str | None) -> str:
        """One line about what the connected store holds, for the router.

        A vague description is a routing bug: told only "a vector index", the
        router decides a perfectly ordinary factual question is something a
        search could not help with and skips retrieval entirely. So this wants
        to be specific, and it can be — the profile already sampled this store
        and wrote a paragraph about what is in it (docs/17-understanding.md).

        Falls back to the backend's own one-line `describe()`, and past that to
        a generic line. Never raises and never blocks: a store connected
        seconds ago has no profile yet, and the right answer for that request
        is a weaker hint rather than a stall while somebody's index is sampled.
        """
        card = ""
        if self._profiles is not None and run.store is not None:
            try:
                card = self._profiles.card(user_id, run.store.name)
            except Exception as error:
                run.ctx.trace.append({"corpus": f"{type(error).__name__}: {error}"})

        if card:
            return card
        return run.backend or "a connected vector index of unknown content"

    def _confidence(self, run: _Run, span_score: float) -> float:
        floor = (
            self._settings.retrieval_floor_cross_lingual
            if run.english
            else self._settings.retrieval_floor
        )
        span = max(1e-6, 1.0 - floor)
        span_term = min(1.0, max(0.0, (span_score - floor) / span))
        return round(min(1.0, 0.6 * run.confidence + 0.4 * span_term), 3)

    def _citations(self, extraction: Extraction | None, run: _Run) -> list[Citation]:
        ordered: list[Hit] = []
        if extraction is not None:
            ordered.append(extraction.hit)
        ordered.extend(
            h for h in run.hits if extraction is None or h.chunk_id != extraction.hit.chunk_id
        )
        return [self._citation(hit, run.english) for hit in ordered[:MAX_CITATIONS]]

    @staticmethod
    def _citation(hit: Hit, english: bool = False) -> Citation:
        origins = hit.payload.get("origins") or []
        return Citation(
            doc_id=hit.chunk_id,
            strategy=hit.strategy,
            score=round(hit.score, 4),
            # The rendering that was answered from, so a citation can be read
            # by whoever is reading the answer.
            text=hit.rendering(english=english),
            source_query_ids=list(hit.payload.get("sourceQueryIds") or []),
            is_gold=any(origin.get("isSelected") for origin in origins),
        )

    def _respond(
        self,
        run: _Run,
        *,
        status: str,
        answer: str | None,
        citations: list[Citation],
        confidence: float,
        reason: str | None,
        method: str | None = None,
        tier: int | None = None,
        cached: bool = False,
    ) -> AskResponse:
        ctx = run.ctx
        timings = Timings(**run.harness.finish())
        budget = self._settings.deadline_for(run.level)
        response = AskResponse(
            status=status,  # type: ignore[arg-type]
            answer=answer,
            citations=citations,
            confidence=confidence,
            tier=run.level if tier is None else tier,
            reason=reason,
            timings=timings,
            request_id=ctx.request_id,
            language=ctx.language,
            flags=run.flags,
            method=method,
            mode=effort.name(run.level),
            cached=cached,
            backend=run.backend,
            escalations=run.escalations,
            budget_ms=budget,
            within_budget=timings.total <= budget,
        )
        self._metrics.record(response, ctx.trace)
        return response


# ---- per-request state ---------------------------------------------------


class _Run:
    """Everything one request accumulates as it climbs the ladder.

    A mutable bag rather than threaded arguments, because the stages genuinely
    share state — a rewrite replaces the query *and* the vector *and* the hits,
    and passing eleven values through six methods to avoid saying so would hide
    that rather than prevent it.
    """

    __slots__ = (
        "request", "level", "ctx", "harness", "settings", "store", "caps", "verdict",
        "query_vector", "english", "hits", "dense", "confidence", "escalations",
        "backend", "rewritten", "direct", "prefer", "_repairs",
    )

    def __init__(
        self,
        *,
        request: AskRequest,
        level: int,
        ctx: Ctx,
        harness: Harness,
        settings: Settings,
    ) -> None:
        self.request = request
        self.level = level
        self.ctx = ctx
        self.harness = harness
        self.settings = settings
        self.store: VectorBackend | None = None
        self.caps = Capabilities()
        self.verdict: guardrails.InputVerdict | None = None
        self.query_vector: np.ndarray | None = None
        self.english = False
        self.hits: list[Hit] = []
        #: The same results in cosine order, across every retrieval round. Only
        #: Gate 2 reads it — see `remember_dense`.
        self.dense: list[Hit] = []
        self.confidence = 0.0
        self.escalations: list[str] = []
        self.backend: str | None = None
        self.rewritten: str | None = None
        self.direct = False
        #: Which connector this question was routed to, when discovery named
        #: one. `None` is "whichever this user has", the standing order.
        self.prefer: str | None = None
        self._repairs = 0

    @property
    def query(self) -> str:
        """What to search with: the rewrite if there was one, else Gate 1's text."""
        if self.rewritten:
            return self.rewritten
        return self.verdict.query if self.verdict else self.request.transcript

    @property
    def flags(self) -> list[str]:
        flags = list(self.verdict.flags) if self.verdict else []
        if self.direct:
            # Read by the voice loop: this was not a corpus question, so answer
            # it conversationally instead of reading out an abstention.
            flags.append("direct")
        return flags

    @property
    def repairs_left(self) -> bool:
        return self._repairs < self.settings.max_repairs

    def spend_repair(self) -> None:
        self._repairs += 1

    def escalate(self, what: str) -> None:
        if what not in self.escalations:
            self.escalations.append(what)

    def remember_dense(self, hits: list[Hit]) -> None:
        """Merge a round of dense results into the cosine-ordered view.

        Rounds accumulate rather than replace, because after a corrective
        rewrite the question Gate 2 has to answer is "did *either* attempt find
        something", not "did the most recent one". A rewrite that drifts
        off-topic would otherwise turn a passable first retrieval into an
        abstention.
        """
        merged = {hit.chunk_id: hit for hit in [*self.dense, *hits]}
        self.dense = sorted(merged.values(), key=lambda h: h.score, reverse=True)


def _describe(store: VectorBackend) -> str:
    try:
        return store.describe()
    except Exception:
        return getattr(store, "name", "unknown")


def get_ask_service() -> AskService:
    return AskService(
        get_settings(),
        get_embedder(),
        get_resolver(),
        get_metrics_service(),
        get_cache(),
        get_profile_service(),
    )
