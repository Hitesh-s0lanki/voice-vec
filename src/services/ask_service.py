"""The RAG pipeline: transcript in, grounded answer or honest abstention out.

Tier 1 of the ladder in docs/02-architecture.md — embed, search, extract, with
a gate before and after. Zero network calls after the transcript arrives, which
is the only reason 200 ms is reachable at all.

Tiers 2 (cross-encoder rerank) and 3 (LLM synthesis) are not in v1. The stage
slots and their timing keys exist so adding them does not reshape the contract.
"""

from __future__ import annotations

import uuid

from src.core.config import Settings, get_settings
from src.rag import guardrails
from src.rag.embed import Embedder, get_embedder
from src.rag.extract import Extraction, extract_span
from src.rag.harness import Abstain, Ctx, Harness, Refuse
from src.rag.manifest import indexed_languages
from src.rag.store import Hit, StoreUnavailable, VectorStore, get_store
from src.schemas.ask import AskRequest, AskResponse, Citation, Timings
from src.services.metrics_service import MetricsService, get_metrics_service

TIER = 1
MAX_CITATIONS = 3


class AskService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        metrics: MetricsService,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._metrics = metrics

    def ask(self, request: AskRequest) -> AskResponse:
        settings = self._settings
        ctx = Ctx(
            request_id=request.request_id or str(uuid.uuid4()),
            language=None,
            effort=request.effort,
            deadline_ms=settings.deadline_ms,
        )
        harness = Harness(ctx)

        if not settings.rag_enabled:
            # Retrieval is switched off for the voice build. Abstaining is both
            # honest and already part of the contract — better than searching an
            # index that was never warmed and reporting whatever comes back.
            return self._respond(
                harness,
                status="abstained",
                answer=None,
                citations=[],
                confidence=0.0,
                reason="Retrieval is switched off — set RAG_ENABLED=true to search the corpus.",
                flags=[],
                tier=0,
            )

        verdict: guardrails.InputVerdict | None = None
        hits: list[Hit] = []
        confidence = 0.0

        try:
            verdict = harness.stage(
                "guard_in",
                lambda: guardrails.gate_input(
                    request.transcript,
                    request.language_code,
                    settings,
                    indexed_languages(),
                ),
            )
            assert verdict is not None
            ctx.language = verdict.language

            if verdict.status == "refused":
                raise Refuse(verdict.reason or "I can't help with that.")
            if verdict.status == "abstained":
                raise Abstain(verdict.reason or "I don't have that in my sources.")

            query_vector = harness.stage(
                "embed",
                lambda: self._embedder.embed_query(verdict.query),
            )
            assert query_vector is not None

            # Filter by language only when the index actually holds more than
            # one. Filtering on the sole indexed value cannot change the result
            # set, and on a filtered ANN search it can actively hurt: pgvector
            # discards non-matching rows *while* walking the HNSW graph, so a
            # predicate that excludes nothing still costs candidates. Cheap to
            # keep, so it stays for the day a second language is indexed.
            languages = indexed_languages()
            filter_language = verdict.language if len(languages) > 1 else None

            def search() -> list[Hit]:
                return self._store.search(
                    query_vector,
                    strategies=settings.search_strategies,
                    limit=settings.search_limit,
                    language=filter_language,
                )

            def unavailable(error: BaseException) -> list[Hit]:
                raise Abstain("My sources are unavailable right now.") from error

            hits = harness.stage(
                "search",
                search,
                # Loopback or in-process: one cheap retry is affordable, and a
                # second would only confirm the same deterministic failure.
                retries=1,
                backoff_ms=20,
                retry_on=(StoreUnavailable,),
                fallback=unavailable,
            ) or []

            retrieval = guardrails.gate_retrieval(hits, settings)
            confidence = retrieval.confidence
            ctx.trace.append(
                {
                    "gate": "retrieval",
                    "top": round(retrieval.top_score, 4),
                    "margin": round(retrieval.margin, 4),
                    "ok": retrieval.ok,
                }
            )
            if not retrieval.ok:
                raise Abstain(
                    retrieval.reason or "I don't have that in my sources.",
                    confidence=retrieval.confidence,
                )

            extraction = harness.stage(
                "extract",
                lambda: extract_span(
                    query=verdict.query,
                    query_vector=query_vector,
                    hits=hits,
                    embedder=self._embedder,
                    settings=settings,
                    # What is left of the 200 ms decides whether the embedding
                    # rerank runs at all.
                    budget_ms=ctx.remaining_ms(),
                ),
            )

            if extraction is None:
                raise Abstain(
                    "I couldn't pull a clear answer out of what I found.",
                    confidence=retrieval.confidence,
                )

            citations = self._citations(extraction, hits)

            grounding = harness.stage(
                "guard_out",
                lambda: guardrails.gate_grounding(
                    extraction.answer, extraction.hit.text, citations
                ),
            )
            if grounding:
                raise Abstain(grounding, confidence=retrieval.confidence)

            return self._respond(
                harness,
                status="answered",
                answer=extraction.answer,
                citations=citations,
                confidence=self._confidence(retrieval.confidence, extraction),
                reason=None,
                flags=verdict.flags,
                method=extraction.method,
            )

        except Refuse as refusal:
            return self._respond(
                harness,
                status="refused",
                answer=None,
                citations=[],
                confidence=0.0,
                reason=refusal.reason,
                flags=verdict.flags if verdict else [],
                tier=0,
            )
        except Abstain as abstention:
            return self._respond(
                harness,
                status="abstained",
                answer=None,
                # The near misses: what was found and rejected. `status` already
                # says nothing was answered, and seeing them is what makes an
                # abstention inspectable rather than a shrug.
                citations=self._citations(None, hits),
                confidence=abstention.confidence or confidence,
                reason=abstention.reason,
                flags=verdict.flags if verdict else [],
            )
        except Exception as error:  # degradation matrix: never 500 at the user
            ctx.trace.append({"unhandled": f"{type(error).__name__}: {error}"})
            return self._respond(
                harness,
                status="refused",
                answer=None,
                citations=[],
                confidence=0.0,
                reason="Something went wrong on my side — try that again.",
                flags=verdict.flags if verdict else [],
                tier=0,
            )

    # ---- helpers --------------------------------------------------------

    def _confidence(self, retrieval_confidence: float, extraction: Extraction) -> float:
        span = max(1e-6, 1.0 - self._settings.retrieval_floor)
        span_term = min(
            1.0, max(0.0, (extraction.score - self._settings.retrieval_floor) / span)
        )
        return round(min(1.0, 0.6 * retrieval_confidence + 0.4 * span_term), 3)

    def _citations(self, extraction: Extraction | None, hits: list[Hit]) -> list[Citation]:
        ordered: list[Hit] = []
        if extraction is not None:
            ordered.append(extraction.hit)
        ordered.extend(h for h in hits if extraction is None or h.chunk_id != extraction.hit.chunk_id)

        return [self._citation(hit) for hit in ordered[:MAX_CITATIONS]]

    @staticmethod
    def _citation(hit: Hit) -> Citation:
        origins = hit.payload.get("origins") or []
        return Citation(
            doc_id=hit.chunk_id,
            strategy=hit.strategy,
            score=round(hit.score, 4),
            text=hit.text,
            source_query_ids=list(hit.payload.get("sourceQueryIds") or []),
            is_gold=any(origin.get("isSelected") for origin in origins),
        )

    def _respond(
        self,
        harness: Harness,
        *,
        status: str,
        answer: str | None,
        citations: list[Citation],
        confidence: float,
        reason: str | None,
        flags: list[str],
        method: str | None = None,
        tier: int = TIER,
    ) -> AskResponse:
        ctx = harness.ctx
        timings = Timings(**harness.finish())
        response = AskResponse(
            status=status,  # type: ignore[arg-type]
            answer=answer,
            citations=citations,
            confidence=confidence,
            tier=tier,
            reason=reason,
            timings=timings,
            request_id=ctx.request_id,
            language=ctx.language,
            flags=flags,
            method=method,
            within_budget=timings.total <= self._settings.deadline_ms,
        )
        self._metrics.record(response, ctx.trace)
        return response


def get_ask_service() -> AskService:
    return AskService(
        get_settings(),
        get_embedder(),
        get_store(),
        get_metrics_service(),
    )
