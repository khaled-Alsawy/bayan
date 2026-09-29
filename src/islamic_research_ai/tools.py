from __future__ import annotations

import time
from typing import Any

import httpx

from islamic_research_ai.contracts import (
    AttributionRole,
    AttributionV1,
    EvidenceItem,
    EvidenceSourceKind,
    ExecutionPlanV1,
    FetchQuranHandlerInputV1,
    FetchTafsirHandlerInputV1,
    Intent,
    QuranPointer,
    QuranSearchCandidateV1,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SearchQuranHandlerInputV1,
    SourceIdentityV1,
    SourceRevisionIdentity,
    TafsirFetchMode,
    TafsirPointer,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    TypedSourceReference,
    VerseKey,
)
from islamic_research_ai.logging import log_event
from islamic_research_ai.ports import (
    EmbeddingModelPort,
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    QuranSourcePort,
    RerankerPort,
    TafsirSourcePort,
)
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingError
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.retrieval.relevance import expanded_topic_query
from islamic_research_ai.rights import SourceOperation, SourcePolicy
from islamic_research_ai.routing import ProviderRouteError, ProviderRoutePolicy


class ExecutionPlanRejected(ValueError):
    pass


class PostgresQuranSource:
    """Searches non-citable candidates and hydrates exact source-scoped evidence."""

    def __init__(
        self,
        repository: PostgresRepository,
        embeddings: EmbeddingModelPort,
        *,
        reranker: RerankerPort | None = None,
        policy: SourcePolicy | None = None,
        qul_topic_arm_enabled: bool = False,
        qul_topic_arm_limit: int = 20,
        qul_topic_rerank_enabled: bool = False,
    ) -> None:
        if qul_topic_arm_limit <= 0:
            raise ValueError("qul_topic_arm_limit must be positive")
        self.repository = repository
        self.embeddings = embeddings
        self.reranker = reranker
        self.policy = policy
        self.qul_topic_arm_enabled = qul_topic_arm_enabled
        self.qul_topic_arm_limit = qul_topic_arm_limit
        self.qul_topic_rerank_enabled = qul_topic_rerank_enabled
        self.last_embedding_completion = None
        self.last_provider_failure: ProviderFailureDiagnostic | None = None
        self.last_embedding_degradation: dict[str, object] | None = None
        # A reranker only refines already-valid RRF candidates.  Its failure is observable in
        # the trace but must never turn a usable retrieval result into a failed-safe response.
        self.last_rerank_degradation: dict[str, object] | None = None
        self.last_qul_topic_rerank: dict[str, object] | None = None
        self.last_fusion_traces: list[dict[str, object]] = []
        self.last_embedding_queries: list[str] = []
        self._embedding_cache: dict[str, tuple[list[float], object]] = {}

    def reset_retrieval_diagnostics(self) -> None:
        """Start a turn-local diagnostic and embedding cache boundary."""

        self.last_fusion_traces = []
        self.last_embedding_queries = []
        self._embedding_cache = {}
        self.last_qul_topic_rerank = None

    async def search(self, query: str, *, limit: int) -> list[QuranSearchCandidateV1]:
        self.last_provider_failure = None
        self.last_embedding_degradation = None
        self.last_rerank_degradation = None
        self.last_qul_topic_rerank = None
        # The lexical arm benefits from morphological expansion; dense retrieval embeds the same
        # raw query after the formal-query replacement experiment measured worse.
        lexical_query = expanded_topic_query(query)
        self.last_embedding_queries.append(query)
        try:
            cached = self._embedding_cache.get(query)
            if cached is None:
                vectors, completion = await self.embeddings.embed([query])
                cached = (vectors[0], completion)
                self._embedding_cache[query] = cached
            vector, completion = cached
        except ProviderUnavailableError as exc:
            # Dense retrieval improves semantic recall but must not turn a usable local
            # lexical index into an opaque outage.  This is not a provider/model fallback:
            # it is the same local corpus searched without a vector query.
            self.last_embedding_degradation = {
                "provider_id": exc.diagnostic.provider_id,
                "category": exc.diagnostic.category,
                "http_status": exc.diagnostic.http_status,
                "safe_detail": exc.diagnostic.safe_detail,
                "mode": "LEXICAL_ONLY",
            }
            log_event("embedding_degraded_to_lexical", **self.last_embedding_degradation)
            return await self.repository.lexical_search(query=lexical_query, limit=limit)
        ProviderRoutePolicy.require_cloudflare_bge_m3_direct_free(completion)
        self.last_embedding_completion = completion
        audited_search = getattr(self.repository, "hybrid_search_with_audit", None)
        if audited_search is None:
            quran_candidates = await self.repository.hybrid_search(
                query=lexical_query, query_embedding=vector, limit=max(limit, 20)
            )
            quran_audit: dict[str, object] = {}
        else:
            quran_candidates, quran_audit = await audited_search(
                query=lexical_query, query_embedding=vector, limit=max(limit, 20)
            )
        # Al-Mukhtasar is an independently indexed Tafsir arm.  It adds topical
        # wording that a verse-only index can miss, but it yields only opaque
        # location candidates.  Canonical Tafsir remains available solely after
        # the normal exact hydration step.
        tafsir_candidates = await self._tafsir_first_candidates(query, vector, limit=20)
        topic_candidates: list[QuranSearchCandidateV1] = []
        topic_trace: dict[str, object] | None = None
        if self.qul_topic_arm_enabled:
            topic_rows = await self.repository.search_qul_topic_arm(
                query_embedding=vector, topic_limit=self.qul_topic_arm_limit
            )
            topic_candidates, topic_trace = self._qul_topic_candidates(topic_rows)
        candidates = self._fuse_candidate_arms(
            quran_candidates, tafsir_candidates, topic_candidates
        )
        if self.qul_topic_rerank_enabled and candidates:
            try:
                topic_rows = await self.repository.qul_topic_similarity_for_candidates(
                    query_embedding=vector,
                    locations=[
                        (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
                        for candidate in candidates
                    ],
                )
                if not topic_rows:
                    raise ValueError("no topic-label similarities for fused candidate pool")
                candidates, rerank_detail = self._rerank_by_qul_topic_similarity(
                    candidates, topic_rows
                )
                self.last_qul_topic_rerank = {"phase": "completed", **rerank_detail}
            except Exception as exc:
                self.last_qul_topic_rerank = {
                    "phase": "degraded",
                    "reason": type(exc).__name__,
                    "safe_detail": str(exc)[:240],
                }
        arms: dict[str, object] = {
            **quran_audit,
            "tafsir_first": self._trace_candidates(tafsir_candidates, "tafsir_rrf_score"),
        }
        if topic_trace is not None:
            arms["qul_topics"] = topic_trace
        self.last_fusion_traces.append(
            {
                "embedded_query": query,
                "arms": arms,
                "fused": self._trace_candidates(candidates, "fused_arm_score"),
            }
        )
        if self.reranker is None or not candidates:
            return candidates[:limit]
        try:
            passages = await self.repository.candidate_rerank_texts(candidates[:20])
            deployment = {
                "gateway_provider_id": "cloudflare",
                "effective_provider_id": "cloudflare",
                "model_id": self.reranker.model,
                "zero_data_retention": False,
                "provider_training": False,
                "aggregator_routing": False,
            }
            if self.policy is not None:
                # This is an authorization audit, not a success record.  It remains truthful if
                # the subsequent external operation degrades to the fused local ordering.
                aliases = {"tanzil-uthmani-v1.1", "quranenc-rowwad-en-v1.0.19"}
                aliases.update(
                    candidate.index_revision.resource_alias
                    for candidate in candidates[:20]
                    if candidate.index_revision.resource_alias.startswith("qul-")
                )
                for alias in sorted(aliases):
                    decision = self.policy.authorize(
                        alias, SourceOperation.EXTERNAL_RERANKING, deployment=deployment
                    )
                    await self.repository.record_activation(decision)
            scores, rerank_completion = await self.reranker.rerank(query=query, passages=passages)
            ProviderRoutePolicy.require_cloudflare_bge_reranker_direct_free(rerank_completion)
        except (
            ProviderUnavailableError,
            ProviderRouteError,
            httpx.HTTPError,
            ValueError,
            CloudflareEmbeddingError,
        ) as exc:
            diagnostic: dict[str, object] = {"error_type": type(exc).__name__}
            if isinstance(exc, ProviderUnavailableError):
                diagnostic.update(
                    {
                        "provider_id": exc.diagnostic.provider_id,
                        "category": exc.diagnostic.category,
                        "http_status": exc.diagnostic.http_status,
                        "safe_detail": exc.diagnostic.safe_detail,
                    }
                )
            else:
                diagnostic["safe_detail"] = str(exc)[:240]
            self.last_rerank_degradation = diagnostic
            log_event("rerank_degraded", model_id=self.reranker.model, **diagnostic)
            return candidates[:limit]
        reranked = sorted(zip(candidates[:20], scores, strict=True), key=lambda item: -item[1])
        result: list[QuranSearchCandidateV1] = []
        for rank, (candidate, score) in enumerate(reranked, start=1):
            ranking = candidate.ranking.model_copy(
                update={
                    "rank": rank,
                    "component_scores": {
                        **candidate.ranking.component_scores,
                        "cross_encoder_score": score,
                    },
                }
            )
            result.append(
                candidate.model_copy(update={"candidate_id": f"C{rank}", "ranking": ranking})
            )
        return result[:limit]

    @staticmethod
    def _rerank_by_qul_topic_similarity(
        candidates: list[QuranSearchCandidateV1], topic_rows: list[dict[str, object]]
    ) -> tuple[list[QuranSearchCandidateV1], dict[str, int]]:
        similarities = {
            (int(row["chapter"]), int(row["verse"])): float(row["topic_similarity"])
            for row in topic_rows
        }
        topic_ordered = sorted(
            candidates,
            key=lambda candidate: (
                -similarities.get(
                    (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse), 0.0
                ),
                candidate.verse_keys[0].chapter,
                candidate.verse_keys[0].verse,
            ),
        )
        topic_ranks = {
            (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse): rank
            for rank, candidate in enumerate(topic_ordered, start=1)
        }
        blend_scores = {
            (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse): (
                1.0 / (10 + candidate.ranking.rank)
                + 1.0
                / (
                    10
                    + topic_ranks[
                        (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
                    ]
                )
            )
            for candidate in candidates
        }
        reranked = sorted(
            candidates,
            key=lambda candidate: (
                -blend_scores[(candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)],
                candidate.verse_keys[0].chapter,
                candidate.verse_keys[0].verse,
            ),
        )
        result: list[QuranSearchCandidateV1] = []
        for rank, candidate in enumerate(reranked, start=1):
            key = (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
            ranking = candidate.ranking.model_copy(
                update={
                    "rank": rank,
                    "component_scores": {
                        **candidate.ranking.component_scores,
                        "qul_topic_similarity": similarities.get(key, 0.0),
                        "qul_topic_rank": float(topic_ranks[key]),
                        "qul_topic_blend_score": blend_scores[key],
                    },
                }
            )
            result.append(
                candidate.model_copy(update={"candidate_id": f"C{rank}", "ranking": ranking})
            )
        original_top = (candidates[0].verse_keys[0].chapter, candidates[0].verse_keys[0].verse)
        top_position = next(
            rank
            for rank, candidate in enumerate(result, start=1)
            if (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse) == original_top
        )
        return result, {
            "candidate_count": len(candidates),
            "labeled_candidate_count": len(similarities),
            "top_fused_candidate_positions_moved": top_position - 1,
        }

    @staticmethod
    def _trace_candidates(
        candidates: list[QuranSearchCandidateV1], score_key: str
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for candidate in candidates[:12]:
            scores = candidate.ranking.component_scores
            output.append(
                {
                    "reference": (
                        f"{candidate.verse_keys[0].chapter}:{candidate.verse_keys[0].verse}"
                    ),
                    "rank": candidate.ranking.rank,
                    "score": scores.get(score_key),
                    "arms": [
                        arm
                        for arm in ("quran", "tafsir", "topic")
                        if f"{arm}_arm_rrf" in scores
                    ],
                    "lexical_rank": scores.get("lexical_rank"),
                    "vector_rank": scores.get("vector_rank"),
                }
            )
        return output

    async def _tafsir_first_candidates(
        self, query: str, query_embedding: list[float], *, limit: int
    ) -> list[QuranSearchCandidateV1]:
        """Map the locally indexed Al-Mukhtasar arm to non-citable verse locations."""

        rows = await self.repository.search_tafsir_tier(
            query=query, query_embedding=query_embedding, limit=limit
        )
        candidates: list[QuranSearchCandidateV1] = []
        for rank, row in enumerate(rows, start=1):
            alias = str(row["resource_alias"])
            revision = SourceRevisionIdentity(
                source_id="qul-tafsir-search-index",
                resource_alias=alias,
                revision_id=str(row["revision_id"]),
                checksum=str(row["revision_checksum"]),
            )
            candidates.append(
                QuranSearchCandidateV1(
                    candidate_id=f"T{rank}",
                    verse_keys=[
                        VerseKey(
                            chapter=int(row["chapter"]),
                            verse=int(row["verse_start"]),
                        )
                    ],
                    index_revision=revision,
                    ranking=RetrievalProvenanceV1(
                        tool_id="search_tafsir_topics.v1",
                        method=RetrievalMethod.RERANKED,
                        index_revision=revision,
                        rank=rank,
                        component_scores={"tafsir_rrf_score": float(row["rrf_score"])},
                    ),
                )
            )
        return candidates

    @staticmethod
    def _qul_topic_candidates(
        rows: list[dict[str, object]],
    ) -> tuple[list[QuranSearchCandidateV1], dict[str, object]]:
        """Expand independently ranked QUL topics to opaque locations for rank-space fusion."""

        topics: dict[int, dict[str, object]] = {}
        expanded: dict[tuple[int, int], dict[str, object]] = {}
        for row in rows:
            topic_id = int(row["topic_id"])
            topic_rank = int(row["topic_rank"])
            topic = topics.setdefault(
                topic_id,
                {
                    "topic_id": topic_id,
                    "rank": topic_rank,
                    "score": float(row["topic_score"]),
                    "label": str(row["label"]),
                    "hierarchy_path": str(row["hierarchy_path"]),
                    "verse_count": 0,
                },
            )
            if row["chapter"] is None or row["verse"] is None:
                continue
            key = (int(row["chapter"]), int(row["verse"]))
            topic["verse_count"] = int(topic["verse_count"]) + 1
            existing = expanded.get(key)
            if existing is None or (topic_rank, -float(row["topic_score"])) < (
                int(existing["topic_rank"]),
                -float(existing["topic_score"]),
            ):
                expanded[key] = dict(row)
        ordered = sorted(
            expanded.values(),
            key=lambda row: (
                int(row["topic_rank"]),
                -float(row["topic_score"]),
                int(row["chapter"]),
                int(row["verse"]),
            ),
        )
        candidates: list[QuranSearchCandidateV1] = []
        for position, row in enumerate(ordered, start=1):
            index_revision = SourceRevisionIdentity(
                source_id="qul-topic-index",
                resource_alias="qul-ayah-topics-v1",
                revision_id=str(row["revision_id"]),
                checksum=str(row["revision_checksum"]),
            )
            topic_rank = int(row["topic_rank"])
            candidates.append(
                QuranSearchCandidateV1(
                    candidate_id=f"Q{position}",
                    verse_keys=[VerseKey(chapter=int(row["chapter"]), verse=int(row["verse"]))],
                    index_revision=index_revision,
                    ranking=RetrievalProvenanceV1(
                        tool_id="search_qul_topics.v1",
                        method=RetrievalMethod.DENSE,
                        index_revision=index_revision,
                        # One topic rank may intentionally fan out to several verses.  The outer
                        # RRF operates on that topic rank, never on concatenated source text.
                        rank=topic_rank,
                        component_scores={
                            "topic_rank": float(topic_rank),
                            "topic_score": float(row["topic_score"]),
                        },
                    ),
                )
            )
        return candidates, {
            "matched_topics": list(topics.values()),
            "expanded_verse_count": len(expanded),
            "candidates": PostgresQuranSource._trace_candidates(candidates, "topic_score"),
        }

    @staticmethod
    def _fuse_candidate_arms(
        quran_candidates: list[QuranSearchCandidateV1],
        tafsir_candidates: list[QuranSearchCandidateV1],
        topic_candidates: list[QuranSearchCandidateV1] | None = None,
    ) -> list[QuranSearchCandidateV1]:
        """Fuse Quran, Tafsir, and optional curated-topic locations before reranking."""

        fused: dict[tuple[int, int], tuple[QuranSearchCandidateV1, dict[str, float]]] = {}
        for arm_name, candidates in (
            ("quran", quran_candidates),
            ("tafsir", tafsir_candidates),
            ("topic", topic_candidates or []),
        ):
            for candidate in candidates:
                key = (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
                score = 1.0 / (60 + candidate.ranking.rank)
                existing = fused.get(key)
                if existing is None:
                    fused[key] = (candidate, {f"{arm_name}_arm_rrf": score})
                else:
                    selected, scores = existing
                    # Prefer the Quran index identity for a duplicate location:
                    # it is the better canonical hydration provenance, while the
                    # Tafsir arm score is still retained as a retrieval signal.
                    if arm_name == "quran":
                        selected = candidate
                    scores[f"{arm_name}_arm_rrf"] = score
                    fused[key] = (selected, scores)
        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -sum(item[1].values()),
                item[0].verse_keys[0].chapter,
                item[0].verse_keys[0].verse,
            ),
        )
        result: list[QuranSearchCandidateV1] = []
        for rank, (candidate, scores) in enumerate(ranked, start=1):
            ranking = candidate.ranking.model_copy(
                update={
                    "rank": rank,
                    "component_scores": {
                        **candidate.ranking.component_scores,
                        **scores,
                        "fused_arm_score": sum(scores.values()),
                    },
                }
            )
            result.append(
                candidate.model_copy(update={"candidate_id": f"C{rank}", "ranking": ranking})
            )
        return result

    async def hydrate_candidates(
        self, candidates: list[QuranSearchCandidateV1], *, limit: int
    ) -> list[EvidenceItem]:
        rows = await self.repository.hydrate_candidates(candidates[:limit])
        return self._rows_to_evidence(rows)

    async def hydrate_candidates_with_context(
        self, candidates: list[QuranSearchCandidateV1], *, limit: int
    ) -> list[EvidenceItem]:
        """Hydrate a bounded structural passage around each topic-search anchor.

        The window is document-structural, never topic-dependent: it lets a result near
        the end or start of a short passage retain the preceding/succeeding relationship
        that gives an individual verse its meaning. Missing verse numbers (including a
        chapter boundary) are simply absent from canonical hydration.
        """

        keys: list[VerseKey] = []
        for candidate in candidates[:limit]:
            anchor = candidate.verse_keys[0]
            keys.extend(
                VerseKey(chapter=anchor.chapter, verse=verse)
                for verse in range(max(1, anchor.verse - 2), anchor.verse + 3)
            )
        unique_keys = list(dict.fromkeys((key.chapter, key.verse) for key in keys))
        return await self.hydrate_references(
            [VerseKey(chapter=chapter, verse=verse) for chapter, verse in unique_keys],
            limit=max(len(unique_keys), limit),
        )

    async def hydrate_references(
        self, verse_keys: list[VerseKey], *, limit: int
    ) -> list[EvidenceItem]:
        rows = await self.repository.hydrate_references(verse_keys[:limit])
        return self._rows_to_evidence(rows)

    @staticmethod
    def _rows_to_evidence(rows: list[dict[str, Any]]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for row_index, row in enumerate(rows, start=1):
            candidate: QuranSearchCandidateV1 = row["candidate"]
            key: VerseKey = row["verse_key"]
            pointer = QuranPointer(kind="QURAN", verse_keys=[key])
            retrieval = RetrievalProvenanceV1(
                tool_id="get_quran_passage.v1",
                method=(
                    RetrievalMethod.EXACT
                    if candidate.ranking.method is RetrievalMethod.EXACT
                    else RetrievalMethod.HYBRID
                ),
                index_revision=candidate.ranking.index_revision,
                rank=candidate.ranking.rank,
                component_scores=candidate.ranking.component_scores,
            )
            arabic_alias = row["arabic_resource_alias"]
            translation_alias = row["translation_resource_alias"]
            arabic_attribution = row.get("arabic_attribution") or {}
            translation_attribution = row.get("translation_attribution") or {}
            reference_value = f"{key.chapter}:{key.verse}"
            items.append(
                EvidenceItem(
                    evidence_id=f"H{row_index}-AR",
                    source_kind=EvidenceSourceKind.QURAN,
                    canonical_pointer=pointer,
                    source_identity=SourceIdentityV1(
                        source_id=row["arabic_source_id"],
                        resource_alias=arabic_alias,
                        work_title=row["arabic_work_title"],
                        author_translator_or_grader=list(row["arabic_contributors"]),
                        canonical_base_url=row["arabic_base_url"],
                    ),
                    source_revision=SourceRevisionIdentity(
                        source_id=row["arabic_source_id"],
                        resource_alias=arabic_alias,
                        revision_id=str(row["arabic_revision_id"]),
                        checksum=row["arabic_revision_checksum"],
                    ),
                    language="ar",
                    exact_text=row["arabic_text"],
                    approved_quote_spans=[],
                    reference=TypedSourceReference(
                        scheme="quran",
                        value=reference_value,
                        canonical_link=f"https://tanzil.net/#{reference_value}",
                    ),
                    attributions=[
                        AttributionV1(
                            role=AttributionRole.SOURCE,
                            display_name=arabic_attribution.get("display_name", "Tanzil Project"),
                            required_notice=arabic_attribution.get("notice"),
                        )
                    ],
                    retrieval=retrieval,
                    content_hash=row["arabic_content_hash"],
                    source_policy_alias=arabic_alias,
                )
            )
            translation_contributors = list(row["translation_contributors"])
            translation_attributions = [
                AttributionV1(
                    role=(
                        AttributionRole.TRANSLATOR
                        if contributor == "Rowwad Translation Center"
                        else AttributionRole.PUBLISHER
                    ),
                    display_name=contributor,
                    required_notice=translation_attribution.get("notice"),
                )
                for contributor in translation_contributors
            ]
            items.append(
                EvidenceItem(
                    evidence_id=f"H{row_index}-EN",
                    source_kind=EvidenceSourceKind.QURAN_TRANSLATION,
                    canonical_pointer=pointer,
                    source_identity=SourceIdentityV1(
                        source_id=row["translation_source_id"],
                        resource_alias=translation_alias,
                        work_title=row["translation_work_title"],
                        author_translator_or_grader=translation_contributors,
                        canonical_base_url=row["translation_base_url"],
                    ),
                    source_revision=SourceRevisionIdentity(
                        source_id=row["translation_source_id"],
                        resource_alias=translation_alias,
                        revision_id=str(row["translation_revision_id"]),
                        checksum=row["translation_revision_checksum"],
                    ),
                    language="en",
                    exact_text=row["translation_text"],
                    approved_quote_spans=[],
                    reference=TypedSourceReference(
                        scheme="quran",
                        value=reference_value,
                        canonical_link=(
                            "https://quranenc.com/en/browse/english_rwwad/"
                            f"{key.chapter}/{key.verse}"
                        ),
                    ),
                    attributions=translation_attributions,
                    retrieval=retrieval,
                    content_hash=row["translation_content_hash"],
                    source_policy_alias=translation_alias,
                )
            )
        return items


class PostgresQulTafsirSource:
    """`TafsirSourcePort` backed by the locally-ingested QUL corpus (WP-0, V3 plan).

    This local source has no per-request network call, no rate limit,
    and no two-surah coverage gap: every passage was ingested once from the downloaded export.
    """

    def __init__(self, repository: PostgresRepository) -> None:
        self.repository = repository
        self.last_provider_failure: ProviderFailureDiagnostic | None = None

    async def fetch_passages(
        self,
        verse_keys: list[VerseKey],
        *,
        resource_aliases: list[str],
        character_budget: int,
    ) -> list[EvidenceItem]:
        self.last_provider_failure = None
        rows = await self.repository.fetch_tafsir_passages(
            resource_aliases=resource_aliases, verse_keys=verse_keys
        )
        items: list[EvidenceItem] = []
        for rank, row in enumerate(rows, start=1):
            items.extend(self._row_to_evidence(row, rank=rank, character_budget=character_budget))
        return items

    @staticmethod
    def _segment_text(value: str, budget: int) -> list[str]:
        """Split long passage text on paragraph/word boundaries within `budget` characters."""

        if len(value) <= budget:
            return [value]
        segments: list[str] = []
        current = ""
        for paragraph in value.splitlines():
            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= budget:
                current = candidate
                continue
            if current:
                segments.append(current)
            remaining = paragraph
            while len(remaining) > budget:
                split_at = remaining.rfind(" ", 0, budget + 1)
                if split_at <= 0:
                    split_at = budget
                segments.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
            current = remaining
        if current:
            segments.append(current)
        return [segment for segment in segments if segment]

    @staticmethod
    def _row_to_evidence(
        row: dict[str, Any], *, rank: int, character_budget: int
    ) -> list[EvidenceItem]:
        alias = row["resource_alias"]
        verse_keys = [
            VerseKey(chapter=row["chapter"], verse=verse)
            for verse in range(row["verse_start"], row["verse_end"] + 1)
        ]
        reference_value = f"{row['chapter']}:{row['verse_start']}-{row['verse_end']}"
        attribution_meta = row.get("attribution") or {}
        contributors = list(row.get("author_translator_or_grader") or [])
        attributions = [
            AttributionV1(role=AttributionRole.AUTHOR, display_name=name, required_notice=None)
            for name in contributors
        ]
        attributions.append(
            AttributionV1(
                role=AttributionRole.SOURCE,
                display_name=attribution_meta.get("display_name", "Quranic Universal Library"),
                required_notice=attribution_meta.get("notice"),
            )
        )
        segments = PostgresQulTafsirSource._segment_text(row["passage_text"], character_budget)
        items: list[EvidenceItem] = []
        for segment_index, segment in enumerate(segments, start=1):
            passage_id = f"qul:{alias}:{row['chapter']}:{row['verse_start']}-{row['verse_end']}"
            items.append(
                EvidenceItem(
                    evidence_id=f"QUL-{passage_id}-{segment_index}",
                    source_kind=EvidenceSourceKind.TAFSIR,
                    canonical_pointer=TafsirPointer(
                        kind="TAFSIR",
                        passage_id=passage_id,
                        verse_keys=verse_keys,
                    ),
                    source_identity=SourceIdentityV1(
                        source_id=row["source_id"],
                        resource_alias=alias,
                        work_title=row["work_title"],
                        author_translator_or_grader=contributors,
                        canonical_base_url=row.get("canonical_base_url"),
                    ),
                    source_revision=SourceRevisionIdentity(
                        source_id=row["source_id"],
                        resource_alias=alias,
                        revision_id=str(row["revision_id"]),
                        checksum=row["revision_checksum"],
                    ),
                    language=row.get("language", "ar"),
                    exact_text=segment,
                    approved_quote_spans=[],
                    reference=TypedSourceReference(
                        scheme="tafsir",
                        value=f"{alias}@{reference_value}",
                        canonical_link=None,
                    ),
                    attributions=attributions,
                    retrieval=RetrievalProvenanceV1(
                        tool_id="get_tafsir_passages.v1",
                        method=RetrievalMethod.EXACT,
                        index_revision=None,
                        rank=rank,
                        component_scores={},
                    ),
                    content_hash=f"sha256:{row['content_hash']}",
                    source_policy_alias=alias,
                    # Only the passage's OWN footnotes are attached -- never a reference
                    # borrowed from another passage or fabricated. See ingestion/tafsir_qul.py.
                    hadith_references=list(row.get("hadith_references") or []),
                )
            )
        return items


class WorkflowExecutor:
    def __init__(
        self,
        source: QuranSourcePort,
        *,
        search_limit: int,
        hydrate_limit: int,
        tafsir_source: TafsirSourcePort | None = None,
    ) -> None:
        self.source = source
        self.tafsir_source = tafsir_source
        self.search_limit = search_limit
        self.hydrate_limit = hydrate_limit
        self.last_candidate_selection: list[dict[str, object]] = []

    async def execute(self, plan: ExecutionPlanV1) -> dict[str, ToolResult[Any]]:
        self._validate_complete_plan(plan)
        self.last_candidate_selection = []
        reset_diagnostics = getattr(self.source, "reset_retrieval_diagnostics", None)
        if reset_diagnostics is not None:
            reset_diagnostics()
        results: dict[str, ToolResult[Any]] = {}
        for node in plan.nodes:
            started = time.perf_counter()
            try:
                if node.tool_id == "search_quran_topics.v1":
                    handler_input = node.validated_input
                    if not isinstance(handler_input, SearchQuranHandlerInputV1):
                        raise ExecutionPlanRejected("Search node input type is invalid")
                    candidates = await self.source.search(
                        handler_input.query,
                        limit=min(node.result_limit, self.search_limit),
                    )
                    results[node.node_id] = ToolResult[list[QuranSearchCandidateV1]](
                        status=(ToolResultStatus.SUCCESS if candidates else ToolResultStatus.EMPTY),
                        data=candidates,
                        evidence_items=[],
                        warnings=[],
                        duration_ms=self._duration_ms(started),
                        error=None,
                    )
                    continue

                if node.tool_id == "get_tafsir_passages.v1":
                    handler_input = node.validated_input
                    if not isinstance(handler_input, FetchTafsirHandlerInputV1):
                        raise ExecutionPlanRejected("Tafsir node input type is invalid")
                    if self.tafsir_source is None:
                        raise ProviderUnavailableError(
                            ProviderFailureDiagnostic(
                                provider_id="qul",
                                category="UNAVAILABLE",
                                http_status=None,
                                retry_after_ms=None,
                                quota_limit_tokens=None,
                                quota_used_tokens=None,
                                quota_requested_tokens=None,
                                request_id=None,
                                safe_detail="Local QUL Tafsir source is not configured",
                            )
                        )
                    verse_keys = handler_input.verse_keys
                    if handler_input.late_binding is not None:
                        candidates = self._late_bound_candidates(
                            results,
                            handler_input.late_binding.upstream_node_id,
                            handler_input.late_binding.additional_upstream_node_ids,
                            handler_input.late_binding.maximum_items,
                        )
                        verse_keys = [
                            key for candidate in candidates for key in candidate.verse_keys
                        ]
                    hydrated = await self.tafsir_source.fetch_passages(
                        verse_keys,
                        resource_aliases=handler_input.tafsir_resource_aliases,
                        character_budget=handler_input.character_budget,
                    )
                    results[node.node_id] = ToolResult[list[VerseKey]](
                        status=(ToolResultStatus.SUCCESS if hydrated else ToolResultStatus.EMPTY),
                        data=[
                            key for item in hydrated for key in item.canonical_pointer.verse_keys
                        ],
                        evidence_items=hydrated,
                        warnings=[],
                        duration_ms=self._duration_ms(started),
                        error=None,
                    )
                    continue

                handler_input = node.validated_input
                if not isinstance(handler_input, FetchQuranHandlerInputV1):
                    raise ExecutionPlanRejected("Fetch node input type is invalid")
                if handler_input.late_binding is not None:
                    candidates = self._late_bound_candidates(
                        results,
                        handler_input.late_binding.upstream_node_id,
                        handler_input.late_binding.additional_upstream_node_ids,
                        handler_input.late_binding.maximum_items,
                    )
                    self._record_late_binding_selection(
                        results,
                        handler_input.late_binding.upstream_node_id,
                        handler_input.late_binding.additional_upstream_node_ids,
                        handler_input.late_binding.maximum_items,
                        node.node_id,
                    )
                    hydration_limit = min(
                        handler_input.late_binding.maximum_items,
                        node.result_limit,
                        self.hydrate_limit,
                    )
                    contextual_hydrator = getattr(
                        self.source, "hydrate_candidates_with_context", None
                    )
                    if handler_input.include_adjacent_verses and contextual_hydrator:
                        hydrated = await contextual_hydrator(candidates, limit=hydration_limit)
                    else:
                        hydrated = await self.source.hydrate_candidates(
                            candidates, limit=hydration_limit
                        )
                else:
                    hydrated = await self.source.hydrate_references(
                        handler_input.verse_keys,
                        limit=min(node.result_limit, self.hydrate_limit),
                    )
                results[node.node_id] = ToolResult[list[VerseKey]](
                    status=(ToolResultStatus.SUCCESS if hydrated else ToolResultStatus.EMPTY),
                    data=[
                        key
                        for item in hydrated
                        if item.source_kind is EvidenceSourceKind.QURAN
                        for key in item.canonical_pointer.verse_keys
                    ],
                    evidence_items=hydrated,
                    warnings=[],
                    duration_ms=self._duration_ms(started),
                    error=None,
                )
            except ExecutionPlanRejected:
                raise
            except ProviderUnavailableError as exc:
                code = {
                    "RATE_LIMITED": ToolErrorCode.RATE_LIMITED,
                    "QUOTA_EXHAUSTED": ToolErrorCode.RATE_LIMITED,
                    "TIMEOUT": ToolErrorCode.TIMEOUT,
                    "UNAVAILABLE": ToolErrorCode.SOURCE_UNAVAILABLE,
                }[exc.diagnostic.category]
                results[node.node_id] = ToolResult[Any](
                    status=ToolResultStatus.ERROR,
                    data=None,
                    evidence_items=[],
                    warnings=[],
                    duration_ms=self._duration_ms(started),
                    error=ToolError(
                        code=code,
                        retryable=True,
                        safe_message_key="research_source.temporarily_unavailable",
                        upstream_status=exc.diagnostic.http_status,
                        retry_after_ms=exc.diagnostic.retry_after_ms,
                    ),
                )
                break
            except Exception:
                results[node.node_id] = ToolResult[Any](
                    status=ToolResultStatus.ERROR,
                    data=None,
                    evidence_items=[],
                    warnings=[],
                    duration_ms=self._duration_ms(started),
                    error=ToolError(
                        code=ToolErrorCode.INTERNAL,
                        retryable=False,
                        safe_message_key="quran_source.failed_safe",
                        upstream_status=None,
                        retry_after_ms=None,
                    ),
                )
                break
        return results

    @staticmethod
    def _late_binding_effective_score(candidate: QuranSearchCandidateV1) -> float:
        """Use the latest available ranking stage when choosing hydrated anchors."""

        scores = candidate.ranking.component_scores
        for score_key in (
            "cross_encoder_score",
            "qul_topic_blend_score",
            "fused_arm_score",
        ):
            score = scores.get(score_key)
            if score is not None:
                return float(score)
        return 0.0

    @staticmethod
    def _late_bound_candidates(
        results: dict[str, ToolResult[Any]],
        primary_id: str,
        additional_ids: list[str],
        limit: int,
    ) -> list[QuranSearchCandidateV1]:
        """Deduplicate then select shared anchors across all topic-search arms."""

        candidates: dict[tuple[int, int], QuranSearchCandidateV1] = {}
        for node_id in [primary_id, *additional_ids]:
            for candidate in list(results[node_id].data or []):
                key = (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
                current = candidates.get(key)
                score = WorkflowExecutor._late_binding_effective_score(candidate)
                current_score = float("-inf")
                if current is not None:
                    current_score = WorkflowExecutor._late_binding_effective_score(current)
                if current is None or score > current_score:
                    candidates[key] = candidate
        return sorted(
            candidates.values(),
            key=lambda item: (
                -WorkflowExecutor._late_binding_effective_score(item),
                item.ranking.rank,
            ),
        )[:limit]

    @staticmethod
    def _trace_late_bound_candidates(
        candidates: list[QuranSearchCandidateV1],
    ) -> list[dict[str, object]]:
        traced = PostgresQuranSource._trace_candidates(candidates, "fused_arm_score")
        for item, candidate in zip(traced, candidates[:12], strict=True):
            item["effective_score"] = WorkflowExecutor._late_binding_effective_score(candidate)
        return traced

    def _record_late_binding_selection(
        self,
        results: dict[str, ToolResult[Any]],
        primary_id: str,
        additional_ids: list[str],
        limit: int,
        node_id: str,
    ) -> None:
        selected = self._late_bound_candidates(results, primary_id, additional_ids, limit)
        all_candidates = self._late_bound_candidates(results, primary_id, additional_ids, 10_000)
        selected_keys = {
            (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse)
            for candidate in selected
        }
        dropped = [
            candidate
            for candidate in all_candidates
            if (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse) not in selected_keys
        ]
        self.last_candidate_selection.append(
            {
                "operation_id": node_id,
                "maximum_items": limit,
                "selected": self._trace_late_bound_candidates(selected),
                "dropped": [
                    {
                        **item,
                        "reason": "LATE_BINDING_MAXIMUM_ITEMS",
                    }
                    for item in self._trace_late_bound_candidates(dropped)
                ],
            }
        )

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))

    @staticmethod
    def _validate_complete_plan(plan: ExecutionPlanV1) -> None:
        if plan.intent is Intent.QURAN_TOPIC_SEARCH:
            searches = [node for node in plan.nodes if node.tool_id == "search_quran_topics.v1"]
            fetches = [node for node in plan.nodes if node.tool_id == "get_quran_passage.v1"]
            tafsirs = [node for node in plan.nodes if node.tool_id == "get_tafsir_passages.v1"]
            expected_nodes = len(searches) + len(fetches) + len(tafsirs)
            if (
                not searches
                or len(fetches) != 1
                or len(tafsirs) > 1
                or len(plan.nodes) != expected_nodes
            ):
                raise ExecutionPlanRejected(
                    "A Quran topic plan must contain well-formed search/fetch groups"
                )
            search_ids = [search.node_id for search in searches]
            fetch = fetches[0]
            for search in searches:
                if (
                    not search.node_id.startswith("node_search_quran")
                    or search.depends_on
                    or not isinstance(search.validated_input, SearchQuranHandlerInputV1)
                ):
                    raise ExecutionPlanRejected("Quran topic execution plan is malformed")
            if (
                fetch.node_id != "node_fetch_quran"
                or fetch.depends_on != search_ids
                or not isinstance(fetch.validated_input, FetchQuranHandlerInputV1)
                or fetch.validated_input.late_binding is None
                or [
                    fetch.validated_input.late_binding.upstream_node_id,
                    *fetch.validated_input.late_binding.additional_upstream_node_ids,
                ]
                != search_ids
            ):
                raise ExecutionPlanRejected("Quran topic execution plan is malformed")
            if tafsirs:
                tafsir = tafsirs[0]
                if (
                    tafsir.node_id != "node_fetch_tafsir"
                    or tafsir.depends_on != search_ids
                    or not isinstance(tafsir.validated_input, FetchTafsirHandlerInputV1)
                    or tafsir.validated_input.late_binding is None
                    or [
                        tafsir.validated_input.late_binding.upstream_node_id,
                        *tafsir.validated_input.late_binding.additional_upstream_node_ids,
                    ]
                    != search_ids
                ):
                    raise ExecutionPlanRejected("Topic Tafsir execution plan is malformed")
        elif plan.intent is Intent.QURAN_PASSAGE_LOOKUP:
            if len(plan.nodes) != 1:
                raise ExecutionPlanRejected("A Quran passage plan must contain one fetch")
            fetch = plan.nodes[0]
            if (
                fetch.node_id != "node_fetch_quran"
                or fetch.tool_id != "get_quran_passage.v1"
                or fetch.depends_on
                or not isinstance(fetch.validated_input, FetchQuranHandlerInputV1)
                or fetch.validated_input.late_binding is not None
                or not fetch.validated_input.verse_keys
            ):
                raise ExecutionPlanRejected("Quran passage execution plan is malformed")
        elif plan.intent in {Intent.EXPLAIN_QURAN_VERSE, Intent.COMPARE_TAFSIR}:
            if len(plan.nodes) != 2:
                raise ExecutionPlanRejected("A Tafsir plan must contain two exact fetches")
            quran, tafsir = plan.nodes
            if (
                quran.node_id != "node_fetch_quran"
                or quran.tool_id != "get_quran_passage.v1"
                or quran.depends_on
                or not isinstance(quran.validated_input, FetchQuranHandlerInputV1)
                or quran.validated_input.late_binding is not None
                or not quran.validated_input.verse_keys
                or tafsir.node_id != "node_fetch_tafsir"
                or tafsir.tool_id != "get_tafsir_passages.v1"
                or tafsir.depends_on
                or not isinstance(tafsir.validated_input, FetchTafsirHandlerInputV1)
                or tafsir.validated_input.late_binding is not None
                or tafsir.validated_input.verse_keys != quran.validated_input.verse_keys
            ):
                raise ExecutionPlanRejected("Tafsir execution plan is malformed")
            expected_mode = (
                TafsirFetchMode.EXPLANATION_ONE
                if plan.intent is Intent.EXPLAIN_QURAN_VERSE
                else TafsirFetchMode.COMPARISON_EXACTLY_TWO
            )
            expected_count = 1 if expected_mode is TafsirFetchMode.EXPLANATION_ONE else 2
            aliases = tafsir.validated_input.tafsir_resource_aliases
            if (
                tafsir.validated_input.mode is not expected_mode
                or len(aliases) != expected_count
                or len(set(aliases)) != expected_count
                or plan.evidence_requirements.required_tafsir_aliases != aliases
                or plan.evidence_requirements.require_two_distinct_tafsir_resources
                is not (expected_count == 2)
            ):
                raise ExecutionPlanRejected("Tafsir mode or resource invariants are malformed")
        else:
            raise ExecutionPlanRejected("Intent is not executable by the Quran core")
