from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from islamic_research_ai.chat_routing import TurnChatRouteState
from islamic_research_ai.composition_layer import AnswerComposer, CompositionProviderFailure
from islamic_research_ai.contracts import (
    AnswerDraftV1,
    AnswerStatus,
    ContextField,
    ContextProvenance,
    ContextValueV1,
    ConversationTurnV1,
    DeterministicResolutionContextV1,
    EvidenceRequirementFailureReason,
    EvidenceRequirementFailureV1,
    EvidenceRequirementsV1,
    EvidenceSet,
    ExecutionPlanV1,
    FetchQuranPassageIntentV1,
    FetchTafsirPassagesIntentV1,
    Intent,
    NoticeCode,
    PersistedEvidenceRecordV1,
    ProviderCompletion,
    QuranSearchCandidateV1,
    RenderedClaimBlockV1,
    RenderedNoticeBlockV1,
    RenderedOutputKind,
    ResearchPlanV1,
    ResearchRequest,
    ResearchResponseV1,
    ResolutionProvenance,
    SanitizedTurnV1,
    ScopeDecisionV1,
    ScopeReason,
    ScopeStatus,
    SearchQuranHandlerInputV1,
    SearchQuranTopicsIntentV1,
    TafsirFetchMode,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    TraceEventV1,
    TurnRole,
    WorkingContextV1,
    derive_rendered_output_kind,
    no_clarification,
)
from islamic_research_ai.evidence import EvidenceAssembler
from islamic_research_ai.logging import log_event
from islamic_research_ai.planner import (
    GroqPlanner,
    PlannerFailedSafe,
    PlannerTemporarilyUnavailable,
    build_deterministic_plan,
    normalize_user_query,
    resolve_deterministically,
)
from islamic_research_ai.ports import ProviderFailureDiagnostic
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.retrieval.relevance import (
    assess_topic_evidence,
    empty_topic_relevance_assessment,
    enforce_ranked_relevance_boundary,
    semantic_subtargets,
)
from islamic_research_ai.scope import ScopePolicy
from islamic_research_ai.synthesis import (
    CanonicalTextGroundingError,
    ExactRenderer,
    GroqSynthesizer,
    GroundingError,
    GroundingValidator,
    SynthesisTemporarilyUnavailable,
    clarification_response,
    deterministic_contextual_topic_draft,
    deterministic_exact_lookup_draft,
    deterministic_tafsir_source_draft,
    deterministic_topic_source_view_draft,
    terminal_response,
)
from islamic_research_ai.tools import PostgresQuranSource, WorkflowExecutor
from islamic_research_ai.tracing import (
    _TRACE_RECORDER,
    TraceRecorder,
    rejected_synthesis_detail,
)
from islamic_research_ai.workflow import WorkflowCompiler, quran_evidence_requirements


@dataclass(frozen=True)
class RequestExecutionIdentity:
    conversation_id: UUID
    turn_id: UUID
    correlation_id: UUID


@dataclass
class ResearchTrace:
    identity: RequestExecutionIdentity
    context: WorkingContextV1
    resolution: DeterministicResolutionContextV1
    scope: ScopeDecisionV1
    plan: ResearchPlanV1 | None
    execution_plan: ExecutionPlanV1 | None
    results: dict[str, ToolResult[Any]]
    candidates: list[QuranSearchCandidateV1]
    evidence: EvidenceSet
    planner_completions: list[ProviderCompletion]
    synthesis_completions: list[ProviderCompletion]
    query_embedding_completion: ProviderCompletion | None
    embedding_degradation: dict[str, object] | None
    rerank_degradation: dict[str, object] | None
    terminal_reason: str | None
    provider_failure: ProviderFailureDiagnostic | None
    chat_route: dict[str, Any]
    chat_completion_records_v22: dict[str, list[dict[str, Any]]]
    events: list[TraceEventV1]


class ResearchService:
    def __init__(
        self,
        *,
        planner: GroqPlanner,
        compiler: WorkflowCompiler,
        executor: WorkflowExecutor,
        source: PostgresQuranSource,
        evidence_assembler: EvidenceAssembler,
        synthesizer: GroqSynthesizer,
        renderer: ExactRenderer,
        repository: PostgresRepository,
        scope_policy: ScopePolicy,
        force_local_chat: bool = False,
        deterministic_topic_plans: bool = False,
        enable_topic_tafsir: bool = False,
        topic_tafsir_alias: str = "qul-tafsir-almukhtasar-ar-v1",
        model_relevance_review_enabled: bool = False,
        relevance_ranked_retention_enabled: bool = False,
        relevance_distinctive_terms_enabled: bool = False,
        honest_insufficient_fallback_enabled: bool = False,
        relevance_retained_passage_limit: int = 5,
        relevance_empty_match_retains_anchors: bool = True,
        answer_composer: AnswerComposer | None = None,
        answer_composer_enabled: bool = True,
        developer_trace_enabled: bool = True,
    ) -> None:
        self.planner = planner
        self.compiler = compiler
        self.executor = executor
        self.source = source
        self.evidence_assembler = evidence_assembler
        self.synthesizer = synthesizer
        self.renderer = renderer
        self.repository = repository
        self.scope_policy = scope_policy
        self.force_local_chat = force_local_chat
        self.deterministic_topic_plans = deterministic_topic_plans
        self.enable_topic_tafsir = enable_topic_tafsir
        self.topic_tafsir_alias = topic_tafsir_alias
        # The local structured reviewer is quarantined after a 16/16 failed-run result.
        # Deterministic lexical relevance is the production path; this switch exists only
        # for controlled diagnosis until the reviewer can prove it adds value.
        self.model_relevance_review_enabled = model_relevance_review_enabled
        self.relevance_ranked_retention_enabled = relevance_ranked_retention_enabled
        self.relevance_distinctive_terms_enabled = relevance_distinctive_terms_enabled
        self.honest_insufficient_fallback_enabled = honest_insufficient_fallback_enabled
        self.relevance_retained_passage_limit = relevance_retained_passage_limit
        self.relevance_empty_match_retains_anchors = relevance_empty_match_retains_anchors
        # WP-2 (V3 plan): composes claims into reader-facing prose, gated by
        # AntiFabricationGate. `answer_composer=None` (e.g. not wired yet) or
        # `answer_composer_enabled=False` both skip composition entirely -- the response is
        # then exactly the pre-WP-2 claims/sources/notices, unchanged.
        self.answer_composer = answer_composer
        self.answer_composer_enabled = answer_composer_enabled
        self.developer_trace_enabled = developer_trace_enabled

    @staticmethod
    def _ambiguity_probe_plan(plan: ResearchPlanV1, message: str) -> ResearchPlanV1:
        """Turn a topic-ambiguity hypothesis into one bounded retrieval pass.

        A model clarification proposal deliberately contains no tools.  That is
        appropriate for missing references, but it must not let a topic-polysemy
        guess make an unobserved claim about what the corpus can answer.
        """

        label = plan.semantic_target_summary.normalized_label or normalize_user_query(message)
        summary = plan.semantic_target_summary.model_copy(
            update={
                "normalized_label": label,
                "entity_labels": plan.semantic_target_summary.entity_labels or [label],
            }
        )
        return plan.model_copy(
            update={
                "intent": Intent.QURAN_TOPIC_SEARCH,
                "semantic_target_summary": summary,
                "clarification": no_clarification(),
                "requested_operations": [
                    SearchQuranTopicsIntentV1(
                        operation="SEARCH_QURAN_TOPICS",
                        query=label,
                        language=plan.response_language,
                        surah_filters=[],
                    ),
                    FetchQuranPassageIntentV1(
                        operation="FETCH_QURAN_PASSAGE",
                        target="QURAN_SEARCH_CANDIDATES",
                        translation_resource_alias="quranenc-rowwad-en-v1.0.19",
                        include_adjacent_verses=True,
                    ),
                ],
            }
        )

    @staticmethod
    def _bare_topic_term_count(message: str) -> int:
        """Count visible terms without assigning a Quran meaning to any of them."""

        return len(re.findall(r"[\w\u0600-\u06ff]+", normalize_user_query(message)))

    @staticmethod
    def _ranking_observation(candidates: list[QuranSearchCandidateV1]) -> dict[str, object]:
        """Expose bounded numeric rank signals without recording candidate/source text."""

        values_by_signal: dict[str, list[float]] = {}
        for candidate in candidates:
            for signal, value in candidate.ranking.component_scores.items():
                values_by_signal.setdefault(signal, []).append(float(value))
        return {
            "ranked_candidate_count": len(candidates),
            "ranking_signal_ranges": {
                signal: {"count": len(values), "min": min(values), "max": max(values)}
                for signal, values in sorted(values_by_signal.items())[:12]
            },
        }

    @staticmethod
    def _retained_references(evidence: EvidenceSet) -> list[str]:
        """Return a bounded, source-text-free view of relevance-retained verse keys."""

        references = {
            f"{key.chapter}:{key.verse}"
            for item in evidence.items_by_id.values()
            for key in item.canonical_pointer.verse_keys
        }
        return sorted(references)[:24]

    async def research(
        self,
        request: ResearchRequest,
        *,
        recent_user_turns: list[ConversationTurnV1] | None = None,
        on_trace_event: Callable[[TraceEventV1], None] | None = None,
    ) -> tuple[ResearchResponseV1, ResearchTrace]:
        recorder = TraceRecorder(self.developer_trace_enabled, on_trace_event)
        _TRACE_RECORDER.set(recorder)
        route_state = TurnChatRouteState(force_local=self.force_local_chat)
        identity = RequestExecutionIdentity(
            conversation_id=request.conversation_id,
            turn_id=uuid4(),
            correlation_id=uuid4(),
        )
        context = self._working_context(request, identity.turn_id, recent_user_turns or [])
        resolution = resolve_deterministically(request.message)
        # A reference may only be inherited from the immediately preceding turn and only when
        # the current wording is explicitly anaphoric. Walking the full transcript leaked a
        # distant verse into unrelated topics (F-10).
        anaphoric = re.search(
            r"(?:هذه\s+الآية|هذه\s+الاية|فيها|اشرحها|فسرها|this\s+verse|\bit\b)",
            request.message,
            re.IGNORECASE,
        )
        if not resolution.resolved_quran_references and anaphoric and recent_user_turns:
            prior_resolution = resolve_deterministically(recent_user_turns[-1].user_message)
            if prior_resolution.resolved_quran_references:
                resolution = resolution.model_copy(
                    update={
                        "resolved_quran_references": [
                            item.model_copy(
                                update={
                                    "provenance": (
                                        ResolutionProvenance.WORKING_CONTEXT_EXACT_REFERENCE
                                    )
                                }
                            )
                            for item in prior_resolution.resolved_quran_references
                        ]
                    }
                )
        scope = self.scope_policy.precheck(request.message, resolution)
        recorder.emit(
            "scope",
            "completed",
            "Scope precheck completed",
            detail={"status": scope.status.value, "reason": scope.reason.value},
        )
        empty_evidence = self._empty_evidence(identity.turn_id)

        if scope.status is ScopeStatus.UNSUPPORTED:
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.UNSUPPORTED_SCOPE,
                notice_code=NoticeCode.UNSUPPORTED_SCOPE,
                text="This request is outside the supported Quran research scope.",
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                evidence=empty_evidence,
                chat_route_state=route_state,
            )
            await self._persist_response(response, empty_evidence)
            return response, trace

        try:
            recorder.emit("planner", "started", "Planning research workflow")
            plan = build_deterministic_plan(
                request.message,
                response_language=request.locale,
                resolution=resolution,
                enable_topics=self.deterministic_topic_plans,
            )
            deterministic_plan = plan is not None
            if plan is None:
                plan, planner_completions = await self.planner.create_plan(
                    user_message=normalize_user_query(request.message),
                    working_context=context,
                    deterministic_resolution=resolution,
                    route_state=route_state,
                )
            else:
                planner_completions = []
            recorder.emit(
                "planner",
                "completed",
                "Research plan accepted",
                detail={
                    "intent": plan.intent.value,
                    "operations": [item.operation for item in plan.requested_operations],
                    "clarification_required": plan.clarification.required,
                },
            )
        except PlannerTemporarilyUnavailable as exc:
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.TEMPORARILY_UNAVAILABLE,
                notice_code=NoticeCode.SOURCE_UNAVAILABLE,
                text="The required planning provider is temporarily unavailable.",
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                evidence=empty_evidence,
                planner_completions=exc.completions,
                terminal_reason=str(exc),
                provider_failure=exc.diagnostic,
                chat_route_state=route_state,
            )
            await self._persist_response(response, empty_evidence)
            return response, trace
        except PlannerFailedSafe as exc:
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.FAILED_SAFE,
                notice_code=NoticeCode.QUESTION_NOT_UNDERSTOOD,
                text="The research plan could not be validated safely.",
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                evidence=empty_evidence,
                planner_completions=exc.completions,
                terminal_reason=str(exc),
                chat_route_state=route_state,
            )
            await self._persist_response(response, empty_evidence)
            return response, trace

        if (
            self.enable_topic_tafsir
            and plan.intent is Intent.QURAN_TOPIC_SEARCH
            and plan.response_language.casefold().startswith("ar")
            and not any(
                isinstance(operation, FetchTafsirPassagesIntentV1)
                for operation in plan.requested_operations
            )
        ):
            # Arabic topic research carries the configured locally stored QUL commentary.
            # English topic turns deliberately carry no Tafsir until an approved English
            # source is available; Arabic commentary must not be substituted for them.
            tafsir_alias = self.topic_tafsir_alias
            plan = plan.model_copy(
                update={
                    "requested_operations": [
                        *plan.requested_operations,
                        FetchTafsirPassagesIntentV1(
                            operation="FETCH_TAFSIR_PASSAGES",
                            target="QURAN_SEARCH_CANDIDATES",
                            mode=TafsirFetchMode.EXPLANATION_ONE,
                            tafsir_resource_aliases=[tafsir_alias],
                            language=plan.response_language,
                        ),
                    ]
                }
            )

        ambiguity_hypothesis = (
            plan.clarification.required
            and plan.intent is Intent.QURAN_TOPIC_SEARCH
            and plan.clarification.reason is not None
            and plan.clarification.reason.value == "AMBIGUOUS_REFERENCE"
            and plan.clarification.missing_fields == [ContextField.TOPIC]
        )
        if plan.clarification.required and not ambiguity_hypothesis:
            scope = ScopeDecisionV1(
                schema_version="1",
                status=ScopeStatus.CLARIFICATION_REQUIRED,
                reason=ScopeReason.NEEDS_CLARIFICATION,
                signals=scope.signals,
                clarification=plan.clarification,
                user_message_key="scope.needs_semantic_clarification",
            )
            response = clarification_response(
                **self._identity_args(identity), clarification=plan.clarification
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                plan=plan,
                evidence=empty_evidence,
                planner_completions=planner_completions,
                terminal_reason="Observable semantic ambiguity requires clarification",
                chat_route_state=route_state,
            )
            await self._persist_response(
                response, empty_evidence, plan=plan, planner_metadata=planner_completions
            )
            return response, trace
        pending_clarification = plan.clarification if ambiguity_hypothesis else None
        if pending_clarification is not None:
            recorder.emit(
                "verification",
                "started",
                "Planner ambiguity hypothesis requires corpus retrieval",
                detail={
                    "reason": pending_clarification.reason.value,
                    "candidate_meaning_count": len(pending_clarification.candidates),
                },
            )
            plan = self._ambiguity_probe_plan(plan, request.message)
        execution_plan = self.compiler.compile(
            plan, deterministic_resolution=resolution, scope=scope
        )
        recorder.emit(
            "retrieval",
            "started",
            "Executing retrieval workflow",
            detail={"operation_count": len(execution_plan.nodes)},
        )
        results = await self.executor.execute(execution_plan)
        embedded_queries = list(getattr(self.source, "last_embedding_queries", []))
        if not embedded_queries:
            embedded_queries = [
                node.validated_input.query
                for node in execution_plan.nodes
                if isinstance(node.validated_input, SearchQuranHandlerInputV1)
            ]
        if embedded_queries:
            recorder.emit(
                "retrieval",
                "completed",
                "Embedded dense retrieval query",
                detail={"queries": embedded_queries},
            )
        for fusion_trace in getattr(self.source, "last_fusion_traces", []):
            recorder.emit(
                "fusion",
                "completed",
                "Fusion arm candidates and ordering",
                detail=fusion_trace,
            )
        for selection in self.executor.last_candidate_selection:
            recorder.emit(
                "fusion",
                "completed",
                "Candidates dropped before evidence assembly",
                detail=selection,
            )
        for node_id, result in results.items():
            node = "tafsir" if "tafsir" in node_id else "retrieval"
            phase = "completed" if result.status is not ToolResultStatus.ERROR else "failed"
            recorder.emit(
                node,
                phase,
                f"{node_id} {result.status.value.lower()}",
                detail={
                    "operation_id": node_id,
                    "result_status": result.status.value,
                    "item_count": len(result.data) if isinstance(result.data, list) else None,
                    "database": "postgres",
                    "stage": (
                        "tafsir_hydration"
                        if node == "tafsir"
                        else "canonical_hydration"
                        if "fetch" in node_id
                        else "candidate_retrieval"
                    ),
                },
            )
        candidates = self._candidates(results)
        recorder.emit(
            "retrieval",
            "completed",
            "Ranked retrieval candidates observed",
            detail=self._ranking_observation(candidates),
        )
        qul_topic_rerank = getattr(self.source, "last_qul_topic_rerank", None)
        if qul_topic_rerank is not None:
            phase = qul_topic_rerank.get("phase")
            recorder.emit(
                "rerank",
                phase if phase in {"completed", "degraded"} else "degraded",
                (
                    "QUL topic-label reranking completed"
                    if phase == "completed"
                    else "QUL topic-label reranking degraded; retained fused retrieval order"
                ),
                detail={key: value for key, value in qul_topic_rerank.items() if key != "phase"},
            )
        rerank_degradation = getattr(self.source, "last_rerank_degradation", None)
        if rerank_degradation is not None:
            recorder.emit(
                "rerank",
                "degraded",
                "Reranking degraded; retained fused retrieval order",
                detail=rerank_degradation,
            )
        elif getattr(self.source, "reranker", None) is not None:
            recorder.emit(
                "rerank",
                "completed",
                "Reranking completed",
                detail={"candidate_count": len(candidates)},
            )

        error_results = [
            result for result in results.values() if result.status is ToolResultStatus.ERROR
        ]
        if error_results:
            availability_codes = {
                ToolErrorCode.RATE_LIMITED,
                ToolErrorCode.SOURCE_UNAVAILABLE,
                ToolErrorCode.TIMEOUT,
            }
            temporarily_unavailable = any(
                result.error is not None and result.error.code in availability_codes
                for result in error_results
            )
            evidence = self._empty_evidence(
                identity.turn_id,
                (
                    EvidenceRequirementFailureReason.UNAVAILABLE
                    if temporarily_unavailable
                    else EvidenceRequirementFailureReason.EMPTY
                ),
                requirements=execution_plan.evidence_requirements,
            )
            response = terminal_response(
                **self._identity_args(identity),
                status=(
                    AnswerStatus.TEMPORARILY_UNAVAILABLE
                    if temporarily_unavailable
                    else AnswerStatus.FAILED_SAFE
                ),
                notice_code=NoticeCode.SOURCE_UNAVAILABLE,
                text=(
                    "A required retrieval provider is temporarily unavailable."
                    if temporarily_unavailable
                    else "Canonical Quran evidence could not be produced safely."
                ),
            )
            provider_failure = getattr(self.source, "last_provider_failure", None)
            tafsir_source = getattr(self.executor, "tafsir_source", None)
            tafsir_failure = getattr(tafsir_source, "last_provider_failure", None)
            if tafsir_failure is not None:
                provider_failure = tafsir_failure
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                plan=plan,
                execution_plan=execution_plan,
                results=results,
                candidates=candidates,
                evidence=evidence,
                planner_completions=planner_completions,
                terminal_reason=(
                    "Required retrieval provider temporarily unavailable"
                    if temporarily_unavailable
                    else "Retrieval failed safely"
                ),
                provider_failure=provider_failure,
                chat_route_state=route_state,
            )
            await self._persist_response(
                response, evidence, plan=plan, planner_metadata=planner_completions
            )
            return response, trace

        try:
            recorder.emit("evidence", "started", "Assembling authorized evidence")
            evidence = await self.evidence_assembler.assemble(
                turn_id=identity.turn_id,
                execution_plan=execution_plan,
                results=results,
            )
            recorder.emit(
                "evidence",
                "completed",
                "Evidence assembly completed",
                detail={"evidence_count": len(evidence.items_by_id)},
            )
        except Exception:
            evidence = self._empty_evidence(
                identity.turn_id,
                EvidenceRequirementFailureReason.RIGHTS_BLOCKED,
                requirements=execution_plan.evidence_requirements,
            )
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.FAILED_SAFE,
                notice_code=NoticeCode.SOURCE_UNAVAILABLE,
                text="Canonical evidence could not be authorized safely.",
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                plan=plan,
                execution_plan=execution_plan,
                results=results,
                candidates=candidates,
                evidence=evidence,
                planner_completions=planner_completions,
                chat_route_state=route_state,
            )
            await self._persist_response(
                response, evidence, plan=plan, planner_metadata=planner_completions
            )
            return response, trace

        if pending_clarification is not None:
            term_count = self._bare_topic_term_count(request.message)
            ranking_observation = self._ranking_observation(candidates)
            recorder.emit(
                "verification",
                "completed",
                "Ambiguity hypothesis evaluated against retrieval",
                detail={
                    "hydrated_evidence_count": len(evidence.items_by_id),
                    "visible_term_count": term_count,
                    **ranking_observation,
                },
            )
            # The supplied run has no persisted component-score distributions, so a
            # numeric confidence floor would be fabricated.  Until the runner records
            # those scores, retain the only observed control boundary: a one-word topic
            # has no stated sense for retrieval to select on the user's behalf.
            if term_count == 1:
                scope = ScopeDecisionV1(
                    schema_version="1",
                    status=ScopeStatus.CLARIFICATION_REQUIRED,
                    reason=ScopeReason.NEEDS_CLARIFICATION,
                    signals=scope.signals,
                    clarification=pending_clarification,
                    user_message_key="scope.needs_semantic_clarification",
                )
                response = clarification_response(
                    **self._identity_args(identity), clarification=pending_clarification
                )
                trace = self._trace(
                    identity,
                    context,
                    resolution,
                    scope,
                    plan=plan,
                    execution_plan=execution_plan,
                    results=results,
                    candidates=candidates,
                    evidence=evidence,
                    planner_completions=planner_completions,
                    terminal_reason=(
                        "Retrieval observed "
                        f"{len(candidates)} ranked candidate(s) and "
                        f"{len(evidence.items_by_id)} hydrated evidence item(s), but the "
                        "request supplied one visible topic term and no selected sense."
                    ),
                    chat_route_state=route_state,
                )
                await self._persist_response(
                    response, evidence, plan=plan, planner_metadata=planner_completions
                )
                return response, trace

        missing_semantic_targets: tuple[str, ...] = ()
        targets = ()
        if plan.intent is Intent.QURAN_TOPIC_SEARCH:
            search_query = next(
                (
                    operation.query
                    for operation in plan.requested_operations
                    if operation.operation == "SEARCH_QURAN_TOPICS"
                ),
                plan.semantic_target_summary.normalized_label or request.message,
            )
            targets = semantic_subtargets(
                plan.semantic_target_summary.entity_labels,
                fallback_query=search_query,
            )
            semantic_judge = (
                getattr(self.planner, "assess_topic_relevance", None)
                if self.model_relevance_review_enabled
                else None
            )
            received_evidence_count = len(evidence.items_by_id)
            review_passage_count = len(
                {
                    str(item.canonical_pointer.model_dump(mode="json"))
                    for item in evidence.items_by_id.values()
                }
            )
            review_model_call_count = 0
            review_outcome = "DETERMINISTIC_RELEVANCE"
            review_drop_reason_counts: dict[str, int] = {}
            review_failure: dict[str, object] | None = None

            def review_failure_diagnostic(exc: PlannerFailedSafe) -> dict[str, object]:
                cause = exc.cause or exc
                provider_id = (
                    "local"
                    if route_state.local_selected
                    else "groq"
                )
                return {
                    "error_type": type(cause).__name__,
                    "provider_id": provider_id,
                    "safe_detail": str(cause)[:240],
                }

            try:
                recorder.emit(
                    "verification",
                    "started",
                    "Assessing retrieved topic relevance",
                    detail={
                        "target_count": len(targets),
                        "received_evidence_count": received_evidence_count,
                        "review_passage_count": review_passage_count,
                        "method": "provider" if semantic_judge is not None else "deterministic",
                    },
                )
                if semantic_judge is not None:
                    relevance, relevance_completions = await semantic_judge(
                        targets=targets,
                        evidence=evidence,
                        response_language=plan.response_language,
                        route_state=route_state,
                    )
                    planner_completions.extend(relevance_completions)
                    review_model_call_count = len(relevance_completions)
                    review_drop_reason_counts = dict(relevance.review_drop_reason_counts)
                    review_outcome = "MODEL_REVIEW_FILTERED"
                else:
                    relevance = assess_topic_evidence(
                        search_query,
                        evidence,
                        targets=targets,
                        retained_passage_limit=self.relevance_retained_passage_limit,
                        ranked_retention_enabled=self.relevance_ranked_retention_enabled,
                        relevance_distinctive_terms_enabled=(
                            self.relevance_distinctive_terms_enabled
                        ),
                    )
            except PlannerTemporarilyUnavailable as exc:
                planner_completions.extend(exc.completions)
                recorder.emit(
                    "verification",
                    "failed",
                    "Topic relevance assessment unavailable",
                    detail={
                        "received_evidence_count": received_evidence_count,
                        "kept_evidence_count": 0,
                        "review_model_call_count": len(exc.completions),
                        "outcome": "REVIEW_PROVIDER_UNAVAILABLE",
                    },
                )
                response = terminal_response(
                    **self._identity_args(identity),
                    status=AnswerStatus.TEMPORARILY_UNAVAILABLE,
                    notice_code=NoticeCode.SOURCE_UNAVAILABLE,
                    text="Semantic evidence validation is temporarily unavailable.",
                )
                trace = self._trace(
                    identity,
                    context,
                    resolution,
                    scope,
                    plan=plan,
                    execution_plan=execution_plan,
                    results=results,
                    candidates=candidates,
                    evidence=evidence,
                    planner_completions=planner_completions,
                    terminal_reason=str(exc),
                    provider_failure=exc.diagnostic,
                    chat_route_state=route_state,
                )
                await self._persist_response(
                    response, evidence, plan=plan, planner_metadata=planner_completions
                )
                return response, trace
            except PlannerFailedSafe as exc:
                planner_completions.extend(exc.completions)
                review_model_call_count = len(exc.completions)
                relevance = empty_topic_relevance_assessment(evidence, targets=targets)
                review_outcome = "MODEL_REVIEW_FAILED"
                review_failure = review_failure_diagnostic(exc)

            relevance = enforce_ranked_relevance_boundary(
                retrieved_evidence=evidence,
                assessment=relevance,
                targets=targets,
                empty_match_retains_anchors=(
                    semantic_judge is None and self.relevance_empty_match_retains_anchors
                ),
            )
            if (
                relevance.boundary_retained_top_ranked
                or relevance.boundary_retained_hydrated_anchors
            ):
                if review_outcome == "MODEL_REVIEW_FAILED":
                    review_outcome = "MODEL_REVIEW_FAILED_RETAINED_TOP_RANKED"
                    summary = "Semantic reviewer failed; retained top-ranked evidence"
                elif semantic_judge is not None:
                    review_outcome = "MODEL_REVIEW_REJECTED_ALL_RETAINED_TOP_RANKED"
                    summary = (
                        "Semantic reviewer rejected all evidence; retained top-ranked evidence"
                    )
                elif relevance.boundary_retained_hydrated_anchors:
                    review_outcome = "DETERMINISTIC_RELEVANCE_EMPTY_RETAINED_HYDRATED_ANCHORS"
                    summary = "Deterministic relevance retained hydrated anchor windows"
                else:
                    review_outcome = "DETERMINISTIC_RELEVANCE_EMPTY_RETAINED_TOP_RANKED"
                    summary = "Deterministic relevance retained top-ranked evidence"
                recorder.emit(
                    "verification",
                    "degraded",
                    summary,
                    detail={
                        "received_evidence_count": received_evidence_count,
                        "retained_evidence_count": len(relevance.evidence.items_by_id),
                        "outcome": review_outcome,
                        "review_drop_reason_counts": review_drop_reason_counts,
                        **({"failure": review_failure} if review_failure is not None else {}),
                    },
                )
            recorder.emit(
                "verification",
                "completed",
                "Topic relevance assessment completed",
                detail={
                    "received_evidence_count": received_evidence_count,
                    "kept_evidence_count": len(relevance.evidence.items_by_id),
                    "dropped_evidence_count": (
                        received_evidence_count - len(relevance.evidence.items_by_id)
                    ),
                    "missing_target_count": len(relevance.missing_target_ids),
                    "review_model_call_count": review_model_call_count,
                    "review_drop_reason_counts": review_drop_reason_counts,
                    "outcome": review_outcome,
                    "retained_references": self._retained_references(relevance.evidence),
                },
            )
            evidence = relevance.evidence
            missing_semantic_targets = relevance.missing_target_ids
            supported_facet_summaries = relevance.supported_facet_summaries
        else:
            supported_facet_summaries = ()

        if not evidence.requirements_satisfied:
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
                text="The retrieved canonical evidence is insufficient for a grounded answer.",
            )
            trace = self._trace(
                identity,
                context,
                resolution,
                scope,
                plan=plan,
                execution_plan=execution_plan,
                results=results,
                candidates=candidates,
                evidence=evidence,
                planner_completions=planner_completions,
                chat_route_state=route_state,
            )
            await self._persist_response(
                response, evidence, plan=plan, planner_metadata=planner_completions
            )
            return response, trace

        synthesis_completions: list[ProviderCompletion] = []
        synthesis_terminal_reason: str | None = None
        honest_fallback_rendered = False
        synthesis_provider_failure: ProviderFailureDiagnostic | None = None
        gate_rejections_by_rule: dict[str, int] = {}

        def validated_topic_fallback(
            *, call_site: str, originating_reason: str
        ) -> tuple[AnswerDraftV1, str | None]:
            summary_pairs = tuple(
                (summary.canonical_pointer, summary.text) for summary in supported_facet_summaries
            )
            first_draft_validated = False
            narrowed_second_attempt_used = False
            fallback_draft = (
                deterministic_contextual_topic_draft(
                    evidence,
                    response_language=plan.response_language,
                    supported_facet_summaries=summary_pairs,
                )
                if summary_pairs
                else deterministic_topic_source_view_draft(
                    evidence,
                    response_language=plan.response_language,
                    honest_insufficient=(
                        self.honest_insufficient_fallback_enabled
                        and call_site == "SYNTHESIZER_INSUFFICIENT_EVIDENCE"
                    ),
                )
            )
            try:
                GroundingValidator().validate(
                    fallback_draft,
                    evidence,
                    response_language=plan.response_language,
                )
                first_draft_validated = True
                return fallback_draft, None
            except GroundingError as fallback_error:
                narrowed_second_attempt_used = True
                required_tafsir_ids = [
                    item.evidence_id
                    for item in evidence.items_by_id.values()
                    if item.source_kind.value == "TAFSIR"
                    and (
                        not evidence.requirements.required_tafsir_aliases
                        or item.source_policy_alias
                        in evidence.requirements.required_tafsir_aliases
                    )
                ]
                repair_guidance = GroqSynthesizer._repair_guidance(
                    fallback_error,
                    response_language=plan.response_language,
                    required_tafsir_ids=required_tafsir_ids,
                )
                recorder.emit(
                    "synthesis",
                    "rejected",
                    "Synthesis draft rejected by validation",
                    detail=rejected_synthesis_detail(
                        attempt="fallback_first_draft",
                        error=fallback_error,
                        repair_guidance=repair_guidance,
                        draft=fallback_draft,
                        is_tafsir_route=bool(required_tafsir_ids),
                    ),
                )
                narrowed = (
                    deterministic_contextual_topic_draft(
                        evidence,
                        response_language=plan.response_language,
                    )
                    if summary_pairs
                    else deterministic_topic_source_view_draft(
                        evidence,
                        response_language=plan.response_language,
                        honest_insufficient=(
                            self.honest_insufficient_fallback_enabled
                            and call_site == "SYNTHESIZER_INSUFFICIENT_EVIDENCE"
                        ),
                    )
                )
                GroundingValidator().validate(
                    narrowed,
                    evidence,
                    response_language=plan.response_language,
                )
                return narrowed, str(fallback_error)
            finally:
                recorder.emit(
                    "synthesis",
                    "degraded",
                    "Validated topic fallback invoked",
                    detail={
                        "call_site": call_site,
                        "originating_reason": originating_reason,
                        "first_draft_validated": first_draft_validated,
                        "narrowed_second_attempt_used": narrowed_second_attempt_used,
                        "supported_facet_summaries_empty": not supported_facet_summaries,
                        **(
                            {"honest_fallback": True}
                            if self.honest_insufficient_fallback_enabled
                            and call_site == "SYNTHESIZER_INSUFFICIENT_EVIDENCE"
                            and not supported_facet_summaries
                            else {}
                        ),
                    },
                )

        def grounding_failure_response(reason: str) -> ResearchResponseV1:
            recorder.emit(
                "synthesis",
                "failed",
                "Grounding validation prevented an answer",
                detail={"reason": reason},
            )
            return terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
                text="The available evidence could not support a safely grounded answer.",
            )

        try:
            recorder.emit("synthesis", "started", "Synthesizing grounded response")
            if deterministic_plan and plan.intent is Intent.QURAN_PASSAGE_LOOKUP:
                draft = deterministic_exact_lookup_draft(evidence)
            else:
                draft, synthesis_completions = await self.synthesizer.synthesize(
                    question=normalize_user_query(
                        plan.semantic_target_summary.normalized_label or request.message
                    ),
                    evidence=evidence,
                    response_language=plan.response_language,
                    route_state=route_state,
                )
                if (
                    draft.status is AnswerStatus.INSUFFICIENT_EVIDENCE
                    and plan.intent is Intent.QURAN_TOPIC_SEARCH
                ):
                    synthesis_terminal_reason = (
                        "Synthesizer returned INSUFFICIENT_EVIDENCE; rendered an "
                        "application-grounded topic fallback"
                    )
                    draft, fallback_error = validated_topic_fallback(
                        call_site="SYNTHESIZER_INSUFFICIENT_EVIDENCE",
                        originating_reason=AnswerStatus.INSUFFICIENT_EVIDENCE.value,
                    )
                    if fallback_error:
                        synthesis_terminal_reason += f"; summary narrowed: {fallback_error}"
                    if (
                        self.honest_insufficient_fallback_enabled
                        and not supported_facet_summaries
                    ):
                        honest_fallback_rendered = True
                        synthesis_terminal_reason = (
                            "Synthesizer returned INSUFFICIENT_EVIDENCE; retained sources "
                            "were rendered as possibly-related results, not as an answer"
                        )
            response = self.renderer.render(
                **self._identity_args(identity), draft=draft, evidence=evidence
            )
            if (
                honest_fallback_rendered
            ):
                notice_text = (
                    "لم أجد آيات كافية تجيب عن هذا السؤال إجابة مباشرة."
                    if plan.response_language.strip().casefold().startswith("ar")
                    else "I couldn't find passages that directly answer this question."
                )
                response = response.model_copy(
                    update={
                        "status": AnswerStatus.INSUFFICIENT_EVIDENCE,
                        "blocks": [
                            RenderedNoticeBlockV1(
                                block_type="NOTICE",
                                notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
                                text=notice_text,
                            ),
                            *response.blocks,
                        ],
                    }
                )
            recorder.emit(
                "synthesis",
                "completed",
                "Grounded response rendered",
                detail={"status": response.status.value},
            )
        except CanonicalTextGroundingError as exc:
            synthesis_terminal_reason = (
                f"Model synthesis required deterministic grounded rendering: {exc}"
            )
            try:
                draft, fallback_error = validated_topic_fallback(
                    call_site="CANONICAL_TEXT_GROUNDING_ERROR",
                    originating_reason=type(exc).__name__,
                )
                response = self.renderer.render(
                    **self._identity_args(identity), draft=draft, evidence=evidence
                )
                if fallback_error:
                    synthesis_terminal_reason += f"; summary narrowed: {fallback_error}"
            except GroundingError as fallback_error:
                synthesis_terminal_reason += f"; fallback grounding failed: {fallback_error}"
                response = grounding_failure_response(str(fallback_error))
        except SynthesisTemporarilyUnavailable as exc:
            synthesis_completions = exc.completions
            synthesis_provider_failure = exc.diagnostic
            synthesis_terminal_reason = str(exc)
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.TEMPORARILY_UNAVAILABLE,
                notice_code=NoticeCode.SOURCE_UNAVAILABLE,
                text="The required synthesis provider is temporarily unavailable.",
            )
        except GroundingError as exc:
            synthesis_terminal_reason = f"Synthesis failed safely: {exc}"
            recorder.emit(
                "synthesis",
                "failed",
                "Synthesis grounding validation failed",
                detail={"reason": str(exc)},
            )
            if route_state.local_selected and plan.intent is Intent.QURAN_TOPIC_SEARCH:
                try:
                    draft, fallback_error = validated_topic_fallback(
                        call_site="GROUNDING_ERROR",
                        originating_reason=type(exc).__name__,
                    )
                    response = self.renderer.render(
                        **self._identity_args(identity), draft=draft, evidence=evidence
                    )
                    synthesis_terminal_reason = (
                        "Local synthesis failed bounded validation; rendered a fresh "
                        f"application-grounded topic draft: {exc}"
                    )
                    if fallback_error:
                        synthesis_terminal_reason += f"; summary narrowed: {fallback_error}"
                except GroundingError as fallback_error:
                    synthesis_terminal_reason += f"; fallback grounding failed: {fallback_error}"
                    response = grounding_failure_response(str(fallback_error))
            elif plan.intent in {Intent.EXPLAIN_QURAN_VERSE, Intent.COMPARE_TAFSIR} and any(
                item.source_kind.value == "TAFSIR" for item in evidence.items_by_id.values()
            ):
                # The generated interpretation failed validation, not the exact
                # source.  Keep the source visible with an application-authored,
                # non-interpretive presentation notice rather than fabricate a
                # replacement summary or hide a long verified Tafsir passage.
                try:
                    draft = deterministic_tafsir_source_draft(
                        evidence, response_language=plan.response_language
                    )
                    GroundingValidator().validate(
                        draft, evidence, response_language=plan.response_language
                    )
                    response = self.renderer.render(
                        **self._identity_args(identity), draft=draft, evidence=evidence
                    )
                    synthesis_terminal_reason += (
                        "; rendered verified Tafsir without generated summary"
                    )
                except GroundingError as fallback_error:
                    synthesis_terminal_reason += (
                        f"; Tafsir fallback grounding failed: {fallback_error}"
                    )
                    response = grounding_failure_response(str(fallback_error))
            else:
                response = terminal_response(
                    **self._identity_args(identity),
                    status=AnswerStatus.FAILED_SAFE,
                    notice_code=NoticeCode.ANSWER_NOT_VERIFIED,
                    text="A fully grounded response could not be produced safely.",
                )
        except (ValueError, RuntimeError) as exc:
            synthesis_terminal_reason = f"Synthesis failed safely: {exc}"
            response = terminal_response(
                **self._identity_args(identity),
                status=AnswerStatus.FAILED_SAFE,
                notice_code=NoticeCode.ANSWER_NOT_VERIFIED,
                text="A fully grounded response could not be produced safely.",
            )

        if (
            missing_semantic_targets
            and response.status is AnswerStatus.ANSWERABLE
            and response.output_kind is not RenderedOutputKind.SOURCE_ONLY
        ):
            missing_labels = [
                target.search_query
                for target in targets
                if target.target_id in missing_semantic_targets
            ]
            if request.locale.casefold().startswith("ar"):
                partial_text = (
                    "لم يثبت الدليل القرآني المسترجع حاليًا الجزء المتعلق بـ: "
                    f"{'، '.join(missing_labels)}؛ لذلك تقتصر الإجابة على الجزء "
                    "الذي سنده الدليل المعروض."
                    if missing_labels
                    else "لم أجد في الاسترجاع الحالي دليلاً قرآنيًا مباشرًا كافيًا لبعض أجزاء "
                    "السؤال؛ لذلك تقتصر الإجابة على الجزء الذي سنده الدليل المعروض."
                )
            else:
                partial_text = (
                    "Current Quran evidence did not establish the part of the question about: "
                    f"{', '.join(missing_labels)}. The answer is limited to the evidenced part."
                    if missing_labels
                    else "Current retrieval did not find sufficient direct Quran evidence for "
                    "every part of the question, so the answer is limited to the evidenced part."
                )
            tafsir_requested = any(
                node.tool_id == "get_tafsir_passages.v1" for node in execution_plan.nodes
            )
            tafsir_returned = any(
                item.source_kind.value == "TAFSIR" for item in evidence.items_by_id.values()
            )
            if tafsir_requested and not tafsir_returned:
                partial_text += (
                    " كما لم يُرجع مصدر التفسير المعتمد نصًا لهذا الموضع، لذلك لم تُنسب "
                    "إليه دلالة تفسيرية إضافية."
                    if request.locale.casefold().startswith("ar")
                    else " The approved Tafsir source returned no passage for this location, "
                    "so no additional interpretive meaning is attributed to it."
                )
            response = response.model_copy(
                update={
                    "status": AnswerStatus.PARTIAL,
                    "blocks": [
                        *response.blocks,
                        RenderedNoticeBlockV1(
                            block_type="NOTICE",
                            notice_code=NoticeCode.PARTIAL_COVERAGE,
                            text=partial_text,
                        ),
                    ],
                }
            )

        if (
            self.answer_composer is not None
            and self.answer_composer_enabled
            and response.status in (AnswerStatus.ANSWERABLE, AnswerStatus.PARTIAL)
        ):
            claim_blocks = [
                block for block in response.blocks if isinstance(block, RenderedClaimBlockV1)
            ]
            if claim_blocks:
                recorder.emit("composition", "started", "Composing reader-facing narrative")
                try:
                    (
                        narrative,
                        composer_completions,
                        gate_rejections_by_rule,
                    ) = await self.answer_composer.compose(
                        question=normalize_user_query(
                            plan.semantic_target_summary.normalized_label or request.message
                        ),
                        response_language=plan.response_language,
                        coverage_status=response.status,
                        claims=claim_blocks,
                        evidence=evidence,
                        route_state=route_state,
                    )
                    synthesis_completions = [*synthesis_completions, *composer_completions]
                    for composer_completion in composer_completions:
                        await self.evidence_assembler.record_chat_completion_activation(
                            evidence, composer_completion
                        )
                    if narrative is not None:
                        # NARRATIVE renders first (WP-4, UI): it is the answer, claims and
                        # sources are the audit trail behind it.
                        response = response.model_copy(
                            update={"blocks": [narrative, *response.blocks]}
                        )
                    recorder.emit(
                        "composition",
                        "completed",
                        "Composition and anti-fabrication gate completed",
                        detail={
                            "narrative_created": narrative is not None,
                            "gate_rejections_by_rule": gate_rejections_by_rule,
                        },
                    )
                except CompositionProviderFailure as exc:
                    synthesis_terminal_reason = (
                        f"Composition provider failed; rendered grounded claims without narrative: "
                        f"{exc.safe_detail}"
                    )
                    synthesis_provider_failure = exc.diagnostic
                    recorder.emit(
                        "composition",
                        "failed",
                        "Composition provider failed; retained grounded claims",
                        detail={"reason": exc.safe_detail},
                    )

        response = response.model_copy(
            update={"output_kind": derive_rendered_output_kind(response.blocks)}
        )

        trace = self._trace(
            identity,
            context,
            resolution,
            scope,
            plan=plan,
            execution_plan=execution_plan,
            results=results,
            candidates=candidates,
            evidence=evidence,
            planner_completions=planner_completions,
            synthesis_completions=synthesis_completions,
            terminal_reason=synthesis_terminal_reason,
            provider_failure=synthesis_provider_failure,
            chat_route_state=route_state,
        )
        await self._persist_response(
            response,
            evidence,
            gate_rejections_by_rule=gate_rejections_by_rule,
            plan=plan,
            planner_metadata=planner_completions,
        )
        return response, trace

    @staticmethod
    def _working_context(
        request: ResearchRequest,
        turn_id: UUID,
        recent_user_turns: list[ConversationTurnV1],
    ) -> WorkingContextV1:
        previous = recent_user_turns[-5:]
        return WorkingContextV1(
            schema_version="1",
            conversation_id=request.conversation_id,
            state_version=len(recent_user_turns),
            recent_turns=[
                *[
                    SanitizedTurnV1(
                        turn_id=item.turn_id,
                        role=TurnRole.USER,
                        sanitized_text=item.user_message,
                    )
                    for item in previous
                ],
                SanitizedTurnV1(
                    turn_id=turn_id,
                    role=TurnRole.USER,
                    sanitized_text=request.message,
                ),
            ],
            values=[
                ContextValueV1(
                    field=ContextField.RESPONSE_LANGUAGE,
                    value=request.locale,
                    provenance=ContextProvenance.CURRENT_USER,
                    source_turn_id=turn_id,
                    turn_age=0,
                )
            ],
            pending_clarification=(
                previous[-1].response.clarification
                if previous and previous[-1].response.status is AnswerStatus.CLARIFICATION_REQUIRED
                else None
            ),
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
        )

    @staticmethod
    def _identity_args(identity: RequestExecutionIdentity) -> dict[str, UUID]:
        return {
            "conversation_id": identity.conversation_id,
            "turn_id": identity.turn_id,
            "correlation_id": identity.correlation_id,
        }

    @staticmethod
    def _candidates(
        results: dict[str, ToolResult[Any]],
    ) -> list[QuranSearchCandidateV1]:
        return [
            item
            for node_id, result in results.items()
            if node_id.startswith("node_search_quran") and result.data is not None
            for item in result.data
            if isinstance(item, QuranSearchCandidateV1)
        ]

    @staticmethod
    def _empty_evidence(
        turn_id: UUID,
        reason: EvidenceRequirementFailureReason = EvidenceRequirementFailureReason.EMPTY,
        *,
        requirements: EvidenceRequirementsV1 | None = None,
    ) -> EvidenceSet:
        effective_requirements = requirements or quran_evidence_requirements()
        missing = [
            EvidenceRequirementFailureV1(
                source_kind=kind,
                resource_alias=None,
                reason=reason,
            )
            for kind in sorted(effective_requirements.required_source_kinds, key=str)
        ]
        missing.extend(
            EvidenceRequirementFailureV1(
                source_kind=None,
                resource_alias=alias,
                reason=reason,
            )
            for alias in effective_requirements.required_tafsir_aliases
        )
        return EvidenceSet(
            schema_version="1",
            turn_id=turn_id,
            items_by_id={},
            requirements=effective_requirements,
            requirements_satisfied=False,
            missing_requirements=missing,
            permitted_synthesis_deployments=set(),
        )

    def _trace(
        self,
        identity: RequestExecutionIdentity,
        context: WorkingContextV1,
        resolution: DeterministicResolutionContextV1,
        scope: ScopeDecisionV1,
        *,
        plan: ResearchPlanV1 | None = None,
        execution_plan: ExecutionPlanV1 | None = None,
        results: dict[str, ToolResult[Any]] | None = None,
        candidates: list[QuranSearchCandidateV1] | None = None,
        evidence: EvidenceSet,
        planner_completions: list[ProviderCompletion] | None = None,
        synthesis_completions: list[ProviderCompletion] | None = None,
        terminal_reason: str | None = None,
        provider_failure: ProviderFailureDiagnostic | None = None,
        chat_route_state: TurnChatRouteState | None = None,
    ) -> ResearchTrace:
        planner_metadata = planner_completions or []
        synthesis_metadata = synthesis_completions or []
        recorder = _TRACE_RECORDER.get()
        if recorder is not None:
            recorder.emit(
                "gate",
                "completed" if evidence.requirements_satisfied else "failed",
                "Evidence and safety gate evaluated",
                detail={
                    "requirements_satisfied": evidence.requirements_satisfied,
                    "missing_requirement_count": len(evidence.missing_requirements),
                },
            )
            recorder.emit(
                "verification",
                "completed",
                "Terminal decision recorded",
                detail={"terminal_reason": terminal_reason, "scope_status": scope.status.value},
            )

        def completion_record(value: ProviderCompletion) -> dict[str, Any]:
            inference_location = (
                "LOCAL" if value.effective_deployment.effective_provider_id == "local" else "CLOUD"
            )
            return {
                "inference_location": inference_location,
                "completion": value.model_dump(mode="json"),
            }

        return ResearchTrace(
            identity=identity,
            context=context,
            resolution=resolution,
            scope=scope,
            plan=plan,
            execution_plan=execution_plan,
            results=results or {},
            candidates=candidates or [],
            evidence=evidence,
            planner_completions=planner_metadata,
            synthesis_completions=synthesis_metadata,
            query_embedding_completion=self.source.last_embedding_completion,
            embedding_degradation=getattr(self.source, "last_embedding_degradation", None),
            rerank_degradation=getattr(self.source, "last_rerank_degradation", None),
            terminal_reason=terminal_reason,
            provider_failure=provider_failure,
            chat_route=(chat_route_state or TurnChatRouteState()).snapshot(),
            chat_completion_records_v22={
                "planner": [completion_record(value) for value in planner_metadata],
                "synthesis": [completion_record(value) for value in synthesis_metadata],
            },
            events=recorder.events if recorder is not None and recorder.enabled else [],
        )

    async def _persist_response(
        self,
        response: ResearchResponseV1,
        evidence: EvidenceSet,
        *,
        gate_rejections_by_rule: dict[str, int] | None = None,
        plan: ResearchPlanV1 | None = None,
        planner_metadata: list[ProviderCompletion] | None = None,
    ) -> None:
        provenance = [
            PersistedEvidenceRecordV1(
                schema_version="1",
                turn_id=response.turn_id,
                evidence_id=item.evidence_id,
                canonical_pointer=item.canonical_pointer,
                source_identity=item.source_identity,
                source_revision=item.source_revision,
                language=item.language,
                reference=item.reference,
                attributions=item.attributions,
                retrieval=item.retrieval,
                content_hash=item.content_hash,
                hadith_references=item.hadith_references,
                permitted_snapshot=(
                    "groq:openai/gpt-oss-20b:zdr"
                    if "groq:openai/gpt-oss-20b:zdr" in evidence.permitted_synthesis_deployments
                    else None
                ),
            ).model_dump(mode="json")
            for item in evidence.items_by_id.values()
        ]
        try:
            if plan is not None:
                await self.repository.persist_plan_and_response(
                    turn_id=response.turn_id,
                    conversation_id=response.conversation_id,
                    correlation_id=response.correlation_id,
                    plan=plan,
                    metadata=planner_metadata or [],
                    response=response,
                    provenance=provenance,
                    gate_rejections_by_rule=gate_rejections_by_rule or {},
                )
            else:
                await self.repository.persist_response(
                    turn_id=response.turn_id,
                    response=response,
                    provenance=provenance,
                    gate_rejections_by_rule=gate_rejections_by_rule or {},
                )
        except Exception as exc:
            # A persistence outage must not convert an already safe terminal response into
            # the API's generic 503.  The repository transaction prevents an orphan plan.
            log_event("research_turn_persistence_degraded", error_type=type(exc).__name__)
