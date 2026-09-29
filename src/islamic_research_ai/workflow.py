from __future__ import annotations

from islamic_research_ai.contracts import (
    DeterministicResolutionContextV1,
    EvidenceRequirementsV1,
    EvidenceSourceKind,
    ExecutionNodeV1,
    ExecutionPlanV1,
    FetchQuranHandlerInputV1,
    FetchQuranPassageIntentV1,
    FetchTafsirHandlerInputV1,
    FetchTafsirPassagesIntentV1,
    Intent,
    LateBindingSelect,
    PartialFailurePolicy,
    ResearchPlanV1,
    ScopeDecisionV1,
    ScopeStatus,
    SearchQuranHandlerInputV1,
    SearchQuranTopicsIntentV1,
    TafsirFetchMode,
    ValidatedLateBindingV1,
)


class WorkflowBuildError(ValueError):
    pass


def quran_evidence_requirements() -> EvidenceRequirementsV1:
    return EvidenceRequirementsV1(
        required_source_kinds={
            EvidenceSourceKind.QURAN,
            EvidenceSourceKind.QURAN_TRANSLATION,
        },
        minimum_items_by_kind={
            EvidenceSourceKind.QURAN: 1,
            EvidenceSourceKind.QURAN_TRANSLATION: 1,
        },
        require_exact_quran=True,
        require_exact_hadith=False,
        required_tafsir_aliases=[],
        require_two_distinct_tafsir_resources=False,
        allow_partial=False,
    )


def tafsir_evidence_requirements(aliases: list[str], *, compare: bool) -> EvidenceRequirementsV1:
    expected_count = 2 if compare else 1
    if len(aliases) != expected_count or len(set(aliases)) != expected_count:
        raise WorkflowBuildError("Tafsir evidence aliases do not match the requested mode")
    return EvidenceRequirementsV1(
        required_source_kinds={
            EvidenceSourceKind.QURAN,
            EvidenceSourceKind.QURAN_TRANSLATION,
            EvidenceSourceKind.TAFSIR,
        },
        minimum_items_by_kind={
            EvidenceSourceKind.QURAN: 1,
            EvidenceSourceKind.QURAN_TRANSLATION: 1,
            EvidenceSourceKind.TAFSIR: expected_count,
        },
        require_exact_quran=True,
        require_exact_hadith=False,
        required_tafsir_aliases=list(aliases),
        require_two_distinct_tafsir_resources=compare,
        allow_partial=False,
    )


def topic_tafsir_evidence_requirements(alias: str) -> EvidenceRequirementsV1:
    """Treat Tafsir as a requested but source-partial topic enhancement.

    The approved provider does not contain every verse in every configured work.
    Missing commentary must not erase otherwise complete Quran evidence; when the
    exact Tafsir exists it is still assembled and forces local-only synthesis.
    """

    del alias
    requirements = quran_evidence_requirements()
    return requirements.model_copy(update={"allow_partial": True})


class WorkflowCompiler:
    def __init__(self, *, topic_anchor_limit: int = 3) -> None:
        if not 1 <= topic_anchor_limit <= 3:
            raise ValueError("topic_anchor_limit must be between 1 and 3")
        self.topic_anchor_limit = topic_anchor_limit

    def compile(
        self,
        plan: ResearchPlanV1,
        *,
        deterministic_resolution: DeterministicResolutionContextV1,
        scope: ScopeDecisionV1,
    ) -> ExecutionPlanV1:
        if scope.status is not ScopeStatus.ALLOWED or plan.clarification.required:
            return ExecutionPlanV1(
                schema_version="1",
                intent=plan.intent,
                nodes=[],
                evidence_requirements=quran_evidence_requirements(),
                overall_deadline_ms=30_000,
                partial_failure_policy=PartialFailurePolicy.REQUIRE_ALL,
            )
        if plan.intent is Intent.QURAN_TOPIC_SEARCH:
            search, fetch = plan.requested_operations[:2]
            if not isinstance(search, SearchQuranTopicsIntentV1) or not isinstance(
                fetch, FetchQuranPassageIntentV1
            ):
                raise WorkflowBuildError("Validated topic plan has incompatible operations")
            # Start with the planner's complete semantic relation, then add at
            # most two independently answerable facets.  This deliberately does
            # not split modifiers/synonyms: the plan validator marks only true
            # coverage obligations with ``part:``.  All candidates are fused by
            # the executor before one shared K=3 anchor selection, so extra
            # searches cannot turn into extra, unrelated answer passages.
            derived_queries = [
                label.removeprefix("part:").strip()
                for label in plan.semantic_target_summary.entity_labels
                if label.startswith("part:")
                and label.removeprefix("part:").strip()
                and label.removeprefix("part:").strip() != search.query
            ]
            queries = list(dict.fromkeys([search.query, *derived_queries]))[:3]
            nodes: list[ExecutionNodeV1] = []
            topic_tafsir = (
                plan.requested_operations[2] if len(plan.requested_operations) == 3 else None
            )
            if topic_tafsir is not None and not isinstance(
                topic_tafsir, FetchTafsirPassagesIntentV1
            ):
                raise WorkflowBuildError("Validated topic Tafsir operation is incompatible")
            search_ids: list[str] = []
            for index, query in enumerate(queries, start=1):
                suffix = f"_{index}" if len(queries) > 1 else ""
                search_id = f"node_search_quran{suffix}"
                search_ids.append(search_id)
                nodes.append(
                    ExecutionNodeV1(
                        node_id=search_id,
                        tool_id="search_quran_topics.v1",
                        validated_input=SearchQuranHandlerInputV1(
                            query=query,
                            language=search.language,
                            surah_filters=search.surah_filters,
                        ),
                        depends_on=[],
                        # The source reranks the top ~20 fused candidates before
                        # returning this bounded pool.  Evidence remains capped
                        # below at K anchors and 12 hydrated source items.
                        result_limit=20,
                        timeout_ms=10_000,
                    )
                )
            late_binding = ValidatedLateBindingV1(
                upstream_node_id=search_ids[0],
                additional_upstream_node_ids=search_ids[1:],
                select=LateBindingSelect.QURAN_CANDIDATE_IDS,
                maximum_items=self.topic_anchor_limit,
            )
            nodes.append(
                ExecutionNodeV1(
                    node_id="node_fetch_quran",
                    tool_id="get_quran_passage.v1",
                    validated_input=FetchQuranHandlerInputV1(
                        verse_keys=[],
                        late_binding=late_binding,
                        translation_resource_alias=fetch.translation_resource_alias,
                        include_adjacent_verses=fetch.include_adjacent_verses,
                    ),
                    depends_on=search_ids,
                    timeout_ms=8_000,
                    result_limit=12,
                )
            )
            if topic_tafsir is not None:
                nodes.append(
                    ExecutionNodeV1(
                        node_id="node_fetch_tafsir",
                        tool_id="get_tafsir_passages.v1",
                        validated_input=FetchTafsirHandlerInputV1(
                            verse_keys=[],
                            late_binding=late_binding,
                            mode=topic_tafsir.mode,
                            tafsir_resource_aliases=topic_tafsir.tafsir_resource_aliases,
                            language=topic_tafsir.language,
                            character_budget=40_000,
                        ),
                        depends_on=search_ids,
                        timeout_ms=15_000,
                        result_limit=12,
                    )
                )
        elif plan.intent is Intent.QURAN_PASSAGE_LOOKUP:
            fetch = plan.requested_operations[0]
            if not isinstance(fetch, FetchQuranPassageIntentV1):
                raise WorkflowBuildError("Validated lookup plan has no fetch operation")
            verse_keys = [
                key
                for reference in deterministic_resolution.resolved_quran_references
                for key in reference.verse_keys
            ]
            nodes = [
                ExecutionNodeV1(
                    node_id="node_fetch_quran",
                    tool_id="get_quran_passage.v1",
                    validated_input=FetchQuranHandlerInputV1(
                        verse_keys=verse_keys,
                        late_binding=None,
                        translation_resource_alias=fetch.translation_resource_alias,
                        include_adjacent_verses=fetch.include_adjacent_verses,
                    ),
                    depends_on=[],
                    timeout_ms=8_000,
                    result_limit=10,
                )
            ]
        elif plan.intent in {Intent.EXPLAIN_QURAN_VERSE, Intent.COMPARE_TAFSIR}:
            quran_fetch, tafsir_fetch = plan.requested_operations
            if not isinstance(quran_fetch, FetchQuranPassageIntentV1) or not isinstance(
                tafsir_fetch, FetchTafsirPassagesIntentV1
            ):
                raise WorkflowBuildError("Validated Tafsir plan has incompatible operations")
            verse_keys = [
                key
                for reference in deterministic_resolution.resolved_quran_references
                for key in reference.verse_keys
            ]
            if not verse_keys:
                raise WorkflowBuildError("Tafsir execution requires exact Quran references")
            compare = plan.intent is Intent.COMPARE_TAFSIR
            expected_mode = (
                TafsirFetchMode.COMPARISON_EXACTLY_TWO
                if compare
                else TafsirFetchMode.EXPLANATION_ONE
            )
            if tafsir_fetch.mode is not expected_mode:
                raise WorkflowBuildError("Tafsir fetch mode conflicts with the validated intent")
            requirements = tafsir_evidence_requirements(
                tafsir_fetch.tafsir_resource_aliases,
                compare=compare,
            )
            nodes = [
                ExecutionNodeV1(
                    node_id="node_fetch_quran",
                    tool_id="get_quran_passage.v1",
                    validated_input=FetchQuranHandlerInputV1(
                        verse_keys=verse_keys,
                        late_binding=None,
                        translation_resource_alias=quran_fetch.translation_resource_alias,
                        include_adjacent_verses=quran_fetch.include_adjacent_verses,
                    ),
                    depends_on=[],
                    timeout_ms=8_000,
                    result_limit=10,
                ),
                ExecutionNodeV1(
                    node_id="node_fetch_tafsir",
                    tool_id="get_tafsir_passages.v1",
                    validated_input=FetchTafsirHandlerInputV1(
                        verse_keys=verse_keys,
                        mode=tafsir_fetch.mode,
                        tafsir_resource_aliases=tafsir_fetch.tafsir_resource_aliases,
                        language=tafsir_fetch.language,
                        character_budget=40_000,
                    ),
                    depends_on=[],
                    timeout_ms=15_000,
                    result_limit=20,
                ),
            ]
        else:
            raise WorkflowBuildError("Intent is not supported by the Quran workflow compiler")

        return ExecutionPlanV1(
            schema_version="1",
            intent=plan.intent,
            nodes=nodes,
            evidence_requirements=(
                requirements
                if plan.intent in {Intent.EXPLAIN_QURAN_VERSE, Intent.COMPARE_TAFSIR}
                else (
                    topic_tafsir_evidence_requirements(topic_tafsir.tafsir_resource_aliases[0])
                    if plan.intent is Intent.QURAN_TOPIC_SEARCH and topic_tafsir is not None
                    else quran_evidence_requirements()
                )
            ),
            overall_deadline_ms=30_000,
            partial_failure_policy=PartialFailurePolicy.REQUIRE_ALL,
        )
