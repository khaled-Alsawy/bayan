from __future__ import annotations

from typing import Any
from uuid import UUID

from islamic_research_ai.contracts import (
    EvidenceItem,
    EvidenceRequirementFailureReason,
    EvidenceRequirementFailureV1,
    EvidenceSet,
    EvidenceSourceKind,
    ExecutionPlanV1,
    ProviderCompletion,
    RetrievalMethod,
    ToolResult,
    ToolResultStatus,
)
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourceAuthorizationError, SourceOperation, SourcePolicy
from islamic_research_ai.routing import LOCAL_GEMMA_DEPLOYMENT_ID

GROQ_DEPLOYMENT = {
    "gateway_provider_id": "groq",
    "effective_provider_id": "groq",
    "model_id": "openai/gpt-oss-20b",
    "zero_data_retention": True,
    "provider_training": False,
    "aggregator_routing": False,
}

LOCAL_GEMMA_DEPLOYMENT = {
    "gateway_provider_id": "local",
    "effective_provider_id": "local",
    "model_id": "gemma-4-26b-a4b-it",
    "inference_location": "LOCAL",
    "zero_data_retention": True,
    "provider_training": False,
    "aggregator_routing": False,
}


class EvidenceAssemblyError(ValueError):
    pass


class EvidenceAssembler:
    def __init__(self, policy: SourcePolicy, repository: PostgresRepository) -> None:
        self.policy = policy
        self.repository = repository

    async def assemble(
        self,
        *,
        turn_id: UUID,
        execution_plan: ExecutionPlanV1,
        results: dict[str, ToolResult[Any]],
    ) -> EvidenceSet:
        expected_nodes = {node.node_id for node in execution_plan.nodes}
        if set(results) != expected_nodes:
            raise EvidenceAssemblyError("Evidence assembly requires complete node results")

        canonical_fetch_tools = {
            "get_quran_passage.v1",
            "get_tafsir_passages.v1",
        }
        hydrated: list[EvidenceItem] = []
        for node in execution_plan.nodes:
            result = results[node.node_id]
            if node.tool_id not in canonical_fetch_tools and result.evidence_items:
                raise EvidenceAssemblyError("Only canonical hydration may create evidence")
            if node.tool_id in canonical_fetch_tools:
                hydrated.extend(result.evidence_items)

        unique_hydrated: list[EvidenceItem] = []
        seen_items: set[tuple[str, str, str]] = set()
        for item in hydrated:
            identity = (
                item.source_kind.value,
                item.source_policy_alias,
                str(item.canonical_pointer.model_dump(mode="json")),
            )
            if identity not in seen_items:
                seen_items.add(identity)
                unique_hydrated.append(item)

        items_by_id: dict[str, EvidenceItem] = {}
        for index, item in enumerate(unique_hydrated, start=1):
            canonical_item = item.model_copy(update={"evidence_id": f"E{index}"})
            items_by_id[canonical_item.evidence_id] = canonical_item

        missing: list[EvidenceRequirementFailureV1] = []
        requirements = execution_plan.evidence_requirements
        for source_kind in sorted(requirements.required_source_kinds, key=str):
            count = sum(item.source_kind is source_kind for item in items_by_id.values())
            minimum = requirements.minimum_items_by_kind.get(source_kind, 1)
            if count < minimum:
                missing.append(
                    EvidenceRequirementFailureV1(
                        source_kind=source_kind,
                        resource_alias=None,
                        reason=(
                            EvidenceRequirementFailureReason.EMPTY
                            if not items_by_id
                            else EvidenceRequirementFailureReason.MISSING
                        ),
                    )
                )

        for alias in requirements.required_tafsir_aliases:
            matching = [
                item
                for item in items_by_id.values()
                if item.source_kind is EvidenceSourceKind.TAFSIR
                and item.source_policy_alias == alias
            ]
            if not matching:
                missing.append(
                    EvidenceRequirementFailureV1(
                        source_kind=EvidenceSourceKind.TAFSIR,
                        resource_alias=alias,
                        reason=(
                            EvidenceRequirementFailureReason.EMPTY
                            if not any(
                                item.source_kind is EvidenceSourceKind.TAFSIR
                                for item in items_by_id.values()
                            )
                            else EvidenceRequirementFailureReason.MISSING
                        ),
                    )
                )
            elif any(item.retrieval.method is not RetrievalMethod.EXACT for item in matching):
                raise EvidenceAssemblyError("Tafsir evidence must come from exact hydration")

        if requirements.require_two_distinct_tafsir_resources:
            distinct_aliases = {
                item.source_policy_alias
                for item in items_by_id.values()
                if item.source_kind is EvidenceSourceKind.TAFSIR
            }
            if len(distinct_aliases) < 2:
                missing.append(
                    EvidenceRequirementFailureV1(
                        source_kind=EvidenceSourceKind.TAFSIR,
                        resource_alias=None,
                        reason=EvidenceRequirementFailureReason.MISSING,
                    )
                )

        fetch_results = [
            results[node.node_id]
            for node in execution_plan.nodes
            if node.tool_id in canonical_fetch_tools
        ]
        if any(result.status is ToolResultStatus.ERROR for result in fetch_results):
            missing.append(
                EvidenceRequirementFailureV1(
                    source_kind=None,
                    resource_alias=None,
                    reason=EvidenceRequirementFailureReason.UNAVAILABLE,
                )
            )

        aliases = sorted({item.source_policy_alias for item in items_by_id.values()})
        permitted_deployments = {"groq:openai/gpt-oss-20b:zdr", LOCAL_GEMMA_DEPLOYMENT_ID}
        for alias in aliases:
            display = self.policy.authorize(alias, SourceOperation.DISPLAY)
            await self.repository.record_activation(display)
            inference = self.policy.authorize(
                alias,
                SourceOperation.EXTERNAL_LLM_INFERENCE,
                deployment=GROQ_DEPLOYMENT,
            )
            await self.repository.record_activation(inference)
            try:
                local_inference = self.policy.authorize(
                    alias,
                    SourceOperation.LOCAL_LLM_INFERENCE,
                    deployment=LOCAL_GEMMA_DEPLOYMENT,
                )
            except SourceAuthorizationError:
                permitted_deployments.discard(LOCAL_GEMMA_DEPLOYMENT_ID)
            else:
                await self.repository.record_activation(local_inference)

        return EvidenceSet(
            schema_version="1",
            turn_id=turn_id,
            items_by_id=items_by_id,
            requirements=requirements,
            requirements_satisfied=not missing,
            missing_requirements=missing,
            permitted_synthesis_deployments=(
                permitted_deployments if items_by_id and not missing else set()
            ),
        )

    async def record_chat_completion_activation(
        self, evidence: EvidenceSet, completion: ProviderCompletion
    ) -> None:
        """Record the actual model identity used for a post-evidence chat completion.

        Synthesis authorization is established during assembly, but the composer can use a
        different approved Groq model or the local route.  Recording the returned completion
        makes that distinction auditable instead of attributing every chat operation to 20B.
        """

        route = completion.effective_deployment
        is_local = route.effective_provider_id == "local"
        operation = (
            SourceOperation.LOCAL_LLM_INFERENCE
            if is_local
            else SourceOperation.EXTERNAL_LLM_INFERENCE
        )
        deployment = {
            "gateway_provider_id": route.gateway_provider_id,
            "effective_provider_id": route.effective_provider_id,
            "model_id": route.model_id,
            "inference_location": "LOCAL" if is_local else "CLOUD",
            "zero_data_retention": route.zero_data_retention_enforced,
            "provider_training": route.provider_training_allowed,
            "aggregator_routing": False,
        }
        for alias in sorted({item.source_policy_alias for item in evidence.items_by_id.values()}):
            decision = self.policy.authorize(alias, operation, deployment=deployment)
            await self.repository.record_activation(decision)
