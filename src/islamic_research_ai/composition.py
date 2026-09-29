from __future__ import annotations

from dataclasses import dataclass

from islamic_research_ai.anti_fabrication import AntiFabricationGate
from islamic_research_ai.chat_deployments import load_role_deployments
from islamic_research_ai.chat_routing import QuotaFallbackPlanner, QuotaFallbackSynthesizer
from islamic_research_ai.composition_layer import AnswerComposer
from islamic_research_ai.conversations import ConversationService
from islamic_research_ai.evidence import EvidenceAssembler
from islamic_research_ai.ingestion.qul_ingest import QUL_RESOURCES
from islamic_research_ai.planner import GroqPlanner, PlanValidator
from islamic_research_ai.providers.cloudflare import (
    CloudflareEmbeddingClient,
    CloudflareRerankerClient,
)
from islamic_research_ai.providers.groq import GroqStructuredClient
from islamic_research_ai.providers.local_mlx import LocalMlxStructuredClient, LocalReadinessV1
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourcePolicy
from islamic_research_ai.routing import LOCAL_GEMMA_DEPLOYMENT_ID, ProviderRoutePolicy
from islamic_research_ai.scope import ScopePolicy
from islamic_research_ai.service import ResearchService
from islamic_research_ai.settings import ROOT, Settings, load_settings
from islamic_research_ai.synthesis import ExactRenderer, GroqSynthesizer, GroundingValidator
from islamic_research_ai.tools import PostgresQulTafsirSource, PostgresQuranSource, WorkflowExecutor
from islamic_research_ai.workflow import WorkflowCompiler


@dataclass
class AppContainer:
    settings: Settings
    repository: PostgresRepository
    groq: GroqStructuredClient
    local_chat: LocalMlxStructuredClient
    local_readiness: LocalReadinessV1
    embeddings: CloudflareEmbeddingClient
    reranker: CloudflareRerankerClient | None
    composer_groq: GroqStructuredClient
    service: ResearchService
    conversations: ConversationService

    async def close(self) -> None:
        await self.groq.close()
        await self.local_chat.close()
        await self.embeddings.close()
        if self.reranker is not None:
            await self.reranker.close()
        await self.composer_groq.close()
        await self.repository.close()


async def build_container() -> AppContainer:
    settings = load_settings()
    if settings.groq_api_key is None:
        raise RuntimeError("GROQ_API_KEY is required")
    if settings.cloudflare_api_token is None or settings.cloudflare_account_id is None:
        raise RuntimeError("Cloudflare embedding credentials are required")
    repository = PostgresRepository(
        settings.database_url, quran_embedding_variant=settings.quran_embedding_variant
    )
    await repository.open()
    await repository.migrate(ROOT / "migrations/001_initial.sql")
    await repository.migrate(ROOT / "migrations/002_archive_pre_v21_contract_records.sql")
    await repository.migrate(ROOT / "migrations/003_redact_qf_tafsir_source_text.sql")
    await repository.migrate(ROOT / "migrations/004_conversation_history.sql")
    await repository.migrate(ROOT / "migrations/005_qul_tafsir.sql")
    await repository.migrate(ROOT / "migrations/006_turn_trace_events.sql")
    await repository.migrate(ROOT / "migrations/007_gate_diagnostics.sql")
    await repository.migrate(ROOT / "migrations/008_enriched_quran_embeddings.sql")
    await repository.migrate(ROOT / "migrations/009_qul_ayah_topics.sql")
    groq = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )
    local_chat = LocalMlxStructuredClient(
        base_url=settings.local_chat_base_url,
        manifest_path=settings.local_model_manifest_path,
        model_root=settings.local_model_root,
        timeout=settings.local_chat_timeout_seconds,
        health_timeout=settings.local_chat_health_timeout_seconds,
        max_concurrency=settings.local_chat_max_concurrency,
        max_prompt_characters=settings.local_chat_max_prompt_characters,
    )
    local_readiness = await local_chat.readiness(run_schema_probe=True)
    embeddings = CloudflareEmbeddingClient(
        api_token=settings.cloudflare_api_token,
        account_id=settings.cloudflare_account_id,
        model=settings.cloudflare_embedding_model,
    )
    reranker = (
        CloudflareRerankerClient(
            api_token=settings.cloudflare_api_token,
            account_id=settings.cloudflare_account_id,
            model=settings.cloudflare_reranker_model,
        )
        if settings.cloudflare_reranker_enabled
        else None
    )
    policy = SourcePolicy(
        settings.source_policy_path,
        environment=settings.app_environment,
        purpose=settings.development_purpose,
        override_acknowledged=settings.project_owner_risk_override_ack,
    )
    source = PostgresQuranSource(
        repository,
        embeddings,
        reranker=reranker,
        policy=policy,
        qul_topic_arm_enabled=settings.qul_topic_arm_enabled,
        qul_topic_arm_limit=settings.qul_topic_arm_limit,
        qul_topic_rerank_enabled=settings.qul_topic_rerank_enabled,
    )
    for qul_resource in QUL_RESOURCES:
        await repository.upsert_resource(
            {
                "resource_alias": qul_resource.resource_alias,
                "source_id": "qul",
                "work_title": qul_resource.work_title,
                "language": "ar",
                "author_translator_or_grader": list(qul_resource.authors),
                "canonical_base_url": (
                    f"https://qul.tarteel.ai/resources/tafsir/{qul_resource.qul_resource_id}"
                ),
                "attribution": {
                    "display_name": "Quranic Universal Library (Tarteel)",
                    "url": "https://qul.tarteel.ai",
                    "notice": (
                        "Digitised export via the Quranic Universal Library; original "
                        f"work by {', '.join(qul_resource.authors)}."
                    ),
                },
                "rights_policy": policy.resources[qul_resource.resource_alias]["rights"],
            }
        )
    qul_tafsir = PostgresQulTafsirSource(repository)
    tafsir_source = qul_tafsir
    # WP-1 (V3 plan): per-role Groq deployment map. `planner`/`relevance`/`synthesis` share the
    # existing fast/cheap model and client (see the role map's own "rationale" fields for why);
    # `composer` gets its own client on a larger model -- the only role worth spending
    # capability on, since it produces the prose the user actually reads (WP-2).
    role_deployments = load_role_deployments(settings.chat_deployments_path)
    synthesis_role = role_deployments["synthesis"]
    synthesis_tafsir_tokens = (
        synthesis_role.max_output_tokens_tafsir or synthesis_role.max_output_tokens
    )
    composer_role = role_deployments["composer"]
    composer_groq = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=composer_role.model_id,
        base_url=settings.groq_base_url,
    )

    plan_validator = PlanValidator()
    grounding_validator = GroundingValidator(
        risk_concept_human_subject_terms_enabled=settings.risk_concept_human_subject_terms_enabled
    )
    groq_planner = GroqPlanner(groq, plan_validator)
    local_planner = GroqPlanner(
        local_chat,
        plan_validator,
        route_validator=ProviderRoutePolicy.require_local_gemma_pinned_free,
    )
    groq_synthesizer = GroqSynthesizer(
        groq,
        grounding_validator,
        max_tokens_quran=synthesis_role.max_output_tokens,
        max_tokens_tafsir=synthesis_tafsir_tokens,
    )
    local_synthesizer = GroqSynthesizer(
        local_chat,
        grounding_validator,
        deployment_id=LOCAL_GEMMA_DEPLOYMENT_ID,
        route_validator=ProviderRoutePolicy.require_local_gemma_pinned_free,
        max_tokens_quran=synthesis_role.max_output_tokens,
        max_tokens_tafsir=synthesis_tafsir_tokens,
    )
    answer_composer = AnswerComposer(
        composer_groq,
        AntiFabricationGate(),
        local_client=local_chat,
        route_validator=lambda completion: ProviderRoutePolicy.require_configured_route(
            completion, role_deployment=composer_role.groq_role_deployment
        ),
        local_route_validator=ProviderRoutePolicy.require_local_gemma_pinned_free,
        max_tokens=composer_role.max_output_tokens,
    )
    service = ResearchService(
        planner=QuotaFallbackPlanner(groq_planner, local_planner),
        compiler=WorkflowCompiler(topic_anchor_limit=settings.topic_anchor_limit),
        executor=WorkflowExecutor(
            source,
            search_limit=settings.search_limit,
            hydrate_limit=settings.hydrate_limit,
            tafsir_source=tafsir_source,
        ),
        source=source,
        evidence_assembler=EvidenceAssembler(policy, repository),
        synthesizer=QuotaFallbackSynthesizer(groq_synthesizer, local_synthesizer),
        renderer=ExactRenderer(),
        repository=repository,
        scope_policy=ScopePolicy(),
        force_local_chat=settings.local_chat_force,
        deterministic_topic_plans=True,
        enable_topic_tafsir=True,
        topic_tafsir_alias=settings.topic_tafsir_alias,
        relevance_ranked_retention_enabled=settings.relevance_ranked_retention_enabled,
        relevance_distinctive_terms_enabled=settings.relevance_distinctive_terms_enabled,
        honest_insufficient_fallback_enabled=settings.honest_insufficient_fallback_enabled,
        relevance_retained_passage_limit=settings.relevance_retained_passage_limit,
        relevance_empty_match_retains_anchors=settings.relevance_empty_match_retains_anchors,
        answer_composer=answer_composer,
        answer_composer_enabled=True,
        developer_trace_enabled=settings.developer_trace_enabled,
    )
    conversations = ConversationService(repository, service)
    return AppContainer(
        settings,
        repository,
        groq,
        local_chat,
        local_readiness,
        embeddings,
        reranker,
        composer_groq,
        service,
        conversations,
    )
