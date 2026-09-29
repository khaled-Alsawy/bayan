from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from islamic_research_ai.chat_routing import QuotaFallbackSynthesizer, TurnChatRouteState
from islamic_research_ai.contracts import (
    AnswerDraftV1,
    AnswerStatus,
    AttributionRole,
    AttributionV1,
    ClaimBlockV1,
    ClaimType,
    ContextField,
    ContextProvenance,
    ContextValueV1,
    EffectiveDeploymentIdentityV1,
    EvidenceItem,
    EvidenceSet,
    EvidenceSourceKind,
    FetchQuranPassageIntentV1,
    FetchTafsirPassagesIntentV1,
    Intent,
    ProviderCompletion,
    QuoteMode,
    QuoteReferenceV1,
    QuranPointer,
    ResearchPlanV1,
    ResearchRequest,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SanitizedTurnV1,
    SemanticTargetKind,
    SemanticTargetSummaryV1,
    SourceBlockRequestV1,
    SourceIdentityV1,
    SourceRevisionIdentity,
    TafsirFetchMode,
    TafsirPointer,
    ToolResultStatus,
    TurnRole,
    TypedSourceReference,
    VerseKey,
    WorkingContextV1,
    no_clarification,
)
from islamic_research_ai.evidence import EvidenceAssembler
from islamic_research_ai.persistence import (
    RESTRICTED_SOURCE_TEXT_REDACTION,
    response_for_durable_storage,
)
from islamic_research_ai.planner import (
    PlanValidationError,
    PlanValidator,
    resolve_deterministically,
)
from islamic_research_ai.ports import (
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from islamic_research_ai.rights import SourceAuthorizationError, SourceOperation, SourcePolicy
from islamic_research_ai.routing import (
    LOCAL_GEMMA_DEPLOYMENT_ID,
    LOCAL_GEMMA_RESOLVED_MODEL,
    LOCAL_GEMMA_ROUTE_POLICY_ID,
    ProviderRoutePolicy,
)
from islamic_research_ai.scope import ScopePolicy
from islamic_research_ai.service import ResearchService
from islamic_research_ai.synthesis import (
    ExactRenderer,
    GroqSynthesizer,
    GroundingError,
    GroundingValidator,
    TafsirAnswerProposalV1,
    TafsirClaimProposalV1,
)
from islamic_research_ai.tools import ExecutionPlanRejected, WorkflowExecutor
from islamic_research_ai.workflow import WorkflowCompiler

ROOT = Path(__file__).resolve().parents[1]
IBN_KATHIR = "qul-tafsir-ibnkathir-ar-v1"
AL_QURTUBI = "qul-tafsir-alqurtubi-ar-v1"


class MemoryRepository:
    def __init__(self) -> None:
        self.activations: list[Any] = []
        self.plans: list[Any] = []
        self.responses: list[Any] = []

    async def record_activation(self, value: Any) -> None:
        self.activations.append(value)

    async def persist_plan(self, **value: Any) -> None:
        self.plans.append(value)

    async def persist_response(self, **value: Any) -> None:
        self.responses.append(value)


def development_policy() -> SourcePolicy:
    return SourcePolicy(
        ROOT / "config/source_policies.development.json",
        environment="development",
        purpose="private_poc",
        override_acknowledged=True,
    )


def production_policy() -> SourcePolicy:
    return SourcePolicy(
        ROOT / "config/source_policies.production.json",
        environment="production",
        purpose="production",
        override_acknowledged=True,
    )


def working_context(language: str = "ar") -> WorkingContextV1:
    turn_id = uuid4()
    return WorkingContextV1(
        schema_version="1",
        conversation_id=uuid4(),
        state_version=0,
        recent_turns=[
            SanitizedTurnV1(
                turn_id=turn_id,
                role=TurnRole.USER,
                sanitized_text="Explain Quran 2:153",
            )
        ],
        values=[
            ContextValueV1(
                field=ContextField.RESPONSE_LANGUAGE,
                value=language,
                provenance=ContextProvenance.CURRENT_USER,
                source_turn_id=turn_id,
                turn_age=0,
            )
        ],
        pending_clarification=None,
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )


def tafsir_plan(*, compare: bool = False, aliases: list[str] | None = None) -> ResearchPlanV1:
    selected = aliases or ([IBN_KATHIR, AL_QURTUBI] if compare else [IBN_KATHIR])
    return ResearchPlanV1(
        schema_version="1",
        intent=Intent.COMPARE_TAFSIR if compare else Intent.EXPLAIN_QURAN_VERSE,
        response_language="ar",
        semantic_target_summary=SemanticTargetSummaryV1(
            kind=SemanticTargetKind.QURAN_PASSAGE,
            normalized_label="Quran 2:153",
            entity_labels=["2:153"],
        ),
        inherited_context=[],
        clarification=no_clarification(),
        scope_signals=set(),
        requested_operations=[
            FetchQuranPassageIntentV1(
                operation="FETCH_QURAN_PASSAGE",
                target="DETERMINISTIC_QURAN_REFERENCES",
                translation_resource_alias="quranenc-rowwad-en-v1.0.19",
                include_adjacent_verses=False,
            ),
            FetchTafsirPassagesIntentV1(
                operation="FETCH_TAFSIR_PASSAGES",
                target="DETERMINISTIC_QURAN_REFERENCES",
                mode=(
                    TafsirFetchMode.COMPARISON_EXACTLY_TWO
                    if compare
                    else TafsirFetchMode.EXPLANATION_ONE
                ),
                tafsir_resource_aliases=selected,
                language="ar",
            ),
        ],
    )


def compile_plan(*, compare: bool = False, aliases: list[str] | None = None):
    resolution = resolve_deterministically("Explain Quran 2:153")
    scope = ScopePolicy().precheck("Explain Quran 2:153", resolution)
    plan = PlanValidator().validate(
        tafsir_plan(compare=compare, aliases=aliases),
        working_context=working_context(),
        deterministic_resolution=resolution,
    )
    return WorkflowCompiler().compile(
        plan,
        deterministic_resolution=resolution,
        scope=scope,
    )


def revision(alias: str, source_id: str) -> SourceRevisionIdentity:
    return SourceRevisionIdentity(
        source_id=source_id,
        resource_alias=alias,
        revision_id=f"{alias}-revision",
        checksum=f"sha256:{alias}",
    )


def quran_evidence() -> list[EvidenceItem]:
    key = VerseKey(chapter=2, verse=153)
    pointer = QuranPointer(kind="QURAN", verse_keys=[key])
    retrieval = RetrievalProvenanceV1(
        tool_id="get_quran_passage.v1",
        method=RetrievalMethod.EXACT,
        index_revision=None,
        rank=1,
        component_scores={},
    )
    return [
        EvidenceItem(
            evidence_id="H1-AR",
            source_kind=EvidenceSourceKind.QURAN,
            canonical_pointer=pointer,
            source_identity=SourceIdentityV1(
                source_id="tanzil",
                resource_alias="tanzil-uthmani-v1.1",
                work_title="The Quran — Uthmani text",
                author_translator_or_grader=["Tanzil Project"],
                canonical_base_url="https://tanzil.net/",
            ),
            source_revision=revision("tanzil-uthmani-v1.1", "tanzil"),
            language="ar",
            exact_text="يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ",
            approved_quote_spans=[],
            reference=TypedSourceReference(
                scheme="quran", value="2:153", canonical_link="https://tanzil.net/#2:153"
            ),
            attributions=[
                AttributionV1(
                    role=AttributionRole.SOURCE,
                    display_name="Tanzil Project",
                    required_notice=None,
                )
            ],
            retrieval=retrieval,
            content_hash="sha256:quran-ar",
            source_policy_alias="tanzil-uthmani-v1.1",
        ),
        EvidenceItem(
            evidence_id="H1-EN",
            source_kind=EvidenceSourceKind.QURAN_TRANSLATION,
            canonical_pointer=pointer,
            source_identity=SourceIdentityV1(
                source_id="quranenc",
                resource_alias="quranenc-rowwad-en-v1.0.19",
                work_title="Translation of the Meanings — Rowwad",
                author_translator_or_grader=["Rowwad Translation Center", "QuranEnc"],
                canonical_base_url="https://quranenc.com",
            ),
            source_revision=revision("quranenc-rowwad-en-v1.0.19", "quranenc"),
            language="en",
            exact_text="Seek help through patience and prayer.",
            approved_quote_spans=[],
            reference=TypedSourceReference(
                scheme="quran",
                value="2:153",
                canonical_link="https://quranenc.com/en/browse/english_rwwad/2/153",
            ),
            attributions=[
                AttributionV1(
                    role=AttributionRole.TRANSLATOR,
                    display_name="Rowwad Translation Center",
                    required_notice=None,
                )
            ],
            retrieval=retrieval,
            content_hash="sha256:quran-en",
            source_policy_alias="quranenc-rowwad-en-v1.0.19",
        ),
    ]


def tafsir_evidence(alias: str, *, text: str | None = None) -> EvidenceItem:
    key = VerseKey(chapter=2, verse=153)
    title = "Tafsir Ibn Kathir" if alias == IBN_KATHIR else "Tafsir al-Qurtubi"
    author = "Ibn Kathir" if alias == IBN_KATHIR else "Al-Qurtubi"
    return EvidenceItem(
        evidence_id=f"RAW-{alias}",
        source_kind=EvidenceSourceKind.TAFSIR,
        canonical_pointer=TafsirPointer(
            kind="TAFSIR",
            passage_id=f"qul:{alias}:2:153",
            verse_keys=[key],
        ),
        source_identity=SourceIdentityV1(
            source_id="qul",
            resource_alias=alias,
            work_title=title,
            author_translator_or_grader=[author],
            canonical_base_url="https://quran.com",
        ),
        source_revision=revision(alias, "qul"),
        language="ar",
        exact_text=text or f"يوضح {title} معنى الصبر والصلاة في الآية.",
        approved_quote_spans=[],
        reference=TypedSourceReference(
            scheme="tafsir",
            value="2:153",
            canonical_link="https://qul.tarteel.ai",
        ),
        attributions=[
            AttributionV1(
                role=AttributionRole.AUTHOR,
                display_name=author,
                required_notice=None,
            ),
            AttributionV1(
                role=AttributionRole.SOURCE,
                display_name="Quranic Universal Library (Tarteel)",
                required_notice="Digitised export via the Quranic Universal Library.",
            ),
        ],
        retrieval=RetrievalProvenanceV1(
            tool_id="get_tafsir_passages.v1",
            method=RetrievalMethod.EXACT,
            index_revision=None,
            rank=1,
            component_scores={},
        ),
        content_hash=f"sha256:{alias}",
        source_policy_alias=alias,
    )


class QuranSourceFixture:
    def __init__(self) -> None:
        self.last_embedding_completion = None
        self.last_provider_failure = None
        self.search_calls = 0

    async def search(self, query: str, *, limit: int) -> list[Any]:
        self.search_calls += 1
        return []

    async def hydrate_candidates(self, candidates: list[Any], *, limit: int) -> list[Any]:
        raise AssertionError("Tafsir exact intents must not hydrate search candidates")

    async def hydrate_references(
        self, verse_keys: list[VerseKey], *, limit: int
    ) -> list[EvidenceItem]:
        assert verse_keys == [VerseKey(chapter=2, verse=153)]
        return quran_evidence()


class TafsirSourceFixture:
    def __init__(self, aliases: list[str], *, empty: bool = False, failure: str | None = None):
        self.aliases = aliases
        self.empty = empty
        self.failure = failure
        self.calls = 0
        self.last_provider_failure: ProviderFailureDiagnostic | None = None

    async def fetch_passages(
        self,
        verse_keys: list[VerseKey],
        *,
        resource_aliases: list[str],
        character_budget: int,
    ) -> list[EvidenceItem]:
        self.calls += 1
        assert verse_keys == [VerseKey(chapter=2, verse=153)]
        assert resource_aliases == self.aliases
        assert character_budget <= 50_000
        if self.failure is not None:
            self.last_provider_failure = ProviderFailureDiagnostic(
                provider_id="qul",
                category=self.failure,  # type: ignore[arg-type]
                http_status=429 if self.failure == "RATE_LIMITED" else None,
                retry_after_ms=1000 if self.failure == "RATE_LIMITED" else None,
                quota_limit_tokens=None,
                quota_used_tokens=None,
                quota_requested_tokens=None,
                request_id="fixture-request",
                safe_detail="fixture provider failure",
            )
            raise ProviderUnavailableError(self.last_provider_failure)
        if self.empty:
            return []
        return [tafsir_evidence(alias) for alias in resource_aliases]


def test_explanation_one_and_comparison_exactly_two_validate() -> None:
    resolution = resolve_deterministically("Explain Quran 2:153")
    validator = PlanValidator()
    validator.validate(
        tafsir_plan(),
        working_context=working_context(),
        deterministic_resolution=resolution,
    )
    validator.validate(
        tafsir_plan(compare=True),
        working_context=working_context(),
        deterministic_resolution=resolution,
    )


@pytest.mark.parametrize(
    "plan",
    [
        tafsir_plan(compare=True, aliases=[IBN_KATHIR, IBN_KATHIR]),
        tafsir_plan(compare=True, aliases=[IBN_KATHIR]),
        tafsir_plan(aliases=["qf-tafsir-ibn-kathir-en"]),
        tafsir_plan(aliases=["unknown-tafsir"]),
    ],
)
def test_invalid_tafsir_alias_invariants_are_rejected(plan: ResearchPlanV1) -> None:
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(
            plan,
            working_context=working_context(),
            deterministic_resolution=resolve_deterministically("Explain Quran 2:153"),
        )


def test_tafsir_compiler_uses_exact_fetches_and_no_semantic_search() -> None:
    for compare in (False, True):
        execution = compile_plan(compare=compare)
        assert [node.tool_id for node in execution.nodes] == [
            "get_quran_passage.v1",
            "get_tafsir_passages.v1",
        ]
        assert all(not node.depends_on for node in execution.nodes)
        assert execution.evidence_requirements.required_tafsir_aliases == (
            [IBN_KATHIR, AL_QURTUBI] if compare else [IBN_KATHIR]
        )
        assert execution.evidence_requirements.require_two_distinct_tafsir_resources is compare


@pytest.mark.asyncio
async def test_executor_runs_both_exact_fetches_and_never_searches() -> None:
    quran = QuranSourceFixture()
    tafsir = TafsirSourceFixture([IBN_KATHIR, AL_QURTUBI])
    results = await WorkflowExecutor(
        quran,
        search_limit=8,
        hydrate_limit=5,
        tafsir_source=tafsir,
    ).execute(compile_plan(compare=True))
    assert quran.search_calls == 0
    assert tafsir.calls == 1
    assert results["node_fetch_quran"].status is ToolResultStatus.SUCCESS
    assert results["node_fetch_tafsir"].status is ToolResultStatus.SUCCESS
    assert all(
        item.source_kind is EvidenceSourceKind.TAFSIR
        for item in results["node_fetch_tafsir"].evidence_items
    )


@pytest.mark.asyncio
async def test_executor_without_tafsir_source_returns_typed_unavailable_result() -> None:
    results = await WorkflowExecutor(QuranSourceFixture(), search_limit=8, hydrate_limit=5).execute(
        compile_plan()
    )
    assert results["node_fetch_tafsir"].status is ToolResultStatus.ERROR
    assert results["node_fetch_tafsir"].error is not None
    assert results["node_fetch_tafsir"].error.code.value == "SOURCE_UNAVAILABLE"


def test_partial_tafsir_execution_plan_never_executes() -> None:
    execution = compile_plan().model_copy(update={"nodes": compile_plan().nodes[:1]})
    with pytest.raises(ExecutionPlanRejected):
        WorkflowExecutor._validate_complete_plan(execution)


async def assembled_evidence(*, compare: bool = False, empty: bool = False) -> EvidenceSet:
    aliases = [IBN_KATHIR, AL_QURTUBI] if compare else [IBN_KATHIR]
    quran = QuranSourceFixture()
    tafsir = TafsirSourceFixture(aliases, empty=empty)
    execution = compile_plan(compare=compare)
    results = await WorkflowExecutor(
        quran,
        search_limit=8,
        hydrate_limit=5,
        tafsir_source=tafsir,
    ).execute(execution)
    return await EvidenceAssembler(development_policy(), MemoryRepository()).assemble(
        turn_id=uuid4(), execution_plan=execution, results=results
    )


@pytest.mark.asyncio
async def test_qul_tafsir_allows_configured_external_and_local_synthesis() -> None:
    aliases = [IBN_KATHIR, AL_QURTUBI]
    execution = compile_plan(compare=True)
    results = await WorkflowExecutor(
        QuranSourceFixture(),
        search_limit=8,
        hydrate_limit=5,
        tafsir_source=TafsirSourceFixture(aliases),
    ).execute(execution)
    repository = MemoryRepository()
    evidence = await EvidenceAssembler(development_policy(), repository).assemble(
        turn_id=uuid4(), execution_plan=execution, results=results
    )
    assert evidence.requirements_satisfied is True
    assert evidence.permitted_synthesis_deployments == {
        "groq:openai/gpt-oss-20b:zdr",
        LOCAL_GEMMA_DEPLOYMENT_ID,
    }
    assert any(
        activation.operation is SourceOperation.EXTERNAL_LLM_INFERENCE
        for activation in repository.activations
    )
    source_policy = development_policy()
    source_policy.authorize(
        IBN_KATHIR,
        SourceOperation.EXTERNAL_LLM_INFERENCE,
        deployment={
            "gateway_provider_id": "groq",
            "effective_provider_id": "groq",
            "model_id": "openai/gpt-oss-20b",
            "zero_data_retention": True,
            "provider_training": False,
            "aggregator_routing": False,
        },
    )


@pytest.mark.asyncio
async def test_empty_tafsir_mapping_fails_evidence_requirements() -> None:
    evidence = await assembled_evidence(empty=True)
    assert evidence.requirements_satisfied is False
    assert any(item.resource_alias == IBN_KATHIR for item in evidence.missing_requirements)


@pytest.mark.asyncio
async def test_production_blocks_unknown_qul_tafsir_inference() -> None:
    execution = compile_plan()
    results = await WorkflowExecutor(
        QuranSourceFixture(),
        search_limit=8,
        hydrate_limit=5,
        tafsir_source=TafsirSourceFixture([IBN_KATHIR]),
    ).execute(execution)
    with pytest.raises(SourceAuthorizationError, match="blocked"):
        await EvidenceAssembler(production_policy(), MemoryRepository()).assemble(
            turn_id=uuid4(), execution_plan=execution, results=results
        )


def local_completion() -> ProviderCompletion:
    return ProviderCompletion(
        effective_deployment=EffectiveDeploymentIdentityV1(
            deployment_id=LOCAL_GEMMA_DEPLOYMENT_ID,
            gateway_provider_id="local",
            effective_provider_id="local",
            model_id="gemma-4-26b-a4b-it",
            resolved_model_version_or_name=LOCAL_GEMMA_RESOLVED_MODEL,
            region="loopback",
            subprocessor_ids=set(),
            zero_data_retention_enforced=True,
            provider_training_allowed=False,
            route_policy_id=LOCAL_GEMMA_ROUTE_POLICY_ID,
        ),
        request_id=None,
        input_tokens=10,
        output_tokens=20,
        latency_ms=10,
        billed_cost_usd=Decimal("0"),
        paid_route=False,
        fallback_used=False,
    )


class Aq15LengthThenValidLocalClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> tuple[Any, ProviderCompletion]:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            raise StructuredOutputValidationError(
                "Local Gemma returned an invalid structured completion "
                "(finish_reason=length; chars=1159,starts_object=False,"
                "ends_object=False,brace_delta=0)"
            )
        return (
            TafsirAnswerProposalV1(
                schema_version="1",
                status="ANSWERABLE",
                claims=[
                    TafsirClaimProposalV1(
                        claim_id="claim_1",
                        text="يفسر ابن كثير الصبر والصلاة في هذه الآية.",
                        evidence_ids=["E3"],
                    )
                ],
            ),
            local_completion(),
        )


@pytest.mark.asyncio
async def test_aq15_length_failure_repairs_once_with_compact_local_schema() -> None:
    evidence = await assembled_evidence()
    long_tafsir = evidence.items_by_id["E3"].model_copy(
        update={
            "exact_text": (
                "يفسر ابن كثير الصبر والصلاة في هذه الآية. " * 100
            ).strip()
        }
    )
    evidence = evidence.model_copy(
        update={"items_by_id": {**evidence.items_by_id, "E3": long_tafsir}}
    )
    client = Aq15LengthThenValidLocalClient()
    draft, completions = await GroqSynthesizer(
        client,  # type: ignore[arg-type]
        GroundingValidator(),
        deployment_id=LOCAL_GEMMA_DEPLOYMENT_ID,
        route_validator=ProviderRoutePolicy.require_local_gemma_pinned_free,
    ).synthesize(
        question="فسر الآية 2:153 من تفسير ابن كثير.",
        evidence=evidence,
        response_language="ar",
    )

    assert len(client.requests) == 2
    assert [request["max_tokens"] for request in client.requests] == [2000, 3000]
    assert all(request["schema_name"] == "tafsir_answer_proposal_v1" for request in client.requests)
    prompt = json.loads(client.requests[0]["user_prompt"])
    tafsir_context = next(item for item in prompt["evidence"] if item["source_kind"] == "TAFSIR")
    assert tafsir_context["evidence_text_is_bounded_excerpt"] is True
    assert len(tafsir_context["evidence_text"]) <= 2_600
    assert "exact_text" not in tafsir_context
    assert "output limit" in client.requests[1]["user_prompt"]
    assert len(completions) == 1

    claims = [block for block in draft.blocks if isinstance(block, ClaimBlockV1)]
    requests = [block for block in draft.blocks if isinstance(block, SourceBlockRequestV1)]
    assert len(claims) == 1 and "E3" not in claims[0].text
    assert [item.quote_reference.evidence_id for item in requests] == ["E1", "E2", "E3"]
    GroundingValidator().validate(draft, evidence, response_language="ar")


@pytest.mark.asyncio
async def test_successful_aq15_response_redacts_qf_text_before_durable_storage() -> None:
    evidence = await assembled_evidence()
    historical_tafsir = evidence.items_by_id["E3"].model_copy(
        update={
            "source_identity": evidence.items_by_id["E3"].source_identity.model_copy(
                update={"source_id": "quran_foundation"}
            ),
            "source_revision": revision("qf-tafsir-ibn-kathir-en", "quran_foundation"),
            "source_policy_alias": "qf-tafsir-ibn-kathir-en",
        }
    )
    evidence = evidence.model_copy(
        update={"items_by_id": {**evidence.items_by_id, "E3": historical_tafsir}}
    )
    response = ExactRenderer().render(
        conversation_id=uuid4(),
        turn_id=evidence.turn_id,
        correlation_id=uuid4(),
        draft=explanation_draft(),
        evidence=evidence,
    )
    durable = response_for_durable_storage(response)

    public_qf_sources = [
        block
        for block in response.blocks
        if block.block_type == "SOURCE"
        and block.citation.source_revision.source_id == "quran_foundation"
    ]
    durable_qf_sources = [
        block
        for block in durable.blocks
        if block.block_type == "SOURCE"
        and block.citation.source_revision.source_id == "quran_foundation"
    ]
    assert public_qf_sources[0].exact_text != RESTRICTED_SOURCE_TEXT_REDACTION
    assert durable_qf_sources[0].exact_text == RESTRICTED_SOURCE_TEXT_REDACTION


def comparison_draft() -> AnswerDraftV1:
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="CL1",
                text="يوضح تفسير ابن كثير معنى الصبر والصلاة في الآية.",
                claim_type=ClaimType.FACTUAL_SUMMARY,
                evidence_ids=["E3"],
            ),
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="CL2",
                text="يوضح تفسير القرطبي معنى الصبر والصلاة في الآية.",
                claim_type=ClaimType.COMPARISON,
                evidence_ids=["E4"],
            ),
            *[
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"S{index}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=f"E{index}", approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                )
                for index in range(1, 5)
            ],
        ],
    )


@pytest.mark.asyncio
async def test_grounding_requires_tafsir_for_every_claim_and_both_comparison_resources() -> None:
    evidence = await assembled_evidence(compare=True)
    validator = GroundingValidator()
    validator.validate(comparison_draft(), evidence, response_language="ar")
    wrong = comparison_draft().model_copy(deep=True)
    assert isinstance(wrong.blocks[1], ClaimBlockV1)
    wrong.blocks[1].evidence_ids = ["E3"]
    with pytest.raises(GroundingError, match="Every required Tafsir resource"):
        validator.validate(wrong, evidence, response_language="ar")
    quran_only_claim = comparison_draft().model_copy(deep=True)
    assert isinstance(quran_only_claim.blocks[0], ClaimBlockV1)
    quran_only_claim.blocks[0].evidence_ids = ["E2"]
    with pytest.raises(GroundingError, match="must cite exact Tafsir"):
        validator.validate(quran_only_claim, evidence, response_language="ar")


@pytest.mark.asyncio
async def test_renderer_keeps_exact_tafsir_text_and_structured_provenance() -> None:
    evidence = await assembled_evidence(compare=True)
    response = ExactRenderer().render(
        conversation_id=uuid4(),
        turn_id=evidence.turn_id,
        correlation_id=uuid4(),
        draft=comparison_draft(),
        evidence=evidence,
    )
    sources = [block for block in response.blocks if block.block_type == "SOURCE"]
    assert [block.exact_text for block in sources] == [
        evidence.items_by_id[f"E{index}"].exact_text for index in range(1, 5)
    ]
    tafsir_citations = [
        item for item in response.citations if item.source_kind is EvidenceSourceKind.TAFSIR
    ]
    assert {item.source_revision.resource_alias for item in tafsir_citations} == {
        IBN_KATHIR,
        AL_QURTUBI,
    }
    assert all(
        evidence.items_by_id[item.evidence_id].source_identity.work_title in item.display_label
        for item in tafsir_citations
    )


class MustNotRun:
    async def synthesize(self, **kwargs: Any) -> Any:
        raise AssertionError("External synthesis must not receive Tafsir evidence")


class RecordingLocalSynthesizer:
    def __init__(self, draft: AnswerDraftV1 | None = None) -> None:
        self.calls = 0
        self.draft = draft

    async def synthesize(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.draft is None:
            return "local-result"
        return self.draft, []


@pytest.mark.asyncio
async def test_qul_tafsir_uses_primary_synthesis_when_policy_permits_it() -> None:
    evidence = await assembled_evidence()
    primary = RecordingLocalSynthesizer()
    state = TurnChatRouteState()
    result = await QuotaFallbackSynthesizer(primary, MustNotRun()).synthesize(
        route_state=state,
        evidence=evidence,
    )
    assert result == "local-result"
    assert primary.calls == 1
    assert state.selection_stage is None
    assert state.snapshot()["inference_location"] == "CLOUD"


class FixedPlanner:
    def __init__(self, plan: ResearchPlanV1) -> None:
        self.plan = plan

    async def create_plan(self, **kwargs: Any) -> tuple[ResearchPlanV1, list[Any]]:
        return self.plan, []


def explanation_draft() -> AnswerDraftV1:
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="CL1",
                text="يوضح تفسير ابن كثير معنى الصبر والصلاة في الآية.",
                claim_type=ClaimType.FACTUAL_SUMMARY,
                evidence_ids=["E3"],
            ),
            *[
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"S{index}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=f"E{index}", approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                )
                for index in range(1, 4)
            ],
        ],
    )


def build_tafsir_service(
    *,
    compare: bool = False,
    empty: bool = False,
    failure: str | None = None,
) -> tuple[ResearchService, RecordingLocalSynthesizer, RecordingLocalSynthesizer]:
    aliases = [IBN_KATHIR, AL_QURTUBI] if compare else [IBN_KATHIR]
    repository = MemoryRepository()
    quran = QuranSourceFixture()
    tafsir = TafsirSourceFixture(aliases, empty=empty, failure=failure)
    primary = RecordingLocalSynthesizer(comparison_draft() if compare else explanation_draft())
    local = RecordingLocalSynthesizer(comparison_draft() if compare else explanation_draft())
    return (
        ResearchService(
            planner=FixedPlanner(tafsir_plan(compare=compare)),  # type: ignore[arg-type]
            compiler=WorkflowCompiler(),
            executor=WorkflowExecutor(
                quran,
                search_limit=8,
                hydrate_limit=5,
                tafsir_source=tafsir,
            ),
            source=quran,  # type: ignore[arg-type]
            evidence_assembler=EvidenceAssembler(development_policy(), repository),
            synthesizer=QuotaFallbackSynthesizer(primary, local),  # type: ignore[arg-type]
            renderer=ExactRenderer(),
            repository=repository,  # type: ignore[arg-type]
            scope_policy=ScopePolicy(),
        ),
        primary,
        local,
    )


@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.asyncio
async def test_full_service_explanation_and_comparison_use_configured_primary(
    compare: bool,
) -> None:
    service, primary, local = build_tafsir_service(compare=compare)
    response, trace = await service.research(
        ResearchRequest(
            message=(
                "Compare tafsir for Quran 2:153" if compare else "Explain Quran 2:153 using tafsir"
            ),
            locale="ar",
        )
    )
    assert response.status is AnswerStatus.ANSWERABLE
    assert primary.calls == 1 and local.calls == 0
    assert trace.chat_route["selection_stage"] is None
    assert trace.evidence.permitted_synthesis_deployments == {
        "groq:openai/gpt-oss-20b:zdr",
        LOCAL_GEMMA_DEPLOYMENT_ID,
    }
    assert all(node.tool_id != "search_quran_topics.v1" for node in trace.execution_plan.nodes)


@pytest.mark.asyncio
async def test_full_service_empty_tafsir_is_insufficient_and_never_synthesizes() -> None:
    service, primary, local = build_tafsir_service(empty=True)
    response, trace = await service.research(
        ResearchRequest(message="Explain Quran 2:153 using tafsir", locale="ar")
    )
    assert response.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert primary.calls == local.calls == 0
    assert trace.evidence.requirements_satisfied is False


@pytest.mark.parametrize("category", ["TIMEOUT", "RATE_LIMITED", "UNAVAILABLE"])
@pytest.mark.asyncio
async def test_full_service_tafsir_provider_failure_is_temporarily_unavailable(
    category: str,
) -> None:
    service, primary, local = build_tafsir_service(failure=category)
    response, trace = await service.research(
        ResearchRequest(message="Explain Quran 2:153 using tafsir", locale="ar")
    )
    assert response.status is AnswerStatus.TEMPORARILY_UNAVAILABLE
    assert primary.calls == local.calls == 0
    assert trace.provider_failure is not None
    assert trace.provider_failure.category == category


def test_tafsir_source_replacement_preserves_public_execution_contract() -> None:
    execution = compile_plan(compare=True)
    serialized = execution.model_dump(mode="json")
    assert serialized["nodes"][1]["tool_id"] == "get_tafsir_passages.v1"
    assert serialized["nodes"][1]["validated_input"]["mode"] == ("COMPARISON_EXACTLY_TWO")
    assert "provider" not in serialized["nodes"][1]["validated_input"]
