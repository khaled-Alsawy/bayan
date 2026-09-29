from __future__ import annotations

from uuid import uuid4

import pytest

from islamic_research_ai.anti_fabrication import AntiFabricationGate
from islamic_research_ai.composition_layer import AnswerComposer
from islamic_research_ai.contracts import (
    AnswerStatus,
    AttributionRole,
    AttributionV1,
    CitationV1,
    EffectiveDeploymentIdentityV1,
    EvidenceItem,
    EvidenceRequirementsV1,
    EvidenceSet,
    EvidenceSourceKind,
    NarrativeDraftV1,
    NarrativeSegmentDraftV1,
    NarrativeSegmentKind,
    ProviderCompletion,
    QuranPointer,
    RenderedClaimBlockV1,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SourceIdentityV1,
    SourceRevisionIdentity,
    TypedSourceReference,
    VerseKey,
)
from islamic_research_ai.routing import GROQ_CHAT_DEPLOYMENT_ID


def _groq_completion() -> ProviderCompletion:
    return ProviderCompletion(
        effective_deployment=EffectiveDeploymentIdentityV1(
            deployment_id=GROQ_CHAT_DEPLOYMENT_ID,
            gateway_provider_id="groq",
            effective_provider_id="groq",
            model_id="openai/gpt-oss-20b",
            resolved_model_version_or_name="openai/gpt-oss-20b",
            region=None,
            subprocessor_ids=set(),
            zero_data_retention_enforced=True,
            provider_training_allowed=False,
            route_policy_id="groq-direct-free-zdr-v1",
        ),
        request_id="r1",
        input_tokens=10,
        output_tokens=10,
        latency_ms=5,
        billed_cost_usd=None,
        paid_route=False,
        fallback_used=False,
    )


def _quran_evidence(chapter: int, verse: int, text: str, evidence_id: str = "E1") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_kind=EvidenceSourceKind.QURAN,
        canonical_pointer=QuranPointer(
            kind="QURAN", verse_keys=[VerseKey(chapter=chapter, verse=verse)]
        ),
        source_identity=SourceIdentityV1(
            source_id="tanzil",
            resource_alias="tanzil-uthmani-v1.1",
            work_title="The Quran",
            author_translator_or_grader=["Tanzil Project"],
            canonical_base_url=None,
        ),
        source_revision=SourceRevisionIdentity(
            source_id="tanzil",
            resource_alias="tanzil-uthmani-v1.1",
            revision_id="r1",
            checksum="c1",
        ),
        language="ar",
        exact_text=text,
        approved_quote_spans=[],
        reference=TypedSourceReference(
            scheme="quran", value=f"{chapter}:{verse}", canonical_link=None
        ),
        attributions=[
            AttributionV1(
                role=AttributionRole.SOURCE, display_name="Tanzil Project", required_notice=None
            )
        ],
        retrieval=RetrievalProvenanceV1(
            tool_id="get_quran_passage.v1",
            method=RetrievalMethod.EXACT,
            index_revision=None,
            rank=1,
            component_scores={},
        ),
        content_hash="sha256:x",
        source_policy_alias="tanzil-uthmani-v1.1",
    )


def _evidence_set(*items: EvidenceItem) -> EvidenceSet:
    return EvidenceSet(
        schema_version="1",
        turn_id=uuid4(),
        items_by_id={item.evidence_id: item for item in items},
        requirements=EvidenceRequirementsV1(
            required_source_kinds=set(),
            minimum_items_by_kind={},
            require_exact_quran=False,
            require_exact_hadith=False,
            required_tafsir_aliases=[],
            require_two_distinct_tafsir_resources=False,
            allow_partial=True,
        ),
        requirements_satisfied=True,
        missing_requirements=[],
        permitted_synthesis_deployments={GROQ_CHAT_DEPLOYMENT_ID},
    )


def _claim(claim_id: str, text: str, evidence_id: str) -> RenderedClaimBlockV1:
    return RenderedClaimBlockV1(
        block_type="CLAIM",
        claim_id=claim_id,
        text=text,
        citations=[
            CitationV1(
                citation_id="C1",
                evidence_id=evidence_id,
                display_label=f"Quran {evidence_id}",
                source_kind=EvidenceSourceKind.QURAN,
                reference=TypedSourceReference(scheme="quran", value="2:155", canonical_link=None),
                attribution=[
                    AttributionV1(
                        role=AttributionRole.SOURCE,
                        display_name="Tanzil Project",
                        required_notice=None,
                    )
                ],
                canonical_link=None,
                source_revision=SourceRevisionIdentity(
                    source_id="tanzil",
                    resource_alias="tanzil-uthmani-v1.1",
                    revision_id="r1",
                    checksum="c1",
                ),
            )
        ],
    )


class FakeChatClient:
    """Returns pre-scripted (draft, completion) pairs in order, one per `complete()` call."""

    def __init__(self, drafts: list[NarrativeDraftV1]) -> None:
        self._drafts = list(drafts)
        self.calls: list[str] = []

    async def complete(
        self, *, response_model, schema_name, system_prompt, user_prompt, max_tokens=1800
    ):
        self.calls.append(user_prompt)
        draft = self._drafts.pop(0)
        return draft, _groq_completion()


def _draft(text: str, supporting_claim_ids: list[str] | None = None) -> NarrativeDraftV1:
    return NarrativeDraftV1(
        schema_version="1",
        segments=[
            NarrativeSegmentDraftV1(
                text=text,
                kind=NarrativeSegmentKind.POINT,
                supporting_claim_ids=supporting_claim_ids or [],
            )
        ],
    )


@pytest.mark.asyncio
async def test_composer_returns_none_when_there_are_no_claims() -> None:
    composer = AnswerComposer(FakeChatClient([]), AntiFabricationGate())
    narrative, completions, _ = await composer.compose(
        question="q",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=[],
        evidence=_evidence_set(),
    )
    assert narrative is None
    assert completions == []


@pytest.mark.asyncio
async def test_composer_produces_a_clean_narrative_and_derives_evidence_ids() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "الابتلاء سنة إلهية.", "E1")]
    client = FakeChatClient([_draft("يوضح القرآن أن الابتلاء سنة إلهية.", ["claim_1"])])
    composer = AnswerComposer(client, AntiFabricationGate())

    narrative, completions, _ = await composer.compose(
        question="ما موقف القرآن من الابتلاء؟",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )

    assert narrative is not None
    assert len(completions) == 1
    assert len(client.calls) == 1
    assert narrative.segments[0].segment_id == "N1"
    assert narrative.segments[0].supporting_claim_ids == ["claim_1"]
    assert narrative.segments[0].supporting_evidence_ids == ["E1"]


@pytest.mark.asyncio
async def test_composer_drops_a_supporting_claim_id_the_model_invented() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "الابتلاء سنة إلهية.", "E1")]
    # The model cites a claim id that does not exist in this turn.
    client = FakeChatClient([_draft("يوضح القرآن ذلك.", ["claim_1", "claim_99"])])
    composer = AnswerComposer(client, AntiFabricationGate())

    narrative, _, _ = await composer.compose(
        question="q",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )
    assert narrative is not None
    assert narrative.segments[0].supporting_claim_ids == ["claim_1"]


@pytest.mark.asyncio
async def test_composer_repairs_once_after_a_gate_failure_and_succeeds() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "الابتلاء سنة إلهية.", "E1")]
    # First attempt cites a chapter:verse absent from evidence; second (repaired) attempt is clean.
    client = FakeChatClient(
        [
            _draft("وهذا مذكور أيضا في 68:51 عن هذا الموضوع.", ["claim_1"]),
            _draft("يوضح القرآن أن الابتلاء سنة إلهية.", ["claim_1"]),
        ]
    )
    composer = AnswerComposer(client, AntiFabricationGate())

    narrative, completions, _ = await composer.compose(
        question="q",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )

    assert narrative is not None
    assert len(completions) == 2
    assert len(client.calls) == 2
    # The repair prompt names the offending rule and span.
    assert "REFERENCE_GUARD" in client.calls[1]
    assert "68:51" in client.calls[1]


@pytest.mark.asyncio
async def test_composer_degrades_to_claims_only_after_a_second_gate_failure() -> None:
    """The golden double-failure test (Part 3b / WP-2 acceptance criterion): when the
    composer cannot produce a clean narrative even after one repair, the caller must render
    claims without a narrative -- never a fabrication."""

    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "الابتلاء سنة إلهية.", "E1")]
    # Both attempts cite a chapter:verse absent from evidence.
    client = FakeChatClient(
        [
            _draft("وهذا مذكور أيضا في 68:51 عن هذا الموضوع.", ["claim_1"]),
            _draft("وهذا مذكور أيضا في سورة القلم عن هذا الموضوع.", ["claim_1"]),
        ]
    )
    composer = AnswerComposer(client, AntiFabricationGate())

    narrative, completions, _ = await composer.compose(
        question="q",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )

    assert narrative is None
    assert len(completions) == 2
    assert len(client.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsafe_text", "rule"),
    [
        ("«إن الله يحب الذين يصبرون في كل وقت»", "QUOTE_GUARD"),
        ("The relevant verse is 68:51.", "REFERENCE_GUARD"),
        ("The relevant verse is 2:300.", "REFERENCE_GUARD"),
        ("Ibn Kathir says this is a promise for every case.", "ATTRIBUTION_GUARD"),
        ("The Prophet said: every hardship is immediately removed.", "HADITH_GUARD"),
    ],
)
async def test_composer_rejects_each_anti_fabrication_rule_and_records_it(
    unsafe_text: str, rule: str
) -> None:
    """Part C: hostile/stubbed composer output cannot become reader-facing prose."""

    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "Hardship is addressed by the retrieved verse.", "E1")]
    composer = AnswerComposer(
        FakeChatClient([_draft(unsafe_text, ["claim_1"]), _draft(unsafe_text, ["claim_1"])]),
        AntiFabricationGate(),
    )

    narrative, completions, rejections = await composer.compose(
        question="q",
        response_language="en",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )

    assert narrative is None
    assert len(completions) == 2
    assert rejections[rule] >= 2


@pytest.mark.asyncio
async def test_composer_cannot_emit_canonical_quran_text_without_retrieved_evidence() -> None:
    """Invariant 1: model prose is rejected; canonical text is renderer-only evidence output."""

    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    claims = [_claim("claim_1", "Hardship is addressed by the retrieved verse.", "E1")]
    invented_canonical_text = "«قل هو الله أحد الله الصمد»"
    composer = AnswerComposer(
        FakeChatClient(
            [
                _draft(invented_canonical_text, ["claim_1"]),
                _draft(invented_canonical_text, ["claim_1"]),
            ]
        ),
        AntiFabricationGate(),
    )

    narrative, _, rejections = await composer.compose(
        question="q",
        response_language="ar",
        coverage_status=AnswerStatus.ANSWERABLE,
        claims=claims,
        evidence=evidence,
    )

    assert narrative is None
    assert rejections == {"QUOTE_GUARD": 2}
