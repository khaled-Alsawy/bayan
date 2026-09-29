from __future__ import annotations

from uuid import uuid4

from islamic_research_ai.anti_fabrication import AntiFabricationGate, FabricationRule
from islamic_research_ai.contracts import (
    AttributionRole,
    AttributionV1,
    EvidenceItem,
    EvidenceRequirementsV1,
    EvidenceSet,
    EvidenceSourceKind,
    QuranPointer,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SourceIdentityV1,
    SourceRevisionIdentity,
    TafsirPointer,
    TypedSourceReference,
    VerseKey,
)

GATE = AntiFabricationGate()


def _quran_evidence(
    chapter: int, verse: int, arabic_text: str, evidence_id: str = "E1"
) -> EvidenceItem:
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
        exact_text=arabic_text,
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


def _tafsir_evidence(
    chapter: int,
    verse: int,
    text: str,
    *,
    author_display_name: str = "Ibn Kathir",
    evidence_id: str = "E2",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_kind=EvidenceSourceKind.TAFSIR,
        canonical_pointer=TafsirPointer(
            kind="TAFSIR", passage_id="p1", verse_keys=[VerseKey(chapter=chapter, verse=verse)]
        ),
        source_identity=SourceIdentityV1(
            source_id="qul",
            resource_alias="qul-tafsir-ibnkathir-ar-v1",
            work_title="Tafsir Ibn Kathir",
            author_translator_or_grader=[author_display_name],
            canonical_base_url=None,
        ),
        source_revision=SourceRevisionIdentity(
            source_id="qul",
            resource_alias="qul-tafsir-ibnkathir-ar-v1",
            revision_id="r2",
            checksum="c2",
        ),
        language="ar",
        exact_text=text,
        approved_quote_spans=[],
        reference=TypedSourceReference(
            scheme="tafsir", value=f"qul@{chapter}:{verse}", canonical_link=None
        ),
        attributions=[
            AttributionV1(
                role=AttributionRole.AUTHOR, display_name=author_display_name, required_notice=None
            )
        ],
        retrieval=RetrievalProvenanceV1(
            tool_id="get_tafsir_passages.v1",
            method=RetrievalMethod.EXACT,
            index_revision=None,
            rank=1,
            component_scores={},
        ),
        content_hash="sha256:y",
        source_policy_alias="qul-tafsir-ibnkathir-ar-v1",
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
        permitted_synthesis_deployments={"groq:openai/gpt-oss-20b:zdr"},
    )


# --- QUOTE_GUARD -----------------------------------------------------------------------------


def test_quote_guard_accepts_a_span_present_in_evidence() -> None:
    evidence = _evidence_set(_quran_evidence(113, 5, "ومن شر حاسد إذا حسد"))
    text = 'يقول القرآن في الآية: "ومن شر حاسد إذا حسد" وهذا واضح.'
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.QUOTE_GUARD]


def test_quote_guard_rejects_a_fabricated_arabic_quotation() -> None:
    evidence = _evidence_set(_quran_evidence(113, 5, "ومن شر حاسد إذا حسد"))
    # Three-plus Arabic words in quotes that were never retrieved as evidence.
    text = 'يقول القرآن: "واعتصموا بحبل الله جميعا ولا تفرقوا" في هذا السياق.'
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.QUOTE_GUARD]
    assert len(matches) == 1
    assert "واعتصموا" in matches[0].offending_span


def test_quote_guard_ignores_short_parenthetical_citations() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    text = "الآية تتحدث عن الابتلاء (2:155) بوضوح."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.QUOTE_GUARD]


# --- REFERENCE_GUARD -------------------------------------------------------------------------


def test_reference_guard_accepts_a_verse_key_present_in_evidence() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "text"))
    text = "هذا مذكور في الآية 2:155 بوضوح."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.REFERENCE_GUARD]


def test_reference_guard_rejects_a_chapter_verse_pair_absent_from_evidence() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "text"))
    # The reported real-world failure case: citing 68:51 when only 2:155 was retrieved.
    text = "وهذا يذكرنا بقوله تعالى في سورة القلم 68:51 عن العين."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.REFERENCE_GUARD]
    assert any("68:51" in v.offending_span for v in matches)


def test_reference_guard_rejects_a_named_surah_with_no_retrieved_evidence() -> None:
    evidence = _evidence_set(_quran_evidence(113, 5, "text"))
    text = "وهذا يشبه ما ورد في سورة القلم عن هذا الموضوع."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.REFERENCE_GUARD]
    assert any("سورة القلم" in v.offending_span for v in matches)


def test_reference_guard_accepts_a_named_surah_with_retrieved_evidence() -> None:
    evidence = _evidence_set(_quran_evidence(113, 5, "text"))
    text = "هذا مذكور في سورة الفلق بوضوح."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.REFERENCE_GUARD]


def test_reference_guard_ignores_a_number_pair_outside_valid_chapter_range() -> None:
    # 150:3 cannot be a verse reference (the Quran has 114 surahs) -- clearly out of range,
    # unlike a small pair such as "5:3" which is *also* a real, famous verse reference (Al-
    # Ma'idah 5:3) and is therefore deliberately still flagged when unsupported: a fabrication
    # gate is conservative by design -- a false rejection costs a repair attempt or a
    # claims-only degrade, never fabricated text.
    evidence = _evidence_set(_quran_evidence(2, 155, "text"))
    text = "النتيجة كانت 150:3 في ذلك اليوم، وهذا لا علاقة له بالقرآن."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.REFERENCE_GUARD]


def test_reference_guard_is_conservative_about_small_ambiguous_number_pairs() -> None:
    # Documents the trade-off above: an in-range, unsupported pair is flagged even though it
    # could, in isolated text, be read as an ordinary ratio or score.
    evidence = _evidence_set(_quran_evidence(2, 155, "text"))
    text = "النسبة كانت 5:3 في تلك المباراة."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert any(v.rule is FabricationRule.REFERENCE_GUARD for v in violations)


# --- ATTRIBUTION_GUARD -----------------------------------------------------------------------


def test_attribution_guard_accepts_a_scholar_present_in_evidence_attributions() -> None:
    evidence = _evidence_set(
        _tafsir_evidence(2, 155, "تفسير الآية", author_display_name="Ibn Kathir")
    )
    text = "كما ذكر ابن كثير في تفسيره لهذه الآية."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.ATTRIBUTION_GUARD]


def test_attribution_guard_rejects_a_scholar_absent_from_evidence() -> None:
    # Evidence only carries Ibn Kathir; narrative claims "some scholars mentioned" via a named
    # exegete (Al-Qurtubi) that was never retrieved.
    evidence = _evidence_set(
        _tafsir_evidence(2, 155, "تفسير الآية", author_display_name="Ibn Kathir")
    )
    text = "وقد ذكر القرطبي في تفسيره رأيا مختلفا حول هذه الآية."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.ATTRIBUTION_GUARD]
    assert len(matches) == 1
    assert matches[0].offending_span == "القرطبي"


# --- HADITH_GUARD -----------------------------------------------------------------------------


def test_hadith_guard_accepts_a_hadith_verbatim_inside_retrieved_tafsir() -> None:
    tafsir_text = "وقال رسول الله صلى الله عليه وسلم: من صبر ظفر. رواه أحمد في مسنده."
    evidence = _evidence_set(_tafsir_evidence(2, 155, tafsir_text))
    text = f"وفي هذا السياق، {tafsir_text}"
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert not [v for v in violations if v.rule is FabricationRule.HADITH_GUARD]


def test_hadith_guard_rejects_a_free_standing_fabricated_hadith() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"))
    text = "وقال رسول الله صلى الله عليه وسلم: من صبر على البلاء نال الأجر العظيم يوم القيامة."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.HADITH_GUARD]
    assert len(matches) == 1


def test_hadith_guard_rejects_english_narrated_phrasing_absent_from_tafsir() -> None:
    evidence = _evidence_set(_quran_evidence(2, 155, "text"))
    text = "It is narrated that the Prophet said patience is half of faith."
    violations = GATE.check(narrative_text=text, evidence=evidence)
    matches = [v for v in violations if v.rule is FabricationRule.HADITH_GUARD]
    assert len(matches) >= 1


# --- Legitimate composed narrative passes clean -----------------------------------------------


def test_a_faithful_narrative_produces_no_violations() -> None:
    evidence = _evidence_set(
        _quran_evidence(2, 155, "ولنبلونكم بشيء من الخوف والجوع"),
        _tafsir_evidence(
            2, 155, "الصبر هو الرضا بقضاء الله دون تسخط.", author_display_name="Ibn Kathir"
        ),
    )
    text = (
        "يوضح القرآن في الآية 2:155 أن الابتلاء سنة إلهية. "
        "وكما ذكر ابن كثير، فإن الصبر المقصود هو الرضا بقضاء الله دون تسخط."
    )
    violations = GATE.check(narrative_text=text, evidence=evidence)
    assert violations == []
