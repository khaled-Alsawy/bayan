from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from islamic_research_ai.contracts import (
    AnswerDraftV1,
    AnswerStatus,
    CitationV1,
    ClaimBlockV1,
    ClaimType,
    ClarificationStateV1,
    EvidenceItem,
    EvidenceRequirementsV1,
    EvidenceSet,
    EvidenceSourceKind,
    NoticeBlockV1,
    NoticeCode,
    ProviderCompletion,
    QuoteMode,
    QuoteReferenceV1,
    RenderedClaimBlockV1,
    RenderedNoticeBlockV1,
    RenderedSourceBlockV1,
    RenderedSourceViewHeadingBlockV1,
    ResearchResponseV1,
    SourceBlockRequestV1,
    SourceViewHeadingRequestV1,
    SourceVisibility,
    ToolWarning,
    derive_rendered_output_kind,
)
from islamic_research_ai.ports import (
    ChatModelPort,
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.retrieval.relevance import evidence_passage_key
from islamic_research_ai.routing import (
    GROQ_CHAT_DEPLOYMENT_ID,
    ProviderRouteError,
    ProviderRoutePolicy,
)
from islamic_research_ai.tracing import _TRACE_RECORDER, rejected_synthesis_detail


class GroundingError(ValueError):
    pass


class CanonicalTextGroundingError(GroundingError):
    """The model kept reproducing canonical source wording after bounded repair."""


class SynthesisTemporarilyUnavailable(RuntimeError):
    def __init__(
        self,
        diagnostic: ProviderFailureDiagnostic,
        completions: list[ProviderCompletion],
    ) -> None:
        self.diagnostic = diagnostic
        self.completions = completions
        super().__init__(
            f"Synthesis required provider temporarily unavailable: {diagnostic.safe_detail}"
        )


def _answer_draft_block_limit() -> int:
    """Read the answer-size constraint from the canonical contract, not a duplicate literal."""

    max_lengths = [
        constraint.max_length
        for constraint in AnswerDraftV1.model_fields["blocks"].metadata
        if hasattr(constraint, "max_length")
    ]
    if len(max_lengths) != 1 or not isinstance(max_lengths[0], int):
        raise RuntimeError("AnswerDraftV1.blocks must expose exactly one max_length constraint")
    return max_lengths[0]


def _bounded_source_ids(
    evidence_ids: list[str],
    *,
    claim_block_count: int,
    cited_evidence_ids: Sequence[str] = (),
) -> list[str]:
    """Leave room for claim blocks while preserving the caller's ranked ID ordering.

    Evidence a claim cites is retained first. Truncating it strands the citation with no source
    block to render, so the answer would cite evidence it never shows and the reader would follow
    a reference to nothing.
    """

    limit = max(0, _answer_draft_block_limit() - claim_block_count)
    ordered = list(dict.fromkeys([*evidence_ids, *cited_evidence_ids]))
    if len(ordered) <= limit:
        return ordered
    cited = set(cited_evidence_ids)
    retained = [evidence_id for evidence_id in ordered if evidence_id in cited][:limit]
    for evidence_id in ordered:
        if len(retained) >= limit:
            break
        if evidence_id not in cited:
            retained.append(evidence_id)
    kept = set(retained)
    return [evidence_id for evidence_id in ordered if evidence_id in kept]


def _bounded_passage_source_ids(
    evidence: EvidenceSet,
    *,
    claim_block_count: int,
    evidence_ids: list[str] | None = None,
) -> list[str]:
    """Fit whole Quran Arabic/translation passage groups into the contract budget.

    A source request is not independent when canonical Quran and translation items share a
    passage pointer: the grounding contract requires both.  Preserve ranked group ordering but
    never use a final slot for only one half of that pair.
    """

    ordered_ids = evidence_ids or _ranked_evidence_ids(evidence)
    remaining = max(0, _answer_draft_block_limit() - claim_block_count)
    pointer_members: dict[str, dict[EvidenceSourceKind, list[str]]] = {}
    for evidence_id in ordered_ids:
        item = evidence.items_by_id[evidence_id]
        if item.source_kind in {EvidenceSourceKind.QURAN, EvidenceSourceKind.QURAN_TRANSLATION}:
            pointer = json.dumps(item.canonical_pointer.model_dump(mode="json"), sort_keys=True)
            pointer_members.setdefault(pointer, {}).setdefault(item.source_kind, []).append(
                evidence_id
            )
    paired_group_for_id: dict[str, str] = {}
    for pointer, members in pointer_members.items():
        arabic_ids = members.get(EvidenceSourceKind.QURAN, [])
        translation_ids = members.get(EvidenceSourceKind.QURAN_TRANSLATION, [])
        for pair_index, (arabic_id, translation_id) in enumerate(
            zip(arabic_ids, translation_ids, strict=False)
        ):
            group_key = f"pair:{pointer}:{pair_index}"
            paired_group_for_id[arabic_id] = group_key
            paired_group_for_id[translation_id] = group_key
    groups: list[list[str]] = []
    group_positions: dict[str, int] = {}
    for evidence_id in ordered_ids:
        item = evidence.items_by_id[evidence_id]
        if item.source_kind in {EvidenceSourceKind.QURAN, EvidenceSourceKind.QURAN_TRANSLATION}:
            key = paired_group_for_id.get(evidence_id, f"single:{evidence_id}")
        else:
            key = f"single:{evidence_id}"
        position = group_positions.get(key)
        if position is None:
            group_positions[key] = len(groups)
            if key.startswith("pair:"):
                groups.append(
                    [
                        item_id
                        for item_id in ordered_ids
                        if paired_group_for_id.get(item_id) == key
                    ]
                )
            else:
                groups.append([evidence_id])
        elif not key.startswith("pair:"):
            groups[position].append(evidence_id)

    selected: list[str] = []
    for group in groups:
        if len(group) > remaining:
            continue
        selected.extend(group)
        remaining -= len(group)
    return selected


def _ranked_evidence_ids(evidence: EvidenceSet) -> list[str]:
    """Return distinct evidence IDs by retrieval rank, retaining input order for rank ties."""

    return [
        evidence_id
        for _, evidence_id in sorted(
            enumerate(evidence.items_by_id),
            key=lambda value: (
                evidence.items_by_id[value[1]].retrieval.rank is None,
                evidence.items_by_id[value[1]].retrieval.rank or 0,
                value[0],
            ),
        )
    ]


def _complete_quran_translation_source_requests(
    draft: AnswerDraftV1, evidence: EvidenceSet
) -> tuple[AnswerDraftV1, list[str]]:
    """Add every available Arabic/translation sibling for requested Quran sources."""

    by_id = evidence.items_by_id
    requested_ids = {
        block.quote_reference.evidence_id
        for block in draft.blocks
        if isinstance(block, SourceBlockRequestV1)
    }
    requested = set(requested_ids)
    existing_block_ids = {
        block.block_id for block in draft.blocks if isinstance(block, SourceBlockRequestV1)
    }
    completed_blocks = []
    added_ids: list[str] = []
    quran_kinds = {EvidenceSourceKind.QURAN, EvidenceSourceKind.QURAN_TRANSLATION}

    for block in draft.blocks:
        completed_blocks.append(block)
        if not isinstance(block, SourceBlockRequestV1):
            continue
        item = by_id.get(block.quote_reference.evidence_id)
        if item is None or item.source_kind not in quran_kinds:
            continue
        pointer = json.dumps(item.canonical_pointer.model_dump(mode="json"), sort_keys=True)
        for evidence_id, paired_item in by_id.items():
            if (
                evidence_id in requested
                or paired_item.source_kind not in quran_kinds
                or json.dumps(paired_item.canonical_pointer.model_dump(mode="json"), sort_keys=True)
                != pointer
            ):
                continue
            block_id = f"PAIR-COMPLETION-{evidence_id}"
            suffix = 2
            while block_id in existing_block_ids:
                block_id = f"PAIR-COMPLETION-{evidence_id}-{suffix}"
                suffix += 1
            existing_block_ids.add(block_id)
            requested.add(evidence_id)
            added_ids.append(evidence_id)
            completed_blocks.append(
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=block_id,
                    quote_reference=QuoteReferenceV1(
                        evidence_id=evidence_id,
                        approved_span_id=None,
                    ),
                    quote_mode=QuoteMode.FULL,
                    visibility=block.visibility,
                    passage_group_id=block.passage_group_id,
                )
            )

    if not added_ids:
        return draft, []
    return draft.model_copy(update={"blocks": completed_blocks}), added_ids


class QuranClaimProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    claim_type: ClaimType
    evidence_ids: list[str]


class QuranAnswerProposalV1(BaseModel):
    """Compact provider transport converted into canonical atomic answer blocks."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    status: Literal["ANSWERABLE", "INSUFFICIENT_EVIDENCE"]
    claims: list[QuranClaimProposalV1] = Field(max_length=_answer_draft_block_limit())
    source_evidence_ids: list[str]

    def to_canonical(self) -> AnswerDraftV1:
        if self.status == "INSUFFICIENT_EVIDENCE":
            if self.claims or self.source_evidence_ids:
                raise GroundingError("Insufficient proposal must not contain answer content")
            return AnswerDraftV1(
                schema_version="1",
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                blocks=[],
            )
        blocks: list[ClaimBlockV1 | SourceBlockRequestV1] = [
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id=claim.claim_id,
                text=claim.text,
                claim_type=claim.claim_type,
                evidence_ids=claim.evidence_ids,
            )
            for claim in self.claims
        ]
        blocks.extend(
            SourceBlockRequestV1(
                block_type="SOURCE_BLOCK_REQUEST",
                block_id=f"SOURCE-{index}",
                quote_reference=QuoteReferenceV1(
                    evidence_id=evidence_id,
                    approved_span_id=None,
                ),
                quote_mode=QuoteMode.FULL,
            )
            for index, evidence_id in enumerate(
                _bounded_source_ids(
                    self.source_evidence_ids,
                    claim_block_count=len(blocks),
                    cited_evidence_ids=[
                        evidence_id for claim in self.claims for evidence_id in claim.evidence_ids
                    ],
                ),
                start=1,
            )
        )
        return AnswerDraftV1(
            schema_version="1",
            status=AnswerStatus.ANSWERABLE,
            blocks=blocks,
        )


class TafsirClaimProposalV1(BaseModel):
    """Minimal local-only transport; source blocks remain application-owned."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class TafsirAnswerProposalV1(BaseModel):
    """Compact schema used when STORE=DENIED Tafsir evidence forces local synthesis."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    status: Literal["ANSWERABLE", "INSUFFICIENT_EVIDENCE"]
    claims: list[TafsirClaimProposalV1] = Field(max_length=2)

    def to_canonical(
        self,
        *,
        source_evidence_ids: list[str],
        comparison: bool,
    ) -> AnswerDraftV1:
        if self.status == "INSUFFICIENT_EVIDENCE":
            if self.claims:
                raise GroundingError("Insufficient proposal must not contain answer content")
            return AnswerDraftV1(
                schema_version="1",
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                blocks=[],
            )
        blocks: list[ClaimBlockV1 | SourceBlockRequestV1] = [
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id=claim.claim_id,
                text=claim.text,
                claim_type=(ClaimType.COMPARISON if comparison else ClaimType.FACTUAL_SUMMARY),
                evidence_ids=claim.evidence_ids,
            )
            for claim in self.claims
        ]
        blocks.extend(
            SourceBlockRequestV1(
                block_type="SOURCE_BLOCK_REQUEST",
                block_id=f"SOURCE-{index}",
                quote_reference=QuoteReferenceV1(
                    evidence_id=evidence_id,
                    approved_span_id=None,
                ),
                quote_mode=QuoteMode.FULL,
            )
            for index, evidence_id in enumerate(
                _bounded_source_ids(
                    source_evidence_ids,
                    claim_block_count=len(blocks),
                    cited_evidence_ids=[
                        evidence_id for claim in self.claims for evidence_id in claim.evidence_ids
                    ],
                ),
                start=1,
            )
        )
        return AnswerDraftV1(
            schema_version="1",
            status=AnswerStatus.ANSWERABLE,
            blocks=blocks,
        )


def deterministic_tafsir_source_draft(
    evidence: EvidenceSet, *, response_language: str
) -> AnswerDraftV1:
    """Render verified Tafsir even when the prose model exhausts its bounded repair.

    This is intentionally *not* an interpretation: it makes the exact retrieved
    source available and tells the reader why there is no generated summary.  It
    prevents a large, legitimate Tafsir passage from disappearing merely because
    a model returned an invalid citation array, while preserving ExactRenderer as
    the only producer of every canonical source character.
    """

    source_ids = _ranked_evidence_ids(evidence)
    tafsir_ids = [
        evidence_id
        for evidence_id in source_ids
        if evidence.items_by_id[evidence_id].source_kind is EvidenceSourceKind.TAFSIR
    ]
    if not tafsir_ids:
        raise GroundingError("Tafsir source-only fallback requires Tafsir evidence")
    language = response_language.strip().casefold().split("-", maxsplit=1)[0]
    text = (
        "يُعرض نص التفسير المسترجع أدناه دون تلخيص مُولَّد، لأن صياغة القراءة لم تجتز "
        "التحقّق الآلي في هذه المحاولة."
        if language == "ar"
        else "The retrieved Tafsir is shown below without a generated summary because the "
        "reading draft did not pass automated verification in this attempt."
    )
    visible_source_ids = _bounded_source_ids(source_ids, claim_block_count=1)
    if not any(evidence_id in tafsir_ids for evidence_id in visible_source_ids):
        # Keep the highest-ranked Tafsir item visible even if a large Quran context bundle
        # occupies the initial answer budget. The claim's evidence and source blocks stay aligned.
        visible_source_ids[-1] = tafsir_ids[0]
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="tafsir-source-only",
                text=text,
                claim_type=ClaimType.FACTUAL_SUMMARY,
                # The claim is strictly about how this answer is presented. It
                # cites every Tafsir resource so the existing requirement that
                # each requested work support a visible claim remains enforced.
                evidence_ids=[
                    evidence_id for evidence_id in visible_source_ids if evidence_id in tafsir_ids
                ],
            ),
            *[
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"tafsir-source-only-{index}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=evidence_id, approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                )
                for index, evidence_id in enumerate(visible_source_ids, start=1)
            ],
        ],
    )


class GroundingValidator:
    _ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
    _LATIN_SCRIPT = re.compile(r"[A-Za-z]")
    _INTERNAL_EVIDENCE_ID = re.compile(r"(?<![\w-])E\d+(?:\s*[,/&+]\s*E\d+)*(?![\w-])", re.I)
    _INTERNAL_SOURCE_TAG = re.compile(r"\b(?:QURAN(?:_TRANSLATION)?|TAFSIR):\s*\d+(?::\d+)?", re.I)
    _PUBLIC_QURAN_REFERENCE = re.compile(
        r"(?<!\d)\d{1,3}:\d{1,3}(?:\s*[-–—]\s*\d{1,3}:\d{1,3})?(?!\d)"
    )
    _HUMANITY_EVIDENCE_TERMS = (
        "humanity",
        "mankind",
        "people",
        "البشريه",
        "الانسانيه",
        "الناس",
    )
    _ABSOLUTE_QUANTIFIER_PATTERN = re.compile(
        r"(?i:\b(?:every|always|never|all)\b)(?!-[A-Z][a-z]+)"
    )
    _RISK_CONCEPTS = (
        (
            re.compile(r"\b(?:believers?|faithful)\b", re.I),
            ("believer", "believers", "believe", "مؤمن", "امنوا"),
        ),
        (
            re.compile(r"(?:المؤمن(?:ون|ين)?|الذين امنوا)"),
            ("مؤمن", "امنوا", "believer", "believers", "believe"),
        ),
        (
            re.compile(r"\b(?:humanity|mankind|all people)\b", re.I),
            _HUMANITY_EVIDENCE_TERMS,
        ),
        (re.compile(r"(?:البشريه|الانسانيه|جميع الناس)"), ("البشريه", "الانسانيه", "الناس")),
        (
            re.compile(r"\b(?:duty|obligatory|mandatory|legal ruling|halal|haram)\b", re.I),
            ("duty", "obligatory", "mandatory", "legal", "halal", "haram"),
        ),
        (
            re.compile(r"(?:واجب|فرض|ملزم|حلال|حرام|حكم شرعي)"),
            ("واجب", "فرض", "ملزم", "حلال", "حرام", "حكم شرعي"),
        ),
        (re.compile(r"\bmercy\b", re.I), ("mercy",)),
        (re.compile(r"(?:رحمه|الرحمه)"), ("رحمه",)),
        (
            re.compile(r"\b(?:sustenance|protection|refuge|seek(?:ing)? refuge)\b", re.I),
            ("sustenance", "protection", "refuge", "seek refuge"),
        ),
        (
            re.compile(r"(?:رزق|حمايه|استعاذه|يلتجئ|لجوء)"),
            ("رزق", "حمايه", "استعاذه", "يلتجئ", "لجوء", "اعوذ", "عوذ"),
        ),
        (
            re.compile(r"\b(?:fundamental attribute|divine nature|his nature)\b", re.I),
            ("fundamental attribute", "divine nature", "his nature"),
        ),
        (re.compile(r"(?:صفه جوهريه|طبيعته)"), ("صفه جوهريه", "طبيعته")),
        (
            re.compile(r"\b(?:leads? to|results? in|causes?)\b", re.I),
            ("leads to", "lead to", "results in", "result in", "causes", "cause"),
        ),
        (re.compile(r"(?:يؤدي الى|يسبب|ينتج عنه)"), ("يؤدي الى", "يسبب", "ينتج عنه")),
        (
            re.compile(r"\b(?:asks?|requests?|seeks?)\b", re.I),
            ("ask", "asks", "request", "requests", "seek", "seeks"),
        ),
        (re.compile(r"(?:طلب|يطلب|تطلب)"), ("طلب", "يطلب", "تطلب")),
        (
            re.compile(r"\b(?:one|people|everyone|we)\s+should\b", re.I),
            ("you who believe", "people should", "everyone should", "we should"),
        ),
        (
            _ABSOLUTE_QUANTIFIER_PATTERN,
            ("every", "always", "never", "all"),
        ),
        (re.compile(r"(?:\bكل\b|دائما|ابدا)"), (" كل ", "دائما", "ابدا")),
    )

    def __init__(self, *, risk_concept_human_subject_terms_enabled: bool = False) -> None:
        self.risk_concept_human_subject_terms_enabled = risk_concept_human_subject_terms_enabled

    def validate(
        self,
        draft: AnswerDraftV1,
        evidence: EvidenceSet,
        *,
        response_language: str | None = None,
        check_semantic_entailment: bool = True,
    ) -> None:
        if not evidence.requirements_satisfied or not evidence.items_by_id:
            raise GroundingError("ANSWERABLE synthesis requires satisfied evidence")
        if draft.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            if draft.blocks:
                raise GroundingError(
                    "Insufficiency drafts cannot contain model-authored religious prose"
                )
            return
        if draft.status is not AnswerStatus.ANSWERABLE:
            raise GroundingError("Synthesis must use ANSWERABLE or INSUFFICIENT_EVIDENCE")

        claims = [block for block in draft.blocks if isinstance(block, ClaimBlockV1)]
        requests = [block for block in draft.blocks if isinstance(block, SourceBlockRequestV1)]
        source_view_headings = [
            block for block in draft.blocks if isinstance(block, SourceViewHeadingRequestV1)
        ]
        notices = [block for block in draft.blocks if isinstance(block, NoticeBlockV1)]
        if notices:
            raise GroundingError("Generated notice/wrapper prose is not permitted")
        is_source_view = bool(source_view_headings)
        if is_source_view and (
            len(source_view_headings) != 1
            or claims
            or draft.blocks[0] is not source_view_headings[0]
        ):
            raise GroundingError("Source-only views require one leading application heading")
        if not claims and not is_source_view:
            raise GroundingError("At least one atomic grounded claim is required")
        if not requests:
            raise GroundingError("At least one source block request is required")

        by_id = evidence.items_by_id
        required_tafsir_aliases = set(evidence.requirements.required_tafsir_aliases)
        claim_tafsir_aliases: set[str] = set()
        for claim in claims:
            if any(evidence_id not in by_id for evidence_id in claim.evidence_ids):
                raise GroundingError(f"Claim {claim.claim_id} cites unavailable evidence")
            claim_tafsir_items = [
                by_id[evidence_id]
                for evidence_id in claim.evidence_ids
                if by_id[evidence_id].source_kind is EvidenceSourceKind.TAFSIR
            ]
            if (
                required_tafsir_aliases
                and not evidence.requirements.allow_partial
                and not claim_tafsir_items
            ):
                raise GroundingError(f"Claim {claim.claim_id} must cite exact Tafsir evidence")
            claim_tafsir_aliases.update(item.source_policy_alias for item in claim_tafsir_items)
            if response_language is not None:
                self._validate_claim_language(claim, response_language)
            if check_semantic_entailment:
                self._validate_semantic_entailment(
                    claim,
                    [by_id[evidence_id] for evidence_id in claim.evidence_ids],
                )
            normalized_claim = self._normalize(claim.text)
            for item in by_id.values():
                normalized_exact = self._normalize(item.exact_text)
                if len(normalized_exact) > 24 and normalized_exact in normalized_claim:
                    raise GroundingError("Model attempted to author canonical Quran text")

        requested_ids: list[str] = []
        for request in requests:
            evidence_id = request.quote_reference.evidence_id
            item = by_id.get(evidence_id)
            if item is None:
                raise GroundingError("Source block request cites unavailable evidence")
            if evidence_id in requested_ids:
                raise GroundingError("Duplicate source block requests are not allowed")
            requested_ids.append(evidence_id)
            if request.quote_mode is QuoteMode.FULL:
                if request.quote_reference.approved_span_id is not None:
                    raise GroundingError("FULL quote cannot name an approved span")
            else:
                approved_id = request.quote_reference.approved_span_id
                if approved_id is None or all(
                    span.span_id != approved_id for span in item.approved_quote_spans
                ):
                    raise GroundingError("Requested quote span is not approved")

        requested = {evidence_id for evidence_id in requested_ids}
        requested_items = [by_id[evidence_id] for evidence_id in requested]
        requested_kinds = {item.source_kind for item in requested_items}
        if not evidence.requirements.required_source_kinds.issubset(requested_kinds):
            raise GroundingError("All required source kinds must have deterministic source blocks")
        requested_tafsir_aliases = {
            item.source_policy_alias
            for item in requested_items
            if item.source_kind is EvidenceSourceKind.TAFSIR
        }
        if not is_source_view and not required_tafsir_aliases.issubset(claim_tafsir_aliases):
            raise GroundingError("Every required Tafsir resource must support a claim")
        if not required_tafsir_aliases.issubset(requested_tafsir_aliases):
            raise GroundingError("Every required Tafsir resource must have a source block")
        pointers = {
            json.dumps(by_id[evidence_id].canonical_pointer.model_dump(mode="json"), sort_keys=True)
            for evidence_id in requested
        }
        for pointer in pointers:
            paired_ids = {
                evidence_id
                for evidence_id, item in by_id.items()
                if json.dumps(item.canonical_pointer.model_dump(mode="json"), sort_keys=True)
                == pointer
                and item.source_kind
                in {EvidenceSourceKind.QURAN, EvidenceSourceKind.QURAN_TRANSLATION}
            }
            if paired_ids and not paired_ids.issubset(requested):
                raise GroundingError(
                    "Arabic and translation source blocks must be requested as a pair"
                )

    def drop_semantically_invalid_claims(
        self, draft: AnswerDraftV1, evidence: EvidenceSet
    ) -> tuple[AnswerDraftV1, list[dict[str, str]], GroundingError | None]:
        """Remove only claims rejected by the semantic entailment check."""

        if draft.status is not AnswerStatus.ANSWERABLE:
            return draft, [], None
        by_id = evidence.items_by_id
        retained_blocks = []
        dropped: list[dict[str, str]] = []
        first_error: GroundingError | None = None
        for block in draft.blocks:
            if not isinstance(block, ClaimBlockV1):
                retained_blocks.append(block)
                continue
            try:
                self._validate_semantic_entailment(
                    block,
                    [by_id[evidence_id] for evidence_id in block.evidence_ids],
                )
            except GroundingError as error:
                first_error = first_error or error
                dropped.append({"claim_id": block.claim_id, "reason": str(error)})
                continue
            retained_blocks.append(block)
        if not dropped:
            return draft, [], None
        return draft.model_copy(update={"blocks": retained_blocks}), dropped, first_error

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split()).casefold()

    @staticmethod
    def _semantic_normalize(value: str) -> str:
        """Use one canonical form for Arabic and English risk matching."""

        return normalize_arabic(normalize_translation(value)).replace("ة", "ه")

    def _validate_semantic_entailment(
        self, claim: ClaimBlockV1, cited_evidence: list[EvidenceItem]
    ) -> None:
        """Conservative, deterministic gate over only the claim's cited evidence.

        This does not ask a second model to judge its own prose. It blocks the high-risk
        semantic expansions observed in QA and requires meaningful lexical support before
        model-authored prose may reach the renderer.
        """

        if self._INTERNAL_EVIDENCE_ID.search(claim.text):
            raise GroundingError(f"Claim {claim.claim_id} exposes an internal evidence ID")
        if self._INTERNAL_SOURCE_TAG.search(claim.text):
            raise GroundingError(f"Claim {claim.claim_id} exposes an internal source tag")
        if self._PUBLIC_QURAN_REFERENCE.search(claim.text):
            raise GroundingError(f"Claim {claim.claim_id} duplicates a public Quran reference")
        normalized_claim = f" {self._semantic_normalize(claim.text)} "
        normalized_evidence = " ".join(
            self._semantic_normalize(item.exact_text) for item in cited_evidence
        )
        padded_evidence = f" {normalized_evidence} "
        for claim_pattern, evidence_terms in self._RISK_CONCEPTS:
            if (
                self.risk_concept_human_subject_terms_enabled
                and evidence_terms is self._HUMANITY_EVIDENCE_TERMS
            ):
                evidence_terms = (*evidence_terms, "الانسان", "man")
            claim_for_pattern = (
            claim.text
                if claim_pattern is self._ABSOLUTE_QUANTIFIER_PATTERN
                else normalized_claim
            )
            if claim_pattern.search(claim_for_pattern) and not any(
                self._semantic_normalize(term) in padded_evidence for term in evidence_terms
            ):
                raise GroundingError(
                    f"Claim {claim.claim_id} adds a high-risk concept absent from cited evidence"
                )

    @classmethod
    def _validate_claim_language(cls, claim: ClaimBlockV1, response_language: str) -> None:
        language = response_language.strip().casefold().split("-", maxsplit=1)[0]
        if language == "ar" and cls._ARABIC_SCRIPT.search(claim.text) is None:
            raise GroundingError(f"Claim {claim.claim_id} does not match Arabic response_language")
        if language == "en" and cls._LATIN_SCRIPT.search(claim.text) is None:
            raise GroundingError(f"Claim {claim.claim_id} does not match English response_language")


def sanitize_model_authored_claims(
    draft: AnswerDraftV1,
) -> tuple[AnswerDraftV1, list[dict[str, str]]]:
    """Remove application-rendered labels from model-authored claim text only."""

    sanitized_blocks = []
    changes: list[dict[str, str]] = []
    for block in draft.blocks:
        if not isinstance(block, ClaimBlockV1):
            sanitized_blocks.append(block)
            continue
        sanitized = _sanitize_claim_text(block.text)
        if sanitized == block.text:
            sanitized_blocks.append(block)
            continue
        changes.append(
            {
                "claim_id": block.claim_id,
                "original": block.text,
                "sanitized": sanitized,
            }
        )
        sanitized_blocks.append(block.model_copy(update={"text": sanitized}))
    return draft.model_copy(update={"blocks": sanitized_blocks}), changes


def _sanitize_claim_text(text: str) -> str:
    """Delete validator-defined rendered labels and only their stranded punctuation."""

    spans = sorted(
        (match.start(), match.end())
        for pattern in (
            GroundingValidator._PUBLIC_QURAN_REFERENCE,
            GroundingValidator._INTERNAL_EVIDENCE_ID,
            GroundingValidator._INTERNAL_SOURCE_TAG,
        )
        for match in pattern.finditer(text)
    )
    if not spans:
        return text

    merged_spans: list[tuple[int, int]] = []
    for start, end in spans:
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1] = (merged_spans[-1][0], max(end, merged_spans[-1][1]))
        else:
            merged_spans.append((start, end))

    marker = "\ue000"
    while marker in text:
        marker += "\ue000"
    fragments: list[str] = []
    position = 0
    for start, end in merged_spans:
        fragments.extend((text[position:start], marker))
        position = end
    fragments.append(text[position:])
    sanitized = "".join(fragments)

    escaped_marker = re.escape(marker)
    stranded_separator = r"[\s,;:/&+–—-]*"
    for opening, closing in (("(", ")"), ("[", "]")):
        sanitized = re.sub(
            rf"{re.escape(opening)}{stranded_separator}"
            rf"(?:{escaped_marker}{stranded_separator})+{re.escape(closing)}",
            "",
            sanitized,
        )
    sanitized = sanitized.replace(marker, "")
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([.,;:!،؟])", r"\1", sanitized)
    return sanitized.strip()


def _reject_too_short_sanitized_claims(draft: AnswerDraftV1) -> None:
    for block in draft.blocks:
        if isinstance(block, ClaimBlockV1) and len(block.text.strip()) < 12:
            raise GroundingError(
                f"Claim {block.claim_id} has fewer than 12 characters after sanitization"
            )


def validate_supported_facet_summary(
    *,
    text: str,
    evidence: EvidenceSet,
    canonical_pointer: str,
    response_language: str,
) -> None:
    """Apply the normal grounded-claim gate to a relevance-produced fallback summary."""

    evidence_ids = [
        evidence_id
        for evidence_id in _ranked_evidence_ids(evidence)
        if evidence_passage_key(evidence.items_by_id[evidence_id]) == canonical_pointer
    ]
    if not evidence_ids:
        raise GroundingError("Supported-facet summary has no retained canonical evidence")
    visible_evidence_ids = _bounded_passage_source_ids(
        evidence,
        claim_block_count=1,
        evidence_ids=evidence_ids,
    )
    draft = AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="supported-facet-summary",
                text=text,
                claim_type=ClaimType.FACTUAL_SUMMARY,
                evidence_ids=visible_evidence_ids,
            ),
            *[
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"supported-facet-source-{index}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=evidence_id, approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                )
                for index, evidence_id in enumerate(visible_evidence_ids, start=1)
            ],
        ],
    )
    # A direct-Quran context summary is checked against its own cited evidence.
    # Topic-level Tafsir requirements are enforced later for the completed answer,
    # not incorrectly imposed on every contextual verse in the surrounding passage.
    validation_evidence = evidence
    if not any(
        evidence.items_by_id[evidence_id].source_kind is EvidenceSourceKind.TAFSIR
        for evidence_id in evidence_ids
    ):
        validation_evidence = evidence.model_copy(
            update={
                "requirements": EvidenceRequirementsV1(
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
            }
        )
    GroundingValidator().validate(draft, validation_evidence, response_language=response_language)


class GroqSynthesizer:
    _TAFSIR_CONTEXT_MAX_CHARACTERS = 2_600
    SYSTEM_PROMPT = """You synthesize a cautious answer using only supplied
Quran and Tafsir evidence.
Return only quran_answer_proposal_v1. For ANSWERABLE, write at most three atomic paraphrased
claims; every claim must cite supplied evidence_ids. Never reproduce source text. Select at
most three Quran references in source_evidence_ids and include both the QURAN Arabic and
QURAN_TRANSLATION IDs for every selected reference. If Tafsir is required, every claim must
cite Tafsir evidence and source_evidence_ids must cover every required Tafsir resource.
Do not write wrapper prose, introduce facts from memory, or produce a fatwa.
Never include internal evidence labels such as E1 in public claim text. Do not change the
audience of a passage, add legal classifications, use absolute quantifiers, or introduce a
concept unless that exact concept is present in the evidence cited by that individual claim.
Do not call a verse an explicit protection from the evil eye when it only acknowledges a
blessing or property; label that narrower meaning unless approved Tafsir evidence says more.
Claims must discuss only the relationship between the supplied reference and the requested
concept. When the evidence directly establishes only a narrower facet of a compound or practical
question, return ANSWERABLE with claims limited to that facet; do not discard the evidence merely
because another requested facet is not established. Do not name, imply, or answer the uncovered
facet. Put no Quran wording, quotation, or close recitation in a claim because the application
renders the canonical Arabic and translation in separate source blocks.
Write every model-authored claim in the exact response_language supplied by the application.
Do not translate, rewrite, or otherwise mutate any canonical evidence or source quotation.
Return INSUFFICIENT_EVIDENCE with empty lists only when no supplied evidence directly establishes
any answerable facet of the question."""
    TAFSIR_SYSTEM_PROMPT = """You synthesize a cautious answer using only the supplied
Quran and Tafsir evidence excerpt. Return only tafsir_answer_proposal_v1 JSON. For ANSWERABLE,
write one short atomic claim for a single Tafsir or at most two short atomic claims for a
comparison. Every claim must cite exact supplied evidence_ids and must cite Tafsir evidence unless
the constraints say direct_quran_claims_are_allowed. In that case, direct Quran context may be a
separate claim, but at least one claim must cite Tafsir and explicitly identify it as a Tafsir
interpretation; never present that interpretation as Quran wording.
Evidence labels such as E1 are permitted only inside evidence_ids arrays and must never appear
in claim text. Do not reproduce source text, write wrapper prose, introduce facts from memory,
change the audience, add legal classifications or absolute quantifiers, or introduce a concept
absent from the evidence cited by that claim. Write claim text in the exact response_language.
The application deterministically renders every required exact source block. If the evidence
does not establish every requested facet, answer the supported Quran or Tafsir facet and leave
the unsupported facet unaddressed. A passage-level relationship may be stated as contextual
inference only when the cited surrounding verses expressly establish it; a Tafsir relationship
must be labelled as that Tafsir's interpretation. Return INSUFFICIENT_EVIDENCE only when no
supplied passage or Tafsir excerpt supports any facet of the question."""

    def __init__(
        self,
        client: ChatModelPort,
        validator: GroundingValidator,
        *,
        deployment_id: str = GROQ_CHAT_DEPLOYMENT_ID,
        route_validator: Callable[[ProviderCompletion], None] | None = None,
        # WP-1 (V3 plan), F-01/task 3: raised from 450/900 so the claims layer is no longer
        # squeezed into a citation table. Still atomic, evidence-bound claims -- only the
        # budget changes, not the contract.
        max_tokens_quran: int = 1400,
        max_tokens_tafsir: int = 2000,
        repair_token_multiplier: float = 1.5,
    ) -> None:
        self.client = client
        self.validator = validator
        self.deployment_id = deployment_id
        self.route_validator = route_validator or ProviderRoutePolicy.require_groq_direct_free
        self.max_tokens_quran = max_tokens_quran
        self.max_tokens_tafsir = max_tokens_tafsir
        self.repair_token_multiplier = repair_token_multiplier

    async def synthesize(
        self,
        *,
        question: str,
        evidence: EvidenceSet,
        response_language: str,
        route_state: object | None = None,
    ) -> tuple[AnswerDraftV1, list[ProviderCompletion]]:
        if self.deployment_id not in evidence.permitted_synthesis_deployments:
            raise GroundingError(
                "Selected synthesis deployment is not permitted for this evidence set"
            )
        selected_evidence = list(evidence.items_by_id.values())
        required_tafsir_ids = [
            item.evidence_id
            for item in selected_evidence
            if item.source_kind is EvidenceSourceKind.TAFSIR
            and (
                not evidence.requirements.required_tafsir_aliases
                or item.source_policy_alias in evidence.requirements.required_tafsir_aliases
            )
        ]
        is_tafsir = bool(required_tafsir_ids)
        source_evidence_ids = [item.evidence_id for item in selected_evidence]
        payload = {
            "question": question,
            "response_language": response_language,
            "evidence_requirements": evidence.requirements.model_dump(mode="json"),
            "synthesis_constraints": {
                "allowed_evidence_ids": source_evidence_ids,
                "required_tafsir_evidence_ids": required_tafsir_ids,
                "required_source_evidence_ids": source_evidence_ids,
                "evidence_ids_may_appear_in_claim_text": False,
                "every_claim_must_cite_tafsir": is_tafsir,
                "direct_quran_claims_are_allowed": bool(
                    is_tafsir and evidence.requirements.allow_partial
                ),
            },
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source_kind": item.source_kind.value,
                    "language": item.language,
                    "reference": item.reference.model_dump(mode="json"),
                    "evidence_text": self._model_evidence_text(item),
                    "evidence_text_is_bounded_excerpt": (
                        len(self._model_evidence_text(item)) < len(item.exact_text)
                    ),
                }
                for item in selected_evidence
            ],
        }
        response_model: type[QuranAnswerProposalV1 | TafsirAnswerProposalV1]
        response_model = TafsirAnswerProposalV1 if is_tafsir else QuranAnswerProposalV1
        schema_name = "tafsir_answer_proposal_v1" if is_tafsir else "quran_answer_proposal_v1"
        system_prompt = self.TAFSIR_SYSTEM_PROMPT if is_tafsir else self.SYSTEM_PROMPT
        metadata: list[ProviderCompletion] = []
        last_error = ""
        last_validation_error: Exception | None = None
        repair_guidance: list[str] = []
        for attempt in range(2):
            draft: AnswerDraftV1 | None = None
            draft_for_rejection: AnswerDraftV1 | None = None
            prompt = json.dumps(payload, ensure_ascii=False)
            if attempt:
                prompt += self._repair_prompt_suffix(
                    response_language=response_language,
                    repair_guidance=repair_guidance,
                )
            try:
                raw_draft, completion = await self.client.complete(
                    response_model=response_model,
                    schema_name=schema_name,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=self._max_tokens_for_attempt(is_tafsir=is_tafsir, attempt=attempt),
                )
                self.route_validator(completion)
                metadata.append(completion)
                if isinstance(raw_draft, AnswerDraftV1):
                    draft = raw_draft
                elif isinstance(raw_draft, TafsirAnswerProposalV1):
                    draft = raw_draft.to_canonical(
                        source_evidence_ids=source_evidence_ids,
                        comparison=evidence.requirements.require_two_distinct_tafsir_resources,
                    )
                elif isinstance(raw_draft, QuranAnswerProposalV1):
                    draft = raw_draft.to_canonical()
                elif is_tafsir:
                    draft = TafsirAnswerProposalV1.model_validate(raw_draft).to_canonical(
                        source_evidence_ids=source_evidence_ids,
                        comparison=evidence.requirements.require_two_distinct_tafsir_resources,
                    )
                else:
                    draft = QuranAnswerProposalV1.model_validate(raw_draft).to_canonical()
                draft, sanitized_claims = sanitize_model_authored_claims(draft)
                if sanitized_claims:
                    recorder = _TRACE_RECORDER.get()
                    if recorder is not None:
                        recorder.emit(
                            "synthesis",
                            "degraded",
                            "Claim text sanitized before validation",
                            detail={
                                "claim_count_changed": len(sanitized_claims),
                                "claims": sanitized_claims,
                            },
                        )
                draft, pair_completion_added = _complete_quran_translation_source_requests(
                    draft, evidence
                )
                draft_for_rejection = draft
                if pair_completion_added:
                    recorder = _TRACE_RECORDER.get()
                    if recorder is not None:
                        recorder.emit(
                            "synthesis",
                            "completed",
                            "Arabic/translation source requests completed",
                            detail={"pair_completion_added": pair_completion_added},
                        )
                self.validator.validate(
                    draft,
                    evidence,
                    response_language=response_language,
                    check_semantic_entailment=False,
                )
                draft, dropped_claims, entailment_error = (
                    self.validator.drop_semantically_invalid_claims(draft, evidence)
                )
                if dropped_claims:
                    recorder = _TRACE_RECORDER.get()
                    if recorder is not None:
                        recorder.emit(
                            "synthesis",
                            "completed",
                            "Claims dropped by entailment gate",
                            detail={"dropped_claims": dropped_claims},
                        )
                    if not any(isinstance(block, ClaimBlockV1) for block in draft.blocks):
                        assert entailment_error is not None
                        raise entailment_error
                self.validator.validate(
                    draft,
                    evidence,
                    response_language=response_language,
                )
                _reject_too_short_sanitized_claims(draft)
                return draft, metadata
            except ProviderUnavailableError as exc:
                raise SynthesisTemporarilyUnavailable(
                    exc.diagnostic,
                    metadata,
                ) from exc
            except ProviderRouteError as exc:
                raise GroundingError("Synthesis effective route was rejected") from exc
            except (GroundingError, StructuredOutputValidationError, ValidationError) as exc:
                last_validation_error = exc
                guidance = self._repair_guidance(
                    exc,
                    response_language=response_language,
                    required_tafsir_ids=required_tafsir_ids,
                )
                repair_guidance = self._accumulate_repair_guidance(repair_guidance, guidance)
                last_error = "\n".join(repair_guidance)
                recorder = _TRACE_RECORDER.get()
                if recorder is not None:
                    recorder.emit(
                        "synthesis",
                        "rejected",
                        "Synthesis draft rejected by validation",
                        detail=rejected_synthesis_detail(
                            attempt=attempt + 1,
                            error=exc,
                            repair_guidance=last_error,
                            draft=draft_for_rejection or draft,
                            is_tafsir_route=is_tafsir,
                        ),
                    )
        if isinstance(last_validation_error, GroundingError) and (
            "canonical Quran text" in str(last_validation_error)
        ):
            raise CanonicalTextGroundingError(
                "Synthesis reproduced canonical source wording after one bounded repair"
            )
        raise GroundingError(f"Synthesis failed after one bounded repair: {last_error}")

    def _max_tokens_for_attempt(self, *, is_tafsir: bool, attempt: int) -> int:
        base_budget = self.max_tokens_tafsir if is_tafsir else self.max_tokens_quran
        return (
            int(base_budget * self.repair_token_multiplier)
            if attempt
            else base_budget
        )

    @staticmethod
    def _repair_prompt_suffix(*, response_language: str, repair_guidance: list[str]) -> str:
        guidance = "\n".join(f"- {item}" for item in repair_guidance)
        return (
            "\nThis is the only repair. Return compact JSON immediately with no analysis or "
            "preamble. Correct every prior issue:\n"
            f"{guidance}\n"
            "Always preserve these existing constraints:\n"
            f"- Write every claim in the exact response_language: {response_language}.\n"
            "- No chapter:verse reference, reference range, E-number, or source-kind tag may "
            "appear in claim text.\n"
            "- Do not copy canonical Quran, translation, or Tafsir wording into claim text.\n"
            "- Return the whole response as compact JSON within the output limit; prefer fewer, "
            "shorter claims over more."
        )

    @staticmethod
    def _accumulate_repair_guidance(
        repair_guidance: list[str], guidance: str
    ) -> list[str]:
        return repair_guidance if guidance in repair_guidance else [*repair_guidance, guidance]

    @classmethod
    def _model_evidence_text(cls, item: EvidenceItem) -> str:
        if (
            item.source_kind is not EvidenceSourceKind.TAFSIR
            or len(item.exact_text) <= cls._TAFSIR_CONTEXT_MAX_CHARACTERS
        ):
            return item.exact_text
        candidate = item.exact_text[: cls._TAFSIR_CONTEXT_MAX_CHARACTERS]
        minimum_boundary = cls._TAFSIR_CONTEXT_MAX_CHARACTERS // 2
        boundary = max(candidate.rfind(marker) for marker in ("\n", ". ", "؟ ", "! "))
        if boundary >= minimum_boundary:
            candidate = candidate[: boundary + 1]
        else:
            whitespace = candidate.rfind(" ")
            if whitespace >= minimum_boundary:
                candidate = candidate[:whitespace]
        return candidate.strip()

    @staticmethod
    def _repair_guidance(
        exc: GroundingError | StructuredOutputValidationError | ValidationError,
        *,
        response_language: str,
        required_tafsir_ids: list[str],
    ) -> str:
        detail = str(exc)
        if isinstance(exc, StructuredOutputValidationError):
            return (
                "The prior completion was not valid compact JSON or reached its output limit. "
                "Use one short claim for a single Tafsir and only schema fields."
            )
        if "exposes an internal evidence ID" in detail:
            return (
                "Remove every E-number label from claims[].text. Keep evidence labels only in "
                "claims[].evidence_ids."
            )
        if "exposes an internal source tag" in detail:
            return (
                "Remove technical source-kind tags such as QURAN_TRANSLATION from claim text. "
                "Keep source metadata only in evidence_ids and source_evidence_ids."
            )
        if "duplicates a public Quran reference" in detail:
            return (
                "Remove chapter:verse numbers and ranges from claim text. The application "
                "already renders reference labels and citations."
            )
        if "must cite exact Tafsir evidence" in detail or (
            "Every required Tafsir resource must support a claim" in detail
        ):
            return (
                "Every claim evidence_ids array must include Tafsir evidence, and the claims "
                f"together must cover all of these Tafsir IDs: {required_tafsir_ids}."
            )
        if "cites unavailable evidence" in detail:
            return "Use only the allowed_evidence_ids supplied in synthesis_constraints."
        if "response_language" in detail:
            language = response_language.strip().casefold().split("-", maxsplit=1)[0]
            language_label = "Arabic" if language == "ar" else "English"
            return (
                f"The prior claim does not match {language_label} response_language; "
                f"every claim must match {response_language} exactly."
            )
        if "high-risk concept absent" in detail:
            return (
                "Remove concepts not stated in the evidence cited by that individual claim; "
                "use a narrower literal paraphrase."
            )
        if "canonical Quran text" in detail:
            return (
                "Remove all Quran wording from every claim. State only how the cited reference "
                "relates to the requested concept; the application renders the full canonical "
                "Arabic and translation in source blocks."
            )
        if "At least one atomic grounded claim" in detail:
            return "For ANSWERABLE, provide one short atomic grounded claim."
        if isinstance(exc, ValidationError):
            return "Return only the exact requested schema with one short atomic claim."
        return "Correct the grounding or schema violation without adding content."


class ResponseValidator:
    def validate(self, response: ResearchResponseV1, evidence: EvidenceSet) -> None:
        by_id = evidence.items_by_id
        canonical_citations: dict[str, CitationV1] = {}
        for citation in response.citations:
            if citation.evidence_id in canonical_citations:
                raise GroundingError("Duplicate response citation provenance")
            item = by_id.get(citation.evidence_id)
            if item is None:
                raise GroundingError("Response citation has no canonical evidence")
            if (
                citation.source_kind is not item.source_kind
                or citation.reference != item.reference
                or citation.attribution != item.attributions
                or citation.canonical_link != item.reference.canonical_link
                or citation.source_revision != item.source_revision
            ):
                raise GroundingError("Response citation provenance does not match evidence")
            canonical_citations[citation.evidence_id] = citation

        for block in response.blocks:
            if isinstance(block, RenderedClaimBlockV1):
                if any(
                    citation.evidence_id not in canonical_citations for citation in block.citations
                ):
                    raise GroundingError("Rendered claim contains an invalid citation")
            elif isinstance(block, RenderedSourceBlockV1):
                item = by_id.get(block.citation.evidence_id)
                if item is None or block.exact_text != item.exact_text:
                    raise GroundingError("Displayed source text is not canonical storage text")


class ExactRenderer:
    def __init__(self, validator: ResponseValidator | None = None) -> None:
        self.validator = validator or ResponseValidator()

    def render(
        self,
        *,
        conversation_id: UUID,
        turn_id: UUID,
        correlation_id: UUID,
        draft: AnswerDraftV1,
        evidence: EvidenceSet,
    ) -> ResearchResponseV1:
        if draft.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            response = terminal_response(
                conversation_id=conversation_id,
                turn_id=turn_id,
                correlation_id=correlation_id,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
                text="The retrieved canonical evidence is insufficient for this exact question.",
            )
            self.validator.validate(response, evidence)
            return response
        referenced_ids: list[str] = []
        for block in draft.blocks:
            if isinstance(block, ClaimBlockV1):
                referenced_ids.extend(block.evidence_ids)
            elif isinstance(block, SourceBlockRequestV1):
                referenced_ids.append(block.quote_reference.evidence_id)
        referenced_ids = list(dict.fromkeys(referenced_ids))
        citations_by_id = {
            evidence_id: self._citation(index, evidence.items_by_id[evidence_id])
            for index, evidence_id in enumerate(referenced_ids, start=1)
        }

        rendered_blocks: list[
            RenderedClaimBlockV1
            | RenderedSourceBlockV1
            | RenderedSourceViewHeadingBlockV1
            | RenderedNoticeBlockV1
        ] = []
        for block in draft.blocks:
            if isinstance(block, ClaimBlockV1):
                rendered_blocks.append(
                    RenderedClaimBlockV1(
                        block_type="CLAIM",
                        claim_id=block.claim_id,
                        text=block.text,
                        citations=[citations_by_id[item] for item in block.evidence_ids],
                    )
                )
            elif isinstance(block, SourceBlockRequestV1):
                item = evidence.items_by_id[block.quote_reference.evidence_id]
                exact_text = self._render_exact_text(item, block)
                rendered_blocks.append(
                    RenderedSourceBlockV1(
                        block_type="SOURCE",
                        block_id=block.block_id,
                        exact_text=exact_text,
                        citation=citations_by_id[item.evidence_id],
                        visibility=block.visibility,
                        passage_group_id=block.passage_group_id,
                    )
                )
            elif isinstance(block, SourceViewHeadingRequestV1):
                rendered_blocks.append(
                    RenderedSourceViewHeadingBlockV1(
                        block_type="SOURCE_VIEW_HEADING", text=block.text
                    )
                )

        response = ResearchResponseV1(
            schema_version="1",
            conversation_id=conversation_id,
            turn_id=turn_id,
            correlation_id=correlation_id,
            status=draft.status,
            output_kind=derive_rendered_output_kind(rendered_blocks),
            blocks=rendered_blocks,
            citations=list(citations_by_id.values()),
            clarification=None,
            warnings=[],
        )
        self.validator.validate(response, evidence)
        return response

    @staticmethod
    def _citation(index: int, item: EvidenceItem) -> CitationV1:
        attribution = ", ".join(value.display_name for value in item.attributions)
        if item.source_kind is EvidenceSourceKind.TAFSIR:
            label = f"Tafsir: {item.source_identity.work_title} — {attribution}"
        else:
            label = f"Quran {item.reference.value} — {attribution}"
        return CitationV1(
            citation_id=f"C{index}",
            evidence_id=item.evidence_id,
            display_label=label,
            source_kind=item.source_kind,
            reference=item.reference,
            attribution=item.attributions,
            canonical_link=item.reference.canonical_link,
            source_revision=item.source_revision,
        )

    @staticmethod
    def _render_exact_text(item: EvidenceItem, request: SourceBlockRequestV1) -> str:
        if request.quote_mode is QuoteMode.FULL:
            return item.exact_text
        span = next(
            value
            for value in item.approved_quote_spans
            if value.span_id == request.quote_reference.approved_span_id
        )
        return item.exact_text[span.start_character : span.end_character_exclusive]


def terminal_response(
    *,
    conversation_id: UUID,
    turn_id: UUID,
    correlation_id: UUID,
    status: AnswerStatus,
    notice_code: NoticeCode,
    text: str,
    warnings: list[ToolWarning] | None = None,
) -> ResearchResponseV1:
    return ResearchResponseV1(
        schema_version="1",
        conversation_id=conversation_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        status=status,
        output_kind=derive_rendered_output_kind([]),
        blocks=[RenderedNoticeBlockV1(block_type="NOTICE", notice_code=notice_code, text=text)],
        citations=[],
        clarification=None,
        warnings=warnings or [],
    )


def clarification_response(
    *,
    conversation_id: UUID,
    turn_id: UUID,
    correlation_id: UUID,
    clarification: ClarificationStateV1,
) -> ResearchResponseV1:
    if not clarification.required:
        raise ValueError("A clarification response requires pending clarification state")
    return ResearchResponseV1(
        schema_version="1",
        conversation_id=conversation_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        status=AnswerStatus.CLARIFICATION_REQUIRED,
        output_kind=derive_rendered_output_kind([]),
        blocks=[
            RenderedNoticeBlockV1(
                block_type="NOTICE",
                notice_code=NoticeCode.CLARIFICATION,
                text=clarification.question or "Clarification is required.",
            )
        ],
        citations=[],
        clarification=clarification,
        warnings=[],
    )


def deterministic_exact_lookup_draft(evidence: EvidenceSet) -> AnswerDraftV1:
    """Render exact lookup evidence without asking a provider to paraphrase it."""

    if not evidence.requirements_satisfied or not evidence.items_by_id:
        raise GroundingError("Exact lookup requires complete canonical evidence")
    evidence_ids = _bounded_passage_source_ids(evidence, claim_block_count=0)
    requests = [
        SourceBlockRequestV1(
            block_type="SOURCE_BLOCK_REQUEST",
            block_id=f"SOURCE-{index}",
            quote_reference=QuoteReferenceV1(
                evidence_id=evidence_id,
                approved_span_id=None,
            ),
            quote_mode=QuoteMode.FULL,
        )
        for index, evidence_id in enumerate(evidence_ids, start=1)
    ]
    return AnswerDraftV1(schema_version="1", status=AnswerStatus.ANSWERABLE, blocks=requests)


def deterministic_grounded_topic_draft(
    evidence: EvidenceSet,
    *,
    response_language: str,
    supported_facet_summaries: tuple[tuple[str, str], ...] = (),
) -> AnswerDraftV1:
    """Render a validated semantic fallback when synthesis keeps copying canonical text.

    Selection is based only on retrieval rank and canonical pointer identity. The model-owned
    summary is accepted only when it was attached to that exact retained pointer; no topic,
    verse, or language-specific religious mapping is embedded here.
    """

    selected = min(
        evidence.items_by_id.values(),
        key=lambda item: (item.retrieval.rank or 10_000, item.evidence_id),
    )
    pointer = evidence_passage_key(selected)
    paired = [
        item for item in evidence.items_by_id.values() if evidence_passage_key(item) == pointer
    ]
    paired.sort(
        key=lambda item: (item.retrieval.rank is None, item.retrieval.rank or 0, item.evidence_id)
    )
    evidence_ids = _bounded_passage_source_ids(
        evidence,
        claim_block_count=1,
        evidence_ids=[item.evidence_id for item in paired],
    )
    selected_pointer = pointer
    claim_text = next(
        (
            summary.strip()
            for summary_pointer, summary in supported_facet_summaries
            if (
                summary_pointer == selected_pointer
                or summary_pointer
                == json.dumps(selected.canonical_pointer.model_dump(mode="json"), sort_keys=True)
            )
            and summary.strip()
        ),
        "",
    )
    if not claim_text:
        raise GroundingError(
            "A deterministic topic fallback requires a validated supported-facet summary"
        )
    language = response_language.strip().casefold().split("-", maxsplit=1)[0]
    facet = claim_text.rstrip(".۔ ")
    claim_text = (
        f"يتناول النص المعروض {facet}."
        if language == "ar"
        else f"The displayed passage addresses {facet}."
    )
    blocks: list[ClaimBlockV1 | SourceBlockRequestV1] = [
        ClaimBlockV1(
            block_type="CLAIM",
            claim_id="claim-1",
            text=claim_text,
            claim_type=ClaimType.FACTUAL_SUMMARY,
            evidence_ids=evidence_ids,
        )
    ]
    blocks.extend(
        SourceBlockRequestV1(
            block_type="SOURCE_BLOCK_REQUEST",
            block_id=f"SOURCE-{index}",
            quote_reference=QuoteReferenceV1(
                evidence_id=evidence_id,
                approved_span_id=None,
            ),
            quote_mode=QuoteMode.FULL,
        )
        for index, evidence_id in enumerate(evidence_ids, start=1)
    )
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=blocks,
    )


def deterministic_contextual_topic_draft(
    evidence: EvidenceSet,
    *,
    response_language: str,
    supported_facet_summaries: tuple[tuple[str, str], ...] = (),
) -> AnswerDraftV1:
    """Last-resort presentation for a retrieved, passage-aware topic bundle.

    It makes no theological inference: source blocks carry the Quran and Tafsir
    evidence while the short connective prose explains why the surrounding passage
    is shown and preserves the current Hadith-source boundary.
    """

    language = response_language.strip().casefold().split("-", maxsplit=1)[0]
    summaries = list(
        dict.fromkeys(
            summary.strip() for _, summary in supported_facet_summaries if summary.strip()
        )
    )[:3]
    if summaries:
        joined = "؛ وفي سياق المقطع: ".join(summaries)
        text = (
            f"الجواب المباشر من الدليل المسترجع: {joined}. "
            "وهذا الموضع جزء من مقطع واحد؛ لذلك تُعرض الآيات السابقة معه لفهم "
            "السياق، لا كآية منفصلة."
            if language == "ar"
            else "Directly supported by the retrieved evidence: "
            + "; in the passage context: ".join(summaries)
            + ". This is part of one connected passage, so the preceding verses are "
            "shown with it for context rather than treating it as an isolated verse."
        )
    else:
        text = (
            "الجواب المباشر: المقطع القرآني المسترجع يُعرض بسياقه المتصل، لا كآية منفصلة. "
            "ولا يضيف هذا الجواب أذكارًا أو عددًا للتلاوة؛ فمصدر حديث معتمدًا لم يُضف إلى "
            "هذا النطاق بعد."
            if language == "ar"
            else "Direct answer: the retrieved Quran passage is shown in its connected context, "
            "not as an isolated verse. This answer does not add supplications or recitation "
            "counts, because an approved Hadith source is not yet available in this scope."
        )
    evidence_ids = _bounded_passage_source_ids(evidence, claim_block_count=1)
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            ClaimBlockV1(
                block_type="CLAIM",
                claim_id="contextual-topic-boundary",
                text=text,
                claim_type=ClaimType.FACTUAL_SUMMARY,
                evidence_ids=evidence_ids,
            ),
            *[
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"SOURCE-{index}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=evidence_id, approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                )
                for index, evidence_id in enumerate(evidence_ids, start=1)
            ],
        ],
    )


def deterministic_topic_source_view_draft(
    evidence: EvidenceSet,
    *,
    response_language: str,
    honest_insufficient: bool = False,
) -> AnswerDraftV1:
    """Render retained topic evidence as an attributed, source-only reading view.

    The ranked evidence and its bounded passage selection are reused exactly as the
    contextual fallback already uses them.  The first three retained Quran passage
    groups are initially visible.  Any additional retained Quran groups are preserved
    as collapsed structural context, explicitly marked so analytics never count them
    as material the reader saw without expanding it.

    Only Al-Mukhtasar evidence whose verse range contains a displayed Quran verse is
    attached to that group.  This is a deterministic pointer mapping, never a new
    retrieval, relevance, or Tafsir selection step.
    """

    if not evidence.requirements_satisfied or not evidence.items_by_id:
        raise GroundingError("Source-only topic view requires complete canonical evidence")

    language = response_language.strip().casefold().split("-", maxsplit=1)[0]
    heading = (
        ("نتائج قد تكون ذات صلة" if language == "ar" else "Possibly related results")
        if honest_insufficient
        else ("آيات للاستكشاف" if language == "ar" else "Passages to explore")
    )
    # Reserve one ordinary stacked block for the application-owned heading, preserving
    # the same bounded selection the previous contextual draft used beside its claim.
    selected_ids = _bounded_passage_source_ids(evidence, claim_block_count=1)
    selected_set = set(selected_ids)

    groups: list[tuple[str, list[str], set[tuple[int, int]]]] = []
    group_index: dict[str, int] = {}
    for evidence_id in selected_ids:
        item = evidence.items_by_id[evidence_id]
        if item.source_kind not in {
            EvidenceSourceKind.QURAN,
            EvidenceSourceKind.QURAN_TRANSLATION,
        }:
            continue
        pointer = json.dumps(item.canonical_pointer.model_dump(mode="json"), sort_keys=True)
        position = group_index.get(pointer)
        if position is None:
            position = len(groups)
            group_index[pointer] = position
            groups.append((pointer, [], set()))
        group_ids = groups[position][1]
        group_ids.append(evidence_id)
        groups[position][2].update(
            (key.chapter, key.verse) for key in item.canonical_pointer.verse_keys
        )

    # A source request must remain unique.  A multi-verse Tafsir passage is therefore
    # attached to its first retained matching verse group rather than duplicated into
    # several groups with the same canonical text.
    tafsir_for_group: dict[int, list[str]] = {}
    for evidence_id in selected_ids:
        item = evidence.items_by_id[evidence_id]
        if (
            item.source_kind is not EvidenceSourceKind.TAFSIR
            or item.source_policy_alias != "qul-tafsir-almukhtasar-ar-v1"
        ):
            continue
        tafsir_keys = {(key.chapter, key.verse) for key in item.canonical_pointer.verse_keys}
        matching_group = next(
            (
                index
                for index, (_, _, verse_keys) in enumerate(groups)
                if tafsir_keys.intersection(verse_keys)
            ),
            None,
        )
        if matching_group is not None:
            tafsir_for_group.setdefault(matching_group, []).append(evidence_id)

    requests: list[SourceBlockRequestV1] = []
    for index, (_, canonical_ids, _) in enumerate(groups, start=1):
        visibility = (
            SourceVisibility.COLLAPSED_CONTEXT
            if honest_insufficient or index > 3
            else SourceVisibility.VISIBLE
        )
        group_id = f"passage-{index}"
        for evidence_id in [*canonical_ids, *tafsir_for_group.get(index - 1, [])]:
            if evidence_id not in selected_set:
                raise RuntimeError("Source-view mapping escaped the retained selection")
            requests.append(
                SourceBlockRequestV1(
                    block_type="SOURCE_BLOCK_REQUEST",
                    block_id=f"{group_id}-source-{len(requests) + 1}",
                    quote_reference=QuoteReferenceV1(
                        evidence_id=evidence_id, approved_span_id=None
                    ),
                    quote_mode=QuoteMode.FULL,
                    visibility=visibility,
                    passage_group_id=group_id,
                )
            )

    if not requests:
        raise GroundingError("Source-only topic view requires Quran source blocks")
    return AnswerDraftV1(
        schema_version="1",
        status=AnswerStatus.ANSWERABLE,
        blocks=[
            SourceViewHeadingRequestV1(
                block_type="SOURCE_VIEW_HEADING_REQUEST", text=heading
            ),
            *requests,
        ],
    )
