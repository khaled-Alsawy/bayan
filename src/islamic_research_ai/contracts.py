from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from islamic_research_ai.contextual_topic_template import (
    is_deterministic_contextual_topic_template_prose,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intent(StrEnum):
    QURAN_TOPIC_SEARCH = "QURAN_TOPIC_SEARCH"
    QURAN_PASSAGE_LOOKUP = "QURAN_PASSAGE_LOOKUP"
    HADITH_TOPIC_SEARCH = "HADITH_TOPIC_SEARCH"
    EXPLAIN_QURAN_VERSE = "EXPLAIN_QURAN_VERSE"
    EXPLAIN_HADITH = "EXPLAIN_HADITH"
    ISLAMIC_TOPIC_RESEARCH = "ISLAMIC_TOPIC_RESEARCH"
    COMPARE_TAFSIR = "COMPARE_TAFSIR"
    FIQH_OR_FATWA = "FIQH_OR_FATWA"
    OTHER = "OTHER"


class ScopeStatus(StrEnum):
    ALLOWED = "ALLOWED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


class AnswerStatus(StrEnum):
    ANSWERABLE = "ANSWERABLE"
    PARTIAL = "PARTIAL"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED_SCOPE = "UNSUPPORTED_SCOPE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    FAILED_SAFE = "FAILED_SAFE"


class RenderedOutputKind(StrEnum):
    """What the reader actually received, independent of response status."""

    SUBSTANTIVE_SYNTHESIS = "SUBSTANTIVE_SYNTHESIS"
    SOURCE_ONLY = "SOURCE_ONLY"
    TEMPLATE = "TEMPLATE"
    NO_CONTENT = "NO_CONTENT"


class ScopeSignal(StrEnum):
    REQUESTS_FIQH_RULING = "REQUESTS_FIQH_RULING"
    REQUESTS_PERSONAL_RELIGIOUS_DECISION = "REQUESTS_PERSONAL_RELIGIOUS_DECISION"
    REQUESTS_MADHHAB_COMPARISON = "REQUESTS_MADHHAB_COMPARISON"
    OUTSIDE_SUPPORTED_RESEARCH = "OUTSIDE_SUPPORTED_RESEARCH"


class ScopeReason(StrEnum):
    SUPPORTED = "SUPPORTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FIQH_OR_FATWA_NOT_SUPPORTED = "FIQH_OR_FATWA_NOT_SUPPORTED"
    OUTSIDE_SUPPORTED_RESEARCH = "OUTSIDE_SUPPORTED_RESEARCH"


class ClarificationReason(StrEnum):
    MISSING_REFERENCE = "MISSING_REFERENCE"
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"
    AMBIGUOUS_HADITH = "AMBIGUOUS_HADITH"
    UNKNOWN_RESOURCE = "UNKNOWN_RESOURCE"
    AMBIGUOUS_CONTEXT = "AMBIGUOUS_CONTEXT"
    OTHER = "OTHER"


class ClarificationField(StrEnum):
    TOPIC = "TOPIC"
    QURAN_REFERENCE = "QURAN_REFERENCE"
    HADITH_TARGET = "HADITH_TARGET"
    SOURCE_KIND = "SOURCE_KIND"
    RESOURCE = "RESOURCE"


class ClarificationDetail(StrEnum):
    MISSING_SURAH_OR_CHAPTER = "MISSING_SURAH_OR_CHAPTER"
    MISSING_VERSE_NUMBER = "MISSING_VERSE_NUMBER"
    MULTIPLE_REFERENCE_MATCHES = "MULTIPLE_REFERENCE_MATCHES"
    MULTIPLE_HADITH_MATCHES = "MULTIPLE_HADITH_MATCHES"
    RESOURCE_NOT_APPROVED = "RESOURCE_NOT_APPROVED"
    CONTEXT_HAS_MULTIPLE_TARGETS = "CONTEXT_HAS_MULTIPLE_TARGETS"
    OTHER = "OTHER"


class EvidenceSourceKind(StrEnum):
    QURAN = "QURAN"
    QURAN_TRANSLATION = "QURAN_TRANSLATION"
    HADITH = "HADITH"
    HADITH_GRADE = "HADITH_GRADE"
    TAFSIR = "TAFSIR"
    HADITH_COMMENTARY = "HADITH_COMMENTARY"


class VerseKey(StrictModel):
    chapter: int = Field(ge=1, le=114)
    verse: int = Field(gt=0)


class QuranPointer(StrictModel):
    kind: Literal["QURAN"]
    verse_keys: list[VerseKey] = Field(min_length=1)


class HadithPointer(StrictModel):
    kind: Literal["HADITH"]
    source_record_id: str


class TafsirPointer(StrictModel):
    kind: Literal["TAFSIR"]
    passage_id: str
    verse_keys: list[VerseKey] = Field(min_length=1)


class HadithCommentaryPointer(StrictModel):
    kind: Literal["HADITH_COMMENTARY"]
    passage_id: str
    hadith_source_record_id: str


CanonicalPointer = Annotated[
    QuranPointer | HadithPointer | TafsirPointer | HadithCommentaryPointer,
    Field(discriminator="kind"),
]


class SourceRevisionIdentity(StrictModel):
    source_id: str
    resource_alias: str
    revision_id: str
    checksum: str


class SourceIdentityV1(StrictModel):
    source_id: str
    resource_alias: str
    work_title: str
    author_translator_or_grader: list[str]
    canonical_base_url: str | None


class TypedSourceReference(StrictModel):
    scheme: str
    value: str
    canonical_link: str | None


class AttributionRole(StrEnum):
    SOURCE = "SOURCE"
    AUTHOR = "AUTHOR"
    TRANSLATOR = "TRANSLATOR"
    GRADER = "GRADER"
    PUBLISHER = "PUBLISHER"


class AttributionV1(StrictModel):
    role: AttributionRole
    display_name: str
    required_notice: str | None


class RetrievalMethod(StrEnum):
    EXACT = "EXACT"
    LEXICAL = "LEXICAL"
    DENSE = "DENSE"
    HYBRID = "HYBRID"
    RERANKED = "RERANKED"


class RetrievalProvenanceV1(StrictModel):
    tool_id: str
    method: RetrievalMethod
    index_revision: SourceRevisionIdentity | None
    rank: int | None = Field(default=None, gt=0)
    component_scores: dict[str, float]


class ApprovedSpanV1(StrictModel):
    span_id: str
    start_character: int = Field(ge=0)
    end_character_exclusive: int = Field(gt=0)
    span_hash: str

    @model_validator(mode="after")
    def validate_range(self) -> ApprovedSpanV1:
        if self.end_character_exclusive <= self.start_character:
            raise ValueError("Approved span end must be greater than start")
        return self


class ResolutionProvenance(StrEnum):
    CURRENT_MESSAGE_EXACT_PARSE = "CURRENT_MESSAGE_EXACT_PARSE"
    WORKING_CONTEXT_EXACT_REFERENCE = "WORKING_CONTEXT_EXACT_REFERENCE"
    APPROVED_ALIAS_TABLE = "APPROVED_ALIAS_TABLE"
    HIGH_PRECISION_SCOPE_PRECHECK = "HIGH_PRECISION_SCOPE_PRECHECK"


class CandidateReferenceV1(StrictModel):
    candidate_id: str
    kind: Literal["QURAN", "HADITH"]
    safe_label: str
    canonical_pointer: CanonicalPointer | None
    source_revision: SourceRevisionIdentity | None
    provenance: ResolutionProvenance


class ClarificationCandidateV1(StrictModel):
    candidate_id: str
    label: str
    reference: CandidateReferenceV1 | None


class ClarificationStateV1(StrictModel):
    required: bool
    reason: ClarificationReason | None
    detail: ClarificationDetail | None
    missing_fields: list[ClarificationField]
    candidates: list[ClarificationCandidateV1]
    question: str | None

    @model_validator(mode="after")
    def validate_state(self) -> ClarificationStateV1:
        if self.required:
            if self.reason is None or self.question is None:
                raise ValueError("Required clarification needs a reason and question")
        elif any(
            (
                self.reason is not None,
                self.detail is not None,
                bool(self.missing_fields),
                bool(self.candidates),
                self.question is not None,
            )
        ):
            raise ValueError("Non-required clarification must have empty nullable fields")
        return self


def no_clarification() -> ClarificationStateV1:
    return ClarificationStateV1(
        required=False,
        reason=None,
        detail=None,
        missing_fields=[],
        candidates=[],
        question=None,
    )


class ScopeDecisionV1(StrictModel):
    schema_version: Literal["1"]
    status: ScopeStatus
    reason: ScopeReason
    signals: set[ScopeSignal]
    clarification: ClarificationStateV1
    user_message_key: str

    @model_validator(mode="after")
    def validate_clarification_status(self) -> ScopeDecisionV1:
        if self.status is ScopeStatus.CLARIFICATION_REQUIRED:
            if not self.clarification.required:
                raise ValueError("Clarification status requires clarification state")
        elif self.clarification.required:
            raise ValueError("Required clarification must use clarification status")
        return self


class ResolvedQuranReferenceV1(StrictModel):
    verse_keys: list[VerseKey] = Field(min_length=1)
    canonical_label: str
    provenance: ResolutionProvenance


class ResolvedHadithReferenceV1(StrictModel):
    canonical_pointer: CanonicalPointer
    reference_set: list[TypedSourceReference] = Field(min_length=1)
    source_revision: SourceRevisionIdentity
    provenance: ResolutionProvenance


class ResolutionIssueCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    INVALID_RANGE = "INVALID_RANGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


class ResolutionIssueV1(StrictModel):
    code: ResolutionIssueCode
    safe_message_key: str
    candidate_ids: list[str]


class DeterministicResolutionContextV1(StrictModel):
    schema_version: Literal["1"]
    resolved_quran_references: list[ResolvedQuranReferenceV1]
    quran_candidates: list[CandidateReferenceV1]
    resolved_hadith_references: list[ResolvedHadithReferenceV1]
    hadith_candidates: list[CandidateReferenceV1]
    deterministic_scope_signals: set[ScopeSignal]
    issues: list[ResolutionIssueV1]


class TurnRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class SanitizedTurnV1(StrictModel):
    turn_id: UUID
    role: TurnRole
    sanitized_text: str


class ContextProvenance(StrEnum):
    CURRENT_USER = "CURRENT_USER"
    PRIOR_USER = "PRIOR_USER"
    TOOL_RESOLVED = "TOOL_RESOLVED"


class ContextField(StrEnum):
    TOPIC = "TOPIC"
    QURAN_REFERENCE = "QURAN_REFERENCE"
    HADITH_TARGET = "HADITH_TARGET"
    SOURCE_KIND = "SOURCE_KIND"
    RESOURCE = "RESOURCE"
    RESPONSE_LANGUAGE = "RESPONSE_LANGUAGE"


class ContextValueV1(StrictModel):
    field: ContextField
    value: str | list[VerseKey]
    provenance: ContextProvenance
    source_turn_id: UUID
    turn_age: int = Field(ge=0)


class WorkingContextV1(StrictModel):
    schema_version: Literal["1"]
    conversation_id: UUID
    state_version: int = Field(ge=0)
    recent_turns: list[SanitizedTurnV1] = Field(max_length=6)
    values: list[ContextValueV1]
    pending_clarification: ClarificationStateV1 | None
    expires_at: datetime


class ToolOperation(StrEnum):
    RESOLVE_QURAN_REFERENCE = "RESOLVE_QURAN_REFERENCE"
    SEARCH_QURAN_TOPICS = "SEARCH_QURAN_TOPICS"
    FETCH_QURAN_PASSAGE = "FETCH_QURAN_PASSAGE"
    RESOLVE_HADITH_IDENTITY = "RESOLVE_HADITH_IDENTITY"
    SEARCH_HADITH_TOPICS = "SEARCH_HADITH_TOPICS"
    FETCH_HADITH_RECORDS = "FETCH_HADITH_RECORDS"
    FETCH_TAFSIR_PASSAGES = "FETCH_TAFSIR_PASSAGES"
    FETCH_HADITH_COMMENTARY = "FETCH_HADITH_COMMENTARY"


class TargetSelector(StrEnum):
    DETERMINISTIC_QURAN_REFERENCES = "DETERMINISTIC_QURAN_REFERENCES"
    QURAN_SEARCH_CANDIDATES = "QURAN_SEARCH_CANDIDATES"
    DETERMINISTIC_HADITH_REFERENCES = "DETERMINISTIC_HADITH_REFERENCES"
    RESOLVED_HADITH_CANDIDATES = "RESOLVED_HADITH_CANDIDATES"
    HADITH_SEARCH_CANDIDATES = "HADITH_SEARCH_CANDIDATES"


class TafsirFetchMode(StrEnum):
    EXPLANATION_ONE = "EXPLANATION_ONE"
    COMPARISON_EXACTLY_TWO = "COMPARISON_EXACTLY_TWO"


class ResolveQuranReferenceIntentV1(StrictModel):
    operation: Literal["RESOLVE_QURAN_REFERENCE"]
    raw_reference: str
    language: str


class SearchQuranTopicsIntentV1(StrictModel):
    operation: Literal["SEARCH_QURAN_TOPICS"]
    query: str
    language: str
    surah_filters: list[int]


class FetchQuranPassageIntentV1(StrictModel):
    operation: Literal["FETCH_QURAN_PASSAGE"]
    target: Literal["DETERMINISTIC_QURAN_REFERENCES", "QURAN_SEARCH_CANDIDATES"]
    translation_resource_alias: str | None
    include_adjacent_verses: bool


class ResolveHadithIdentityIntentV1(StrictModel):
    operation: Literal["RESOLVE_HADITH_IDENTITY"]
    supplied_reference_or_excerpt: str
    language: str
    collection_filters: list[str]


class SearchHadithTopicsIntentV1(StrictModel):
    operation: Literal["SEARCH_HADITH_TOPICS"]
    query: str
    language: str
    collection_filters: list[str]
    attributed_grade_filters: list[str]


class FetchHadithRecordsIntentV1(StrictModel):
    operation: Literal["FETCH_HADITH_RECORDS"]
    target: Literal[
        "DETERMINISTIC_HADITH_REFERENCES",
        "RESOLVED_HADITH_CANDIDATES",
        "HADITH_SEARCH_CANDIDATES",
    ]
    language: str


class FetchTafsirPassagesIntentV1(StrictModel):
    operation: Literal["FETCH_TAFSIR_PASSAGES"]
    target: Literal["DETERMINISTIC_QURAN_REFERENCES", "QURAN_SEARCH_CANDIDATES"]
    mode: TafsirFetchMode
    tafsir_resource_aliases: list[str]
    language: str


class FetchHadithCommentaryIntentV1(StrictModel):
    operation: Literal["FETCH_HADITH_COMMENTARY"]
    target: Literal[
        "DETERMINISTIC_HADITH_REFERENCES",
        "RESOLVED_HADITH_CANDIDATES",
        "HADITH_SEARCH_CANDIDATES",
    ]
    commentary_resource_alias: str
    language: str


ToolCallIntentV1 = Annotated[
    ResolveQuranReferenceIntentV1
    | SearchQuranTopicsIntentV1
    | FetchQuranPassageIntentV1
    | ResolveHadithIdentityIntentV1
    | SearchHadithTopicsIntentV1
    | FetchHadithRecordsIntentV1
    | FetchTafsirPassagesIntentV1
    | FetchHadithCommentaryIntentV1,
    Field(discriminator="operation"),
]


class SemanticTargetKind(StrEnum):
    TOPIC = "TOPIC"
    QURAN_PASSAGE = "QURAN_PASSAGE"
    HADITH = "HADITH"
    CROSS_SOURCE_TOPIC = "CROSS_SOURCE_TOPIC"
    UNSUPPORTED = "UNSUPPORTED"


class SemanticTargetSummaryV1(StrictModel):
    kind: SemanticTargetKind
    normalized_label: str | None
    entity_labels: list[str]


class InheritedContextUseV1(StrictModel):
    field: ContextField
    source_turn_id: UUID


class ResearchPlanV1(StrictModel):
    schema_version: Literal["1"]
    intent: Intent
    response_language: str
    semantic_target_summary: SemanticTargetSummaryV1
    inherited_context: list[InheritedContextUseV1]
    clarification: ClarificationStateV1
    scope_signals: set[ScopeSignal]
    requested_operations: list[ToolCallIntentV1] = Field(max_length=6)


class PartialFailurePolicy(StrEnum):
    REQUIRE_ALL = "REQUIRE_ALL"
    ALLOW_DECLARED_SOURCE_PARTIAL = "ALLOW_DECLARED_SOURCE_PARTIAL"


class EvidenceRequirementsV1(StrictModel):
    required_source_kinds: set[EvidenceSourceKind]
    minimum_items_by_kind: dict[EvidenceSourceKind, int]
    require_exact_quran: bool
    require_exact_hadith: bool
    required_tafsir_aliases: list[str]
    require_two_distinct_tafsir_resources: bool
    allow_partial: bool


class EvidenceRequirementFailureReason(StrEnum):
    MISSING = "MISSING"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"
    RIGHTS_BLOCKED = "RIGHTS_BLOCKED"
    WRONG_REVISION = "WRONG_REVISION"


class EvidenceRequirementFailureV1(StrictModel):
    source_kind: EvidenceSourceKind | None
    resource_alias: str | None
    reason: EvidenceRequirementFailureReason


class LateBindingSelect(StrEnum):
    QURAN_CANDIDATE_IDS = "QURAN_CANDIDATE_IDS"
    HADITH_CANDIDATE_IDS = "HADITH_CANDIDATE_IDS"
    RESOLVED_QURAN_REFERENCES = "RESOLVED_QURAN_REFERENCES"
    RESOLVED_HADITH_REFERENCES = "RESOLVED_HADITH_REFERENCES"


class ValidatedLateBindingV1(StrictModel):
    upstream_node_id: str
    # Topic retrieval may fan out to a small number of independently stated
    # coverage obligations.  The first source remains explicit for backwards
    # compatible audit records; these are the additional, bounded search nodes
    # whose candidates are fused before exact hydration.
    additional_upstream_node_ids: list[str] = Field(default_factory=list, max_length=2)
    select: LateBindingSelect
    maximum_items: int = Field(gt=0)


class SearchQuranHandlerInputV1(StrictModel):
    query: str
    language: str
    surah_filters: list[int]


class FetchQuranHandlerInputV1(StrictModel):
    verse_keys: list[VerseKey]
    late_binding: ValidatedLateBindingV1 | None
    translation_resource_alias: str | None
    include_adjacent_verses: bool


class FetchTafsirHandlerInputV1(StrictModel):
    # Exact references are supplied directly.  Topic plans bind this to the
    # non-citable search candidates at execution time; the fetched Tafsir is
    # still exact, source-scoped evidence.
    verse_keys: list[VerseKey] = Field(default_factory=list)
    late_binding: ValidatedLateBindingV1 | None = None
    mode: TafsirFetchMode
    tafsir_resource_aliases: list[str] = Field(min_length=1, max_length=2)
    language: str
    character_budget: int = Field(gt=0, le=50_000)

    @model_validator(mode="after")
    def validate_target(self) -> FetchTafsirHandlerInputV1:
        if bool(self.verse_keys) == bool(self.late_binding):
            raise ValueError("Tafsir hydration requires exactly one verse target")
        return self


ValidatedHandlerInputV1 = (
    SearchQuranHandlerInputV1 | FetchQuranHandlerInputV1 | FetchTafsirHandlerInputV1
)


class ExecutionNodeV1(StrictModel):
    node_id: str
    tool_id: str
    validated_input: ValidatedHandlerInputV1
    depends_on: list[str]
    timeout_ms: int = Field(gt=0)
    result_limit: int = Field(gt=0)


class ExecutionPlanV1(StrictModel):
    schema_version: Literal["1"]
    intent: Intent
    nodes: list[ExecutionNodeV1] = Field(max_length=6)
    evidence_requirements: EvidenceRequirementsV1
    overall_deadline_ms: int = Field(gt=0)
    partial_failure_policy: PartialFailurePolicy


class ToolResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class ToolErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    RIGHTS_POLICY_BLOCKED = "RIGHTS_POLICY_BLOCKED"
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"
    INTERNAL = "INTERNAL"


class ToolError(StrictModel):
    code: ToolErrorCode
    retryable: bool
    safe_message_key: str
    upstream_status: int | None
    retry_after_ms: int | None = Field(default=None, ge=0)


class ToolWarningCode(StrEnum):
    PARTIAL_RESULTS = "PARTIAL_RESULTS"
    TRUNCATED_BY_POLICY = "TRUNCATED_BY_POLICY"
    OPTIONAL_RESOURCE_UNAVAILABLE = "OPTIONAL_RESOURCE_UNAVAILABLE"
    STALE_OPTIONAL_INDEX = "STALE_OPTIONAL_INDEX"
    RATE_LIMIT_APPROACHING = "RATE_LIMIT_APPROACHING"
    ATTRIBUTION_NOTICE = "ATTRIBUTION_NOTICE"


JSONPrimitive = str | int | float | bool | None


class ToolWarning(StrictModel):
    code: ToolWarningCode
    safe_message_key: str
    metadata: dict[str, JSONPrimitive]


class QuranSearchCandidateV1(StrictModel):
    candidate_id: str
    verse_keys: list[VerseKey] = Field(min_length=1)
    index_revision: SourceRevisionIdentity
    ranking: RetrievalProvenanceV1


class EvidenceItem(StrictModel):
    evidence_id: str
    source_kind: EvidenceSourceKind
    canonical_pointer: CanonicalPointer
    source_identity: SourceIdentityV1
    source_revision: SourceRevisionIdentity
    language: str
    exact_text: str
    approved_quote_spans: list[ApprovedSpanV1]
    reference: TypedSourceReference
    attributions: list[AttributionV1]
    retrieval: RetrievalProvenanceV1
    content_hash: str
    source_policy_alias: str
    # WP-0 (V3 plan): hadith/athar takhrij extracted from this tafsir excerpt's own footnotes,
    # rendered verbatim as part of the excerpt's attribution -- never an independent hadith
    # lookup. Additive and empty by default so every other producer of EvidenceItem is
    # unaffected.
    hadith_references: list[str] = Field(default_factory=list)


class ToolResult[T](StrictModel):
    status: ToolResultStatus
    data: T | None
    evidence_items: list[EvidenceItem]
    warnings: list[ToolWarning]
    duration_ms: int = Field(ge=0)
    error: ToolError | None


class EvidenceSet(StrictModel):
    schema_version: Literal["1"]
    turn_id: UUID
    items_by_id: dict[str, EvidenceItem]
    requirements: EvidenceRequirementsV1
    requirements_satisfied: bool
    missing_requirements: list[EvidenceRequirementFailureV1]
    permitted_synthesis_deployments: set[str]

    @model_validator(mode="after")
    def validate_ledger(self) -> EvidenceSet:
        if any(key != item.evidence_id for key, item in self.items_by_id.items()):
            raise ValueError("Evidence map key must equal the item's evidence_id")
        if self.requirements_satisfied and self.missing_requirements:
            raise ValueError("Satisfied requirements cannot have missing requirements")
        return self


class PersistedEvidenceRecordV1(StrictModel):
    schema_version: Literal["1"]
    turn_id: UUID
    evidence_id: str
    canonical_pointer: CanonicalPointer
    source_identity: SourceIdentityV1
    source_revision: SourceRevisionIdentity
    language: str
    reference: TypedSourceReference
    attributions: list[AttributionV1]
    retrieval: RetrievalProvenanceV1
    content_hash: str
    permitted_snapshot: str | None
    hadith_references: list[str] = Field(default_factory=list)


class ClaimType(StrEnum):
    FACTUAL_SUMMARY = "FACTUAL_SUMMARY"
    INTERPRETIVE_SUMMARY = "INTERPRETIVE_SUMMARY"
    COMPARISON = "COMPARISON"
    COVERAGE_LIMITATION = "COVERAGE_LIMITATION"


class QuoteMode(StrEnum):
    FULL = "FULL"
    APPROVED_SPAN = "APPROVED_SPAN"


class ClaimBlockV1(StrictModel):
    block_type: Literal["CLAIM"]
    claim_id: str
    text: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(min_length=1)


class QuoteReferenceV1(StrictModel):
    evidence_id: str
    approved_span_id: str | None


class SourceVisibility(StrEnum):
    """Whether a source is initially shown or available as collapsed context."""

    VISIBLE = "VISIBLE"
    COLLAPSED_CONTEXT = "COLLAPSED_CONTEXT"


class SourceBlockRequestV1(StrictModel):
    block_type: Literal["SOURCE_BLOCK_REQUEST"]
    block_id: str
    quote_reference: QuoteReferenceV1
    quote_mode: QuoteMode
    # Source-view presentation metadata is application-authored.  It does not alter
    # the requested source or its quote span; it only tells a renderer whether a
    # retained structural-context source is initially visible to the reader.
    visibility: SourceVisibility = SourceVisibility.VISIBLE
    passage_group_id: str | None = None


class SourceViewHeadingRequestV1(StrictModel):
    """An application-owned heading for a deterministic source-only response."""

    block_type: Literal["SOURCE_VIEW_HEADING_REQUEST"]
    text: str


class NoticeCode(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    QUESTION_NOT_UNDERSTOOD = "QUESTION_NOT_UNDERSTOOD"
    ANSWER_NOT_VERIFIED = "ANSWER_NOT_VERIFIED"
    CLARIFICATION = "CLARIFICATION"
    UNSUPPORTED_SCOPE = "UNSUPPORTED_SCOPE"


class NoticeBlockV1(StrictModel):
    block_type: Literal["NOTICE"]
    notice_code: NoticeCode
    text: str


AnswerDraftBlockV1 = Annotated[
    ClaimBlockV1 | SourceBlockRequestV1 | SourceViewHeadingRequestV1 | NoticeBlockV1,
    Field(discriminator="block_type"),
]


class AnswerDraftV1(StrictModel):
    schema_version: Literal["1"]
    status: AnswerStatus
    blocks: list[AnswerDraftBlockV1] = Field(max_length=24)


class CitationV1(StrictModel):
    citation_id: str
    evidence_id: str
    display_label: str
    source_kind: EvidenceSourceKind
    reference: TypedSourceReference
    attribution: list[AttributionV1]
    canonical_link: str | None
    source_revision: SourceRevisionIdentity


class RenderedClaimBlockV1(StrictModel):
    block_type: Literal["CLAIM"]
    claim_id: str
    text: str
    citations: list[CitationV1] = Field(min_length=1)


class RenderedSourceBlockV1(StrictModel):
    block_type: Literal["SOURCE"]
    block_id: str
    exact_text: str
    citation: CitationV1
    visibility: SourceVisibility = SourceVisibility.VISIBLE
    passage_group_id: str | None = None


class RenderedSourceViewHeadingBlockV1(StrictModel):
    """A standard stacked heading preceding deterministic source material."""

    block_type: Literal["SOURCE_VIEW_HEADING"]
    text: str


class RenderedNoticeBlockV1(StrictModel):
    block_type: Literal["NOTICE"]
    notice_code: NoticeCode
    text: str


# WP-2 (Conversational Quality Technical Plan V3): the composed narrative. Additive contract --
# every existing block type and consumer is unaffected. `supporting_claim_ids` is model-authored
# (which verified claims this segment is organising); `supporting_evidence_ids` is always
# derived in code from those claims' own `evidence_ids`, never emitted by the model, so a
# segment can never point at evidence its claims did not already cite.
class NarrativeSegmentKind(StrEnum):
    FRAMING = "FRAMING"
    POINT = "POINT"
    TRANSITION = "TRANSITION"
    SUMMARY = "SUMMARY"
    LIMITATION = "LIMITATION"


class NarrativeSegmentV1(StrictModel):
    segment_id: str
    text: str
    kind: NarrativeSegmentKind
    supporting_claim_ids: list[str]
    supporting_evidence_ids: list[str]


class RenderedNarrativeBlockV1(StrictModel):
    block_type: Literal["NARRATIVE"]
    segments: list[NarrativeSegmentV1] = Field(min_length=1)


# The composer's own proposal schema (model-authored). `NarrativeSegmentV1` above is the
# rendered/derived form the response carries; this is what the LLM actually returns.
class NarrativeSegmentDraftV1(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    kind: NarrativeSegmentKind
    # No default: every field in an LLM-facing structured-output schema must be always-required
    # (Groq's strict JSON-schema dialect rejects a schema with an optional property) -- the
    # model must always emit the key, using [] when a segment supports no claim directly
    # (e.g. FRAMING/TRANSITION).
    supporting_claim_ids: list[str] = Field(max_length=16)


class NarrativeDraftV1(StrictModel):
    schema_version: Literal["1"]
    segments: list[NarrativeSegmentDraftV1] = Field(min_length=1, max_length=12)


RenderedAnswerBlockV1 = Annotated[
    RenderedClaimBlockV1
    | RenderedSourceBlockV1
    | RenderedSourceViewHeadingBlockV1
    | RenderedNoticeBlockV1
    | RenderedNarrativeBlockV1,
    Field(discriminator="block_type"),
]


def derive_rendered_output_kind(blocks: object) -> RenderedOutputKind:
    """Classify rendered answer material without relying on the producing branch."""

    if not isinstance(blocks, list):
        return RenderedOutputKind.NO_CONTENT

    def value(block: object, field: str) -> object:
        return block.get(field) if isinstance(block, dict) else getattr(block, field, None)

    def template_candidate_texts(block: object) -> list[str]:
        texts = [value(block, "text")]
        segments = value(block, "segments")
        if isinstance(segments, list):
            texts.extend(value(segment, "text") for segment in segments)
        return [text.strip() for text in texts if isinstance(text, str) and text.strip()]

    candidate_texts = [text for block in blocks for text in template_candidate_texts(block)]
    claim_texts = [
        text
        for block in blocks
        if value(block, "block_type") == "CLAIM"
        for text in template_candidate_texts(block)
    ]
    has_narrative = any(value(block, "block_type") == "NARRATIVE" for block in blocks)
    if any(is_deterministic_contextual_topic_template_prose(text) for text in candidate_texts):
        return RenderedOutputKind.TEMPLATE
    if claim_texts or has_narrative:
        return RenderedOutputKind.SUBSTANTIVE_SYNTHESIS
    if any(value(block, "block_type") == "SOURCE" for block in blocks):
        return RenderedOutputKind.SOURCE_ONLY
    return RenderedOutputKind.NO_CONTENT


class ResearchResponseV1(StrictModel):
    schema_version: Literal["1"]
    conversation_id: UUID
    turn_id: UUID
    correlation_id: UUID
    status: AnswerStatus
    output_kind: RenderedOutputKind
    blocks: list[RenderedAnswerBlockV1]
    citations: list[CitationV1]
    clarification: ClarificationStateV1 | None
    warnings: list[ToolWarning]

    @model_validator(mode="before")
    @classmethod
    def derive_output_kind_from_rendered_blocks(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        rendered = dict(value)
        rendered["output_kind"] = derive_rendered_output_kind(rendered.get("blocks"))
        return rendered


class TraceEventV1(StrictModel):
    """A bounded, redacted execution event for the owner-only development trace."""

    seq: int = Field(ge=1)
    elapsed_ms: int = Field(ge=0)
    node: Literal[
        "scope",
        "planner",
        "retrieval",
        "fusion",
        "tafsir",
        "rerank",
        "evidence",
        "verification",
        "synthesis",
        "composition",
        "gate",
    ]
    phase: Literal["started", "completed", "degraded", "skipped", "failed", "rejected"]
    summary: str = Field(min_length=1, max_length=240)
    detail: dict[str, object] = Field(default_factory=dict, max_length=24)
    provider: dict[str, object] | None = None


class ResearchRequest(StrictModel):
    message: str = Field(min_length=2, max_length=2000)
    locale: str = "en"
    conversation_id: UUID = Field(default_factory=uuid4)


class ConversationCreateRequest(StrictModel):
    locale: Literal["ar", "en"] = "ar"
    title: str | None = Field(default=None, min_length=1, max_length=120)


class ConversationUpdateRequest(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    archived: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> ConversationUpdateRequest:
        if self.title is None and self.archived is None:
            raise ValueError("At least one conversation field must be changed")
        return self


class ConversationTurnCreateRequest(StrictModel):
    message: str = Field(min_length=2, max_length=2000)
    locale: Literal["ar", "en"] = "ar"
    idempotency_key: str = Field(min_length=8, max_length=128)


class ConversationSummaryV1(StrictModel):
    conversation_id: UUID
    title: str
    locale: Literal["ar", "en"]
    archived: bool
    created_at: datetime
    updated_at: datetime
    turn_count: int = Field(ge=0)


class ConversationTurnV1(StrictModel):
    turn_id: UUID
    user_message: str
    locale: Literal["ar", "en"]
    response: ResearchResponseV1
    created_at: datetime
    trace_events: list[TraceEventV1] = Field(default_factory=list, max_length=200)


class ConversationDetailV1(ConversationSummaryV1):
    turns: list[ConversationTurnV1]


class ConversationPageV1(StrictModel):
    items: list[ConversationSummaryV1]
    next_cursor: str | None


class ConversationTurnPageV1(StrictModel):
    items: list[ConversationTurnV1]
    next_cursor: str | None


class EffectiveDeploymentIdentityV1(StrictModel):
    deployment_id: str
    gateway_provider_id: str
    effective_provider_id: str
    model_id: str
    resolved_model_version_or_name: str | None
    region: str | None
    subprocessor_ids: set[str]
    zero_data_retention_enforced: bool
    provider_training_allowed: bool
    route_policy_id: str


class ProviderCompletion(StrictModel):
    effective_deployment: EffectiveDeploymentIdentityV1
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    billed_cost_usd: Decimal | None
    paid_route: bool
    fallback_used: bool
