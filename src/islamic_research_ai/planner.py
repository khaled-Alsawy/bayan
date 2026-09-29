from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from islamic_research_ai.contracts import (
    ClarificationCandidateV1,
    ClarificationDetail,
    ClarificationField,
    ClarificationReason,
    ClarificationStateV1,
    DeterministicResolutionContextV1,
    EvidenceSet,
    FetchQuranPassageIntentV1,
    FetchTafsirPassagesIntentV1,
    Intent,
    ProviderCompletion,
    ResearchPlanV1,
    ResolutionIssueCode,
    ResolutionIssueV1,
    ResolutionProvenance,
    ResolvedQuranReferenceV1,
    ScopeSignal,
    SearchQuranTopicsIntentV1,
    SemanticTargetKind,
    SemanticTargetSummaryV1,
    TafsirFetchMode,
    ToolOperation,
    VerseKey,
    WorkingContextV1,
    no_clarification,
)
from islamic_research_ai.ports import (
    ChatModelPort,
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.retrieval.relevance import (
    SemanticSubtarget,
    SupportedFacetSummary,
    TopicRelevanceAssessment,
    assess_topic_evidence_from_matches,
    conservative_compound_anchor_matches,
    evidence_passage_key,
)
from islamic_research_ai.routing import ProviderRouteError, ProviderRoutePolicy
from islamic_research_ai.synthesis import GroundingError, validate_supported_facet_summary

_QURAN_REFERENCE = re.compile(
    r"\b(?:qur(?:an|'an)\s*)?(\d{1,3})\s*[:/]\s*(\d{1,3})\b", re.IGNORECASE
)

_QUERY_TYPOS = {
    "koran": "quran",
    "quraan": "quran",
    "forgivness": "forgiveness",
    "forgivenes": "forgiveness",
    "adversety": "adversity",
}


def normalize_user_query(message: str) -> str:
    normalized = normalize_arabic(normalize_translation(message))
    for typo, replacement in _QUERY_TYPOS.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", replacement, normalized)
    return normalized


def build_deterministic_plan(
    message: str,
    *,
    response_language: str,
    resolution: DeterministicResolutionContextV1,
    enable_topics: bool = True,
) -> ResearchPlanV1 | None:
    """Build only high-confidence plans; all other semantics remain provider-planned."""

    del enable_topics
    normalized = normalize_user_query(message)
    tafsir_markers = re.compile(r"\b(?:explain|interpret|tafsir|compare)\b|(?:فسر|تفسير|اشرح|قارن)")
    exact_display_markers = re.compile(
        r"\b(?:show|display|quote|translation|exact text)\b|"
        r"\bdoes\s+(?:the\s+)?quran\s+\d{1,3}\s*[:/]\s*\d{1,3}\s+say\b|"
        r"(?:اعرض|اظهر|اقتبس|النص|ترجمه)"
    )
    if (
        resolution.resolved_quran_references
        and not tafsir_markers.search(normalized)
        and exact_display_markers.search(normalized)
    ):
        label = ", ".join(item.canonical_label for item in resolution.resolved_quran_references)
        return ResearchPlanV1(
            schema_version="1",
            intent=Intent.QURAN_PASSAGE_LOOKUP,
            response_language=response_language,
            semantic_target_summary=SemanticTargetSummaryV1(
                kind=SemanticTargetKind.QURAN_PASSAGE,
                normalized_label=label,
                entity_labels=[label],
            ),
            inherited_context=[],
            clarification=no_clarification(),
            scope_signals=set(resolution.deterministic_scope_signals),
            requested_operations=[
                FetchQuranPassageIntentV1(
                    operation="FETCH_QURAN_PASSAGE",
                    target="DETERMINISTIC_QURAN_REFERENCES",
                    translation_resource_alias="quranenc-rowwad-en-v1.0.19",
                    include_adjacent_verses=False,
                )
            ],
        )

    return None


def resolve_deterministically(message: str) -> DeterministicResolutionContextV1:
    resolved: list[ResolvedQuranReferenceV1] = []
    issues: list[ResolutionIssueV1] = []
    for raw_chapter, raw_verse in _QURAN_REFERENCE.findall(message):
        chapter, verse = int(raw_chapter), int(raw_verse)
        if not 1 <= chapter <= 114 or verse < 1:
            issues.append(
                ResolutionIssueV1(
                    code=ResolutionIssueCode.INVALID_RANGE,
                    safe_message_key="quran_reference.invalid_range",
                    candidate_ids=[],
                )
            )
            continue
        key = VerseKey(chapter=chapter, verse=verse)
        resolved.append(
            ResolvedQuranReferenceV1(
                verse_keys=[key],
                canonical_label=f"{chapter}:{verse}",
                provenance=ResolutionProvenance.CURRENT_MESSAGE_EXACT_PARSE,
            )
        )
    return DeterministicResolutionContextV1(
        schema_version="1",
        resolved_quran_references=resolved,
        quran_candidates=[],
        resolved_hadith_references=[],
        hadith_candidates=[],
        deterministic_scope_signals=set(),
        issues=issues,
    )


class PlanValidationError(ValueError):
    pass


class PlannerFailedSafe(RuntimeError):
    def __init__(
        self,
        message: str,
        completions: list[ProviderCompletion],
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.completions = completions
        self.cause = cause


class PlannerTemporarilyUnavailable(RuntimeError):
    def __init__(
        self,
        diagnostic: ProviderFailureDiagnostic,
        completions: list[ProviderCompletion],
    ) -> None:
        self.diagnostic = diagnostic
        self.completions = completions
        super().__init__(
            f"Planner required provider temporarily unavailable: {diagnostic.safe_detail}"
        )


class QuranPlannerProposalV1(BaseModel):
    """Small provider transport for the implemented Quran vertical slice.

    The full cross-domain ResearchPlanV1 remains the canonical planner output. Keeping
    this transport local to the Groq planner prevents unimplemented Hadith/Tafsir union
    branches from burdening constrained decoding for a Quran-only request.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    intent: Literal[
        "QURAN_TOPIC_SEARCH",
        "QURAN_PASSAGE_LOOKUP",
        "EXPLAIN_QURAN_VERSE",
        "COMPARE_TAFSIR",
    ]
    response_language: str
    semantic_target_label: str
    entity_labels: list[str]
    requested_operations: list[
        Literal[
            "SEARCH_QURAN_TOPICS",
            "FETCH_QURAN_PASSAGE",
            "FETCH_TAFSIR_PASSAGES",
        ]
    ]
    search_query: str
    fetch_target: Literal["DETERMINISTIC_QURAN_REFERENCES", "QURAN_SEARCH_CANDIDATES"]
    tafsir_mode: Literal["EXPLANATION_ONE", "COMPARISON_EXACTLY_TWO"] | None
    tafsir_resource_aliases: list[str]
    clarification_required: bool
    observable_ambiguity: (
        Literal[
            "POLYSEMY",
            "MISSING_REQUIRED_TARGET",
            "CONFLICTING_MEANINGS",
            "UNKNOWN_TAFSIR_ALIAS",
            "UNRESOLVED_REFERENCE",
            "AMBIGUOUS_INHERITED_CONTEXT",
        ]
        | None
    )
    clarification_question: str
    clarification_candidates: list[str]

    def to_canonical(self, *, scope_signals: set[ScopeSignal]) -> ResearchPlanV1:
        operations: list[
            SearchQuranTopicsIntentV1 | FetchQuranPassageIntentV1 | FetchTafsirPassagesIntentV1
        ] = []
        if self.clarification_required:
            if self.observable_ambiguity is None:
                raise PlanValidationError("Clarification requires an observable ambiguity")
            if self.requested_operations or self.search_query:
                raise PlanValidationError("Clarification proposals cannot request retrieval")
            if len(self.clarification_candidates) not in {0, 2, 3}:
                raise PlanValidationError("Clarification choices must contain 2–3 meanings")
            reason, detail, field = {
                "POLYSEMY": (
                    ClarificationReason.AMBIGUOUS_REFERENCE,
                    ClarificationDetail.MULTIPLE_REFERENCE_MATCHES,
                    ClarificationField.TOPIC,
                ),
                "MISSING_REQUIRED_TARGET": (
                    ClarificationReason.MISSING_REFERENCE,
                    ClarificationDetail.OTHER,
                    ClarificationField.TOPIC,
                ),
                "CONFLICTING_MEANINGS": (
                    ClarificationReason.AMBIGUOUS_CONTEXT,
                    ClarificationDetail.CONTEXT_HAS_MULTIPLE_TARGETS,
                    ClarificationField.TOPIC,
                ),
                "UNKNOWN_TAFSIR_ALIAS": (
                    ClarificationReason.UNKNOWN_RESOURCE,
                    ClarificationDetail.RESOURCE_NOT_APPROVED,
                    ClarificationField.RESOURCE,
                ),
                "UNRESOLVED_REFERENCE": (
                    ClarificationReason.MISSING_REFERENCE,
                    ClarificationDetail.MISSING_VERSE_NUMBER,
                    ClarificationField.QURAN_REFERENCE,
                ),
                "AMBIGUOUS_INHERITED_CONTEXT": (
                    ClarificationReason.AMBIGUOUS_CONTEXT,
                    ClarificationDetail.CONTEXT_HAS_MULTIPLE_TARGETS,
                    ClarificationField.TOPIC,
                ),
            }[self.observable_ambiguity]
            return ResearchPlanV1(
                schema_version="1",
                intent=self.intent,
                response_language=self.response_language,
                semantic_target_summary=SemanticTargetSummaryV1(
                    kind=SemanticTargetKind.TOPIC,
                    normalized_label=self.semantic_target_label or None,
                    entity_labels=self.entity_labels,
                ),
                inherited_context=[],
                clarification=ClarificationStateV1(
                    required=True,
                    reason=reason,
                    detail=detail,
                    missing_fields=[field],
                    candidates=[
                        ClarificationCandidateV1(
                            candidate_id=f"meaning-{index}",
                            label=label,
                            reference=None,
                        )
                        for index, label in enumerate(self.clarification_candidates, start=1)
                    ],
                    question=self.clarification_question,
                ),
                scope_signals=scope_signals,
                requested_operations=[],
            )
        if any(
            (
                self.observable_ambiguity is not None,
                bool(self.clarification_question),
                bool(self.clarification_candidates),
            )
        ):
            raise PlanValidationError("Non-clarification proposal contains ambiguity fields")
        if self.requested_operations == [
            "SEARCH_QURAN_TOPICS",
            "FETCH_QURAN_PASSAGE",
        ]:
            if not self.search_query:
                raise PlanValidationError("Topic search requires a query")
            operations.append(
                SearchQuranTopicsIntentV1(
                    operation="SEARCH_QURAN_TOPICS",
                    query=self.search_query,
                    language=self.response_language,
                    surah_filters=[],
                )
            )
            target_kind = SemanticTargetKind.TOPIC
        elif self.requested_operations == [
            "SEARCH_QURAN_TOPICS",
            "FETCH_QURAN_PASSAGE",
            "FETCH_TAFSIR_PASSAGES",
        ]:
            if not self.search_query or self.tafsir_mode is None:
                raise PlanValidationError(
                    "Contextual topic research requires query and Tafsir mode"
                )
            target_kind = SemanticTargetKind.TOPIC
        elif self.requested_operations == ["FETCH_QURAN_PASSAGE"]:
            if self.search_query:
                raise PlanValidationError("Passage lookup contains an unexpected search query")
            target_kind = SemanticTargetKind.QURAN_PASSAGE
        elif self.requested_operations == [
            "FETCH_QURAN_PASSAGE",
            "FETCH_TAFSIR_PASSAGES",
        ]:
            if self.search_query:
                raise PlanValidationError("Exact Tafsir intent contains a search query")
            if self.tafsir_mode is None:
                raise PlanValidationError("Tafsir operation requires an explicit mode")
            target_kind = SemanticTargetKind.QURAN_PASSAGE
        else:
            raise PlanValidationError("Planner proposed an invalid semantic operation sequence")

        operations.append(
            FetchQuranPassageIntentV1(
                operation="FETCH_QURAN_PASSAGE",
                target=self.fetch_target,
                translation_resource_alias="quranenc-rowwad-en-v1.0.19",
                include_adjacent_verses=(target_kind is SemanticTargetKind.TOPIC),
            )
        )
        if self.requested_operations[-1] == "FETCH_TAFSIR_PASSAGES":
            assert self.tafsir_mode is not None
            operations.append(
                FetchTafsirPassagesIntentV1(
                    operation="FETCH_TAFSIR_PASSAGES",
                    target=(
                        "QURAN_SEARCH_CANDIDATES"
                        if target_kind is SemanticTargetKind.TOPIC
                        else "DETERMINISTIC_QURAN_REFERENCES"
                    ),
                    mode=TafsirFetchMode(self.tafsir_mode),
                    tafsir_resource_aliases=self.tafsir_resource_aliases,
                    language=self.response_language,
                )
            )
        return ResearchPlanV1(
            schema_version="1",
            intent=self.intent,
            response_language=self.response_language,
            semantic_target_summary=SemanticTargetSummaryV1(
                kind=target_kind,
                normalized_label=self.semantic_target_label,
                entity_labels=self.entity_labels,
            ),
            inherited_context=[],
            clarification=no_clarification(),
            scope_signals=scope_signals,
            requested_operations=operations,
        )


class PlanValidator:
    _APPROVED_TRANSLATION = "quranenc-rowwad-en-v1.0.19"
    _APPROVED_TAFSIR = {
        # WP-0 (V3 plan): QUL evidence-tier resources, full-Quran coverage, Arabic only.
        # Al-Mukhtasar is deliberately absent here -- it is the embedded search tier, not an
        # evidence-tier citation source (see the two-tier design in the V3 plan, WP-0).
        "qul-tafsir-ibnkathir-ar-v1",
        "qul-tafsir-alqurtubi-ar-v1",
        "qul-tafsir-albaghawi-ar-v1",
    }

    def validate(
        self,
        plan: ResearchPlanV1,
        *,
        working_context: WorkingContextV1,
        deterministic_resolution: DeterministicResolutionContextV1,
    ) -> ResearchPlanV1:
        if plan.scope_signals != deterministic_resolution.deterministic_scope_signals:
            raise PlanValidationError("Plan conflicts with deterministic scope signals")
        response_languages = {
            value.value
            for value in working_context.values
            if value.field.value == "RESPONSE_LANGUAGE" and isinstance(value.value, str)
        }
        if response_languages and plan.response_language not in response_languages:
            raise PlanValidationError("Plan response language conflicts with working context")
        if plan.clarification.required:
            self._validate_clarification(plan)
            return plan
        operations = [ToolOperation(item.operation) for item in plan.requested_operations]
        if len(operations) != len(set(operations)):
            raise PlanValidationError("Duplicate semantic operations are forbidden")

        if plan.intent is Intent.QURAN_TOPIC_SEARCH:
            expected = [ToolOperation.SEARCH_QURAN_TOPICS, ToolOperation.FETCH_QURAN_PASSAGE]
            expected_with_tafsir = [*expected, ToolOperation.FETCH_TAFSIR_PASSAGES]
            if tuple(operations) not in {tuple(expected), tuple(expected_with_tafsir)}:
                raise PlanValidationError(
                    "Quran topic search requires SEARCH then FETCH, with optional Tafsir"
                )
            search, fetch = plan.requested_operations[:2]
            if not isinstance(search, SearchQuranTopicsIntentV1):
                raise PlanValidationError("First semantic operation must be Quran topic search")
            if search.language != plan.response_language:
                raise PlanValidationError("Search language conflicts with response language")
            self._validate_fetch(fetch, expected_target="QURAN_SEARCH_CANDIDATES")
            if len(plan.requested_operations) == 3:
                self._validate_tafsir_fetch(
                    plan.requested_operations[2],
                    mode=TafsirFetchMode.EXPLANATION_ONE,
                    response_language=plan.response_language,
                    expected_target="QURAN_SEARCH_CANDIDATES",
                )
        elif plan.intent is Intent.QURAN_PASSAGE_LOOKUP:
            if not deterministic_resolution.resolved_quran_references:
                raise PlanValidationError("Passage lookup requires a deterministic Quran reference")
            if operations != [ToolOperation.FETCH_QURAN_PASSAGE]:
                raise PlanValidationError("Resolved passage lookup must bypass semantic search")
            self._validate_fetch(
                plan.requested_operations[0],
                expected_target="DETERMINISTIC_QURAN_REFERENCES",
            )
        elif plan.intent in {Intent.EXPLAIN_QURAN_VERSE, Intent.COMPARE_TAFSIR}:
            if not deterministic_resolution.resolved_quran_references:
                raise PlanValidationError("Tafsir intents require a deterministic Quran reference")
            if operations != [
                ToolOperation.FETCH_QURAN_PASSAGE,
                ToolOperation.FETCH_TAFSIR_PASSAGES,
            ]:
                raise PlanValidationError("Tafsir intents require Quran fetch then Tafsir fetch")
            quran_fetch, tafsir_fetch = plan.requested_operations
            self._validate_fetch(
                quran_fetch,
                expected_target="DETERMINISTIC_QURAN_REFERENCES",
            )
            self._validate_tafsir_fetch(
                tafsir_fetch,
                mode=(
                    TafsirFetchMode.EXPLANATION_ONE
                    if plan.intent is Intent.EXPLAIN_QURAN_VERSE
                    else TafsirFetchMode.COMPARISON_EXACTLY_TWO
                ),
                response_language=plan.response_language,
            )
        else:
            raise PlanValidationError("Intent is not implemented in the hardened Quran core")
        return plan

    @staticmethod
    def _validate_clarification(plan: ResearchPlanV1) -> None:
        state = plan.clarification
        if plan.requested_operations:
            raise PlanValidationError("Clarification plans must not request retrieval operations")
        if not state.missing_fields:
            raise PlanValidationError("Clarification plans must identify the unresolved field")
        if len(state.candidates) not in {0, 2, 3}:
            raise PlanValidationError("Clarification choices must be absent or contain 2–3 items")
        candidate_ids = [candidate.candidate_id for candidate in state.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise PlanValidationError("Clarification candidate IDs must be unique")
        question = (state.question or "").strip()
        if not question or question.count("?") + question.count("؟") > 1:
            raise PlanValidationError("Clarification must contain exactly one focused question")

    def _validate_fetch(self, value: Any, *, expected_target: str) -> None:
        if not isinstance(value, FetchQuranPassageIntentV1):
            raise PlanValidationError("Required fetch operation is missing")
        if value.target != expected_target:
            raise PlanValidationError("Fetch target conflicts with the validated intent")
        if value.translation_resource_alias != self._APPROVED_TRANSLATION:
            raise PlanValidationError("Fetch uses an unapproved translation resource")

    def _validate_tafsir_fetch(
        self,
        value: Any,
        *,
        mode: TafsirFetchMode,
        response_language: str,
        expected_target: str = "DETERMINISTIC_QURAN_REFERENCES",
    ) -> None:
        if not isinstance(value, FetchTafsirPassagesIntentV1):
            raise PlanValidationError("Required Tafsir fetch operation is missing")
        if value.target != expected_target:
            raise PlanValidationError("Tafsir fetch target conflicts with the validated intent")
        if value.mode is not mode:
            raise PlanValidationError("Tafsir fetch mode conflicts with the intent")
        expected_count = 1 if mode is TafsirFetchMode.EXPLANATION_ONE else 2
        if len(value.tafsir_resource_aliases) != expected_count:
            raise PlanValidationError("Tafsir resource count conflicts with the fetch mode")
        if len(set(value.tafsir_resource_aliases)) != expected_count:
            raise PlanValidationError("Tafsir comparison resources must be distinct")
        if not set(value.tafsir_resource_aliases).issubset(self._APPROVED_TAFSIR):
            raise PlanValidationError("Tafsir fetch uses an unapproved resource")
        if not response_language.casefold().startswith("ar"):
            raise PlanValidationError("The available Tafsir resources are Arabic only")
        if value.language != response_language:
            raise PlanValidationError("Tafsir operation language conflicts with response language")


class SemanticAmbiguityProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    clarification_required: bool
    observable_ambiguity: (
        Literal[
            "POLYSEMY",
            "MISSING_REQUIRED_TARGET",
            "CONFLICTING_MEANINGS",
            "UNKNOWN_TAFSIR_ALIAS",
            "UNRESOLVED_REFERENCE",
            "AMBIGUOUS_INHERITED_CONTEXT",
        ]
        | None
    )
    resolved_target_summary: str
    clarification_question: str
    clarification_candidates: list[str]


class SemanticPassageDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    directly_relevant: bool
    matched_target_ids: list[str]
    partial_target_ids: list[str]
    supported_facet_summary: str
    rejection_reason: Literal[
        "NOT_DROPPED",
        "NO_DIRECT_ENTAILMENT",
        "DIFFERENT_TARGET",
        "INSUFFICIENT_CONTEXT",
        "NO_REASON_RECORDED",
    ] = "NO_REASON_RECORDED"


class SemanticRelevanceProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    decisions: list[SemanticPassageDecisionV1]


class SemanticFacetSummaryDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    supported_facet_summary: str


class SemanticFacetSummaryProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    decisions: list[SemanticFacetSummaryDecisionV1]


class SemanticCoverageProposalV1(BaseModel):
    """Model-owned evidence-obligation decomposition for a clear topic request."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    target_labels: list[str] = Field(min_length=1, max_length=3)


class GroqPlanner:
    AMBIGUITY_SYSTEM_PROMPT = """You are the semantic ambiguity resolver for a Quran research POC.
Return only semantic_ambiguity_v1 JSON. Inspect the full user phrase, colloquial language,
morphology, and any pending clarification. Decide whether an observable ambiguity materially
prevents reliable corpus retrieval. Allowed signals are POLYSEMY, MISSING_REQUIRED_TARGET,
CONFLICTING_MEANINGS, UNKNOWN_TAFSIR_ALIAS, UNRESOLVED_REFERENCE, and
AMBIGUOUS_INHERITED_CONTEXT. Never use confidence alone. A request for passages about a bare
polysemous noun with multiple common, disjoint meanings requires clarification when no modifier,
verb, coordination, or surrounding phrase selects a dominant sense. When context selects one
meaning, do not clarify. A request that explicitly coordinates several related targets or
subquestions is compound, not ambiguous: retain every requested part for planning and never ask
the user to choose between parts they already requested together. If clarification is required,
ask exactly one focused question in the
requested response language and give 2–3 short, mutually exclusive, established common meanings
when useful. Include common idiomatic senses used in the user's language and domain; do not invent
obscure senses or offer a mere word-occurrence search unless the user asked for a concordance.
Candidate coverage is part of the decision: before returning only two meanings, check whether a
third materially common disjoint sense exists across ordinary literal usage, natural features,
objects, anatomy, and idiomatic or domain usage as applicable. Use all three slots when it does.
Candidates must be distinct dictionary-level senses, not different corpus themes or settings that
reuse the same sense. Prefer materially common general-language meanings, including applicable
natural-feature or geographic meanings, before narrow domain contexts.
Otherwise set the
signal to null, question to an empty string, candidates to [], and provide a neutral resolved
target summary in the provided response language. Never answer the religious question, quote
scripture, or name a verse."""
    AMBIGUITY_COVERAGE_PROMPT = """Audit a proposed semantic clarification for coverage. Return
only semantic_ambiguity_v1 JSON. Re-read the full user phrase and the proposed candidates. Check
systematically for an omitted materially common disjoint sense across ordinary literal usage,
natural features or geography, objects or artifacts, anatomy, and idiomatic, social, or domain
usage as applicable. Use three short mutually exclusive candidates when three common senses
exist; keep two only when no third common sense exists. Do not add obscure, technical, archaic,
or mere concordance senses. Candidates must be dictionary-level meanings rather than corpus
themes, locations, or narratives that reuse one meaning. Prefer common general-language senses,
including natural-feature or geographic senses when applicable, over narrow domain contexts. If
the proposed candidates merely split explicitly coordinated parts
that the user requested together, reject the clarification and return clarification_required=false
with a resolved summary containing all parts. Otherwise return one localized question covering
the complete candidate set.
Never answer the underlying religious question or name evidence."""
    SYSTEM_PROMPT = """You are the bounded multilingual semantic planner for a Quran research POC.
Understand colloquial wording, morphology, word sense from the full context, and compound
questions generically. Return only strict quran_plan_proposal_v1 JSON. Never answer the user,
name evidence, quote Quran text, or map a topic to a verse. For a topic query, use
QURAN_TOPIC_SEARCH, requested_operations [SEARCH_QURAN_TOPICS, FETCH_QURAN_PASSAGE], a concise
neutral semantic search_query in the provided response_language, and fetch_target
QURAN_SEARCH_CANDIDATES. Use include_adjacent_verses=true for topic retrieval so the application
can retrieve the document context around a matched verse. When the question asks for meaning, an
implied concept, a relationship established across a passage, or an interpretation rather than a
literal word occurrence, Arabic turns may append FETCH_TAFSIR_PASSAGES with EXPLANATION_ONE. Its
target must be QURAN_SEARCH_CANDIDATES and it must name exactly one approved Arabic QUL source:
qul-tafsir-ibnkathir-ar-v1, qul-tafsir-alqurtubi-ar-v1, or qul-tafsir-albaghawi-ar-v1. English
turns must not request Tafsir until an approved English source is available. Do not request
Tafsir for a plain request to display an exact verse. semantic_target_label and entity_labels
must use that same language.
Put only independently answerable obligations in entity_labels as
`part:<neutral semantic target>`; otherwise use ordinary entity labels. Coordination alone does
not make several obligations: retain one complete target when coordinated words are modifiers,
near-synonyms, or otherwise jointly name one inseparable phenomenon, relation, or outcome. Split
when coordinated terms identify distinct phenomena, factual predicates, actions, outcomes, or
separately attributable objects/persons unless direct support for one logically entails direct
support for every other coordinated term. Treat identity or true synonymy as entailment; do not
collapse distinct terms merely because a cultural, technical, or practical association often
links them. Preserve the intended sense
in every independent subtarget so a polysemous word
does not become a bare lexical query. For an exact resolved reference, use QURAN_PASSAGE_LOOKUP,
[FETCH_QURAN_PASSAGE], an empty search_query, and DETERMINISTIC_QURAN_REFERENCES. Copy the
provided BCP47 response_language exactly. A request to explain an exact verse must use
EXPLAIN_QURAN_VERSE with [FETCH_QURAN_PASSAGE, FETCH_TAFSIR_PASSAGES], EXPLANATION_ONE,
and exactly one approved Arabic QUL alias (qul-tafsir-ibnkathir-ar-v1,
qul-tafsir-alqurtubi-ar-v1, or qul-tafsir-albaghawi-ar-v1). English turns must not request
Tafsir until an approved English source is available. A request
to compare Tafsir must use COMPARE_TAFSIR, the same two operations, COMPARISON_EXACTLY_TWO,
and both distinct approved aliases -- for an Arabic comparison prefer two of the three QUL
classical works. Tafsir intents require a resolved exact reference. A request
only to show or translate a verse remains QURAN_PASSAGE_LOOKUP. For non-Tafsir intents, set
tafsir_mode to null and tafsir_resource_aliases to an empty list.

Set clarification_required=true only when an observable ambiguity materially prevents reliable
retrieval: POLYSEMY, MISSING_REQUIRED_TARGET, CONFLICTING_MEANINGS, UNKNOWN_TAFSIR_ALIAS,
UNRESOLVED_REFERENCE, or AMBIGUOUS_INHERITED_CONTEXT. Do not use self-reported confidence.
Apply a lexical-sense check before planning: if the request is merely asking for passages about
a bare noun that has multiple common, disjoint meanings and no modifier or surrounding phrase
selects one meaning, that is observable POLYSEMY and must be clarified before retrieval. Candidate
labels must name those distinct meanings. Conversely, modifiers, coordinated concepts, verbs,
and the whole phrase may select a dominant sense; when they do, proceed without clarification.
Do not clarify merely because a clear request can be framed at a broader or narrower research
level, or because coordinated concepts may later have different evidence coverage. A complete
predicate that names a requested action, effect, or relation selects the relevant sense unless
there are multiple common disjoint senses that remain plausible in that complete predicate.
When clarifying, request no operations, leave search_query empty, ask exactly one short localized
question, and provide 2–3 localized candidate meanings when useful. If the full phrase clearly
resolves a word sense, do not clarify. If pending_clarification is present, resolve the current
user reply only against those candidate meanings, then create a fresh plan; never treat the prior
assistant question as evidence. When app_verified_clarification_choice is present, the current
reply exactly matched a retained model-authored candidate, so treat that sense as resolved and
set clarification_required=false. For a normal plan set observable_ambiguity=null, question="", and
candidates=[]."""
    RELEVANCE_SYSTEM_PROMPT = """You are a strict semantic relevance judge for a Quran research
POC. Return only semantic_relevance_v1 JSON. Judge each hydrated passage against the supplied
planner subtargets using the meaning of the complete target and passage, in Arabic and English.
First test each distinct coordinated term, participant, and outcome separately. A cultural,
technical, practical, or commentary-based association is not textual entailment: do not use it to
cover a term that the supplied passage itself does not establish.
A shared word, stem, root, named entity, or broad theme is never sufficient by itself. The
passage must directly express the requested central phenomenon, action, property, or relation,
with the same word sense. Judge semantic entailment rather than literal overlap: verbal and
nominal paraphrases may express the same relation, and evidence that directly describes the
relevant harm, refuge, response, or condition may support that evidenced facet of a requested
protective or preventive relation even if it does not repeat the user's practical wording. Mark
only the target/facet actually established; do not infer a broader technical label, mechanism,
or unstated outcome. A passage may be directly relevant when it explicitly names the central
phenomenon even if it does not answer every practical part of the user's question.
A passage that establishes only a narrower facet of a matched target must still be retained:
set directly_relevant=true, include that target in matched_target_ids, and also include it in
partial_target_ids. Use partial_target_ids exactly when the passage leaves a distinct requested
term, participant, relation, or outcome of that target unestablished. Do not mark a target partial
when the passage establishes all of it.
For every directly relevant passage, write one short supported_facet_summary phrase in the
supplied response language. It must state only what that passage directly establishes for its
matched target(s), without quotation, reference labels, canonical wording, or a finite verb; the
application supplies the neutral sentence frame. For an irrelevant passage,
supported_facet_summary must be empty and rejection_reason must be one of
NO_DIRECT_ENTAILMENT, DIFFERENT_TARGET, or INSUFFICIENT_CONTEXT. For a directly relevant
passage set rejection_reason=NOT_DROPPED. This summary is a bounded fallback claim and
must not mention an uncovered part or infer a broader technical concept. Do not import an action,
instruction, outcome, grammatical context, or neighboring passage merely because it appears in
the user target or is commonly associated with the fragment; state only the proposition present
in the supplied exact passage text. When the target asks for an action or outcome absent from the
passage, summarize the directly named harm, entity, condition, event, or property instead. Before
returning the summary, compare every action verb and normative/modal phrase in it with the exact
passage text; remove it if the passage does not establish that action or phrase.
For compound targets distinguished by a requested person, object, possession, situation, or
outcome, the passage must express that distinguishing element or its relation; a shared hazard or
action alone must not be used to mark every compound target as covered.
Reject lexical neighbors, homonyms, incidental mentions, and passages that mention only one
participant while omitting the requested relation. Do not use outside scripture, commentary,
memory, chapter names, or expected verse mappings. Evaluate every supplied passage exactly once.
Set directly_relevant=false with no matched targets when none is directly supported."""
    FACET_SUMMARY_REPAIR_PROMPT = """Repair only short supported-facet summaries for Quran
passages. Return semantic_facet_summary_v1 JSON. Every supplied passage was already accepted as
relevant for its listed target. For each passage, write one short phrase in the supplied response
language that states only an entity, harm, condition, event, or property directly present in that
exact passage. Do not quote it or use a finite verb. Do not import an action, instruction, refuge,
protection, outcome, grammatical context, neighboring passage, or common association unless it is
established by the exact supplied passage. Ignore broader user-requested actions absent from the
passage. Return every supplied passage exactly once."""
    COMPOUND_COVERAGE_PROMPT = """Decompose the evidence obligations in a clear Quran topic
request. Return only semantic_coverage_v1 JSON. Compare the original colloquial request and its
proposed research plan. Produce one short neutral target label for every independently
evidence-verifiable requested meaning, preserving the full relation and intended word sense in
each label. First test every distinct coordinated term, participant, and outcome separately; a
cultural, technical, practical, or commentary-based association does not merge two evidence
obligations. This is an evidence-obligation analysis, not a paraphrase exercise: split coordinated
terms unless direct support for one logically entails direct support for every other term, even if
they share a practical outcome. Do not mechanically split modifiers, near-synonyms, or terms that
jointly name one inseparable phenomenon, relation, or outcome. Preserve every explicitly
requested person, object, possession, situation, relation, and outcome. Correct colloquial
coordination and obvious spelling noise. Treat identity or true synonymy as entailment; do not
collapse distinct terms merely because a cultural, technical, or practical association links them.
without changing meaning. Do not add evidence, scripture, verse references, technical labels, or
clarification. Return one label when all requested meaning rises or falls together."""

    def __init__(
        self,
        client: ChatModelPort,
        validator: PlanValidator,
        *,
        route_validator: Callable[[ProviderCompletion], None] | None = None,
    ) -> None:
        self.client = client
        self.validator = validator
        self.route_validator = route_validator or ProviderRoutePolicy.require_groq_direct_free

    async def create_plan(
        self,
        *,
        user_message: str,
        working_context: WorkingContextV1,
        deterministic_resolution: DeterministicResolutionContextV1,
        route_state: object | None = None,
    ) -> tuple[ResearchPlanV1, list[ProviderCompletion]]:
        response_languages = [
            value.value
            for value in working_context.values
            if value.field.value == "RESPONSE_LANGUAGE" and isinstance(value.value, str)
        ]
        prompt_payload = {
            "user_message": user_message,
            "response_language": response_languages[0] if response_languages else "en",
            "resolved_quran_references": [
                item.canonical_label for item in deterministic_resolution.resolved_quran_references
            ],
            "pending_clarification": (
                working_context.pending_clarification.model_dump(mode="json")
                if working_context.pending_clarification is not None
                else None
            ),
            "app_verified_clarification_choice": None,
        }
        verified_choice: str | None = None
        if working_context.pending_clarification is not None:
            normalized_reply = normalize_user_query(user_message).casefold()
            verified_choice = next(
                (
                    candidate.label
                    for candidate in working_context.pending_clarification.candidates
                    if normalize_user_query(candidate.label).casefold() == normalized_reply
                ),
                None,
            )
        prompt_payload["app_verified_clarification_choice"] = verified_choice
        prompt = json.dumps(prompt_payload, ensure_ascii=False)
        metadata: list[ProviderCompletion] = []
        last_error = ""
        attempt_errors: list[str] = []
        for attempt in range(2):
            user_prompt = prompt
            if attempt:
                user_prompt += (
                    "\nThe previous proposal failed schema/policy validation. "
                    "Correct it once without broadening scope. Error: " + last_error
                )
            try:
                semantic_analysis_performed = False
                if verified_choice is None:
                    raw_analysis, completion = await self.client.complete(
                        response_model=SemanticAmbiguityProposalV1,
                        schema_name="semantic_ambiguity_v1",
                        system_prompt=self.AMBIGUITY_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        max_tokens=300,
                    )
                    self.route_validator(completion)
                    metadata.append(completion)
                else:
                    raw_analysis = SemanticAmbiguityProposalV1(
                        schema_version="1",
                        clarification_required=False,
                        observable_ambiguity=None,
                        resolved_target_summary=verified_choice,
                        clarification_question="",
                        clarification_candidates=[],
                    )
                if isinstance(raw_analysis, (ResearchPlanV1, QuranPlannerProposalV1)):
                    raw_plan = raw_analysis
                else:
                    semantic_analysis_performed = True
                    analysis = SemanticAmbiguityProposalV1.model_validate(raw_analysis)
                    if analysis.clarification_required:
                        if len(analysis.clarification_candidates) == 2:
                            audit_prompt = json.dumps(
                                {
                                    "original_request": user_message,
                                    "response_language": (
                                        response_languages[0] if response_languages else "en"
                                    ),
                                    "proposed_clarification": analysis.model_dump(mode="json"),
                                },
                                ensure_ascii=False,
                            )
                            audited, audit_completion = await self.client.complete(
                                response_model=SemanticAmbiguityProposalV1,
                                schema_name="semantic_ambiguity_v1",
                                system_prompt=self.AMBIGUITY_COVERAGE_PROMPT,
                                user_prompt=audit_prompt,
                                max_tokens=360,
                            )
                            self.route_validator(audit_completion)
                            metadata.append(audit_completion)
                            audited_analysis = (
                                audited
                                if isinstance(audited, SemanticAmbiguityProposalV1)
                                else SemanticAmbiguityProposalV1.model_validate(audited)
                            )
                            if audited_analysis.clarification_required and len(
                                audited_analysis.clarification_candidates
                            ) >= len(analysis.clarification_candidates):
                                analysis = audited_analysis
                        proposal = QuranPlannerProposalV1(
                            schema_version="1",
                            intent="QURAN_TOPIC_SEARCH",
                            response_language=(
                                response_languages[0] if response_languages else "en"
                            ),
                            semantic_target_label=analysis.resolved_target_summary,
                            entity_labels=[],
                            requested_operations=[],
                            search_query="",
                            fetch_target="QURAN_SEARCH_CANDIDATES",
                            tafsir_mode=None,
                            tafsir_resource_aliases=[],
                            clarification_required=True,
                            observable_ambiguity=analysis.observable_ambiguity,
                            clarification_question=analysis.clarification_question,
                            clarification_candidates=analysis.clarification_candidates,
                        )
                        plan = proposal.to_canonical(
                            scope_signals=(deterministic_resolution.deterministic_scope_signals)
                        )
                        return (
                            self.validator.validate(
                                plan,
                                working_context=working_context,
                                deterministic_resolution=deterministic_resolution,
                            ),
                            metadata,
                        )
                    if any(
                        (
                            analysis.observable_ambiguity is not None,
                            bool(analysis.clarification_question),
                            bool(analysis.clarification_candidates),
                        )
                    ):
                        raise PlanValidationError(
                            "Resolved semantic analysis contains clarification fields"
                        )
                    planning_prompt = (
                        user_prompt
                        + "\nResolved semantic target: "
                        + (analysis.resolved_target_summary)
                    )
                    raw_plan, plan_completion = await self.client.complete(
                        response_model=QuranPlannerProposalV1,
                        schema_name="quran_plan_proposal_v1",
                        system_prompt=self.SYSTEM_PROMPT,
                        user_prompt=planning_prompt,
                        max_tokens=500,
                    )
                    self.route_validator(plan_completion)
                    metadata.append(plan_completion)
                if (
                    semantic_analysis_performed
                    and verified_choice is None
                    and isinstance(raw_plan, QuranPlannerProposalV1)
                    and raw_plan.intent == "QURAN_TOPIC_SEARCH"
                    and not raw_plan.clarification_required
                ):
                    coverage_prompt = json.dumps(
                        {
                            "original_request": user_message,
                            "proposed_plan": raw_plan.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                    raw_coverage, coverage_completion = await self.client.complete(
                        response_model=SemanticCoverageProposalV1,
                        schema_name="semantic_coverage_v1",
                        system_prompt=self.COMPOUND_COVERAGE_PROMPT,
                        user_prompt=coverage_prompt,
                        max_tokens=360,
                    )
                    self.route_validator(coverage_completion)
                    metadata.append(coverage_completion)
                    coverage = (
                        raw_coverage
                        if isinstance(raw_coverage, SemanticCoverageProposalV1)
                        else SemanticCoverageProposalV1.model_validate(raw_coverage)
                    )
                    labels = list(dict.fromkeys(label.strip() for label in coverage.target_labels))
                    if not all(labels) or len(labels) > 3:
                        raise PlanValidationError(
                            "Semantic coverage labels must be non-empty and bounded"
                        )
                    raw_plan = raw_plan.model_copy(
                        update={
                            "entity_labels": (
                                [f"part:{label}" for label in labels] if len(labels) > 1 else labels
                            )
                        }
                    )
                if isinstance(raw_plan, ResearchPlanV1):
                    # Test/double compatibility; real provider responses use the
                    # bounded transport and are converted below.
                    plan = raw_plan
                elif isinstance(raw_plan, QuranPlannerProposalV1):
                    if verified_choice is not None and raw_plan.clarification_required:
                        raise PlanValidationError(
                            "An app-verified clarification choice is already resolved"
                        )
                    plan = raw_plan.to_canonical(
                        scope_signals=deterministic_resolution.deterministic_scope_signals
                    )
                else:
                    validated_proposal = QuranPlannerProposalV1.model_validate(raw_plan)
                    if verified_choice is not None and validated_proposal.clarification_required:
                        raise PlanValidationError(
                            "An app-verified clarification choice is already resolved"
                        )
                    plan = validated_proposal.to_canonical(
                        scope_signals=deterministic_resolution.deterministic_scope_signals
                    )
                return (
                    self.validator.validate(
                        plan,
                        working_context=working_context,
                        deterministic_resolution=deterministic_resolution,
                    ),
                    metadata,
                )
            except (
                PlanValidationError,
                StructuredOutputValidationError,
                ValidationError,
            ) as exc:
                last_error = str(exc)
                attempt_errors.append(last_error)
            except ProviderUnavailableError as exc:
                raise PlannerTemporarilyUnavailable(
                    exc.diagnostic,
                    metadata,
                ) from exc
            except ProviderRouteError as exc:
                raise PlannerFailedSafe("Planner effective route was rejected", metadata) from exc
            except RuntimeError as exc:
                raise PlannerFailedSafe(
                    f"Planner provider unavailable without repair: {exc}", metadata
                ) from exc
        raise PlannerFailedSafe(
            "Planner failed after exactly one repair: " + " | ".join(attempt_errors),
            metadata,
        )

    async def assess_topic_relevance(
        self,
        *,
        targets: tuple[SemanticSubtarget, ...],
        evidence: EvidenceSet,
        response_language: str = "en",
        route_state: object | None = None,
    ) -> tuple[TopicRelevanceAssessment, list[ProviderCompletion]]:
        del route_state
        grouped: dict[str, list[Any]] = {}
        for item in evidence.items_by_id.values():
            pointer = evidence_passage_key(item)
            grouped.setdefault(pointer, []).append(item)
        passage_to_pointer: dict[str, str] = {}
        passages: list[dict[str, Any]] = []
        for index, (pointer, items) in enumerate(grouped.items(), start=1):
            passage_id = f"passage-{index}"
            passage_to_pointer[passage_id] = pointer
            passages.append(
                {
                    "passage_id": passage_id,
                    "reference": items[0].reference.value,
                    "texts": [
                        {"language": item.language, "text": item.exact_text} for item in items
                    ],
                }
            )
        payload = json.dumps(
            {
                "targets": [
                    {"target_id": target.target_id, "meaning": target.search_query}
                    for target in targets
                ],
                "response_language": response_language,
                "passages": passages,
            },
            ensure_ascii=False,
        )
        expected_passages = set(passage_to_pointer)
        expected_targets = {target.target_id for target in targets}
        metadata: list[ProviderCompletion] = []
        last_error = ""
        last_exception: Exception | None = None
        for attempt in range(2):
            user_prompt = payload
            if attempt:
                user_prompt += (
                    "\nThis is the only repair. Evaluate every passage once and fix: " + last_error
                )
            try:
                raw, completion = await self.client.complete(
                    response_model=SemanticRelevanceProposalV1,
                    schema_name="semantic_relevance_v1",
                    system_prompt=self.RELEVANCE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=900,
                )
                self.route_validator(completion)
                metadata.append(completion)
                proposal = (
                    raw
                    if isinstance(raw, SemanticRelevanceProposalV1)
                    else SemanticRelevanceProposalV1.model_validate(raw)
                )
                decision_ids = [item.passage_id for item in proposal.decisions]
                if len(decision_ids) != len(set(decision_ids)):
                    raise PlanValidationError("Duplicate semantic passage decisions")
                if set(decision_ids) != expected_passages:
                    raise PlanValidationError("Semantic decisions must cover every passage")
                matches = {target.target_id: set() for target in targets}
                partial_target_ids: set[str] = set()
                summary_entries: list[tuple[str, str, set[str], str]] = []
                drop_reason_counts: dict[str, int] = {}
                for decision in proposal.decisions:
                    matched = set(decision.matched_target_ids)
                    partial = set(decision.partial_target_ids)
                    if not matched.issubset(expected_targets):
                        raise PlanValidationError("Semantic decision names an unknown target")
                    if not partial.issubset(matched):
                        raise PlanValidationError(
                            "Partial semantic targets must also be matched targets"
                        )
                    if decision.directly_relevant != bool(matched):
                        raise PlanValidationError(
                            "Semantic relevance flag conflicts with matched targets"
                        )
                    summary = decision.supported_facet_summary.strip()
                    if bool(matched) != bool(summary):
                        raise PlanValidationError(
                            "Semantic summary must be present exactly for direct evidence"
                        )
                    pointer = passage_to_pointer[decision.passage_id]
                    if not matched:
                        drop_reason_counts[decision.rejection_reason] = (
                            drop_reason_counts.get(decision.rejection_reason, 0) + 1
                        )
                    for target_id in matched:
                        matches[target_id].add(pointer)
                    partial_target_ids.update(partial)
                    if summary:
                        summary_entries.append((decision.passage_id, pointer, matched, summary))
                if attempt == 0 and passages and not any(matches.values()):
                    last_error = (
                        "The first semantic pass admitted no passage. Re-evaluate once: a passage "
                        "that explicitly names a central requested phenomenon may be directly "
                        "relevant even when it does not cover every practical part, but shared "
                        "entities or roots alone must still be rejected."
                    )
                    continue
                if not any(matches.values()):
                    matches = conservative_compound_anchor_matches(evidence, targets)
                supported_summaries: list[SupportedFacetSummary] = []
                invalid_summary_entries = []
                for passage_id, pointer, matched, summary in summary_entries:
                    try:
                        validate_supported_facet_summary(
                            text=summary,
                            evidence=evidence,
                            canonical_pointer=pointer,
                            response_language=response_language,
                        )
                    except GroundingError:
                        invalid_summary_entries.append((passage_id, pointer, matched))
                    else:
                        supported_summaries.extend(
                            SupportedFacetSummary(
                                target_id=target_id,
                                canonical_pointer=pointer,
                                text=summary,
                            )
                            for target_id in matched
                        )
                if invalid_summary_entries:
                    if attempt:
                        raise PlanValidationError(
                            "Supported-facet summary is not grounded after one repair"
                        )
                    repair_payload = json.dumps(
                        {
                            "response_language": response_language,
                            "passages": [
                                {
                                    "passage_id": passage_id,
                                    "targets": [
                                        {
                                            "target_id": target_id,
                                            "meaning": next(
                                                target.search_query
                                                for target in targets
                                                if target.target_id == target_id
                                            ),
                                        }
                                        for target_id in matched
                                    ],
                                    "texts": next(
                                        passage["texts"]
                                        for passage in passages
                                        if passage["passage_id"] == passage_id
                                    ),
                                }
                                for passage_id, _, matched in invalid_summary_entries
                            ],
                        },
                        ensure_ascii=False,
                    )
                    repaired_raw, repaired_completion = await self.client.complete(
                        response_model=SemanticFacetSummaryProposalV1,
                        schema_name="semantic_facet_summary_v1",
                        system_prompt=self.FACET_SUMMARY_REPAIR_PROMPT,
                        user_prompt=repair_payload,
                        max_tokens=360,
                    )
                    self.route_validator(repaired_completion)
                    metadata.append(repaired_completion)
                    repaired = (
                        repaired_raw
                        if isinstance(repaired_raw, SemanticFacetSummaryProposalV1)
                        else SemanticFacetSummaryProposalV1.model_validate(repaired_raw)
                    )
                    repaired_by_id = {
                        item.passage_id: item.supported_facet_summary.strip()
                        for item in repaired.decisions
                    }
                    expected_repaired_ids = {
                        passage_id for passage_id, _, _ in invalid_summary_entries
                    }
                    if set(repaired_by_id) != expected_repaired_ids:
                        raise PlannerFailedSafe(
                            "Supported-facet summary repair did not cover every passage", metadata
                        )
                    for passage_id, pointer, matched in invalid_summary_entries:
                        repaired_summary = repaired_by_id[passage_id]
                        try:
                            validate_supported_facet_summary(
                                text=repaired_summary,
                                evidence=evidence,
                                canonical_pointer=pointer,
                                response_language=response_language,
                            )
                        except GroundingError as exc:
                            raise PlannerFailedSafe(
                                "Supported-facet summary failed its bounded repair: " + str(exc),
                                metadata,
                            ) from exc
                        supported_summaries.extend(
                            SupportedFacetSummary(
                                target_id=target_id,
                                canonical_pointer=pointer,
                                text=repaired_summary,
                            )
                            for target_id in matched
                        )
                assessment = assess_topic_evidence_from_matches(
                    evidence,
                    targets=targets,
                    matches_by_target=matches,
                    partial_target_ids=partial_target_ids,
                    supported_facet_summaries=tuple(supported_summaries),
                )
                return (
                    TopicRelevanceAssessment(
                        evidence=assessment.evidence,
                        covered_target_ids=assessment.covered_target_ids,
                        missing_target_ids=assessment.missing_target_ids,
                        supported_facet_summaries=assessment.supported_facet_summaries,
                        review_drop_reason_counts=tuple(sorted(drop_reason_counts.items())),
                    ),
                    metadata,
                )
            except ProviderUnavailableError as exc:
                raise PlannerTemporarilyUnavailable(exc.diagnostic, metadata) from exc
            except ProviderRouteError as exc:
                raise PlannerFailedSafe(
                    "Semantic relevance route was rejected", metadata, cause=exc
                ) from exc
            except (
                PlanValidationError,
                StructuredOutputValidationError,
                ValidationError,
            ) as exc:
                last_error = str(exc)
                last_exception = exc
        raise PlannerFailedSafe(
            "Semantic relevance failed after exactly one repair: " + last_error,
            metadata,
            cause=last_exception,
        )
