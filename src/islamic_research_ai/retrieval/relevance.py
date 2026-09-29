from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from islamic_research_ai.contracts import (
    EvidenceItem,
    EvidenceRequirementFailureReason,
    EvidenceRequirementFailureV1,
    EvidenceSet,
    EvidenceSourceKind,
)
from islamic_research_ai.retrieval.normalization import (
    arabic_light_stem,
    normalize_arabic,
    normalize_translation,
)


def evidence_passage_key(item: EvidenceItem) -> str:
    """Stable passage identity shared by Quran, translation, and Tafsir evidence.

    Source-specific canonical pointers intentionally differ.  Relevance, however,
    must retain the full same-turn evidence bundle for one Quran passage rather than
    accidentally dropping its Tafsir or surrounding translation.
    """

    return json.dumps(
        [key.model_dump(mode="json") for key in item.canonical_pointer.verse_keys],
        sort_keys=True,
    )


@dataclass(frozen=True)
class SemanticSubtarget:
    target_id: str
    search_query: str
    required_terms: tuple[str, ...] = ()


def semantic_subtargets(
    entity_labels: list[str], *, fallback_query: str
) -> tuple[SemanticSubtarget, ...]:
    parts = [
        label.removeprefix("part:").strip()
        for label in entity_labels
        if label.startswith("part:") and label.removeprefix("part:").strip()
    ]
    fallback = [fallback_query.strip()] if fallback_query.strip() else []
    queries = list(dict.fromkeys(parts or fallback))
    return tuple(
        SemanticSubtarget(target_id=f"target-{index}", search_query=query)
        for index, query in enumerate(queries, start=1)
    )


_WORD = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_QUERY_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "quran",
    "say",
    "the",
    "to",
    "what",
    "which",
    "who",
    "القران",
    "الى",
    "عن",
    "عند",
    "على",
    "في",
    "قال",
    "كيف",
    "ما",
    "ماذا",
    "من",
    "هل",
    "يقول",
}


def _content_tokens(value: str) -> list[str]:
    normalized = normalize_arabic(normalize_translation(value))
    tokens: list[str] = []
    for raw_token in _WORD.findall(normalized):
        token = raw_token.casefold().strip("_؟،؛")
        if token and token not in _QUERY_STOPWORDS and len(token) >= 3:
            tokens.append(token)
    return tokens


def arabic_light_root(token: str, *, enable_taf3il: bool = False) -> str | None:
    """Return a conservative three-letter key for common Arabic derived forms.

    This is deliberately a relevance key, not a linguistic stemmer. It only handles
    high-confidence clitics, suffixes, and common أفعل/إفعال/مفعل shapes.
    """

    if not _ARABIC.search(token):
        return None
    value = arabic_light_stem(token)
    if len(value) == 6 and value.startswith("ا") and value[3] == "ا" and value.endswith("ا"):
        value = value[:-1]
    if len(value) >= 6 and value.startswith(("مست", "است")):
        value = value[3:]
    if len(value) == 5 and value.startswith("م") and value[3] == "و":
        value = value[1:3] + value[4]
    elif (
        enable_taf3il
        and len(value) == 5
        and value.startswith("ت")
        and value[3] == "ي"
    ):
        value = value[1:3] + value[4]
    elif len(value) == 4 and "ا" in value[1:3]:
        long_vowel_index = value.index("ا", 1, 3)
        value = value[:long_vowel_index] + value[long_vowel_index + 1 :]
    if len(value) == 5 and value.startswith("ا") and value[3] == "ا":
        value = value[1:3] + value[4]
    elif len(value) == 4 and value[0] in "اميتن":
        value = value[1:]
    return value if len(value) == 3 else None


@lru_cache(maxsize=1)
def _quran_root_df() -> tuple[int, dict[str, int]]:
    path = Path(__file__).with_name("quran_root_df.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["total_verses"], payload["df"]


def _arabic_match_key(token: str, *, distinctive_terms_enabled: bool) -> str:
    return arabic_light_root(token, enable_taf3il=distinctive_terms_enabled) or token


def _distinctive_idf(key: str) -> float:
    total_verses, df = _quran_root_df()
    return math.log((total_verses + 1) / (df.get(key, 0) + 1))


def expanded_topic_query(query: str) -> str:
    """Add morphological search forms while preserving the original query semantics."""

    tokens = _content_tokens(query)
    expanded: list[str] = list(tokens)
    for token in tokens:
        if not _ARABIC.search(token):
            continue
        unclitic = token[1:] if len(token) >= 5 and token[0] in "وفبكل" else token
        without_article = (
            unclitic[2:] if unclitic.startswith("ال") and len(unclitic) >= 5 else unclitic
        )
        expanded.extend((without_article, f"ال{without_article}", f"وال{without_article}"))
        root = arabic_light_root(token)
        if root is not None:
            expanded.append(root)
    return " ".join(dict.fromkeys(item for item in expanded if item)) or query


def arabic_root_regex(query: str) -> str:
    roots = {
        root for token in _content_tokens(query) if (root := arabic_light_root(token)) is not None
    }
    if not roots:
        return r"a^"
    patterns = [
        rf"(^| )[^ ]*{re.escape(root[0])}[^ ]*{re.escape(root[1])}"
        rf"[^ ]*{re.escape(root[2])}[^ ]*( |$)"
        for root in sorted(roots)
    ]
    return "(?:" + "|".join(patterns) + ")"


def _english_related(query_token: str, evidence_token: str) -> bool:
    if query_token == evidence_token:
        return True
    common = 0
    for left, right in zip(query_token, evidence_token, strict=False):
        if left != right:
            break
        common += 1
    return common >= 5 or (common >= 4 and min(len(query_token), len(evidence_token)) <= 7)


def lexical_relevance_score(query: str, evidence_text: str) -> float:
    """Score generic lexical support without encoding any domain concepts.

    Distinct matches reward breadth while a repeated Arabic root rewards passages
    that actually develop a concept instead of mentioning a nearby word once.
    The score is used only to reorder the bounded hybrid candidate pool; the
    stricter evidence gate below still decides whether a passage may be cited.
    """

    query_tokens = _content_tokens(query)
    evidence_tokens = _content_tokens(evidence_text)
    query_roots = {root for token in query_tokens if (root := arabic_light_root(token)) is not None}
    evidence_root_list = [
        root for token in evidence_tokens if (root := arabic_light_root(token)) is not None
    ]
    root_score = sum(
        min(evidence_root_list.count(root), 2) * 2.0
        for root in query_roots
        if root in evidence_root_list
    )
    literal_score = float(len(set(query_tokens).intersection(evidence_tokens)))
    query_english = [token for token in query_tokens if token.isascii()]
    evidence_english = [token for token in evidence_tokens if token.isascii()]
    english_score = float(
        sum(
            any(
                _english_related(query_token, evidence_token) for evidence_token in evidence_english
            )
            for query_token in query_english
        )
    )
    return root_score + literal_score + english_score


def evidence_is_topic_relevant(question: str, items: list[EvidenceItem]) -> bool:
    query_tokens = _content_tokens(question)
    query_roots = {root for token in query_tokens if (root := arabic_light_root(token)) is not None}
    evidence_text = " ".join(item.exact_text for item in items)
    evidence_tokens = _content_tokens(evidence_text)
    evidence_roots = {
        root for token in evidence_tokens if (root := arabic_light_root(token)) is not None
    }
    if query_roots and query_roots.intersection(evidence_roots):
        return True

    query_english = [token for token in query_tokens if token.isascii()]
    evidence_english = [token for token in evidence_tokens if token.isascii()]
    return any(
        _english_related(query_token, evidence_token)
        for query_token in query_english
        for evidence_token in evidence_english
    )


def _subtarget_matches(
    target: SemanticSubtarget,
    items: list[EvidenceItem],
    *,
    distinctive_terms_enabled: bool = False,
    distinctive_terms_minimum_fraction: float = 0.5,
) -> bool:
    def semantic_text(value: str) -> str:
        return normalize_arabic(normalize_translation(value)).replace("ة", "ه")

    evidence_text = semantic_text(" ".join(item.exact_text for item in items))
    if target.required_terms:
        evidence_tokens = _content_tokens(evidence_text)
        evidence_features = set(evidence_tokens)
        evidence_features.update(arabic_light_stem(token) for token in evidence_tokens)
        evidence_features.update(
            root for token in evidence_tokens if (root := arabic_light_root(token)) is not None
        )
        for term in target.required_terms:
            normalized_term = semantic_text(term)
            term_tokens = _content_tokens(normalized_term)
            term_features = set(term_tokens)
            term_features.update(arabic_light_stem(token) for token in term_tokens)
            term_features.update(
                root for token in term_tokens if (root := arabic_light_root(token)) is not None
            )
            if term_features.intersection(evidence_features):
                return True
        return False

    target_tokens = _content_tokens(target.search_query)
    evidence_token_list = _content_tokens(evidence_text)
    evidence_tokens = set(evidence_token_list)
    if distinctive_terms_enabled:
        target_arabic = [token for token in target_tokens if _ARABIC.search(token)]
        evidence_arabic_keys = {
            _arabic_match_key(token, distinctive_terms_enabled=True)
            for token in evidence_token_list
            if _ARABIC.search(token)
        }
        target_weights: dict[str, float] = {}
        for token in target_arabic:
            key = _arabic_match_key(token, distinctive_terms_enabled=True)
            target_weights[key] = _distinctive_idf(key)
        matched_keys = set(target_weights).intersection(evidence_arabic_keys)
        if target_weights:
            total_weight = sum(target_weights.values())
            matched_weight = sum(target_weights[key] for key in matched_keys)
            highest_key = max(target_weights, key=target_weights.get)
            if (
                matched_weight >= total_weight * distinctive_terms_minimum_fraction
                or highest_key in matched_keys
            ):
                return True
        best_rank = min(
            (item.retrieval.rank for item in items if item.retrieval.rank is not None),
            default=None,
        )
        if target_weights and best_rank == 1 and matched_keys:
            sorted_by_weight = sorted(target_weights, key=target_weights.get)
            lowest_half = set(sorted_by_weight[: len(sorted_by_weight) // 2])
            if matched_keys.difference(lowest_half):
                return True
        # English matching retains the pre-flag count and rank-1 rules.
        target_english = [token for token in target_tokens if token.isascii()]
        evidence_english = [token for token in evidence_tokens if token.isascii()]
        english_matches = {
            query_token
            for query_token in target_english
            if any(
                _english_related(query_token, evidence_token)
                for evidence_token in evidence_english
            )
        }
        required_english = 1 if len(target_english) <= 1 else 2
        return len(english_matches) >= required_english or (
            best_rank == 1 and bool(english_matches)
        )

    target_roots = {
        root for token in target_tokens if (root := arabic_light_root(token)) is not None
    }
    evidence_root_list = [
        root for token in evidence_token_list if (root := arabic_light_root(token)) is not None
    ]
    evidence_roots = set(evidence_root_list)
    root_overlap = target_roots.intersection(evidence_roots)
    matched_query_tokens = {
        token
        for token in target_tokens
        if token in evidence_tokens
        or ((root := arabic_light_root(token)) is not None and root in evidence_roots)
    }
    minimum = 1 if len(target_tokens) <= 1 else 2
    if len(matched_query_tokens) >= minimum:
        return True
    best_rank = min(
        (item.retrieval.rank for item in items if item.retrieval.rank is not None),
        default=None,
    )
    if best_rank == 1 and root_overlap:
        return True
    target_english = [token for token in target_tokens if token.isascii()]
    evidence_english = [token for token in evidence_tokens if token.isascii()]
    english_matches = {
        query_token
        for query_token in target_english
        if any(_english_related(query_token, evidence_token) for evidence_token in evidence_english)
    }
    required_english = 1 if len(target_english) <= 1 else 2
    if len(english_matches) >= required_english:
        return True
    return best_rank == 1 and bool(english_matches)


@dataclass(frozen=True)
class TopicRelevanceAssessment:
    evidence: EvidenceSet
    covered_target_ids: tuple[str, ...]
    missing_target_ids: tuple[str, ...]
    supported_facet_summaries: tuple[SupportedFacetSummary, ...] = ()
    review_drop_reason_counts: tuple[tuple[str, int], ...] = ()
    boundary_retained_top_ranked: bool = False
    boundary_retained_hydrated_anchors: bool = False


@dataclass(frozen=True)
class SupportedFacetSummary:
    target_id: str
    canonical_pointer: str
    text: str


def assess_topic_evidence(
    question: str,
    evidence: EvidenceSet,
    *,
    targets: tuple[SemanticSubtarget, ...] = (),
    retained_passage_limit: int = 5,
    ranked_retention_enabled: bool = False,
    relevance_distinctive_terms_enabled: bool = False,
    distinctive_terms_minimum_fraction: float = 0.5,
) -> TopicRelevanceAssessment:
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence.items_by_id.values():
        pointer = evidence_passage_key(item)
        grouped[pointer].append(item)

    matches_by_target = {
        target.target_id: {
            pointer
            for pointer, items in grouped.items()
            if _subtarget_matches(
                target,
                items,
                distinctive_terms_enabled=relevance_distinctive_terms_enabled,
                distinctive_terms_minimum_fraction=distinctive_terms_minimum_fraction,
            )
        }
        for target in targets
    }
    if not ranked_retention_enabled:
        if targets:
            retained_pointers = set().union(*matches_by_target.values())
        else:
            retained_pointers = {
                pointer
                for pointer, items in grouped.items()
                if evidence_is_topic_relevant(question, items)
            }
        return assess_topic_evidence_from_matches(
            evidence,
            targets=targets,
            matches_by_target=matches_by_target,
            retained_pointers=retained_pointers,
        )

    ranked_pointers = sorted(
        grouped,
        key=lambda pointer: (
            not any(item.retrieval.rank is not None for item in grouped[pointer]),
            min(
                (
                    item.retrieval.rank
                    for item in grouped[pointer]
                    if item.retrieval.rank is not None
                ),
                default=0,
            ),
            pointer,
        ),
    )[:retained_passage_limit]
    retained_pointers = set(ranked_pointers)
    assessment = assess_topic_evidence_from_matches(
        evidence,
        targets=targets,
        matches_by_target=matches_by_target,
        retained_pointers=retained_pointers,
    )
    retained_pointers = {
        evidence_passage_key(item) for item in assessment.evidence.items_by_id.values()
    }
    retained_matches_by_target = {
        target_id: pointers.intersection(retained_pointers)
        for target_id, pointers in matches_by_target.items()
    }
    return TopicRelevanceAssessment(
        evidence=assessment.evidence,
        covered_target_ids=tuple(
            target.target_id
            for target in targets
            if retained_matches_by_target.get(target.target_id)
        ),
        missing_target_ids=tuple(
            target.target_id
            for target in targets
            if not retained_matches_by_target.get(target.target_id)
        ),
        supported_facet_summaries=assessment.supported_facet_summaries,
        review_drop_reason_counts=assessment.review_drop_reason_counts,
        boundary_retained_top_ranked=assessment.boundary_retained_top_ranked,
    )


def retain_top_ranked_topic_evidence(
    evidence: EvidenceSet,
    *,
    targets: tuple[SemanticSubtarget, ...],
) -> TopicRelevanceAssessment | None:
    """Keep the best hydrated passage when an advisory review rejects everything.

    This is a retrieval-backed hypothesis check, not a semantic endorsement.  The retained
    passage remains subject to the normal grounding gates and every requested target stays
    marked missing, so the response is explicitly partial rather than claiming coverage that
    the reviewer did not establish.
    """

    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence.items_by_id.values():
        grouped[evidence_passage_key(item)].append(item)
    ranked = [
        (min(item.retrieval.rank for item in items if item.retrieval.rank is not None), pointer)
        for pointer, items in grouped.items()
        if any(item.retrieval.rank is not None for item in items)
    ]
    if not ranked:
        return None
    best_rank = min(rank for rank, _ in ranked)
    retained_pointers = {pointer for rank, pointer in ranked if rank == best_rank}
    assessment = assess_topic_evidence_from_matches(
        evidence,
        targets=targets,
        matches_by_target={target.target_id: set() for target in targets},
        retained_pointers=retained_pointers,
    )
    return TopicRelevanceAssessment(
        evidence=assessment.evidence,
        covered_target_ids=(),
        missing_target_ids=tuple(target.target_id for target in targets),
        supported_facet_summaries=(),
        review_drop_reason_counts=(),
    )


def enforce_ranked_relevance_boundary(
    *,
    retrieved_evidence: EvidenceSet,
    assessment: TopicRelevanceAssessment,
    targets: tuple[SemanticSubtarget, ...],
    empty_match_retains_anchors: bool = False,
) -> TopicRelevanceAssessment:
    """Enforce the non-emptying invariant for every relevance implementation.

    Relevance producers may narrow evidence, but none may turn a non-empty ranked retrieval
    result into an empty evidence set.  The retained top-ranked passage is deliberately marked
    as uncovered, so this is a grounded partial response rather than a semantic endorsement.
    Keeping the rule here, after every producer, prevents a future deterministic or model-based
    implementation from recreating the emptying defect through a new control-flow branch.
    """

    if assessment.evidence.items_by_id or not retrieved_evidence.items_by_id:
        return assessment
    if empty_match_retains_anchors:
        # Evidence assembly has already deduplicated canonical items shared by overlapping
        # hydrated windows. Retain that existing bounded set as-is: K1 changes survival only,
        # never lexical matching, target coverage, source attachments, or a later retrieval cap.
        retained = assess_topic_evidence_from_matches(
            retrieved_evidence,
            targets=targets,
            matches_by_target={target.target_id: set() for target in targets},
            retained_pointers={
                evidence_passage_key(item) for item in retrieved_evidence.items_by_id.values()
            },
        )
        return TopicRelevanceAssessment(
            evidence=retained.evidence,
            covered_target_ids=(),
            missing_target_ids=tuple(target.target_id for target in targets),
            supported_facet_summaries=(),
            review_drop_reason_counts=assessment.review_drop_reason_counts,
            boundary_retained_hydrated_anchors=True,
        )
    retained = retain_top_ranked_topic_evidence(retrieved_evidence, targets=targets)
    if retained is None:
        return assessment
    return TopicRelevanceAssessment(
        evidence=retained.evidence,
        covered_target_ids=(),
        missing_target_ids=tuple(target.target_id for target in targets),
        supported_facet_summaries=(),
        review_drop_reason_counts=assessment.review_drop_reason_counts,
        boundary_retained_top_ranked=True,
    )


def empty_topic_relevance_assessment(
    evidence: EvidenceSet, *, targets: tuple[SemanticSubtarget, ...]
) -> TopicRelevanceAssessment:
    """Produce an explicit empty proposal for a failed relevance producer.

    Callers must pass this through :func:`enforce_ranked_relevance_boundary` before using it.
    """

    return assess_topic_evidence_from_matches(
        evidence,
        targets=targets,
        matches_by_target={target.target_id: set() for target in targets},
    )


def assess_topic_evidence_from_matches(
    evidence: EvidenceSet,
    *,
    targets: tuple[SemanticSubtarget, ...],
    matches_by_target: dict[str, set[str]],
    partial_target_ids: set[str] | None = None,
    supported_facet_summaries: tuple[SupportedFacetSummary, ...] = (),
    retained_pointers: set[str] | None = None,
) -> TopicRelevanceAssessment:
    """Apply model-produced semantic matches through deterministic evidence policy."""

    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence.items_by_id.values():
        pointer = evidence_passage_key(item)
        grouped[pointer].append(item)
    retained_pointers = retained_pointers or set().union(*matches_by_target.values())
    # Once semantic review establishes a verse as relevant, retain the bounded
    # structural passage fetched around it. This is deliberately based only on
    # chapter/verse document proximity, never on topic vocabulary or a verse ID.
    retained_locations = {
        (item.canonical_pointer.verse_keys[0].chapter, item.canonical_pointer.verse_keys[0].verse)
        for pointer, items in grouped.items()
        if pointer in retained_pointers
        for item in items
        if item.source_kind is EvidenceSourceKind.QURAN
    }
    context_pointers = {
        pointer
        for pointer, items in grouped.items()
        if any(
            item.source_kind is EvidenceSourceKind.QURAN
            and any(
                item.canonical_pointer.verse_keys[0].chapter == chapter
                and abs(item.canonical_pointer.verse_keys[0].verse - verse) <= 4
                for chapter, verse in retained_locations
            )
            for item in items
        )
    }
    retained_pointers = retained_pointers.union(context_pointers)
    retained_ids = {
        item.evidence_id
        for pointer, items in grouped.items()
        if pointer in retained_pointers
        for item in items
    }
    items_by_id = {
        evidence_id: item
        for evidence_id, item in evidence.items_by_id.items()
        if evidence_id in retained_ids
    }

    missing: list[EvidenceRequirementFailureV1] = []
    for source_kind in sorted(evidence.requirements.required_source_kinds, key=str):
        count = sum(item.source_kind is source_kind for item in items_by_id.values())
        minimum = evidence.requirements.minimum_items_by_kind.get(source_kind, 1)
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
    satisfied = not missing and bool(items_by_id)
    filtered = evidence.model_copy(
        update={
            "items_by_id": items_by_id,
            "requirements_satisfied": satisfied,
            "missing_requirements": missing,
            "permitted_synthesis_deployments": (
                evidence.permitted_synthesis_deployments if satisfied else set()
            ),
        }
    )
    partial_target_ids = partial_target_ids or set()
    covered = tuple(
        target.target_id
        for target in targets
        if matches_by_target.get(target.target_id) and target.target_id not in partial_target_ids
    )
    missing_targets = tuple(
        target.target_id
        for target in targets
        if not matches_by_target.get(target.target_id) or target.target_id in partial_target_ids
    )
    retained_summaries = tuple(
        summary
        for summary in supported_facet_summaries
        if summary.canonical_pointer in retained_pointers
    )
    return TopicRelevanceAssessment(filtered, covered, missing_targets, retained_summaries)


def conservative_compound_anchor_matches(
    evidence: EvidenceSet,
    targets: tuple[SemanticSubtarget, ...],
) -> dict[str, set[str]]:
    """Find one bounded passage for a repeated concept shared by compound targets.

    This is a fail-closed fallback for a semantic judge that returns no matches twice. It
    never decides a word sense and never marks every part covered: a repeated shared concept
    can support at most the first compatible subtarget, leaving distinct objects for explicit
    coverage. Single-target requests receive no fallback.
    """

    matches = {target.target_id: set() for target in targets}
    if len(targets) < 2:
        return matches
    roots_by_target = {
        target.target_id: {
            root
            for token in _content_tokens(target.search_query)
            if (root := arabic_light_root(token)) is not None
        }
        for target in targets
    }
    root_target_counts: dict[str, int] = defaultdict(int)
    for roots in roots_by_target.values():
        for root in roots:
            root_target_counts[root] += 1
    shared_roots = {root for root, count in root_target_counts.items() if count >= 2}
    if not shared_roots:
        return matches

    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence.items_by_id.values():
        pointer = evidence_passage_key(item)
        grouped[pointer].append(item)
    candidates: list[tuple[int, int, str, str]] = []
    for pointer, items in grouped.items():
        roots = [
            root
            for token in _content_tokens(" ".join(item.exact_text for item in items))
            if (root := arabic_light_root(token)) is not None
        ]
        for root in shared_roots:
            occurrences = roots.count(root)
            if occurrences >= 2:
                best_rank = min((item.retrieval.rank or 10_000 for item in items), default=10_000)
                candidates.append((-occurrences, best_rank, pointer, root))
    if not candidates:
        return matches
    _, _, pointer, root = min(candidates)
    target_id = next(
        target.target_id for target in targets if root in roots_by_target[target.target_id]
    )
    matches[target_id].add(pointer)
    return matches


def retain_relevant_topic_evidence(question: str, evidence: EvidenceSet) -> EvidenceSet:
    return assess_topic_evidence(question, evidence).evidence
