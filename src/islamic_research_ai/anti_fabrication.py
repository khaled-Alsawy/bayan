"""`AntiFabricationGate` -- WP-2 of `CONVERSATIONAL_QUALITY_TECHNICAL_PLAN_V3.md`.

This is the core safety deliverable of the plan. It is the mechanical enforcement of the
"no invented text" contract (Part 2 of the plan) for the composed narrative:

1. QUOTE_GUARD -- any quoted span of 3+ consecutive Arabic words must be a normalized
   substring of some retrieved `EvidenceItem.exact_text` this turn.
2. REFERENCE_GUARD -- every `chapter:verse` pair, and every `surah <name>` mention, must
   resolve to a verse actually present in this turn's evidence.
3. ATTRIBUTION_GUARD -- every named exegete/hadith collection must appear in this turn's
   evidence attributions.
4. HADITH_GUARD -- a hadith-shaped sentence is rejected unless it is a verbatim substring of
   retrieved tafsir text (a hadith may only appear as part of a tafsir excerpt's own takhrij,
   never as a free-standing claim -- see the safety contract, Part 2 §4 of the plan).

On failure the caller gets back the list of violations, each naming its rule and the offending
span, so `AnswerComposer` can attempt exactly one bounded repair. On a second failure the
caller must drop the narrative and render claims only -- this module never repairs by itself
and never renders anything; it only decides pass/fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from islamic_research_ai.contracts import EvidenceItem, EvidenceSet, EvidenceSourceKind
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.surah_names import NAME_TO_CHAPTER

# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------


def _normalize_for_match(value: str) -> str:
    """One normalization used for both narrative spans and evidence text, so a substring
    check is not defeated by diacritics, alef variants, or casing alone."""

    return normalize_translation(normalize_arabic(value))


_ARABIC_WORD_RE = re.compile(r"[ء-ي]+")


def _consecutive_arabic_word_count(text: str) -> int:
    return len(_ARABIC_WORD_RE.findall(text))


_SENTENCE_END_RE = re.compile(r"[.!?؟۔\n]")


def _signal_to_sentence_end(text: str, start: int) -> str:
    """From a hadith-signal match to the end of its sentence -- not the whole sentence.

    Framing prose commonly precedes the signal in the same sentence ("وفي هذا السياق: قال
    رسول الله ..."); that lead-in was never part of the tafsir excerpt and must not be
    required to match it. What must match verbatim is the hadith wording itself, starting at
    the signal.
    """

    end_match = _SENTENCE_END_RE.search(text, start)
    end = end_match.start() if end_match else len(text)
    return text[start:end]


class FabricationRule(StrEnum):
    QUOTE_GUARD = "QUOTE_GUARD"
    REFERENCE_GUARD = "REFERENCE_GUARD"
    ATTRIBUTION_GUARD = "ATTRIBUTION_GUARD"
    HADITH_GUARD = "HADITH_GUARD"


@dataclass(frozen=True)
class FabricationViolation:
    rule: FabricationRule
    offending_span: str
    detail: str


# --------------------------------------------------------------------------------------------
# QUOTE_GUARD
# --------------------------------------------------------------------------------------------

_QUOTE_SPAN_RE = re.compile(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”|\(([^)]+)\)|﴾([^﴿]+)﴿")


def _quote_guard(text: str, evidence_texts: list[str]) -> list[FabricationViolation]:
    violations: list[FabricationViolation] = []
    for match in _QUOTE_SPAN_RE.finditer(text):
        span = next(group for group in match.groups() if group is not None)
        if _consecutive_arabic_word_count(span) < 3:
            continue  # short parentheticals/citations, not a quotation risk
        normalized_span = _normalize_for_match(span)
        if not normalized_span:
            continue
        if not any(normalized_span in haystack for haystack in evidence_texts):
            violations.append(
                FabricationViolation(
                    rule=FabricationRule.QUOTE_GUARD,
                    offending_span=span.strip(),
                    detail=(
                        "Quoted span of 3+ Arabic words is not a verbatim substring of any "
                        "evidence item retrieved this turn."
                    ),
                )
            )
    return violations


# --------------------------------------------------------------------------------------------
# REFERENCE_GUARD
# --------------------------------------------------------------------------------------------

_VERSE_REF_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[:：]\s*(\d{1,3})(?!\d)")
_SURAH_NAME_RE = re.compile(
    r"(?:سورة|surah|surat)\s+([ء-ي][ء-ي\s]{0,30}|[A-Za-z][A-Za-z'\- ]{0,30})",
    re.IGNORECASE,
)


def _known_verse_keys(evidence: EvidenceSet) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for item in evidence.items_by_id.values():
        pointer = item.canonical_pointer
        verse_keys = getattr(pointer, "verse_keys", None)
        if verse_keys:
            keys.update((key.chapter, key.verse) for key in verse_keys)
    return keys


def _known_chapters(evidence: EvidenceSet) -> set[int]:
    return {chapter for chapter, _ in _known_verse_keys(evidence)}


def _resolve_surah_name(candidate: str) -> int | None:
    tokens = candidate.strip().split()
    for length in range(min(3, len(tokens)), 0, -1):
        phrase = " ".join(tokens[:length])
        chapter = NAME_TO_CHAPTER.get(phrase) or NAME_TO_CHAPTER.get(phrase.lower())
        if chapter is not None:
            return chapter
    return None


def _reference_guard(text: str, evidence: EvidenceSet) -> list[FabricationViolation]:
    violations: list[FabricationViolation] = []
    known_verses = _known_verse_keys(evidence)
    known_chapters = _known_chapters(evidence)

    for match in _VERSE_REF_RE.finditer(text):
        chapter, verse = int(match.group(1)), int(match.group(2))
        if not (1 <= chapter <= 114) or verse <= 0:
            continue  # not shaped like a verse reference (e.g. a ratio, a time, a score)
        if (chapter, verse) not in known_verses:
            violations.append(
                FabricationViolation(
                    rule=FabricationRule.REFERENCE_GUARD,
                    offending_span=match.group(0),
                    detail=(
                        f"{chapter}:{verse} is not among the verse keys retrieved as evidence "
                        "this turn."
                    ),
                )
            )

    for match in _SURAH_NAME_RE.finditer(text):
        chapter = _resolve_surah_name(match.group(1))
        if chapter is None:
            continue  # not a recognized surah name -- do not flag ordinary prose
        if chapter not in known_chapters:
            violations.append(
                FabricationViolation(
                    rule=FabricationRule.REFERENCE_GUARD,
                    offending_span=match.group(0),
                    detail=f"Surah {chapter} has no evidence retrieved this turn.",
                )
            )
    return violations


# --------------------------------------------------------------------------------------------
# ATTRIBUTION_GUARD
# --------------------------------------------------------------------------------------------

# (key, arabic surface forms, latin surface forms) -- exactly the static list in the plan.
_ATTRIBUTION_NAMES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("ibn_kathir", ("ابن كثير",), ("ibn kathir",)),
    ("al_tabari", ("الطبري",), ("al-tabari", "at-tabari")),
    ("al_qurtubi", ("القرطبي",), ("al-qurtubi",)),
    ("al_jalalayn", ("الجلالين",), ("jalalayn",)),
    ("al_saadi", ("السعدي",), ("al-saadi", "as-sa'di", "al-sa'di")),
    ("al_baghawi", ("البغوي",), ("al-baghawi",)),
    ("ibn_ashur", ("ابن عاشور",), ("ibn ashur",)),
    ("al_shawkani", ("الشوكاني",), ("al-shawkani",)),
    ("al_bukhari", ("البخاري",), ("bukhari",)),
    ("muslim", ("مسلم",), ("sahih muslim",)),  # bare "muslim" is too ambiguous in English
    ("al_tirmidhi", ("الترمذي",), ("tirmidhi",)),
    ("abu_dawud", ("أبو داود",), ("abu dawud", "abu dawood")),
    ("al_nasai", ("النسائي",), ("al-nasai", "an-nasa'i")),
    ("ibn_majah", ("ابن ماجه", "ابن ماجة"), ("ibn majah",)),
    ("ahmad", ("مسند أحمد",), ("musnad ahmad",)),  # bare "ahmad" is a common given name
)


def _attribution_surface_forms(item: EvidenceItem) -> str:
    parts = [item.source_identity.work_title]
    parts.extend(attribution.display_name for attribution in item.attributions)
    return _normalize_for_match(" ".join(parts))


def _attribution_guard(text: str, evidence: EvidenceSet) -> list[FabricationViolation]:
    violations: list[FabricationViolation] = []
    normalized_text = _normalize_for_match(text)
    evidence_forms = [_attribution_surface_forms(item) for item in evidence.items_by_id.values()]
    for _key, arabic_forms, latin_forms in _ATTRIBUTION_NAMES:
        mentioned_form = next(
            (
                form
                for form in (*arabic_forms, *latin_forms)
                if _normalize_for_match(form) in normalized_text
            ),
            None,
        )
        if mentioned_form is None:
            continue
        supported = any(
            _normalize_for_match(form) in haystack
            for form in (*arabic_forms, *latin_forms)
            for haystack in evidence_forms
        )
        if not supported:
            violations.append(
                FabricationViolation(
                    rule=FabricationRule.ATTRIBUTION_GUARD,
                    offending_span=mentioned_form,
                    detail=(
                        f"'{mentioned_form}' is named in the narrative but does not appear in "
                        "any evidence attribution retrieved this turn."
                    ),
                )
            )
    return violations


# --------------------------------------------------------------------------------------------
# HADITH_GUARD
# --------------------------------------------------------------------------------------------

_HADITH_SIGNAL_RE = re.compile(
    "|".join(
        [
            r"قال\s+رسول\s+الله",
            r"قال\s+النبي",
            r"عن\s+النبي",
            r"روى",
            r"أخرجه",
            r"حديث\s*(?:صحيح|حسن|ضعيف)",
            r"ﷺ",  # ﷺ
            r"the\s+messenger\s+of\s+god\s+said",
            r"the\s+prophet\s+said",
            r"narrated\s+(?:by|that)",
            r"reported\s+by",
            r"\bhadith\b",
        ]
    ),
    re.IGNORECASE,
)


def _hadith_guard(text: str, tafsir_texts: list[str]) -> list[FabricationViolation]:
    violations: list[FabricationViolation] = []
    for match in _HADITH_SIGNAL_RE.finditer(text):
        span = _signal_to_sentence_end(text, match.start())
        normalized_span = _normalize_for_match(span)
        if not normalized_span or not any(normalized_span in haystack for haystack in tafsir_texts):
            violations.append(
                FabricationViolation(
                    rule=FabricationRule.HADITH_GUARD,
                    offending_span=span.strip(),
                    detail=(
                        "Hadith-shaped text is not a verbatim substring of any tafsir excerpt "
                        "retrieved this turn; a hadith may only appear verbatim inside a "
                        "retrieved tafsir passage, never as a free-standing claim."
                    ),
                )
            )
    return violations


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------


class AntiFabricationGate:
    def check(self, *, narrative_text: str, evidence: EvidenceSet) -> list[FabricationViolation]:
        evidence_texts = [
            _normalize_for_match(item.exact_text) for item in evidence.items_by_id.values()
        ]
        tafsir_texts = [
            _normalize_for_match(item.exact_text)
            for item in evidence.items_by_id.values()
            if item.source_kind is EvidenceSourceKind.TAFSIR
        ]
        violations: list[FabricationViolation] = []
        violations += _quote_guard(narrative_text, evidence_texts)
        violations += _reference_guard(narrative_text, evidence)
        violations += _attribution_guard(narrative_text, evidence)
        violations += _hadith_guard(narrative_text, tafsir_texts)
        return violations
