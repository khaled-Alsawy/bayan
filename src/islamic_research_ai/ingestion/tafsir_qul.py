"""Parser and normaliser for the QUL (Quranic Universal Library) tafsir exports.

WP-0 of `CONVERSATIONAL_QUALITY_TECHNICAL_PLAN_V3.md`. Pure shape work: this module has no
network or database dependency and does not depend on any pending rights answer. It turns a
raw QUL grouped-tafsir JSON export into `TafsirPassage` rows ready for ingestion, exactly the
granularity passage-level retrieval needs.

Source shape (verified against the four downloaded exports in `data/raw/qul/extracted/`):

    { "113:1": { "text": "<div class=ar>...</div>", "ayah_keys": ["113:1", ..., "113:5"] },
      "113:5": "113:1" }

A grouped tafsir stores the passage on its first ayah key; every other key in the group is a
bare string pointing back at the owning key. `ayah_keys` gives the passage's full verse range.

Footnotes are `[[...]]` spans inline in the HTML. Measured on Ibn Kathir (31,203 footnotes):
~72% are manuscript-variant notes ("in ms. X: ...", "addition from ms. Y") that belong to the
critical edition's apparatus, not the tafsir itself, and are dropped; ~26% are takhrij
(hadith/athar source citations) kept as structured `hadith_references`; the remainder is a
long tail of cross-references and citations to non-hadith works, which default to STRIP -- the
safe direction, since an unclassified footnote is far more likely to be noise than a hadith
citation, and `HADITH_GUARD` (WP-2) never trusts this list as an allow-list, only as display
metadata attached to the excerpt it came from.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_BREAK_RE = re.compile(r"</?(?:p|div|br)[^>]*>", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_SPACE_RUN_RE = re.compile(r"[ \t\f\v]+")

# Manuscript-variant apparatus: "زيادة من <siglum>" (addition from ms. X), "بياض" (blank in the
# manuscript), or the inline "في <siglum(s)>: ..." template recording a variant reading.
_MANUSCRIPT_VARIANT_RE = re.compile(r"^(?:زيادة من|زياده من|بياض|سقط من|كذا في|تكرر في)\b")
_MANUSCRIPT_INLINE_RE = re.compile(r"^في\s+[ء-يـ،\s]{1,15}[:：]{1,2}")

# A takhrij / source-citation footnote: either an explicit narration verb, or a citation that
# carries a volume/page locator (Arabic-Indic or Western digits) -- the shape every real
# footnote in the sampled reference bucket shares, hadith or not.
_NARRATION_VERB_RE = re.compile(r"(رواه|أخرجه)")
_LOCATOR_DIGIT_RE = re.compile(r"[٠-٩\d]")

FOOTNOTE_STRIP = "STRIP"
FOOTNOTE_REFERENCE = "HADITH_REFERENCE"


def _flatten_html(value: str) -> str:
    """Strip HTML/entities from a footnote body, collapsing to a single line."""

    value = html.unescape(value)
    value = _BLOCK_BREAK_RE.sub(" ", value)
    value = _TAG_RE.sub("", value)
    return " ".join(value.split())


def strip_html(value: str) -> str:
    """Strip HTML/entities from a passage body, preserving paragraph breaks."""

    value = html.unescape(value)
    value = _BLOCK_BREAK_RE.sub("\n", value)
    value = _TAG_RE.sub("", value)
    lines = [_SPACE_RUN_RE.sub(" ", line).strip() for line in value.split("\n")]
    value = "\n".join(line for line in lines if line)
    return _BLANK_RUN_RE.sub("\n\n", value).strip()


def classify_footnote(flattened_note: str) -> str:
    """Classify an already-flattened (HTML-stripped) footnote body.

    Order matters: the manuscript-variant template is checked first because a few variant
    notes ("في ت: \"ورواه الترمذي وقال: حديث حسن\"") quote a takhrij-shaped fragment as the
    *variant reading itself* -- the template match, not the keyword, decides those.
    """

    text = flattened_note.strip()
    if _MANUSCRIPT_VARIANT_RE.search(text) or _MANUSCRIPT_INLINE_RE.search(text):
        return FOOTNOTE_STRIP
    if _NARRATION_VERB_RE.search(text) or _LOCATOR_DIGIT_RE.search(text):
        return FOOTNOTE_REFERENCE
    return FOOTNOTE_STRIP


def split_footnotes(raw_html: str) -> tuple[str, list[str]]:
    """Remove ``[[...]]`` footnotes from raw HTML, returning (remaining_html, references).

    ``references`` holds the flattened text of every footnote classified as a hadith/source
    reference, in document order, duplicates included (a resolved passage may cite the same
    work twice).
    """

    references: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        flattened = _flatten_html(match.group(1))
        if flattened and classify_footnote(flattened) == FOOTNOTE_REFERENCE:
            references.append(flattened)
        return ""

    remaining = _FOOTNOTE_RE.sub(_replace, raw_html)
    return remaining, references


@dataclass(frozen=True)
class TafsirPassage:
    resource_alias: str
    chapter: int
    verse_start: int
    verse_end: int
    text: str
    hadith_references: tuple[str, ...]

    @property
    def ayah_keys(self) -> list[str]:
        return [f"{self.chapter}:{verse}" for verse in range(self.verse_start, self.verse_end + 1)]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class TafsirShapeError(ValueError):
    """The QUL export does not match the shape verified for this plan."""


def _parse_key(key: str) -> tuple[int, int]:
    chapter_str, verse_str = key.split(":")
    return int(chapter_str), int(verse_str)


def load_qul_export(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_passages(raw: dict[str, Any], *, resource_alias: str) -> list[TafsirPassage]:
    """Resolve string-pointer keys to their owning passage and deduplicate.

    Every key in the export is either an owning entry (a dict with ``text`` and ``ayah_keys``)
    or a pointer (a bare string naming the owning key). Each owning key is emitted exactly
    once, keyed by ``ayah_keys`` for its verse range -- so a five-ayah passage that occupies
    five keys in the raw export becomes exactly one `TafsirPassage`.
    """

    owners: dict[str, dict[str, Any]] = {
        key: value for key, value in raw.items() if isinstance(value, dict)
    }
    passages: list[TafsirPassage] = []
    seen_owner_keys: set[str] = set()
    for key, value in raw.items():
        owner_key = value if isinstance(value, str) else key
        if owner_key in seen_owner_keys:
            continue
        owner = owners.get(owner_key)
        if owner is None:
            # A pointer to a key that is itself neither a dict nor resolvable -- a shape
            # violation. Skip rather than fabricate a passage; surfaced in ingestion counts.
            continue
        seen_owner_keys.add(owner_key)
        ayah_keys = owner.get("ayah_keys") or [owner_key]
        first_chapter, first_verse = _parse_key(ayah_keys[0])
        last_chapter, last_verse = _parse_key(ayah_keys[-1])
        if first_chapter != last_chapter:
            raise TafsirShapeError(
                f"{resource_alias}: passage at {owner_key!r} crosses a surah boundary "
                f"({ayah_keys[0]} .. {ayah_keys[-1]})"
            )
        remaining_html, references = split_footnotes(owner.get("text", ""))
        clean_text = strip_html(remaining_html)
        if not clean_text:
            continue
        passages.append(
            TafsirPassage(
                resource_alias=resource_alias,
                chapter=first_chapter,
                verse_start=min(first_verse, last_verse),
                verse_end=max(first_verse, last_verse),
                text=clean_text,
                hadith_references=tuple(references),
            )
        )
    passages.sort(key=lambda passage: (passage.chapter, passage.verse_start))
    return passages


def parse_qul_export(path: Path, *, resource_alias: str) -> list[TafsirPassage]:
    return resolve_passages(load_qul_export(path), resource_alias=resource_alias)
