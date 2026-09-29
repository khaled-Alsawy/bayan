from __future__ import annotations

from pathlib import Path

import pytest

from islamic_research_ai.ingestion.tafsir_qul import (
    FOOTNOTE_REFERENCE,
    FOOTNOTE_STRIP,
    TafsirShapeError,
    classify_footnote,
    parse_qul_export,
    resolve_passages,
    split_footnotes,
    strip_html,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUL_DIR = PROJECT_ROOT / "data" / "raw" / "qul" / "extracted"

QUL_EXPORTS = {
    "qul-tafsir-almukhtasar-ar-v1": (
        "arabic-al-mukhtasar-in-interpreting-the-noble-quran.json",
        6211,
    ),
    "qul-tafsir-ibnkathir-ar-v1": ("ar-tafsir-ibn-kathir.json", 1911),
    "qul-tafsir-alqurtubi-ar-v1": ("ar-tafseer-al-qurtubi.json", 3258),
    "qul-tafsir-albaghawi-ar-v1": ("ar-tafsir-al-baghawi.json", 2152),
}

requires_qul_corpus = pytest.mark.skipif(
    not QUL_DIR.exists(), reason="QUL corpus not downloaded to data/raw/qul/extracted/"
)


# --- unit tests: footnote classification and HTML stripping (no fixtures needed) ----------


def test_strip_html_preserves_paragraph_breaks_and_removes_tags() -> None:
    raw = "<div class=ar lang=ar><p>الأول.</p><p>الثاني.</p></div>"
    assert strip_html(raw) == "الأول.\nالثاني."


def test_strip_html_unescapes_entities() -> None:
    assert strip_html("<p>a &amp; b</p>") == "a & b"


def test_classify_footnote_manuscript_addition_is_stripped() -> None:
    assert classify_footnote("زيادة من جـ، ر، أ، و، والمسند.") == FOOTNOTE_STRIP


def test_classify_footnote_manuscript_inline_variant_is_stripped() -> None:
    assert classify_footnote('في ت: "ورواه الترمذي وقال: حديث حسن".') == FOOTNOTE_STRIP


def test_classify_footnote_takhrij_with_narration_verb_is_kept() -> None:
    note = "رواه البخاري في صحيحه برقم (٥٥٣) والنسائي في السنن (١/٢٣٦)."
    assert classify_footnote(note) == FOOTNOTE_REFERENCE


def test_classify_footnote_page_locator_without_narration_verb_is_kept() -> None:
    assert classify_footnote("تفسير الطبري (١/٢٧٣) .") == FOOTNOTE_REFERENCE


def test_classify_footnote_bare_prose_defaults_to_strip() -> None:
    # No manuscript template and no locator/narration-verb hint: the safe default is STRIP,
    # never a fabricated reference.
    assert classify_footnote("قول لا يستند إلى شيء مما سبق") == FOOTNOTE_STRIP


def test_split_footnotes_removes_bracket_spans_and_collects_references() -> None:
    raw = "<p>نص المتن. [[زيادة من جـ.]] وتتمة النص. [[رواه البخاري برقم (١) .]]</p>"
    remaining, references = split_footnotes(raw)
    assert "زيادة من" not in remaining
    assert "رواه البخاري" not in remaining
    assert references == ["رواه البخاري برقم (١) ."]
    assert strip_html(remaining) == "نص المتن. وتتمة النص."


# --- pointer resolution on synthetic data (no fixtures needed) -----------------------------


def test_resolve_passages_dedupes_grouped_pointer_keys() -> None:
    raw = {
        "113:1": {"text": "<p>نص الآيات ١-٥.</p>", "ayah_keys": ["113:1", "113:2", "113:3"]},
        "113:2": "113:1",
        "113:3": "113:1",
    }
    passages = resolve_passages(raw, resource_alias="test-resource")
    assert len(passages) == 1
    passage = passages[0]
    assert (passage.chapter, passage.verse_start, passage.verse_end) == (113, 1, 3)
    assert passage.ayah_keys == ["113:1", "113:2", "113:3"]
    assert passage.resource_alias == "test-resource"


def test_resolve_passages_single_ayah_entries_are_independent() -> None:
    raw = {
        "1:1": {"text": "<p>أ</p>"},
        "1:2": {"text": "<p>ب</p>"},
    }
    passages = resolve_passages(raw, resource_alias="test-resource")
    assert [(p.chapter, p.verse_start, p.verse_end) for p in passages] == [(1, 1, 1), (1, 2, 2)]


def test_resolve_passages_rejects_surah_boundary_crossing() -> None:
    raw = {
        "1:7": {"text": "<p>x</p>", "ayah_keys": ["1:7", "2:1"]},
        "2:1": "1:7",
    }
    with pytest.raises(TafsirShapeError):
        resolve_passages(raw, resource_alias="test-resource")


def test_resolve_passages_drops_entries_that_are_empty_after_stripping() -> None:
    raw = {"34:54": {"text": "<div class=ar lang=ar></div>"}}
    assert resolve_passages(raw, resource_alias="test-resource") == []


def test_resolve_passages_content_hash_is_stable_and_text_dependent() -> None:
    raw_a = {"1:1": {"text": "<p>نص أ</p>"}}
    raw_b = {"1:1": {"text": "<p>نص ب</p>"}}
    (passage_a,) = resolve_passages(raw_a, resource_alias="r")
    (passage_b,) = resolve_passages(raw_b, resource_alias="r")
    assert passage_a.content_hash != passage_b.content_hash
    (passage_a_again,) = resolve_passages(raw_a, resource_alias="r")
    assert passage_a.content_hash == passage_a_again.content_hash


# --- acceptance tests against the real, downloaded QUL corpus ------------------------------
#
# These are the numbers WP-0 was verified against. They are skipped, not failed, when the
# corpus archives have not been extracted -- the plan treats the corpus as an already-acquired
# external asset, not something ingestion tests should download.


@requires_qul_corpus
@pytest.mark.parametrize(("resource_alias", "filename_and_count"), list(QUL_EXPORTS.items()))
def test_qul_export_passage_count_matches_verified_baseline(
    resource_alias: str, filename_and_count: tuple[str, int]
) -> None:
    filename, expected_count = filename_and_count
    passages = parse_qul_export(QUL_DIR / filename, resource_alias=resource_alias)
    assert len(passages) == expected_count


@requires_qul_corpus
def test_qul_exports_cover_the_full_quran_with_at_most_a_handful_of_source_gaps() -> None:
    # Every resource must resolve ayah 1:1 through 114:6; a handful of ayat with genuinely
    # empty commentary in the source print (verified: Qurtubi 34:54, 59:19) is expected and
    # is not a parser defect -- see the WP-0 section of
    # CONVERSATIONAL_QUALITY_IMPLEMENTATION_REPORT.md for the spot-check.
    for resource_alias, (filename, _) in QUL_EXPORTS.items():
        passages = parse_qul_export(QUL_DIR / filename, resource_alias=resource_alias)
        covered = {key for passage in passages for key in passage.ayah_keys}
        raw = {
            key
            for key in (__import__("json").loads((QUL_DIR / filename).read_text(encoding="utf-8")))
        }
        missing = raw - covered
        assert len(missing) <= 3, f"{resource_alias} unexpectedly missing {sorted(missing)}"


@requires_qul_corpus
def test_al_mukhtasar_is_html_free_search_tier_text() -> None:
    passages = parse_qul_export(
        QUL_DIR / QUL_EXPORTS["qul-tafsir-almukhtasar-ar-v1"][0],
        resource_alias="qul-tafsir-almukhtasar-ar-v1",
    )
    assert all("<" not in passage.text and ">" not in passage.text for passage in passages)
    average_chars = sum(len(p.text) for p in passages) / len(passages)
    assert 100 <= average_chars <= 250


@requires_qul_corpus
def test_ibn_kathir_hadith_reference_share_is_in_the_measured_range() -> None:
    # Measured: ~26-28% of Ibn Kathir's 31,203 footnotes are takhrij kept as references.
    passages = parse_qul_export(
        QUL_DIR / QUL_EXPORTS["qul-tafsir-ibnkathir-ar-v1"][0],
        resource_alias="qul-tafsir-ibnkathir-ar-v1",
    )
    reference_count = sum(len(p.hadith_references) for p in passages)
    assert 7000 <= reference_count <= 9500
