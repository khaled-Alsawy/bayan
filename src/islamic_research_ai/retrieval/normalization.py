from __future__ import annotations

import re
import unicodedata

_ARABIC_MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_NON_WORD = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)
_ARABIC_WORD = re.compile(r"[\u0621-\u064a]+")
_PROCLITICS = ("و", "ف", "ب", "ك", "ل")
_INFLECTIONAL_SUFFIXES = (
    "كما",
    "هما",
    "يات",
    "ات",
    "ون",
    "ين",
    "ان",
    "وا",
    "ها",
    "هم",
    "هن",
    "كم",
    "كن",
    "نا",
    "ية",
    "ك",
    "ي",
    "ه",
    "ة",
    "ا",
)


def normalize_arabic(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = _ARABIC_MARKS.sub("", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))
    return " ".join(_NON_WORD.sub(" ", value).split())


def normalize_translation(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_NON_WORD.sub(" ", value).split())


def arabic_words(value: str) -> list[str]:
    return _ARABIC_WORD.findall(normalize_arabic(value))


def arabic_light_stem(token: str) -> str:
    """Strip productive clitics/inflections without assigning a lexical meaning."""

    value = normalize_arabic(token)
    if len(value) >= 5 and value[0] in _PROCLITICS:
        value = value[1:]
    if value.startswith("ال") and len(value) >= 5:
        value = value[2:]
    if value.endswith("انا") and len(value) >= 6:
        return value[:-1]
    changed = True
    while changed:
        changed = False
        for suffix in _INFLECTIONAL_SUFFIXES:
            minimum = 4 if suffix in {"يات", "ات", "ون", "ين", "ان", "وا", "ية"} else 3
            if suffix == "ات":
                minimum = 3
            if value.endswith(suffix) and len(value) - len(suffix) >= minimum:
                value = value[: -len(suffix)]
                changed = suffix != "ا"
                break
    return value
