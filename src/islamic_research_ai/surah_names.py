"""The 114 surah names, Arabic and transliterated, for `AntiFabricationGate.REFERENCE_GUARD`
(WP-2, V3 plan).

Names are the standard indices used across essentially every Quran edition/index. A second
Arabic name is included for the handful of surahs with a common alternate (Ghafir/Al-Mu'min,
Al-Insan/Ad-Dahr, Ash-Sharh/Al-Inshirah, Al-Masad/Al-Lahab). Transliterations follow the
common English-Quran-translation convention (the form most likely to appear in an English
narrative, e.g. "Al-Baqarah", "Yaseen").
"""

from __future__ import annotations

# chapter -> (arabic names, transliterated names)
SURAH_NAMES: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {
    1: (("الفاتحة",), ("Al-Fatiha", "Al-Fatihah")),
    2: (("البقرة",), ("Al-Baqarah", "Al-Baqara")),
    3: (("آل عمران",), ("Al-Imran", "Aal-e-Imran")),
    4: (("النساء",), ("An-Nisa", "An-Nisaa")),
    5: (("المائدة",), ("Al-Maidah", "Al-Ma'idah")),
    6: (("الأنعام",), ("Al-Anam", "Al-An'am")),
    7: (("الأعراف",), ("Al-Araf", "Al-A'raf")),
    8: (("الأنفال",), ("Al-Anfal",)),
    9: (("التوبة",), ("At-Tawbah", "At-Tawba")),
    10: (("يونس",), ("Yunus",)),
    11: (("هود",), ("Hud",)),
    12: (("يوسف",), ("Yusuf",)),
    13: (("الرعد",), ("Ar-Rad", "Ar-Ra'd")),
    14: (("إبراهيم",), ("Ibrahim",)),
    15: (("الحجر",), ("Al-Hijr",)),
    16: (("النحل",), ("An-Nahl",)),
    17: (("الإسراء",), ("Al-Isra",)),
    18: (("الكهف",), ("Al-Kahf",)),
    19: (("مريم",), ("Maryam",)),
    20: (("طه",), ("Taha",)),
    21: (("الأنبياء",), ("Al-Anbiya",)),
    22: (("الحج",), ("Al-Hajj",)),
    23: (("المؤمنون",), ("Al-Muminun", "Al-Mu'minun")),
    24: (("النور",), ("An-Nur",)),
    25: (("الفرقان",), ("Al-Furqan",)),
    26: (("الشعراء",), ("Ash-Shuara", "Ash-Shu'ara")),
    27: (("النمل",), ("An-Naml",)),
    28: (("القصص",), ("Al-Qasas",)),
    29: (("العنكبوت",), ("Al-Ankabut",)),
    30: (("الروم",), ("Ar-Rum",)),
    31: (("لقمان",), ("Luqman",)),
    32: (("السجدة",), ("As-Sajdah", "As-Sajda")),
    33: (("الأحزاب",), ("Al-Ahzab",)),
    34: (("سبأ",), ("Saba",)),
    35: (("فاطر",), ("Fatir",)),
    36: (("يس", "يسين"), ("Yaseen", "Ya-Sin")),
    37: (("الصافات",), ("As-Saffat",)),
    38: (("ص",), ("Sad",)),
    39: (("الزمر",), ("Az-Zumar",)),
    40: (("غافر", "المؤمن"), ("Ghafir", "Al-Mumin", "Al-Ghafir")),
    41: (("فصلت",), ("Fussilat",)),
    42: (("الشورى",), ("Ash-Shura",)),
    43: (("الزخرف",), ("Az-Zukhruf",)),
    44: (("الدخان",), ("Ad-Dukhan",)),
    45: (("الجاثية",), ("Al-Jathiyah", "Al-Jathiya")),
    46: (("الأحقاف",), ("Al-Ahqaf",)),
    47: (("محمد",), ("Muhammad",)),
    48: (("الفتح",), ("Al-Fath",)),
    49: (("الحجرات",), ("Al-Hujurat",)),
    50: (("ق",), ("Qaf",)),
    51: (("الذاريات",), ("Adh-Dhariyat",)),
    52: (("الطور",), ("At-Tur",)),
    53: (("النجم",), ("An-Najm",)),
    54: (("القمر",), ("Al-Qamar",)),
    55: (("الرحمن",), ("Ar-Rahman",)),
    56: (("الواقعة",), ("Al-Waqiah", "Al-Waqi'ah")),
    57: (("الحديد",), ("Al-Hadid",)),
    58: (("المجادلة",), ("Al-Mujadila", "Al-Mujadilah")),
    59: (("الحشر",), ("Al-Hashr",)),
    60: (("الممتحنة",), ("Al-Mumtahanah", "Al-Mumtahina")),
    61: (("الصف",), ("As-Saff",)),
    62: (("الجمعة",), ("Al-Jumuah", "Al-Jumu'ah")),
    63: (("المنافقون",), ("Al-Munafiqun",)),
    64: (("التغابن",), ("At-Taghabun",)),
    65: (("الطلاق",), ("At-Talaq",)),
    66: (("التحريم",), ("At-Tahrim",)),
    67: (("الملك",), ("Al-Mulk",)),
    68: (("القلم",), ("Al-Qalam",)),
    69: (("الحاقة",), ("Al-Haqqah", "Al-Haqqa")),
    70: (("المعارج",), ("Al-Maarij", "Al-Ma'arij")),
    71: (("نوح",), ("Nuh",)),
    72: (("الجن",), ("Al-Jinn",)),
    73: (("المزمل",), ("Al-Muzzammil",)),
    74: (("المدثر",), ("Al-Muddaththir",)),
    75: (("القيامة",), ("Al-Qiyamah", "Al-Qiyama")),
    76: (("الإنسان", "الدهر"), ("Al-Insan", "Ad-Dahr")),
    77: (("المرسلات",), ("Al-Mursalat",)),
    78: (("النبأ",), ("An-Naba",)),
    79: (("النازعات",), ("An-Naziat", "An-Nazi'at")),
    80: (("عبس",), ("Abasa",)),
    81: (("التكوير",), ("At-Takwir",)),
    82: (("الانفطار",), ("Al-Infitar",)),
    83: (("المطففين",), ("Al-Mutaffifin",)),
    84: (("الانشقاق",), ("Al-Inshiqaq",)),
    85: (("البروج",), ("Al-Buruj",)),
    86: (("الطارق",), ("At-Tariq",)),
    87: (("الأعلى",), ("Al-Ala", "Al-A'la")),
    88: (("الغاشية",), ("Al-Ghashiyah", "Al-Ghashiya")),
    89: (("الفجر",), ("Al-Fajr",)),
    90: (("البلد",), ("Al-Balad",)),
    91: (("الشمس",), ("Ash-Shams",)),
    92: (("الليل",), ("Al-Layl",)),
    93: (("الضحى",), ("Ad-Duha",)),
    94: (("الشرح", "الانشراح"), ("Ash-Sharh", "Al-Inshirah")),
    95: (("التين",), ("At-Tin",)),
    96: (("العلق",), ("Al-Alaq", "Al-'Alaq")),
    97: (("القدر",), ("Al-Qadr",)),
    98: (("البينة",), ("Al-Bayyinah", "Al-Bayyina")),
    99: (("الزلزلة",), ("Az-Zalzalah", "Az-Zalzala")),
    100: (("العاديات",), ("Al-Adiyat",)),
    101: (("القارعة",), ("Al-Qariah", "Al-Qari'ah")),
    102: (("التكاثر",), ("At-Takathur",)),
    103: (("العصر",), ("Al-Asr",)),
    104: (("الهمزة",), ("Al-Humazah",)),
    105: (("الفيل",), ("Al-Fil",)),
    106: (("قريش",), ("Quraysh", "Quraish")),
    107: (("الماعون",), ("Al-Maun", "Al-Ma'un")),
    108: (("الكوثر",), ("Al-Kawthar",)),
    109: (("الكافرون",), ("Al-Kafirun",)),
    110: (("النصر",), ("An-Nasr",)),
    111: (("المسد", "اللهب"), ("Al-Masad", "Al-Lahab")),
    112: (("الإخلاص",), ("Al-Ikhlas",)),
    113: (("الفلق",), ("Al-Falaq",)),
    114: (("الناس",), ("An-Nas",)),
}

assert len(SURAH_NAMES) == 114, "SURAH_NAMES must cover exactly the 114 surahs"


def build_name_to_chapter() -> dict[str, int]:
    """Every surface form (Arabic and transliterated), normalized-lowercase, to its chapter."""

    table: dict[str, int] = {}
    for chapter, (arabic_names, latin_names) in SURAH_NAMES.items():
        for name in arabic_names:
            table[name] = chapter
        for name in latin_names:
            table[name.lower()] = chapter
    return table


NAME_TO_CHAPTER = build_name_to_chapter()
