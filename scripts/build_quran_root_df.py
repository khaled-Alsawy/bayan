#!/usr/bin/env python3
"""Build verse document frequencies for conservative Quran relevance keys."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from islamic_research_ai.retrieval.relevance import (  # noqa: E402
    _arabic_match_key,
    _content_tokens,
    arabic_light_root,
)


def main() -> None:
    verse_keys: Counter[str] = Counter()
    total_verses = 0
    changed_token_occurrences = 0
    source = ROOT / "data/raw/quran-uthmani.txt"
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        chapter, verse, text = line.split("|", maxsplit=2)
        del chapter, verse
        total_verses += 1
        keys: set[str] = set()
        for token in _content_tokens(text):
            before = arabic_light_root(token)
            after = arabic_light_root(token, enable_taf3il=True)
            if before != after:
                changed_token_occurrences += 1
            keys.add(_arabic_match_key(token, distinctive_terms_enabled=True))
        verse_keys.update(keys)

    target = ROOT / "src/islamic_research_ai/retrieval/quran_root_df.json"
    target.write_text(
        json.dumps(
            {"total_verses": total_verses, "df": dict(sorted(verse_keys.items()))},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"total_verses={total_verses}")
    print(f"df_keys={len(verse_keys)}")
    print(f"changed_token_occurrences={changed_token_occurrences}")


if __name__ == "__main__":
    main()
