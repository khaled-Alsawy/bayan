from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from islamic_research_ai.composition import build_container
from islamic_research_ai.contracts import QuranSearchCandidateV1
from islamic_research_ai.routing import ProviderRoutePolicy

DEVELOPMENT_QUERIES: list[dict[str, Any]] = [
    {
        "id": "en-patience-hardship",
        "query": "What does the Quran say about patience during hardship?",
        "language_mix": "English",
        "expected": ["2:153", "2:155", "2:156", "2:157", "16:127", "46:35"],
    },
    {
        "id": "ar-patience-hardship",
        "query": "ماذا يقول القرآن عن الصبر عند الشدائد؟",
        "language_mix": "Arabic",
        "expected": ["2:153", "2:155", "2:156", "2:157", "16:127", "46:35"],
    },
    {
        "id": "en-paraphrase-steadfast",
        "query": "How should someone stay steadfast when life becomes painful?",
        "language_mix": "English paraphrase; no required exact lexical match",
        "expected": ["2:153", "2:155", "2:156", "2:157", "3:200", "94:5", "94:6"],
    },
    {
        "id": "en-gratitude",
        "query": "gratitude for blessings",
        "language_mix": "English",
        "expected": ["2:152", "14:7", "31:12"],
    },
    {
        "id": "ar-gratitude",
        "query": "الشكر على النعم",
        "language_mix": "Arabic",
        "expected": ["2:152", "14:7", "31:12"],
    },
    {
        "id": "en-parents",
        "query": "kindness and care for parents",
        "language_mix": "English",
        "expected": ["17:23", "17:24", "31:14", "46:15"],
    },
    {
        "id": "mixed-parents",
        "query": "الوالدين mercy and kindness",
        "language_mix": "Arabic/English mixed",
        "expected": ["17:23", "17:24", "31:14", "46:15"],
    },
    {
        "id": "en-charity",
        "query": "charity and spending for those in need",
        "language_mix": "English",
        "expected": ["2:261", "2:267", "2:271", "2:274", "9:60"],
    },
    {
        "id": "en-charity-paraphrase",
        "query": "giving quietly to people who need help",
        "language_mix": "English paraphrase; no required exact lexical match",
        "expected": ["2:271", "2:274"],
    },
    {
        "id": "en-mercy",
        "query": "God's mercy for people who have sinned",
        "language_mix": "English",
        "expected": ["7:156", "39:53"],
    },
    {
        "id": "ar-repentance-forgiveness",
        "query": "التوبة والمغفرة بعد الذنب",
        "language_mix": "Arabic",
        "expected": ["25:70", "39:53", "66:8"],
    },
    {
        "id": "en-forgiveness",
        "query": "forgiving other people after they wrong you",
        "language_mix": "English",
        "expected": ["3:134", "24:22", "42:40", "42:43"],
    },
    {
        "id": "en-justice",
        "query": "stand for justice even against yourself",
        "language_mix": "English",
        "expected": ["4:135", "5:8"],
    },
    {
        "id": "ar-justice",
        "query": "العدل وعدم الظلم",
        "language_mix": "Arabic",
        "expected": ["4:135", "5:8", "16:90"],
    },
    {
        "id": "en-arrogance",
        "query": "arrogance and looking down on people",
        "language_mix": "English",
        "expected": ["7:146", "16:23", "31:18"],
    },
    {
        "id": "mixed-prayer-distress",
        "query": "الصلاة when facing distress",
        "language_mix": "Arabic/English mixed",
        "expected": ["2:45", "2:153"],
    },
    {
        "id": "en-knowledge",
        "query": "seeking and increasing knowledge",
        "language_mix": "English",
        "expected": ["20:114", "39:9", "58:11"],
    },
    {
        "id": "mixed-knowledge",
        "query": "طلب العلم and people who know",
        "language_mix": "Arabic/English mixed",
        "expected": ["20:114", "39:9", "58:11"],
    },
    {
        "id": "en-gentle-response-paraphrase",
        "query": "responding gently to people who insult you",
        "language_mix": "English paraphrase; no required exact lexical match",
        "expected": ["25:63", "41:34"],
    },
    {
        "id": "en-over-specific-no-answer",
        "query": "Which verse guarantees my exact business recovers before Friday?",
        "language_mix": "English; intentionally no good answer",
        "expected": [],
    },
]


MANUAL_REVIEWS: dict[str, dict[str, Any]] = {
    "en-patience-hardship": {
        "reasonable_additions": ["70:5", "11:115"],
        "false_positives": [],
        "assessment": {
            "70:5": "BROADLY_RELEVANT: explicit patience, but hardship is not explicit.",
            "16:127": "DIRECTLY_RELEVANT: patience is joined to grief and distress.",
            "2:153": "DIRECTLY_RELEVANT: patience and prayer are presented as help.",
            "46:35": "BROADLY_RELEVANT: perseverance under prophetic opposition.",
            "11:115": "BROADLY_RELEVANT: patience and preserved reward, without explicit hardship.",
        },
        "note": "Strong top five; the ranking distinguishes direct from broader patience verses.",
    },
    "ar-patience-hardship": {
        "reasonable_additions": ["8:66", "8:46", "3:17"],
        "false_positives": [],
        "assessment": {
            "2:153": "DIRECTLY_RELEVANT",
            "8:66": "BROADLY_RELEVANT: steadfastness in a battle context.",
            "8:46": "BROADLY_RELEVANT: steadfastness and endurance in conflict.",
            "2:155": "DIRECTLY_RELEVANT: trials are explicitly paired with patience.",
            "3:17": "BROADLY_RELEVANT: praises the patient without naming hardship.",
        },
        "note": "Arabic stopword normalization materially improved the result set.",
    },
    "en-paraphrase-steadfast": {
        "reasonable_additions": ["8:46", "8:66", "16:126"],
        "false_positives": ["70:23", "14:27"],
        "assessment": {
            "8:46": "BROADLY_RELEVANT: commands steadfastness.",
            "70:23": "LIKELY_FALSE_POSITIVE: constancy in prayer, not painful hardship.",
            "8:66": "BROADLY_RELEVANT: steadfastness under battle burden.",
            "16:126": "BROADLY_RELEVANT: patience after retaliation or injury.",
            "14:27": "LIKELY_FALSE_POSITIVE: steadfast speech, not coping with pain.",
        },
        "note": (
            "Semantic paraphrase retrieves the steadfastness theme but misses stronger "
            "hardship passages."
        ),
    },
    "en-gratitude": {
        "reasonable_additions": ["54:35", "16:114"],
        "false_positives": ["51:4", "19:50", "33:56"],
        "assessment": {
            "54:35": "DIRECTLY_RELEVANT: reward for those who give thanks.",
            "16:114": "DIRECTLY_RELEVANT: gratitude for blessings.",
            "51:4": "LIKELY_FALSE_POSITIVE: distributing affairs, unrelated to gratitude.",
            "33:56": "LIKELY_FALSE_POSITIVE: blessing polysemy, not gratitude for blessings.",
            "19:50": "LIKELY_FALSE_POSITIVE: being blessed, not expressing gratitude.",
        },
        "note": "Polysemy around 'blessings' causes lexical false positives.",
    },
    "ar-gratitude": {
        "reasonable_additions": ["54:35", "76:3"],
        "false_positives": ["24:61", "2:143", "24:58"],
        "assessment": {
            "24:61": "LIKELY_FALSE_POSITIVE",
            "54:35": "DIRECTLY_RELEVANT",
            "2:143": "LIKELY_FALSE_POSITIVE",
            "76:3": "DIRECTLY_RELEVANT: gratitude versus ingratitude.",
            "24:58": "LIKELY_FALSE_POSITIVE",
        },
        "note": "Arabic lexical matching still has morphological collisions for this short query.",
    },
    "en-parents": {
        "reasonable_additions": ["29:8", "93:6"],
        "false_positives": [],
        "assessment": {
            "17:23": "DIRECTLY_RELEVANT",
            "31:14": "DIRECTLY_RELEVANT",
            "29:8": "DIRECTLY_RELEVANT: kindness to parents with a faith boundary.",
            "93:6": "BROADLY_RELEVANT: care for an orphan, not specifically parents.",
            "46:15": "DIRECTLY_RELEVANT",
        },
        "note": "High-quality result set; 93:6 is broader caregiving evidence.",
    },
    "mixed-parents": {
        "reasonable_additions": [],
        "false_positives": ["31:3", "27:77", "36:44", "107:7"],
        "assessment": {
            "31:3": "LIKELY_FALSE_POSITIVE",
            "17:24": "DIRECTLY_RELEVANT",
            "27:77": "LIKELY_FALSE_POSITIVE",
            "36:44": "LIKELY_FALSE_POSITIVE",
            "107:7": "LIKELY_FALSE_POSITIVE: kindness is not parent-specific.",
        },
        "note": "The mixed-language query overweights generic mercy/kindness meanings.",
    },
    "en-charity": {
        "reasonable_additions": ["2:272", "2:215", "8:3", "34:39"],
        "false_positives": [],
        "assessment": {
            "2:272": "DIRECTLY_RELEVANT",
            "2:215": "DIRECTLY_RELEVANT",
            "2:274": "DIRECTLY_RELEVANT",
            "34:39": "DIRECTLY_RELEVANT: spending and divine provision.",
            "8:3": "BROADLY_RELEVANT: spending from provision.",
        },
        "note": "All selected results concern spending or charity.",
    },
    "en-charity-paraphrase": {
        "reasonable_additions": ["90:16", "90:14", "17:28"],
        "false_positives": ["8:26"],
        "assessment": {
            "90:16": "DIRECTLY_RELEVANT in context: feeding a needy person.",
            "8:26": "LIKELY_FALSE_POSITIVE",
            "17:28": "BROADLY_RELEVANT: kind words when unable to give to those in need.",
            "8:72": "BROADLY_RELEVANT: shelter and practical help, not quiet charity.",
            "90:14": "DIRECTLY_RELEVANT in context: feeding during famine.",
        },
        "note": "It finds the needy-feeding passage but misses the expected secrecy passages.",
    },
    "en-mercy": {
        "reasonable_additions": ["9:27"],
        "false_positives": ["2:282", "27:77", "2:196", "31:3"],
        "assessment": {
            "2:282": "LIKELY_FALSE_POSITIVE",
            "27:77": "LIKELY_FALSE_POSITIVE: generic mercy.",
            "2:196": "LIKELY_FALSE_POSITIVE",
            "9:27": "DIRECTLY_RELEVANT: repentance and forgiveness.",
            "31:3": "LIKELY_FALSE_POSITIVE: generic mercy.",
        },
        "note": "The intended sin/repentance qualifier is underweighted.",
    },
    "ar-repentance-forgiveness": {
        "reasonable_additions": ["7:153", "2:52", "16:119", "3:89"],
        "false_positives": ["2:109"],
        "assessment": {
            "7:153": "DIRECTLY_RELEVANT",
            "2:52": "BROADLY_RELEVANT: divine pardon after wrongdoing.",
            "16:119": "DIRECTLY_RELEVANT",
            "2:109": "LIKELY_FALSE_POSITIVE",
            "3:89": "DIRECTLY_RELEVANT",
        },
        "note": "Good thematic alternatives despite missing the initially expected references.",
    },
    "en-forgiveness": {
        "reasonable_additions": [],
        "false_positives": ["9:102", "2:199", "5:39", "4:110", "27:11"],
        "assessment": {
            "9:102": "LIKELY_FALSE_POSITIVE: seeking divine forgiveness.",
            "2:199": "LIKELY_FALSE_POSITIVE: seeking divine forgiveness.",
            "5:39": "LIKELY_FALSE_POSITIVE: repentance by the wrongdoer.",
            "4:110": "LIKELY_FALSE_POSITIVE: seeking divine forgiveness.",
            "27:11": "LIKELY_FALSE_POSITIVE: wronging oneself.",
        },
        "note": (
            "Critical semantic distinction failure: divine forgiveness displaces forgiving "
            "other people."
        ),
    },
    "en-justice": {
        "reasonable_additions": [],
        "false_positives": ["75:14", "77:46", "51:21"],
        "assessment": {
            "4:135": "DIRECTLY_RELEVANT",
            "5:8": "DIRECTLY_RELEVANT",
            "75:14": "LIKELY_FALSE_POSITIVE: self-witness wording without justice.",
            "77:46": "LIKELY_FALSE_POSITIVE",
            "51:21": "LIKELY_FALSE_POSITIVE: self-reference without justice.",
        },
        "note": "The two most direct verses rank first and second.",
    },
    "ar-justice": {
        "reasonable_additions": ["3:57", "26:209", "11:85"],
        "false_positives": ["2:257", "21:87"],
        "assessment": {
            "3:57": "BROADLY_RELEVANT: rejection of wrongdoing.",
            "2:257": "LIKELY_FALSE_POSITIVE",
            "26:209": "BROADLY_RELEVANT: divine non-injustice.",
            "11:85": "DIRECTLY_RELEVANT: just weights and no defrauding.",
            "21:87": "LIKELY_FALSE_POSITIVE",
        },
        "note": "Mixed quality; one direct justice passage and two broad non-injustice passages.",
    },
    "en-arrogance": {
        "reasonable_additions": [],
        "false_positives": ["4:11", "37:14", "2:282", "26:55"],
        "assessment": {
            "31:18": "DIRECTLY_RELEVANT",
            "4:11": "LIKELY_FALSE_POSITIVE",
            "37:14": "LIKELY_FALSE_POSITIVE",
            "2:282": "LIKELY_FALSE_POSITIVE",
            "26:55": "LIKELY_FALSE_POSITIVE",
        },
        "note": "Two direct hits followed by weak semantic matches.",
    },
    "mixed-prayer-distress": {
        "reasonable_additions": [],
        "false_positives": ["3:153", "76:10", "16:127", "6:64", "26:13"],
        "assessment": {
            "3:153": "LIKELY_FALSE_POSITIVE: distress without prayer guidance.",
            "76:10": "LIKELY_FALSE_POSITIVE: distress without prayer guidance.",
            "16:127": "LIKELY_FALSE_POSITIVE for the full prayer-and-distress concept.",
            "6:64": "LIKELY_FALSE_POSITIVE for the full prayer-and-distress concept.",
            "26:13": "LIKELY_FALSE_POSITIVE: distress narrative without prayer guidance.",
        },
        "note": (
            "Material conjunction failure: distress matches displace the required prayer concept."
        ),
    },
    "en-knowledge": {
        "reasonable_additions": ["26:83", "28:14"],
        "false_positives": ["78:11", "26:112", "52:41"],
        "assessment": {
            "26:83": "DIRECTLY_RELEVANT: prayer for knowledge/wisdom.",
            "78:11": "LIKELY_FALSE_POSITIVE: seeking livelihood, not knowledge.",
            "28:14": "BROADLY_RELEVANT: wisdom and knowledge granted.",
            "26:112": "LIKELY_FALSE_POSITIVE",
            "52:41": "LIKELY_FALSE_POSITIVE: unseen knowledge, not seeking knowledge.",
        },
        "note": "The exact 20:114 match appears lexically but is lost in fusion.",
    },
    "mixed-knowledge": {
        "reasonable_additions": ["22:54", "34:6", "20:110"],
        "false_positives": ["33:63"],
        "assessment": {
            "39:9": "DIRECTLY_RELEVANT",
            "22:54": "DIRECTLY_RELEVANT: those given knowledge recognize truth.",
            "34:6": "DIRECTLY_RELEVANT: people with knowledge recognize truth.",
            "33:63": "LIKELY_FALSE_POSITIVE: knowledge of the Hour, not seeking knowledge.",
            "20:110": "BROADLY_RELEVANT: limits of human knowledge.",
        },
        "note": "Useful knowledge passages dominate four of five results.",
    },
    "en-gentle-response-paraphrase": {
        "reasonable_additions": ["20:44", "28:55"],
        "false_positives": ["79:2", "27:62", "27:28"],
        "assessment": {
            "79:2": "LIKELY_FALSE_POSITIVE: gentle extraction, not interpersonal response.",
            "20:44": "DIRECTLY_RELEVANT: command to speak gently.",
            "27:62": "LIKELY_FALSE_POSITIVE: divine response, not responding to insults.",
            "28:55": "DIRECTLY_RELEVANT: peaceful response to vain speech.",
            "27:28": "LIKELY_FALSE_POSITIVE: delivery instruction using response vocabulary.",
        },
        "note": "One strong paraphrase hit; four lexical/semantic false positives.",
    },
    "en-over-specific-no-answer": {
        "reasonable_additions": [],
        "false_positives": ["14:31", "2:41", "20:130", "12:100", "77:7"],
        "assessment": {
            "2:41": "LIKELY_FALSE_POSITIVE",
            "20:130": "LIKELY_FALSE_POSITIVE: promised reward, not business recovery.",
            "77:7": "LIKELY_FALSE_POSITIVE: eschatological promise, not a business guarantee.",
            "12:100": "LIKELY_FALSE_POSITIVE: fulfilled dream narrative.",
            "14:31": "LIKELY_FALSE_POSITIVE: an eschatological deadline, not Friday.",
        },
        "note": (
            "Correct product behavior is insufficiency; retrieval alone cannot establish no-answer."
        ),
    },
}


def _reference(item: QuranSearchCandidateV1) -> str:
    key = item.verse_keys[0]
    return f"{key.chapter}:{key.verse}"


def _ranked(
    candidates: list[QuranSearchCandidateV1], component: str | None = None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for position, item in enumerate(candidates, start=1):
        output.append(
            {
                "rank": (
                    int(item.ranking.component_scores[component])
                    if component is not None
                    else position
                ),
                "reference": _reference(item),
                "candidate_id": item.candidate_id,
                "rrf_score": item.ranking.component_scores.get("rrf_score"),
                "lexical_rank": item.ranking.component_scores.get("lexical_rank"),
                "vector_rank": item.ranking.component_scores.get("vector_rank"),
            }
        )
    return output


async def run_sanity(artifact_path: Path | None = None) -> dict[str, Any]:
    container = await build_container()
    try:
        queries = [item["query"] for item in DEVELOPMENT_QUERIES]
        vectors, completion = await container.embeddings.embed(queries)
        ProviderRoutePolicy.require_cloudflare_bge_m3_direct_free(completion)
        evaluations: list[dict[str, Any]] = []
        for definition, vector in zip(DEVELOPMENT_QUERIES, vectors, strict=True):
            review = MANUAL_REVIEWS[definition["id"]]
            components = await container.repository.hybrid_search_components(
                query=definition["query"], query_embedding=vector, limit=8
            )
            selected = components["fused"][:5]
            evidence = await container.service.source.hydrate_candidates(selected, limit=5)
            fused_references = [_reference(item) for item in selected]
            expected = list(
                dict.fromkeys([*definition["expected"], *review["reasonable_additions"]])
            )
            evaluations.append(
                {
                    "id": definition["id"],
                    "query": definition["query"],
                    "language_mix": definition["language_mix"],
                    "lexical_top_candidates": _ranked(components["lexical"], "lexical_rank"),
                    "vector_top_candidates": _ranked(components["vector"], "vector_rank"),
                    "fused_rrf_candidates": _ranked(components["fused"]),
                    "hydrated_selections": fused_references,
                    "hydrated_evidence_items": [item.model_dump(mode="json") for item in evidence],
                    "manually_expected_or_reasonable_references": expected,
                    "expected_references_retrieved_in_top_5": [
                        reference for reference in fused_references if reference in expected
                    ],
                    "manual_top_5_assessment": review["assessment"],
                    "obvious_false_positives": review["false_positives"],
                    "manual_review_note": review["note"],
                }
            )
        artifact = {
            "artifact_kind": "DEVELOPMENT_RETRIEVAL_SANITY_SET",
            "holdout_status": "NOT_FROZEN_FINAL_HOLDOUT",
            "created_at": datetime.now(UTC).isoformat(),
            "query_count": len(evaluations),
            "provider_completion": completion.model_dump(mode="json"),
            "evaluations": evaluations,
        }
        path = artifact_path or (
            container.settings.artifact_dir / "quran-retrieval-sanity-development.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return artifact
    finally:
        await container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    artifact = asyncio.run(run_sanity(args.artifact))
    print(
        json.dumps(
            {
                "query_count": artifact["query_count"],
                "artifact_kind": artifact["artifact_kind"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
