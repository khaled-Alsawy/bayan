"""Frozen, training-free question-bank retrieval pilot.

This is deliberately an *offline harness*, not a retrieval implementation.  In
particular, pool reconstruction calls the application's search arms, fusion and
late binding code; changing any of those production behaviours must make the
anchor assertion fail rather than silently changing this experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field


def cloudflare_account_id() -> str:
    """Account IDs stay out of source control: env first, then the gitignored .env.feasibility."""

    value = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if value:
        return value
    local = Path(__file__).resolve().parents[1] / ".env.feasibility"
    if local.is_file():
        for line in local.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLOUDFLARE_ACCOUNT_ID="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("Set CLOUDFLARE_ACCOUNT_ID or add it to .env.feasibility")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "artifacts/qa/semantic/semantic_responses_v16.jsonl"
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/qa/question-bank-pilot"
ALMUKHTASAR_ALIAS = "qul-tafsir-almukhtasar-ar-v1"
ARM_DEPTH = 20
TOP_K = 3
WINDOW_RADIUS = 2
GENERATION_MAX_TOKENS = 700
WINDOW_PROMPT_CHARACTER_BUDGET = 24_000
SCORERS = ("production_rrf", "tafsir_window", "bank_msa", "bank_colloquial", "bank_all")
GENERATION_FAILURE_RATE_LIMIT = 0.10
GENERATION_FAILURE_MIN_ATTEMPTED_WINDOWS = 20
GENERATION_SUCCEEDED = "generation_succeeded"
GENERATION_FAILED = "generation_failed"


class QuestionPair(BaseModel):
    """One structurally valid bank intent in the two fixed language registers."""

    model_config = ConfigDict(extra="forbid")

    supporting_clause_ids: list[str]
    msa: str
    egyptian_colloquial: str


class GeneratedQuestionPair(BaseModel):
    """Permissive transport schema; structural acceptance happens per pair below."""

    model_config = ConfigDict(extra="forbid")

    supporting_clause_ids: list[str] | None = None
    msa: str | None = None
    egyptian_colloquial: str | None = None


class GeneratedWindowQuestionBank(BaseModel):
    """Raw local response, allowing invalid pairs to be atomically discarded."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GeneratedQuestionPair] = Field(default_factory=list, max_length=2)


class WindowQuestionBank(BaseModel):
    """The structurally accepted question bank cached for one source window."""

    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionPair] = Field(default_factory=list, max_length=2)


@dataclass(frozen=True)
class CandidateRecord:
    reference: str
    fused_score: float
    rank: int


@dataclass(frozen=True)
class PoolRecord:
    case_id: str
    original_question: str
    acceptable_anchors: frozenset[str]
    candidates: tuple[CandidateRecord, ...]
    production_top3: tuple[str, ...]


@dataclass(frozen=True)
class GenerationWindow:
    """A deliberately non-evaluation-shaped input to generation.

    It contains canonical source material only.  There is no case id, user
    question, planner query, target label, score, rank, or anchor-set field.
    """

    window_id: str
    verses: tuple[dict[str, str], ...]
    almukhtasar_passages: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ScoredSelection:
    anchors: tuple[str, ...]
    scores: tuple[float, ...]


class GenerationFailureRateExceeded(RuntimeError):
    """Too many windows failed structured generation for this experiment to be useful."""

    def __init__(self, failed_windows: int, total_windows: int) -> None:
        self.failed_windows = failed_windows
        self.total_windows = total_windows
        super().__init__(
            "generation failed for "
            f"{failed_windows}/{total_windows} windows "
            f"({failed_windows / total_windows:.1%}), above the "
            f"{GENERATION_FAILURE_RATE_LIMIT:.0%} limit"
        )


def _reference(chapter: int, verse: int) -> str:
    return f"{chapter}:{verse}"


def _parse_reference(reference: str) -> tuple[int, int]:
    chapter, verse = reference.split(":", 1)
    return int(chapter), int(verse)


def _window_reaches(anchors: Iterable[str], acceptable: frozenset[str]) -> bool:
    """Measure the production ±2 structural hydration unit, not anchor equality."""

    targets = {_parse_reference(reference) for reference in acceptable}
    for anchor in anchors:
        chapter, verse = _parse_reference(anchor)
        if any(
            target_chapter == chapter and abs(target_verse - verse) <= WINDOW_RADIUS
            for target_chapter, target_verse in targets
        ):
            return True
    return False


def _normalise(vector: Sequence[float]) -> list[float]:
    """L2 normalization used before cosine comparisons of BGE-M3 embeddings."""

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise ValueError("embedding vector has zero magnitude")
    return [value / magnitude for value in vector]


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right, strict=True))


def production_top3(pool: PoolRecord) -> ScoredSelection:
    candidates = sorted(pool.candidates, key=lambda item: (item.rank, item.reference))[:TOP_K]
    return ScoredSelection(
        anchors=tuple(item.reference for item in candidates),
        scores=tuple(item.fused_score for item in candidates),
    )


def select_by_window_similarity(
    pool: PoolRecord,
    similarities: dict[str, float],
) -> ScoredSelection:
    """Fixed max scorer with only a deterministic reference tie break."""

    ranked = sorted(
        pool.candidates,
        key=lambda candidate: (
            -similarities.get(candidate.reference, float("-inf")),
            candidate.reference,
        ),
    )[:TOP_K]
    return ScoredSelection(
        anchors=tuple(candidate.reference for candidate in ranked),
        scores=tuple(similarities.get(candidate.reference, float("-inf")) for candidate in ranked),
    )


def assert_production_top3(pool: PoolRecord) -> None:
    actual = production_top3(pool).anchors
    if actual != pool.production_top3:
        raise AssertionError(
            f"production pool reconstruction mismatch for {pool.case_id}: "
            f"trace={pool.production_top3!r}, rebuilt={actual!r}"
        )


def classify_pools(
    pools: Sequence[PoolRecord],
) -> tuple[list[PoolRecord], list[PoolRecord], list[PoolRecord]]:
    """Return current successes, oracle-recoverable misses, and unreachable pools."""

    successes: list[PoolRecord] = []
    recoverable: list[PoolRecord] = []
    unreachable: list[PoolRecord] = []
    for pool in pools:
        baseline = _window_reaches(pool.production_top3, pool.acceptable_anchors)
        union = _window_reaches(
            (candidate.reference for candidate in pool.candidates), pool.acceptable_anchors
        )
        if baseline:
            successes.append(pool)
        elif union:
            recoverable.append(pool)
        else:
            unreachable.append(pool)
    return successes, recoverable, unreachable


def _trace_top3(record: dict[str, Any]) -> tuple[str, ...] | None:
    for event in record.get("events", []):
        if event.get("summary") != "Candidates dropped before evidence assembly":
            continue
        selected = event.get("detail", {}).get("selected", [])
        references = tuple(item["reference"] for item in selected[:TOP_K] if "reference" in item)
        if len(references) == TOP_K:
            return references
    return None


def _trace_queries(record: dict[str, Any]) -> list[str] | None:
    for event in record.get("events", []):
        if event.get("summary") == "Embedded dense retrieval query":
            queries = event.get("detail", {}).get("queries")
            if isinstance(queries, list) and all(isinstance(query, str) for query in queries):
                return queries
    return None


def load_trace_inputs(
    trace_path: Path, cases_path: Path
) -> list[tuple[dict[str, Any], list[str], tuple[str, ...]]]:
    """Read only the trace queries and top-three anchors needed for reconstruction."""

    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases if case.get("intent") == "answer"}
    result: list[tuple[dict[str, Any], list[str], tuple[str, ...]]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case = by_id.get(record.get("id"))
            queries = _trace_queries(record)
            top3 = _trace_top3(record)
            if case is not None and queries is not None and top3 is not None:
                result.append((case, queries, top3))
    if len(result) != 42:
        raise ValueError(f"expected exactly 42 traced answer cases, found {len(result)}")
    return result


def _pool_path(output: Path) -> Path:
    return output / "pools.json"


def _bank_path(output: Path) -> Path:
    return output / "window_banks.jsonl"


def _report_path(output: Path) -> Path:
    return output / "report.json"


def _atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pool_to_json(pool: PoolRecord) -> dict[str, object]:
    return {
        "case_id": pool.case_id,
        "original_question": pool.original_question,
        "acceptable_anchors": sorted(pool.acceptable_anchors),
        "candidates": [candidate.__dict__ for candidate in pool.candidates],
        "production_top3": list(pool.production_top3),
    }


def _pool_from_json(data: dict[str, Any]) -> PoolRecord:
    return PoolRecord(
        case_id=data["case_id"],
        original_question=data["original_question"],
        acceptable_anchors=frozenset(data["acceptable_anchors"]),
        candidates=tuple(CandidateRecord(**candidate) for candidate in data["candidates"]),
        production_top3=tuple(data["production_top3"]),
    )


async def reconstruct_pools(output: Path, trace_path: Path, cases_path: Path) -> list[PoolRecord]:
    """Rebuild all candidate unions through the production search pieces.

    The direct calls below are intentionally kept visible: repository hybrid search,
    ``_tafsir_first_candidates``, ``_fuse_candidate_arms`` and production late
    binding.  Do not replace any of them with an offline approximation.
    """

    existing = _pool_path(output)
    if existing.exists():
        pools = [_pool_from_json(item) for item in json.loads(existing.read_text(encoding="utf-8"))]
        for pool in pools:
            assert_production_top3(pool)
        return pools

    _configure_runtime_environment()
    from islamic_research_ai.composition import build_container
    from islamic_research_ai.contracts import ToolResult, ToolResultStatus
    from islamic_research_ai.retrieval.relevance import expanded_topic_query
    from islamic_research_ai.tools import PostgresQuranSource, WorkflowExecutor

    inputs = load_trace_inputs(trace_path, cases_path)
    container = await build_container()
    try:
        source = container.service.source
        if not isinstance(source, PostgresQuranSource):
            raise TypeError("pilot requires PostgresQuranSource production retrieval")
        pools: list[PoolRecord] = []
        for position, (case, queries, trace_top3) in enumerate(inputs, start=1):
            source.reset_retrieval_diagnostics()
            results: dict[str, ToolResult[Any]] = {}
            for query_index, query in enumerate(queries, start=1):
                vectors, completion = await container.embeddings.embed([query])
                # The production source validates this exact direct BGE-M3 route.
                from islamic_research_ai.routing import ProviderRoutePolicy

                ProviderRoutePolicy.require_cloudflare_bge_m3_direct_free(completion)
                vector = vectors[0]
                quran = await container.repository.hybrid_search(
                    query=expanded_topic_query(query), query_embedding=vector, limit=ARM_DEPTH
                )
                tafsir = await source._tafsir_first_candidates(query, vector, limit=ARM_DEPTH)
                fused = source._fuse_candidate_arms(quran, tafsir)
                results[f"variant-{query_index}"] = ToolResult(
                    status=ToolResultStatus.SUCCESS if fused else ToolResultStatus.EMPTY,
                    data=fused,
                    evidence_items=[],
                    warnings=[],
                    duration_ms=0,
                    error=None,
                )
            identifiers = list(results)
            all_candidates = WorkflowExecutor._late_bound_candidates(
                results, identifiers[0], identifiers[1:], 10_000
            )
            candidates = tuple(
                CandidateRecord(
                    reference=_reference(
                        candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse
                    ),
                    fused_score=float(candidate.ranking.component_scores["fused_arm_score"]),
                    # Preserve WorkflowExecutor's production late-bound ordering.
                    rank=position,
                )
                for position, candidate in enumerate(all_candidates, start=1)
            )
            pool = PoolRecord(
                case_id=case["id"],
                original_question=case["prompt"],
                acceptable_anchors=frozenset(case["acceptable_anchors"]),
                candidates=candidates,
                production_top3=trace_top3,
            )
            assert_production_top3(pool)
            pools.append(pool)
            print(f"reconstructed {position}/{len(inputs)} {pool.case_id}", flush=True)
        _atomic_json(_pool_path(output), [_pool_to_json(pool) for pool in pools])
        return pools
    finally:
        await container.close()


def _configure_runtime_environment() -> None:
    """Match the existing local collector without importing credentials in tests."""

    account = "islamic-research-ai"
    for variable, service in {
        "GROQ_API_KEY": "groq-api-key",
        "CLOUDFLARE_API_TOKEN": "cloudflare-api-token",
    }.items():
        if not os.environ.get(variable):
            os.environ[variable] = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-a", account, "-s", service, "-w"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    os.environ.update(
        {
            "CLOUDFLARE_ACCOUNT_ID": cloudflare_account_id(),
            "PROJECT_OWNER_RISK_OVERRIDE_ACK": "1",
            "LOCAL_CHAT_FORCE": "1",
        }
    )


def _safe_generation_prompt(window: GenerationWindow) -> str:
    """Serialize source-only data and assert banned evaluation fields cannot enter it."""

    payload = {
        "source_clauses": _source_clauses(window),
    }
    forbidden = {
        "case_id",
        "original_question",
        "evaluation_question",
        "planner_query",
        "acceptable_anchors",
        "target",
        "topic",
        "rank",
        "score",
    }
    if forbidden.intersection(payload):
        raise AssertionError("evaluation metadata attempted to enter a generation prompt")
    source = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(source) > WINDOW_PROMPT_CHARACTER_BUDGET:
        # The source extractor bounds each field before this point.  Silently truncating
        # here would make treatment depend on accidental JSON layout.
        raise ValueError(f"window {window.window_id} exceeds the fixed prompt budget")
    return source


GENERATION_OUTPUT_EXAMPLE_JSON = json.dumps(
    {
        "questions": [
            {
                "supporting_clause_ids": [
                    "verse:2:1:arabic",
                    "verse:2:1:translation",
                ],
                "msa": "ما الذي ينبغي فعله عند توثيق دين إلى أجل مسمى؟",
                "egyptian_colloquial": "نعمل إيه لما نوثق دين لحد وقت محدد؟",
            },
            {
                "supporting_clause_ids": [
                    "verse:2:1:arabic",
                    "verse:2:1:translation",
                ],
                "msa": "ما الشرط المتعلق بموعد سداد الدين المذكور؟",
                "egyptian_colloquial": "إيه الشرط اللي يخص ميعاد سداد الدين المذكور؟",
            }
        ]
    },
    ensure_ascii=False,
    separators=(",", ":"),
)


GENERATION_SYSTEM_PROMPT = f"""You are building a question bank from supplied Quran verses and
Al-Mukhtasar tafsir source clauses. Perform the following steps silently, in exactly this order for
each possible pair. They are internal work, not text to include in the response:

1. Select an explicit relationship, action, condition, contrast, or claim in the source, and
   identify the supporting source clause IDs.
2. Identify what makes that content more specific than its broad topic. The distinguishing detail
   must itself be supported by the source. Do not manufacture circumstances to satisfy this step.
3. Write the question an ordinary person seeking that specific information would ask. For practical
   guidance, a situational question is appropriate; for narrative or doctrinal material, ask about
   the stated event or claim and do not force it into a personal dilemma.
4. Render that same question in Modern Standard Arabic (msa) and Egyptian Arabic
   (egyptian_colloquial), preserving participants, conditions, negation, and the information
   requested.
5. Produce a second pair only if there is a distinct supported intent. Zero or one pair is a
   valid, non-failing outcome.

Hard cap for the entire window: return at most two question pairs total, never one pair per
source clause. Select the strongest distinct supported intents across all supplied clauses before
producing pairs. A single pair may draw on and cite several clauses. Returning zero pairs or one
pair is correct when the window does not support two distinct intents.

Each output pair must cite one or more supporting_clause_ids from the supplied source_clauses.
Do not reproduce Quranic or tafsir quotations.

Prohibitions:
- No topic-only templates such as "What does the Quran say about [topic]?".
- Do not add motives, settings, diagnoses, ages, or circumstances unsupported by the source.
- Do not turn a narrated event into a universal instruction, and do not strengthen the source's
  claim.
- Do not produce a second question that merely paraphrases the first.
- Do not strip useful topic words just to make the question look colloquial.

Required output shape: return exactly one JSON object with exactly one top-level key, `questions`.
The output must begin with `{{` and end with `}}`; do not use a Markdown fence. Do not wrap it in a
schema-name key or any other key. `questions` is an array of zero, one, or two objects. Each array
item is directly the question pair; do not nest it under `question_pair` or any other key. Every
item has exactly these keys: `supporting_clause_ids`, `msa`, and
`egyptian_colloquial`. Do not use `question_msa`, `question_egyptian_colloquial`, or variant
spellings.

Return an object in this exact shape (the clause ID and question text here are only an example):
{GENERATION_OUTPUT_EXAMPLE_JSON}

Output no preamble, commentary, explanation, or text before or after the object, and do not use a
Markdown code fence.
"""


def _source_clauses(window: GenerationWindow) -> list[dict[str, str]]:
    """Mechanically expose every source text unit with a stable, prompt-local clause ID."""

    clauses: list[dict[str, str]] = []
    for verse in window.verses:
        reference = verse["reference"]
        for language in ("arabic", "translation"):
            text = verse.get(language, "")
            if text:
                clauses.append({"id": f"verse:{reference}:{language}", "text": text})
    for passage_index, passage in enumerate(window.almukhtasar_passages, start=1):
        text = passage.get("text", "")
        if text:
            clauses.append(
                {
                    "id": f"tafsir:{passage_index}",
                    "text": text,
                }
            )
    return clauses


def _normalise_question_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def validate_question_bank(
    raw_bank: GeneratedWindowQuestionBank, window: GenerationWindow
) -> WindowQuestionBank:
    """Apply only frozen structural acceptance rules, atomically per language pair."""

    valid_clause_ids = {clause["id"] for clause in _source_clauses(window)}
    accepted: list[QuestionPair] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in raw_bank.questions:
        if (
            pair.msa is None
            or pair.egyptian_colloquial is None
            or pair.supporting_clause_ids is None
            or not pair.supporting_clause_ids
        ):
            continue
        msa = _normalise_question_text(pair.msa)
        colloquial = _normalise_question_text(pair.egyptian_colloquial)
        if (
            not msa
            or not colloquial
            or not set(pair.supporting_clause_ids).issubset(valid_clause_ids)
        ):
            continue
        identity = (msa, colloquial)
        if identity in seen_pairs:
            continue
        seen_pairs.add(identity)
        accepted.append(
            QuestionPair(
                supporting_clause_ids=list(pair.supporting_clause_ids),
                msa=pair.msa,
                egyptian_colloquial=pair.egyptian_colloquial,
            )
        )
    return WindowQuestionBank(questions=accepted)


def _bounded(value: str, budget: int) -> str:
    return value if len(value) <= budget else value[:budget]


async def _window_for_candidate(
    source: Any, tafsir_source: Any, candidate: CandidateRecord
) -> GenerationWindow:
    """Fetch precisely one candidate's production structural window and Al-Mukhtasar."""

    from islamic_research_ai.contracts import VerseKey

    chapter, verse = _parse_reference(candidate.reference)
    # hydrate_references gives the same canonical structural material without needing
    # candidate-index identity; the intended ±2 coordinates are exactly production's.
    keys = [VerseKey(chapter=chapter, verse=value) for value in range(max(1, verse - 2), verse + 3)]
    quran_items = await source.hydrate_references(keys, limit=len(keys))
    actual_keys = []
    verses: dict[str, dict[str, str]] = {}
    for item in quran_items:
        key = item.canonical_pointer.verse_keys[0]
        reference = _reference(key.chapter, key.verse)
        actual_keys.append(key)
        entry = verses.setdefault(reference, {"reference": reference})
        if item.language == "ar":
            entry["arabic"] = _bounded(item.exact_text, 2_000)
        elif item.language == "en":
            entry["translation"] = _bounded(item.exact_text, 2_000)
    # Duplicated keys are harmless to source fetch, but avoid giving a window extra text.
    unique_keys = list({(key.chapter, key.verse): key for key in actual_keys}.values())
    tafsir_items = await tafsir_source.fetch_passages(
        unique_keys, resource_aliases=[ALMUKHTASAR_ALIAS], character_budget=4_000
    )
    tafsir: list[dict[str, str]] = []
    for item in tafsir_items:
        references = ",".join(
            _reference(key.chapter, key.verse) for key in item.canonical_pointer.verse_keys
        )
        tafsir.append({"verses": references, "text": _bounded(item.exact_text, 4_000)})
    identity = ";".join(sorted(verses))
    return GenerationWindow(
        window_id=identity,
        verses=tuple(verses[key] for key in sorted(verses, key=_parse_reference)),
        almukhtasar_passages=tuple(tafsir),
    )


def _load_banks(path: Path) -> dict[str, dict[str, Any]]:
    banks: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return banks
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # a killed process may leave only its final line incomplete
        banks[item["window_id"]] = item
    return banks


def _generation_status(item: dict[str, Any]) -> str:
    """Treat cache records written before outcome statuses as successful banks."""

    return str(item.get("generation_status", GENERATION_SUCCEEDED))


def _is_generation_failed(item: dict[str, Any]) -> bool:
    return _generation_status(item) == GENERATION_FAILED


def _pending_windows(
    windows: dict[str, GenerationWindow], known: dict[str, dict[str, Any]]
) -> list[GenerationWindow]:
    """A recorded failure is complete work for resume purposes, just like a bank."""

    return [window for window_id, window in windows.items() if window_id not in known]


def _assert_generation_failure_rate(
    banks: dict[str, dict[str, Any]], window_ids: Iterable[str]
) -> None:
    """Fail closed once failures dominate a sufficiently large attempted sample."""

    expected_window_ids = list(window_ids)
    attempted_window_ids = [window_id for window_id in expected_window_ids if window_id in banks]
    failures = sum(_is_generation_failed(banks[window_id]) for window_id in attempted_window_ids)
    attempted = len(attempted_window_ids)
    if (
        attempted >= GENERATION_FAILURE_MIN_ATTEMPTED_WINDOWS
        and failures / attempted > GENERATION_FAILURE_RATE_LIMIT
    ):
        raise GenerationFailureRateExceeded(failures, attempted)


async def _complete_question_bank(
    local_chat: Any,
    *,
    prompt: str,
) -> tuple[GeneratedWindowQuestionBank | None, Any | None, str | None, int]:
    """Make the fixed one-or-two-attempt structured-generation request.

    A retry is reserved for a malformed structured completion and deliberately reuses every
    request parameter, including the prompt. Other provider failures are cached as failures
    immediately rather than being misrepresented as empty question banks.
    """

    from islamic_research_ai.ports import StructuredOutputValidationError

    arguments = {
        "response_model": GeneratedWindowQuestionBank,
        "schema_name": "question_bank_window_v1",
        "system_prompt": GENERATION_SYSTEM_PROMPT,
        "user_prompt": prompt,
        "max_tokens": GENERATION_MAX_TOKENS,
    }
    try:
        return (*await local_chat.complete(**arguments), None, 1)
    except StructuredOutputValidationError:
        try:
            return (*await local_chat.complete(**arguments), None, 2)
        except StructuredOutputValidationError as second_error:
            return None, None, type(second_error).__name__, 2
        except Exception as second_error:  # noqa: BLE001 - provider failure is a cacheable outcome
            return None, None, type(second_error).__name__, 2
    except Exception as error:  # noqa: BLE001 - provider failure is a cacheable outcome
        return None, None, type(error).__name__, 1


async def generate_banks(output: Path, pools: Sequence[PoolRecord], *, phase: str) -> None:
    """Sequentially create/cache every distinct candidate window in the supplied pools."""

    _configure_runtime_environment()
    from islamic_research_ai.composition import build_container
    from islamic_research_ai.tools import PostgresQulTafsirSource

    path = _bank_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = _load_banks(path)
    container = await build_container()
    try:
        source = container.service.source
        tafsir_source = PostgresQulTafsirSource(container.repository)
        candidates = [candidate for pool in pools for candidate in pool.candidates]
        # Candidate records are not passed through to the model. The candidate only selects
        # source material; the resulting identity is what is deduplicated and cached.
        windows: dict[str, GenerationWindow] = {}
        for candidate in candidates:
            window = await _window_for_candidate(source, tafsir_source, candidate)
            windows.setdefault(window.window_id, window)
        pending = _pending_windows(windows, known)
        print(f"{phase}: {len(known)} cached, {len(pending)} new windows", flush=True)
        _assert_generation_failure_rate(known, windows)
        with path.open("a", encoding="utf-8") as handle:
            for index, window in enumerate(pending, start=1):
                prompt = _safe_generation_prompt(window)
                raw_bank, completion, error_type, attempts = await _complete_question_bank(
                    container.local_chat, prompt=prompt
                )
                bank = (
                    validate_question_bank(raw_bank, window) if raw_bank is not None else None
                )
                texts = [
                    question
                    for pair in (bank.questions if bank is not None else [])
                    for question in (pair.msa, pair.egyptian_colloquial)
                ]
                vectors: list[list[float]] = []
                if texts:
                    embedded, _ = await container.embeddings.embed(texts)
                    vectors = [_normalise(vector) for vector in embedded]
                tafsir_texts = [passage["text"] for passage in window.almukhtasar_passages]
                tafsir_vectors: list[list[float]] = []
                if tafsir_texts:
                    embedded_tafsir, _ = await container.embeddings.embed(tafsir_texts)
                    tafsir_vectors = [_normalise(vector) for vector in embedded_tafsir]
                clause_texts = [clause["text"] for clause in _source_clauses(window)]
                clause_vectors: list[list[float]] = []
                if clause_texts:
                    embedded_clauses, _ = await container.embeddings.embed(clause_texts)
                    clause_vectors = [_normalise(vector) for vector in embedded_clauses]
                item = {
                    "window_id": window.window_id,
                    "bank_embeddings": vectors,
                    "tafsir_embeddings": tafsir_vectors,
                    "source_clause_embeddings": clause_vectors,
                    "phase": phase,
                    "generation_status": (
                        GENERATION_SUCCEEDED if bank is not None else GENERATION_FAILED
                    ),
                    "generation_attempts": attempts,
                }
                if bank is not None:
                    item["bank"] = bank.model_dump(mode="json")
                    item["completion"] = completion.model_dump(mode="json")
                    item["accepted_pair_count"] = len(bank.questions)
                else:
                    item["generation_error_type"] = error_type
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                known[window.window_id] = item
                outcome = "failed" if bank is None else "generated"
                print(f"{phase}: {outcome} {index}/{len(pending)} {window.window_id}", flush=True)
                _assert_generation_failure_rate(known, windows)
    finally:
        await container.close()


def _current_citation_successes(trace_path: Path, pools: Sequence[PoolRecord]) -> set[str]:
    """Read the frozen run's citations only for the required regression report."""

    from qa.semantic_quality_runner import _citation_anchor, _response_citations

    by_id = {pool.case_id: pool for pool in pools}
    successes: set[str] = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        pool = by_id.get(record.get("id"))
        if pool is None:
            continue
        cited = {
            anchor
            for citation in _response_citations(record)
            if (anchor := _citation_anchor(citation)) is not None
        }
        if cited.intersection(pool.acceptable_anchors):
            successes.add(pool.case_id)
    return successes


async def _all_windows(
    source: Any, tafsir_source: Any, pools: Sequence[PoolRecord]
) -> dict[str, GenerationWindow]:
    """Materialize source-only windows once per anchor reference for a scoring run."""

    windows: dict[str, GenerationWindow] = {}
    for candidate in (candidate for pool in pools for candidate in pool.candidates):
        windows.setdefault(
            candidate.reference,
            await _window_for_candidate(source, tafsir_source, candidate),
        )
    return windows


async def _acceptable_evidence_after_filter(
    source: Any,
    tafsir_source: Any,
    pool: PoolRecord,
    anchors: Sequence[str],
) -> bool:
    """Use the production deterministic filter over the unchanged selected windows."""

    from islamic_research_ai.contracts import EvidenceRequirementsV1, EvidenceSet, VerseKey
    from islamic_research_ai.retrieval.relevance import assess_topic_evidence, semantic_subtargets

    keys = [
        VerseKey(chapter=chapter, verse=window_verse)
        for reference in anchors
        for chapter, verse in [_parse_reference(reference)]
        for window_verse in range(max(1, verse - 2), verse + 3)
    ]
    unique = list({(key.chapter, key.verse): key for key in keys}.values())
    quran = await source.hydrate_references(unique, limit=len(unique))
    anchor_keys = [
        VerseKey(chapter=chapter, verse=verse) for chapter, verse in map(_parse_reference, anchors)
    ]
    tafsir = await tafsir_source.fetch_passages(
        anchor_keys,
        resource_aliases=[ALMUKHTASAR_ALIAS],
        character_budget=40_000,
    )
    all_items = [*quran, *tafsir]
    items_by_id = {
        f"P{index}": item.model_copy(update={"evidence_id": f"P{index}"})
        for index, item in enumerate(all_items, start=1)
    }
    evidence = EvidenceSet(
        schema_version="1",
        turn_id=uuid4(),
        items_by_id=items_by_id,
        requirements=EvidenceRequirementsV1(
            required_source_kinds=set(),
            minimum_items_by_kind={},
            require_exact_quran=False,
            require_exact_hadith=False,
            required_tafsir_aliases=[],
            require_two_distinct_tafsir_resources=False,
            allow_partial=True,
        ),
        requirements_satisfied=True,
        missing_requirements=[],
        permitted_synthesis_deployments=set(),
    )
    # This follows service.py's target construction, with the experiment's original
    # user question as its fixed fallback rather than exposing planner labels to Gemma.
    targets = semantic_subtargets([], fallback_query=pool.original_question)
    assessment = assess_topic_evidence(pool.original_question, evidence, targets=targets)
    retained = {
        _reference(key.chapter, key.verse)
        for item in assessment.evidence.items_by_id.values()
        for key in item.canonical_pointer.verse_keys
    }
    return bool(retained.intersection(pool.acceptable_anchors))


async def score_and_filter(
    output: Path,
    pools: Sequence[PoolRecord],
    trace_path: Path,
) -> dict[str, Any]:
    """Run all five frozen scorers and the unchanged deterministic filter."""

    _configure_runtime_environment()
    from islamic_research_ai.composition import build_container
    from islamic_research_ai.tools import PostgresQulTafsirSource

    banks = _load_banks(_bank_path(output))
    container = await build_container()
    try:
        tafsir_source = PostgresQulTafsirSource(container.repository)
        current, recoverable, unreachable = classify_pools(pools)
        solvable = [*recoverable, *current]
        windows = await _all_windows(container.service.source, tafsir_source, solvable)
        maps = {
            pool.case_id: await _similarities_for_pool(container, pool, banks, windows)
            for pool in solvable
        }
        for pool in unreachable:
            # The fixed protocol still embeds each original question once, even where
            # the predeclared union ceiling makes every reranking outcome a failure.
            query, _ = await container.embeddings.embed([pool.original_question])
            _normalise(query[0])
            maps[pool.case_id] = {
                scorer: {candidate.reference: float("-inf") for candidate in pool.candidates}
                for scorer in ("tafsir_window", "bank_msa", "bank_colloquial", "bank_all")
            }
        report = score_pools(pools, maps)
        for pool in pools:
            report["cases"][pool.case_id]["question_diagnostics"] = (
                _question_diagnostics(pool, banks, windows) if pool in solvable else []
            )
            for scorer in SCORERS:
                anchors = report["cases"][pool.case_id]["scorers"][scorer]["anchors"]
                if pool in unreachable:
                    retained = False
                else:
                    retained = await _acceptable_evidence_after_filter(
                        container.service.source, tafsir_source, pool, anchors
                    )
                report["cases"][pool.case_id]["scorers"][scorer]["acceptable_after_filter"] = (
                    retained
                )
        citation_successes = _current_citation_successes(trace_path, current)
        summary: dict[str, Any] = {}
        for scorer in SCORERS:
            recovered = sum(
                report["cases"][pool.case_id]["scorers"][scorer]["hydration_success"]
                for pool in recoverable
            )
            lost = sum(
                not report["cases"][pool.case_id]["scorers"][scorer]["hydration_success"]
                for pool in current
            )
            post_filter = sum(
                report["cases"][pool.case_id]["scorers"][scorer]["acceptable_after_filter"]
                for pool in pools
            )
            lost_citation_evidence = sorted(
                case_id
                for case_id in citation_successes
                if not report["cases"][case_id]["scorers"][scorer]["acceptable_after_filter"]
            )
            summary[scorer] = {
                "recovered_of_oracle_recoverable_misses": recovered,
                "oracle_recoverable_miss_count": len(recoverable),
                "lost_of_19_current_successes": lost,
                "total_hydration_successes_of_42": report["hydration_successes"][scorer],
                "acceptable_evidence_after_deterministic_filter_of_42": post_filter,
                "current_citation_successes_losing_acceptable_evidence": lost_citation_evidence,
            }
        report["summary"] = summary
        report["generation_outcomes"] = _bank_outcome_counts(banks)
        report["current_citation_success_count"] = len(citation_successes)
        if len(citation_successes) != 13:
            raise AssertionError(
                "expected 13 current citation successes in frozen trace, found "
                f"{len(citation_successes)}"
            )
        return report
    finally:
        await container.close()


async def _early_bank_score(output: Path, pools: Sequence[PoolRecord]) -> dict[str, Any]:
    """Score the eight first-pass pools before their expensive remaining generation."""

    _configure_runtime_environment()
    from islamic_research_ai.composition import build_container
    from islamic_research_ai.tools import PostgresQulTafsirSource

    container = await build_container()
    try:
        tafsir_source = PostgresQulTafsirSource(container.repository)
        windows = await _all_windows(container.service.source, tafsir_source, pools)
        banks = _load_banks(_bank_path(output))
        maps = {
            pool.case_id: await _similarities_for_pool(container, pool, banks, windows)
            for pool in pools
        }
        report = score_pools(pools, maps)
        for pool in pools:
            report["cases"][pool.case_id]["question_diagnostics"] = _question_diagnostics(
                pool, banks, windows
            )
        report["generation_outcomes"] = _bank_outcome_counts(banks)
        return report
    finally:
        await container.close()


async def _similarities_for_pool(
    container: Any,
    pool: PoolRecord,
    banks: dict[str, dict[str, Any]],
    windows_by_reference: dict[str, GenerationWindow],
) -> dict[str, dict[str, float]]:
    """Embed the original question once and apply only fixed max comparisons."""

    query, _ = await container.embeddings.embed([pool.original_question])
    query_vector = _normalise(query[0])
    result = {"tafsir_window": {}, "bank_msa": {}, "bank_colloquial": {}, "bank_all": {}}
    for candidate in pool.candidates:
        window = windows_by_reference[candidate.reference]
        item = banks.get(window.window_id)
        if item is None:
            raise ValueError(f"missing cached bank for window {window.window_id}")
        pairs = [] if _is_generation_failed(item) else item["bank"]["questions"]
        vectors = item.get("bank_embeddings", [])
        if len(vectors) != len(pairs) * 2:
            raise ValueError(f"invalid cached question embeddings for {item['window_id']}")
        msa = [_similarity(query_vector, vector) for vector in vectors[::2]]
        colloquial = [_similarity(query_vector, vector) for vector in vectors[1::2]]
        # The Tafsir control is cached alongside the source-only window material and is
        # intentionally never combined with a generated-question score.
        result["bank_msa"][candidate.reference] = max(msa, default=float("-inf"))
        result["bank_colloquial"][candidate.reference] = max(colloquial, default=float("-inf"))
        result["bank_all"][candidate.reference] = max([*msa, *colloquial], default=float("-inf"))
        tafsir_vectors = item.get("tafsir_embeddings", [])
        result["tafsir_window"][candidate.reference] = max(
            (_similarity(query_vector, vector) for vector in tafsir_vectors),
            default=float("-inf"),
        )
    return result


def _question_diagnostics(
    pool: PoolRecord,
    banks: dict[str, dict[str, Any]],
    windows_by_reference: dict[str, GenerationWindow],
) -> list[dict[str, Any]]:
    """Compute diagnostic-only question metrics without changing acceptance or ranking."""

    all_other_questions: dict[str, list[list[float]]] = {}
    for window_id, item in banks.items():
        all_other_questions[window_id] = list(item.get("bank_embeddings", []))
    diagnostics: list[dict[str, Any]] = []
    emitted_windows: set[str] = set()
    for candidate in pool.candidates:
        window = windows_by_reference[candidate.reference]
        if window.window_id in emitted_windows:
            continue
        emitted_windows.add(window.window_id)
        item = banks.get(window.window_id)
        if item is None:
            raise ValueError(f"missing cached bank for window {window.window_id}")
        pairs = [] if _is_generation_failed(item) else item["bank"]["questions"]
        vectors = item.get("bank_embeddings", [])
        clauses = item.get("source_clause_embeddings", [])
        if len(vectors) != len(pairs) * 2:
            raise ValueError(f"invalid cached question embeddings for {window.window_id}")
        others = [
            vector
            for other_window_id, values in all_other_questions.items()
            if other_window_id != window.window_id
            for vector in values
        ]
        questions: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(pairs):
            for register, text, vector in (
                ("msa", pair["msa"], vectors[pair_index * 2]),
                ("egyptian_colloquial", pair["egyptian_colloquial"], vectors[pair_index * 2 + 1]),
            ):
                questions.append(
                    {
                        "register": register,
                        "question": text,
                        "own_window_similarity": max(
                            (_similarity(vector, clause) for clause in clauses),
                            default=float("-inf"),
                        ),
                        "max_other_window_question_similarity": max(
                            (_similarity(vector, other) for other in others),
                            default=float("-inf"),
                        ),
                        "length": len(text),
                    }
                )
        diagnostics.append(
            {
                "candidate_reference": candidate.reference,
                "window_id": window.window_id,
                "questions": questions,
            }
        )
    return diagnostics


def _bank_outcome_counts(banks: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count valid empty banks separately from cacheable generation failures."""

    counts = {
        "two_pairs": 0,
        "one_pair": 0,
        "legitimately_empty_bank": 0,
        "generation_failed": 0,
    }
    for item in banks.values():
        if _is_generation_failed(item):
            counts["generation_failed"] += 1
            continue
        count = int(item.get("accepted_pair_count", len(item.get("bank", {}).get("questions", []))))
        if count == 0:
            counts["legitimately_empty_bank"] += 1
        elif count == 1:
            counts["one_pair"] += 1
        elif count == 2:
            counts["two_pairs"] += 1
        else:
            raise ValueError(f"window {item['window_id']} has an invalid accepted-pair count")
    return counts


def score_pools(
    pools: Sequence[PoolRecord], score_maps: dict[str, dict[str, dict[str, float]]]
) -> dict[str, Any]:
    """Pure scoring core used both by the runnable harness and synthetic tests."""

    details: dict[str, Any] = {}
    totals = {name: 0 for name in SCORERS}
    for pool in pools:
        selections: dict[str, ScoredSelection] = {"production_rrf": production_top3(pool)}
        for scorer in ("tafsir_window", "bank_msa", "bank_colloquial", "bank_all"):
            selections[scorer] = select_by_window_similarity(pool, score_maps[pool.case_id][scorer])
        row = {"production_top3": list(pool.production_top3), "scorers": {}}
        for scorer, selection in selections.items():
            hydration = _window_reaches(selection.anchors, pool.acceptable_anchors)
            totals[scorer] += int(hydration)
            row["scorers"][scorer] = {
                "anchors": list(selection.anchors),
                "scores": list(selection.scores),
                "hydration_success": hydration,
            }
        details[pool.case_id] = row
    return {"hydration_successes": totals, "cases": details}


def early_rejection(recoverable_pools: Sequence[PoolRecord], partial: dict[str, Any]) -> bool:
    recovered = sum(
        bool(partial["cases"][pool.case_id]["scorers"]["bank_all"]["hydration_success"])
        for pool in recoverable_pools
    )
    return recovered <= 2


async def run(output: Path, trace_path: Path, cases_path: Path) -> int:
    pools = await reconstruct_pools(output, trace_path, cases_path)
    current, recoverable, unreachable = classify_pools(pools)
    # Measured by --reconstruct-only over semantic_responses_v18.jsonl with all 42
    # top-3 assertions passing.
    if (len(current), len(recoverable), len(unreachable)) != (18, 9, 15):
        raise AssertionError(
            "frozen pool partition mismatch: expected 18 current, 9 recoverable, 15 unreachable; "
            f"got {len(current)}, {len(recoverable)}, {len(unreachable)}"
        )
    # Fixed predeclared ordering: only recoverable failures are generated before the stop gate.
    try:
        await generate_banks(output, recoverable, phase="recoverable_failures_first")
    except GenerationFailureRateExceeded as error:
        _atomic_json(
            _report_path(output),
            {
                "status": "GENERATION_FAILURE_RATE_EXCEEDED",
                "reason": str(error),
                "generation_outcomes": _bank_outcome_counts(_load_banks(_bank_path(output))),
            },
        )
        print(f"generation stopped: {error}", flush=True)
        return 1
    partial = await _early_bank_score(output, recoverable)
    if early_rejection(recoverable, partial):
        _atomic_json(
            _report_path(output),
            {
                "status": "EARLY_REJECTED",
                "reason": "bank_all recovered 0-2 of the oracle-recoverable misses",
                "oracle_recoverable_miss_count": len(recoverable),
                "recoverable_failure_first_pass": partial,
            },
        )
        print("early rejection: bank_all recovered 0-2; remaining pools not generated")
        return 0
    try:
        await generate_banks(output, current, phase="current_successes_after_early_gate")
    except GenerationFailureRateExceeded as error:
        _atomic_json(
            _report_path(output),
            {
                "status": "GENERATION_FAILURE_RATE_EXCEEDED",
                "reason": str(error),
                "generation_outcomes": _bank_outcome_counts(_load_banks(_bank_path(output))),
            },
        )
        print(f"generation stopped: {error}", flush=True)
        return 1
    report = await score_and_filter(output, pools, trace_path)
    report["status"] = "COMPLETE"
    report["stopping_rules"] = {
        "generation_not_worth_cost": abs(
            report["summary"]["bank_all"]["total_hydration_successes_of_42"]
            - report["summary"]["tafsir_window"]["total_hydration_successes_of_42"]
        )
        <= 1,
        "threshold_to_reconsider_36_percent": (
            report["summary"]["bank_all"]["total_hydration_successes_of_42"] >= 25
            and report["summary"]["bank_all"][
                "acceptable_evidence_after_deterministic_filter_of_42"
            ]
            >= 25
            and not report["summary"]["bank_all"][
                "current_citation_successes_losing_acceptable_evidence"
            ]
        ),
    }
    _atomic_json(_report_path(output), report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reconstruct-only", action="store_true")
    arguments = parser.parse_args()
    lock_path = arguments.output.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"question-bank pilot already running for {arguments.output}", file=sys.stderr)
            return 1
        if arguments.reconstruct_only:
            asyncio.run(reconstruct_pools(arguments.output, arguments.trace, arguments.cases))
            return 0
        return asyncio.run(run(arguments.output, arguments.trace, arguments.cases))


if __name__ == "__main__":
    raise SystemExit(main())
