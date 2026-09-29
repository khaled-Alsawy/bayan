"""Offline semantic candidate-pool judgement experiment using isolated Codex CLI calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shlex
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from semantic_funnel import (
        _citation_references,
        _in_structural_window,
        _selection_references,
        compute_funnel,
        load_cases,
        read_records,
    )
except ModuleNotFoundError:
    from qa.semantic_funnel import (
        _citation_references,
        _in_structural_window,
        _selection_references,
        compute_funnel,
        load_cases,
        read_records,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
DEFAULT_RESPONSES = ROOT / "artifacts/qa/semantic/semantic_responses_v27.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/qa/semantic/luna_judge_v27.jsonl"
DEFAULT_SUMMARY = ROOT / "artifacts/qa/semantic/LUNA_JUDGE_SUMMARY.md"
SCHEMA = ROOT / "qa/strong_judge_schema.json"
CODEX = Path.home() / ".npm-global/bin/codex"
ALIAS = "qul-tafsir-almukhtasar-ar-v1"
INFRA_IDS = {
    "sem_ar_no_way_back",
    "sem_ar_prayer_keyword",
    "sem_ar_public_hypocrisy",
    "sem_en_justice_keyword",
}
_REF = re.compile(r"^(\d+):(\d+)$")
_JUDGE_INSTRUCTION = (
    "اقرأ السؤال والنص القرآني وتفسير كل مرشح. قرر لكل مرشح هل تتناول الآية مباشرة ما يسأل عنه "
    "السؤال، لا لمجرد اشتراكها معه في كلمة. أعد حكمًا لكل ملصق كما هو، ولا تضف أي نص خارج JSON. "
    "يجب أن يتضمن كل حكم: label، relevant (true أو false)، score عدد صحيح من 0 إلى 3، وreason "
    "جملة عربية قصيرة واحدة."
)


def _load_lines(path: Path) -> list[dict[str, Any]]:
    return read_records(path)


def _ranked_pool(record: dict[str, Any]) -> list[str] | None:
    """Reuse the funnel parser; retain the original trace rank for opaque label order."""
    parsed = _selection_references(record)
    if parsed is None:
        return None
    events = record.get("events")
    rows: list[tuple[int, str]] = []
    for event in events if isinstance(events, list) else []:
        if (
            not isinstance(event, dict)
            or event.get("summary") != "Candidates dropped before evidence assembly"
        ):
            continue
        detail = event.get("detail")
        if not isinstance(detail, dict):
            continue
        for field in ("selected", "dropped"):
            candidates = detail.get(field)
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict) or not isinstance(
                    candidate.get("reference"), str
                ):
                    continue
                ref = candidate["reference"]
                if not _REF.fullmatch(ref):
                    continue
                rank = candidate.get("rank")
                rows.append((rank if isinstance(rank, int) else 1_000_000, ref))
    ordered = []
    for _, ref in sorted(rows, key=lambda item: (item[0], item[1])):
        if ref not in ordered:
            ordered.append(ref)
    selected, dropped = parsed
    if set(ordered) != selected | dropped:
        raise ValueError(f"trace pool parser mismatch for {record.get('id')}")
    return ordered


async def _read_verse_text(references: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Read active canonical Uthmani and Al-Mukhtasar text using the repository pool."""
    from islamic_research_ai.repositories.postgres import PostgresRepository
    from islamic_research_ai.settings import load_settings

    parsed = [_REF.fullmatch(reference) for reference in references]
    if any(match is None for match in parsed):
        raise ValueError("invalid verse reference in reconstructed pool")
    keys = {(int(match.group(1)), int(match.group(2))) for match in parsed if match}
    chapters = [chapter for chapter, _ in sorted(keys)]
    verses = [verse for _, verse in sorted(keys)]
    repository = PostgresRepository(load_settings().database_url)
    await repository.open()
    try:
        async with repository.pool.connection() as connection:
            canonical_cursor = await connection.execute(
                """SELECT text.chapter, text.verse, text.canonical_text
                   FROM quran_canonical_text AS text
                   JOIN source_revisions AS revision USING (revision_id)
                   WHERE revision.active
                     AND text.resource_alias = 'tanzil-uthmani-v1.1'
                     AND (text.chapter, text.verse) IN
                         (SELECT * FROM unnest(%s::smallint[], %s::smallint[]))""",
                (chapters, verses),
            )
            canonical_rows = await canonical_cursor.fetchall()
            tafsir_cursor = await connection.execute(
                """SELECT requested.chapter, requested.verse AS verse_start, passage.passage_text
                   FROM unnest(%s::smallint[], %s::smallint[]) AS requested(chapter, verse)
                   JOIN tafsir_passages AS passage
                     ON passage.chapter = requested.chapter
                    AND requested.verse BETWEEN passage.verse_start AND passage.verse_end
                   JOIN source_revisions AS revision USING (revision_id)
                   WHERE revision.active
                     AND passage.resource_alias = %s
                   ORDER BY requested.chapter, requested.verse,
                            (passage.verse_end - passage.verse_start), passage.verse_start""",
                (chapters, verses, ALIAS),
            )
            tafsir_rows = await tafsir_cursor.fetchall()
    finally:
        await repository.close()
    canonical = {
        f"{row['chapter']}:{row['verse']}": row["canonical_text"] for row in canonical_rows
    }
    tafsir = {}
    for row in tafsir_rows:
        tafsir.setdefault(f"{row['chapter']}:{row['verse_start']}", row["passage_text"])
    missing_text = sorted(references - canonical.keys())
    missing_tafsir = sorted(references - tafsir.keys())
    if missing_text or missing_tafsir:
        raise ValueError(f"missing text: {missing_text[:8]}; missing tafsir: {missing_tafsir[:8]}")
    return canonical, tafsir


def _prompt(question: str, candidates: list[dict[str, str]]) -> str:
    pool = "\n\n".join(
        f"{item['label']}\nالنص: {item['verse_text']}\nالتفسير: {item['tafsir_text']}"
        for item in candidates
    )
    return f"{_JUDGE_INSTRUCTION}\n\nالسؤال: {question}\n\nالمرشحون:\n{pool}\n"


def _decode_judge(stdout: str) -> tuple[dict[str, Any] | None, bool, str | None, dict[str, Any]]:
    """Return structured final message, contamination marker, error, and usage data."""
    events: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        event_type = str(event.get("type", "")).casefold()
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type", "")).casefold()
        if any(
            term in item_type
            for term in ("command_execution", "file_read", "file_search", "shell", "mcp_tool")
        ):
            return None, True, "judge used a tool", usage
        if any(
            term in event_type
            for term in ("command_execution", "file_read", "file_search", "shell", "mcp_tool")
        ):
            return None, True, "judge used a tool", usage
        if "usage" in event_type or "token" in event_type:
            usage[event_type] = event.get("usage", event.get("data", event))
        if event_type == "turn.completed":
            usage["turn.completed"] = event.get("usage", event.get("token_usage"))

    messages = []
    for event in events:
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if item.get("type") in {"agent_message", "assistant_message"}:
            content = item.get("text") or item.get("content")
            if isinstance(content, str):
                messages.append(content)
        if event.get("type") in {"agent_message", "assistant_message"}:
            content = event.get("text")
            if isinstance(content, str):
                messages.append(content)
    if not messages:
        return None, False, "no final assistant message in Codex JSONL", usage
    try:
        result = json.loads(messages[-1])
    except json.JSONDecodeError as exc:
        return None, False, f"malformed JSON: {exc}", usage
    return result if isinstance(result, dict) else None, False, None, usage


def _validate_result(result: dict[str, Any] | None, labels: set[str]) -> tuple[bool, str | None]:
    if not isinstance(result, dict) or set(result) != {"judgements"}:
        return False, "wrong top-level JSON shape"
    judgements = result["judgements"]
    if not isinstance(judgements, list):
        return False, "judgements is not an array"
    received = []
    for row in judgements:
        if not isinstance(row, dict) or set(row) != {"label", "relevant", "score", "reason"}:
            return False, "wrong judgement object shape"
        if (
            not isinstance(row["label"], str)
            or not isinstance(row["relevant"], bool)
            or not isinstance(row["score"], int)
            or isinstance(row["score"], bool)
            or not 0 <= row["score"] <= 3
            or not isinstance(row["reason"], str)
        ):
            return False, "wrong judgement field type or score range"
        received.append(row["label"])
    if len(received) != len(set(received)) or set(received) != labels:
        return False, "labels are missing, duplicated, or unexpected"
    return True, None


async def _judge_case(
    case: dict[str, Any], candidate_text: list[dict[str, str]], retries: int = 1
) -> dict[str, Any]:
    prompt = _prompt(case["prompt"], candidate_text)
    label_set = {candidate["label"] for candidate in candidate_text}
    command = [
        str(CODEX),
        "exec",
        "-m",
        "gpt-6-luna",
        "-c",
        'model_reasoning_effort="high"',
        "-s",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--output-schema",
        str(SCHEMA),
        "--json",
    ]
    attempts = 0
    total_usage: list[dict[str, Any]] = []
    started = time.monotonic()
    while attempts <= retries:
        attempts += 1
        empty_dir = Path(tempfile.mkdtemp(prefix="luna-judge-")).resolve()
        actual_command = [*command, "-C", str(empty_dir), "-"]
        try:
            process = await asyncio.create_subprocess_exec(
                *actual_command,
                cwd=empty_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate(prompt.encode("utf-8"))
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            result, contaminated, error, usage = _decode_judge(stdout)
            if usage:
                total_usage.append(usage)
            if contaminated:
                return {
                    "status": "CONTAMINATED",
                    "attempts": attempts,
                    "error": error,
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "usage": total_usage,
                    "command": actual_command,
                }
            valid, validation_error = _validate_result(result, label_set)
            if process.returncode == 0 and valid:
                return {
                    "status": "ok",
                    "attempts": attempts,
                    "judgements": result["judgements"],
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "usage": total_usage,
                    "command": actual_command,
                    "cli_jsonl": stdout,
                }
            error = error or validation_error or f"Codex exited {process.returncode}"
            if (
                "usage limit" in (stderr + stdout).casefold()
                or "rate limit" in (stderr + stdout).casefold()
            ):
                return {
                    "status": "usage_limit",
                    "attempts": attempts,
                    "error": error,
                    "stderr": stderr[-1000:],
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "usage": total_usage,
                    "command": actual_command,
                    "cli_jsonl": stdout,
                }
            if attempts > retries:
                return {
                    "status": "failed",
                    "attempts": attempts,
                    "error": error,
                    "stderr": stderr[-1000:],
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "usage": total_usage,
                    "command": actual_command,
                    "cli_jsonl": stdout,
                }
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)
    raise AssertionError("unreachable")


def _hit(references: set[str], anchors: set[str]) -> bool:
    return any(
        anchor in references or _in_structural_window(anchor, references) for anchor in anchors
    )


def _exact_anchor_hit(references: set[str], anchors: set[str]) -> bool:
    return bool(references & anchors)


def _summary(
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    skipped: list[dict[str, str]],
    baseline_pool_validation: dict[str, Any],
    run_info: dict[str, Any],
) -> str:
    valid_cases = [
        case
        for case in cases
        if case["id"] in results and results[case["id"]].get("status") == "ok"
    ]
    arms: dict[str, dict[str, list[bool | int]]] = {
        arm: {band: [] for band in ("overall", "none", "low", "high")}
        for arm in ("baseline", "pool oracle", "judge top-3", "judge any-relevant")
    }
    selected_counts: dict[str, dict[str, list[int]]] = {
        arm: {band: [] for band in ("overall", "none", "low", "high")} for arm in arms
    }
    judge_distractor: dict[str, list[bool]] = {"top-3": [], "any-relevant": []}
    relevant_empty = 0
    discrepancies: list[dict[str, Any]] = []
    raw_judge_rows = []
    for case in valid_cases:
        cid = case["id"]
        record = records[cid]
        pool = record["pool_references"]
        anchors = set(case.get("acceptable_anchors", []))
        unacceptable = set(case.get("unacceptable_anchors", []))
        baseline = set(record["baseline_references"])
        judgements = results[cid]["judgements"]
        refs_by_label = record["label_references"]
        relevant = [row for row in judgements if row["relevant"]]
        if not relevant:
            relevant_empty += 1
        top = sorted(relevant, key=lambda row: (-row["score"], int(row["label"][1:])))[:3]
        top_refs = {refs_by_label[row["label"]] for row in top}
        any_refs = {refs_by_label[row["label"]] for row in relevant}
        selections = {
            "baseline": baseline,
            "pool oracle": set(pool),
            "judge top-3": top_refs,
            "judge any-relevant": any_refs,
        }
        for arm, selected in selections.items():
            hit = _hit(selected, anchors)
            band = case.get("indirection")
            arms[arm]["overall"].append(hit)
            selected_counts[arm]["overall"].append(len(selected))
            if band in {"none", "low", "high"}:
                arms[arm][band].append(hit)
                selected_counts[arm][band].append(len(selected))
            if arm.startswith("judge"):
                arm_name = "top-3" if arm == "judge top-3" else "any-relevant"
                judge_distractor[arm_name].append(_hit(selected, unacceptable))
        baseline_hit = _hit(baseline, anchors)
        judge_hit = _hit(top_refs, anchors)
        if baseline_hit != judge_hit:
            discrepancies.append(
                {
                    "case": case,
                    "baseline": sorted(baseline),
                    "judge_refs": sorted(top_refs),
                    "judge": top,
                    "baseline_hit": baseline_hit,
                    "judge_hit": judge_hit,
                }
            )
        raw_judge_rows.append((cid, judgements))

    def metrics(values: list[bool | int]) -> tuple[int, float, float]:
        return (
            sum(bool(value) for value in values),
            (sum(bool(value) for value in values) / len(values) if values else 0.0),
            (sum(int(value) for value in values) / len(values) if values else 0.0),
        )

    lines = [
        "# Luna judge experiment (v27)",
        "",
        "| Arm | Cases | Anchor hit | Mean selected verses |",
        "|---|---:|---:|---:|",
    ]
    for arm, bands in arms.items():
        hits, rate, mean = metrics(bands["overall"])
        selections = selected_counts[arm]["overall"]
        mean_selected = sum(selections) / len(selections) if selections else 0.0
        lines.append(
            f"| {arm} | {len(bands['overall'])} | "
            f"{hits}/{len(bands['overall'])} ({rate:.1%}) | {mean_selected:.2f} |"
        )
    lines += [
        "",
        "| Arm | none | low | high | Unacceptable-anchor rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        name = arm.replace("judge ", "")
        cells = []
        for band in ("none", "low", "high"):
            hits, rate, _ = metrics(arms[arm][band])
            cells.append(f"{hits}/{len(arms[arm][band])} ({rate:.1%})")
        if arm.startswith("judge"):
            distractors = judge_distractor[name]
            d_rate = (
                f"{sum(distractors)}/{len(distractors)} ({sum(distractors) / len(distractors):.1%})"
                if distractors
                else "n/a"
            )
        else:
            d_rate = "—"
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {d_rate} |")
    lines += [
        "",
        f"Judge marked nothing relevant: {relevant_empty}/{len(valid_cases)} valid cases.",
        f"Valid judged cases: {len(valid_cases)}; failed/contaminated results are excluded.",
        (
            f"Recorded calls: {len(results)}; failed: "
            f"{sum(row.get('status') == 'failed' for row in results.values())}; "
            f"contaminated: "
            f"{sum(row.get('status') == 'CONTAMINATED' for row in results.values())}; "
            f"retried: {sum(row.get('attempts', 1) > 1 for row in results.values())}."
        ),
        "",
        "## Pool validation",
        "",
        (
            f"Reconstructed late-binding pool hits: "
            f"{baseline_pool_validation['reconstructed_hits']}/"
            f"{baseline_pool_validation['case_count']} "
            f"({baseline_pool_validation['reconstructed_rate']:.1%})."
        ),
        (
            f"`semantic_funnel.py` v27 late-binding pool hits: "
            f"{baseline_pool_validation['funnel_hits']}/"
            f"{baseline_pool_validation['funnel_cases']} "
            f"({baseline_pool_validation['funnel_rate']:.1%})."
        ),
        f"Excluded v27 infra IDs: {', '.join(sorted(INFRA_IDS))}.",
        "Cases with no recoverable pool: "
        + (", ".join(item["id"] for item in skipped) if skipped else "none")
        + ".",
        "",
        "## sem_ar_envy_keyword",
        "",
    ]
    envy = next((case for case in valid_cases if case["id"] == "sem_ar_envy_keyword"), None)
    if envy:
        rec = records[envy["id"]]
        mapping = rec["label_references"]
        judgement_map = {item["label"]: item for item in results[envy["id"]]["judgements"]}
        lines += ["| Ref | Label | Relevant | Score | Reason |", "|---|---|---:|---:|---|"]
        for label, ref in mapping.items():
            j = judgement_map[label]
            escaped_reason = j["reason"].replace("|", "\\|")
            lines.append(
                f"| {ref} | {label} | {str(j['relevant']).lower()} | "
                f"{j['score']} | {escaped_reason} |"
            )
        envy_related = {
            case_id: next((row for row in valid_cases if row["id"] == case_id), None)
            for case_id in ("sem_ar_envy_keyword", "sem_ar_jealous_friend")
        }
        related_109: list[str] = []
        related_sixteen: list[str] = []
        for case_id, related_case in envy_related.items():
            if related_case is None:
                continue
            related_record = records[case_id]
            related_mapping = related_record["label_references"]
            related_judgement_map = {item["label"]: item for item in results[case_id]["judgements"]}
            for label, ref in related_mapping.items():
                if ref == "2:109":
                    status = "relevant" if related_judgement_map[label]["relevant"] else "rejected"
                    reason = related_judgement_map[label]["reason"].rstrip().rstrip(".")
                    related_109.append(f"{case_id}: {status} — {reason}")
                if ref.startswith("16:"):
                    status = "relevant" if related_judgement_map[label]["relevant"] else "rejected"
                    related_sixteen.append(f"{ref} ({case_id}: {status})")
        lines += [
            "",
            "2:109 across the keyword and situational envy pools: "
            + ("; ".join(related_109) or "absent from both pools")
            + ".",
            "16:x candidates across those pools: "
            + ("; ".join(related_sixteen) or "none in either pool")
            + ".",
        ]
    lines += ["", "## Ten baseline/judge disagreements", ""]
    if discrepancies:
        for item in discrepancies[:10]:
            lines += [
                (
                    f"### {item['case']['id']} — baseline "
                    f"{'hit' if item['baseline_hit'] else 'miss'}, judge "
                    f"{'hit' if item['judge_hit'] else 'miss'}"
                ),
                f"Question: {item['case']['prompt']}",
                f"Baseline refs: {', '.join(item['baseline']) or 'none'}",
                f"Judge top-3 refs: {', '.join(item['judge_refs']) or 'none'}",
            ]
            lines.extend(
                f"- {row['label']} ({row['score']}): {row['reason']}" for row in item["judge"]
            )
            lines.append("")
    else:
        lines.append("Fewer than ten disagreements occurred; all are listed above if present.")
    if len(discrepancies) < 10:
        lines.append(
            f"Observed hit/miss disagreements: {len(discrepancies)}; "
            "no cases were added to reach ten."
        )
    lines += [
        "## Run details",
        "",
        f"Command: `{run_info['command_text']}`",
        f"Judge wall time: {run_info['wall_seconds']:.1f}s across parallel batches; "
        "max concurrency: 6.",
        f"Aggregate CLI token usage: {json.dumps(run_info['usage_totals'], ensure_ascii=False)}",
        "Scoring note: an anchor hits when selected at the same verse or within ±2 verses in the "
        "same surah. The same window is applied to unacceptable anchors. Means include "
        "zero-selection cases.",
        "Limitations: single-model judgement, potentially correlated model error, no blinded human "
        "adjudication yet, and the baseline compares a pool-level offline selection with citations "
        "generated from full pipeline synthesis. The pool oracle is a recall ceiling, not a "
        "deployable arm. Completed calls used subprocess cwd for the empty per-case directory "
        "instead of passing the equivalent directory with -C; the runner now passes -C explicitly.",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    answer_cases = [case for case in cases if case.get("intent") == "answer"]
    records_list = _load_lines(args.responses)
    records_by_id = {record.get("id", record.get("case_id")): record for record in records_list}
    funnel = compute_funnel(cases, records_list)
    valid_source_cases: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    excluded_ids: list[str] = []
    pools: dict[str, list[str]] = {}
    baseline_by_id: dict[str, set[str]] = {}
    for case in answer_cases:
        cid = case["id"]
        if cid in INFRA_IDS:
            excluded_ids.append(cid)
            continue
        record = records_by_id.get(cid)
        pool = _ranked_pool(record) if record else None
        if not pool:
            skipped.append({"id": cid, "reason": "no recoverable nonempty fusion pool"})
            continue
        pools[cid] = pool
        valid_source_cases.append(case)
        baseline_by_id[cid] = _citation_references(record)
    funnel_ids = set(funnel["unavailable_trace_case_ids"])
    funnel_answer = [
        case
        for case in answer_cases
        if case["id"] not in INFRA_IDS and case["id"] not in funnel_ids
    ]
    # The funnel's late-binding stage is exact candidate-pool presence. Apply the
    # ±2 hydration window only to downstream selection arms, not this validation.
    reconstructed_hits = sum(
        bool(set(pools[case["id"]]) & set(case.get("acceptable_anchors", [])))
        for case in valid_source_cases
    )
    funnel_hits = sum(
        bool(
            set(case.get("acceptable_anchors", []))
            & (
                set(_selection_references(records_by_id[case["id"]])[0])
                | set(_selection_references(records_by_id[case["id"]])[1])
            )
        )
        for case in funnel_answer
    )
    reconstructed_rate = reconstructed_hits / len(valid_source_cases) if valid_source_cases else 0.0
    funnel_rate = funnel_hits / len(funnel_answer) if funnel_answer else 0.0
    if reconstructed_hits != funnel_hits or len(valid_source_cases) != len(funnel_answer):
        raise RuntimeError(
            f"Pool validation failed: reconstructed {reconstructed_hits}/"
            f"{len(valid_source_cases)}, funnel {funnel_hits}/{len(funnel_answer)}. "
            "No judge calls launched."
        )
    validation = {
        "reconstructed_hits": reconstructed_hits,
        "case_count": len(valid_source_cases),
        "reconstructed_rate": reconstructed_rate,
        "funnel_hits": funnel_hits,
        "funnel_cases": len(funnel_answer),
        "funnel_rate": funnel_rate,
    }
    print(
        f"Pool validation passed: {reconstructed_hits}/{len(valid_source_cases)} "
        f"({reconstructed_rate:.1%})",
        flush=True,
    )
    if args.validate_only:
        print(
            json.dumps(
                {"validation": validation, "infra_excluded": excluded_ids, "no_pool": skipped},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    canonical, tafsir = await _read_verse_text({ref for pool in pools.values() for ref in pool})
    by_id: dict[str, dict[str, Any]] = {}
    prompts: dict[str, list[dict[str, str]]] = {}
    for case in valid_source_cases:
        labels = {ref: f"V{index}" for index, ref in enumerate(pools[case["id"]], 1)}
        prompts[case["id"]] = [
            {"label": labels[ref], "verse_text": canonical[ref], "tafsir_text": tafsir[ref]}
            for ref in pools[case["id"]]
        ]
        by_id[case["id"]] = {
            "pool_references": pools[case["id"]],
            "baseline_references": sorted(baseline_by_id[case["id"]]),
            "label_references": {labels[ref]: ref for ref in pools[case["id"]]},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        kept = []
        for line in args.output.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id") in by_id:
                results[row["id"]] = row
                kept.append(json.dumps(row, ensure_ascii=False))
        args.output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    todo = [case for case in valid_source_cases if case["id"] not in results]
    usage: list[dict[str, Any]] = []
    command_text = (
        shlex.join(
            [
                str(CODEX),
                "exec",
                "-m",
                "gpt-6-luna",
                "-c",
                'model_reasoning_effort="high"',
                "-s",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--output-schema",
                str(SCHEMA),
                "--json",
                "-",
            ]
        )
        + "  (Arabic prompt on stdin; subprocess cwd=<empty per-case temp dir>)"
    )
    with args.output.open("a", encoding="utf-8") as out:
        for offset in range(0, len(todo), 6):
            batch = todo[offset : offset + 6]
            batch_results = await asyncio.gather(
                *(_judge_case(case, prompts[case["id"]]) for case in batch)
            )
            for case, result in zip(batch, batch_results, strict=True):
                row = {"id": case["id"], **result, **by_id[case["id"]]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                results[case["id"]] = row
                usage.extend(result.get("usage", []))
                print(
                    f"{len(results)}/{len(valid_source_cases)} {case['id']} {result['status']}",
                    flush=True,
                )
                if result["status"] == "usage_limit":
                    raise SystemExit(
                        f"Usage limit stopped experiment after {len(results)} "
                        "completed/recorded cases."
                    )
    ordered_rows = [
        results[case["id"]]
        for case in valid_source_cases
        if case["id"] in results and results[case["id"]].get("status") == "ok"
    ]
    batch_wall = sum(
        max((row.get("wall_seconds", 0.0) for row in ordered_rows[index : index + 6]), default=0.0)
        for index in range(0, len(ordered_rows), 6)
    )
    usage_totals: Counter[str] = Counter()
    for row in results.values():
        for usage_event in row.get("usage", []):
            if isinstance(usage_event, dict) and isinstance(
                usage_event.get("turn.completed"), dict
            ):
                usage_totals.update(
                    {
                        key: value
                        for key, value in usage_event["turn.completed"].items()
                        if isinstance(value, int)
                    }
                )
    run_info = {
        "wall_seconds": batch_wall,
        "command_text": command_text,
        "usage_totals": dict(usage_totals),
    }
    summary = _summary(valid_source_cases, results, by_id, skipped, validation, run_info)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary, encoding="utf-8")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
