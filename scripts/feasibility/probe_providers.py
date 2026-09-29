#!/usr/bin/env python3
"""Disposable, synthetic-only provider qualification probes.

This file is intentionally independent from application code. It records only
synthetic inputs, parsed outputs, and a small allow-list of response headers.
Authorization headers and environment-variable values are never serialized.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "artifacts" / "feasibility" / "results"

GROQ_MODEL = "openai/gpt-oss-20b"
CF_BGE_MODEL = "@cf/baai/bge-m3"
CF_QWEN_EMBED_MODEL = "@cf/qwen/qwen3-embedding-0.6b"
CF_QWEN_CHAT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
CF_LLAMA_CHAT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CF_DEEPSEEK_CHAT_MODEL = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"

ALLOWED_GROQ_MODELS = {GROQ_MODEL}
ALLOWED_CF_EMBED_MODELS = {CF_BGE_MODEL, CF_QWEN_EMBED_MODEL}
ALLOWED_CF_CHAT_MODELS = {
    CF_QWEN_CHAT_MODEL,
    CF_LLAMA_CHAT_MODEL,
    CF_DEEPSEEK_CHAT_MODEL,
}

SAFE_RESPONSE_HEADERS = {
    "cf-ray",
    "content-type",
    "request-id",
    "retry-after",
    "server",
    "x-groq-region",
    "x-request-id",
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Required environment variable is missing: {name}")
    return value


def require_assertion(name: str) -> None:
    if os.environ.get(name) != "1":
        raise SystemExit(
            f"Safety assertion {name}=1 is required after visual account verification."
        )


def sanitize_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in SAFE_RESPONSE_HEADERS or lowered.startswith("x-ratelimit-"):
            result[lowered] = str(value)
    return result


def safe_error_body(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")[:2000]
    # Defensive removal in case a provider unexpectedly reflects a bearer token.
    for secret_name in ("GROQ_API_KEY", "CLOUDFLARE_API_TOKEN"):
        secret = os.environ.get(secret_name)
        if secret:
            text = text.replace(secret, "<REDACTED>")
    text = re.sub(r"org_[A-Za-z0-9_-]+", "<REDACTED_ORGANIZATION>", text)
    return text


def post_json(
    url: str,
    payload: dict[str, Any],
    token: str,
    timeout: float = 45.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "islamic-research-ai-feasibility/1",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_ms": elapsed_ms,
                "headers": sanitize_headers(response.headers),
                "body": json.loads(raw.decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": elapsed_ms,
            "headers": sanitize_headers(exc.headers),
            "error": safe_error_body(exc.read()),
        }
    except Exception as exc:  # Network/timeout evidence, never credentials.
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "ok": False,
            "status": None,
            "latency_ms": elapsed_ms,
            "headers": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def exact_keys(value: Any, keys: Iterable[str]) -> bool:
    return isinstance(value, dict) and set(value) == set(keys)


SEARCH_OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["SEARCH_QURAN_TOPICS"]},
        "query": {"type": "string"},
        "language": {"type": "string", "enum": ["ar", "en", "mixed"]},
        "surah_filters": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["operation", "query", "language", "surah_filters"],
    "additionalProperties": False,
}

FETCH_OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["FETCH_QURAN_PASSAGE"]},
        "target": {
            "type": "string",
            "enum": ["QURAN_SEARCH_CANDIDATES"],
        },
        "translation_resource_alias": {"type": ["string", "null"]},
        "include_adjacent_verses": {"type": "boolean"},
    },
    "required": [
        "operation",
        "target",
        "translation_resource_alias",
        "include_adjacent_verses",
    ],
    "additionalProperties": False,
}

RESEARCH_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "enum": ["1"]},
        "intent": {"type": "string", "enum": ["QURAN_TOPIC_SEARCH"]},
        "response_language": {
            "type": "string",
            "enum": ["ar", "en", "mixed"],
        },
        "semantic_target_summary": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["TOPIC"]},
                "normalized_label": {"type": ["string", "null"]},
                "entity_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["kind", "normalized_label", "entity_labels"],
            "additionalProperties": False,
        },
        # The live test expects these arrays to be empty. String items keep every
        # object in the strict schema fully closed, which Groq requires even for
        # object types that the prompt never asks the model to instantiate.
        "inherited_context": {"type": "array", "items": {"type": "string"}},
        "clarification": {
            "type": "object",
            "properties": {
                "required": {"type": "boolean"},
                "reason": {"type": ["string", "null"]},
                "detail": {"type": ["string", "null"]},
                "missing_fields": {"type": "array", "items": {"type": "string"}},
                "candidates": {"type": "array", "items": {"type": "string"}},
                "question": {"type": ["string", "null"]},
            },
            "required": [
                "required",
                "reason",
                "detail",
                "missing_fields",
                "candidates",
                "question",
            ],
            "additionalProperties": False,
        },
        "scope_signals": {"type": "array", "items": {"type": "string"}},
        "requested_operations": {
            "type": "array",
            "items": {"anyOf": [SEARCH_OPERATION_SCHEMA, FETCH_OPERATION_SCHEMA]},
        },
    },
    "required": [
        "schema_version",
        "intent",
        "response_language",
        "semantic_target_summary",
        "inherited_context",
        "clarification",
        "scope_signals",
        "requested_operations",
    ],
    "additionalProperties": False,
}

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_type": {"type": "string", "enum": ["CLAIM"]},
        "claim_id": {"type": "string"},
        "text": {"type": "string"},
        "claim_type": {
            "type": "string",
            "enum": [
                "FACTUAL_SUMMARY",
                "INTERPRETIVE_SUMMARY",
                "COMPARISON",
                "COVERAGE_LIMITATION",
            ],
        },
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "enum": ["E1", "E2"]},
        },
    },
    "required": ["block_type", "claim_id", "text", "claim_type", "evidence_ids"],
    "additionalProperties": False,
}

SOURCE_BLOCK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_type": {"type": "string", "enum": ["SOURCE_BLOCK_REQUEST"]},
        "block_id": {"type": "string"},
        "quote_reference": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "enum": ["E1", "E2"]},
                "approved_span_id": {"type": ["string", "null"]},
            },
            "required": ["evidence_id", "approved_span_id"],
            "additionalProperties": False,
        },
        "quote_mode": {"type": "string", "enum": ["FULL", "APPROVED_SPAN"]},
    },
    "required": ["block_type", "block_id", "quote_reference", "quote_mode"],
    "additionalProperties": False,
}

NOTICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_type": {"type": "string", "enum": ["NOTICE"]},
        "notice_code": {
            "type": "string",
            "enum": [
                "INSUFFICIENT_EVIDENCE",
                "PARTIAL_COVERAGE",
                "SOURCE_UNAVAILABLE",
                "CLARIFICATION",
                "UNSUPPORTED_SCOPE",
            ],
        },
        "text": {"type": "string"},
    },
    "required": ["block_type", "notice_code", "text"],
    "additionalProperties": False,
}

ANSWER_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "enum": ["1"]},
        "status": {
            "type": "string",
            "enum": [
                "ANSWERABLE",
                "PARTIAL",
                "CLARIFICATION_REQUIRED",
                "UNSUPPORTED_SCOPE",
                "INSUFFICIENT_EVIDENCE",
                "TEMPORARILY_UNAVAILABLE",
                "FAILED_SAFE",
            ],
        },
        "blocks": {
            "type": "array",
            "items": {
                "anyOf": [CLAIM_SCHEMA, SOURCE_BLOCK_SCHEMA, NOTICE_SCHEMA],
            },
        },
    },
    "required": ["schema_version", "status", "blocks"],
    "additionalProperties": False,
}


RESEARCH_FIXTURES: Sequence[tuple[str, str, str]] = (
    ("ar", "ما الآيات التي تتحدث عن الصبر وقت الشدة؟", "ar"),
    ("en", "What Quran passages discuss patience during hardship?", "en"),
    ("mixed", "عاوز Quran verses about الصبر وقت الشدة", "mixed"),
)

SYNTHETIC_EVIDENCE = (
    "E1: الثبات عند الصعوبات يساعد الإنسان على مواصلة العمل.\n"
    "E2: Community support can reduce stress during difficult periods."
)


def research_messages(text: str, expected_language: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return only a ResearchPlanV1 JSON object matching the supplied schema. "
                "This is a capability test, not religious research. Classify the request "
                "as QURAN_TOPIC_SEARCH, preserve meaningful Arabic wording in the search "
                "query when Arabic is present, request exactly SEARCH_QURAN_TOPICS then "
                "FETCH_QURAN_PASSAGE, use no inherited context or scope signals, and do "
                "not add facts, citations, or verse text."
            ),
        },
        {
            "role": "user",
            "content": f"Expected response_language={expected_language}. Input: {text}",
        },
    ]


def answer_messages(language: str) -> list[dict[str, str]]:
    instruction = {
        "ar": "اكتب الادعاء الموجز بالعربية.",
        "en": "Write the short claim in English.",
        "mixed": "Write a short mixed Arabic/English claim.",
    }[language]
    return [
        {
            "role": "system",
            "content": (
                "Return only an AnswerDraftV1 JSON object matching the supplied schema. "
                "The evidence is synthetic and non-religious. Return status ANSWERABLE, "
                "one small FACTUAL_SUMMARY claim referencing E1, and one FULL "
                "SOURCE_BLOCK_REQUEST for E1. Do not quote the evidence inside the claim."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\nSynthetic evidence:\n{SYNTHETIC_EVIDENCE}",
        },
    ]


def repair_research_messages(
    text: str,
    expected_language: str,
    validation_errors: Sequence[str],
) -> list[dict[str, str]]:
    """Build the sole permitted schema/policy-guided planner repair request.

    The repair deliberately does not include the original model output. Its user
    message contains only the same semantic request/context and local validation
    errors. The same native strict JSON Schema is supplied separately in
    ``response_format`` by the caller. There is no recursive retry path.
    """
    errors = "\n".join(f"- {error}" for error in validation_errors)
    return [
        {
            "role": "system",
            "content": (
                "Return only a corrected ResearchPlanV1 JSON object matching the "
                "supplied schema. This is a capability test, not religious research."
            ),
        },
        {
            "role": "user",
            "content": (
                "Semantic user request/context:\n"
                f"Expected response_language={expected_language}. Input: {text}\n\n"
                "Local validation errors from the initial output:\n"
                f"{errors}"
            ),
        },
    ]


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def validate_research(value: Any, expected_language: str) -> list[str]:
    errors: list[str] = []
    root = {
        "schema_version",
        "intent",
        "response_language",
        "semantic_target_summary",
        "inherited_context",
        "clarification",
        "scope_signals",
        "requested_operations",
    }
    if not exact_keys(value, root):
        return ["root fields do not exactly match ResearchPlanV1 probe schema"]
    if value["schema_version"] != "1":
        errors.append("schema_version")
    if value["intent"] != "QURAN_TOPIC_SEARCH":
        errors.append("intent")
    if value["response_language"] != expected_language:
        errors.append("response_language")
    summary = value["semantic_target_summary"]
    if not exact_keys(summary, {"kind", "normalized_label", "entity_labels"}):
        errors.append("semantic_target_summary fields")
    elif summary["kind"] != "TOPIC":
        errors.append("semantic_target_summary.kind")
    clarification = value["clarification"]
    clarification_fields = {
        "required",
        "reason",
        "detail",
        "missing_fields",
        "candidates",
        "question",
    }
    if not exact_keys(clarification, clarification_fields):
        errors.append("clarification fields")
    elif not (
        clarification["required"] is False
        and clarification["reason"] is None
        and clarification["detail"] is None
        and clarification["missing_fields"] == []
        and clarification["candidates"] == []
        and clarification["question"] is None
    ):
        errors.append("clarification values")
    if value["inherited_context"] != []:
        errors.append("inherited_context")
    if value["scope_signals"] != []:
        errors.append("scope_signals")
    operations = value["requested_operations"]
    if not isinstance(operations, list) or len(operations) != 2:
        errors.append("requested_operations count")
        return errors
    search, fetch = operations
    if not exact_keys(search, {"operation", "query", "language", "surah_filters"}):
        errors.append("search operation fields")
    else:
        if search["operation"] != "SEARCH_QURAN_TOPICS":
            errors.append("search operation")
        if search["language"] != expected_language:
            errors.append("search language")
        if not isinstance(search["query"], str) or not search["query"].strip():
            errors.append("search query")
        if expected_language in {"ar", "mixed"} and not contains_arabic(search["query"]):
            errors.append("Arabic preservation")
        if search["surah_filters"] != []:
            errors.append("surah_filters")
    if not exact_keys(
        fetch,
        {
            "operation",
            "target",
            "translation_resource_alias",
            "include_adjacent_verses",
        },
    ):
        errors.append("fetch operation fields")
    else:
        if fetch["operation"] != "FETCH_QURAN_PASSAGE":
            errors.append("fetch operation")
        if fetch["target"] != "QURAN_SEARCH_CANDIDATES":
            errors.append("fetch target")
        if fetch["translation_resource_alias"] is not None:
            errors.append("unapproved translation alias")
        if not isinstance(fetch["include_adjacent_verses"], bool):
            errors.append("include_adjacent_verses")
    return errors


def validate_answer(value: Any) -> list[str]:
    errors: list[str] = []
    if not exact_keys(value, {"schema_version", "status", "blocks"}):
        return ["root fields do not exactly match AnswerDraftV1 probe schema"]
    if value["schema_version"] != "1":
        errors.append("schema_version")
    if value["status"] != "ANSWERABLE":
        errors.append("status")
    blocks = value["blocks"]
    if not isinstance(blocks, list):
        return errors + ["blocks type"]
    claims = [
        item for item in blocks if isinstance(item, dict) and item.get("block_type") == "CLAIM"
    ]
    sources = [
        item
        for item in blocks
        if isinstance(item, dict) and item.get("block_type") == "SOURCE_BLOCK_REQUEST"
    ]
    notices = [
        item for item in blocks if isinstance(item, dict) and item.get("block_type") == "NOTICE"
    ]
    if len(claims) != 1 or len(sources) != 1 or notices:
        errors.append("block composition")
        return errors
    claim = claims[0]
    if not exact_keys(claim, {"block_type", "claim_id", "text", "claim_type", "evidence_ids"}):
        errors.append("claim fields")
    else:
        if claim["claim_type"] != "FACTUAL_SUMMARY":
            errors.append("claim_type")
        if claim["evidence_ids"] != ["E1"]:
            errors.append("claim evidence_ids")
    source = sources[0]
    if not exact_keys(source, {"block_type", "block_id", "quote_reference", "quote_mode"}):
        errors.append("source block fields")
    elif not (
        source["quote_mode"] == "FULL"
        and exact_keys(source["quote_reference"], {"evidence_id", "approved_span_id"})
        and source["quote_reference"]["evidence_id"] == "E1"
        and source["quote_reference"]["approved_span_id"] is None
    ):
        errors.append("source block values")
    return errors


def research_tool_plan_errors(value: Any, expected_language: str) -> list[str]:
    """Return only intent/tool-plan semantic errors, independent of repair state.

    This intentionally excludes clarification and response-language fields so a
    valid plan's tool selection can be measured separately from the full planner
    contract. It is not an execution authority and does not relax validation.
    """
    if not isinstance(value, dict):
        return ["no JSON object"]
    errors: list[str] = []
    if value.get("intent") != "QURAN_TOPIC_SEARCH":
        errors.append("wrong intent")
    operations = value.get("requested_operations")
    if not isinstance(operations, list) or len(operations) != 2:
        return errors + ["wrong ToolOperation set"]
    search, fetch = operations
    if not isinstance(search, dict) or not isinstance(fetch, dict):
        return errors + ["wrong ToolOperation set"]
    if (
        search.get("operation") != "SEARCH_QURAN_TOPICS"
        or fetch.get("operation") != "FETCH_QURAN_PASSAGE"
    ):
        errors.append("wrong ToolOperation set")
    if search.get("language") != expected_language:
        errors.append("search language")
    query = search.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append("search query")
    elif expected_language in {"ar", "mixed"} and not contains_arabic(query):
        errors.append("Arabic preservation")
    if fetch.get("target") != "QURAN_SEARCH_CANDIDATES":
        errors.append("wrong ToolOperation set")
    if fetch.get("translation_resource_alias") is not None:
        errors.append("wrong ToolOperation set")
    if not isinstance(fetch.get("include_adjacent_verses"), bool):
        errors.append("wrong ToolOperation set")
    return errors


def classify_research_failure(record: dict[str, Any], expected_language: str) -> str:
    """Classify a failed planner attempt using the requested A-H taxonomy."""
    if record.get("status") != 200:
        return "A. Transport/API failure"
    errors = record.get("validation_errors", [])
    if any("JSONDecodeError" in error for error in errors):
        return "B. Invalid JSON"
    value = record.get("value")
    if not isinstance(value, dict):
        return "C. JSON Schema violation"
    if value.get("response_language") != expected_language:
        return "G. Language-detection/response-language disagreement"
    if value.get("intent") != "QURAN_TOPIC_SEARCH":
        return "E. Valid ResearchPlan schema but wrong intent"
    tool_errors = research_tool_plan_errors(value, expected_language)
    if any(error == "wrong ToolOperation set" for error in tool_errors):
        return "F. Valid intent but wrong ToolOperation set"
    if any(error in {"schema_version", "search language"} for error in errors + tool_errors):
        return "D. Valid schema but invalid enum/value"
    if any("fields" in error or "root fields" in error for error in errors):
        return "C. JSON Schema violation"
    return "H. Other semantic-policy failure"


def extract_chat_content(provider: str, body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if provider == "groq":
        content = body["choices"][0]["message"]["content"]
        return content, {
            "response_id": body.get("id"),
            "reported_model": body.get("model"),
            "usage": body.get("usage"),
            "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
        }
    result = body.get("result", body)
    if isinstance(result, dict) and "choices" in result:
        content = result["choices"][0]["message"]["content"]
    else:
        content = result.get("response") if isinstance(result, dict) else None
    return content, {
        "response_id": result.get("id") if isinstance(result, dict) else None,
        "reported_model": result.get("model") if isinstance(result, dict) else None,
        "usage": result.get("usage") if isinstance(result, dict) else None,
        "finish_reason": (
            result.get("choices", [{}])[0].get("finish_reason")
            if isinstance(result, dict) and result.get("choices")
            else None
        ),
    }


def parse_json_content(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError(f"unsupported response content type: {type(content).__name__}")
    return json.loads(content)


def summarize_calls(calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in calls if item.get("valid")]
    latencies = [
        item["latency_ms"] for item in calls if isinstance(item.get("latency_ms"), (int, float))
    ]
    return {
        "attempted": len(calls),
        "valid": len(successes),
        "valid_percent": round(100 * len(successes) / len(calls), 2) if calls else 0,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "http_statuses": sorted({item.get("status") for item in calls}, key=str),
    }


def run_chat_suite(
    provider: str,
    model: str,
    repetitions: int,
    pace_seconds: float,
    max_tokens: int,
    token: str,
    account_id: str | None = None,
) -> dict[str, Any]:
    if provider == "groq":
        if model not in ALLOWED_GROQ_MODELS:
            raise SystemExit("Refusing non-allow-listed Groq model")
        url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        if model not in ALLOWED_CF_CHAT_MODELS:
            raise SystemExit("Refusing non-allow-listed Cloudflare chat model")
        quoted_model = urllib.parse.quote(model, safe="@/")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{quoted_model}"

    calls: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for language, text, expected_language in RESEARCH_FIXTURES:
            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "research_plan_v1_probe",
                        "strict": True,
                        "schema": RESEARCH_PLAN_SCHEMA,
                    },
                }
                if provider == "groq"
                else {"type": "json_schema", "json_schema": RESEARCH_PLAN_SCHEMA}
            )
            payload: dict[str, Any] = {
                "messages": research_messages(text, expected_language),
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            if provider == "groq":
                payload["model"] = model
            response = post_json(url, payload, token)
            record: dict[str, Any] = {
                "contract": "ResearchPlanV1",
                "language": language,
                "repetition": repetition,
                "status": response.get("status"),
                "latency_ms": response.get("latency_ms"),
                "headers": response.get("headers", {}),
                "valid": False,
            }
            if response.get("ok"):
                try:
                    content, metadata = extract_chat_content(provider, response["body"])
                    value = parse_json_content(content)
                    errors = validate_research(value, expected_language)
                    record.update(metadata)
                    record["value"] = value
                    record["validation_errors"] = errors
                    record["valid"] = not errors and metadata.get("reported_model") in {None, model}
                    if metadata.get("reported_model") not in {None, model}:
                        record["validation_errors"].append("reported model mismatch")
                except Exception as exc:
                    record["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
            else:
                record["error"] = response.get("error")
            calls.append(record)
            time.sleep(pace_seconds)

        for language, _, _ in RESEARCH_FIXTURES:
            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_draft_v1_probe",
                        "strict": True,
                        "schema": ANSWER_DRAFT_SCHEMA,
                    },
                }
                if provider == "groq"
                else {"type": "json_schema", "json_schema": ANSWER_DRAFT_SCHEMA}
            )
            payload = {
                "messages": answer_messages(language),
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            if provider == "groq":
                payload["model"] = model
            response = post_json(url, payload, token)
            record = {
                "contract": "AnswerDraftV1",
                "language": language,
                "repetition": repetition,
                "status": response.get("status"),
                "latency_ms": response.get("latency_ms"),
                "headers": response.get("headers", {}),
                "valid": False,
            }
            if response.get("ok"):
                try:
                    content, metadata = extract_chat_content(provider, response["body"])
                    value = parse_json_content(content)
                    errors = validate_answer(value)
                    record.update(metadata)
                    record["value"] = value
                    record["validation_errors"] = errors
                    record["valid"] = not errors and metadata.get("reported_model") in {None, model}
                    if metadata.get("reported_model") not in {None, model}:
                        record["validation_errors"].append("reported model mismatch")
                except Exception as exc:
                    record["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
            else:
                record["error"] = response.get("error")
            calls.append(record)
            time.sleep(pace_seconds)

    by_contract_language: dict[str, Any] = {}
    for contract in ("ResearchPlanV1", "AnswerDraftV1"):
        for language, _, _ in RESEARCH_FIXTURES:
            selected = [
                item
                for item in calls
                if item["contract"] == contract and item["language"] == language
            ]
            by_contract_language[f"{contract}:{language}"] = summarize_calls(selected)
    return {
        "probe": f"{provider}-chat",
        "checked_at": utc_now(),
        "provider": provider,
        "effective_provider": provider,
        "pinned_model": model,
        "synthetic_only": True,
        "paid_calls_enabled": False,
        "repetitions_per_language_contract": repetitions,
        "summary": summarize_calls(calls),
        "summary_by_contract_language": by_contract_language,
        "calls": calls,
    }


def groq_research_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "research_plan_v1_probe",
            "strict": True,
            "schema": RESEARCH_PLAN_SCHEMA,
        },
    }


def run_groq_research_attempt(
    messages: list[dict[str, str]],
    expected_language: str,
    token: str,
    max_tokens: int,
    phase: str,
) -> dict[str, Any]:
    response = post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": groq_research_response_format(),
        },
        token,
    )
    record: dict[str, Any] = {
        "phase": phase,
        "status": response.get("status"),
        "latency_ms": response.get("latency_ms"),
        "headers": response.get("headers", {}),
        "valid": False,
        "semantic_tool_plan_correct": False,
    }
    if not response.get("ok"):
        record["error"] = response.get("error")
        record["failure_taxonomy"] = classify_research_failure(record, expected_language)
        return record
    try:
        content, metadata = extract_chat_content("groq", response["body"])
        value = parse_json_content(content)
        errors = validate_research(value, expected_language)
        record.update(metadata)
        record["value"] = value
        record["validation_errors"] = errors
        if metadata.get("reported_model") != GROQ_MODEL:
            record["validation_errors"].append("reported model mismatch")
        record["valid"] = not record["validation_errors"]
        record["semantic_tool_plan_errors"] = research_tool_plan_errors(value, expected_language)
        record["semantic_tool_plan_correct"] = not record["semantic_tool_plan_errors"]
    except Exception as exc:
        record["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
    if not record["valid"]:
        record["failure_taxonomy"] = classify_research_failure(record, expected_language)
    return record


def summarize_repair_phase(cases: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    attempts = [case[field] for case in cases]
    return {
        "attempted": len(attempts),
        "valid": sum(bool(attempt.get("valid")) for attempt in attempts),
        "semantic_tool_plan_correct": sum(
            bool(attempt.get("semantic_tool_plan_correct")) for attempt in attempts
        ),
        "failure_taxonomy": {
            category: sum(attempt.get("failure_taxonomy") == category for attempt in attempts)
            for category in (
                "A. Transport/API failure",
                "B. Invalid JSON",
                "C. JSON Schema violation",
                "D. Valid schema but invalid enum/value",
                "E. Valid ResearchPlan schema but wrong intent",
                "F. Valid intent but wrong ToolOperation set",
                "G. Language-detection/response-language disagreement",
                "H. Other semantic-policy failure",
            )
        },
    }


def run_groq_one_repair_case(
    case_id: str,
    text: str,
    expected_language: str,
    token: str,
    max_tokens: int,
    pace_seconds: float,
) -> dict[str, Any]:
    initial = run_groq_research_attempt(
        research_messages(text, expected_language),
        expected_language,
        token,
        max_tokens,
        "initial",
    )
    repair: dict[str, Any] | None = None
    # A transport failure has no invalid model object to repair. Every parsed or
    # malformed model response gets exactly one repair attempt, and never more.
    if not initial["valid"] and initial.get("status") == 200:
        time.sleep(pace_seconds)
        repair = run_groq_research_attempt(
            repair_research_messages(
                text,
                expected_language,
                initial.get("validation_errors", ["invalid planner output"]),
            ),
            expected_language,
            token,
            max_tokens,
            "repair",
        )
    final = repair if repair is not None else initial
    case = {
        "case_id": case_id,
        "language": expected_language,
        "initial": initial,
        "repair": repair,
        "final": final,
    }
    time.sleep(pace_seconds)
    return case


def run_groq_one_repair_suite(
    token: str,
    max_tokens: int,
    pace_seconds: float,
    mixed_repetitions: int,
) -> dict[str, Any]:
    mixed = next(item for item in RESEARCH_FIXTURES if item[0] == "mixed")
    cases: list[dict[str, Any]] = []
    # The five case identifiers map to the five failing iterations in the prior
    # 120-call run. Replays use the same synthetic semantic fixture, never prior
    # output text or religious content.
    for repetition in (1, 14, 15, 18, 20):
        cases.append(
            run_groq_one_repair_case(
                f"prior-mixed-failure-replay-{repetition}",
                mixed[1],
                mixed[2],
                token,
                max_tokens,
                pace_seconds,
            )
        )
    representative: list[dict[str, Any]] = []
    for repetition in range(1, mixed_repetitions + 1):
        representative.append(
            run_groq_one_repair_case(
                f"representative-mixed-{repetition}",
                mixed[1],
                mixed[2],
                token,
                max_tokens,
                pace_seconds,
            )
        )
    return {
        "probe": "groq-one-repair",
        "checked_at": utc_now(),
        "provider": "groq",
        "effective_provider": "groq",
        "pinned_model": GROQ_MODEL,
        "synthetic_only": True,
        "paid_calls_enabled": False,
        "repair_policy": {
            "maximum_repairs_per_case": 1,
            "recursive_retries": False,
            "repair_contains": [
                "same semantic user request/context",
                "local validation errors",
                "same native output schema",
            ],
        },
        "prior_failure_replays": {
            "cases": cases,
            "first_pass": summarize_repair_phase(cases, "initial"),
            "final_after_one_repair": summarize_repair_phase(cases, "final"),
        },
        "representative_mixed_reliability": {
            "cases": representative,
            "first_pass": summarize_repair_phase(representative, "initial"),
            "final_after_one_repair": summarize_repair_phase(representative, "final"),
        },
    }


def extract_vectors(body: dict[str, Any]) -> tuple[list[list[float]], dict[str, Any]]:
    result = body.get("result", body)
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list):
        raise ValueError("response has no vector data array")
    vectors: list[list[float]] = []
    for item in data:
        vector = item.get("embedding") if isinstance(item, dict) else item
        if not isinstance(vector, list):
            raise ValueError("vector item has unexpected shape")
        converted = [float(number) for number in vector]
        if not converted or not all(math.isfinite(number) for number in converted):
            raise ValueError("vector is empty or contains a non-finite value")
        vectors.append(converted)
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("vectors have inconsistent dimensions")
    return vectors, {
        "shape": result.get("shape") if isinstance(result, dict) else None,
        "usage": result.get("usage") if isinstance(result, dict) else None,
        "reported_model": result.get("model") if isinstance(result, dict) else None,
    }


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def run_cf_embedding_call(
    account_id: str,
    token: str,
    model: str,
    texts: Sequence[str],
) -> dict[str, Any]:
    if model not in ALLOWED_CF_EMBED_MODELS:
        raise SystemExit("Refusing non-allow-listed Cloudflare embedding model")
    quoted_model = urllib.parse.quote(model, safe="@/")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{quoted_model}"
    response = post_json(url, {"text": list(texts)}, token)
    record: dict[str, Any] = {
        "model": model,
        "input_count": len(texts),
        "status": response.get("status"),
        "latency_ms": response.get("latency_ms"),
        "headers": response.get("headers", {}),
        "valid": False,
    }
    if response.get("ok"):
        try:
            vectors, metadata = extract_vectors(response["body"])
            record.update(metadata)
            record["vector_count"] = len(vectors)
            record["dimension"] = len(vectors[0]) if vectors else None
            record["valid"] = len(vectors) == len(texts)
            record["vectors"] = vectors
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    else:
        record["error"] = response.get("error")
    return record


def run_cf_async_probe(account_id: str, token: str) -> dict[str, Any]:
    model = CF_BGE_MODEL
    quoted_model = urllib.parse.quote(model, safe="@/")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        f"{quoted_model}?queueRequest=true"
    )
    request_payload = {
        "requests": [
            {
                "query": "الصبر وقت الشدة",
                "contexts": [
                    {"text": "الثبات عند الصعوبات يساعد الإنسان على مواصلة العمل."},
                    {"text": "تحتاج النباتات إلى الماء والضوء كي تنمو."},
                    {"text": "Patience and perseverance during hardship."},
                ],
                "external_reference": "synthetic-arabic-feasibility-1",
            }
        ]
    }
    queued = post_json(url, request_payload, token)
    record: dict[str, Any] = {
        "submitted_status": queued.get("status"),
        "submitted_latency_ms": queued.get("latency_ms"),
        "submitted_headers": queued.get("headers", {}),
        "valid": False,
    }
    if not queued.get("ok"):
        record["error"] = queued.get("error")
        return record
    result = queued["body"].get("result", {})
    request_id = result.get("request_id")
    record["reported_model"] = result.get("model")
    record["queue_status"] = result.get("status")
    record["request_id"] = request_id
    if not request_id:
        record["error"] = "queue response did not contain request_id"
        return record
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        polled = post_json(url, {"request_id": request_id}, token)
        record["poll_status"] = polled.get("status")
        record["poll_headers"] = polled.get("headers", {})
        if not polled.get("ok"):
            record["error"] = polled.get("error")
            return record
        poll_result = polled["body"].get("result", {})
        if "responses" in poll_result:
            record["responses"] = poll_result.get("responses")
            record["usage"] = poll_result.get("usage")
            record["valid"] = bool(poll_result.get("responses"))
            return record
        record["queue_status"] = poll_result.get("status", record.get("queue_status"))
        time.sleep(2)
    record["error"] = "async result did not complete within 45 seconds"
    return record


def run_cf_embedding_suite(
    account_id: str,
    token: str,
    include_oversized: bool,
    include_async: bool,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    fixtures = [
        "الصبر وقت الشدة",
        "الثبات عند الصعوبات يساعد الإنسان على مواصلة العمل.",
        "تحتاج النباتات إلى الماء والضوء كي تنمو.",
        "Patience and perseverance during hardship.",
    ]
    for model in (CF_BGE_MODEL, CF_QWEN_EMBED_MODEL):
        semantic = run_cf_embedding_call(account_id, token, model, fixtures)
        vectors = semantic.pop("vectors", None)
        if semantic.get("valid") and vectors:
            semantic["cosine"] = {
                "arabic_related": cosine(vectors[0], vectors[1]),
                "arabic_unrelated": cosine(vectors[0], vectors[2]),
                "english_related": cosine(vectors[0], vectors[3]),
            }
            semantic["ordering_pass"] = (
                semantic["cosine"]["arabic_related"] > semantic["cosine"]["arabic_unrelated"]
                and semantic["cosine"]["english_related"] > semantic["cosine"]["arabic_unrelated"]
            )

        batch_attempts: list[dict[str, Any]] = []
        for size in (1, 3, 16, 32, 64, 128):
            texts = [f"Synthetic resilience sentence number {index}." for index in range(size)]
            attempt = run_cf_embedding_call(account_id, token, model, texts)
            attempt.pop("vectors", None)
            batch_attempts.append(attempt)
            if not attempt.get("valid"):
                break

        oversized = None
        if include_oversized:
            limit = 60050 if model == CF_BGE_MODEL else 8250
            oversized_text = "test " * limit
            oversized = run_cf_embedding_call(account_id, token, model, [oversized_text])
            oversized.pop("vectors", None)

        models[model] = {
            "semantic_probe": semantic,
            "sync_batch_attempts": batch_attempts,
            "max_successful_sync_batch": max(
                (item["input_count"] for item in batch_attempts if item.get("valid")),
                default=0,
            ),
            "oversized_input_probe": oversized,
        }

    async_result = run_cf_async_probe(account_id, token) if include_async else None
    return {
        "probe": "cloudflare-embeddings",
        "checked_at": utc_now(),
        "provider": "cloudflare",
        "effective_provider": "cloudflare",
        "synthetic_only": True,
        "paid_calls_enabled": False,
        "models": models,
        "async_bge_m3_probe": async_result,
    }


def write_result(result: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{stamp}-{result['probe']}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    groq_parser = subparsers.add_parser("groq")
    groq_parser.add_argument("--repetitions", type=int, default=20)
    groq_parser.add_argument("--pace-seconds", type=float, default=2.1)
    groq_parser.add_argument("--max-tokens", type=int, default=700)

    repair_parser = subparsers.add_parser("groq-one-repair")
    repair_parser.add_argument("--mixed-repetitions", type=int, default=10)
    repair_parser.add_argument("--pace-seconds", type=float, default=14)
    repair_parser.add_argument("--max-tokens", type=int, default=1000)

    embedding_parser = subparsers.add_parser("cloudflare-embeddings")
    embedding_parser.add_argument("--include-oversized", action="store_true")
    embedding_parser.add_argument("--skip-async", action="store_true")

    cf_chat_parser = subparsers.add_parser("cloudflare-chat")
    cf_chat_parser.add_argument("--repetitions", type=int, default=20)
    cf_chat_parser.add_argument("--pace-seconds", type=float, default=0.25)
    cf_chat_parser.add_argument("--max-tokens", type=int, default=700)
    cf_chat_parser.add_argument(
        "--model",
        choices=sorted(ALLOWED_CF_CHAT_MODELS),
        default=CF_QWEN_CHAT_MODEL,
    )

    args = parser.parse_args()
    repetitions = getattr(args, "repetitions", getattr(args, "mixed_repetitions", 1))
    if repetitions < 1 or repetitions > 20:
        raise SystemExit("Repetitions must be between 1 and 20")
    if getattr(args, "max_tokens", 1) < 100 or getattr(args, "max_tokens", 1) > 2000:
        raise SystemExit("max-tokens must be between 100 and 2000")

    if args.command == "groq":
        require_assertion("GROQ_FREE_ONLY_CONFIRMED")
        require_assertion("GROQ_ZDR_CONFIRMED")
        result = run_chat_suite(
            "groq",
            GROQ_MODEL,
            args.repetitions,
            args.pace_seconds,
            args.max_tokens,
            require_env("GROQ_API_KEY"),
        )
    elif args.command == "groq-one-repair":
        require_assertion("GROQ_FREE_ONLY_CONFIRMED")
        require_assertion("GROQ_ZDR_CONFIRMED")
        result = run_groq_one_repair_suite(
            require_env("GROQ_API_KEY"),
            args.max_tokens,
            args.pace_seconds,
            args.mixed_repetitions,
        )
    elif args.command == "cloudflare-embeddings":
        require_assertion("CLOUDFLARE_FREE_ONLY_CONFIRMED")
        require_assertion("CLOUDFLARE_NO_GATEWAY_CREDITS_CONFIRMED")
        result = run_cf_embedding_suite(
            require_env("CLOUDFLARE_ACCOUNT_ID"),
            require_env("CLOUDFLARE_API_TOKEN"),
            args.include_oversized,
            not args.skip_async,
        )
    else:
        require_assertion("CLOUDFLARE_FREE_ONLY_CONFIRMED")
        require_assertion("CLOUDFLARE_NO_GATEWAY_CREDITS_CONFIRMED")
        result = run_chat_suite(
            "cloudflare",
            args.model,
            args.repetitions,
            args.pace_seconds,
            args.max_tokens,
            require_env("CLOUDFLARE_API_TOKEN"),
            require_env("CLOUDFLARE_ACCOUNT_ID"),
        )

    path = write_result(result)
    if result.get("probe") == "groq-one-repair":
        summary = {
            "prior_failure_replays": {
                "first_pass": result["prior_failure_replays"]["first_pass"],
                "final_after_one_repair": result["prior_failure_replays"]["final_after_one_repair"],
            },
            "representative_mixed_reliability": {
                "first_pass": result["representative_mixed_reliability"]["first_pass"],
                "final_after_one_repair": result["representative_mixed_reliability"][
                    "final_after_one_repair"
                ],
            },
        }
    else:
        summary = result.get("summary") or {
            model: {
                "semantic_valid": details["semantic_probe"].get("valid"),
                "dimension": details["semantic_probe"].get("dimension"),
                "ordering_pass": details["semantic_probe"].get("ordering_pass"),
                "max_successful_sync_batch": details["max_successful_sync_batch"],
            }
            for model, details in result.get("models", {}).items()
        }
    print(json.dumps({"result_file": str(path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
