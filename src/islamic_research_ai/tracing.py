from __future__ import annotations

import re
from collections.abc import Callable
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from islamic_research_ai.contracts import (
    AnswerDraftV1,
    ClaimBlockV1,
    SourceBlockRequestV1,
    TraceEventV1,
)

_SENSITIVE_TRACE_KEY = re.compile(
    r"(?:authorization|token|secret|api[_-]?key|password|database_url|connection|"
    r"exact_text|passage_text|source_text|canonical.*text)",
    re.I,
)


class TraceRecorder:
    """The single event source for a turn; details are bounded before they leave process memory."""

    def __init__(
        self, enabled: bool, callback: Callable[[TraceEventV1], None] | None = None
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.started = perf_counter()
        self.events: list[TraceEventV1] = []

    def emit(
        self,
        node: str,
        phase: str,
        summary: str,
        *,
        detail: dict[str, object] | None = None,
        provider: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = TraceEventV1(
            seq=len(self.events) + 1,
            elapsed_ms=round((perf_counter() - self.started) * 1000),
            node=node,
            phase=phase,
            summary=summary[:240],
            detail=self._safe(detail or {}),
            provider=self._safe(provider) if provider is not None else None,
        )
        self.events.append(event)
        if self.callback is not None:
            self.callback(event)

    @classmethod
    def _safe(cls, value: object, *, depth: int = 0) -> Any:
        if depth > 4:
            return "[TRUNCATED_DEPTH]"
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]"
                if _SENSITIVE_TRACE_KEY.search(str(key))
                else cls._safe(item, depth=depth + 1)
                for key, item in list(value.items())[:24]
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._safe(item, depth=depth + 1) for item in list(value)[:24]]
        if isinstance(value, str):
            return value[:240] + ("[TRUNCATED]" if len(value) > 240 else "")
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)[:240]


_TRACE_RECORDER: ContextVar[TraceRecorder | None] = ContextVar("trace_recorder", default=None)


def rejected_synthesis_detail(
    *,
    attempt: int | str,
    error: Exception,
    repair_guidance: str,
    draft: AnswerDraftV1 | None,
    is_tafsir_route: bool,
) -> dict[str, object]:
    """Describe a rejected synthesized draft without including canonical source text."""

    if draft is None:
        return {
            "attempt": attempt,
            "error_type": type(error).__name__,
            "error": str(error),
            "repair_guidance": repair_guidance,
            "draft_status": None,
            "claim_count": None,
            "claims": None,
            "requested_evidence_ids": None,
            "is_tafsir_route": is_tafsir_route,
        }

    claims = [block for block in draft.blocks if isinstance(block, ClaimBlockV1)]
    requests = [block for block in draft.blocks if isinstance(block, SourceBlockRequestV1)]
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "repair_guidance": repair_guidance,
        "draft_status": draft.status.value,
        "claim_count": len(claims),
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
                "claim_type": claim.claim_type.value,
            }
            for claim in claims
        ],
        "requested_evidence_ids": [
            request.quote_reference.evidence_id for request in requests
        ],
        "is_tafsir_route": is_tafsir_route,
    }
