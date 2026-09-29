from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from islamic_research_ai import api
from islamic_research_ai.contracts import (
    AnswerDraftV1,
    AnswerStatus,
    NoticeBlockV1,
    NoticeCode,
    ResearchRequest,
)


def answer_draft_block_limit_error() -> ValidationError:
    with pytest.raises(ValidationError) as raised:
        AnswerDraftV1(
            schema_version="1",
            status=AnswerStatus.ANSWERABLE,
            blocks=[
                NoticeBlockV1(
                    block_type="NOTICE",
                    notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
                    text="Invariant test notice",
                )
                for _ in range(25)
            ],
        )
    return raised.value


@pytest.mark.asyncio
async def test_api_logs_contract_validation_as_an_internal_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariant_error = answer_draft_block_limit_error()
    events: list[tuple[str, dict[str, object]]] = []

    class BrokenService:
        async def research(self, _: ResearchRequest) -> None:
            raise invariant_error

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                container=SimpleNamespace(
                    service=BrokenService(),
                    settings=SimpleNamespace(public_access_token=None),
                )
            )
        ),
        client=SimpleNamespace(host="127.0.0.1"),
        headers={},
    )
    monkeypatch.setattr(api, "log_event", lambda event, **fields: events.append((event, fields)))

    with pytest.raises(HTTPException) as raised:
        await api.research(ResearchRequest(message="patience"), request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "Research request failed safely"
    assert events == [
        (
            "research_request_internal_invariant_violation",
            {
                "error_type": "ValidationError",
                "validation_model": "AnswerDraftV1",
                "validation_errors": [
                    {
                        "location": ["blocks"],
                        "constraint": "too_long",
                        "message": "List should have at most 24 items after validation, not 25",
                        "context": {"field_type": "List", "max_length": 24, "actual_length": 25},
                        "invalid_collection_count": 25,
                    }
                ],
            },
        )
    ]
