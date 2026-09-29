from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from islamic_research_ai.contracts import (
    AnswerStatus,
    ConversationCreateRequest,
    ConversationTurnCreateRequest,
    NoticeCode,
    ResearchResponseV1,
    TraceEventV1,
)
from islamic_research_ai.conversations import ConversationService
from islamic_research_ai.persistence import RESTRICTED_SOURCE_TEXT_REDACTION
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourcePolicy
from islamic_research_ai.settings import ROOT
from islamic_research_ai.synthesis import terminal_response

DATABASE_URL = "postgresql://127.0.0.1:55432/islamic_research_ai"


def trace_events() -> list[TraceEventV1]:
    return [
        TraceEventV1(
            seq=1,
            elapsed_ms=4,
            node="retrieval",
            phase="completed",
            summary="Hydrated canonical evidence",
            detail={"database": "postgres", "stage": "canonical_hydration"},
            provider=None,
        )
    ]


class RecordingResearchService:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts = []
        self.trace_callback_events: list[TraceEventV1] = []

    async def research(self, request, *, recent_user_turns, on_trace_event=None):
        self.calls += 1
        self.contexts.append(list(recent_user_turns))
        events = trace_events()
        if on_trace_event is not None:
            for event in events:
                on_trace_event(event)
                self.trace_callback_events.append(event)
        await asyncio.sleep(0)
        response = terminal_response(
            conversation_id=request.conversation_id,
            turn_id=uuid4(),
            correlation_id=uuid4(),
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            notice_code=NoticeCode.INSUFFICIENT_EVIDENCE,
            text="Stable application notice",
        )
        return response, SimpleNamespace(events=events)


def qf_response(*, conversation_id=None, turn_id=None) -> ResearchResponseV1:
    conversation_id = conversation_id or uuid4()
    turn_id = turn_id or uuid4()
    citation = {
        "citation_id": "C1",
        "evidence_id": "E1",
        "display_label": "Qualified Tafsir — Quran Foundation",
        "source_kind": "TAFSIR",
        "reference": {
            "scheme": "quran",
            "value": "2:153",
            "canonical_link": "https://quran.com/2:153/tafsirs/en-tafisr-ibn-kathir",
        },
        "attribution": [
            {
                "role": "PUBLISHER",
                "display_name": "Quran Foundation",
                "required_notice": None,
            }
        ],
        "canonical_link": "https://quran.com/2:153/tafsirs/en-tafisr-ibn-kathir",
        "source_revision": {
            "source_id": "quran_foundation",
            "resource_alias": "qf-tafsir-ibn-kathir-en",
            "revision_id": "request-time:test",
            "checksum": "sha256:test-only",
        },
    }
    return ResearchResponseV1.model_validate(
        {
            "schema_version": "1",
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "correlation_id": uuid4(),
            "status": "ANSWERABLE",
            "blocks": [
                {
                    "block_type": "CLAIM",
                    "claim_id": "CL1",
                    "text": "The cited commentary discusses patience.",
                    "citations": [citation],
                },
                {
                    "block_type": "SOURCE",
                    "block_id": "S1",
                    "exact_text": "RESTRICTED_TEST_SENTINEL",
                    "citation": citation,
                },
            ],
            "citations": [citation],
            "clarification": None,
            "warnings": [],
        }
    )


def test_historical_qf_citation_loads_without_an_active_source_policy_entry() -> None:
    policy = SourcePolicy(
        ROOT / "config/source_policies.development.json",
        environment="development",
        purpose="private_poc",
        override_acknowledged=True,
    )
    historical = qf_response()

    assert "qf-tafsir-ibn-kathir-en" not in policy.resources
    restored = ResearchResponseV1.model_validate(historical.model_dump(mode="json"))
    assert restored.citations[0].source_revision.resource_alias == "qf-tafsir-ibn-kathir-en"


@pytest.mark.asyncio
async def test_database_history_crud_pagination_isolation_and_qf_redaction() -> None:
    repository = PostgresRepository(DATABASE_URL)
    try:
        await repository.open()
    except Exception as exc:
        pytest.skip(f"Local PostgreSQL is unavailable: {type(exc).__name__}")

    owner = f"pytest:{uuid4()}"
    other_owner = f"pytest:{uuid4()}"
    created_ids = []
    try:
        # Both migrations are intentionally safe to replay.
        await repository.migrate(ROOT / "migrations/003_redact_qf_tafsir_source_text.sql")
        await repository.migrate(ROOT / "migrations/004_conversation_history.sql")
        await repository.migrate(ROOT / "migrations/006_turn_trace_events.sql")
        await repository.migrate(ROOT / "migrations/004_conversation_history.sql")
        await repository.migrate(ROOT / "migrations/006_turn_trace_events.sql")

        for index in range(3):
            conversation = await repository.create_conversation(
                conversation_id=uuid4(),
                owner_key=owner,
                title=f"Conversation {index}",
                locale="en",
            )
            created_ids.append(conversation.conversation_id)

        first_page, cursor = await repository.list_conversations(
            owner_key=owner,
            include_archived=False,
            limit=2,
            cursor=None,
        )
        assert len(first_page) == 2 and cursor is not None
        second_page, final_cursor = await repository.list_conversations(
            owner_key=owner,
            include_archived=False,
            limit=2,
            cursor=cursor,
        )
        assert len(second_page) == 1 and final_cursor is None
        assert {item.conversation_id for item in first_page + second_page} == set(created_ids)
        isolated, _ = await repository.list_conversations(
            owner_key=other_owner,
            include_archived=True,
            limit=10,
            cursor=None,
        )
        assert isolated == []

        selected = created_ids[0]
        response = qf_response(conversation_id=selected)
        persisted_events = trace_events()
        turn = await repository.persist_conversation_turn(
            conversation_id=selected,
            owner_key=owner,
            idempotency_key="idem-history-0001",
            request_hash="hash-1",
            user_message="Explain Quran 2:153.",
            locale="en",
            response=response,
            trace_events=persisted_events,
        )
        assert turn.response.blocks[1].exact_text == RESTRICTED_SOURCE_TEXT_REDACTION
        assert turn.trace_events == persisted_events
        repeated = await repository.persist_conversation_turn(
            conversation_id=selected,
            owner_key=owner,
            idempotency_key="idem-history-0001",
            request_hash="hash-1",
            user_message="Explain Quran 2:153.",
            locale="en",
            response=response,
            trace_events=persisted_events,
        )
        assert repeated.turn_id == turn.turn_id
        with pytest.raises(ValueError, match="Idempotency key"):
            await repository.persist_conversation_turn(
                conversation_id=selected,
                owner_key=owner,
                idempotency_key="idem-history-0001",
                request_hash="different-hash",
                user_message="Different request",
                locale="en",
                response=qf_response(conversation_id=selected),
                trace_events=persisted_events,
            )

        detail, next_turn_cursor = await repository.get_conversation(
            conversation_id=selected,
            owner_key=owner,
            turn_limit=10,
        )
        assert detail is not None and next_turn_cursor is None
        assert [item.user_message for item in detail.turns] == ["Explain Quran 2:153."]
        assert detail.turns[0].response.blocks[1].exact_text == RESTRICTED_SOURCE_TEXT_REDACTION
        assert detail.turns[0].trace_events == persisted_events

        renamed = await repository.update_conversation(
            conversation_id=selected,
            owner_key=owner,
            title="Renamed",
            archived=True,
        )
        assert renamed is not None and renamed.title == "Renamed" and renamed.archived
        visible, _ = await repository.list_conversations(
            owner_key=owner,
            include_archived=False,
            limit=10,
            cursor=None,
        )
        assert selected not in {item.conversation_id for item in visible}
        assert await repository.delete_conversation(conversation_id=selected, owner_key=owner)
        created_ids.remove(selected)

        async with repository.pool.connection() as connection:
            for table in (
                "research_response_records",
                "legacy_research_response_records",
                "conversation_turns",
            ):
                cursor_result = await connection.execute(
                    f"""
                    SELECT count(*) AS violations FROM {table}
                    WHERE jsonb_path_exists(
                        response,
                        '$.blocks[*] ? (@.block_type == "SOURCE" && '
                        '@.citation.source_kind == "TAFSIR" && '
                        '@.citation.source_revision.source_id == "quran_foundation" && '
                        '@.exact_text != "[SOURCE_TEXT_NOT_STORED_POLICY]")'
                    )
                    """
                )
                assert (await cursor_result.fetchone())["violations"] == 0
            constraint_cursor = await connection.execute(
                """
                SELECT count(*) AS count FROM pg_constraint
                WHERE conname IN (
                    'research_response_records_no_qf_raw_text',
                    'legacy_research_response_records_no_qf_raw_text',
                    'conversation_turns_no_qf_raw_text'
                )
                """
            )
            assert (await constraint_cursor.fetchone())["count"] == 3
    finally:
        if repository.pool._opened:
            async with repository.pool.connection() as connection:
                await connection.execute("DELETE FROM conversations WHERE owner_key = %s", (owner,))
            await repository.close()


@pytest.mark.asyncio
async def test_conversation_service_auto_title_context_and_concurrent_idempotency() -> None:
    repository = PostgresRepository(DATABASE_URL)
    try:
        await repository.open()
    except Exception as exc:
        pytest.skip(f"Local PostgreSQL is unavailable: {type(exc).__name__}")
    owner = f"pytest:{uuid4()}"
    research = RecordingResearchService()
    service = ConversationService(repository, research)  # type: ignore[arg-type]
    try:
        await repository.migrate(ROOT / "migrations/004_conversation_history.sql")
        await repository.migrate(ROOT / "migrations/006_turn_trace_events.sql")
        conversation = await service.create(ConversationCreateRequest(locale="en"), owner_key=owner)
        first = ConversationTurnCreateRequest(
            message="What does the Quran say about patience during hardship?",
            locale="en",
            idempotency_key="concurrent-submit-0001",
        )
        first_a, first_b = await asyncio.gather(
            service.submit_turn(conversation.conversation_id, first, owner_key=owner),
            service.submit_turn(conversation.conversation_id, first, owner_key=owner),
        )
        assert first_a.turn_id == first_b.turn_id
        assert research.calls == 1

        received_trace_events: list[TraceEventV1] = []
        second = await service.submit_turn(
            conversation.conversation_id,
            ConversationTurnCreateRequest(
                message="Show that verse and its translation.",
                locale="en",
                idempotency_key="follow-up-submit-0002",
            ),
            owner_key=owner,
            on_trace_event=received_trace_events.append,
        )
        assert second.user_message.startswith("Show that verse")
        assert research.calls == 2
        assert [item.user_message for item in research.contexts[1]] == [first.message]
        assert received_trace_events == trace_events()
        assert research.trace_callback_events == trace_events()

        detail, _ = await service.get(conversation.conversation_id, owner_key=owner)
        assert detail.title == first.message
        assert [item.user_message for item in detail.turns] == [first.message, second.user_message]
    finally:
        async with repository.pool.connection() as connection:
            await connection.execute("DELETE FROM conversations WHERE owner_key = %s", (owner,))
        await repository.close()
