from __future__ import annotations

from pathlib import Path

import pytest

from qa.question_bank_pilot import (
    GENERATION_FAILED,
    GENERATION_OUTPUT_EXAMPLE_JSON,
    GENERATION_SYSTEM_PROMPT,
    CandidateRecord,
    GeneratedQuestionPair,
    GeneratedWindowQuestionBank,
    GenerationFailureRateExceeded,
    GenerationWindow,
    PoolRecord,
    QuestionPair,
    _assert_generation_failure_rate,
    _bank_outcome_counts,
    _complete_question_bank,
    _load_banks,
    _pending_windows,
    _safe_generation_prompt,
    assert_production_top3,
    early_rejection,
    generate_banks,
    score_pools,
    validate_question_bank,
)


def _pool(case_id: str = "case") -> PoolRecord:
    return PoolRecord(
        case_id=case_id,
        original_question="unused in pure fixture",
        acceptable_anchors=frozenset({"2:5"}),
        candidates=(
            CandidateRecord("2:1", 0.9, 1),
            CandidateRecord("2:7", 0.8, 2),
            CandidateRecord("2:10", 0.7, 3),
            CandidateRecord("2:5", 0.6, 4),
        ),
        production_top3=("2:1", "2:7", "2:10"),
    )


def test_pool_reconstruction_assertion_aborts_on_nonproduction_order() -> None:
    pool = _pool()
    assert_production_top3(pool)
    wrong = PoolRecord(
        case_id=pool.case_id,
        original_question=pool.original_question,
        acceptable_anchors=pool.acceptable_anchors,
        candidates=pool.candidates,
        production_top3=("2:7", "2:1", "2:10"),
    )
    with pytest.raises(AssertionError, match="pool reconstruction mismatch"):
        assert_production_top3(wrong)


def test_fixed_max_scoring_uses_windows_and_early_rejection_range() -> None:
    pool = _pool()
    maps = {
        pool.case_id: {
            "tafsir_window": {"2:1": 0.1, "2:7": 0.4, "2:10": 0.3, "2:5": 0.9},
            "bank_msa": {"2:1": 0.1, "2:7": 0.2, "2:10": 0.8, "2:5": 0.7},
            "bank_colloquial": {"2:1": 0.1, "2:7": 0.2, "2:10": 0.3, "2:5": 0.99},
            "bank_all": {"2:1": 0.1, "2:7": 0.2, "2:10": 0.3, "2:5": 0.99},
        }
    }
    result = score_pools([pool], maps)
    # 2:7 is within the target's ±2 hydration window, so the production baseline succeeds.
    assert result["hydration_successes"]["production_rrf"] == 1
    assert result["cases"][pool.case_id]["scorers"]["bank_all"]["anchors"][0] == "2:5"
    assert early_rejection([pool], result) is True


def test_question_bank_validation_is_pair_atomic_and_structural_only() -> None:
    window = GenerationWindow(
        window_id="2:1",
        verses=({"reference": "2:1", "arabic": "نص", "translation": "source text"},),
        almukhtasar_passages=(),
    )
    raw = GeneratedWindowQuestionBank(
        questions=[
            GeneratedQuestionPair(
                supporting_clause_ids=["verse:2:1:arabic"],
                msa="x",
                egyptian_colloquial=" س  ",
            ),
            # Exact duplicates collapse after Unicode and whitespace normalization.
            GeneratedQuestionPair(
                supporting_clause_ids=["verse:2:1:translation"],
                msa="ｘ",
                egyptian_colloquial="س",
            ),
        ]
    )
    accepted = validate_question_bank(raw, window)
    assert len(accepted.questions) == 1
    assert accepted.questions[0].msa == "x"  # no hidden minimum-length or semantic rule

    invalid = GeneratedWindowQuestionBank(
        questions=[
            GeneratedQuestionPair(
                supporting_clause_ids=["missing-clause"],
                msa="valid MSA",
                egyptian_colloquial="valid Egyptian",
            ),
            GeneratedQuestionPair(
                supporting_clause_ids=["verse:2:1:arabic"],
                msa=None,
                egyptian_colloquial="orphaned pair",
            ),
        ]
    )
    assert validate_question_bank(invalid, window).questions == []
    assert "verse:2:1:arabic" in _safe_generation_prompt(window)


def test_documented_generation_output_example_round_trips_through_validation() -> None:
    window = GenerationWindow(
        window_id="2:1",
        verses=({"reference": "2:1", "arabic": "نص", "translation": "source text"},),
        almukhtasar_passages=(),
    )

    raw = GeneratedWindowQuestionBank.model_validate_json(GENERATION_OUTPUT_EXAMPLE_JSON)
    accepted = validate_question_bank(raw, window)

    assert GENERATION_OUTPUT_EXAMPLE_JSON in GENERATION_SYSTEM_PROMPT
    assert accepted.questions == [
        QuestionPair(
            supporting_clause_ids=["verse:2:1:arabic", "verse:2:1:translation"],
            msa="ما الذي ينبغي فعله عند توثيق دين إلى أجل مسمى؟",
            egyptian_colloquial="نعمل إيه لما نوثق دين لحد وقت محدد؟",
        ),
        QuestionPair(
            supporting_clause_ids=["verse:2:1:arabic", "verse:2:1:translation"],
            msa="ما الشرط المتعلق بموعد سداد الدين المذكور؟",
            egyptian_colloquial="إيه الشرط اللي يخص ميعاد سداد الدين المذكور؟",
        ),
    ]


def test_generation_prompt_pins_window_wide_two_pair_cap_and_example() -> None:
    assert "Hard cap for the entire window: return at most two question pairs total" in (
        GENERATION_SYSTEM_PROMPT
    )

    example = GeneratedWindowQuestionBank.model_validate_json(GENERATION_OUTPUT_EXAMPLE_JSON)
    assert len(example.questions) == 2
    assert all(len(pair.supporting_clause_ids) > 1 for pair in example.questions)


@pytest.mark.asyncio
async def test_parse_failure_is_recorded_as_generation_failed_not_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import qa.question_bank_pilot as pilot
    from islamic_research_ai import composition, tools
    from islamic_research_ai.ports import StructuredOutputValidationError

    class Completion:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            return {"mode": mode}

    class LocalChat:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self, **kwargs: object
        ) -> tuple[GeneratedWindowQuestionBank, Completion]:
            self.calls += 1
            if self.calls <= 2:
                raise StructuredOutputValidationError("not JSON")
            return GeneratedWindowQuestionBank(), Completion()

    class Container:
        def __init__(self) -> None:
            self.local_chat = LocalChat()
            self.service = type("Service", (), {"source": object()})()
            self.repository = object()

        async def close(self) -> None:
            return None

    class TafsirSource:
        def __init__(self, repository: object) -> None:
            self.repository = repository

    async def window_for_candidate(
        source: object, tafsir_source: object, candidate: CandidateRecord
    ) -> GenerationWindow:
        return GenerationWindow(candidate.reference, verses=(), almukhtasar_passages=())

    async def build_container() -> Container:
        return container

    container = Container()
    monkeypatch.setattr(composition, "build_container", build_container)
    monkeypatch.setattr(tools, "PostgresQulTafsirSource", TafsirSource)
    monkeypatch.setattr(pilot, "_configure_runtime_environment", lambda: None)
    monkeypatch.setattr(pilot, "_window_for_candidate", window_for_candidate)
    pool = PoolRecord(
        case_id="case",
        original_question="unused",
        acceptable_anchors=frozenset(),
        candidates=tuple(CandidateRecord(f"2:{index}", 0.0, index) for index in range(1, 11)),
        production_top3=(),
    )

    await generate_banks(tmp_path, [pool], phase="test")
    counts = _bank_outcome_counts(_load_banks(tmp_path / "window_banks.jsonl"))

    assert counts == {
        "two_pairs": 0,
        "one_pair": 0,
        "legitimately_empty_bank": 9,
        "generation_failed": 1,
    }


@pytest.mark.asyncio
async def test_parse_failure_retries_once_with_the_identical_prompt() -> None:
    from islamic_research_ai.ports import StructuredOutputValidationError

    class AlwaysMalformed:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def complete(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            raise StructuredOutputValidationError("not JSON")

    local_chat = AlwaysMalformed()
    bank, completion, error_type, attempts = await _complete_question_bank(
        local_chat, prompt='{"source_clauses":[]}'
    )

    assert bank is None
    assert completion is None
    assert error_type == "StructuredOutputValidationError"
    assert attempts == 2
    assert len(local_chat.calls) == 2
    assert local_chat.calls[0] == local_chat.calls[1]
    assert local_chat.calls[0]["user_prompt"] == '{"source_clauses":[]}'


def test_resume_skips_a_window_with_a_cached_generation_failure() -> None:
    window = GenerationWindow(window_id="2:1", verses=(), almukhtasar_passages=())

    assert _pending_windows(
        {window.window_id: window},
        {
            window.window_id: {
                "window_id": window.window_id,
                "generation_status": GENERATION_FAILED,
                "generation_error_type": "StructuredOutputValidationError",
            }
        },
    ) == []


def test_generation_does_not_abort_before_minimum_attempted_sample() -> None:
    windows = {"0": GenerationWindow("0", verses=(), almukhtasar_passages=())}
    banks = {
        "0": {
            "window_id": "0",
            "generation_status": GENERATION_FAILED,
            "generation_error_type": "StructuredOutputValidationError",
        }
    }

    _assert_generation_failure_rate(banks, windows)


def test_generation_aborts_when_failures_exceed_ten_percent_after_minimum_sample() -> None:
    windows = {
        str(index): GenerationWindow(str(index), verses=(), almukhtasar_passages=())
        for index in range(20)
    }
    banks = {
        **{
            str(index): {
                "window_id": str(index),
                "generation_status": GENERATION_FAILED,
                "generation_error_type": "StructuredOutputValidationError",
            }
            for index in range(3)
        },
        **{
            str(index): {"window_id": str(index), "generation_status": "generation_succeeded"}
            for index in range(3, 20)
        },
    }

    with pytest.raises(GenerationFailureRateExceeded, match="above the 10% limit"):
        _assert_generation_failure_rate(banks, windows)
