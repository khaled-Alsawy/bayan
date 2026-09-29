from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def runner_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "qa/equivalent_query_runner.py"
    spec = importlib.util.spec_from_file_location("equivalent_query_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_part_d_manifest_has_ten_valid_equivalent_pairs() -> None:
    runner = runner_module()
    cases = runner.load_yaml(runner.DEFAULT_CASES)
    pairs = runner.load_yaml(runner.DEFAULT_PAIRS)
    runner.validate_pairs(cases, pairs)
    assert len(pairs) == 10


def test_part_d_reports_status_delta_across_both_orders() -> None:
    runner = runner_module()
    pairs = [{"id": "same_meaning", "first": "a", "second": "b"}]
    records = {
        "same_meaning:forward": {"a": "ANSWERABLE", "b": "ANSWERABLE"},
        "same_meaning:reverse": {"a": "FAILED_SAFE", "b": "ANSWERABLE"},
    }
    result = runner.score_pairs(pairs, records)
    assert result["status_delta_count"] == 1
    assert result["outcomes"][0]["consistent"] is False
