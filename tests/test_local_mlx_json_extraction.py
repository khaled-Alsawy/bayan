from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from islamic_research_ai.providers.local_mlx import extract_first_json_object


class _ValueModel(BaseModel):
    value: int


@pytest.mark.parametrize(
    "completion",
    [
        'Here is the requested result:\n{"value":1}',
        '<think>Let me reason through the request.</think>\n{"value":1}',
        '{"value":1}\nThe JSON above is the final answer.',
        '{"value":1}\n{"value":2}',
    ],
)
def test_extract_first_balanced_json_object_from_completion(completion: str) -> None:
    extracted = extract_first_json_object(completion)

    assert _ValueModel.model_validate_json(extracted).value == 1
    assert json.loads(extracted) == {"value": 1}


def test_extract_first_json_object_does_not_repair_invalid_object_content() -> None:
    completion = 'preamble {"value": not-json} commentary'

    with pytest.raises(ValueError):
        _ValueModel.model_validate_json(extract_first_json_object(completion))
