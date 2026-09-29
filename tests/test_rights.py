from __future__ import annotations

import json
from pathlib import Path

import pytest

from islamic_research_ai.rights import (
    OverrideType,
    RightDecision,
    SourceAuthorizationError,
    SourceOperation,
    SourcePolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_policy(tmp_path: Path, decision: RightDecision) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "override_type": "PROJECT_OWNER_ACCEPTED_RISK",
                "override_reason": "test owner decision",
                "override_operations": ["INDEX_VECTOR"],
                "resources": [
                    {
                        "resource_alias": "source",
                        "rights": {"INDEX_VECTOR": decision.value},
                        "external_constraints": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def policy(path: Path, *, environment: str, acknowledged: bool) -> SourcePolicy:
    return SourcePolicy(
        path,
        environment=environment,
        purpose="private_poc",
        override_acknowledged=acknowledged,
    )


def test_unknown_without_override_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(SourceAuthorizationError):
        policy(
            make_policy(tmp_path, RightDecision.UNKNOWN),
            environment="development",
            acknowledged=False,
        ).authorize("source", SourceOperation.INDEX_VECTOR)


def test_unknown_with_override_is_permitted_in_development(tmp_path: Path) -> None:
    decision = policy(
        make_policy(tmp_path, RightDecision.UNKNOWN),
        environment="development",
        acknowledged=True,
    ).authorize("source", SourceOperation.INDEX_VECTOR)
    assert decision.rights_decision is RightDecision.UNKNOWN
    assert decision.override_type is OverrideType.PROJECT_OWNER_ACCEPTED_RISK
    assert decision.effective_permission is True


def test_unknown_with_override_is_blocked_in_production(tmp_path: Path) -> None:
    with pytest.raises(SourceAuthorizationError):
        policy(
            make_policy(tmp_path, RightDecision.UNKNOWN),
            environment="production",
            acknowledged=True,
        ).authorize("source", SourceOperation.INDEX_VECTOR)


def test_denied_always_blocks(tmp_path: Path) -> None:
    with pytest.raises(SourceAuthorizationError):
        policy(
            make_policy(tmp_path, RightDecision.DENIED),
            environment="development",
            acknowledged=True,
        ).authorize("source", SourceOperation.INDEX_VECTOR)


def test_allowed_does_not_need_override(tmp_path: Path) -> None:
    decision = policy(
        make_policy(tmp_path, RightDecision.ALLOWED),
        environment="production",
        acknowledged=False,
    ).authorize("source", SourceOperation.INDEX_VECTOR)
    assert decision.override_type is OverrideType.NONE
    assert decision.effective_permission is True


def test_approved_development_policy_authorizes_local_llm_inference() -> None:
    source_policy = SourcePolicy(
        PROJECT_ROOT / "config/source_policies.development.json",
        environment="development",
        purpose="private_poc",
        override_acknowledged=True,
    )
    decision = source_policy.authorize(
        "tanzil-uthmani-v1.1",
        SourceOperation.LOCAL_LLM_INFERENCE,
        deployment={
            "gateway_provider_id": "local",
            "effective_provider_id": "local",
            "model_id": "gemma-4-26b-a4b-it",
            "inference_location": "LOCAL",
            "zero_data_retention": True,
            "provider_training": False,
            "aggregator_routing": False,
        },
    )
    assert decision.rights_decision is RightDecision.UNKNOWN
    assert decision.override_type is OverrideType.PROJECT_OWNER_ACCEPTED_RISK
    assert decision.effective_permission is True


def test_production_policy_blocks_local_llm_inference() -> None:
    source_policy = SourcePolicy(
        PROJECT_ROOT / "config/source_policies.production.json",
        environment="production",
        purpose="production",
        override_acknowledged=True,
    )
    with pytest.raises(SourceAuthorizationError):
        source_policy.authorize(
            "tanzil-uthmani-v1.1",
            SourceOperation.LOCAL_LLM_INFERENCE,
            deployment={
                "gateway_provider_id": "local",
                "effective_provider_id": "local",
                "model_id": "gemma-4-26b-a4b-it",
                "inference_location": "LOCAL",
                "zero_data_retention": True,
                "provider_training": False,
                "aggregator_routing": False,
            },
        )


def test_local_llm_inference_cannot_omit_deployment_metadata() -> None:
    source_policy = SourcePolicy(
        PROJECT_ROOT / "config/source_policies.development.json",
        environment="development",
        purpose="private_poc",
        override_acknowledged=True,
    )
    with pytest.raises(SourceAuthorizationError):
        source_policy.authorize(
            "tanzil-uthmani-v1.1",
            SourceOperation.LOCAL_LLM_INFERENCE,
        )
