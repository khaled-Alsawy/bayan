from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from islamic_research_ai.logging import log_event


class RightDecision(StrEnum):
    ALLOWED = "ALLOWED"
    UNKNOWN = "UNKNOWN"
    DENIED = "DENIED"


class OverrideType(StrEnum):
    NONE = "NONE"
    PROJECT_OWNER_ACCEPTED_RISK = "PROJECT_OWNER_ACCEPTED_RISK"


class SourceOperation(StrEnum):
    FETCH_REMOTE = "FETCH_REMOTE"
    DISPLAY = "DISPLAY"
    STORE = "STORE"
    CACHE = "CACHE"
    INDEX_LEXICAL = "INDEX_LEXICAL"
    INDEX_VECTOR = "INDEX_VECTOR"
    LOCAL_EMBEDDING = "LOCAL_EMBEDDING"
    EXTERNAL_EMBEDDING = "EXTERNAL_EMBEDDING"
    EXTERNAL_LLM_INFERENCE = "EXTERNAL_LLM_INFERENCE"
    LOCAL_LLM_INFERENCE = "LOCAL_LLM_INFERENCE"
    EXTERNAL_RERANKING = "EXTERNAL_RERANKING"
    REDISTRIBUTE = "REDISTRIBUTE"
    COMMERCIAL_USE = "COMMERCIAL_USE"


class AuthorizationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_alias: str
    operation: SourceOperation
    rights_decision: RightDecision
    override_type: OverrideType
    effective_permission: bool
    environment: str
    purpose: str
    reason: str
    effective_deployment: dict[str, object] | None


class SourceAuthorizationError(PermissionError):
    pass


class SourcePolicy:
    _DEVELOPMENT_PURPOSES = {
        "private_poc",
        "non_public_demo",
        "technical_evaluation",
    }

    def __init__(
        self,
        policy_path: Path,
        *,
        environment: str,
        purpose: str,
        override_acknowledged: bool,
    ) -> None:
        self.data = json.loads(policy_path.read_text(encoding="utf-8"))
        self.environment = environment
        self.purpose = purpose
        self.override_acknowledged = override_acknowledged
        self.resources = {r["resource_alias"]: r for r in self.data["resources"]}

    def authorize(
        self,
        resource_alias: str,
        operation: SourceOperation,
        *,
        deployment: dict[str, object] | None = None,
    ) -> AuthorizationDecision:
        resource = self.resources.get(resource_alias)
        if resource is None:
            raise SourceAuthorizationError(f"Unknown source resource: {resource_alias}")
        original = RightDecision(resource["rights"].get(operation.value, "UNKNOWN"))
        override = OverrideType.NONE
        permitted = original is RightDecision.ALLOWED
        reason = f"Original right is {original.value}."

        if original is RightDecision.DENIED:
            permitted = False
            reason = "Explicit DENIED always blocks the operation."
        elif original is RightDecision.UNKNOWN:
            configured_override = OverrideType(self.data.get("override_type", "NONE"))
            override_ops = set(self.data.get("override_operations", []))
            development_override = (
                self.environment == "development"
                and self.purpose in self._DEVELOPMENT_PURPOSES
                and configured_override is OverrideType.PROJECT_OWNER_ACCEPTED_RISK
                and self.override_acknowledged
                and operation.value in override_ops
            )
            if development_override:
                override = OverrideType.PROJECT_OWNER_ACCEPTED_RISK
                permitted = True
                reason = self.data.get("override_reason", "Project-owner accepted risk.")
            else:
                reason = "UNKNOWN right is blocked without the bounded development override."

        if permitted and operation is SourceOperation.LOCAL_LLM_INFERENCE and deployment is None:
            raise SourceAuthorizationError("Local inference requires effective deployment metadata")
        if permitted and deployment is not None:
            if operation is SourceOperation.LOCAL_LLM_INFERENCE:
                self._validate_local_deployment(deployment)
            else:
                self._validate_deployment(resource, operation, deployment)

        decision = AuthorizationDecision(
            resource_alias=resource_alias,
            operation=operation,
            rights_decision=original,
            override_type=override,
            effective_permission=permitted,
            environment=self.environment,
            purpose=self.purpose,
            reason=reason,
            effective_deployment=deployment,
        )
        log_event(
            "source_authorization_decision",
            resource_alias=resource_alias,
            operation=operation.value,
            rights_decision=original.value,
            override_type=override.value,
            effective_permission=permitted,
            environment=self.environment,
            purpose=self.purpose,
            effective_deployment=deployment,
        )
        if not permitted:
            raise SourceAuthorizationError(
                f"{resource_alias}:{operation.value} blocked ({original.value}; {reason})"
            )
        return decision

    @staticmethod
    def _validate_deployment(
        resource: dict[str, object],
        operation: SourceOperation,
        deployment: dict[str, object],
    ) -> None:
        constraints = resource.get("external_constraints", {}).get(operation.value)
        if constraints is None:
            raise SourceAuthorizationError(
                f"No approved deployment constraint for {operation.value}"
            )
        for supplied_key, allowed_key in (
            ("gateway_provider_id", "gateway_provider_ids"),
            ("effective_provider_id", "effective_provider_ids"),
            ("model_id", "model_ids"),
        ):
            if deployment.get(supplied_key) not in constraints.get(allowed_key, []):
                raise SourceAuthorizationError(
                    f"Deployment {supplied_key} is outside the source policy"
                )
        if constraints.get("zero_data_retention_required") and not deployment.get(
            "zero_data_retention"
        ):
            raise SourceAuthorizationError("Zero-data-retention is required")
        if not constraints.get("provider_training_allowed", True) and deployment.get(
            "provider_training"
        ):
            raise SourceAuthorizationError("Provider training is not permitted")
        if not constraints.get("aggregator_routing_allowed", True) and deployment.get(
            "aggregator_routing"
        ):
            raise SourceAuthorizationError("Aggregator routing is not permitted")

    @staticmethod
    def _validate_local_deployment(deployment: dict[str, object]) -> None:
        if (
            deployment.get("gateway_provider_id") != "local"
            or deployment.get("effective_provider_id") != "local"
            or deployment.get("model_id") != "gemma-4-26b-a4b-it"
            or deployment.get("inference_location") != "LOCAL"
            or not deployment.get("zero_data_retention")
            or deployment.get("provider_training")
            or deployment.get("aggregator_routing")
        ):
            raise SourceAuthorizationError("Local inference deployment is outside policy")
