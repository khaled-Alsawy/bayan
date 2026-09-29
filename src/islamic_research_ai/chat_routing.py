from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from islamic_research_ai.local_inference import local_completion_attribution
from islamic_research_ai.planner import PlannerTemporarilyUnavailable
from islamic_research_ai.ports import ProviderFailureDiagnostic
from islamic_research_ai.retrieval.relevance import assess_topic_evidence
from islamic_research_ai.routing import GROQ_CHAT_DEPLOYMENT_ID, LOCAL_GEMMA_DEPLOYMENT_ID
from islamic_research_ai.synthesis import SynthesisTemporarilyUnavailable

FailoverStage = Literal[
    "PLANNER",
    "SYNTHESIS",
    "COMPOSITION",
    "FORCED_LOCAL_ACCEPTANCE",
    "SOURCE_POLICY_LOCAL_SYNTHESIS",
]


@dataclass
class TurnChatRouteState:
    """Mutable, request-local routing state; never share it between turns."""

    force_local: bool = False
    local_selected: bool = False
    selection_stage: FailoverStage | None = None
    provider_failures: list[ProviderFailureDiagnostic] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.force_local:
            self.local_selected = True
            self.selection_stage = "FORCED_LOCAL_ACCEPTANCE"
            self.decisions.append("FORCED_LOCAL_ACCEPTANCE")

    def record_failure(self, diagnostic: ProviderFailureDiagnostic) -> None:
        self.provider_failures.append(diagnostic)

    def select_local_after_quota(
        self, diagnostic: ProviderFailureDiagnostic, *, stage: FailoverStage
    ) -> bool:
        self.record_failure(diagnostic)
        eligible = diagnostic.provider_id == "groq" and diagnostic.category in {
            "QUOTA_EXHAUSTED",
            "RATE_LIMITED",
        }
        if not eligible:
            self.decisions.append(f"NO_FAILOVER_{stage}_{diagnostic.category}")
            return False
        self.local_selected = True
        self.selection_stage = stage
        self.decisions.append(f"GROQ_QUOTA_TO_LOCAL_{stage}")
        return True

    def select_local_for_source_policy(self) -> None:
        self.local_selected = True
        self.selection_stage = "SOURCE_POLICY_LOCAL_SYNTHESIS"
        self.decisions.append("SOURCE_POLICY_LOCAL_SYNTHESIS")

    def snapshot(self) -> dict[str, Any]:
        return {
            "selected_route": "LOCAL" if self.local_selected else "GROQ_DIRECT",
            "selection_stage": self.selection_stage,
            "sticky_for_turn": self.local_selected,
            "inference_location": "LOCAL" if self.local_selected else "CLOUD",
            "decisions": list(self.decisions),
            "provider_failures": [item.model_dump(mode="json") for item in self.provider_failures],
        }


class QuotaFallbackPlanner:
    def __init__(self, primary: Any, local: Any) -> None:
        self.primary = primary
        self.local = local

    async def create_plan(self, *, route_state: TurnChatRouteState, **kwargs: Any) -> Any:
        if route_state.local_selected:
            return await self._run_local(route_state=route_state, **kwargs)
        try:
            return await self.primary.create_plan(route_state=route_state, **kwargs)
        except PlannerTemporarilyUnavailable as exc:
            if not route_state.select_local_after_quota(exc.diagnostic, stage="PLANNER"):
                raise
            return await self._run_local(route_state=route_state, **kwargs)

    async def _run_local(self, *, route_state: TurnChatRouteState, **kwargs: Any) -> Any:
        try:
            quota_fallback = route_state.selection_stage in {"PLANNER", "SYNTHESIS"}
            with local_completion_attribution(quota_fallback=quota_fallback):
                return await self.local.create_plan(route_state=route_state, **kwargs)
        except PlannerTemporarilyUnavailable as exc:
            route_state.record_failure(exc.diagnostic)
            raise

    async def assess_topic_relevance(
        self, *, route_state: TurnChatRouteState, **kwargs: Any
    ) -> Any:
        if not all(
            hasattr(planner, "assess_topic_relevance") for planner in (self.primary, self.local)
        ):
            targets = kwargs["targets"]
            question = " ".join(target.search_query for target in targets)
            return assess_topic_evidence(question, kwargs["evidence"], targets=targets), []
        if route_state.local_selected:
            return await self._run_local_relevance(route_state=route_state, **kwargs)
        try:
            return await self.primary.assess_topic_relevance(route_state=route_state, **kwargs)
        except PlannerTemporarilyUnavailable as exc:
            if not route_state.select_local_after_quota(exc.diagnostic, stage="PLANNER"):
                raise
            return await self._run_local_relevance(route_state=route_state, **kwargs)

    async def _run_local_relevance(self, *, route_state: TurnChatRouteState, **kwargs: Any) -> Any:
        try:
            quota_fallback = route_state.selection_stage in {"PLANNER", "SYNTHESIS"}
            with local_completion_attribution(quota_fallback=quota_fallback):
                return await self.local.assess_topic_relevance(route_state=route_state, **kwargs)
        except PlannerTemporarilyUnavailable as exc:
            route_state.record_failure(exc.diagnostic)
            raise


class QuotaFallbackSynthesizer:
    def __init__(self, primary: Any, local: Any) -> None:
        self.primary = primary
        self.local = local

    async def synthesize(self, *, route_state: TurnChatRouteState, **kwargs: Any) -> Any:
        evidence = kwargs.get("evidence")
        permitted = getattr(evidence, "permitted_synthesis_deployments", set())
        if (
            not route_state.local_selected
            and GROQ_CHAT_DEPLOYMENT_ID not in permitted
            and LOCAL_GEMMA_DEPLOYMENT_ID in permitted
        ):
            route_state.select_local_for_source_policy()
        if route_state.local_selected:
            return await self._run_local(route_state=route_state, **kwargs)
        try:
            return await self.primary.synthesize(route_state=route_state, **kwargs)
        except SynthesisTemporarilyUnavailable as exc:
            if not route_state.select_local_after_quota(exc.diagnostic, stage="SYNTHESIS"):
                raise
            return await self._run_local(route_state=route_state, **kwargs)

    async def _run_local(self, *, route_state: TurnChatRouteState, **kwargs: Any) -> Any:
        try:
            quota_fallback = route_state.selection_stage in {"PLANNER", "SYNTHESIS"}
            with local_completion_attribution(quota_fallback=quota_fallback):
                return await self.local.synthesize(route_state=route_state, **kwargs)
        except SynthesisTemporarilyUnavailable as exc:
            route_state.record_failure(exc.diagnostic)
            raise
