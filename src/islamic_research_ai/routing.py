from __future__ import annotations

from dataclasses import dataclass

from islamic_research_ai.contracts import ProviderCompletion

GROQ_ROUTE_POLICY_ID = "groq-direct-free-zdr-v1"
GROQ_CHAT_DEPLOYMENT_ID = "groq:openai/gpt-oss-20b:zdr"


@dataclass(frozen=True)
class GroqRoleDeployment:
    """WP-1 (V3 plan): one role's approved Groq deployment identity.

    Model identity is data (see `config/chat_deployments.json`); the route itself -- ZDR, no
    provider training, no subprocessors, no paid route, no fallback -- is still asserted in
    code by `ProviderRoutePolicy.require_configured_route` and cannot be changed by config.
    """

    role: str
    model_id: str

    @property
    def deployment_id(self) -> str:
        return f"groq:{self.model_id}:zdr"


# Back-compat default: the route every existing caller asserted before WP-1 introduced a
# role map. `require_groq_direct_free` still checks exactly this.
DEFAULT_GROQ_ROLE_DEPLOYMENT = GroqRoleDeployment(role="default", model_id="openai/gpt-oss-20b")
LOCAL_GEMMA_REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
LOCAL_GEMMA_DEPLOYMENT_ID = "local:gemma-4-26b-a4b-it:mlx-lm:" + LOCAL_GEMMA_REVISION
LOCAL_GEMMA_RESOLVED_MODEL = (
    "mlx-community/gemma-4-26b-a4b-it-4bit@" + LOCAL_GEMMA_REVISION + ":mlx-4bit-affine-g64-router8"
)
LOCAL_GEMMA_ROUTE_POLICY_ID = "local-loopback-gemma4-q4-v22"


class ProviderRouteError(PermissionError):
    pass


class ProviderRoutePolicy:
    @staticmethod
    def require_groq_direct_free(completion: ProviderCompletion) -> None:
        """Back-compat alias for `require_configured_route` with the default role."""

        ProviderRoutePolicy.require_configured_route(
            completion, role_deployment=DEFAULT_GROQ_ROLE_DEPLOYMENT
        )

    @staticmethod
    def require_configured_route(
        completion: ProviderCompletion, *, role_deployment: GroqRoleDeployment
    ) -> None:
        """WP-1 (V3 plan): assert the Groq route for one role's configured model.

        Everything about the route except model identity is still hard-asserted here, not
        read from config: gateway/provider must be Groq, ZDR must be enforced, no provider
        training, no subprocessors, no paid route, no fallback. Only `role_deployment.model_id`
        (and the deployment id it implies) comes from `config/chat_deployments.json`.
        """

        route = completion.effective_deployment
        if (
            route.deployment_id != role_deployment.deployment_id
            or route.gateway_provider_id != "groq"
            or route.effective_provider_id != "groq"
            or route.model_id != role_deployment.model_id
            or route.route_policy_id != GROQ_ROUTE_POLICY_ID
            or not route.zero_data_retention_enforced
            or route.provider_training_allowed
            or route.subprocessor_ids
        ):
            raise ProviderRouteError("Unexpected Groq effective deployment route")
        ProviderRoutePolicy._require_free_no_fallback(completion)

    @staticmethod
    def require_local_gemma_pinned_free(completion: ProviderCompletion) -> None:
        route = completion.effective_deployment
        if (
            route.deployment_id != LOCAL_GEMMA_DEPLOYMENT_ID
            or route.gateway_provider_id != "local"
            or route.effective_provider_id != "local"
            or route.model_id != "gemma-4-26b-a4b-it"
            or route.resolved_model_version_or_name != LOCAL_GEMMA_RESOLVED_MODEL
            or route.region != "loopback"
            or route.route_policy_id != LOCAL_GEMMA_ROUTE_POLICY_ID
            or not route.zero_data_retention_enforced
            or route.provider_training_allowed
            or route.subprocessor_ids
        ):
            raise ProviderRouteError("Unexpected local Gemma effective deployment route")
        if completion.paid_route:
            raise ProviderRouteError("A paid local route is prohibited")
        if completion.billed_cost_usd is not None and completion.billed_cost_usd > 0:
            raise ProviderRouteError("A billed local route is prohibited")

    @staticmethod
    def require_cloudflare_bge_m3_direct_free(completion: ProviderCompletion) -> None:
        route = completion.effective_deployment
        if (
            route.deployment_id != "cloudflare-workers-ai:direct:@cf/baai/bge-m3"
            or route.gateway_provider_id != "cloudflare"
            or route.effective_provider_id != "cloudflare"
            or route.model_id != "@cf/baai/bge-m3"
            or route.route_policy_id != "cloudflare-workers-ai-direct-free-v1"
            or route.provider_training_allowed
            or route.subprocessor_ids
        ):
            raise ProviderRouteError("Unexpected Cloudflare effective deployment route")
        ProviderRoutePolicy._require_free_no_fallback(completion)

    @staticmethod
    def require_cloudflare_bge_reranker_direct_free(completion: ProviderCompletion) -> None:
        route = completion.effective_deployment
        if (
            route.deployment_id != "cloudflare-workers-ai:direct:@cf/baai/bge-reranker-base"
            or route.gateway_provider_id != "cloudflare"
            or route.effective_provider_id != "cloudflare"
            or route.model_id != "@cf/baai/bge-reranker-base"
            or route.route_policy_id != "cloudflare-workers-ai-direct-free-v1"
            or route.provider_training_allowed
            or route.subprocessor_ids
        ):
            raise ProviderRouteError("Unexpected Cloudflare reranker route")
        ProviderRoutePolicy._require_free_no_fallback(completion)

    @staticmethod
    def _require_free_no_fallback(completion: ProviderCompletion) -> None:
        if completion.paid_route or completion.fallback_used:
            raise ProviderRouteError("Paid or fallback provider routes are prohibited")
        if completion.billed_cost_usd is not None and completion.billed_cost_usd > 0:
            raise ProviderRouteError("A billed provider route is prohibited")
