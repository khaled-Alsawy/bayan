from __future__ import annotations

import time

import httpx
from pydantic import SecretStr

from islamic_research_ai.contracts import EffectiveDeploymentIdentityV1, ProviderCompletion
from islamic_research_ai.ports import ProviderFailureDiagnostic, ProviderUnavailableError


class CloudflareEmbeddingError(RuntimeError):
    pass


def _retry_after_ms(response: httpx.Response) -> int | None:
    value = response.headers.get("retry-after")
    try:
        return max(0, round(float(value) * 1000)) if value is not None else None
    except ValueError:
        return None


def _availability_error(
    response: httpx.Response | None, *, category: str, safe_detail: str
) -> ProviderUnavailableError:
    return ProviderUnavailableError(
        ProviderFailureDiagnostic(
            provider_id="cloudflare",
            category=category,
            http_status=response.status_code if response is not None else None,
            retry_after_ms=_retry_after_ms(response) if response is not None else None,
            quota_limit_tokens=None,
            quota_used_tokens=None,
            quota_requested_tokens=None,
            request_id=response.headers.get("cf-ray") if response is not None else None,
            safe_detail=safe_detail,
        )
    )


class CloudflareEmbeddingClient:
    dimension = 1024

    def __init__(
        self,
        *,
        api_token: SecretStr,
        account_id: str,
        model: str,
        timeout: float = 20.0,
    ) -> None:
        self.model = model
        self.deployment_id = f"cloudflare-workers-ai:direct:{model}"
        self._client = httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            headers={"Authorization": f"Bearer {api_token.get_secret_value()}"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
        if not texts or len(texts) > 128:
            raise ValueError("Cloudflare embedding batches must contain 1..128 texts")
        started = time.perf_counter()
        try:
            response = await self._client.post(f"/{self.model}", json={"text": texts})
        except httpx.TimeoutException as exc:
            raise _availability_error(
                None,
                category="TIMEOUT",
                safe_detail="The Cloudflare embedding request timed out.",
            ) from exc
        except httpx.RequestError as exc:
            raise _availability_error(
                None,
                category="UNAVAILABLE",
                safe_detail="The Cloudflare embedding network request failed.",
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if response.is_error:
            error_detail = ""
            try:
                errors = response.json().get("errors", [])
                error_detail = "; ".join(
                    f"{item.get('code', 'unknown')}: {item.get('message', 'unknown')}"
                    for item in errors
                    if isinstance(item, dict)
                )
            except (TypeError, ValueError):
                pass
            if response.status_code == 429:
                raise _availability_error(
                    response,
                    category="RATE_LIMITED",
                    safe_detail="Cloudflare embedding rate limit or quota exhausted.",
                )
            if response.status_code in {408, 500, 502, 503, 504}:
                raise _availability_error(
                    response,
                    category=("TIMEOUT" if response.status_code == 408 else "UNAVAILABLE"),
                    safe_detail="Cloudflare embedding is temporarily unavailable.",
                )
            raise CloudflareEmbeddingError(
                f"Cloudflare embedding failed with HTTP {response.status_code}"
                + (f" ({error_detail})" if error_detail else "")
            )
        body = response.json()
        if not body.get("success"):
            raise CloudflareEmbeddingError("Cloudflare embedding response was unsuccessful")
        result = body.get("result", {})
        vectors = result.get("data") if isinstance(result, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise CloudflareEmbeddingError("Cloudflare returned an unexpected embedding count")
        if any(not isinstance(v, list) or len(v) != self.dimension for v in vectors):
            raise CloudflareEmbeddingError("Cloudflare returned an unexpected embedding dimension")
        vectors = [[float(component) for component in vector] for vector in vectors]
        metadata = ProviderCompletion(
            effective_deployment=EffectiveDeploymentIdentityV1(
                deployment_id=self.deployment_id,
                gateway_provider_id="cloudflare",
                effective_provider_id="cloudflare",
                model_id=self.model,
                resolved_model_version_or_name=self.model,
                region=None,
                subprocessor_ids=set(),
                zero_data_retention_enforced=False,
                provider_training_allowed=False,
                route_policy_id="cloudflare-workers-ai-direct-free-v1",
            ),
            request_id=response.headers.get("cf-ray"),
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            billed_cost_usd=None,
            paid_route=False,
            fallback_used=False,
        )
        return vectors, metadata


class CloudflareRerankerClient:
    """Direct Workers AI bge-reranker-base client.

    The model returns scores only. It never gets to generate or alter canonical text; the
    calling retrieval layer uses the score to order opaque candidates before hydration.
    """

    def __init__(
        self, *, api_token: SecretStr, account_id: str, model: str, timeout: float = 20.0
    ) -> None:
        self.model = model
        self.deployment_id = f"cloudflare-workers-ai:direct:{model}"
        self._client = httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            headers={"Authorization": f"Bearer {api_token.get_secret_value()}"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank(
        self, *, query: str, passages: list[str]
    ) -> tuple[list[float], ProviderCompletion]:
        if not passages or len(passages) > 32:
            raise ValueError("Cloudflare reranking batches must contain 1..32 passages")
        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"/{self.model}",
                json={"query": query, "contexts": [{"text": passage} for passage in passages]},
            )
        except httpx.TimeoutException as exc:
            raise _availability_error(
                None, category="TIMEOUT", safe_detail="The Cloudflare reranking request timed out."
            ) from exc
        except httpx.RequestError as exc:
            raise _availability_error(
                None,
                category="UNAVAILABLE",
                safe_detail="The Cloudflare reranking network request failed.",
            ) from exc
        if response.is_error or not response.json().get("success"):
            if response.status_code == 429:
                raise _availability_error(
                    response,
                    category="RATE_LIMITED",
                    safe_detail="Cloudflare reranking rate limit or quota exhausted.",
                )
            raise CloudflareEmbeddingError(
                f"Cloudflare reranking failed with HTTP {response.status_code}"
            )
        body = response.json().get("result", {})
        values = body.get("response", body.get("data", body)) if isinstance(body, dict) else body
        if not isinstance(values, list) or len(values) != len(passages):
            raise CloudflareEmbeddingError("Cloudflare returned an unexpected reranking count")
        if all(isinstance(value, dict) and isinstance(value.get("id"), int) for value in values):
            values = sorted(values, key=lambda value: value["id"])
        scores: list[float] = []
        for value in values:
            score = value.get("score") if isinstance(value, dict) else value
            if not isinstance(score, (int, float)):
                raise CloudflareEmbeddingError("Cloudflare returned an invalid reranking score")
            scores.append(float(score))
        return scores, ProviderCompletion(
            effective_deployment=EffectiveDeploymentIdentityV1(
                deployment_id=self.deployment_id,
                gateway_provider_id="cloudflare",
                effective_provider_id="cloudflare",
                model_id=self.model,
                resolved_model_version_or_name=self.model,
                region=None,
                subprocessor_ids=set(),
                zero_data_retention_enforced=False,
                provider_training_allowed=False,
                route_policy_id="cloudflare-workers-ai-direct-free-v1",
            ),
            request_id=response.headers.get("cf-ray"),
            input_tokens=None,
            output_tokens=None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            billed_cost_usd=None,
            paid_route=False,
            fallback_used=False,
        )
