from __future__ import annotations

import re
import time
from copy import deepcopy
from typing import TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from islamic_research_ai.contracts import EffectiveDeploymentIdentityV1, ProviderCompletion
from islamic_research_ai.ports import (
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)

T = TypeVar("T", bound=BaseModel)


def _safe_error_detail(value: str) -> str:
    value = re.sub(r"organization `[^`]+`", "organization `[redacted]`", value)
    value = re.sub(r"https://console\.groq\.com/\S+", "[Groq billing page]", value)
    return value[:800]


def _retry_after_ms(response: httpx.Response, detail: str) -> int | None:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return max(0, round(float(header) * 1000))
        except ValueError:
            pass
    match = re.search(
        r"Please try again in\s+(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?P<seconds>\d+(?:\.\d+)?)s",
        detail,
        re.IGNORECASE,
    )
    if match is None:
        return None
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))
    return max(0, round((minutes * 60 + seconds) * 1000))


def _rate_limit_reset_ms(response: httpx.Response, header: str) -> int | None:
    """Parse Groq's compact duration headers without retaining raw headers."""

    value = response.headers.get(header)
    if value is None:
        return None
    duration = value.strip()
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)(ms|h|m|s)", duration))
    if not matches or "".join(match.group(0) for match in matches) != duration:
        return None
    multipliers = {"h": 3_600_000, "m": 60_000, "s": 1_000, "ms": 1}
    return max(
        0,
        round(sum(float(match.group(1)) * multipliers[match.group(2)] for match in matches)),
    )


def _quota_tokens(detail: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(
        r"Limit\s+(?P<limit>\d+),\s+Used\s+(?P<used>\d+),\s+Requested\s+(?P<requested>\d+)",
        detail,
        re.IGNORECASE,
    )
    if match is None:
        return None, None, None
    return (
        int(match.group("limit")),
        int(match.group("used")),
        int(match.group("requested")),
    )


def _availability_error(
    *,
    response: httpx.Response | None,
    category: str,
    safe_detail: str,
) -> ProviderUnavailableError:
    quota_limit, quota_used, quota_requested = _quota_tokens(safe_detail)
    resolved_category = "QUOTA_EXHAUSTED" if quota_limit is not None else category
    return ProviderUnavailableError(
        ProviderFailureDiagnostic(
            provider_id="groq",
            category=resolved_category,
            http_status=response.status_code if response is not None else None,
            retry_after_ms=(
                _retry_after_ms(response, safe_detail) if response is not None else None
            ),
            rate_limit_reset_tokens_ms=(
                _rate_limit_reset_ms(response, "x-ratelimit-reset-tokens")
                if response is not None
                else None
            ),
            quota_limit_tokens=quota_limit,
            quota_used_tokens=quota_used,
            quota_requested_tokens=quota_requested,
            request_id=response.headers.get("x-request-id") if response is not None else None,
            safe_detail=safe_detail,
        )
    )


def _inline_local_refs(schema: dict[str, object]) -> dict[str, object]:
    """Inline Pydantic $defs for Groq's stricter JSON-schema transport dialect."""
    source = deepcopy(schema)
    definitions = source.get("$defs", {})

    def expand(node: object) -> object:
        if isinstance(node, list):
            return [expand(item) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            key = reference.removeprefix("#/$defs/")
            target = definitions.get(key)
            if not isinstance(target, dict):
                raise GroqError(f"Unresolved local JSON schema reference: {reference}")
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return expand(merged)
        expanded_node = {
            key: expand(value)
            for key, value in node.items()
            # Groq's strict-schema dialect does not accept JSON Schema's
            # ``uniqueItems`` keyword. Pydantic still enforces set semantics
            # when the returned JSON is validated locally.
            # Titles/descriptions/default annotations do not affect validation
            # and are omitted to conserve the bounded free-route token budget.
            if key
            not in {
                "$defs",
                "default",
                "description",
                "discriminator",
                "title",
                "uniqueItems",
            }
        }
        alternatives = expanded_node.get("anyOf")
        if isinstance(alternatives, list):
            flattened: list[object] = []
            for alternative in alternatives:
                if isinstance(alternative, dict) and set(alternative) in (
                    {"oneOf"},
                    {"anyOf"},
                ):
                    nested = alternative.get("oneOf", alternative.get("anyOf"))
                    flattened.extend(nested if isinstance(nested, list) else [alternative])
                else:
                    flattened.append(alternative)
            expanded_node["anyOf"] = flattened
        if "oneOf" in expanded_node and "anyOf" not in expanded_node:
            expanded_node["anyOf"] = expanded_node.pop("oneOf")
        return expanded_node

    expanded = expand(source)
    if not isinstance(expanded, dict):
        raise GroqError("Expanded response schema is not an object")
    return expanded


class GroqError(RuntimeError):
    pass


class GroqStructuredClient:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        base_url: str,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.deployment_id = f"groq:{model}:zdr"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        *,
        response_model: type[T],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1800,
    ) -> tuple[T, ProviderCompletion]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "reasoning_effort": "low",
            "max_completion_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _inline_local_refs(response_model.model_json_schema()),
                },
            },
        }
        started = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise _availability_error(
                response=None,
                category="TIMEOUT",
                safe_detail="The Groq request timed out.",
            ) from exc
        except httpx.RequestError as exc:
            raise _availability_error(
                response=None,
                category="UNAVAILABLE",
                safe_detail="The Groq network request failed.",
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if response.is_error:
            error_detail = ""
            try:
                error = response.json().get("error", {})
                if isinstance(error, dict):
                    error_detail = _safe_error_detail(str(error.get("message", "")))
            except (TypeError, ValueError):
                pass
            if response.status_code == 400 and any(
                marker in error_detail
                for marker in ("generate JSON", "validate JSON", "expected schema")
            ):
                raise StructuredOutputValidationError(
                    "Groq generated JSON that did not satisfy the structured-output schema"
                )
            if response.status_code == 429:
                raise _availability_error(
                    response=response,
                    category="RATE_LIMITED",
                    safe_detail=error_detail or "Groq rate limit or quota exhausted.",
                )
            if response.status_code in {408, 500, 502, 503, 504}:
                raise _availability_error(
                    response=response,
                    category=("TIMEOUT" if response.status_code == 408 else "UNAVAILABLE"),
                    safe_detail=error_detail or "Groq is temporarily unavailable.",
                )
            raise GroqError(
                f"Groq request failed with HTTP {response.status_code}"
                + (f" ({error_detail})" if error_detail else "")
            )
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = response_model.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StructuredOutputValidationError(
                "Groq returned an invalid structured completion"
            ) from exc
        usage = body.get("usage", {})

        def _integer_header(name: str) -> int | None:
            value = response.headers.get(name)
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None

        resolved_model = body.get("model", self.model)
        metadata = ProviderCompletion(
            effective_deployment=EffectiveDeploymentIdentityV1(
                deployment_id=self.deployment_id,
                gateway_provider_id="groq",
                effective_provider_id="groq",
                model_id=resolved_model,
                resolved_model_version_or_name=resolved_model,
                region=response.headers.get("x-groq-region"),
                subprocessor_ids=set(),
                zero_data_retention_enforced=True,
                provider_training_allowed=False,
                route_policy_id="groq-direct-free-zdr-v1",
            ),
            request_id=response.headers.get("x-request-id") or body.get("id"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            remaining_requests=_integer_header("x-ratelimit-remaining-requests"),
            remaining_tokens=_integer_header("x-ratelimit-remaining-tokens"),
            billed_cost_usd=None,
            paid_route=False,
            fallback_used=False,
        )
        if metadata.effective_deployment.model_id != self.model:
            raise GroqError("Groq resolved to an unexpected model")
        return parsed, metadata
