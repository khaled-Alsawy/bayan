from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict

from islamic_research_ai.contracts import EffectiveDeploymentIdentityV1, ProviderCompletion
from islamic_research_ai.local_inference import local_quota_fallback_used
from islamic_research_ai.ports import (
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from islamic_research_ai.routing import (
    LOCAL_GEMMA_DEPLOYMENT_ID,
    LOCAL_GEMMA_RESOLVED_MODEL,
    LOCAL_GEMMA_ROUTE_POLICY_ID,
)
from islamic_research_ai.tracing import _TRACE_RECORDER

T = TypeVar("T", bound=BaseModel)


def extract_first_json_object(content: str) -> str:
    """Return the first balanced top-level JSON object without rewriting its contents."""

    start = content.find("{")
    if start < 0:
        raise ValueError("No JSON object found in local-model completion")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        character = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    raise ValueError("No balanced JSON object found in local-model completion")


class ManifestFileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int
    sha256: str


class LocalRuntimeQualificationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str
    architecture: Literal["arm64"]
    soc: str
    unified_memory_bytes: int
    bind_host: Literal["127.0.0.1"]
    max_concurrency: Literal[1]
    max_output_tokens: int
    max_prompt_characters: int
    runtime_network_access_disabled: bool
    runtime_telemetry_disabled: bool


class LocalModelManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    artifact_repo: str
    artifact_url: str
    artifact_revision: str
    upstream_repo: str
    upstream_revision: str
    quantization: str
    runtime: str
    runtime_version: str
    runtime_source_url: str
    runtime_revision: str
    runtime_packages: dict[str, str]
    qualification: LocalRuntimeQualificationV1
    license: str
    license_terms_version: str
    license_evidence_url: str
    total_size_bytes: int
    artifact_tree_sha256: str
    files: list[ManifestFileV1]


class LocalReadinessV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    manifest_verified: bool
    loopback_only: bool
    server_reachable: bool
    schema_probe_passed: bool
    safe_detail: str


class _SchemaProbeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["OK"]
    value: int


class LocalPromptLimitError(RuntimeError):
    pass


class LocalRuntimeRequestError(RuntimeError):
    """A non-transient local request/config/auth/route defect."""

    def __init__(self, *, http_status: int) -> None:
        self.http_status = http_status
        super().__init__(f"Local runtime rejected the request with HTTP {http_status}")


def _local_failure(
    category: Literal["RATE_LIMITED", "UNAVAILABLE", "TIMEOUT"],
    detail: str,
    *,
    http_status: int | None = None,
    retry_after_ms: int | None = None,
    request_id: str | None = None,
) -> ProviderUnavailableError:
    return ProviderUnavailableError(
        ProviderFailureDiagnostic(
            provider_id="local",
            category=category,
            http_status=http_status,
            retry_after_ms=retry_after_ms,
            quota_limit_tokens=None,
            quota_used_tokens=None,
            quota_requested_tokens=None,
            request_id=request_id,
            safe_detail=detail,
        )
    )


class LocalMlxStructuredClient:
    """Loopback-only MLX-LM adapter with application-owned schema validation."""

    def __init__(
        self,
        *,
        base_url: str,
        manifest_path: Path,
        model_root: Path,
        timeout: float = 90.0,
        health_timeout: float = 5.0,
        max_concurrency: int = 1,
        readiness_cache_seconds: float = 300.0,
        max_prompt_characters: int = 100_000,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Local MLX endpoint must use HTTP on a loopback host")
        if max_concurrency != 1:
            raise ValueError("The qualified local runtime permits exactly one concurrent request")
        self.base_url = base_url.rstrip("/")
        self.manifest_path = manifest_path
        self.model_root = model_root
        self._timeout = timeout
        self._health_timeout = health_timeout
        self._readiness_cache_seconds = readiness_cache_seconds
        self._max_prompt_characters = max_prompt_characters
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
            headers={"User-Agent": "islamic-research-ai-local-fallback/2.2"},
        )
        self._manifest_verified = False
        self.last_readiness: LocalReadinessV1 | None = None
        self._readiness_checked_at: float | None = None

    async def close(self) -> None:
        await self._client.aclose()

    def _load_and_verify_manifest(self) -> LocalModelManifestV1:
        try:
            manifest = LocalModelManifestV1.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("The pinned local-model manifest is unavailable or invalid") from exc
        expected_identity = {
            "artifact_repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "artifact_url": (
                "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/"
                "tree/0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
            ),
            "artifact_revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
            "upstream_repo": "google/gemma-4-26b-a4b-it",
            "upstream_revision": "20da991ab4afab98e8f910c4a2e8f4fbefc404ad",
            "quantization": "mlx-4bit-affine-g64-router-projections-8bit",
            "runtime": "mlx-lm",
            "runtime_version": "0.31.3",
            "runtime_source_url": "https://github.com/ml-explore/mlx-lm",
            "runtime_revision": "v0.31.3",
            "license": "Apache-2.0",
            "license_terms_version": "Apache-2.0-January-2004",
        }
        if any(getattr(manifest, key) != value for key, value in expected_identity.items()):
            raise RuntimeError("The local-model manifest identity is not the approved pin")
        expected_packages = {
            "huggingface-hub": "1.31.0",
            "mlx": "0.32.2",
            "mlx-lm": "0.31.3",
            "mlx-metal": "0.32.2",
            "transformers": "5.17.0",
        }
        if manifest.runtime_packages != expected_packages:
            raise RuntimeError("The local-model runtime package set is not the approved pin")
        manifest_paths = [entry.path for entry in manifest.files]
        if manifest_paths != sorted(set(manifest_paths)):
            raise RuntimeError("The local-model manifest paths must be unique and sorted")
        total = 0
        tree_digest = hashlib.sha256()
        for entry in manifest.files:
            path = (self.model_root / entry.path).resolve()
            root = self.model_root.resolve()
            if root not in path.parents:
                raise RuntimeError("The local-model manifest contains an unsafe file path")
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise RuntimeError("A pinned local-model file is unavailable") from exc
            if size != entry.size_bytes:
                raise RuntimeError("A pinned local-model file size does not match the manifest")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry.sha256:
                raise RuntimeError("A pinned local-model checksum does not match the manifest")
            total += size
            tree_digest.update(f"{entry.path}\0{size}\0{entry.sha256}\n".encode())
        if total != manifest.total_size_bytes:
            raise RuntimeError("The pinned local-model total size does not match the manifest")
        if tree_digest.hexdigest() != manifest.artifact_tree_sha256:
            raise RuntimeError("The pinned local-model tree checksum does not match the manifest")
        self._manifest_verified = True
        return manifest

    async def readiness(
        self, *, run_schema_probe: bool = True, force: bool = False
    ) -> LocalReadinessV1:
        now = time.monotonic()
        if (
            not force
            and self.last_readiness is not None
            and self._readiness_checked_at is not None
            and now - self._readiness_checked_at < self._readiness_cache_seconds
            and (not run_schema_probe or self.last_readiness.schema_probe_passed)
        ):
            return self.last_readiness
        server_reachable = False
        try:
            if not self._manifest_verified:
                await asyncio.to_thread(self._load_and_verify_manifest)
            response = await self._client.get("/models", timeout=self._health_timeout)
            response.raise_for_status()
            server_reachable = True
            schema_probe_passed = False
            if run_schema_probe:
                probe, _ = await self.complete(
                    response_model=_SchemaProbeV1,
                    schema_name="local_readiness_probe_v1",
                    system_prompt="Return only JSON matching the requested fields.",
                    user_prompt='Return {"status":"OK","value":1}.',
                    max_tokens=40,
                )
                schema_probe_passed = probe.status == "OK" and probe.value == 1
            result = LocalReadinessV1(
                ready=(not run_schema_probe or schema_probe_passed),
                manifest_verified=True,
                loopback_only=True,
                server_reachable=True,
                schema_probe_passed=schema_probe_passed,
                safe_detail="Pinned local Gemma runtime is ready.",
            )
        except Exception as exc:
            result = LocalReadinessV1(
                ready=False,
                manifest_verified=self._manifest_verified,
                loopback_only=True,
                server_reachable=server_reachable,
                schema_probe_passed=False,
                safe_detail=f"Local readiness failed safely ({type(exc).__name__}).",
            )
        self.last_readiness = result
        self._readiness_checked_at = now
        return result

    async def complete(
        self,
        *,
        response_model: type[T],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1800,
    ) -> tuple[T, ProviderCompletion]:
        if len(system_prompt) + len(user_prompt) > self._max_prompt_characters:
            raise LocalPromptLimitError("The local-model prompt exceeds the qualified POC limit")
        if not self._manifest_verified:
            try:
                await asyncio.to_thread(self._load_and_verify_manifest)
            except RuntimeError as exc:
                raise _local_failure("UNAVAILABLE", str(exc)) from exc
        schema = response_model.model_json_schema()
        schema_instruction = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        payload = {
            # MLX-LM maps this private transport key to the CLI-pinned local path.
            # The runtime value is never exposed as domain/provider identity.
            "model": "default_model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\nReturn exactly one JSON object. "
                        + "Do not use Markdown or reasoning tags. "
                        + "The JSON must satisfy this JSON Schema: "
                        + schema_instruction
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            # WP-1 (V3 plan), task 5: raised 900 -> 2000. Direct measurements show 60.8,
            # 67.9, and 65.4 tok/s for 256, 1024, and 1555 generated tokens respectively;
            # a 2,000-token decode is therefore about 31 seconds. The 90-second per-call
            # timeout retains substantial cold-start/contended-runtime headroom while keeping
            # a stuck local fallback bounded.
            "max_tokens": min(max_tokens, 2000),
            "stream": False,
        }
        started = time.perf_counter()
        try:
            async with self._semaphore:
                response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise _local_failure("TIMEOUT", "The loopback local-model request timed out.") from exc
        except httpx.RequestError as exc:
            raise _local_failure(
                "UNAVAILABLE", "The loopback local-model runtime is unavailable."
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if response.is_error:
            status = response.status_code
            request_id = response.headers.get("x-request-id")
            retry_after_ms = None
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    retry_after_ms = max(0, round(float(retry_after) * 1000))
                except ValueError:
                    pass
            if status == 408:
                raise _local_failure(
                    "TIMEOUT",
                    "The loopback local-model runtime timed out the request.",
                    http_status=status,
                    request_id=request_id,
                )
            if status == 429:
                raise _local_failure(
                    "RATE_LIMITED",
                    "The loopback local-model runtime is temporarily saturated.",
                    http_status=status,
                    retry_after_ms=retry_after_ms,
                    request_id=request_id,
                )
            if 500 <= status <= 599:
                raise _local_failure(
                    "UNAVAILABLE",
                    f"The loopback local-model runtime returned transient HTTP {status}.",
                    http_status=status,
                    request_id=request_id,
                )
            raise LocalRuntimeRequestError(http_status=status)
        content: object = None
        body: dict[str, object] = {}
        try:
            body = response.json()
            choice = body["choices"][0]  # type: ignore[index]
            content = choice["message"]["content"]  # type: ignore[index]
            if not isinstance(content, str):
                raise TypeError
            cleaned = extract_first_json_object(content)
            parsed = response_model.model_validate_json(cleaned)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if (
                isinstance(content, str)
                and os.environ.get("APP_ENVIRONMENT", "").casefold() == "development"
            ):
                recorder = _TRACE_RECORDER.get()
                if recorder is not None:
                    recorder.emit(
                        "synthesis",
                        "rejected",
                        "Local Gemma returned an invalid structured completion",
                        detail={
                            "raw_completion_first_200": content[:200],
                            "raw_completion_last_80": content[-80:],
                        },
                    )
            safe_shape = "content_unavailable"
            if isinstance(content, str):
                stripped = content.strip()
                safe_shape = (
                    f"chars={len(content)},starts_object={stripped.startswith('{')},"
                    f"ends_object={stripped.endswith('}')},"
                    f"brace_delta={content.count('{') - content.count('}')}"
                )
            finish_reason = None
            choices = body.get("choices") if isinstance(body, dict) else None
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                finish_reason = choices[0].get("finish_reason")
            raise StructuredOutputValidationError(
                "Local Gemma returned an invalid structured completion "
                f"(finish_reason={finish_reason!s}; {safe_shape})"
            ) from exc
        usage = body.get("usage", {})
        completion = ProviderCompletion(
            effective_deployment=EffectiveDeploymentIdentityV1(
                deployment_id=LOCAL_GEMMA_DEPLOYMENT_ID,
                gateway_provider_id="local",
                effective_provider_id="local",
                model_id="gemma-4-26b-a4b-it",
                resolved_model_version_or_name=LOCAL_GEMMA_RESOLVED_MODEL,
                region="loopback",
                subprocessor_ids=set(),
                zero_data_retention_enforced=True,
                provider_training_allowed=False,
                route_policy_id=LOCAL_GEMMA_ROUTE_POLICY_ID,
            ),
            request_id=None,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            billed_cost_usd=None,
            paid_route=False,
            fallback_used=local_quota_fallback_used(),
        )
        return parsed, completion
