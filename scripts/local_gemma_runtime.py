from __future__ import annotations

import argparse
import asyncio
import json
import os

from islamic_research_ai.providers.local_mlx import LocalMlxStructuredClient
from islamic_research_ai.settings import ROOT, load_settings

RUNTIME_PYTHON = ROOT / ".runtime/mlx-lm-v0.31.3/bin/python"


async def readiness() -> int:
    settings = load_settings()
    client = LocalMlxStructuredClient(
        base_url=settings.local_chat_base_url,
        manifest_path=settings.local_model_manifest_path,
        model_root=settings.local_model_root,
        timeout=settings.local_chat_timeout_seconds,
        health_timeout=settings.local_chat_health_timeout_seconds,
        max_concurrency=settings.local_chat_max_concurrency,
        max_prompt_characters=settings.local_chat_max_prompt_characters,
    )
    try:
        result = await client.readiness(run_schema_probe=True)
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0 if result.ready else 1
    finally:
        await client.close()


def verify() -> int:
    settings = load_settings()
    client = LocalMlxStructuredClient(
        base_url=settings.local_chat_base_url,
        manifest_path=settings.local_model_manifest_path,
        model_root=settings.local_model_root,
        timeout=settings.local_chat_timeout_seconds,
        health_timeout=settings.local_chat_health_timeout_seconds,
        max_concurrency=settings.local_chat_max_concurrency,
        max_prompt_characters=settings.local_chat_max_prompt_characters,
    )
    manifest = client._load_and_verify_manifest()
    print(
        json.dumps(
            {
                "verified": True,
                "artifact_repo": manifest.artifact_repo,
                "artifact_revision": manifest.artifact_revision,
                "quantization": manifest.quantization,
                "total_size_bytes": manifest.total_size_bytes,
                "file_count": len(manifest.files),
            },
            indent=2,
        )
    )
    return 0


def serve() -> None:
    settings = load_settings()
    if not RUNTIME_PYTHON.is_file():
        raise SystemExit("The pinned MLX-LM 0.31.3 environment is not installed.")
    if not settings.local_model_root.is_dir():
        raise SystemExit("The pinned local model is not present.")
    parsed_port = settings.local_chat_base_url.split(":")[-1].split("/")[0]
    command = [
        str(RUNTIME_PYTHON),
        "-m",
        "mlx_lm",
        "server",
        "--model",
        str(settings.local_model_root),
        "--host",
        "127.0.0.1",
        "--port",
        parsed_port,
        "--allowed-origins",
        "http://127.0.0.1",
        "--temp",
        "0",
        "--max-tokens",
        "2000",  # WP-1 (V3 plan), task 5: raised from 900; see local_mlx.py for the rationale.
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--prefill-step-size",
        "1024",
        "--prompt-cache-size",
        "1",
        "--prompt-cache-bytes",
        str(1024 * 1024 * 1024),
        "--chat-template-args",
        '{"enable_thinking":false}',
        "--log-level",
        "WARNING",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    os.execve(str(RUNTIME_PYTHON), command, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned local Gemma 4 fallback runtime")
    parser.add_argument("action", choices=("verify", "serve", "readiness"))
    args = parser.parse_args()
    if args.action == "verify":
        raise SystemExit(verify())
    if args.action == "readiness":
        raise SystemExit(asyncio.run(readiness()))
    serve()


if __name__ == "__main__":
    main()
