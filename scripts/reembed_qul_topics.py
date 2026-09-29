"""Resumable, idempotent QUL Ayah Topics backfill; failures always return a non-zero status."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import NoReturn

from islamic_research_ai.ingestion.qul_topics import QulTopicReembedding
from islamic_research_ai.logging import configure_logging
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourcePolicy
from islamic_research_ai.settings import ROOT, load_settings


async def run(*, batch_size: int, dry_run: bool) -> dict[str, object]:
    settings = load_settings()
    if settings.cloudflare_api_token is None or settings.cloudflare_account_id is None:
        raise RuntimeError("Cloudflare credentials are required for QUL topic re-embedding")
    repository = PostgresRepository(settings.database_url)
    embeddings = CloudflareEmbeddingClient(
        api_token=settings.cloudflare_api_token,
        account_id=settings.cloudflare_account_id,
        model=settings.cloudflare_embedding_model,
    )
    policy = SourcePolicy(
        settings.source_policy_path,
        environment=settings.app_environment,
        purpose=settings.development_purpose,
        override_acknowledged=settings.project_owner_risk_override_ack,
    )
    await repository.open()
    try:
        for migration in (
            "001_initial.sql",
            "002_archive_pre_v21_contract_records.sql",
            "003_redact_qf_tafsir_source_text.sql",
            "004_conversation_history.sql",
            "005_qul_tafsir.sql",
            "006_turn_trace_events.sql",
            "007_gate_diagnostics.sql",
            "008_enriched_quran_embeddings.sql",
            "009_qul_ayah_topics.sql",
        ):
            await repository.migrate(ROOT / "migrations" / migration)
        return await QulTopicReembedding(
            settings=settings,
            repository=repository,
            policy=policy,
            embeddings=embeddings,
            batch_size=batch_size,
        ).run(dry_run=dry_run)
    finally:
        await embeddings.close()
        await repository.close()


def _failure(message: str) -> NoReturn:
    print(f"QUL topic backfill failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        _failure("--batch-size must be positive")
    configure_logging()
    try:
        summary = asyncio.run(run(batch_size=args.batch_size, dry_run=args.dry_run))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _failure(f"{type(exc).__name__}: {exc}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
