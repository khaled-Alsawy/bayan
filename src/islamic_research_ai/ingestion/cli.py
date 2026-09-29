from __future__ import annotations

import asyncio
import json

from islamic_research_ai.ingestion.qul_ingest import QulTafsirIngestion
from islamic_research_ai.ingestion.quran import QuranIngestion
from islamic_research_ai.logging import configure_logging
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourcePolicy
from islamic_research_ai.settings import ROOT, load_settings


def _build_dependencies() -> tuple[PostgresRepository, CloudflareEmbeddingClient, SourcePolicy]:
    settings = load_settings()
    if settings.cloudflare_api_token is None or settings.cloudflare_account_id is None:
        raise SystemExit("Cloudflare credentials are required for real ingestion")
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
    return repository, embeddings, policy


async def run_ingestion() -> dict[str, object]:
    settings = load_settings()
    repository, embeddings, policy = _build_dependencies()
    await repository.open()
    try:
        await repository.migrate(ROOT / "migrations/001_initial.sql")
        return await QuranIngestion(
            settings=settings,
            repository=repository,
            policy=policy,
            embeddings=embeddings,
        ).run()
    finally:
        await embeddings.close()
        await repository.close()


async def run_qul_tafsir_ingestion() -> dict[str, object]:
    """WP-0 (V3 plan): ingest the four downloaded QUL tafsir exports."""

    settings = load_settings()
    repository, embeddings, policy = _build_dependencies()
    await repository.open()
    try:
        for migration in (
            "001_initial.sql",
            "002_archive_pre_v21_contract_records.sql",
            "003_redact_qf_tafsir_source_text.sql",
            "004_conversation_history.sql",
            "005_qul_tafsir.sql",
            "006_turn_trace_events.sql",
        ):
            await repository.migrate(ROOT / "migrations" / migration)
        return await QulTafsirIngestion(
            settings=settings,
            repository=repository,
            policy=policy,
            embeddings=embeddings,
        ).run()
    finally:
        await embeddings.close()
        await repository.close()


def main() -> None:
    configure_logging()
    print(json.dumps(asyncio.run(run_ingestion()), indent=2))


def main_qul_tafsir() -> None:
    configure_logging()
    print(json.dumps(asyncio.run(run_qul_tafsir_ingestion()), indent=2))


if __name__ == "__main__":
    main()
