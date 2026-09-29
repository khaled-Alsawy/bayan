from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from islamic_research_ai.contracts import EffectiveDeploymentIdentityV1, ProviderCompletion
from islamic_research_ai.ingestion.enriched_quran import (
    DEFAULT_TAFSIR_EXCERPT_CHARACTERS,
    QuranEnrichedReembedding,
    build_enriched_document,
)
from islamic_research_ai.retrieval.relevance import lexical_relevance_score
from islamic_research_ai.settings import Settings


def test_enriched_vector_is_opt_in_at_query_time() -> None:
    assert Settings.model_fields["quran_embedding_variant"].default == "base"


def test_enriched_document_bounds_tafsir_and_lifts_semantic_context_over_short_garbage() -> None:
    short_unrelated = build_enriched_document(
        document_id=uuid4(),
        arabic_context="واكيد كيدا",
        translation_context="They are plotting.",
        tafsir_excerpt=None,
    )
    parent_care = build_enriched_document(
        document_id=uuid4(),
        arabic_context="وقضى ربك ألا تعبدوا إلا إياه وبالوالدين إحسانا",
        translation_context="Your Lord has commanded kindness and care for parents.",
        tafsir_excerpt="شرح موجز عن الإحسان بالوالدين " * 100,
    )

    assert lexical_relevance_score("care for parents", parent_care.text) > lexical_relevance_score(
        "care for parents", short_unrelated.text
    )
    tafsir_part = parent_care.text.split("Tafsir context: ", maxsplit=1)[1]
    assert len(tafsir_part) <= DEFAULT_TAFSIR_EXCERPT_CHARACTERS
    assert parent_care.tafsir_covered is True


def test_enriched_document_hash_is_the_idempotent_backfill_checkpoint() -> None:
    document_id = uuid4()
    first = build_enriched_document(
        document_id=document_id,
        arabic_context="نص عربي",
        translation_context="English text",
        tafsir_excerpt=None,
    )
    completed = build_enriched_document(
        document_id=document_id,
        arabic_context="نص عربي",
        translation_context="English text",
        tafsir_excerpt=None,
        already_embedded=True,
        stored_sha256=first.sha256,
    )
    changed = build_enriched_document(
        document_id=document_id,
        arabic_context="نص عربي مختلف",
        translation_context="English text",
        tafsir_excerpt=None,
        already_embedded=True,
        stored_sha256=first.sha256,
    )

    assert first.needs_embedding is True
    assert completed.needs_embedding is False
    assert changed.needs_embedding is True


@pytest.mark.asyncio
async def test_enriched_reembedding_dry_run_reports_coverage_without_provider_calls() -> None:
    document_id = uuid4()

    class Repository:
        async def enriched_quran_document_inputs(self, **_: object) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "arabic_context": "نص عربي",
                    "translation_context": "English text",
                    "tafsir_excerpt": None,
                    "enriched_embedding_present": False,
                    "enriched_document_sha256": None,
                }
            ]

    runner = QuranEnrichedReembedding(
        repository=Repository(),  # type: ignore[arg-type]
        embeddings=None,  # type: ignore[arg-type]
        policy=None,  # type: ignore[arg-type]
    )

    summary = await runner.run(dry_run=True)

    assert summary["document_count"] == 1
    assert summary["verse_context_only_document_count"] == 1
    assert summary["pending_document_count"] == 1
    assert summary["dry_run"] is True


@pytest.mark.asyncio
async def test_enriched_reembedding_commits_a_durable_idempotent_checkpoint() -> None:
    document_id = uuid4()

    class Repository:
        def __init__(self) -> None:
            self.embedded = False
            self.sha256: str | None = None
            self.stored_batches: list[list[tuple[object, ...]]] = []
            self.activations: list[object] = []

        async def enriched_quran_document_inputs(self, **_: object) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "arabic_context": "نص عربي",
                    "translation_context": "English text",
                    "tafsir_excerpt": None,
                    "enriched_embedding_present": self.embedded,
                    "enriched_document_sha256": self.sha256,
                }
            ]

        async def store_enriched_embeddings(
            self, rows: list[tuple[object, ...]]
        ) -> None:
            self.stored_batches.append(rows)
            self.embedded = True
            self.sha256 = str(rows[0][5])

        async def enriched_quran_document_counts(self) -> tuple[int, int, int]:
            return 1, int(self.embedded), 0

        async def record_activation(self, decision: object) -> None:
            self.activations.append(decision)

    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            self.calls += 1
            return [[0.0] for _ in texts], ProviderCompletion(
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
                request_id="test",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                billed_cost_usd=Decimal("0"),
                paid_route=False,
                fallback_used=False,
            )

    class Policy:
        def authorize(self, *_: object, **__: object) -> object:
            return object()

    repository = Repository()
    embeddings = Embeddings()
    first = await QuranEnrichedReembedding(
        repository=repository,  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
        policy=Policy(),  # type: ignore[arg-type]
    ).run()
    second = await QuranEnrichedReembedding(
        repository=repository,  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
        policy=Policy(),  # type: ignore[arg-type]
    ).run()

    assert first["provider_call_count"] == 1
    assert second["pending_document_count"] == 0
    assert second["provider_call_count"] == 0
    assert embeddings.calls == 1
    assert len(repository.stored_batches) == 1
