"""Build and backfill the optional context-plus-Tafsir Quran embedding representation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from islamic_research_ai.ingestion.quran import ARABIC_ALIAS, CF_DEPLOYMENT, TRANSLATION_ALIAS
from islamic_research_ai.logging import log_event
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.rights import SourceOperation, SourcePolicy
from islamic_research_ai.routing import ProviderRoutePolicy

ENRICHMENT_VERSION = "quran-context-tafsir-v1"
TAFSIR_SEARCH_ALIAS = "qul-tafsir-almukhtasar-ar-v1"
DEFAULT_TAFSIR_EXCERPT_CHARACTERS = 480


@dataclass(frozen=True)
class EnrichedQuranDocument:
    document_id: UUID
    text: str
    sha256: str
    tafsir_covered: bool
    already_embedded: bool
    stored_sha256: str | None

    @property
    def needs_embedding(self) -> bool:
        return not self.already_embedded or self.stored_sha256 != self.sha256


def build_enriched_document(
    *,
    document_id: UUID,
    arabic_context: str,
    translation_context: str,
    tafsir_excerpt: str | None,
    tafsir_excerpt_characters: int = DEFAULT_TAFSIR_EXCERPT_CHARACTERS,
    already_embedded: bool = False,
    stored_sha256: str | None = None,
) -> EnrichedQuranDocument:
    """Create bounded, index-only embedding input; it is never canonical evidence."""

    if tafsir_excerpt_characters <= 0:
        raise ValueError("tafsir_excerpt_characters must be positive")
    normalized_tafsir = normalize_arabic(tafsir_excerpt or "")[:tafsir_excerpt_characters]
    parts = [
        f"Quran Arabic context: {normalize_arabic(arabic_context)}",
        f"Quran English context: {normalize_translation(translation_context)}",
    ]
    if normalized_tafsir:
        parts.append(f"Tafsir context: {normalized_tafsir}")
    text = "\n".join(parts)
    digest = hashlib.sha256(f"{ENRICHMENT_VERSION}\n{text}".encode()).hexdigest()
    return EnrichedQuranDocument(
        document_id=document_id,
        text=text,
        sha256=digest,
        tafsir_covered=bool(normalized_tafsir),
        already_embedded=already_embedded,
        stored_sha256=stored_sha256,
    )


class QuranEnrichedReembedding:
    """Resumable backfill whose database row state is its durable checkpoint."""

    def __init__(
        self,
        *,
        repository: PostgresRepository,
        embeddings: CloudflareEmbeddingClient,
        policy: SourcePolicy,
        tafsir_excerpt_characters: int = DEFAULT_TAFSIR_EXCERPT_CHARACTERS,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.repository = repository
        self.embeddings = embeddings
        self.policy = policy
        self.tafsir_excerpt_characters = tafsir_excerpt_characters
        self.batch_size = batch_size
        self.provider_call_count = 0

    async def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        rows = await self.repository.enriched_quran_document_inputs(
            tafsir_resource_alias=TAFSIR_SEARCH_ALIAS,
            tafsir_excerpt_characters=self.tafsir_excerpt_characters,
        )
        documents = [
            build_enriched_document(
                document_id=row["document_id"],
                arabic_context=str(row["arabic_context"] or ""),
                translation_context=str(row["translation_context"] or ""),
                tafsir_excerpt=(
                    str(row["tafsir_excerpt"]) if row["tafsir_excerpt"] is not None else None
                ),
                tafsir_excerpt_characters=self.tafsir_excerpt_characters,
                already_embedded=bool(row["enriched_embedding_present"]),
                stored_sha256=row["enriched_document_sha256"],
            )
            for row in rows
        ]
        pending = [document for document in documents if document.needs_embedding]
        summary: dict[str, Any] = {
            "enrichment_version": ENRICHMENT_VERSION,
            "document_count": len(documents),
            "tafsir_covered_document_count": sum(document.tafsir_covered for document in documents),
            "verse_context_only_document_count": sum(
                not document.tafsir_covered for document in documents
            ),
            "pending_document_count": len(pending),
            "tafsir_excerpt_characters": self.tafsir_excerpt_characters,
            "batch_size": self.batch_size,
            "dry_run": dry_run,
        }
        if dry_run:
            return summary

        await self._record_activations()
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors, completion = await self.embeddings.embed([document.text for document in batch])
            ProviderRoutePolicy.require_cloudflare_bge_m3_direct_free(completion)
            await self.repository.store_enriched_embeddings(
                [
                    (
                        vector,
                        self.embeddings.model,
                        self.embeddings.deployment_id,
                        document.document_id,
                        document.text,
                        document.sha256,
                        document.tafsir_covered,
                    )
                    for document, vector in zip(batch, vectors, strict=True)
                ]
            )
            self.provider_call_count += 1
            log_event(
                "quran_enriched_embedding_batch_complete",
                batch_size=len(batch),
                remaining=len(pending) - start - len(batch),
                model_id=self.embeddings.model,
                latency_ms=completion.latency_ms,
            )
        total, embedded, tafsir_covered = await self.repository.enriched_quran_document_counts()
        summary.update(
            {
                "provider_call_count": self.provider_call_count,
                "final_document_count": total,
                "final_embedded_document_count": embedded,
                "final_tafsir_covered_document_count": tafsir_covered,
            }
        )
        return summary

    async def _record_activations(self) -> None:
        for alias in (ARABIC_ALIAS, TRANSLATION_ALIAS, TAFSIR_SEARCH_ALIAS):
            for operation in (SourceOperation.INDEX_VECTOR, SourceOperation.EXTERNAL_EMBEDDING):
                decision = self.policy.authorize(
                    alias,
                    operation,
                    deployment=CF_DEPLOYMENT
                    if operation is SourceOperation.EXTERNAL_EMBEDDING
                    else None,
                )
                await self.repository.record_activation(decision)
