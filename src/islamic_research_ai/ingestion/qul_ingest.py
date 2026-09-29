"""Ingestion orchestration for the QUL tafsir corpus (WP-0, V3 plan).

Reuses the exact rights/revision/embedding pattern `ingestion/quran.py` already established:
authorize each operation through `SourcePolicy` before performing it, record every decision,
version content by revision, embed in bounded batches. The only new pieces are the QUL-shaped
rows from `ingestion.tafsir_qul` and the `tafsir_passages` / `tafsir_search_documents` tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from islamic_research_ai.ingestion.quran import sha256_file
from islamic_research_ai.ingestion.tafsir_qul import TafsirPassage, parse_qul_export
from islamic_research_ai.logging import log_event
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.retrieval.normalization import normalize_arabic
from islamic_research_ai.rights import SourceOperation, SourcePolicy
from islamic_research_ai.settings import Settings

CF_DEPLOYMENT = {
    "gateway_provider_id": "cloudflare",
    "effective_provider_id": "cloudflare",
    "model_id": "@cf/baai/bge-m3",
    "zero_data_retention": False,
    "provider_training": False,
    "aggregator_routing": False,
}


@dataclass(frozen=True)
class QulResourceSpec:
    resource_alias: str
    filename: str
    work_title: str
    authors: tuple[str, ...]
    tier: str  # "search" | "evidence"
    qul_resource_id: int


QUL_RESOURCES: tuple[QulResourceSpec, ...] = (
    QulResourceSpec(
        resource_alias="qul-tafsir-almukhtasar-ar-v1",
        filename="arabic-al-mukhtasar-in-interpreting-the-noble-quran.json",
        work_title="Al-Mukhtasar fi Tafsir al-Quran al-Karim",
        authors=("Tafsir Center for Quranic Studies (King Fahd Quran Printing Complex)",),
        tier="search",
        qul_resource_id=251,
    ),
    QulResourceSpec(
        resource_alias="qul-tafsir-ibnkathir-ar-v1",
        filename="ar-tafsir-ibn-kathir.json",
        work_title="Tafsir Ibn Kathir",
        authors=("Ibn Kathir",),
        tier="evidence",
        qul_resource_id=22,
    ),
    QulResourceSpec(
        resource_alias="qul-tafsir-alqurtubi-ar-v1",
        filename="ar-tafseer-al-qurtubi.json",
        work_title="Al-Jami' li-Ahkam al-Quran (Tafsir al-Qurtubi)",
        authors=("Al-Qurtubi",),
        tier="evidence",
        qul_resource_id=23,
    ),
    QulResourceSpec(
        resource_alias="qul-tafsir-albaghawi-ar-v1",
        filename="ar-tafsir-al-baghawi.json",
        work_title="Ma'alim al-Tanzil (Tafsir al-Baghawi)",
        authors=("Al-Baghawi",),
        tier="evidence",
        qul_resource_id=27,
    ),
)


class QulTafsirIngestion:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: PostgresRepository,
        policy: SourcePolicy,
        embeddings: CloudflareEmbeddingClient,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.policy = policy
        self.embeddings = embeddings
        self.embedding_completions: list[dict[str, Any]] = []

    async def run(self) -> dict[str, Any]:
        qul_dir = self.settings.raw_data_dir / "qul" / "extracted"
        summary: dict[str, Any] = {"completed_at": datetime.now(UTC).isoformat(), "resources": {}}
        for spec in QUL_RESOURCES:
            summary["resources"][spec.resource_alias] = await self._ingest_one(spec, qul_dir)
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.settings.artifact_dir / "qul-tafsir-ingestion-summary.json"
        artifact.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log_event("qul_tafsir_ingestion_complete", resource_count=len(QUL_RESOURCES))
        return summary

    async def _ingest_one(self, spec: QulResourceSpec, qul_dir: Path) -> dict[str, Any]:
        path = qul_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"QUL export missing for {spec.resource_alias}: {path}. "
                "This ingestion assumes the corpus is already downloaded to data/raw/qul/ "
                "(see the WP-0 acquisition note); it does not fetch it."
            )
        await self.repository.upsert_resource(
            {
                "resource_alias": spec.resource_alias,
                "source_id": "qul",
                "work_title": spec.work_title,
                "language": "ar",
                "author_translator_or_grader": list(spec.authors),
                "canonical_base_url": (
                    f"https://qul.tarteel.ai/resources/tafsir/{spec.qul_resource_id}"
                ),
                "attribution": {
                    "display_name": "Quranic Universal Library (Tarteel)",
                    "url": "https://qul.tarteel.ai",
                    "notice": (
                        "Digitised export via the Quranic Universal Library; original "
                        f"work by {', '.join(spec.authors)}."
                    ),
                },
                "rights_policy": self.policy.resources[spec.resource_alias]["rights"],
            }
        )

        passages = parse_qul_export(path, resource_alias=spec.resource_alias)
        revision_id = await self.repository.upsert_revision(
            resource_alias=spec.resource_alias,
            version="qul-export-1",
            checksum_sha256=sha256_file(path),
            acquired_at=datetime.now(UTC).isoformat(),
            record_count=len(passages),
            source_url=f"https://qul.tarteel.ai/resources/tafsir/{spec.qul_resource_id}",
            metadata={
                "tier": spec.tier,
                "qul_resource_id": spec.qul_resource_id,
                "canonical_exact": False,  # AI-facing prose, not scriptural canonical text
            },
        )

        store_decision = self.policy.authorize(spec.resource_alias, SourceOperation.STORE)
        await self.repository.record_activation(store_decision)
        await self.repository.replace_tafsir_passages(
            revision_id=revision_id,
            resource_alias=spec.resource_alias,
            rows=(
                (
                    p.chapter,
                    p.verse_start,
                    p.verse_end,
                    p.text,
                    list(p.hadith_references),
                    p.content_hash,
                )
                for p in passages
            ),
        )
        display_decision = self.policy.authorize(spec.resource_alias, SourceOperation.DISPLAY)
        await self.repository.record_activation(display_decision)

        result: dict[str, Any] = {
            "tier": spec.tier,
            "revision_id": str(revision_id),
            "passage_count": len(passages),
            "hadith_reference_count": sum(len(p.hadith_references) for p in passages),
            "checksum_sha256": sha256_file(path),
        }
        if spec.tier == "search":
            result["search_index"] = await self._embed_search_tier(spec, revision_id, passages)
        return result

    async def _embed_search_tier(
        self, spec: QulResourceSpec, revision_id: UUID, passages: list[TafsirPassage]
    ) -> dict[str, Any]:
        for operation in (SourceOperation.INDEX_LEXICAL, SourceOperation.INDEX_VECTOR):
            decision = self.policy.authorize(spec.resource_alias, operation)
            await self.repository.record_activation(decision)
        embedding_decision = self.policy.authorize(
            spec.resource_alias, SourceOperation.EXTERNAL_EMBEDDING, deployment=CF_DEPLOYMENT
        )
        await self.repository.record_activation(embedding_decision)

        counts = await self.repository.tafsir_search_document_counts(revision_id)
        if counts[0] != len(passages):
            rows = [
                (
                    p.chapter,
                    p.verse_start,
                    p.verse_end,
                    normalize_arabic(p.text),
                    normalize_arabic(p.text),
                )
                for p in passages
            ]
            await self.repository.replace_tafsir_search_documents(
                revision_id=revision_id, resource_alias=spec.resource_alias, rows=rows
            )
        await self._embed_missing(revision_id)
        final_counts = await self.repository.tafsir_search_document_counts(revision_id)
        return {"lexical_rows": final_counts[0], "vector_rows": final_counts[1]}

    async def _embed_missing(self, revision_id: UUID) -> None:
        while True:
            missing = await self.repository.unembedded_tafsir_documents(revision_id)
            if not missing:
                return
            batch = missing[:64]
            vectors, completion = await self.embeddings.embed(
                [row["combined_search_text"] for row in batch]
            )
            self.embedding_completions.append(completion.model_dump(mode="json"))
            await self.repository.store_tafsir_embeddings(
                [
                    (
                        vector,
                        self.embeddings.model,
                        self.embeddings.deployment_id,
                        row["document_id"],
                    )
                    for row, vector in zip(batch, vectors, strict=True)
                ]
            )
            log_event(
                "qul_tafsir_embedding_batch_complete",
                batch_size=len(batch),
                remaining=max(0, len(missing) - len(batch)),
                model_id=self.embeddings.model,
                latency_ms=completion.latency_ms,
            )
