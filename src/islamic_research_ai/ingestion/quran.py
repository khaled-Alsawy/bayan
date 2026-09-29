from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from islamic_research_ai.logging import log_event
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.rights import SourceOperation, SourcePolicy
from islamic_research_ai.settings import Settings

ARABIC_ALIAS = "tanzil-uthmani-v1.1"
TRANSLATION_ALIAS = "quranenc-rowwad-en-v1.0.19"
EXPECTED_VERSE_COUNT = 6236
CF_DEPLOYMENT = {
    "gateway_provider_id": "cloudflare",
    "effective_provider_id": "cloudflare",
    "model_id": "@cf/baai/bge-m3",
    "zero_data_retention": False,
    "provider_training": False,
    "aggregator_routing": False,
}


@dataclass(frozen=True)
class CanonicalRow:
    chapter: int
    verse: int
    text: str
    metadata: dict[str, Any]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tanzil(path: Path) -> list[CanonicalRow]:
    rows: list[CanonicalRow] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("|", 2)
        if len(fields) != 3:
            raise ValueError("Unexpected Tanzil source line")
        chapter, verse, text = fields
        rows.append(CanonicalRow(int(chapter), int(verse), text, {}))
    _validate_rows(rows, "Tanzil")
    return rows


def parse_quranenc(path: Path) -> list[CanonicalRow]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        records = connection.execute(
            "SELECT sura, aya, translation, footnotes FROM translations ORDER BY sura, aya"
        ).fetchall()
    finally:
        connection.close()
    rows = [
        CanonicalRow(
            int(chapter),
            int(verse),
            str(translation),
            {"footnotes": footnotes} if footnotes else {},
        )
        for chapter, verse, translation, footnotes in records
    ]
    _validate_rows(rows, "QuranEnc/Rowwad")
    return rows


def _validate_rows(rows: list[CanonicalRow], label: str) -> None:
    keys = [(row.chapter, row.verse) for row in rows]
    if len(rows) != EXPECTED_VERSE_COUNT:
        raise ValueError(f"{label} has {len(rows)} rows; expected {EXPECTED_VERSE_COUNT}")
    if len(set(keys)) != EXPECTED_VERSE_COUNT:
        raise ValueError(f"{label} contains duplicate or missing verse keys")
    if not all(row.text for row in rows):
        raise ValueError(f"{label} contains an empty canonical text")


class QuranIngestion:
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
        self.settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        tanzil_path = self.settings.raw_data_dir / "quran-uthmani.txt"
        quranenc_path = self.settings.raw_data_dir / "english_rwwad.sqlite"
        await self._bootstrap_resource_records()
        await self._ensure_download(ARABIC_ALIAS, self.settings.tanzil_download_url, tanzil_path)
        await self._ensure_download(
            TRANSLATION_ALIAS, self.settings.quranenc_download_url, quranenc_path
        )

        arabic_rows = parse_tanzil(tanzil_path)
        translation_rows = parse_quranenc(quranenc_path)
        if {(r.chapter, r.verse) for r in arabic_rows} != {
            (r.chapter, r.verse) for r in translation_rows
        }:
            raise ValueError("Arabic and translation verse identities do not match")

        for alias in (ARABIC_ALIAS, TRANSLATION_ALIAS):
            decision = self.policy.authorize(alias, SourceOperation.STORE)
            await self.repository.record_activation(decision)

        arabic_revision = await self._persist_resource(
            alias=ARABIC_ALIAS,
            source_id="tanzil",
            work_title="The Quran — Uthmani text",
            language="ar",
            contributors=["Tanzil Project"],
            canonical_base_url="https://tanzil.net/",
            attribution={
                "display_name": "Tanzil Project",
                "url": "https://tanzil.net/",
                "notice": "Text obtained from Tanzil Project; verbatim canonical text retained.",
            },
            version="1.1",
            path=tanzil_path,
            source_url=self.settings.tanzil_download_url,
            rows=arabic_rows,
        )
        translation_revision = await self._persist_resource(
            alias=TRANSLATION_ALIAS,
            source_id="quranenc",
            work_title="Translation of the Meanings of the Noble Quran — Rowwad Translation Center",
            language="en",
            contributors=["Rowwad Translation Center", "QuranEnc"],
            canonical_base_url="https://quranenc.com/en/browse/english_rwwad",
            attribution={
                "display_name": "QuranEnc / Rowwad Translation Center",
                "url": "https://quranenc.com/en/browse/english_rwwad",
            },
            version="1.0.19",
            path=quranenc_path,
            source_url=self.settings.quranenc_download_url,
            rows=translation_rows,
        )

        for alias in (ARABIC_ALIAS, TRANSLATION_ALIAS):
            decision = self.policy.authorize(alias, SourceOperation.INDEX_LEXICAL)
            await self.repository.record_activation(decision)

        counts = await self.repository.search_document_counts(arabic_revision, translation_revision)
        if counts[0] != EXPECTED_VERSE_COUNT:
            translations = {(row.chapter, row.verse): row for row in translation_rows}
            search_rows = []
            for arabic in arabic_rows:
                translation = translations[(arabic.chapter, arabic.verse)]
                normalized_arabic = normalize_arabic(arabic.text)
                normalized_translation = normalize_translation(translation.text)
                search_rows.append(
                    (
                        arabic.chapter,
                        arabic.verse,
                        normalized_arabic,
                        normalized_translation,
                        f"{normalized_arabic}\n{normalized_translation}",
                    )
                )
            await self.repository.replace_search_documents(
                arabic_revision_id=arabic_revision,
                translation_revision_id=translation_revision,
                rows=search_rows,
            )

        for alias in (ARABIC_ALIAS, TRANSLATION_ALIAS):
            for operation in (
                SourceOperation.EXTERNAL_EMBEDDING,
                SourceOperation.INDEX_VECTOR,
            ):
                decision = self.policy.authorize(
                    alias,
                    operation,
                    deployment=CF_DEPLOYMENT
                    if operation is SourceOperation.EXTERNAL_EMBEDDING
                    else None,
                )
                await self.repository.record_activation(decision)

        await self._embed_missing(arabic_revision, translation_revision)
        final_counts = await self.repository.search_document_counts(
            arabic_revision, translation_revision
        )
        if final_counts != (EXPECTED_VERSE_COUNT, EXPECTED_VERSE_COUNT):
            raise ValueError(f"Incomplete search index: {final_counts}")

        summary = {
            "completed_at": datetime.now(UTC).isoformat(),
            "record_count": EXPECTED_VERSE_COUNT,
            "arabic": {
                "resource_alias": ARABIC_ALIAS,
                "revision_id": str(arabic_revision),
                "version": "1.1",
                "checksum_sha256": sha256_file(tanzil_path),
            },
            "translation": {
                "resource_alias": TRANSLATION_ALIAS,
                "revision_id": str(translation_revision),
                "version": "1.0.19",
                "checksum_sha256": sha256_file(quranenc_path),
            },
            "index": {
                "lexical_rows": final_counts[0],
                "vector_rows": final_counts[1],
                "embedding_model": self.embeddings.model,
                "embedding_dimension": self.embeddings.dimension,
                "provider_requests_this_run": len(self.embedding_completions),
                "provider_latency_ms_this_run": sum(
                    item["latency_ms"] for item in self.embedding_completions
                ),
            },
            "gate_3": "OPEN_AWAITING_RESPONSES",
            "development_override": "PROJECT_OWNER_ACCEPTED_RISK",
        }
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.settings.artifact_dir / "ingestion-summary.json"
        artifact.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log_event("quran_ingestion_complete", **summary)
        return summary

    async def _bootstrap_resource_records(self) -> None:
        resources = [
            {
                "resource_alias": ARABIC_ALIAS,
                "source_id": "tanzil",
                "work_title": "The Quran — Uthmani text",
                "language": "ar",
                "author_translator_or_grader": ["Tanzil Project"],
                "canonical_base_url": "https://tanzil.net/",
                "attribution": {
                    "display_name": "Tanzil Project",
                    "url": "https://tanzil.net/",
                },
                "rights_policy": self.policy.resources[ARABIC_ALIAS]["rights"],
            },
            {
                "resource_alias": TRANSLATION_ALIAS,
                "source_id": "quranenc",
                "work_title": (
                    "Translation of the Meanings of the Noble Quran — Rowwad Translation Center"
                ),
                "language": "en",
                "author_translator_or_grader": ["Rowwad Translation Center", "QuranEnc"],
                "canonical_base_url": "https://quranenc.com/en/browse/english_rwwad",
                "attribution": {
                    "display_name": "QuranEnc / Rowwad Translation Center",
                    "url": "https://quranenc.com/en/browse/english_rwwad",
                },
                "rights_policy": self.policy.resources[TRANSLATION_ALIAS]["rights"],
            },
        ]
        for resource in resources:
            await self.repository.upsert_resource(resource)

    async def _ensure_download(self, alias: str, url: str, path: Path) -> None:
        decision = self.policy.authorize(alias, SourceOperation.FETCH_REMOTE)
        await self.repository.record_activation(decision)
        if path.exists() and path.stat().st_size:
            return
        async with httpx.AsyncClient(follow_redirects=True, timeout=90.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            path.write_bytes(response.content)

    async def _persist_resource(
        self,
        *,
        alias: str,
        source_id: str,
        work_title: str,
        language: str,
        contributors: list[str],
        canonical_base_url: str,
        attribution: dict[str, str],
        version: str,
        path: Path,
        source_url: str,
        rows: list[CanonicalRow],
    ) -> UUID:
        policy_resource = self.policy.resources[alias]
        await self.repository.upsert_resource(
            {
                "resource_alias": alias,
                "source_id": source_id,
                "work_title": work_title,
                "language": language,
                "author_translator_or_grader": contributors,
                "canonical_base_url": canonical_base_url,
                "attribution": attribution,
                "rights_policy": policy_resource["rights"],
            }
        )
        revision = await self.repository.upsert_revision(
            resource_alias=alias,
            version=version,
            checksum_sha256=sha256_file(path),
            acquired_at=datetime.now(UTC).isoformat(),
            record_count=len(rows),
            source_url=source_url,
            metadata={"canonical_exact": True, "encoding": "UTF-8"},
        )
        await self.repository.replace_canonical_rows(
            revision_id=revision,
            resource_alias=alias,
            language=language,
            rows=((row.chapter, row.verse, row.text, row.digest, row.metadata) for row in rows),
        )
        return revision

    async def _embed_missing(self, arabic_revision: UUID, translation_revision: UUID) -> None:
        while True:
            missing = await self.repository.unembedded_documents(
                arabic_revision, translation_revision
            )
            if not missing:
                return
            batch = missing[:64]
            vectors, completion = await self.embeddings.embed(
                [row["combined_search_text"] for row in batch]
            )
            self.embedding_completions.append(completion.model_dump(mode="json"))
            await self.repository.store_embeddings(
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
                "quran_embedding_batch_complete",
                batch_size=len(batch),
                remaining=max(0, len(missing) - len(batch)),
                model_id=self.embeddings.model,
                latency_ms=completion.latency_ms,
            )
