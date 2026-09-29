"""Ingest QUL's curated Ayah Topics as a separate retrieval-only vector index.

Topic labels are metadata used only to retrieve verse locations.  They never enter Quran search
documents and are never eligible to be rendered as canonical source text.
"""

from __future__ import annotations

import hashlib
import html
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from islamic_research_ai.ingestion.quran import CF_DEPLOYMENT, sha256_file
from islamic_research_ai.logging import log_event
from islamic_research_ai.providers.cloudflare import CloudflareEmbeddingClient
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.rights import SourceOperation, SourcePolicy
from islamic_research_ai.settings import Settings

QUL_TOPIC_RESOURCE_ALIAS = "qul-ayah-topics-v1"
QUL_TOPIC_VERSION = "qul-ayah-topics-export-1"
QUL_TOPIC_SOURCE_URL = "https://qul.tarteel.ai"
QUL_TOPIC_LABEL_VERSION = "qul-topic-hierarchy-label-v1"
DEFAULT_HIERARCHY_DEPTH_CAP = 6

_AYAH_REFERENCE = re.compile(r"([1-9][0-9]{0,2}):([1-9][0-9]*)\Z")
_HTML_TAG = re.compile(r"<[^>]+>")


class QulTopicShapeError(ValueError):
    """Raised when the local export cannot be safely mapped to verse locations."""

    def __init__(self, message: str, *, malformed_reference_count: int = 0) -> None:
        super().__init__(message)
        self.malformed_reference_count = malformed_reference_count


@dataclass(frozen=True)
class QulTopic:
    topic_id: int
    name: str
    arabic_name: str | None
    parent_id: int | None
    thematic_parent_id: int | None
    ontology_parent_id: int | None
    description: str | None
    thematic: bool
    ontology: bool
    ayahs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QulTopicDocument:
    topic: QulTopic
    hierarchy_path: str
    label: str
    label_sha256: str


def _optional_int(value: object) -> int | None:
    if value is None or value == "" or value == 0:
        return None
    return int(value)


def _clean_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = " ".join(html.unescape(_HTML_TAG.sub(" ", str(value))).split())
    return text or None


def parse_ayahs(raw: object, *, topic_id: int) -> tuple[tuple[int, int], ...]:
    """Parse every QUL ``chapter:verse`` value, rejecting rather than dropping bad data."""

    if raw is None or not str(raw).strip():
        return ()
    references: list[tuple[int, int]] = []
    malformed: list[str] = []
    for item in str(raw).split(","):
        value = item.strip()
        match = _AYAH_REFERENCE.fullmatch(value)
        if match is None:
            malformed.append(value)
            continue
        chapter, verse = (int(part) for part in match.groups())
        if chapter > 114:
            malformed.append(value)
            continue
        references.append((chapter, verse))
    if malformed:
        raise QulTopicShapeError(
            f"topic {topic_id} has {len(malformed)} malformed ayah reference(s): {malformed[:3]}",
            malformed_reference_count=len(malformed),
        )
    return tuple(dict.fromkeys(references))


def load_qul_topics(path: Path) -> list[QulTopic]:
    """Read the already-acquired SQLite export without any network access."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT topic_id, name, arabic_name, parent_id, thematic_parent_id,
                   ontology_parent_id, description, thematic, ontology, ayahs
            FROM topics ORDER BY topic_id
            """
        ).fetchall()
    finally:
        connection.close()
    topics: list[QulTopic] = []
    malformed_errors: list[QulTopicShapeError] = []
    for row in rows:
        topic_id = int(row[0])
        name = _clean_text(row[1])
        if name is None:
            raise QulTopicShapeError(f"topic {topic_id} has no name")
        try:
            ayahs = parse_ayahs(row[9], topic_id=topic_id)
        except QulTopicShapeError as exc:
            malformed_errors.append(exc)
            continue
        topics.append(
            QulTopic(
                topic_id=topic_id,
                name=name,
                arabic_name=_clean_text(row[2]),
                parent_id=_optional_int(row[3]),
                thematic_parent_id=_optional_int(row[4]),
                ontology_parent_id=_optional_int(row[5]),
                description=_clean_text(row[6]),
                thematic=bool(row[7]),
                ontology=bool(row[8]),
                ayahs=ayahs,
            )
        )
    if malformed_errors:
        malformed_count = sum(error.malformed_reference_count for error in malformed_errors)
        raise QulTopicShapeError(
            "QUL topic export rejected "
            f"malformed_reference_count={malformed_count}; first error: {malformed_errors[0]}",
            malformed_reference_count=malformed_count,
        )
    if not topics:
        raise QulTopicShapeError("QUL topic export contains no topics")
    return topics


def _preferred_parent(topic: QulTopic) -> int | None:
    return topic.thematic_parent_id or topic.ontology_parent_id or topic.parent_id


def build_topic_documents(
    topics: list[QulTopic], *, depth_cap: int = DEFAULT_HIERARCHY_DEPTH_CAP
) -> list[QulTopicDocument]:
    """Construct the measured hierarchy-path label for every topic.

    The preferred QUL parent relation is thematic, then ontology, then generic parent.  A bounded
    walk prevents a malformed cycle from making an ingestion run non-terminating.
    """

    if depth_cap <= 0:
        raise ValueError("depth_cap must be positive")
    by_id = {topic.topic_id: topic for topic in topics}
    documents: list[QulTopicDocument] = []
    for topic in topics:
        names = [topic.name]
        seen = {topic.topic_id}
        parent_id = _preferred_parent(topic)
        for _ in range(depth_cap):
            if parent_id is None:
                break
            if parent_id in seen:
                raise QulTopicShapeError(f"topic hierarchy cycle at topic {topic.topic_id}")
            parent = by_id.get(parent_id)
            if parent is None:
                break
            names.append(parent.name)
            seen.add(parent_id)
            parent_id = _preferred_parent(parent)
        hierarchy_path = " > ".join(reversed(names))
        label = hierarchy_path
        if topic.arabic_name:
            label += f" ({topic.arabic_name})"
        if topic.description:
            label += f" — {topic.description[:120]}"
        digest = hashlib.sha256(f"{QUL_TOPIC_LABEL_VERSION}\n{label}".encode()).hexdigest()
        documents.append(
            QulTopicDocument(
                topic=topic,
                hierarchy_path=hierarchy_path,
                label=label,
                label_sha256=digest,
            )
        )
    return documents


class QulTopicReembedding:
    """Resumable, idempotent topic index ingestion using stored label hashes as checkpoints."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: PostgresRepository,
        policy: SourcePolicy,
        embeddings: CloudflareEmbeddingClient,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.settings = settings
        self.repository = repository
        self.policy = policy
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.provider_call_count = 0

    async def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        path = self.settings.raw_data_dir / "qul" / "ayah-topics.db"
        if not path.exists():
            raise FileNotFoundError(f"QUL topic export is missing: {path}")
        documents = build_topic_documents(load_qul_topics(path))
        summary: dict[str, Any] = {
            "resource_alias": QUL_TOPIC_RESOURCE_ALIAS,
            "topic_count": len(documents),
            "mapped_topic_count": sum(bool(document.topic.ayahs) for document in documents),
            "ayah_mapping_count": sum(len(document.topic.ayahs) for document in documents),
            "malformed_reference_count": 0,
            "batch_size": self.batch_size,
            "dry_run": dry_run,
        }
        if dry_run:
            summary["pending_topic_count"] = len(documents)
            return summary

        await self.repository.upsert_resource(
            {
                "resource_alias": QUL_TOPIC_RESOURCE_ALIAS,
                "source_id": "qul",
                "work_title": "QUL Ayah Topics",
                "language": "en",
                "author_translator_or_grader": ["Quranic Universal Library (Tarteel)"],
                "canonical_base_url": QUL_TOPIC_SOURCE_URL,
                "attribution": {
                    "display_name": "Quranic Universal Library (Tarteel)",
                    "url": QUL_TOPIC_SOURCE_URL,
                    "notice": "Curated topic taxonomy used only as retrieval metadata.",
                },
                "rights_policy": self.policy.resources[QUL_TOPIC_RESOURCE_ALIAS]["rights"],
            }
        )
        revision_id = await self.repository.upsert_revision(
            resource_alias=QUL_TOPIC_RESOURCE_ALIAS,
            version=QUL_TOPIC_VERSION,
            checksum_sha256=sha256_file(path),
            acquired_at=datetime.now(UTC).isoformat(),
            record_count=len(documents),
            source_url=QUL_TOPIC_SOURCE_URL,
            metadata={
                "label_version": QUL_TOPIC_LABEL_VERSION,
                "hierarchy_depth_cap": DEFAULT_HIERARCHY_DEPTH_CAP,
                "renderable_as_source_text": False,
            },
        )
        await self._record_activations()
        expected_hashes = {document.topic.topic_id: document.label_sha256 for document in documents}
        existing_hashes = await self.repository.qul_topic_document_hashes(revision_id)
        if existing_hashes != expected_hashes:
            await self.repository.replace_qul_topic_documents(
                revision_id=revision_id,
                resource_alias=QUL_TOPIC_RESOURCE_ALIAS,
                rows=documents,
            )
        pending = await self.repository.unembedded_qul_topic_documents(revision_id)
        summary["pending_topic_count"] = len(pending)
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors, completion = await self.embeddings.embed([str(row["label"]) for row in batch])
            from islamic_research_ai.routing import ProviderRoutePolicy

            ProviderRoutePolicy.require_cloudflare_bge_m3_direct_free(completion)
            await self.repository.store_qul_topic_embeddings(
                [
                    (
                        vector,
                        self.embeddings.model,
                        self.embeddings.deployment_id,
                        revision_id,
                        int(row["topic_id"]),
                        str(row["label_sha256"]),
                    )
                    for row, vector in zip(batch, vectors, strict=True)
                ]
            )
            self.provider_call_count += 1
            log_event(
                "qul_topic_embedding_batch_complete",
                batch_size=len(batch),
                remaining=len(pending) - start - len(batch),
                model_id=self.embeddings.model,
                latency_ms=completion.latency_ms,
            )
        total, embedded = await self.repository.qul_topic_document_counts(revision_id)
        summary.update(
            {
                "provider_call_count": self.provider_call_count,
                "final_topic_count": total,
                "final_embedded_topic_count": embedded,
                "revision_id": str(revision_id),
            }
        )
        return summary

    async def _record_activations(self) -> None:
        for operation in (
            SourceOperation.STORE,
            SourceOperation.INDEX_VECTOR,
            SourceOperation.EXTERNAL_EMBEDDING,
        ):
            decision = self.policy.authorize(
                QUL_TOPIC_RESOURCE_ALIAS,
                operation,
                deployment=(
                    CF_DEPLOYMENT if operation is SourceOperation.EXTERNAL_EMBEDDING else None
                ),
            )
            await self.repository.record_activation(decision)
