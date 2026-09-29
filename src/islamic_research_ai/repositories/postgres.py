from __future__ import annotations

import hashlib
import re
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from islamic_research_ai.contracts import (
    ConversationDetailV1,
    ConversationSummaryV1,
    ConversationTurnV1,
    ProviderCompletion,
    QuranSearchCandidateV1,
    ResearchPlanV1,
    ResearchResponseV1,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SourceRevisionIdentity,
    TraceEventV1,
    VerseKey,
)
from islamic_research_ai.persistence import response_for_durable_storage
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation
from islamic_research_ai.retrieval.relevance import (
    arabic_root_regex,
    lexical_relevance_score,
)
from islamic_research_ai.rights import AuthorizationDecision

_QUERY_STOPWORDS = {
    "a",
    "about",
    "after",
    "and",
    "are",
    "be",
    "becomes",
    "does",
    "during",
    "even",
    "facing",
    "for",
    "how",
    "in",
    "is",
    "it",
    "life",
    "of",
    "people",
    "quran",
    "say",
    "should",
    "someone",
    "the",
    "they",
    "those",
    "to",
    "what",
    "when",
    "which",
    "who",
    "you",
    "your",
    "القران",
    "القرآن",
    "الى",
    "إلى",
    "عن",
    "عند",
    "على",
    "في",
    "قال",
    "كيف",
    "ما",
    "ماذا",
    "من",
    "هل",
    "يقول",
}

# Keep the conventional RRF constant. The development sanity artifact records
# channel ranks separately so relevance weaknesses remain visible rather than
# being hidden by query-specific rank tuning.
_RRF_RANK_CONSTANT = 60


def _encode_cursor(timestamp: datetime, value_id: UUID) -> str:
    raw = f"{timestamp.isoformat()}|{value_id}".encode()
    return urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        timestamp, value_id = urlsafe_b64decode(padded.encode()).decode().split("|", 1)
        return datetime.fromisoformat(timestamp), UUID(value_id)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid pagination cursor") from exc


def _lexical_tsquery(query: str) -> str:
    normalized_query = normalize_arabic(normalize_translation(query))
    tokens = [
        token.casefold()
        for token in re.findall(r"[\w\u0600-\u06ff]+", normalized_query)
        if token.casefold() not in _QUERY_STOPWORDS
    ]
    if not tokens:
        tokens = re.findall(r"[\w\u0600-\u06ff]+", normalized_query)
    prefixes = [token[:5] if token.isascii() and len(token) >= 7 else token for token in tokens]
    return " | ".join(f"{token}:*" for token in dict.fromkeys(prefixes[:12]))


class PostgresRepository:
    def __init__(self, database_url: str, *, quran_embedding_variant: str = "base") -> None:
        if quran_embedding_variant not in {"base", "enriched"}:
            raise ValueError("quran_embedding_variant must be 'base' or 'enriched'")

        async def configure(connection: Any) -> None:
            await register_vector_async(connection)

        self.quran_embedding_variant = quran_embedding_variant
        self.pool = AsyncConnectionPool(
            database_url,
            min_size=1,
            max_size=6,
            open=False,
            kwargs={"row_factory": dict_row},
            configure=configure,
        )

    @property
    def _quran_embedding_columns(self) -> tuple[str, str]:
        if getattr(self, "quran_embedding_variant", "base") == "enriched":
            return "enriched_embedding", "enriched_embedding_model"
        return "embedding", "embedding_model"

    async def open(self) -> None:
        await self.pool.open()
        await self.pool.wait()

    async def close(self) -> None:
        await self.pool.close()

    async def migrate(self, path: Path) -> None:
        sql = path.read_text(encoding="utf-8")
        async with self.pool.connection() as connection:
            await connection.execute(sql, prepare=False)

    async def upsert_resource(self, resource: dict[str, Any]) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO source_resources (
                    resource_alias, source_id, work_title, language,
                    author_translator_or_grader, canonical_base_url, attribution, rights_policy
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_alias) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    work_title = EXCLUDED.work_title,
                    language = EXCLUDED.language,
                    author_translator_or_grader = EXCLUDED.author_translator_or_grader,
                    canonical_base_url = EXCLUDED.canonical_base_url,
                    attribution = EXCLUDED.attribution,
                    rights_policy = EXCLUDED.rights_policy
                """,
                (
                    resource["resource_alias"],
                    resource["source_id"],
                    resource["work_title"],
                    resource["language"],
                    Jsonb(resource["author_translator_or_grader"]),
                    resource.get("canonical_base_url"),
                    Jsonb(resource["attribution"]),
                    Jsonb(resource["rights_policy"]),
                ),
            )

    async def upsert_revision(
        self,
        *,
        resource_alias: str,
        version: str,
        checksum_sha256: str,
        acquired_at: str,
        record_count: int,
        source_url: str,
        metadata: dict[str, Any],
    ) -> UUID:
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE source_revisions SET active = false WHERE resource_alias = %s",
                    (resource_alias,),
                )
                cursor = await connection.execute(
                    """
                    INSERT INTO source_revisions (
                        resource_alias, version, checksum_sha256, acquired_at,
                        record_count, source_url, active, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, true, %s)
                    ON CONFLICT (resource_alias, version, checksum_sha256) DO UPDATE SET
                        acquired_at = EXCLUDED.acquired_at,
                        record_count = EXCLUDED.record_count,
                        source_url = EXCLUDED.source_url,
                        active = true,
                        metadata = EXCLUDED.metadata
                    RETURNING revision_id
                    """,
                    (
                        resource_alias,
                        version,
                        checksum_sha256,
                        acquired_at,
                        record_count,
                        source_url,
                        Jsonb(metadata),
                    ),
                )
                row = await cursor.fetchone()
                return row["revision_id"]

    async def replace_canonical_rows(
        self,
        *,
        revision_id: UUID,
        resource_alias: str,
        language: str,
        rows: Iterable[tuple[int, int, str, str, dict[str, Any]]],
    ) -> None:
        values = [
            (revision_id, resource_alias, chapter, verse, language, text, digest, Jsonb(meta))
            for chapter, verse, text, digest, meta in rows
        ]
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM quran_canonical_text WHERE revision_id = %s",
                    (revision_id,),
                )
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO quran_canonical_text (
                            revision_id, resource_alias, chapter, verse, language,
                            canonical_text, content_hash, source_metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )

    async def replace_search_documents(
        self,
        *,
        arabic_revision_id: UUID,
        translation_revision_id: UUID,
        rows: Iterable[tuple[int, int, str, str, str]],
    ) -> None:
        values = [
            (arabic_revision_id, translation_revision_id, chapter, verse, ar, en, combined)
            for chapter, verse, ar, en, combined in rows
        ]
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM quran_search_documents
                    WHERE arabic_revision_id = %s AND translation_revision_id = %s
                    """,
                    (arabic_revision_id, translation_revision_id),
                )
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO quran_search_documents (
                            arabic_revision_id, translation_revision_id, chapter, verse,
                            normalized_arabic, normalized_translation, combined_search_text
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )

    async def search_document_counts(
        self, arabic_revision_id: UUID, translation_revision_id: UUID
    ) -> tuple[int, int]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS total, count(embedding) AS embedded
                FROM quran_search_documents
                WHERE arabic_revision_id = %s AND translation_revision_id = %s
                """,
                (arabic_revision_id, translation_revision_id),
            )
            row = await cursor.fetchone()
            return int(row["total"]), int(row["embedded"])

    async def unembedded_documents(
        self, arabic_revision_id: UUID, translation_revision_id: UUID
    ) -> list[dict[str, Any]]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT document_id, combined_search_text
                FROM quran_search_documents
                WHERE arabic_revision_id = %s AND translation_revision_id = %s
                  AND embedding IS NULL
                ORDER BY chapter, verse
                """,
                (arabic_revision_id, translation_revision_id),
            )
            return list(await cursor.fetchall())

    async def store_embeddings(
        self,
        rows: list[tuple[list[float], str, str, UUID]],
    ) -> None:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    UPDATE quran_search_documents SET
                        embedding = %s, embedding_model = %s,
                        embedding_deployment_id = %s, embedded_at = now()
                    WHERE document_id = %s
                    """,
                    rows,
                )

    async def enriched_quran_document_inputs(
        self, *, tafsir_resource_alias: str, tafsir_excerpt_characters: int
    ) -> list[dict[str, Any]]:
        """Return index-only context and bounded Tafsir text for active Quran documents."""

        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT d.document_id, d.chapter, d.verse, d.enriched_document_sha256,
                       d.enriched_embedding IS NOT NULL AS enriched_embedding_present,
                       context.arabic_context, context.translation_context, tafsir.tafsir_excerpt
                FROM quran_search_documents d
                JOIN source_revisions ar_rev ON ar_rev.revision_id = d.arabic_revision_id
                JOIN source_revisions tr_rev ON tr_rev.revision_id = d.translation_revision_id
                JOIN LATERAL (
                    SELECT string_agg(ar.canonical_text, chr(10) ORDER BY ar.verse)
                               AS arabic_context,
                           string_agg(tr.canonical_text, chr(10) ORDER BY tr.verse)
                               AS translation_context
                    FROM quran_canonical_text ar
                    JOIN quran_canonical_text tr
                      ON tr.revision_id = d.translation_revision_id
                     AND tr.chapter = ar.chapter AND tr.verse = ar.verse
                    WHERE ar.revision_id = d.arabic_revision_id
                      AND ar.chapter = d.chapter
                      AND ar.verse BETWEEN greatest(1, d.verse - 1) AND d.verse + 1
                ) context ON true
                LEFT JOIN LATERAL (
                    SELECT left(tp.passage_text, %s) AS tafsir_excerpt
                    FROM tafsir_passages tp
                    JOIN source_revisions tp_rev ON tp_rev.revision_id = tp.revision_id
                    WHERE tp_rev.active
                      AND tp.resource_alias = %s
                      AND tp.chapter = d.chapter
                      AND d.verse BETWEEN tp.verse_start AND tp.verse_end
                    ORDER BY (tp.verse_end - tp.verse_start), tp.verse_start
                    LIMIT 1
                ) tafsir ON true
                WHERE ar_rev.active AND tr_rev.active
                ORDER BY d.chapter, d.verse
                """,
                (tafsir_excerpt_characters, tafsir_resource_alias),
            )
            return list(await cursor.fetchall())

    async def store_enriched_embeddings(
        self, rows: list[tuple[list[float], str, str, UUID, str, str, bool]]
    ) -> None:
        """Persist a batch atomically; existing document rows are the durable checkpoint."""

        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    UPDATE quran_search_documents SET
                        enriched_embedding = %s,
                        enriched_embedding_model = %s,
                        enriched_embedding_deployment_id = %s,
                        enriched_document_text = %s,
                        enriched_document_sha256 = %s,
                        enriched_tafsir_covered = %s,
                        enriched_embedded_at = now()
                    WHERE document_id = %s
                    """,
                    [
                        (vector, model, deployment, text, digest, covered, document_id)
                        for vector, model, deployment, document_id, text, digest, covered in rows
                    ],
                )

    async def enriched_quran_document_counts(self) -> tuple[int, int, int]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS total,
                       count(enriched_embedding) AS embedded,
                       count(*) FILTER (WHERE enriched_tafsir_covered) AS tafsir_covered
                FROM quran_search_documents
                """
            )
            row = await cursor.fetchone()
            return int(row["total"]), int(row["embedded"]), int(row["tafsir_covered"])

    # -- WP-0 (V3 plan): QUL tafsir passages -------------------------------------------------

    async def replace_tafsir_passages(
        self,
        *,
        revision_id: UUID,
        resource_alias: str,
        rows: Iterable[tuple[int, int, int, str, list[str], str]],
    ) -> None:
        """Replace every passage for one (revision, resource) pair.

        ``rows`` is (chapter, verse_start, verse_end, text, hadith_references, content_hash),
        exactly the shape ``ingestion.tafsir_qul.TafsirPassage`` emits.
        """

        values = [
            (
                revision_id,
                resource_alias,
                chapter,
                verse_start,
                verse_end,
                text,
                Jsonb(list(hadith_references)),
                content_hash,
            )
            for chapter, verse_start, verse_end, text, hadith_references, content_hash in rows
        ]
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM tafsir_passages WHERE revision_id = %s AND resource_alias = %s",
                    (revision_id, resource_alias),
                )
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO tafsir_passages (
                            revision_id, resource_alias, chapter, verse_start, verse_end,
                            passage_text, hadith_references, content_hash
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )

    async def fetch_tafsir_passages(
        self, *, resource_aliases: list[str], verse_keys: list[VerseKey]
    ) -> list[dict[str, Any]]:
        """Passages from the active revision of each alias whose range covers a requested verse.

        One row per (alias, passage) match; a verse inside a five-ayah passage returns that
        whole passage once, not once per requested verse in its range.
        """

        if not resource_aliases or not verse_keys:
            return []
        chapters = [key.chapter for key in verse_keys]
        verses = [key.verse for key in verse_keys]
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT DISTINCT ON (tp.resource_alias, tp.chapter, tp.verse_start)
                       tp.resource_alias, tp.chapter, tp.verse_start, tp.verse_end,
                       tp.passage_text, tp.hadith_references, tp.content_hash,
                       rev.revision_id, rev.checksum_sha256 AS revision_checksum,
                       res.source_id, res.work_title, res.author_translator_or_grader,
                       res.canonical_base_url, res.attribution, res.language
                FROM tafsir_passages tp
                JOIN source_revisions rev ON rev.revision_id = tp.revision_id
                JOIN source_resources res ON res.resource_alias = tp.resource_alias
                JOIN unnest(%s::smallint[], %s::smallint[]) AS requested(chapter, verse)
                  ON requested.chapter = tp.chapter
                 AND requested.verse BETWEEN tp.verse_start AND tp.verse_end
                WHERE rev.active AND tp.resource_alias = ANY(%s)
                ORDER BY tp.resource_alias, tp.chapter, tp.verse_start
                """,
                (chapters, verses, resource_aliases),
            )
            return list(await cursor.fetchall())

    # -- WP-0 (V3 plan): QUL Al-Mukhtasar search tier ----------------------------------------

    async def replace_tafsir_search_documents(
        self,
        *,
        revision_id: UUID,
        resource_alias: str,
        rows: Iterable[tuple[int, int, int, str, str]],
    ) -> None:
        values = [
            (revision_id, resource_alias, chapter, verse_start, verse_end, normalized, combined)
            for chapter, verse_start, verse_end, normalized, combined in rows
        ]
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM tafsir_search_documents
                    WHERE revision_id = %s AND resource_alias = %s
                    """,
                    (revision_id, resource_alias),
                )
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO tafsir_search_documents (
                            revision_id, resource_alias, chapter, verse_start, verse_end,
                            normalized_text, combined_search_text
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )

    async def tafsir_search_document_counts(self, revision_id: UUID) -> tuple[int, int]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS total, count(embedding) AS embedded
                FROM tafsir_search_documents WHERE revision_id = %s
                """,
                (revision_id,),
            )
            row = await cursor.fetchone()
            return int(row["total"]), int(row["embedded"])

    async def unembedded_tafsir_documents(self, revision_id: UUID) -> list[dict[str, Any]]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT document_id, combined_search_text FROM tafsir_search_documents
                WHERE revision_id = %s AND embedding IS NULL
                ORDER BY chapter, verse_start
                """,
                (revision_id,),
            )
            return list(await cursor.fetchall())

    async def store_tafsir_embeddings(self, rows: list[tuple[list[float], str, str, UUID]]) -> None:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    UPDATE tafsir_search_documents SET
                        embedding = %s, embedding_model = %s,
                        embedding_deployment_id = %s, embedded_at = now()
                    WHERE document_id = %s
                    """,
                    rows,
                )

    # -- QUL curated Ayah Topics -------------------------------------------------------------

    async def replace_qul_topic_documents(
        self,
        *,
        revision_id: UUID,
        resource_alias: str,
        rows: Iterable[Any],
    ) -> None:
        """Atomically replace one immutable QUL topic revision and its verse mappings.

        The topic label/vector tables are intentionally separate from Quran documents.  ``rows``
        contains ``ingestion.qul_topics.QulTopicDocument`` instances, kept untyped here to avoid
        an ingestion-to-repository import cycle.
        """

        documents = list(rows)
        values = [
            (
                revision_id,
                resource_alias,
                document.topic.topic_id,
                document.topic.name,
                document.hierarchy_path,
                document.topic.arabic_name,
                document.topic.description,
                document.topic.thematic,
                document.topic.ontology,
                document.label,
                document.label_sha256,
            )
            for document in documents
        ]
        mappings = [
            (revision_id, document.topic.topic_id, chapter, verse)
            for document in documents
            for chapter, verse in document.topic.ayahs
        ]
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM qul_topic_documents WHERE revision_id = %s", (revision_id,)
                )
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO qul_topic_documents (
                            revision_id, resource_alias, topic_id, name, hierarchy_path,
                            arabic_name, description, thematic, ontology, label, label_sha256
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        values,
                    )
                    await cursor.executemany(
                        """
                        INSERT INTO qul_topic_ayah_mappings (revision_id, topic_id, chapter, verse)
                        VALUES (%s, %s, %s, %s)
                        """,
                        mappings,
                    )

    async def qul_topic_document_counts(self, revision_id: UUID) -> tuple[int, int]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS total, count(embedding) AS embedded
                FROM qul_topic_documents WHERE revision_id = %s
                """,
                (revision_id,),
            )
            row = await cursor.fetchone()
            return int(row["total"]), int(row["embedded"])

    async def qul_topic_document_hashes(self, revision_id: UUID) -> dict[int, str]:
        """Return the durable label-input checkpoints for one topic revision."""

        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT topic_id, label_sha256 FROM qul_topic_documents
                WHERE revision_id = %s ORDER BY topic_id
                """,
                (revision_id,),
            )
            rows = await cursor.fetchall()
            return {int(row["topic_id"]): str(row["label_sha256"]) for row in rows}

    async def unembedded_qul_topic_documents(self, revision_id: UUID) -> list[dict[str, Any]]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT topic_id, label, label_sha256 FROM qul_topic_documents
                WHERE revision_id = %s AND embedding IS NULL
                ORDER BY topic_id
                """,
                (revision_id,),
            )
            return list(await cursor.fetchall())

    async def store_qul_topic_embeddings(
        self, rows: list[tuple[list[float], str, str, UUID, int, str]]
    ) -> None:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    UPDATE qul_topic_documents SET
                        embedding = %s, embedding_model = %s, embedding_deployment_id = %s,
                        embedded_at = now()
                    WHERE revision_id = %s AND topic_id = %s AND label_sha256 = %s
                    """,
                    rows,
                )

    async def search_qul_topic_arm(
        self, *, query_embedding: list[float], topic_limit: int
    ) -> list[dict[str, Any]]:
        """Return top topic matches expanded to verse locations without touching Quran vectors."""

        vector = Vector(query_embedding)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                WITH topic_matches AS (
                    SELECT d.revision_id, d.topic_id, d.name, d.hierarchy_path, d.label,
                           d.embedding_model, d.embedding <=> %s AS distance,
                           row_number() OVER (ORDER BY d.embedding <=> %s, d.topic_id) AS topic_rank
                    FROM qul_topic_documents d
                    JOIN source_revisions rev ON rev.revision_id = d.revision_id
                    WHERE rev.active AND d.embedding IS NOT NULL
                    ORDER BY d.embedding <=> %s, d.topic_id
                    LIMIT %s
                )
                SELECT m.revision_id, rev.checksum_sha256 AS revision_checksum,
                       m.topic_id, m.name, m.hierarchy_path, m.label, m.embedding_model,
                       m.topic_rank, 1 - m.distance AS topic_score,
                       map.chapter, map.verse
                FROM topic_matches m
                JOIN source_revisions rev ON rev.revision_id = m.revision_id
                LEFT JOIN qul_topic_ayah_mappings map
                  ON map.revision_id = m.revision_id AND map.topic_id = m.topic_id
                ORDER BY m.topic_rank, map.chapter, map.verse
                """,
                (vector, vector, vector, topic_limit),
            )
            return list(await cursor.fetchall())

    async def qul_topic_similarity_for_candidates(
        self,
        *,
        query_embedding: list[float],
        locations: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """Return each supplied verse's best active curated-topic label similarity."""

        if not locations:
            return []
        vector = Vector(query_embedding)
        chapters, verses = zip(*locations, strict=True)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                WITH locations AS (
                    SELECT * FROM unnest(%s::integer[], %s::integer[]) AS location(chapter, verse)
                )
                SELECT location.chapter, location.verse,
                       max(1 - (document.embedding <=> %s)) AS topic_similarity
                FROM locations AS location
                JOIN qul_topic_ayah_mappings AS mapping
                  ON mapping.chapter = location.chapter AND mapping.verse = location.verse
                JOIN qul_topic_documents AS document
                  ON document.revision_id = mapping.revision_id
                 AND document.topic_id = mapping.topic_id
                JOIN source_revisions AS revision ON revision.revision_id = document.revision_id
                WHERE revision.active AND document.embedding IS NOT NULL
                GROUP BY location.chapter, location.verse
                ORDER BY location.chapter, location.verse
                """,
                (list(chapters), list(verses), vector),
            )
            return list(await cursor.fetchall())

    async def search_tafsir_tier(
        self, *, query: str, query_embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Lexical + dense RRF search over the embedded Al-Mukhtasar search tier.

        Returns rows mappable to verse ranges so a topical query can resolve to tafsir wording
        directly (WP-3 tafsir-first retrieval), independent of the Quran hybrid_search index.
        Mirrors ``hybrid_search`` structurally but over one resource with a passage range
        instead of two resources keyed by a single verse.
        """

        vector = Vector(query_embedding)
        lexical_query = _lexical_tsquery(query)
        normalized_query = normalize_arabic(normalize_translation(query))
        root_regex = arabic_root_regex(query)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                WITH eligible AS (
                    SELECT d.* FROM tafsir_search_documents d
                    JOIN source_revisions rev ON rev.revision_id = d.revision_id
                    WHERE rev.active AND d.embedding IS NOT NULL
                ), lexical AS (
                    SELECT document_id,
                           row_number() OVER (ORDER BY
                               CASE WHEN normalized_text ~ %s THEN 1 ELSE 0 END DESC,
                               ts_rank_cd(search_vector, to_tsquery('simple', %s)) DESC,
                               similarity(normalized_text, %s) DESC,
                               chapter, verse_start
                           ) AS lexical_rank
                    FROM eligible
                    WHERE search_vector @@ to_tsquery('simple', %s)
                       OR normalized_text %% %s
                       OR normalized_text ~ %s
                    LIMIT 60
                ), dense AS (
                    SELECT document_id,
                           row_number() OVER (
                               ORDER BY embedding <=> %s, chapter, verse_start
                           ) AS vector_rank
                    FROM eligible
                    ORDER BY embedding <=> %s, chapter, verse_start
                    LIMIT 60
                ), fused AS (
                    SELECT coalesce(l.document_id, d.document_id) AS document_id,
                           coalesce(1.0 / (%s + l.lexical_rank), 0) +
                           coalesce(1.0 / (%s + d.vector_rank), 0) AS rrf_score
                    FROM lexical l FULL OUTER JOIN dense d USING (document_id)
                )
                SELECT e.resource_alias, e.chapter, e.verse_start, e.verse_end, f.rrf_score,
                       rev.revision_id, rev.checksum_sha256 AS revision_checksum
                FROM fused f JOIN eligible e USING (document_id)
                JOIN source_revisions rev ON rev.revision_id = e.revision_id
                ORDER BY f.rrf_score DESC, e.chapter, e.verse_start
                LIMIT %s
                """,
                (
                    root_regex,
                    lexical_query,
                    normalized_query,
                    lexical_query,
                    normalized_query,
                    root_regex,
                    vector,
                    vector,
                    _RRF_RANK_CONSTANT,
                    _RRF_RANK_CONSTANT,
                    limit,
                ),
            )
            return list(await cursor.fetchall())

    async def record_activation(self, decision: AuthorizationDecision) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO source_activation_records (
                    resource_alias, operation, rights_decision, override_type,
                    effective_permission, environment, purpose, reason, effective_deployment
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    decision.resource_alias,
                    decision.operation.value,
                    decision.rights_decision.value,
                    decision.override_type.value,
                    decision.effective_permission,
                    decision.environment,
                    decision.purpose,
                    decision.reason,
                    Jsonb(decision.effective_deployment)
                    if decision.effective_deployment is not None
                    else None,
                ),
            )

    async def hybrid_search(
        self, *, query: str, query_embedding: list[float], limit: int
    ) -> list[QuranSearchCandidateV1]:
        candidates, _ = await self._hybrid_search(
            query=query, query_embedding=query_embedding, limit=limit
        )
        return candidates

    async def hybrid_search_with_audit(
        self, *, query: str, query_embedding: list[float], limit: int
    ) -> tuple[
        list[QuranSearchCandidateV1],
        dict[str, list[dict[str, float | int | str | None]]],
    ]:
        """Return the production result plus bounded, non-canonical ranking diagnostics."""

        return await self._hybrid_search(query=query, query_embedding=query_embedding, limit=limit)

    async def _hybrid_search(
        self, *, query: str, query_embedding: list[float], limit: int
    ) -> tuple[
        list[QuranSearchCandidateV1],
        dict[str, list[dict[str, float | int | str | None]]],
    ]:
        embedding_column, embedding_model_column = self._quran_embedding_columns
        vector = Vector(query_embedding)
        lexical_query = _lexical_tsquery(query)
        normalized_query = normalize_arabic(normalize_translation(query))
        root_regex = arabic_root_regex(query)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                WITH eligible AS (
                    SELECT d.* FROM quran_search_documents d
                    JOIN source_revisions ar ON ar.revision_id = d.arabic_revision_id
                    JOIN source_revisions tr ON tr.revision_id = d.translation_revision_id
                    WHERE ar.active AND tr.active AND d.{embedding_column} IS NOT NULL
                ), lexical AS (
                    SELECT document_id,
                           row_number() OVER (ORDER BY
                               CASE WHEN normalized_arabic ~ %s THEN 1 ELSE 0 END DESC,
                               ts_rank_cd(search_vector, to_tsquery('simple', %s)) DESC,
                               greatest(
                                   similarity(normalized_arabic, %s),
                                   similarity(normalized_translation, lower(%s))
                               ) DESC,
                               chapter, verse
                           ) AS lexical_rank
                    FROM eligible
                    WHERE search_vector @@ to_tsquery('simple', %s)
                       OR normalized_arabic %% %s
                       OR normalized_translation %% lower(%s)
                       OR normalized_arabic ~ %s
                    LIMIT 60
                ), dense AS (
                    SELECT document_id,
                           row_number() OVER (
                               ORDER BY {embedding_column} <=> %s, chapter, verse
                           ) AS vector_rank
                    FROM eligible
                    ORDER BY {embedding_column} <=> %s, chapter, verse
                    LIMIT 60
                ), fused AS (
                    SELECT coalesce(l.document_id, d.document_id) AS document_id,
                           l.lexical_rank, d.vector_rank,
                           coalesce(1.0 / (%s + l.lexical_rank), 0) +
                           coalesce(1.0 / (%s + d.vector_rank), 0) AS rrf_score
                    FROM lexical l FULL OUTER JOIN dense d USING (document_id)
                )
                SELECT e.document_id, e.chapter, e.verse, e.arabic_revision_id,
                       e.translation_revision_id, e.{embedding_model_column} AS embedding_model,
                       e.normalized_arabic, e.normalized_translation,
                       ars.source_id AS arabic_source_id,
                       ar.checksum_sha256 AS arabic_revision_checksum,
                       trs.source_id AS translation_source_id,
                       tr.checksum_sha256 AS translation_revision_checksum,
                       f.lexical_rank, f.vector_rank, f.rrf_score
                FROM fused f JOIN eligible e USING (document_id)
                JOIN source_revisions ar ON ar.revision_id = e.arabic_revision_id
                JOIN source_resources ars ON ars.resource_alias = ar.resource_alias
                JOIN source_revisions tr ON tr.revision_id = e.translation_revision_id
                JOIN source_resources trs ON trs.resource_alias = tr.resource_alias
                ORDER BY f.rrf_score DESC, e.chapter, e.verse
                LIMIT %s
                """,
                (
                    root_regex,
                    lexical_query,
                    normalized_query,
                    normalized_query,
                    lexical_query,
                    normalized_query,
                    normalized_query,
                    root_regex,
                    vector,
                    vector,
                    _RRF_RANK_CONSTANT,
                    _RRF_RANK_CONSTANT,
                    max(limit * 8, 60),
                ),
            )
            rows = await cursor.fetchall()
        for row in rows:
            row["lexical_relevance_score"] = lexical_relevance_score(
                query,
                f"{row.get('normalized_arabic', '')} {row.get('normalized_translation', '')}",
            )
        audit_rows = [dict(row) for row in rows]
        rows.sort(
            key=lambda row: (
                -float(row["rrf_score"]),
                -float(row["lexical_relevance_score"]),
                int(row["chapter"]),
                int(row["verse"]),
            )
        )
        rows = rows[:limit]
        result: list[QuranSearchCandidateV1] = []
        for index, row in enumerate(rows, start=1):
            index_revision = self._index_revision(row)
            scores = {
                "rrf_score": float(row["rrf_score"]),
                "lexical_relevance_score": float(row["lexical_relevance_score"]),
            }
            if row["lexical_rank"] is not None:
                scores["lexical_rank"] = float(row["lexical_rank"])
            if row["vector_rank"] is not None:
                scores["vector_rank"] = float(row["vector_rank"])
            result.append(
                QuranSearchCandidateV1(
                    candidate_id=f"C{index}",
                    verse_keys=[VerseKey(chapter=row["chapter"], verse=row["verse"])],
                    index_revision=index_revision,
                    ranking=RetrievalProvenanceV1(
                        tool_id="search_quran_topics.v1",
                        method=RetrievalMethod.RERANKED,
                        index_revision=index_revision,
                        rank=index,
                        component_scores=scores,
                    ),
                )
            )
        def audit_candidates(
            candidates: list[dict[str, Any]], *, component: str | None = None
        ) -> list[dict[str, float | int | str | None]]:
            output: list[dict[str, float | int | str | None]] = []
            for position, row in enumerate(candidates[:12], start=1):
                output.append(
                    {
                        "reference": f"{row['chapter']}:{row['verse']}",
                        "rank": (
                            int(row[component])
                            if component and row[component] is not None
                            else position
                        ),
                        "rrf_score": float(row["rrf_score"]),
                        "lexical_rank": (
                            float(row["lexical_rank"])
                            if row["lexical_rank"] is not None
                            else None
                        ),
                        "vector_rank": (
                            float(row["vector_rank"])
                            if row["vector_rank"] is not None
                            else None
                        ),
                        "lexical_relevance_score": float(
                            row.get("lexical_relevance_score", 0.0)
                        ),
                    }
                )
            return output

        retained_references = {
            (item.verse_keys[0].chapter, item.verse_keys[0].verse) for item in result
        }
        pre_source_dropped = [
            row
            for row in audit_rows
            if (int(row["chapter"]), int(row["verse"])) not in retained_references
        ]
        return result, {
            "lexical": audit_candidates(
                sorted(
                    (row for row in audit_rows if row["lexical_rank"] is not None),
                    key=lambda row: float(row["lexical_rank"]),
                ),
                component="lexical_rank",
            ),
            "dense": audit_candidates(
                sorted(
                    (row for row in audit_rows if row["vector_rank"] is not None),
                    key=lambda row: float(row["vector_rank"]),
                ),
                component="vector_rank",
            ),
            "quran_selected": audit_candidates(rows),
            "quran_dropped_before_arm_fusion": audit_candidates(pre_source_dropped),
        }

    async def lexical_search(self, *, query: str, limit: int) -> list[QuranSearchCandidateV1]:
        """Search the local lexical index when the remote embedding service is unavailable.

        This is deliberately a separate query rather than a zero-vector approximation:
        it must neither invoke pgvector nor represent its ordering as semantic/RRF ranking.
        """

        _, embedding_model_column = self._quran_embedding_columns
        lexical_query = _lexical_tsquery(query)
        normalized_query = normalize_arabic(normalize_translation(query))
        root_regex = arabic_root_regex(query)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT d.chapter, d.verse, d.arabic_revision_id, d.translation_revision_id,
                       d.{embedding_model_column} AS embedding_model,
                       d.normalized_arabic, d.normalized_translation,
                       ars.source_id AS arabic_source_id,
                       ar.checksum_sha256 AS arabic_revision_checksum,
                       trs.source_id AS translation_source_id,
                       tr.checksum_sha256 AS translation_revision_checksum,
                       row_number() OVER (ORDER BY
                           CASE WHEN d.normalized_arabic ~ %s THEN 1 ELSE 0 END DESC,
                           ts_rank_cd(d.search_vector, to_tsquery('simple', %s)) DESC,
                           greatest(
                               similarity(d.normalized_arabic, %s),
                               similarity(d.normalized_translation, lower(%s))
                           ) DESC,
                           d.chapter, d.verse
                       ) AS lexical_rank
                FROM quran_search_documents d
                JOIN source_revisions ar ON ar.revision_id = d.arabic_revision_id
                JOIN source_resources ars ON ars.resource_alias = ar.resource_alias
                JOIN source_revisions tr ON tr.revision_id = d.translation_revision_id
                JOIN source_resources trs ON trs.resource_alias = tr.resource_alias
                WHERE ar.active AND tr.active
                  AND (
                    d.search_vector @@ to_tsquery('simple', %s)
                    OR d.normalized_arabic %% %s
                    OR d.normalized_translation %% lower(%s)
                    OR d.normalized_arabic ~ %s
                  )
                LIMIT %s
                """,
                (
                    root_regex,
                    lexical_query,
                    normalized_query,
                    normalized_query,
                    lexical_query,
                    normalized_query,
                    normalized_query,
                    root_regex,
                    limit,
                ),
            )
            rows = list(await cursor.fetchall())
        for row in rows:
            row["lexical_relevance_score"] = lexical_relevance_score(
                query,
                f"{row.get('normalized_arabic', '')} {row.get('normalized_translation', '')}",
            )
        rows.sort(
            key=lambda row: (
                -float(row["lexical_relevance_score"]),
                float(row["lexical_rank"]),
                int(row["chapter"]),
                int(row["verse"]),
            )
        )
        return [
            QuranSearchCandidateV1(
                candidate_id=f"L{index}",
                verse_keys=[VerseKey(chapter=row["chapter"], verse=row["verse"])],
                index_revision=self._index_revision(row),
                ranking=RetrievalProvenanceV1(
                    tool_id="search_quran_topics.v1",
                    method=RetrievalMethod.RERANKED,
                    index_revision=self._index_revision(row),
                    rank=index,
                    component_scores={
                        "lexical_rank": float(row["lexical_rank"]),
                        "lexical_relevance_score": float(row["lexical_relevance_score"]),
                        "lexical_only_degraded": 1.0,
                    },
                ),
            )
            for index, row in enumerate(rows[:limit], start=1)
        ]

    async def candidate_rerank_texts(self, candidates: list[QuranSearchCandidateV1]) -> list[str]:
        """Return normalized search text in candidate order for scoring only.

        These are index fields, not evidence. Canonical source text remains unavailable until
        exact hydration and therefore remains exclusively rendered by ``ExactRenderer``.
        """
        texts: list[str] = []
        async with self.pool.connection() as connection:
            for candidate in candidates:
                key = candidate.verse_keys[0]
                if candidate.index_revision.resource_alias.startswith("qul-tafsir-"):
                    cursor = await connection.execute(
                        """SELECT normalized_text AS normalized_arabic, '' AS normalized_translation
                           FROM tafsir_search_documents d
                           JOIN source_revisions rev ON rev.revision_id = d.revision_id
                           WHERE rev.active AND d.resource_alias = %s
                             AND d.chapter = %s AND d.verse_start <= %s AND d.verse_end >= %s
                           ORDER BY d.verse_start LIMIT 1""",
                        (
                            candidate.index_revision.resource_alias,
                            key.chapter,
                            key.verse,
                            key.verse,
                        ),
                    )
                else:
                    cursor = await connection.execute(
                        """SELECT normalized_arabic, normalized_translation
                           FROM quran_search_documents
                           WHERE chapter = %s AND verse = %s
                           ORDER BY document_id DESC LIMIT 1""",
                        (key.chapter, key.verse),
                    )
                row = await cursor.fetchone()
                # Cross-encoder input is a non-canonical index field, not source evidence.
                # A QUL passage may be a complete multi-page tafsir chapter (Ibn Kathir on
                # Al-Falaq is over 23k characters after ingestion).  Passing several such
                # records in one batch breaches the rerank endpoint's request limit and used
                # to turn otherwise answerable topic searches into a generic failed-safe
                # result.  Keep a bounded, deterministic prefix strictly for relevance
                # scoring; exact, untruncated source text remains fetched later and is only
                # emitted by ExactRenderer.
                text = f"{row['normalized_arabic']} {row['normalized_translation']}" if row else ""
                texts.append(text[:1600])
        return texts

    @staticmethod
    def _index_revision(row: dict[str, Any]) -> SourceRevisionIdentity:
        identity = (
            f"{row['arabic_revision_id']}:{row['translation_revision_id']}:{row['embedding_model']}"
        )
        digest = hashlib.sha256(
            (
                f"{row['arabic_revision_checksum']}:"
                f"{row['translation_revision_checksum']}:{row['embedding_model']}"
            ).encode()
        ).hexdigest()
        return SourceRevisionIdentity(
            source_id="quran-bilingual-search-index",
            resource_alias="quran-bilingual-search-index-v1",
            revision_id=identity,
            checksum=digest,
        )

    async def hybrid_search_components(
        self, *, query: str, query_embedding: list[float], limit: int
    ) -> dict[str, list[QuranSearchCandidateV1]]:
        pool = await self.hybrid_search(query=query, query_embedding=query_embedding, limit=60)
        lexical = sorted(
            (item for item in pool if "lexical_rank" in item.ranking.component_scores),
            key=lambda item: item.ranking.component_scores["lexical_rank"],
        )[:limit]
        vector = sorted(
            (item for item in pool if "vector_rank" in item.ranking.component_scores),
            key=lambda item: item.ranking.component_scores["vector_rank"],
        )[:limit]
        return {"lexical": lexical, "vector": vector, "fused": pool[:limit]}

    async def hydrate_candidates(
        self, candidates: list[QuranSearchCandidateV1]
    ) -> list[dict[str, Any]]:
        _, embedding_model_column = self._quran_embedding_columns
        result: list[dict[str, Any]] = []
        async with self.pool.connection() as connection:
            for candidate in candidates:
                key = candidate.verse_keys[0]
                cursor = await connection.execute(
                    f"""
                    SELECT ar.canonical_text AS arabic_text,
                           ar.content_hash AS arabic_content_hash,
                           ar.resource_alias AS arabic_resource_alias,
                           ar_rev.revision_id AS arabic_revision_id,
                           ar_rev.checksum_sha256 AS arabic_revision_checksum,
                           ar_res.source_id AS arabic_source_id,
                           ar_res.work_title AS arabic_work_title,
                           ar_res.author_translator_or_grader AS arabic_contributors,
                           ar_res.canonical_base_url AS arabic_base_url,
                           ar_res.attribution AS arabic_attribution,
                           tr.canonical_text AS translation_text,
                           tr.content_hash AS translation_content_hash,
                           tr.resource_alias AS translation_resource_alias,
                           tr_rev.revision_id AS translation_revision_id,
                           tr_rev.checksum_sha256 AS translation_revision_checksum,
                           tr_res.source_id AS translation_source_id,
                           tr_res.work_title AS translation_work_title,
                           tr_res.author_translator_or_grader AS translation_contributors,
                           tr_res.canonical_base_url AS translation_base_url,
                           tr_res.attribution AS translation_attribution,
                           sd.{embedding_model_column} AS embedding_model
                    FROM quran_canonical_text ar
                    JOIN source_revisions ar_rev ON ar_rev.revision_id = ar.revision_id
                    JOIN source_resources ar_res ON ar_res.resource_alias = ar.resource_alias
                    JOIN quran_canonical_text tr
                      ON tr.chapter = ar.chapter AND tr.verse = ar.verse
                    JOIN source_revisions tr_rev ON tr_rev.revision_id = tr.revision_id
                    JOIN source_resources tr_res ON tr_res.resource_alias = tr.resource_alias
                    JOIN quran_search_documents sd
                      ON sd.arabic_revision_id = ar.revision_id
                     AND sd.translation_revision_id = tr.revision_id
                     AND sd.chapter = ar.chapter AND sd.verse = ar.verse
                    WHERE ar_rev.active AND tr_rev.active
                      AND ar.resource_alias = 'tanzil-uthmani-v1.1'
                      AND tr.resource_alias = 'quranenc-rowwad-en-v1.0.19'
                      AND ar.chapter = %s AND ar.verse = %s
                    """,
                    (key.chapter, key.verse),
                )
                row = await cursor.fetchone()
                if row:
                    expected = self._index_revision(
                        {
                            **row,
                            "arabic_revision_id": row["arabic_revision_id"],
                            "translation_revision_id": row["translation_revision_id"],
                        }
                    )
                    if expected != candidate.index_revision:
                        continue
                    result.append({**row, "candidate": candidate, "verse_key": key})
        return result

    async def hydrate_references(self, verse_keys: list[VerseKey]) -> list[dict[str, Any]]:
        candidates: list[QuranSearchCandidateV1] = []
        if not verse_keys:
            return []
        _, embedding_model_column = self._quran_embedding_columns
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT sd.arabic_revision_id, sd.translation_revision_id,
                       sd.{embedding_model_column} AS embedding_model,
                       ar.checksum_sha256 AS arabic_revision_checksum,
                       tr.checksum_sha256 AS translation_revision_checksum
                FROM quran_search_documents sd
                JOIN source_revisions ar ON ar.revision_id = sd.arabic_revision_id
                JOIN source_revisions tr ON tr.revision_id = sd.translation_revision_id
                WHERE ar.active AND tr.active LIMIT 1
                """
            )
            row = await cursor.fetchone()
        if not row:
            return []
        index_revision = self._index_revision(row)
        for index, key in enumerate(verse_keys, start=1):
            candidates.append(
                QuranSearchCandidateV1(
                    candidate_id=f"D{index}",
                    verse_keys=[key],
                    index_revision=index_revision,
                    ranking=RetrievalProvenanceV1(
                        tool_id="resolve_quran_reference.v1",
                        method=RetrievalMethod.EXACT,
                        index_revision=None,
                        rank=index,
                        component_scores={},
                    ),
                )
            )
        return await self.hydrate_candidates(candidates)

    async def persist_plan(
        self,
        *,
        turn_id: UUID,
        conversation_id: UUID,
        correlation_id: UUID,
        plan: ResearchPlanV1,
        metadata: list[ProviderCompletion],
    ) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_plan_records (
                    turn_id, conversation_id, correlation_id, plan,
                    completion_metadata, validation_status
                ) VALUES (%s, %s, %s, %s, %s, 'VALID')
                ON CONFLICT (turn_id) DO UPDATE SET
                    plan = EXCLUDED.plan,
                    completion_metadata = EXCLUDED.completion_metadata,
                    validation_status = EXCLUDED.validation_status
                """,
                (
                    turn_id,
                    conversation_id,
                    correlation_id,
                    Jsonb(plan.model_dump(mode="json")),
                    Jsonb([item.model_dump(mode="json") for item in metadata]),
                ),
            )

    async def persist_response(
        self,
        *,
        turn_id: UUID,
        response: ResearchResponseV1,
        provenance: list[dict[str, Any]],
        gate_rejections_by_rule: dict[str, int] | None = None,
    ) -> None:
        durable_response = response_for_durable_storage(response)
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_response_records (
                    turn_id, response, evidence_provenance, gate_rejections_by_rule
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (turn_id) DO UPDATE SET
                    response = EXCLUDED.response,
                    evidence_provenance = EXCLUDED.evidence_provenance,
                    gate_rejections_by_rule = EXCLUDED.gate_rejections_by_rule
                """,
                (
                    turn_id,
                    Jsonb(durable_response.model_dump(mode="json")),
                    Jsonb(provenance),
                    Jsonb(gate_rejections_by_rule or {}),
                ),
            )

    async def persist_plan_and_response(
        self,
        *,
        turn_id: UUID,
        conversation_id: UUID,
        correlation_id: UUID,
        plan: ResearchPlanV1,
        metadata: list[ProviderCompletion],
        response: ResearchResponseV1,
        provenance: list[dict[str, Any]],
        gate_rejections_by_rule: dict[str, int],
    ) -> None:
        """Atomically record a planned turn, or record neither half of it."""

        durable_response = response_for_durable_storage(response)
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO research_plan_records (
                        turn_id, conversation_id, correlation_id, plan,
                        completion_metadata, validation_status
                    ) VALUES (%s, %s, %s, %s, %s, 'VALID')
                    ON CONFLICT (turn_id) DO UPDATE SET
                        plan = EXCLUDED.plan,
                        completion_metadata = EXCLUDED.completion_metadata,
                        validation_status = EXCLUDED.validation_status
                    """,
                    (
                        turn_id,
                        conversation_id,
                        correlation_id,
                        Jsonb(plan.model_dump(mode="json")),
                        Jsonb([item.model_dump(mode="json") for item in metadata]),
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO research_response_records (
                        turn_id, response, evidence_provenance, gate_rejections_by_rule
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (turn_id) DO UPDATE SET
                        response = EXCLUDED.response,
                        evidence_provenance = EXCLUDED.evidence_provenance,
                        gate_rejections_by_rule = EXCLUDED.gate_rejections_by_rule
                    """,
                    (
                        turn_id,
                        Jsonb(durable_response.model_dump(mode="json")),
                        Jsonb(provenance),
                        Jsonb(gate_rejections_by_rule),
                    ),
                )

    async def create_conversation(
        self,
        *,
        conversation_id: UUID,
        owner_key: str,
        title: str,
        locale: str,
    ) -> ConversationSummaryV1:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO conversations (conversation_id, owner_key, title, locale)
                VALUES (%s, %s, %s, %s)
                RETURNING conversation_id, title, locale, archived_at,
                          created_at, updated_at, 0::bigint AS turn_count
                """,
                (conversation_id, owner_key, title, locale),
            )
            row = await cursor.fetchone()
        assert row is not None
        return self._conversation_summary(row)

    async def list_conversations(
        self,
        *,
        owner_key: str,
        include_archived: bool,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[ConversationSummaryV1], str | None]:
        values: list[Any] = [owner_key, include_archived]
        cursor_clause = ""
        if cursor is not None:
            cursor_timestamp, cursor_id = _decode_cursor(cursor)
            cursor_clause = "AND (c.updated_at, c.conversation_id) < (%s, %s)"
            values.extend([cursor_timestamp, cursor_id])
        values.append(limit + 1)
        async with self.pool.connection() as connection:
            result = await connection.execute(
                f"""
                SELECT c.conversation_id, c.title, c.locale, c.archived_at,
                       c.created_at, c.updated_at, count(t.turn_id)::bigint AS turn_count
                FROM conversations c
                LEFT JOIN conversation_turns t ON t.conversation_id = c.conversation_id
                WHERE c.owner_key = %s AND (%s OR c.archived_at IS NULL)
                {cursor_clause}
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC, c.conversation_id DESC
                LIMIT %s
                """,
                values,
            )
            rows = list(await result.fetchall())
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = (
            _encode_cursor(rows[-1]["updated_at"], rows[-1]["conversation_id"])
            if has_more and rows
            else None
        )
        return [self._conversation_summary(row) for row in rows], next_cursor

    async def get_conversation(
        self,
        *,
        conversation_id: UUID,
        owner_key: str,
        turn_limit: int = 50,
        turn_cursor: str | None = None,
    ) -> tuple[ConversationDetailV1 | None, str | None]:
        turn_values: list[Any] = [conversation_id, owner_key]
        turn_clause = ""
        if turn_cursor is not None:
            timestamp, turn_id = _decode_cursor(turn_cursor)
            turn_clause = "AND (created_at, turn_id) > (%s, %s)"
            turn_values.extend([timestamp, turn_id])
        turn_values.append(turn_limit + 1)
        async with self.pool.connection() as connection:
            summary_cursor = await connection.execute(
                """
                SELECT c.conversation_id, c.title, c.locale, c.archived_at,
                       c.created_at, c.updated_at, count(t.turn_id)::bigint AS turn_count
                FROM conversations c
                LEFT JOIN conversation_turns t ON t.conversation_id = c.conversation_id
                WHERE c.conversation_id = %s AND c.owner_key = %s
                GROUP BY c.conversation_id
                """,
                (conversation_id, owner_key),
            )
            summary = await summary_cursor.fetchone()
            if summary is None:
                return None, None
            turns_cursor = await connection.execute(
                f"""
                SELECT turn_id, user_message, locale, response, trace_events, created_at
                FROM conversation_turns
                WHERE conversation_id = %s AND owner_key = %s {turn_clause}
                ORDER BY created_at ASC, turn_id ASC
                LIMIT %s
                """,
                turn_values,
            )
            rows = list(await turns_cursor.fetchall())
        has_more = len(rows) > turn_limit
        rows = rows[:turn_limit]
        next_cursor = (
            _encode_cursor(rows[-1]["created_at"], rows[-1]["turn_id"])
            if has_more and rows
            else None
        )
        base = self._conversation_summary(summary)
        return (
            ConversationDetailV1(
                **base.model_dump(),
                turns=[self._conversation_turn(row) for row in rows],
            ),
            next_cursor,
        )

    async def update_conversation(
        self,
        *,
        conversation_id: UUID,
        owner_key: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> ConversationSummaryV1 | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE conversations
                SET title = COALESCE(%s, title),
                    archived_at = CASE
                        WHEN %s::boolean IS NULL THEN archived_at
                        WHEN %s THEN COALESCE(archived_at, now())
                        ELSE NULL
                    END,
                    updated_at = now()
                WHERE conversation_id = %s AND owner_key = %s
                RETURNING conversation_id, title, locale, archived_at,
                          created_at, updated_at,
                          (SELECT count(*) FROM conversation_turns
                           WHERE conversation_id = conversations.conversation_id)::bigint
                           AS turn_count
                """,
                (title, archived, archived, conversation_id, owner_key),
            )
            row = await cursor.fetchone()
        return self._conversation_summary(row) if row is not None else None

    async def delete_conversation(self, *, conversation_id: UUID, owner_key: str) -> bool:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM conversations WHERE conversation_id = %s AND owner_key = %s",
                (conversation_id, owner_key),
            )
        return cursor.rowcount == 1

    async def get_turn_by_idempotency_key(
        self,
        *,
        conversation_id: UUID,
        owner_key: str,
        idempotency_key: str,
    ) -> tuple[str, ConversationTurnV1] | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT request_hash, turn_id, user_message, locale, response,
                       trace_events, created_at
                FROM conversation_turns
                WHERE conversation_id = %s AND owner_key = %s AND idempotency_key = %s
                """,
                (conversation_id, owner_key, idempotency_key),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return row["request_hash"], self._conversation_turn(row)

    async def persist_conversation_turn(
        self,
        *,
        conversation_id: UUID,
        owner_key: str,
        idempotency_key: str,
        request_hash: str,
        user_message: str,
        locale: str,
        response: ResearchResponseV1,
        trace_events: list[TraceEventV1],
    ) -> ConversationTurnV1:
        durable_response = response_for_durable_storage(response)
        async with self.pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO conversation_turns (
                        turn_id, conversation_id, owner_key, idempotency_key,
                        request_hash, user_message, locale, response, trace_events
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (conversation_id, idempotency_key) DO NOTHING
                    """,
                    (
                        response.turn_id,
                        conversation_id,
                        owner_key,
                        idempotency_key,
                        request_hash,
                        user_message,
                        locale,
                        Jsonb(durable_response.model_dump(mode="json")),
                        Jsonb([event.model_dump(mode="json") for event in trace_events]),
                    ),
                )
                await connection.execute(
                    """
                    UPDATE conversations SET updated_at = now()
                    WHERE conversation_id = %s AND owner_key = %s
                    """,
                    (conversation_id, owner_key),
                )
                cursor = await connection.execute(
                    """
                    SELECT request_hash, turn_id, user_message, locale, response,
                           trace_events, created_at
                    FROM conversation_turns
                    WHERE conversation_id = %s AND owner_key = %s AND idempotency_key = %s
                    """,
                    (conversation_id, owner_key, idempotency_key),
                )
                row = await cursor.fetchone()
        assert row is not None
        if row["request_hash"] != request_hash:
            raise ValueError("Idempotency key was reused with a different request")
        return self._conversation_turn(row)

    @staticmethod
    def _conversation_summary(row: dict[str, Any]) -> ConversationSummaryV1:
        return ConversationSummaryV1(
            conversation_id=row["conversation_id"],
            title=row["title"],
            locale=row["locale"],
            archived=row["archived_at"] is not None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            turn_count=row["turn_count"],
        )

    @staticmethod
    def _conversation_turn(row: dict[str, Any]) -> ConversationTurnV1:
        return ConversationTurnV1(
            turn_id=row["turn_id"],
            user_message=row["user_message"],
            locale=row["locale"],
            response=ResearchResponseV1.model_validate(row["response"]),
            created_at=row["created_at"],
            trace_events=[
                TraceEventV1.model_validate(item) for item in row.get("trace_events", [])
            ],
        )
