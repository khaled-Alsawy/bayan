from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from islamic_research_ai.contracts import (
    EffectiveDeploymentIdentityV1,
    ProviderCompletion,
    QuranSearchCandidateV1,
    RetrievalMethod,
    RetrievalProvenanceV1,
    SourceRevisionIdentity,
    VerseKey,
)
from islamic_research_ai.ingestion.qul_topics import (
    QUL_TOPIC_RESOURCE_ALIAS,
    QulTopicReembedding,
    QulTopicShapeError,
    build_topic_documents,
    load_qul_topics,
    parse_ayahs,
)
from islamic_research_ai.settings import Settings
from islamic_research_ai.tools import PostgresQuranSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPIC_DB = PROJECT_ROOT / "data" / "raw" / "qul" / "ayah-topics.db"


def _completion() -> ProviderCompletion:
    return ProviderCompletion(
        effective_deployment=EffectiveDeploymentIdentityV1(
            deployment_id="cloudflare-workers-ai:direct:@cf/baai/bge-m3",
            gateway_provider_id="cloudflare",
            effective_provider_id="cloudflare",
            model_id="@cf/baai/bge-m3",
            resolved_model_version_or_name="@cf/baai/bge-m3",
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


def _quran_candidate(reference: str, *, rank: int = 1) -> QuranSearchCandidateV1:
    chapter, verse = (int(part) for part in reference.split(":"))
    revision = SourceRevisionIdentity(
        source_id="quran-bilingual-search-index",
        resource_alias="quran-bilingual-search-index-v1",
        revision_id="quran-index",
        checksum="quran-checksum",
    )
    return QuranSearchCandidateV1(
        candidate_id="C1",
        verse_keys=[VerseKey(chapter=chapter, verse=verse)],
        index_revision=revision,
        ranking=RetrievalProvenanceV1(
            tool_id="search_quran_topics.v1",
            method=RetrievalMethod.HYBRID,
            index_revision=revision,
            rank=rank,
            component_scores={"rrf_score": 0.01},
        ),
    )


def test_hierarchy_label_disambiguates_children_and_parents_with_inheritance_path() -> None:
    topics = load_qul_topics(TOPIC_DB)
    documents = {
        document.topic.topic_id: document
        for document in build_topic_documents(topics)
    }
    document = documents[757]

    assert len(topics) == 2512
    assert sum(bool(topic.ayahs) for topic in topics) == 2330
    assert document.hierarchy_path == (
        "Doctraine > Relationship with people > Financial > Inheritance > apportionment to "
        "> children and parents"
    )
    assert document.label.startswith(document.hierarchy_path)
    assert document.topic.ayahs == ((4, 11),)


def test_ayah_parser_accepts_the_export_format_and_rejects_malformed_references() -> None:
    assert parse_ayahs("2:255, 17:23,2:255", topic_id=1) == ((2, 255), (17, 23))
    with pytest.raises(QulTopicShapeError, match="malformed ayah reference"):
        parse_ayahs("2:255, not-a-reference, 115:1", topic_id=1)


@pytest.mark.asyncio
async def test_topic_only_verse_survives_rank_space_fusion() -> None:
    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        async def embed(self, _: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            return [[0.0]], _completion()

    class Repository:
        async def hybrid_search(self, **_: object) -> list[QuranSearchCandidateV1]:
            return [_quran_candidate("2:153", rank=20)]

        async def search_tafsir_tier(self, **_: object) -> list[dict[str, object]]:
            return []

        async def search_qul_topic_arm(self, **_: object) -> list[dict[str, object]]:
            return [
                {
                    "revision_id": "topic-revision",
                    "revision_checksum": "topic-checksum",
                    "topic_id": 757,
                    "name": "children and parents",
                    "hierarchy_path": "Inheritance > children and parents",
                    "label": "Inheritance > children and parents",
                    "embedding_model": "@cf/baai/bge-m3",
                    "topic_rank": 1,
                    "topic_score": 0.91,
                    "chapter": 4,
                    "verse": 11,
                }
            ]

    source = PostgresQuranSource(
        Repository(),  # type: ignore[arg-type]
        Embeddings(),  # type: ignore[arg-type]
        qul_topic_arm_enabled=True,
    )
    result = await source.search("inheritance shares for children", limit=1)

    assert [
        (candidate.verse_keys[0].chapter, candidate.verse_keys[0].verse) for candidate in result
    ] == [(4, 11)]
    assert result[0].ranking.component_scores["topic_arm_rrf"] == pytest.approx(1 / 61)
    trace = source.last_fusion_traces[0]["arms"]["qul_topics"]
    assert trace["expanded_verse_count"] == 1  # type: ignore[index]


@pytest.mark.asyncio
async def test_topic_arm_off_is_byte_identical_and_does_not_query_the_topic_index() -> None:
    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        async def embed(self, _: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            return [[0.0]], _completion()

    class Repository:
        def __init__(self) -> None:
            self.topic_calls = 0

        async def hybrid_search(self, **_: object) -> list[QuranSearchCandidateV1]:
            return [_quran_candidate("17:23")]

        async def search_tafsir_tier(self, **_: object) -> list[dict[str, object]]:
            return []

        async def search_qul_topic_arm(self, **_: object) -> list[dict[str, object]]:
            self.topic_calls += 1
            return []

    repository = Repository()
    baseline = PostgresQuranSource(repository, Embeddings())  # type: ignore[arg-type]
    explicitly_off = PostgresQuranSource(  # type: ignore[arg-type]
        repository, Embeddings(), qul_topic_arm_enabled=False
    )
    baseline_result = await baseline.search("kindness to parents", limit=3)
    off_result = await explicitly_off.search("kindness to parents", limit=3)

    assert [item.model_dump(mode="json") for item in off_result] == [
        item.model_dump(mode="json") for item in baseline_result
    ]
    assert repository.topic_calls == 0
    assert "qul_topics" not in explicitly_off.last_fusion_traces[0]["arms"]
    assert Settings.model_fields["qul_topic_arm_enabled"].default is False
    assert Settings.model_fields["qul_topic_arm_limit"].default == 20


@pytest.mark.asyncio
async def test_topic_label_reranker_reorders_without_widening_the_fused_pool() -> None:
    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        async def embed(self, _: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            return [[0.0]], _completion()

    class Repository:
        def __init__(self) -> None:
            self.similarity_calls = 0

        async def hybrid_search(self, **_: object) -> list[QuranSearchCandidateV1]:
            return [
                _quran_candidate("2:153", rank=1),
                _quran_candidate("2:154", rank=2),
                _quran_candidate("2:155", rank=3),
            ]

        async def search_tafsir_tier(self, **_: object) -> list[dict[str, object]]:
            return []

        async def qul_topic_similarity_for_candidates(
            self, **_: object
        ) -> list[dict[str, object]]:
            self.similarity_calls += 1
            return [
                {"chapter": 2, "verse": 154, "topic_similarity": 0.9},
                {"chapter": 2, "verse": 155, "topic_similarity": 0.5},
            ]

    repository = Repository()
    fused = await PostgresQuranSource(repository, Embeddings()).search("patience", limit=3)  # type: ignore[arg-type]
    reranked_source = PostgresQuranSource(  # type: ignore[arg-type]
        repository,
        Embeddings(),
        qul_topic_rerank_enabled=True,
    )
    reranked = await reranked_source.search("patience", limit=3)

    fused_locations = [(item.verse_keys[0].chapter, item.verse_keys[0].verse) for item in fused]
    reranked_locations = [
        (item.verse_keys[0].chapter, item.verse_keys[0].verse) for item in reranked
    ]
    assert fused_locations == [(2, 153), (2, 154), (2, 155)]
    assert set(reranked_locations) == set(fused_locations)
    assert reranked_locations == [(2, 154), (2, 153), (2, 155)]
    assert [item.candidate_id for item in reranked] == ["C1", "C2", "C3"]
    assert [item.ranking.rank for item in reranked] == [1, 2, 3]
    assert reranked[1].ranking.component_scores["qul_topic_similarity"] == 0.0
    assert reranked[1].ranking.component_scores["qul_topic_rank"] == 3.0
    assert reranked[0].ranking.component_scores["qul_topic_blend_score"] == pytest.approx(
        1.0 / 12 + 1.0 / 11
    )
    fused_scores = {
        (item.verse_keys[0].chapter, item.verse_keys[0].verse): item.ranking.component_scores[
            "fused_arm_score"
        ]
        for item in fused
    }
    assert {
        (item.verse_keys[0].chapter, item.verse_keys[0].verse): item.ranking.component_scores[
            "fused_arm_score"
        ]
        for item in reranked
    } == fused_scores
    assert repository.similarity_calls == 1
    assert reranked_source.last_qul_topic_rerank == {
        "phase": "completed",
        "candidate_count": 3,
        "labeled_candidate_count": 2,
        "top_fused_candidate_positions_moved": 1,
    }
    assert Settings.model_fields["qul_topic_rerank_enabled"].default is False


@pytest.mark.asyncio
async def test_topic_label_reranker_failure_retains_fused_order() -> None:
    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        async def embed(self, _: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            return [[0.0]], _completion()

    class Repository:
        async def hybrid_search(self, **_: object) -> list[QuranSearchCandidateV1]:
            return [_quran_candidate("2:153", rank=1), _quran_candidate("2:154", rank=2)]

        async def search_tafsir_tier(self, **_: object) -> list[dict[str, object]]:
            return []

        async def qul_topic_similarity_for_candidates(
            self, **_: object
        ) -> list[dict[str, object]]:
            raise RuntimeError("curated topic index unavailable")

    source = PostgresQuranSource(  # type: ignore[arg-type]
        Repository(),
        Embeddings(),
        qul_topic_rerank_enabled=True,
    )
    result = await source.search("patience", limit=2)

    assert [(item.verse_keys[0].chapter, item.verse_keys[0].verse) for item in result] == [
        (2, 153),
        (2, 154),
    ]
    assert source.last_qul_topic_rerank == {
        "phase": "degraded",
        "reason": "RuntimeError",
        "safe_detail": "curated topic index unavailable",
    }


def _write_minimal_topic_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE topics (
                topic_id INTEGER PRIMARY KEY, name TEXT, arabic_name TEXT, parent_id INTEGER,
                thematic_parent_id INTEGER, ontology_parent_id INTEGER, description TEXT,
                wiki_link TEXT, thematic INTEGER, ontology INTEGER, ayahs TEXT, related_topics TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO topics VALUES "
            "(1, 'parenting', NULL, NULL, NULL, NULL, NULL, NULL, 1, 0, '17:23', NULL)"
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_topic_backfill_uses_durable_per_topic_checkpoints_on_rerun(tmp_path: Path) -> None:
    raw_qul = tmp_path / "qul"
    raw_qul.mkdir()
    _write_minimal_topic_db(raw_qul / "ayah-topics.db")
    revision_id = uuid4()

    class Repository:
        def __init__(self) -> None:
            self.documents: list[object] = []
            self.embedded: set[int] = set()
            self.stored_batches: list[list[tuple[object, ...]]] = []

        async def upsert_resource(self, _: object) -> None:
            return None

        async def upsert_revision(self, **_: object) -> UUID:
            return revision_id

        async def record_activation(self, _: object) -> None:
            return None

        async def qul_topic_document_counts(self, _: UUID) -> tuple[int, int]:
            return len(self.documents), len(self.embedded)

        async def qul_topic_document_hashes(self, _: UUID) -> dict[int, str]:
            return {
                document.topic.topic_id: document.label_sha256  # type: ignore[attr-defined]
                for document in self.documents
            }

        async def replace_qul_topic_documents(self, **kwargs: object) -> None:
            self.documents = list(kwargs["rows"])  # type: ignore[arg-type]

        async def unembedded_qul_topic_documents(self, _: UUID) -> list[dict[str, object]]:
            return [
                {
                    "topic_id": document.topic.topic_id,  # type: ignore[attr-defined]
                    "label": document.label,  # type: ignore[attr-defined]
                    "label_sha256": document.label_sha256,  # type: ignore[attr-defined]
                }
                for document in self.documents
                if document.topic.topic_id not in self.embedded  # type: ignore[attr-defined]
            ]

        async def store_qul_topic_embeddings(self, rows: list[tuple[object, ...]]) -> None:
            self.stored_batches.append(rows)
            self.embedded.update(int(row[4]) for row in rows)

    class Embeddings:
        model = "@cf/baai/bge-m3"
        deployment_id = "cloudflare-workers-ai:direct:@cf/baai/bge-m3"

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts: list[str]) -> tuple[list[list[float]], ProviderCompletion]:
            self.calls += 1
            return [[0.0] for _ in texts], _completion()

    class Policy:
        resources = {QUL_TOPIC_RESOURCE_ALIAS: {"rights": {}}}

        def authorize(self, *_: object, **__: object) -> object:
            return object()

    repository = Repository()
    embeddings = Embeddings()
    settings = Settings(raw_data_dir=tmp_path)
    first = await QulTopicReembedding(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        policy=Policy(),  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
    ).run()
    second = await QulTopicReembedding(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        policy=Policy(),  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
    ).run()

    assert first["provider_call_count"] == 1
    assert second["pending_topic_count"] == 0
    assert second["provider_call_count"] == 0
    assert embeddings.calls == 1
    assert len(repository.stored_batches) == 1
