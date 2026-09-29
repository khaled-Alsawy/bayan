BEGIN;

-- WP-0 (Conversational Quality Technical Plan V3): QUL tafsir corpus.
--
-- Evidence tier (Ibn Kathir, Al-Qurtubi, Al-Baghawi): passage-range text, displayed and cited
-- exactly like Quran Foundation Tafsir, but sourced from a locally-owned export with full
-- Quran coverage instead of a two-surah prelive API.
--
-- Search tier (Al-Mukhtasar): short, modern-Arabic, per-ayah commentary, embedded so a topical
-- Arabic query can match tafsir wording directly instead of only Quranic wording (F-07).

CREATE TABLE IF NOT EXISTS tafsir_passages (
    revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    chapter smallint NOT NULL CHECK (chapter BETWEEN 1 AND 114),
    verse_start smallint NOT NULL CHECK (verse_start > 0),
    verse_end smallint NOT NULL CHECK (verse_end >= verse_start),
    passage_text text NOT NULL,
    hadith_references jsonb NOT NULL DEFAULT '[]'::jsonb,
    content_hash text NOT NULL,
    PRIMARY KEY (revision_id, chapter, verse_start)
);

CREATE INDEX IF NOT EXISTS tafsir_passages_range_idx
ON tafsir_passages(resource_alias, chapter, verse_start, verse_end);

CREATE TABLE IF NOT EXISTS tafsir_search_documents (
    document_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    chapter smallint NOT NULL CHECK (chapter BETWEEN 1 AND 114),
    verse_start smallint NOT NULL CHECK (verse_start > 0),
    verse_end smallint NOT NULL CHECK (verse_end >= verse_start),
    normalized_text text NOT NULL,
    combined_search_text text NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', combined_search_text)) STORED,
    embedding vector(1024),
    embedding_model text,
    embedding_deployment_id text,
    embedded_at timestamptz,
    UNIQUE(revision_id, chapter, verse_start)
);

CREATE INDEX IF NOT EXISTS tafsir_search_documents_fts_idx
ON tafsir_search_documents USING gin(search_vector);

CREATE INDEX IF NOT EXISTS tafsir_search_documents_trgm_idx
ON tafsir_search_documents USING gin(normalized_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS tafsir_search_documents_embedding_hnsw_idx
ON tafsir_search_documents USING hnsw (embedding vector_cosine_ops);

COMMIT;
