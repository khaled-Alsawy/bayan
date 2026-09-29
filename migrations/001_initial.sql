BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS source_resources (
    resource_alias text PRIMARY KEY,
    source_id text NOT NULL,
    work_title text NOT NULL,
    language text NOT NULL,
    author_translator_or_grader jsonb NOT NULL DEFAULT '[]'::jsonb,
    canonical_base_url text,
    attribution jsonb NOT NULL,
    rights_policy jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_revisions (
    revision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    version text NOT NULL,
    checksum_sha256 text NOT NULL,
    acquired_at timestamptz NOT NULL,
    record_count integer NOT NULL CHECK (record_count > 0),
    source_url text NOT NULL,
    active boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(resource_alias, version, checksum_sha256)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_revision_per_resource
ON source_revisions(resource_alias) WHERE active;

CREATE TABLE IF NOT EXISTS source_activation_records (
    activation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    operation text NOT NULL,
    rights_decision text NOT NULL,
    override_type text NOT NULL,
    effective_permission boolean NOT NULL,
    environment text NOT NULL,
    purpose text NOT NULL,
    reason text NOT NULL,
    effective_deployment jsonb,
    recorded_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quran_canonical_text (
    revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    chapter smallint NOT NULL CHECK (chapter BETWEEN 1 AND 114),
    verse smallint NOT NULL CHECK (verse > 0),
    language text NOT NULL,
    canonical_text text NOT NULL,
    content_hash text NOT NULL,
    source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY(revision_id, chapter, verse)
);

CREATE TABLE IF NOT EXISTS quran_search_documents (
    document_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    arabic_revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    translation_revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    chapter smallint NOT NULL CHECK (chapter BETWEEN 1 AND 114),
    verse smallint NOT NULL CHECK (verse > 0),
    normalized_arabic text NOT NULL,
    normalized_translation text NOT NULL,
    combined_search_text text NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', combined_search_text)) STORED,
    embedding vector(1024),
    embedding_model text,
    embedding_deployment_id text,
    embedded_at timestamptz,
    UNIQUE(arabic_revision_id, translation_revision_id, chapter, verse)
);

CREATE INDEX IF NOT EXISTS quran_search_documents_fts_idx
ON quran_search_documents USING gin(search_vector);

CREATE INDEX IF NOT EXISTS quran_search_documents_ar_trgm_idx
ON quran_search_documents USING gin(normalized_arabic gin_trgm_ops);

CREATE INDEX IF NOT EXISTS quran_search_documents_en_trgm_idx
ON quran_search_documents USING gin(normalized_translation gin_trgm_ops);

CREATE INDEX IF NOT EXISTS quran_search_documents_embedding_hnsw_idx
ON quran_search_documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS research_plan_records (
    turn_id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL,
    correlation_id uuid NOT NULL,
    plan jsonb NOT NULL,
    completion_metadata jsonb NOT NULL,
    validation_status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_response_records (
    turn_id uuid PRIMARY KEY,
    response jsonb NOT NULL,
    evidence_provenance jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

COMMIT;
