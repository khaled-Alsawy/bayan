BEGIN;

-- The original verse-level vector remains intact for A/B measurement. This separate,
-- index-only representation stores a context-plus-tafsir embedding and its input fingerprint.
ALTER TABLE quran_search_documents
ADD COLUMN IF NOT EXISTS enriched_document_text text,
ADD COLUMN IF NOT EXISTS enriched_document_sha256 text,
ADD COLUMN IF NOT EXISTS enriched_tafsir_covered boolean NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS enriched_embedding vector(1024),
ADD COLUMN IF NOT EXISTS enriched_embedding_model text,
ADD COLUMN IF NOT EXISTS enriched_embedding_deployment_id text,
ADD COLUMN IF NOT EXISTS enriched_embedded_at timestamptz;

CREATE INDEX IF NOT EXISTS quran_search_documents_enriched_embedding_hnsw_idx
ON quran_search_documents USING hnsw (enriched_embedding vector_cosine_ops);

COMMIT;
