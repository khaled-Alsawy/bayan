BEGIN;

-- QUL's curated topic hierarchy is a retrieval-only third arm.  It deliberately has no
-- relationship to quran_search_documents or their embeddings: topic labels are ranked separately
-- and expand only to opaque verse locations before rank-space fusion.
CREATE TABLE IF NOT EXISTS qul_topic_documents (
    revision_id uuid NOT NULL REFERENCES source_revisions(revision_id),
    resource_alias text NOT NULL REFERENCES source_resources(resource_alias),
    topic_id integer NOT NULL,
    name text NOT NULL,
    hierarchy_path text NOT NULL,
    arabic_name text,
    description text,
    thematic boolean NOT NULL,
    ontology boolean NOT NULL,
    label text NOT NULL,
    label_sha256 text NOT NULL,
    embedding vector(1024),
    embedding_model text,
    embedding_deployment_id text,
    embedded_at timestamptz,
    PRIMARY KEY (revision_id, topic_id)
);

CREATE TABLE IF NOT EXISTS qul_topic_ayah_mappings (
    revision_id uuid NOT NULL,
    topic_id integer NOT NULL,
    chapter smallint NOT NULL CHECK (chapter BETWEEN 1 AND 114),
    verse smallint NOT NULL CHECK (verse > 0),
    PRIMARY KEY (revision_id, topic_id, chapter, verse),
    FOREIGN KEY (revision_id, topic_id)
        REFERENCES qul_topic_documents(revision_id, topic_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS qul_topic_documents_embedding_hnsw_idx
ON qul_topic_documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS qul_topic_ayah_mappings_verse_idx
ON qul_topic_ayah_mappings(revision_id, chapter, verse);

COMMIT;
