BEGIN;

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_key text NOT NULL DEFAULT 'local',
    title text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 120),
    locale text NOT NULL CHECK (locale IN ('ar', 'en')),
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_owner_updated_idx
ON conversations(owner_key, archived_at, updated_at DESC, conversation_id DESC);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    owner_key text NOT NULL DEFAULT 'local',
    idempotency_key text NOT NULL CHECK (char_length(idempotency_key) BETWEEN 8 AND 128),
    request_hash text NOT NULL,
    user_message text NOT NULL CHECK (char_length(user_message) BETWEEN 2 AND 2000),
    locale text NOT NULL CHECK (locale IN ('ar', 'en')),
    response jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS conversation_turns_order_idx
ON conversation_turns(conversation_id, created_at ASC, turn_id ASC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'conversation_turns_no_qf_raw_text'
    ) THEN
        ALTER TABLE conversation_turns
        ADD CONSTRAINT conversation_turns_no_qf_raw_text CHECK (
            NOT jsonb_path_exists(
                response,
                '$.blocks[*] ? (@.block_type == "SOURCE" && @.citation.source_kind == "TAFSIR" && @.citation.source_revision.source_id == "quran_foundation" && @.exact_text != "[SOURCE_TEXT_NOT_STORED_POLICY]")'
            )
        );
    END IF;
END $$;

COMMIT;
