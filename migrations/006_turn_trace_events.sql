BEGIN;

ALTER TABLE conversation_turns
ADD COLUMN IF NOT EXISTS trace_events jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMIT;
