BEGIN;

CREATE TABLE IF NOT EXISTS legacy_research_plan_records (
    turn_id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL,
    correlation_id uuid NOT NULL,
    plan jsonb NOT NULL,
    completion_metadata jsonb NOT NULL,
    validation_status text NOT NULL,
    original_created_at timestamptz NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now(),
    archive_reason text NOT NULL
);

CREATE TABLE IF NOT EXISTS legacy_research_response_records (
    turn_id uuid PRIMARY KEY,
    response jsonb NOT NULL,
    evidence_provenance jsonb NOT NULL,
    original_created_at timestamptz NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now(),
    archive_reason text NOT NULL
);

INSERT INTO legacy_research_plan_records (
    turn_id, conversation_id, correlation_id, plan, completion_metadata,
    validation_status, original_created_at, archive_reason
)
SELECT turn_id, conversation_id, correlation_id, plan, completion_metadata,
       validation_status, created_at, 'PRE_VERSION_2_1_CONTRACT_DRIFT'
FROM research_plan_records
WHERE plan ? 'scope_status' OR plan ? 'tool_calls'
ON CONFLICT (turn_id) DO NOTHING;

DELETE FROM research_plan_records
WHERE plan ? 'scope_status' OR plan ? 'tool_calls';

INSERT INTO legacy_research_response_records (
    turn_id, response, evidence_provenance, original_created_at, archive_reason
)
SELECT turn_id, response, evidence_provenance, created_at,
       'PRE_VERSION_2_1_CONTRACT_DRIFT'
FROM research_response_records
WHERE response->>'status' = 'ANSWERED'
   OR response ? 'introduction'
   OR response ? 'conclusion'
   OR response ? 'source_blocks'
ON CONFLICT (turn_id) DO NOTHING;

DELETE FROM research_response_records
WHERE response->>'status' = 'ANSWERED'
   OR response ? 'introduction'
   OR response ? 'conclusion'
   OR response ? 'source_blocks';

COMMIT;
