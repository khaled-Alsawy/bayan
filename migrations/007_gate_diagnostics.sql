BEGIN;

-- Rule names and counts are durable observability data.  Never store model output,
-- quotations, or offending spans here: those remain process-local and are not client data.
ALTER TABLE research_response_records
ADD COLUMN IF NOT EXISTS gate_rejections_by_rule jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMIT;
