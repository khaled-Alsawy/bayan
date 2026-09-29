BEGIN;

CREATE TABLE IF NOT EXISTS data_remediation_records (
    remediation_id text PRIMARY KEY,
    affected_rows integer NOT NULL CHECK (affected_rows >= 0),
    affected_blocks integer NOT NULL CHECK (affected_blocks >= 0),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    applied_at timestamptz NOT NULL DEFAULT now()
);

DO $$
DECLARE
    current_rows integer := 0;
    current_blocks integer := 0;
    legacy_rows integer := 0;
    legacy_blocks integer := 0;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM data_remediation_records
        WHERE remediation_id = '003_redact_qf_tafsir_source_text'
    ) THEN
        SELECT count(DISTINCT r.turn_id), count(*)
        INTO current_rows, current_blocks
        FROM research_response_records AS r
        CROSS JOIN LATERAL jsonb_array_elements(r.response->'blocks') AS block
        WHERE block->>'block_type' = 'SOURCE'
          AND block->'citation'->>'source_kind' = 'TAFSIR'
          AND block->'citation'->'source_revision'->>'source_id' = 'quran_foundation'
          AND block->>'exact_text' <> '[SOURCE_TEXT_NOT_STORED_POLICY]';

        UPDATE research_response_records AS r
        SET response = jsonb_set(
            r.response,
            '{blocks}',
            (
                SELECT jsonb_agg(
                    CASE
                        WHEN block->>'block_type' = 'SOURCE'
                         AND block->'citation'->>'source_kind' = 'TAFSIR'
                         AND block->'citation'->'source_revision'->>'source_id' =
                             'quran_foundation'
                        THEN jsonb_set(
                            block,
                            '{exact_text}',
                            to_jsonb('[SOURCE_TEXT_NOT_STORED_POLICY]'::text)
                        )
                        ELSE block
                    END
                    ORDER BY ordinal
                )
                FROM jsonb_array_elements(r.response->'blocks')
                    WITH ORDINALITY AS values(block, ordinal)
            )
        )
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_array_elements(r.response->'blocks') AS block
            WHERE block->>'block_type' = 'SOURCE'
              AND block->'citation'->>'source_kind' = 'TAFSIR'
              AND block->'citation'->'source_revision'->>'source_id' = 'quran_foundation'
              AND block->>'exact_text' <> '[SOURCE_TEXT_NOT_STORED_POLICY]'
        );

        IF to_regclass('public.legacy_research_response_records') IS NOT NULL THEN
            SELECT count(DISTINCT r.turn_id), count(*)
            INTO legacy_rows, legacy_blocks
            FROM legacy_research_response_records AS r
            CROSS JOIN LATERAL jsonb_array_elements(r.response->'blocks') AS block
            WHERE block->>'block_type' = 'SOURCE'
              AND block->'citation'->>'source_kind' = 'TAFSIR'
              AND block->'citation'->'source_revision'->>'source_id' = 'quran_foundation'
              AND block->>'exact_text' <> '[SOURCE_TEXT_NOT_STORED_POLICY]';

            UPDATE legacy_research_response_records AS r
            SET response = jsonb_set(
                r.response,
                '{blocks}',
                (
                    SELECT jsonb_agg(
                        CASE
                            WHEN block->>'block_type' = 'SOURCE'
                             AND block->'citation'->>'source_kind' = 'TAFSIR'
                             AND block->'citation'->'source_revision'->>'source_id' =
                                 'quran_foundation'
                            THEN jsonb_set(
                                block,
                                '{exact_text}',
                                to_jsonb('[SOURCE_TEXT_NOT_STORED_POLICY]'::text)
                            )
                            ELSE block
                        END
                        ORDER BY ordinal
                    )
                    FROM jsonb_array_elements(r.response->'blocks')
                        WITH ORDINALITY AS values(block, ordinal)
                )
            )
            WHERE EXISTS (
                SELECT 1
                FROM jsonb_array_elements(r.response->'blocks') AS block
                WHERE block->>'block_type' = 'SOURCE'
                  AND block->'citation'->>'source_kind' = 'TAFSIR'
                  AND block->'citation'->'source_revision'->>'source_id' =
                      'quran_foundation'
                  AND block->>'exact_text' <> '[SOURCE_TEXT_NOT_STORED_POLICY]'
            );
        END IF;

        INSERT INTO data_remediation_records (
            remediation_id, affected_rows, affected_blocks, details
        ) VALUES (
            '003_redact_qf_tafsir_source_text',
            current_rows + legacy_rows,
            current_blocks + legacy_blocks,
            jsonb_build_object(
                'current_rows', current_rows,
                'current_blocks', current_blocks,
                'legacy_rows', legacy_rows,
                'legacy_blocks', legacy_blocks
            )
        );
    END IF;
END $$;

-- Defense in depth: application redaction remains the primary boundary, while
-- these checks make a future regression fail closed at the database boundary.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_response_records_no_qf_raw_text'
    ) THEN
        ALTER TABLE research_response_records
        ADD CONSTRAINT research_response_records_no_qf_raw_text CHECK (
            NOT jsonb_path_exists(
                response,
                '$.blocks[*] ? (@.block_type == "SOURCE" && @.citation.source_kind == "TAFSIR" && @.citation.source_revision.source_id == "quran_foundation" && @.exact_text != "[SOURCE_TEXT_NOT_STORED_POLICY]")'
            )
        );
    END IF;

    IF to_regclass('public.legacy_research_response_records') IS NOT NULL
       AND NOT EXISTS (
           SELECT 1 FROM pg_constraint
           WHERE conname = 'legacy_research_response_records_no_qf_raw_text'
       ) THEN
        ALTER TABLE legacy_research_response_records
        ADD CONSTRAINT legacy_research_response_records_no_qf_raw_text CHECK (
            NOT jsonb_path_exists(
                response,
                '$.blocks[*] ? (@.block_type == "SOURCE" && @.citation.source_kind == "TAFSIR" && @.citation.source_revision.source_id == "quran_foundation" && @.exact_text != "[SOURCE_TEXT_NOT_STORED_POLICY]")'
            )
        );
    END IF;
END $$;

COMMIT;
