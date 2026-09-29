import { z } from "zod";

import type { components } from "@/contracts/openapi.generated";

export const ANSWER_STATUSES = [
  "ANSWERABLE",
  "PARTIAL",
  "CLARIFICATION_REQUIRED",
  "UNSUPPORTED_SCOPE",
  "INSUFFICIENT_EVIDENCE",
  "TEMPORARILY_UNAVAILABLE",
  "FAILED_SAFE",
] as const;

const AnswerStatusSchema = z.enum(ANSWER_STATUSES);
const EvidenceSourceKindSchema = z.enum([
  "QURAN",
  "QURAN_TRANSLATION",
  "HADITH",
  "HADITH_GRADE",
  "TAFSIR",
  "HADITH_COMMENTARY",
]);
const AttributionRoleSchema = z.enum([
  "SOURCE",
  "AUTHOR",
  "TRANSLATOR",
  "GRADER",
  "PUBLISHER",
]);
const NoticeCodeSchema = z.enum([
  "INSUFFICIENT_EVIDENCE",
  "PARTIAL_COVERAGE",
  "SOURCE_UNAVAILABLE",
  "QUESTION_NOT_UNDERSTOOD",
  "ANSWER_NOT_VERIFIED",
  "CLARIFICATION",
  "UNSUPPORTED_SCOPE",
]);
const ToolWarningCodeSchema = z.enum([
  "PARTIAL_RESULTS",
  "TRUNCATED_BY_POLICY",
  "OPTIONAL_RESOURCE_UNAVAILABLE",
  "STALE_OPTIONAL_INDEX",
  "RATE_LIMIT_APPROACHING",
  "ATTRIBUTION_NOTICE",
]);

const SourceRevisionSchema = z
  .object({
    source_id: z.string(),
    resource_alias: z.string(),
    revision_id: z.string(),
    checksum: z.string(),
  })
  .strict();

const TypedSourceReferenceSchema = z
  .object({
    scheme: z.string(),
    value: z.string(),
    canonical_link: z.url().nullable(),
  })
  .strict();

const AttributionSchema = z
  .object({
    role: AttributionRoleSchema,
    display_name: z.string(),
    required_notice: z.string().nullable(),
  })
  .strict();

export const CitationSchema = z
  .object({
    citation_id: z.string(),
    evidence_id: z.string(),
    display_label: z.string(),
    source_kind: EvidenceSourceKindSchema,
    reference: TypedSourceReferenceSchema,
    attribution: z.array(AttributionSchema),
    canonical_link: z.url().nullable(),
    source_revision: SourceRevisionSchema,
  })
  .strict();

const ClaimBlockSchema = z
  .object({
    block_type: z.literal("CLAIM"),
    claim_id: z.string(),
    text: z.string(),
    citations: z.array(CitationSchema).min(1),
  })
  .strict();

const NarrativeBlockSchema = z
  .object({
    block_type: z.literal("NARRATIVE"),
    segments: z.array(z.object({
      segment_id: z.string(),
      text: z.string(),
      kind: z.enum(["FRAMING", "POINT", "TRANSITION", "SUMMARY", "LIMITATION"]),
      supporting_claim_ids: z.array(z.string()),
      supporting_evidence_ids: z.array(z.string()),
    }).strict()).min(1),
  })
  .strict();

const SourceBlockSchema = z
  .object({
    block_type: z.literal("SOURCE"),
    block_id: z.string(),
    exact_text: z.string(),
    citation: CitationSchema,
    visibility: z.enum(["VISIBLE", "COLLAPSED_CONTEXT"]),
    passage_group_id: z.string().nullable().optional(),
  })
  .strict();

const NoticeBlockSchema = z
  .object({
    block_type: z.literal("NOTICE"),
    notice_code: NoticeCodeSchema,
    text: z.string(),
  })
  .strict();

const SourceViewHeadingBlockSchema = z
  .object({
    block_type: z.literal("SOURCE_VIEW_HEADING"),
    text: z.string(),
  })
  .strict();

const VerseKeySchema = z
  .object({ chapter: z.number().int().min(1).max(114), verse: z.number().int().positive() })
  .strict();
const SourcePointerSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("QURAN"), verse_keys: z.array(VerseKeySchema).min(1) }).strict(),
  z.object({ kind: z.literal("HADITH"), source_record_id: z.string() }).strict(),
  z
    .object({
      kind: z.literal("TAFSIR"),
      passage_id: z.string(),
      verse_keys: z.array(VerseKeySchema).min(1),
    })
    .strict(),
  z
    .object({
      kind: z.literal("HADITH_COMMENTARY"),
      passage_id: z.string(),
      hadith_source_record_id: z.string(),
    })
    .strict(),
]);
const CandidateReferenceSchema = z
  .object({
    candidate_id: z.string(),
    kind: z.enum(["QURAN", "HADITH"]),
    safe_label: z.string(),
    canonical_pointer: SourcePointerSchema.nullable(),
    source_revision: SourceRevisionSchema.nullable(),
    provenance: z.enum([
      "CURRENT_MESSAGE_EXACT_PARSE",
      "WORKING_CONTEXT_EXACT_REFERENCE",
      "APPROVED_ALIAS_TABLE",
      "HIGH_PRECISION_SCOPE_PRECHECK",
    ]),
  })
  .strict();
const ClarificationCandidateSchema = z
  .object({
    candidate_id: z.string(),
    label: z.string(),
    reference: CandidateReferenceSchema.nullable(),
  })
  .strict();

const ClarificationSchema = z
  .object({
    required: z.boolean(),
    reason: z
      .enum([
        "MISSING_REFERENCE",
        "AMBIGUOUS_REFERENCE",
        "AMBIGUOUS_HADITH",
        "UNKNOWN_RESOURCE",
        "AMBIGUOUS_CONTEXT",
        "OTHER",
      ])
      .nullable(),
    detail: z
      .enum([
        "MISSING_SURAH_OR_CHAPTER",
        "MISSING_VERSE_NUMBER",
        "MULTIPLE_REFERENCE_MATCHES",
        "MULTIPLE_HADITH_MATCHES",
        "RESOURCE_NOT_APPROVED",
        "CONTEXT_HAS_MULTIPLE_TARGETS",
        "OTHER",
      ])
      .nullable(),
    missing_fields: z.array(
      z.enum(["TOPIC", "QURAN_REFERENCE", "HADITH_TARGET", "SOURCE_KIND", "RESOURCE"]),
    ),
    candidates: z.array(ClarificationCandidateSchema),
    question: z.string().nullable(),
  })
  .strict();

const ToolWarningSchema = z
  .object({
    code: ToolWarningCodeSchema,
    safe_message_key: z.string(),
    metadata: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])),
  })
  .strict();

export const ResearchRequestSchema = z
  .object({
    message: z.string().trim().min(2).max(2000),
    locale: z.string(),
    conversation_id: z.uuid().optional(),
  })
  .strict();

export const ResearchResponseSchema = z
  .object({
    schema_version: z.literal("1"),
    conversation_id: z.uuid(),
    turn_id: z.uuid(),
    correlation_id: z.uuid(),
    status: AnswerStatusSchema,
    output_kind: z.enum(["SUBSTANTIVE_SYNTHESIS", "SOURCE_ONLY", "TEMPLATE", "NO_CONTENT"]),
    blocks: z.array(z.discriminatedUnion("block_type", [ClaimBlockSchema, NarrativeBlockSchema, SourceBlockSchema, SourceViewHeadingBlockSchema, NoticeBlockSchema])),
    citations: z.array(CitationSchema),
    clarification: ClarificationSchema.nullable(),
    warnings: z.array(ToolWarningSchema),
  })
  .strict();

export const ConversationSummarySchema = z
  .object({
    conversation_id: z.uuid(),
    title: z.string().min(1).max(120),
    locale: z.enum(["ar", "en"]),
    archived: z.boolean(),
    created_at: z.string(),
    updated_at: z.string(),
    turn_count: z.number().int().nonnegative(),
  })
  .strict();

export const TraceEventSchema = z
  .object({
    seq: z.number().int().positive(),
    elapsed_ms: z.number().int().nonnegative(),
    node: z.enum(["scope", "planner", "retrieval", "fusion", "tafsir", "rerank", "evidence", "verification", "synthesis", "composition", "gate"]),
    phase: z.enum(["started", "completed", "degraded", "skipped", "failed", "rejected"]),
    summary: z.string().min(1).max(240),
    detail: z.record(z.string(), z.unknown()).optional(),
    provider: z.record(z.string(), z.unknown()).nullable().optional(),
  })
  .strict();

export const ConversationTurnSchema = z
  .object({
    turn_id: z.uuid(),
    user_message: z.string().min(2).max(2000),
    locale: z.enum(["ar", "en"]),
    response: ResearchResponseSchema,
    created_at: z.string(),
    trace_events: z.array(TraceEventSchema).max(200).optional(),
  })
  .strict();

export const ConversationDetailSchema = ConversationSummarySchema.extend({
  turns: z.array(ConversationTurnSchema),
}).strict();

export const ConversationPageSchema = z
  .object({
    items: z.array(ConversationSummarySchema),
    next_cursor: z.string().nullable(),
  })
  .strict();

export const GatewayErrorSchema = z
  .object({
    error: z
      .object({
        code: z.enum([
          "INVALID_REQUEST",
          "BACKEND_UNAVAILABLE",
          "BACKEND_TIMEOUT",
          "INVALID_BACKEND_RESPONSE",
        ]),
        message: z.string(),
      })
      .strict(),
  })
  .strict();

export type AnswerStatus = z.infer<typeof AnswerStatusSchema>;
export type Citation = z.infer<typeof CitationSchema>;
export type ResearchRequest = z.infer<typeof ResearchRequestSchema>;
export type ResearchResponse = z.infer<typeof ResearchResponseSchema>;
export type RenderedBlock = ResearchResponse["blocks"][number];
export type GatewayError = z.infer<typeof GatewayErrorSchema>;
export type ConversationSummary = z.infer<typeof ConversationSummarySchema>;
export type ConversationTurn = z.infer<typeof ConversationTurnSchema>;
export type TraceEvent = z.infer<typeof TraceEventSchema>;
export type ConversationDetail = z.infer<typeof ConversationDetailSchema>;
export type ConversationPage = z.infer<typeof ConversationPageSchema>;

type OpenApiResearchResponse = components["schemas"]["ResearchResponseV1"];
type OpenApiResearchRequest = components["schemas"]["ResearchRequest"];
type OpenApiConversationSummary = components["schemas"]["ConversationSummaryV1"];
type OpenApiConversationTurn = components["schemas"]["ConversationTurnV1"];
type Assert<T extends true> = T;
type Assignable<From, To> = [From] extends [To] ? true : false;
export type ResponseContractParity = [
  Assert<Assignable<ResearchResponse, OpenApiResearchResponse>>,
  Assert<Assignable<OpenApiResearchResponse, ResearchResponse>>,
];
export type RequestContractParity = [
  Assert<Assignable<ResearchRequest, OpenApiResearchRequest>>,
  Assert<Assignable<OpenApiResearchRequest, ResearchRequest>>,
];
export type ConversationContractParity = [
  Assert<Assignable<ConversationSummary, OpenApiConversationSummary>>,
  Assert<Assignable<OpenApiConversationSummary, ConversationSummary>>,
  Assert<Assignable<ConversationTurn, OpenApiConversationTurn>>,
  Assert<Assignable<OpenApiConversationTurn, ConversationTurn>>,
];

export function sourceDirection(citation: Citation): "rtl" | "ltr" {
  return citation.source_kind === "QURAN" ? "rtl" : "ltr";
}

export function textDirection(text: string): "rtl" | "ltr" {
  return /[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]/.test(text) ? "rtl" : "ltr";
}
