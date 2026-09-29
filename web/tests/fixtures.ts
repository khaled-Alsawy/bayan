import type { AnswerStatus, Citation, ResearchResponse } from "@/lib/contracts";

export const ARABIC_EXACT = "إِنَّ مَعَ الْعُسْرِ يُسْرًا";
export const ENGLISH_EXACT = "Indeed, with hardship comes ease.";

export const arabicCitation: Citation = {
  citation_id: "cite-arabic",
  evidence_id: "evidence-arabic",
  display_label: "Quran 94:6 — Tanzil Project",
  source_kind: "QURAN",
  reference: {
    scheme: "quran",
    value: "94:6",
    canonical_link: "https://tanzil.net/#94:6",
  },
  attribution: [
    { role: "SOURCE", display_name: "Tanzil Project", required_notice: null },
  ],
  canonical_link: "https://tanzil.net/#94:6",
  source_revision: {
    source_id: "tanzil",
    resource_alias: "tanzil-uthmani-v1.1",
    revision_id: "rev-arabic-1",
    checksum: "sha256:arabic",
  },
};

export const translationCitation: Citation = {
  citation_id: "cite-translation",
  evidence_id: "evidence-translation",
  display_label: "Quran 94:6 — Rowwad Translation Center, QuranEnc",
  source_kind: "QURAN_TRANSLATION",
  reference: {
    scheme: "quran",
    value: "94:6",
    canonical_link: "https://quranenc.com/en/browse/english_rwwad/94/6",
  },
  attribution: [
    { role: "TRANSLATOR", display_name: "Rowwad Translation Center / QuranEnc", required_notice: "Translation of the meanings." },
  ],
  canonical_link: "https://quranenc.com/en/browse/english_rwwad/94/6",
  source_revision: {
    source_id: "quranenc",
    resource_alias: "quranenc-rowwad-en-v1.0.19",
    revision_id: "rev-translation-1",
    checksum: "sha256:translation",
  },
};

export const ibnKathirCitation: Citation = {
  citation_id: "cite-ibn-kathir",
  evidence_id: "evidence-ibn-kathir",
  display_label: "Tafsir Ibn Kathir — Quranic Universal Library (Tarteel)",
  source_kind: "TAFSIR",
  reference: {
    scheme: "tafsir",
    value: "94:6-94:8",
    canonical_link: "https://qul.tarteel.ai",
  },
  attribution: [{ role: "SOURCE", display_name: "Quranic Universal Library (Tarteel)", required_notice: null }],
  canonical_link: "https://qul.tarteel.ai",
  source_revision: {
    source_id: "qul",
    resource_alias: "qul-tafsir-ibnkathir-ar-v1",
    revision_id: "qul-export",
    checksum: "sha256:ibn-kathir",
  },
};

export const qurtubiCitation: Citation = {
  citation_id: "cite-qurtubi",
  evidence_id: "evidence-qurtubi",
  display_label: "تفسير القرطبي — Quranic Universal Library (Tarteel)",
  source_kind: "TAFSIR",
  reference: {
    scheme: "tafsir",
    value: "94:6",
    canonical_link: "https://qul.tarteel.ai",
  },
  attribution: [{ role: "SOURCE", display_name: "Quranic Universal Library (Tarteel)", required_notice: null }],
  canonical_link: "https://qul.tarteel.ai",
  source_revision: {
    source_id: "qul",
    resource_alias: "qul-tafsir-alqurtubi-ar-v1",
    revision_id: "qul-export",
    checksum: "sha256:qurtubi",
  },
};

export function makeTafsirResponse(comparison = false): ResearchResponse {
  const response = makeResearchResponse();
  const tafsirCitations = comparison ? [ibnKathirCitation, qurtubiCitation] : [qurtubiCitation];
  response.citations = [...response.citations, ...tafsirCitations];
  response.blocks = [
    {
      block_type: "CLAIM",
      claim_id: "tafsir-claim-1",
      text: "يربط النص المسترجع اليسر المذكور في الآية بالعسر.",
      citations: tafsirCitations,
    },
    ...response.blocks.filter((block) => block.block_type === "SOURCE"),
    ...tafsirCitations.map((citation, index) => ({
      block_type: "SOURCE" as const,
      block_id: `tafsir-source-${index + 1}`,
      exact_text: index === 0 ? "A concise retrieved Tafsir excerpt." : "نص تفسيري عربي موجز مسترجع.",
      citation,
      visibility: "VISIBLE" as const,
    })),
  ];
  return response;
}

export function makeResearchResponse(status: AnswerStatus = "ANSWERABLE"): ResearchResponse {
  const answered = status === "ANSWERABLE" || status === "PARTIAL";
  return {
    schema_version: "1",
    conversation_id: "11111111-1111-4111-8111-111111111111",
    turn_id: "22222222-2222-4222-8222-222222222222",
    correlation_id: "33333333-3333-4333-8333-333333333333",
    status,
    output_kind: answered ? "SUBSTANTIVE_SYNTHESIS" : "NO_CONTENT",
    blocks: answered
      ? [
          {
            block_type: "CLAIM",
            claim_id: "claim-1",
            text: "تبيّن الآية أن اليسر يقترن بالعسر.",
            citations: [arabicCitation, translationCitation],
          },
          {
            block_type: "SOURCE",
            block_id: "source-arabic",
            exact_text: ARABIC_EXACT,
            citation: arabicCitation,
            visibility: "VISIBLE",
          },
          {
            block_type: "SOURCE",
            block_id: "source-translation",
            exact_text: ENGLISH_EXACT,
            citation: translationCitation,
            visibility: "VISIBLE",
          },
        ]
      : [],
    citations: answered ? [arabicCitation, translationCitation] : [],
    clarification:
      status === "CLARIFICATION_REQUIRED"
        ? {
            required: true,
            reason: "AMBIGUOUS_CONTEXT",
            detail: "CONTEXT_HAS_MULTIPLE_TARGETS",
            missing_fields: ["TOPIC"],
            candidates: [
              { candidate_id: "sense-eye-envy", label: "الحسد والإصابة بالعين", reference: null },
              { candidate_id: "sense-eye-vision", label: "العين الباصرة", reference: null },
              { candidate_id: "sense-eye-water", label: "عيون الماء", reference: null },
            ],
            question: "هل تقصد بالعين: الحسد والإصابة بالعين، أم العين الباصرة، أم عيون الماء؟",
          }
        : null,
    warnings: [],
  };
}

export function makePartialResearchResponse(locale: "ar" | "en" = "ar"): ResearchResponse {
  const response = makeResearchResponse("PARTIAL");
  const claim = response.blocks.find((block) => block.block_type === "CLAIM");
  if (claim?.block_type === "CLAIM") {
    claim.text = locale === "ar"
      ? "تشير الآية مباشرة إلى أن اليسر يأتي مع العسر."
      : "The verse directly indicates that ease comes with hardship.";
  }
  response.blocks.push({
    block_type: "NOTICE",
    notice_code: "PARTIAL_COVERAGE",
    text: locale === "ar"
      ? "لم يثبت الدليل القرآني المسترجع حاليًا الجزء المتعلق بطرق عملية لتجاوز العسر؛ لذلك تقتصر الإجابة على الجزء الذي سنده الدليل المعروض."
      : "Current Quran evidence did not establish the part of the question about practical ways to move through hardship. The answer is limited to the evidenced part.",
  });
  return response;
}

export function makeLongResearchResponse(locale: "ar" | "en"): ResearchResponse {
  const response = makeResearchResponse();
  const claim =
    locale === "ar"
      ? "توجّه الآية المؤمن إلى الاستعانة بالصبر والصلاة عند الشدة، مع التزام الدليل المذكور."
      : "The passage directs believers to seek help through patience and prayer during hardship.";
  response.blocks = Array.from({ length: 10 }, (_, index) => [
    {
      block_type: "CLAIM" as const,
      claim_id: `long-claim-${index + 1}`,
      text: `${claim} ${index + 1}`,
      citations: [arabicCitation, translationCitation],
    },
    {
      block_type: "SOURCE" as const,
      block_id: `long-source-arabic-${index + 1}`,
      exact_text: `${ARABIC_EXACT} ﴿${index + 1}﴾`,
      citation: arabicCitation,
      visibility: "VISIBLE" as const,
    },
    {
      block_type: "SOURCE" as const,
      block_id: `long-source-translation-${index + 1}`,
      exact_text: `${ENGLISH_EXACT} [${index + 1}]`,
      citation: translationCitation,
      visibility: "VISIBLE" as const,
    },
  ]).flat();
  return response;
}
