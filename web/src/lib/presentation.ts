import type { Citation, ResearchResponse } from "@/lib/contracts";
import type { UiLocale } from "@/lib/i18n";
import { SOURCE_NAMES_AR } from "@/lib/source-names";
import { formatSurahReference } from "@/lib/surah-names";

export type PresentationKind = "topic" | "exact-verse" | "single-tafsir" | "tafsir-comparison" | "terminal";
export type SourceBlock = Extract<ResearchResponse["blocks"][number], { block_type: "SOURCE" }>;
export type ClaimBlock = Extract<ResearchResponse["blocks"][number], { block_type: "CLAIM" }>;
export type NarrativeBlock = Extract<ResearchResponse["blocks"][number], { block_type: "NARRATIVE" }>;

export interface EvidenceGroup {
  reference: string;
  chapter: number | null;
  verseStart: number | null;
  verseEnd: number | null;
  quran: SourceBlock[];
  translations: SourceBlock[];
  tafsirs: SourceBlock[];
}

export interface AnswerPresentation {
  kind: PresentationKind;
  citations: Citation[];
  citedEvidenceIds: Set<string>;
  claims: ClaimBlock[];
  narrative: NarrativeBlock | null;
  groups: EvidenceGroup[];
}

const EXACT_REFERENCE = /(?:\b(?:qur(?:an|\u2019an)|koran|verse|ayah)\b|(?:الآية|اية|آية|سورة))[^\n]{0,36}?\d{1,3}\s*[:：]\s*\d{1,3}\b/iu;
const COMPARISON = /\b(?:compare|comparison|versus|vs\.?|between)\b|(?:قارن|مقارنة|بين\s+تفسير)/iu;
const REFERENCE = /(\d{1,3})\s*[:：]\s*(\d{1,3})(?:\s*-\s*(\d{1,3}))?/u;
const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";

function toArabicDigits(value: string): string {
  return value.replace(/\d/gu, (digit) => AR_DIGITS[Number(digit)]);
}

function range(citation: Citation): { chapter: number; start: number; end: number } | null {
  const value = citation.reference.value.match(REFERENCE);
  if (!value) return null;
  const chapter = Number(value[1]);
  const start = Number(value[2]);
  const end = Number(value[3] ?? value[2]);
  return chapter > 0 && start > 0 && end >= start ? { chapter, start, end } : null;
}

function label(chapter: number, start: number, end: number): string {
  return `${chapter}:${start}${start === end ? "" : `-${end}`}`;
}

function usedEvidenceIds(response: ResearchResponse): Set<string> {
  const ids = new Set<string>();
  for (const block of response.blocks) {
    if (block.block_type === "SOURCE") ids.add(block.citation.evidence_id);
    if (block.block_type === "CLAIM") block.citations.forEach((citation) => ids.add(citation.evidence_id));
    if (block.block_type === "NARRATIVE") block.segments.forEach((segment) => segment.supporting_evidence_ids.forEach((id) => ids.add(id)));
  }
  return ids;
}

/** Evidence referenced by the generated reading, in first-cited order. */
export function citedEvidenceIds(response: ResearchResponse): Set<string> {
  const ids = new Set<string>();
  const narrative = response.blocks.find((block): block is NarrativeBlock => block.block_type === "NARRATIVE");
  narrative?.segments.forEach((segment) => segment.supporting_evidence_ids.forEach((id) => ids.add(id)));
  response.blocks.filter((block): block is ClaimBlock => block.block_type === "CLAIM")
    .forEach((claim) => claim.citations.forEach((citation) => ids.add(citation.evidence_id)));
  return ids;
}

/** Groups consecutive retrieved ayat as a passage, never as search-result cards. */
function passageGroups(sources: SourceBlock[]): EvidenceGroup[] {
  const byVerse = new Map<string, { chapter: number; verse: number; quran: SourceBlock[]; translations: SourceBlock[] }>();
  for (const source of sources) {
    if (source.citation.source_kind !== "QURAN" && source.citation.source_kind !== "QURAN_TRANSLATION") continue;
    const parsed = range(source.citation);
    if (!parsed || parsed.start !== parsed.end) continue;
    const key = `${parsed.chapter}:${parsed.start}`;
    const entry = byVerse.get(key) ?? { chapter: parsed.chapter, verse: parsed.start, quran: [], translations: [] };
    (source.citation.source_kind === "QURAN" ? entry.quran : entry.translations).push(source);
    byVerse.set(key, entry);
  }
  const ordered = [...byVerse.values()].sort((a, b) => a.chapter - b.chapter || a.verse - b.verse);
  const groups: EvidenceGroup[] = [];
  for (const verse of ordered) {
    const previous = groups.at(-1);
    if (previous && previous.chapter === verse.chapter && previous.verseEnd === verse.verse - 1) {
      previous.verseEnd = verse.verse;
      previous.reference = label(verse.chapter, previous.verseStart!, verse.verse);
      previous.quran.push(...verse.quran);
      previous.translations.push(...verse.translations);
    } else {
      groups.push({ reference: label(verse.chapter, verse.verse, verse.verse), chapter: verse.chapter, verseStart: verse.verse, verseEnd: verse.verse, quran: verse.quran, translations: verse.translations, tafsirs: [] });
    }
  }
  for (const source of sources.filter((item) => item.citation.source_kind === "TAFSIR")) {
    const parsed = range(source.citation);
    const target = parsed ? groups.find((group) => group.chapter === parsed.chapter && group.verseStart! <= parsed.end && group.verseEnd! >= parsed.start) : undefined;
    if (target) target.tafsirs.push(source);
    else groups.push({ reference: parsed ? label(parsed.chapter, parsed.start, parsed.end) : source.citation.reference.value, chapter: parsed?.chapter ?? null, verseStart: parsed?.start ?? null, verseEnd: parsed?.end ?? null, quran: [], translations: [], tafsirs: [source] });
  }
  return groups;
}

export function buildAnswerPresentation(response: ResearchResponse, userMessage: string): AnswerPresentation {
  const ids = usedEvidenceIds(response);
  const readingEvidenceIds = citedEvidenceIds(response);
  const citations = response.citations.filter((citation) => ids.has(citation.evidence_id));
  const claims = response.blocks.filter((block): block is ClaimBlock => block.block_type === "CLAIM");
  const narrative = response.blocks.find((block): block is NarrativeBlock => block.block_type === "NARRATIVE") ?? null;
  const groups = passageGroups(response.blocks.filter((block): block is SourceBlock => block.block_type === "SOURCE"));
  const tafsirAliases = new Set(citations.filter((citation) => citation.source_kind === "TAFSIR").map((citation) => citation.source_revision.resource_alias));
  const kind = response.status !== "ANSWERABLE" && response.status !== "PARTIAL" ? "terminal"
    : tafsirAliases.size === 2 && COMPARISON.test(userMessage) ? "tafsir-comparison"
      : tafsirAliases.size === 1 ? "single-tafsir"
        : EXACT_REFERENCE.test(userMessage) ? "exact-verse" : "topic";
  if ((kind === "topic" || kind === "single-tafsir") && readingEvidenceIds.size > 0) {
    const citationOrder = new Map([...readingEvidenceIds].map((id, index) => [id, index]));
    const firstCitationIndex = (group: EvidenceGroup) => [...group.quran, ...group.translations, ...group.tafsirs]
      .reduce((earliest, source) => Math.min(earliest, citationOrder.get(source.citation.evidence_id) ?? Number.POSITIVE_INFINITY), Number.POSITIVE_INFINITY);
    groups.sort((a, b) => firstCitationIndex(a) - firstCitationIndex(b)
      || (a.chapter ?? Number.POSITIVE_INFINITY) - (b.chapter ?? Number.POSITIVE_INFINITY)
      || (a.verseStart ?? Number.POSITIVE_INFINITY) - (b.verseStart ?? Number.POSITIVE_INFINITY));
  }
  return { kind, citations, citedEvidenceIds: readingEvidenceIds, claims, narrative, groups };
}

/** Renders the reader-facing reference only. `reference.value` may carry an internal
 *  resource alias ("qul-tafsir-almukhtasar-ar-v1@70:22-22"), which must never reach the page. */
export function citationLabel(citation: Citation, locale: UiLocale = "ar"): string {
  const rawReference = citation.reference.value.split("@").at(-1) ?? citation.reference.value;
  if (citation.source_kind === "QURAN" || citation.source_kind === "QURAN_TRANSLATION" || citation.source_kind === "TAFSIR") {
    const formatted = formatSurahReference(rawReference, locale, "short");
    if (formatted !== rawReference) return formatted;
  }
  const parsed = range(citation);
  if (!parsed) return rawReference;
  const legacyLabel = `${parsed.chapter}:${parsed.start === parsed.end ? `${parsed.start}` : `${parsed.start}-${parsed.end}`}`;
  return locale === "ar" ? toArabicDigits(legacyLabel) : legacyLabel;
}

/** The legal attribution name, kept verbatim: required notices depend on it. */
export function legalSourceName(citation: Citation): string {
  const names = citation.attribution.map((item) => item.display_name).filter(Boolean);
  return names.join(" / ") || citation.display_label;
}

export function sourceDisplayName(citation: Citation, locale: UiLocale = "en"): string {
  if (locale === "ar") {
    const arabic = SOURCE_NAMES_AR[citation.source_revision.resource_alias];
    if (arabic) return arabic;
  }
  return legalSourceName(citation);
}

export function safeCitationHref(citation: Citation): string | null {
  const candidate = citation.canonical_link ?? citation.reference.canonical_link;
  if (!candidate) return null;
  try { const url = new URL(candidate); return url.protocol === "https:" || url.protocol === "http:" ? url.toString() : null; } catch { return null; }
}
