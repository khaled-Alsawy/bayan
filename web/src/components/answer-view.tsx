"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown, Clipboard, ExternalLink, Flag, MessageCircleQuestion, ShieldAlert, TriangleAlert } from "lucide-react";

import { textDirection, type Citation, type ResearchResponse, type TraceEvent } from "@/lib/contracts";
import { getMessages, type Messages, type UiLocale } from "@/lib/i18n";
import { buildAnswerPresentation, citationLabel, legalSourceName, safeCitationHref, sourceDisplayName, type EvidenceGroup, type SourceBlock } from "@/lib/presentation";
import { formatSurahReference } from "@/lib/surah-names";
import { traceElapsed, traceHeadline, traceLine } from "@/lib/trace-copy";
import { cn } from "@/lib/utils";

function sourceLinkLabel(citation: Citation, copy: Messages): string {
  const source = sourceDisplayName(citation, copy.locale);
  if (citation.source_kind === "QURAN") return copy.openVerseSource(source);
  if (citation.source_kind === "QURAN_TRANSLATION") return copy.openTranslationSource(source);
  if (citation.source_kind === "TAFSIR") return copy.openTafsirSource(source);
  return copy.openCanonicalSource;
}

function SourceLink({ citation, copy }: { citation: Citation; copy: Messages }) {
  const href = safeCitationHref(citation);
  const label = sourceLinkLabel(citation, copy);
  const source = sourceDisplayName(citation, copy.locale);
  return href ? <a className="source-link" href={href} aria-label={label} title={label} target="_blank" rel="noopener noreferrer">{copy.openSource}<ExternalLink aria-hidden size={13} /></a> : <bdi className="source-link source-link-static">{source}</bdi>;
}

function sourceRecordLabel(citation: Citation, locale: UiLocale): string {
  if (citation.source_kind !== "QURAN" && citation.source_kind !== "QURAN_TRANSLATION" && citation.source_kind !== "TAFSIR") return citation.display_label;
  return citation.display_label.replace(/\d{1,3}\s*[:：]\s*\d{1,3}(?:\s*-\s*(?:\d{1,3}\s*[:：]\s*)?\d{1,3})?/u, (reference) => formatSurahReference(reference, locale, "short"));
}

function CitationChip({ citation, copy }: { citation: Citation; copy: Messages }) {
  const source = sourceDisplayName(citation, copy.locale);
  const title = citation.source_kind === "QURAN" || citation.source_kind === "QURAN_TRANSLATION" || citation.source_kind === "TAFSIR"
    ? `${source} — ${citation.reference.value.split("@").at(-1) ?? citation.reference.value}`
    : `${source} — ${copy.locale === "ar" ? "الآية" : "Verse"} ${citationLabel(citation, copy.locale)}`;
  return <a className="citation-chip" href={`#evidence-${citation.evidence_id}`} title={title}>{citationLabel(citation, copy.locale)}</a>;
}

/** Kept as an inspectable provenance view for keyboard users and existing consumers. */
export function SourcesPanel({ citations, locale }: { citations: Citation[]; locale: UiLocale }) {
  const copy = getMessages(locale);
  return <section aria-label={copy.usedSources(citations.length)}>{citations.map((citation) => <div key={citation.citation_id} className="source-record"><p>{sourceRecordLabel(citation, copy.locale)}</p><p>{citation.source_revision.resource_alias} · {citation.source_revision.revision_id}</p><p>{sourceDisplayName(citation, copy.locale)}</p><SourceLink citation={citation} copy={copy} /></div>)}</section>;
}

function StatusPanel({ response, copy, onClarificationChoice, onEditQuestion }: { response: ResearchResponse; copy: Messages; onClarificationChoice?: (choice: string) => void; onEditQuestion?: () => void }) {
  const status = copy.status[response.status];
  const Icon = response.status === "CLARIFICATION_REQUIRED" ? MessageCircleQuestion : response.status === "TEMPORARILY_UNAVAILABLE" ? ShieldAlert : TriangleAlert;
  const notice = response.blocks.find((block) => block.block_type === "NOTICE");
  const detail = notice?.block_type === "NOTICE" ? copy.notices[notice.notice_code] : status.detail;
  return <section className={cn("terminal-panel", `terminal-panel-${status.tone}`)} role="status"><span className="terminal-icon"><Icon aria-hidden size={20} /></span><div><h2 className="text-lg font-bold">{status.label}</h2><p className="mt-2 text-sm leading-7 text-[var(--ink-soft)]">{response.status === "CLARIFICATION_REQUIRED" && response.clarification?.question ? response.clarification.question : detail}</p>{response.status === "FAILED_SAFE" && onEditQuestion ? <div className="mt-4"><button type="button" className="follow-up-button" onClick={onEditQuestion}>{copy.editQuestion}</button></div> : null}{response.status === "CLARIFICATION_REQUIRED" && response.clarification?.candidates.length ? <div className="mt-4 flex flex-wrap gap-2">{response.clarification.candidates.map((candidate) => <button key={candidate.candidate_id} type="button" dir={textDirection(candidate.label)} className="follow-up-button" onClick={() => onClarificationChoice?.(candidate.label)}>{candidate.label}</button>)}</div> : null}</div></section>;
}

function VerseCard({ group, copy, defaultTranslationOpen = false }: { group: EvidenceGroup; copy: Messages; defaultTranslationOpen?: boolean }) {
  const [translationOpen, setTranslationOpen] = useState(defaultTranslationOpen);
  const [copied, setCopied] = useState(false);
  const translationAnchors = useMemo(() => group.translations.map((source) => `#evidence-${source.citation.evidence_id}`).join("|"), [group.translations]);
  useEffect(() => {
    const openIfTargeted = () => { if (translationAnchors.split("|").includes(window.location.hash)) setTranslationOpen(true); };
    openIfTargeted();
    window.addEventListener("hashchange", openIfTargeted);
    return () => window.removeEventListener("hashchange", openIfTargeted);
  }, [translationAnchors]);
  const quranText = group.quran.map((source) => source.exact_text).join(" ");
  const canCopy = Boolean(quranText && !quranText.includes("[SOURCE_TEXT_NOT_STORED_POLICY]"));
  const copyVerse = async () => { if (!canCopy) return; try { await navigator.clipboard.writeText(quranText); setCopied(true); window.setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } };
  const citation = group.quran[0]?.citation;
  return <figure className="verse-card" data-testid="verse-card" id={`passage-${group.reference}`}>
    <figcaption className="evidence-header"><div><p className="evidence-kicker">{copy.quranPassage}</p><p className="passage-title">{copy.verseReference(group.reference)}</p></div><div className="evidence-actions">{canCopy ? <button type="button" className="icon-button" onClick={() => void copyVerse()} aria-label={copy.copyVerse}>{copied ? <Check aria-hidden size={15} /> : <Clipboard aria-hidden size={15} />}</button> : null}</div></figcaption>
    {quranText ? <blockquote dir="rtl" className="quran-text">{group.quran.map((source, index) => <span key={source.block_id} dir="rtl"><span id={`evidence-${source.citation.evidence_id}`} className="evidence-anchor" />{source.exact_text}<span className="ayah-number">{((group.verseStart ?? 1) + index).toLocaleString("ar-EG")}</span>{" "}</span>)}</blockquote> : null}
    {group.translations.length ? <div className="translation-area">{group.translations.map((source) => <span key={`anchor-${source.block_id}`} id={`evidence-${source.citation.evidence_id}`} className="evidence-anchor" />)}<button type="button" className="translation-toggle" aria-expanded={translationOpen} onClick={() => setTranslationOpen((value) => !value)}><ChevronDown aria-hidden className={cn("transition-transform", translationOpen && "rotate-180")} size={15} />{translationOpen ? copy.hideTranslation : copy.showTranslation}</button>{translationOpen ? <div className="translation-block">{group.translations.map((source) => <p key={source.block_id} dir={textDirection(source.exact_text)}>{source.exact_text}</p>)}</div> : null}</div> : null}
    <footer className="evidence-footer">{citation ? <SourceLink citation={citation} copy={copy} /> : null}</footer>
  </figure>;
}

function TafsirCard({ source, copy, onReloadSource, onReturnToAnswer }: { source: SourceBlock; copy: Messages; onReloadSource?: () => void; onReturnToAnswer?: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const redacted = source.exact_text === "[SOURCE_TEXT_NOT_STORED_POLICY]";
  const wordCount = source.exact_text.trim().split(/\s+/u).filter(Boolean).length;
  const toggle = () => setExpanded((current) => !current);
  const copyText = async () => { try { await navigator.clipboard.writeText(source.exact_text); setCopied(true); window.setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } };
  return <figure className={cn("tafsir-card", expanded && "tafsir-card-expanded")} data-testid="tafsir-card" data-tafsir-expanded={expanded} id={`evidence-${source.citation.evidence_id}`}>
    <figcaption className="evidence-header tafsir-header"><div><p className="evidence-kicker">{copy.tafsirExcerpt}</p><p className="passage-title">{copy.verseReference(source.citation.reference.value.split("@").at(-1) ?? source.citation.reference.value)}</p></div><div className="evidence-actions"><button type="button" className="icon-button" onClick={() => void copyText()} aria-label={copy.copyVerse}>{copied ? <Check aria-hidden size={15} /> : <Clipboard aria-hidden size={15} />}</button></div></figcaption>
    {redacted ? <div className="p-5 text-sm text-[var(--ink-soft)]"><p>{copy.restrictedSourceNotice}</p><button type="button" onClick={onReloadSource} className="reload-source-button">{copy.reloadSource}</button></div> : <div className={cn("tafsir-reading", !expanded && "tafsir-collapsed")}><blockquote dir={textDirection(source.exact_text)}>{source.exact_text}</blockquote></div>}
    {!redacted ? <><footer className="evidence-footer"><button type="button" className="translation-toggle" aria-expanded={expanded} onClick={toggle}><ChevronDown aria-hidden className={cn("transition-transform", expanded && "rotate-180")} size={15} />{expanded ? copy.collapseExcerpt : copy.readFullExcerpt(wordCount)}</button><SourceLink citation={source.citation} copy={copy} /></footer>{expanded ? <button type="button" className="return-to-answer" onClick={onReturnToAnswer}>{copy.backToAnswer}</button> : null}</> : null}
  </figure>;
}

/** Renders cited evidence whose SOURCE block never reached the response, so a citation
 *  always lands somewhere and the gap is stated instead of silently linking nowhere. */
function UnrenderedEvidenceCard({ citation, copy }: { citation: Citation; copy: Messages }) {
  return <figure className="evidence-missing" data-testid="unrendered-evidence" id={`evidence-${citation.evidence_id}`}>
    <figcaption className="evidence-header"><div><p className="evidence-kicker">{copy.sourceKinds[citation.source_kind]}</p><p className="passage-title">{copy.verseReference(citation.reference.value.split("@").at(-1) ?? citation.reference.value)}</p></div></figcaption>
    <div className="evidence-missing-body"><p><strong>{copy.evidenceNotRendered}</strong></p><p>{copy.evidenceNotRenderedDetail}</p></div>
    <footer className="evidence-footer"><SourceLink citation={citation} copy={copy} /></footer>
  </figure>;
}

function Narrative({ response, copy, citations }: { response: ResearchResponse; copy: Messages; citations: Citation[] }) {
  const narrative = response.blocks.find((block) => block.block_type === "NARRATIVE");
  const claims = response.blocks.filter((block) => block.block_type === "CLAIM");
  const segments = narrative?.block_type === "NARRATIVE" ? narrative.segments : claims.map((claim) => ({ segment_id: claim.claim_id, text: claim.text, kind: "POINT" as const, supporting_claim_ids: [claim.claim_id], supporting_evidence_ids: claim.citations.map((citation) => citation.evidence_id) }));
  // Each evidence chip appears once, on the segment that first relies on it: repeating the same
  // three chips under every paragraph adds no provenance and doubles the reading length.
  const firstMention = useMemo(() => {
    const bySegment = new Map<string, string[]>();
    const seen = new Set<string>();
    for (const segment of segments) {
      const fresh = segment.supporting_evidence_ids.filter((id) => !seen.has(id));
      fresh.forEach((id) => seen.add(id));
      if (fresh.length) bySegment.set(segment.segment_id, fresh);
    }
    return bySegment;
  }, [segments]);
  if (!segments.length) return null;
  return <section className="narrative" aria-label={copy.generatedReadingAria}><p className="generated-label">{copy.generatedReadingLabel}</p><div className="narrative-copy">{segments.map((segment) => <p key={segment.segment_id} dir={textDirection(segment.text)}>{segment.text}{(firstMention.get(segment.segment_id) ?? []).map((id) => { const citation = citations.find((item) => item.evidence_id === id); return citation ? <CitationChip key={id} citation={citation} copy={copy} /> : null; })}</p>)}</div></section>;
}

function PartialCoverageNote({ response, copy }: { response: ResearchResponse; copy: Messages }) {
  const notice = response.blocks.find((block) => block.block_type === "NOTICE" && block.notice_code === "PARTIAL_COVERAGE");
  const text = notice?.block_type === "NOTICE" ? copy.notices[notice.notice_code] : copy.notices.PARTIAL_COVERAGE;
  return <section className="partial-coverage-note" data-testid="partial-coverage-note"><strong>{copy.partialCoverageLabel}</strong><p dir={textDirection(text)}>{text}</p></section>;
}

function ProvenanceFooter({ citations, copy }: { citations: Citation[]; copy: Messages }) {
  const rows = ["QURAN", "QURAN_TRANSLATION", "TAFSIR"].map((kind) => citations.find((citation) => citation.source_kind === kind)).filter((citation): citation is Citation => Boolean(citation));
  return rows.length ? <section className="provenance-footer" data-testid="provenance-footer" aria-label={copy.provenanceSources}><h2>{copy.provenanceSources}</h2>{rows.map((citation) => { const name = sourceDisplayName(citation, copy.locale); const legal = legalSourceName(citation); return <div key={citation.citation_id} className="provenance-row"><span>{copy.sourceKinds[citation.source_kind]}</span><span className="provenance-name"><bdi>{name}</bdi>{legal !== name ? <bdi className="provenance-legal">{legal}</bdi> : null}</span>{safeCitationHref(citation) ? <SourceLink citation={citation} copy={copy} /> : <span />}</div>; })}</section> : null;
}

/** The workflow panel: compact rows while the answer is still running, then a collapsed
 *  dropdown pinned above the answer. Each row expands to its explanation and raw payload. */
export function ExecutionTrace({ events, locale, running = false, open: openProp }: { events: TraceEvent[]; locale: UiLocale; running?: boolean; open?: boolean }) {
  const copy = getMessages(locale);
  const [open, setOpen] = useState(openProp ?? running);
  const [expanded, setExpanded] = useState<number | null>(null);
  if (!events.length && !running) return null;
  return <section className={cn("trace-panel", running && "trace-panel-running")} data-testid="execution-trace">
    <button type="button" className="trace-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
      <ChevronDown aria-hidden size={14} className={cn("trace-caret", open && "trace-caret-open")} />
      <span className="trace-headline">{traceHeadline(events, locale, running)}</span>
      {running ? <span className="trace-spinner" aria-hidden /> : <span className="trace-count">{copy.executionTrace}</span>}
    </button>
    {open ? <ol className="trace-steps">{events.map((event) => {
      const step = traceLine(event, locale);
      const isOpen = expanded === event.seq;
      return <li key={event.seq} className={cn("trace-step", `trace-step-${step.tone}`)}>
        <button type="button" className="trace-step-head" aria-expanded={isOpen} onClick={() => setExpanded(isOpen ? null : event.seq)}>
          <span className="trace-dot" aria-hidden />
          <span className="trace-step-label">{step.label}</span>
          <span className="trace-step-line" dir={textDirection(step.line)}>{step.line}</span>
          <span className="trace-step-time">{traceElapsed(event, locale)}</span>
        </button>
        {isOpen ? <div className="trace-step-body">
          <p className="trace-step-summary">{event.summary}</p>
          {event.provider ? <p className="trace-step-provider">{String(event.provider.effective_provider_id ?? event.provider.gateway_provider_id ?? "provider")}</p> : null}
          <details className="trace-step-raw"><summary>{copy.traceRawDetail}</summary><pre>{JSON.stringify(event.detail, null, 2)}</pre></details>
        </div> : null}
      </li>;
    })}</ol> : null}
  </section>;
}

export function AnswerView({ response, userMessage = "", locale = "ar", traceEvents = [], onReloadSource, onFollowUp, onClarificationChoice, onEditQuestion }: { response: ResearchResponse; userMessage?: string; locale?: UiLocale; traceEvents?: TraceEvent[]; onReloadSource?: () => void; onFollowUp?: (prompt: string) => void; onClarificationChoice?: (choice: string) => void; onEditQuestion?: () => void }) {
  const copy = getMessages(locale);
  const presentation = useMemo(() => buildAnswerPresentation(response, userMessage), [response, userMessage]);
  const sourceViewHeading = response.blocks.find((block) => block.block_type === "SOURCE_VIEW_HEADING");
  const honestInsufficientSourceView = response.status === "INSUFFICIENT_EVIDENCE" && Boolean(sourceViewHeading) && presentation.groups.length > 0;
  // Evidence the answer cites but the response never shipped a SOURCE block for. Without this the
  // narrative renders chips pointing at element ids that do not exist anywhere on the page.
  const unrendered = useMemo(() => {
    const rendered = new Set(presentation.groups.flatMap((group) => [...group.quran, ...group.translations, ...group.tafsirs].map((source) => source.citation.evidence_id)));
    const cited = new Set(response.blocks.flatMap((block) => block.block_type === "NARRATIVE" ? block.segments.flatMap((segment) => segment.supporting_evidence_ids) : block.block_type === "CLAIM" ? block.citations.map((citation) => citation.evidence_id) : []));
    return response.citations.filter((citation) => cited.has(citation.evidence_id) && !rendered.has(citation.evidence_id));
  }, [presentation, response]);
  const [reportOpen, setReportOpen] = useState(false);
  const returnToAnswer = () => document.getElementById(`answer-${response.turn_id}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  const groupIsCited = (group: EvidenceGroup) => [...group.quran, ...group.translations, ...group.tafsirs]
    .some((source) => presentation.citedEvidenceIds.has(source.citation.evidence_id));
  const collapseUncitedGroups = (presentation.kind === "topic" || presentation.kind === "single-tafsir") && !honestInsufficientSourceView
    && presentation.citedEvidenceIds.size > 0 && presentation.groups.some((group) => !groupIsCited(group));
  const visibleGroups = collapseUncitedGroups ? presentation.groups.filter(groupIsCited) : presentation.groups;
  const collapsedGroups = collapseUncitedGroups ? presentation.groups.filter((group) => !groupIsCited(group)) : [];
  const renderGroup = (group: EvidenceGroup) => <div key={group.reference} className="passage-group">{group.quran.length ? <VerseCard group={group} copy={copy} defaultTranslationOpen={presentation.kind === "exact-verse"} /> : null}{group.tafsirs.map((source) => <TafsirCard key={source.block_id} source={source} copy={copy} onReloadSource={onReloadSource} onReturnToAnswer={returnToAnswer} />)}</div>;
  return <article id={`answer-${response.turn_id}`} className="assistant-answer" aria-live="polite" data-testid="assistant-answer" data-answer-status={response.status} onClick={(event) => {
    if (!(event.target instanceof Element)) return;
    const link = event.target.closest<HTMLAnchorElement>("a.citation-chip[href^='#evidence-']");
    if (!link) return;
    const target = document.getElementById(link.hash.slice(1));
    const details = target?.closest("details");
    if (details) details.open = true;
  }}>
    <ExecutionTrace events={traceEvents} locale={locale} />
    {honestInsufficientSourceView ? <><StatusPanel response={response} copy={copy} onClarificationChoice={onClarificationChoice} onEditQuestion={onEditQuestion} /><section className="evidence-section"><details data-testid="possibly-related-results"><summary>{sourceViewHeading?.block_type === "SOURCE_VIEW_HEADING" ? sourceViewHeading.text : ""}</summary>{presentation.groups.map(renderGroup)}</details></section></> : presentation.kind === "terminal" ? <StatusPanel response={response} copy={copy} onClarificationChoice={onClarificationChoice} onEditQuestion={onEditQuestion} /> : <><Narrative response={response} copy={copy} citations={presentation.citations} />{presentation.kind === "tafsir-comparison" ? <section className="comparison-scope"><h2>{copy.comparisonScope}</h2><p>{copy.comparisonDetail}</p></section> : null}{response.status === "PARTIAL" ? <PartialCoverageNote response={response} copy={copy} /> : null}<section className="evidence-section"><div className="evidence-section-heading"><h2>{copy.evidence}</h2></div>{visibleGroups.map(renderGroup)}{collapsedGroups.length ? <details data-testid="other-passages"><summary>{copy.otherPassagesFound}</summary>{collapsedGroups.map(renderGroup)}</details> : null}{unrendered.map((citation) => <UnrenderedEvidenceCard key={citation.citation_id} citation={citation} copy={copy} />)}</section><ProvenanceFooter citations={presentation.citations} copy={copy} />{response.status === "PARTIAL" && onFollowUp && presentation.groups.length === 1 ? <button type="button" className="follow-up-button" onClick={() => onFollowUp(copy.expandWithTafsirPrompt(presentation.groups[0].reference))}>{copy.expandWithTafsir(presentation.groups[0].reference)}</button> : null}<button type="button" className="report-answer" onClick={() => setReportOpen(true)}><Flag aria-hidden size={14} />{copy.reportAnswer}</button></>}
    {reportOpen ? <div className="answer-report-dialog" role="dialog" aria-modal="true" aria-label={copy.reportAnswerAria}><div><h2>{copy.reportAnswer}</h2><p>{copy.reportAnswerDetail}</p><code>{response.turn_id}</code><button type="button" className="follow-up-button" onClick={() => setReportOpen(false)}>{copy.close}</button></div></div> : null}
  </article>;
}
