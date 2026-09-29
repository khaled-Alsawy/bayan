import type { TraceEvent } from "@/lib/contracts";
import type { UiLocale } from "@/lib/i18n";

/** A trace event turned into something a reader can follow: what the step did, and what came back. */
export interface TraceLine {
  /** Short step name, e.g. "بحث دلالي". */
  label: string;
  /** What actually happened, with the numbers and references the step produced. */
  line: string;
  tone: "ok" | "warn" | "fail" | "running";
}

const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";

function num(value: unknown, locale: UiLocale): string {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) return "—";
  const text = String(parsed);
  return locale === "ar" ? text.replace(/\d/gu, (digit) => AR_DIGITS[Number(digit)]) : text;
}

/** Arabic counts a noun four different ways. `forms` is [one, two, few (3-10), many (11+)]. */
function counted(value: number, locale: UiLocale, forms: [string, string, string, string], english: string): string {
  if (locale !== "ar") return `${value} ${english}`;
  if (value === 1) return forms[0];
  if (value === 2) return forms[1];
  const digits = num(value, locale);
  return value >= 3 && value <= 10 ? `${digits} ${forms[2]}` : `${digits} ${forms[3]}`;
}

function ref(value: unknown, locale: UiLocale): string {
  const text = String(value ?? "").trim();
  return locale === "ar" ? text.replace(/\d/gu, (digit) => AR_DIGITS[Number(digit)]) : text;
}

function seconds(ms: number, locale: UiLocale): string {
  const value = (ms / 1000).toFixed(1);
  return locale === "ar" ? `${value.replace(/\d/gu, (d) => AR_DIGITS[Number(d)])} ث` : `${value}s`;
}

function list(values: unknown, locale: UiLocale, limit = 3): string {
  if (!Array.isArray(values) || !values.length) return "";
  return values.slice(0, limit).map((item) => ref(item, locale)).join(" · ");
}

/** Ranked arms carry `reference` per candidate; surface the top few so the ordering is inspectable. */
function topReferences(arm: unknown, locale: UiLocale, limit = 3): string {
  if (!Array.isArray(arm)) return "";
  return arm
    .map((item) => (item && typeof item === "object" ? String((item as Record<string, unknown>).reference ?? "") : ""))
    .filter((value) => value && !value.startsWith("["))
    .slice(0, limit)
    .map((value) => ref(value, locale))
    .join(" · ");
}

function toneOf(phase: TraceEvent["phase"]): TraceLine["tone"] {
  if (phase === "started") return "running";
  if (phase === "degraded" || phase === "skipped" || phase === "rejected") return "warn";
  if (phase === "failed") return "fail";
  return "ok";
}

const LABELS_AR: Record<TraceEvent["node"], string> = {
  scope: "فحص النطاق",
  planner: "فهم السؤال",
  retrieval: "الاسترجاع",
  fusion: "الدمج والترتيب",
  tafsir: "جلب التفسير",
  rerank: "إعادة الترتيب",
  evidence: "تجميع الأدلة",
  verification: "التحقّق",
  synthesis: "صياغة الإجابة",
  composition: "صياغة القراءة",
  gate: "بوابة الأدلة",
};

const LABELS_EN: Record<TraceEvent["node"], string> = {
  scope: "Scope check",
  planner: "Understanding the question",
  retrieval: "Retrieval",
  fusion: "Fusion and ranking",
  tafsir: "Fetching Tafsir",
  rerank: "Reranking",
  evidence: "Assembling evidence",
  verification: "Verification",
  synthesis: "Drafting the answer",
  composition: "Composing the reading",
  gate: "Evidence gate",
};

/** Dispatches on the detail payload, not only the node: one node emits several distinct steps
 *  (a dense query, a passage fetch and a ranking pass all arrive as `retrieval` + `completed`). */
function describe(event: TraceEvent, locale: UiLocale): { label: string; line: string } {
  const ar = locale === "ar";
  const detail = (event.detail ?? {}) as Record<string, unknown>;
  const labels = ar ? LABELS_AR : LABELS_EN;
  const label = labels[event.node];
  const n = (key: string) => num(detail[key], locale);

  if (event.node === "scope") {
    const status = String(detail.status ?? "");
    return {
      label,
      line: ar
        ? `السؤال داخل نطاق الخدمة${status ? ` · ${status}` : ""}`
        : `Question is in scope${status ? ` · ${status}` : ""}`,
    };
  }

  if (event.node === "planner") {
    if (event.phase === "started") return { label, line: ar ? "يخطّط لخطوات البحث…" : "Planning the research steps…" };
    const intent = String(detail.intent ?? "");
    const operations = Array.isArray(detail.operations) ? detail.operations : [];
    const needsClarification = detail.clarification_required === true;
    if (needsClarification) {
      return { label, line: ar ? "السؤال يحتاج توضيحًا قبل البحث" : "The question needs clarification before searching" };
    }
    return {
      label,
      line: ar
        ? `صنّف السؤال كـ ${intent} وخطّط لتنفيذ ${counted(operations.length, locale, ["عملية واحدة", "عمليتين", "عمليات", "عملية"], "")}: ${operations.join("، ")}`
        : `Classified as ${intent}; planned ${operations.length} operations: ${operations.join(", ")}`,
    };
  }

  if ("operation_count" in detail) {
    const count = Number(detail.operation_count ?? 0);
    return {
      label: ar ? "تنفيذ البحث" : "Running retrieval",
      line: ar
        ? `يشغّل ${counted(count, locale, ["عملية واحدة", "عمليتين", "عمليات", "عملية"], "")} من خطة البحث…`
        : `Running ${count} planned operations…`,
    };
  }

  if ("queries" in detail) {
    const queries = Array.isArray(detail.queries) ? detail.queries : [];
    const quoted = queries.map((query) => `«${String(query)}»`).join("، ");
    return {
      label: ar ? "بحث دلالي" : "Dense search",
      line: ar
        ? `حوّل ${quoted} إلى متجّه وبحث به في فهرس الآيات`
        : `Embedded ${queries.map((q) => `"${String(q)}"`).join(", ")} and searched the verse index`,
    };
  }

  if ("arms" in detail) {
    const arms = (detail.arms ?? {}) as Record<string, unknown>;
    const dense = Array.isArray(arms.dense) ? arms.dense.length : 0;
    const lexical = Array.isArray(arms.lexical) ? arms.lexical.length : 0;
    const tafsirFirst = Array.isArray(arms.tafsir_first) ? arms.tafsir_first.length : 0;
    const top = topReferences(arms.dense, locale) || topReferences(arms.tafsir_first, locale);
    return {
      label: ar ? "دمج نتائج المسارات" : "Fusing retrieval arms",
      line: ar
        ? `دلالي ${num(dense, locale)} · لفظي ${num(lexical, locale)} · تفسير ${num(tafsirFirst, locale)}${top ? ` — أعلى الترتيب: ${top}` : ""}`
        : `dense ${dense} · lexical ${lexical} · tafsir ${tafsirFirst}${top ? ` — top ranked: ${top}` : ""}`,
    };
  }

  if ("dropped" in detail || "selected" in detail) {
    const dropped = Array.isArray(detail.dropped) ? detail.dropped.length : Number(detail.dropped ?? 0);
    const selected = Array.isArray(detail.selected) ? detail.selected.length : Number(detail.selected ?? 0);
    return {
      label: ar ? "تصفية المرشّحين" : "Trimming candidates",
      line: ar
        ? `أبقى ${num(selected, locale)} واستبعد ${num(dropped, locale)} مرشّحًا قبل تجميع الأدلة`
        : `kept ${selected}, dropped ${dropped} before assembling evidence`,
    };
  }

  if ("item_count" in detail) {
    const stage = String(detail.stage ?? "");
    const isTafsir = event.node === "tafsir" || stage.includes("tafsir");
    return {
      label: ar ? (isTafsir ? "جلب التفسير" : "جلب النصوص") : isTafsir ? "Fetching Tafsir" : "Fetching text",
      line: ar
        ? `${counted(Number(detail.item_count ?? 0), locale, ["عنصر واحد", "عنصران", "عناصر", "عنصرًا"], "")} من ${String(detail.database ?? "قاعدة البيانات")} · ${String(detail.result_status ?? "")}`
        : `${n("item_count")} items from ${String(detail.database ?? "the database")} · ${String(detail.result_status ?? "")}`,
    };
  }

  if ("ranked_candidate_count" in detail) {
    return {
      label: ar ? "ترتيب المرشّحين" : "Ranking candidates",
      line: ar
        ? `رتّب ${n("ranked_candidate_count")} مرشّحًا بإشارات الترتيب المسجّلة`
        : `Ranked ${n("ranked_candidate_count")} candidates using the recorded ranking signals`,
    };
  }

  if (event.node === "evidence") {
    if (event.phase === "started") return { label, line: ar ? "يجمّع الأدلة المصرّح بعرضها…" : "Assembling authorized evidence…" };
    return {
      label,
      line: ar ? `جهّز ${n("evidence_count")} شاهدًا للعرض` : `Prepared ${n("evidence_count")} evidence items`,
    };
  }

  if (event.node === "verification") {
    if ("terminal_reason" in detail || "scope_status" in detail) {
      const reason = String(detail.terminal_reason ?? "");
      return {
        label: ar ? "القرار النهائي" : "Terminal decision",
        line: reason || (ar ? "اكتمل التحقّق دون اعتراض" : "Verification completed with no objection"),
      };
    }
    if ("kept_evidence_count" in detail) {
      const kept = Number(detail.kept_evidence_count ?? 0);
      const dropped = Number(detail.dropped_evidence_count ?? 0);
      const refs = list(detail.retained_references, locale, 4);
      return {
        label: ar ? "فلترة الصلة بالسؤال" : "Relevance filtering",
        line: ar
          ? `احتفظ بـ${counted(kept, locale, ["شاهد واحد", "شاهدين", "شواهد", "شاهدًا"], "")} واستبعد ${num(dropped, locale)}${refs ? ` — أبقى: ${refs}` : ""}`
          : `kept ${kept}, dropped ${dropped}${refs ? ` — retained: ${refs}` : ""}`,
      };
    }
    if (event.phase === "degraded") {
      return {
        label: ar ? "فلترة الصلة بالسؤال" : "Relevance filtering",
        line: ar
          ? `رجع للفلترة الحتمية بدل المراجعة بالنموذج · أبقى ${n("retained_evidence_count")}`
          : `Fell back to deterministic filtering instead of model review · retained ${n("retained_evidence_count")}`,
      };
    }
    return {
      label: ar ? "تقييم الصلة" : "Assessing relevance",
      line: ar
        ? `يراجع ${n("received_evidence_count")} شاهدًا بطريقة ${String(detail.method ?? "")}`
        : `Reviewing ${n("received_evidence_count")} items via ${String(detail.method ?? "")}`,
    };
  }

  if (event.node === "synthesis") {
    if (event.phase === "started") return { label, line: ar ? "يصوغ إجابة مستندة للأدلة…" : "Drafting a grounded answer…" };
    if (event.phase === "degraded") {
      return {
        label,
        line: ar ? "نُقّح نص الادّعاءات قبل التحقّق منها" : "Claim text was sanitized before validation",
      };
    }
    return {
      label,
      line: ar ? `أخرج إجابة بحالة ${String(detail.status ?? "")}` : `Produced an answer with status ${String(detail.status ?? "")}`,
    };
  }

  if (event.node === "composition") {
    if (event.phase === "started") return { label, line: ar ? "يحوّل الادّعاءات لقراءة متصلة…" : "Turning claims into a connected reading…" };
    const rejections = (detail.gate_rejections_by_rule ?? {}) as Record<string, unknown>;
    const rejected = Object.keys(rejections).length;
    return {
      label,
      line: detail.narrative_created === false
        ? ar ? "لم تُنتج قراءة؛ عُرضت الادّعاءات كما هي" : "No reading produced; claims shown as-is"
        : ar
          ? `كُتبت القراءة${rejected ? ` · رفضت بوابة منع التلفيق ${num(rejected, locale)} قاعدة` : " · اجتازت بوابة منع التلفيق"}`
          : `Reading composed${rejected ? ` · anti-fabrication gate rejected ${rejected} rules` : " · passed the anti-fabrication gate"}`,
    };
  }

  if (event.node === "gate") {
    const satisfied = detail.requirements_satisfied === true;
    return {
      label,
      line: satisfied
        ? ar ? "كل متطلبات الأدلة مستوفاة" : "All evidence requirements satisfied"
        : ar
          ? `${n("missing_requirement_count")} من متطلبات الأدلة غير مستوفاة`
          : `${n("missing_requirement_count")} evidence requirements unmet`,
    };
  }

  return { label, line: event.summary };
}

export function traceLine(event: TraceEvent, locale: UiLocale): TraceLine {
  return { ...describe(event, locale), tone: toneOf(event.phase) };
}

/** The one-line summary on the collapsed panel, e.g. "٧ خطوات · ٦.٣ ث". */
export function traceHeadline(events: TraceEvent[], locale: UiLocale, running: boolean): string {
  const elapsed = events.at(-1)?.elapsed_ms ?? 0;
  if (locale === "ar") {
    if (running) {
      const current = events.at(-1);
      return current ? traceLine(current, "ar").label : "يبدأ البحث…";
    }
    return `${counted(events.length, locale, ["خطوة واحدة", "خطوتان", "خطوات", "خطوة"], "")} · ${seconds(elapsed, locale)}`;
  }
  if (running) {
    const current = events.at(-1);
    return current ? traceLine(current, "en").label : "Starting…";
  }
  return `${events.length} steps · ${seconds(elapsed, locale)}`;
}

export function traceElapsed(event: TraceEvent, locale: UiLocale): string {
  return seconds(event.elapsed_ms, locale);
}
