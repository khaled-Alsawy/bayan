"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Archive,
  ArrowUp,
  BookMarked,
  History,
  Pencil,
  Trash2,
  CircleDot,
  Info,
  Languages,
  Moon,
  Plus,
  Search,
  ShieldCheck,
  Sun,
} from "lucide-react";

import { AnswerView, ExecutionTrace } from "@/components/answer-view";
import { Button } from "@/components/ui/button";
import {
  ConversationDetailSchema,
  ConversationPageSchema,
  ConversationSummarySchema,
  ConversationTurnSchema,
  TraceEventSchema,
  textDirection,
  type ConversationSummary,
  type ConversationTurn,
  type TraceEvent,
} from "@/lib/contracts";
import {
  getMessages,
  localeDirection,
  resolveResponseLocale,
  UI_LOCALE_COOKIE,
  UI_LOCALE_STORAGE_KEY,
  type Messages,
  type UiLocale,
} from "@/lib/i18n";
import { cn } from "@/lib/utils";

const ACTIVE_CONVERSATION_KEY = "quran-active-conversation-id";

function scrollToLatest(container: HTMLDivElement, behavior: ScrollBehavior) {
  if (typeof container.scrollTo === "function") {
    container.scrollTo({ top: container.scrollHeight, behavior });
  }
  else container.scrollTop = container.scrollHeight;
}

function LoadingState({ copy, events, onStop }: { copy: Messages; events: TraceEvent[]; onStop: () => void }) {
  const startedAt = useRef<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  useEffect(() => {
    startedAt.current = Date.now();
    const timer = window.setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - (startedAt.current ?? Date.now())) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, []);
  const latest = events.at(-1);
  return (
    <section className="loading-state" aria-label={copy.loadingAria}>
      <p className="sr-only" aria-live="polite">{latest ? `${latest.node}: ${latest.phase}` : ""}</p>
      <ExecutionTrace events={events} locale={copy.locale} running open />
      <div className="loading-actions"><span>{copy.locale === "ar" ? `${elapsedSeconds.toLocaleString("ar-EG")} ث` : `${elapsedSeconds}s`}</span><button type="button" onClick={onStop}>{copy.locale === "ar" ? "إيقاف وتعديل السؤال" : "Stop and edit question"}</button></div>
    </section>
  );
}

function EmptyState({ copy, onChoose }: { copy: Messages; onChoose: (prompt: string) => void }) {
  const scope = copy.locale === "ar"
    ? "النطاق الحالي: القرآن الكريم وترجماته وتفاسير مختارة. لا يشمل الحديث ولا الفتوى ولا الأحكام."
    : "Current scope: the Qur'an, its translations, and selected tafsir. It does not include hadith, fatwas, or legal rulings.";
  return (
    <section className="empty-state-card paper-card relative overflow-hidden p-6 sm:p-10">
      <div aria-hidden className="absolute -right-14 -top-14 size-52 rounded-full border border-[#d8c49f] opacity-40" />
      <div className="relative max-w-3xl">
        <div className="inline-flex items-center gap-2 rounded-full border border-[#d8c49f] bg-[var(--brass-soft)] px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.16em] text-[#77511c]"><BookMarked aria-hidden size={14} /> {copy.workspaceBadge}</div>
        <h1 className="empty-hero-title mt-6 max-w-2xl text-4xl font-semibold leading-[1.08] tracking-[-0.045em] text-[var(--ink)] sm:text-[3.25rem]">{copy.heroTitle}</h1>
        <p className="empty-hero-detail mt-5 max-w-2xl text-base leading-7 text-[var(--ink-soft)] sm:text-lg sm:leading-8">{copy.heroDetail}</p>
        <p className="empty-hero-accent mt-5 text-xl leading-9 text-[var(--emerald-deep)]">{copy.heroAccent}</p>
        <div className="empty-suggestions mt-9">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-[var(--ink-soft)]">{copy.tryQuestion}</p>
          <div className="mt-3 flex flex-wrap gap-2.5">
            {copy.suggestions.map((item) => <Button key={item.label} variant="outline" size="sm" onClick={() => onChoose(item.prompt)} dir={textDirection(item.label)}>{item.label}</Button>)}
          </div>
        </div>
        <aside className="empty-scope-note"><Info aria-hidden size={16} /><span>{scope}</span></aside>
      </div>
    </section>
  );
}

function LanguageControl({ copy, locale, onChange }: { copy: Messages; locale: UiLocale; onChange: (locale: UiLocale) => void }) {
  return (
    <div role="group" aria-label={copy.languageControl} dir="ltr" className="flex items-center gap-1 rounded-full border bg-[rgba(255,253,249,.72)] p-1 text-xs font-bold backdrop-blur">
      <Languages aria-hidden size={14} className="mx-1 text-[var(--emerald)]" />
      {(["ar", "en"] as const).map((item) => <button key={item} type="button" lang={item} dir={localeDirection(item)} aria-pressed={locale === item} onClick={() => onChange(item)} className={cn("rounded-full px-2.5 py-1.5 focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[var(--focus)]", locale === item ? "bg-[var(--ink)] text-white" : "text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]")}>{item === "ar" ? copy.arabic : copy.english}</button>)}
    </div>
  );
}

export function ResearchWorkspace({ initialLocale = "ar" }: { initialLocale?: UiLocale }) {
  const [locale, setLocale] = useState<UiLocale>(initialLocale);
  const [message, setMessage] = useState("");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [liveEvents, setLiveEvents] = useState<TraceEvent[]>([]);
  const [answerHandoffTurnId, setAnswerHandoffTurnId] = useState<string | null>(null);
  const [isPinnedToBottom, setIsPinnedToBottom] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [dialog, setDialog] = useState<{ kind: "rename" | "archive" | "delete"; item: ConversationSummary } | null>(null);
  const [renameValue, setRenameValue] = useState("");
  // null = follow the viewport: open beside the answer on desktop, closed drawer on narrow screens.
  const [historyPref, setHistoryPref] = useState<boolean | null>(null);
  const [historyAutoOpen, setHistoryAutoOpen] = useState(false);
  const historyOpen = historyPref ?? historyAutoOpen;
  const initialized = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const conversationScrollRef = useRef<HTMLDivElement>(null);
  const pendingTurnRef = useRef<HTMLDivElement>(null);
  const activeRequest = useRef<AbortController | null>(null);
  const programmaticScroll = useRef(false);
  const scrollResetTimer = useRef<number | null>(null);
  const lastTailScrollAt = useRef(0);
  const copy = getMessages(locale);
  const direction = useMemo(() => message.trim() ? textDirection(message) : localeDirection(locale), [locale, message]);

  useEffect(() => { document.documentElement.lang = locale; document.documentElement.dir = localeDirection(locale); }, [locale]);
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => {
    const query = window.matchMedia?.("(min-width: 1024px)");
    if (!query) return;
    const update = () => setHistoryAutoOpen(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  function runProgrammaticScroll(container: HTMLDivElement, scroll: () => void) {
    programmaticScroll.current = true;
    if (scrollResetTimer.current !== null) window.clearTimeout(scrollResetTimer.current);
    scroll();
    // Smooth scrolling emits several scroll events. Keep the guard through the animation so
    // our own movement never looks like the owner deliberately scrolling away.
    scrollResetTimer.current = window.setTimeout(() => { programmaticScroll.current = false; }, 500);
  }

  function scrollBehaviorForTail(): ScrollBehavior {
    const now = performance.now();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const rapid = now - lastTailScrollAt.current < 180;
    lastTailScrollAt.current = now;
    return reducedMotion || rapid ? "auto" : "smooth";
  }

  function scrollElementToTop(container: HTMLDivElement, target: HTMLElement) {
    const top = target.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
    runProgrammaticScroll(container, () => {
      if (typeof container.scrollTo === "function") container.scrollTo({ top, behavior: "auto" });
      else container.scrollTop = top;
    });
  }

  useEffect(() => {
    const container = conversationScrollRef.current;
    if (!container || !loading || !isPinnedToBottom || liveEvents.length === 0) return;
    runProgrammaticScroll(container, () => scrollToLatest(container, scrollBehaviorForTail()));
  }, [liveEvents, loading, isPinnedToBottom]);

  // The handoff is intentionally keyed only to the finished turn; helper functions are
  // render-local wrappers around refs and must not restart the settling window on rerender.
  useLayoutEffect(() => {
    if (!answerHandoffTurnId) return;
    const container = conversationScrollRef.current;
    const answer = document.getElementById(`answer-${answerHandoffTurnId}`);
    if (!container || !(answer instanceof HTMLElement)) return;

    // AnswerView contains evidence and tafsir sections that can grow during the browser's
    // layout work.  A single rAF can therefore scroll to a location that immediately moves.
    // Wait for a quiet ResizeObserver interval, then cross two paint boundaries before the
    // handoff.  If a measured answer section grows meanwhile, restart the quiet interval.
    let firstFrame = 0;
    let secondFrame = 0;
    let settleTimer: number | null = null;
    const handoff = () => {
      firstFrame = window.requestAnimationFrame(() => {
        secondFrame = window.requestAnimationFrame(() => {
          scrollElementToTop(container, answer);
          setAnswerHandoffTurnId(null);
        });
      });
    };
    const scheduleAfterLayoutSettles = () => {
      if (settleTimer !== null) window.clearTimeout(settleTimer);
      settleTimer = window.setTimeout(handoff, 80);
    };
    // ResizeObserver is available in supported browsers; the timer + double-rAF path keeps
    // non-layout test environments and older browsers safe without reviving the one-rAF race.
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleAfterLayoutSettles);
    observer?.observe(answer);
    scheduleAfterLayoutSettles();
    return () => {
      observer?.disconnect();
      if (settleTimer !== null) window.clearTimeout(settleTimer);
      window.cancelAnimationFrame(firstFrame);
      window.cancelAnimationFrame(secondFrame);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [answerHandoffTurnId]);

  useEffect(() => {
    const container = conversationScrollRef.current;
    const pendingTurn = pendingTurnRef.current;
    if (!container || !pendingTurn || !loading) return;
    const top = pendingTurn.getBoundingClientRect().top
      - container.getBoundingClientRect().top
      + container.scrollTop;
    programmaticScroll.current = true;
    if (scrollResetTimer.current !== null) window.clearTimeout(scrollResetTimer.current);
    if (typeof container.scrollTo === "function") container.scrollTo({ top, behavior: "auto" });
    else container.scrollTop = top;
    scrollResetTimer.current = window.setTimeout(() => { programmaticScroll.current = false; }, 500);
  }, [loading, pendingQuestion]);

  useEffect(() => () => {
    if (scrollResetTimer.current !== null) window.clearTimeout(scrollResetTimer.current);
  }, []);

  async function loadConversationList() {
    setHistoryLoading(true);
    try {
      const result = await fetch("/api/conversations?limit=50", { cache: "no-store" });
      const parsed = ConversationPageSchema.safeParse(await result.json());
      if (!result.ok || !parsed.success) throw new Error();
      setConversations(parsed.data.items);
      const saved = window.localStorage.getItem(ACTIVE_CONVERSATION_KEY);
      const selected = parsed.data.items.find((item) => item.conversation_id === saved)?.conversation_id ?? parsed.data.items[0]?.conversation_id ?? null;
      if (selected) {
        const detailResult = await fetch(`/api/conversations/${selected}?limit=100`, { cache: "no-store" });
        const detail = ConversationDetailSchema.safeParse(await detailResult.json());
        if (!detailResult.ok || !detail.success) throw new Error();
        setActiveId(selected); setTurns(detail.data.turns);
        window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, selected);
      }
    } catch { setHistoryError(getMessages(initialLocale).historyUnavailable); }
    finally { setHistoryLoading(false); }
  }

  // The initial history load is intentionally one-shot; later refreshes are explicit mutations.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (!initialized.current) { initialized.current = true; void loadConversationList(); } }, []);

  // Evidence anchors live inside .conversation-scroll, so the browser does not honour a hash
  // present at load or one set before its target has rendered. Resolve it ourselves.
  useEffect(() => {
    const scrollToHash = () => {
      const id = window.location.hash.slice(1);
      if (!id) return;
      const target = document.getElementById(id);
      if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
    };
    const timer = window.setTimeout(scrollToHash, 120);
    window.addEventListener("hashchange", scrollToHash);
    return () => { window.clearTimeout(timer); window.removeEventListener("hashchange", scrollToHash); };
  }, [turns]);

  async function refreshConversationList() {
    const result = await fetch("/api/conversations?limit=50", { cache: "no-store" });
    const parsed = ConversationPageSchema.safeParse(await result.json());
    if (result.ok && parsed.success) setConversations(parsed.data.items);
  }

  async function createConversation(): Promise<string | null> {
    setError(null);
    const result = await fetch("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ locale }) });
    const parsed = ConversationSummarySchema.safeParse(await result.json());
    if (!result.ok || !parsed.success) { setError(copy.genericUnavailable); return null; }
    setConversations((items) => [parsed.data, ...items]); setActiveId(parsed.data.conversation_id); setTurns([]);
    window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, parsed.data.conversation_id);
    requestAnimationFrame(() => textareaRef.current?.focus()); return parsed.data.conversation_id;
  }

  async function selectConversation(id: string) {
    setHistoryLoading(true); setError(null);
    try {
      const result = await fetch(`/api/conversations/${id}?limit=100`, { cache: "no-store" });
      const parsed = ConversationDetailSchema.safeParse(await result.json());
      if (!result.ok || !parsed.success) throw new Error();
      setActiveId(id); setTurns(parsed.data.turns); window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, id);
    } catch { setHistoryError(copy.historyUnavailable); }
    finally { setHistoryLoading(false); }
  }

  async function updateConversation(item: ConversationSummary, update: { title?: string; archived?: boolean }) {
    const result = await fetch(`/api/conversations/${item.conversation_id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(update) });
    const parsed = ConversationSummarySchema.safeParse(await result.json());
    if (!result.ok || !parsed.success) { setError(copy.genericUnavailable); return; }
    if (update.archived) {
      const remaining = conversations.filter((value) => value.conversation_id !== item.conversation_id); setConversations(remaining);
      if (activeId === item.conversation_id) { setActiveId(null); setTurns([]); window.localStorage.removeItem(ACTIVE_CONVERSATION_KEY); if (remaining[0]) void selectConversation(remaining[0].conversation_id); }
    } else setConversations((items) => items.map((value) => value.conversation_id === item.conversation_id ? parsed.data : value));
  }

  async function deleteConversation(item: ConversationSummary) {
    const result = await fetch(`/api/conversations/${item.conversation_id}`, { method: "DELETE" });
    if (!result.ok) { setError(copy.genericUnavailable); return; }
    const remaining = conversations.filter((value) => value.conversation_id !== item.conversation_id); setConversations(remaining);
    if (activeId === item.conversation_id) { setActiveId(null); setTurns([]); window.localStorage.removeItem(ACTIVE_CONVERSATION_KEY); if (remaining[0]) void selectConversation(remaining[0].conversation_id); }
  }

  async function submitResearch(question: string) {
    const trimmed = question.trim(); if (trimmed.length < 2 || loading) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setLoading(true); setError(null); setPendingQuestion(trimmed); setLiveEvents([]); setIsPinnedToBottom(true); setMessage("");
    try {
      const conversationId = activeId ?? await createConversation(); if (!conversationId) return;
      const result = await fetch(`/api/conversations/${conversationId}/turns/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ message: trimmed, locale: resolveResponseLocale(trimmed, locale), idempotency_key: crypto.randomUUID() }) });
      if (!result.ok) throw new Error();
      let finalTurn: ConversationTurn | null = null;
      if (result.headers.get("content-type")?.includes("application/json")) {
        const turn = ConversationTurnSchema.safeParse(await result.json()); if (turn.success) finalTurn = turn.data;
      } else {
        if (!result.body) throw new Error(); const reader = result.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
        while (true) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const frames = buffer.split("\n\n"); buffer = frames.pop() ?? ""; for (const frame of frames) { const event = frame.match(/^event: (.+)$/m)?.[1]; const data = frame.match(/^data: (.+)$/m)?.[1]; if (!event || !data) continue; const parsedData = JSON.parse(data); if (event === "trace") { const trace = TraceEventSchema.safeParse(parsedData); if (trace.success) setLiveEvents((items) => [...items, trace.data]); } else if (event === "final") { const turn = ConversationTurnSchema.safeParse(parsedData); if (turn.success) finalTurn = turn.data; } else throw new Error(); } }
      }
      if (!finalTurn) throw new Error();
      setTurns((items) => [...items, finalTurn]); setPendingQuestion(null); setLiveEvents([]); setIsPinnedToBottom(false); setAnswerHandoffTurnId(finalTurn.response.turn_id); await refreshConversationList();
    } catch (caught) { if (!(caught instanceof DOMException && caught.name === "AbortError")) setError(copy.genericUnavailable); }
    finally { if (activeRequest.current === controller) activeRequest.current = null; setLoading(false); }
  }

  function stopResearch() {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setLoading(false);
    setMessage((current) => current || pendingQuestion || "");
    setPendingQuestion(null);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function onConversationScroll() {
    if (programmaticScroll.current) return;
    const container = conversationScrollRef.current;
    if (!container) return;
    setIsPinnedToBottom(container.scrollHeight - container.scrollTop - container.clientHeight <= 48);
  }

  function unpinForOwnerReading() { programmaticScroll.current = false; setIsPinnedToBottom(false); }
  function jumpToLatest() { const container = conversationScrollRef.current; if (!container) return; setIsPinnedToBottom(true); runProgrammaticScroll(container, () => scrollToLatest(container, "auto")); }

  function changeLocale(nextLocale: UiLocale) { setLocale(nextLocale); document.title = getMessages(nextLocale).metaTitle; window.localStorage.setItem(UI_LOCALE_STORAGE_KEY, nextLocale); document.cookie = `${UI_LOCALE_COOKIE}=${nextLocale}; Max-Age=31536000; Path=/; SameSite=Lax`; }
  function onSubmit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void submitResearch(message); }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submitResearch(message); } }
  function choosePrompt(prompt: string) { setMessage(prompt); requestAnimationFrame(() => textareaRef.current?.focus()); }
  function startNewChat() {
    void createConversation();
    if (window.matchMedia?.("(max-width: 1023px)").matches) setHistoryPref(false);
  }
  const showEmpty = !historyLoading && turns.length === 0 && !pendingQuestion;
  const composer = <div data-testid="composer-dock" className="composer-dock">
    <form onSubmit={onSubmit} className="mx-auto max-w-4xl" aria-label={copy.formAria}><div className="composer-row rounded-[1.35rem] border border-[var(--border-strong)] bg-[var(--paper)] p-2 shadow-[0_12px_38px_rgba(35,49,44,.12)] focus-within:ring-3 focus-within:ring-[rgba(46,139,116,.15)]"><textarea ref={textareaRef} value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={onKeyDown} dir={direction} rows={1} maxLength={2000} disabled={loading} placeholder={copy.questionPlaceholder} aria-label={copy.questionAria} className={cn("min-h-12 w-full resize-none bg-transparent px-3 py-2 text-[15px] leading-6 outline-none", direction === "rtl" && "font-arabic text-right text-lg")} /><Button type="submit" size="icon" disabled={message.trim().length < 2 || loading} aria-label={loading ? copy.researching : copy.submit}><ArrowUp aria-hidden size={18} /></Button></div><div className="composer-hints flex items-center justify-between gap-3 border-t px-2 pt-2"><div className="flex min-w-0 items-center gap-2 text-[11px] font-semibold text-[var(--ink-soft)]"><CircleDot aria-hidden size={12} className="text-[var(--emerald)]" /><span className="truncate">{direction === "rtl" ? copy.arabicDetected : copy.englishDetected} · {copy.enterToSend}</span></div></div><p className="mt-2 text-center text-[10px] leading-4 text-[var(--ink-soft)]">{copy.scopeNotice}</p></form>
  </div>;
  return (
    <div className="workspace-root">
      <header className="workspace-header">
        <div className="app-header-inner"><div><h1>{locale === "ar" ? "بَيان" : "Bayān"}</h1><p>{locale === "ar" ? "بحثٌ في القرآن وتفاسيره — يجيب من نصوص معروضة" : "Qur'an and tafsir research — answers only from shown sources"}</p></div><div className="app-header-actions"><button type="button" className="theme-button" onClick={() => setHistoryPref((value) => !(value ?? historyAutoOpen))} aria-expanded={historyOpen} aria-controls="history-sidebar" aria-label={historyOpen ? copy.closeHistory : copy.openHistory}><History aria-hidden size={16} /></button>{!historyOpen ? <button type="button" className="header-new-chat-button" onClick={startNewChat} aria-label={copy.newChat}><Plus aria-hidden size={18} /></button> : null}<button type="button" className="theme-button" onClick={() => setTheme((value) => value === "light" ? "dark" : "light")} aria-label={theme === "light" ? (locale === "ar" ? "تبديل الوضع الليلي" : "Switch to dark mode") : (locale === "ar" ? "تبديل الوضع النهاري" : "Switch to light mode")}>{theme === "light" ? <Moon aria-hidden size={16} /> : <Sun aria-hidden size={16} />}</button><LanguageControl copy={copy} locale={locale} onChange={changeLocale} /></div></div>
      </header>
      <div className={cn("chat-layout", !historyOpen && "history-collapsed", historyPref === null && "history-auto")}>
        {historyOpen ? <button type="button" aria-hidden tabIndex={-1} className="history-backdrop" onClick={() => setHistoryPref(false)} /> : null}
        {historyOpen ? <aside id="history-sidebar" className="history-sidebar" aria-label={copy.history}>
          <div className="history-sidebar-head"><button type="button" className="sidebar-new-chat-button" onClick={startNewChat}><Plus aria-hidden size={17} />{copy.newChat}</button><h2>{copy.history}</h2></div>
          {conversations.length === 0 ? <p className="history-empty">{copy.noChats}</p> : <ul className="history-list">{conversations.map((item) => <li key={item.conversation_id} className={cn("history-item", item.conversation_id === activeId && "history-item-active")}>
            <button type="button" className="history-item-open" dir={textDirection(item.title)} aria-current={item.conversation_id === activeId ? "true" : undefined} onClick={() => { void selectConversation(item.conversation_id); if (window.matchMedia("(max-width: 1023px)").matches) setHistoryPref(false); }}>{item.title}</button>
            <div className="history-item-actions">
              <button type="button" className="history-action" aria-label={copy.rename} title={copy.rename} onClick={() => { setRenameValue(item.title); setDialog({ kind: "rename", item }); }}><Pencil aria-hidden size={13} /></button>
              <button type="button" className="history-action" aria-label={copy.archive} title={copy.archive} onClick={() => setDialog({ kind: "archive", item })}><Archive aria-hidden size={13} /></button>
              <button type="button" className="history-action" aria-label={copy.delete} title={copy.delete} onClick={() => setDialog({ kind: "delete", item })}><Trash2 aria-hidden size={13} /></button>
            </div>
          </li>)}</ul>}
        </aside> : null}
        <main id="main-content" className="conversation-shell min-w-0">
          <div ref={conversationScrollRef} data-testid="conversation-scroll" className="conversation-scroll" onScroll={onConversationScroll} onWheel={unpinForOwnerReading} onTouchMove={unpinForOwnerReading} onKeyDown={unpinForOwnerReading}>
            <div className="conversation-stream">
              {historyError ? <p role="status" className="rounded-xl border bg-[var(--brass-soft)] p-3 text-sm">{historyError}</p> : null}
              {historyLoading && turns.length === 0 ? <LoadingState copy={copy} events={liveEvents} onStop={stopResearch} /> : null}
              {showEmpty ? <EmptyState copy={copy} onChoose={choosePrompt} /> : null}
              {turns.map((turn) => (
                <div key={turn.turn_id} data-testid="conversation-turn" className="conversation-turn">
                  <section className="user-message">
                    <div className="flex items-center gap-2 text-[11px] font-bold text-[var(--ink-soft)]"><Search aria-hidden size={13} />{copy.yourQuestion}</div>
                    <p dir={textDirection(turn.user_message)} className="mt-2 text-lg font-medium leading-8">{turn.user_message}</p>
                  </section>
                  <AnswerView response={turn.response} userMessage={turn.user_message} locale={locale} traceEvents={turn.trace_events} onReloadSource={() => void submitResearch(turn.user_message)} onFollowUp={choosePrompt} onEditQuestion={() => choosePrompt(turn.user_message)} onClarificationChoice={loading ? undefined : (choice) => void submitResearch(choice)} />
                </div>
              ))}
              {pendingQuestion ? <div ref={pendingTurnRef} className="conversation-turn" data-testid="pending-turn"><section className="user-message"><div className="flex items-center gap-2 text-[11px] font-bold text-[var(--ink-faint)]"><Search aria-hidden size={13} />{copy.yourQuestion}</div><p dir={textDirection(pendingQuestion)} className="mt-2 text-lg font-medium leading-8">{pendingQuestion}</p></section>{loading ? <LoadingState copy={copy} events={liveEvents} onStop={stopResearch} /> : <button type="button" className="follow-up-button" onClick={() => void submitResearch(pendingQuestion)}>{copy.retry}</button>}</div> : null}
              {error ? <section className="paper-card p-6" role="alert"><ShieldCheck aria-hidden size={20} /><p className="mt-3 text-sm">{error}</p></section> : null}
              {loading && !isPinnedToBottom ? <button type="button" className="jump-to-latest" onClick={jumpToLatest}>{copy.jumpToLatest}</button> : null}
            </div>
          </div>
          {composer}
        </main>
      </div>
      {dialog ? <div className="fixed inset-0 z-50 grid place-items-center bg-black/35 p-4" role="dialog" aria-modal="true" aria-label={dialog.kind === "rename" ? copy.rename : dialog.kind === "archive" ? copy.archive : copy.delete}><form className="w-full max-w-sm rounded-xl border bg-[var(--paper)] p-5 shadow-xl" onSubmit={(event) => { event.preventDefault(); if (dialog.kind === "rename" && renameValue.trim()) void updateConversation(dialog.item, { title: renameValue.trim() }); if (dialog.kind === "archive") void updateConversation(dialog.item, { archived: true }); if (dialog.kind === "delete") void deleteConversation(dialog.item); setDialog(null); }}><h2 className="text-lg font-semibold">{dialog.kind === "rename" ? copy.rename : dialog.kind === "archive" ? copy.archive : copy.delete}</h2>{dialog.kind === "rename" ? <input autoFocus value={renameValue} onChange={(event) => setRenameValue(event.target.value)} className="mt-4 w-full rounded-lg border bg-transparent p-2" aria-label={copy.renamePrompt} /> : <p className="mt-3 text-sm leading-6 text-[var(--ink-soft)]">{dialog.kind === "archive" ? copy.archiveConfirm : copy.deleteConfirm}</p>}<div className="mt-5 flex justify-end gap-2"><button type="button" className="follow-up-button" onClick={() => setDialog(null)}>{copy.locale === "ar" ? "إلغاء" : "Cancel"}</button><Button type="submit">{dialog.kind === "rename" ? copy.rename : dialog.kind === "archive" ? copy.archive : copy.delete}</Button></div></form></div> : null}
    </div>
  );
}
