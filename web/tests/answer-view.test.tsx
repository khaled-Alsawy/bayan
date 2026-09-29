import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AnswerView, ExecutionTrace, SourcesPanel } from "@/components/answer-view";
import { ANSWER_STATUSES } from "@/lib/contracts";
import { buildAnswerPresentation } from "@/lib/presentation";
import {
  ARABIC_EXACT,
  ENGLISH_EXACT,
  arabicCitation,
  makeTafsirResponse,
  makePartialResearchResponse,
  makeResearchResponse,
  translationCitation,
} from "./fixtures";

function makePassageOrderResponse(citedReferences: string[] = ["113:5"]) {
  const response = makeResearchResponse();
  const references = ["2:107", "2:108", "2:109", "2:110", "2:111", "113:3", "113:4", "113:5"];
  const citations = references.map((reference) => ({
    ...arabicCitation,
    citation_id: `cite-${reference}`,
    evidence_id: `evidence-${reference}`,
    display_label: `Quran ${reference}`,
    reference: { ...arabicCitation.reference, value: reference },
  }));
  response.citations = citations;
  response.blocks = [
    {
      block_type: "CLAIM",
      claim_id: "passage-order-claim",
      text: "تقتصر القراءة على الشاهد المذكور.",
      citations: citations.filter((citation) => citedReferences.includes(citation.reference.value)),
    },
    ...citations.map((citation) => ({
      block_type: "SOURCE" as const,
      block_id: `source-${citation.reference.value}`,
      exact_text: `نص الآية ${citation.reference.value}`,
      citation,
      visibility: "VISIBLE" as const,
    })),
  ];
  return response;
}

describe("AnswerView", () => {
  it("keeps a persisted execution trace collapsed until the owner opens it", async () => {
    render(<ExecutionTrace locale="en" events={[{ seq: 1, elapsed_ms: 12, node: "retrieval", phase: "completed", summary: "Retrieved two candidates", detail: { item_count: 2, database: "quran", result_status: "success" }, provider: null }]} />);
    const user = userEvent.setup();
    const toggle = screen.getByRole("button", { name: /1 steps/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Fetching text")).not.toBeInTheDocument();
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Fetching text")).toBeInTheDocument();
  });

  it("explains a step in readable terms and keeps the raw payload one click away", async () => {
    render(<ExecutionTrace locale="en" open events={[{ seq: 1, elapsed_ms: 3275, node: "retrieval", phase: "completed", summary: "Embedded dense retrieval query", detail: { queries: ["المصايب"] }, provider: null }]} />);
    // The reader sees what was searched for, not just that a search happened.
    expect(screen.getByText(/Embedded "المصايب" and searched the verse index/)).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: /Dense search/ }));
    expect(screen.getByText("Raw payload")).toBeInTheDocument();
    expect(screen.getByText("Embedded dense retrieval query")).toBeInTheDocument();
  });

  it("renders the workflow above the answer it produced", () => {
    render(<AnswerView response={makeResearchResponse()} locale="en" traceEvents={[{ seq: 1, elapsed_ms: 12, node: "planner", phase: "completed", summary: "Research plan accepted", detail: { intent: "QURAN_TOPIC_SEARCH", operations: ["FETCH_QURAN_PASSAGE"], clarification_required: false }, provider: null }]} />);
    const answer = screen.getByTestId("assistant-answer");
    const trace = screen.getByTestId("execution-trace");
    expect(answer.firstElementChild).toBe(trace);
  });

  it("renders exact Arabic and exposes its translation through a passage-level toggle", async () => {
    render(<AnswerView response={makeResearchResponse()} locale="en" />);
    expect(screen.getByText(ARABIC_EXACT)).toHaveAttribute("dir", "rtl");
    await userEvent.setup().click(screen.getByRole("button", { name: "Show translation" }));
    expect(screen.getByText(ENGLISH_EXACT)).toHaveAttribute("dir", "ltr");
    expect(screen.getAllByRole("link", { name: /Open verse text/ }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: /Open translation/ }).length).toBeGreaterThan(0);
  });

  it("shows cited 113:3–5 before uncited 2:107–111 and collapses the latter", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="en" />);
    const cards = screen.getAllByTestId("verse-card");
    expect(cards[0]).toHaveAttribute("id", "passage-113:3-5");
    const otherPassages = screen.getByTestId("other-passages") as HTMLDetailsElement;
    expect(otherPassages.open).toBe(false);
    expect(within(otherPassages).getByTestId("verse-card")).toHaveAttribute("id", "passage-2:107-111");
    expect(within(otherPassages).getByText("Other passages found in the search")).toBeInTheDocument();
  });

  it("formats Arabic passage titles with the surah name", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="ar" />);
    expect(screen.getByText("سورة البقرة ١٠٧–١١١")).toBeInTheDocument();
  });

  it("formats Quran citation chips in Arabic and keeps the numeric reference in the tooltip", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="ar" />);
    const chip = screen.getByRole("link", { name: "الفلق ٥" });
    expect(chip).toHaveAttribute("title", expect.stringContaining("113:5"));
  });

  it("formats passage titles and citation chips in English", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="en" />);
    expect(screen.getByText("Al-Baqarah 107–111")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Al-Falaq 5" })).toHaveAttribute("title", expect.stringContaining("113:5"));
  });

  it("keeps uncited passages collapsed for an answer that cites Tafsir Al-Mukhtasar", () => {
    const response = makePassageOrderResponse();
    const tafsirCitation = {
      ...arabicCitation,
      citation_id: "cite-tafsir-mukhtasar",
      evidence_id: "evidence-tafsir-mukhtasar",
      source_kind: "TAFSIR" as const,
      reference: { ...arabicCitation.reference, value: "113:5" },
      source_revision: { ...arabicCitation.source_revision, resource_alias: "qul-tafsir-almukhtasar-ar-v1" },
    };
    response.citations.push(tafsirCitation);
    response.blocks.push({
      block_type: "SOURCE",
      block_id: "source-tafsir-mukhtasar",
      exact_text: "نص تفسير المختصر المسترجع.",
      citation: tafsirCitation,
      visibility: "VISIBLE",
    });
    expect(buildAnswerPresentation(response, "").kind).toBe("single-tafsir");

    render(<AnswerView response={response} locale="ar" />);

    expect(screen.getAllByTestId("verse-card")[0]).toHaveAttribute("id", "passage-113:3-5");
    const otherPassages = screen.getByTestId("other-passages") as HTMLDetailsElement;
    expect(otherPassages.open).toBe(false);
    expect(within(otherPassages).getByTestId("verse-card")).toHaveAttribute("id", "passage-2:107-111");
  });

  it("does not render an empty details element when every passage is cited", () => {
    render(<AnswerView response={makePassageOrderResponse(["2:107", "113:3"])} locale="en" />);
    expect(screen.queryByTestId("other-passages")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("verse-card").map((card) => card.id)).toEqual([
      "passage-2:107-111",
      "passage-113:3-5",
    ]);
  });

  it("leaves source-only topic evidence in its existing expanded list", () => {
    const response = makePassageOrderResponse();
    response.blocks = response.blocks.filter((block) => block.block_type === "SOURCE");
    render(<AnswerView response={response} userMessage="ماذا يقول القرآن عن الآيات؟" locale="en" />);
    expect(screen.getByRole("heading", { name: "Evidence" })).toBeInTheDocument();
    expect(screen.getAllByTestId("verse-card").map((card) => card.id)).toEqual([
      "passage-2:107-111",
      "passage-113:3-5",
    ]);
    expect(screen.queryByTestId("other-passages")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Generated reading")).not.toBeInTheDocument();
  });

  it("orders two cited groups by their first citation in the reading", () => {
    const response = makePassageOrderResponse(["113:5", "2:109"]);
    const claim = response.blocks.find((block) => block.block_type === "CLAIM");
    if (claim?.block_type !== "CLAIM") throw new Error("fixture claim missing");
    claim.citations.reverse();
    render(<AnswerView response={response} locale="en" />);
    const visibleGroups = [...document.querySelectorAll(".evidence-section > .passage-group .verse-card")];
    expect(visibleGroups.map((card) => card.id)).toEqual(["passage-113:3-5", "passage-2:107-111"]);
  });

  it("opens the collapsed passages when a reading chip targets one", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="en" />);
    const otherPassages = screen.getByTestId("other-passages") as HTMLDetailsElement;
    const chip = screen.getByRole("link", { name: "Al-Falaq 5" });
    chip.setAttribute("href", "#evidence-evidence-2:107");
    fireEvent.click(chip);
    expect(otherPassages.open).toBe(true);
    expect(document.getElementById("evidence-evidence-2:107")).toBeInTheDocument();
  });

  it("renders honest insufficient notices above collapsed possibly-related sources", () => {
    const response = makeResearchResponse("INSUFFICIENT_EVIDENCE");
    const answered = makeResearchResponse();
    response.output_kind = "SOURCE_ONLY";
    response.citations = answered.citations;
    response.blocks = [
      {
        block_type: "NOTICE",
        notice_code: "INSUFFICIENT_EVIDENCE",
        text: "I couldn't find passages that directly answer this question.",
      },
      { block_type: "SOURCE_VIEW_HEADING", text: "Possibly related results" },
      ...answered.blocks.filter((block) => block.block_type === "SOURCE").map((block) => ({ ...block, visibility: "COLLAPSED_CONTEXT" as const })),
    ];

    render(<AnswerView response={response} locale="en" />);

    const notice = screen.getByRole("status");
    const results = screen.getByTestId("possibly-related-results");
    expect(notice).toHaveTextContent("I couldn't find passages that directly answer this question.");
    expect((results as HTMLDetailsElement).open).toBe(false);
    expect(screen.getByText("Possibly related results")).toBeInTheDocument();
    expect(notice.compareDocumentPosition(results) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("uses the formal Arabic insufficient-evidence copy and related-results heading", () => {
    const response = makeResearchResponse("INSUFFICIENT_EVIDENCE");
    const answered = makeResearchResponse();
    response.output_kind = "SOURCE_ONLY";
    response.citations = answered.citations;
    response.blocks = [
      { block_type: "NOTICE", notice_code: "INSUFFICIENT_EVIDENCE", text: "backend notice" },
      { block_type: "SOURCE_VIEW_HEADING", text: "نتائج قد تكون ذات صلة" },
      ...answered.blocks.filter((block) => block.block_type === "SOURCE").map((block) => ({ ...block, visibility: "COLLAPSED_CONTEXT" as const })),
    ];
    render(<AnswerView response={response} locale="ar" />);
    expect(screen.getByRole("status")).toHaveTextContent("لم أجد آيات كافية تجيب عن هذا السؤال إجابة مباشرة.");
    expect(screen.getByText("نتائج قد تكون ذات صلة")).toBeInTheDocument();
  });

  it("renders the formal other-passages label in Arabic", () => {
    render(<AnswerView response={makePassageOrderResponse()} locale="ar" />);
    expect(screen.getByText("آيات أخرى ظهرت في البحث")).toBeInTheDocument();
  });

  it("renders clarification candidates as keyboard-operable localized buttons", async () => {
    const onChoice = vi.fn();
    render(
      <AnswerView
        response={makeResearchResponse("CLARIFICATION_REQUIRED")}
        locale="ar"
        onClarificationChoice={onChoice}
      />,
    );
    const firstChoice = screen.getByRole("button", { name: "الحسد والإصابة بالعين" });
    expect(firstChoice).toHaveAttribute("dir", "rtl");
    const user = userEvent.setup();
    await user.tab();
    expect(firstChoice).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(onChoice).toHaveBeenCalledWith("الحسد والإصابة بالعين");
  });

  it.each(ANSWER_STATUSES)("renders the %s state safely", (status) => {
    render(<AnswerView response={makeResearchResponse(status)} />);
    if (status === "ANSWERABLE") {
      expect(screen.getByText("تبيّن الآية أن اليسر يقترن بالعسر.")).toBeInTheDocument();
    } else if (status === "PARTIAL") {
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      expect(screen.getByTestId("partial-coverage-note")).toBeInTheDocument();
    } else {
      expect(screen.getByRole("status")).toBeInTheDocument();
    }
  });

  it("renders structured provenance and safe canonical links", () => {
    render(<SourcesPanel citations={[arabicCitation, translationCitation]} locale="en" />);
    expect(screen.getByText("Quran Ash-Sharh 6 — Tanzil Project")).toBeInTheDocument();
    expect(screen.getByText("tanzil-uthmani-v1.1 · rev-arabic-1")).toBeInTheDocument();
    expect(screen.getByText("Rowwad Translation Center / QuranEnc")).toBeInTheDocument();
    const links = screen.getAllByRole("link", { name: /Open (?:verse text|translation)/ });
    expect(links[0]).toHaveAttribute("target", "_blank");
    expect(links[0]).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("never mutates canonical source text when the shell language changes", async () => {
    const { rerender } = render(<AnswerView response={makeResearchResponse()} locale="ar" />);
    expect(screen.getByText(ARABIC_EXACT)).toHaveTextContent(ARABIC_EXACT);

    rerender(<AnswerView response={makeResearchResponse()} locale="en" />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Show translation" }));
    expect(screen.getByText(ARABIC_EXACT)).toHaveTextContent(ARABIC_EXACT);
    expect(screen.getByText(ARABIC_EXACT)).toHaveAttribute("dir", "rtl");
    expect(screen.getByText(ENGLISH_EXACT)).toHaveTextContent(ENGLISH_EXACT);
    expect(screen.getByText(ENGLISH_EXACT)).toHaveAttribute("dir", "ltr");
  });

  it("shows an actionable planner failure with notice tone and no shield icon", () => {
    const response = makeResearchResponse("FAILED_SAFE");
    response.blocks = [{
      block_type: "NOTICE",
      notice_code: "QUESTION_NOT_UNDERSTOOD",
      text: "The research plan could not be validated safely.",
    }];
    render(<AnswerView response={response} locale="ar" />);
    expect(screen.getByText("تعذّرت الإجابة هذه المرة")).toBeInTheDocument();
    expect(screen.getByText("لم أتمكن من فهم السؤال بهذه الصيغة. جرّب صياغته بشكل أبسط، أو اسأل عن الموضوع وحده.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveClass("terminal-panel-notice");
    expect(document.querySelector(".lucide-shield-alert")).not.toBeInTheDocument();
    expect(screen.queryByText("The research plan could not be validated safely.")).not.toBeInTheDocument();
  });

  it("localizes answer-verification and source-unavailable notices in English", () => {
    const response = makeResearchResponse("FAILED_SAFE");
    response.blocks = [{ block_type: "NOTICE", notice_code: "ANSWER_NOT_VERIFIED", text: "hidden backend detail" }];
    const { rerender } = render(<AnswerView response={response} locale="en" />);
    expect(screen.getByText("I couldn't answer this one")).toBeInTheDocument();
    expect(screen.getByText("I drafted an answer but couldn't verify every sentence against the sources, so I didn't show it. Try again or rephrase.")).toBeInTheDocument();
    expect(screen.queryByText("hidden backend detail")).not.toBeInTheDocument();

    response.blocks = [{ block_type: "NOTICE", notice_code: "SOURCE_UNAVAILABLE", text: "hidden backend detail" }];
    rerender(<AnswerView response={response} locale="en" />);
    expect(screen.getByText("I couldn't reach the sources right now. Try again shortly.")).toBeInTheDocument();
  });

  it("offers a localized reload action for policy-redacted Tafsir source text", () => {
    const response = makeResearchResponse();
    const source = response.blocks.find((block) => block.block_type === "SOURCE");
    if (!source || source.block_type !== "SOURCE") throw new Error("fixture source missing");
    source.exact_text = "[SOURCE_TEXT_NOT_STORED_POLICY]";
    source.citation = { ...source.citation, source_kind: "TAFSIR" };
    render(<AnswerView response={response} locale="en" onReloadSource={() => undefined} />);
    expect(screen.getByRole("button", { name: "Reload source text" })).toBeInTheDocument();
    expect(screen.queryByText("[SOURCE_TEXT_NOT_STORED_POLICY]")).not.toBeInTheDocument();
  });

  it("uses a passage card when the request contains a verse reference", () => {
    render(<AnswerView response={makeResearchResponse()} userMessage="Show Quran 94:6" locale="ar" />);
    expect(screen.getByText("الشواهد")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "إخفاء الترجمة" })).toHaveAttribute("aria-expanded", "true");
  });

  it("keeps a two-source Tafsir comparison as two retrieved documents", () => {
    render(<AnswerView response={makeTafsirResponse(true)} userMessage="قارن بين تفسير ابن كثير والجلالين للآية 94:6" locale="ar" />);
    expect(screen.getByText("الشواهد")).toBeInTheDocument();
    expect(screen.getAllByTestId("tafsir-card")).toHaveLength(2);
  });

  it("expands Tafsir in place with a return-to-answer affordance", async () => {
    render(<AnswerView response={makeTafsirResponse()} locale="ar" />);
    const expand = screen.getByRole("button", { name: /اقرأ كاملًا/ });
    await userEvent.setup().click(expand);
    expect(screen.getByTestId("tafsir-card")).toHaveAttribute("data-tafsir-expanded", "true");
    expect(screen.getByRole("button", { name: "العودة إلى الجواب" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "طَيّ المقتطف" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("keeps report-answer handling inside the application", async () => {
    render(<AnswerView response={makeResearchResponse()} locale="en" />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Report a problem" }));
    expect(screen.getByRole("dialog", { name: "Report an answer" })).toBeInTheDocument();
    expect(screen.getByText("22222222-2222-4222-8222-222222222222")).toBeInTheDocument();
  });

  it("renders a compact provenance footer for only used citations", () => {
    const response = makeResearchResponse();
    response.citations.push({ ...translationCitation, citation_id: "unused", evidence_id: "unused" });
    render(<AnswerView response={response} locale="ar" />);
    expect(screen.getByRole("region", { name: "المصادر" })).toBeInTheDocument();
    expect(screen.queryByText("unused")).not.toBeInTheDocument();
  });

  it("never labels a source-only topic group as a meaning", () => {
    const response = makeResearchResponse();
    response.blocks = response.blocks.filter((block) => block.block_type === "SOURCE");
    render(<AnswerView response={response} userMessage="ماذا يقول القرآن عن الإحسان؟" locale="ar" />);
    expect(screen.getByTestId("verse-card")).toBeInTheDocument();
    expect(screen.queryByText(/المعنى 1/)).not.toBeInTheDocument();
  });

  it("puts partial coverage directly under the generated reading, before evidence", () => {
    const onFollowUp = vi.fn();
    render(<AnswerView response={makePartialResearchResponse()} locale="ar" onFollowUp={onFollowUp} />);
    const claim = screen.getByText("تشير الآية مباشرة إلى أن اليسر يأتي مع العسر.");
    const verse = screen.getByTestId("verse-card");
    const coverage = screen.getByTestId("partial-coverage-note");
    expect(claim.compareDocumentPosition(verse) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(coverage.compareDocumentPosition(verse) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getByText("تدعم المصادر المتاحة جزءًا من السؤال فقط.")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /فتح نص الآية/ })).toHaveLength(2);
    expect(screen.getByRole("button", { name: "أضف تفسيرًا معتمدًا لـ سورة الشرح ٦" })).toBeInTheDocument();
  });

  it("localizes the partial coverage note and optional Tafsir action in English", () => {
    render(<AnswerView response={makePartialResearchResponse("en")} locale="en" onFollowUp={() => undefined} />);
    expect(screen.getByTestId("partial-coverage-note")).toHaveTextContent("available sources support only part");
    expect(screen.getByRole("button", { name: "Add approved Tafsir for Ash-Sharh 6" })).toBeInTheDocument();
    expect(screen.queryByText("Useful follow-ups")).not.toBeInTheDocument();
  });
});
