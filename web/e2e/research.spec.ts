import { readFile } from "node:fs/promises";
import path from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import type { AnswerStatus } from "@/lib/contracts";
import { ENGLISH_EXACT, makeLongResearchResponse, makePartialResearchResponse, makeResearchResponse, makeTafsirResponse } from "../tests/fixtures";

async function mockResearch(
  page: import("@playwright/test").Page,
  status: AnswerStatus = "ANSWERABLE",
) {
  await mockConversations(page, makeResearchResponse(status));
}

async function mockConversations(
  page: import("@playwright/test").Page,
  response: ReturnType<typeof makeResearchResponse>,
  traceEvents: Array<Record<string, unknown>> = [],
) {
  const turns: Array<Record<string, unknown>> = [];
  const summary = {
    conversation_id: response.conversation_id,
    title: "ماذا يقول القرآن عن الصبر وقت الشدة؟",
    locale: "ar",
    archived: false,
    created_at: "2026-09-16T00:00:00Z",
    updated_at: "2026-09-16T00:00:00Z",
    turn_count: 0,
  };
  await page.route("**/api/conversations**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    if (method === "GET" && url.pathname === "/api/conversations") {
      await route.fulfill({ json: { items: turns.length ? [{ ...summary, turn_count: turns.length }] : [], next_cursor: null } });
      return;
    }
    if (method === "POST" && url.pathname === "/api/conversations") {
      await route.fulfill({ status: 201, json: summary });
      return;
    }
    if (method === "GET" && url.pathname === `/api/conversations/${response.conversation_id}`) {
      await route.fulfill({ json: { ...summary, turn_count: turns.length, turns } });
      return;
    }
    if (method === "POST" && url.pathname.endsWith("/turns/stream")) {
      const body = request.postDataJSON() as { message: string; locale: "ar" | "en" };
      const turn = { turn_id: response.turn_id, user_message: body.message, locale: body.locale, response, created_at: "2026-09-16T00:01:00Z", trace_events: traceEvents };
      turns.push(turn);
      await route.fulfill({ status: 201, json: turn });
      return;
    }
    await route.fulfill({ status: 404, json: {} });
  });
}

async function mockStreamingTurn(
  page: import("@playwright/test").Page,
  response: ReturnType<typeof makeResearchResponse>,
  events: Array<Record<string, unknown>>,
  delayMs = 80,
  holdFinal = false,
) : Promise<{ releaseFinal: () => Promise<void> }> {
  const releaseEvent = `release-stream-${response.turn_id}`;
  const turn = {
    turn_id: response.turn_id,
    user_message: "ماذا يقول القرآن عن الصبر وقت الشدة؟",
    locale: "ar",
    response,
    created_at: "2026-09-16T00:01:00Z",
    trace_events: events,
  };
  await page.addInitScript(({ finalTurn, streamEvents, interval, shouldHoldFinal, releaseSignal }) => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/turns/stream") && init?.method === "POST") {
        const encoder = new TextEncoder();
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            streamEvents.forEach((event, index) => {
              window.setTimeout(() => {
                controller.enqueue(encoder.encode(`event: trace\ndata: ${JSON.stringify(event)}\n\n`));
              }, interval * (index + 1));
            });
            const finish = () => {
              controller.enqueue(encoder.encode(`event: final\ndata: ${JSON.stringify(finalTurn)}\n\n`));
              controller.close();
            };
            if (shouldHoldFinal) window.addEventListener(releaseSignal, finish, { once: true });
            else window.setTimeout(finish, interval * (streamEvents.length + 1));
          },
        });
        return Promise.resolve(new Response(body, {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        }));
      }
      return originalFetch(input, init);
    }) as typeof window.fetch;
  }, { finalTurn: turn, streamEvents: events, interval: delayMs, shouldHoldFinal: holdFinal, releaseSignal: releaseEvent });
  return {
    releaseFinal: async () => { await page.evaluate((signal) => { window.dispatchEvent(new Event(signal)); }, releaseEvent); },
  };
}

async function submitArabicQuestion(page: import("@playwright/test").Page, question = "ماذا يقول القرآن عن الصبر وقت الشدة؟") {
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill(question);
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
}

async function expectFocusRing(page: import("@playwright/test").Page) {
  const focus = page.locator(":focus");
  await expect(focus).toHaveCount(1);
  const outline = await focus.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return { color: style.outlineColor, style: style.outlineStyle, width: style.outlineWidth };
  });
  expect(outline.color).not.toBe("rgba(0, 0, 0, 0)");
  expect(outline.style).not.toBe("none");
  expect(Number.parseFloat(outline.width)).toBeGreaterThan(0);
}

async function tabTo(page: import("@playwright/test").Page, target: import("@playwright/test").Locator) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await page.keyboard.press("Tab");
    if (await target.evaluate((element) => document.activeElement === element)) return;
  }
  throw new Error("Keyboard focus did not reach the expected control");
}

function traceEvents(count: number): Array<Record<string, unknown>> {
  return Array.from({ length: count }, (_, index) => ({
    seq: index + 1,
    elapsed_ms: (index + 1) * 100,
    node: index % 2 === 0 ? "retrieval" : "evidence",
    phase: index % 3 === 0 ? "started" : "completed",
    summary: `Step ${index + 1}`,
    detail: { candidate_count: index + 1 },
    provider: null,
  }));
}

function intersects(
  first: { x: number; y: number; width: number; height: number },
  second: { x: number; y: number; width: number; height: number },
) {
  return !(
    first.x + first.width <= second.x ||
    second.x + second.width <= first.x ||
    first.y + first.height <= second.y ||
    second.y + second.height <= first.y
  );
}

async function expectMobileControlsDoNotOverlap(page: import("@playwright/test").Page) {
  const evidence = page.getByTestId("verse-card").first();
  const composer = page.getByRole("textbox", { name: /(?:سؤال البحث|Research question)/ });
  const submit = page.getByRole("button", { name: /(?:إرسال سؤال البحث|Submit research question)/ });
  const [evidenceBox, composerBox, submitBox] = await Promise.all([
    evidence.boundingBox(),
    composer.boundingBox(),
    submit.boundingBox(),
  ]);
  expect(evidenceBox).not.toBeNull();
  expect(composerBox).not.toBeNull();
  expect(submitBox).not.toBeNull();
  expect(intersects(evidenceBox!, composerBox!)).toBe(false);
  expect(intersects(evidenceBox!, submitBox!)).toBe(false);
}

async function expectFullyAboveComposer(
  page: import("@playwright/test").Page,
  target: import("@playwright/test").Locator,
) {
  const composerDock = page.getByTestId("composer-dock");
  const [targetBox, composerBox] = await Promise.all([
    target.boundingBox(),
    composerDock.boundingBox(),
  ]);
  expect(targetBox).not.toBeNull();
  expect(composerBox).not.toBeNull();
  expect(intersects(targetBox!, composerBox!)).toBe(false);
  expect(targetBox!.y + targetBox!.height).toBeLessThanOrEqual(composerBox!.y);
}

test("tail-follows streamed events, respects owner scroll, and hands off at answer top", async ({ page }) => {
  const response = makeLongResearchResponse("ar");
  const events = traceEvents(20);
  await mockConversations(page, response);
  // The final frame stays under test control: the assertions below must never race the
  // mock's completion timer and unmount the live trace before owner-scroll is observed.
  const stream = await mockStreamingTurn(page, response, events, 80, true);
  await page.setViewportSize({ width: 390, height: 700 });
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("ماذا يقول القرآن عن الصبر وقت الشدة؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();

  const scrollArea = page.getByTestId("conversation-scroll");
  await expect.poll(() => page.locator(".loading-phase").count()).toBeGreaterThan(5);
  await expect.poll(() => scrollArea.evaluate(
    (element) => Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight - 2,
  )).toBe(true);

  const bottomPosition = await scrollArea.evaluate((element) => element.scrollTop);
  await scrollArea.hover();
  await page.mouse.wheel(0, -600);
  await expect.poll(() => scrollArea.evaluate((element) => element.scrollTop)).toBeLessThan(bottomPosition);
  await expect(page.getByRole("button", { name: "الانتقال إلى الأحدث" })).toBeVisible();
  const heldPosition = await scrollArea.evaluate((element) => element.scrollTop);
  await expect.poll(() => page.locator(".loading-phase").count()).toBeGreaterThan(10);
  expect(await scrollArea.evaluate((element) => element.scrollTop)).toBe(heldPosition);

  await page.mouse.wheel(0, 10_000);
  await expect.poll(() => scrollArea.evaluate(
    (element) => Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight - 2,
  )).toBe(true);
  await stream.releaseFinal();
  await expect(page.getByTestId("assistant-answer")).toBeVisible();

  // The answer contains measured evidence/tafsir content. Check after its layout has been
  // quiet, then again after another paint: a one-frame sample would hide the old race.
  await page.waitForTimeout(250);
  const answerOffset = async () => page.getByTestId("assistant-answer").evaluate((answer) => {
    const container = document.querySelector<HTMLElement>("[data-testid='conversation-scroll']");
    if (!container) return Number.POSITIVE_INFINITY;
    return Math.abs(answer.getBoundingClientRect().top - container.getBoundingClientRect().top);
  });
  expect(await answerOffset()).toBeLessThan(3);
  await page.waitForTimeout(100);
  expect(await answerOffset()).toBeLessThan(3);

});

test("keeps the persisted trace collapsed and does not jump while expanding it", async ({ page }) => {
  const response = makeLongResearchResponse("ar");
  await mockConversations(page, response, traceEvents(3));
  await page.setViewportSize({ width: 390, height: 700 });
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("ماذا يقول القرآن عن الصبر وقت الشدة؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();

  const trace = page.getByTestId("execution-trace");
  await expect(trace).toBeVisible();
  await expect(trace).not.toHaveAttribute("open", "");
  const scrollArea = page.getByTestId("conversation-scroll");
  await trace.evaluate((element) => {
    const container = document.querySelector<HTMLElement>("[data-testid='conversation-scroll']");
    if (container) container.scrollTop = element.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop - 40;
  });
  const before = await scrollArea.evaluate((element) => element.scrollTop);
  await trace.locator(":scope > summary").click();
  expect(await scrollArea.evaluate((element) => element.scrollTop)).toBe(before);
});

test("renders a grounded bilingual answer accessibly", async ({ page }, testInfo) => {
  await mockResearch(page);
  await page.setViewportSize(testInfo.project.name === "mobile-chromium"
    ? { width: 390, height: 844 }
    : { width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("lang", "ar");
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  const composer = page.getByRole("textbox", { name: "سؤال البحث" });
  await page.getByRole("button", { name: "ماذا يقول القرآن عن الصبر عند الشدّة؟" }).click();
  await expect(composer).toHaveValue("ماذا يقول القرآن عن الصبر عند الشدة؟");
  await expect(page.getByRole("button", { name: "إرسال سؤال البحث" })).toBeEnabled();
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByTestId("assistant-answer").last()).toBeVisible();
  await expect(page.getByText("إِنَّ مَعَ الْعُسْرِ يُسْرًا")).toHaveAttribute("dir", "rtl");
  await page.getByRole("button", { name: "عرض الترجمة" }).click();
  await expect(page.getByText("Indeed, with hardship comes ease.")).toHaveAttribute("dir", "ltr");

  if (testInfo.project.name === "mobile-chromium") {
    await expectMobileControlsDoNotOverlap(page);
    await expect(page.getByTestId("provenance-footer")).toBeVisible();
  } else {
    await expect(page.getByTestId("provenance-footer")).toBeVisible();
  }

  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "..",
      "artifacts", "qa",
      testInfo.project.name === "mobile-chromium"
        ? "conversational-arabic-mobile-390x844.png"
        : "conversational-arabic-desktop-1440x900.png",
    ),
    fullPage: false,
  });

  const arabicText = await page.getByText("إِنَّ مَعَ الْعُسْرِ يُسْرًا").textContent();
  const translationText = await page.getByText("Indeed, with hardship comes ease.").textContent();
  // Locale selection lives outside the deliberately minimal artboard header.
  await page.evaluate(() => {
    window.localStorage.setItem("quran-ui-locale", "en");
    document.cookie = "quran-ui-locale=en; Path=/; SameSite=Lax";
  });
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
  // The clarification remains visible as part of the same conversation; assert the newly
  // answered turn rather than assuming there is only one assistant response in history.
  await expect(page.getByTestId("assistant-answer").last()).toBeVisible();
  await expect(page.getByText("إِنَّ مَعَ الْعُسْرِ يُسْرًا")).toHaveText(arabicText ?? "");
  await page.getByRole("button", { name: "Show translation" }).click();
  await expect(page.getByText("Indeed, with hardship comes ease.")).toHaveText(translationText ?? "");

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
  await expect(page.getByTestId("assistant-answer")).toBeVisible();
  await expect.poll(() => page.evaluate(() => Object.keys(window.localStorage).sort())).toEqual([
    "quran-active-conversation-id",
    "quran-ui-locale",
  ]);

  // The approved artboards deliberately replace the application sidebar with a reading surface.
  await expect(page.getByRole("button", { name: "New thread" })).toBeVisible();
  await expect(page.getByLabel("Conversations")).toHaveCount(0);
});

test("renders partial evidence before its quiet coverage limit on desktop and mobile", async ({ page }, testInfo) => {
  await mockConversations(page, makePartialResearchResponse());
  await page.setViewportSize(testInfo.project.name === "mobile-chromium"
    ? { width: 390, height: 844 }
    : { width: 1440, height: 900 });
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("ما الذي تقوله الآية عن اليسر مع العسر وطرق تجاوز الشدة؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();

  const answer = page.getByTestId("assistant-answer");
  const claim = page.getByText("تشير الآية مباشرة إلى أن اليسر يأتي مع العسر.");
  const verse = page.getByTestId("verse-card");
  const coverage = page.getByTestId("partial-coverage-note");
  await expect(answer).toHaveAttribute("data-answer-status", "PARTIAL");
  await expect(page.getByRole("status")).toHaveCount(0);
  await expect(coverage).toContainText("تدعم المصادر المتاحة جزءًا من السؤال فقط.");
  await expect(page.getByRole("link", { name: /فتح نص الآية/ })).toHaveCount(2);
  await expect(page.getByRole("button", { name: "أضف تفسيرًا معتمدًا للآية 94:6" })).toBeVisible();
  const [claimBox, verseBox, coverageBox] = await Promise.all([
    claim.boundingBox(),
    verse.boundingBox(),
    coverage.boundingBox(),
  ]);
  expect(claimBox).not.toBeNull();
  expect(verseBox).not.toBeNull();
  expect(coverageBox).not.toBeNull();
  expect(coverageBox!.y).toBeGreaterThanOrEqual(claimBox!.y + claimBox!.height);
  expect(verseBox!.y).toBeGreaterThanOrEqual(coverageBox!.y + coverageBox!.height);

  await page.screenshot({
    path: path.resolve(process.cwd(), "..", "artifacts", "qa", testInfo.project.name === "mobile-chromium" ? "partial-mobile-390x844.png" : "partial-desktop-1440x900.png"),
    fullPage: false,
  });

  if (testInfo.project.name === "mobile-chromium") {
    await expectMobileControlsDoNotOverlap(page);
    await page.setViewportSize({ width: 320, height: 640 });
    await page.getByTestId("conversation-scroll").evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await expect(coverage).toBeVisible();
    await expect(page.getByRole("textbox", { name: "سؤال البحث" })).toBeVisible();
    await page.screenshot({ path: path.resolve(process.cwd(), "..", "artifacts", "qa", "partial-narrow-320x640.png"), fullPage: false });
  }
});

test("clarification candidates continue the same conversation and survive reload", async ({ page }, testInfo) => {
  const clarification = makeResearchResponse("CLARIFICATION_REQUIRED");
  const answered = makeResearchResponse();
  answered.turn_id = "55555555-5555-4555-8555-555555555555";
  answered.correlation_id = "66666666-6666-4666-8666-666666666666";
  const turns: Array<Record<string, unknown>> = [];
  const submitted: string[] = [];
  const summary = {
    conversation_id: clarification.conversation_id,
    title: "آيات العين",
    locale: "ar",
    archived: false,
    created_at: "2026-09-16T00:00:00Z",
    updated_at: "2026-09-16T00:02:00Z",
    turn_count: 0,
  };
  await page.route("**/api/conversations**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === "/api/conversations") {
      await route.fulfill({ json: { items: turns.length ? [{ ...summary, turn_count: turns.length }] : [], next_cursor: null } });
      return;
    }
    if (request.method() === "POST" && url.pathname === "/api/conversations") {
      await route.fulfill({ status: 201, json: summary });
      return;
    }
    if (request.method() === "GET" && url.pathname === `/api/conversations/${clarification.conversation_id}`) {
      await route.fulfill({ json: { ...summary, turn_count: turns.length, turns } });
      return;
    }
    if (request.method() === "POST" && url.pathname.endsWith("/turns/stream")) {
      const body = request.postDataJSON() as { message: string; locale: "ar" | "en" };
      submitted.push(body.message);
      const response = turns.length === 0 ? clarification : answered;
      const turn = { turn_id: response.turn_id, user_message: body.message, locale: body.locale, response, created_at: `2026-09-16T00:0${turns.length + 1}:00Z` };
      turns.push(turn);
      await route.fulfill({ status: 201, json: turn });
      return;
    }
    await route.fulfill({ status: 404, json: {} });
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("ما هي آيات العين؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  const choice = page.getByRole("button", { name: "الحسد والإصابة بالعين" });
  await expect(choice).toBeVisible();
  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "..",
      "artifacts",
      "qa",
      testInfo.project.name === "mobile-chromium"
        ? "clarification-before-mobile-390x844.png"
        : "clarification-before-desktop-1440x900.png",
    ),
    fullPage: false,
  });
  await choice.focus();
  await page.keyboard.press("Enter");
  // A clarification is retained in history; the newly answered turn is the last response.
  await expect(page.getByTestId("assistant-answer").last()).toBeVisible();
  expect(submitted).toEqual(["ما هي آيات العين؟", "الحسد والإصابة بالعين"]);

  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "..",
      "artifacts",
      "qa",
      testInfo.project.name === "mobile-chromium"
        ? "clarification-mobile-390x844.png"
        : "clarification-desktop-1440x900.png",
    ),
    fullPage: false,
  });
  await page.reload();
  await expect(page.getByText("هل تقصد بالعين: الحسد والإصابة بالعين، أم العين الباصرة، أم عيون الماء؟")).toBeVisible();
  await expect(page.getByTestId("assistant-answer").last()).toBeVisible();
  await expect(page.getByLabel("المحادثات")).toHaveCount(0);
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("renders a safe unsupported-scope terminal state", async ({ page }) => {
  await mockResearch(page, "UNSUPPORTED_SCOPE");
  await page.goto("/");
  const composer = page.getByRole("textbox", { name: "سؤال البحث" });
  await composer.fill("هل هذا الخيار الشخصي حرام؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByText("خارج نطاق البحث القرآني")).toBeVisible();
  await expect(page.getByRole("button", { name: /المصادر المستخدمة/ })).toHaveCount(0);
});

test("renders an exact verse as a focused passage with translation controls", async ({ page }) => {
  await mockConversations(page, makeResearchResponse());
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("اعرض الآية 94:6 مع ترجمتها المعتمدة.");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByRole("heading", { name: "الشواهد" })).toBeVisible();
  await expect(page.getByRole("button", { name: "إخفاء الترجمة" })).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("button", { name: "نسخ الآية" })).toBeVisible();
});

test("renders one Tafsir without turning the whole response into cards", async ({ page }) => {
  await mockConversations(page, makeTafsirResponse(false));
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("فسّر الآية 94:6 من تفسير الجلالين.");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByText("نص التفسير المسترجع")).toBeVisible();
  await expect(page.getByTestId("tafsir-card")).toHaveCount(1);
});

test("renders exactly two Tafsirs side by side on desktop and stacked on mobile", async ({ page }, testInfo) => {
  await mockConversations(page, makeTafsirResponse(true));
  await page.setViewportSize(testInfo.project.name === "mobile-chromium" ? { width: 390, height: 844 } : { width: 1440, height: 900 });
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("قارن بين تفسير ابن كثير والجلالين للآية 94:6.");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByText("نطاق المقارنة")).toBeVisible();
  const cards = page.getByTestId("tafsir-card");
  await expect(cards).toHaveCount(2);
  const [first, second] = await Promise.all([cards.nth(0).boundingBox(), cards.nth(1).boundingBox()]);
  expect(first).not.toBeNull();
  expect(second).not.toBeNull();
  expect(second!.y).toBeGreaterThanOrEqual(first!.y + first!.height);
});

for (const state of ["INSUFFICIENT_EVIDENCE", "FAILED_SAFE", "TEMPORARILY_UNAVAILABLE"] as const) {
  test(`renders the ${state} state without fabricated evidence`, async ({ page }) => {
    await mockResearch(page, state);
    await page.goto("/");
    await page.getByRole("textbox", { name: "سؤال البحث" }).fill("سؤال يحتاج إلى تحقق آمن");
    await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
    await expect(page.getByRole("status")).toBeVisible();
    await expect(page.getByTestId("verse-card")).toHaveCount(0);
    await expect(page.getByRole("button", { name: /المصادر المستخدمة/ })).toHaveCount(0);
  });
}

test("shows the policy-safe reload state for restricted Tafsir history", async ({ page }) => {
  const response = makeTafsirResponse(false);
  const tafsir = response.blocks.find((block) => block.block_type === "SOURCE" && block.citation.source_kind === "TAFSIR");
  if (!tafsir || tafsir.block_type !== "SOURCE") throw new Error("Tafsir fixture missing");
  tafsir.exact_text = "[SOURCE_TEXT_NOT_STORED_POLICY]";
  await mockConversations(page, response);
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("فسّر الآية 94:6 من تفسير الجلالين.");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await expect(page.getByRole("button", { name: "إعادة تحميل نص المصدر" })).toBeVisible();
  await expect(page.getByText("[SOURCE_TEXT_NOT_STORED_POLICY]")).toHaveCount(0);
});

for (const locale of ["ar", "en"] as const) {
  test(`keeps the full long ${locale} response clear of the fixed composer`, async ({ page }) => {
    await mockConversations(page, makeLongResearchResponse(locale));
    if (locale === "en") {
      await page.context().addCookies([{ name: "quran-ui-locale", value: "en", url: "http://localhost:3000" }]);
    }
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("dir", locale === "ar" ? "rtl" : "ltr");
    const textbox = page.getByRole("textbox", {
      name: locale === "ar" ? "سؤال البحث" : "Research question",
    });
    await textbox.fill(locale === "ar" ? "الصبر وقت الشدة" : "patience during hardship");
    await page.getByRole("button", {
      name: locale === "ar" ? "إرسال سؤال البحث" : "Submit research question",
    }).click();

    const claims = page.locator(`.narrative-copy > p[dir="${locale === "ar" ? "rtl" : "ltr"}"]`);
    await expect(claims).toHaveCount(10);
    for (let index = 0; index < 10; index += 1) {
      await expect(claims.nth(index)).toHaveAttribute("dir", locale === "ar" ? "rtl" : "ltr");
    }
    await page.getByRole("button", { name: locale === "ar" ? "عرض الترجمة" : "Show translation" }).click();
    await expect(page.getByText(`${ENGLISH_EXACT} [10]`)).toHaveAttribute("dir", "ltr");

    const scrollArea = page.getByTestId("conversation-scroll");
    await scrollArea.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await expect.poll(() => scrollArea.evaluate(
      (element) => Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight - 1,
    )).toBe(true);
    await expectFullyAboveComposer(page, page.getByTestId("assistant-answer").last());
    await expect(textbox).toBeVisible();
    await expect(textbox).toBeEditable();
  });
}

test("keeps first-use suggestions clear of the composer at required breakpoints", async ({ page }) => {
  await mockResearch(page);
  for (const viewport of [{ width: 1280, height: 720 }, { width: 320, height: 640 }]) {
    await page.setViewportSize(viewport);
    await page.goto("/");
    const suggestions = page.getByRole("button", { name: "ماذا يقول القرآن عن الصبر عند الشدّة؟" });
    await expect(suggestions).toBeAttached();
    await page.getByTestId("conversation-scroll").evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    const [suggestionBox, dockBox] = await Promise.all([
      suggestions.boundingBox(),
      page.getByTestId("composer-dock").boundingBox(),
    ]);
    expect(suggestionBox).not.toBeNull();
    expect(dockBox).not.toBeNull();
    expect(intersects(suggestionBox!, dockBox!)).toBe(false);
    await page.screenshot({
      path: path.resolve(process.cwd(), "..", "artifacts", "qa", `conversational-empty-${viewport.width}x${viewport.height}.png`),
      fullPage: false,
    });
  }
});

for (const locale of ["ar", "en"] as const) {
  for (const theme of ["light", "dark"] as const) {
    test(`has no axe violations in ${locale}/${theme} with the trace collapsed and expanded`, async ({ page }) => {
      const response = makeResearchResponse();
      await mockConversations(page, response, traceEvents(3));
      if (locale === "en") {
        await page.context().addCookies([{ name: "quran-ui-locale", value: "en", url: "http://localhost:3000" }]);
      }
      await page.goto("/");
      if (theme === "dark") {
        await page.getByRole("button", { name: locale === "ar" ? "تبديل الوضع الليلي" : "Switch to dark mode" }).click();
      }
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      const textbox = page.getByRole("textbox", { name: locale === "ar" ? "سؤال البحث" : "Research question" });
      await textbox.fill(locale === "ar" ? "ماذا يقول القرآن عن الصبر وقت الشدة؟" : "What does the Quran say about patience in hardship?");
      await page.getByRole("button", { name: locale === "ar" ? "إرسال سؤال البحث" : "Submit research question" }).click();

      const trace = page.getByTestId("execution-trace");
      await expect(trace).toBeVisible();
      await expect(trace).not.toHaveAttribute("open", "");
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
      await trace.locator(":scope > summary").click();
      await expect(trace).toHaveAttribute("open", "");
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    });
  }
}

test("keeps the Arabic RTL reading surface within 375px, 1440px, and 320px bounds", async ({ page }) => {
  await mockConversations(page, makeLongResearchResponse("ar"));
  await page.goto("/");
  await submitArabicQuestion(page);
  await expect(page.getByTestId("assistant-answer")).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");

  for (const viewport of [{ width: 375, height: 812 }, { width: 1440, height: 900 }, { width: 320, height: 640 }]) {
    await page.setViewportSize(viewport);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
    await expect.poll(() => page.getByTestId("conversation-scroll").evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    )).toBe(true);
  }
});

test("visually separates generated narrative from source material by ground and type", async ({ page }) => {
  await mockResearch(page);
  await page.goto("/");
  await submitArabicQuestion(page);
  await expect(page.getByTestId("assistant-answer")).toBeVisible();

  const styles = await page.evaluate(() => {
    const narrative = document.querySelector<HTMLElement>(".narrative");
    const narrativeText = document.querySelector<HTMLElement>(".narrative-copy > p");
    const source = document.querySelector<HTMLElement>("[data-testid='verse-card']");
    const sourceText = document.querySelector<HTMLElement>(".quran-text");
    if (!narrative || !narrativeText || !source || !sourceText) throw new Error("Expected answer blocks are missing");
    return {
      narrativeGround: window.getComputedStyle(narrative).backgroundColor,
      narrativeFont: window.getComputedStyle(narrativeText).fontFamily,
      sourceGround: window.getComputedStyle(source).backgroundColor,
      sourceFont: window.getComputedStyle(sourceText).fontFamily,
    };
  });
  expect(styles.narrativeGround).not.toBe(styles.sourceGround);
  expect(styles.narrativeFont).not.toBe(styles.sourceFont);
});

test("renders the real 26,975-character Ibn Kathir Tafsir without horizontal layout breakage", async ({ page }) => {
  const sourceData = JSON.parse(await readFile(
    path.resolve(process.cwd(), "..", "data", "raw", "qul", "extracted", "ar-tafsir-ibn-kathir.json"),
    "utf8",
  )) as Record<string, { text: string }>;
  const exactText = sourceData["113:1"]?.text;
  if (!exactText || exactText.length !== 26_975) throw new Error("Expected 26,975-character Ibn Kathir source fixture");
  const response = makeTafsirResponse(false);
  const tafsir = response.blocks.find((block) => block.block_type === "SOURCE" && block.citation.source_kind === "TAFSIR");
  if (!tafsir || tafsir.block_type !== "SOURCE") throw new Error("Tafsir fixture missing");
  tafsir.exact_text = exactText;
  await mockConversations(page, response);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await submitArabicQuestion(page, "فسّر المعوذتين من تفسير ابن كثير.");

  const card = page.getByTestId("tafsir-card");
  await expect(card).toBeVisible();
  await page.getByRole("button", { name: /اقرأ كاملًا/ }).click();
  await expect(card).toHaveAttribute("data-tafsir-expanded", "true");
  const layout = await page.locator(".tafsir-reading").evaluate((element) => ({
    characters: element.textContent?.length ?? 0,
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(layout.characters).toBe(26_975);
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.clientWidth);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
});

test("supports keyboard-only submission, Tafsir and trace expansion, and report dialog controls", async ({ page }) => {
  await mockConversations(page, makeTafsirResponse(false), traceEvents(3));
  await page.goto("/");
  const textbox = page.getByRole("textbox", { name: "سؤال البحث" });
  await textbox.fill("فسّر الآية 94:6 من تفسير الجلالين.");
  await expectFocusRing(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("assistant-answer")).toBeVisible();

  const tafsirToggle = page.getByRole("button", { name: /اقرأ كاملًا/ });
  await tabTo(page, tafsirToggle);
  await expectFocusRing(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("tafsir-card")).toHaveAttribute("data-tafsir-expanded", "true");

  const traceToggle = page.getByTestId("execution-trace").locator(":scope > summary");
  await tabTo(page, traceToggle);
  await expectFocusRing(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("execution-trace")).toHaveAttribute("open", "");

  const report = page.getByRole("button", { name: "أبلغ عن جواب غير دقيق" });
  await tabTo(page, report);
  await expectFocusRing(page);
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "الإبلاغ عن جواب" });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole("button", { name: "إغلاق" });
  await tabTo(page, close);
  await expectFocusRing(page);
  await page.keyboard.press("Enter");
  await expect(dialog).toHaveCount(0);
});

test("uses instant tail scrolling under reduced motion and still hands off to the answer top", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const behaviors: string[] = [];
    const prototype = HTMLElement.prototype as unknown as { scrollTo: (...arguments_: unknown[]) => void };
    const originalScrollTo = prototype.scrollTo;
    Object.defineProperty(window, "__conversationScrollBehaviors", { value: behaviors });
    prototype.scrollTo = function patchedScrollTo(this: HTMLElement, ...arguments_: unknown[]) {
      const options = arguments_[0];
      if (this.dataset.testid === "conversation-scroll" && typeof options === "object" && options !== null && "behavior" in options) {
        behaviors.push((options as { behavior?: string }).behavior ?? "auto");
      }
      return originalScrollTo.apply(this, arguments_);
    };
  });
  const response = makeLongResearchResponse("ar");
  await mockConversations(page, response);
  const stream = await mockStreamingTurn(page, response, traceEvents(12), 80, true);
  await page.setViewportSize({ width: 390, height: 700 });
  await page.goto("/");
  await submitArabicQuestion(page);
  await expect.poll(() => page.locator(".loading-phase").count()).toBeGreaterThan(5);
  const behaviors = await page.evaluate(() => (
    window as Window & { __conversationScrollBehaviors?: string[] }
  ).__conversationScrollBehaviors ?? []);
  expect(behaviors.length).toBeGreaterThan(0);
  expect(behaviors.every((behavior) => behavior === "auto")).toBe(true);

  const scrollArea = page.getByTestId("conversation-scroll");
  const bottomPosition = await scrollArea.evaluate((element) => element.scrollTop);
  await scrollArea.hover();
  await page.mouse.wheel(0, -600);
  await expect.poll(() => scrollArea.evaluate((element) => element.scrollTop)).toBeLessThan(bottomPosition);
  await expect(page.getByRole("button", { name: "الانتقال إلى الأحدث" })).toBeVisible();
  const heldPosition = await scrollArea.evaluate((element) => element.scrollTop);
  await expect.poll(() => page.locator(".loading-phase").count()).toBeGreaterThan(6);
  expect(await scrollArea.evaluate((element) => element.scrollTop)).toBe(heldPosition);
  await page.mouse.wheel(0, 10_000);
  await expect.poll(() => scrollArea.evaluate(
    (element) => Math.ceil(element.scrollTop + element.clientHeight) >= element.scrollHeight - 2,
  )).toBe(true);

  await stream.releaseFinal();
  const answer = page.getByTestId("assistant-answer");
  await expect(answer).toBeVisible();
  const answerTop = await answer.evaluate((element) => {
    const container = document.querySelector<HTMLElement>("[data-testid='conversation-scroll']");
    return container ? Math.abs(element.getBoundingClientRect().top - container.getBoundingClientRect().top) : Number.POSITIVE_INFINITY;
  });
  expect(answerTop).toBeLessThan(3);
});
