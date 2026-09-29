import path from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Response as PlaywrightResponse } from "@playwright/test";

const publicUrl = process.env.PUBLIC_APP_URL;

async function finalTurnFromStream<T>(response: PlaywrightResponse): Promise<T> {
  const finalFrame = (await response.text())
    .split("\n\n")
    .find((frame) => frame.startsWith("event: final\n"));
  const payload = finalFrame?.match(/^data: (.+)$/m)?.[1];
  if (!payload) throw new Error("Research stream did not contain a final turn");
  return JSON.parse(payload) as T;
}

test.skip(!publicUrl, "Run only when the external Cloudflare Tunnel URL is supplied.");

test("@public real localized flow works over the external HTTPS origin", async ({ page }, testInfo) => {
  test.setTimeout(360_000);
  const browserFailures: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") browserFailures.push(`console:${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    browserFailures.push(`request:${request.url()}:${request.failure()?.errorText ?? "failed"}`);
  });

  const mobile = testInfo.project.name === "mobile-chromium";
  await page.setViewportSize(mobile ? { width: 390, height: 844 } : { width: 1440, height: 900 });
  await page.goto(publicUrl!, { waitUntil: "networkidle" });
  await expect(page.locator("html")).toHaveAttribute("lang", "ar");
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  expect(new URL(page.url()).protocol).toBe("https:");

  const question = "ايه الآيات الي بتتكلم عن تحصين الإنسان من الحسد والعين ؟";
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill(question);
  const turnResponse = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && response.url().includes("/api/conversations/")
      && response.url().endsWith("/turns/stream"),
  );
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  const response = await turnResponse;
  expect(response.status()).toBe(200);
  const payload = await finalTurnFromStream<{
    response: {
      status: string;
      citations: Array<{ source_kind: string; reference: { value: string } }>;
      blocks: Array<
        | { block_type: "CLAIM"; text: string }
        | { block_type: "SOURCE"; citation: { reference: { value: string } } }
        | { block_type: "NOTICE" }
      >;
    };
  }>(response);
  expect(["ANSWERABLE", "PARTIAL"]).toContain(payload.response.status);
  expect(payload.response.citations.length).toBeGreaterThan(0);
  const claims = payload.response.blocks.filter(
    (block): block is { block_type: "CLAIM"; text: string } => block.block_type === "CLAIM",
  );
  const references = payload.response.blocks
    .filter((block) => block.block_type === "SOURCE")
    .map((block) => block.citation.reference.value);
  expect(claims.length).toBeGreaterThan(0);
  expect(claims.every((claim) => /[\u0600-\u06ff]/u.test(claim.text))).toBe(true);
  expect(references).toContain("113:1");
  expect(references).toContain("113:5");
  expect(references.every((reference) => /^113:[1-5]$/u.test(reference))).toBe(true);

  await expect(page.getByTestId("assistant-answer")).toBeVisible({ timeout: 330_000 });
  if (payload.response.status === "PARTIAL") {
    await expect(page.getByTestId("partial-coverage-note")).toBeVisible();
    await expect(page.getByRole("status")).toHaveCount(0);
  }
  const sources = page.getByRole("button", { name: /المصادر المستخدمة \(\d+\)/ });
  await expect(sources).toBeVisible();
  await sources.click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  const links = drawer.getByRole("link");
  expect(await links.count()).toBeGreaterThan(0);
  for (const href of await links.evaluateAll((items) => items.map((item) => (item as HTMLAnchorElement).href))) {
    expect(new URL(href).protocol).toBe("https:");
  }
  await page.getByRole("button", { name: "إغلاق المصادر" }).click();
  await expect(drawer).toBeHidden();
  await page.waitForTimeout(500);
  await page.getByTestId("conversation-scroll").evaluate((element) => { element.scrollTop = 0; });

  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "..",
      "artifacts",
      "qa",
      mobile ? "public-mobile-390x844.png" : "public-desktop-1440x900.png",
    ),
    fullPage: false,
  });
  if (mobile) {
    await page.getByTestId("conversation-scroll").evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    await page.screenshot({
      path: path.resolve(
        process.cwd(),
        "..",
        "artifacts",
        "qa",
        "public-mobile-conclusion-390x844.png",
      ),
      fullPage: false,
    });
    await page.setViewportSize({ width: 320, height: 640 });
    await expect(page.getByRole("textbox", { name: "سؤال البحث" })).toBeVisible();
    await page.screenshot({
      path: path.resolve(process.cwd(), "..", "artifacts", "qa", "public-partial-narrow-320x640.png"),
      fullPage: false,
    });
  }

  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByRole("heading", { name: question })).toBeVisible();
  await expect(page.getByTestId("assistant-answer")).toBeVisible();
  if (mobile) {
    await page.getByRole("button", { name: "فتح سجل المحادثات" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
  } else {
    await expect(page.getByRole("complementary", { name: "المحادثات" })).toBeVisible();
  }

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  expect(browserFailures.filter((item) => /mixed content|cors|blocked/i.test(item))).toEqual([]);
});

test("@public real ambiguity choice stays in one conversation and survives reload", async ({ page }, testInfo) => {
  test.setTimeout(360_000);
  const mobile = testInfo.project.name === "mobile-chromium";
  await page.setViewportSize(mobile ? { width: 390, height: 844 } : { width: 1440, height: 900 });
  await page.goto(publicUrl!, { waitUntil: "networkidle" });
  if (mobile) {
    await page.getByRole("button", { name: "فتح سجل المحادثات" }).click();
  }
  await page.getByRole("button", { name: "محادثة جديدة" }).click();

  const question = "ما هي آيات العين؟";
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill(question);
  const clarificationResponse = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && response.url().includes("/api/conversations/")
      && response.url().endsWith("/turns/stream"),
  );
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  const clarificationPayload = await clarificationResponse.then((response) => finalTurnFromStream<{
    response: {
      conversation_id: string;
      status: string;
      citations: unknown[];
      clarification: { candidates: Array<{ label: string }> };
    };
  }>(response));
  expect(clarificationPayload.response.status).toBe("CLARIFICATION_REQUIRED");
  expect(clarificationPayload.response.citations).toEqual([]);
  expect(clarificationPayload.response.clarification.candidates).toHaveLength(3);
  expect(
    clarificationPayload.response.clarification.candidates.some(({ label }) => /ماء|ينابيع/u.test(label)),
  ).toBe(true);
  const selected = clarificationPayload.response.clarification.candidates.find(({ label }) => /حسد/u.test(label));
  expect(selected).toBeTruthy();
  await expect(page.getByRole("button", { name: selected!.label })).toBeVisible();
  await page.screenshot({
    path: path.resolve(
      process.cwd(), "..", "artifacts", "qa",
      mobile ? "live-clarification-before-mobile-390x844.png" : "live-clarification-before-desktop-1440x900.png",
    ),
    fullPage: false,
  });

  const resolvedResponse = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && response.url().endsWith(`/api/conversations/${clarificationPayload.response.conversation_id}/turns/stream`),
  );
  await page.getByRole("button", { name: selected!.label }).click();
  const resolvedPayload = await resolvedResponse.then((response) => finalTurnFromStream<{
    response: { conversation_id: string; status: string };
  }>(response));
  expect(resolvedPayload.response.conversation_id).toBe(clarificationPayload.response.conversation_id);
  expect(resolvedPayload.response.status).not.toBe("CLARIFICATION_REQUIRED");
  await page.screenshot({
    path: path.resolve(
      process.cwd(), "..", "artifacts", "qa",
      mobile ? "live-clarification-after-mobile-390x844.png" : "live-clarification-after-desktop-1440x900.png",
    ),
    fullPage: false,
  });

  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByRole("heading", { name: question, exact: true })).toBeVisible();
  await expect(page.getByText(selected!.label, { exact: true }).first()).toBeVisible();
});
