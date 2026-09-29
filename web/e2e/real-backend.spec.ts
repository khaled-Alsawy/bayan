import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Response as PlaywrightResponse } from "@playwright/test";

test.skip(!process.env.REAL_BACKEND, "Run only against the live local FastAPI service.");
test.use({ trace: "off", screenshot: "off", video: "off" });

type TurnPayload = {
  response: {
    status: string;
    blocks: Array<
      | { block_type: "CLAIM"; text: string }
      | {
          block_type: "SOURCE";
          exact_text: string;
          citation: { source_kind: string; evidence_id: string; reference: { value: string } };
        }
      | { block_type: "NOTICE"; text: string }
    >;
    citations: Array<{ source_kind: string; evidence_id: string }>;
  };
};

async function finalTurnFromStream(response: PlaywrightResponse): Promise<TurnPayload> {
  const finalFrame = (await response.text())
    .split("\n\n")
    .find((frame) => frame.startsWith("event: final\n"));
  const payload = finalFrame?.match(/^data: (.+)$/m)?.[1];
  if (!payload) throw new Error("Research stream did not contain a final turn");
  return JSON.parse(payload) as TurnPayload;
}

async function submitQuestion(page: Page, question: string): Promise<TurnPayload> {
  await page.goto("/");
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill(question);
  const apiResponse = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && response.url().includes("/api/conversations/")
      && response.url().endsWith("/turns/stream"),
  );
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  const response = await apiResponse;
  expect(response.status()).toBe(200);
  return finalTurnFromStream(response);
}

test("@real broad Quran topic renders a continuous grounded answer", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "One live desktop proof is sufficient.");
  test.setTimeout(360_000);
  const turn = await submitQuestion(page, "ماذا يقول القرآن عن الإحسان؟");

  expect(turn.response.status).toBe("ANSWERABLE");
  expect(turn.response.citations.length).toBeGreaterThan(0);
  expect(turn.response.citations.every((citation) =>
    citation.source_kind === "QURAN" || citation.source_kind === "QURAN_TRANSLATION"
  )).toBe(true);
  const claims = turn.response.blocks.filter(
    (block): block is { block_type: "CLAIM"; text: string } => block.block_type === "CLAIM",
  );
  const references = turn.response.blocks
    .filter((block) => block.block_type === "SOURCE")
    .map((block) => block.citation.reference.value);
  expect(claims.length).toBeGreaterThan(0);
  expect(claims.every((claim) => /[\u0600-\u06ff]/u.test(claim.text))).toBe(true);
  expect(references.some((reference) => reference === "2:83" || reference === "16:90")).toBe(true);
  expect(references).not.toContain("94:6");
  expect(references).not.toContain("55:3");
  await expect(page.getByTestId("assistant-answer")).toBeVisible({ timeout: 330_000 });
  await expect(page.getByTestId("assistant-answer")).toHaveCount(1);
  await expect(page.getByRole("button", {
    name: `المصادر المستخدمة (${turn.response.citations.length})`,
  })).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("@real exact verse keeps the passage focused with its canonical translation", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "One live desktop proof is sufficient.");
  test.setTimeout(180_000);
  const turn = await submitQuestion(page, "اعرض الآية 94:6 مع ترجمتها المعتمدة.");

  expect(turn.response.status).toBe("ANSWERABLE");
  expect(turn.response.citations.map((citation) => citation.source_kind).sort()).toEqual([
    "QURAN",
    "QURAN_TRANSLATION",
  ]);
  await expect(page.getByText("الآية المطلوبة")).toBeVisible({ timeout: 150_000 });
  await expect(page.getByTestId("verse-card")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "إخفاء الترجمة" })).toHaveAttribute(
    "aria-expanded",
    "true",
  );
});

test("@real Tafsir is fresh in-turn and policy-redacted after reload", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "Avoid duplicate restricted-source retrieval.");
  test.setTimeout(420_000);
  const turn = await submitQuestion(page, "اشرح الآية 2:153 باستخدام تفسير ابن كثير.");

  expect(turn.response.status).toBe("ANSWERABLE");
  expect(turn.response.citations.filter((citation) => citation.source_kind === "TAFSIR")).toHaveLength(1);
  const liveTafsir = turn.response.blocks.find(
    (block) => block.block_type === "SOURCE" && block.citation.source_kind === "TAFSIR",
  );
  expect(liveTafsir?.block_type).toBe("SOURCE");
  if (liveTafsir?.block_type === "SOURCE") {
    expect(liveTafsir.exact_text).not.toBe("[SOURCE_TEXT_NOT_STORED_POLICY]");
  }
  await expect(page.getByText("نص التفسير المعتمد في الإجابة")).toBeVisible({ timeout: 390_000 });
  await expect(page.getByTestId("tafsir-card")).toHaveCount(1);

  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByRole("button", { name: "إعادة تحميل نص المصدر" })).toBeVisible();
  await expect(page.getByText("[SOURCE_TEXT_NOT_STORED_POLICY]")).toHaveCount(0);
});

test("@real two-Tafsir comparison uses two side-by-side evidence cards", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "Desktop validates the two-column live layout.");
  test.setTimeout(420_000);
  const turn = await submitQuestion(page, "قارن بين تفسير ابن كثير والجلالين للآية 2:153.");

  expect(turn.response.status).toBe("ANSWERABLE");
  expect(turn.response.citations.filter((citation) => citation.source_kind === "TAFSIR")).toHaveLength(2);
  await expect(page.getByText("نطاق المقارنة")).toBeVisible({ timeout: 390_000 });
  const cards = page.getByTestId("tafsir-card");
  await expect(cards).toHaveCount(2);
  const [first, second] = await Promise.all([cards.nth(0).boundingBox(), cards.nth(1).boundingBox()]);
  expect(first).not.toBeNull();
  expect(second).not.toBeNull();
  expect(Math.abs(first!.y - second!.y)).toBeLessThan(2);
});

test("@real unsupported personal ruling renders a source-free terminal state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "One deterministic live terminal proof is sufficient.");
  test.setTimeout(120_000);
  const turn = await submitQuestion(page, "هل هذا الخيار الشخصي حرام؟");

  expect(turn.response.status).toBe("UNSUPPORTED_SCOPE");
  expect(turn.response.citations).toHaveLength(0);
  await expect(page.getByText("خارج نطاق البحث القرآني", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /المصادر المستخدمة/ })).toHaveCount(0);
});
