import fs from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";


const root = path.resolve(process.cwd(), "..");
const outputDir = path.join(root, "artifacts", "qa", "ui");
fs.mkdirSync(outputDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
const page = await context.newPage();
const requests = [];
page.on("request", (request) => {
  if (request.url().includes("/api/research")) requests.push(request.url());
});

try {
  await page.goto("http://localhost:3001", { waitUntil: "networkidle" });
  await page.getByRole("textbox", { name: "سؤال البحث" }).fill("ماذا يقول القرآن عن الصبر؟");
  await page.getByRole("button", { name: "إرسال سؤال البحث" }).click();
  await page.getByText("لم يتم عرض أي محتوى غير آمن").waitFor({ state: "visible" });
  const firstError = await page.getByText("خدمة البحث الخاصة غير متاحة مؤقتًا.").isVisible();
  await page.screenshot({
    path: path.join(outputDir, "mobile-backend-unavailable-retry.png"),
    fullPage: false,
  });
  await page.getByRole("button", { name: "حاول مرة أخرى" }).click();
  await page.getByText("لم يتم عرض أي محتوى غير آمن").waitFor({ state: "visible" });
  await page.waitForTimeout(200);
  console.log(
    JSON.stringify(
      {
        localizedErrorVisible: firstError,
        retryControlVisible: await page.getByRole("button", { name: "حاول مرة أخرى" }).isVisible(),
        requestCountAfterRetry: requests.length,
      },
      null,
      2,
    ),
  );
} finally {
  await context.close();
  await browser.close();
}
