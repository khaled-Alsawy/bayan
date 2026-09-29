import { defineConfig } from "@playwright/test";

const baseURL = process.env.DEMO_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "../demo",
  testMatch: "record.spec.ts",
  fullyParallel: false,
  workers: 1,
  timeout: 30 * 60 * 1000,
  // A live turn can come back ungrounded (record.spec throws); record the take again.
  retries: 2,
  expect: { timeout: 180_000 },
  outputDir: "test-results-demo",
  reporter: "list",
  use: {
    baseURL,
    browserName: "chromium",
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    video: { mode: "on", size: { width: 1920, height: 1080 } },
    trace: "off",
    screenshot: "off",
  },
});
