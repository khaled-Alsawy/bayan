import { readFile, mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { test, expect, type BrowserContext, type Page } from "@playwright/test";

type Overlay = { kind: "card" | "caption" | "pipeline"; title?: string; lines?: string[]; highlight?: string[] };
type Step = { say?: string; overlay?: Overlay; act?: string; text?: string; candidates?: string[]; wait?: string; expect?: string; selector?: string; value?: string; ms?: number };
type Script = { captions: boolean; scenes: { id: string; title: string; steps: Step[] }[] };
type Manifest = Record<string, { file: string; duration_s: number }>;

const DEMO_DIR = path.resolve(__dirname);
const OUT_DIR = path.join(DEMO_DIR, "out");
const escapeHtml = (value: string) => value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);

test("record narrated Bayān demo", async ({ browser }, testInfo) => {
  await mkdir(OUT_DIR, { recursive: true });
  const script = JSON.parse(await readFile(path.join(DEMO_DIR, "script.json"), "utf8")) as Script;
  const manifest = JSON.parse(await readFile(path.join(DEMO_DIR, "audio/manifest.json"), "utf8")) as Manifest;
  const preflight = JSON.parse(await readFile(path.join(OUT_DIR, "preflight.json"), "utf8")) as { selected_questions?: Record<string, string> };
  const css = await readFile(path.join(DEMO_DIR, "overlays/overlay.css"), "utf8");
  const pipeline = await readFile(path.join(DEMO_DIR, "overlays/pipeline.html"), "utf8");
  const context: BrowserContext = await browser.newContext({
    baseURL: process.env.DEMO_BASE_URL ?? "http://localhost:3000",
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    recordVideo: { dir: testInfo.outputDir, size: { width: 1920, height: 1080 } },
  });
  const t0 = performance.now();
  const timeline: { segment_key?: string; marker?: string; t_start_ms: number }[] = [];
  let turnsBeforeAsk = 0;
  let expectedKind: string | undefined;
  let answerPending = false;
  const page: Page = await context.newPage();
  await page.emulateMedia({ reducedMotion: "reduce" });
  let video = page.video();
  try {
    await page.addStyleTag({ content: ":root { --demo-recording-context: 1; }" });
    await page.goto("/");
    await page.getByTestId("composer-dock").waitFor({ state: "visible" });
    // The product hides the language switcher in the header (display: none), so the director
    // invokes the real button's handler directly instead of clicking something the viewer can't see.
    const setLocale = async (value: "ar" | "en") => {
      const label = value === "en" ? "English" : "العربية";
      await page.evaluate((label) => {
        const button = [...document.querySelectorAll<HTMLButtonElement>('[role="group"] button')].find((b) => b.textContent?.trim() === label);
        if (!button) throw new Error(`Language button ${label} is missing`);
        button.click();
      }, label);
      await page.waitForFunction((value) => document.documentElement.lang === value || document.cookie.includes(`quran-ui-locale=${value}`), value);
    };
    await setLocale("ar");
    await page.locator(".sidebar-new-chat-button:visible, .header-new-chat-button:visible").first().click();
    await expect(page.getByTestId("conversation-turn")).toHaveCount(0, { timeout: 15_000 });
    await closeHistory();

    await page.evaluate(({ overlayCss }) => {
      const host = document.createElement("div");
      host.id = "bayan-demo-overlay-host";
      const shadow = host.attachShadow({ mode: "open" });
      shadow.innerHTML = `<style>${overlayCss}</style><div class="demo-root"><div class="demo-layer"></div></div>`;
      document.body.append(host);
    }, { overlayCss: css });

    const overlay = async (item: Overlay | undefined, caption?: string, magenta = false, livePipeline = false) => {
      await page.evaluate(({ item, caption, magenta, pipelineHtml, livePipeline }) => {
        const shadow = document.querySelector("#bayan-demo-overlay-host")?.shadowRoot;
        const layer = shadow?.querySelector<HTMLElement>(".demo-layer");
        if (!layer) throw new Error("Demo overlay root is missing");
        layer.classList.remove("is-hidden");
        const captionHtml = caption ? `<div class="demo-caption" dir="rtl">${caption}</div>` : "";
        if (magenta) layer.innerHTML = '<div class="demo-magenta"></div>';
        else if (item?.kind === "pipeline") {
          layer.innerHTML = `<div class="demo-pipeline${livePipeline ? " demo-pipeline-live" : ""}">${pipelineHtml}</div>${captionHtml}`;
          const highlighted = new Set(item.highlight ?? []);
          shadow!.querySelectorAll<HTMLElement>("[data-stage]").forEach((node) => {
            node.classList.toggle("is-active", highlighted.has(node.dataset.stage ?? ""));
          });
        } else {
          let card = "";
          if (item?.kind === "card") {
            const lines = (item.lines ?? []).map((line) => `<p dir="auto">${line}</p>`).join("");
            const title = item.title ?? "";
            const titleDirection = /^[A-Za-z0-9]/.test(title) ? "ltr" : "rtl";
            card = `<aside class="demo-card"><h2 dir="${titleDirection}">${title}</h2>${lines}</aside>`;
          }
          layer.innerHTML = `${card}${captionHtml}`;
        }
        requestAnimationFrame(() => layer.classList.remove("is-hidden"));
      }, {
        item: item ? { ...item, title: escapeHtml(item.title ?? ""), lines: (item.lines ?? []).map(escapeHtml) } : undefined,
        caption: caption ? escapeHtml(caption) : undefined,
        magenta,
        pipelineHtml: pipeline,
        livePipeline,
      });
      await assertOverlayClear(item);
    };
    const assertOverlayClear = async (item: Overlay | undefined) => {
      if (item?.kind === "pipeline" && !answerPending) return;
      {
        const failures = await page.evaluate(() => {
          const shadow = document.querySelector("#bayan-demo-overlay-host")?.shadowRoot;
          const overlayNodes = [...(shadow?.querySelectorAll<HTMLElement>(".demo-card, .demo-caption, .demo-pipeline, .demo-pulse-ring") ?? [])];
          // conversation-scroll spans the whole pane; the visible column is the stream inside it.
          const column = document.querySelector<HTMLElement>('[data-testid="conversation-scroll"] .conversation-stream');
          const headers = [...document.querySelectorAll<HTMLElement>("header, .app-header")];
          if (!column) return ["missing .conversation-stream inside conversation-scroll"];
          const boxes = (nodes: HTMLElement[]) => nodes.map((node) => ({
            name: node.className.toString(), rect: node.getBoundingClientRect().toJSON(),
          }));
          const intersects = (a: DOMRect, b: DOMRect) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
          const contentRect = column.getBoundingClientRect();
          const headerRects = headers.map((node) => node.getBoundingClientRect());
          return boxes(overlayNodes).flatMap(({ name, rect }) => {
            const box = new DOMRect(rect.x, rect.y, rect.width, rect.height);
            const problems: string[] = [];
            if (intersects(box, contentRect)) {
              problems.push(`${name} overlaps conversation content ${JSON.stringify(contentRect.toJSON())}`);
            }
            if (headerRects.some((header) => intersects(box, header))) problems.push(`${name} overlaps header`);
            return problems.map((problem) => `${problem}: ${JSON.stringify(rect)}`);
          });
        });
        expect(failures, `Rejected overlay position: ${failures.join("; ")}`).toEqual([]);
      }
    };
    const clearOverlay = async () => {
      await page.evaluate(() => document.querySelector<HTMLElement>("#bayan-demo-overlay-host")?.shadowRoot?.querySelector(".demo-layer")?.classList.add("is-hidden"));
      await page.waitForTimeout(200);
      await page.evaluate(() => {
        const layer = document.querySelector("#bayan-demo-overlay-host")?.shadowRoot?.querySelector(".demo-layer");
        if (layer) layer.innerHTML = "";
      });
    };
    const mark = async (name: string) => {
      timeline.push({ marker: name, t_start_ms: performance.now() - t0 });
      await overlay(undefined, undefined, true);
      await page.waitForTimeout(500);
      await clearOverlay();
    };
    await mark("start");

    for (const scene of script.scenes) {
      if (["architecture", "engineering", "outro"].includes(scene.id)) await returnToFirstConversation();
      for (const [stepIndex, step] of scene.steps.entries()) {
        if (step.say !== undefined) {
          const key = `${scene.id}/${stepIndex}`;
          const clip = manifest[key];
          if (!clip) throw new Error(`Audio manifest is missing ${key}`);
          // Keep narration about a generated answer after it is visible. Pipeline narration is
          // intentionally spoken while the first answer streams, with the trace open.
          if (answerPending && scene.id !== "first_question") await awaitAnswer();
          if (scene.id === "answer_walkthrough" && stepIndex === 2) {
            await page.locator('[data-testid="assistant-answer"] a.citation-chip').first().scrollIntoViewIfNeeded();
          }
          timeline.push({ segment_key: key, t_start_ms: performance.now() - t0 });
          await overlay(step.overlay, script.captions ? step.say : undefined, false,
            scene.id === "first_question" && answerPending && step.overlay?.kind === "pipeline");
          if (scene.id === "answer_walkthrough" && stepIndex === 2) {
            await pulseFirstCitation();
          }
          await page.waitForTimeout(clip.duration_s * 1000 + 250);
          if (scene.id === "answer_walkthrough" && stepIndex === 2) {
            await removePulse();
            await page.locator('[data-testid="assistant-answer"] a.citation-chip').first().click();
            // The chip sets #evidence-…; left in the URL, every later re-render jumps back to
            // the first answer's evidence, under narration about a different turn.
            await page.waitForTimeout(1200);
            await page.evaluate(() => history.replaceState(null, "", location.pathname + location.search));
          }
          await clearOverlay();
          continue;
        }
        if (!step.act) continue;
        if (step.act === "ask") {
          turnsBeforeAsk = await page.getByTestId("conversation-turn").count();
          expectedKind = step.expect;
          const question = preflight.selected_questions?.[`${scene.id}/${stepIndex}`] ?? step.text ?? step.candidates?.[0];
          if (!question) throw new Error(`No preflight-selected question for ${scene.id}/${stepIndex}`);
          const composer = page.locator("textarea[aria-label]");
          await composer.fill("");
          await composer.pressSequentially(question, { delay: 35 });
          await composer.press("Enter");
          answerPending = true;
          if (step.wait === "final") await awaitAnswer();
          if (scene.id === "first_question") await openTrace();
        } else if (step.act === "await_answer") {
          await awaitAnswer();
        } else if (step.act === "click") {
          const target = page.locator(step.selector!).first();
          await target.waitFor({ state: "visible", timeout: 180_000 });
          await target.click();
          await page.evaluate(() => history.replaceState(null, "", location.pathname + location.search));
        } else if (step.act === "scroll_to") {
          const target = page.locator(step.selector!).first();
          await target.waitFor({ state: "visible", timeout: 180_000 });
          await target.scrollIntoViewIfNeeded();
        } else if (step.act === "open_trace") {
          await openTrace();
        } else if (step.act === "set_locale") {
          await setLocale(step.value === "en" ? "en" : "ar");
        } else if (step.act === "new_chat") {
          await page.locator(".sidebar-new-chat-button:visible, .header-new-chat-button:visible").first().click();
          // The old thread's turns stay in the DOM briefly; counting them would make the next ask
          // wait for a turn index that the fresh conversation never reaches.
          await expect(page.getByTestId("conversation-turn")).toHaveCount(0, { timeout: 15_000 });
          answerPending = false;
        } else if (step.act === "pause") {
          await page.waitForTimeout(step.ms ?? 500);
        } else {
          throw new Error(`Unsupported director action: ${step.act}`);
        }
      }
    }
    await mark("end");
    await page.evaluate(() => document.querySelector("#bayan-demo-overlay-host")?.remove());
    await page.close();
    await context.close();
    video = page.video();
    if (!video) throw new Error("Playwright did not create a recording");
    const tempVideo = await video.path();
    await rename(tempVideo, path.join(OUT_DIR, "raw.webm"));
    await writeTimeline(timeline);
  } catch (error) {
    await context.close().catch(() => undefined);
    throw error;
  }

  async function awaitAnswer() {
    const turn = page.getByTestId("conversation-turn").nth(turnsBeforeAsk);
    await turn.waitFor({ state: "attached", timeout: 180_000 });
    await expect(turn.getByTestId("assistant-answer")).toBeVisible({ timeout: 180_000 });
    // Preflight passing minutes earlier does not make a live turn deterministic. A grounded
    // question that renders without citations would put the narration over the wrong outcome,
    // so fail and let the config's retries record the whole take again.
    if (expectedKind === "grounded" && (await turn.locator("a.citation-chip").count()) === 0) {
      throw new Error(`Live answer #${turnsBeforeAsk + 1} was not grounded; re-recording`);
    }
    // The honesty narration says the system declined. A cited answer here (it once answered
    // «أول من صنع الساعة» with a verse about the Hour) must never be recorded under it.
    if (expectedKind === "honest_refusal" && (await turn.locator("a.citation-chip").count()) > 0) {
      throw new Error(`Live answer #${turnsBeforeAsk + 1} answered instead of declining; re-recording`);
    }
    // The app does not scroll to a finished answer; without this the honesty narration played
    // over the first question's tafsir cards instead of the refusal it describes.
    await turn.evaluate((node) => node.scrollIntoView({ block: "start", behavior: "smooth" }));
    await page.waitForTimeout(700);
    answerPending = false;
  }

  async function openTrace() {
    const trace = page.getByTestId("execution-trace").last();
    await trace.waitFor({ state: "visible", timeout: 180_000 });
    const toggle = trace.getByRole("button").first();
    if ((await toggle.getAttribute("aria-expanded")) !== "true") await toggle.click();
  }

  async function pulseFirstCitation() {
    const chip = page.locator('[data-testid="assistant-answer"] a.citation-chip').first();
    await chip.waitFor({ state: "visible", timeout: 180_000 });
    const rect = await chip.boundingBox();
    if (!rect) throw new Error("First citation chip has no visible bounding box");
    await page.evaluate(({ rect }) => {
      const layer = document.querySelector("#bayan-demo-overlay-host")?.shadowRoot?.querySelector<HTMLElement>(".demo-layer");
      if (!layer) throw new Error("Demo overlay root is missing");
      const ring = document.createElement("div");
      ring.className = "demo-pulse-ring";
      ring.style.left = `${rect.x - 7}px`;
      ring.style.top = `${rect.y - 7}px`;
      ring.style.width = `${rect.width + 14}px`;
      ring.style.height = `${rect.height + 14}px`;
      layer.append(ring);
    }, { rect });
  }

  async function removePulse() {
    await page.evaluate(() => document.querySelector("#bayan-demo-overlay-host")?.shadowRoot?.querySelector(".demo-pulse-ring")?.remove());
  }

  async function returnToFirstConversation() {
    const historyToggle = page.locator('button[aria-controls="history-sidebar"]');
    if ((await historyToggle.getAttribute("aria-expanded")) !== "true") await historyToggle.click();
    const firstConversation = page.locator("#history-sidebar .history-item-open").filter({ hasText: "ماذا يقول القرآن عن الصبر" }).first();
    await firstConversation.waitFor({ state: "visible", timeout: 30_000 });
    await firstConversation.click();
    await page.getByTestId("conversation-turn").first().waitFor({ state: "visible", timeout: 30_000 });
    await closeHistory();
  }

  // Card and pipeline lanes live in the side margins, which exist only while the history
  // sidebar is closed (content column 512-1408 at 1920 px instead of 380-1276).
  async function closeHistory() {
    const historyToggle = page.locator('button[aria-controls="history-sidebar"]');
    if ((await historyToggle.getAttribute("aria-expanded")) === "true") await historyToggle.click();
    await expect(historyToggle).toHaveAttribute("aria-expanded", "false");
    await page.waitForFunction(() => {
      const column = document.querySelector(".conversation-stream");
      return !!column && column.getBoundingClientRect().left >= 480;
    }, undefined, { timeout: 5_000 });
  }

  async function writeTimeline(entries: typeof timeline) {
    await mkdir(OUT_DIR, { recursive: true });
    await writeFile(path.join(OUT_DIR, "timeline.json"), JSON.stringify(entries, null, 2) + "\n");
  }
});
