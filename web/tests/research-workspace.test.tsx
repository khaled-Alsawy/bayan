import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ResearchWorkspace } from "@/components/research-workspace";
import { UI_LOCALE_STORAGE_KEY } from "@/lib/i18n";
import { makeResearchResponse } from "./fixtures";

const CONVERSATION_ID = "11111111-1111-4111-8111-111111111111";

function mockHistoryFetch(response = makeResearchResponse()) {
  return vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (url === "/api/conversations?limit=50") {
      return Response.json({ items: [], next_cursor: null });
    }
    if (url === "/api/conversations" && method === "POST") {
      return Response.json({
        conversation_id: CONVERSATION_ID,
        title: "محادثة جديدة",
        locale: "ar",
        archived: false,
        created_at: "2026-09-16T00:00:00Z",
        updated_at: "2026-09-16T00:00:00Z",
        turn_count: 0,
      }, { status: 201 });
    }
    if (url.endsWith("/turns/stream") && method === "POST") {
      const request = JSON.parse(String(init?.body)) as { message: string; locale: "ar" | "en" };
      return Response.json({
        turn_id: response.turn_id,
        user_message: request.message,
        locale: request.locale,
        response,
        created_at: "2026-09-16T00:01:00Z",
      }, { status: 201 });
    }
    return Response.json({ error: { code: "BACKEND_UNAVAILABLE" } }, { status: 503 });
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  document.cookie = "quran-ui-locale=; Max-Age=0; Path=/";
  document.documentElement.lang = "";
  document.documentElement.dir = "";
});

describe("ResearchWorkspace localization", () => {
  it("continues a clarification choice in the same conversation and restores it after reload", async () => {
    const clarification = makeResearchResponse("CLARIFICATION_REQUIRED");
    const answered = makeResearchResponse();
    answered.turn_id = "55555555-5555-4555-8555-555555555555";
    answered.correlation_id = "66666666-6666-4666-8666-666666666666";
    const turns: Array<Record<string, unknown>> = [];
    const submitted: string[] = [];
    const summary = {
      conversation_id: CONVERSATION_ID,
      title: "آيات العين",
      locale: "ar" as const,
      archived: false,
      created_at: "2026-09-16T00:00:00Z",
      updated_at: "2026-09-16T00:02:00Z",
      turn_count: 0,
    };
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/api/conversations?limit=50") {
        return Response.json({
          items: turns.length ? [{ ...summary, turn_count: turns.length }] : [],
          next_cursor: null,
        });
      }
      if (url === "/api/conversations" && method === "POST") {
        return Response.json(summary, { status: 201 });
      }
      if (url === `/api/conversations/${CONVERSATION_ID}?limit=100`) {
        return Response.json({ ...summary, turn_count: turns.length, turns });
      }
      if (url.endsWith("/turns/stream") && method === "POST") {
        const body = JSON.parse(String(init?.body)) as { message: string; locale: "ar" | "en" };
        submitted.push(body.message);
        const response = turns.length === 0 ? clarification : answered;
        const turn = {
          turn_id: response.turn_id,
          user_message: body.message,
          locale: body.locale,
          response,
          created_at: `2026-09-16T00:0${turns.length + 1}:00Z`,
        };
        turns.push(turn);
        return Response.json(turn, { status: 201 });
      }
      return Response.json({}, { status: 404 });
    }));

    const view = render(<ResearchWorkspace />);
    const user = userEvent.setup();
    const input = await screen.findByRole("textbox", { name: "سؤال البحث" });
    await user.type(input, "ما هي آيات العين؟");
    await user.keyboard("{Enter}");
    const choice = await screen.findByRole("button", { name: "الحسد والإصابة بالعين" });
    choice.focus();
    await user.keyboard("{Enter}");
    expect(await screen.findAllByTestId("assistant-answer")).toHaveLength(2);
    expect(submitted).toEqual(["ما هي آيات العين؟", "الحسد والإصابة بالعين"]);
    expect(window.localStorage.getItem("quran-active-conversation-id")).toBe(CONVERSATION_ID);

    view.unmount();
    render(<ResearchWorkspace />);
    expect(await screen.findByText("هل تقصد بالعين: الحسد والإصابة بالعين، أم العين الباصرة، أم عيون الماء؟")).toBeInTheDocument();
    expect(screen.getAllByTestId("assistant-answer")).toHaveLength(2);
  });

  it("renders Arabic and RTL by default and submits the Arabic response locale", async () => {
    const fetchMock = mockHistoryFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<ResearchWorkspace />);
    const user = userEvent.setup();

    await waitFor(() => expect(document.documentElement).toHaveAttribute("lang", "ar"));
    expect(document.documentElement).toHaveAttribute("dir", "rtl");
    expect(screen.getByRole("heading", { name: "اسأل، ثم اقرأ الدليل" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "ماذا يقول القرآن عن الصبر عند الشدّة؟" }));
    await user.click(screen.getByRole("button", { name: "إرسال سؤال البحث" }));
    expect(await screen.findByTestId("assistant-answer")).toBeInTheDocument();
    const request = fetchMock.mock.calls.find(([input]) => String(input).endsWith("/turns/stream"))?.[1] as RequestInit;
    expect(JSON.parse(String(request.body))).toMatchObject({ locale: "ar" });
  });

  it("switches the whole shell to English/LTR and persists the choice locally", async () => {
    vi.stubGlobal("fetch", mockHistoryFetch());
    render(<ResearchWorkspace />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "English" }));
    expect(document.documentElement).toHaveAttribute("lang", "en");
    expect(document.documentElement).toHaveAttribute("dir", "ltr");
    expect(screen.getByRole("heading", { name: "Ask with the answer and its sources in view." })).toBeInTheDocument();
    expect(window.localStorage.getItem(UI_LOCALE_STORAGE_KEY)).toBe("en");
    expect(document.cookie).toContain("quran-ui-locale=en");
  });

  it("keeps input direction independent from the Arabic application shell", async () => {
    vi.stubGlobal("fetch", mockHistoryFetch());
    render(<ResearchWorkspace />);
    const user = userEvent.setup();
    const input = screen.getByRole("textbox", { name: "سؤال البحث" });

    await user.type(input, "patience during hardship");
    expect(input).toHaveAttribute("dir", "ltr");
    expect(document.documentElement).toHaveAttribute("dir", "rtl");
  });

  it("honors an explicit response-language request without changing the UI language", async () => {
    const fetchMock = mockHistoryFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<ResearchWorkspace />);
    const user = userEvent.setup();
    const input = screen.getByRole("textbox", { name: "سؤال البحث" });

    await user.type(input, "Answer in English: what does the Quran say about patience?");
    await user.click(screen.getByRole("button", { name: "إرسال سؤال البحث" }));
    await screen.findByTestId("assistant-answer");

    const request = fetchMock.mock.calls.find(([input]) => String(input).endsWith("/turns/stream"))?.[1] as RequestInit;
    expect(JSON.parse(String(request.body))).toMatchObject({ locale: "en" });
    expect(document.documentElement).toHaveAttribute("lang", "ar");
    expect(document.documentElement).toHaveAttribute("dir", "rtl");
  });

  it("restores the active conversation from server history and isolates New chat", async () => {
    const response = makeResearchResponse();
    const summary = {
      conversation_id: CONVERSATION_ID,
      title: "Patience",
      locale: "en" as const,
      archived: false,
      created_at: "2026-09-16T00:00:00Z",
      updated_at: "2026-09-16T00:01:00Z",
      turn_count: 1,
    };
    const newId = "44444444-4444-4444-8444-444444444444";
    window.localStorage.setItem("quran-active-conversation-id", CONVERSATION_ID);
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      if (url === "/api/conversations?limit=50") {
        return Response.json({ items: [summary], next_cursor: null });
      }
      if (url === `/api/conversations/${CONVERSATION_ID}?limit=100`) {
        return Response.json({ ...summary, turns: [{
          turn_id: response.turn_id,
          user_message: "What does the Quran say about patience?",
          locale: "en",
          response,
          created_at: "2026-09-16T00:01:00Z",
        }] });
      }
      if (url === "/api/conversations" && init?.method === "POST") {
        return Response.json({ ...summary, conversation_id: newId, title: "New conversation", turn_count: 0 }, { status: 201 });
      }
      return Response.json({}, { status: 404 });
    }));

    render(<ResearchWorkspace />);
    expect(await screen.findByTestId("assistant-answer")).toBeInTheDocument();
    expect(screen.getByText("What does the Quran say about patience?")).toHaveAttribute("dir", "ltr");
    await userEvent.setup().click(screen.getByRole("button", { name: "محادثة جديدة" }));
    await waitFor(() => expect(screen.queryByTestId("assistant-answer")).not.toBeInTheDocument());
    expect(window.localStorage.getItem("quran-active-conversation-id")).toBe(newId);
  });

  it("puts the failed question back in the composer without submitting it", async () => {
    const response = makeResearchResponse("FAILED_SAFE");
    response.blocks = [{ block_type: "NOTICE", notice_code: "QUESTION_NOT_UNDERSTOOD", text: "hidden backend detail" }];
    const fetchMock = mockHistoryFetch(response);
    vi.stubGlobal("fetch", fetchMock);
    render(<ResearchWorkspace />);
    const question = "ايه الآيات في سورة الفلق؟";
    const input = await screen.findByRole("textbox", { name: "سؤال البحث" });
    await userEvent.setup().type(input, question);
    await userEvent.setup().click(screen.getByRole("button", { name: "إرسال سؤال البحث" }));
    await screen.findByRole("button", { name: "تعديل السؤال" });

    await userEvent.setup().click(screen.getByRole("button", { name: "تعديل السؤال" }));

    expect(input).toHaveValue(question);
    await waitFor(() => expect(input).toHaveFocus());
    expect(fetchMock.mock.calls.filter(([url, init]) => String(url).endsWith("/turns/stream") && init?.method === "POST")).toHaveLength(1);
  });

  it("keeps New chat first in the open sidebar and uses the header icon when closed", async () => {
    vi.stubGlobal("matchMedia", ((query: string) => ({
      matches: query.includes("min-width: 1024px"),
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    })) as typeof window.matchMedia);
    const fetchMock = mockHistoryFetch();
    vi.stubGlobal("fetch", fetchMock);
    const openView = render(<ResearchWorkspace />);
    const sidebar = await screen.findByRole("complementary", { name: "المحادثات" });
    const sidebarNewChat = within(sidebar).getByRole("button", { name: "محادثة جديدة" });
    expect(within(sidebar).getAllByRole("button")[0]).toBe(sidebarNewChat);
    expect(sidebar.querySelectorAll(".sidebar-new-chat-button")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "محادثة جديدة" })?.className).toContain("sidebar-new-chat-button");
    await userEvent.setup().click(sidebarNewChat);
    await waitFor(() => expect(fetchMock.mock.calls.some(([url, init]) => String(url) === "/api/conversations" && init?.method === "POST")).toBe(true));
    expect(sidebar.querySelectorAll(".sidebar-new-chat-button")).toHaveLength(1);
    openView.unmount();

    vi.stubGlobal("matchMedia", ((query: string) => ({
      matches: query.includes("max-width: 1023px"),
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    })) as typeof window.matchMedia);
    const closed = render(<ResearchWorkspace />);
    expect(screen.queryByRole("complementary", { name: "المحادثات" })).not.toBeInTheDocument();
    const headerNewChat = screen.getByRole("button", { name: "محادثة جديدة" });
    expect(headerNewChat).toHaveClass("header-new-chat-button");
    expect(headerNewChat.querySelector("svg")).not.toBeNull();
    await userEvent.setup().click(screen.getByRole("button", { name: "فتح سجل المحادثات" }));
    const mobileSidebar = await screen.findByRole("complementary", { name: "المحادثات" });
    await userEvent.setup().click(within(mobileSidebar).getByRole("button", { name: "محادثة جديدة" }));
    await waitFor(() => expect(screen.queryByRole("complementary", { name: "المحادثات" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "محادثة جديدة" })).toHaveClass("header-new-chat-button");
    closed.unmount();
  });
});
