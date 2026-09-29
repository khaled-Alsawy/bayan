import { describe, expect, it, vi } from "vitest";

import { proxyResearchRequest } from "@/lib/research-gateway";
import { makeResearchResponse } from "./fixtures";

function browserRequest(body: unknown, contentType = "application/json") {
  return new Request("http://app.test/api/research", {
    method: "POST",
    headers: { "Content-Type": contentType },
    body: JSON.stringify(body),
  });
}

describe("research BFF", () => {
  it("validates both boundaries and disables caching", async () => {
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.cache).toBe("no-store");
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      return Response.json(makeResearchResponse());
    }) as typeof fetch;
    const response = await proxyResearchRequest(
      browserRequest({ message: "patience", locale: "en" }),
      { backendBaseUrl: "http://127.0.0.1:8000", fetchImpl, timeoutMs: 1000 },
    );
    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toContain("no-store");
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it("rejects malformed browser input before calling the backend", async () => {
    const fetchImpl = vi.fn() as unknown as typeof fetch;
    const response = await proxyResearchRequest(
      browserRequest({ message: "x", locale: "en", provider: "groq" }),
      { fetchImpl },
    );
    expect(response.status).toBe(400);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("rejects an invalid upstream envelope without returning its contents", async () => {
    const fetchImpl = vi.fn(async () =>
      Response.json({ secret_trace: "must-not-cross-boundary", status: "ANSWERABLE" }),
    ) as unknown as typeof fetch;
    const response = await proxyResearchRequest(
      browserRequest({ message: "patience", locale: "en" }),
      { fetchImpl },
    );
    expect(response.status).toBe(502);
    expect(await response.text()).not.toContain("secret_trace");
  });

  it("normalizes backend failures without exposing backend details", async () => {
    const fetchImpl = vi.fn(async () =>
      Response.json({ detail: "/private/path provider key failed" }, { status: 503 }),
    ) as unknown as typeof fetch;
    const response = await proxyResearchRequest(
      browserRequest({ message: "patience", locale: "en" }),
      { fetchImpl },
    );
    expect(response.status).toBe(503);
    const text = await response.text();
    expect(text).not.toContain("private/path");
    expect(text).not.toContain("provider key");
  });

  it("rejects cross-site browser mutations before they reach the private backend", async () => {
    const fetchImpl = vi.fn() as unknown as typeof fetch;
    const request = new Request("http://app.test/api/research", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "https://attacker.test",
        "Sec-Fetch-Site": "cross-site",
      },
      body: JSON.stringify({ message: "patience", locale: "en" }),
    });
    const response = await proxyResearchRequest(request, { fetchImpl });
    expect(response.status).toBe(403);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
