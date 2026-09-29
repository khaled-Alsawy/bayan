import { cookies } from "next/headers";

import { rejectUnsafeBrowserRequest } from "@/lib/request-security";

const OWNER_COOKIE = "quran-history-owner";
const MAX_BODY_BYTES = 16_000;
const MAX_RESPONSE_BYTES = 2_000_000;

function backendBaseUrl(): string {
  const parsed = new URL(process.env.BACKEND_BASE_URL ?? "http://127.0.0.1:8000");
  if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error("Invalid private backend URL configuration");
  }
  return parsed.toString().replace(/\/$/, "");
}

export async function proxyConversationRequest(
  request: Request,
  path: string[],
): Promise<Response> {
  const rejected = rejectUnsafeBrowserRequest(request);
  if (rejected) return rejected;
  const cookieStore = await cookies();
  let ownerKey = cookieStore.get(OWNER_COOKIE)?.value;
  const shouldSetOwner = !ownerKey;
  ownerKey ??= crypto.randomUUID();

  const url = new URL(request.url);
  const suffix = path.length ? `/${path.map(encodeURIComponent).join("/")}` : "";
  const target = `${backendBaseUrl()}/v1/conversations${suffix}${url.search}`;
  const body = request.method === "GET" || request.method === "DELETE"
    ? undefined
    : await request.text();
  if (body && body.length > MAX_BODY_BYTES) {
    return Response.json({ error: { code: "INVALID_REQUEST" } }, { status: 413 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers: {
        "Content-Type": "application/json",
        "X-Owner-Key": ownerKey,
        ...(process.env.PUBLIC_ACCESS_TOKEN ? { "X-Public-Access-Token": process.env.PUBLIC_ACCESS_TOKEN } : {}),
      },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(350_000),
    });
  } catch {
    return Response.json({ error: { code: "BACKEND_UNAVAILABLE" } }, { status: 503 });
  }
  const contentType = upstream.headers.get("content-type") ?? "application/json";
  const headers = new Headers({
    "Cache-Control": "no-store, max-age=0",
    "Content-Type": contentType,
  });
  const nextCursor = upstream.headers.get("x-next-cursor");
  if (nextCursor) headers.set("X-Next-Cursor", nextCursor);
  if (shouldSetOwner) {
    headers.append(
      "Set-Cookie",
      `${OWNER_COOKIE}=${ownerKey}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000`,
    );
  }
  // A trace stream is only useful while the answer is still being produced. Buffering it with
  // `.text()` holds every event until the request finishes, which turns live progress into a
  // single blob delivered after the answer is already on screen.
  if (contentType.includes("text/event-stream") && upstream.body) {
    headers.set("Connection", "keep-alive");
    headers.set("X-Accel-Buffering", "no");
    return new Response(upstream.body, { status: upstream.status, headers });
  }
  const text = await upstream.text();
  if (text.length > MAX_RESPONSE_BYTES) {
    return Response.json({ error: { code: "INVALID_BACKEND_RESPONSE" } }, { status: 502 });
  }
  return new Response(text || null, { status: upstream.status, headers });
}
