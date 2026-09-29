import {
  ResearchRequestSchema,
  ResearchResponseSchema,
  type GatewayError,
} from "@/lib/contracts";
import { rejectUnsafeBrowserRequest } from "@/lib/request-security";

const MAX_REQUEST_BYTES = 12_000;
const MAX_RESPONSE_BYTES = 1_000_000;
const DEFAULT_TIMEOUT_MS = 330_000;

type FetchLike = typeof fetch;

export interface ResearchGatewayOptions {
  backendBaseUrl?: string;
  fetchImpl?: FetchLike;
  timeoutMs?: number;
}

function noStoreHeaders() {
  return {
    "Cache-Control": "no-store, max-age=0",
    "Content-Type": "application/json; charset=utf-8",
  };
}

function safeError(
  status: number,
  code: GatewayError["error"]["code"],
  message: string,
) {
  return Response.json(
    { error: { code, message } } satisfies GatewayError,
    { status, headers: noStoreHeaders() },
  );
}

function resolveBackendBaseUrl(value: string | undefined): string {
  const candidate = value ?? "http://127.0.0.1:8000";
  const parsed = new URL(candidate);
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error("Invalid private backend URL configuration");
  }
  return parsed.toString().replace(/\/$/, "");
}

function resolveTimeout(value: number | undefined): number {
  const timeout = value ?? DEFAULT_TIMEOUT_MS;
  return Math.max(1_000, Math.min(timeout, 360_000));
}

export async function proxyResearchRequest(
  request: Request,
  options: ResearchGatewayOptions = {},
): Promise<Response> {
  const rejected = rejectUnsafeBrowserRequest(request);
  if (rejected) return rejected;
  const contentType = request.headers.get("content-type") ?? "";
  const declaredLength = Number(request.headers.get("content-length") ?? "0");
  if (!contentType.toLowerCase().startsWith("application/json")) {
    return safeError(415, "INVALID_REQUEST", "Submit the request as JSON.");
  }
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REQUEST_BYTES) {
    return safeError(413, "INVALID_REQUEST", "The research request is too large.");
  }

  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return safeError(400, "INVALID_REQUEST", "The research request is invalid.");
  }
  const parsedRequest = ResearchRequestSchema.safeParse(rawBody);
  if (!parsedRequest.success) {
    return safeError(400, "INVALID_REQUEST", "Enter a valid Quran research question.");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), resolveTimeout(options.timeoutMs));
  let upstream: Response;
  try {
    const backendBaseUrl = resolveBackendBaseUrl(
      options.backendBaseUrl ?? process.env.BACKEND_BASE_URL,
    );
    upstream = await (options.fetchImpl ?? fetch)(`${backendBaseUrl}/v1/research`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.PUBLIC_ACCESS_TOKEN ? { "X-Public-Access-Token": process.env.PUBLIC_ACCESS_TOKEN } : {}),
      },
      body: JSON.stringify(parsedRequest.data),
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    if (error instanceof DOMException && error.name === "AbortError") {
      return safeError(
        504,
        "BACKEND_TIMEOUT",
        "Research is taking longer than expected. Please try again.",
      );
    }
    return safeError(
      503,
      "BACKEND_UNAVAILABLE",
      "The private research service is temporarily unavailable.",
    );
  }
  clearTimeout(timeout);

  if (!upstream.ok) {
    return safeError(
      upstream.status === 408 || upstream.status === 504 ? 504 : 503,
      upstream.status === 408 || upstream.status === 504
        ? "BACKEND_TIMEOUT"
        : "BACKEND_UNAVAILABLE",
      upstream.status === 408 || upstream.status === 504
        ? "Research is taking longer than expected. Please try again."
        : "The private research service is temporarily unavailable.",
    );
  }

  let text: string;
  try {
    text = await upstream.text();
  } catch {
    return safeError(
      502,
      "INVALID_BACKEND_RESPONSE",
      "The research response could not be verified.",
    );
  }
  if (text.length > MAX_RESPONSE_BYTES) {
    return safeError(
      502,
      "INVALID_BACKEND_RESPONSE",
      "The research response could not be verified.",
    );
  }

  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    return safeError(
      502,
      "INVALID_BACKEND_RESPONSE",
      "The research response could not be verified.",
    );
  }
  const parsedResponse = ResearchResponseSchema.safeParse(payload);
  if (!parsedResponse.success) {
    return safeError(
      502,
      "INVALID_BACKEND_RESPONSE",
      "The research response could not be verified.",
    );
  }

  return Response.json(parsedResponse.data, {
    status: 200,
    headers: noStoreHeaders(),
  });
}
