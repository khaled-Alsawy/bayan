const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

function firstHeaderValue(value: string | null): string | null {
  return value?.split(",", 1)[0]?.trim() || null;
}

function requestHost(request: Request): string | null {
  return firstHeaderValue(request.headers.get("x-forwarded-host"))
    ?? request.headers.get("host")
    ?? new URL(request.url).host;
}

function hostnameOf(host: string): string | null {
  try {
    return new URL(`http://${host}`).hostname.toLowerCase();
  } catch {
    return null;
  }
}

function allowedHost(host: string): boolean {
  const hostname = hostnameOf(host);
  if (!hostname) return false;
  if (["localhost", "127.0.0.1", "::1"].includes(hostname)) return true;
  if (hostname.endsWith(".trycloudflare.com")) return true;
  if (process.env.NODE_ENV === "test" && hostname.endsWith(".test")) return true;
  const configured = (process.env.PUBLIC_WEB_HOSTS ?? "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return configured.includes(hostname);
}

function sameOrigin(request: Request, host: string): boolean {
  const site = request.headers.get("sec-fetch-site");
  if (site === "cross-site") return false;
  const origin = request.headers.get("origin");
  if (!origin) return true;
  try {
    const parsed = new URL(origin);
    const expectedProtocol = firstHeaderValue(request.headers.get("x-forwarded-proto"))
      ?? new URL(request.url).protocol.replace(":", "");
    return parsed.host.toLowerCase() === host.toLowerCase()
      && parsed.protocol === `${expectedProtocol}:`;
  } catch {
    return false;
  }
}

export function rejectUnsafeBrowserRequest(request: Request): Response | null {
  const host = requestHost(request);
  if (!host || !allowedHost(host)) {
    return Response.json({ error: { code: "INVALID_REQUEST" } }, { status: 421 });
  }
  if (!SAFE_METHODS.has(request.method) && !sameOrigin(request, host)) {
    return Response.json({ error: { code: "INVALID_REQUEST" } }, { status: 403 });
  }
  // A tunnel is intentionally opt-in. When enabled, every public browser request must carry
  // the configured bearer; local development remains credential-free.
  const publicToken = process.env.PUBLIC_ACCESS_TOKEN;
  if (publicToken && hostnameOf(host)?.endsWith(".trycloudflare.com")) {
    if (request.headers.get("authorization") !== `Bearer ${publicToken}`) {
      return Response.json({ error: { code: "INVALID_REQUEST" } }, { status: 401 });
    }
  }
  return null;
}
