import { proxyConversationRequest } from "@/lib/conversation-gateway";

export const dynamic = "force-dynamic";
export const maxDuration = 360;

type Context = { params: Promise<{ path?: string[] }> };

async function proxy(request: Request, context: Context) {
  return proxyConversationRequest(request, (await context.params).path ?? []);
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
