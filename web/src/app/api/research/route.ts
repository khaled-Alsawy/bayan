import { proxyResearchRequest } from "@/lib/research-gateway";

export const dynamic = "force-dynamic";
export const maxDuration = 360;

export async function POST(request: Request) {
  return proxyResearchRequest(request);
}
