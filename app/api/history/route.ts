import { proxyToBridge } from "../bridge-helpers";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return proxyToBridge(request, "/v1/history", { method: "GET", requireSession: true });
}
