import { corsHeaders, proxyToBridge } from "../../bridge-helpers";

export const dynamic = "force-dynamic";

export async function OPTIONS(request: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(request) });
}

export async function POST(request: Request) {
  const body = await request.text();
  return proxyToBridge(request, "/v1/auth/redeem", { method: "POST", body, cors: true });
}
