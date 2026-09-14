import { proxyToBridge } from "../../bridge-helpers";

export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ id: string }> };

function safeId(value: string) {
  return /^[A-Za-z0-9._:-]{1,64}$/.test(value) ? value : "";
}

export async function PUT(request: Request, context: RouteContext) {
  const { id } = await context.params;
  const convId = safeId(id);
  if (!convId) return Response.json({ error: "bad_conversation_id" }, { status: 400 });
  const body = await request.text();
  return proxyToBridge(request, `/v1/history/${encodeURIComponent(convId)}`, {
    method: "PUT",
    body,
    requireSession: true,
  });
}

export async function DELETE(request: Request, context: RouteContext) {
  const { id } = await context.params;
  const convId = safeId(id);
  if (!convId) return Response.json({ error: "bad_conversation_id" }, { status: 400 });
  return proxyToBridge(request, `/v1/history/${encodeURIComponent(convId)}`, {
    method: "DELETE",
    requireSession: true,
  });
}
