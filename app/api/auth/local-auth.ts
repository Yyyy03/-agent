export const LOCAL_AUTH_USERNAME = "tl-finagent";
export const LOCAL_AUTH_PASSWORD = "123456";
const LOCAL_AUTH_TOKEN = "local-session-tl-finagent";

export function isLocalSessionToken(value: string | null): boolean {
  return value?.trim() === LOCAL_AUTH_TOKEN;
}

export function localSessionResponse(): Response {
  return Response.json(
    { token: LOCAL_AUTH_TOKEN, username: LOCAL_AUTH_USERNAME },
    { headers: { "Cache-Control": "no-store" } },
  );
}

export function authRequiredResponse(): Response {
  return Response.json(
    { error: "auth_required", message: "请先登录 Mint-Agent 账号。" },
    { status: 401, headers: { "Cache-Control": "no-store" } },
  );
}
