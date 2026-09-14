import {
  LOCAL_AUTH_PASSWORD,
  LOCAL_AUTH_USERNAME,
  localSessionResponse,
} from "../local-auth";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let payload: { username?: unknown; password?: unknown };
  try {
    payload = (await request.json()) as { username?: unknown; password?: unknown };
  } catch {
    return Response.json({ error: "invalid_request" }, { status: 400 });
  }

  const username =
    typeof payload.username === "string" ? payload.username.trim().toLowerCase() : "";
  const password = typeof payload.password === "string" ? payload.password : "";

  if (username !== LOCAL_AUTH_USERNAME || password !== LOCAL_AUTH_PASSWORD) {
    return Response.json(
      { error: "invalid_credentials", message: "用户名或密码不正确。" },
      { status: 401, headers: { "Cache-Control": "no-store" } },
    );
  }

  return localSessionResponse();
}
