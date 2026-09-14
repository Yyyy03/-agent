const REDEEM_ALLOWED_ORIGINS = new Set([
  "https://mint-fin.github.io",
  "http://127.0.0.1:4181",
  "http://localhost:4181",
]);

export function bridgeBase() {
  return (process.env.FIRE_AGENT_BRIDGE_URL?.trim() || "http://127.0.0.1:4180").replace(/\/+$/, "");
}

export function bridgeToken() {
  return process.env.FIRE_AGENT_BRIDGE_TOKEN?.trim() || process.env.FIRE_AGENT_API_KEY?.trim() || "";
}

export function clientIp(request: Request) {
  return (
    request.headers.get("CF-Connecting-IP")?.trim() ||
    request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    ""
  );
}

export function authHeaders(request: Request): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    Authorization: `Bearer ${bridgeToken()}`,
  };
  const session = request.headers.get("X-Mint-Session")?.trim();
  if (session) headers["X-Mint-Session"] = session;
  const ip = clientIp(request);
  if (ip) headers["X-Client-IP"] = ip;
  return headers;
}

export function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("Origin") || "";
  if (!REDEEM_ALLOWED_ORIGINS.has(origin)) return {};
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

export function requireSessionHeader(request: Request): Response | null {
  if (request.headers.get("X-Mint-Session")?.trim()) return null;
  return Response.json(
    { error: "auth_required", message: "请先登录 Mint-Agent 账号。" },
    { status: 401 },
  );
}

export async function proxyToBridge(
  request: Request,
  path: string,
  options: { method: string; body?: string; cors?: boolean; requireSession?: boolean },
): Promise<Response> {
  if (options.requireSession) {
    const denied = requireSessionHeader(request);
    if (denied) return denied;
  }
  if (!bridgeToken()) {
    return Response.json({ error: "bridge_unconfigured", message: "服务未配置。" }, { status: 503 });
  }
  const extra = options.cors ? corsHeaders(request) : {};
  let upstream: Response;
  try {
    upstream = await fetch(`${bridgeBase()}${path}`, {
      method: options.method,
      headers: {
        ...authHeaders(request),
        ...(options.body ? { "Content-Type": "application/json" } : {}),
      },
      body: options.body,
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { error: "bridge_unavailable", message: "服务暂时不可用，请稍后再试。" },
      { status: 502, headers: extra },
    );
  }
  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": upstream.headers.get("Content-Type") || "application/json; charset=utf-8",
      ...extra,
    },
  });
}
