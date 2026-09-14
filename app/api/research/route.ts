export const dynamic = "force-dynamic";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

type ModelId = "mint-cu" | "mint-sg";

type BrowserTimeContext = {
  capturedAtMs?: number;
  iso?: string;
  timeZone?: string;
  utcOffsetMinutes?: number;
  locale?: string;
};

type ResearchRequest = {
  question?: string;
  history?: ChatTurn[];
  mode?: "deep" | "fast";
  model?: ModelId;
  runId?: string;
  browserTime?: BrowserTimeContext;
};

function runtimeValue(key: string) {
  return process.env[key]?.trim() || "";
}

function cleanBaseUrl(value: string) {
  return value.replace(/\/+$/, "");
}

function compactText(value: string, limit: number) {
  const compact = value
    .replace(/\u0000/g, "")
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return compact.length <= limit ? compact : `${compact.slice(0, limit).trimEnd()}…`;
}

function errorResponse(error: string, status = 502) {
  return Response.json({ error }, { status });
}

function sanitizeRunId(value: unknown) {
  if (typeof value !== "string") return "";
  const compact = value.trim();
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(compact) ? compact : "";
}

function isSimpleGreeting(question: string) {
  return /^(你好|您好|嗨|哈喽|hello|hi|hey)[！!。.，,\s]*$/i.test(question);
}

function sanitizeBrowserTime(value: BrowserTimeContext | undefined) {
  if (!value || typeof value !== "object") return undefined;
  const capturedAtMs =
    typeof value.capturedAtMs === "number" &&
    Number.isFinite(value.capturedAtMs) &&
    value.capturedAtMs >= 946684800000 &&
    value.capturedAtMs <= 4102444800000
      ? Math.trunc(value.capturedAtMs)
      : undefined;
  const iso =
    typeof value.iso === "string" ? compactText(value.iso, 64) : undefined;
  const timeZone =
    typeof value.timeZone === "string" &&
    /^[A-Za-z0-9_+\-/]{1,80}$/.test(value.timeZone)
      ? value.timeZone
      : undefined;
  const utcOffsetMinutes =
    typeof value.utcOffsetMinutes === "number" &&
    Number.isFinite(value.utcOffsetMinutes) &&
    value.utcOffsetMinutes >= -840 &&
    value.utcOffsetMinutes <= 840
      ? Math.trunc(value.utcOffsetMinutes)
      : undefined;
  const locale =
    typeof value.locale === "string" &&
    /^[A-Za-z0-9_-]{1,32}$/.test(value.locale)
      ? value.locale
      : undefined;
  if (!capturedAtMs && !iso && !timeZone && utcOffsetMinutes === undefined && !locale) {
    return undefined;
  }
  return { capturedAtMs, iso, timeZone, utcOffsetMinutes, locale };
}

function greetingResponse(mode: "deep" | "fast", model: ModelId, runId: string) {
  const startedAt = Date.now();
  const answer =
    "你好，我是 Mint Finance Agent。直接给我一个需要调查的金融问题，我会通过 FIRE Agent Harness 检索、阅读并核验证据。";
  const lines = [
    { type: "run_started", startedAt, mode, model, runId, backend: "fire-agent-harness" },
    { type: "answer_delta", delta: answer },
    { type: "done", elapsedMs: 1, evidenceCount: 0, model, backend: "fire-agent-harness" },
  ];
  return new Response(`${lines.map((line) => JSON.stringify(line)).join("\n")}\n`, {
    headers: {
      "Cache-Control": "no-cache, no-transform",
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      "X-Mint-Agent-Backend": "fire-agent-harness",
    },
  });
}

async function cancelBridgeResearch(bridgeUrl: string, bridgeToken: string, runId: string) {
  try {
    return await fetch(`${bridgeUrl}/v1/research/${encodeURIComponent(runId)}`, {
      method: "DELETE",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${bridgeToken}`,
      },
      cache: "no-store",
    });
  } catch {
    return null;
  }
}

export async function GET() {
  const bridgeUrl = cleanBaseUrl(runtimeValue("FIRE_AGENT_BRIDGE_URL") || "http://127.0.0.1:4180");
  try {
    const response = await fetch(`${bridgeUrl}/health`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    const payload = await response.text();
    return new Response(payload, {
      status: response.status,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": response.headers.get("Content-Type") || "application/json; charset=utf-8",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "无法连接 FIRE Agent Harness。";
    return errorResponse(`FIRE Agent Harness bridge unavailable: ${compactText(message, 500)}`);
  }
}

export async function POST(request: Request) {
  let body: ResearchRequest;
  try {
    body = (await request.json()) as ResearchRequest;
  } catch {
    return errorResponse("请求不是有效的 JSON。", 400);
  }

  const question = typeof body.question === "string" ? compactText(body.question, 12_000) : "";
  if (!question) return errorResponse("请输入要研究的问题。", 400);
  const mode = body.mode === "fast" ? "fast" : "deep";
  const model: ModelId = body.model === "mint-sg" ? "mint-sg" : "mint-cu";
  const runId =
    sanitizeRunId(body.runId) ||
    `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  if (isSimpleGreeting(question)) return greetingResponse(mode, model, runId);
  const browserTime = sanitizeBrowserTime(body.browserTime);

  const history = Array.isArray(body.history)
    ? body.history
        .filter(
          (turn): turn is ChatTurn =>
            Boolean(turn) &&
            (turn.role === "user" || turn.role === "assistant") &&
            typeof turn.content === "string",
        )
        .slice(-8)
        .map((turn) => ({ role: turn.role, content: compactText(turn.content, 4_000) }))
    : [];

  const bridgeUrl = cleanBaseUrl(runtimeValue("FIRE_AGENT_BRIDGE_URL") || "http://127.0.0.1:4180");
  const bridgeToken = runtimeValue("FIRE_AGENT_BRIDGE_TOKEN") || runtimeValue("FIRE_AGENT_API_KEY");
  if (!bridgeToken) {
    return errorResponse(
      "FIRE Agent Harness bridge 尚未配置：缺少 FIRE_AGENT_BRIDGE_TOKEN。",
      503,
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${bridgeUrl}/v1/research`, {
      method: "POST",
      headers: {
        Accept: "application/x-ndjson",
        Authorization: `Bearer ${bridgeToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        history,
        mode,
        model,
        runId,
        browserTime,
        serverReceivedAtMs: Date.now(),
      }),
      cache: "no-store",
      signal: request.signal,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "连接失败";
    return errorResponse(`无法连接 FIRE Agent Harness：${compactText(message, 500)}`);
  }

  if (!upstream.ok || !upstream.body) {
    const detail = compactText(await upstream.text().catch(() => ""), 800);
    return errorResponse(
      `FIRE Agent Harness 返回 ${upstream.status}${detail ? `：${detail}` : ""}`,
      upstream.status >= 400 && upstream.status < 600 ? upstream.status : 502,
    );
  }

  const upstreamReader = upstream.body.getReader();
  let upstreamFinished = false;
  let cancellationStarted = false;
  const cancelUpstream = async (reason?: unknown) => {
    if (upstreamFinished || cancellationStarted) return;
    cancellationStarted = true;
    await upstreamReader.cancel(reason).catch(() => undefined);
    await cancelBridgeResearch(bridgeUrl, bridgeToken, runId);
  };
  const onRequestAbort = () => {
    void cancelUpstream(request.signal.reason);
  };
  request.signal.addEventListener("abort", onRequestAbort, { once: true });

  const downstream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await upstreamReader.read();
        if (done) {
          upstreamFinished = true;
          request.signal.removeEventListener("abort", onRequestAbort);
          controller.close();
          return;
        }
        controller.enqueue(value);
      } catch (error) {
        request.signal.removeEventListener("abort", onRequestAbort);
        if (request.signal.aborted) {
          await cancelUpstream(request.signal.reason);
          controller.close();
          return;
        }
        await cancelUpstream(error);
        controller.error(error);
      }
    },
    async cancel(reason) {
      request.signal.removeEventListener("abort", onRequestAbort);
      await cancelUpstream(reason);
    },
  });

  return new Response(downstream, {
    status: 200,
    headers: {
      "Cache-Control": "no-cache, no-transform",
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      "X-Mint-Agent-Backend": "fire-agent-harness",
      "X-Mint-Agent-Run-Id": runId,
    },
  });
}

export async function DELETE(request: Request) {
  const runId = sanitizeRunId(new URL(request.url).searchParams.get("runId"));
  if (!runId) return errorResponse("缺少有效的 runId。", 400);
  const bridgeUrl = cleanBaseUrl(runtimeValue("FIRE_AGENT_BRIDGE_URL") || "http://127.0.0.1:4180");
  const bridgeToken = runtimeValue("FIRE_AGENT_BRIDGE_TOKEN") || runtimeValue("FIRE_AGENT_API_KEY");
  if (!bridgeToken) {
    return errorResponse(
      "FIRE Agent Harness bridge 尚未配置：缺少 FIRE_AGENT_BRIDGE_TOKEN。",
      503,
    );
  }
  const response = await cancelBridgeResearch(bridgeUrl, bridgeToken, runId);
  if (!response) return errorResponse("无法连接 FIRE Agent Harness 取消任务。");
  const payload = await response.text();
  return new Response(payload, {
    status: response.status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": response.headers.get("Content-Type") || "application/json; charset=utf-8",
    },
  });
}
