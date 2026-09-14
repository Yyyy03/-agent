import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function createWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker;
}

const bindings = {
  ASSETS: {
    fetch: async () => new Response("Not found", { status: 404 }),
  },
};

const context = {
  waitUntil() {},
  passThroughOnException() {},
};

test("server-renders the Mint Agent product landing page", async () => {
  const worker = await createWorker();
  const response = await worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    bindings,
    context,
  );

  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /Mint Agent — Evidence-first Financial Research/);
  assert.match(html, /让每一个金融判断/);
  assert.match(html, /都有证据可循/);
  assert.match(html, /开始一项真实研究/);
  assert.match(html, /查看完整评测/);
  assert.match(html, /已完成研究案例/);
  assert.match(html, /增长归因/);
  assert.match(html, /递延收入/);
  assert.match(html, /风险偏好/);
  assert.match(html, /NVIDIA · FY2025 Q3/);
  assert.match(html, /\$30\.8B/);
  assert.match(html, /How Mint Perform/);
  assert.match(html, /Try Mint Now/);
  assert.match(html, /Switch to English/);
  assert.doesNotMatch(html, /LIVE AGENT RUN/);
  assert.doesNotMatch(html, /mint-finance-agent-hero-v1\.png/);
  assert.doesNotMatch(html, /Adobe deferred revenue check/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site|codex-preview/i);
});

test("answers a greeting without pretending to run financial research", async () => {
  const worker = await createWorker();
  const response = await worker.fetch(
    new Request("http://localhost/api/research", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "你好", history: [], mode: "deep" }),
    }),
    bindings,
    context,
  );

  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^application\/x-ndjson\b/i);
  const body = await response.text();
  assert.match(body, /Mint Finance Agent/);
  assert.match(body, /直接给我一个需要调查的金融问题/);
  assert.match(body, /"evidenceCount":0/);
  assert.doesNotMatch(body, /Adobe|18\.9%/);
});

test("proxies research to the FIRE Agent Harness and preserves evidence streaming", async () => {
  const [page, route, layout, packageJson, styles, bridge] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/research/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../../bridge/fire_agent_bridge.py", import.meta.url), "utf8"),
  ]);

  assert.match(page, /conversation-history/);
  assert.match(page, /ResearchTimeline/);
  assert.match(page, /answer_delta/);
  assert.match(page, /ReactMarkdown/);
  assert.match(page, /remarkGfm/);
  assert.match(page, /remarkCitationLinks/);
  assert.match(page, /normalizeModelMarkdown/);
  assert.match(page, /cleanToolStepText/);
  assert.match(page, /markdownLinkComponents/);
  assert.match(page, /copyMessageText/);
  assert.match(page, /copyUserMessage/);
  assert.match(page, /copyAgentAnswer/);
  assert.match(page, /className="user-message-group"/);
  assert.match(page, /className=\{`message-copy-button/);
  assert.match(page, /localStorage/);
  assert.match(page, /Evidence ledger/);
  assert.match(page, /Mint-Cu · 9B/);
  assert.match(page, /Mint-Ag · 27B/);
  assert.doesNotMatch(page, /Mint-Sg/);
  assert.match(page, /选择研究模型/);
  assert.match(page, /LOCALE_STORAGE_KEY/);
  assert.match(page, /PerformancePage/);
  assert.match(page, /LandingPage/);
  assert.match(page, /type ProductView = "home" \| "performance" \| "workspace"/);
  assert.match(page, /requestedView === "workbench"/);
  assert.match(page, /landingCaseStudies/);
  assert.match(page, /moveLandingShowcase/);
  assert.match(page, /resetLandingShowcase/);
  assert.match(page, /landing-case-controls/);
  assert.doesNotMatch(page, /LIVE AGENT RUN/);
  assert.match(page, /BenchmarkScoreCard/);
  assert.match(page, /Financial Expertise Knowledge/);
  assert.match(page, /Financial Deep Research/);
  assert.match(page, /SEC filing retrieval · numerical reasoning · evidence-grounded QA/);
  assert.match(page, /Reference-free misinformation detection · comparative diagnosis/);
  assert.match(page, /providerLogos/);
  assert.match(page, /score-model-logo/);
  assert.match(page, /score-bar-marker/);
  assert.match(page, /kimi-icon-rounded-corner\.png/);
  assert.match(page, /const axisMin = Math\.max\(0, Math\.floor\(Math\.min\(\.\.\.percentages\) - 3\)\)/);
  assert.match(page, /Math\.min\(Math\.ceil\(Math\.max\(\.\.\.percentages\) \+ 3\), 100\)/);
  assert.match(page, /Math\.round\(value\)\.toString\(\)/);
  assert.match(page, /Cursor \(Grok 4\.5\)/);
  assert.match(page, /Cursor \(Composer 2\.5\)/);
  assert.match(page, /Codex \(GPT 5\.6\)/);
  assert.doesNotMatch(page, /moveBenchmarkGlow/);
  assert.match(styles, /\.score-bars:has\(> \.score-bar-item:hover\) > \.score-bar-item:not\(:hover\)/);
  assert.match(styles, /transform: translateY\(-10px\) scaleX\(1\.15\)/);
  assert.match(styles, /\.score-bar-item:hover \.score-model-logo/);
  assert.match(page, /tone-\$\{item\.provider\}/);
  assert.match(page, /name: "BizFinBench"/);
  assert.match(page, /name: "FinanceBench"/);
  assert.match(page, /name: "RFC Bench"/);
  assert.match(page, /name: "FinanceAgentBench v1\.1"/);
  assert.match(page, /name: "FinanceAgentBench v2"/);
  assert.match(page, /name: "FinSearchComp T2"/);
  assert.match(page, /name: "FinSearchComp T3"/);
  assert.match(page, /const sortedScores = \[\.\.\.benchmark\.scores\]\.sort\(\(a, b\) => b\.score - a\.score\)/);
  assert.match(page, /sortedScores\.findIndex\(\(candidate\) => candidate\.model === item\.model\) \+ 1/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.5386, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.9, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.9667, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.68, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.4198, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.6986, true\)/);
  assert.match(page, /score\(modelNames\.mintCu, 0\.3488, true\)/);
  assert.equal(page.match(/score\(modelNames\.mintCu,/g)?.length, 7);
  assert.match(page, /score\(modelNames\.mintAg, 0\.5571, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.9133, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.9833, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.76, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.6049, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.8904, true\)/);
  assert.match(page, /score\(modelNames\.mintAg, 0\.5407, true\)/);
  assert.equal(page.match(/score\(modelNames\.mintAg,/g)?.length, 7);
  assert.doesNotMatch(page, /GPT-5\.5/);
  assert.match(page, /TRACE_CONNECTION_NOISE/);
  assert.match(page, /message\.run\?\.mode === "deep"/);
  assert.match(page, /message\.run\?\.mode === "fast"/);
  assert.match(page, /function FastThinking/);
  assert.match(page, /fast-thinking-copy/);
  assert.match(page, /fast-thinking-preview/);
  assert.doesNotMatch(page, /fast-thinking-fade/);
  assert.match(page, /expandThinking/);
  assert.match(styles, /-webkit-line-clamp: 3/);
  assert.match(page, /conversationActivityKey/);
  assert.match(page, /const preservedScrollTop = canvas\?\.scrollTop/);
  assert.match(page, /canvas\.scrollTop = preservedScrollTop/);
  assert.match(styles, /overflow-anchor: none/);
  assert.match(page, /canvas\.scrollTop = canvas\.scrollHeight/);
  assert.match(page, /conversationHasActiveRun/);
  assert.match(page, /abortControllersRef = useRef<Map<string, AbortController>>/);
  assert.match(page, /abortControllersRef\.current\.set\(conversationId, controller\)/);
  assert.match(page, /runId: assistantMessage\.run!\.id/);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /patchAssistantMessage\(conversationId, runningMessage\.id/);
  assert.match(page, /historyRunning/);
  assert.match(page, /activeIsRunning/);
  assert.doesNotMatch(page, /const \[isRunning, setIsRunning\]/);
  assert.doesNotMatch(page, /const abortRef = useRef/);
  assert.match(page, /browserTimeContext\(\)/);
  assert.match(page, /Intl\.DateTimeFormat\(\)\.resolvedOptions\(\)\.timeZone/);
  assert.match(page, /isExternalSourceUrl\(selectedEvidence\.url\)/);
  assert.match(page, /复制原始工具数据/);
  assert.match(page, /rawContent\?: string/);
  assert.match(page, /item\.rawContent \|\|/);
  assert.match(page, /buildEvidencePresentation/);
  assert.match(page, /decodeStructuredValue/);
  assert.match(page, /evidence-result-list/);
  assert.match(page, /evidence-raw-data/);
  assert.match(page, /可在下方打开来源与原始数据/);
  assert.doesNotMatch(page, /可在右侧查看来源与原始数据/);
  assert.match(page, /navigator\.clipboard\?\.writeText/);
  assert.match(page, /buildEvidenceGraph/);
  assert.match(page, /evidenceGraphPath/);
  assert.match(page, /hoveredEvidenceId/);
  assert.match(page, /evidenceMessageId/);
  assert.match(page, /openEvidence\(message\.id\)/);
  assert.match(page, /openEvidenceRecord\(message\.id, message\.run\?\.evidence\[index - 1\]\)/);
  assert.match(page, /DEFAULT_EVIDENCE_DRAWER_WIDTH = 860/);
  assert.match(page, /clampEvidenceDrawerWidth/);
  assert.match(page, /startEvidenceResize/);
  assert.match(page, /moveEvidenceResize/);
  assert.match(page, /role="separator"/);
  assert.match(page, /onDoubleClick=\{resetEvidenceDrawerWidth\}/);
  assert.match(page, /evidenceQuestion \|\| activeEvidence\[0\]\?\.query/);
  assert.doesNotMatch(page, /for \(const message of activeConversation\?\.messages \?\? \[\]\)/);
  assert.match(page, /evidence-graph-node/);
  assert.doesNotMatch(page, /className="evidence-list"/);
  assert.doesNotMatch(page, /<a href=\{selectedEvidence\.url\}/);
  assert.match(page, /FIRE Agent Harness · 已连接/);
  assert.match(page, /From search trajectory to verifiable evidence/);
  assert.match(page, /Ask a question worth investigating/);
  assert.match(route, /FIRE_AGENT_BRIDGE_URL/);
  assert.match(route, /FIRE_AGENT_BRIDGE_TOKEN/);
  assert.match(route, /\/v1\/research/);
  assert.match(route, /Authorization: `Bearer \$\{bridgeToken\}`/);
  assert.match(route, /cancelBridgeResearch/);
  assert.match(route, /export async function DELETE/);
  assert.match(route, /X-Mint-Agent-Run-Id/);
  assert.match(route, /runId,/);
  assert.match(route, /fire-agent-harness/);
  assert.match(route, /body\.model === "mint-sg"/);
  assert.match(route, /sanitizeBrowserTime/);
  assert.match(route, /serverReceivedAtMs: Date\.now\(\)/);
  assert.match(route, /application\/x-ndjson/);
  assert.doesNotMatch(route, /planResearch|findEvidenceGap|searchWeb|readSource|composeAnswer/);
  assert.match(bridge, /"label": "Mint-Ag"/);
  assert.match(bridge, /values\["max_steps"\] = 1/);
  assert.match(bridge, /values\["single_turn_reasoning_pass"\] = True/);
  assert.match(bridge, /enabled_tools = \[\] if mode == "fast" else self\.enabled_tools/);
  assert.match(bridge, /task_family = "single_turn_qa"/);
  assert.match(bridge, /emit_steps=mode == "deep"/);
  assert.match(bridge, /emit_native_thinking=mode == "fast"/);
  assert.match(bridge, /step\.reasoning_content or step\.think/);
  assert.match(bridge, /request\.app\["run_semaphores"\]\[mode\]/);
  assert.match(bridge, /"deep": asyncio\.Semaphore\(deep_limit\)/);
  assert.match(bridge, /"fast": asyncio\.Semaphore\(fast_limit\)/);
  assert.match(bridge, /multiprocessing\.get_context\("spawn"\)/);
  assert.match(bridge, /run_harness_process/);
  assert.match(bridge, /stop_run_process/);
  assert.match(bridge, /add_delete\("\/v1\/research\/\{run_id\}", cancel_research\)/);
  assert.match(bridge, /control\["cancel_event"\]\.set\(\)/);
  assert.match(bridge, /FIRE_AGENT_BRIDGE_FAST_MAX_CONCURRENT/);
  assert.match(bridge, /def resolve_time_context/);
  assert.match(bridge, /Interpret 今天\/today\/当前\/现在 relative to the browser local date/);
  assert.match(bridge, /"rawContent": evidence_copy_content/);
  assert.match(bridge, /raw_tool_result/);
  assert.match(layout, /themeColor/);
  assert.match(styles, /Mint Noto Serif SC/);
  assert.match(styles, /Mint Playfair Display/);
  assert.match(styles, /"DengXian", "等线"/);
  assert.match(styles, /html\[lang="en"\]/);
  assert.match(styles, /font-variation-settings: normal/);
  assert.match(styles, /@keyframes editorialReveal/);
  assert.match(styles, /@keyframes emptyOrbitFloat/);
  assert.match(styles, /\.evidence-tool-origin/);
  assert.match(styles, /\.evidence-record-summary/);
  assert.doesNotMatch(styles, /\.evidence-fact-grid/);
  assert.match(styles, /\.evidence-result-list/);
  assert.match(styles, /\.evidence-raw-data/);
  assert.match(styles, /\.markdown-table-scroll/);
  assert.match(styles, /\.trace-markdown/);
  assert.match(styles, /\.fast-thinking/);
  assert.match(styles, /\.fast-thinking-expand/);
  assert.match(styles, /\.message-copy-button/);
  assert.match(styles, /\.message-actions-assistant/);
  assert.match(styles, /\.answer-copy blockquote/);
  assert.match(styles, /\.answer-copy pre/);
  assert.match(styles, /\.evidence-graph-canvas/);
  assert.match(styles, /\.evidence-graph-lines path\.is-focused/);
  assert.match(styles, /\.evidence-graph-node\.is-muted/);
  assert.match(styles, /\.evidence-resize-handle/);
  assert.match(styles, /body\.is-resizing-evidence/);
  assert.match(styles, /\.score-model-logo/);
  assert.match(styles, /\.landing-visual-frame/);
  assert.match(styles, /\.landing-case-window/);
  assert.match(styles, /\.landing-case-tabs/);
  assert.match(styles, /@keyframes caseSwapIn/);
  assert.match(styles, /\.landing-pathway-card/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(packageJson, /react-markdown/);
  assert.match(packageJson, /remark-gfm/);

  await Promise.all([
    access(new URL("../public/fonts/noto-serif-sc-medium-ui.ttf", import.meta.url)),
    access(new URL("../public/fonts/playfair-display-ui.ttf", import.meta.url)),
    access(new URL("../public/model-logos/gemini.svg", import.meta.url)),
    access(new URL("../public/model-logos/claude.svg", import.meta.url)),
    access(new URL("../public/model-logos/openai.svg", import.meta.url)),
    access(new URL("../public/model-logos/deepseek.svg", import.meta.url)),
    access(new URL("../public/model-logos/qwen.svg", import.meta.url)),
    access(new URL("../public/mint-finance-agent-hero-v1.png", import.meta.url)),
    assert.rejects(access(new URL("app/_sites-preview", templateRoot))),
  ]);
});
