/// <reference types="react/canary" />
"use client";

import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type IconName =
  | "archive"
  | "arrow"
  | "book"
  | "check"
  | "chevron"
  | "clock"
  | "close"
  | "copy"
  | "file"
  | "globe"
  | "history"
  | "menu"
  | "paperclip"
  | "pencil"
  | "plus"
  | "search"
  | "send"
  | "spark"
  | "stop"
  | "tool"
  | "trash";

type EvidenceItem = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  content: string;
  rawContent?: string;
  source: string;
  query: string;
  publishedAt?: string;
};

type EvidenceResult = {
  title: string;
  url: string;
  domain: string;
  snippet: string;
};

type EvidencePresentation = {
  summary: string;
  results: EvidenceResult[];
  rawPreview: string;
  rawTruncated: boolean;
};

type EvidenceGraphNode = {
  item: EvidenceItem;
  x: number;
  y: number;
  kind: "web" | "market" | "filing";
};

type EvidenceGraphEdge = {
  id: string;
  from: string;
  to: string;
};

type EvidenceGraph = {
  nodes: EvidenceGraphNode[];
  edges: EvidenceGraphEdge[];
  height: number;
};

type JsonRecord = Record<string, unknown>;

const evidenceDetailCopy = {
  zh: {
    structuredRecord: "结构化证据",
    structuredSummary: "原始工具结果已整理为可阅读的数据视图。",
    searchSummary: (count: number) =>
      `本次检索返回 ${count} 条候选结果。标题、来源域名与关键摘要已整理如下。`,
    sourceResults: "来源结果",
    rawData: "原始数据",
    rawPreview: "折叠查看结构化记录",
    rawTruncated: "预览内容较长，复制按钮仍会复制完整原始数据。",
  },
  en: {
    structuredRecord: "Structured evidence",
    structuredSummary: "The original tool result has been organized into a readable data view.",
    searchSummary: (count: number) =>
      `${count} candidate result${count === 1 ? "" : "s"} returned. Titles, source domains, and key excerpts are organized below.`,
    sourceResults: "Source results",
    rawData: "Raw data",
    rawPreview: "Expand structured record",
    rawTruncated: "The preview is shortened; the copy action still copies the complete raw record.",
  },
} as const;

function isJsonRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function decodeStructuredValue(value: unknown, depth = 0): unknown {
  if (depth > 5) return value;
  if (typeof value === "string") {
    const trimmed = value
      .trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "");
    if (!trimmed || (!trimmed.startsWith("{") && !trimmed.startsWith("["))) return value;
    try {
      return decodeStructuredValue(JSON.parse(trimmed), depth + 1);
    } catch {
      return value;
    }
  }
  if (Array.isArray(value)) {
    return value.map((entry) => decodeStructuredValue(entry, depth + 1));
  }
  if (isJsonRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        decodeStructuredValue(entry, depth + 1),
      ]),
    );
  }
  return value;
}

function firstRecordText(record: JsonRecord | null, keys: string[]) {
  if (!record) return "";
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" || typeof value === "boolean") return String(value);
  }
  return "";
}

function findStructuredRecord(
  value: unknown,
  keys: string[],
  depth = 0,
): JsonRecord | null {
  if (depth > 7) return null;
  if (isJsonRecord(value)) {
    if (keys.some((key) => key in value)) return value;
    for (const nested of Object.values(value)) {
      const match = findStructuredRecord(nested, keys, depth + 1);
      if (match) return match;
    }
  }
  if (Array.isArray(value)) {
    for (const nested of value) {
      const match = findStructuredRecord(nested, keys, depth + 1);
      if (match) return match;
    }
  }
  return null;
}

function readableEvidenceText(value: unknown) {
  if (typeof value !== "string") return "";
  const text = value.trim();
  if (!text || text.startsWith("{") || text.startsWith("[") || text.includes('\\"tool\\"')) {
    return "";
  }
  return text;
}

function buildEvidencePresentation(
  item: EvidenceItem,
  locale: Locale,
): EvidencePresentation {
  const labels = evidenceDetailCopy[locale];
  const rawSource =
    item.rawContent ||
    JSON.stringify(
      {
        title: item.title,
        source: item.source,
        query: item.query,
        content: item.content,
      },
      null,
      2,
    );
  const decoded = decodeStructuredValue(rawSource);
  const root = isJsonRecord(decoded) ? decoded : null;
  const evidence = isJsonRecord(root?.evidence) ? root.evidence : root;
  const searchRecord =
    findStructuredRecord(decoded, ["top_results", "result_count"]) ||
    findStructuredRecord(
      [
        decodeStructuredValue(evidence?.quote_or_row_excerpt),
        decodeStructuredValue(evidence?.snippet),
        decodeStructuredValue(item.snippet),
        decodeStructuredValue(item.content),
      ],
      ["top_results", "result_count"],
    );

  const rawResults = Array.isArray(searchRecord?.top_results)
    ? searchRecord.top_results
    : [];
  const results = rawResults
    .filter(isJsonRecord)
    .map((result) => ({
      title:
        firstRecordText(result, ["title", "name"]) ||
        (locale === "zh" ? "候选来源" : "Candidate source"),
      url: firstRecordText(result, ["url", "link", "source_url"]),
      domain: firstRecordText(result, ["domain", "source", "provider"]),
      snippet: firstRecordText(result, ["snippet", "description", "quote", "text"]),
    }))
    .slice(0, 8);

  const fallbackSummary =
    readableEvidenceText(firstRecordText(evidence, ["fact", "quote_or_row_excerpt", "snippet"])) ||
    readableEvidenceText(item.snippet) ||
    readableEvidenceText(item.content);
  const summary = results.length
    ? labels.searchSummary(
        Number(firstRecordText(searchRecord, ["result_count"])) || results.length,
      )
    : fallbackSummary || labels.structuredSummary;
  const prettyRaw =
    typeof decoded === "string" ? decoded : JSON.stringify(decoded, null, 2);
  const rawLimit = 16000;

  return {
    summary,
    results,
    rawPreview: prettyRaw.slice(0, rawLimit),
    rawTruncated: prettyRaw.length > rawLimit,
  };
}

const EVIDENCE_GRAPH_X = [34, 236, 112, 258, 26, 204, 78, 248, 132, 28, 226, 102];

function evidenceGraphKind(item: EvidenceItem): EvidenceGraphNode["kind"] {
  const text = `${item.title} ${item.source} ${item.url}`.toLowerCase();
  if (/market_data|akshare|quote|snapshot|行情|交易所/.test(text)) return "market";
  if (/10-k|10-q|annual|filing|sec\.gov|investor\.|年报|财报/.test(text)) return "filing";
  return "web";
}

function evidenceGraphSourceKey(item: EvidenceItem) {
  const source = item.source.trim().toLowerCase();
  if (source) return source;
  try {
    return new URL(item.url).hostname.toLowerCase();
  } catch {
    return item.title.split(/\s+/)[0]?.toLowerCase() ?? item.id;
  }
}

function buildEvidenceGraph(items: EvidenceItem[]): EvidenceGraph {
  const nodes = items.map((item, index) => ({
    item,
    x: EVIDENCE_GRAPH_X[index % EVIDENCE_GRAPH_X.length],
    y: 92 + index * 74,
    kind: evidenceGraphKind(item),
  }));
  const edges: EvidenceGraphEdge[] = [];
  const edgeKeys = new Set<string>();

  function connect(from: string, to: string) {
    if (!from || !to || from === to) return;
    const key = [from, to].sort().join("::");
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edges.push({ id: key, from, to });
  }

  for (let index = 0; index < Math.min(items.length, 3); index += 1) {
    connect("root", items[index].id);
  }
  for (let index = 1; index < items.length; index += 1) {
    connect(items[index - 1].id, items[index].id);
  }

  const latestBySource = new Map<string, string>();
  for (const item of items) {
    const sourceKey = evidenceGraphSourceKey(item);
    const related = latestBySource.get(sourceKey);
    if (related) connect(related, item.id);
    latestBySource.set(sourceKey, item.id);
  }

  return {
    nodes,
    edges,
    height: Math.max(360, 190 + Math.max(items.length - 1, 0) * 74),
  };
}

function evidenceGraphPoint(node: EvidenceGraphNode | undefined) {
  if (!node) return { x: 220, y: 47 };
  return { x: node.x + 82, y: node.y + 37 };
}

function evidenceGraphPath(from: EvidenceGraphNode | undefined, to: EvidenceGraphNode | undefined) {
  const start = evidenceGraphPoint(from);
  const end = evidenceGraphPoint(to);
  const middleY = start.y + (end.y - start.y) * 0.5;
  return `M ${start.x} ${start.y} C ${start.x} ${middleY}, ${end.x} ${middleY}, ${end.x} ${end.y}`;
}

type ResearchStep = {
  id: string;
  kind: "thought" | "tool";
  status: "running" | "complete";
  title: string;
  text: string;
  tool?: string;
  count?: number;
};

type ModelId = "mint-cu" | "mint-sg";
type Locale = "zh" | "en";
type ProductView = "home" | "performance" | "workspace";

type BrowserTimeContext = {
  capturedAtMs: number;
  iso: string;
  timeZone: string;
  utcOffsetMinutes: number;
  locale: string;
};

type ResearchRun = {
  id: string;
  status: "running" | "complete" | "error" | "cancelled";
  mode: "deep" | "fast";
  model: ModelId;
  startedAt: number;
  elapsedMs: number;
  steps: ResearchStep[];
  evidence: EvidenceItem[];
  expanded: boolean;
  error?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  run?: ResearchRun;
};

type Conversation = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
};

function conversationHasActiveRun(conversation?: Conversation) {
  return Boolean(
    conversation?.messages.some(
      (message) => message.role === "assistant" && message.run?.status === "running",
    ),
  );
}

function conversationActivityKey(conversation?: Conversation) {
  if (!conversation) return "";
  return conversation.messages
    .map((message) => {
      const run = message.run;
      const steps =
        run?.steps.map((step) => `${step.id}:${step.status}:${step.text.length}`).join(",") ?? "";
      return [
        message.id,
        message.content.length,
        run?.status ?? "",
        steps,
        run?.evidence.length ?? 0,
      ].join(":");
    })
    .join("|");
}

type StreamEvent =
  | { type: "run_started"; startedAt: number; mode: "deep" | "fast"; model: ModelId; runId?: string }
  | { type: "step"; step: ResearchStep }
  | { type: "evidence"; evidence: EvidenceItem }
  | { type: "answer_delta"; delta: string }
  | { type: "done"; elapsedMs: number; evidenceCount: number }
  | { type: "error"; message: string };

const icons: Record<IconName, string[]> = {
  archive: ["M4 7h16v13H4V7Z", "M7 3h10l3 4H4l3-4Z", "M9 12h6"],
  arrow: ["M5 12h14", "m13-6 6 6-6 6"],
  book: ["M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 1 4 16.5v-11Z", "M4 16.5A2.5 2.5 0 0 1 6.5 14H20"],
  check: ["m5 12 4 4L19 6"],
  chevron: ["m9 18 6-6-6-6"],
  clock: ["M12 6v6l4 2", "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"],
  close: ["M6 6l12 12M18 6 6 18"],
  copy: ["M9 9h11v11H9V9Z", "M4 4h11v11H4V4Z"],
  file: ["M7 3h7l5 5v13H7V3Z", "M14 3v6h5", "M9 13h6M9 17h6"],
  globe: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M3 12h18", "M12 3c2.2 2.5 3.4 5.5 3.4 9S14.2 18.5 12 21c-2.2-2.5-3.4-5.5-3.4-9S9.8 5.5 12 3Z"],
  history: ["M3 12a9 9 0 1 0 3-6.7", "M3 4v5h5", "M12 7v5l3 2"],
  menu: ["M4 7h16M4 12h16M4 17h16"],
  paperclip: ["m20.5 11.5-8.9 8.9a6 6 0 0 1-8.5-8.5l9.6-9.6a4 4 0 0 1 5.7 5.7l-9.6 9.6a2 2 0 0 1-2.8-2.8l8.9-8.9"],
  pencil: ["M4 20l1-4L16.5 4.5a2.1 2.1 0 0 1 3 3L8 19l-4 1Z", "M13.5 7.5l3 3"],
  plus: ["M12 5v14M5 12h14"],
  search: ["M11 17a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z", "m16 16 4 4"],
  send: ["M12 19V5", "m6 11 6-6 6 6"],
  spark: ["m12 3 1.25 4.25L17.5 8.5l-4.25 1.25L12 14l-1.25-4.25L6.5 8.5l4.25-1.25L12 3Z", "m18 15 .75 2.25L21 18l-2.25.75L18 21l-.75-2.25L15 18l2.25-.75L18 15Z"],
  stop: ["M7 7h10v10H7z"],
  tool: ["M14.7 6.3a4 4 0 0 0-5-5L12 3.6 8.4 7.2 6.1 4.9a4 4 0 0 0 5 5L4 17l3 3 7.7-7.7a4 4 0 0 0 5-5L17.4 9.6 14.7 6.3Z"],
  trash: ["M4 7h16", "M9 7V4h6v3", "M8 10v8M12 10v8M16 10v8", "M6 7l1 14h10l1-14"],
};

const STORAGE_KEY = "mint-agent-conversations-v2";
const LOCALE_STORAGE_KEY = "mint-agent-locale-v1";
const EVIDENCE_WIDTH_STORAGE_KEY = "mint-agent-evidence-width-v1";
const SESSION_STORAGE_KEY = "mint-agent-session-v1";
const PROJECT_SITE_URL = "https://mint-fin.github.io/mint-agent/#contact";
const DEFAULT_LOCAL_SESSION = { token: "local-session-tl-finagent", username: "tl-finagent" };

function readStoredSession(): { token: string; username: string } | null {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { token?: string; username?: string };
    if (typeof parsed.token === "string" && parsed.token && typeof parsed.username === "string") {
      return { token: parsed.token, username: parsed.username };
    }
  } catch {
    /* fall through */
  }
  return DEFAULT_LOCAL_SESSION;
}

function sessionFetch(token: string | null, input: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  if (token) headers.set("X-Mint-Session", token);
  return fetch(input, { ...init, headers });
}
const DEFAULT_EVIDENCE_DRAWER_WIDTH = 860;
const MIN_EVIDENCE_DRAWER_WIDTH = 520;
const MAX_EVIDENCE_DRAWER_WIDTH = 1240;
const modelLabels: Record<ModelId, string> = {
  "mint-cu": "Mint-Cu · 9B",
  "mint-sg": "Mint-Ag · 27B",
};
const initialConversation: Conversation = {
  id: "welcome",
  title: "新研究",
  updatedAt: 0,
  messages: [],
};

const uiCopy = {
  zh: {
    brandTagline: "金融研究",
    navPerformanceHint: "能力与评测",
    navWorkspaceHint: "开始研究",
    landingEyebrow: "MINT · FINANCIAL RESEARCH AGENT",
    landingTitleFirst: "让每一个金融判断，",
    landingTitleSecond: "都有证据可循。",
    landingIntro:
      "从一个问题出发，Mint 自主规划检索、调用金融工具、阅读原始来源并交叉核验，最终交付可追溯的判断，而不是一段无法复核的文字。",
    landingPathwaysEyebrow: "EXPLORE MINT",
    landingPathwaysTitle: "先看能力，或直接开始。",
    landingPerformanceDescription: "查看 Mint-Cu 与 Mint-Ag 在七项金融知识与深度研究评测中的完整表现。",
    landingWorkspaceDescription: "把真实问题交给 FIRE Agent Harness，实时观察计划、工具调用与 evidence 积累。",
    landingAgentTitle: "一个问题，完整走完研究链路。",
    landingAgentDescription: "每一步都留痕，每一条关键结论都能回到来源。",
    switchLanguage: "Switch to English",
    chinese: "中",
    english: "EN",
    newResearch: "新研究",
    conversationHistory: "对话历史",
    questionCount: (count: number) => `${count} 个问题`,
    historyRunning: (count: number) => `研究中 · ${count} 个问题`,
    waitingForQuestion: "等待第一个问题",
    now: "现在",
    deleteConversation: (title: string) => `删除对话 ${title}`,
    deleteTitle: "删除对话",
    deleteConfirm: (title: string) => `删除「${title}」？此操作无法恢复。`,
    agentReady: "Agent 已就绪",
    evidenceFirst: "证据优先研究",
    openHistory: "打开对话历史",
    closeSidebar: "关闭侧边栏",
    deepResearch: "深度研究",
    fastAnswer: "快速回答",
    evidenceCount: (count: number) => `${count} 条 evidence`,
    createResearch: "新建研究",
    emptyEyebrow: "Mint Finance Agent",
    emptyTitleFirst: "从一个问题开始，",
    emptyTitleSecond: "让证据自己说话。",
    emptyDescription: "我会进行多轮检索、阅读来源、补齐证据缺口，并把研究过程与引用一并交付。",
    fastDescription: "单轮 Harness 会直接回答，不启动检索时间线；需要实时取证时可切换到深度研究。",
    quickPrompts: [
      "比较 Adobe 最近两个财年的递延收入变化，并引用年报证据",
      "分析最新 A 股融资余额变化是否说明风险偏好改善",
      "英伟达最近一季数据中心收入增长的主要驱动是什么？",
    ],
    viewSources: (count: number) => `查看 ${count} 条来源`,
    inputAria: "输入金融研究问题",
    inputPlaceholder: "输入一个需要调查的问题…",
    selectModel: "选择研究模型",
    stopResearch: "停止研究",
    sendQuestion: "发送问题",
    composerHint: "Enter 发送 · Shift + Enter 换行 · 重要结论会附带来源编号",
    closeEvidence: "关闭 evidence",
    resizeEvidence: "拖拽调整研究来源面板宽度；双击恢复默认宽度",
    evidenceLedger: "Evidence ledger",
    researchSources: "研究来源",
    evidenceGraph: "证据关系图",
    evidenceGraphHint: "悬停节点以追踪来源关系",
    evidenceGraphRoot: "研究问题",
    evidenceGraphCount: (count: number) => `${count} 个证据节点`,
    collected: "已采集",
    foundByQuery: "由此查询发现",
    openOriginal: "打开原始来源",
    toolEvidenceNoLink: "这是工具返回的结构化数据，没有网页原文；可复制原始证据记录。",
    copyToolEvidence: "复制原始工具数据",
    copiedToolEvidence: "已复制",
    copiedMessage: "已复制",
    copyUserMessage: "复制用户问题",
    copyAgentAnswer: "复制 Agent 回答",
    evidenceEmpty: "研究开始后，来源会在这里逐条积累。",
    traceRunning: (elapsed: string) => `研究中（已用时 ${elapsed}）`,
    traceError: "研究遇到问题",
    traceCancelled: "研究已停止",
    traceComplete: (elapsed: string) => `已研究（用时 ${elapsed}）`,
    fastThinking: "思考过程",
    fastThinkingRunning: "模型正在思考",
    fastThinkingSource: "Mint 原生 thinking",
    expandThinking: "展开完整思考",
    collapseThinking: "收起思考过程",
    buildingPlan: "正在建立研究计划",
    buildingPlanDetail: "将问题转换为可以验证的搜索任务。",
    incompleteAnswer: "这次研究没有完成。我保留了已经取得的步骤和 evidence，便于你检查。",
    stoppedAnswer: "研究已停止。",
    serviceStatus: (status: number) => `研究服务返回 ${status}`,
    noStream: "浏览器没有收到研究事件流。",
    connectionFailed: "无法连接研究服务。",
    performanceEyebrow: "MINT 研究能力",
    performanceTitle: "从检索轨迹到可核验证据",
    performanceIntro:
      "Mint 不只生成答案。它通过 FIRE Agent Harness 规划、搜索、阅读、补证并综合结论，让每一个重要判断都能回到来源。",
    heroPrimaryAction: "开始一项真实研究",
    heroSecondaryAction: "查看完整评测",
    heroSignals: ["多轮自主检索", "可审计 Evidence Ledger", "来源级引用"],
    caseShowcaseAria: "Mint 真实研究案例",
    caseCompleted: "已完成研究案例",
    caseArchive: "Research archive",
    caseVerified: "证据核验通过",
    caseKeyFinding: "关键判断",
    caseResearchPath: "研究轨迹",
    caseEvidence: "核心证据",
    casePrimarySources: "原始来源",
    caseOpenSource: "打开来源",
    casePrevious: "上一个案例",
    caseNext: "下一个案例",
    caseSelectedEvidence: "展示 3 条关键证据",
    agentRunMeta: ["12 条来源", "6 次工具调用"],
    liveHarness: "FIRE Agent Harness · 已连接",
    scoreEyebrow: "BENCHMARK SCORE",
    scoreTitle: "Mint 表现",
    scoreDescription: "同一评测口径下的模型准确率对比；柱状图按分数从高到低排列。",
    scoreSource: "数据来自 7 月评测汇总",
    singleTurn: "单轮评测",
    multiTurn: "多轮评测",
    financialExpertiseKnowledge: "金融专业知识",
    financialDeepResearch: "金融深度研究",
    modelsEvaluated: (count: number) => `${count} 个模型`,
    notEvaluated: "未评测",
    scrollHint: "横向滚动查看全部模型",
    scoreLegendMint: "Mint 模型",
    scoreLegendOther: "对比模型",
    modelsTitle: "当前模型",
    live: "在线",
    cuDescription: "9B 快速研究模型，面向高效率的金融检索与证据整理。",
    agDescription: "27B 深度研究模型，面向复杂、多步骤和交叉验证任务。",
    methodologyEyebrow: "评测方法",
    methodologyTitle: "以完整研究轨迹评测，而不只看最终答案",
    methodologyDescription:
      "Harness 会保留每轮计划、工具调用、来源阅读、证据积累与最终综合，便于复现和审计。",
    pipeline: ["计划", "搜索", "阅读", "证据", "综合"],
    dimensionsTitle: "核心评测维度",
    dimensions: [
      ["来源可追溯", "检查关键结论能否回到原始来源"],
      ["工具执行", "检查搜索与阅读工具是否真实完成"],
      ["多步研究", "检查是否根据证据缺口继续补充检索"],
      ["引用覆盖", "检查重要事实是否带有对应引用"],
    ],
    audited: "可审计",
    suitesTitle: "金融评测套件",
    suites: [
      ["FinanceBench", "财报问答、数值核验与跨年度比较"],
      ["FinSearchComp", "金融搜索、来源选择与综合判断"],
      ["RFC Task 2", "复杂披露、组合分析与研究轨迹"],
    ],
    harnessConnected: "Harness 已连接",
    ctaEyebrow: "真实运行，不是静态演示",
    ctaTitle: "现在给 Mint 一个金融问题",
    ctaDescription: "选择 Mint-Cu 或 Mint-Ag，实时查看它如何规划、调用工具并积累 evidence。",
    ctaButton: "进入研究工作台",
    authTitle: "登录 Mint-Agent",
    authSubtitle: "使用邀请码兑换的账号登录，对话历史将安全保存在服务端。",
    authUsername: "用户名",
    authPassword: "密码",
    authSubmit: "登录",
    authSubmitting: "登录中…",
    authFailed: "用户名或密码不正确。",
    authRateLimited: "尝试次数过多，请稍后再试。",
    authNetworkError: "无法连接服务，请稍后再试。",
    authExpired: "会话已过期，请重新登录。",
    authRequired: "请先登录后再开始研究。",
    authNoAccount: "还没有账号？",
    authGetCode: "前往项目主页用邀请码兑换 →",
    signedInAs: "当前账号",
    signOut: "退出登录",
    historySynced: "历史对话已同步云端",
    renameTitle: "重命名",
    renameConversation: (title: string) => `重命名对话 ${title}`,
    renamePrompt: "输入新的对话名称：",
    accountDetails: "点击查看账号详情",
    accountQuotaTitle: "今日提问额度",
    accountQuotaUsed: (used: number, limit: number) => `已用 ${used} / ${limit} 次`,
    accountQuotaRemaining: (left: number) => `剩余 ${left} 次`,
    accountQuotaReset: "北京时间每日 00:00 重置",
    accountQuotaLoading: "额度加载中…",
    accountQuotaError: "额度信息加载失败，请稍后再试。",
    accountHistoryNote: "对话历史永久保存在服务端",
  },
  en: {
    brandTagline: "Financial Research",
    navPerformanceHint: "Benchmarks",
    navWorkspaceHint: "Start researching",
    landingEyebrow: "MINT · FINANCIAL RESEARCH AGENT",
    landingTitleFirst: "Every financial judgment,",
    landingTitleSecond: "grounded in evidence.",
    landingIntro:
      "Start with one question. Mint plans the search, calls financial tools, reads primary sources, cross-checks the evidence, and delivers a judgment you can trace—not a paragraph you cannot audit.",
    landingPathwaysEyebrow: "EXPLORE MINT",
    landingPathwaysTitle: "See the capability, or put it to work.",
    landingPerformanceDescription:
      "Explore how Mint-Cu and Mint-Ag perform across seven financial knowledge and deep-research evaluations.",
    landingWorkspaceDescription:
      "Give the FIRE Agent Harness a real question and watch the plan, tool calls, and evidence accumulate live.",
    landingAgentTitle: "One question. The complete research trajectory.",
    landingAgentDescription: "Every step is preserved. Every material conclusion can return to its source.",
    switchLanguage: "切换到中文",
    chinese: "中",
    english: "EN",
    newResearch: "New research",
    conversationHistory: "Conversation history",
    questionCount: (count: number) => `${count} ${count === 1 ? "question" : "questions"}`,
    historyRunning: (count: number) =>
      `Researching · ${count} ${count === 1 ? "question" : "questions"}`,
    waitingForQuestion: "Waiting for the first question",
    now: "Now",
    deleteConversation: (title: string) => `Delete conversation ${title}`,
    deleteTitle: "Delete conversation",
    deleteConfirm: (title: string) => `Delete “${title}”? This action cannot be undone.`,
    agentReady: "Agent ready",
    evidenceFirst: "Evidence-first research",
    openHistory: "Open conversation history",
    closeSidebar: "Close sidebar",
    deepResearch: "Deep research",
    fastAnswer: "Fast answer",
    evidenceCount: (count: number) => `${count} ${count === 1 ? "evidence item" : "evidence items"}`,
    createResearch: "Create research",
    emptyEyebrow: "Mint Finance Agent",
    emptyTitleFirst: "Start with a question.",
    emptyTitleSecond: "Let the evidence speak.",
    emptyDescription:
      "I will search iteratively, read the sources, close evidence gaps, and deliver the full research trail with citations.",
    fastDescription:
      "The single-turn Harness answers directly without a research timeline; switch to Deep Research for live evidence gathering.",
    quickPrompts: [
      "Compare Adobe deferred revenue across its two latest fiscal years and cite the filings",
      "Does the latest change in China A-share margin financing suggest improving risk appetite?",
      "What drove NVIDIA’s data center revenue growth in its latest quarter?",
    ],
    viewSources: (count: number) => `View ${count} ${count === 1 ? "source" : "sources"}`,
    inputAria: "Enter a financial research question",
    inputPlaceholder: "Ask a question worth investigating…",
    selectModel: "Select research model",
    stopResearch: "Stop research",
    sendQuestion: "Send question",
    composerHint: "Enter to send · Shift + Enter for a new line · Material claims include source numbers",
    closeEvidence: "Close evidence",
    resizeEvidence: "Drag to resize the research sources panel; double-click to reset",
    evidenceLedger: "Evidence ledger",
    researchSources: "Research sources",
    evidenceGraph: "Evidence graph",
    evidenceGraphHint: "Hover a node to trace its relationships",
    evidenceGraphRoot: "Research question",
    evidenceGraphCount: (count: number) => `${count} evidence nodes`,
    collected: "Collected",
    foundByQuery: "Discovered through",
    openOriginal: "Open original source",
    toolEvidenceNoLink: "This is structured tool data with no source page; copy the original evidence record instead.",
    copyToolEvidence: "Copy raw tool data",
    copiedToolEvidence: "Copied",
    copiedMessage: "Copied",
    copyUserMessage: "Copy user question",
    copyAgentAnswer: "Copy agent answer",
    evidenceEmpty: "Sources will accumulate here as the research progresses.",
    traceRunning: (elapsed: string) => `Researching (${elapsed} elapsed)`,
    traceError: "Research encountered an issue",
    traceCancelled: "Research stopped",
    traceComplete: (elapsed: string) => `Researched in ${elapsed}`,
    fastThinking: "Thinking",
    fastThinkingRunning: "The model is thinking",
    fastThinkingSource: "Native Mint thinking",
    expandThinking: "Expand full thinking",
    collapseThinking: "Collapse thinking",
    buildingPlan: "Building the research plan",
    buildingPlanDetail: "Turning the question into verifiable search tasks.",
    incompleteAnswer: "This research did not complete. The steps and evidence collected so far are preserved for review.",
    stoppedAnswer: "Research stopped.",
    serviceStatus: (status: number) => `Research service returned ${status}`,
    noStream: "The browser did not receive a research event stream.",
    connectionFailed: "Unable to connect to the research service.",
    performanceEyebrow: "MINT RESEARCH CAPABILITY",
    performanceTitle: "From search trajectory to verifiable evidence",
    performanceIntro:
      "Mint does more than generate an answer. Through the FIRE Agent Harness, it plans, searches, reads, closes evidence gaps, and synthesizes conclusions so every material claim can return to its source.",
    heroPrimaryAction: "Start a real investigation",
    heroSecondaryAction: "Explore the benchmarks",
    heroSignals: ["Iterative autonomous search", "Audit-ready evidence ledger", "Source-level citations"],
    caseShowcaseAria: "Mint real-world research cases",
    caseCompleted: "Completed research case",
    caseArchive: "Research archive",
    caseVerified: "Evidence verified",
    caseKeyFinding: "Key finding",
    caseResearchPath: "Research trajectory",
    caseEvidence: "Material evidence",
    casePrimarySources: "Primary sources",
    caseOpenSource: "Open source",
    casePrevious: "Previous case",
    caseNext: "Next case",
    caseSelectedEvidence: "Showing 3 material items",
    agentRunMeta: ["12 sources", "6 tool calls"],
    liveHarness: "FIRE Agent Harness · Connected",
    scoreEyebrow: "BENCHMARK SCORE",
    scoreTitle: "How Mint performs",
    scoreDescription: "Model accuracy under the same evaluation protocol, ranked from highest to lowest.",
    scoreSource: "Source: July evaluation summary",
    singleTurn: "Single-turn",
    multiTurn: "Multi-turn",
    financialExpertiseKnowledge: "Financial Expertise Knowledge",
    financialDeepResearch: "Financial Deep Research",
    modelsEvaluated: (count: number) => `${count} models evaluated`,
    notEvaluated: "Not evaluated",
    scrollHint: "Scroll horizontally to view every model",
    scoreLegendMint: "Mint models",
    scoreLegendOther: "Comparison models",
    modelsTitle: "Models available now",
    live: "Live",
    cuDescription: "A 9B fast-research model for efficient financial search and evidence organization.",
    agDescription: "A 27B deep-research model for complex, multi-step, cross-validated investigations.",
    methodologyEyebrow: "METHODOLOGY",
    methodologyTitle: "Evaluate the full research trajectory, not only the final answer",
    methodologyDescription:
      "The harness preserves every planning turn, tool call, source read, evidence item, and final synthesis for reproducible audits.",
    pipeline: ["Plan", "Search", "Read", "Evidence", "Synthesize"],
    dimensionsTitle: "Core evaluation dimensions",
    dimensions: [
      ["Source traceability", "Can material claims be traced to primary sources?"],
      ["Tool execution", "Did search and reader tools actually complete?"],
      ["Multi-step research", "Did the agent search again when evidence was missing?"],
      ["Citation coverage", "Are important facts paired with the right citations?"],
    ],
    audited: "Audited",
    suitesTitle: "Financial evaluation suites",
    suites: [
      ["FinanceBench", "Filing QA, numerical verification, and period comparisons"],
      ["FinSearchComp", "Financial search, source selection, and synthesis"],
      ["RFC Task 2", "Complex disclosures, portfolio analysis, and research trajectories"],
    ],
    harnessConnected: "Harness connected",
    ctaEyebrow: "LIVE RESEARCH, NOT A STATIC DEMO",
    ctaTitle: "Give Mint a financial question now",
    ctaDescription:
      "Choose Mint-Cu or Mint-Ag and watch it plan, call tools, and accumulate evidence in real time.",
    ctaButton: "Open the research workspace",
    authTitle: "Sign in to Mint-Agent",
    authSubtitle:
      "Use the account issued with your invitation code. Conversations are stored securely on the server.",
    authUsername: "Username",
    authPassword: "Password",
    authSubmit: "Sign in",
    authSubmitting: "Signing in…",
    authFailed: "Incorrect username or password.",
    authRateLimited: "Too many attempts. Please try again later.",
    authNetworkError: "Cannot reach the service. Please try again later.",
    authExpired: "Session expired. Please sign in again.",
    authRequired: "Please sign in before starting research.",
    authNoAccount: "No account yet?",
    authGetCode: "Redeem an invitation code on the project page →",
    signedInAs: "Signed in as",
    signOut: "Sign out",
    historySynced: "History synced to the server",
    renameTitle: "Rename",
    renameConversation: (title: string) => `Rename conversation ${title}`,
    renamePrompt: "Enter a new conversation name:",
    accountDetails: "View account details",
    accountQuotaTitle: "Today's question quota",
    accountQuotaUsed: (used: number, limit: number) => `Used ${used} of ${limit}`,
    accountQuotaRemaining: (left: number) => `${left} left`,
    accountQuotaReset: "Resets daily at 00:00 Beijing time",
    accountQuotaLoading: "Loading quota…",
    accountQuotaError: "Couldn't load quota info. Try again later.",
    accountHistoryNote: "Conversations are stored permanently on the server",
  },
};

type LandingCaseStudy = {
  id: string;
  tab: Record<Locale, string>;
  eyebrow: Record<Locale, string>;
  question: Record<Locale, string>;
  finding: Record<Locale, string>;
  metric: string;
  metricLabel: Record<Locale, string>;
  delta: string;
  sourceCount: number;
  toolCalls: number;
  chart: Array<{ label: string; value: number }>;
  evidence: Array<{
    source: string;
    detail: Record<Locale, string>;
    status: Record<Locale, string>;
    url: string;
  }>;
};

const landingCaseStudies: LandingCaseStudy[] = [
  {
    id: "nvidia-q3-fy25",
    tab: { zh: "增长归因", en: "Growth drivers" },
    eyebrow: { zh: "NVIDIA · FY2025 Q3", en: "NVIDIA · FY2025 Q3" },
    question: {
      zh: "英伟达数据中心收入增长的主要驱动是什么？",
      en: "What drove NVIDIA data center revenue growth?",
    },
    finding: {
      zh: "数据中心收入达到 307.71 亿美元，同比上升 112%。主要增量来自 Hopper 计算平台需求；网络业务环比回落，因此增长归因应集中在 Compute，而不是笼统归于全部数据中心产品。",
      en: "Data Center revenue reached $30.771B, up 112% year over year. Hopper compute demand supplied the principal increment; networking declined sequentially, so the attribution belongs primarily to Compute—not the entire platform indiscriminately.",
    },
    metric: "$30.8B",
    metricLabel: { zh: "数据中心收入", en: "Data Center revenue" },
    delta: "+112% YoY",
    sourceCount: 12,
    toolCalls: 6,
    chart: [
      { label: "Q3’24", value: 14.5 },
      { label: "Q4’24", value: 18.4 },
      { label: "Q1’25", value: 22.6 },
      { label: "Q2’25", value: 26.3 },
      { label: "Q3’25", value: 30.8 },
    ],
    evidence: [
      {
        source: "CFO Commentary",
        detail: {
          zh: "Data Center $30.771B · QoQ +17% · YoY +112%",
          en: "Data Center $30.771B · QoQ +17% · YoY +112%",
        },
        status: { zh: "数值核验", en: "Metric verified" },
        url: "https://investor.nvidia.com/files/doc_financials/2025/Q325/Q3FY25-CFO-Commentary.pdf",
      },
      {
        source: "Form 10-Q",
        detail: {
          zh: "Compute 收入增长与 Hopper GPU 计算平台需求一致",
          en: "Compute growth aligns with demand for the Hopper GPU computing platform",
        },
        status: { zh: "原始披露", en: "Primary filing" },
        url: "https://investor.nvidia.com/financial-info/financial-reports-and-sec-filings/default.aspx",
      },
      {
        source: "Earnings materials",
        detail: {
          zh: "Networking 环比下降 15%，排除其作为主要环比驱动",
          en: "Networking declined 15% sequentially, ruling it out as the main QoQ driver",
        },
        status: { zh: "交叉验证", en: "Cross-checked" },
        url: "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx",
      },
    ],
  },
  {
    id: "adobe-deferred-revenue",
    tab: { zh: "递延收入", en: "Deferred revenue" },
    eyebrow: { zh: "ADOBE · FY2017 10-K", en: "ADOBE · FY2017 10-K" },
    question: {
      zh: "Adobe 的递延收入变化是否支持订阅业务继续扩张？",
      en: "Does Adobe deferred revenue support continued subscription expansion?",
    },
    finding: {
      zh: "递延收入由 24.92 亿美元升至 29.63 亿美元，增幅约 18.9%。结合 Digital Media 与 Experience Cloud 的订阅收入披露，余额增长与订阅模式扩张方向一致，但不能脱离收入确认政策单独解释为未来收入增速。",
      en: "Deferred revenue increased from $2.492B to $2.963B, approximately 18.9%. Together with subscription disclosures for Digital Media and Experience Cloud, the balance supports continued subscription expansion, while remaining subject to revenue-recognition timing.",
    },
    metric: "$2.963B",
    metricLabel: { zh: "FY2017 递延收入", en: "FY2017 deferred revenue" },
    delta: "+18.9% YoY",
    sourceCount: 8,
    toolCalls: 4,
    chart: [
      { label: "FY2016", value: 2.492 },
      { label: "FY2017", value: 2.963 },
    ],
    evidence: [
      {
        source: "Adobe FY2017 10-K",
        detail: {
          zh: "递延收入期末余额 $2.963B",
          en: "Period-end deferred revenue of $2.963B",
        },
        status: { zh: "数值核验", en: "Metric verified" },
        url: "https://www.sec.gov/Archives/edgar/data/796343/000079634318000015/adbe10kfy17.htm",
      },
      {
        source: "Adobe FY2016 10-K",
        detail: {
          zh: "可比基期余额 $2.492B",
          en: "Comparable prior-year balance of $2.492B",
        },
        status: { zh: "口径对齐", en: "Basis aligned" },
        url: "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_ADBE_2016.pdf",
      },
      {
        source: "Revenue note",
        detail: {
          zh: "订阅增长与递延收入上升方向一致，保留收入确认时点限制",
          en: "Subscription growth aligns with the increase, with recognition timing preserved as a caveat",
        },
        status: { zh: "解释核验", en: "Interpretation checked" },
        url: "https://www.sec.gov/Archives/edgar/data/796343/000079634318000015/adbe10kfy17.htm",
      },
    ],
  },
  {
    id: "a-share-risk-appetite",
    tab: { zh: "风险偏好", en: "Risk appetite" },
    eyebrow: { zh: "A 股 · 60 个交易日快照", en: "A-SHARES · 60-SESSION SNAPSHOT" },
    question: {
      zh: "融资余额与成交活跃度是否共同确认风险偏好改善？",
      en: "Do margin balances and turnover jointly confirm improving risk appetite?",
    },
    finding: {
      zh: "单日放量不足以确认趋势。研究框架先观察成交额是否持续抬升，再检查融资余额是否同向跟进；只有两个序列在多个窗口内共同改善，才把信号从“交易活跃”升级为“风险偏好改善”。",
      en: "A single high-volume day is insufficient. The framework first tests whether turnover remains elevated, then whether margin balances follow in the same direction; only multi-window confirmation upgrades the signal from activity to improving risk appetite.",
    },
    metric: "60D",
    metricLabel: { zh: "双序列诊断窗口", en: "Dual-series diagnostic window" },
    delta: "2-series check",
    sourceCount: 9,
    toolCalls: 8,
    chart: [
      { label: "T−60", value: 42 },
      { label: "T−45", value: 46 },
      { label: "T−30", value: 45 },
      { label: "T−15", value: 53 },
      { label: "T", value: 58 },
    ],
    evidence: [
      {
        source: "上交所市场数据",
        detail: {
          zh: "沪市融资交易与市场成交序列",
          en: "Shanghai margin-trading and market-turnover series",
        },
        status: { zh: "交易所来源", en: "Exchange source" },
        url: "https://www.sse.com.cn/market/stockdata/overview/day/",
      },
      {
        source: "深交所市场数据",
        detail: {
          zh: "深市融资余额与成交活跃度序列",
          en: "Shenzhen margin-balance and turnover series",
        },
        status: { zh: "交易所来源", en: "Exchange source" },
        url: "https://www.szse.cn/market/overview/index.html",
      },
      {
        source: "AKShare snapshot",
        detail: {
          zh: "结构化行情快照用于统一日期、单位与缺失值检查",
          en: "Structured snapshots align dates, units, and missing-value checks",
        },
        status: { zh: "结构核验", en: "Schema checked" },
        url: "https://akshare.akfamily.xyz/data/stock/stock.html",
      },
    ],
  },
];

type BenchmarkScore = {
  model: string;
  score: number;
  provider: ModelProvider;
  mint?: boolean;
};

type ModelProvider =
  | "mint"
  | "google"
  | "anthropic"
  | "openai"
  | "zai"
  | "minimax"
  | "kimi"
  | "deepseek"
  | "xiaomi"
  | "alibaba"
  | "meta"
  | "cursor"
  | "agent";

type BenchmarkChart = {
  id: string;
  name: string;
  subtitle: Record<Locale, string>;
  scores: BenchmarkScore[];
};

// Public brand marks are sourced from @lobehub/icons-static-svg (MIT).
// Research models without a confirmed public mark are intentionally left blank.
const providerLogos: Partial<Record<ModelProvider, string>> = {
  mint: "/favicon.svg",
  google: "/model-logos/gemini.svg",
  anthropic: "/model-logos/claude.svg",
  openai: "/model-logos/openai.svg",
  zai: "/model-logos/zai.svg",
  minimax: "/model-logos/minimax.svg",
  kimi: "https://moonshotai.github.io/Branding-Guide/scenarios/03-icon-without-kimi/kimi-icon-rounded-corner.png",
  deepseek: "/model-logos/deepseek.svg",
  xiaomi: "/model-logos/xiaomimimo.svg",
  alibaba: "/model-logos/qwen.svg",
  cursor: "/model-logos/cursor.svg",
};

const modelNames = {
  gemini: "Gemini 3.5 Flash",
  claude: "Claude Opus 4.8",
  gpt56: "GPT-5.6 Sol",
  glm: "GLM 5.2",
  minimax: "MiniMax M3",
  kimi: "Kimi K2.7 Code",
  deepseekPro: "DeepSeek V4 Pro",
  deepseekFlash: "DeepSeek V4 Flash",
  mimo: "MiMo V2.5 Pro",
  qwen: "Qwen3.7 Plus",
  mintCu: "Mint-Cu",
  mintAg: "Mint-Ag",
  agentsA1: "Agents-A1-35B",
  nex: "Nex-N2-mini",
  asearcher: "ASearcher-32B",
  openThinker: "OpenThinkerAgent-32B",
  openResearcher: "OpenResearcher-30B-A3B",
  tongyi: "Tongyi-DeepResearch-30B-A3B",
  codex: "Codex (GPT 5.6)",
  cursorGrok: "Cursor (Grok 4.5)",
  cursorComposer: "Cursor (Composer 2.5)",
} as const;

function modelProvider(model: string): ModelProvider {
  if (model.startsWith("Mint-")) return "mint";
  if (model.startsWith("Gemini")) return "google";
  if (model.startsWith("Claude")) return "anthropic";
  if (model.startsWith("GPT-") || model.startsWith("Codex")) return "openai";
  if (model.startsWith("GLM")) return "zai";
  if (model.startsWith("MiniMax")) return "minimax";
  if (model.startsWith("Kimi")) return "kimi";
  if (model.startsWith("DeepSeek")) return "deepseek";
  if (model.startsWith("MiMo") || model.startsWith("Mimo")) return "xiaomi";
  if (model.startsWith("Qwen") || model.startsWith("Tongyi")) return "alibaba";
  if (model.startsWith("Cursor")) return "cursor";
  return "agent";
}

function score(model: string, value: number, mint = false): BenchmarkScore {
  return { model, score: value, provider: modelProvider(model), mint };
}

const benchmarkCharts: BenchmarkChart[] = [
  {
    id: "bizfinbench",
    name: "BizFinBench",
    subtitle: {
      zh: "数值计算 · 推理 · 信息抽取 · 预测识别 · 知识问答",
      en: "Calculation · reasoning · extraction · forecasting · knowledge QA",
    },
    scores: [
      score(modelNames.gemini, 0.5071),
      score(modelNames.claude, 0.4971),
      score(modelNames.gpt56, 0.5157),
      score(modelNames.glm, 0.4757),
      score(modelNames.minimax, 0.4071),
      score(modelNames.kimi, 0.4657),
      score(modelNames.deepseekPro, 0.44),
      score(modelNames.deepseekFlash, 0.4571),
      score(modelNames.mimo, 0.45),
      score(modelNames.qwen, 0.4843),
      score(modelNames.agentsA1, 0.4186),
      score(modelNames.nex, 0.2743),
      score(modelNames.asearcher, 0.2814),
      score(modelNames.openThinker, 0.25),
      score(modelNames.openResearcher, 0.1071),
      score(modelNames.tongyi, 0.1957),
      score(modelNames.codex, 0.4614),
      score(modelNames.cursorGrok, 0.4829),
      score(modelNames.cursorComposer, 0.47),
      score(modelNames.mintCu, 0.5386, true),
      score(modelNames.mintAg, 0.5571, true),
    ],
  },
  {
    id: "financebench",
    name: "FinanceBench",
    subtitle: {
      zh: "SEC 文件检索 · 数值推理 · 证据问答",
      en: "SEC filing retrieval · numerical reasoning · evidence-grounded QA",
    },
    scores: [
      score(modelNames.gemini, 0.8533),
      score(modelNames.claude, 0.9),
      score(modelNames.gpt56, 0.8667),
      score(modelNames.glm, 0.88),
      score(modelNames.minimax, 0.86),
      score(modelNames.kimi, 0.82),
      score(modelNames.deepseekPro, 0.8733),
      score(modelNames.deepseekFlash, 0.8867),
      score(modelNames.mimo, 0.8733),
      score(modelNames.qwen, 0.88),
      score(modelNames.agentsA1, 0.84),
      score(modelNames.nex, 0.6933),
      score(modelNames.asearcher, 0.8133),
      score(modelNames.openThinker, 0.6667),
      score(modelNames.openResearcher, 0.5),
      score(modelNames.tongyi, 0.4267),
      score(modelNames.codex, 0.8533),
      score(modelNames.cursorGrok, 0.88),
      score(modelNames.cursorComposer, 0.8467),
      score(modelNames.mintCu, 0.9, true),
      score(modelNames.mintAg, 0.9133, true),
    ],
  },
  {
    id: "rfcbench",
    name: "RFC-Bench",
    subtitle: {
      zh: "无参照金融错误信息识别 · 对照诊断",
      en: "Reference-free misinformation detection · comparative diagnosis",
    },
    scores: [
      score(modelNames.gemini, 0.9233),
      score(modelNames.claude, 0.9533),
      score(modelNames.gpt56, 0.9467),
      score(modelNames.glm, 0.9433),
      score(modelNames.minimax, 0.9133),
      score(modelNames.kimi, 0.9267),
      score(modelNames.deepseekPro, 0.9133),
      score(modelNames.deepseekFlash, 0.8967),
      score(modelNames.mimo, 0.9133),
      score(modelNames.qwen, 0.9333),
      score(modelNames.agentsA1, 0.9),
      score(modelNames.nex, 0.8333),
      score(modelNames.asearcher, 0.84),
      score(modelNames.openThinker, 0.71),
      score(modelNames.openResearcher, 0.22),
      score(modelNames.tongyi, 0.5767),
      score(modelNames.codex, 0.8633),
      score(modelNames.cursorGrok, 0.6333),
      score(modelNames.cursorComposer, 0.6967),
      score(modelNames.mintCu, 0.9667, true),
      score(modelNames.mintAg, 0.9833, true),
    ],
  },
  {
    id: "financeagent-v1",
    name: "FinanceAgentBench v1.1",
    subtitle: {
      zh: "多步金融任务执行 · 工具协作",
      en: "Multi-step financial task execution · tool orchestration",
    },
    scores: [
      score(modelNames.gemini, 0.62),
      score(modelNames.claude, 0.66),
      score(modelNames.gpt56, 0.66),
      score(modelNames.glm, 0.58),
      score(modelNames.minimax, 0.7),
      score(modelNames.kimi, 0.62),
      score(modelNames.deepseekPro, 0.6),
      score(modelNames.deepseekFlash, 0.72),
      score(modelNames.mimo, 0.66),
      score(modelNames.qwen, 0.64),
      score(modelNames.agentsA1, 0.42),
      score(modelNames.nex, 0.38),
      score(modelNames.asearcher, 0.24),
      score(modelNames.openThinker, 0.34),
      score(modelNames.openResearcher, 0.42),
      score(modelNames.tongyi, 0.58),
      score(modelNames.codex, 0.66),
      score(modelNames.cursorGrok, 0.66),
      score(modelNames.cursorComposer, 0.72),
      score(modelNames.mintCu, 0.68, true),
      score(modelNames.mintAg, 0.76, true),
    ],
  },
  {
    id: "financeagent-v2",
    name: "FinanceAgentBench v2",
    subtitle: {
      zh: "长程金融推理 · 鲁棒工具调用",
      en: "Long-horizon financial reasoning · robust tool use",
    },
    scores: [
      score(modelNames.gemini, 0.2593),
      score(modelNames.claude, 0.5556),
      score(modelNames.gpt56, 0.5679),
      score(modelNames.glm, 0.5185),
      score(modelNames.minimax, 0.3827),
      score(modelNames.kimi, 0.4321),
      score(modelNames.deepseekPro, 0.4074),
      score(modelNames.deepseekFlash, 0.4321),
      score(modelNames.mimo, 0.3704),
      score(modelNames.qwen, 0.4815),
      score(modelNames.agentsA1, 0.1111),
      score(modelNames.nex, 0.0741),
      score(modelNames.asearcher, 0.037),
      score(modelNames.openThinker, 0),
      score(modelNames.openResearcher, 0),
      score(modelNames.tongyi, 0.0988),
      score(modelNames.codex, 0.5432),
      score(modelNames.cursorGrok, 0.5309),
      score(modelNames.cursorComposer, 0.4938),
      score(modelNames.mintCu, 0.4198, true),
      score(modelNames.mintAg, 0.6049, true),
    ],
  },
  {
    id: "finsearch-t2",
    name: "FinSearchComp T2",
    subtitle: {
      zh: "金融检索 · 来源选择 · 证据综合",
      en: "Financial retrieval · source selection · evidence synthesis",
    },
    scores: [
      score(modelNames.gemini, 0.3653),
      score(modelNames.claude, 0.4566),
      score(modelNames.gpt56, 0.411),
      score(modelNames.glm, 0.6849),
      score(modelNames.minimax, 0.6027),
      score(modelNames.kimi, 0.5662),
      score(modelNames.deepseekPro, 0.6073),
      score(modelNames.deepseekFlash, 0.5982),
      score(modelNames.mimo, 0.5845),
      score(modelNames.qwen, 0.621),
      score(modelNames.agentsA1, 0.4703),
      score(modelNames.nex, 0.5708),
      score(modelNames.asearcher, 0.4292),
      score(modelNames.openThinker, 0.3607),
      score(modelNames.openResearcher, 0.5388),
      score(modelNames.tongyi, 0.7215),
      score(modelNames.codex, 0.8037),
      score(modelNames.cursorGrok, 0.8174),
      score(modelNames.cursorComposer, 0.7763),
      score(modelNames.mintCu, 0.6986, true),
      score(modelNames.mintAg, 0.8904, true),
    ],
  },
  {
    id: "finsearch-t3",
    name: "FinSearchComp T3",
    subtitle: {
      zh: "深度搜索 · 跨来源核验 · 复杂综合",
      en: "Deep search · cross-source verification · complex synthesis",
    },
    scores: [
      score(modelNames.gemini, 0.4419),
      score(modelNames.claude, 0.5116),
      score(modelNames.gpt56, 0.5058),
      score(modelNames.glm, 0.3488),
      score(modelNames.minimax, 0.3023),
      score(modelNames.kimi, 0.25),
      score(modelNames.deepseekPro, 0.314),
      score(modelNames.deepseekFlash, 0.25),
      score(modelNames.mimo, 0.2616),
      score(modelNames.qwen, 0.3198),
      score(modelNames.agentsA1, 0.1163),
      score(modelNames.nex, 0.2035),
      score(modelNames.asearcher, 0.0407),
      score(modelNames.openThinker, 0.1453),
      score(modelNames.openResearcher, 0.1686),
      score(modelNames.tongyi, 0.1802),
      score(modelNames.codex, 0.5407),
      score(modelNames.cursorGrok, 0.5233),
      score(modelNames.cursorComposer, 0.5116),
      score(modelNames.mintCu, 0.3488, true),
      score(modelNames.mintAg, 0.5407, true),
    ],
  },
];

function Icon({ name }: { name: IconName }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      {icons[name].map((path) => (
        <path d={path} key={path} />
      ))}
    </svg>
  );
}

function uid(prefix: string) {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function timestamp() {
  return Date.now();
}

async function writeClipboardText(text: string) {
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
}

function clampEvidenceDrawerWidth(width: number, viewportWidth: number) {
  const viewportMaximum = Math.max(
    Math.min(MIN_EVIDENCE_DRAWER_WIDTH, viewportWidth),
    viewportWidth - 240,
  );
  const maximum = Math.min(MAX_EVIDENCE_DRAWER_WIDTH, viewportMaximum);
  const minimum = Math.min(MIN_EVIDENCE_DRAWER_WIDTH, maximum);
  return Math.round(Math.min(maximum, Math.max(minimum, width)));
}

function browserTimeContext(): BrowserTimeContext {
  const now = new Date();
  let timeZone = "UTC";
  try {
    timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    // UTC is a stable fallback for browsers that do not expose an IANA zone.
  }
  return {
    capturedAtMs: now.getTime(),
    iso: now.toISOString(),
    timeZone,
    utcOffsetMinutes: -now.getTimezoneOffset(),
    locale: navigator.language || "zh-CN",
  };
}

function conversationTitle(question: string) {
  const compact = question.replace(/\s+/g, " ").trim();
  return compact.length > 26 ? `${compact.slice(0, 26)}…` : compact;
}

function formatElapsed(milliseconds: number, locale: Locale) {
  const seconds = Math.max(1, Math.round(milliseconds / 1000));
  if (locale === "en") {
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
}

function formatHistoryTime(timestamp: number, locale: Locale) {
  if (!timestamp) return uiCopy[locale].now;
  const date = new Date(timestamp);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString(locale === "zh" ? "zh-CN" : "en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return date.toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric" });
}

function upsertById<T extends { id: string }>(items: T[], item: T) {
  const index = items.findIndex((entry) => entry.id === item.id);
  if (index < 0) return [...items, item];
  return items.map((entry, itemIndex) => (itemIndex === index ? item : entry));
}

function isExternalSourceUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

const TRACE_CONNECTION_NOISE =
  /\b(?:connection(?:error|failed|refused|reset|closed)?|clientconnectorerror|connectorerror|connecterror|econn(?:refused|reset)|fetch failed|network error|request timeout|read timeout|connect timeout|timed out|server disconnected)\b|连接(?:失败|错误|超时|中断|被拒绝)|无法连接|网络(?:错误|异常)|请求超时|读取超时/i;

function isCleanResearchStep(step: ResearchStep) {
  return !TRACE_CONNECTION_NOISE.test(`${step.title}\n${step.text}`);
}

function isCleanTraceError(message?: string) {
  return Boolean(message && !TRACE_CONNECTION_NOISE.test(message));
}

type MarkdownAstNode = {
  type?: string;
  value?: string;
  url?: string;
  children?: MarkdownAstNode[];
};

function remarkCitationLinks() {
  return (tree: unknown) => {
    const visit = (node: MarkdownAstNode) => {
      if (!node.children || node.type === "code" || node.type === "inlineCode" || node.type === "link") {
        return;
      }
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text" || !child.value || !/\[\d+]/.test(child.value)) {
          visit(child);
          return [child];
        }
        return child.value
          .split(/(\[\d+])/g)
          .filter(Boolean)
          .map((part): MarkdownAstNode => {
            const citation = part.match(/^\[(\d+)]$/);
            if (!citation) return { type: "text", value: part };
            return {
              type: "link",
              url: `#mint-citation-${citation[1]}`,
              children: [{ type: "text", value: citation[1] }],
            };
          });
      });
    };
    visit(tree as MarkdownAstNode);
  };
}

function markdownLinkComponents(onCitation?: (index: number) => void): Components {
  return {
    a({ children, href }) {
      const citation = href?.match(/^#mint-citation-(\d+)$/);
      if (citation && onCitation) {
        return (
          <button
            className="inline-citation"
            onClick={() => onCitation(Number(citation[1]))}
            type="button"
          >
            {children}
          </button>
        );
      }
      const external = Boolean(href && isExternalSourceUrl(href));
      return (
        <a
          className="markdown-link"
          href={href}
          rel={external ? "noreferrer" : undefined}
          target={external ? "_blank" : undefined}
        >
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="markdown-table-scroll">
          <table>{children}</table>
        </div>
      );
    },
  };
}

function normalizeModelMarkdown(content: string) {
  return content.replace(/\*\*([^\n*]+?\S)\s+\*\*/g, "**$1**");
}

function AnswerText({
  content,
  onCitation,
}: {
  content: string;
  onCitation: (index: number) => void;
}) {
  return (
    <div className="answer-copy">
      <ReactMarkdown
        components={markdownLinkComponents(onCitation)}
        remarkPlugins={[remarkGfm, remarkCitationLinks]}
      >
        {normalizeModelMarkdown(content)}
      </ReactMarkdown>
    </div>
  );
}

function cleanToolStepText(step: ResearchStep, locale: Locale) {
  const text = step.text.trim();
  const containsStructuredPayload =
    text.length > 520 || /structured_result|accessionNo|companyNameLong|^\s*[\[{].*[}\]]\s*$/s.test(text);
  if (!containsStructuredPayload) return text;

  const rawValue = text.match(/(?:^|\n)raw:\s*([^\n]{1,120})/i)?.[1]?.trim();
  const conciseRawValue =
    rawValue && !/^[\[{]/.test(rawValue) ? `\n\n**${locale === "zh" ? "返回值" : "Result"}：** ${rawValue}` : "";
  return locale === "zh"
    ? `工具调用成功。完整结构化结果已收录至 **Evidence Ledger**，可在下方打开来源与原始数据。${conciseRawValue}`
    : `Tool call completed. The full structured result is preserved in the **Evidence Ledger** for source and raw-data review.${conciseRawValue}`;
}

function TraceMarkdown({ content }: { content: string }) {
  return (
    <div className="trace-markdown">
      <ReactMarkdown components={markdownLinkComponents()} remarkPlugins={[remarkGfm]}>
        {normalizeModelMarkdown(content)}
      </ReactMarkdown>
    </div>
  );
}

function ResearchTimeline({
  run,
  now,
  onToggle,
  locale,
}: {
  run: ResearchRun;
  now: number;
  onToggle: () => void;
  locale: Locale;
}) {
  const t = uiCopy[locale];
  const elapsed = run.status === "running" ? now - run.startedAt : run.elapsedMs;
  const visibleSteps = run.steps.filter(isCleanResearchStep);
  const title =
    run.status === "running"
      ? t.traceRunning(formatElapsed(elapsed, locale))
      : run.status === "error"
        ? t.traceError
        : run.status === "cancelled"
          ? t.traceCancelled
          : t.traceComplete(formatElapsed(elapsed, locale));

  return (
    <section className={`research-trace is-${run.status}`}>
      <button aria-expanded={run.expanded} className="trace-toggle" onClick={onToggle} type="button">
        <span className="trace-orbit">
          <Icon name={run.status === "running" ? "spark" : run.status === "complete" ? "check" : "clock"} />
        </span>
        <strong>{title}</strong>
        <span className="trace-summary">
          {run.evidence.length > 0
            ? t.evidenceCount(run.evidence.length)
            : run.mode === "deep"
              ? t.deepResearch
              : t.fastAnswer}
        </span>
        <span className={`trace-chevron ${run.expanded ? "is-open" : ""}`}>
          <Icon name="chevron" />
        </span>
      </button>

      {run.expanded && (
        <div className="trace-steps" aria-live="polite">
          {visibleSteps.map((step) => (
            <div className={`trace-step is-${step.status}`} key={step.id}>
              <span className="trace-node">
                {step.status === "complete" ? <Icon name="check" /> : <span className="trace-pulse" />}
              </span>
              <div className="trace-step-copy">
                <div className="trace-step-title">
                  <strong>{step.title}</strong>
                  {step.tool && (
                    <span className="tool-chip">
                      <Icon name={step.tool.includes("reader") ? "book" : "search"} />
                      {step.tool}
                    </span>
                  )}
                </div>
                <TraceMarkdown
                  content={step.kind === "tool" ? cleanToolStepText(step, locale) : step.text}
                />
              </div>
            </div>
          ))}
          {run.status === "running" && visibleSteps.length === 0 && (
            <div className="trace-step is-running">
              <span className="trace-node">
                <span className="trace-pulse" />
              </span>
              <div className="trace-step-copy">
                <strong>{t.buildingPlan}</strong>
                <p>{t.buildingPlanDetail}</p>
              </div>
            </div>
          )}
          {isCleanTraceError(run.error) && <p className="trace-error">{run.error}</p>}
        </div>
      )}
    </section>
  );
}

function FastThinking({
  run,
  onToggle,
  locale,
}: {
  run: ResearchRun;
  onToggle: () => void;
  locale: Locale;
}) {
  const t = uiCopy[locale];
  const thinking = run.steps
    .filter((step) => step.kind === "thought" && isCleanResearchStep(step))
    .map((step) => step.text.trim())
    .filter(Boolean)
    .join("\n\n");
  const isWaiting = run.status === "running" && !thinking;
  if (!thinking && !isWaiting) return null;
  const contentId = `fast-thinking-content-${run.id}`;

  return (
    <section
      className={`fast-thinking${run.expanded ? " is-expanded" : ""}${isWaiting ? " is-waiting" : ""}`}
    >
      <button
        aria-controls={thinking ? contentId : undefined}
        aria-expanded={run.expanded}
        aria-label={run.expanded ? t.collapseThinking : t.expandThinking}
        className="fast-thinking-toggle"
        disabled={isWaiting}
        onClick={onToggle}
        type="button"
      >
        <span className="fast-thinking-mark">
          <Icon name="spark" />
        </span>
        <span className="fast-thinking-heading">
          <strong>{isWaiting ? t.fastThinkingRunning : t.fastThinking}</strong>
          <small>{t.fastThinkingSource}</small>
        </span>
        {isWaiting ? (
          <span className="fast-thinking-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        ) : (
          <span className={`fast-thinking-chevron${run.expanded ? " is-open" : ""}`}>
            <Icon name="chevron" />
          </span>
        )}
      </button>

      {thinking && (
        <div className="fast-thinking-body" id={contentId}>
          <div className="fast-thinking-copy">
            {run.expanded ? (
              <TraceMarkdown content={thinking} />
            ) : (
              <p className="fast-thinking-preview">{thinking}</p>
            )}
          </div>
          <button
            aria-controls={contentId}
            aria-expanded={run.expanded}
            className="fast-thinking-expand"
            onClick={onToggle}
            type="button"
          >
            {run.expanded ? t.collapseThinking : t.expandThinking}
            <Icon name="chevron" />
          </button>
        </div>
      )}
    </section>
  );
}

function ProductHeader({
  locale,
  view,
  onLocaleChange,
  onViewChange,
}: {
  locale: Locale;
  view: ProductView;
  onLocaleChange: (locale: Locale) => void;
  onViewChange: (view: ProductView) => void;
}) {
  const t = uiCopy[locale];

  return (
    <header className="product-header">
      <button className="product-brand" onClick={() => onViewChange("home")} type="button">
        <span className="brand-mark">
          <Icon name="spark" />
        </span>
        <span>
          <strong>Mint Agent</strong>
          <small>{t.brandTagline}</small>
        </span>
      </button>

      <nav className="product-nav" aria-label={locale === "zh" ? "产品导航" : "Product navigation"}>
        <button
          aria-current={view === "performance" ? "page" : undefined}
          className={view === "performance" ? "is-active" : ""}
          onClick={() => onViewChange("performance")}
          type="button"
        >
          <span>How Mint Perform</span>
          <small>{t.navPerformanceHint}</small>
        </button>
        <button
          aria-current={view === "workspace" ? "page" : undefined}
          className={view === "workspace" ? "is-active" : ""}
          onClick={() => onViewChange("workspace")}
          type="button"
        >
          <span>Try Mint Now</span>
          <small>{t.navWorkspaceHint}</small>
        </button>
      </nav>

      <div className="language-switch" aria-label={locale === "zh" ? "语言" : "Language"}>
        <Icon name="globe" />
        <button
          aria-label={t.switchLanguage}
          onClick={() => onLocaleChange(locale === "zh" ? "en" : "zh")}
          type="button"
        >
          <span className={locale === "zh" ? "is-active" : ""}>{t.chinese}</span>
          <i aria-hidden="true" />
          <span className={locale === "en" ? "is-active" : ""}>{t.english}</span>
        </button>
      </div>
    </header>
  );
}

function moveLandingShowcase(event: React.PointerEvent<HTMLElement>) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const x = ((event.clientX - bounds.left) / bounds.width) * 100;
  const y = ((event.clientY - bounds.top) / bounds.height) * 100;
  const horizontal = (x - 50) / 50;
  const vertical = (y - 50) / 50;

  event.currentTarget.style.setProperty("--pointer-x", `${x}%`);
  event.currentTarget.style.setProperty("--pointer-y", `${y}%`);
  event.currentTarget.style.setProperty("--case-tilt-x", `${(-vertical * 1.15).toFixed(2)}deg`);
  event.currentTarget.style.setProperty("--case-tilt-y", `${(horizontal * 1.45).toFixed(2)}deg`);
  event.currentTarget.style.setProperty("--case-shift-x", `${(horizontal * 7).toFixed(2)}px`);
  event.currentTarget.style.setProperty("--case-shift-y", `${(vertical * 6).toFixed(2)}px`);
  event.currentTarget.style.setProperty("--case-float-x", `${(-horizontal * 10).toFixed(2)}px`);
  event.currentTarget.style.setProperty("--case-float-y", `${(-vertical * 8).toFixed(2)}px`);
}

function resetLandingShowcase(event: React.PointerEvent<HTMLElement>) {
  event.currentTarget.style.setProperty("--pointer-x", "50%");
  event.currentTarget.style.setProperty("--pointer-y", "50%");
  event.currentTarget.style.setProperty("--case-tilt-x", "0deg");
  event.currentTarget.style.setProperty("--case-tilt-y", "0deg");
  event.currentTarget.style.setProperty("--case-shift-x", "0px");
  event.currentTarget.style.setProperty("--case-shift-y", "0px");
  event.currentTarget.style.setProperty("--case-float-x", "0px");
  event.currentTarget.style.setProperty("--case-float-y", "0px");
}

function BenchmarkScoreCard({ benchmark, locale }: { benchmark: BenchmarkChart; locale: Locale }) {
  const t = uiCopy[locale];
  const sortedScores = [...benchmark.scores].sort((a, b) => b.score - a.score);
  const mintScores = sortedScores.filter((item) => item.mint);
  const percentages = sortedScores.map((item) => item.score * 100);
  const axisMin = Math.max(0, Math.floor(Math.min(...percentages) - 3));
  const axisMax = Math.min(Math.ceil(Math.max(...percentages) + 3), 100);
  const axisRange = Math.max(axisMax - axisMin, 1);
  const axisTicks = Array.from({ length: 5 }, (_, index) => axisMax - (axisRange * index) / 4);
  const formatAxisTick = (value: number) => Math.round(value).toString();

  return (
    <article className="benchmark-card">
      <header>
        <div>
          <h3>{benchmark.name}</h3>
          <p>{benchmark.subtitle[locale]}</p>
        </div>
        <span>{t.modelsEvaluated(sortedScores.length)}</span>
      </header>

      <div className="mint-score-summary">
        {mintScores.map((item) => (
          <span key={item.model}>
            <i aria-hidden="true" />
            <strong>{item.model}</strong>
            <b>{(item.score * 100).toFixed(1)}%</b>
            <small>#{sortedScores.findIndex((candidate) => candidate.model === item.model) + 1}</small>
          </span>
        ))}
        {!benchmark.scores.some((item) => item.model === "Mint-Ag") && (
          <span className="is-missing">
            <i aria-hidden="true" />
            <strong>Mint-Ag</strong>
            <b>—</b>
            <small>{t.notEvaluated}</small>
          </span>
        )}
      </div>

      <div
        aria-label={`${benchmark.name}: ${t.scrollHint}`}
        className="score-chart-viewport"
        role="region"
        tabIndex={0}
      >
        <div className="score-chart" style={{ width: `${Math.max(100, sortedScores.length * 52)}px` }}>
          <div className="score-gridlines" aria-hidden="true">
            {axisTicks.map((tick) => (
              <span key={tick}>{formatAxisTick(tick)}</span>
            ))}
          </div>
          <div className="score-bars">
            {sortedScores.map((item, index) => {
              const logo = providerLogos[item.provider];
              const scaledHeight = Math.max(
                1.2,
                Math.min(100, ((item.score * 100 - axisMin) / axisRange) * 100),
              );
              return (
                <div
                  aria-label={`${index + 1}. ${item.model}: ${(item.score * 100).toFixed(1)}%`}
                  className={`score-bar-item tone-${item.provider}${item.mint ? " is-mint" : ""}`}
                  key={item.model}
                  title={`${item.model}: ${(item.score * 100).toFixed(1)}%`}
                >
                  <div className="score-bar-track">
                    <span style={{ height: `${scaledHeight}%` }}>
                      {logo && (
                        <i
                          aria-hidden="true"
                          className="score-model-logo"
                          style={{ backgroundImage: `url("${logo}")` }}
                        />
                      )}
                      <span className="score-bar-marker">
                        <strong>{(item.score * 100).toFixed(1)}</strong>
                      </span>
                    </span>
                  </div>
                  <span className="score-model-label">{item.model}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </article>
  );
}

function LandingPage({
  locale,
  onPerformance,
  onTryMint,
}: {
  locale: Locale;
  onPerformance: () => void;
  onTryMint: () => void;
}) {
  const t = uiCopy[locale];
  const [activeCaseIndex, setActiveCaseIndex] = useState(0);
  const activeCase = landingCaseStudies[activeCaseIndex];
  const chartMaximum = Math.max(...activeCase.chart.map((point) => point.value), 1);

  function shiftCase(direction: number) {
    setActiveCaseIndex((current) => (current + direction + landingCaseStudies.length) % landingCaseStudies.length);
  }

  return (
    <main className="landing-page" id="main-content">
      <section className="landing-hero">
        <div className="landing-hero-copy">
          <p className="eyebrow">{t.landingEyebrow}</p>
          <h1>
            <span>{t.landingTitleFirst}</span>
            <span>{t.landingTitleSecond}</span>
          </h1>
          <p className="landing-hero-intro">{t.landingIntro}</p>
          <div className="landing-actions">
            <button className="is-primary" onClick={onTryMint} type="button">
              {t.heroPrimaryAction}
              <Icon name="arrow" />
            </button>
            <button className="is-secondary" onClick={onPerformance} type="button">
              {t.heroSecondaryAction}
              <Icon name="arrow" />
            </button>
          </div>
          <div className="landing-signals">
            {t.heroSignals.map((signal) => (
              <span key={signal}>
                <Icon name="check" />
                {signal}
              </span>
            ))}
          </div>
        </div>

        <div
          aria-label={t.caseShowcaseAria}
          className="landing-visual-frame landing-case-showcase"
          onPointerLeave={resetLandingShowcase}
          onPointerMove={moveLandingShowcase}
        >
          <span className="landing-case-ambient" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>

          <div aria-label={t.caseShowcaseAria} className="landing-case-tabs" role="tablist">
            {landingCaseStudies.map((caseStudy, index) => (
              <button
                aria-controls="landing-case-panel"
                aria-selected={activeCaseIndex === index}
                className={activeCaseIndex === index ? "is-active" : ""}
                id={`landing-case-tab-${caseStudy.id}`}
                key={caseStudy.id}
                onClick={() => setActiveCaseIndex(index)}
                onMouseEnter={() => setActiveCaseIndex(index)}
                role="tab"
                tabIndex={activeCaseIndex === index ? 0 : -1}
                type="button"
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                {caseStudy.tab[locale]}
              </button>
            ))}
          </div>

          <article
            aria-labelledby={`landing-case-tab-${activeCase.id}`}
            aria-live="polite"
            className="landing-case-window"
            id="landing-case-panel"
            key={`${activeCase.id}-${locale}`}
            role="tabpanel"
          >
            <header className="landing-case-window-header">
              <div>
                <span className="landing-agent-mark">
                  <Icon name="spark" />
                </span>
                <span>
                  <strong>Mint Finance Agent</strong>
                  <small>{t.caseArchive}</small>
                </span>
              </div>
              <span className="landing-case-complete">
                <Icon name="check" />
                {t.caseCompleted}
              </span>
            </header>

            <div className="landing-case-window-body">
              <aside className="landing-case-trajectory">
                <small>{t.caseResearchPath}</small>
                <ol>
                  {t.pipeline.map((step, index) => (
                    <li key={step}>
                      <span>
                        <Icon name="check" />
                      </span>
                      <div>
                        <strong>{step}</strong>
                        <small>
                          {index === 0
                            ? activeCase.eyebrow[locale]
                            : index === 1
                              ? `${activeCase.toolCalls} ${locale === "zh" ? "次查询" : "queries"}`
                              : index === 2
                                ? `${activeCase.sourceCount} ${locale === "zh" ? "条来源" : "sources"}`
                                : index === 3
                                  ? t.caseVerified
                                  : t.audited}
                        </small>
                      </div>
                    </li>
                  ))}
                </ol>
              </aside>

              <section className="landing-case-analysis">
                <p>{activeCase.eyebrow[locale]}</p>
                <h2>{activeCase.question[locale]}</h2>
                <div className="landing-case-metrics">
                  <span>
                    <small>{activeCase.metricLabel[locale]}</small>
                    <strong>{activeCase.metric}</strong>
                  </span>
                  <span>
                    <small>{locale === "zh" ? "变化 / 口径" : "Change / basis"}</small>
                    <strong>{activeCase.delta}</strong>
                  </span>
                </div>
                <div className="landing-case-chart" aria-label={activeCase.metricLabel[locale]}>
                  <header>
                    <span>{locale === "zh" ? "历史对比" : "Historical comparison"}</span>
                    <small>{locale === "zh" ? "统一口径" : "Comparable basis"}</small>
                  </header>
                  <div>
                    {activeCase.chart.map((point, index) => (
                      <span key={point.label}>
                        <i style={{ height: `${Math.max(24, (point.value / chartMaximum) * 100)}%` }}>
                          <b>{point.value.toFixed(point.value < 10 ? 3 : 1)}</b>
                        </i>
                        <small>{point.label}</small>
                        {index === activeCase.chart.length - 1 && <em>{t.audited}</em>}
                      </span>
                    ))}
                  </div>
                </div>
              </section>

              <aside className="landing-case-evidence">
                <header>
                  <span>
                    <Icon name="file" />
                  </span>
                  <div>
                    <small>{t.caseEvidence}</small>
                    <strong>{t.caseSelectedEvidence}</strong>
                  </div>
                </header>
                <div>
                  {activeCase.evidence.map((item, index) => (
                    <a href={item.url} key={`${activeCase.id}-${item.source}`} rel="noreferrer" target="_blank">
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <div>
                        <strong>{item.source}</strong>
                        <p>{item.detail[locale]}</p>
                        <small>
                          {item.status[locale]}
                          <Icon name="arrow" />
                        </small>
                      </div>
                    </a>
                  ))}
                </div>
                <footer>
                  <span>{t.casePrimarySources}</span>
                  <strong>{activeCase.sourceCount}</strong>
                </footer>
              </aside>
            </div>
          </article>

          <aside className="landing-case-finding" key={`finding-${activeCase.id}-${locale}`}>
            <header>
              <span>{t.caseKeyFinding}</span>
              <small>{activeCase.eyebrow[locale]}</small>
            </header>
            <p>{activeCase.finding[locale]}</p>
            <footer>
              <span>{activeCase.sourceCount} {locale === "zh" ? "条来源" : "sources"}</span>
              <span>{activeCase.toolCalls} {locale === "zh" ? "次工具调用" : "tool calls"}</span>
              <strong>
                <Icon name="check" />
                {t.caseVerified}
              </strong>
            </footer>
          </aside>

          <aside className="landing-case-proof" key={`proof-${activeCase.id}`}>
            <span>
              <Icon name="file" />
            </span>
            <div>
              <small>{activeCase.metricLabel[locale]}</small>
              <strong>{activeCase.metric}</strong>
              <em>{activeCase.delta}</em>
            </div>
          </aside>

          <div className="landing-case-controls">
            <button aria-label={t.casePrevious} onClick={() => shiftCase(-1)} type="button">
              <Icon name="arrow" />
            </button>
            <span>
              <strong>{String(activeCaseIndex + 1).padStart(2, "0")}</strong>
              <i />
              {String(landingCaseStudies.length).padStart(2, "0")}
            </span>
            <button aria-label={t.caseNext} onClick={() => shiftCase(1)} type="button">
              <Icon name="arrow" />
            </button>
          </div>
        </div>
      </section>

      <section className="landing-agent-story">
        <div>
          <p className="eyebrow">{t.methodologyEyebrow}</p>
          <h2>{t.landingAgentTitle}</h2>
          <p>{t.landingAgentDescription}</p>
        </div>
        <ol>
          {t.pipeline.map((step, index) => (
            <li key={step}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{step}</strong>
            </li>
          ))}
        </ol>
      </section>

      <section className="landing-pathways">
        <header>
          <p className="eyebrow">{t.landingPathwaysEyebrow}</p>
          <h2>{t.landingPathwaysTitle}</h2>
        </header>
        <div>
          <button className="landing-pathway-card is-performance" onClick={onPerformance} type="button">
            <span className="pathway-index">01</span>
            <span className="pathway-kicker">How Mint Perform</span>
            <span className="pathway-title">{t.scoreTitle}</span>
            <span className="pathway-description">{t.landingPerformanceDescription}</span>
            <span className="landing-benchmark-preview" aria-hidden="true">
              {[46, 63, 56, 82, 68, 91, 76].map((height, index) => (
                <i className={index === 5 ? "is-mint" : ""} key={height} style={{ height: `${height}%` }} />
              ))}
            </span>
            <span className="pathway-action">
              {t.heroSecondaryAction}
              <Icon name="arrow" />
            </span>
          </button>

          <button className="landing-pathway-card is-workspace" onClick={onTryMint} type="button">
            <span className="pathway-index">02</span>
            <span className="pathway-kicker">Try Mint Now</span>
            <span className="pathway-title">{t.ctaTitle}</span>
            <span className="pathway-description">{t.landingWorkspaceDescription}</span>
            <span className="landing-agent-preview" aria-hidden="true">
              <span>
                <i />
                <b>{t.pipeline[0]}</b>
                <small>Complete</small>
              </span>
              <span>
                <i />
                <b>{t.pipeline[1]}</b>
                <small>6 queries</small>
              </span>
              <span>
                <i />
                <b>{t.pipeline[3]}</b>
                <small>{t.agentRunMeta[0]}</small>
              </span>
            </span>
            <span className="pathway-action">
              {t.heroPrimaryAction}
              <Icon name="arrow" />
            </span>
          </button>
        </div>
      </section>
    </main>
  );
}

function PerformancePage({ locale, onTryMint }: { locale: Locale; onTryMint: () => void }) {
  const t = uiCopy[locale];
  const singleTurnBenchmarks = benchmarkCharts.slice(0, 3);
  const financeAgentBenchmarks = benchmarkCharts.slice(3, 5);
  const finSearchBenchmarks = benchmarkCharts.slice(5, 7);

  return (
    <main className="performance-page" id="main-content">
      <section className="performance-hero">
        <div className="performance-intro">
          <p className="eyebrow">{t.performanceEyebrow}</p>
          <h1>{t.performanceTitle}</h1>
          <p>{t.performanceIntro}</p>
          <span className="live-harness">
            <i aria-hidden="true" />
            {t.liveHarness}
          </span>
        </div>
        <div className="performance-models">
          <div className="section-label">
            <span>{t.modelsTitle}</span>
            <Icon name="arrow" />
          </div>
          <article>
            <div>
              <strong>Mint-Cu</strong>
              <span>9B</span>
            </div>
            <p>{t.cuDescription}</p>
            <small>
              <i aria-hidden="true" />
              {t.live}
            </small>
          </article>
          <article>
            <div>
              <strong>Mint-Ag</strong>
              <span>27B</span>
            </div>
            <p>{t.agDescription}</p>
            <small>
              <i aria-hidden="true" />
              {t.live}
            </small>
          </article>
        </div>
      </section>

      <section className="score-dashboard">
        <div className="score-dashboard-heading">
          <div>
            <p className="eyebrow">{t.scoreEyebrow}</p>
            <h2>
              <span aria-hidden="true" />
              {t.scoreTitle}
            </h2>
            <p>{t.scoreDescription}</p>
          </div>
          <div className="score-legend">
            <span className="is-mint">
              <i aria-hidden="true" />
              {t.scoreLegendMint}
            </span>
            <span>
              <i aria-hidden="true" />
              {t.scoreLegendOther}
            </span>
            <small>{t.scoreSource}</small>
          </div>
        </div>

        <div className="benchmark-group">
          <div className="benchmark-group-label">
            <span>01</span>
            <strong>{t.financialExpertiseKnowledge}</strong>
            <i aria-hidden="true" />
          </div>
          <div className="benchmark-row is-three">
            {singleTurnBenchmarks.map((benchmark) => (
              <BenchmarkScoreCard benchmark={benchmark} key={benchmark.id} locale={locale} />
            ))}
          </div>
        </div>

        <div className="benchmark-group">
          <div className="benchmark-group-label">
            <span>02</span>
            <strong>{t.financialDeepResearch}</strong>
            <i aria-hidden="true" />
          </div>
          <div className="benchmark-row is-two">
            {financeAgentBenchmarks.map((benchmark) => (
              <BenchmarkScoreCard benchmark={benchmark} key={benchmark.id} locale={locale} />
            ))}
          </div>
        </div>

        <div className="benchmark-group">
          <div className="benchmark-group-label">
            <span>03</span>
            <strong>{t.financialDeepResearch}</strong>
            <i aria-hidden="true" />
          </div>
          <div className="benchmark-row is-two">
            {finSearchBenchmarks.map((benchmark) => (
              <BenchmarkScoreCard benchmark={benchmark} key={benchmark.id} locale={locale} />
            ))}
          </div>
        </div>
      </section>

      <section className="methodology-panel">
        <div>
          <p className="eyebrow">{t.methodologyEyebrow}</p>
          <h2>{t.methodologyTitle}</h2>
          <p>{t.methodologyDescription}</p>
        </div>
        <div className="research-pipeline" aria-label={t.methodologyTitle}>
          {t.pipeline.map((item, index) => (
            <React.Fragment key={item}>
              <span>
                <i>{index + 1}</i>
                {item}
              </span>
              {index < t.pipeline.length - 1 && <Icon name="arrow" />}
            </React.Fragment>
          ))}
        </div>
      </section>

      <section className="performance-section">
        <div className="performance-section-heading">
          <span aria-hidden="true" />
          <h2>{t.dimensionsTitle}</h2>
        </div>
        <div className="dimension-grid">
          {t.dimensions.map(([title, description], index) => (
            <article key={title}>
              <span className="dimension-number">0{index + 1}</span>
              <div>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
              <small>
                <Icon name="check" />
                {t.audited}
              </small>
            </article>
          ))}
        </div>
      </section>

      <section className="performance-cta">
        <div>
          <p className="eyebrow">{t.ctaEyebrow}</p>
          <h2>{t.ctaTitle}</h2>
          <p>{t.ctaDescription}</p>
        </div>
        <button onClick={onTryMint} type="button">
          {t.ctaButton}
          <Icon name="arrow" />
        </button>
      </section>
    </main>
  );
}

function LoginGate({
  locale,
  notice,
  onSuccess,
}: {
  locale: Locale;
  notice: string | null;
  onSuccess: (session: { token: string; username: string }) => void;
}) {
  const t = uiCopy[locale];
  const [username, setUsername] = useState("tl-finagent");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    const name = username.trim().toLowerCase();
    const pass = password.trim();
    if (!name || !pass) {
      setError(t.authFailed);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: name, password: pass }),
      });
      if (response.status === 429) {
        setError(t.authRateLimited);
        return;
      }
      if (!response.ok) {
        setError(t.authFailed);
        return;
      }
      const payload = (await response.json()) as { token?: string; username?: string };
      if (!payload.token || !payload.username) {
        setError(t.authNetworkError);
        return;
      }
      onSuccess({ token: payload.token, username: payload.username });
    } catch {
      setError(t.authNetworkError);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-gate" id="main-content">
      <form className="login-card" onSubmit={submit}>
        <span className="login-eyebrow">MINT · FINANCIAL RESEARCH AGENT</span>
        <h1>{t.authTitle}</h1>
        <p className="login-subtitle">{t.authSubtitle}</p>
        {notice ? <p className="login-notice">{notice}</p> : null}
        <label className="login-field">
          <span>{t.authUsername}</span>
          <input
            autoCapitalize="none"
            autoComplete="username"
            autoCorrect="off"
            name="username"
            onChange={(event) => setUsername(event.target.value)}
            placeholder="tl-finagent"
            spellCheck={false}
            value={username}
          />
        </label>
        <label className="login-field">
          <span>{t.authPassword}</span>
          <input
            autoComplete="current-password"
            name="password"
            onChange={(event) => setPassword(event.target.value)}
            type="password"
            value={password}
          />
        </label>
        {error ? (
          <p className="login-error" role="alert">
            {error}
          </p>
        ) : null}
        <button className="login-submit" disabled={submitting} type="submit">
          {submitting ? t.authSubmitting : t.authSubmit}
        </button>
        <p className="login-footnote">
          {t.authNoAccount}{" "}
          <a href={PROJECT_SITE_URL} rel="noreferrer" target="_blank">
            {t.authGetCode}
          </a>
        </p>
      </form>
    </main>
  );
}

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([initialConversation]);
  const [activeConversationId, setActiveConversationId] = useState(initialConversation.id);
  const [locale, setLocale] = useState<Locale>("zh");
  const [productView, setProductView] = useState<ProductView>("home");
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<"deep" | "fast">("deep");
  const [model, setModel] = useState<ModelId>("mint-cu");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidenceMessageId, setEvidenceMessageId] = useState<string | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [hoveredEvidenceId, setHoveredEvidenceId] = useState<string | null>(null);
  const [copiedEvidenceId, setCopiedEvidenceId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [evidenceDrawerWidth, setEvidenceDrawerWidth] = useState(DEFAULT_EVIDENCE_DRAWER_WIDTH);
  const [hydrated, setHydrated] = useState(false);
  const [now, setNow] = useState(() => timestamp());
  const [session, setSession] = useState<{ token: string; username: string } | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authNoticeKey, setAuthNoticeKey] = useState<"expired" | "required" | null>(null);
  const [accountOpen, setAccountOpen] = useState(false);
  const [accountQuota, setAccountQuota] = useState<
    { used: number; limit: number; remaining: number } | "loading" | "error" | null
  >(null);
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const lastSyncedRef = useRef<Map<string, number>>(new Map());
  const historyLoadedRef = useRef(false);
  const evidenceResizingRef = useRef(false);
  const conversationCanvasRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const t = uiCopy[locale];

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0];
  const activeIsRunning = conversationHasActiveRun(activeConversation);
  const activeConversationActivityKey = conversationActivityKey(activeConversation);
  const anyConversationRunning = conversations.some(conversationHasActiveRun);
  const evidenceMessage = useMemo(
    () => activeConversation?.messages.find((message) => message.id === evidenceMessageId) ?? null,
    [activeConversation, evidenceMessageId],
  );
  const activeEvidence = evidenceMessage?.run?.evidence ?? [];
  const evidenceQuestion = useMemo(() => {
    if (!activeConversation || !evidenceMessageId) return "";
    const answerIndex = activeConversation.messages.findIndex(
      (message) => message.id === evidenceMessageId,
    );
    for (let index = answerIndex - 1; index >= 0; index -= 1) {
      const message = activeConversation.messages[index];
      if (message.role === "user") return message.content;
    }
    return "";
  }, [activeConversation, evidenceMessageId]);
  const selectedEvidence =
    activeEvidence.find((item) => item.id === selectedEvidenceId) ?? activeEvidence[0] ?? null;
  const selectedEvidenceView = useMemo(
    () => (selectedEvidence ? buildEvidencePresentation(selectedEvidence, locale) : null),
    [locale, selectedEvidence],
  );
  const evidenceGraph = useMemo(() => buildEvidenceGraph(activeEvidence), [activeEvidence]);
  const focusedEvidenceId = hoveredEvidenceId;
  const relatedEvidenceIds = useMemo(() => {
    const related = new Set<string>();
    if (!focusedEvidenceId) return related;
    related.add(focusedEvidenceId);
    for (const edge of evidenceGraph.edges) {
      if (edge.from === focusedEvidenceId && edge.to !== "root") related.add(edge.to);
      if (edge.to === focusedEvidenceId && edge.from !== "root") related.add(edge.from);
    }
    return related;
  }, [evidenceGraph, focusedEvidenceId]);

  useEffect(() => {
    let restored: Conversation[] | null = null;
    let restoredLocale: Locale | null = null;
    let restoredEvidenceWidth: number | null = null;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Conversation[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          restored = parsed.slice(0, 30).map((conversation) => ({
            ...conversation,
            messages: conversation.messages.map((message) => ({
              ...message,
              run: message.run?.status === "running" ? { ...message.run, status: "cancelled" as const } : message.run,
            })),
          }));
        }
      }
      const storedLocale = localStorage.getItem(LOCALE_STORAGE_KEY);
      if (storedLocale === "zh" || storedLocale === "en") restoredLocale = storedLocale;
      const storedEvidenceWidth = Number(localStorage.getItem(EVIDENCE_WIDTH_STORAGE_KEY));
      if (Number.isFinite(storedEvidenceWidth) && storedEvidenceWidth > 0) {
        restoredEvidenceWidth = clampEvidenceDrawerWidth(storedEvidenceWidth, window.innerWidth);
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }

    const timer = window.setTimeout(() => {
      if (restored) {
        setConversations(restored);
        setActiveConversationId(restored[0].id);
      }
      if (restoredLocale) setLocale(restoredLocale);
      if (restoredEvidenceWidth) setEvidenceDrawerWidth(restoredEvidenceWidth);
      setHydrated(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.slice(0, 30)));
  }, [conversations, hydrated]);

  useEffect(() => {
    const stored = readStoredSession();
    if (!stored) {
      setAuthChecked(true);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const response = await sessionFetch(stored.token, "/api/auth/me");
        if (cancelled) return;
        if (response.ok) {
          const payload = (await response.json()) as { username?: string };
          setSession({ token: stored.token, username: payload.username || stored.username });
        } else if (response.status === 401) {
          localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(DEFAULT_LOCAL_SESSION));
          setSession(DEFAULT_LOCAL_SESSION);
          setAuthNoticeKey(null);
        } else {
          // Transient server trouble: keep the stored session; the research
          // API will reject it later if it is really invalid.
          setSession(stored);
        }
      } catch {
        setSession(stored);
      } finally {
        if (!cancelled) setAuthChecked(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!session || !hydrated || historyLoadedRef.current) return;
    historyLoadedRef.current = true;
    let cancelled = false;
    (async () => {
      try {
        const response = await sessionFetch(session.token, "/api/history");
        if (cancelled || !response.ok) return;
        const payload = (await response.json()) as {
          conversations?: { id: string; updatedAt: number; payload: Conversation }[];
        };
        const server = (payload.conversations ?? [])
          .map((item) => item.payload)
          .filter(
            (conversation): conversation is Conversation =>
              Boolean(conversation) &&
              typeof conversation.id === "string" &&
              Array.isArray(conversation.messages),
          )
          .map((conversation) => ({
            ...conversation,
            messages: conversation.messages.map((message) => ({
              ...message,
              run:
                message.run?.status === "running"
                  ? { ...message.run, status: "cancelled" as const }
                  : message.run,
            })),
          }));
        if (cancelled || server.length === 0) return;
        for (const conversation of server) {
          lastSyncedRef.current.set(conversation.id, conversation.updatedAt);
        }
        setConversations((local) => {
          const merged = new Map<string, Conversation>();
          for (const conversation of server) merged.set(conversation.id, conversation);
          for (const conversation of local) {
            if (conversation.messages.length === 0) continue;
            const existing = merged.get(conversation.id);
            if (!existing || conversation.updatedAt > existing.updatedAt) {
              merged.set(conversation.id, conversation);
            }
          }
          const list = [...merged.values()].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 100);
          if (list.length === 0) return local;
          setActiveConversationId((activeId) =>
            list.some((conversation) => conversation.id === activeId) ? activeId : list[0].id,
          );
          return list;
        });
      } catch {
        /* offline: local copy remains usable */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session, hydrated]);

  useEffect(() => {
    if (!session || !hydrated) return;
    const token = session.token;
    const timer = window.setTimeout(() => {
      for (const conversation of conversations) {
        if (conversation.messages.length === 0) continue;
        if (conversationHasActiveRun(conversation)) continue;
        const lastSynced = lastSyncedRef.current.get(conversation.id) ?? 0;
        if (conversation.updatedAt <= lastSynced) continue;
        lastSyncedRef.current.set(conversation.id, conversation.updatedAt);
        void sessionFetch(token, `/api/history/${encodeURIComponent(conversation.id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: conversation.title,
            updatedAt: conversation.updatedAt,
            payload: conversation,
          }),
        })
          .then((response) => {
            if (!response.ok) lastSyncedRef.current.delete(conversation.id);
          })
          .catch(() => {
            lastSyncedRef.current.delete(conversation.id);
          });
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [conversations, session, hydrated]);

  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
    if (hydrated) localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  }, [hydrated, locale]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(EVIDENCE_WIDTH_STORAGE_KEY, String(evidenceDrawerWidth));
  }, [evidenceDrawerWidth, hydrated]);

  useEffect(() => {
    const keepEvidenceDrawerInViewport = () => {
      setEvidenceDrawerWidth((width) => clampEvidenceDrawerWidth(width, window.innerWidth));
    };
    keepEvidenceDrawerInViewport();
    window.addEventListener("resize", keepEvidenceDrawerInViewport);
    return () => {
      evidenceResizingRef.current = false;
      document.body.classList.remove("is-resizing-evidence");
      window.removeEventListener("resize", keepEvidenceDrawerInViewport);
    };
  }, []);

  useEffect(() => {
    const syncViewFromUrl = () => {
      const requestedView = new URLSearchParams(window.location.search).get("view");
      if (requestedView === "performance") {
        setProductView("performance");
      } else if (requestedView === "workspace" || requestedView === "workbench") {
        setProductView("workspace");
      } else {
        setProductView("home");
      }
    };
    syncViewFromUrl();
    window.addEventListener("popstate", syncViewFromUrl);
    return () => window.removeEventListener("popstate", syncViewFromUrl);
  }, []);

  useEffect(() => {
    if (!anyConversationRunning) return;
    const timer = window.setInterval(() => setNow(timestamp()), 1000);
    return () => window.clearInterval(timer);
  }, [anyConversationRunning]);

  useEffect(() => {
    if (productView !== "workspace") return;
    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        const canvas = conversationCanvasRef.current;
        if (!canvas) return;
        if (activeIsRunning) {
          canvas.scrollTop = canvas.scrollHeight;
          return;
        }
        canvas.scrollTo({ top: canvas.scrollHeight, behavior: "smooth" });
      });
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      if (secondFrame) window.cancelAnimationFrame(secondFrame);
    };
  }, [activeConversationActivityKey, activeConversationId, activeIsRunning, productView]);

  function updateConversation(id: string, updater: (conversation: Conversation) => Conversation) {
    setConversations((items) => items.map((item) => (item.id === id ? updater(item) : item)));
  }

  function patchAssistantMessage(
    conversationId: string,
    messageId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) {
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      updatedAt: timestamp(),
      messages: conversation.messages.map((message) => (message.id === messageId ? updater(message) : message)),
    }));
  }

  function handleLoginSuccess(next: { token: string; username: string }) {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
    setAuthNoticeKey(null);
    historyLoadedRef.current = false;
    lastSyncedRef.current.clear();
    setSession(next);
  }

  function resetWorkspace() {
    const replacement: Conversation = {
      ...initialConversation,
      id: uid("conversation"),
      title: t.newResearch,
      updatedAt: timestamp(),
    };
    setConversations([replacement]);
    setActiveConversationId(replacement.id);
    setPrompt("");
    setEvidenceOpen(false);
    setEvidenceMessageId(null);
  }

  function clearClientSession(noticeKey: "expired" | "required" | null) {
    localStorage.removeItem(SESSION_STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY);
    historyLoadedRef.current = false;
    lastSyncedRef.current.clear();
    setSession(null);
    setAuthNoticeKey(noticeKey);
    setAccountOpen(false);
    setAccountQuota(null);
    resetWorkspace();
  }

  function toggleAccountPanel() {
    const next = !accountOpen;
    setAccountOpen(next);
    if (!next || !session) return;
    setAccountQuota("loading");
    void sessionFetch(session.token, "/api/auth/me")
      .then(async (response) => {
        if (response.status === 401) {
          clearClientSession("expired");
          return;
        }
        if (!response.ok) throw new Error(String(response.status));
        const payload = (await response.json()) as {
          quota?: { used: number; limit: number; remaining: number };
        };
        setAccountQuota(
          payload.quota && typeof payload.quota.used === "number" ? payload.quota : "error",
        );
      })
      .catch(() => setAccountQuota("error"));
  }

  function handleLogout() {
    const token = session?.token;
    if (token) {
      void fetch("/api/auth/logout", {
        method: "POST",
        headers: { "X-Mint-Session": token },
        keepalive: true,
      }).catch(() => undefined);
    }
    clearClientSession(null);
  }

  function newConversation() {
    const conversation: Conversation = {
      id: uid("conversation"),
      title: t.newResearch,
      updatedAt: timestamp(),
      messages: [],
    };
    setConversations((items) => [conversation, ...items]);
    setActiveConversationId(conversation.id);
    setPrompt("");
    setSidebarOpen(false);
    setEvidenceOpen(false);
    setEvidenceMessageId(null);
    window.setTimeout(() => composerRef.current?.focus(), 60);
  }

  function deleteConversation(id: string) {
    const conversation = conversations.find((item) => item.id === id);
    if (!conversation || !window.confirm(t.deleteConfirm(conversation.title))) return;

    stopResearch(id);
    lastSyncedRef.current.delete(id);
    if (session) {
      void sessionFetch(session.token, `/api/history/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }).catch(() => undefined);
    }

    const remaining = conversations.filter((item) => item.id !== id);
    if (remaining.length > 0) {
      setConversations(remaining);
      if (id === activeConversationId) {
        setActiveConversationId(remaining[0].id);
        setPrompt("");
        setEvidenceOpen(false);
        setEvidenceMessageId(null);
      }
      return;
    }

    const replacement: Conversation = {
      ...initialConversation,
      id: uid("conversation"),
      title: t.newResearch,
      updatedAt: timestamp(),
    };
    setConversations([replacement]);
    setActiveConversationId(replacement.id);
    setPrompt("");
    setEvidenceOpen(false);
    setEvidenceMessageId(null);
  }

  function renameConversation(id: string) {
    const conversation = conversations.find((item) => item.id === id);
    if (!conversation) return;
    const input = window.prompt(t.renamePrompt, conversation.title);
    if (input === null) return;
    const title = input.trim().slice(0, 80);
    if (!title || title === conversation.title) return;
    setConversations((items) =>
      items.map((item) => (item.id === id ? { ...item, title, updatedAt: timestamp() } : item)),
    );
  }

  function selectConversation(id: string) {
    setActiveConversationId(id);
    setSidebarOpen(false);
    setEvidenceOpen(false);
    setEvidenceMessageId(null);
  }

  function handleStreamEvent(conversationId: string, messageId: string, event: StreamEvent) {
    patchAssistantMessage(conversationId, messageId, (message) => {
      const run = message.run;
      if (!run) return message;

      if (event.type === "run_started") {
        return {
          ...message,
          run: {
            ...run,
            startedAt: event.startedAt,
            mode: event.mode,
            model: event.model,
            status: "running",
          },
        };
      }
      if (event.type === "step") {
        return {
          ...message,
          run: { ...run, steps: upsertById(run.steps, event.step) },
        };
      }
      if (event.type === "evidence") {
        return {
          ...message,
          run: { ...run, evidence: upsertById(run.evidence, event.evidence) },
        };
      }
      if (event.type === "answer_delta") {
        return { ...message, content: `${message.content}${event.delta}` };
      }
      if (event.type === "done") {
        return {
          ...message,
          run: { ...run, status: "complete", elapsedMs: event.elapsedMs },
        };
      }
      if (event.type === "error") {
        return {
          ...message,
          content: message.content || t.incompleteAnswer,
          run: { ...run, status: "error", elapsedMs: timestamp() - run.startedAt, error: event.message },
        };
      }
      return message;
    });
  }

  async function askAgent(questionOverride?: string) {
    const question = (questionOverride ?? prompt).trim();
    if (!question || activeIsRunning || !activeConversation) return;

    const conversationId = activeConversation.id;
    const assistantMessageId = uid("assistant");
    const startedAt = timestamp();
    const userMessage: ChatMessage = {
      id: uid("user"),
      role: "user",
      content: question,
      createdAt: startedAt,
    };
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: startedAt,
      run: {
        id: uid("run"),
        status: "running",
        mode,
        model,
        startedAt,
        elapsedMs: 0,
        steps: [],
        evidence: [],
        expanded: mode === "deep",
      },
    };
    const history = activeConversation.messages
      .filter((message) => message.content.trim())
      .slice(-8)
      .map((message) => ({ role: message.role, content: message.content }));

    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      title: conversation.messages.length === 0 ? conversationTitle(question) : conversation.title,
      updatedAt: startedAt,
      messages: [...conversation.messages, userMessage, assistantMessage],
    }));
    setConversations((items) => {
      const selected = items.find((item) => item.id === conversationId);
      if (!selected) return items;
      return [selected, ...items.filter((item) => item.id !== conversationId)];
    });
    setPrompt("");
    setNow(startedAt);

    const controller = new AbortController();
    abortControllersRef.current.set(conversationId, controller);

    try {
      const response = await sessionFetch(session?.token ?? null, "/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history,
          mode,
          model,
          runId: assistantMessage.run!.id,
          browserTime: browserTimeContext(),
        }),
        signal: controller.signal,
      });

      if (response.status === 401) {
        clearClientSession("expired");
        throw new Error(t.authExpired);
      }
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { error?: string; message?: string }
          | null;
        throw new Error(payload?.message || payload?.error || t.serviceStatus(response.status));
      }
      if (!response.body) throw new Error(t.noStream);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          handleStreamEvent(conversationId, assistantMessageId, JSON.parse(line) as StreamEvent);
        }
      }

      if (buffer.trim()) {
        handleStreamEvent(conversationId, assistantMessageId, JSON.parse(buffer) as StreamEvent);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        patchAssistantMessage(conversationId, assistantMessageId, (message) => ({
          ...message,
          content: message.content || t.stoppedAnswer,
          run: message.run
            ? {
                ...message.run,
                status: "cancelled",
                elapsedMs: timestamp() - message.run.startedAt,
              }
            : message.run,
        }));
      } else {
        const message = error instanceof Error ? error.message : t.connectionFailed;
        handleStreamEvent(conversationId, assistantMessageId, { type: "error", message });
      }
    } finally {
      if (abortControllersRef.current.get(conversationId) === controller) {
        abortControllersRef.current.delete(conversationId);
      }
    }
  }

  function stopResearch(conversationId: string) {
    const conversation = conversations.find((item) => item.id === conversationId);
    const runningMessage = [...(conversation?.messages ?? [])]
      .reverse()
      .find((message) => message.role === "assistant" && message.run?.status === "running");
    const runId = runningMessage?.run?.id;

    abortControllersRef.current.get(conversationId)?.abort();
    if (runId && runningMessage) {
      void sessionFetch(session?.token ?? null, `/api/research?runId=${encodeURIComponent(runId)}`, {
        method: "DELETE",
        keepalive: true,
      }).catch(() => undefined);
      patchAssistantMessage(conversationId, runningMessage.id, (message) => ({
        ...message,
        content: message.content || t.stoppedAnswer,
        run: message.run
          ? {
              ...message.run,
              status: "cancelled",
              elapsedMs: timestamp() - message.run.startedAt,
            }
          : message.run,
      }));
    }
  }

  function toggleRun(messageId: string) {
    if (!activeConversation) return;
    const canvas = conversationCanvasRef.current;
    const preservedScrollTop = canvas?.scrollTop;
    patchAssistantMessage(activeConversation.id, messageId, (message) => ({
      ...message,
      run: message.run ? { ...message.run, expanded: !message.run.expanded } : message.run,
    }));
    if (!canvas || preservedScrollTop === undefined) return;
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        const previousScrollBehavior = canvas.style.scrollBehavior;
        canvas.style.scrollBehavior = "auto";
        canvas.scrollTop = preservedScrollTop;
        canvas.style.scrollBehavior = previousScrollBehavior;
      });
    });
  }

  function openEvidence(messageId: string, index = 1) {
    const evidence =
      activeConversation?.messages.find((message) => message.id === messageId)?.run?.evidence ?? [];
    const item = evidence[index - 1] ?? evidence[0];
    setEvidenceMessageId(messageId);
    setSelectedEvidenceId(item?.id ?? null);
    setHoveredEvidenceId(null);
    setEvidenceOpen(true);
  }

  function openEvidenceRecord(messageId: string, item?: EvidenceItem) {
    const evidence =
      activeConversation?.messages.find((message) => message.id === messageId)?.run?.evidence ?? [];
    setEvidenceMessageId(messageId);
    setSelectedEvidenceId(item?.id ?? evidence[0]?.id ?? null);
    setHoveredEvidenceId(null);
    setEvidenceOpen(true);
  }

  function resizeEvidenceDrawer(clientX: number) {
    if (window.innerWidth <= 760) return;
    setEvidenceDrawerWidth(
      clampEvidenceDrawerWidth(window.innerWidth - clientX, window.innerWidth),
    );
  }

  function startEvidenceResize(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || window.innerWidth <= 760) return;
    event.preventDefault();
    evidenceResizingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    document.body.classList.add("is-resizing-evidence");
    resizeEvidenceDrawer(event.clientX);
  }

  function moveEvidenceResize(event: React.PointerEvent<HTMLDivElement>) {
    if (!evidenceResizingRef.current) return;
    event.preventDefault();
    resizeEvidenceDrawer(event.clientX);
  }

  function finishEvidenceResize(event: React.PointerEvent<HTMLDivElement>) {
    if (!evidenceResizingRef.current) return;
    evidenceResizingRef.current = false;
    document.body.classList.remove("is-resizing-evidence");
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function resetEvidenceDrawerWidth() {
    setEvidenceDrawerWidth(
      clampEvidenceDrawerWidth(DEFAULT_EVIDENCE_DRAWER_WIDTH, window.innerWidth),
    );
  }

  function handleEvidenceResizeKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    let nextWidth: number | null = null;
    if (event.key === "ArrowLeft") nextWidth = evidenceDrawerWidth + 32;
    if (event.key === "ArrowRight") nextWidth = evidenceDrawerWidth - 32;
    if (event.key === "Home") nextWidth = MIN_EVIDENCE_DRAWER_WIDTH;
    if (event.key === "End") nextWidth = MAX_EVIDENCE_DRAWER_WIDTH;
    if (nextWidth === null) return;
    event.preventDefault();
    setEvidenceDrawerWidth(clampEvidenceDrawerWidth(nextWidth, window.innerWidth));
  }

  async function copyEvidenceData(item: EvidenceItem) {
    const legacyDetail = [item.snippet, item.content]
      .filter((value): value is string => Boolean(value))
      .sort((left, right) => right.length - left.length)[0];
    const text =
      item.rawContent ||
      [item.title, legacyDetail].filter(Boolean).join("\n\n");
    if (!text) return;
    await writeClipboardText(text);
    setCopiedEvidenceId(item.id);
    window.setTimeout(() => {
      setCopiedEvidenceId((current) => (current === item.id ? null : current));
    }, 1800);
  }

  async function copyMessageText(messageId: string, text: string) {
    if (!text.trim()) return;
    await writeClipboardText(text);
    setCopiedMessageId(messageId);
    window.setTimeout(() => {
      setCopiedMessageId((current) => (current === messageId ? null : current));
    }, 1800);
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void askAgent();
  }

  function navigateToView(view: ProductView) {
    setProductView(view);
    const url = new URL(window.location.href);
    if (view === "home") {
      url.searchParams.delete("view");
    } else {
      url.searchParams.set("view", view === "workspace" ? "workbench" : view);
    }
    window.history.pushState({}, "", url);
  }

  return (
    <div className="research-app" data-sidebar-open={sidebarOpen} data-evidence-open={evidenceOpen}>
      <ProductHeader
        locale={locale}
        onLocaleChange={setLocale}
        onViewChange={navigateToView}
        view={productView}
      />
      {productView === "home" ? (
        <LandingPage
          locale={locale}
          onPerformance={() => navigateToView("performance")}
          onTryMint={() => navigateToView("workspace")}
        />
      ) : productView === "performance" ? (
        <PerformancePage locale={locale} onTryMint={() => navigateToView("workspace")} />
      ) : !authChecked ? (
        <main className="login-gate" id="main-content" aria-busy="true" />
      ) : !session ? (
        <LoginGate
          locale={locale}
          notice={
            authNoticeKey === "expired"
              ? t.authExpired
              : authNoticeKey === "required"
                ? t.authRequired
                : null
          }
          onSuccess={handleLoginSuccess}
        />
      ) : (
        <>
      <button
        aria-label={t.closeSidebar}
        className="mobile-scrim"
        onClick={() => setSidebarOpen(false)}
        type="button"
      />

      <aside className="history-sidebar">
        <button className="new-research-button" onClick={newConversation} type="button">
          <Icon name="plus" />
          <span>{t.newResearch}</span>
        </button>

        <div className="history-heading">
          <span>{t.conversationHistory}</span>
          <Icon name="history" />
        </div>

        <nav className="conversation-history" aria-label={t.conversationHistory}>
          {conversations.map((conversation) => {
            const conversationIsRunning = conversationHasActiveRun(conversation);
            const questionCount = conversation.messages.filter((message) => message.role === "user").length;
            return (
              <div
                className={`conversation-item${conversation.id === activeConversationId ? " is-active" : ""}${
                  conversationIsRunning ? " is-running" : ""
                }`}
                key={conversation.id}
              >
                <button
                  aria-current={conversation.id === activeConversationId ? "page" : undefined}
                  className="conversation-select"
                  onClick={() => selectConversation(conversation.id)}
                  type="button"
                >
                  <span>
                    <strong>{conversation.messages.length === 0 ? t.newResearch : conversation.title}</strong>
                    <small className={conversationIsRunning ? "is-running" : undefined}>
                      {conversationIsRunning
                        ? t.historyRunning(questionCount)
                        : conversation.messages.length > 0
                          ? t.questionCount(questionCount)
                          : t.waitingForQuestion}
                    </small>
                  </span>
                  <time>{formatHistoryTime(conversation.updatedAt, locale)}</time>
                </button>
                {conversation.messages.length > 0 && (
                  <button
                    aria-label={t.renameConversation(conversation.title)}
                    className="conversation-rename"
                    onClick={() => renameConversation(conversation.id)}
                    title={t.renameTitle}
                    type="button"
                  >
                    <Icon name="pencil" />
                  </button>
                )}
                <button
                  aria-label={t.deleteConversation(conversation.title)}
                  className="conversation-delete"
                  onClick={() => deleteConversation(conversation.id)}
                  title={t.deleteTitle}
                  type="button"
                >
                  <Icon name="trash" />
                </button>
              </div>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="runtime-state">
            <span aria-hidden="true" />
            <p>
              <strong>{t.agentReady}</strong>
              <small>{t.evidenceFirst}</small>
            </p>
          </div>
          <div className="account-chip-wrap">
            {accountOpen && (
              <>
                <div
                  aria-hidden="true"
                  className="account-pop-overlay"
                  onClick={() => setAccountOpen(false)}
                />
                <div aria-label={t.accountQuotaTitle} className="account-pop" role="dialog">
                  <strong>{session?.username}</strong>
                  <p className="account-pop-title">{t.accountQuotaTitle}</p>
                  {accountQuota === "loading" && (
                    <p className="account-pop-line">{t.accountQuotaLoading}</p>
                  )}
                  {accountQuota === "error" && (
                    <p className="account-pop-line">{t.accountQuotaError}</p>
                  )}
                  {accountQuota !== null &&
                    typeof accountQuota === "object" && (
                      <>
                        <div
                          aria-label={t.accountQuotaUsed(accountQuota.used, accountQuota.limit)}
                          className="account-pop-meter"
                          role="img"
                        >
                          <i
                            style={{
                              width: `${Math.min(
                                100,
                                (accountQuota.used / Math.max(1, accountQuota.limit)) * 100,
                              )}%`,
                            }}
                          />
                        </div>
                        <p className="account-pop-line">
                          {t.accountQuotaUsed(accountQuota.used, accountQuota.limit)} ·{" "}
                          {t.accountQuotaRemaining(accountQuota.remaining)}
                        </p>
                      </>
                    )}
                  <small>{t.accountQuotaReset}</small>
                  <small>{t.accountHistoryNote}</small>
                </div>
              </>
            )}
            <div className="account-chip">
              <button
                className="account-chip-main"
                onClick={toggleAccountPanel}
                title={t.accountDetails}
                type="button"
              >
                <small>{t.signedInAs}</small>
                <strong title={session?.username}>{session?.username}</strong>
              </button>
              <button onClick={handleLogout} type="button">
                {t.signOut}
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main className="research-main" id="main-content">
        <header className="topbar">
          <button
            aria-label={t.openHistory}
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
            type="button"
          >
            <Icon name="menu" />
          </button>
          <div className="conversation-title">
            <strong>
              {(activeConversation?.messages.length ?? 0) === 0
                ? t.newResearch
                : activeConversation?.title || t.newResearch}
            </strong>
            <span>
              {modelLabels[model]} · {mode === "deep" ? t.deepResearch : t.fastAnswer}
            </span>
          </div>
          <div className="topbar-actions">
            <button aria-label={t.createResearch} className="icon-button" onClick={newConversation} type="button">
              <Icon name="plus" />
            </button>
          </div>
        </header>

        <div className="conversation-canvas" ref={conversationCanvasRef}>
          <div className="conversation-column">
            {(activeConversation?.messages.length ?? 0) === 0 ? (
              <section className="empty-state">
                <span className="empty-orbit">
                  <Icon name="spark" />
                </span>
                <p className="eyebrow">{t.emptyEyebrow}</p>
                <h1>{`${t.emptyTitleFirst}${locale === "en" ? " " : ""}${t.emptyTitleSecond}`}</h1>
                <p className="empty-description">
                  {mode === "fast" ? t.fastDescription : t.emptyDescription}
                </p>
                <div className="quick-prompts">
                  {t.quickPrompts.map((item, index) => (
                    <button key={item} onClick={() => void askAgent(item)} type="button">
                      <i aria-hidden="true">{String(index + 1).padStart(2, "0")}</i>
                      <span>{item}</span>
                      <b aria-hidden="true">
                        <Icon name="arrow" />
                      </b>
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <div className="message-list">
                {activeConversation?.messages.map((message) =>
                  message.role === "user" ? (
                    <article className="user-message-group" key={message.id}>
                      <div className="user-message">
                        <p>{message.content}</p>
                      </div>
                      <div className="message-actions message-actions-user">
                        <button
                          aria-label={
                            copiedMessageId === message.id ? t.copiedMessage : t.copyUserMessage
                          }
                          className={`message-copy-button${copiedMessageId === message.id ? " is-copied" : ""}`}
                          onClick={() => void copyMessageText(message.id, message.content)}
                          title={copiedMessageId === message.id ? t.copiedMessage : t.copyUserMessage}
                          type="button"
                        >
                          <Icon name={copiedMessageId === message.id ? "check" : "copy"} />
                        </button>
                      </div>
                    </article>
                  ) : (
                    <article className="assistant-message" key={message.id}>
                      <div className="assistant-identity">
                        <span>
                          <Icon name="spark" />
                        </span>
                        <div>
                          <strong>Mint Finance Agent</strong>
                          <small>
                            {modelLabels[message.run?.model ?? "mint-cu"]} ·{" "}
                            {message.run?.mode === "fast" ? t.fastAnswer : t.deepResearch}
                          </small>
                        </div>
                      </div>

                      {message.run?.mode === "deep" &&
                        (message.run.steps.length > 0 || message.run.status !== "complete") && (
                        <ResearchTimeline
                          now={now}
                          onToggle={() => toggleRun(message.id)}
                          run={message.run}
                          locale={locale}
                        />
                      )}

                      {message.run?.mode === "fast" &&
                        (message.run.steps.length > 0 || message.run.status === "running") && (
                          <FastThinking
                            locale={locale}
                            onToggle={() => toggleRun(message.id)}
                            run={message.run}
                          />
                        )}

                      {message.content ? (
                        <AnswerText
                          content={message.content}
                          onCitation={(index) =>
                            openEvidenceRecord(message.id, message.run?.evidence[index - 1])
                          }
                        />
                      ) : (
                        <div className="answer-loading">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}

                      {(message.content || (message.run?.evidence.length ?? 0) > 0) && (
                        <div className="message-actions message-actions-assistant">
                          {message.content && (
                            <button
                              aria-label={
                                copiedMessageId === message.id ? t.copiedMessage : t.copyAgentAnswer
                              }
                              className={`message-copy-button${copiedMessageId === message.id ? " is-copied" : ""}`}
                              onClick={() => void copyMessageText(message.id, message.content)}
                              title={
                                copiedMessageId === message.id ? t.copiedMessage : t.copyAgentAnswer
                              }
                              type="button"
                            >
                              <Icon name={copiedMessageId === message.id ? "check" : "copy"} />
                            </button>
                          )}

                          {(message.run?.evidence.length ?? 0) > 0 && (
                            <button
                              className="answer-sources"
                              onClick={() => openEvidence(message.id)}
                              type="button"
                            >
                              <span className="source-stack" aria-hidden="true">
                                {message.run!.evidence.slice(0, 3).map((item) => (
                                  <i key={item.id}>{item.source.slice(0, 1).toUpperCase()}</i>
                                ))}
                              </span>
                              {t.viewSources(message.run!.evidence.length)}
                              <Icon name="chevron" />
                            </button>
                          )}
                        </div>
                      )}
                    </article>
                  ),
                )}
              </div>
            )}
          </div>
        </div>

        <div className="composer-dock">
          <div className="composer-shell">
            <textarea
              aria-label={t.inputAria}
              autoComplete="off"
              onChange={(event) => setPrompt(event.currentTarget.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder={t.inputPlaceholder}
              ref={composerRef}
              rows={2}
              value={prompt}
            />
            <div className="composer-toolbar">
              <div className="composer-left">
                <label className="model-picker">
                  <span className="model-status" aria-hidden="true" />
                  <select
                    aria-label={t.selectModel}
                    disabled={activeIsRunning}
                    onChange={(event) => setModel(event.currentTarget.value as ModelId)}
                    value={model}
                  >
                    <option value="mint-cu">Mint-Cu · 9B</option>
                    {/* Mint-Ag · 27B 依赖本机 8002 服务；未启动时隐藏，避免请求无回答。 */}
                  </select>
                  <Icon name="chevron" />
                </label>
                <span className="composer-divider" aria-hidden="true" />
                <button
                  className="mode-button"
                  onClick={() => setMode((value) => (value === "deep" ? "fast" : "deep"))}
                  type="button"
                >
                  <Icon name={mode === "deep" ? "globe" : "spark"} />
                  {mode === "deep" ? t.deepResearch : t.fastAnswer}
                </button>
              </div>
              {activeIsRunning ? (
                <button
                  aria-label={t.stopResearch}
                  className="send-button is-stop"
                  onClick={() => stopResearch(activeConversation.id)}
                  type="button"
                >
                  <Icon name="stop" />
                </button>
              ) : (
                <button
                  aria-label={t.sendQuestion}
                  className="send-button"
                  disabled={!prompt.trim()}
                  onClick={() => void askAgent()}
                  type="button"
                >
                  <Icon name="send" />
                </button>
              )}
            </div>
          </div>
          <p>{t.composerHint}</p>
        </div>
      </main>

      <button
        aria-label={t.closeEvidence}
        className="evidence-scrim"
        onClick={() => setEvidenceOpen(false)}
        type="button"
      />
      <aside
        className="evidence-drawer"
        aria-label={t.evidenceLedger}
        id="evidence-ledger"
        style={{ width: evidenceDrawerWidth }}
      >
        <div
          aria-controls="evidence-ledger"
          aria-label={t.resizeEvidence}
          aria-orientation="vertical"
          aria-valuemax={MAX_EVIDENCE_DRAWER_WIDTH}
          aria-valuemin={MIN_EVIDENCE_DRAWER_WIDTH}
          aria-valuenow={evidenceDrawerWidth}
          className="evidence-resize-handle"
          onDoubleClick={resetEvidenceDrawerWidth}
          onKeyDown={handleEvidenceResizeKeyDown}
          onPointerCancel={finishEvidenceResize}
          onPointerDown={startEvidenceResize}
          onPointerMove={moveEvidenceResize}
          onPointerUp={finishEvidenceResize}
          role="separator"
          tabIndex={0}
          title={t.resizeEvidence}
        />
        <header>
          <div>
            <p className="eyebrow">{t.evidenceLedger}</p>
            <h2>{t.researchSources}</h2>
          </div>
          <button aria-label={t.closeEvidence} className="icon-button" onClick={() => setEvidenceOpen(false)} type="button">
            <Icon name="close" />
          </button>
        </header>

        {activeEvidence.length > 0 ? (
          <>
            <section className="evidence-graph-panel">
              <header>
                <div>
                  <strong>{t.evidenceGraph}</strong>
                  <small>{t.evidenceGraphHint}</small>
                </div>
                <span>{t.evidenceGraphCount(activeEvidence.length)}</span>
              </header>
              <div
                className="evidence-graph-viewport"
                onMouseLeave={() => setHoveredEvidenceId(null)}
              >
                <div className="evidence-graph-canvas" style={{ height: evidenceGraph.height }}>
                  <svg
                    aria-hidden="true"
                    className="evidence-graph-lines"
                    preserveAspectRatio="none"
                    viewBox={`0 0 440 ${evidenceGraph.height}`}
                  >
                    <defs>
                      <linearGradient id="evidence-line-gradient" x1="0" x2="1" y1="0" y2="1">
                        <stop offset="0%" stopColor="#b8ccc3" />
                        <stop offset="100%" stopColor="#79a997" />
                      </linearGradient>
                      <filter id="evidence-line-glow" height="180%" width="180%" x="-40%" y="-40%">
                        <feGaussianBlur result="blur" stdDeviation="2.2" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>
                    {evidenceGraph.edges.map((edge) => {
                      const fromNode = evidenceGraph.nodes.find((node) => node.item.id === edge.from);
                      const toNode = evidenceGraph.nodes.find((node) => node.item.id === edge.to);
                      const isFocused =
                        focusedEvidenceId !== null &&
                        (edge.from === focusedEvidenceId || edge.to === focusedEvidenceId);
                      return (
                        <path
                          className={
                            isFocused
                              ? "is-focused"
                              : focusedEvidenceId
                                ? "is-muted"
                                : ""
                          }
                          d={evidenceGraphPath(fromNode, toNode)}
                          key={edge.id}
                          vectorEffect="non-scaling-stroke"
                        />
                      );
                    })}
                  </svg>

                  <div
                    className={
                      evidenceGraph.edges.some(
                        (edge) =>
                          (edge.from === "root" && edge.to === focusedEvidenceId) ||
                          (edge.to === "root" && edge.from === focusedEvidenceId),
                      )
                        ? "evidence-graph-root is-related"
                        : "evidence-graph-root"
                    }
                  >
                    <span>
                      <Icon name="search" />
                    </span>
                    <div>
                      <small>{t.evidenceGraphRoot}</small>
                      <strong>{evidenceQuestion || activeEvidence[0]?.query || t.researchSources}</strong>
                    </div>
                  </div>

                  {evidenceGraph.nodes.map((node, index) => {
                    const isSelected = selectedEvidence?.id === node.item.id;
                    const isHovered = hoveredEvidenceId === node.item.id;
                    const isRelated = relatedEvidenceIds.has(node.item.id);
                    const isMuted = focusedEvidenceId !== null && !isRelated;
                    const iconName = node.kind === "market" ? "tool" : node.kind === "filing" ? "file" : "globe";
                    return (
                      <button
                        aria-label={`${index + 1}. ${node.item.title}, ${node.item.source}`}
                        aria-pressed={isSelected}
                        className={[
                          "evidence-graph-node",
                          `is-${node.kind}`,
                          isSelected ? "is-selected" : "",
                          isHovered ? "is-hovered" : "",
                          isRelated ? "is-related" : "",
                          isMuted ? "is-muted" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        key={node.item.id}
                        onBlur={() => setHoveredEvidenceId(null)}
                        onClick={() => setSelectedEvidenceId(node.item.id)}
                        onFocus={() => setHoveredEvidenceId(node.item.id)}
                        onMouseEnter={() => setHoveredEvidenceId(node.item.id)}
                        onMouseLeave={() => setHoveredEvidenceId(null)}
                        style={{ left: node.x, top: node.y }}
                        type="button"
                      >
                        <span className="evidence-graph-node-topline">
                          <i>
                            <Icon name={iconName} />
                          </i>
                          <b>{String(index + 1).padStart(2, "0")}</b>
                        </span>
                        <strong>{node.item.title}</strong>
                        <small>{node.item.source}</small>
                      </button>
                    );
                  })}
                </div>
              </div>
            </section>
            {selectedEvidence && (
              <section className="evidence-detail">
                <div className="evidence-source-line">
                  <span>
                    <Icon name="globe" />
                    {selectedEvidence.source}
                  </span>
                  <span>{t.collected}</span>
                </div>
                <h3>{selectedEvidence.title}</h3>
                {selectedEvidenceView && (
                  <div className="evidence-record">
                    <div className="evidence-record-summary">
                      <span>
                        <Icon name="spark" />
                      </span>
                      <div>
                        <small>{evidenceDetailCopy[locale].structuredRecord}</small>
                        <p>{selectedEvidenceView.summary}</p>
                      </div>
                    </div>

                    {selectedEvidenceView.results.length > 0 && (
                      <section className="evidence-result-section">
                        <header>
                          <span>{evidenceDetailCopy[locale].sourceResults}</span>
                          <b>{selectedEvidenceView.results.length}</b>
                        </header>
                        <div className="evidence-result-list">
                          {selectedEvidenceView.results.map((result, index) => {
                            const content = (
                              <>
                                <span className="evidence-result-index">
                                  {String(index + 1).padStart(2, "0")}
                                </span>
                                <span className="evidence-result-copy">
                                  <strong>{result.title}</strong>
                                  <small>{result.domain || result.url}</small>
                                  {result.snippet && <p>{result.snippet}</p>}
                                </span>
                                {result.url && <Icon name="arrow" />}
                              </>
                            );
                            return result.url && isExternalSourceUrl(result.url) ? (
                              <a
                                href={result.url}
                                key={`${result.url}-${index}`}
                                rel="noreferrer"
                                target="_blank"
                              >
                                {content}
                              </a>
                            ) : (
                              <div key={`${result.title}-${index}`}>{content}</div>
                            );
                          })}
                        </div>
                      </section>
                    )}

                    <details className="evidence-raw-data">
                      <summary>
                        <span>
                          <Icon name="file" />
                          {evidenceDetailCopy[locale].rawPreview}
                        </span>
                        <Icon name="chevron" />
                      </summary>
                      <div>
                        <small>{evidenceDetailCopy[locale].rawData}</small>
                        <pre>{selectedEvidenceView.rawPreview}</pre>
                        {selectedEvidenceView.rawTruncated && (
                          <p>{evidenceDetailCopy[locale].rawTruncated}</p>
                        )}
                      </div>
                    </details>
                  </div>
                )}
                {isExternalSourceUrl(selectedEvidence.url) ? (
                  <a
                    className="evidence-origin-action"
                    href={selectedEvidence.url}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {t.openOriginal}
                    <Icon name="arrow" />
                  </a>
                ) : (
                  <div className="evidence-tool-origin">
                    <span>
                      <Icon name="tool" />
                      {t.toolEvidenceNoLink}
                    </span>
                    <button
                      className="evidence-origin-action"
                      onClick={() => void copyEvidenceData(selectedEvidence)}
                      type="button"
                    >
                      {copiedEvidenceId === selectedEvidence.id ? t.copiedToolEvidence : t.copyToolEvidence}
                      <Icon name={copiedEvidenceId === selectedEvidence.id ? "check" : "copy"} />
                    </button>
                  </div>
                )}
              </section>
            )}
          </>
        ) : (
          <div className="evidence-empty">
            <Icon name="archive" />
            <p>{t.evidenceEmpty}</p>
          </div>
        )}
      </aside>
        </>
      )}
    </div>
  );
}
