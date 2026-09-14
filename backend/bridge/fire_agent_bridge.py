#!/usr/bin/env python3
"""HTTP streaming bridge from the web UI to the real FIRE Agent Harness.

The bridge deliberately lives outside ``src/fire_agent`` so the aligned
harness stays byte-for-byte compatible with the reference machine.  It
translates the harness' public trajectory (plan, ``think`` summaries, tool
calls, observations, and evidence ledger) into the NDJSON protocol consumed by
``frontend-site/app/page.tsx``.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import multiprocessing
import os
import queue as queue_module
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiohttp import web


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qa" / "fin_deepsearch_sft.yaml"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ToolResult debug payloads let the bridge preserve the original structured
# result for the evidence copy action. Secret-like fields are redacted before
# the payload is sent to the browser.
os.environ.setdefault("FIRE_AGENT_KEEP_TOOL_RESULTS", "1")

from fire_agent import runtime as runtime_module  # noqa: E402
from fire_agent.benchmark_config import (  # noqa: E402
    agent_settings_from_config,
    benchmark_template_variables,
    expand_benchmark_templates,
    load_benchmark_config,
    model_reasoning_settings_from_config,
    prompt_settings_from_config,
    tool_settings_from_config,
    tools_from_config,
    validate_benchmark_config,
)
from fire_agent.benchmarks.base import NO_SEED_URL  # noqa: E402
from fire_agent.benchmarks.fin_deepsearch_sft import FinDeepSearchSFTRunner  # noqa: E402
from fire_agent.cli import build_auxiliary_model  # noqa: E402
from fire_agent.config import ModelSettings, load_fire_agent_env  # noqa: E402
from fire_agent.llm import OpenAICompatibleChatModel  # noqa: E402
from fire_agent.runtime import FIREAgent  # noqa: E402
from fire_agent.schemas import AgentConfig, AgentState, BenchTaskContext, ToolCall, TrajectoryStep  # noqa: E402
from fire_agent.tool_families import build_default_tool_registry  # noqa: E402


JsonDict = Dict[str, Any]
_ACTIVE_RUN = threading.local()
_URL_RE = re.compile(r"https?://[^\s<>()\]}\",]+")
_CONNECTION_NOISE_RE = re.compile(
    r"(?:connection\s*(?:error|failed|refused|reset|closed)?|"
    r"clientconnectorerror|connectorerror|connecterror|"
    r"econn(?:refused|reset)|fetch failed|network error|"
    r"request timeout|read timeout|connect timeout|timed out|server disconnected|"
    r"连接(?:失败|错误|超时|中断|被拒绝)|无法连接|网络(?:错误|异常)|请求超时|读取超时)",
    re.IGNORECASE,
)
MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "mint-cu": {
        "label": "Mint-Cu",
        "size": "9B",
        "env_prefix": "FIRE_AGENT_MINT_CU",
        "base_url": "http://127.0.0.1:8001/v1",
        "served_model": "Mint-Cu",
    },
    "mint-sg": {
        "label": "Mint-Ag",
        "size": "27B",
        "env_prefix": "FIRE_AGENT_MINT_SG",
        "base_url": "http://127.0.0.1:8002/v1",
        # Keep the serving identifier stable; Mint-Ag is the public product name.
        "served_model": "Mint-Sg",
    },
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def normalize_model_id(value: Any) -> str:
    candidate = str(value or "").strip().lower().replace("_", "-")
    return candidate if candidate in MODEL_CATALOG else "mint-cu"


def normalize_run_id(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if _RUN_ID_RE.fullmatch(candidate) else ""


def compact_text(value: Any, limit: int = 900) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    value = value.replace("\x00", "").replace("\r", "")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


def resolve_time_context(payload: Mapping[str, Any]) -> JsonDict:
    server_now = datetime.now(timezone.utc)
    raw_browser = payload.get("browserTime")
    browser = raw_browser if isinstance(raw_browser, Mapping) else {}

    zone_name = compact_text(browser.get("timeZone"), 80) or "UTC"
    try:
        browser_zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone_name = "UTC"
        browser_zone = timezone.utc

    captured_at_ms = browser.get("capturedAtMs")
    try:
        captured_at_ms = int(captured_at_ms)
    except (TypeError, ValueError, OverflowError):
        captured_at_ms = int(server_now.timestamp() * 1000)
    if not 946684800000 <= captured_at_ms <= 4102444800000:
        captured_at_ms = int(server_now.timestamp() * 1000)

    browser_instant = datetime.fromtimestamp(captured_at_ms / 1000, tz=timezone.utc)
    browser_local = browser_instant.astimezone(browser_zone)
    skew_seconds = round(browser_instant.timestamp() - server_now.timestamp(), 3)
    raw_offset = browser.get("utcOffsetMinutes")
    try:
        offset_minutes = int(raw_offset)
    except (TypeError, ValueError, OverflowError):
        offset = browser_local.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    offset_minutes = min(840, max(-840, offset_minutes))

    return {
        "browserCapturedAt": browser_instant.isoformat(),
        "browserLocalDateTime": browser_local.isoformat(),
        "browserLocalDate": browser_local.date().isoformat(),
        "browserTimeZone": zone_name,
        "browserUtcOffsetMinutes": offset_minutes,
        "browserLocale": compact_text(browser.get("locale"), 32),
        "serverReceivedAt": server_now.isoformat(),
        "browserServerSkewSeconds": skew_seconds,
    }


def time_context_prompt(time_context: Mapping[str, Any]) -> str:
    return (
        "Runtime time context for this web session (authoritative for relative date/time expressions):\n"
        f"- Browser local datetime: {time_context.get('browserLocalDateTime')}\n"
        f"- Browser IANA timezone: {time_context.get('browserTimeZone')}\n"
        f"- Browser local date: {time_context.get('browserLocalDate')}\n"
        f"- Server received at (UTC): {time_context.get('serverReceivedAt')}\n"
        f"- Browser/server clock skew: {time_context.get('browserServerSkewSeconds')} seconds\n"
        "Interpret 今天/today/当前/现在 relative to the browser local date and timezone above. "
        "For exchange-traded market data, then apply the security's exchange timezone, trading calendar, "
        "and session state. Never silently substitute a different trading date; state the actual data date "
        "and whether the observation is live, delayed, or end-of-day."
    )


def is_connection_noise(value: Any) -> bool:
    return bool(_CONNECTION_NOISE_RE.search(compact_text(value, 4000)))


def first_text(mapping: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return compact_text(value, 2000)
    return ""


def first_url(mapping: Mapping[str, Any]) -> str:
    for key in (
        "url",
        "source_url",
        "page_url",
        "filing_url",
        "api_url",
        "source_ref",
        "source",
    ):
        value = mapping.get(key)
        if isinstance(value, str):
            match = _URL_RE.search(value)
            if match:
                return match.group(0).rstrip(".,;:")
    match = _URL_RE.search(compact_text(mapping, 5000))
    return match.group(0).rstrip(".,;:") if match else ""


def source_name(url: str, fallback: str = "FIRE Agent") -> str:
    if url:
        try:
            hostname = urlparse(url).hostname
            if hostname:
                return hostname.removeprefix("www.")
        except Exception:
            pass
    return compact_text(fallback, 80) or "FIRE Agent"


def call_arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = call.get("arguments")
    return arguments if isinstance(arguments, Mapping) else {}


def tool_label(call: Mapping[str, Any], *, complete: bool) -> tuple[str, str]:
    name = str(call.get("name") or "tool")
    args = call_arguments(call)
    verb = "已完成" if complete else "正在"
    if name == "web_search":
        query = first_text(args, ("query", "q", "search_query", "keywords"))
        return f"{verb}搜索「{compact_text(query, 96)}」", "检索公开网页并筛选相关来源。"
    if name == "web_reader":
        url = first_text(args, ("url", "source_url", "page_url"))
        return f"{verb}阅读 {source_name(url, '网页来源')}", compact_text(url, 180)
    if name == "structured_table_reader":
        url = first_text(args, ("url", "source_url", "file_path", "path"))
        return f"{verb}提取结构化表格", compact_text(url, 180) or "读取精确行列与单位。"
    if name == "sec_search":
        query = first_text(args, ("query", "company", "ticker", "cik"))
        return f"{verb}检索 SEC 文件", compact_text(query, 160)
    if name == "sec_reader":
        url = first_text(args, ("url", "filing_url", "accession_number"))
        return f"{verb}核验 SEC 披露", compact_text(url, 180)
    if name == "market_data":
        target = first_text(args, ("symbol", "ticker", "code", "function", "query"))
        return f"{verb}查询市场数据", compact_text(target, 160)
    if name == "calculator":
        expression = first_text(args, ("expression", "formula", "query"))
        return f"{verb}计算", compact_text(expression, 180)
    return f"{verb}调用 {name}", compact_text(args, 180)


_COPY_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "set-cookie",
    "token",
}
_NORMALIZED_COPY_SECRET_KEYS = {item.replace("-", "_") for item in _COPY_SECRET_KEYS}


def copy_safe_value(value: Any) -> Any:
    """Return a JSON-safe evidence payload without leaking credentials."""

    if isinstance(value, Mapping):
        cleaned: JsonDict = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized_key = key.lower().replace("-", "_")
            if (
                normalized_key in _NORMALIZED_COPY_SECRET_KEYS
                or normalized_key.endswith("_api_key")
                or normalized_key.endswith("_access_token")
            ):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = copy_safe_value(raw_value)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [copy_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def evidence_copy_content(
    item: Mapping[str, Any],
    raw_tool_result: Optional[Mapping[str, Any]] = None,
) -> str:
    """Serialize the actual evidence record, optionally with its full tool result.

    This intentionally has no presentation-length truncation. The UI summary
    remains compact, while copying a schema evidence item preserves returned
    rows/tables instead of copying only ``value · unit · period``.
    """

    payload: JsonDict = {"evidence": copy_safe_value(item)}
    if raw_tool_result:
        payload["tool_result"] = copy_safe_value(raw_tool_result)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def raw_result_matches_evidence(result: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    family = str(result.get("tool_family") or "").strip().lower()
    action = str(result.get("action") or "").strip().lower()
    metric = first_text(item, ("metric", "title", "fact")).lower()
    provider = first_text(item, ("provider", "origin", "source")).lower()
    result_provider = str(result.get("provider") or "").strip().lower()
    names = [family]
    if family and action and action != "default":
        names.insert(0, f"{family}_{action}")
    name_match = any(name and (metric == name or metric.startswith(f"{name} ")) for name in names)
    provider_match = bool(provider and result_provider and provider == result_provider)
    return name_match or (provider_match and bool(family))


def is_primary_tool_evidence(item: Mapping[str, Any]) -> bool:
    metric = first_text(item, ("metric", "title", "fact")).lower()
    unit = first_text(item, ("unit",)).lower()
    return (
        unit == "schema"
        or metric.endswith(" top_results")
        or metric.endswith(" observation_excerpt")
    )


def evidence_to_browser(
    item: Mapping[str, Any],
    serial: int,
    raw_tool_result: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    url = first_url(item)
    fact = first_text(
        item,
        (
            "fact",
            "title",
            "metric",
            "quote_or_row_excerpt",
            "snippet",
            "text",
            "value",
            "search_text",
        ),
    )
    title = first_text(item, ("title", "fact", "metric", "document_type"))
    if not title:
        title = source_name(url, first_text(item, ("provider", "origin", "source")) or f"Evidence {serial}")
    snippet = first_text(item, ("quote_or_row_excerpt", "snippet", "fact", "text", "search_text"))
    detail_parts = [
        first_text(item, ("fact",)),
        first_text(item, ("value",)),
        first_text(item, ("unit",)),
        first_text(item, ("period", "date")),
        first_text(item, ("quote_or_row_excerpt", "snippet", "text")),
    ]
    content = compact_text(" · ".join(part for part in detail_parts if part), 2600)
    if not content:
        content = compact_text(item, 2600)
    return {
        "id": str(item.get("evidence_id") or f"fire-evidence-{serial}"),
        "title": compact_text(title or fact or f"Evidence {serial}", 220),
        "url": url,
        "snippet": compact_text(snippet or content, 720),
        "content": content,
        "rawContent": evidence_copy_content(item, raw_tool_result),
        "source": source_name(url, first_text(item, ("provider", "origin", "source"))),
        "query": first_text(item, ("query", "search_query", "metric", "fact")),
        "publishedAt": first_text(item, ("published_at", "publishedAt", "date", "period")) or None,
    }


def is_material_evidence(item: Mapping[str, Any]) -> bool:
    status = first_text(item, ("status", "result_status")).lower()
    origin = first_text(item, ("origin", "provider", "source")).lower()
    kind = first_text(item, ("kind", "type", "evidence_type")).lower()
    if status in {"error", "failed", "blocked"}:
        return False
    if any(marker in origin or marker in kind for marker in ("tool_error", "policy_error", "failure")):
        return False
    return bool(
        first_url(item)
        or first_text(
            item,
            ("fact", "value", "quote_or_row_excerpt", "snippet", "title", "metric"),
        )
    )


class RunEmitter:
    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop],
        queue: Any,
        *,
        emit_steps: bool = True,
        emit_native_thinking: bool = False,
    ) -> None:
        self.loop = loop
        self.queue = queue
        self.emit_steps = emit_steps
        self.emit_native_thinking = emit_native_thinking
        self.serial = 0
        self.evidence_count = 0
        self.seen_evidence: set[str] = set()
        self.pending_tool_ids: Deque[str] = deque()

    def next_id(self, prefix: str) -> str:
        self.serial += 1
        return f"fire-{prefix}-{self.serial}"

    def send(self, event: JsonDict) -> None:
        if self.loop is None:
            self.queue.put(event)
            return
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def start_tools(self, calls: Iterable[ToolCall]) -> None:
        if not self.emit_steps:
            return
        for call in calls:
            call_dict = call.to_lean_dict()
            event_id = self.next_id("tool")
            self.pending_tool_ids.append(event_id)
            title, text = tool_label(call_dict, complete=False)
            self.send(
                {
                    "type": "step",
                    "step": {
                        "id": event_id,
                        "kind": "tool",
                        "status": "running",
                        "title": title,
                        "text": text,
                        "tool": str(call_dict.get("name") or "tool"),
                    },
                }
            )

    def finish_step(self, state: AgentState, step: TrajectoryStep) -> None:
        if not self.emit_steps and not self.emit_native_thinking:
            return
        if step.name == "plan":
            if not self.emit_steps:
                return
            plan_text = compact_text(step.value or step.think, 1400)
            if not plan_text or is_connection_noise(plan_text):
                return
            self.send(
                {
                    "type": "step",
                    "step": {
                        "id": self.next_id("plan"),
                        "kind": "thought",
                        "status": "complete",
                        "title": "研究计划",
                        "text": plan_text,
                    },
                }
            )
            return

        thinking_text = (
            step.reasoning_content or step.think
            if self.emit_native_thinking
            else step.think
        )
        if thinking_text and not is_connection_noise(thinking_text):
            self.send(
                {
                    "type": "step",
                    "step": {
                        "id": self.next_id("thought"),
                        "kind": "thought",
                        "status": "complete",
                        "title": "思考过程" if self.emit_native_thinking else "分析与下一步",
                        "text": compact_text(
                            thinking_text,
                            40_000 if self.emit_native_thinking else 1200,
                        ),
                    },
                }
            )

        if not self.emit_steps:
            return

        new_evidence: List[JsonDict] = []
        raw_tool_results = [
            result for result in (step.raw_tool_results or []) if isinstance(result, Mapping)
        ]
        claimed_raw_results: set[int] = set()
        for entry in state.evidence_ledger:
            if not is_material_evidence(entry):
                continue
            evidence_id = str(entry.get("evidence_id") or "")
            fingerprint = evidence_id or str(entry.get("fingerprint") or compact_text(entry, 500))
            if not fingerprint or fingerprint in self.seen_evidence:
                continue
            self.seen_evidence.add(fingerprint)
            self.evidence_count += 1
            raw_tool_result: Optional[Mapping[str, Any]] = None
            if is_primary_tool_evidence(entry):
                for raw_index, candidate in enumerate(raw_tool_results):
                    if raw_index in claimed_raw_results:
                        continue
                    if raw_result_matches_evidence(candidate, entry):
                        raw_tool_result = candidate
                        claimed_raw_results.add(raw_index)
                        break
            new_evidence.append(
                evidence_to_browser(entry, self.evidence_count, raw_tool_result)
            )

        observation = compact_text(step.obs, 720)
        if step.error or is_connection_noise(observation):
            observation = ""
        for index, call in enumerate(step.tool_calls or []):
            if str(call.get("name") or "") == "final_answer":
                continue
            event_id = self.pending_tool_ids.popleft() if self.pending_tool_ids else self.next_id("tool")
            title, fallback_text = tool_label(call, complete=True)
            if step.error:
                title = "已调整研究路径"
                fallback_text = "该来源暂未返回可用结果，继续核验其他来源。"
            self.send(
                {
                    "type": "step",
                    "step": {
                        "id": event_id,
                        "kind": "tool",
                        "status": "complete",
                        "title": title,
                        "text": observation or fallback_text,
                        "tool": str(call.get("name") or "tool"),
                        "count": len(new_evidence) if index == 0 and new_evidence else None,
                    },
                }
            )

        if step.error:
            while self.pending_tool_ids:
                event_id = self.pending_tool_ids.popleft()
                self.send(
                    {
                        "type": "step",
                        "step": {
                            "id": event_id,
                            "kind": "tool",
                            "status": "complete",
                            "title": "已调整研究路径",
                            "text": "该来源暂未返回可用结果，继续核验其他来源。",
                            "tool": "tool",
                        },
                    }
                )

        for evidence in new_evidence:
            self.send({"type": "evidence", "evidence": evidence})


class StreamingAgentState(AgentState):
    def add_step(self, step: TrajectoryStep) -> None:
        super().add_step(step)
        emitter: Optional[RunEmitter] = getattr(_ACTIVE_RUN, "emitter", None)
        if emitter is not None:
            emitter.finish_step(self, step)


class StreamingFIREAgent(FIREAgent):
    def _execute_tool_round(self, state: AgentState, calls: List[ToolCall]) -> Any:
        emitter: Optional[RunEmitter] = getattr(_ACTIVE_RUN, "emitter", None)
        if emitter is not None:
            emitter.start_tools(calls)
        return super()._execute_tool_round(state, calls)


# FIREAgent.run resolves AgentState from its module global. Replacing that
# symbol instruments state creation without editing the aligned harness.
runtime_module.AgentState = StreamingAgentState


class HarnessFactory:
    def __init__(self, config_path: Path, env_file: Path) -> None:
        load_fire_agent_env(str(env_file))
        self.config_path = config_path
        raw_config = load_benchmark_config(config_path)
        self.bench_config = expand_benchmark_templates(raw_config, benchmark_template_variables(raw_config))
        validate_benchmark_config(self.bench_config, "fin_deepsearch_sft", config_path)
        self.agent_settings = agent_settings_from_config(self.bench_config, config_path)
        self.prompt_settings = prompt_settings_from_config(self.bench_config, config_path)
        self.reasoning_settings = model_reasoning_settings_from_config(self.bench_config, config_path)
        self.enabled_tools = tools_from_config(self.bench_config, config_path)
        self.tool_settings = tool_settings_from_config(self.bench_config, config_path)
        self.runner = FinDeepSearchSFTRunner(
            PROJECT_ROOT / "sft" / "fin_deepsearch_sft.jsonl",
            PROJECT_ROOT / "output" / "bridge-unused.jsonl",
            agent=None,  # type: ignore[arg-type]
        )

    def agent_config(self, mode: str) -> AgentConfig:
        allowed = {field.name for field in fields(AgentConfig)}
        values = {key: value for key, value in self.agent_settings.items() if key in allowed}
        if mode == "fast":
            values["max_steps"] = 1
            values["max_tool_calls_per_round"] = 1
            values["final_answer_only_round"] = True
            values["single_turn_reasoning_pass"] = True
            values["benchmark_system_prompt"] = (
                "Answer the user's current question directly from the task context. "
                "No external retrieval or research tools are available. Return the complete "
                "user-facing response only through final_answer.arguments.answer."
            )
            values["benchmark_planning_prompt"] = ""
            values["tool_prompt_constraints"] = {}
        else:
            values["max_steps"] = min(int(values.get("max_steps", 40)), 40)
            values["benchmark_system_prompt"] = str(self.prompt_settings.get("system_constraints") or "").strip()
            values["benchmark_planning_prompt"] = str(self.prompt_settings.get("planning_constraints") or "").strip()
            values["tool_prompt_constraints"] = {
                name: ([constraints] if isinstance(constraints, str) else list(constraints))
                for name, constraints in (self.prompt_settings.get("tool_constraints") or {}).items()
            }
        return AgentConfig(**values)

    def model_settings(self, model_id: str) -> ModelSettings:
        normalized = normalize_model_id(model_id)
        definition = MODEL_CATALOG[normalized]
        prefix = definition["env_prefix"]
        primary = ModelSettings.from_env()

        def value(name: str, fallback: str) -> str:
            return str(os.getenv(f"{prefix}_{name}") or fallback).strip()

        model = value("MODEL", definition["served_model"])
        base_url = value("BASE_URL", definition["base_url"]).rstrip("/")
        raw_thinking = str(os.getenv(f"{prefix}_THINKING_ENABLED") or "").strip().lower()
        if raw_thinking:
            thinking_enabled: Optional[bool] = raw_thinking in {"1", "true", "yes", "y", "on", "enabled"}
        else:
            # Mint models are reasoning models. Keep thinking enabled by
            # default even after their serving name/base URL changes from the
            # temporary DeepSeek channel to a local checkpoint.
            thinking_enabled = True

        extra_body: JsonDict = {}
        raw_template_kwargs = str(
            os.getenv(f"{prefix}_CHAT_TEMPLATE_KWARGS") or ""
        ).strip()
        if raw_template_kwargs:
            try:
                template_kwargs = json.loads(raw_template_kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{prefix}_CHAT_TEMPLATE_KWARGS must be a JSON object"
                ) from exc
            if not isinstance(template_kwargs, dict):
                raise ValueError(
                    f"{prefix}_CHAT_TEMPLATE_KWARGS must be a JSON object"
                )
            extra_body["chat_template_kwargs"] = template_kwargs
        elif "deepseek" not in f"{model} {base_url}".lower():
            # Qwen/SGLang/vLLM-style Mint endpoints commonly expose thinking
            # through the chat template rather than the provider-specific
            # top-level field. Send both signals for local Mint checkpoints.
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": bool(thinking_enabled)
            }

        return ModelSettings(
            model=model,
            api_key=value("API_KEY", "local-fire-agent"),
            base_url=base_url,
            timeout_seconds=int(value("TIMEOUT_SECONDS", str(max(primary.timeout_seconds, 180)))),
            max_tokens=int(value("MAX_TOKENS", str(primary.max_tokens))),
            temperature=float(value("TEMPERATURE", str(primary.temperature))),
            thinking_enabled=thinking_enabled,
            reasoning_effort=primary.reasoning_effort,
            extra_body=extra_body,
        )

    def build_agent(self, mode: str, model_id: str) -> StreamingFIREAgent:
        config = self.agent_config(mode)
        settings = self.model_settings(model_id)
        if self.reasoning_settings:
            settings = ModelSettings(
                **{
                    **settings.__dict__,
                    **self.reasoning_settings,
                }
            )
        model = OpenAICompatibleChatModel(settings)
        auxiliary_model = build_auxiliary_model()
        enabled_tools = [] if mode == "fast" else self.enabled_tools
        tools = build_default_tool_registry(enabled_tools, tool_settings=self.tool_settings)
        return StreamingFIREAgent(model=model, auxiliary_model=auxiliary_model, config=config, tools=tools)

    def task_context(
        self,
        question: str,
        history: List[Mapping[str, str]],
        mode: str,
        time_context: Mapping[str, Any],
    ) -> BenchTaskContext:
        recent = history[-8:]
        context_lines = [
            f"{'User' if turn.get('role') == 'user' else 'Assistant'}: {compact_text(turn.get('content'), 1000)}"
            for turn in recent
            if turn.get("role") in {"user", "assistant"} and compact_text(turn.get("content"), 1000)
        ]
        contextual_question = time_context_prompt(time_context) + "\n\nCurrent user request:\n" + question
        if context_lines:
            contextual_question = (
                time_context_prompt(time_context)
                + "\n\n"
                "Use the following conversation only to resolve references and follow-up context. "
                "Answer the current question, not the earlier questions.\n\n"
                + "\n".join(context_lines)
                + "\n\nCurrent question:\n"
                + question
            )
        if mode == "fast":
            task_prompt = (
                "Respond directly to the user's current question. Use only the task context and "
                "the model's existing knowledge; do not claim to have searched, browsed, or called "
                "external tools. Give a concise but complete user-facing answer through the native "
                "final_answer tool.\n\n"
                + contextual_question
            )
            bench_name = "mint_fast_answer"
            task_family = "single_turn_qa"
            evaluator_name = "generic"
        else:
            task_prompt = self.runner.format_task_prompt(contextual_question, NO_SEED_URL)
            bench_name = "fin_deepsearch_sft"
            task_family = "deep_search_qa"
            evaluator_name = "fin_deepsearch_sft"
        return BenchTaskContext(
            bench_name=bench_name,
            task_index=0,
            task_prompt=task_prompt,
            raw_question=question,
            raw_record={"question": question},
            task_family=task_family,
            evaluator_name=evaluator_name,
        )


def answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("answer", "final_answer", "content", "text"):
            if value.get(key) not in (None, ""):
                return compact_text(value.get(key), 100_000)
    return compact_text(value, 100_000)


def run_harness(
    factory: HarnessFactory,
    question: str,
    history: List[Mapping[str, str]],
    mode: str,
    model_id: str,
    time_context: Mapping[str, Any],
    emitter: RunEmitter,
) -> None:
    started_at = time.time()
    _ACTIVE_RUN.emitter = emitter
    try:
        agent = factory.build_agent(mode, model_id)
        result = agent.run(factory.task_context(question, history, mode, time_context))
        if result.status != "success":
            raise RuntimeError(result.error or f"FIRE Agent stopped with status: {result.status}")
        final = answer_text(result.agent_result)
        if not final:
            raise RuntimeError("FIRE Agent returned an empty final answer")
        if mode == "deep":
            emitter.send(
                {
                    "type": "step",
                    "step": {
                        "id": emitter.next_id("synthesis"),
                        "kind": "thought",
                        "status": "complete",
                        "title": "研究完成",
                        "text": (
                            f"FIRE Agent Harness 完成 {len(result.state.trajectory)} 个轨迹步骤、"
                            f"{result.state.tool_call_count} 次工具调用。"
                        ),
                    },
                }
            )
        for offset in range(0, len(final), 64):
            emitter.send({"type": "answer_delta", "delta": final[offset : offset + 64]})
        emitter.send(
            {
                "type": "done",
                "elapsedMs": int((time.time() - started_at) * 1000),
                "evidenceCount": emitter.evidence_count,
                "backend": "fire-agent-harness",
                "model": model_id,
                "stats": {
                    "steps": len(result.state.trajectory),
                    "toolCalls": result.state.tool_call_count,
                    "toolsByName": result.state.tools_by_name,
                },
            }
        )
    except Exception as exc:
        traceback.print_exc()
        emitter.send({"type": "error", "message": f"FIRE Agent Harness：{compact_text(exc, 1000)}"})
    finally:
        _ACTIVE_RUN.emitter = None


def run_harness_process(
    config_path: str,
    env_file: str,
    question: str,
    history: List[Mapping[str, str]],
    mode: str,
    model_id: str,
    time_context: Mapping[str, Any],
    event_queue: Any,
) -> None:
    """Run one Harness request in an isolated process.

    Process isolation makes a user cancellation real: terminating this worker
    stops the synchronous model/tool loop instead of merely closing the
    browser stream while the Harness keeps occupying a concurrency slot.
    """

    factory = HarnessFactory(Path(config_path), Path(env_file))
    emitter = RunEmitter(
        None,
        event_queue,
        emit_steps=mode == "deep",
        emit_native_thinking=mode == "fast",
    )
    run_harness(
        factory,
        question,
        history,
        mode,
        model_id,
        time_context,
        emitter,
    )


async def stop_run_process(process: Any) -> None:
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        await asyncio.to_thread(process.join, 2.0)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join, 2.0)
    elif process.exitcode is None:
        await asyncio.to_thread(process.join, 0.2)


REQUEST_LOG_DIR = Path(os.getenv("FIRE_AGENT_BRIDGE_LOG_DIR") or (PROJECT_ROOT / "logs"))

# How long a run may stay silent before the bridge records a stall marker.
STALL_AFTER_SECONDS = 60.0


class BridgeRequestLog:
    """Append-only JSONL log describing every request's full lifecycle.

    One line per lifecycle event, so a stuck run can be located by its last
    logged line. Logging must never break request handling: write failures
    print one traceback and are otherwise swallowed.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._handle: Any = None
        self._handle_date = ""
        self._warned = False

    def write(self, run_id: str, event: str, **fields: Any) -> None:
        now = datetime.now(timezone.utc)
        record: JsonDict = {
            "ts": now.isoformat(timespec="milliseconds"),
            "epochMs": int(now.timestamp() * 1000),
            "runId": run_id,
            "event": event,
        }
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            line = json.dumps(
                {"ts": record["ts"], "runId": run_id, "event": event, "serializeError": True}
            )
        try:
            with self._lock:
                handle = self._rotated_handle(now)
                handle.write(line + "\n")
                handle.flush()
        except Exception:
            if not self._warned:
                self._warned = True
                traceback.print_exc()

    def _rotated_handle(self, now: datetime) -> Any:
        date_key = now.strftime("%Y%m%d")
        if self._handle is None or self._handle_date != date_key:
            if self._handle is not None:
                try:
                    self._handle.close()
                except Exception:
                    pass
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"bridge_requests_{date_key}.jsonl"
            self._handle = path.open("a", encoding="utf-8")
            self._handle_date = date_key
        return self._handle


def summarize_stream_event(event: Mapping[str, Any]) -> JsonDict:
    """Compact per-event detail for the request log (answer deltas excluded)."""
    etype = str(event.get("type") or "")
    if etype == "step":
        step = event.get("step") if isinstance(event.get("step"), Mapping) else {}
        return {
            "stepId": step.get("id"),
            "kind": step.get("kind"),
            "status": step.get("status"),
            "tool": step.get("tool"),
            "title": compact_text(step.get("title"), 200),
            "text": compact_text(step.get("text"), 2000),
        }
    if etype == "evidence":
        evidence = event.get("evidence") if isinstance(event.get("evidence"), Mapping) else {}
        return {
            "evidenceId": evidence.get("id"),
            "title": compact_text(evidence.get("title"), 200),
            "url": compact_text(evidence.get("url"), 500),
        }
    if etype == "error":
        return {"message": compact_text(event.get("message"), 2000)}
    if etype == "done":
        return {
            "elapsedMs": event.get("elapsedMs"),
            "evidenceCount": event.get("evidenceCount"),
            "stats": event.get("stats"),
        }
    return {"raw": compact_text(event, 500)}


def client_fields(request: web.Request) -> JsonDict:
    return {
        "remote": request.remote,
        "cfConnectingIp": request.headers.get("CF-Connecting-IP"),
        "xForwardedFor": request.headers.get("X-Forwarded-For"),
        "userAgent": compact_text(request.headers.get("User-Agent"), 300),
    }


def bearer_token(request: web.Request) -> str:
    value = request.headers.get("Authorization", "")
    return value[7:].strip() if value.lower().startswith("bearer ") else ""


@web.middleware
async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    if request.path == "/health":
        return await handler(request)
    expected = request.app["bridge_token"]
    if not expected or not hmac.compare_digest(bearer_token(request), expected):
        raise web.HTTPUnauthorized(text="Unauthorized")
    return await handler(request)


async def health(request: web.Request) -> web.Response:
    factory: HarnessFactory = request.app["factory"]
    return web.json_response(
        {
            "ok": True,
            "backend": "fire-agent-harness",
            "bench": "fin_deepsearch_sft",
            "config": str(factory.config_path.relative_to(PROJECT_ROOT)),
            "models": [
                {
                    "id": model_id,
                    "label": definition["label"],
                    "size": definition["size"],
                    "baseUrl": factory.model_settings(model_id).base_url,
                    "servedModel": factory.model_settings(model_id).model,
                }
                for model_id, definition in MODEL_CATALOG.items()
            ],
            "tools": factory.enabled_tools,
            "concurrency": {
                "deep": request.app["deep_max_concurrent"],
                "fast": request.app["fast_max_concurrent"],
            },
            "activeRuns": {
                "total": len(request.app["run_controls"]),
                "deep": sum(
                    1
                    for control in request.app["run_controls"].values()
                    if control.get("mode") == "deep"
                ),
                "fast": sum(
                    1
                    for control in request.app["run_controls"].values()
                    if control.get("mode") == "fast"
                ),
            },
        }
    )


async def research(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text=f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, MutableMapping):
        raise web.HTTPBadRequest(text="Request body must be an object")
    question = compact_text(payload.get("question"), 12_000)
    if not question:
        raise web.HTTPBadRequest(text="question is required")
    mode = "fast" if payload.get("mode") == "fast" else "deep"
    model_id = normalize_model_id(payload.get("model"))
    run_id = normalize_run_id(payload.get("runId"))
    if not run_id:
        run_id = f"bridge-{int(time.time() * 1000)}-{id(request):x}"
    time_context = resolve_time_context(payload)
    raw_history = payload.get("history")
    history: List[Mapping[str, str]] = []
    if isinstance(raw_history, list):
        history = [
            {"role": str(turn.get("role") or ""), "content": compact_text(turn.get("content"), 4000)}
            for turn in raw_history[-8:]
            if isinstance(turn, Mapping)
        ]

    run_controls: Dict[str, JsonDict] = request.app["run_controls"]
    request_log: BridgeRequestLog = request.app["request_log"]
    if run_id in run_controls:
        request_log.write(
            run_id,
            "rejected_duplicate_run_id",
            mode=mode,
            model=model_id,
            client=client_fields(request),
        )
        raise web.HTTPConflict(text=f"runId is already active: {run_id}")
    control: JsonDict = {
        "runId": run_id,
        "mode": mode,
        "model": model_id,
        "cancelled": False,
        "cancel_event": asyncio.Event(),
        "process": None,
        "startedAt": int(time.time() * 1000),
    }
    run_controls[run_id] = control
    request_log.write(
        run_id,
        "request_received",
        mode=mode,
        model=model_id,
        question=question,
        historyTurns=len(history),
        client=client_fields(request),
        activeRunsIncludingThis={
            "total": len(run_controls),
            "deep": sum(1 for c in run_controls.values() if c.get("mode") == "deep"),
            "fast": sum(1 for c in run_controls.values() if c.get("mode") == "fast"),
        },
        limits={
            "deep": request.app["deep_max_concurrent"],
            "fast": request.app["fast_max_concurrent"],
        },
    )

    response = web.StreamResponse(
        status=200,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Type": "application/x-ndjson; charset=utf-8",
            "X-Content-Type-Options": "nosniff",
            "X-Mint-Agent-Backend": "fire-agent-harness",
        },
    )
    try:
        await response.prepare(request)
        started_at_ms = int(time.time() * 1000)
        await response.write(
            (
                json.dumps(
                    {
                        "type": "run_started",
                        "startedAt": started_at_ms,
                        "mode": mode,
                        "model": model_id,
                        "runId": run_id,
                        "backend": "fire-agent-harness",
                        "timeContext": time_context,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    except (ConnectionResetError, asyncio.CancelledError):
        request_log.write(run_id, "client_gone_before_start")
        run_controls.pop(run_id, None)
        return response
    request_log.write(run_id, "run_started_sent")

    event_queue: Any = None
    process: Any = None
    semaphore_acquired = False
    acquire_task: Optional[asyncio.Task[Any]] = None
    cancel_wait_task: Optional[asyncio.Task[Any]] = None
    run_semaphore: asyncio.Semaphore = request.app["run_semaphores"][mode]
    queue_wait_started = time.time()
    queue_wait_ms: Optional[int] = None
    worker_pid: Optional[int] = None
    event_counts: Dict[str, int] = {}
    answer_parts: List[str] = []
    last_event_at = time.time()
    last_stall_logged_at = 0.0
    stall_count = 0
    last_terminal_type = ""
    disconnected = False
    try:
        acquire_task = asyncio.create_task(run_semaphore.acquire())
        cancel_wait_task = asyncio.create_task(control["cancel_event"].wait())
        await asyncio.wait(
            {acquire_task, cancel_wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire_task.done() and not acquire_task.cancelled() and acquire_task.exception() is None:
            semaphore_acquired = True
        if control["cancelled"]:
            if not acquire_task.done():
                acquire_task.cancel()
                await asyncio.gather(acquire_task, return_exceptions=True)
            request_log.write(
                run_id,
                "cancelled_while_queued",
                queueWaitMs=int((time.time() - queue_wait_started) * 1000),
            )
            return response
        if not semaphore_acquired:
            await acquire_task
            semaphore_acquired = True
        cancel_wait_task.cancel()
        await asyncio.gather(cancel_wait_task, return_exceptions=True)
        queue_wait_ms = int((time.time() - queue_wait_started) * 1000)
        request_log.write(run_id, "slot_acquired", queueWaitMs=queue_wait_ms)

        process_context = request.app["process_context"]
        event_queue = process_context.Queue()
        process = process_context.Process(
            target=run_harness_process,
            args=(
                str(request.app["config_path"]),
                str(request.app["env_file"]),
                question,
                history,
                mode,
                model_id,
                time_context,
                event_queue,
            ),
            daemon=True,
            name=f"fire-agent-{run_id[:32]}",
        )
        control["process"] = process
        spawn_started = time.time()
        process.start()
        worker_pid = process.pid
        last_event_at = time.time()
        request_log.write(
            run_id,
            "worker_started",
            pid=worker_pid,
            spawnMs=int((time.time() - spawn_started) * 1000),
        )

        terminal = False
        while not terminal and not control["cancelled"]:
            try:
                event = await asyncio.to_thread(event_queue.get, True, 0.25)
            except queue_module.Empty:
                idle_s = time.time() - last_event_at
                if (
                    idle_s >= STALL_AFTER_SECONDS
                    and time.time() - last_stall_logged_at >= STALL_AFTER_SECONDS
                ):
                    stall_count += 1
                    last_stall_logged_at = time.time()
                    request_log.write(
                        run_id,
                        "stall",
                        idleMs=int(idle_s * 1000),
                        processAlive=process.is_alive(),
                        pid=worker_pid,
                        eventCounts=dict(event_counts),
                        answerChars=sum(len(part) for part in answer_parts),
                    )
                if process.is_alive():
                    continue
                if not control["cancelled"]:
                    request_log.write(
                        run_id,
                        "worker_exited_without_final",
                        exitcode=process.exitcode,
                        pid=worker_pid,
                    )
                    event = {
                        "type": "error",
                        "message": (
                            "FIRE Agent worker exited before returning a final event "
                            f"(exit code {process.exitcode})."
                        ),
                    }
                else:
                    break
            now = time.time()
            gap_ms = int((now - last_event_at) * 1000)
            last_event_at = now
            etype = str(event.get("type") or "")
            event_counts[etype] = event_counts.get(etype, 0) + 1
            if etype == "answer_delta":
                answer_parts.append(str(event.get("delta") or ""))
            else:
                if etype in {"done", "error"}:
                    last_terminal_type = etype
                request_log.write(
                    run_id,
                    "stream_event",
                    eventType=etype,
                    gapMs=gap_ms,
                    seq=sum(event_counts.values()),
                    detail=summarize_stream_event(event),
                )
            terminal = event.get("type") in {"done", "error"}
            await response.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))

        if not control["cancelled"]:
            await asyncio.to_thread(process.join, 2.0)
    except (ConnectionResetError, asyncio.CancelledError):
        disconnected = True
        request_log.write(
            run_id,
            "client_disconnected",
            eventCounts=dict(event_counts),
            answerChars=sum(len(part) for part in answer_parts),
        )
        control["cancelled"] = True
        control["cancel_event"].set()
    except Exception as exc:
        request_log.write(
            run_id,
            "handler_exception",
            error=compact_text(exc, 1000),
            traceback=compact_text(traceback.format_exc(), 4000),
        )
        raise
    finally:
        if acquire_task is not None and not acquire_task.done():
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
        if cancel_wait_task is not None and not cancel_wait_task.done():
            cancel_wait_task.cancel()
            await asyncio.gather(cancel_wait_task, return_exceptions=True)
        await stop_run_process(process)
        if semaphore_acquired:
            run_semaphore.release()
        if run_controls.get(run_id) is control:
            run_controls.pop(run_id, None)
        if event_queue is not None:
            event_queue.close()
        if last_terminal_type:
            outcome = last_terminal_type
        elif disconnected:
            outcome = "client_disconnected"
        elif control["cancelled"]:
            outcome = "cancelled" if queue_wait_ms is not None else "cancelled_while_queued"
        else:
            outcome = "unknown"
        final_answer = "".join(answer_parts)
        request_log.write(
            run_id,
            "run_finished",
            outcome=outcome,
            mode=mode,
            model=model_id,
            totalMs=int(time.time() * 1000) - control["startedAt"],
            queueWaitMs=queue_wait_ms,
            eventCounts=dict(event_counts),
            stallCount=stall_count,
            answerChars=len(final_answer),
            finalAnswer=compact_text(final_answer, 100_000),
            workerPid=worker_pid,
            workerExitcode=process.exitcode if process is not None else None,
        )

    try:
        await response.write_eof()
    except ConnectionResetError:
        pass
    return response


async def cancel_research(request: web.Request) -> web.Response:
    run_id = normalize_run_id(request.match_info.get("run_id"))
    if not run_id:
        raise web.HTTPBadRequest(text="A valid runId is required")
    request_log: BridgeRequestLog = request.app["request_log"]
    control = request.app["run_controls"].get(run_id)
    if control is None:
        request_log.write(run_id, "cancel_requested", found=False, client=client_fields(request))
        return web.json_response({"ok": True, "runId": run_id, "cancelled": False})
    request_log.write(
        run_id,
        "cancel_requested",
        found=True,
        mode=control.get("mode"),
        runningForMs=int(time.time() * 1000) - int(control.get("startedAt") or 0),
        client=client_fields(request),
    )
    control["cancelled"] = True
    control["cancel_event"].set()
    await stop_run_process(control.get("process"))
    return web.json_response({"ok": True, "runId": run_id, "cancelled": True})


async def cleanup_active_runs(app: web.Application) -> None:
    controls = list(app["run_controls"].values())
    for control in controls:
        control["cancelled"] = True
        control["cancel_event"].set()
    await asyncio.gather(
        *(stop_run_process(control.get("process")) for control in controls),
        return_exceptions=True,
    )
    app["run_controls"].clear()


def build_app(
    config_path: Path,
    env_file: Path,
    bridge_token: str,
    max_concurrent: int,
    fast_max_concurrent: int,
) -> web.Application:
    app = web.Application(
        middlewares=[auth_middleware],
        client_max_size=128 * 1024,
    )
    app["factory"] = HarnessFactory(config_path, env_file)
    app["config_path"] = config_path
    app["env_file"] = env_file
    app["bridge_token"] = bridge_token
    deep_limit = max(1, max_concurrent)
    fast_limit = max(1, fast_max_concurrent)
    app["deep_max_concurrent"] = deep_limit
    app["fast_max_concurrent"] = fast_limit
    app["run_semaphores"] = {
        "deep": asyncio.Semaphore(deep_limit),
        "fast": asyncio.Semaphore(fast_limit),
    }
    app["run_controls"] = {}
    app["request_log"] = BridgeRequestLog(REQUEST_LOG_DIR)
    app["process_context"] = multiprocessing.get_context("spawn")
    app.router.add_get("/health", health)
    app.router.add_post("/v1/research", research)
    app.router.add_delete("/v1/research/{run_id}", cancel_research)
    app.on_cleanup.append(cleanup_active_runs)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream FIRE Agent Harness events to the Mint Agent frontend.")
    parser.add_argument("--host", default=os.getenv("FIRE_AGENT_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FIRE_AGENT_BRIDGE_PORT", "4180")))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--token",
        default=(
            os.getenv("FIRE_AGENT_BRIDGE_TOKEN")
            or os.getenv("FIRE_AGENT_API_KEY")
            or ""
        ),
        help="Bearer token. Defaults to FIRE_AGENT_BRIDGE_TOKEN, then FIRE_AGENT_API_KEY.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Deep-research concurrency. Defaults to FIRE_AGENT_BRIDGE_MAX_CONCURRENT after loading --env-file.",
    )
    parser.add_argument(
        "--fast-max-concurrent",
        type=int,
        default=None,
        help=(
            "Fast-answer concurrency. Defaults to FIRE_AGENT_BRIDGE_FAST_MAX_CONCURRENT "
            "after loading --env-file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Without a configured handler the aiohttp access log (INFO level) is
    # dropped entirely, leaving no per-request trace in the process log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # The env file may contain the fallback token, so load it once before the
    # final validation and app construction.
    load_fire_agent_env(str(args.env_file))
    token = args.token or os.getenv("FIRE_AGENT_BRIDGE_TOKEN") or os.getenv("FIRE_AGENT_API_KEY") or ""
    if not token:
        raise SystemExit("Set FIRE_AGENT_BRIDGE_TOKEN (or FIRE_AGENT_API_KEY) before starting the bridge.")
    max_concurrent = (
        args.max_concurrent
        if args.max_concurrent is not None
        else int(os.getenv("FIRE_AGENT_BRIDGE_MAX_CONCURRENT", "1"))
    )
    fast_max_concurrent = (
        args.fast_max_concurrent
        if args.fast_max_concurrent is not None
        else int(os.getenv("FIRE_AGENT_BRIDGE_FAST_MAX_CONCURRENT", "4"))
    )
    app = build_app(
        args.config.resolve(),
        args.env_file.resolve(),
        token,
        max_concurrent,
        fast_max_concurrent,
    )
    web.run_app(app, host=args.host, port=args.port, access_log_format='%a "%r" %s %Tf')


if __name__ == "__main__":
    main()
