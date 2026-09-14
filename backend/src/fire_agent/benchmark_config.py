from __future__ import annotations

import os
from dataclasses import MISSING, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .benchmarks.registry import canonical_bench_name
from .config import PROJECT_ROOT
from .schemas import AgentConfig


JsonDict = Dict[str, Any]


_DYNAMIC_DATE_VALUES = {"current", "current_date", "today", "run_date", "now"}


def _agent_config_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(AgentConfig))


def _agent_config_bool_fields() -> set[str]:
    return {
        field.name
        for field in fields(AgentConfig)
        if field.default is not MISSING and isinstance(field.default, bool)
    }


def _agent_config_int_fields() -> set[str]:
    return {
        field.name
        for field in fields(AgentConfig)
        if field.default is not MISSING and isinstance(field.default, int) and not isinstance(field.default, bool)
    }


def _agent_config_str_fields() -> set[str]:
    return {
        field.name
        for field in fields(AgentConfig)
        if field.default is not MISSING and isinstance(field.default, str)
    }


def resolve_benchmark_config_path(
    bench: str,
    explicit_path: Optional[str] = None,
    configs_dir: Optional[str] = "configs/qa",
) -> Optional[Path]:
    """Resolve an optional benchmark YAML config.

    ``explicit_path`` is strict and must exist. ``configs_dir`` is best-effort:
    if no matching YAML is found, callers should fall back to runner defaults.
    """

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {path}")
        return path

    if not configs_dir:
        return None

    bench_name = canonical_bench_name(bench)
    raw_dir = Path(configs_dir).expanduser()
    candidate_dirs = [raw_dir if raw_dir.is_absolute() else Path.cwd() / raw_dir]
    project_candidate = PROJECT_ROOT / raw_dir if not raw_dir.is_absolute() else raw_dir
    if project_candidate not in candidate_dirs:
        candidate_dirs.append(project_candidate)

    for base_dir in candidate_dirs:
        for suffix in (".yaml", ".yml"):
            candidate = base_dir / f"{bench_name}{suffix}"
            if candidate.exists():
                return candidate
    return None


def load_benchmark_config(path: Optional[Path]) -> JsonDict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark config must contain a YAML mapping: {path}")
    return data


def benchmark_template_variables(config: JsonDict) -> JsonDict:
    """Resolve small YAML template variables used by benchmark configs.

    Static historical benchmarks can keep literal dates in their YAML. Benchmarks
    that should follow the current run date can set:

        date_context:
          source_cutoff_date: current_date

    and then refer to ``{source_cutoff_date}`` in prompt text or tool defaults.
    """

    run_date = datetime.now(timezone.utc).date().isoformat()
    raw_context = config.get("date_context") or config.get("benchmark_context") or {}
    if not isinstance(raw_context, dict):
        raw_context = {}

    raw_cutoff = (
        os.getenv("FIRE_AGENT_SOURCE_CUTOFF_DATE")
        or raw_context.get("source_cutoff_date")
        or raw_context.get("current_date")
        or config.get("source_cutoff_date")
    )
    cutoff = str(raw_cutoff or run_date).strip()
    if cutoff.lower() in _DYNAMIC_DATE_VALUES:
        cutoff = run_date

    return {
        "run_date": run_date,
        "current_date": cutoff,
        "source_cutoff_date": cutoff,
    }


def expand_benchmark_templates(config: JsonDict, variables: Optional[JsonDict] = None) -> JsonDict:
    """Recursively expand ``{name}`` placeholders in benchmark YAML values."""

    resolved = variables or benchmark_template_variables(config)

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            text = value
            for key, replacement in resolved.items():
                text = text.replace("{" + str(key) + "}", str(replacement))
            return text
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(config)


def tools_from_config(config: JsonDict, path: Optional[Path] = None) -> Optional[List[str]]:
    raw_tools = config.get("tools", config.get("tool_families"))
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list) or not all(isinstance(tool, str) for tool in raw_tools):
        location = f" in {path}" if path else ""
        raise ValueError(f"'tools' must be a list of tool names{location}")
    if any(tool.strip() == "final_answer" for tool in raw_tools):
        location = f" in {path}" if path else ""
        raise ValueError(
            f"'tools' must list only external tool families; final_answer is injected by the runtime{location}"
        )
    return raw_tools


def tool_settings_from_config(config: JsonDict, path: Optional[Path] = None) -> JsonDict:
    raw_settings = config.get("tool_settings") or {}
    if not isinstance(raw_settings, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"'tool_settings' must be a mapping keyed by tool name{location}")
    for tool_name, settings in raw_settings.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            location = f" in {path}" if path else ""
            raise ValueError(f"'tool_settings' keys must be non-empty tool names{location}")
        if not isinstance(settings, dict):
            location = f" in {path}" if path else ""
            raise ValueError(f"'tool_settings.{tool_name}' must be a mapping{location}")
    return raw_settings


def agent_settings_from_config(config: JsonDict, path: Optional[Path] = None) -> JsonDict:
    raw_settings: JsonDict = {}
    agent_keys = _agent_config_field_names()
    bool_agent_keys = _agent_config_bool_fields()
    int_agent_keys = _agent_config_int_fields()
    str_agent_keys = _agent_config_str_fields()
    raw_agent = config.get("agent") or {}
    if raw_agent:
        if not isinstance(raw_agent, dict):
            location = f" in {path}" if path else ""
            raise ValueError(f"'agent' must be a mapping{location}")
        unknown_agent_keys = sorted(set(raw_agent) - set(agent_keys))
        if unknown_agent_keys:
            location = f" in {path}" if path else ""
            keys = ", ".join(unknown_agent_keys)
            raise ValueError(f"'agent' contains unsupported keys: {keys}{location}")
        raw_settings.update(raw_agent)

    for key in agent_keys:
        if key in config:
            raw_settings[key] = config[key]

    for key in agent_keys:
        if key not in raw_settings:
            continue
        value = raw_settings[key]
        if key == "context_mode":
            if not isinstance(value, str) or value.strip().lower() not in {
                "legacy",
                "auto",
                "packet",
                "hybrid",
                "full",
                "full_compatible",
            }:
                location = f" in {path}" if path else ""
                raise ValueError(
                    f"'agent.{key}' must be one of legacy, auto, packet, hybrid, full, or full_compatible{location}"
                )
            continue
        if key == "task_state_render_mode":
            if not isinstance(value, str) or value.strip().lower() not in {"full", "view", "auto"}:
                location = f" in {path}" if path else ""
                raise ValueError(f"'agent.{key}' must be one of full, view, or auto{location}")
            continue
        if key == "context_compression_mode":
            if not isinstance(value, str) or value.strip().lower() not in {
                "llm",
                "always",
                "deterministic",
                "off",
                "disabled",
                "none",
                "false",
                "0",
            }:
                location = f" in {path}" if path else ""
                raise ValueError(f"'agent.{key}' must be llm, deterministic, or off{location}")
            continue
        if key in bool_agent_keys:
            if not isinstance(value, bool):
                location = f" in {path}" if path else ""
                raise ValueError(f"'agent.{key}' must be a boolean{location}")
            continue
        if key in str_agent_keys:
            if value is not None and not isinstance(value, str):
                location = f" in {path}" if path else ""
                raise ValueError(f"'agent.{key}' must be a string{location}")
            continue
        if key == "tool_prompt_constraints":
            if not isinstance(value, dict):
                location = f" in {path}" if path else ""
                raise ValueError(f"'agent.{key}' must be a mapping{location}")
            continue
        if key not in int_agent_keys:
            continue
        if not isinstance(value, int) or value < 0:
            location = f" in {path}" if path else ""
            raise ValueError(f"'agent.{key}' must be a non-negative integer{location}")
        if key in {"max_steps", "max_tool_calls_per_round", "tool_timeout_seconds"} and value <= 0:
            location = f" in {path}" if path else ""
            raise ValueError(f"'agent.{key}' must be a positive integer{location}")
    return raw_settings


def prompt_settings_from_config(config: JsonDict, path: Optional[Path] = None) -> JsonDict:
    raw_prompt = config.get("prompt") or {}
    if not isinstance(raw_prompt, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"'prompt' must be a mapping{location}")

    tool_constraints = raw_prompt.get("tool_constraints") or {}
    if not isinstance(tool_constraints, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"'prompt.tool_constraints' must be a mapping keyed by tool name{location}")
    for tool_name, constraints in tool_constraints.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            location = f" in {path}" if path else ""
            raise ValueError(f"'prompt.tool_constraints' keys must be non-empty tool names{location}")
        if isinstance(constraints, str):
            continue
        if not isinstance(constraints, list) or not all(isinstance(item, str) for item in constraints):
            location = f" in {path}" if path else ""
            raise ValueError(f"'prompt.tool_constraints.{tool_name}' must be a string or list of strings{location}")
    return raw_prompt


def model_reasoning_settings_from_config(config: JsonDict, path: Optional[Path] = None) -> JsonDict:
    raw_model = config.get("model") or {}
    if not isinstance(raw_model, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"'model' must be a mapping{location}")
    raw_reasoning = raw_model.get("reasoning") or {}
    if raw_reasoning in (None, ""):
        return {}
    if not isinstance(raw_reasoning, dict):
        location = f" in {path}" if path else ""
        raise ValueError(f"'model.reasoning' must be a mapping{location}")

    settings: JsonDict = {}
    if "enabled" in raw_reasoning:
        enabled = raw_reasoning.get("enabled")
        if not isinstance(enabled, bool):
            location = f" in {path}" if path else ""
            raise ValueError(f"'model.reasoning.enabled' must be a boolean{location}")
        settings["thinking_enabled"] = enabled

    mode = raw_reasoning.get("mode")
    if mode not in (None, ""):
        mode_text = str(mode).strip().lower()
        if mode_text in {"off", "disabled", "false", "0", "none"}:
            settings["thinking_enabled"] = False
        elif mode_text in {"on", "enabled", "true", "1"}:
            settings["thinking_enabled"] = True
        elif mode_text in {"low", "medium", "high", "max"}:
            settings["thinking_enabled"] = True
            settings["reasoning_effort"] = mode_text
        else:
            location = f" in {path}" if path else ""
            raise ValueError(
                f"'model.reasoning.mode' must be off/on/low/medium/high/max{location}"
            )

    effort = raw_reasoning.get("effort", raw_reasoning.get("reasoning_effort"))
    if effort not in (None, ""):
        effort_text = str(effort).strip().lower()
        if effort_text not in {"low", "medium", "high", "max"}:
            location = f" in {path}" if path else ""
            raise ValueError(f"'model.reasoning.effort' must be low, medium, high, or max{location}")
        settings["reasoning_effort"] = effort_text

    return settings


def validate_benchmark_config(config: JsonDict, bench: str, path: Optional[Path] = None) -> None:
    configured_bench = config.get("bench") or config.get("benchmark")
    if not configured_bench:
        return
    expected = canonical_bench_name(bench)
    actual = canonical_bench_name(str(configured_bench))
    if actual != expected:
        location = f" in {path}" if path else ""
        raise ValueError(f"Benchmark config bench mismatch{location}: expected {expected}, got {actual}")
