from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json_repair

from ..constants import (
    FINDOCRESEARCH_FIELD_SCHEMA,
    FINDOCRESEARCH_STRUCTURE,
    FINRPT_OFFICIAL_RESPONSE_FIELDS,
    FINRPT_SOLVER_EXCLUDED_FIELDS,
)


JsonRow = Dict[str, Any]


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_prompt_text(value: Any, default: str = "") -> str:
    """Extract user-visible prompt text from plain strings or chat messages."""

    if value is None:
        return default
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
            else:
                text = normalize_text(item)
                if text:
                    parts.append(text)
        text = "\n\n".join(parts).strip()
        return text if text else default
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return normalize_text(value, default)


def value_at_path(record: JsonRow, key: str) -> Any:
    if key in record:
        return record[key]
    if "." not in key:
        return None
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_present(record: JsonRow, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = value_at_path(record, key)
        if value not in (None, ""):
            return value
    return default


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    try:
        return json_repair.loads(text)
    except Exception:
        for candidate in candidates[1:]:
            try:
                return json_repair.loads(candidate)
            except Exception:
                continue
        return default


def jsonish_string(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    parsed = parse_jsonish(value)
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False)
    return str(value).strip()


def sanitize_file_stem(value: Any, default: str) -> str:
    text = normalize_text(value, default)
    text = re.sub(r"\.md$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or default


def extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s)>\]\"']+", text or "")


def read_jsonl(path: str | Path) -> List[JsonRow]:
    rows: List[JsonRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                try:
                    row = json_repair.loads(line)
                except Exception as repair_exc:
                    raise ValueError(f"Invalid JSONL row {line_number} in {path}: {repair_exc}") from repair_exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} must be an object, got {type(row).__name__}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[JsonRow], mode: str = "w") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


# ---------------------------------------------------------------------------
# Legacy → minimal-schema migration helpers.
#
# Old result rows had many top-level keys (agent_output, evaluator_metadata,
# benchmark_scoring_input, metrics, judge_output, token_stats, evidence_board,
# scoring_input, …). The new schema only has 9 keys. These helpers convert old
# rows on the fly so aggregation / rescoring scripts keep working on existing
# JSONLs without a forced re-run.
# ---------------------------------------------------------------------------


def _stats_from_trajectory(trajectory: List[JsonRow], token_stats: JsonRow | None = None) -> JsonRow:
    token_stats = token_stats or {}
    tools_by_name: JsonRow = {}
    tool_call_count = 0
    for step in trajectory:
        if not isinstance(step, dict) or step.get("name") != "action":
            continue
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            if name in {"", "final_answer"}:
                continue
            tool_call_count += 1
            tools_by_name[name] = int(tools_by_name.get(name) or 0) + 1
    return {
        "steps": len(trajectory),
        "tool_calls": tool_call_count,
        "tools_by_name": tools_by_name,
        "tokens": {
            "prompt": int(token_stats.get("prompt_tokens") or token_stats.get("prompt") or 0),
            "completion": int(token_stats.get("completion_tokens") or token_stats.get("completion") or 0),
            "total": int(token_stats.get("total_tokens") or token_stats.get("total") or 0),
            "llm_calls": int(token_stats.get("llm_calls") or 0),
        },
    }


def _legacy_stats(row: JsonRow, trajectory: List[JsonRow]) -> JsonRow:
    token_stats = row.get("token_stats") or {}
    return _stats_from_trajectory(trajectory, token_stats if isinstance(token_stats, dict) else {})


def _legacy_bench_extra(row: JsonRow) -> JsonRow:
    extra: JsonRow = {}
    for source_key in ("evaluator_metadata", "eval_metadata", "solver_metadata"):
        source = row.get(source_key) or {}
        if isinstance(source, dict):
            for key, value in source.items():
                if value in (None, ""):
                    continue
                extra.setdefault(key, value)
    bsi = row.get("benchmark_scoring_input") or {}
    if isinstance(bsi, dict):
        for key, value in bsi.items():
            if value in (None, "") or key in extra:
                continue
            extra[key] = value
    for key in ("id", "stock_code", "date"):
        if row.get(key) is not None and key not in extra:
            extra[key] = row[key]
    for key in FINRPT_OFFICIAL_RESPONSE_FIELDS:
        if row.get(key) not in (None, "") and key not in extra:
            extra[key] = row[key]
    return extra


def _legacy_trajectory(value: Any) -> List[JsonRow]:
    """Convert old verbose trajectory steps into the lean ``{name, ...}`` shape."""
    if not isinstance(value, list):
        return []
    new_steps: List[JsonRow] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        if "name" in step and ("value" in step or "obs" in step):
            new_steps.append(step)
            continue
        role = step.get("role")
        if role == "planning":
            metadata = step.get("metadata") or {}
            plan_text = ""
            if isinstance(metadata, dict):
                plan_text = str(metadata.get("plan") or "")
            new_steps.append({"name": "plan", "value": plan_text, "think": ""})
            continue
        if role == "action":
            metadata = step.get("metadata") or {}
            think = ""
            if isinstance(metadata, dict):
                think = str(metadata.get("think") or "")
            tool_calls = []
            for call in step.get("tool_calls") or []:
                if isinstance(call, dict):
                    tool_calls.append(
                        {
                            "name": call.get("name"),
                            "action": call.get("action") or "default",
                            "arguments": call.get("arguments") or {},
                        }
                    )
            new_steps.append(
                {
                    "name": "action",
                    "tool_calls": tool_calls,
                    "obs": str(step.get("summary") or ""),
                    "think": think,
                }
            )
    return new_steps


def normalize_legacy_row(row: JsonRow) -> JsonRow:
    """Return ``row`` upgraded to the new lean schema (idempotent)."""

    if "score" in row and "stats" in row and "bench_extra" in row:
        return row
    trajectory = _legacy_trajectory(row.get("agent_trajectory"))
    score = row.get("metrics") or row.get("judge_output") or {}
    if not isinstance(score, dict):
        score = {}
    new_row: JsonRow = {
        "bench_name": row.get("bench_name"),
        "task_index": row.get("task_index"),
        "question": row.get("question"),
        "golden_answer": row.get("golden_answer"),
        "agent_result": row.get("agent_result"),
        "agent_trajectory": trajectory,
        "bench_extra": _legacy_bench_extra(row),
        "status": row.get("status"),
        "stats": _legacy_stats(row, trajectory),
        "score": dict(score),
    }
    if row.get("error"):
        new_row["error"] = row["error"]
    sample_id = row.get("sample_id")
    if sample_id is not None:
        new_row["sample_id"] = sample_id
    return new_row


def normalize_legacy_rows(rows: Iterable[JsonRow]) -> List[JsonRow]:
    return [normalize_legacy_row(row) for row in rows]
