from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from .memory import ActionStep, AgentMemory, PlanningStep, TaskStep
from .schemas import AgentConfig, AgentState, ToolCall, ToolResult, compact_text


Message = Dict[str, Any]
JsonDict = Dict[str, Any]

TASK_STATE_VIEW_KEYS = (
    "schema_version",
    "revision",
    "contract",
    "progress",
    "evidence",
    "answer_state",
    "answer_pins",
    "batch_coverage",
    "open_slots",
    "blocked_slots",
    "gaps",
    "checks",
    "basis_notes",
    "stale_paths",
    "next_focus",
)

TASK_STATE_SCHEMA_VERSION = "fire_task_state.v1"

TASK_STATE_LIST_MERGE_KEYS = {
    "evidence",
    "answer_state",
    "answer_pins",
    "batch_coverage",
    "stale_paths",
}

TASK_STATE_MAPPING_MERGE_KEYS = {
    "contract",
    "progress",
}

TASK_STATE_LEGACY_PATCH_LIST_MERGE_KEYS = {
    "open_slots",
    "blocked_slots",
    "gaps",
    "checks",
    "basis_notes",
}

TASK_STATE_DERIVED_KEYS = {
    "evidence_count",
    "stale_path_count",
    "batch_coverage_omitted_count",
}

TASK_STATE_SCHEMA_INSTRUCTION = (
    "TaskState is the agent workflow's structured working memory across steps: it tells the "
    "next step what the task asks for, what evidence or values are already confirmed, what "
    "gaps remain, which paths are stale, and what objective/tool focus comes next. It is not "
    "the final answer and does not replace tool calls. For every non-terminal multi-turn step, "
    "content.task_state must be a complete mergeable snapshot, not a delta. Use only this "
    "fixed top-level schema: "
    "schema_version(string, exactly fire_task_state.v1), revision(integer), "
    "contract(object), progress(object), "
    "evidence(list), answer_state(list), answer_pins(list), batch_coverage(list), "
    "open_slots(list), blocked_slots(list), gaps(list), checks(list), basis_notes(list), "
    "stale_paths(list), and next_focus(string or object). Do not create free top-level "
    "keys. Always carry forward prior still-valid fields. contract.task, progress.status, "
    "progress.completed_actions, and next_focus are required. Put entity/metric/period/"
    "source-basis/output requirements under contract; "
    "put status/found/values under progress; put unresolved work under gaps/open_slots; "
    "and put the immediate next objective under next_focus. Omit unused fields. Minimal example: "
    '{"schema_version":"fire_task_state.v1","revision":1,'
    '"contract":{"task":"...","entities":["..."],"metrics":["..."],"periods":["..."]},'
    '"progress":{"status":"searching","completed_actions":0,"values":{}},'
    '"gaps":["..."],"next_focus":{"objective":"...","tools":["..."]}}.'
)

_TASK_STATE_CONTRACT_ALIASES = {
    "entity": "entities",
    "entities": "entities",
    "metric": "metrics",
    "metrics": "metrics",
    "period": "periods",
    "periods": "periods",
    "date": "periods",
    "dates": "periods",
    "as_of": "periods",
    "year": "periods",
    "years": "periods",
    "source_basis": "source_basis",
    "basis": "source_basis",
    "unit": "units",
    "units": "units",
    "currency": "currency",
    "scale": "scale",
    "formula": "formula",
    "output": "output_format",
    "output_format": "output_format",
    "question": "question",
}

_TASK_STATE_PROGRESS_ALIASES = {
    "status": "status",
    "found": "found",
    "have": "found",
    "values": "values",
    "value": "values",
    "candidate": "candidates",
    "candidates": "candidates",
    "candidate_value": "candidates",
    "answer": "candidate_answer",
    "answer_ready": "answer_ready",
    "precision": "precision",
}

_TASK_STATE_GAP_ALIASES = {
    "gap",
    "need",
    "needs",
    "pending",
    "missing",
    "unresolved",
}

_TASK_STATE_CHECK_ALIASES = {
    "calc",
    "calculation",
    "calculations",
    "check",
}

_TASK_STATE_NEXT_ALIASES = {
    "next",
    "next_action",
    "focus",
}

_TASK_STATE_EVIDENCE_ALIASES = {
    "source",
    "sources",
    "url",
    "source_url",
    "filing",
    "filings",
    "filing_url",
    "citations",
}

TASK_STATE_SUPPORTED_STATUSES = {"supported", "closed", "done", "complete"}
TASK_STATE_UNRESOLVED_STATUSES = {"empty", "candidate", "partial", "missing"}


def _tool_call_display_name(call: ToolCall) -> str:
    native_name = str(getattr(call, "native_name", "") or "").strip()
    if native_name:
        return native_name
    tool_name = str(getattr(call, "name", "") or "").strip()
    action_name = str(getattr(call, "action", "") or "default").strip() or "default"
    if not tool_name:
        return action_name
    if action_name == "default":
        return tool_name
    if action_name in {"search", "run", "calculate"} and tool_name in {"web_search", "dataframe_query", "calculator"}:
        return tool_name
    return f"{tool_name}_{action_name}"

MARKET_DATA_METADATA_KEYS = (
    "provider",
    "function",
    "requested_function",
    "endpoint",
    "ts_code",
    "index_code",
    "fund_code",
    "contract",
    "series",
    "series_id",
    "series_title",
    "requested_ticker",
    "symbol",
    "query",
    "start_date",
    "end_date",
    "requested_start_date",
    "requested_end_date",
    "requested_start_year",
    "requested_end_year",
    "actual_start_date",
    "actual_end_date",
    "country",
    "requested_country",
    "indicator",
    "source_authority",
    "value_origin",
    "inference_risk",
    "matched_rows",
    "total_rows",
    "result_count",
    "source",
    "url",
    "page_url",
    "api_url",
    "fallback_from",
    "data_source",
    "params",
    "fields",
    "field_schema",
    "semantic_candidates",
    "typed_schema",
    "unit_schema",
    "currency",
    "scale",
    "adjust",
    "amount_unit",
    "volume_unit",
    "date_filter_applied",
    "doc_id",
    "doc_title",
    "publish_date",
    "doc_source",
    "attachment_count",
    "attachment_urls",
    "attachment_errors",
    "source_url",
    "sheet",
    "header_rows",
)

TABLE_METADATA_KEYS = (
    "provider",
    "source",
    "url",
    "page_url",
    "api_url",
    "document_key",
    "document_kind",
    "stored_chars",
    "cached",
    "query",
    "parser",
    "docling_version",
    "table_count",
    "table_index",
    "returned",
    "returned_rows",
    "previewed",
    "preview_rows",
    "unit_hint",
    "period_hint",
    "selection",
    "page_or_sheet",
    "caption",
    "source_url",
    "sheet",
    "doc_id",
    "doc_title",
    "publish_date",
    "doc_source",
    "attachment_count",
    "attachment_urls",
    "attachment_errors",
    "warnings",
    "source_authority",
    "value_origin",
    "inference_risk",
)


def messages_char_count(messages: List[Message]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


def tool_schema_char_count(tools: Any) -> int:
    if not isinstance(tools, list) or not tools:
        return 0
    return len(json_compact(tools))


def json_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_json_value(item) for item in value), key=lambda item: json_compact(item))
    return value


def json_canonical(value: Any) -> str:
    try:
        return json.dumps(
            _canonical_json_value(value),
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        return str(value)


def args_fingerprint(arguments: Dict[str, Any]) -> str:
    digest = hashlib.sha1(json_canonical(arguments).encode("utf-8", errors="replace")).hexdigest()
    return digest[:12]


def should_use_context_system(config: AgentConfig) -> bool:
    return str(config.context_mode or "auto").strip().lower() != "legacy"


def should_render_task_state_view(config: AgentConfig, render_mode: str) -> bool:
    if str(config.context_mode or "").strip().lower() == "legacy":
        return False
    mode = str(config.task_state_render_mode or "view").strip().lower()
    if mode == "full":
        return False
    if mode == "view":
        return True
    # Auto keeps the old full block only for explicit legacy rendering.
    return str(render_mode or "").strip().lower() != "legacy"


def _task_state_list(value: Any) -> List[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    return [copy.deepcopy(value)]


def _task_state_mapping(value: Any, *, scalar_key: str) -> JsonDict:
    if value in (None, "", [], {}):
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {scalar_key: copy.deepcopy(value)}


def _append_task_state_values(target: List[Any], value: Any) -> None:
    for item in _task_state_list(value):
        if item not in target:
            target.append(item)


def _task_state_evidence_items(key: str, value: Any) -> List[Any]:
    values = value if isinstance(value, list) else [value]
    items: List[Any] = []
    for item in values:
        if item in (None, "", [], {}):
            continue
        if isinstance(item, dict):
            items.append(copy.deepcopy(item))
        else:
            items.append({key: copy.deepcopy(item)})
    return items


def _task_state_missing_slots(key: str, value: Any) -> List[JsonDict]:
    if value in (None, "", [], {}):
        return [{"slot_id": key, "status": "missing"}]
    if isinstance(value, list):
        slots: List[JsonDict] = []
        for index, child_value in enumerate(value):
            slots.extend(_task_state_missing_slots(f"{key}[{index}]", child_value))
        return slots
    if not isinstance(value, dict):
        return []
    slots: List[JsonDict] = []
    for child_key, child_value in value.items():
        child_path = f"{key}.{child_key}" if key else str(child_key)
        slots.extend(_task_state_missing_slots(child_path, child_value))
    return slots


def canonicalize_task_state(task_state: Optional[Dict[str, Any]]) -> JsonDict:
    """Normalize model-authored TaskState into the prompt-visible schema.

    Older trajectories commonly use free top-level keys such as ``entity``,
    ``metric``, ``gap``, and ``next``. Preserve their information while moving
    it under the canonical control sections so TaskStateView does not silently
    filter useful working memory.
    """

    if not isinstance(task_state, dict) or not task_state:
        return {}

    source = copy.deepcopy(task_state)
    canonical: JsonDict = {}

    schema_version = copy.deepcopy(source.pop("schema_version", None))
    revision = copy.deepcopy(source.pop("revision", None))
    contract = _task_state_mapping(source.pop("contract", None), scalar_key="summary")
    progress = _task_state_mapping(source.pop("progress", None), scalar_key="status")
    evidence = _task_state_list(source.pop("evidence", None))
    answer_state = _task_state_list(source.pop("answer_state", None))
    answer_pins = _task_state_list(source.pop("answer_pins", None))
    batch_coverage = _task_state_list(source.pop("batch_coverage", None))
    open_slots = _task_state_list(source.pop("open_slots", None))
    blocked_slots = _task_state_list(source.pop("blocked_slots", None))
    gaps = _task_state_list(source.pop("gaps", None))
    checks = _task_state_list(source.pop("checks", None))
    basis_notes = _task_state_list(source.pop("basis_notes", None))
    stale_paths = _task_state_list(source.pop("stale_paths", None))
    next_focus = copy.deepcopy(source.pop("next_focus", None))

    for alias, destination in _TASK_STATE_CONTRACT_ALIASES.items():
        if alias not in source:
            continue
        value = source.pop(alias)
        if destination in {"entities", "metrics", "periods", "units"}:
            existing = contract.get(destination)
            merged = _task_state_list(existing)
            _append_task_state_values(merged, value)
            contract[destination] = merged
        elif value not in (None, "", [], {}):
            contract[destination] = value

    for alias, destination in _TASK_STATE_PROGRESS_ALIASES.items():
        if alias not in source:
            continue
        value = source.pop(alias)
        if value not in (None, "", [], {}):
            progress[destination] = value

    for alias in _TASK_STATE_GAP_ALIASES:
        if alias in source:
            _append_task_state_values(gaps, source.pop(alias))

    for alias in _TASK_STATE_CHECK_ALIASES:
        if alias in source:
            _append_task_state_values(checks, source.pop(alias))

    for alias in _TASK_STATE_NEXT_ALIASES:
        if alias in source and next_focus in (None, "", [], {}):
            next_focus = source.pop(alias)
        else:
            source.pop(alias, None)

    for alias in _TASK_STATE_EVIDENCE_ALIASES:
        if alias in source:
            evidence.extend(_task_state_evidence_items(alias, source.pop(alias)))

    # Dynamic fact keys such as ``kingston_2020_density_sqmi`` remain useful,
    # but they belong under progress.values rather than as invisible root keys.
    if source:
        values = _task_state_mapping(progress.get("values"), scalar_key="value")
        for key, value in source.items():
            if key in TASK_STATE_DERIVED_KEYS:
                continue
            _append_task_state_values(
                open_slots,
                _task_state_missing_slots(str(key), value),
            )
            compacted_value = _drop_empty(value)
            if compacted_value not in (None, "", [], {}):
                values[str(key)] = compacted_value
        if values:
            progress["values"] = values

    candidates = {
        "schema_version": schema_version,
        "revision": revision,
        "contract": contract,
        "progress": progress,
        "evidence": evidence,
        "answer_state": answer_state,
        "answer_pins": answer_pins,
        "batch_coverage": batch_coverage,
        "open_slots": open_slots,
        "blocked_slots": blocked_slots,
        "gaps": gaps,
        "checks": checks,
        "basis_notes": basis_notes,
        "stale_paths": stale_paths,
        "next_focus": next_focus,
    }
    return _drop_empty(candidates)


def task_state_view(task_state: Optional[Dict[str, Any]]) -> JsonDict:
    normalized = canonicalize_task_state(task_state)
    if not normalized:
        return {}
    view: JsonDict = {}
    for key in TASK_STATE_VIEW_KEYS:
        if key in normalized:
            if key == "evidence" and isinstance(normalized[key], list):
                view[key] = _compact_task_state_items(normalized[key], limit=32, char_limit=900)
            elif key == "stale_paths" and isinstance(normalized[key], list):
                view[key] = _compact_task_state_items(normalized[key], limit=24, char_limit=700)
            else:
                view[key] = normalized[key]
    if "evidence" in normalized and isinstance(normalized["evidence"], list):
        view["evidence_count"] = len(normalized["evidence"])
    if "stale_paths" in normalized and isinstance(normalized["stale_paths"], list):
        view["stale_path_count"] = len(normalized["stale_paths"])
    return view


def _task_state_storage_payload(task_state: Optional[Dict[str, Any]]) -> JsonDict:
    normalized = canonicalize_task_state(task_state)
    if not normalized:
        return {}
    return {
        str(key): value
        for key, value in normalized.items()
        if str(key) not in TASK_STATE_DERIVED_KEYS
    }


def _compact_task_state_items(items: List[Any], *, limit: int, char_limit: int) -> List[Any]:
    selected = items[-limit:] if len(items) > limit else items
    return [_compact_coverage_value(item, char_limit) for item in selected]


def task_state_prompt_view(task_state: Optional[Dict[str, Any]]) -> JsonDict:
    """Prompt-facing TaskState view.

    The internal TaskState must keep the full batch coverage history for merge
    correctness. Prompt rendering only needs the unresolved coverage plus a
    recent audit trail, so this view compacts batch_coverage lazily at render
    time instead of mutating the stored state.
    """

    view = task_state_view(task_state)
    coverage = view.get("batch_coverage")
    if isinstance(coverage, list):
        compacted, omitted = _compact_batch_coverage_for_prompt(coverage)
        view["batch_coverage"] = compacted
        if omitted > 0:
            view["batch_coverage_omitted_count"] = omitted
    return view


def task_state_support_audit(task_state: Optional[Dict[str, Any]]) -> JsonDict:
    """Summarize whether TaskState is closed enough to trust in packet mode.

    TaskState is model-owned control state, not source evidence. This audit is
    intentionally conservative but not a verifier: it exposes gaps to the
    prompt/runtime without discarding useful supported slots.
    """

    if not isinstance(task_state, dict) or not task_state:
        return {
            "answer_state_count": 0,
            "supported_count": 0,
            "proven_supported_count": 0,
            "unproven_supported_count": 0,
            "unresolved_count": 0,
            "blocked_count": 0,
            "open_slots_count": 0,
            "blocked_slots_count": 0,
            "can_natural_final": False,
            "blocking_reasons": ["missing_task_state"],
        }

    answer_state = task_state.get("answer_state")
    answer_items = answer_state if isinstance(answer_state, list) else []
    supported: List[str] = []
    proven_supported: List[str] = []
    unproven_supported: List[str] = []
    unresolved: List[str] = []
    blocked: List[str] = []

    for index, item in enumerate(answer_items):
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or item.get("id") or f"slot_{index + 1}")
        status = str(item.get("status") or "").strip().lower()
        if status in TASK_STATE_SUPPORTED_STATUSES:
            supported.append(slot_id)
            if _task_state_slot_has_provenance(item):
                proven_supported.append(slot_id)
            else:
                unproven_supported.append(slot_id)
        elif status == "blocked":
            blocked.append(slot_id)
        elif status in TASK_STATE_UNRESOLVED_STATUSES:
            unresolved.append(slot_id)

    open_slots = task_state.get("open_slots")
    blocked_slots = task_state.get("blocked_slots")
    open_slots_count = len(open_slots) if isinstance(open_slots, list) else 0
    blocked_slots_count = len(blocked_slots) if isinstance(blocked_slots, list) else 0

    reasons: List[str] = []
    if open_slots_count:
        reasons.append("open_slots_present")
    if not answer_items and not blocked_slots_count:
        reasons.append("missing_answer_state")
    if answer_items and supported and not proven_supported:
        reasons.append("supported_slots_without_evidence_refs")
    if answer_items and not supported and not blocked and unresolved:
        reasons.append("only_unresolved_answer_slots")

    return _drop_empty(
        {
            "answer_state_count": len(answer_items),
            "supported_count": len(supported),
            "proven_supported_count": len(proven_supported),
            "unproven_supported_count": len(unproven_supported),
            "unresolved_count": len(unresolved),
            "blocked_count": len(blocked),
            "open_slots_count": open_slots_count,
            "blocked_slots_count": blocked_slots_count,
            "unproven_supported_slots": unproven_supported[:12],
            "unresolved_slots": unresolved[:12],
            "can_natural_final": not reasons,
            "blocking_reasons": reasons,
        }
    )


def _task_state_slot_has_provenance(item: JsonDict) -> bool:
    for key in ("evidence_refs", "calc_refs", "artifact_refs", "source_refs", "depends_on"):
        value = item.get(key)
        if isinstance(value, list) and any(v not in (None, "", [], {}) for v in value):
            return True
        if value not in (None, "", [], {}):
            return True
    basis = item.get("basis") if isinstance(item.get("basis"), dict) else {}
    for key in ("evidence_refs", "calc_refs", "artifact_refs", "document_key", "locator", "row_locator"):
        value = basis.get(key)
        if value not in (None, "", [], {}):
            return True
    # Source fields alone are not source evidence, but exact reader/table
    # locators are strong enough to keep older correct SEC/table trajectories
    # from being downgraded when their evidence ref uses a legacy name.
    source_field = str(item.get("source_field") or basis.get("source_field") or "")
    if any(marker in source_field.lower() for marker in ("table_index", " row ", "row=", "sec_reader", "read_tables")):
        return True
    return False


def _compact_batch_coverage_for_prompt(items: List[Any], *, limit: int = 24) -> Tuple[List[Any], int]:
    if len(items) <= limit:
        return [_compact_batch_coverage_item(item) for item in items], 0

    important_indexes = []
    for idx, item in enumerate(items):
        if _batch_coverage_item_is_open(item):
            important_indexes.append(idx)

    selected_indexes = set(important_indexes[:limit])
    for idx in range(len(items) - 1, -1, -1):
        if len(selected_indexes) >= limit:
            break
        selected_indexes.add(idx)

    selected = [_compact_batch_coverage_item(items[idx]) for idx in sorted(selected_indexes)]
    return selected, max(0, len(items) - len(selected))


def _batch_coverage_item_is_open(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status") or "").strip().lower()
    if status and status not in {"success", "complete", "covered", "success_no_matching_rows"}:
        return True
    if item.get("missing_slots") not in (None, "", [], {}):
        return True
    next_action = str(item.get("next_action") or "").strip().lower()
    if next_action and next_action not in {"final", "answer", "calculate", "done"}:
        return True
    if item.get("can_answer_now") is False:
        return True
    return False


def _compact_batch_coverage_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return compact_text(json_compact(item), 500)

    kept_keys = (
        "coverage_id",
        "tool",
        "action",
        "status",
        "requested",
        "result_rows",
        "filled_slots",
        "filled_candidates",
        "missing_slots",
        "basis",
        "diagnostics",
        "retry_same_plan_allowed",
        "retry_same_args_allowed",
        "can_answer_now",
        "next_action",
    )
    compacted: JsonDict = {}
    for key in kept_keys:
        value = item.get(key)
        if value in (None, "", [], {}):
            continue
        if key in {"filled_slots", "filled_candidates", "missing_slots"} and isinstance(value, list):
            compacted[key] = [_compact_coverage_value(row, 500) for row in value[:8]]
            omitted = len(value) - len(compacted[key])
            if omitted > 0:
                compacted[f"{key}_omitted_count"] = omitted
            continue
        compacted[key] = _compact_coverage_value(value, 900)
    return _drop_empty(compacted)


def _compact_coverage_value(value: Any, char_limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_coverage_value(item, max(220, char_limit // 2))
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_compact_coverage_value(item, max(220, char_limit // 2)) for item in value[:12]]
    text = str(value)
    if len(text) > char_limit:
        return compact_text(text, char_limit)
    return value


def _task_state_item_identity(item: Any) -> str:
    if not isinstance(item, dict):
        return json_compact(item)
    for key in (
        "slot_id",
        "slot",
        "coverage_id",
        "id",
        "key",
        "evidence_id",
        "call_id",
        "calc_id",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return f"{key}:{value}"
    composite = {
        key: item.get(key)
        for key in (
            "entity",
            "period",
            "date",
            "metric",
            "requested_metric",
            "subfield",
            "required_basis",
            "source_field",
            "source",
        )
        if item.get(key) not in (None, "", [], {})
    }
    if composite:
        return "composite:" + json_compact(composite)
    return "item:" + json_compact(item)


def _merge_task_state_list(previous: Any, candidate: Any) -> List[Any]:
    previous_items = previous if isinstance(previous, list) else []
    candidate_items = candidate if isinstance(candidate, list) else []
    merged: List[Any] = []
    positions: Dict[str, int] = {}

    for item in previous_items:
        identity = _task_state_item_identity(item)
        positions[identity] = len(merged)
        merged.append(item)

    for item in candidate_items:
        identity = _task_state_item_identity(item)
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(merged)
            merged.append(item)
            continue
        if isinstance(merged[position], dict) and isinstance(item, dict):
            merged[position] = {**merged[position], **item}
        else:
            merged[position] = item
    return merged


def _merge_task_state_mapping(previous: Any, candidate: Any) -> JsonDict:
    merged = copy.deepcopy(previous) if isinstance(previous, dict) else {}
    if not isinstance(candidate, dict):
        return merged
    for key, value in candidate.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_task_state_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_task_state(previous: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> JsonDict:
    """Merge model TaskState updates without letting answer-critical slots vanish.

    The model is still responsible for reasoning, but omission of a previously
    supported answer slot in a later compact step should not delete that slot.
    Lists that carry answer slots are merged by stable slot/evidence identity;
    returned control lists such as basis_notes replace prior values so obsolete
    source-basis notes can be cleared.
    """

    normalized_candidate = canonicalize_task_state(candidate)
    if not normalized_candidate:
        return _drop_empty(_task_state_storage_payload(previous))

    merged = _task_state_storage_payload(previous)
    candidate_payload = _task_state_storage_payload(normalized_candidate)
    candidate_is_snapshot = (
        candidate_payload.get("schema_version") == TASK_STATE_SCHEMA_VERSION
    )
    for key, value in candidate_payload.items():
        if key in TASK_STATE_LIST_MERGE_KEYS or (
            not candidate_is_snapshot and key in TASK_STATE_LEGACY_PATCH_LIST_MERGE_KEYS
        ):
            merged[key] = _merge_task_state_list(merged.get(key), value)
        elif key in TASK_STATE_MAPPING_MERGE_KEYS:
            merged[key] = _merge_task_state_mapping(merged.get(key), value)
        else:
            merged[key] = value
    return _drop_empty(merged)


def complete_task_state_snapshot(
    previous: Optional[Dict[str, Any]],
    candidate: Optional[Dict[str, Any]],
    *,
    task: Any,
    revision: int,
    next_focus: Any = None,
) -> JsonDict:
    """Build a full action-state snapshot from a model patch and prior state."""

    normalized_revision = max(1, int(revision or 1))
    task_text = compact_text(str(task or "").strip(), 600)
    envelope: JsonDict = {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "revision": normalized_revision,
        "contract": {"task": task_text} if task_text else {"task": "current user task"},
        "progress": {
            "status": "in_progress",
            "completed_actions": max(0, normalized_revision - 1),
        },
        "next_focus": (
            copy.deepcopy(next_focus)
            if next_focus not in (None, "", [], {})
            else "continue the current evidence plan"
        ),
    }
    # The envelope supplies required fields; the model-authored patch wins
    # where it contains a more specific status, contract, or next objective.
    merged = merge_task_state(previous, envelope)
    merged = merge_task_state(merged, candidate or {})
    merged["schema_version"] = TASK_STATE_SCHEMA_VERSION
    merged["revision"] = normalized_revision

    contract = merged.get("contract") if isinstance(merged.get("contract"), dict) else {}
    if not contract.get("task"):
        contract["task"] = task_text or "current user task"
    merged["contract"] = contract

    progress = merged.get("progress") if isinstance(merged.get("progress"), dict) else {}
    progress.setdefault("status", "in_progress")
    progress["completed_actions"] = max(0, normalized_revision - 1)
    merged["progress"] = progress
    if merged.get("next_focus") in (None, "", [], {}):
        merged["next_focus"] = copy.deepcopy(envelope["next_focus"])
    return _drop_empty(merged)


def task_state_snapshot_issues(task_state: Optional[Dict[str, Any]]) -> List[str]:
    """Return structural reasons why a non-terminal state is not a full snapshot."""

    if not isinstance(task_state, dict) or not task_state:
        return ["empty"]
    issues: List[str] = []
    extra = sorted(set(map(str, task_state)) - set(TASK_STATE_VIEW_KEYS))
    if extra:
        issues.append("free_top_level_keys")
    if task_state.get("schema_version") != TASK_STATE_SCHEMA_VERSION:
        issues.append("schema_version")
    revision = task_state.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        issues.append("revision")
    contract = task_state.get("contract")
    if not isinstance(contract, dict) or not str(contract.get("task") or "").strip():
        issues.append("contract.task")
    progress = task_state.get("progress")
    if not isinstance(progress, dict):
        issues.append("progress")
    else:
        if not str(progress.get("status") or "").strip():
            issues.append("progress.status")
        completed = progress.get("completed_actions")
        if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
            issues.append("progress.completed_actions")
    if task_state.get("next_focus") in (None, "", [], {}):
        issues.append("next_focus")
    return issues


def task_state_patch_retention_issues(
    candidate: Optional[Dict[str, Any]],
    snapshot: Optional[Dict[str, Any]],
) -> List[str]:
    """Return candidate paths that were lost while building a full snapshot."""

    expected = canonicalize_task_state(candidate)
    actual = canonicalize_task_state(snapshot)
    issues: List[str] = []

    def visit(expected_value: Any, actual_value: Any, path: str) -> None:
        if path in {"schema_version", "revision", "progress.completed_actions"}:
            return
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                issues.append(path or "<root>")
                return
            for key, value in expected_value.items():
                child = f"{path}.{key}" if path else str(key)
                if key not in actual_value:
                    issues.append(child)
                    continue
                visit(value, actual_value[key], child)
            return
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list):
                issues.append(path)
                return
            actual_by_identity = {
                _task_state_item_identity(item): item
                for item in actual_value
            }
            for item in expected_value:
                identity = _task_state_item_identity(item)
                if identity not in actual_by_identity:
                    issues.append(f"{path}[{identity[:80]}]")
                    continue
                visit(item, actual_by_identity[identity], f"{path}[{identity[:32]}]")
            return
        if expected_value != actual_value:
            issues.append(path)

    visit(expected, actual, "")
    return issues


def evidence_fingerprint(value: Any) -> str:
    if isinstance(value, dict):
        stable = {
            key: item
            for key, item in value.items()
            if key
            not in {
                "evidence_id",
                "fingerprint",
                "origin",
                "ledger_source",
                "first_seen_step",
                "last_seen_step",
            }
        }
    else:
        stable = {"text": str(value or "")}
    digest = hashlib.sha1(json_canonical(stable).encode("utf-8", errors="replace")).hexdigest()
    return digest[:16]


def _search_text_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: item
            for key, item in value.items()
            if key not in {"search_text", "fingerprint", "pin_key"}
        }
    return value


def ledger_search_text(value: Any) -> str:
    if isinstance(value, dict):
        cached = value.get("search_text")
        if isinstance(cached, str) and cached:
            return cached
    return json_compact(_search_text_payload(value)).lower()


def _coerce_step_number(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _evidence_fingerprint_index(state: AgentState) -> Dict[str, Dict[str, Any]]:
    index = getattr(state, "evidence_fingerprint_index", None)
    if not isinstance(index, dict):
        index = {}
        setattr(state, "evidence_fingerprint_index", index)

    initialized = bool(getattr(state, "_evidence_fingerprint_index_initialized", False))
    if not initialized and state.evidence_ledger:
        for entry in state.evidence_ledger:
            if not isinstance(entry, dict):
                continue
            fp = str(entry.get("fingerprint") or "")
            if fp:
                index[fp] = entry
        setattr(state, "_evidence_fingerprint_index_initialized", True)
    elif not initialized:
        setattr(state, "_evidence_fingerprint_index_initialized", True)
    return index


def _evidence_ref_index(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    index = getattr(state, "evidence_ref_index", None)
    if not isinstance(index, dict):
        index = {}
        setattr(state, "evidence_ref_index", index)
    initialized = bool(getattr(state, "_evidence_ref_index_initialized", False))
    if not initialized and state.evidence_ledger:
        for entry in state.evidence_ledger:
            if isinstance(entry, dict):
                _index_evidence_ref(index, entry)
        setattr(state, "_evidence_ref_index_initialized", True)
    elif not initialized:
        setattr(state, "_evidence_ref_index_initialized", True)
    return index


def _index_evidence_ref(index: Dict[str, List[Dict[str, Any]]], entry: Dict[str, Any]) -> None:
    seen_keys = set()
    for key_name in ("evidence_id", "call_id"):
        key = str(entry.get(key_name) or "")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)
    for key_name in ("step", "first_seen_step", "last_seen_step"):
        step = _coerce_step_number(entry.get(key_name))
        if step is None:
            continue
        key = f"step:{step}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)


def _index_evidence_schema_item(state: AgentState, entry: Dict[str, Any]) -> None:
    metric = str(entry.get("metric") or "").lower()
    if "schema" not in metric and entry.get("schema") in (None, "", [], {}):
        return
    items = getattr(state, "evidence_schema_items", None)
    if not isinstance(items, list):
        items = []
        setattr(state, "evidence_schema_items", items)
    items.append(entry)


def record_evidence_items(
    state: AgentState,
    items: Any,
    *,
    origin: str,
    step_number: Optional[int] = None,
) -> List[str]:
    """Append model-extracted evidence into the runtime EvidenceLedger.

    The model may return the same compact evidence on every step. We dedupe by a
    stable content fingerprint so TaskStateView can stay small without losing the
    curated fact cache that older correct trajectories relied on.
    """

    if isinstance(items, dict):
        raw_items: List[Any] = [items]
    elif isinstance(items, list):
        raw_items = items
    else:
        return []

    seen = _evidence_fingerprint_index(state)
    ref_index = _evidence_ref_index(state)

    added_ids: List[str] = []
    for item in raw_items:
        if item in (None, ""):
            continue
        if isinstance(item, dict):
            payload: JsonDict = dict(item)
        else:
            payload = {"text": str(item)}
        if not payload:
            continue

        fp = evidence_fingerprint(payload)
        existing = seen.get(fp)
        if existing is not None:
            if step_number is not None:
                existing["last_seen_step"] = step_number
                existing.pop("search_text", None)
                existing["search_text"] = ledger_search_text(existing)
                _index_evidence_ref(ref_index, existing)
            added_ids.append(str(existing.get("evidence_id") or ""))
            continue

        evidence_id = str(payload.get("evidence_id") or f"ev_{len(state.evidence_ledger) + 1}")
        payload.pop("evidence_id", None)
        record: JsonDict = {
            "evidence_id": evidence_id,
            "fingerprint": fp,
            "origin": origin,
            **payload,
        }
        if step_number is not None:
            record["first_seen_step"] = step_number
            record["last_seen_step"] = step_number
        record["search_text"] = ledger_search_text(record)
        state.evidence_ledger.append(record)
        seen[fp] = record
        _index_evidence_ref(ref_index, record)
        _index_evidence_schema_item(state, record)
        added_ids.append(evidence_id)
    return [item for item in added_ids if item]


ANSWER_PIN_KEEP_KEYS = (
    "pin_id",
    "slot_id",
    "entity",
    "period",
    "date",
    "metric",
    "subfield",
    "value",
    "unit",
    "currency",
    "scale",
    "source",
    "source_ref",
    "source_field",
    "source_authority",
    "value_origin",
    "inference_risk",
    "provider",
    "endpoint",
    "document_key",
    "table_index",
    "row_locator",
    "row_values",
    "call_id",
    "evidence_id",
    "calc_ref",
    "formula",
    "raw_result",
    "rounding",
    "date_basis",
    "adjustment",
    "amount_unit",
    "volume_unit",
    "basis",
    "caveat",
    "status",
    "supersedes",
    "origin",
    "first_seen_step",
    "last_seen_step",
    "quote",
    "text",
)


def _answer_pin_index(state: AgentState) -> Dict[str, Dict[str, Any]]:
    index = getattr(state, "answer_pin_index", None)
    if not isinstance(index, dict):
        index = {}
        setattr(state, "answer_pin_index", index)
    initialized = bool(getattr(state, "_answer_pin_index_initialized", False))
    if not initialized and state.answer_critical_pins:
        for entry in state.answer_critical_pins:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("pin_key") or _answer_pin_identity(entry))
            if key:
                index[key] = entry
        setattr(state, "_answer_pin_index_initialized", True)
    elif not initialized:
        setattr(state, "_answer_pin_index_initialized", True)
    return index


def _answer_pin_ref_index(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    pins = getattr(state, "answer_critical_pins", None)
    if not isinstance(pins, list):
        pins = []
    index = getattr(state, "answer_pin_ref_index", None)
    built_count = getattr(state, "_answer_pin_ref_index_built_count", 0)
    if not isinstance(index, dict) or not isinstance(built_count, int) or built_count > len(pins):
        index = {}
        built_count = 0
        setattr(state, "answer_pin_ref_index", index)

    for entry in pins[built_count:]:
        if isinstance(entry, dict):
            _index_answer_pin_ref(index, entry)
    setattr(state, "_answer_pin_ref_index_built_count", len(pins))
    return index


def _index_answer_pin_ref(index: Dict[str, List[Dict[str, Any]]], entry: Dict[str, Any]) -> None:
    seen_keys = set()
    for key_name in ("pin_id", "call_id", "evidence_id", "calc_ref", "source_ref"):
        key = str(entry.get(key_name) or "")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)
    for key_name in ("step", "first_seen_step", "last_seen_step"):
        step = _coerce_step_number(entry.get(key_name))
        if step is None:
            continue
        key = f"step:{step}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)


def _answer_pin_identity(item: Any) -> str:
    if not isinstance(item, dict):
        return json_canonical(item)
    for key in ("slot_id", "pin_id", "id", "key"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return f"{key}:{value}"
    composite = {
        key: item.get(key)
        for key in (
            "entity",
            "period",
            "date",
            "metric",
            "subfield",
            "source_field",
            "formula",
            "call_id",
        )
        if item.get(key) not in (None, "", [], {})
    }
    if composite:
        return "composite:" + json_canonical(composite)
    return "item:" + json_canonical(item)


def _normalize_answer_pin_payload(item: Any) -> JsonDict:
    if isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {"text": str(item)}
    payload.pop("search_text", None)
    payload.pop("pin_key", None)
    payload.pop("fingerprint", None)
    return _drop_empty(payload)


def record_answer_pins(
    state: AgentState,
    items: Any,
    *,
    origin: str,
    step_number: Optional[int] = None,
) -> List[str]:
    """Persist answer-critical facts in a monotonic runtime lane.

    Pins are prompt-visible memory, not a final-answer gate. Omitting a prior pin
    in a later TaskState update never deletes it; a newer pin with the same
    identity only fills or overwrites non-empty fields for that identity.
    """

    if isinstance(items, dict):
        raw_items: List[Any] = [items]
    elif isinstance(items, list):
        raw_items = items
    else:
        return []

    index = _answer_pin_index(state)
    ref_index = _answer_pin_ref_index(state)
    added_ids: List[str] = []
    for item in raw_items:
        if item in (None, ""):
            continue
        payload = _normalize_answer_pin_payload(item)
        if not payload:
            continue
        identity = _answer_pin_identity(payload)
        if not identity:
            continue
        existing = index.get(identity)
        if existing is None:
            pin_id = str(payload.get("pin_id") or f"ap_{len(state.answer_critical_pins) + 1}")
            payload.pop("pin_id", None)
            record: JsonDict = {
                "pin_id": pin_id,
                "pin_key": identity,
                "origin": origin,
                **payload,
            }
            if step_number is not None:
                record["first_seen_step"] = step_number
                record["last_seen_step"] = step_number
            record["fingerprint"] = evidence_fingerprint(record)
            record["search_text"] = ledger_search_text(record)
            state.answer_critical_pins.append(record)
            index[identity] = record
            _index_answer_pin_ref(ref_index, record)
            setattr(state, "_answer_pin_ref_index_built_count", len(state.answer_critical_pins))
            added_ids.append(pin_id)
            continue

        for key, value in payload.items():
            if value not in (None, "", [], {}):
                existing[key] = value
        existing.setdefault("origin", origin)
        if step_number is not None:
            existing.setdefault("first_seen_step", step_number)
            existing["last_seen_step"] = step_number
        existing["fingerprint"] = evidence_fingerprint(existing)
        existing["search_text"] = ledger_search_text(existing)
        _index_answer_pin_ref(ref_index, existing)
        added_ids.append(str(existing.get("pin_id") or ""))

    if added_ids:
        state.context_stats["answer_critical_pin_count"] = len(state.answer_critical_pins)
    return [item for item in added_ids if item]


def _iter_strings(value: Any) -> Iterable[str]:
    if value in (None, ""):
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)
        return
    yield str(value)


def _extract_terms(task_prompt: str, arguments: Dict[str, Any]) -> List[str]:
    text_parts = [task_prompt or ""]
    for key in (
        "query",
        "ticker",
        "symbol",
        "code",
        "name",
        "company",
        "entity",
        "search_query",
    ):
        if key in arguments:
            text_parts.extend(_iter_strings(arguments.get(key)))
    text = "\n".join(part for part in text_parts if part)

    terms: List[str] = []
    for raw in re.split(r"[\s,;，；、。！？!?()\[\]{}<>《》:：\"'|/\\\\]+", text):
        item = raw.strip()
        if 2 <= len(item) <= 40:
            terms.append(item)
    terms.extend(re.findall(r"\b[A-Za-z]{1,6}\d{0,4}(?:\.[A-Za-z]{1,4})?\b", text))
    terms.extend(re.findall(r"\b\d{4,6}\b", text))

    cleaned: List[str] = []
    seen = set()
    stopwords = {
        "status",
        "success",
        "error",
        "market",
        "data",
        "source",
        "query",
        "limit",
        "function",
        "provider",
    }
    for term in terms:
        term = str(term or "").strip()
        if len(term) < 2 or term.lower() in stopwords:
            continue
        if term not in seen:
            seen.add(term)
            cleaned.append(term)
    return cleaned[:80]


def _rows_to_csv(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return ""
    if columns is None:
        seen: Dict[str, None] = {}
        for row in rows:
            for key, value in row.items():
                if value is not None:
                    seen.setdefault(str(key), None)
        columns = list(seen)
    columns = _dedupe_columns([str(column or "") for column in columns])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "N/A" if row.get(key) is None else row.get(key) for key in columns})
    return output.getvalue().strip()


def _row_lite(row: Dict[str, Any], columns: List[str], *, max_fields: int = 24) -> JsonDict:
    if not isinstance(row, dict):
        return {"value": row}
    output: JsonDict = {}
    ordered = list(columns or [])
    for key in row:
        if key not in ordered:
            ordered.append(str(key))
    for key in ordered:
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        output[str(key)] = value
        if len(output) >= max_fields:
            break
    return output


def _column_name(column: Any, index: int) -> str:
    if isinstance(column, dict):
        for key in ("name", "label", "text", "title", "header", "column", "field", "key"):
            value = column.get(key)
            if value not in (None, "", [], {}):
                return str(value).strip()
        value = column.get("index")
        if value not in (None, ""):
            return f"column_{value}"
    elif column not in (None, ""):
        return str(column).strip()
    return f"column_{index}"


def _dedupe_columns(columns: List[str]) -> List[str]:
    output: List[str] = []
    counts: Dict[str, int] = {}
    for index, column in enumerate(columns):
        base = str(column or "").strip() or f"column_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        output.append(base if count == 1 else f"{base}_{count}")
    return output


def _normalize_columns(raw_columns: Any) -> List[str]:
    if not isinstance(raw_columns, list):
        return []
    return _dedupe_columns([_column_name(column, index) for index, column in enumerate(raw_columns)])


def _append_column(columns: List[str], column: str) -> None:
    if column and column not in columns:
        columns.append(column)


def _cell_column_name(cell: Dict[str, Any], columns: List[str], index: int) -> str:
    column_index = cell.get("column_index", cell.get("col_index", cell.get("col_start")))
    try:
        parsed_index = int(column_index)
    except (TypeError, ValueError):
        parsed_index = None
    if parsed_index is not None and 0 <= parsed_index < len(columns):
        return columns[parsed_index]
    for key in ("column", "column_name", "name", "label", "header", "field", "key"):
        value = cell.get(key)
        if value not in (None, "", [], {}):
            return str(value).strip()
    if parsed_index is not None:
        return f"column_{parsed_index}"
    return f"cell_{index}"


def _infer_columns(rows: List[Any], raw_columns: Any) -> List[str]:
    columns = _normalize_columns(raw_columns)
    if columns:
        return columns

    max_width = 0
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("cells"), list):
            cells = row.get("cells") or []
            max_width = max(max_width, len(cells))
            for index, cell in enumerate(cells):
                if isinstance(cell, dict):
                    _append_column(columns, _cell_column_name(cell, columns, index))
            continue
        if isinstance(row, dict):
            for key, value in row.items():
                if key == "cells" or value is None:
                    continue
                _append_column(columns, str(key))
            continue
        if isinstance(row, (list, tuple)):
            max_width = max(max_width, len(row))

    if not columns and max_width:
        columns = [f"column_{index}" for index in range(max_width)]
    return _dedupe_columns(columns)


def _put_row_value(row: Dict[str, Any], key: str, value: Any) -> None:
    if not key:
        return
    if key not in row:
        row[key] = value
        return
    suffix = 2
    while f"{key}_{suffix}" in row:
        suffix += 1
    row[f"{key}_{suffix}"] = value


def _normalize_cell_row(raw_row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    row: JsonDict = {}
    for key in ("row", "row_index", "row_number", "table_index", "page", "page_or_sheet", "sheet", "source_url"):
        value = raw_row.get(key)
        if value not in (None, "", [], {}):
            row[str(key)] = value

    cells = raw_row.get("cells") or []
    for index, cell in enumerate(cells):
        if isinstance(cell, dict):
            key = _cell_column_name(cell, columns, index)
            value = cell.get("value")
            if value is None:
                value = cell.get("text", cell.get("content", ""))
            _put_row_value(row, key, value)
        else:
            key = columns[index] if index < len(columns) else f"column_{index}"
            _put_row_value(row, key, cell)
    return row


def _normalize_table_rows(raw_rows: List[Any], columns: List[str]) -> List[Dict[str, Any]]:
    dict_rows: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, dict) and isinstance(raw_row.get("cells"), list):
            dict_rows.append(_normalize_cell_row(raw_row, columns))
        elif isinstance(raw_row, dict):
            dict_rows.append({str(key): value for key, value in raw_row.items()})
        elif isinstance(raw_row, (list, tuple)):
            dict_rows.append(
                {
                    (columns[index] if index < len(columns) else f"column_{index}"): value
                    for index, value in enumerate(raw_row)
                }
            )
    return dict_rows


def _table_rows(result: ToolResult) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not result.tables:
        return [], []
    table = result.tables[0]
    if not isinstance(table, dict):
        return [], []
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        returned_rows = table.get("returned_rows")
        if isinstance(returned_rows, list):
            rows = returned_rows
    columns = table.get("columns")
    raw_had_columns = bool(_normalize_columns(columns))
    if not isinstance(rows, list) or not rows:
        if all(isinstance(item, dict) for item in result.tables) and any(
            isinstance(item, dict) and item.get("table_index") not in (None, "")
            for item in result.tables
        ):
            catalogue_rows = [item for item in result.tables if isinstance(item, dict)]
            catalogue_columns = _infer_columns(catalogue_rows, None)
            return catalogue_columns, _normalize_table_rows(catalogue_rows, catalogue_columns)
        return _normalize_columns(columns), []
    has_cell_rows = any(isinstance(row, dict) and isinstance(row.get("cells"), list) for row in rows)
    normalized_columns = _infer_columns(rows, columns)
    dict_rows = _normalize_table_rows(rows, normalized_columns)
    if not normalized_columns and dict_rows:
        normalized_columns = _infer_columns(dict_rows, None)
    if not raw_had_columns or has_cell_rows:
        for row in dict_rows:
            for key, value in row.items():
                if value is not None:
                    _append_column(normalized_columns, str(key))
    return normalized_columns, dict_rows


def _json_object(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text.startswith("{"):
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _summary_key_arguments(arguments: Dict[str, Any]) -> JsonDict:
    keep = (
        "query",
        "search_query",
        "ticker",
        "symbol",
        "code",
        "index_code",
        "fund_code",
        "company",
        "entity",
        "function",
        "provider",
        "start_date",
        "end_date",
        "date",
        "trade_date",
        "period",
        "fields",
        "document_key",
        "table_index",
        "row_start",
        "row_end",
        "url",
        "limit",
    )
    return {key: arguments.get(key) for key in keep if arguments.get(key) not in (None, "", [], {})}


def _argument_entity(arguments: Dict[str, Any]) -> str:
    for key in ("ticker", "symbol", "code", "index_code", "fund_code", "company", "entity", "function", "url"):
        value = arguments.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _argument_period(arguments: Dict[str, Any]) -> str:
    start = arguments.get("start_date")
    end = arguments.get("end_date")
    if start or end:
        return f"{start or ''}..{end or ''}"
    for key in ("period", "date", "trade_date", "ann_date", "record_date"):
        value = arguments.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _copy_keys(source: Dict[str, Any], keys: Iterable[str]) -> JsonDict:
    return {key: source.get(key) for key in keys if source.get(key) not in (None, "", [], {})}


def _schema_data_columns(columns: List[str]) -> List[str]:
    locator_columns = {
        "row",
        "row_index",
        "row_number",
        "table_index",
        "page",
        "page_or_sheet",
        "sheet",
        "source",
        "source_url",
        "url",
        "document_key",
        "column_index",
    }
    selected = [str(column) for column in columns if str(column or "").strip().lower() not in locator_columns]
    return selected or columns


def _date_basis_from_columns(columns: List[str]) -> str:
    lowered = {str(column or "").strip().lower(): str(column or "").strip() for column in columns}
    for candidate in (
        "trade_date",
        "date",
        "observation_date",
        "end_date",
        "ann_date",
        "f_ann_date",
        "nav_date",
        "period",
        "cal_date",
        "pub_date",
        "month",
        "quarter",
    ):
        if candidate in lowered:
            return lowered[candidate]
    return ""


def _actual_date_range_from_rows(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return ""
    basis = _date_basis_from_columns(columns)
    candidates = [basis] if basis else []
    candidates.extend(
        [
            "date",
            "observation_date",
            "trade_date",
            "cal_date",
            "end_date",
            "ann_date",
            "f_ann_date",
            "nav_date",
            "period",
            "year",
            "month",
            "quarter",
        ]
    )
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        key = str(candidate or "")
        lowered = key.lower()
        if not key or lowered in seen:
            continue
        seen.add(lowered)
        ordered_candidates.append(key)
    for candidate in ordered_candidates:
        values = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get(candidate)
            if value in (None, "", [], {}):
                lower_map = {str(key).lower(): key for key in row}
                value = row.get(lower_map.get(candidate.lower(), ""))
            if value in (None, "", [], {}):
                continue
            values.append(str(value))
        if values:
            return f"{min(values)}..{max(values)}"
    return ""


def _field_unit_for_market_column(action: str, endpoint: str, column: str, metadata: Dict[str, Any]) -> str:
    unit_schema = metadata.get("unit_schema")
    if isinstance(unit_schema, dict):
        explicit = unit_schema.get(column) or unit_schema.get(str(column).lower())
        if explicit not in (None, "", [], {}):
            return str(explicit)

    lowered = str(column or "").strip().lower()
    amount_unit = metadata.get("amount_unit")
    volume_unit = metadata.get("volume_unit")
    if lowered in {"amount", "amt", "成交额"} and amount_unit not in (None, "", [], {}):
        return str(amount_unit)
    if lowered in {"vol", "volume", "成交量"} and volume_unit not in (None, "", [], {}):
        return str(volume_unit)
    if lowered in {"pct_chg", "pct_change", "change_pct", "涨跌幅", "turnover_rate", "turnover_rate_f"}:
        return "percent"
    if lowered in {"pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "volume_ratio"}:
        return "ratio"
    if lowered in {"total_mv", "circ_mv", "float_mv"} and endpoint == "daily_basic":
        return "10k CNY"
    if lowered in {"nav", "unit_nav", "accum_nav", "adj_nav"}:
        return "CNY/unit"
    if lowered in {"open", "high", "low", "close", "pre_close", "change", "price"}:
        if action == "index_history" or endpoint == "index_daily":
            return "index points"
        if action in {"equity_price_history", "fund_history"} or endpoint in {"daily", "fund_daily"}:
            return "CNY/share_or_unit"
        return "source price unit"
    return ""


def _market_field_units(action: str, endpoint: str, columns: List[str], metadata: Dict[str, Any]) -> JsonDict:
    units: JsonDict = {}
    for column in columns:
        unit = _field_unit_for_market_column(action, endpoint, column, metadata)
        if unit:
            units[column] = unit
    return units


def _market_requested_fields(call: ToolCall, metadata: Dict[str, Any]) -> Any:
    params = metadata.get("params")
    if isinstance(params, dict) and params.get("fields") not in (None, "", [], {}):
        return params.get("fields")
    if metadata.get("fields") not in (None, "", [], {}):
        return metadata.get("fields")
    if call.arguments.get("fields") not in (None, "", [], {}):
        return call.arguments.get("fields")
    return None


def _market_entity(call: ToolCall, metadata: Dict[str, Any]) -> str:
    for key in ("ts_code", "index_code", "fund_code", "contract", "series", "symbol", "requested_ticker", "ticker", "code"):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return _argument_entity(call.arguments)


def _source_domain(source: Any) -> str:
    try:
        return urlparse(str(source or "")).netloc.lower()
    except Exception:
        return ""


def _source_authority(call: ToolCall, result: ToolResult, source: Any = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    provider = str(result.provider or metadata.get("provider") or "").lower()
    tool = str(call.name or "").lower()
    action = str(call.action or "").lower()
    domain = _source_domain(source or metadata.get("url") or metadata.get("source") or metadata.get("page_url") or metadata.get("api_url"))
    joined = " ".join(str(value or "").lower() for value in (source, metadata.get("source"), metadata.get("doc_source"), metadata.get("doc_title")))

    if tool == "sec_reader" or "sec" in provider or "sec.gov" in domain:
        return "official_filing"
    if provider in {"world_bank", "fred", "alfred", "oecd", "iea", "comtrade", "wits", "lbma", "safe", "nfra", "tinyshare", "tushare", "tushare_http"}:
        return "official_structured_provider"
    if any(token in domain for token in ("gov", "worldbank.org", "stlouisfed.org", "federalreserve.gov", "cninfo.com.cn", "sse.com.cn", "szse.cn", "hkexnews.hk")):
        return "official_source"
    if tool == "structured_table_reader" and action == "discover_tables":
        return "source_discovery"
    if tool == "structured_table_reader" and action in {"read_tables", "default"}:
        return "table_document"
    if any(token in joined or token in domain for token in ("rating", "ratings", "research", "news", "summary", "reporter", "prnewswire", "businesswire")):
        return "secondary_summary"
    if tool == "web_reader":
        return "narrative_source"
    if tool == "web_search":
        return "search_snippet"
    return ""


def _value_origin(call: ToolCall, result: ToolResult) -> str:
    tool = str(call.name or "").lower()
    action = str(call.action or "").lower()
    provider = str(result.provider or "").lower()
    if tool == "market_data":
        return "structured_rows"
    if tool == "structured_table_reader" and action == "discover_tables":
        return "source_discovery"
    if tool == "structured_table_reader" or (tool == "sec_reader" and "table" in action):
        return "exact_table_extraction"
    if action == "official_attachment_table" or provider in {"safe", "nfra"}:
        return "official_attachment_table"
    if tool == "web_reader":
        return "reader_summary"
    if tool == "web_search":
        return "search_snippet"
    if tool == "calculator":
        return "calculation"
    return ""


def _inference_risk_text(text: Any) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("inferred", "back-solved", "back solved", "reverse", "yoy", "year-over-year", "year over year", "同比", "反推", "倒推", "增长率")):
        return "possible_inference_or_derived_value"
    if any(token in lowered for token in ("rating report", "research report", "news", "summary", "评级报告", "研究报告", "新闻", "摘要")):
        return "secondary_source_summary"
    return ""


def _inference_risk(call: ToolCall, result: ToolResult, source: Any = None, metadata: Optional[Dict[str, Any]] = None, text: Any = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    for value in (
        text,
        source,
        metadata.get("source"),
        metadata.get("url"),
        metadata.get("doc_title"),
        metadata.get("doc_source"),
        call.arguments,
    ):
        risk = _inference_risk_text(value)
        if risk:
            return risk
    return ""


def market_data_observation_schema(call: ToolCall, result: ToolResult) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    columns, rows = _table_rows(result)
    data_columns = _schema_data_columns(columns)
    endpoint = str(metadata.get("endpoint") or metadata.get("function") or "")
    params = metadata.get("params") if isinstance(metadata.get("params"), dict) else {}
    start_date = (
        params.get("start_date")
        or metadata.get("requested_start_date")
        or metadata.get("start_date")
        or metadata.get("requested_start_year")
        or call.arguments.get("start_date")
        or call.arguments.get("year")
    )
    end_date = (
        params.get("end_date")
        or metadata.get("requested_end_date")
        or metadata.get("end_date")
        or metadata.get("requested_end_year")
        or call.arguments.get("end_date")
        or call.arguments.get("year")
        or start_date
    )
    requested_date_range = f"{start_date or ''}..{end_date or ''}" if start_date or end_date else _argument_period(call.arguments)
    actual_date_range = (
        f"{metadata.get('actual_start_date')}..{metadata.get('actual_end_date')}"
        if metadata.get("actual_start_date") or metadata.get("actual_end_date")
        else _actual_date_range_from_rows(rows, columns)
    )
    basis: JsonDict = {
        "adjustment": metadata.get("adjust"),
        "date_basis": _date_basis_from_columns(columns),
        "data_source": metadata.get("data_source"),
    }
    schema: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider or metadata.get("provider"),
        "endpoint": endpoint,
        "entity": _market_entity(call, metadata),
        "source_authority": _source_authority(call, result, metadata.get("url") or metadata.get("source"), metadata),
        "value_origin": _value_origin(call, result),
        "inference_risk": _inference_risk(call, result, metadata.get("url") or metadata.get("source"), metadata, rows[:3]),
        "date_range": actual_date_range or requested_date_range,
        "requested_date_range": requested_date_range if requested_date_range and requested_date_range != actual_date_range else None,
        "requested_fields": _market_requested_fields(call, metadata),
        "columns": data_columns[:80],
        "row_count": len(rows) if rows else metadata.get("result_count"),
        "field_units": _market_field_units(str(call.action or ""), endpoint, data_columns, metadata),
        "basis": basis,
        "params": params,
        "metadata": _copy_keys(metadata, MARKET_DATA_METADATA_KEYS),
    }
    return _drop_empty(schema)


def _first_table(result: ToolResult) -> JsonDict:
    if not result.tables:
        return {}
    table = result.tables[0]
    return table if isinstance(table, dict) else {}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def table_observation_schema(call: ToolCall, result: ToolResult) -> JsonDict:
    columns, rows = _table_rows(result)
    if not columns and not rows:
        return {}
    data_columns = _schema_data_columns(columns)
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    table = _first_table(result)
    unit_hint = _first_nonempty(metadata.get("unit_hint"), table.get("unit_hint"))
    period_hint = _first_nonempty(metadata.get("period_hint"), table.get("period_hint"))
    table_index = _first_nonempty(metadata.get("table_index"), table.get("table_index"))
    source = _first_nonempty(metadata.get("url"), metadata.get("source"), table.get("source"), table.get("source_url"))
    page_or_sheet = _first_nonempty(metadata.get("page_or_sheet"), table.get("page_or_sheet"), table.get("sheet"))
    schema: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider or metadata.get("provider"),
        "source": source,
        "source_authority": _source_authority(call, result, source, metadata),
        "value_origin": _value_origin(call, result),
        "inference_risk": _inference_risk(call, result, source, metadata, rows[:3]),
        "document_key": metadata.get("document_key"),
        "table_index": table_index,
        "table_count": metadata.get("table_count"),
        "columns": data_columns[:80],
        "row_count": len(rows) if rows else _first_nonempty(table.get("row_count"), metadata.get("returned_rows")),
        "unit_hint": unit_hint,
        "period_hint": period_hint,
        "basis": {
            "date_basis": _date_basis_from_columns(columns),
            "unit_hint": unit_hint,
            "period_hint": period_hint,
            "page_or_sheet": page_or_sheet,
            "caption": table.get("caption"),
        },
        "execution_card": metadata.get("execution_card") if isinstance(metadata.get("execution_card"), dict) else None,
        "selection": table.get("selection") or metadata.get("selection"),
        "metadata": _copy_keys(metadata, TABLE_METADATA_KEYS),
    }
    return _drop_empty(schema)


def tool_observation_schema(call: ToolCall, result: ToolResult) -> JsonDict:
    if call.name == "market_data":
        return market_data_observation_schema(call, result)
    if result.tables:
        return table_observation_schema(call, result)
    return {}


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: JsonDict = {}
        for key, item in value.items():
            compacted = _drop_empty(item)
            if compacted in (None, "", [], {}):
                continue
            cleaned[key] = compacted
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            compacted = _drop_empty(item)
            if compacted in (None, "", [], {}):
                continue
            cleaned_list.append(compacted)
        return cleaned_list
    return value


def _parse_numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.strip("()").replace(",", "").replace("$", "").replace("¥", "").replace("￥", "")
    cleaned = cleaned.replace("%", "").strip()
    if not re.match(r"^-?\d+(?:\.\d+)?$", cleaned):
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -number if negative else number


def _numeric_extrema(rows: List[Dict[str, Any]], columns: List[str], *, max_columns: int = 6) -> List[JsonDict]:
    if not rows or not columns:
        return []
    skip_tokens = {
        "date",
        "time",
        "period",
        "year",
        "quarter",
        "code",
        "symbol",
        "ticker",
        "name",
        "url",
        "source",
        "form",
        "cik",
        "row",
        "row_index",
        "row_number",
        "table_index",
        "page",
        "page_or_sheet",
        "sheet",
        "column_index",
    }
    preferred_tokens = (
        "amount",
        "volume",
        "vol",
        "value",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct",
        "turnover",
        "price",
        "mv",
        "market",
        "margin",
        "ratio",
        "rate",
        "profit",
        "revenue",
        "income",
        "shares",
    )
    candidates: List[Tuple[int, str, List[Tuple[float, Dict[str, Any]]]]] = []
    for column in columns:
        name = str(column)
        lowered = name.lower()
        if any(token in lowered for token in skip_tokens):
            continue
        values: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            number = _parse_numeric(row.get(column))
            if number is not None:
                values.append((number, row))
        if len(values) < 2:
            continue
        preference = 1 if any(token in lowered for token in preferred_tokens) else 0
        candidates.append((preference, name, values))
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    output: List[JsonDict] = []
    for _, column, values in candidates[:max_columns]:
        minimum = min(values, key=lambda item: item[0])
        maximum = max(values, key=lambda item: item[0])
        output.append(
            {
                "column": column,
                "min": {"value": minimum[0], "row": _row_identity(minimum[1])},
                "max": {"value": maximum[0], "row": _row_identity(maximum[1])},
            }
        )
    return output


def _row_identity(row: Dict[str, Any]) -> JsonDict:
    if not isinstance(row, dict):
        return {"row": row}
    preferred = (
        "trade_date",
        "date",
        "period",
        "period_end",
        "fiscal_period",
        "ts_code",
        "symbol",
        "ticker",
        "name",
        "line_item",
        "metric",
        "item",
        "value",
        "amount",
        "close",
        "unit",
    )
    selected = {key: row.get(key) for key in preferred if row.get(key) not in (None, "", [], {})}
    if selected:
        return selected
    compact: JsonDict = {}
    for key, value in row.items():
        if value not in (None, "", [], {}):
            compact[str(key)] = value
        if len(compact) >= 8:
            break
    return compact


def _table_preview_summary(
    result: ToolResult,
    task_prompt: str,
    arguments: Dict[str, Any],
    *,
    max_rows: int = 10,
    max_scan_rows: int = 800,
) -> JsonDict:
    preview = selected_table_preview(
        result,
        task_prompt,
        arguments,
        max_rows=max_rows,
        max_scan_rows=max_scan_rows,
    )
    columns, rows = _table_rows(result)
    if rows:
        preview["numeric_extrema"] = _numeric_extrema(rows, columns)
    return preview


def selected_table_preview(
    result: ToolResult,
    task_prompt: str,
    arguments: Dict[str, Any],
    *,
    max_rows: int = 16,
    max_scan_rows: int = 600,
) -> JsonDict:
    columns, rows = _table_rows(result)
    if not rows:
        return {}
    scan_limit = max(0, int(max_scan_rows or 0))
    scan_rows = rows if scan_limit <= 0 else rows[:scan_limit]
    terms = _extract_terms(task_prompt, arguments)
    matched: List[Dict[str, Any]] = []
    if terms:
        lowered_terms = [term.lower() for term in terms]
        for row in scan_rows:
            row_text = json_compact(row).lower()
            if any(term.lower() in row_text for term in lowered_terms):
                matched.append(row)
                if len(matched) >= max_rows:
                    break
    if matched:
        selected = matched
        mode = "task_term_match"
    else:
        head_count = min(max_rows, len(rows))
        selected = rows[:head_count]
        mode = "head"
    return {
        "columns": columns[:80],
        "total_rows_in_result": len(rows),
        "scanned_rows": len(scan_rows),
        "selected_mode": mode,
        "selected_rows": [_row_lite(row, columns) for row in selected],
        "selected_rows_csv": _rows_to_csv(selected, columns=columns)[:16000],
    }


def store_artifact(
    state: AgentState,
    *,
    kind: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> JsonDict:
    artifact_index = int(state.context_stats.get("artifact_count", 0) or 0) + 1
    artifact_id = f"artifact:{artifact_index}"
    record = {
        "artifact_id": artifact_id,
        "kind": kind,
        "chars": len(content or ""),
        "sha1": hashlib.sha1((content or "").encode("utf-8", errors="replace")).hexdigest()[:16],
        "metadata": dict(metadata or {}),
        "content": content or "",
    }
    state.artifact_store[artifact_id] = record
    state.context_stats["artifact_count"] = artifact_index
    state.context_stats["artifact_chars"] = int(state.context_stats.get("artifact_chars", 0) or 0) + record["chars"]
    max_chars = int(state.context_stats.get("max_artifact_chars", 0) or 0)
    state.context_stats["max_artifact_chars"] = max(max_chars, record["chars"])
    ref = {key: value for key, value in record.items() if key != "content"}
    return ref


def build_artifact_preview(
    state: AgentState,
    call: ToolCall,
    result: ToolResult,
    full_observation: str,
) -> Tuple[str, str]:
    metadata = {
        "tool": call.name,
        "function": _tool_call_display_name(call),
        "action": call.action,
        "status": result.status,
        "provider": result.provider,
        "arguments_fingerprint": args_fingerprint(call.arguments),
        "result_metadata": result.metadata or {},
    }
    ref = store_artifact(
        state,
        kind="tool_observation",
        content=full_observation,
        metadata=metadata,
    )
    table_preview = selected_table_preview(
        result,
        state.task_context.task_prompt,
        call.arguments,
    )
    preview: JsonDict = {
        "artifact": ref,
        "tool": {"function": _tool_call_display_name(call), "status": result.status},
        "replay": {
            "function": _tool_call_display_name(call),
            "arguments": call.arguments,
            "note": "Full observation is retained in RawTrace/ArtifactStore; replay with a narrower query, limit, document_key, table_index, row range, or the same arguments if needed.",
        },
    }
    if table_preview:
        preview["table_preview"] = table_preview
    else:
        preview["text_preview"] = compact_text(full_observation, state.config.context_artifact_preview_chars)
    return (
        "Large tool observation stored as artifact:\n"
        + json.dumps(preview, ensure_ascii=False, default=str, indent=2),
        str(ref["artifact_id"]),
    )


def _section(text: str, start_label: str, end_labels: Iterable[str]) -> str:
    lowered = text.lower()
    start = lowered.find(start_label.lower())
    if start < 0:
        return ""
    start += len(start_label)
    end = len(text)
    for label in end_labels:
        pos = lowered.find(label.lower(), start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip(" \n:-")


def _reader_summary_payload(call: ToolCall, result: ToolResult, *, text_limit: int = 3200) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    text = result.observation_text()
    evidence = _section(text, "EVIDENCE", ("KEY_VALUES", "COVERAGE", "SUMMARY"))
    key_values = _section(text, "KEY_VALUES", ("COVERAGE", "SUMMARY"))
    coverage = _section(text, "COVERAGE", ("SUMMARY",))
    summary = _section(text, "SUMMARY", ())
    payload: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider,
        "query": call.arguments.get("query") or call.arguments.get("search_query"),
        "document_key": metadata.get("document_key"),
        "source": metadata.get("url") or metadata.get("source"),
        "source_authority": _source_authority(call, result, metadata.get("url") or metadata.get("source"), metadata),
        "value_origin": _value_origin(call, result),
        "inference_risk": _inference_risk(call, result, metadata.get("url") or metadata.get("source"), metadata, text),
        "stored_chars": metadata.get("stored_chars"),
        "cached": metadata.get("cached"),
    }
    if evidence:
        payload["evidence"] = compact_text(evidence, text_limit)
    if key_values:
        payload["key_values"] = compact_text(key_values, 1200)
    if coverage:
        payload["coverage"] = compact_text(coverage, 1000)
    if summary:
        payload["summary"] = compact_text(summary, 1000)
    if not any(key in payload for key in ("evidence", "key_values", "coverage", "summary")):
        payload["evidence"] = compact_text(text, text_limit)
    return _drop_empty(payload)


def _web_search_payload(call: ToolCall, result: ToolResult, *, result_limit: int = 6) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    query = call.arguments.get("query") or call.arguments.get("search_query")
    rows: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(result.observation_text() or "[]")
    except Exception:
        parsed = []
    if isinstance(parsed, list):
        for item in parsed[: max(1, result_limit)]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("link") or "")
            domain = ""
            if url:
                try:
                    domain = urlparse(url).netloc
                except Exception:
                    domain = ""
            rows.append(
                _drop_empty(
                    {
                        "title": item.get("title"),
                        "url": url,
                        "domain": domain,
                        "source_authority": _source_authority(call, result, url, {"provider": item.get("provider") or result.provider}),
                        "inference_risk": _inference_risk(call, result, url, {}, item.get("snippet")),
                        "snippet": compact_text(str(item.get("snippet") or ""), 360),
                        "date": item.get("date"),
                        "provider": item.get("provider") or result.provider,
                    }
                )
            )
    payload: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider,
        "query": query,
        "result_count": metadata.get("result_count") if metadata.get("result_count") is not None else len(rows),
        "value_origin": _value_origin(call, result),
        "top_results": rows,
    }
    return _drop_empty(payload)


def _sec_tables_payload(call: ToolCall, result: ToolResult, *, table_limit: int = 8, row_limit: int = 8) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    obs = _json_object(result.observation)
    payload: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider,
        "query": metadata.get("query") or call.arguments.get("query"),
        "document_key": metadata.get("document_key") or obs.get("document_key"),
        "source": metadata.get("url") or obs.get("url"),
        "table_count": metadata.get("table_count"),
        "table_index": metadata.get("table_index") or obs.get("table_index"),
    }
    if isinstance(obs.get("tables"), list):
        tables = [item for item in obs.get("tables") if isinstance(item, dict)]
        payload["table_count"] = payload.get("table_count") or len(tables)
        candidates = [
            item
            for item in tables
            if item.get("preview_rows") not in (None, "", [], {}) or int(item.get("score") or 0) > 0
        ]
        if not candidates:
            candidates = tables[:table_limit]
        selected = []
        for item in candidates[:table_limit]:
            selected.append(
                {
                    "table_index": item.get("table_index"),
                    "score": item.get("score"),
                    "row_count": item.get("row_count"),
                    "column_count": item.get("column_count"),
                    "columns": (item.get("columns") or [])[:16] if isinstance(item.get("columns"), list) else item.get("columns"),
                    "unit_hint": item.get("unit_hint"),
                    "period_hint": item.get("period_hint"),
                    "matching_row_numbers": (item.get("matching_row_numbers") or [])[:20]
                    if isinstance(item.get("matching_row_numbers"), list)
                    else item.get("matching_row_numbers"),
                    "preview_rows": (item.get("preview_rows") or [])[:row_limit]
                    if isinstance(item.get("preview_rows"), list)
                    else item.get("preview_rows"),
                }
            )
        payload["candidate_tables"] = selected
    if isinstance(obs.get("returned_rows"), list):
        payload.update(
            {
                "row_count": obs.get("row_count"),
                "column_count": obs.get("column_count"),
                "columns": (obs.get("columns") or [])[:24] if isinstance(obs.get("columns"), list) else obs.get("columns"),
                "unit_hint": obs.get("unit_hint"),
                "period_hint": obs.get("period_hint"),
                "selection": obs.get("selection"),
                "returned_rows": obs.get("returned_rows")[: max(1, row_limit)]
                if isinstance(obs.get("returned_rows"), list)
                else obs.get("returned_rows"),
            }
        )
    return _drop_empty(payload)


def _market_data_payload(state: AgentState, call: ToolCall, result: ToolResult) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    columns, rows = _table_rows(result)
    schema = market_data_observation_schema(call, result)
    preview = _table_preview_summary(
        result,
        state.task_context.task_prompt,
        call.arguments,
        max_rows=12,
        max_scan_rows=1200,
    )
    payload: JsonDict = {
        "tool": _tool_call_display_name(call),
        "provider": result.provider,
        "arguments": _summary_key_arguments(call.arguments),
        "schema": schema,
        "metadata": _copy_keys(metadata, MARKET_DATA_METADATA_KEYS),
        "columns": columns[:80],
        "row_count": len(rows) if rows else metadata.get("result_count"),
    }
    if preview:
        payload["selected_rows"] = preview
    return _drop_empty(payload)


def _tool_error_payload(call: ToolCall, result: ToolResult) -> JsonDict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    return {
        "tool": _tool_call_display_name(call),
        "provider": result.provider,
        "arguments": _summary_key_arguments(call.arguments),
        "status": result.status,
        "error": compact_text(str(result.error or "tool returned non-success status"), 1200),
        "metadata": _copy_keys(
            metadata,
            (
                "provider",
                "function",
                "endpoint",
                "query",
                "url",
                "source",
                "page_url",
                "api_url",
                "document_key",
                "table_index",
                "result_count",
                "matched_rows",
                "total_rows",
                "fallback_from",
                "akshare_error",
                "yahoo_error",
                "doc_id",
                "attachment_count",
                "attachment_urls",
                "source_url",
                "sheet",
            ),
        ),
    }


def _result_summary(state: AgentState, call: ToolCall, result: Optional[ToolResult]) -> str:
    if result is None:
        return "No tool result was returned."
    if result.status != "success":
        return compact_text(json_compact(_tool_error_payload(call, result)), 700)
    if call.name == "market_data":
        return compact_text(json_compact(_market_data_payload(state, call, result)), 1800)
    if call.name == "web_search":
        return compact_text(json_compact(_web_search_payload(call, result, result_limit=5)), 900)
    if call.name == "sec_reader" and call.action == "read_tables":
        return compact_text(json_compact(_sec_tables_payload(call, result, table_limit=4, row_limit=4)), 900)
    if call.name == "web_reader" or (call.name == "sec_reader" and call.action == "read_filing"):
        return compact_text(json_compact(_reader_summary_payload(call, result, text_limit=1200)), 900)
    metadata = result.metadata or {}
    parts: List[str] = []
    document_key = metadata.get("document_key")
    if document_key:
        parts.append(f"document_key={document_key}")
    source = metadata.get("url") or metadata.get("source")
    if source:
        parts.append(f"source={source}")
    if result.tables:
        table = result.tables[0] if result.tables else {}
        if isinstance(table, dict):
            rows = table.get("rows")
            cols = table.get("columns")
            row_count = len(rows) if isinstance(rows, list) else metadata.get("total_rows")
            col_count = len(cols) if isinstance(cols, list) else None
            parts.append(f"table rows={row_count} columns={col_count}")
    observation = result.observation
    if isinstance(observation, str):
        if observation:
            parts.append(f"observation_chars={len(observation)}")
    elif observation not in (None, ""):
        if isinstance(observation, (dict, list, tuple, set)):
            parts.append(f"observation_type={type(observation).__name__} size={len(observation)}")
        else:
            parts.append(f"observation_type={type(observation).__name__}")
    return compact_text(" | ".join(str(part) for part in parts if part), 1000)


def record_tool_result_evidence(
    state: AgentState,
    *,
    step_number: int,
    call_id: str,
    call: ToolCall,
    result: Optional[ToolResult],
) -> None:
    """Shadow-extract compact, verifiable facts from tool outputs.

    This is intentionally conservative: it records table previews, calculator
    raw results, and bounded text excerpts so packet mode can remember what an
    older tool call found without replaying the full observation.
    """

    if result is None:
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    source = metadata.get("url") or metadata.get("source") or result.provider or call.name
    document_key = metadata.get("document_key")
    entries: List[JsonDict] = []

    if result.status != "success":
        entries.append(
            {
                "entity": str(call.arguments.get("ticker") or call.arguments.get("symbol") or call.arguments.get("code") or ""),
                "period": _argument_period(call.arguments),
                "metric": f"{_tool_call_display_name(call)} error",
                "value": compact_text(str(result.error or "tool returned non-success status"), 400),
                "unit": "",
                "source": source,
                "document_key": document_key,
                "locator": f"step {step_number}",
                "text": compact_text(json_compact(_tool_error_payload(call, result)), 1800),
            }
        )
        if call_id:
            for entry in entries:
                entry["call_id"] = call_id
        record_evidence_items(
            state,
            entries,
            origin="tool_error",
            step_number=step_number,
        )
        return

    table_schema: JsonDict = {}
    if call.name != "market_data" and result.tables:
        table_schema = table_observation_schema(call, result)
        if table_schema:
            table_source_authority = table_schema.get("source_authority")
            table_value_origin = table_schema.get("value_origin")
            table_inference_risk = table_schema.get("inference_risk")
            entries.append(
                {
                    "entity": _argument_entity(call.arguments),
                    "period": _argument_period(call.arguments),
                    "metric": f"{_tool_call_display_name(call)} table_schema",
                    "value": table_schema.get("table_index") or table_schema.get("row_count") or call.action,
                    "unit": "schema",
                    "source": table_schema.get("source") or source,
                    "provider": table_schema.get("provider") or result.provider,
                    "source_authority": table_source_authority,
                    "value_origin": table_value_origin,
                    "inference_risk": table_inference_risk,
                    "document_key": table_schema.get("document_key") or document_key,
                    "locator": f"step {step_number}; {_tool_call_display_name(call)}",
                    "schema": table_schema,
                    "basis": table_schema.get("basis"),
                    "quote": compact_text(json_compact(table_schema), 3200),
                }
            )

    if call.name == "market_data":
        payload = _market_data_payload(state, call, result)
        schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
        field_units = schema.get("field_units") if isinstance(schema.get("field_units"), dict) else {}
        basis = schema.get("basis") if isinstance(schema.get("basis"), dict) else {}
        endpoint = schema.get("endpoint")
        provider = schema.get("provider") or result.provider
        source_authority = schema.get("source_authority")
        value_origin = schema.get("value_origin")
        inference_risk = schema.get("inference_risk")
        entries.append(
            {
                "entity": schema.get("entity") or _argument_entity(call.arguments),
                "period": schema.get("date_range") or _argument_period(call.arguments),
                "metric": f"{_tool_call_display_name(call)} schema",
                "value": endpoint or call.action,
                "unit": "schema",
                "source": source,
                "provider": provider,
                "source_authority": source_authority,
                "value_origin": value_origin,
                "inference_risk": inference_risk,
                "endpoint": endpoint,
                "document_key": document_key,
                "locator": f"step {step_number}; {_tool_call_display_name(call)}",
                "schema": schema,
                "basis": basis,
                "quote": compact_text(json_compact(schema), 3200),
            }
        )
        entries.append(
            {
                "entity": schema.get("entity") or _argument_entity(call.arguments),
                "period": schema.get("date_range") or _argument_period(call.arguments),
                "metric": f"{_tool_call_display_name(call)} selected_rows",
                "value": f"{payload.get('row_count', '')} rows".strip(),
                "unit": "rows",
                "source": source,
                "provider": provider,
                "source_authority": source_authority,
                "value_origin": value_origin,
                "inference_risk": inference_risk,
                "endpoint": endpoint,
                "document_key": document_key,
                "locator": f"step {step_number}",
                "schema": schema,
                "basis": basis,
                "quote": compact_text(json_compact(payload), 5200),
            }
        )
        preview = payload.get("selected_rows") if isinstance(payload.get("selected_rows"), dict) else {}
        selected_rows = preview.get("selected_rows") if isinstance(preview.get("selected_rows"), list) else []
        for row_index, row in enumerate(selected_rows[:12], start=1):
            if not isinstance(row, dict):
                continue
            row_locator = _row_identity(row)
            entries.append(
                {
                    "entity": schema.get("entity") or _argument_entity(call.arguments),
                    "period": schema.get("date_range") or _argument_period(call.arguments),
                    "metric": f"{_tool_call_display_name(call)} selected_row",
                    "value": compact_text(json_compact(row_locator), 300),
                    "unit": "row",
                    "source": source,
                    "provider": provider,
                    "source_authority": source_authority,
                    "value_origin": value_origin,
                    "inference_risk": inference_risk or _inference_risk(call, result, source, metadata, row),
                    "endpoint": endpoint,
                    "document_key": document_key,
                    "locator": f"step {step_number}; selected_row[{row_index}]",
                    "row_locator": row_locator,
                    "row_values": row,
                    "basis": basis,
                    "schema": schema,
                    "quote": compact_text(json_compact(row), 1600),
                }
            )
        extrema = preview.get("numeric_extrema") if isinstance(preview.get("numeric_extrema"), list) else []
        for item in extrema:
            if not isinstance(item, dict):
                continue
            field = str(item.get("column") or "").strip()
            if not field:
                continue
            for kind in ("min", "max"):
                detail = item.get(kind)
                if not isinstance(detail, dict):
                    continue
                value = detail.get("value")
                if value in (None, ""):
                    continue
                entries.append(
                    {
                        "entity": schema.get("entity") or _argument_entity(call.arguments),
                        "period": schema.get("date_range") or _argument_period(call.arguments),
                        "metric": f"{_tool_call_display_name(call)} {field} {kind}",
                        "field": field,
                        "source_field": field,
                        "value": value,
                        "unit": field_units.get(field) or "",
                        "source": source,
                        "provider": provider,
                        "source_authority": source_authority,
                        "value_origin": value_origin,
                        "inference_risk": inference_risk,
                        "endpoint": endpoint,
                        "document_key": document_key,
                        "locator": f"step {step_number}; numeric_extrema.{field}.{kind}",
                        "row_locator": detail.get("row"),
                        "basis": basis,
                        "schema": schema,
                        "quote": compact_text(json_compact(detail), 1200),
                    }
                )
    elif call.name == "web_search":
        payload = _web_search_payload(call, result, result_limit=8)
        entries.append(
            {
                "entity": _argument_entity(call.arguments),
                "period": _argument_period(call.arguments),
                "metric": "web_search top_results",
                "value": f"{payload.get('result_count', '')} results".strip(),
                "unit": "results",
                "source": result.provider or "web_search",
                "source_authority": "search_snippet",
                "value_origin": payload.get("value_origin") or _value_origin(call, result),
                "locator": f"step {step_number}",
                "quote": compact_text(json_compact(payload), 5200),
            }
        )
    elif call.name == "structured_table_reader" and call.action == "discover_tables":
        payload = metadata.get("discovery_payload") if isinstance(metadata.get("discovery_payload"), dict) else {}
        if not payload:
            payload = {
                "source": metadata.get("source") or source,
                "source_status": metadata.get("source_status"),
                "table_count": metadata.get("table_count"),
                "asset_count": metadata.get("asset_count"),
                "state_control_count": metadata.get("state_control_count"),
                "recommended_next": metadata.get("recommended_next"),
                "observation": compact_text(result.observation_text(), 2000),
            }
        entries.append(
            {
                "entity": _argument_entity(call.arguments),
                "period": _argument_period(call.arguments),
                "metric": "structured_table_reader_discover_tables source_discovery",
                "value": f"{metadata.get('table_count', 0) or 0} tables; {metadata.get('asset_count', 0) or 0} assets; {metadata.get('state_control_count', 0) or 0} state controls",
                "unit": "discovery",
                "source": payload.get("source") or source,
                "source_authority": "source_discovery",
                "value_origin": "source_discovery",
                "locator": f"step {step_number}; structured_table_reader_discover_tables",
                "schema": {
                    "source_status": payload.get("source_status"),
                    "recommended_next": payload.get("recommended_next"),
                    "failure_class": payload.get("failure_class") or metadata.get("failure_class"),
                },
                "quote": compact_text(json_compact(payload), 5200),
            }
        )
    elif call.name == "sec_reader" and call.action == "read_tables":
        payload = _sec_tables_payload(call, result, table_limit=10, row_limit=10)
        entries.append(
            {
                "entity": _argument_entity(call.arguments),
                "period": _argument_period(call.arguments),
                "metric": "sec_reader_read_tables candidates_or_rows",
                "value": f"{payload.get('table_count', '')} tables".strip(),
                "unit": "tables",
                "source": source,
                "source_authority": table_schema.get("source_authority") if isinstance(table_schema, dict) else _source_authority(call, result, source, metadata),
                "value_origin": table_schema.get("value_origin") if isinstance(table_schema, dict) else _value_origin(call, result),
                "inference_risk": table_schema.get("inference_risk") if isinstance(table_schema, dict) else _inference_risk(call, result, source, metadata, payload),
                "document_key": payload.get("document_key") or document_key,
                "locator": f"step {step_number}",
                "schema": table_schema,
                "basis": table_schema.get("basis") if isinstance(table_schema, dict) else {},
                "quote": compact_text(json_compact(payload), 6500),
            }
        )
    elif call.name == "web_reader" or (call.name == "sec_reader" and call.action == "read_filing"):
        payload = _reader_summary_payload(call, result, text_limit=3600)
        entries.append(
            {
                "entity": _argument_entity(call.arguments),
                "period": _argument_period(call.arguments),
                "metric": f"{_tool_call_display_name(call)} query_evidence",
                "value": "query-focused evidence",
                "unit": "",
                "source": payload.get("source") or source,
                "source_authority": payload.get("source_authority") or _source_authority(call, result, payload.get("source") or source, metadata),
                "value_origin": payload.get("value_origin") or _value_origin(call, result),
                "inference_risk": payload.get("inference_risk") or _inference_risk(call, result, payload.get("source") or source, metadata, payload),
                "document_key": payload.get("document_key") or document_key,
                "locator": f"step {step_number}",
                "quote": compact_text(json_compact(payload), 5200),
            }
        )
    else:
        table_preview = _table_preview_summary(
            result,
            state.task_context.task_prompt,
            call.arguments,
            max_rows=8,
        )
        if table_preview:
            rows = table_preview.get("total_rows_in_result")
            mode = table_preview.get("selected_mode")
            entries.append(
                {
                    "entity": str(call.arguments.get("ticker") or call.arguments.get("symbol") or call.arguments.get("code") or ""),
                    "period": str(call.arguments.get("period") or call.arguments.get("date") or call.arguments.get("trade_date") or ""),
                    "metric": f"{_tool_call_display_name(call)} table_preview",
                    "value": f"{rows} rows",
                    "unit": "rows",
                    "source": source,
                    "document_key": document_key,
                    "locator": f"step {step_number}; selected={mode}",
                    "schema": table_schema,
                    "basis": table_schema.get("basis") if isinstance(table_schema, dict) else {},
                    "quote": compact_text(json_compact(table_preview), 2600),
                }
            )

    structured = metadata.get("structured_result")
    if call.name == "calculator" and isinstance(structured, dict):
        raw = structured.get("raw")
        if raw not in (None, ""):
            entries.append(
                {
                    "entity": "",
                    "period": "",
                    "metric": "calculator_result",
                    "value": raw,
                    "unit": "",
                    "source": "calculator",
                    "locator": f"step {step_number}",
                    "text": compact_text(
                        json_compact(
                            {
                                "expression": call.arguments.get("expression")
                                or call.arguments.get("formula")
                                or call.arguments,
                                "structured_result": structured,
                            }
                        ),
                        1000,
                    ),
                }
            )

    observation_text = result.observation_text()
    if observation_text and not entries and not (call.name == "calculator" and isinstance(structured, dict)):
        excerpt_limit = 1600 if len(observation_text) <= 6000 else 1200
        entries.append(
            {
                "entity": str(call.arguments.get("ticker") or call.arguments.get("symbol") or call.arguments.get("code") or ""),
                "period": str(call.arguments.get("period") or call.arguments.get("date") or call.arguments.get("trade_date") or ""),
                "metric": f"{_tool_call_display_name(call)} observation_excerpt",
                "value": "",
                "unit": "",
                "source": source,
                "document_key": document_key,
                "locator": f"step {step_number}",
                "text": compact_text(observation_text, excerpt_limit),
            }
        )

    if entries:
        if call_id:
            for entry in entries:
                entry["call_id"] = call_id
        record_evidence_items(
            state,
            entries,
            origin="tool_result",
            step_number=step_number,
        )


def record_tool_result_answer_pins(
    state: AgentState,
    *,
    step_number: int,
    call_id: str,
    call: ToolCall,
    result: Optional[ToolResult],
) -> None:
    if result is None or result.status != "success":
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    source = metadata.get("url") or metadata.get("source") or metadata.get("page_url") or result.provider or call.name
    base: JsonDict = {
        "call_id": call_id,
        "source": source,
        "source_ref": metadata.get("document_key") or metadata.get("url") or metadata.get("source"),
        "provider": result.provider or metadata.get("provider"),
        "endpoint": metadata.get("endpoint") or metadata.get("function"),
        "source_authority": _source_authority(call, result, source, metadata),
        "value_origin": _value_origin(call, result),
        "inference_risk": _inference_risk(call, result, source, metadata, result.observation_text()[:1200]),
        "status": "tool_supported_candidate",
    }
    pins: List[JsonDict] = []

    if call.name == "calculator":
        structured = metadata.get("structured_result") if isinstance(metadata.get("structured_result"), dict) else {}
        raw = structured.get("raw")
        if raw not in (None, ""):
            pins.append(
                {
                    **base,
                    "slot_id": f"{call_id}:calculator_result",
                    "metric": "calculator_result",
                    "value": raw,
                    "formula": call.arguments.get("expression") or call.arguments.get("formula") or json_compact(call.arguments),
                    "raw_result": raw,
                    "calc_ref": call_id,
                }
            )

    elif call.name == "market_data":
        schema = market_data_observation_schema(call, result)
        columns, rows = _table_rows(result)
        field_units = schema.get("field_units") if isinstance(schema.get("field_units"), dict) else {}
        basis = schema.get("basis") if isinstance(schema.get("basis"), dict) else {}
        metadata_copy = schema.get("metadata") if isinstance(schema.get("metadata"), dict) else metadata
        common = {
            **base,
            "entity": schema.get("entity") or _argument_entity(call.arguments),
            "period": schema.get("date_range") or _argument_period(call.arguments),
            "metric": _tool_call_display_name(call),
            "basis": basis,
            "date_basis": basis.get("date_basis"),
            "adjustment": basis.get("adjustment") or metadata.get("adjust"),
            "amount_unit": metadata_copy.get("amount_unit"),
            "volume_unit": metadata_copy.get("volume_unit"),
            "currency": metadata_copy.get("currency"),
            "scale": metadata_copy.get("scale"),
        }
        if schema:
            pins.append(
                {
                    **common,
                    "slot_id": f"{call_id}:schema",
                    "subfield": "observation_schema",
                    "value": schema.get("endpoint") or call.action,
                    "unit": "schema",
                    "quote": compact_text(json_compact(schema), 2200),
                }
            )
        for row_index, row in enumerate(rows[:12], start=1):
            if not isinstance(row, dict):
                continue
            row_values = _row_lite(row, columns)
            row_locator = _row_identity(row_values)
            pins.append(
                {
                    **common,
                    "slot_id": f"{call_id}:row:{row_index}",
                    "subfield": f"selected_row[{row_index}]",
                    "value": compact_text(json_compact(row_locator), 360),
                    "unit": "row",
                    "row_locator": row_locator,
                    "row_values": row_values,
                    "quote": compact_text(json_compact(row_values), 1600),
                }
            )
        extrema = _numeric_extrema(rows, columns)
        for item in extrema[:8]:
            field = str(item.get("column") or "")
            if not field:
                continue
            for kind in ("min", "max"):
                detail = item.get(kind)
                if not isinstance(detail, dict) or detail.get("value") in (None, ""):
                    continue
                pins.append(
                    {
                        **common,
                        "slot_id": f"{call_id}:{field}:{kind}",
                        "metric": f"{_tool_call_display_name(call)} {field} {kind}",
                        "source_field": field,
                        "subfield": kind,
                        "value": detail.get("value"),
                        "unit": field_units.get(field) or "",
                        "row_locator": detail.get("row"),
                    }
                )

    elif call.name in {"structured_table_reader", "sec_reader"} and result.tables:
        schema = table_observation_schema(call, result)
        columns, rows = _table_rows(result)
        common = {
            **base,
            "entity": _argument_entity(call.arguments),
            "period": schema.get("period_hint") or _argument_period(call.arguments),
            "metric": _tool_call_display_name(call),
            "document_key": schema.get("document_key") or metadata.get("document_key"),
            "table_index": schema.get("table_index") or metadata.get("table_index"),
            "basis": schema.get("basis") if isinstance(schema.get("basis"), dict) else {},
            "unit": schema.get("unit_hint") or "",
        }
        if schema:
            pins.append(
                {
                    **common,
                    "slot_id": f"{call_id}:table_schema",
                    "subfield": "table_schema",
                    "value": schema.get("table_index") or schema.get("row_count") or call.action,
                    "quote": compact_text(json_compact(schema), 2200),
                }
            )
        for row_index, row in enumerate(rows[:10], start=1):
            if not isinstance(row, dict):
                continue
            row_values = _row_lite(row, columns)
            row_locator = _row_identity(row_values)
            pins.append(
                {
                    **common,
                    "slot_id": f"{call_id}:table_row:{row_index}",
                    "subfield": f"table_row[{row_index}]",
                    "value": compact_text(json_compact(row_locator), 360),
                    "row_locator": row_locator,
                    "row_values": row_values,
                    "quote": compact_text(json_compact(row_values), 1600),
                }
            )

    elif call.name in {"web_reader", "sec_reader"}:
        payload = _reader_summary_payload(call, result, text_limit=1800)
        evidence_text = payload.get("evidence") or payload.get("key_values") or payload.get("summary")
        if evidence_text:
            pins.append(
                {
                    **base,
                    "slot_id": f"{call_id}:reader_evidence",
                    "entity": _argument_entity(call.arguments),
                    "period": _argument_period(call.arguments),
                    "metric": f"{_tool_call_display_name(call)} query_evidence",
                    "value": "query-focused evidence",
                    "document_key": payload.get("document_key") or metadata.get("document_key"),
                    "status": "reader_candidate",
                    "quote": compact_text(str(evidence_text), 1800),
                }
            )

    if pins:
        record_answer_pins(
            state,
            pins,
            origin="tool_result",
            step_number=step_number,
        )


def _action_duplicate_key(entry: Dict[str, Any]) -> str:
    tool = str(entry.get("tool") or "")
    action = str(entry.get("action") or "")
    fp = str(entry.get("args_fingerprint") or "")
    if not tool or not action or not fp:
        return ""
    return "\t".join((tool, action, fp))


def _action_duplicate_index(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    index = getattr(state, "action_duplicate_index", None)
    if not isinstance(index, dict):
        index = {}
        setattr(state, "action_duplicate_index", index)
    initialized = bool(getattr(state, "_action_duplicate_index_initialized", False))
    if not initialized and state.action_ledger:
        for entry in state.action_ledger:
            if isinstance(entry, dict):
                key = _action_duplicate_key(entry)
                if key:
                    index.setdefault(key, []).append(entry)
        setattr(state, "_action_duplicate_index_initialized", True)
    elif not initialized:
        setattr(state, "_action_duplicate_index_initialized", True)
    return index


def _action_artifact_index(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    index = getattr(state, "action_artifact_index", None)
    if not isinstance(index, dict):
        index = {}
        setattr(state, "action_artifact_index", index)
    initialized = bool(getattr(state, "_action_artifact_index_initialized", False))
    if not initialized and state.action_ledger:
        for entry in state.action_ledger:
            if isinstance(entry, dict):
                _index_action_artifacts(index, entry)
        setattr(state, "_action_artifact_index_initialized", True)
    elif not initialized:
        setattr(state, "_action_artifact_index_initialized", True)
    return index


def _index_action_artifacts(index: Dict[str, List[Dict[str, Any]]], entry: Dict[str, Any]) -> None:
    artifact_ids = entry.get("artifact_ids")
    if not isinstance(artifact_ids, list):
        return
    for artifact_id in artifact_ids:
        key = str(artifact_id or "")
        if key:
            index.setdefault(key, []).append(entry)


def record_action_ledger(
    state: AgentState,
    *,
    step_number: int,
    call: ToolCall,
    result: Optional[ToolResult],
    full_chars: int,
    prompt_chars: int,
    call_index: int = 0,
    think: str = "",
    artifact_ids: Optional[List[str]] = None,
    prompt_observation: str = "",
) -> None:
    duplicate_index = _action_duplicate_index(state)
    artifact_index = _action_artifact_index(state)
    call_id = f"s{step_number}_c{call_index + 1}"
    native_arguments = call.native_arguments if isinstance(call.native_arguments, dict) and call.native_arguments else call.arguments
    entry: JsonDict = {
        "step": step_number,
        "call_id": call_id,
        "tool": call.name,
        "action": call.action,
        "function": _tool_call_display_name(call),
        "args_fingerprint": args_fingerprint(call.arguments),
        "arguments": call.arguments,
        "native_arguments": native_arguments,
        "status": getattr(result, "status", "missing") if result is not None else "missing",
        "provider": getattr(result, "provider", "") if result is not None else "",
        "result_summary": _result_summary(state, call, result),
        "full_observation_chars": full_chars,
        "prompt_observation_chars": prompt_chars,
    }
    if think and call_index == 0:
        entry["step_think"] = compact_text(think, 500)
    if prompt_observation:
        excerpt_limit = max(400, int(state.config.context_recent_tool_call_excerpt_chars or 0))
        entry["prompt_observation_excerpt"] = compact_text(prompt_observation, excerpt_limit)
    if result is not None and result.error:
        entry["error"] = compact_text(str(result.error), 400)
    metadata = getattr(result, "metadata", None) if result is not None else None
    projected_metadata: JsonDict = {}
    if isinstance(metadata, dict) and metadata:
        projected_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "document_key",
                "document_kind",
                "url",
                "source",
                "page_url",
                "api_url",
                "function",
                "requested_function",
                "endpoint",
                "query",
                "result_count",
                "matched_rows",
                "total_rows",
                "table_count",
                "table_index",
                "returned_rows",
                "provider",
                "data_source",
                "params",
                "fields",
                "ts_code",
                "index_code",
                "fund_code",
                "contract",
                "series",
                "requested_ticker",
                "symbol",
                "start_date",
                "end_date",
                "field_schema",
                "semantic_candidates",
                "typed_schema",
                "unit_schema",
                "currency",
                "scale",
                "adjust",
                "amount_unit",
                "volume_unit",
                "date_filter_applied",
                "doc_id",
                "doc_title",
                "publish_date",
                "doc_source",
                "attachment_count",
                "attachment_urls",
                "attachment_errors",
                "source_url",
                "sheet",
                "header_rows",
                "unit_hint",
                "period_hint",
                "parser",
                "source_authority",
                "value_origin",
                "inference_risk",
                "execution_card",
            }
        }
    if result is not None:
        schema = tool_observation_schema(call, result)
        if isinstance(schema, dict) and schema:
            for key in ("source_authority", "value_origin", "inference_risk"):
                if schema.get(key) not in (None, "", [], {}):
                    projected_metadata.setdefault(key, schema.get(key))
    if projected_metadata:
        entry["metadata"] = projected_metadata
    if artifact_ids:
        entry["artifact_ids"] = artifact_ids
    entry["search_text"] = ledger_search_text(entry)
    state.action_ledger.append(entry)
    key = _action_duplicate_key(entry)
    if key:
        duplicate_index.setdefault(key, []).append(entry)
    _index_action_artifacts(artifact_index, entry)
    record_tool_result_evidence(
        state,
        step_number=step_number,
        call_id=call_id,
        call=call,
        result=result,
    )
    record_tool_result_answer_pins(
        state,
        step_number=step_number,
        call_id=call_id,
        call=call,
        result=result,
    )
    state.context_stats["total_full_observation_chars"] = (
        int(state.context_stats.get("total_full_observation_chars", 0) or 0) + full_chars
    )
    state.context_stats["total_prompt_observation_chars"] = (
        int(state.context_stats.get("total_prompt_observation_chars", 0) or 0) + prompt_chars
    )
    state.context_stats["max_full_observation_chars"] = max(
        int(state.context_stats.get("max_full_observation_chars", 0) or 0),
        full_chars,
    )

    if call.name == "calculator" and result is not None and result.status == "success":
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        state.calc_ledger.append(
            {
                "step": step_number,
                "call_id": call_id,
                "expression": call.arguments.get("expression") or call.arguments.get("formula") or call.arguments,
                "output": compact_text(result.observation_text(), 400),
                "structured_result": metadata.get("structured_result") or {},
                "depends_on": call.arguments.get("depends_on") or [],
            }
        )


class ContextAssembler:
    def __init__(self, config: AgentConfig):
        self.config = config

    def _effective_context_budgets(self, state: AgentState, *, tools: Optional[List[JsonDict]] = None) -> Tuple[int, int]:
        raw_threshold = max(1, int(self.config.context_hybrid_char_threshold or 0))
        raw_full_limit = max(raw_threshold, int(getattr(self.config, "context_full_compatible_char_limit", 0) or 0))
        active_tools = tools if tools is not None else getattr(state, "tools", [])
        tools_chars = tool_schema_char_count(active_tools)
        reserve_chars = max(0, int(getattr(self.config, "context_request_reserve_chars", 0) or 0))
        deduction = tools_chars + reserve_chars
        threshold = max(1, raw_threshold - deduction)
        full_limit = max(threshold, raw_full_limit - deduction)
        state.context_stats["context_hybrid_char_threshold"] = raw_threshold
        state.context_stats["context_full_compatible_char_limit"] = raw_full_limit
        state.context_stats["tool_schema_chars"] = tools_chars
        state.context_stats["context_request_reserve_chars"] = reserve_chars
        state.context_stats["effective_context_hybrid_char_threshold"] = threshold
        state.context_stats["effective_context_full_compatible_char_limit"] = full_limit
        return threshold, full_limit

    def select_render_mode(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        tools: Optional[List[JsonDict]] = None,
    ) -> Tuple[str, int]:
        legacy_chars = self._estimate_legacy_prompt_chars(memory, instruction)
        setattr(state, "_context_legacy_messages_cache", None)
        setattr(state, "_context_legacy_instruction_cache", None)
        state.context_stats["last_legacy_prompt_chars"] = legacy_chars
        mode = str(self.config.context_mode or "auto").strip().lower()
        threshold, full_limit = self._effective_context_budgets(state, tools=tools)

        if mode == "legacy":
            legacy_messages, legacy_chars = self._build_legacy_messages(memory, instruction)
            state.context_stats["last_legacy_prompt_chars"] = legacy_chars
            self._cache_legacy_messages(state, legacy_messages, instruction)
            return "legacy", legacy_chars
        if mode in {"full", "full_compatible"}:
            legacy_messages, legacy_chars = self._build_legacy_messages(memory, instruction)
            state.context_stats["last_legacy_prompt_chars"] = legacy_chars
            self._cache_legacy_messages(state, legacy_messages, instruction)
            return "full_compatible", legacy_chars
        if mode == "packet":
            return "lean_packet", legacy_chars

        if legacy_chars <= threshold:
            legacy_messages, exact_chars = self._build_legacy_messages(memory, instruction)
            state.context_stats["last_legacy_prompt_chars"] = exact_chars
            if exact_chars <= threshold:
                self._cache_legacy_messages(state, legacy_messages, instruction)
                return "full_compatible", exact_chars
            legacy_chars = exact_chars

        readiness = self._packet_semantic_readiness(state)
        state.context_stats["packet_semantic_readiness"] = readiness
        if not readiness.get("ready", True):
            guard_full_limit = full_limit
            exact_chars = legacy_chars
            legacy_messages: Optional[List[Message]] = None
            if legacy_chars <= guard_full_limit:
                legacy_messages, exact_chars = self._build_legacy_messages(memory, instruction)
                state.context_stats["last_legacy_prompt_chars"] = exact_chars
            if exact_chars <= guard_full_limit and legacy_messages is not None:
                self._cache_legacy_messages(state, legacy_messages, instruction)
                state.context_stats["packet_deferred_reason"] = "old_tool_results_not_semantically_covered"
                return "full_compatible", exact_chars
            state.context_stats["packet_forced_compact_reason"] = "semantic_gaps_need_selective_replay"
            return "compact_packet", legacy_chars

        if self._should_use_compact_packet(state, legacy_chars, tools=tools):
            return "compact_packet", legacy_chars
        return "lean_packet", legacy_chars

    def _build_legacy_messages(self, memory: AgentMemory, instruction: str) -> Tuple[List[Message], int]:
        legacy_messages = memory.to_messages() + [{"role": "user", "content": instruction}]
        return legacy_messages, messages_char_count(legacy_messages)

    @staticmethod
    def _cache_legacy_messages(state: AgentState, messages: List[Message], instruction: str) -> None:
        setattr(state, "_context_legacy_messages_cache", messages)
        setattr(state, "_context_legacy_instruction_cache", instruction)

    def _packet_cache_key(self, state: AgentState) -> Tuple[Any, ...]:
        return (
            len(state.action_ledger),
            len(state.evidence_ledger),
            len(state.calc_ledger),
            len(state.artifact_store),
            len(getattr(state, "trajectory_action_steps", []) or []),
            id(state.current_task_state),
            int(self.config.context_action_ledger_limit or 0),
            int(self.config.context_pinned_evidence_limit or 0),
            int(self.config.context_pinned_evidence_chars or 0),
            int(self.config.context_recent_steps or 0),
            int(getattr(state, "_context_recent_steps_override", -1) or -1),
        )

    def _packet_context(self, state: AgentState) -> JsonDict:
        key = self._packet_cache_key(state)
        cache = getattr(state, "_packet_build_context", None)
        if not isinstance(cache, dict) or cache.get("_key") != key:
            cache = {"_key": key}
            setattr(state, "_packet_build_context", cache)
        return cache

    def _estimate_legacy_prompt_chars(self, memory: AgentMemory, instruction: str) -> int:
        estimator = getattr(memory, "estimated_prompt_chars", None)
        if callable(estimator):
            try:
                return int(estimator(instruction))
            except Exception:
                pass
        total = len(getattr(memory.system_prompt, "system_prompt", "") or "")
        for step in memory.steps:
            if isinstance(step, TaskStep):
                total += len("New task:\n") + len(step.task or "")
            elif isinstance(step, PlanningStep):
                total += len("Now write a short initial plan for the task above. Do not call tools yet.")
                total += len("[PLAN]\n") + len((step.plan or "").strip())
            elif isinstance(step, ActionStep):
                assistant_payload: Dict[str, Any] = {
                    "think": step.think or "",
                    "tools": step.tool_calls or [],
                }
                total += len("Calling tools:\n") + len(
                    json.dumps(assistant_payload, ensure_ascii=False, indent=2, default=str)
                )
                if step.observations:
                    total += len(f"Tool calling observation (step {step.step_number}):\n")
                    total += len(step.observations)
                if step.error:
                    total += len(f"Error in step {step.step_number}: ")
                    total += len(step.error)
                    total += len("\nTake a different approach in your next step.")
            else:
                total += messages_char_count(step.to_messages())
        total += len(instruction or "")
        return total

    def action_messages(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        render_mode: str,
        legacy_chars: int,
        packet_override: Optional[str] = None,
        tools: Optional[List[JsonDict]] = None,
    ) -> List[Message]:
        mode = str(render_mode or "full_compatible").strip().lower()
        if mode in {"legacy", "full_compatible"}:
            self._record_render_mode(state, mode)
            cached = getattr(state, "_context_legacy_messages_cache", None)
            cached_instruction = getattr(state, "_context_legacy_instruction_cache", None)
            if isinstance(cached, list) and cached_instruction == instruction:
                setattr(state, "_context_legacy_messages_cache", None)
                setattr(state, "_context_legacy_instruction_cache", None)
                return cached
            return memory.to_messages() + [{"role": "user", "content": instruction}]

        packet_mode = "compact" if mode == "compact_packet" else "lean"
        self._record_render_mode(state, mode)
        return self._budgeted_packet_messages(
            state,
            memory,
            instruction,
            legacy_chars=legacy_chars,
            packet_mode=packet_mode,
            packet_content=packet_override or "",
            packet_override=packet_override,
            tools=tools,
        )

    def _budgeted_packet_messages(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        legacy_chars: int,
        packet_mode: str,
        packet_content: str,
        packet_override: Optional[str],
        tools: Optional[List[JsonDict]],
    ) -> List[Message]:
        _, full_limit = self._effective_context_budgets(state, tools=tools)
        action_steps = getattr(memory, "action_steps", None)
        action_count = len(action_steps) if isinstance(action_steps, list) else 0
        configured_keep = min(max(0, int(self.config.context_recent_steps or 0)), action_count)
        packet_min_keep = 0
        if packet_override:
            raw_tail_start = getattr(state, "_context_compression_raw_tail_start_action_index", None)
            if isinstance(raw_tail_start, int):
                packet_min_keep = max(0, min(action_count, action_count - raw_tail_start))
        default_keep = min(action_count, max(configured_keep, packet_min_keep))
        candidates = [default_keep]
        for keep in (3, 1, 0):
            keep = min(keep, action_count)
            if keep >= packet_min_keep and keep not in candidates:
                candidates.append(keep)
        if packet_min_keep and packet_min_keep not in candidates:
            candidates.append(packet_min_keep)

        original_override = getattr(state, "_context_recent_steps_override", None)
        original_tail_start = getattr(state, "_context_compression_raw_tail_start_action_index", None)
        best_messages: List[Message] = []
        best_chars = 0
        chosen_keep = default_keep
        chosen_packet = packet_content

        for keep in candidates:
            self._reset_packet_context_cache(state)
            setattr(state, "_context_recent_steps_override", keep)
            if packet_override:
                raw_tail_start = max(0, action_count - keep)
                setattr(state, "_context_compression_raw_tail_start_action_index", raw_tail_start)
                current_packet = packet_content
            else:
                current_packet = self._working_context_packet(
                    state,
                    memory,
                    legacy_chars,
                    packet_mode=packet_mode,
                )
            messages = self._packet_messages_for_current_tail(
                state,
                memory,
                instruction,
                packet_content=current_packet,
                packet_override=packet_override,
            )
            chars = messages_char_count(messages)
            best_messages = messages
            best_chars = chars
            chosen_keep = keep
            chosen_packet = current_packet
            if chars <= full_limit:
                break

        if best_chars > full_limit and chosen_packet:
            trimmed = self._budget_trim_packet_content(chosen_packet, max_chars=max(1, len(chosen_packet) - (best_chars - full_limit) - 256))
            if trimmed != chosen_packet:
                for keep in candidates:
                    self._reset_packet_context_cache(state)
                    setattr(state, "_context_recent_steps_override", keep)
                    if packet_override:
                        raw_tail_start = max(0, action_count - keep)
                        setattr(state, "_context_compression_raw_tail_start_action_index", raw_tail_start)
                        current_packet = trimmed
                    else:
                        current_packet = self._budget_trim_packet_content(
                            self._working_context_packet(
                                state,
                                memory,
                                legacy_chars,
                                packet_mode=packet_mode,
                            ),
                            max_chars=max(1, len(trimmed)),
                        )
                    messages = self._packet_messages_for_current_tail(
                        state,
                        memory,
                        instruction,
                        packet_content=current_packet,
                        packet_override=packet_override,
                    )
                    chars = messages_char_count(messages)
                    if chars <= best_chars:
                        best_messages = messages
                        best_chars = chars
                        chosen_keep = keep
                        chosen_packet = current_packet
                    if chars <= full_limit:
                        break

        state.context_stats["last_packet_prompt_chars"] = best_chars
        state.context_stats["last_packet_prompt_budget_chars"] = full_limit
        state.context_stats["last_packet_prompt_over_budget"] = best_chars > full_limit
        state.context_stats["last_packet_recent_raw_steps_selected"] = chosen_keep
        state.context_stats["last_packet_content_chars"] = len(chosen_packet or "")
        if isinstance(chosen_packet, str) and "\"prompt_budget_trim\"" in chosen_packet:
            state.context_stats["last_packet_prompt_budget_trim"] = True
        else:
            state.context_stats.pop("last_packet_prompt_budget_trim", None)
        if chosen_keep < default_keep:
            state.context_stats["last_packet_recent_raw_reduction"] = {
                "from": default_keep,
                "to": chosen_keep,
                "reason": "final_packet_prompt_exceeds_budget",
            }
        else:
            state.context_stats.pop("last_packet_recent_raw_reduction", None)
        state.context_stats["last_historical_tool_call_turns"] = 0
        state.context_stats["last_historical_tool_call_chars"] = 0

        if not best_messages:
            if original_override is None:
                try:
                    delattr(state, "_context_recent_steps_override")
                except AttributeError:
                    pass
            else:
                setattr(state, "_context_recent_steps_override", original_override)
            if original_tail_start is None:
                try:
                    delattr(state, "_context_compression_raw_tail_start_action_index")
                except AttributeError:
                    pass
            else:
                setattr(state, "_context_compression_raw_tail_start_action_index", original_tail_start)
        return best_messages

    def _budget_trim_packet_content(self, packet_content: str, *, max_chars: int) -> str:
        if not isinstance(packet_content, str) or len(packet_content) <= max_chars:
            return packet_content
        packet = self._parse_packet_json(packet_content)
        if not packet:
            return self._minimum_budget_packet(
                {},
                max_chars=max_chars,
                original_chars=len(packet_content),
            )
        original_chars = len(packet_content)
        packet["prompt_budget_trim"] = {
            "applied": True,
            "original_chars": original_chars,
            "target_chars": max_chars,
            "policy": "preserve task_state, answer-critical memory, compact ledgers, and recent raw turns; trim bulky auxiliary packet sections first",
        }
        replay_hints = packet.get("replay_hints")
        if isinstance(replay_hints, list) and len(replay_hints) > 6:
            packet["replay_hints"] = replay_hints[:6]

        drop_order = (
            "pinned_raw_slices",
            "artifact_selective_slices",
            "artifact_refs",
            "duplicate_call_warnings",
            "semantic_replay_requests",
        )
        for key in drop_order:
            if self._packet_text_len(packet) <= max_chars:
                break
            if key in packet:
                packet.pop(key, None)

        list_limits = (
            ("old_action_ledger", "items", (120, 80, 40, 20, 8, 0)),
            ("evidence_ledger", "items", (48, 32, 16, 8, 0)),
            ("answer_critical_pins", None, (48, 32, 16, 8, 0)),
            ("calc_ledger", None, (64, 32, 16, 8, 0)),
        )
        for section, child_key, limits in list_limits:
            for limit in limits:
                if self._packet_text_len(packet) <= max_chars:
                    break
                self._trim_packet_list(packet, section, child_key, limit)

        if self._packet_text_len(packet) > max_chars:
            packet = self._clip_packet_value(packet, max_chars=max(1200, max_chars - len("WorkingContextPacket:\n")))
        text = "WorkingContextPacket:\n" + json_compact(packet)
        if len(text) > max_chars:
            packet["prompt_budget_trim"]["hard_clipped"] = True
            return self._minimum_budget_packet(
                packet,
                max_chars=max_chars,
                original_chars=original_chars,
            )
        return text

    def _minimum_budget_packet(
        self,
        packet: JsonDict,
        *,
        max_chars: int,
        original_chars: int,
    ) -> str:
        max_chars = max(1, int(max_chars or 0))
        render_mode = str(packet.get("render_mode") or "lean")
        trim_record: JsonDict = {
            "applied": True,
            "minimal_packet": True,
            "original_chars": int(original_chars or 0),
            "target_chars": max_chars,
            "policy": "legal_json_budget_fallback",
        }
        old_action = packet.get("old_action_ledger") if isinstance(packet.get("old_action_ledger"), dict) else {}
        evidence = packet.get("evidence_ledger") if isinstance(packet.get("evidence_ledger"), dict) else {}
        minimal: JsonDict = {
            "render_mode": render_mode,
            "task_card": self._clip_packet_value(packet.get("task_card") or {}, max_chars=1200),
            "recent_raw_window": self._clip_packet_value(packet.get("recent_raw_window") or {}, max_chars=1200),
            "task_state_view": self._clip_packet_value(packet.get("task_state_view") or {}, max_chars=2400),
            "task_state_support_audit": self._clip_packet_value(packet.get("task_state_support_audit") or {}, max_chars=1200),
            "answer_critical_memory": self._clip_packet_value(packet.get("answer_critical_memory") or {}, max_chars=3200),
            "answer_critical_pins": self._clip_packet_value(packet.get("answer_critical_pins") or [], max_chars=2400),
            "old_action_ledger": {
                "total_tool_calls": old_action.get("total_tool_calls", 0),
                "old_tool_calls_in_packet": old_action.get("old_tool_calls_in_packet", 0),
                "omitted_recent_steps": old_action.get("omitted_recent_steps") or [],
                "items": self._clip_packet_value(old_action.get("items") or [], max_chars=2400),
            },
            "evidence_ledger": {
                "total_evidence_items": evidence.get("total_evidence_items", 0),
                "items": self._clip_packet_value(evidence.get("items") or [], max_chars=2400),
            },
            "calc_ledger": self._clip_packet_value(packet.get("calc_ledger") or [], max_chars=1200),
            "replay_hints": [
                "WorkingContextPacket was minimized to fit the prompt budget.",
                "Use selected recent raw turns first; use compact ledgers only as recovery memory.",
            ],
            "prompt_budget_trim": trim_record,
        }
        drop_order = (
            "calc_ledger",
            "task_state_support_audit",
            "answer_critical_pins",
            "evidence_ledger",
            "old_action_ledger",
            "answer_critical_memory",
            "task_state_view",
            "replay_hints",
            "task_card",
        )
        for key in drop_order:
            text = "WorkingContextPacket:\n" + json_compact(minimal)
            if len(text) <= max_chars:
                return text
            minimal.pop(key, None)
        text = "WorkingContextPacket:\n" + json_compact(minimal)
        if len(text) <= max_chars:
            return text
        tiny: JsonDict = {
            "render_mode": render_mode,
            "prompt_budget_trim": {
                "applied": True,
                "minimal_packet": True,
                "original_chars": int(original_chars or 0),
                "target_chars": max_chars,
            },
        }
        text = "WorkingContextPacket:\n" + json_compact(tiny)
        if len(text) <= max_chars:
            return text
        return "WorkingContextPacket:\n" + json_compact({"prompt_budget_trim": {"applied": True}})

    @staticmethod
    def _parse_packet_json(packet_content: str) -> JsonDict:
        text = str(packet_content or "").strip()
        if text.startswith("WorkingContextPacket:"):
            text = text[len("WorkingContextPacket:") :].strip()
        text = ContextAssembler._first_balanced_json_object(text) or text
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _first_balanced_json_object(text: str) -> str:
        start = str(text or "").find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

    @staticmethod
    def _packet_text_len(packet: JsonDict) -> int:
        return len("WorkingContextPacket:\n") + len(json_compact(packet))

    @staticmethod
    def _trim_packet_list(packet: JsonDict, section: str, child_key: Optional[str], limit: int) -> None:
        limit = max(0, int(limit or 0))
        container = packet.get(section)
        if child_key is None:
            if isinstance(container, list):
                packet[section] = container[-limit:] if limit else []
            return
        if isinstance(container, dict):
            items = container.get(child_key)
            if isinstance(items, list):
                container[child_key] = items[-limit:] if limit else []
                container[f"{child_key}_prompt_trimmed_to"] = limit

    def _clip_packet_value(self, value: Any, *, max_chars: int, depth: int = 0) -> Any:
        if max_chars <= 0:
            return {}
        if depth > 5:
            return compact_text(json_compact(value), min(800, max_chars))
        if isinstance(value, str):
            return compact_text(value, min(len(value), max(120, max_chars // 4)))
        if isinstance(value, list):
            if not value:
                return []
            selected = value[-min(len(value), 8):]
            per_item = max(200, max_chars // max(1, len(selected)))
            return [self._clip_packet_value(item, max_chars=per_item, depth=depth + 1) for item in selected]
        if isinstance(value, dict):
            output: JsonDict = {}
            priority = (
                "render_mode",
                "task_card",
                "recent_raw_window",
                "task_state_view",
                "task_state_support_audit",
                "answer_critical_memory",
                "answer_critical_pins",
                "old_action_ledger",
                "evidence_ledger",
                "calc_ledger",
                "prompt_budget_trim",
                "replay_hints",
            )
            keys = [key for key in priority if key in value] + [key for key in value.keys() if key not in priority]
            per_item = max(300, max_chars // max(1, len(keys)))
            for key in keys:
                output[key] = self._clip_packet_value(value.get(key), max_chars=per_item, depth=depth + 1)
                if len(json_compact(output)) > max_chars:
                    output.pop(key, None)
                    break
            return output
        return value

    def _packet_messages_for_current_tail(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        packet_content: str,
        packet_override: Optional[str],
    ) -> List[Message]:
        messages = self._base_messages(memory)
        messages.append({"role": "user", "content": packet_content})
        raw_tail_start = getattr(state, "_context_compression_raw_tail_start_action_index", None)
        if packet_override and isinstance(raw_tail_start, int):
            messages.extend(self._compression_raw_tail_messages(state, memory, raw_tail_start))
        else:
            state.context_stats["last_historical_tool_call_turns"] = 0
            state.context_stats["last_historical_tool_call_chars"] = 0
            messages.extend(self._recent_action_messages(state, memory))
        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _reset_packet_context_cache(state: AgentState) -> None:
        try:
            delattr(state, "_packet_build_context")
        except AttributeError:
            pass

    def terminal_messages(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        tools: Optional[List[JsonDict]] = None,
    ) -> List[Message]:
        render_mode, legacy_chars = self.select_render_mode(state, memory, instruction, tools=tools)
        return self.action_messages(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            tools=tools,
        )

    def _base_messages(self, memory: AgentMemory) -> List[Message]:
        messages = memory.system_prompt.to_messages()
        for step in memory.steps:
            if isinstance(step, ActionStep):
                continue
            messages.extend(step.to_messages())
        return messages

    def build_working_context_packet(
        self,
        state: AgentState,
        memory: AgentMemory,
        legacy_chars: int,
        *,
        packet_mode: str,
    ) -> str:
        return self._working_context_packet(
            state,
            memory,
            legacy_chars,
            packet_mode=packet_mode,
        )

    def build_rolling_compression_delta_packet(
        self,
        state: AgentState,
        memory: AgentMemory,
        *,
        start_action_index: int,
        end_action_index: int,
        char_limit: int,
    ) -> str:
        """Compact deterministic overlay for steps that just left the raw tail."""

        action_steps = getattr(memory, "action_steps", None)
        if not isinstance(action_steps, list):
            action_steps = [step for step in memory.steps if isinstance(step, ActionStep)]
        start = max(0, min(int(start_action_index or 0), len(action_steps)))
        end = max(start, min(int(end_action_index or 0), len(action_steps)))
        selected_steps = action_steps[start:end]
        if not selected_steps:
            return ""

        step_numbers = {
            int(step.step_number)
            for step in selected_steps
            if self._coerce_int(getattr(step, "step_number", None)) is not None
        }
        if not step_numbers:
            return ""
        step_numbers_list = sorted(step_numbers)

        action_items = [
            item
            for item in state.action_ledger
            if isinstance(item, dict) and self._coerce_int(item.get("step")) in step_numbers
        ]
        call_ids = {str(item.get("call_id") or "") for item in action_items if item.get("call_id") not in (None, "")}
        for step in selected_steps:
            for call in getattr(step, "tool_calls", None) or []:
                if isinstance(call, dict) and call.get("id") not in (None, ""):
                    call_ids.add(str(call.get("id")))
        action_lite = self._action_ledger_lite(
            action_items,
            omit_result_steps=set(),
            include_result_summary=True,
            args_char_limit=360,
        )

        terms = self._contract_terms(state)
        evidence_candidates = self._records_for_delta_refs(
            _evidence_ref_index(state),
            call_ids=call_ids,
            step_numbers=step_numbers,
        )
        evidence_rows = [
            (sum(1 for term in terms if term and term in ledger_search_text(item)), idx, item)
            for idx, item in enumerate(evidence_candidates)
            if isinstance(item, dict)
        ]
        evidence_rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        evidence_lite = [self._evidence_entry_lite(item) for _, _, item in evidence_rows[:10]]

        calc_items = [
            item
            for item in state.calc_ledger
            if isinstance(item, dict) and self._record_matches_delta_refs(item, call_ids=call_ids, step_numbers=step_numbers)
        ]
        calc_lite = self._calc_ledger_lite(calc_items)

        pin_candidates = self._records_for_delta_refs(
            _answer_pin_ref_index(state),
            call_ids=call_ids,
            step_numbers=step_numbers,
        )
        pin_rows = [
            (self._answer_pin_score(item, terms), idx, item)
            for idx, item in enumerate(pin_candidates)
            if isinstance(item, dict)
        ]
        pin_rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        pin_lite = [self._answer_pin_lite(item) for _, _, item in pin_rows[:8]]

        delta: JsonDict = _drop_empty(
            {
                "purpose": (
                    "Deterministic overlay for action steps that left the raw recent window "
                    "since the cached compressed packet. Latest raw turns are appended after the packet."
                ),
                "action_index_range": [start, end],
                "steps": step_numbers_list,
                "action_ledger_delta": action_lite,
                "evidence_delta": evidence_lite,
                "calc_delta": calc_lite,
                "answer_pin_delta": pin_lite,
            }
        )
        if not delta:
            return ""

        max_chars = max(1200, int(char_limit or 0))
        for section in ("answer_pin_delta", "evidence_delta", "calc_delta"):
            while section in delta and len(json_compact(delta)) > max_chars:
                records = delta.get(section)
                if not isinstance(records, list) or not records:
                    delta.pop(section, None)
                    break
                records.pop()
                if not records:
                    delta.pop(section, None)
        text = json_compact(delta)
        if len(text) > max_chars:
            return ""
        return "RollingCompressionDelta:\n" + text

    @staticmethod
    def _records_for_delta_refs(
        index: Dict[str, List[Dict[str, Any]]],
        *,
        call_ids: set,
        step_numbers: set,
    ) -> List[JsonDict]:
        selected: List[JsonDict] = []
        seen = set()
        for key in list(call_ids) + [f"step:{step}" for step in sorted(step_numbers)]:
            for item in index.get(str(key), []):
                if not isinstance(item, dict):
                    continue
                marker = str(item.get("fingerprint") or item.get("pin_key") or item.get("pin_id") or id(item))
                if marker in seen:
                    continue
                seen.add(marker)
                selected.append(item)
        return selected

    @staticmethod
    def _record_matches_delta_refs(item: JsonDict, *, call_ids: set, step_numbers: set) -> bool:
        call_id = str(item.get("call_id") or "")
        if call_id and call_id in call_ids:
            return True
        for key in ("step", "first_seen_step", "last_seen_step"):
            value = _coerce_step_number(item.get(key))
            if value in step_numbers:
                return True
        return False

    def _compression_raw_tail_messages(
        self,
        state: AgentState,
        memory: AgentMemory,
        start_action_index: int,
    ) -> List[Message]:
        """Append exact prompt-facing ActionStep turns after the compression point.

        LLM compression is a rolling base: the compressed packet replaces the
        prefix up to ``start_action_index``; the uncompressed tail after that
        remains normal ReAct history until it crosses the recompression
        threshold.
        """

        cached_action_steps = getattr(memory, "action_steps", None)
        action_steps = (
            cached_action_steps
            if isinstance(cached_action_steps, list)
            else [step for step in memory.steps if isinstance(step, ActionStep)]
        )
        start = max(0, min(int(start_action_index or 0), len(action_steps)))
        selected = action_steps[start:]
        messages: List[Message] = []
        for step in selected:
            messages.extend(step.to_messages())

        raw_chars = messages_char_count(messages)
        state.context_stats["last_historical_tool_call_turns"] = 0
        state.context_stats["last_historical_tool_call_chars"] = 0
        state.context_stats["last_recent_raw_turns"] = len(selected)
        state.context_stats["last_recent_raw_turn_chars"] = raw_chars
        state.context_stats["last_compression_raw_tail_steps"] = len(selected)
        state.context_stats["last_compression_raw_tail_chars"] = raw_chars
        state.context_stats["last_compression_raw_tail_start_action_index"] = start
        return messages

    def _recent_action_messages(self, state: AgentState, memory: AgentMemory) -> List[Message]:
        recent_steps = self._effective_recent_steps(state)
        if recent_steps <= 0:
            state.context_stats["last_recent_raw_turns"] = 0
            state.context_stats["last_recent_raw_turn_chars"] = 0
            return []

        cached_action_steps = getattr(memory, "action_steps", None)
        action_steps = (
            cached_action_steps
            if isinstance(cached_action_steps, list)
            else [step for step in memory.steps if isinstance(step, ActionStep)]
        )
        selected = action_steps[-recent_steps:]
        if not selected:
            state.context_stats["last_recent_raw_turns"] = 0
            state.context_stats["last_recent_raw_turn_chars"] = 0
            return []

        messages: List[Message] = []
        for step in selected:
            messages.extend(step.to_messages())

        state.context_stats["last_recent_raw_turns"] = len(selected)
        state.context_stats["last_recent_raw_turn_chars"] = messages_char_count(messages)
        return messages

    def _working_context_packet(
        self,
        state: AgentState,
        memory: AgentMemory,
        legacy_chars: int,
        *,
        packet_mode: str,
    ) -> str:
        ts_view = task_state_prompt_view(state.current_task_state)
        recent_action_numbers_list = self._recent_action_numbers(state)
        recent_action_numbers = set(recent_action_numbers_list)
        old_action_ledger = [
            item
            for item in state.action_ledger
            if self._coerce_int(item.get("step")) not in recent_action_numbers
        ]
        packet: JsonDict = {
            "render_mode": packet_mode,
            "legacy_prompt_chars_if_full": legacy_chars,
            "task_card": {
                "task_index": state.task_context.task_index,
                "bench_name": state.task_context.bench_name,
                "question_location": "The original task is in the prior New task message.",
            },
            "recent_raw_window": {
                "location": "Immediately after this packet as original assistant/user ReAct messages.",
                "step_count": len(recent_action_numbers_list),
                "steps": recent_action_numbers_list,
                "raw_observation_policy": "untruncated selected recent turns; default window is 5, adaptive 3/1/0 only when packet plus recent raw turns exceeds the prompt budget",
                "note": (
                    "This packet does not duplicate recent raw observations. Use the following assistant/user "
                    "messages for short-term continuity and latest tool results."
                ),
            },
            "task_state_view": ts_view,
            "task_state_support_audit": task_state_support_audit(state.current_task_state),
            "answer_critical_pins": self._answer_critical_pins(state),
            "answer_critical_memory": self._answer_critical_memory(state),
            "old_action_ledger": {
                "total_tool_calls": len(state.action_ledger),
                "old_tool_calls_in_packet": len(old_action_ledger),
                "omitted_recent_steps": recent_action_numbers_list,
                "items": self._action_ledger_lite(
                    old_action_ledger,
                    omit_result_steps=set(),
                    include_result_summary=True,
                    args_char_limit=420,
                ),
            },
            "evidence_ledger": {
                "total_evidence_items": len(state.evidence_ledger),
                "items": self._pinned_evidence(state),
            },
            "calc_ledger": self._calc_ledger_lite(state.calc_ledger),
            "replay_hints": [
                "RawTrace keeps the full transcript on disk. In this online prompt, older action/observation history is compressed into this packet, and only the selected recent raw turns are appended after it.",
                "Read the assistant/user messages immediately after this packet first; they preserve recent ReAct continuity.",
                "Use OldActionLedger for compact older function/action chronology, key arguments, statuses, result summaries, and failure/retry memory.",
                "Use EvidenceLedger for verified facts and compact tool-result findings that are older than the raw recent turns.",
                "Use AnswerCriticalPins first for answer-slot facts, calculation inputs, source fields, units, and basis notes that must not be lost during compression.",
                "Promote compressed facts into task_state.evidence only when the same packet item preserves the needed entity, period/date, metric or source field, value, unit/scale, source locator, and basis/caveat.",
                "If a packet item has a source locator or recovery index but lacks the exact value, unit, period, or source basis required by the answer cell, treat it as a narrow replay target rather than a final answer fact.",
                "Do not declare a required cell not found merely because old raw observations are compressed. A not-found conclusion needs concrete failed attempts or source-gap records in stale_paths, OldActionLedger, FAILURE_CARD, or semantic_replay_requests.",
                "Do not replace exact source-bound packet facts with later, broader, secondary, estimated, or different-unit facts unless the task asks for that basis or a better primary source directly supports it.",
                "Decision cards may appear inside web-reader observations: ANSWER_SLOT_CARD, ENTITY_RESOLUTION_CARD, WEB_SOURCE_CARD, STRUCTURED_NEXT, and FAILURE_CARD. Use them to preserve routing state, not as evidence unless backed by quoted rows/cells.",
                "Before repeating web_search/web_reader, check FAILURE_CARD and WEB_SOURCE_CARD entries for blocked domains, no_relevant_info, source_discovery-only pages, retry_same_source_allowed=no, and structured_next recommendations.",
                "If a prior card reports entity/code/date keys and should_switch_to_structured_tool=yes, prefer a narrow structured tool over another broad web query.",
                "Use DuplicateCallWarnings before repeating the exact same function/arguments; replay only when a specific required row, unit, period, quote, or source detail is still missing.",
                "Observation schemas preserve selected rows, dates, fields, units, table indices, numeric extrema, and provider errors; use them before repeating a tool call.",
                "Structured/table/batch observations may include execution_card, coverage, filled_slots, missing_slots, selected_rows, numeric_extrema, no_rows reasons, provider errors, retry_same_plan_allowed, and can_answer_now; preserve these coverage fields before generic history and use them to target only missing contract cells.",
                "TaskStateView is compact working memory, not a verifier. Use its evidence, gaps, checks, basis notes, and stale paths to continue the task.",
                "AnswerCriticalMemory carries current answer-critical facts, gaps, basis notes, and optional slot records; do not overwrite supported facts without better evidence.",
                "Web-search summaries preserve query, provider, top result title/url/domain/snippet/date, and provider errors; use them before repeating a search query.",
                "SEC table summaries preserve document_key, table_count, candidate table_index values, unit/period hints, matching rows, and returned rows; replay sec_reader_read_tables only when an exact missing row/range is needed.",
                "Web-reader summaries preserve query-focused source evidence and coverage gaps; use them before reading the same document again.",
                "Use CalcLedger for calculation lineage; replay artifacts/document slices only when source, period, unit, or quote detail is missing.",
                "Semantic replay requests are advisory recovery/coverage reports only; they do not create evidence, are not mandatory, and must not be cited as facts.",
            ],
        }
        duplicate_warnings = self._duplicate_call_warnings(state)
        if duplicate_warnings:
            packet["duplicate_call_warnings"] = duplicate_warnings
        replay_requests = self._semantic_replay_requests(state)
        if replay_requests:
            packet["semantic_replay_requests"] = replay_requests
        if packet_mode == "compact":
            packet["pinned_raw_slices"] = self._pinned_raw_slices(state, memory)
            artifact_slices = self._artifact_selective_slices(state)
            if artifact_slices:
                packet["artifact_selective_slices"] = artifact_slices
            packet["artifact_refs"] = self._artifact_refs(state)

        state.context_stats["last_packet_mode"] = packet_mode
        return "WorkingContextPacket:\n" + json_compact(packet)

    def _answer_critical_pins(self, state: AgentState) -> List[JsonDict]:
        cache = self._packet_context(state)
        if "answer_critical_pins" in cache:
            return cache["answer_critical_pins"]
        pins = getattr(state, "answer_critical_pins", None)
        if not isinstance(pins, list) or not pins:
            cache["answer_critical_pins"] = []
            state.context_stats["last_visible_answer_critical_pins"] = 0
            return []
        terms = self._contract_terms(state)
        ranked: List[Tuple[int, int, JsonDict]] = []
        for idx, item in enumerate(pins):
            if not isinstance(item, dict):
                continue
            score = self._answer_pin_score(item, terms)
            ranked.append((score, idx, item))
        if terms:
            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        else:
            ranked.sort(key=lambda row: row[1], reverse=True)

        grouped_ranked = self._aggregate_answer_pin_ranked(ranked)
        limit = max(0, int(self.config.context_answer_pin_limit or 0))
        char_limit = max(0, int(self.config.context_answer_pin_chars or 0))
        group_count = self._answer_pin_group_count(ranked)
        wide_pin_set = len(pins) >= 48 or group_count >= 24
        if wide_pin_set:
            if limit:
                limit = min(limit, 24)
            else:
                limit = 24
            if char_limit:
                char_limit = min(char_limit, 12000)
            else:
                char_limit = 12000
        selected: List[JsonDict] = []
        selected_keys: set = set()
        total_chars = 0
        for _, _, item in grouped_ranked:
            key = str(item.get("pin_key") or item.get("pin_id") or id(item))
            if key in selected_keys:
                continue
            lite = self._answer_pin_lite(item)
            item_chars = len(json_compact(lite))
            if limit and len(selected) >= limit:
                break
            if char_limit and selected and total_chars + item_chars > char_limit:
                continue
            selected.append(lite)
            selected_keys.add(key)
            total_chars += item_chars
        selected.sort(key=lambda item: (int(item.get("first_seen_step") or 0), str(item.get("pin_id") or "")))
        cache["answer_critical_pins"] = selected
        state.context_stats["last_visible_answer_critical_pins"] = len(selected)
        state.context_stats["last_aggregated_answer_critical_pin_groups"] = group_count
        state.context_stats["last_answer_pin_wide_projection"] = bool(wide_pin_set)
        return selected

    def _answer_pin_lite(self, item: JsonDict) -> JsonDict:
        entry = {key: item.get(key) for key in ANSWER_PIN_KEEP_KEYS if item.get(key) not in (None, "", [], {})}
        if "basis" in entry:
            entry["basis"] = self._compact_value(entry["basis"], 1200)
        if "row_values" in entry:
            entry["row_values"] = self._compact_value(entry["row_values"], 900)
        for key in ("quote", "text"):
            if key in entry:
                entry[key] = compact_text(str(entry[key]), 1400)
        return entry

    def _answer_pin_score(self, item: JsonDict, terms: List[str]) -> int:
        text = ledger_search_text(item)
        score = sum(1 for term in terms if term and term in text)
        origin = str(item.get("origin") or "").lower()
        status = str(item.get("status") or "").lower()
        authority = str(item.get("source_authority") or "").lower()
        evidence_strength = str(item.get("evidence_strength") or item.get("value_origin") or "").lower()
        if "task_state" in origin:
            score += 1000
        if status in {"supported", "closed", "done", "complete", "tool_supported_candidate", "selected", "final"}:
            score += 140
        if status in {"candidate", "alternative", "conflict", "conflicting"}:
            score += 80
        if authority in {"official_structured_provider", "official_filing", "official_source", "table_document"}:
            score += 80
        if evidence_strength in {"exact_table_cell", "structured_row", "official_table", "official_filing"}:
            score += 80
        for key in ("value", "unit", "scale", "currency", "source_ref", "document_key", "table_index", "row_locator", "calc_ref"):
            if item.get(key) not in (None, "", [], {}):
                score += 8
        if item.get("superseded_by") not in (None, "", [], {}):
            score -= 500
        return score

    def _aggregate_answer_pin_ranked(self, ranked: List[Tuple[int, int, JsonDict]]) -> List[Tuple[int, int, JsonDict]]:
        """Keep a few high-strength pins per answer slot for prompt visibility.

        The runtime state remains monotonic. This projection prevents wide
        tables or daily panels from emitting one prompt-visible pin per
        intermediate row while preserving selected/conflicting candidates and
        strong provenance for each logical answer slot.
        """

        groups: Dict[str, List[Tuple[int, int, JsonDict]]] = {}
        fallback_groups: Dict[str, List[Tuple[int, int, JsonDict]]] = {}
        fallback: List[Tuple[int, int, JsonDict]] = []
        for row in ranked:
            _, _, item = row
            slot_key = self._answer_pin_slot_key(item)
            if slot_key:
                groups.setdefault(slot_key, []).append(row)
                continue
            fallback_key = self._answer_pin_fallback_group_key(item)
            if fallback_key:
                fallback_groups.setdefault(fallback_key, []).append(row)
            else:
                fallback.append(row)

        output: List[Tuple[int, int, JsonDict]] = []
        for rows in groups.values():
            rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
            output.extend(self._select_answer_pin_group_rows(rows, per_group_limit=2))
        for rows in fallback_groups.values():
            rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
            output.extend(self._select_answer_pin_group_rows(rows, per_group_limit=1))
        output.extend(fallback)
        output.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return output

    @staticmethod
    def _answer_pin_group_count(ranked: List[Tuple[int, int, JsonDict]]) -> int:
        groups = set()
        for _, _, item in ranked:
            if not isinstance(item, dict):
                continue
            key = ContextAssembler._answer_pin_slot_key(item)
            if not key:
                key = ContextAssembler._answer_pin_fallback_group_key(item)
            if not key:
                key = str(item.get("pin_key") or item.get("pin_id") or id(item))
            groups.add(key)
        return len(groups)

    @staticmethod
    def _select_answer_pin_group_rows(
        rows: List[Tuple[int, int, JsonDict]],
        *,
        per_group_limit: int,
    ) -> List[Tuple[int, int, JsonDict]]:
        if per_group_limit <= 0:
            return []
        selected: List[Tuple[int, int, JsonDict]] = []
        selected.append(rows[0])
        if per_group_limit == 1:
            return selected
        for row in rows[1:]:
            status = str(row[2].get("status") or "").lower()
            if status in {"conflict", "conflicting", "alternative", "rejected", "superseded", "reopened"}:
                selected.append(row)
                break
        if len(selected) < per_group_limit and len(rows) > 1:
            selected.append(rows[1])
        return selected[:per_group_limit]

    @staticmethod
    def _answer_pin_slot_key(item: JsonDict) -> str:
        if not isinstance(item, dict):
            return ""
        entity = item.get("entity") or item.get("ticker") or item.get("symbol") or item.get("name")
        period = item.get("period") or item.get("date") or item.get("as_of_date") or item.get("as_of")
        metric = item.get("metric") or item.get("source_field") or item.get("formula") or item.get("subfield")
        if entity in (None, "", [], {}) or period in (None, "", [], {}) or metric in (None, "", [], {}):
            return ""
        composite = {
            "entity": entity,
            "period": period,
            "metric": metric,
            "subfield": item.get("subfield"),
            "unit": item.get("unit") or item.get("scale") or item.get("currency"),
            "basis": item.get("basis") or item.get("source_basis") or item.get("date_basis") or item.get("adjustment"),
        }
        return "slot:" + json_canonical(_drop_empty(composite))

    @staticmethod
    def _answer_pin_fallback_group_key(item: JsonDict) -> str:
        if not isinstance(item, dict):
            return ""
        for key in ("slot_id", "key", "source_ref", "document_key", "artifact_ref"):
            value = item.get(key)
            if value not in (None, "", [], {}):
                return f"{key}:{json_canonical(value)}"
        call_id = item.get("call_id")
        table_index = item.get("table_index")
        source_field = item.get("source_field") or item.get("metric")
        if call_id not in (None, "", [], {}):
            composite = _drop_empty(
                {
                    "call_id": call_id,
                    "table_index": table_index,
                    "source_field": source_field,
                }
            )
            return "call:" + json_canonical(composite)
        return ""

    def _answer_critical_memory(self, state: AgentState) -> JsonDict:
        ts_view = task_state_prompt_view(state.current_task_state)
        answer_pins = self._answer_critical_pins(state)
        memory: JsonDict = {}
        for key in (
            "contract",
            "progress",
            "evidence",
            "gaps",
            "checks",
            "stale_paths",
            "basis_notes",
            "answer_state",
            "answer_pins",
            "batch_coverage",
            "open_slots",
            "blocked_slots",
        ):
            if ts_view.get(key) not in (None, "", [], {}):
                memory[key] = ts_view.get(key)
        pin_coverage = self._answer_pin_batch_coverage(state)
        if pin_coverage:
            existing = memory.get("batch_coverage")
            if isinstance(existing, dict):
                existing.setdefault("pin_slot_coverage", pin_coverage)
            elif existing not in (None, "", [], {}):
                memory["batch_coverage"] = {
                    "task_state_batch_coverage": existing,
                    "pin_slot_coverage": pin_coverage,
                }
            else:
                memory["batch_coverage"] = {"pin_slot_coverage": pin_coverage}

        refs = self._answer_state_refs(ts_view)
        if answer_pins:
            refs.update(self._answer_state_refs({"answer_pins": answer_pins}))
        if refs:
            evidence = self._referenced_evidence(state, refs)
            if evidence:
                memory["referenced_evidence"] = evidence
            calc = self._referenced_calc_records(state, refs)
            if calc:
                memory["referenced_calculations"] = calc

        schema_records = self._answer_relevant_schema_records(state, refs)
        if schema_records:
            memory["observation_schemas"] = schema_records
        return _drop_empty(memory)

    def _answer_pin_batch_coverage(self, state: AgentState) -> JsonDict:
        pins = getattr(state, "answer_critical_pins", None)
        if not isinstance(pins, list) or len(pins) < 24:
            return {}
        terms = self._contract_terms(state)
        groups: Dict[str, Tuple[int, JsonDict]] = {}
        for item in pins:
            if not isinstance(item, dict):
                continue
            slot_key = self._answer_pin_slot_key(item)
            if not slot_key:
                continue
            score = self._answer_pin_score(item, terms)
            current = groups.get(slot_key)
            if current is None or score > current[0]:
                groups[slot_key] = (score, item)
        if len(groups) < 12:
            return {}

        slots: List[JsonDict] = []
        for _, item in sorted(groups.values(), key=lambda row: row[0], reverse=True):
            slot = _drop_empty(
                {
                    "entity": item.get("entity") or item.get("ticker") or item.get("symbol") or item.get("name"),
                    "period": item.get("period") or item.get("date") or item.get("as_of_date") or item.get("as_of"),
                    "metric": item.get("metric") or item.get("source_field") or item.get("formula") or item.get("subfield"),
                    "status": item.get("status") or "candidate",
                    "value": item.get("value"),
                    "unit": item.get("unit") or item.get("scale") or item.get("currency"),
                    "basis": item.get("basis") or item.get("source_basis") or item.get("date_basis") or item.get("adjustment"),
                    "source_ref": item.get("source_ref") or item.get("call_id") or item.get("document_key"),
                    "locator": item.get("row_locator") or item.get("table_index") or item.get("page"),
                }
            )
            if slot:
                slots.append(slot)

        char_limit = max(4000, min(18000, int(self.config.context_answer_pin_chars or 0)))
        selected: List[JsonDict] = []
        total_chars = 0
        for slot in slots:
            item_chars = len(json_compact(slot))
            if selected and total_chars + item_chars > char_limit:
                break
            selected.append(slot)
            total_chars += item_chars
        return _drop_empty(
            {
                "slot_count": len(groups),
                "visible_slots": selected,
                "truncated": len(selected) < len(slots),
                "note": "Aggregated from answer-critical pins by entity/period/metric/unit/basis; detailed intermediate rows stay out of prompt-visible pins.",
            }
        )

    def _answer_state_refs(self, task_state: JsonDict) -> set:
        refs = set()

        def visit(value: Any, key_hint: str = "") -> None:
            if value in (None, ""):
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, str(key))
                return
            if isinstance(value, list):
                for item in value:
                    visit(item, key_hint)
                return
            text = str(value).strip()
            if not text:
                return
            lowered_key = key_hint.lower()
            if lowered_key in {
                "evidence_id",
                "evidence_ref",
                "evidence_refs",
                "source_ref",
                "source_refs",
                "calc_ref",
                "calc_refs",
                "call_id",
                "depends_on",
            }:
                refs.add(text)
                return
            if re.match(r"^(?:ev_\d+|s\d+_c\d+|calc_\d+)$", text):
                refs.add(text)

        for key in ("answer_state", "answer_pins", "batch_coverage", "basis_notes", "progress"):
            visit(task_state.get(key), key)
        return refs

    def _referenced_evidence(self, state: AgentState, refs: set) -> List[JsonDict]:
        selected: List[JsonDict] = []
        seen = set()
        ref_index = _evidence_ref_index(state)
        for ref in refs:
            for item in ref_index.get(str(ref), []):
                if not isinstance(item, dict):
                    continue
                fp = str(item.get("fingerprint") or id(item))
                if fp in seen:
                    continue
                seen.add(fp)
                selected.append(self._evidence_entry_lite(item))
        return selected

    def _referenced_calc_records(self, state: AgentState, refs: set) -> List[JsonDict]:
        selected: List[JsonDict] = []
        for item in self._calc_ledger_lite(state.calc_ledger):
            call_id = str(item.get("call_id") or "")
            if call_id and call_id in refs:
                selected.append(item)
        return selected

    def _answer_relevant_schema_records(self, state: AgentState, refs: set) -> List[JsonDict]:
        terms = self._contract_terms(state)
        scored: List[Tuple[int, int, JsonDict]] = []
        schema_items = getattr(state, "evidence_schema_items", None)
        if not isinstance(schema_items, list) or (not schema_items and state.evidence_ledger):
            schema_items = [
                item
                for item in state.evidence_ledger
                if isinstance(item, dict)
                and ("schema" in str(item.get("metric") or "").lower() or item.get("schema") not in (None, "", [], {}))
            ]
            setattr(state, "evidence_schema_items", schema_items)
        for idx, item in enumerate(schema_items):
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric") or "").lower()
            if "schema" not in metric:
                continue
            call_id = str(item.get("call_id") or "")
            score = 1000 if call_id and call_id in refs else 0
            text = ledger_search_text(item)
            score += sum(1 for term in terms if term and term in text)
            scored.append((score, idx, item))
        if not scored:
            return []
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [self._evidence_entry_lite(item) for _, _, item in scored[:16]]
        selected.sort(key=lambda item: str(item.get("evidence_id") or ""))
        return selected

    def _action_ledger_lite(
        self,
        items: List[JsonDict],
        *,
        omit_result_steps: Optional[set] = None,
        include_result_summary: bool = False,
        args_char_limit: int = 420,
    ) -> List[JsonDict]:
        limit = int(self.config.context_action_ledger_limit or 0)
        selected = items if limit <= 0 else items[-limit:]
        return [
            self._action_entry_lite(
                item,
                omit_result_steps=omit_result_steps or set(),
                include_result_summary=include_result_summary,
                args_char_limit=args_char_limit,
            )
            for item in selected
        ]

    def _action_entry_lite(
        self,
        item: JsonDict,
        *,
        omit_result_steps: set,
        include_result_summary: bool,
        args_char_limit: int,
    ) -> JsonDict:
        keys = (
            "step",
            "call_id",
            "function",
            "args_fingerprint",
            "status",
            "provider",
            "error",
            "metadata",
            "artifact_ids",
            "full_observation_chars",
            "prompt_observation_chars",
        )
        entry = {key: item.get(key) for key in keys if item.get(key) not in (None, "", [], {})}
        if (
            include_result_summary
            and item.get("step") not in omit_result_steps
            and item.get("result_summary") not in (None, "", [], {})
        ):
            entry["result_summary"] = compact_text(str(item.get("result_summary")), 650)
        arguments = item.get("native_arguments") if isinstance(item.get("native_arguments"), dict) else item.get("arguments")
        if isinstance(arguments, dict):
            entry["key_args"] = self._key_arguments(arguments, args_char_limit)
        return entry

    def _key_arguments(self, arguments: Any, char_limit: int) -> Any:
        if not isinstance(arguments, dict):
            return self._compact_value(arguments, char_limit)
        keep = (
            "query",
            "search_query",
            "ticker",
            "symbol",
            "code",
            "fund_code",
            "company",
            "entity",
            "function",
            "provider",
            "start_date",
            "end_date",
            "date",
            "trade_date",
            "period",
            "fields",
            "document_key",
            "table_index",
            "row_start",
            "row_end",
            "url",
            "limit",
        )
        selected = {key: arguments.get(key) for key in keep if arguments.get(key) not in (None, "", [], {})}
        if not selected:
            selected = dict(arguments)
        return self._compact_value(selected, char_limit)

    def _pinned_evidence(self, state: AgentState) -> List[JsonDict]:
        cache = self._packet_context(state)
        if "pinned_evidence" in cache:
            return cache["pinned_evidence"]
        evidence = state.evidence_ledger
        if not evidence:
            cache["pinned_evidence"] = []
            return []
        terms = self._contract_terms(state)
        mandatory_call_ids = self._mandatory_evidence_call_ids(state)
        scored: List[Tuple[int, int, JsonDict]] = []
        mandatory: List[Tuple[int, int, JsonDict]] = []
        for idx, item in enumerate(evidence):
            text = ledger_search_text(item)
            score = sum(1 for term in terms if term and term in text)
            row = (score, idx, item)
            call_id = str(item.get("call_id") or "")
            if call_id and call_id in mandatory_call_ids:
                mandatory.append(row)
            else:
                scored.append(row)
        if terms:
            scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
            mandatory.sort(key=lambda row: (row[0], row[1]), reverse=True)
        else:
            scored.sort(key=lambda row: row[1], reverse=True)
            mandatory.sort(key=lambda row: row[1], reverse=True)

        limit = max(0, int(self.config.context_pinned_evidence_limit or 0))
        char_limit = max(0, int(self.config.context_pinned_evidence_chars or 0))
        selected: List[JsonDict] = []
        selected_fps: set = set()
        total_chars = 0
        for _, _, item in mandatory + scored:
            fp = str(item.get("fingerprint") or id(item))
            if fp in selected_fps:
                continue
            lite = self._evidence_entry_lite(item)
            item_chars = len(json_compact(lite))
            if limit and len(selected) >= limit:
                break
            if char_limit and selected and total_chars + item_chars > char_limit:
                continue
            selected.append(lite)
            selected_fps.add(fp)
            total_chars += item_chars
        selected.sort(key=lambda item: str(item.get("evidence_id") or ""))
        cache["pinned_evidence"] = selected
        return selected

    def _evidence_entry_lite(self, item: JsonDict) -> JsonDict:
        keep = (
            "evidence_id",
            "call_id",
            "entity",
            "period",
            "metric",
            "field",
            "field_label",
            "source_field",
            "value",
            "unit",
            "currency",
            "scale",
            "basis",
            "schema",
            "source",
            "provider",
            "endpoint",
            "url",
            "document_key",
            "locator",
            "row_locator",
            "row_values",
            "quote",
            "text",
            "origin",
            "first_seen_step",
            "last_seen_step",
        )
        entry = {key: item.get(key) for key in keep if item.get(key) not in (None, "", [], {})}
        if "schema" in entry:
            entry["schema"] = self._compact_value(entry["schema"], 1800)
        if "row_values" in entry:
            entry["row_values"] = self._compact_value(entry["row_values"], 1800)
        for key in ("quote", "text"):
            if key in entry:
                entry[key] = compact_text(str(entry[key]), self._evidence_text_limit(item))
        return entry

    @staticmethod
    def _evidence_text_limit(item: JsonDict) -> int:
        metric = str(item.get("metric") or "").lower()
        origin = str(item.get("origin") or "").lower()
        if "market_data" in metric or "selected_rows" in metric:
            return 2400
        if "web_search" in metric or "top_results" in metric:
            return 1800
        if "sec_reader_read_tables" in metric or "candidates_or_rows" in metric:
            return 3000
        if "web_reader" in metric or "query_evidence" in metric or "read_filing" in metric:
            return 1800
        if "calculator" in metric:
            return 1200
        if "tool_error" in origin:
            return 1400
        return 900

    def _calc_ledger_lite(self, items: List[JsonDict]) -> List[JsonDict]:
        selected = self._tail(items, self.config.context_calc_ledger_limit)
        output: List[JsonDict] = []
        for item in selected:
            structured = item.get("structured_result") if isinstance(item.get("structured_result"), dict) else {}
            entry = {
                key: item.get(key)
                for key in ("step", "call_id", "expression", "depends_on")
                if item.get(key) not in (None, "", [], {})
            }
            if structured.get("raw") not in (None, ""):
                entry["raw_result"] = structured.get("raw")
            elif item.get("output") not in (None, ""):
                entry["raw_result"] = compact_text(str(item.get("output")), 160)
            output.append(entry)
        return output

    def _packet_semantic_readiness(self, state: AgentState) -> JsonDict:
        cache = self._packet_context(state)
        if "packet_semantic_readiness" in cache:
            return cache["packet_semantic_readiness"]
        recent_steps = set(self._recent_action_numbers(state))
        visible_evidence = self._pinned_evidence(state)
        visible_evidence_by_call: Dict[str, List[JsonDict]] = {}
        for item in visible_evidence:
            if not isinstance(item, dict):
                continue
            call_id = str(item.get("call_id") or "")
            if call_id:
                visible_evidence_by_call.setdefault(call_id, []).append(item)
        visible_evidence_call_ids = {
            str(item.get("call_id") or "")
            for item in visible_evidence
            if isinstance(item, dict) and item.get("call_id")
        }
        visible_calc = self._calc_ledger_lite(state.calc_ledger)
        visible_calc_call_ids = {
            str(item.get("call_id") or "")
            for item in visible_calc
            if isinstance(item, dict) and item.get("call_id")
        }
        visible_action_call_ids = self._visible_action_call_ids(state)
        old_entries: List[JsonDict] = []
        direct_covered = 0
        recoverable_indexes: List[JsonDict] = []
        critical_recoverable_indexes: List[JsonDict] = []
        thin_summaries = 0
        blocking_gaps: List[JsonDict] = []
        artifact_sensitive = self._task_requires_artifact_coverage(state)
        relevance_terms = self._active_slot_terms(state) or self._contract_terms(state)
        has_open_answer_gaps = self._task_state_has_open_answer_gaps(state)
        for entry in state.action_ledger:
            if not isinstance(entry, dict):
                continue
            step = self._coerce_int(entry.get("step"))
            if step is None or step in recent_steps:
                continue
            old_entries.append(entry)
            coverage_level = self._old_action_entry_coverage_level(
                entry,
                visible_action_call_ids,
                visible_evidence_by_call,
                visible_calc_call_ids,
            )
            if coverage_level == "answer_covered":
                direct_covered += 1
                continue
            if coverage_level == "recoverable_index":
                gap = self._semantic_gap_lite(entry, coverage_level=coverage_level)
                recoverable_indexes.append(gap)
                if artifact_sensitive and (
                    has_open_answer_gaps or self._replay_candidate_score(gap, relevance_terms) > 0
                ):
                    critical_recoverable_indexes.append(gap)
                continue
            if coverage_level == "thin_summary_covered":
                gap = self._semantic_gap_lite(entry, coverage_level=coverage_level)
                if artifact_sensitive and (
                    has_open_answer_gaps or self._replay_candidate_score(gap, relevance_terms) > 0
                ):
                    blocking_gaps.append(gap)
                else:
                    thin_summaries += 1
                continue
            blocking_gaps.append(self._semantic_gap_lite(entry, coverage_level=coverage_level))

        limit = 8
        readiness: JsonDict = {
            "ready": not blocking_gaps and not critical_recoverable_indexes,
            "older_tool_calls": len(old_entries),
            "answer_covered_old_tool_calls": direct_covered,
            "recoverable_index_old_tool_calls": len(recoverable_indexes),
            "critical_recoverable_index_old_tool_calls": len(critical_recoverable_indexes),
            "thin_summary_covered_old_tool_calls": thin_summaries,
            "blocking_gap_old_tool_calls": len(blocking_gaps),
            "artifact_sensitive_task": artifact_sensitive,
            "visible_evidence_items": len(visible_evidence),
            "visible_evidence_call_ids": len(visible_evidence_call_ids),
        }
        if recoverable_indexes:
            readiness["recovery_indexes"] = recoverable_indexes[:limit]
        if critical_recoverable_indexes:
            readiness["critical_recovery_indexes"] = critical_recoverable_indexes[:limit]
        if blocking_gaps:
            readiness["blocking_gaps"] = blocking_gaps[:limit]
            readiness["uncovered"] = blocking_gaps[:limit]
        cache["packet_semantic_readiness"] = readiness
        return readiness

    def _semantic_replay_requests(self, state: AgentState) -> List[JsonDict]:
        readiness = state.context_stats.get("packet_semantic_readiness")
        if not isinstance(readiness, dict):
            readiness = self._packet_semantic_readiness(state)
        active_terms = self._active_slot_terms(state) or self._contract_terms(state)
        if not active_terms:
            return []
        candidates: List[JsonDict] = []
        for key in ("recovery_indexes", "blocking_gaps"):
            items = readiness.get(key)
            if isinstance(items, list):
                candidates.extend(item for item in items if isinstance(item, dict))
        items = readiness.get("critical_recovery_indexes")
        critical_signatures = set()
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    candidates.append(item)
                    critical_signatures.add(
                        (
                            str(item.get("call_id") or ""),
                            str(item.get("coverage_level") or ""),
                            json_compact(item.get("key_args") or {}),
                        )
                    )
        if not candidates:
            return []
        deduped_candidates: List[JsonDict] = []
        seen_candidates = set()
        for candidate in candidates:
            key = (
                str(candidate.get("call_id") or ""),
                str(candidate.get("coverage_level") or ""),
                json_compact(candidate.get("key_args") or {}),
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            deduped_candidates.append(candidate)
        candidates = deduped_candidates
        ranked: List[Tuple[int, int, JsonDict]] = []
        for idx, candidate in enumerate(candidates):
            score = self._replay_candidate_score(candidate, active_terms)
            signature = (
                str(candidate.get("call_id") or ""),
                str(candidate.get("coverage_level") or ""),
                json_compact(candidate.get("key_args") or {}),
            )
            if signature in critical_signatures:
                score = max(score, 1)
            if score > 0:
                ranked.append((score, idx, candidate))
        if not ranked:
            return []
        ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        requests: List[JsonDict] = []
        for _, _, gap in ranked[:3]:
            item = dict(gap)
            item["instruction"] = (
                "This is an advisory recovery request, not evidence. Replay only if a current open or "
                "unsupported answer slot still needs this call's exact row/date/unit/source and the value is "
                "not already available in AnswerCriticalMemory, EvidenceLedger, CalcLedger, "
                "the raw recent assistant/user turns, or pinned slices."
            )
            requests.append(item)
        return requests

    def _old_action_entry_coverage_level(
        self,
        entry: JsonDict,
        visible_action_call_ids: set,
        visible_evidence_by_call: Dict[str, List[JsonDict]],
        visible_calc_call_ids: set,
    ) -> str:
        call_id = str(entry.get("call_id") or "")
        if call_id and call_id not in visible_action_call_ids:
            return "blocking_gap"
        status = str(entry.get("status") or "").lower()
        if status in {"error", "missing"}:
            return "thin_summary_covered"
        evidence_items = visible_evidence_by_call.get(call_id, []) if call_id else []
        evidence_level = self._visible_evidence_coverage_level(evidence_items)
        if evidence_level in {"answer_covered", "recoverable_index", "thin_summary_covered"}:
            return evidence_level
        if self._action_entry_needs_visible_evidence(entry):
            if self._action_entry_has_recovery_index(entry):
                return "recoverable_index"
            return "blocking_gap"
        if str(entry.get("tool") or "") == "calculator":
            return "answer_covered" if call_id and call_id in visible_calc_call_ids else "blocking_gap"
        return "thin_summary_covered" if self._result_summary_is_semantic(entry) else "blocking_gap"

    def _visible_evidence_coverage_level(self, items: List[JsonDict]) -> str:
        if not items:
            return ""
        has_recovery = False
        has_path = False
        for item in items:
            if not isinstance(item, dict):
                continue
            if self._evidence_item_is_direct(item):
                return "answer_covered"
            if self._evidence_item_is_recovery_index(item):
                has_recovery = True
            elif self._evidence_item_is_path_summary(item):
                has_path = True
        if has_recovery:
            return "recoverable_index"
        if has_path:
            return "thin_summary_covered"
        return ""

    @staticmethod
    def _evidence_item_is_direct(item: JsonDict) -> bool:
        metric = str(item.get("metric") or "").lower()
        unit = str(item.get("unit") or "").lower()
        if item.get("row_values") not in (None, "", [], {}):
            return True
        direct_markers = (
            "selected_row",
            "selected_rows",
            "numeric_extrema",
            "calculator_result",
            "query_evidence",
            "candidates_or_rows",
        )
        if any(marker in metric for marker in direct_markers):
            return True
        if unit not in {"", "schema", "results", "tables", "rows"} and item.get("value") not in (None, "", [], {}):
            return True
        return False

    @staticmethod
    def _evidence_item_is_recovery_index(item: JsonDict) -> bool:
        metric = str(item.get("metric") or "").lower()
        unit = str(item.get("unit") or "").lower()
        if "table_schema" in metric or "table_preview" in metric or metric.endswith(" schema") or unit == "schema":
            return True
        if item.get("schema") not in (None, "", [], {}):
            return True
        for key in ("document_key", "row_locator", "artifact_ids"):
            if item.get(key) not in (None, "", [], {}):
                return True
        return False

    @staticmethod
    def _evidence_item_is_path_summary(item: JsonDict) -> bool:
        metric = str(item.get("metric") or "").lower()
        if "web_search" in metric or "top_results" in metric:
            return True
        if item.get("source") not in (None, "", [], {}) or item.get("url") not in (None, "", [], {}):
            return True
        return False

    def _action_entry_has_recovery_index(self, entry: JsonDict) -> bool:
        if entry.get("artifact_ids") not in (None, "", [], {}):
            return True
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        for source in (metadata, arguments):
            for key in (
                "document_key",
                "table_index",
                "row_start",
                "row_end",
                "url",
                "source",
                "source_url",
                "page_url",
                "api_url",
                "sheet",
                "doc_id",
                "table_count",
            ):
                if source.get(key) not in (None, "", [], {}):
                    return True
        return False

    def _active_slot_terms(self, state: AgentState) -> List[str]:
        cache = self._packet_context(state)
        if "active_slot_terms" in cache:
            return cache["active_slot_terms"]
        ts = state.current_task_state if isinstance(state.current_task_state, dict) else {}
        values: List[Any] = []
        for key in ("open_slots", "blocked_slots", "gaps", "next_focus"):
            if ts.get(key) not in (None, "", [], {}):
                values.append(ts.get(key))
        answer_state = ts.get("answer_state")
        if isinstance(answer_state, list):
            for item in answer_state:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "").strip().lower()
                if status not in {"supported", "closed", "done"}:
                    values.append(item)
        if not values:
            cache["active_slot_terms"] = []
            return []
        terms = self._terms_from_values(values, limit=80)
        cache["active_slot_terms"] = terms
        return terms

    @staticmethod
    def _terms_from_values(values: List[Any], limit: int = 80) -> List[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "what",
            "was",
            "were",
            "are",
            "is",
            "in",
            "on",
            "of",
            "to",
            "by",
            "as",
            "at",
            "or",
            "an",
            "a",
            "it",
            "its",
            "this",
            "that",
            "slot",
            "status",
            "empty",
            "candidate",
            "supported",
            "blocked",
            "metric",
            "period",
            "source",
            "value",
            "unit",
        }
        terms: List[str] = []
        seen = set()
        for value in values:
            for text in _iter_strings(value):
                for term in re.split(r"[\s,;，；、。！？!?()\[\]{}<>《》:：\"'|/\\\\]+", str(text or "")):
                    term = term.strip().lower()
                    if 2 <= len(term) <= 50 and term not in seen and term not in stopwords:
                        seen.add(term)
                        terms.append(term)
                        if len(terms) >= limit:
                            return terms
        return terms

    @staticmethod
    def _replay_candidate_score(candidate: JsonDict, active_terms: List[str]) -> int:
        text = json_compact(candidate).lower()
        score = 0
        for term in active_terms:
            if term and term in text:
                score += max(1, min(8, len(term) // 4))
        level = str(candidate.get("coverage_level") or "")
        if score and level == "recoverable_index":
            score += 2
        return score

    def _task_requires_artifact_coverage(self, state: AgentState) -> bool:
        cache = self._packet_context(state)
        if "task_requires_artifact_coverage" in cache:
            return bool(cache["task_requires_artifact_coverage"])

        ts = state.current_task_state if isinstance(state.current_task_state, dict) else {}
        text = " ".join(
            str(value)
            for value in (
                state.task_context.task_prompt,
                ts.get("contract"),
                ts.get("batch_coverage"),
                ts.get("open_slots"),
                ts.get("next_focus"),
            )
            if value not in (None, "", [], {})
        ).lower()
        markers = (
            "top ",
            "bottom ",
            "rank",
            "ranking",
            "sort",
            "constituent",
            "constituents",
            "every ",
            "each ",
            "daily",
            "monthly",
            "quarterly",
            "between ",
            "highest",
            "lowest",
            "largest",
            "smallest",
            "number of",
            "list",
            "筛选",
            "排序",
            "排名",
            "排行",
            "前十",
            "前三",
            "跌幅",
            "涨幅",
            "最高",
            "最低",
            "全部",
            "所有",
            "每",
            "多少天",
            "连续",
            "区间",
            "成分",
            "持仓",
            "表格",
            "明细",
        )
        requires = any(marker in text for marker in markers)
        cache["task_requires_artifact_coverage"] = requires
        return requires

    @staticmethod
    def _task_state_has_open_answer_gaps(state: AgentState) -> bool:
        ts = state.current_task_state if isinstance(state.current_task_state, dict) else {}
        if ts.get("open_slots") not in (None, "", [], {}):
            return True
        answer_state = ts.get("answer_state")
        if isinstance(answer_state, list):
            for item in answer_state:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "").strip().lower()
                if status in TASK_STATE_UNRESOLVED_STATUSES:
                    return True
        return False

    def _mandatory_evidence_call_ids(self, state: AgentState) -> set:
        cache = self._packet_context(state)
        if "mandatory_evidence_call_ids" in cache:
            return cache["mandatory_evidence_call_ids"]
        recent_steps = set(self._recent_action_numbers(state))
        call_ids = set()
        for entry in state.action_ledger:
            if not isinstance(entry, dict):
                continue
            step = self._coerce_int(entry.get("step"))
            if step is None or step in recent_steps:
                continue
            if not self._action_entry_needs_visible_evidence(entry):
                continue
            call_id = str(entry.get("call_id") or "")
            if call_id:
                call_ids.add(call_id)
        cache["mandatory_evidence_call_ids"] = call_ids
        return call_ids

    def _visible_action_call_ids(self, state: AgentState) -> set:
        cache = self._packet_context(state)
        if "visible_action_call_ids" in cache:
            return cache["visible_action_call_ids"]
        limit = int(self.config.context_action_ledger_limit or 0)
        visible_entries = state.action_ledger if limit <= 0 else state.action_ledger[-limit:]
        call_ids = {
            str(entry.get("call_id") or "")
            for entry in visible_entries
            if isinstance(entry, dict) and entry.get("call_id")
        }
        cache["visible_action_call_ids"] = call_ids
        return call_ids

    def _action_entry_needs_visible_evidence(self, entry: JsonDict) -> bool:
        status = str(entry.get("status") or "").lower()
        if status != "success":
            return False
        tool = str(entry.get("tool") or "")
        action = str(entry.get("action") or "")
        if tool == "market_data":
            return True
        if tool == "sec_reader" and action in {"read_tables", "read_filing"}:
            return True
        if tool in {"web_reader", "structured_table_reader"}:
            return True
        if entry.get("artifact_ids") not in (None, "", [], {}):
            return True
        full_chars = self._coerce_int(entry.get("full_observation_chars")) or 0
        return full_chars > max(4000, int(self.config.context_recent_observation_step_chars or 0))

    def _result_summary_is_semantic(self, entry: JsonDict) -> bool:
        summary = str(entry.get("result_summary") or "").strip()
        if not summary:
            return False
        lowered = summary.lower()
        generic_markers = ("observation_chars=", "observation_type=")
        if any(marker in lowered for marker in generic_markers) and not any(
            marker in lowered
            for marker in (
                "selected_rows",
                "candidate_tables",
                "returned_rows",
                "key_values",
                "evidence",
                "document_key",
                "source",
                "row_count",
                "columns",
                "numeric_extrema",
            )
        ):
            return False

        tool = str(entry.get("tool") or "")
        action = str(entry.get("action") or "")
        if tool == "market_data":
            return any(marker in lowered for marker in ("selected_rows", "row_count", "columns", "numeric_extrema", "provider"))
        if tool == "web_search":
            return any(marker in lowered for marker in ("top_results", "url", "snippet", "query", "result_count"))
        if tool == "sec_reader" and action == "read_tables":
            return any(marker in lowered for marker in ("candidate_tables", "returned_rows", "table_count", "table_index"))
        if tool == "structured_table_reader" and action == "discover_tables":
            return any(marker in lowered for marker in ("source_status", "download_assets", "state_controls", "recommended_next", "table_count", "asset_count"))
        if tool == "web_reader" or (tool == "sec_reader" and action == "read_filing"):
            return any(marker in lowered for marker in ("evidence", "key_values", "coverage", "document_key", "source"))
        if tool == "calculator":
            return "raw" in lowered or "structured_result" in lowered
        return len(summary) >= 80

    def _semantic_gap_lite(self, entry: JsonDict, coverage_level: str = "blocking_gap") -> JsonDict:
        reason = (
            "old tool result is represented by a recovery index, not direct answer evidence"
            if coverage_level == "recoverable_index"
            else "old tool result is not fully represented in prompt-visible ledger evidence"
        )
        lite: JsonDict = {
            "step": entry.get("step"),
            "call_id": entry.get("call_id"),
            "function": entry.get("function") or f"{entry.get('tool')}_{entry.get('action')}",
            "status": entry.get("status"),
            "coverage_level": coverage_level,
            "full_observation_chars": entry.get("full_observation_chars"),
            "artifact_ids": entry.get("artifact_ids"),
            "key_args": self._key_arguments(
                entry.get("native_arguments") if isinstance(entry.get("native_arguments"), dict) else entry.get("arguments"),
                360,
            ),
            "recovery_index": self._action_recovery_index_lite(entry),
            "reason": reason,
        }
        return _drop_empty(lite)

    def _action_recovery_index_lite(self, entry: JsonDict) -> JsonDict:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        keep = (
            "document_key",
            "document_kind",
            "table_index",
            "row_start",
            "row_end",
            "url",
            "source",
            "source_url",
            "page_url",
            "api_url",
            "sheet",
            "doc_id",
            "doc_title",
            "table_count",
            "unit_hint",
            "period_hint",
            "source_status",
            "source_authority",
            "asset_count",
            "state_control_count",
            "ajax_candidate_count",
            "recommended_next",
            "state_recommended_next",
            "requested_state",
            "state_applicable",
            "state_match",
            "state_applied",
            "failure_class",
            "retry_same_source_allowed",
            "suggested_recovery",
            "value_origin",
            "inference_risk",
        )
        output: JsonDict = {}
        for source in (metadata, arguments):
            for key in keep:
                if key not in output and source.get(key) not in (None, "", [], {}):
                    output[key] = source.get(key)
        return self._compact_value(output, 900) if output else {}

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _should_use_compact_packet(
        self,
        state: AgentState,
        legacy_chars: int,
        *,
        tools: Optional[List[JsonDict]] = None,
    ) -> bool:
        _, full_limit = self._effective_context_budgets(state, tools=tools)
        if legacy_chars > full_limit:
            return True
        total_obs = int(state.context_stats.get("total_full_observation_chars", 0) or 0)
        max_obs = int(state.context_stats.get("max_full_observation_chars", 0) or 0)
        if total_obs <= 0 and max_obs <= 0:
            for _, step in self._trajectory_action_steps(state):
                obs_len = len(getattr(step, "obs", "") or "")
                total_obs += obs_len
                if obs_len > max_obs:
                    max_obs = obs_len
        state.context_stats["last_total_observation_chars"] = total_obs
        state.context_stats["last_max_observation_chars"] = max_obs
        return max_obs > 64000 or total_obs > 200000

    def _duplicate_call_warnings(self, state: AgentState) -> List[JsonDict]:
        cache = self._packet_context(state)
        if "duplicate_call_warnings" in cache:
            return cache["duplicate_call_warnings"]
        limit = max(0, int(self.config.context_duplicate_warning_limit or 0))
        if limit <= 0:
            cache["duplicate_call_warnings"] = []
            return []
        groups = _action_duplicate_index(state)

        repeated: List[Tuple[int, int, Tuple[str, str, str], List[JsonDict]]] = []
        for key_text, items in groups.items():
            if len(items) < 2:
                continue
            parts = str(key_text).split("\t", 2)
            if len(parts) != 3:
                continue
            key = (parts[0], parts[1], parts[2])
            last_step = max(int(item.get("step") or 0) for item in items)
            repeated.append((len(items), last_step, key, items))
        repeated.sort(key=lambda row: (row[0], row[1]), reverse=True)

        warnings: List[JsonDict] = []
        for count, _, (tool, action, fp), items in repeated[:limit]:
            latest = items[-1]
            statuses = {}
            for item in items:
                status = str(item.get("status") or "missing")
                statuses[status] = statuses.get(status, 0) + 1
            warnings.append(
                _drop_empty(
                    {
                        "function": latest.get("function") or f"{tool}_{action}",
                        "args_fingerprint": fp,
                        "count": count,
                        "call_ids": [item.get("call_id") for item in items[-8:] if item.get("call_id")],
                        "statuses": statuses,
                        "key_args": self._key_arguments(
                            latest.get("native_arguments") if isinstance(latest.get("native_arguments"), dict) else latest.get("arguments"),
                            420,
                        ),
                        "latest_status": latest.get("status"),
                        "latest_result_summary": compact_text(str(latest.get("result_summary") or ""), 700),
                        "instruction": "This exact function/arguments was already tried. Do not repeat it unless a specific required row, unit, period, quote, or source detail is missing from the visible ledgers.",
                    }
                )
            )
        cache["duplicate_call_warnings"] = warnings
        return warnings

    def _artifact_selective_slices(self, state: AgentState) -> List[JsonDict]:
        limit = max(0, int(self.config.context_artifact_slice_limit or 0))
        char_limit = max(0, int(self.config.context_artifact_slice_chars or 0))
        slice_limit = max(1000, int(self.config.context_artifact_slice_step_chars or 0))
        if limit <= 0 or char_limit <= 0 or not state.artifact_store:
            return []

        entries_by_artifact = self._action_entries_by_artifact_id(state)
        candidates: List[Tuple[int, int, JsonDict]] = []
        for idx, (artifact_id, record) in enumerate(state.artifact_store.items()):
            content = str(record.get("content") or "")
            if not content:
                continue
            terms = self._artifact_slice_terms(state, entries_by_artifact.get(artifact_id, []))
            if not terms:
                continue
            match = self._best_observation_slice(content, terms, slice_limit)
            if match is None:
                continue
            score, matched_terms, text = match
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            candidates.append(
                (
                    score,
                    idx,
                    _drop_empty(
                        {
                            "artifact_id": artifact_id,
                            "kind": record.get("kind"),
                            "chars": record.get("chars"),
                            "tool": metadata.get("tool"),
                            "action": metadata.get("action"),
                            "status": metadata.get("status"),
                            "provider": metadata.get("provider"),
                            "matched_terms": matched_terms[:10],
                            "related_call_ids": [
                                entry.get("call_id")
                                for entry in entries_by_artifact.get(artifact_id, [])[:8]
                                if entry.get("call_id")
                            ],
                            "slice": text,
                        }
                    ),
                )
            )

        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected: List[JsonDict] = []
        used_chars = 0
        for _, _, item in candidates:
            item_chars = len(json_compact(item))
            if len(selected) >= limit:
                break
            if selected and used_chars + item_chars > char_limit:
                continue
            if not selected and item_chars > char_limit:
                item = dict(item)
                item["slice"] = compact_text(str(item.get("slice") or ""), max(1000, char_limit))
                item_chars = len(json_compact(item))
            selected.append(item)
            used_chars += item_chars
        selected.sort(key=lambda item: str(item.get("artifact_id") or ""))
        return selected

    @staticmethod
    def _action_entries_by_artifact_id(state: AgentState) -> Dict[str, List[JsonDict]]:
        return _action_artifact_index(state)

    def _artifact_slice_terms(self, state: AgentState, entries: List[JsonDict]) -> List[str]:
        terms = list(self._contract_terms(state))
        for entry in entries:
            arguments = entry.get("arguments")
            if isinstance(arguments, dict):
                terms.extend(_extract_terms(state.task_context.task_prompt, arguments))
        seen = set()
        deduped: List[str] = []
        for term in terms:
            text = str(term or "").strip().lower()
            if len(text) < 2 or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return sorted(deduped[:160], key=len, reverse=True)

    def _pinned_raw_slices(self, state: AgentState, memory: AgentMemory) -> List[JsonDict]:
        limit = max(0, int(self.config.context_pinned_raw_slice_limit or 0))
        char_limit = max(0, int(self.config.context_pinned_raw_slice_chars or 0))
        step_char_limit = max(1000, int(self.config.context_pinned_raw_slice_step_chars or 0))
        if limit <= 0 or char_limit <= 0:
            return []

        recent_steps = self._effective_recent_steps(state)
        action_steps = self._trajectory_action_steps(state)
        older_steps = action_steps[:-recent_steps] if recent_steps else action_steps
        if not older_steps:
            return []

        terms = self._contract_terms(state)
        if not terms:
            return []
        ranked_terms = sorted(terms, key=len, reverse=True)

        candidates: List[Tuple[int, int, JsonDict]] = []
        for idx, (action_number, step) in enumerate(older_steps):
            observation = getattr(step, "obs", "") or ""
            if not observation:
                continue
            match = self._best_observation_slice(observation, ranked_terms, step_char_limit)
            if match is None:
                tool_names = {
                    str(call.get("name") or "")
                    for call in (getattr(step, "tool_calls", []) or [])
                    if isinstance(call, dict)
                }
                high_value_tools = {
                    "calculator",
                    "market_data",
                    "structured_table_reader",
                }
                if not (tool_names & high_value_tools):
                    continue
                score = 1
                matched_terms = [f"tool:{name}" for name in sorted(tool_names & high_value_tools)]
                text = compact_text(observation, step_char_limit)
            else:
                score, matched_terms, text = match
            if score <= 0:
                continue
            candidates.append(
                (
                    score,
                    idx,
                    {
                        "step": action_number,
                        "matched_terms": matched_terms[:8],
                        "tool_calls": self._compact_value(getattr(step, "tool_calls", []) or [], 1200),
                        "slice": text,
                    },
                )
            )

        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected: List[JsonDict] = []
        used_chars = 0
        for _, _, item in candidates:
            item_chars = len(json_compact(item))
            if len(selected) >= limit:
                break
            if selected and used_chars + item_chars > char_limit:
                continue
            if not selected and item_chars > char_limit:
                item = dict(item)
                item["slice"] = compact_text(str(item.get("slice") or ""), max(1000, char_limit))
                item_chars = len(json_compact(item))
            selected.append(item)
            used_chars += item_chars
        selected.sort(key=lambda item: int(item.get("step") or 0))
        return selected

    def _best_observation_slice(
        self,
        observation: str,
        terms: List[str],
        char_limit: int,
    ) -> Optional[Tuple[int, List[str], str]]:
        lowered = observation.lower()
        best: Optional[Tuple[int, int, List[str]]] = None
        for term in terms:
            pos = lowered.find(term)
            if pos < 0:
                continue
            start = max(0, pos - char_limit // 2)
            end = min(len(observation), start + char_limit)
            start = max(0, end - char_limit)
            window = lowered[start:end]
            matched = [candidate for candidate in terms if candidate in window]
            score = sum(max(1, min(8, len(candidate) // 4)) for candidate in matched)
            if best is None or score > best[0]:
                best = (score, start, matched)
        if best is None:
            return None
        score, start, matched = best
        end = min(len(observation), start + char_limit)
        text = observation[start:end]
        if start > 0:
            text = "...[pinned slice starts mid-observation]...\n" + text
        if end < len(observation):
            text = text + "\n...[pinned slice ends mid-observation]..."
        return score, matched, text

    def _contract_terms(self, state: AgentState) -> List[str]:
        cache = self._packet_context(state)
        if "contract_terms" in cache:
            return cache["contract_terms"]
        values: List[str] = [state.task_context.task_prompt]
        ts = state.current_task_state or {}
        if isinstance(ts, dict):
            for key in (
                "contract",
                "progress",
                "answer_state",
                "answer_pins",
                "batch_coverage",
                "open_slots",
                "blocked_slots",
                "gaps",
                "checks",
                "basis_notes",
                "next_focus",
            ):
                values.extend(_iter_strings(ts.get(key)))
        answer_pins = getattr(state, "answer_critical_pins", None)
        if isinstance(answer_pins, list):
            values.extend(_iter_strings(answer_pins))
        terms: List[str] = []
        seen = set()
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "what",
            "was",
            "were",
            "are",
            "is",
            "in",
            "on",
            "of",
            "to",
            "by",
            "as",
            "at",
            "or",
            "an",
            "a",
            "it",
            "its",
            "this",
            "that",
            "format",
            "response",
            "answer",
            "value",
            "metric",
            "period",
            "source",
            "final",
        }
        for value in values:
            for term in re.split(r"[\s,;，；、。！？!?()\[\]{}<>《》:：\"'|/\\\\]+", str(value or "")):
                term = term.strip().lower()
                if 2 <= len(term) <= 50 and term not in seen and term not in stopwords:
                    seen.add(term)
                    terms.append(term)
        output = terms[:120]
        cache["contract_terms"] = output
        return output

    @staticmethod
    def _compact_value(value: Any, char_limit: int) -> Any:
        text = json_compact(value)
        if len(text) <= char_limit:
            return value
        return {"_compact_json": compact_text(text, char_limit)}

    def _artifact_refs(self, state: AgentState) -> List[JsonDict]:
        limit = max(0, int(self.config.context_artifact_ref_limit or 0))
        if limit <= 0:
            return []
        refs: List[JsonDict] = []
        for artifact_id, record in list(state.artifact_store.items())[-limit:]:
            refs.append({key: value for key, value in record.items() if key != "content"})
        return refs

    @staticmethod
    def _tail(items: List[JsonDict], limit: int) -> List[JsonDict]:
        limit = max(0, int(limit or 0))
        if limit <= 0:
            return []
        return items[-limit:]

    def _recent_action_numbers(self, state: AgentState) -> List[int]:
        cache = self._packet_context(state)
        if "recent_action_numbers" in cache:
            return cache["recent_action_numbers"]
        recent_steps = self._effective_recent_steps(state)
        if recent_steps <= 0:
            cache["recent_action_numbers"] = []
            return []
        numbers = [number for number, _ in self._trajectory_action_steps(state)[-recent_steps:]]
        cache["recent_action_numbers"] = numbers
        return numbers

    def _effective_recent_steps(self, state: AgentState) -> int:
        override = getattr(state, "_context_recent_steps_override", None)
        if isinstance(override, int):
            return max(0, override)
        return max(0, int(self.config.context_recent_steps or 0))

    @staticmethod
    def _trajectory_action_steps(state: AgentState) -> List[Tuple[int, Any]]:
        cached = getattr(state, "trajectory_action_steps", None)
        if isinstance(cached, list):
            return cached
        action_steps: List[Tuple[int, Any]] = []
        action_number = 0
        for step in state.trajectory:
            if getattr(step, "name", "") != "action":
                continue
            action_number += 1
            action_steps.append((action_number, step))
        return action_steps

    @staticmethod
    def _record_render_mode(state: AgentState, mode: str) -> None:
        counts = state.context_stats.setdefault("render_mode_counts", {})
        if isinstance(counts, dict):
            counts[mode] = int(counts.get(mode, 0) or 0) + 1
        state.context_stats["last_render_mode"] = mode
