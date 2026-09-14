from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import json_repair

from .schemas import AgentConfig, AgentState, compact_text


JsonDict = Dict[str, Any]
Message = Dict[str, Any]


REQUIRED_PACKET_KEYS = (
    "render_mode",
    "legacy_prompt_chars_if_full",
    "task_card",
    "recent_raw_window",
    "task_state_view",
    "task_state_support_audit",
    "answer_critical_pins",
    "answer_critical_memory",
    "old_action_ledger",
    "evidence_ledger",
    "calc_ledger",
    "duplicate_call_warnings",
    "semantic_replay_requests",
    "pinned_raw_slices",
    "artifact_selective_slices",
    "artifact_refs",
    "compression_audit",
    "replay_hints",
)

PACKET_OBJECT_KEYS = {
    "task_card",
    "recent_raw_window",
    "task_state_view",
    "task_state_support_audit",
    "answer_critical_memory",
    "old_action_ledger",
    "evidence_ledger",
    "compression_audit",
}

PACKET_LIST_KEYS = {
    "answer_critical_pins",
    "calc_ledger",
    "duplicate_call_warnings",
    "semantic_replay_requests",
    "pinned_raw_slices",
    "artifact_selective_slices",
    "artifact_refs",
    "replay_hints",
}

SECTION_OBJECT_KEYS = {
    "task_state_view",
    "task_state_support_audit",
    "answer_critical_memory",
    "old_action_ledger",
    "evidence_ledger",
    "compression_audit",
}

SECTION_LIST_KEYS = {
    "answer_critical_pins",
    "calc_ledger",
    "duplicate_call_warnings",
    "semantic_replay_requests",
    "pinned_raw_slices",
    "artifact_selective_slices",
    "artifact_refs",
    "replay_hints",
}

SECTION_KEYS = SECTION_OBJECT_KEYS | SECTION_LIST_KEYS
PACKET_ROOT_FRAME_KEYS = {
    "render_mode",
    "legacy_prompt_chars_if_full",
    "task_card",
    "recent_raw_window",
}
WRAPPER_KEYS = (
    "recompressed_memory_sections",
    "memory_sections",
    "compression_sections",
    "compression_patch",
    "working_context_packet",
    "WorkingContextPacket",
    "packet",
    "data",
    "response",
)

ANSWER_MEMORY_LIST_FIELDS = (
    "supported_facts",
    "answer_pins",
    "referenced_evidence",
    "referenced_calculations",
    "answer_candidates",
    "alternative_candidates",
    "rejected_candidates",
    "gaps",
    "basis_notes",
)

SECTION_LIST_LIMITS = {
    "answer_critical_pins": 128,
    "calc_ledger": 120,
    "duplicate_call_warnings": 96,
    "semantic_replay_requests": 96,
    "pinned_raw_slices": 48,
    "artifact_selective_slices": 48,
    "artifact_refs": 128,
    "replay_hints": 96,
}

LEDGER_ITEM_LIMITS = {
    "old_action_ledger": 180,
    "evidence_ledger": 180,
}


CONTEXT_COMPRESSION_SYSTEM_PROMPT = """Compress finance-agent context into the next memory sections for a prompt-facing WorkingContextPacket.

Hard output rules:
1. Return pure JSON only. The first non-whitespace character must be "{" and the last must be "}".
2. Do not output markdown fences, prose, comments, apologies, analysis, or a "WorkingContextPacket:" label.
3. Do not answer the task, plan, call tools, or write runtime memory.
4. Use only source_payload. Do not invent facts, values, dates, units, sources, or confidence.
5. Return only memory section keys. The runtime owns the root frame keys such as render_mode, task_card, recent_raw_window, and legacy_prompt_chars_if_full.

Rolling-compression model:
- The task and plan are already before this packet.
- deterministic_packet_base is the reliable code-generated packet frame and fallback.
- Your output replaces or refines the memory sections inside that base; do not copy the full base.
- The runtime appends raw prompt-facing ReAct action history after the compression point, then the current step instruction.
- If source_payload contains previous_compressed_packet, reconcile that compressed prefix with raw_action_trace into the next concise memory sections.
- You may forget obsolete old compressed details, but the next sections must still preserve the current answer contract, unresolved gaps, supported answer-critical facts, source basis, and replay constraints needed to continue or answer.
- If a deterministic_packet_base section already contains an answer-critical fact and your source_payload does not provide stronger contradictory evidence, either preserve that fact in your returned section or omit the section so the runtime keeps the base. Do not return empty or sparse critical sections just because you are unsure.
- Do not duplicate raw-tail observations that will be appended after this packet.

Preserve in this order:
1. Answer contract: entities, periods, metrics, units, output shape, and required slots.
2. Coverage state: filled/open/blocked/reopened/contradicted slots and missing cells.
3. Answer-critical facts: exact values, units, dates, formulas, comparison basis, rounding basis, caveats.
4. Provenance: step, call_id, function name, source/document_key, table/row/column/page locator, artifact_ref, evidence strength.
5. Tool outcomes: useful result, no rows, field mismatch, retry target, duplicate-call risk.
6. Calculation lineage: inputs, expression, raw result, conversion, rounding, final display basis.
7. Answer-readiness: chosen candidate values, alternative/conflicting basis values, rejected basis values only when disqualified, and whether every required slot is closed.
8. Raw-tail boundary: which steps are appended raw after this packet.

Batch/table coverage invariants:
- First decide whether the task is an explicit batch/table task: multiple named entities, tickers, countries, contracts, dates, rows, or slots must be filled and compared/listed/ranked.
- For ordinary single-fact questions, keep 1-8 high-value records per list.
- For explicit batch/table tasks, preserve every contract slot's compact coverage status. Do not apply the 1-8 record cap to slot coverage.
- Store the compact coverage map inside task_state_view and/or answer_critical_memory; do not add a new root key.
- Each slot record should keep: stable entity key/name, status, metric, period/date, value, unit, conversion/display basis if available, source step/call_id.
- Filled slots are monotonic across rolling compression. A filled slot from previous_compressed_packet or raw_action_trace must not disappear.
- filled_count must not decrease and missing_slots must not grow unless a slot is explicitly reopened due to stronger contradictory evidence.
- If reopened, record entity key/name, previous value, new uncertainty/contradiction, stronger source/provenance, and reason in answer_critical_memory.gaps or compression_audit.caveats.
- answer_critical_pins are high-value handles, not the complete coverage set for a wide table. Do not emit one pin per intermediate row, daily row, or broad candidate row. Put wide coverage in answer_critical_memory.batch_coverage as compact slot records, and keep only final candidates, key conflicting candidates, calculation outputs, and strong source locators in pins.

Evidence discipline:
- Evidence strength order: structured row/table cell > reader quote/extract > TaskState note > search snippet.
- Search snippets and TaskState notes are not source truth unless backed by structured/quoted evidence in source_payload.
- A lower-strength source must not overwrite a higher-strength source for the same entity, period/date, metric, unit, and basis. If they conflict, preserve both compactly and mark the basis conflict in answer_critical_memory.gaps or compression_audit.caveats.
- If uncertain, preserve it as a gap/caveat. Never fill by inference.
- Prefer compact structured records over prose. Shorten excerpts before dropping slot keys, values, units, or source refs.
- For completed historical periods, preserve whether a value is final/actual, forecast/planned/budget, estimate, snippet-only, or inferred. Do not turn an available final official value into "not found" solely because the question used expected/planned/预计 wording.

Answer-closing discipline:
- For each required answer cell that has a plausible value, keep a compact answer candidate with value, unit/scale, period/date, source basis, evidence strength, and step/call_id.
- If multiple candidate values exist for one cell, keep the likely chosen value plus the most important alternative/conflicting value and its basis. Mark a value as rejected only when entity, period/as-of date, metric/source field, unit/scale, source basis, or stronger evidence shows it does not answer the asked cell. This is more useful than a long search chronology.
- If a calculator result or final display rounding is visible, preserve the expression, raw result, conversion, and final display value together. Do not preserve only one number from the calculation.
- If a task appears nearly answerable, prefer preserving the exact cell/value needed to answer over preserving additional broad search attempts.

Required JSON shape:
- Return a single object whose keys are memory sections only.
- Object section keys: task_state_view, task_state_support_audit, answer_critical_memory, old_action_ledger, evidence_ledger, compression_audit.
- Array section keys: answer_critical_pins, calc_ledger, duplicate_call_warnings, semantic_replay_requests, pinned_raw_slices, artifact_selective_slices, artifact_refs, replay_hints.
- You may omit an unchanged memory section; the runtime will keep the deterministic base section.
- If you include a section, it is the next compressed version of that section, not an append-only patch.
- old_action_ledger.items and evidence_ledger.items must be arrays. Use [] for no records.
- evidence_ledger items should use stable fields: fact, value, unit, period, source_ref, locator, quote_or_row_excerpt, evidence_strength.
- Stay under max_packet_chars. Aim near target_packet_chars. Put uncertainty in gaps/caveats, not in long prose.

Recommended output schema:
{
  "task_state_view": {},
  "task_state_support_audit": {},
  "answer_critical_pins": [],
  "answer_critical_memory": {
    "supported_facts": [],
    "answer_pins": [],
    "referenced_evidence": [],
    "referenced_calculations": [],
    "answer_candidates": [],
    "alternative_candidates": [],
    "rejected_candidates": [],
    "gaps": [],
    "basis_notes": []
  },
  "old_action_ledger": {
    "total_tool_calls": 0,
    "old_tool_calls_in_packet": 0,
    "omitted_recent_steps": [],
    "items": [
      {
        "step": 0,
        "call_id": "...",
        "tool": "...",
        "action": "...",
        "status": "success|error|empty|unknown",
        "result_summary": "...",
        "evidence_refs": [],
        "failure_or_retry": ""
      }
    ]
  },
  "evidence_ledger": {
    "total_evidence_items": 0,
    "items": [
      {
        "fact": "...",
        "value": "...",
        "unit": "...",
        "period": "...",
        "source_ref": "...",
        "locator": "...",
        "quote_or_row_excerpt": "...",
        "evidence_strength": "structured_row|reader_quote|task_state|search_snippet"
      }
    ]
  },
  "calc_ledger": [],
  "duplicate_call_warnings": [],
  "semantic_replay_requests": [],
  "pinned_raw_slices": [],
  "artifact_selective_slices": [],
  "artifact_refs": [],
  "compression_audit": {
    "preserved": [],
    "dropped_or_deferred": [],
    "gaps": [],
    "caveats": [],
    "source_basis": "compressed from source_payload only"
  },
  "replay_hints": []
}
"""


CONTEXT_COMPRESSION_MINIMUM_JSON_SKELETON = """{
  "task_state_view": {},
  "task_state_support_audit": {},
  "answer_critical_pins": [],
  "answer_critical_memory": {"supported_facts": [], "answer_pins": [], "referenced_evidence": [], "referenced_calculations": [], "answer_candidates": [], "alternative_candidates": [], "rejected_candidates": [], "gaps": [], "basis_notes": []},
  "old_action_ledger": {"total_tool_calls": 0, "old_tool_calls_in_packet": 0, "omitted_recent_steps": [], "items": []},
  "evidence_ledger": {"total_evidence_items": 0, "items": []},
  "calc_ledger": [],
  "duplicate_call_warnings": [],
  "semantic_replay_requests": [],
  "pinned_raw_slices": [],
  "artifact_selective_slices": [],
  "artifact_refs": [],
  "compression_audit": {"preserved": [], "dropped_or_deferred": [], "gaps": [], "caveats": [], "source_basis": "compressed from source_payload only"},
  "replay_hints": []
}
"""


def should_attempt_context_compression(config: AgentConfig, render_mode: str) -> bool:
    mode = str(getattr(config, "context_compression_mode", "llm") or "llm").strip().lower()
    if mode in {"off", "disabled", "none", "false", "0", "deterministic"}:
        return False
    render = str(render_mode or "").strip().lower()
    return render in {"lean_packet", "compact_packet"}


def build_context_compression_messages(
    state: AgentState,
    *,
    source_payload: JsonDict,
    attempt: int = 1,
    rejected_feedback: Optional[List[JsonDict]] = None,
) -> List[Message]:
    max_chars = max(8000, int(getattr(state.config, "context_compression_max_source_chars", 0) or 0))
    max_packet_chars = max(4000, int(getattr(state.config, "context_compression_max_packet_chars", 0) or 0))
    target_packet_chars = min(max_packet_chars, max(6000, min(26000, max_packet_chars - 3000)))
    configured_retries = max(0, int(getattr(state.config, "context_compression_retries", 0) or 0))
    source_payload = dict(source_payload)
    source_payload["compression_limits"] = {
        "attempt": attempt,
        "target_packet_chars": target_packet_chars,
        "max_packet_chars": max_packet_chars,
        "retry_available": attempt <= configured_retries,
    }
    if rejected_feedback:
        source_payload["verifier_feedback_from_previous_attempt"] = rejected_feedback[:8]
        source_payload["retry_instruction"] = _context_compression_retry_instruction(rejected_feedback)
    payload_text = json.dumps(source_payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(payload_text) > max_chars:
        source_payload = _shrink_source_payload_for_prompt(source_payload)
        payload_text = json.dumps(source_payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(payload_text) > max_chars:
        payload_text = compact_text(payload_text, max_chars)
    retry_preamble = ""
    if rejected_feedback:
        retry_preamble = (
            "This is a retry after the verifier rejected the previous output. "
            "Correct the verifier error before preserving more detail. "
            "Return pure JSON only; no markdown, no prose, no label.\n\n"
        )
    return [
        {"role": "system", "content": CONTEXT_COMPRESSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                retry_preamble
                + "Minimum valid memory-sections skeleton. Copy this structure if needed, then fill only "
                "high-value memory sections from source_payload:\n"
                + CONTEXT_COMPRESSION_MINIMUM_JSON_SKELETON
                + "\n\n"
                "Compress source_payload into the next WorkingContextPacket memory sections described in the system "
                "message. Do not reproduce deterministic_packet_base or any root frame fields. Preserve "
                "answer-critical evidence, provenance, gaps, compressed-prefix outcomes, and replay handles. "
                "If source_payload contains previous_compressed_packet, reconcile that compressed prefix with "
                "raw_action_trace into the next memory sections. Do not duplicate raw-tail observations that the "
                "runtime appends after the packet. Return JSON only and respect compression_limits.\n\n"
                + payload_text
            ),
        },
    ]


def _shrink_source_payload_for_prompt(source_payload: JsonDict) -> JsonDict:
    payload = deepcopy(source_payload)
    actions = payload.get("raw_action_trace")
    if isinstance(actions, list):
        compact_actions: List[JsonDict] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            item = deepcopy(action)
            is_recent = bool(item.get("is_recent_raw_window"))
            assistant_action = item.get("assistant_action")
            if isinstance(assistant_action, dict):
                assistant_action["think"] = compact_text(str(assistant_action.get("think") or ""), 260 if not is_recent else 500)
                tools = assistant_action.get("tools")
                if len(json.dumps(tools, ensure_ascii=False, default=str)) > (900 if not is_recent else 1600):
                    assistant_action["tools"] = compact_text(
                        json.dumps(tools, ensure_ascii=False, default=str),
                        900 if not is_recent else 1600,
                    )
            item["observation_excerpt"] = compact_text(
                str(item.get("observation_excerpt") or ""),
                800 if not is_recent else 2200,
            )
            compact_actions.append(item)
        payload["raw_action_trace"] = compact_actions

    ledgers = payload.get("runtime_ledgers")
    if isinstance(ledgers, dict):
        for key, limit in (
            ("action_ledger", 48),
            ("evidence_ledger", 56),
            ("answer_critical_pins", 56),
            ("calc_ledger", 40),
            ("artifact_refs", 24),
        ):
            value = ledgers.get(key)
            if isinstance(value, list) and len(value) > limit:
                ledgers[key] = value[-limit:]

    base = payload.get("deterministic_packet_base")
    if isinstance(base, dict):
        for key in ("pinned_raw_slices", "artifact_selective_slices", "artifact_refs"):
            value = base.get(key)
            if isinstance(value, list) and len(value) > 12:
                base[key] = value[-12:]
    payload["source_payload_compaction"] = {
        "applied": True,
        "method": "structured_shrink_before_prompt_text_compaction",
    }
    return payload


def _context_compression_retry_instruction(rejected_feedback: List[JsonDict]) -> str:
    reasons = [str(item.get("reason") or "") for item in rejected_feedback if isinstance(item, dict)]
    reason_text = " ".join(reasons).lower()
    base = (
        "The previous compression output was not accepted. Return one valid JSON object containing memory sections "
        "only, stay under max_packet_chars, and preserve only answer-critical facts, provenance, gaps, calculation "
        "lineage, compressed-prefix outcomes, and replay handles. Do not copy deterministic_packet_base or root "
        "frame fields. For explicit batch/table tasks, preserve every still-relevant slot's compact coverage status; "
        "retire obsolete details only when the current answer contract remains covered."
    )
    if "not_json_object" in reason_text or "invalid_json_syntax" in reason_text:
        return (
            base
            + " Verifier error: invalid_json_syntax/not_json_object. Your next output must be valid JSON, "
            "start with '{', and end with '}'. "
            "Do not include markdown fences, a WorkingContextPacket label, natural-language explanation, "
            "analysis text, or any characters outside the JSON object. If uncertain, copy the minimum valid "
            "output skeleton and fill sparse fields."
        )
    if "not_pure_json_object" in reason_text:
        return (
            base
            + " Verifier error: not_pure_json_object. Output the JSON object itself only. The first character "
            "must be '{' and the last character must be '}'. Remove code fences, labels, prose, and explanations."
        )
    if "schema_missing_major_sections" in reason_text or "schema_missing_required_keys" in reason_text:
        return (
            base
            + " Verifier error: schema_missing_required_keys. Use the memory-section keys from the minimum "
            "skeleton. The runtime supplies root frame keys, so do not emit render_mode, task_card, or recent_raw_window."
        )
    if "no_memory_sections" in reason_text:
        return (
            base
            + " Verifier error: no_memory_sections. Include at least one recognized memory section such as "
            "answer_critical_memory, evidence_ledger, old_action_ledger, calc_ledger, semantic_replay_requests, "
            "task_state_view, or compression_audit."
        )
    if "schema_invalid_type" in reason_text:
        return (
            base
            + " Verifier error: schema_invalid_type. Memory object sections must be JSON objects and list sections "
            "must be JSON arrays. In particular, old_action_ledger and evidence_ledger are objects with an "
            "items array, not bare arrays or prose strings."
        )
    if "oversize:" in reason_text:
        return (
            base
            + " Verifier error: oversize. Keep at most 1-4 records per list, shorten excerpts, and move uncertainty "
            "to compact gaps/caveats instead of long prose."
        )
    return base


def parse_context_compression_packet(
    content: Any,
    *,
    packet_mode: str,
    base_packet: Any = None,
) -> Optional[str]:
    packet, _ = parse_context_compression_packet_with_reason(
        content,
        packet_mode=packet_mode,
        base_packet=base_packet,
    )
    return packet


def is_direct_memory_sections_json(content: Any) -> Tuple[bool, str]:
    if not isinstance(content, str):
        return False, "not_string"
    stripped = content.strip()
    if not stripped:
        return False, "empty"
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False, "not_exact_json"
    if not isinstance(parsed, dict):
        return False, "not_json_object"

    keys = set(parsed.keys())
    if keys & PACKET_ROOT_FRAME_KEYS:
        return False, "contains_working_context_packet_root_keys"
    if keys & set(WRAPPER_KEYS):
        return False, "contains_wrapper_key"
    if not keys:
        return False, "empty_object"
    unknown = keys - SECTION_KEYS
    if unknown:
        return False, "unknown_memory_section_keys:" + ",".join(sorted(str(key) for key in unknown)[:8])
    if not keys & SECTION_KEYS:
        return False, "no_memory_sections"

    for key in SECTION_OBJECT_KEYS:
        if key in parsed and not isinstance(parsed.get(key), dict):
            return False, f"invalid_object_section:{key}"
    for key in SECTION_LIST_KEYS:
        if key in parsed and not isinstance(parsed.get(key), list):
            return False, f"invalid_list_section:{key}"
    return True, ""


def parse_context_compression_packet_with_reason(
    content: Any,
    *,
    packet_mode: str,
    max_packet_chars: Optional[int] = None,
    base_packet: Any = None,
) -> Tuple[Optional[str], str]:
    base = context_packet_object_from_text(base_packet, packet_mode=packet_mode)
    parsed, parse_reason = _parse_json_object_tolerant(content)
    if not isinstance(parsed, dict) or not parsed:
        return None, parse_reason or "not_json_object"

    sections = _unwrap_compression_payload(parsed)
    if not _has_compression_payload(sections):
        return None, "no_memory_sections"
    merged = _merge_memory_sections_into_base(base, sections, packet_mode=packet_mode)

    schema_error = _packet_schema_error(merged)
    if schema_error:
        return None, schema_error

    audit = merged.get("compression_audit")
    if not isinstance(audit, dict):
        merged["compression_audit"] = {"source_basis": "compressed from source_payload only"}
    else:
        audit.setdefault("source_basis", "compressed from source_payload only")
        if parse_reason:
            repairs = audit.setdefault("parser_repairs", [])
            if isinstance(repairs, list) and parse_reason not in repairs:
                repairs.append(parse_reason)
    text = "WorkingContextPacket:\n" + json.dumps(merged, ensure_ascii=False, default=str, separators=(",", ":"))
    if max_packet_chars is not None and max_packet_chars > 0 and len(text) > max_packet_chars:
        return None, f"oversize:{len(text)}>{max_packet_chars}"
    return text, ""


def context_packet_object_from_text(content: Any, *, packet_mode: str) -> JsonDict:
    parsed, _ = _parse_json_object_tolerant(content)
    if isinstance(parsed, dict):
        unwrapped = _unwrap_compression_payload(parsed)
        if isinstance(unwrapped, dict) and not _packet_schema_error(unwrapped):
            base = deepcopy(unwrapped)
            base["render_mode"] = packet_mode
            return _normalize_packet_defaults(base, packet_mode=packet_mode)
    return _minimum_packet_object(packet_mode)


def _looks_like_packet(value: JsonDict) -> bool:
    return not _packet_schema_error(value)


def _parse_json_object_tolerant(content: Any) -> Tuple[JsonDict, str]:
    if isinstance(content, dict):
        return deepcopy(content), "dict_input"
    if not isinstance(content, str):
        return {}, "not_json_object"
    stripped = content.strip()
    if not stripped:
        return {}, "not_json_object"

    candidates: List[Tuple[str, str]] = [(stripped, "exact")]
    cleaned = _strip_json_fence(stripped)
    if cleaned != stripped:
        candidates.append((cleaned, "code_fence"))
    label_stripped = _strip_packet_label(cleaned)
    if label_stripped != cleaned:
        candidates.append((label_stripped, "label_prefix"))
    extracted = _extract_balanced_json_object(label_stripped)
    if extracted and extracted != label_stripped:
        candidates.append((extracted, "json_substring"))

    seen = set()
    for candidate, reason in candidates:
        text = candidate.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed, "" if reason == "exact" else reason
        except Exception:
            pass
        try:
            parsed = json_repair.loads(text)
            if isinstance(parsed, dict):
                return parsed, "json_repair" if reason == "exact" else f"{reason}+json_repair"
        except Exception:
            pass
    return {}, "not_json_object"


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _strip_packet_label(text: str) -> str:
    stripped = text.strip()
    for label in ("WorkingContextPacket:", "working_context_packet:", "MemorySections:", "memory_sections:"):
        if stripped.startswith(label):
            return stripped[len(label) :].strip()
    return stripped


def _extract_balanced_json_object(text: str) -> str:
    start_positions = [idx for idx, char in enumerate(text) if char == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1].strip()
    return ""


def _unwrap_compression_payload(parsed: JsonDict) -> JsonDict:
    current = parsed
    for _ in range(3):
        if not isinstance(current, dict):
            return {}
        if current.keys() & (SECTION_KEYS | set(REQUIRED_PACKET_KEYS)):
            return current
        next_value = None
        for key in WRAPPER_KEYS:
            value = current.get(key)
            if isinstance(value, dict):
                next_value = value
                break
        if next_value is None:
            return current
        current = next_value
    return current if isinstance(current, dict) else {}


def _has_compression_payload(value: JsonDict) -> bool:
    if not isinstance(value, dict):
        return False
    if value.keys() & (SECTION_KEYS | set(REQUIRED_PACKET_KEYS)):
        return True
    legacy_keys = {
        "answer_critical_memory_patch",
        "task_state_view_patch",
        "compression_audit_patch",
        "old_action_ledger_upsert",
        "evidence_ledger_upsert",
        "calc_ledger_upsert",
        "semantic_replay_requests_upsert",
    }
    return bool(value.keys() & legacy_keys)


def _minimum_packet_object(packet_mode: str) -> JsonDict:
    return {
        "render_mode": packet_mode,
        "legacy_prompt_chars_if_full": 0,
        "task_card": {
            "task_index": 0,
            "bench_name": "",
            "question_location": "The original task is in the prior New task message.",
        },
        "recent_raw_window": {
            "location": "Immediately after this packet as original assistant/user ReAct messages.",
            "step_count": 0,
            "steps": [],
            "note": "Do not duplicate raw-tail observations in the packet.",
        },
        "task_state_view": {},
        "task_state_support_audit": {},
        "answer_critical_pins": [],
        "answer_critical_memory": {
            "supported_facts": [],
            "answer_pins": [],
            "referenced_evidence": [],
            "referenced_calculations": [],
            "answer_candidates": [],
            "alternative_candidates": [],
            "rejected_candidates": [],
            "gaps": [],
            "basis_notes": [],
        },
        "old_action_ledger": {"total_tool_calls": 0, "old_tool_calls_in_packet": 0, "omitted_recent_steps": [], "items": []},
        "evidence_ledger": {"total_evidence_items": 0, "items": []},
        "calc_ledger": [],
        "duplicate_call_warnings": [],
        "semantic_replay_requests": [],
        "pinned_raw_slices": [],
        "artifact_selective_slices": [],
        "artifact_refs": [],
        "compression_audit": {
            "preserved": [],
            "dropped_or_deferred": [],
            "gaps": [],
            "caveats": [],
            "source_basis": "compressed from source_payload only",
        },
        "replay_hints": [],
    }


def _normalize_packet_defaults(packet: JsonDict, *, packet_mode: str) -> JsonDict:
    base = _minimum_packet_object(packet_mode)
    for key in REQUIRED_PACKET_KEYS:
        value = packet.get(key)
        if value not in (None, ""):
            base[key] = deepcopy(value)
    base["render_mode"] = packet_mode
    return _coerce_packet_sections(base)


def _merge_memory_sections_into_base(base_packet: JsonDict, sections: JsonDict, *, packet_mode: str) -> JsonDict:
    merged = _normalize_packet_defaults(base_packet or {}, packet_mode=packet_mode)
    if not isinstance(sections, dict):
        return merged
    normalized = _normalize_legacy_section_aliases(sections)
    for key in ("task_state_view", "task_state_support_audit", "compression_audit"):
        value = normalized.get(key)
        if isinstance(value, dict):
            merged[key] = _merge_dict_section(merged.get(key), value)
    value = normalized.get("answer_critical_memory")
    if isinstance(value, dict):
        merged["answer_critical_memory"] = _merge_answer_critical_memory(
            merged.get("answer_critical_memory"),
            value,
        )
    for key in ("old_action_ledger", "evidence_ledger"):
        value = normalized.get(key)
        if isinstance(value, dict):
            merged[key] = _merge_ledger_section(key, merged.get(key), value)
    for key in SECTION_LIST_KEYS:
        value = normalized.get(key)
        if isinstance(value, list):
            merged[key] = _merge_list_unique(
                _coerce_list(merged.get(key)),
                value,
                limit=SECTION_LIST_LIMITS.get(key),
            )
    return _coerce_packet_sections(merged)


def _merge_dict_section(base: Any, new: JsonDict, *, list_limit: int = 96) -> JsonDict:
    output = deepcopy(base) if isinstance(base, dict) else {}
    for key, value in new.items():
        if isinstance(value, list) and isinstance(output.get(key), list):
            output[key] = _merge_list_unique(output.get(key), value, limit=list_limit)
        elif isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge_dict_section(output.get(key), value, list_limit=list_limit)
        else:
            output[key] = deepcopy(value)
    return output


def _merge_answer_critical_memory(base: Any, new: JsonDict) -> JsonDict:
    output = deepcopy(base) if isinstance(base, dict) else {}
    for key, value in new.items():
        if key in ANSWER_MEMORY_LIST_FIELDS and isinstance(value, list):
            output[key] = _merge_list_unique(_coerce_list(output.get(key)), value, limit=128)
        elif isinstance(value, list) and isinstance(output.get(key), list):
            output[key] = _merge_list_unique(output.get(key), value, limit=96)
        elif isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge_dict_section(output.get(key), value)
        else:
            output[key] = deepcopy(value)
    return output


def _merge_ledger_section(section_name: str, base: Any, new: JsonDict) -> JsonDict:
    output = deepcopy(base) if isinstance(base, dict) else {}
    base_items = _coerce_list(output.get("items"))
    new_items = _coerce_list(new.get("items"))
    for key, value in new.items():
        if key == "items":
            continue
        if isinstance(value, int) and isinstance(output.get(key), int):
            output[key] = max(int(output.get(key) or 0), int(value or 0))
        elif isinstance(value, list) and isinstance(output.get(key), list):
            output[key] = _merge_list_unique(output.get(key), value, limit=96)
        elif isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge_dict_section(output.get(key), value)
        else:
            output[key] = deepcopy(value)
    output["items"] = _merge_list_unique(
        base_items,
        new_items,
        limit=LEDGER_ITEM_LIMITS.get(section_name),
    )
    if section_name == "old_action_ledger":
        output["old_tool_calls_in_packet"] = max(
            int(output.get("old_tool_calls_in_packet", 0) or 0),
            len(output["items"]),
        )
        output["total_tool_calls"] = max(
            int(output.get("total_tool_calls", 0) or 0),
            len(output["items"]),
        )
    elif section_name == "evidence_ledger":
        output["total_evidence_items"] = max(
            int(output.get("total_evidence_items", 0) or 0),
            len(output["items"]),
        )
    return output


def _coerce_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _merge_list_unique(base: List[Any], new: List[Any], *, limit: Optional[int] = None) -> List[Any]:
    merged: List[Any] = []
    positions: Dict[str, int] = {}
    for item in list(base or []) + list(new or []):
        item_copy = deepcopy(item)
        fingerprint = _memory_item_fingerprint(item_copy)
        if fingerprint in positions:
            merged[positions[fingerprint]] = item_copy
            continue
        if limit is not None and limit > 0 and len(merged) >= limit:
            continue
        positions[fingerprint] = len(merged)
        merged.append(item_copy)
    return merged


def _memory_item_fingerprint(item: Any) -> str:
    if isinstance(item, dict):
        preferred_keys = (
            "id",
            "pin_id",
            "card_id",
            "evidence_id",
            "artifact_id",
            "call_id",
            "step",
            "tool",
            "action",
            "entity",
            "name",
            "metric",
            "field",
            "source_field",
            "field_label",
            "period",
            "period_basis",
            "date",
            "as_of_date",
            "as_of",
            "value",
            "unit",
            "scale",
            "unit_scale",
            "currency",
            "basis",
            "source_basis",
            "basis_type",
            "date_basis",
            "evidence_strength",
            "status",
            "source",
            "document_key",
            "source_ref",
            "locator",
            "table_index",
            "row",
            "column",
            "formula",
            "expression",
            "result",
        )
        projection = {key: item.get(key) for key in preferred_keys if key in item and item.get(key) not in (None, "")}
        if projection:
            return json.dumps(projection, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _normalize_legacy_section_aliases(value: JsonDict) -> JsonDict:
    output = dict(value)
    if isinstance(output.get("answer_critical_memory_patch"), dict) and "answer_critical_memory" not in output:
        output["answer_critical_memory"] = output["answer_critical_memory_patch"]
    if isinstance(output.get("task_state_view_patch"), dict) and "task_state_view" not in output:
        output["task_state_view"] = output["task_state_view_patch"]
    if isinstance(output.get("compression_audit_patch"), dict) and "compression_audit" not in output:
        output["compression_audit"] = output["compression_audit_patch"]
    if isinstance(output.get("old_action_ledger_upsert"), list) and "old_action_ledger" not in output:
        output["old_action_ledger"] = {"items": output["old_action_ledger_upsert"]}
    if isinstance(output.get("evidence_ledger_upsert"), list) and "evidence_ledger" not in output:
        output["evidence_ledger"] = {"items": output["evidence_ledger_upsert"]}
    if isinstance(output.get("calc_ledger_upsert"), list) and "calc_ledger" not in output:
        output["calc_ledger"] = output["calc_ledger_upsert"]
    if isinstance(output.get("semantic_replay_requests_upsert"), list) and "semantic_replay_requests" not in output:
        output["semantic_replay_requests"] = output["semantic_replay_requests_upsert"]
    return output


def _coerce_packet_sections(packet: JsonDict) -> JsonDict:
    for key in PACKET_OBJECT_KEYS:
        if not isinstance(packet.get(key), dict):
            packet[key] = {}
    for key in PACKET_LIST_KEYS:
        if not isinstance(packet.get(key), list):
            packet[key] = []
    old_action_ledger = packet.get("old_action_ledger")
    if isinstance(old_action_ledger, dict):
        if not isinstance(old_action_ledger.get("items"), list):
            old_action_ledger["items"] = []
        old_action_ledger.setdefault("total_tool_calls", len(old_action_ledger.get("items") or []))
        old_action_ledger.setdefault("old_tool_calls_in_packet", len(old_action_ledger.get("items") or []))
        old_action_ledger.setdefault("omitted_recent_steps", [])
    evidence_ledger = packet.get("evidence_ledger")
    if isinstance(evidence_ledger, dict):
        if not isinstance(evidence_ledger.get("items"), list):
            evidence_ledger["items"] = []
        evidence_ledger.setdefault("total_evidence_items", len(evidence_ledger.get("items") or []))
    answer_memory = packet.get("answer_critical_memory")
    if isinstance(answer_memory, dict):
        for key in ANSWER_MEMORY_LIST_FIELDS:
            if not isinstance(answer_memory.get(key), list):
                answer_memory[key] = []
    audit = packet.get("compression_audit")
    if isinstance(audit, dict):
        for key in ("preserved", "dropped_or_deferred", "gaps", "caveats"):
            if not isinstance(audit.get(key), list):
                audit[key] = []
        audit.setdefault("source_basis", "compressed from source_payload only")
    return packet


def _strict_json_shell_reason(content: Any) -> str:
    if isinstance(content, dict):
        return ""
    if not isinstance(content, str):
        return "not_json_object"
    stripped = content.strip()
    if not stripped:
        return "not_json_object"
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return "not_pure_json_object"
    return ""



def _packet_schema_error(value: JsonDict) -> str:
    if not isinstance(value, dict) or not value:
        return "not_json_object"
    missing = [key for key in REQUIRED_PACKET_KEYS if key not in value]
    if missing:
        return "schema_missing_required_keys:" + ",".join(missing[:8])
    for key in PACKET_OBJECT_KEYS:
        if not isinstance(value.get(key), dict):
            return f"schema_invalid_type:{key}:expected_object"
    for key in PACKET_LIST_KEYS:
        if not isinstance(value.get(key), list):
            return f"schema_invalid_type:{key}:expected_array"
    old_action_ledger = value.get("old_action_ledger") or {}
    if not isinstance(old_action_ledger.get("items"), list):
        return "schema_invalid_type:old_action_ledger.items:expected_array"
    evidence_ledger = value.get("evidence_ledger") or {}
    if not isinstance(evidence_ledger.get("items"), list):
        return "schema_invalid_type:evidence_ledger.items:expected_array"
    return ""
