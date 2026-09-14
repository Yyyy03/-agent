from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

from fire_agent.context import (
    TASK_STATE_SCHEMA_INSTRUCTION,
    TASK_STATE_VIEW_KEYS,
    canonicalize_task_state,
    complete_task_state_snapshot,
    merge_task_state,
    task_state_patch_retention_issues,
    task_state_prompt_view,
    task_state_snapshot_issues,
)


JsonDict = Dict[str, Any]
TASK_STATE_POLICIES = {"preserve", "carry", "scaffold"}

_EMPTY_VIEW_RE = re.compile(r"Current TaskStateView: \(empty\)\.[^\n]*")
_PACKET_VIEW_RE = re.compile(r"TaskStateView is included in WorkingContextPacket\.[^\n]*")
_STEP_PROMPT_MARKERS = (
    "Decide your next action",
    "choose the next available tool or final_answer",
    "Use at most ",
    "The step limit has been reached.",
)


def _parse_working_content(content: Any) -> Optional[JsonDict]:
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    if "task_state" not in parsed and "think" not in parsed:
        return None
    return parsed


def _tool_names(message: JsonDict) -> list[str]:
    names: list[str] = []
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = str((function or {}).get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _is_final_message(message: JsonDict, target_type: str) -> bool:
    return target_type in {"final", "final_answer"} or "final_answer" in _tool_names(message)


def _scaffold_state(message: JsonDict, *, target_type: str, completed_actions: int) -> JsonDict:
    final = _is_final_message(message, target_type)
    scaffold: JsonDict = {
        "progress": {
            "status": "answer_ready" if final else "in_progress",
            "completed_actions": max(0, completed_actions),
        }
    }
    tools = _tool_names(message)
    if tools:
        scaffold["next_focus"] = {
            "tools": tools,
        }
    elif final:
        scaffold["next_focus"] = "final_answer"
    return scaffold


def _normalize_assistant_message(
    message: JsonDict,
    state: Optional[JsonDict],
    *,
    policy: str,
    target_type: str,
    completed_actions: int,
    single_turn: bool,
    task: Any,
    counters: Optional[Counter[str]] = None,
) -> Tuple[JsonDict, JsonDict, bool]:
    parsed = _parse_working_content(message.get("content"))
    if parsed is None:
        return message, canonicalize_task_state(state), False

    raw_state = parsed.get("task_state")
    candidate = canonicalize_task_state(raw_state)
    final = _is_final_message(message, target_type)
    if counters is not None:
        counters["working_messages"] += 1
        if not isinstance(raw_state, dict):
            counters["raw_state_missing_or_malformed"] += 1
        elif not raw_state:
            counters["raw_state_empty"] += 1
        else:
            counters["raw_state_nonempty"] += 1
            if single_turn or final:
                counters["raw_state_nonempty_normalizable_terminal_state"] += 1
            free_keys = {str(key) for key in raw_state} - set(TASK_STATE_VIEW_KEYS)
            if free_keys:
                counters["raw_state_free_schema"] += 1
            if not any(str(key) in TASK_STATE_VIEW_KEYS for key in raw_state):
                counters["raw_state_invisible_to_legacy_view"] += 1

    previous = canonicalize_task_state(state)
    if single_turn or final:
        next_state: JsonDict = {}
    elif policy == "preserve":
        next_state = candidate
    elif policy == "scaffold":
        scaffold = _scaffold_state(
            message,
            target_type=target_type,
            completed_actions=completed_actions,
        )
        next_state = complete_task_state_snapshot(
            previous,
            candidate,
            task=task,
            revision=completed_actions + 1,
            next_focus=scaffold.get("next_focus"),
        )
    else:
        next_state = merge_task_state(previous, candidate)

    think = parsed.get("think")
    if not isinstance(think, str):
        think = "" if think in (None, "", [], {}) else str(think)
    normalized_content = json.dumps(
        {
            "think": think,
            "task_state": next_state,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    changed = normalized_content != message.get("content")
    normalized_message = copy.deepcopy(message)
    normalized_message["content"] = normalized_content

    if counters is not None:
        if changed:
            counters["working_messages_changed"] += 1
        if not next_state:
            counters["normalized_state_empty"] += 1
        else:
            counters["normalized_state_nonempty"] += 1
        normalized_kind = "final" if final else "action"
        state_kind = "nonempty" if next_state else "empty"
        counters[f"normalized_{normalized_kind}_state_{state_kind}"] += 1
        if not final and next_state:
            issues = task_state_snapshot_issues(next_state)
            if issues:
                counters["normalized_action_snapshot_incomplete"] += 1
                for issue in issues:
                    counters[f"snapshot_issue_{issue}"] += 1
            else:
                counters["normalized_action_snapshot_complete"] += 1
        if candidate and not final:
            retention_issues = task_state_patch_retention_issues(candidate, next_state)
            if retention_issues:
                counters["raw_nonempty_action_candidate_retention_failed"] += 1
                for issue in retention_issues[:24]:
                    counters[f"candidate_retention_issue_{issue}"] += 1
            else:
                counters["raw_nonempty_action_candidate_retained"] += 1
        if not candidate and previous and next_state:
            counters["empty_state_carried"] += 1
        if policy == "scaffold" and not candidate and not final:
            counters["empty_state_scaffolded"] += 1
    return normalized_message, next_state, changed


def _replace_existing_view_block(text: str, replacement: str) -> Tuple[str, bool]:
    for marker in (
        "Current TaskStateView (compact working memory;",
        "Current TaskStateView (normalized compact working memory):",
    ):
        start = text.find(marker)
        if start < 0:
            continue
        line_start = text.rfind("\n", 0, start) + 1
        use_marker = "\nUse this compact working memory"
        use_start = text.find(use_marker, start)
        if use_start < 0:
            continue
        line_end = text.find("\n", use_start + 1)
        if line_end < 0:
            line_end = len(text)
        return text[:line_start] + replacement + text[line_end:], True
    return text, False


def _dedupe_schema_instruction(text: str) -> str:
    first = text.find(TASK_STATE_SCHEMA_INSTRUCTION)
    if first < 0:
        return text
    search_from = first + len(TASK_STATE_SCHEMA_INSTRUCTION)
    while True:
        duplicate = text.find(TASK_STATE_SCHEMA_INSTRUCTION, search_from)
        if duplicate < 0:
            return text
        remove_start = duplicate - 1 if duplicate > 0 and text[duplicate - 1] == "\n" else duplicate
        text = text[:remove_start] + text[duplicate + len(TASK_STATE_SCHEMA_INSTRUCTION) :]
        search_from = first + len(TASK_STATE_SCHEMA_INSTRUCTION)


def rewrite_task_state_prompt(content: Any, state: Optional[JsonDict]) -> Tuple[Any, bool]:
    if not isinstance(content, str):
        return content, False
    normalized = canonicalize_task_state(state)
    if normalized:
        rendered = json.dumps(task_state_prompt_view(normalized), ensure_ascii=False, indent=2)
        replacement = (
            "Current TaskStateView (normalized compact working memory):\n"
            + rendered
            + "\nUse this compact working memory and return its full update in content.task_state.\n"
            + TASK_STATE_SCHEMA_INSTRUCTION
        )
    else:
        replacement = (
            "Current TaskStateView: (empty). Start a non-empty state for this multi-turn task.\n"
            + TASK_STATE_SCHEMA_INSTRUCTION
        )

    rewritten, changed = _replace_existing_view_block(content, replacement)
    if not changed:
        rewritten, count = _EMPTY_VIEW_RE.subn(replacement, content, count=1)
        changed = count > 0
    if not changed:
        rewritten, count = _PACKET_VIEW_RE.subn(replacement, content, count=1)
        changed = count > 0
    if not changed and "TaskStateView" not in content and any(
        marker in content for marker in _STEP_PROMPT_MARKERS
    ):
        rewritten = content.rstrip() + "\n\n" + replacement + "\n"
        changed = True
    if changed:
        rewritten = _dedupe_schema_instruction(rewritten)
    return rewritten, changed


def _normalize_input_messages(
    messages: Any,
    *,
    policy: str,
    single_turn: bool,
    task: Any,
) -> Tuple[Any, JsonDict, int, int]:
    if not isinstance(messages, list):
        return messages, {}, 0, 0
    normalized_messages = copy.deepcopy(messages)
    state: JsonDict = {}
    action_count = 0
    prompt_rewrites = 0
    for index, message in enumerate(normalized_messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            parsed = _parse_working_content(message.get("content"))
            if parsed is None:
                continue
            message, state, _ = _normalize_assistant_message(
                message,
                state,
                policy=policy,
                target_type="history",
                completed_actions=action_count,
                single_turn=single_turn,
                task=task,
            )
            normalized_messages[index] = message
            action_count += 1
        elif role == "user" and not single_turn:
            rewritten, changed = rewrite_task_state_prompt(message.get("content"), state)
            if changed:
                message["content"] = rewritten
                prompt_rewrites += 1
    return normalized_messages, state, action_count, prompt_rewrites


def normalize_trace_row(
    row: JsonDict,
    *,
    policy: str = "carry",
    single_turn: bool = False,
    counters: Optional[Counter[str]] = None,
) -> JsonDict:
    if policy not in TASK_STATE_POLICIES:
        raise ValueError(f"unsupported TaskState policy: {policy}")
    normalized_row = copy.deepcopy(row)
    turns = normalized_row.get("turns")
    if not isinstance(turns, list):
        return normalized_row
    task = (
        normalized_row.get("question")
        or normalized_row.get("task")
        or normalized_row.get("raw_question")
        or "current user task"
    )

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        target_type = str(turn.get("target_type") or turn.get("type") or "")
        if target_type == "plan":
            continue
        input_messages, state, action_count, prompt_rewrites = _normalize_input_messages(
            turn.get("input_messages"),
            policy=policy,
            single_turn=single_turn,
            task=task,
        )
        turn["input_messages"] = input_messages
        if counters is not None:
            counters["prompt_blocks_rewritten"] += prompt_rewrites

        output = turn.get("output_message")
        if not isinstance(output, dict):
            continue
        output, _, _ = _normalize_assistant_message(
            output,
            state,
            policy=policy,
            target_type=target_type,
            completed_actions=action_count,
            single_turn=single_turn,
            task=task,
            counters=counters,
        )
        turn["output_message"] = output
        if counters is not None and not single_turn:
            user_messages = [
                message
                for message in input_messages
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            target_prompt = str((user_messages[-1] if user_messages else {}).get("content") or "")
            if TASK_STATE_SCHEMA_INSTRUCTION in target_prompt:
                counters["target_prompts_with_task_state_schema"] += 1
            else:
                counters["target_prompts_missing_task_state_schema"] += 1
    return normalized_row
