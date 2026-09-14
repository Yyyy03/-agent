from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from fire_agent.sft.task_state_normalization import normalize_trace_row


JsonDict = Dict[str, Any]
FINAL_TOOL_NAMES = {"final_answer"}


def _read_jsonl(path: Path) -> Iterator[Tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                yield line_number, row


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _safe_id_part(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text.replace("/", "_") or "unknown"


def _episode_id(row: JsonDict) -> str:
    if row.get("episode_id"):
        return str(row["episode_id"])
    return "/".join(
        [
            _safe_id_part(row.get("bench_name")),
            _safe_id_part(row.get("task_index")),
            _safe_id_part(row.get("sample_id")),
        ]
    )


def _get_dotted(mapping: JsonDict, key: str) -> Any:
    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "correct", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "incorrect", "failed"}:
        return False
    return None


def _episode_allowed(row: JsonDict, args: argparse.Namespace) -> Tuple[bool, str]:
    allowed = {part.strip() for part in str(args.allowed_status or "").split(",") if part.strip()}
    if allowed and str(row.get("status") or "") not in allowed:
        return False, "episode_status"
    if args.require_correct:
        value = _get_dotted(row, f"score.{args.correct_key}")
        if _coerce_bool(value) is not True:
            return False, "episode_not_correct"
    return True, ""


def _expand_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        expanded = glob.glob(str(Path(value).expanduser()))
        if expanded:
            paths.extend(Path(item).resolve() for item in expanded)
        else:
            paths.append(Path(value).expanduser().resolve())
    return paths


def _json_loads_object(text: str) -> Tuple[Optional[JsonDict], str]:
    try:
        parsed = json.loads(text) if text.strip() else {}
    except Exception as exc:
        return None, f"arguments_json_decode:{exc}"
    if not isinstance(parsed, dict):
        return None, "arguments_not_object"
    return parsed, ""


def _normalize_tool_call(call: Any) -> Tuple[Optional[JsonDict], str]:
    if not isinstance(call, dict):
        return None, "tool_call_not_object"
    normalized = deepcopy(call)
    function = normalized.get("function")
    if not isinstance(function, dict):
        return None, "tool_call_missing_function"
    function = deepcopy(function)
    name = str(function.get("name") or "").strip()
    if not name:
        return None, "tool_call_missing_name"
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments, reason = _json_loads_object(arguments)
        if arguments is None:
            return None, reason
    elif arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict):
        return None, "arguments_not_object"
    function["arguments"] = arguments
    normalized["function"] = function
    normalized.setdefault("type", "function")
    if not str(normalized.get("id") or "").strip():
        return None, "tool_call_missing_id"
    return normalized, ""


def _normalize_message(message: Any, *, step_loss_mask: int, keep_reasoning: bool) -> Tuple[Optional[JsonDict], str]:
    if not isinstance(message, dict):
        return None, "message_not_object"
    role = str(message.get("role") or "").strip()
    if role not in {"system", "user", "assistant", "tool"}:
        return None, f"unsupported_role:{role or '<empty>'}"
    content = message.get("content", "")
    content_text = content if isinstance(content, str) else str(content or "")
    normalized: JsonDict = {"role": role, "content": content_text, "step_loss_mask": int(step_loss_mask)}
    if role == "tool":
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        if not tool_call_id:
            return None, "tool_message_missing_tool_call_id"
        normalized["tool_call_id"] = tool_call_id
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_calls: List[JsonDict] = []
            for call in tool_calls:
                normalized_call, reason = _normalize_tool_call(call)
                if normalized_call is None:
                    return None, reason
                normalized_calls.append(normalized_call)
            normalized["tool_calls"] = normalized_calls
        reasoning = message.get("reasoning_content")
        if keep_reasoning and isinstance(reasoning, str) and reasoning.strip():
            normalized["reasoning_content"] = reasoning
    return normalized, ""


def _tool_name(call: JsonDict) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    return str((function or {}).get("name") or "").strip()


def _tool_call_ids(calls: Sequence[JsonDict]) -> List[str]:
    return [str(call.get("id") or "").strip() for call in calls if isinstance(call, dict)]


def _schema_by_name(tools: Any) -> Dict[str, JsonDict]:
    schemas: Dict[str, JsonDict] = {}
    if not isinstance(tools, list):
        return schemas
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if name:
            schemas[name] = function
    return schemas


def _type_matches(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for type_name in types:
        if type_name == "string" and isinstance(value, str):
            return True
        if type_name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if type_name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if type_name == "boolean" and isinstance(value, bool):
            return True
        if type_name == "array" and isinstance(value, list):
            return True
        if type_name == "object" and isinstance(value, dict):
            return True
        if type_name in {None, "null"} and value is None:
            return True
    return False


def _validate_arguments_against_tools(calls: Sequence[JsonDict], tools: Any) -> Tuple[bool, str]:
    schemas = _schema_by_name(tools)
    for call in calls:
        name = _tool_name(call)
        if name not in schemas:
            return False, f"target_tool_not_in_schema:{name}"
        arguments = call["function"].get("arguments") or {}
        parameters = schemas[name].get("parameters") if isinstance(schemas[name].get("parameters"), dict) else {}
        required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
        for key in required:
            if key not in arguments:
                return False, f"target_tool_missing_required:{name}.{key}"
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        if parameters.get("additionalProperties") is False:
            extra = sorted(set(arguments) - set(properties))
            if extra:
                return False, f"target_tool_extra_arguments:{name}.{','.join(extra[:5])}"
        for key, value in arguments.items():
            spec = properties.get(key)
            if isinstance(spec, dict) and "type" in spec and not _type_matches(value, spec.get("type")):
                return False, f"target_tool_bad_type:{name}.{key}"
    return True, ""


def _coerce_schema_value(value: Any, expected: Any) -> Tuple[Any, bool]:
    expected_types = expected if isinstance(expected, list) else [expected]
    if _type_matches(value, expected_types):
        return value, True
    if "integer" in expected_types:
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], int):
                return parsed[0], True
            match = re.search(r"-?\d+", text)
            if match:
                return int(match.group()), True
        if isinstance(value, float) and value.is_integer():
            return int(value), True
    if "number" in expected_types and isinstance(value, str):
        try:
            return float(value.strip()), True
        except ValueError:
            pass
    if "array" in expected_types and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return parsed, True
    if "object" in expected_types and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, True
    return value, False


def _canonicalize_target_calls(
    calls: Sequence[JsonDict],
    tools: Any,
    counters: Counter[str],
) -> List[JsonDict]:
    schemas = _schema_by_name(tools)
    canonical: List[JsonDict] = []
    aliases = {
        "table_indices": "table_index",
        "_precision": "precision",
    }
    for call in calls:
        copied = deepcopy(call)
        name = _tool_name(copied)
        schema = schemas.get(name)
        if not isinstance(schema, dict):
            counters[f"dropped_call_tool_not_in_schema:{name}"] += 1
            continue
        parameters = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
        function = copied["function"]
        arguments = dict(function.get("arguments") or {})

        for old_key, new_key in aliases.items():
            if old_key in arguments and old_key not in properties and new_key in properties and new_key not in arguments:
                arguments[new_key] = arguments.pop(old_key)
                counters[f"argument_alias:{name}.{old_key}->{new_key}"] += 1

        if parameters.get("additionalProperties") is False:
            extra = sorted(set(arguments) - set(properties))
            for key in extra:
                arguments.pop(key, None)
                counters[f"removed_extra_argument:{name}.{key}"] += 1

        invalid = False
        for key, value in list(arguments.items()):
            spec = properties.get(key)
            if not isinstance(spec, dict) or "type" not in spec:
                continue
            coerced, ok = _coerce_schema_value(value, spec.get("type"))
            if not ok:
                counters[f"dropped_call_bad_type:{name}.{key}"] += 1
                invalid = True
                break
            if coerced != value:
                arguments[key] = coerced
                counters[f"coerced_argument:{name}.{key}"] += 1
        missing = [key for key in required if key not in arguments]
        if missing:
            counters[f"dropped_call_missing_required:{name}.{','.join(missing)}"] += 1
            invalid = True
        if invalid:
            continue
        function["arguments"] = arguments
        canonical.append(copied)
    return canonical


def _validate_current_observation_links(turn: JsonDict, target: JsonDict) -> Tuple[bool, str]:
    calls = target.get("tool_calls") if isinstance(target.get("tool_calls"), list) else []
    non_final_calls = [call for call in calls if _tool_name(call) not in FINAL_TOOL_NAMES]
    if not non_final_calls:
        return True, ""
    target_ids = _tool_call_ids(non_final_calls)
    if len(set(target_ids)) != len(target_ids):
        return False, "duplicate_target_tool_call_id"
    observations = turn.get("observations") if isinstance(turn.get("observations"), list) else []
    observed_ids = [
        str(item.get("tool_call_id") or "").strip()
        for item in observations
        if isinstance(item, dict) and item.get("role") == "tool" and str(item.get("tool_call_id") or "").strip()
    ]
    retained_observed_ids = [value for value in observed_ids if value in set(target_ids)]
    if retained_observed_ids != target_ids:
        return False, "target_tool_observation_order_mismatch"
    return True, ""


def _validate_history_tool_links(messages: Sequence[JsonDict]) -> Tuple[bool, str]:
    pending: List[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if pending:
                return False, "history_assistant_before_tool_responses_complete"
            calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
            pending = _tool_call_ids(calls)
        elif role == "tool":
            if not pending:
                return False, "history_tool_without_pending_call"
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            expected = pending.pop(0)
            if tool_call_id != expected:
                return False, "history_tool_call_id_order_mismatch"
        elif role in {"system", "user"}:
            if pending:
                return False, "history_user_before_tool_responses_complete"
    return True, ""


def _target_type(turn: JsonDict) -> str:
    value = str(turn.get("target_type") or turn.get("type") or "").strip()
    if value == "final_answer":
        return "final"
    return value


def _is_final_answer_only_episode(row: JsonDict) -> bool:
    tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    tool_names = set(_schema_by_name(tools))
    turns = row.get("turns") if isinstance(row.get("turns"), list) else []
    working_turns = [
        turn
        for turn in turns
        if isinstance(turn, dict) and _target_type(turn) in {"action", "final"}
    ]
    return bool(working_turns) and len(working_turns) <= 1 and tool_names <= FINAL_TOOL_NAMES


class Qwen35Validator:
    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def validate(self, messages: List[JsonDict], tools: List[JsonDict]) -> Tuple[bool, str, Dict[str, int]]:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=tools or None,
                return_dict=False,
            )
            tokenized = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            token_ids = tokenized["input_ids"]
            offsets = tokenized.get("offset_mapping")
            expected = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                tools=tools or None,
                return_dict=False,
            )
        except Exception as exc:
            return False, f"qwen_template:{exc}", {}
        if offsets is None:
            return False, "qwen_tokenizer_no_offsets", {}
        if list(token_ids) != list(expected):
            return False, "qwen_render_tokenize_mismatch", {}
        try:
            loss_mask = self._loss_mask(rendered, offsets, messages)
            loss_tokens = int(sum(loss_mask))
            if loss_tokens <= 0:
                return False, "empty_loss_mask", {}
            masked_text = self._masked_text(rendered, offsets, loss_mask)
        except Exception as exc:
            return False, f"qwen_mask:{exc}", {}
        target = messages[-1] if messages else {}
        reasoning = str(target.get("reasoning_content") or "").strip()
        if reasoning and reasoning[: min(80, len(reasoning))] not in masked_text:
            return False, "masked_span_missing_reasoning", {}
        if "<tool_response>" in masked_text:
            return False, "masked_span_contains_tool_response", {}
        return True, "", {"tokens": len(token_ids), "loss_tokens": loss_tokens}

    @staticmethod
    def _loss_mask(rendered: str, offsets: Sequence[Tuple[int, int]], messages: Sequence[JsonDict]) -> List[int]:
        assistant_header = "<|im_start|>assistant\n"
        think_prefix = "<think>\n"
        end_marker = "<|im_end|>"
        char_mask = [0] * len(rendered)
        cursor = 0
        for message in messages:
            if message.get("role") != "assistant":
                continue
            header_pos = rendered.find(assistant_header, cursor)
            if header_pos < 0:
                raise ValueError("assistant header not found")
            content_start = header_pos + len(assistant_header)
            end_pos = rendered.find(end_marker, content_start)
            if end_pos < 0:
                raise ValueError("assistant end marker not found")
            span_end = end_pos + len(end_marker)
            if span_end < len(rendered) and rendered[span_end] == "\n":
                span_end += 1
            cursor = span_end
            if int(message.get("step_loss_mask", 1)) != 1:
                continue
            mask_start = content_start + len(think_prefix) if rendered[content_start : content_start + len(think_prefix)] == think_prefix else content_start
            for pos in range(mask_start, span_end):
                char_mask[pos] = 1
        prefix = [0]
        for value in char_mask:
            prefix.append(prefix[-1] + value)
        loss_mask: List[int] = []
        for start, end in offsets:
            if end <= start:
                loss_mask.append(0)
            else:
                loss_mask.append(1 if prefix[end] - prefix[start] > 0 else 0)
        return loss_mask

    @staticmethod
    def _masked_text(rendered: str, offsets: Sequence[Tuple[int, int]], loss_mask: Sequence[int]) -> str:
        chunks: List[str] = []
        last_end = -1
        for (start, end), mask in zip(offsets, loss_mask):
            if not mask or end <= start:
                continue
            if start >= last_end:
                chunks.append(rendered[start:end])
                last_end = end
        return "".join(chunks)


def _build_example(
    *,
    row: JsonDict,
    turn: JsonDict,
    turn_index: int,
    source_path: Path,
    line_number: int,
    args: argparse.Namespace,
    validator: Optional[Qwen35Validator],
    repair_counters: Counter[str],
) -> Tuple[Optional[JsonDict], str, Dict[str, int]]:
    target_type = _target_type(turn)
    wanted = {part.strip() for part in args.target_types.split(",") if part.strip()}
    if wanted and target_type not in wanted:
        return None, "target_type_filtered", {}
    input_messages = turn.get("input_messages")
    output_message = turn.get("output_message")
    if not isinstance(input_messages, list) or not isinstance(output_message, dict):
        return None, "bad_turn_shape", {}
    reasoning = output_message.get("reasoning_content")
    if args.require_reasoning and not (isinstance(reasoning, str) and reasoning.strip()):
        return None, "missing_reasoning", {}

    messages: List[JsonDict] = []
    for message in input_messages:
        normalized, reason = _normalize_message(message, step_loss_mask=0, keep_reasoning=False)
        if normalized is None:
            return None, reason, {}
        if normalized.get("reasoning_content"):
            return None, "history_reasoning_not_stripped", {}
        messages.append(normalized)
    target, reason = _normalize_message(output_message, step_loss_mask=1, keep_reasoning=True)
    if target is None:
        return None, reason, {}
    if args.require_reasoning and not str(target.get("reasoning_content") or "").strip():
        return None, "missing_reasoning", {}
    tools = turn.get("tools") if isinstance(turn.get("tools"), list) else row.get("tools") if isinstance(row.get("tools"), list) else []
    target_calls = target.get("tool_calls") if isinstance(target.get("tool_calls"), list) else []
    if args.tool_argument_policy == "canonicalize" and target_calls:
        target_calls = _canonicalize_target_calls(target_calls, tools, repair_counters)
        if target_calls:
            target["tool_calls"] = target_calls
        else:
            target.pop("tool_calls", None)
    if args.drop_no_tool_action and target_type == "action" and not target_calls:
        return None, "action_without_tool_calls", {}
    messages.append(target)

    ok, reason = _validate_history_tool_links(messages[:-1])
    if not ok:
        return None, reason, {}
    ok, reason = _validate_current_observation_links(turn, target)
    if not ok:
        return None, reason, {}
    calls = target.get("tool_calls") if isinstance(target.get("tool_calls"), list) else []
    if calls:
        ok, reason = _validate_arguments_against_tools(calls, tools)
        if not ok:
            return None, reason, {}

    episode_id = _episode_id(row)
    turn_id = str(turn.get("id") or f"{episode_id}:turn_{turn_index + 1:04d}_{target_type}")
    example = {
        "messages": messages,
        "tools": tools,
        "metadata": {
            "episode_id": episode_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "step_number": turn.get("step_number"),
            "target_type": target_type,
            "source_path": str(source_path),
            "source_line": line_number,
            "bench_name": row.get("bench_name"),
            "task_index": row.get("task_index"),
            "sample_id": row.get("sample_id"),
            "sft_quality": row.get("sft_quality"),
            "sft_source_run": row.get("sft_source_run"),
            "sft_answer_correct_judge_fixed": _get_dotted(row, "score.sft_answer_correct_judge_fixed"),
        },
    }
    token_stats: Dict[str, int] = {}
    if validator is not None:
        ok, reason, token_stats = validator.validate(messages, tools)
        if not ok:
            return None, reason, token_stats
        if args.max_seq_length and token_stats.get("tokens", 0) > args.max_seq_length:
            return None, "overlength", token_stats
        example["metadata"]["token_count"] = token_stats.get("tokens")
        example["metadata"]["loss_token_count"] = token_stats.get("loss_tokens")
    return example, "", token_stats


def _split_by_episode(examples: Sequence[JsonDict], *, eval_ratio: float, seed: int) -> Tuple[List[JsonDict], List[JsonDict]]:
    if eval_ratio <= 0:
        return list(examples), []
    by_episode: Dict[str, List[JsonDict]] = defaultdict(list)
    for example in examples:
        episode = str((example.get("metadata") or {}).get("episode_id") or "unknown")
        by_episode[episode].append(example)
    episodes = list(by_episode)
    if len(episodes) <= 1:
        return list(examples), []
    rng = random.Random(seed)
    rng.shuffle(episodes)
    eval_count = max(1, int(round(len(episodes) * eval_ratio)))
    eval_count = min(eval_count, len(episodes) - 1)
    eval_episodes = set(episodes[:eval_count])
    train: List[JsonDict] = []
    eval_rows: List[JsonDict] = []
    for episode in episodes:
        (eval_rows if episode in eval_episodes else train).extend(by_episode[episode])
    return train, eval_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export FIREAgent thinking trajectories to Slime Qwen3.5 SFT JSONL. "
            "The current assistant reasoning_content/content/tool_calls are supervised; "
            "current observations are never included in the same row."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Trace JSONL files or globs.")
    parser.add_argument("--output_dir", default="output/slime_qwen35_agent_sft", help="Directory for train/eval/quarantine/manifest.")
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--quarantine_output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3.5-9B", help="Qwen3.5 tokenizer path for chat-template and mask validation.")
    parser.add_argument("--no_tokenizer_validation", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=65536)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_status", default="success", help="Comma-separated accepted episode statuses; empty disables status filtering.")
    parser.add_argument("--require_correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correct_key", default="is_correct", help="Dotted under score, e.g. is_correct.")
    parser.add_argument("--require_reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_types", default="plan,action,final")
    parser.add_argument(
        "--drop_no_tool_action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop action turns whose current assistant target has no native tool_calls. Plan/final turns are unaffected.",
    )
    parser.add_argument(
        "--task_state_policy",
        choices=("off", "preserve", "carry", "scaffold"),
        default="off",
        help=(
            "Normalize multi-turn TaskState before export. scaffold is recommended for legacy traces "
            "with empty/free-form state; final-answer-only episodes remain exactly {}."
        ),
    )
    parser.add_argument(
        "--tool_argument_policy",
        choices=("strict", "canonicalize"),
        default="strict",
        help=(
            "strict quarantines targets that do not match the current tool schema. "
            "canonicalize removes unsupported arguments, applies safe aliases/type coercions, "
            "and drops only individual calls that still violate required fields."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_output = Path(args.train_output).expanduser().resolve() if args.train_output else output_dir / "train.jsonl"
    eval_output = Path(args.eval_output).expanduser().resolve() if args.eval_output else output_dir / "eval.jsonl"
    quarantine_output = (
        Path(args.quarantine_output).expanduser().resolve() if args.quarantine_output else output_dir / "quarantine.jsonl"
    )
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "manifest.json"
    input_paths = _expand_paths(args.input)
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing input trace files:\n" + "\n".join(missing))

    validator = None if args.no_tokenizer_validation else Qwen35Validator(args.tokenizer)
    counters: Counter[str] = Counter()
    examples: List[JsonDict] = []
    quarantine: List[JsonDict] = []
    token_counts: List[int] = []
    loss_token_counts: List[int] = []
    target_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    task_state_counters: Counter[str] = Counter()
    tool_repair_counters: Counter[str] = Counter()
    episode_id_occurrences: Counter[str] = Counter()

    for path in input_paths:
        for line_number, row in _read_jsonl(path):
            counters["episodes_seen"] += 1
            allowed, reason = _episode_allowed(row, args)
            if not allowed:
                counters[f"dropped_{reason}"] += 1
                continue
            base_episode_id = _episode_id(row)
            episode_id_occurrences[base_episode_id] += 1
            occurrence = episode_id_occurrences[base_episode_id]
            if occurrence > 1:
                sample_id = _safe_id_part(row.get("sample_id"))
                row = deepcopy(row)
                row["episode_id"] = f"{base_episode_id}__dup{occurrence:02d}_{sample_id}"
                counters["duplicate_episode_ids_rewritten"] += 1
            if args.task_state_policy != "off":
                row = normalize_trace_row(
                    row,
                    policy=args.task_state_policy,
                    single_turn=_is_final_answer_only_episode(row),
                    counters=task_state_counters,
                )
            turns = row.get("turns")
            if not isinstance(turns, list) or not turns:
                counters["episodes_without_turns"] += 1
                continue
            for turn_index, turn in enumerate(turns):
                counters["turns_seen"] += 1
                if not isinstance(turn, dict):
                    counters["dropped_bad_turn_shape"] += 1
                    continue
                if turn.get("include_in_sft") is False:
                    counters["dropped_include_in_sft_false"] += 1
                    continue
                example, drop_reason, token_stats = _build_example(
                    row=row,
                    turn=turn,
                    turn_index=turn_index,
                    source_path=path,
                    line_number=line_number,
                    args=args,
                    validator=validator,
                    repair_counters=tool_repair_counters,
                )
                if example is None:
                    counters[f"dropped_{drop_reason}"] += 1
                    quarantine.append(
                        {
                            "reason": drop_reason,
                            "source_path": str(path),
                            "source_line": line_number,
                            "episode_id": _episode_id(row),
                            "turn_index": turn_index,
                            "target_type": _target_type(turn),
                            "token_stats": token_stats,
                        }
                    )
                    continue
                examples.append(example)
                metadata = example["metadata"]
                target_counts[str(metadata.get("target_type") or "unknown")] += 1
                episode_counts[str(metadata.get("episode_id") or "unknown")] += 1
                if metadata.get("token_count"):
                    token_counts.append(int(metadata["token_count"]))
                if metadata.get("loss_token_count"):
                    loss_token_counts.append(int(metadata["loss_token_count"]))
                counters["examples_kept"] += 1
                if args.max_examples and len(examples) >= args.max_examples:
                    break
            if args.max_examples and len(examples) >= args.max_examples:
                break
        if args.max_examples and len(examples) >= args.max_examples:
            break

    train_rows, eval_rows = _split_by_episode(examples, eval_ratio=args.eval_ratio, seed=args.seed)
    written_train = _write_jsonl(train_output, train_rows)
    written_eval = _write_jsonl(eval_output, eval_rows)
    written_quarantine = _write_jsonl(quarantine_output, quarantine)

    def summary(values: Sequence[int]) -> JsonDict:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[max(0, int(len(ordered) * 0.9) - 1)],
            "p99": ordered[max(0, int(len(ordered) * 0.99) - 1)],
            "max": ordered[-1],
        }

    manifest = {
        "inputs": [str(path) for path in input_paths],
        "outputs": {
            "train": str(train_output),
            "eval": str(eval_output),
            "quarantine": str(quarantine_output),
        },
        "counters": dict(counters),
        "written": {
            "train": written_train,
            "eval": written_eval,
            "quarantine": written_quarantine,
        },
        "episodes_kept": len(episode_counts),
        "target_type_counts": dict(target_counts),
        "task_state_counters": dict(task_state_counters),
        "tool_repair_counters": dict(tool_repair_counters),
        "token_count_summary": summary(token_counts),
        "loss_token_count_summary": summary(loss_token_counts),
        "settings": {
            "tokenizer": args.tokenizer,
            "tokenizer_validation": not args.no_tokenizer_validation,
            "max_seq_length": args.max_seq_length,
            "eval_ratio": args.eval_ratio,
            "require_correct": args.require_correct,
            "require_reasoning": args.require_reasoning,
            "target_types": args.target_types,
            "drop_no_tool_action": args.drop_no_tool_action,
            "task_state_policy": args.task_state_policy,
            "tool_argument_policy": args.tool_argument_policy,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
