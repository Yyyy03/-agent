from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]


AGENT_TARGET_TYPES = {"plan", "action", "final_answer"}
CONTEXT_COMPRESSION_TARGET_TYPE = "context_compression_sections"
CONTENT_READER_TARGET_TYPE = "content_reader_summary"
DEFAULT_SAMPLE_TYPES = ("agent_policy", "context_compression", "content_reader")
TURN_TARGET_TYPES = {
    "plan": "plan",
    "action": "action",
    "final": "final_answer",
    "context_compression": CONTEXT_COMPRESSION_TARGET_TYPE,
    "compression": CONTEXT_COMPRESSION_TARGET_TYPE,
    "content_reader_summary": CONTENT_READER_TARGET_TYPE,
    "reader": CONTENT_READER_TARGET_TYPE,
}


def _read_jsonl_with_line_numbers(path: Path) -> Iterator[Tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
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


def episode_id(row: JsonDict) -> str:
    return "/".join(
        [
            _safe_id_part(row.get("bench_name")),
            _safe_id_part(row.get("task_index")),
            _safe_id_part(row.get("sample_id")),
        ]
    )


def _split_csv(values: Optional[str], default: Sequence[str]) -> List[str]:
    if values is None:
        return list(default)
    parsed = [part.strip() for part in values.split(",") if part.strip()]
    return parsed or list(default)


def _allowed_statuses(values: Optional[str]) -> List[str]:
    if values is None:
        return ["success"]
    if not str(values).strip():
        return []
    return [part.strip() for part in str(values).split(",") if part.strip()]


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _score_raw_value(row: JsonDict, key: Optional[str]) -> Any:
    if not key:
        return None
    current: Any = row.get("score")
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _score_value(row: JsonDict, key: Optional[str]) -> Optional[float]:
    return _coerce_numeric(_score_raw_value(row, key))


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


def _episode_allowed(row: JsonDict, args: argparse.Namespace) -> bool:
    allowed_statuses = set(_allowed_statuses(args.allowed_status))
    if allowed_statuses and str(row.get("status") or "") not in allowed_statuses:
        return False
    if getattr(args, "require_correct", False):
        if _coerce_bool(_score_raw_value(row, getattr(args, "correct_key", "is_correct"))) is not True:
            return False
    if args.min_score is not None:
        score = _score_value(row, args.score_key)
        if score is None or score < float(args.min_score):
            return False
    return True


def _message_has_target(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str) and bool(content.strip()):
        return True
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _turn_has_required_shape(turn: JsonDict) -> bool:
    messages = turn.get("input_messages")
    output_message = turn.get("output_message")
    if not isinstance(messages, list) or not messages:
        return False
    if not all(isinstance(message, dict) for message in messages):
        return False
    return _message_has_target(output_message)


def _turn_has_policy_or_validation_observation(turn: JsonDict) -> bool:
    observations = turn.get("observations")
    if not isinstance(observations, list):
        return False
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        content = str(observation.get("content") or "").lower()
        status = str(observation.get("status") or "").lower()
        if "[policy." in content or "policy_error" in content:
            return True
        if "validation_hint:" in content:
            return True
        if status == "error" and (
            "missing required" in content
            or "must be a" in content
            or "must be an" in content
            or "invalid argument" in content
        ):
            return True
    return False


def _message_for_export(message: Any, *, strip_reasoning_content: bool) -> Any:
    if not strip_reasoning_content or not isinstance(message, dict):
        return message
    cleaned = dict(message)
    cleaned.pop("reasoning_content", None)
    return cleaned


def _sample_type_for_turn(turn: JsonDict) -> Optional[str]:
    module = str(turn.get("module") or "").strip()
    if module == "context_compression":
        return "context_compression"
    if module == "content_reader":
        return "content_reader"
    if module == "agent_policy":
        return "agent_policy"
    turn_type = str(turn.get("type") or "").strip()
    target_type = str(turn.get("target_type") or "").strip() or TURN_TARGET_TYPES.get(turn_type, turn_type)
    if target_type in AGENT_TARGET_TYPES:
        return "agent_policy"
    if target_type == CONTEXT_COMPRESSION_TARGET_TYPE:
        return "context_compression"
    if target_type == CONTENT_READER_TARGET_TYPE:
        return "content_reader"
    return None


def _target_type_for_turn(turn: JsonDict) -> str:
    turn_type = str(turn.get("type") or "").strip()
    target_type = str(turn.get("target_type") or "").strip()
    return target_type or TURN_TARGET_TYPES.get(turn_type, turn_type)


def _turn_allowed_for_sample_type(
    turn: JsonDict,
    sample_type: str,
    *,
    include_rejected_compression: bool,
) -> bool:
    if sample_type == "agent_policy":
        return bool(turn.get("include_in_sft", True))
    if sample_type == "context_compression":
        if include_rejected_compression:
            return _target_type_for_turn(turn) == CONTEXT_COMPRESSION_TARGET_TYPE
        return bool(turn.get("include_in_sft", True))
    if sample_type == "content_reader":
        return bool(turn.get("include_in_sft", True))
    return False


def _exported_turn_row(
    *,
    row: JsonDict,
    turn: JsonDict,
    source_path: Path,
    line_number: int,
    turn_index: int,
    sample_type: str,
    strip_reasoning_content: bool = False,
) -> JsonDict:
    target_type = _target_type_for_turn(turn)
    return {
        "episode_id": row.get("episode_id") or episode_id(row),
        "bench_name": row.get("bench_name"),
        "task_index": row.get("task_index"),
        "sample_id": row.get("sample_id"),
        "turn_id": turn.get("id") or f"{sample_type}_{turn_index + 1:04d}",
        "turn_index": turn_index,
        "module": sample_type,
        "role": turn.get("type"),
        "target_type": target_type,
        "render_mode": turn.get("render_mode"),
        "step_number": turn.get("step_number"),
        "messages": turn.get("input_messages"),
        "output_message": _message_for_export(
            turn.get("output_message"),
            strip_reasoning_content=strip_reasoning_content,
        ),
        "observations": turn.get("observations") or [],
        "tools": turn.get("tools") if isinstance(turn.get("tools"), list) else [],
        "source": turn.get("source") or {},
        "include_in_sft": bool(turn.get("include_in_sft", True)),
    }


def iter_export_rows(paths: Sequence[Path], args: argparse.Namespace) -> Iterator[JsonDict]:
    wanted_sample_types = set(_split_csv(args.sample_types, DEFAULT_SAMPLE_TYPES))
    counters: Counter[str] = Counter()
    for source_path in paths:
        for line_number, row in _read_jsonl_with_line_numbers(source_path):
            counters["episodes_seen"] += 1
            if not _episode_allowed(row, args):
                counters["episodes_filtered"] += 1
                continue
            exported_for_episode = False
            turn_sources = (
                ("turns", row.get("turns")),
                ("auxiliary_turns", row.get("auxiliary_turns")),
            )
            for source_view, turns in turn_sources:
                if not isinstance(turns, list) or not turns:
                    counters[f"episodes_without_{source_view}"] += 1
                    continue
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        counters["turns_bad_shape"] += 1
                        continue
                    sample_type = _sample_type_for_turn(turn)
                    if sample_type is None or sample_type not in wanted_sample_types:
                        counters["turns_filtered_by_type"] += 1
                        continue
                    if not _turn_allowed_for_sample_type(
                        turn,
                        sample_type,
                        include_rejected_compression=bool(args.include_rejected_compression),
                    ):
                        counters["turns_filtered_by_flags"] += 1
                        continue
                    if not _turn_has_required_shape(turn):
                        counters["turns_bad_shape"] += 1
                        continue
                    if (
                        sample_type == "agent_policy"
                        and not bool(getattr(args, "include_policy_error_turns", False))
                        and _turn_has_policy_or_validation_observation(turn)
                    ):
                        counters["turns_policy_or_validation_filtered"] += 1
                        continue
                    counters[f"exported_{sample_type}"] += 1
                    exported_for_episode = True
                    yield _exported_turn_row(
                        row=row,
                        turn=turn,
                        source_path=source_path,
                        line_number=line_number,
                        turn_index=turn_index,
                        sample_type=sample_type,
                        strip_reasoning_content=bool(getattr(args, "strip_reasoning_content", False)),
                    )
            if not exported_for_episode and args.fail_on_missing_turns:
                raise ValueError(
                    f"{source_path}:{line_number}: row produced no exportable turns. "
                    "Regenerate traces with turns and auxiliary_turns enabled."
                )
    args._export_counters = counters


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand fire_sft_trace_v2 benchmark traces into turn-level prompt-completion "
            "SFT rows. Each output row keeps messages and output_message; training masks messages."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input trace JSONL files.")
    parser.add_argument("--output", default=None, help="Output turn-level SFT JSONL file.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional directory for three module files: sft_agent_policy_events.jsonl, "
            "sft_context_compression_events.jsonl, sft_content_reader_events.jsonl."
        ),
    )
    parser.add_argument(
        "--sample_types",
        default=",".join(DEFAULT_SAMPLE_TYPES),
        help="Comma-separated sample types: agent_policy,context_compression,content_reader.",
    )
    parser.add_argument(
        "--allowed_status",
        default="success",
        help="Comma-separated episode statuses to export. Use empty string to disable status filtering.",
    )
    parser.add_argument("--score_key", default=None, help="Optional dotted score key, e.g. exact_match or metrics.f1.")
    parser.add_argument("--min_score", type=float, default=None, help="Require score_key >= min_score.")
    parser.add_argument(
        "--require_correct",
        action="store_true",
        help="Require a truthy score correctness field before exporting any episode events.",
    )
    parser.add_argument(
        "--correct_key",
        default="is_correct",
        help="Dotted key inside row['score'] used by --require_correct. Defaults to is_correct.",
    )
    parser.add_argument(
        "--include_rejected_compression",
        action="store_true",
        help="Export rejected context_compression events too. Defaults to accepted-for-SFT only.",
    )
    parser.add_argument(
        "--fail_on_missing_turns",
        action="store_true",
        help="Fail if an accepted episode produces no exportable turns instead of skipping it.",
    )
    parser.add_argument(
        "--strip_reasoning_content",
        action="store_true",
        help="Drop output_message.reasoning_content from exported SFT rows while keeping output_message.content unchanged.",
    )
    parser.add_argument(
        "--include_policy_error_turns",
        action="store_true",
        help="Also export agent_policy turns whose observations contain policy or deterministic validation errors.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <output>.manifest.json.",
    )
    return parser.parse_args(argv)


def export_events(
    paths: Sequence[Path | str],
    *,
    output: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    sample_types: Optional[str | Sequence[str]] = None,
    allowed_status: Optional[str] = "success",
    score_key: Optional[str] = None,
    min_score: Optional[float] = None,
    require_correct: bool = False,
    correct_key: str = "is_correct",
    include_rejected_compression: bool = False,
    fail_on_missing_turns: bool = False,
    strip_reasoning_content: bool = False,
    include_policy_error_turns: bool = False,
    manifest: Optional[Path | str] = None,
) -> JsonDict:
    if not output and not output_dir:
        raise ValueError("Either output or output_dir is required.")
    if isinstance(sample_types, str) or sample_types is None:
        sample_types_arg = sample_types or ",".join(DEFAULT_SAMPLE_TYPES)
    else:
        sample_types_arg = ",".join(str(value) for value in sample_types)
    args = argparse.Namespace(
        sample_types=sample_types_arg,
        allowed_status=allowed_status,
        score_key=score_key,
        min_score=min_score,
        require_correct=require_correct,
        correct_key=correct_key,
        include_rejected_compression=include_rejected_compression,
        fail_on_missing_turns=fail_on_missing_turns,
        strip_reasoning_content=strip_reasoning_content,
        include_policy_error_turns=include_policy_error_turns,
    )
    input_paths = [Path(path).expanduser().resolve() for path in paths]
    rows = list(iter_export_rows(input_paths, args))
    counters = Counter(getattr(args, "_export_counters", Counter()))
    sample_counts = Counter(str(row.get("module") or "") for row in rows)
    target_counts = Counter(str(row.get("target_type") or "") for row in rows)
    outputs: Dict[str, Any] = {}
    written_total = 0
    if output_dir:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        for module in DEFAULT_SAMPLE_TYPES:
            module_rows = [row for row in rows if row.get("module") == module]
            module_path = resolved_output_dir / f"sft_{module}_events.jsonl"
            written = _write_jsonl(module_path, module_rows)
            outputs[module] = str(module_path)
            counters[f"rows_written_{module}"] = written
            written_total += written
    if output:
        output_path = Path(output).expanduser().resolve()
        written = _write_jsonl(output_path, rows)
        outputs["combined"] = str(output_path)
        counters["rows_written_combined"] = written
        written_total += written
    counters["rows_written"] = written_total
    manifest_data = {
        "outputs": outputs,
        "inputs": [str(path) for path in input_paths],
        "counters": dict(counters),
        "module_counts": dict(sample_counts),
        "target_type_counts": dict(target_counts),
    }
    manifest_path = (
        Path(manifest).expanduser().resolve()
        if manifest
        else (
            Path(output).expanduser().resolve().with_suffix(Path(output).suffix + ".manifest.json")
            if output
            else Path(output_dir).expanduser().resolve() / "sft_export_manifest.json"
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_data


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        manifest_data = export_events(
            args.input,
            output=args.output,
            output_dir=args.output_dir,
            sample_types=args.sample_types,
            allowed_status=args.allowed_status,
            score_key=args.score_key,
            min_score=args.min_score,
            require_correct=bool(args.require_correct),
            correct_key=args.correct_key,
            include_rejected_compression=bool(args.include_rejected_compression),
            fail_on_missing_turns=bool(args.fail_on_missing_turns),
            strip_reasoning_content=bool(args.strip_reasoning_content),
            include_policy_error_turns=bool(args.include_policy_error_turns),
            manifest=args.manifest,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    counters = manifest_data.get("counters") or {}
    outputs = manifest_data.get("outputs") or {}
    print(f"[fire-agent-sft-export] wrote {counters.get('rows_written', 0)} rows")
    for key, path in outputs.items():
        print(f"[fire-agent-sft-export] {key}: {path}")
    print(f"[fire-agent-sft-export] manifest: {manifest_data.get('manifest_path')}")


if __name__ == "__main__":
    main()
