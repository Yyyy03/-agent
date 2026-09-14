from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmarks.common import read_jsonl, write_jsonl
from .api_balance_circuit_breaker import abort_details, abort_requested
from .benchmark_config import (
    benchmark_template_variables,
    expand_benchmark_templates,
    agent_settings_from_config,
    load_benchmark_config,
    model_reasoning_settings_from_config,
    prompt_settings_from_config,
    resolve_benchmark_config_path,
    tool_settings_from_config,
    tools_from_config,
    validate_benchmark_config,
)
from .config import ModelSettings, load_fire_agent_env
from .llm import OpenAICompatibleChatModel
from .benchmarks import RUNNER_REGISTRY
from .runtime import FIREAgent
from .schemas import AgentConfig
from .tool_families import build_default_tool_registry


def build_model(
    model_name: str | None,
    *,
    reasoning_overrides: Optional[Dict[str, Any]] = None,
) -> OpenAICompatibleChatModel | None:
    settings = ModelSettings.from_env(model_name)
    if reasoning_overrides:
        settings = replace(settings, **reasoning_overrides)
    if not settings.is_configured:
        return None
    return OpenAICompatibleChatModel(settings)


def build_auxiliary_model() -> OpenAICompatibleChatModel | None:
    settings = ModelSettings.from_role_env("AUXILIARY")
    if not settings.is_configured:
        return None
    return OpenAICompatibleChatModel(settings)


def _bool_setting(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _int_setting(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _load_record_count(runner_cls: type, infile: str, outfile: str) -> int:
    # Loading records through the runner preserves benchmark-specific JSON/CSV
    # handling without constructing model clients or tool instances.
    runner = runner_cls(infile, outfile, agent=None)  # type: ignore[arg-type]
    return len(runner.load_records())


def _completed_indices(outfile: str) -> set[int]:
    output_path = Path(outfile)
    if not output_path.exists():
        return set()
    terminal_statuses = {
        "success",
        "max_steps",
        "error",
        "empty_final_answer",
        "protocol_no_answer",
    }
    completed: set[int] = set()
    for row in read_jsonl(output_path):
        if row.get("status") not in terminal_statuses:
            continue
        try:
            completed.add(int(row.get("task_index")))
        except Exception:
            continue
    return completed


def _selected_task_indices(args: argparse.Namespace, runner_cls: type) -> List[int]:
    total_records = _load_record_count(runner_cls, args.infile, args.outfile)
    if args.task_indices is None:
        selected = list(range(total_records))
    else:
        selected = [idx for idx in args.task_indices if 0 <= idx < total_records]
    if args.sample_num is not None:
        selected = selected[: args.sample_num]
    if args.resume:
        done = _completed_indices(args.outfile)
        selected = [idx for idx in selected if idx not in done]
    return selected


def _extend_optional(command: List[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _child_command(args: argparse.Namespace, shard_indices: List[int], part_path: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "fire_agent.cli",
        "--bench",
        args.bench,
        "--infile",
        args.infile,
        "--outfile",
        str(part_path),
        "--max_workers",
        "1",
        "--parallel_backend",
        "thread",
    ]
    _extend_optional(command, "--bench_config", args.bench_config)
    _extend_optional(command, "--bench_configs_dir", args.bench_configs_dir)
    _extend_optional(command, "--model", args.model)
    _extend_optional(command, "--max_steps", args.max_steps)
    _extend_optional(command, "--max_tool_calls_per_round", args.max_tool_calls_per_round)
    _extend_optional(command, "--context_mode", args.context_mode)
    _extend_optional(command, "--task_state_render_mode", args.task_state_render_mode)
    _extend_optional(command, "--env_file", args.env_file)
    if args.no_paid_tools:
        command.append("--no_paid_tools")
    if args.heuristic_only:
        command.append("--heuristic_only")
    if args.tools is not None:
        command.append("--tools")
        command.extend(str(tool) for tool in args.tools)
    if shard_indices:
        command.append("--task_indices")
        command.extend(str(idx) for idx in shard_indices)
    return command


def _read_rows_by_index(path: Path) -> Dict[int, Dict[str, Any]]:
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return rows_by_index
    # A global abort can terminate a child while it is appending its final row.
    # Keep every complete JSON object and ignore only a truncated tail line.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                idx = int(row.get("task_index"))
            except Exception:
                continue
            if isinstance(row, dict):
                rows_by_index[idx] = row
    return rows_by_index


def _merge_process_shards(args: argparse.Namespace, runner_cls: type, part_paths: List[Path]) -> None:
    output_path = Path(args.outfile)
    rows_by_index = _read_rows_by_index(output_path) if args.resume else {}
    for part_path in part_paths:
        rows_by_index.update(_read_rows_by_index(part_path))

    merged_rows = [
        rows_by_index[idx]
        for idx in sorted(rows_by_index)
        if idx in rows_by_index
    ]
    write_jsonl(output_path, merged_rows, mode="w")
    runner = runner_cls(args.infile, args.outfile, agent=None)  # type: ignore[arg-type]
    runner.write_summary(merged_rows)


def _cleanup_process_shards(part_paths: List[Path]) -> None:
    if _bool_setting(os.getenv("FIRE_AGENT_KEEP_PROCESS_PARTS"), False):
        return
    for part_path in part_paths:
        for path in (part_path, part_path.with_suffix(part_path.suffix + ".summary.json")):
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def _cleanup_stale_process_shards(output_path: Path) -> None:
    prefix = f"{output_path.name}.part"
    for path in output_path.parent.iterdir() if output_path.parent.exists() else []:
        if not path.name.startswith(prefix):
            continue
        if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.summary.json")):
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _stale_process_shards(output_path: Path) -> List[Path]:
    prefix = f"{output_path.name}.part"
    if not output_path.parent.exists():
        return []
    return sorted(
        path
        for path in output_path.parent.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(".jsonl")
    )


def _recover_process_shards_for_resume(
    args: argparse.Namespace, runner_cls: type, output_path: Path
) -> None:
    if not args.resume:
        return
    part_paths = _stale_process_shards(output_path)
    if not part_paths:
        return
    print(
        f"[fire_agent.cli] recovering {len(part_paths)} process shards before resume",
        flush=True,
    )
    # Merge first, then delete. If parsing or writing fails, the original shards
    # remain available for the next recovery attempt.
    _merge_process_shards(args, runner_cls, part_paths)
    _cleanup_process_shards(part_paths)


def _process_chunk_size(args: argparse.Namespace, selected_count: int, worker_count: int) -> int:
    configured = max(0, int(getattr(args, "process_chunk_size", 0) or 0))
    if configured > 0:
        return configured
    if selected_count <= worker_count * 2:
        return 1
    # Auto mode keeps chunks small enough for dynamic load balancing while
    # avoiding one Python cold start per task on larger benchmark runs.
    return max(1, min(4, (selected_count + worker_count * 8 - 1) // (worker_count * 8)))


def _chunk_task_indices(selected: List[int], chunk_size: int) -> List[List[int]]:
    return [selected[offset : offset + chunk_size] for offset in range(0, len(selected), chunk_size)]


def _run_process_backend(args: argparse.Namespace, runner_cls: type) -> None:
    output_path = Path(args.outfile)
    _recover_process_shards_for_resume(args, runner_cls, output_path)
    selected = _selected_task_indices(args, runner_cls)
    if not selected:
        print("[fire_agent.cli] No selected tasks to run.", flush=True)
        return

    worker_count = min(max(1, int(args.max_workers or 1)), len(selected))
    chunk_size = _process_chunk_size(args, len(selected), worker_count)
    chunks = _chunk_task_indices(selected, chunk_size)
    _cleanup_stale_process_shards(output_path)
    part_paths = [output_path.with_name(f"{output_path.name}.part{idx}.jsonl") for idx in range(len(chunks))]

    env = os.environ.copy()
    env["FIRE_AGENT_PARALLEL_CHILD"] = "1"
    pending_chunk_ids = list(range(len(chunks)))
    active: Dict[int, tuple[int, subprocess.Popen[Any]]] = {}
    completed_tasks = 0
    print(
        (
            f"[fire_agent.cli] process backend: {len(selected)} tasks across {worker_count} processes; "
            f"dynamic chunks={len(chunks)} chunk_size={chunk_size}"
        ),
        flush=True,
    )

    def launch_next() -> bool:
        if abort_requested():
            return False
        if not pending_chunk_ids:
            return False
        chunk_id = pending_chunk_ids.pop(0)
        chunk_indices = chunks[chunk_id]
        command = _child_command(args, chunk_indices, part_paths[chunk_id])
        process = subprocess.Popen(command, cwd=str(Path.cwd()), env=env)
        active[chunk_id] = (len(chunk_indices), process)
        return True

    for _ in range(worker_count):
        if not launch_next():
            break

    failures: List[str] = []
    while active:
        if abort_requested():
            details = abort_details()
            reason = str(details.get("reason") or "global API balance circuit breaker triggered")
            print(f"[fire_agent.cli] GLOBAL ABORT: {reason}", flush=True)
            processes = [process for _, process in active.values()]
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            deadline = time.time() + 10.0
            while time.time() < deadline and any(process.poll() is None for process in processes):
                time.sleep(0.2)
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            active.clear()
            _merge_process_shards(args, runner_cls, part_paths)
            _cleanup_process_shards(part_paths)
            raise SystemExit(f"Global experiment abort: {reason}")
        finished: List[tuple[int, int, int]] = []
        for chunk_id, (task_count, process) in list(active.items()):
            return_code = process.poll()
            if return_code is not None:
                finished.append((chunk_id, task_count, return_code))
        if not finished:
            time.sleep(0.5)
            continue
        for chunk_id, task_count, return_code in finished:
            active.pop(chunk_id, None)
            if return_code != 0:
                failures.append(f"chunk={chunk_id} returncode={return_code}")
                continue
            completed_tasks += task_count
            print(
                f"[fire_agent.cli] process chunk {chunk_id + 1}/{len(chunks)} finished "
                f"({completed_tasks}/{len(selected)} tasks)",
                flush=True,
            )
            if not failures:
                launch_next()
    if failures:
        raise SystemExit("Process backend failed: " + "; ".join(failures))

    _merge_process_shards(args, runner_cls, part_paths)
    _cleanup_process_shards(part_paths)
    print(f"[fire_agent.cli] merged process shards into {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FIRE single-agent framework on a benchmark file.")
    parser.add_argument("--bench", choices=sorted(RUNNER_REGISTRY), required=True)
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--task_indices", nargs="*", type=int, help="0-based task indices.")
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_tool_calls_per_round", type=int, default=None)
    parser.add_argument(
        "--context_mode",
        choices=["legacy", "auto", "packet", "hybrid", "full", "full_compatible"],
        default=None,
        help="Prompt context rendering mode. auto keeps short runs full-compatible and uses lean/compact packets for longer runs.",
    )
    parser.add_argument(
        "--task_state_render_mode",
        choices=["full", "view", "auto"],
        default=None,
        help="TaskState prompt rendering. view removes repeated evidence/stale_paths from the per-step prompt.",
    )
    parser.add_argument("--no_paid_tools", action="store_true")
    parser.add_argument(
        "--heuristic_only",
        action="store_true",
        help="Disable FIRE agent role model calls; benchmark evaluators may still call their judge models.",
    )
    parser.add_argument("--tools", nargs="*", default=None, help="Override runner default tools by name.")
    parser.add_argument(
        "--bench_config",
        default=None,
        help="Optional YAML config for this benchmark. Defaults to configs/qa/<bench>.yaml when present.",
    )
    parser.add_argument(
        "--bench_configs_dir",
        default="configs/qa",
        help="Directory containing per-benchmark YAML configs. Use an empty string to disable auto-discovery.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to outfile and skip completed task_index rows.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel sample threads. >1 enables concurrent API calls.")
    parser.add_argument(
        "--parallel_backend",
        choices=["thread", "process"],
        default=os.getenv("FIRE_AGENT_PARALLEL_BACKEND", "thread"),
        help=(
            "Task-level parallel backend. thread preserves legacy behavior; process shards the CLI into child "
            "processes so CPU-heavy context assembly/serialization is not serialized by the GIL."
        ),
    )
    parser.add_argument(
        "--process_chunk_size",
        type=int,
        default=_int_setting(os.getenv("FIRE_AGENT_PROCESS_CHUNK_SIZE"), 0),
        help=(
            "Number of task indices per subprocess chunk when --parallel_backend process is used. "
            "0 enables auto-sized dynamic chunks; 1 gives maximum long-tail load balancing."
        ),
    )
    parser.add_argument(
        "--env_file",
        default=None,
        help="Path to a FIRE Agent .env file. Defaults to fire_agent/.env if it exists.",
    )
    args = parser.parse_args()
    load_fire_agent_env(args.env_file)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.outfile).exists() and not args.resume:
        Path(args.outfile).unlink()

    runner_cls = RUNNER_REGISTRY[args.bench]
    config_path = resolve_benchmark_config_path(args.bench, args.bench_config, args.bench_configs_dir)
    bench_config = load_benchmark_config(config_path)
    bench_config = expand_benchmark_templates(bench_config, benchmark_template_variables(bench_config))
    validate_benchmark_config(bench_config, args.bench, config_path)
    configured_tools = tools_from_config(bench_config, config_path)
    configured_tool_settings = tool_settings_from_config(bench_config, config_path)
    agent_settings = agent_settings_from_config(bench_config, config_path)
    prompt_settings = prompt_settings_from_config(bench_config, config_path)
    model_reasoning_settings = model_reasoning_settings_from_config(bench_config, config_path)
    enabled_tools = (
        args.tools
        if args.tools is not None
        else configured_tools
        if configured_tools is not None
        else runner_cls.default_tools
    )
    max_steps = args.max_steps if args.max_steps is not None else int(agent_settings.get("max_steps", AgentConfig.max_steps))
    max_tool_calls_per_round = (
        args.max_tool_calls_per_round
        if args.max_tool_calls_per_round is not None
        else int(agent_settings.get("max_tool_calls_per_round", AgentConfig.max_tool_calls_per_round))
    )
    tool_timeout_seconds = int(agent_settings.get("tool_timeout_seconds", AgentConfig.tool_timeout_seconds))
    context_mode = args.context_mode or str(agent_settings.get("context_mode", AgentConfig.context_mode))
    task_state_render_mode = args.task_state_render_mode or str(
        agent_settings.get("task_state_render_mode", AgentConfig.task_state_render_mode)
    )
    allow_paid_tools = False if args.no_paid_tools else bool(
        agent_settings.get("allow_paid_tools", AgentConfig.allow_paid_tools)
    )
    use_model_for_roles = False if args.heuristic_only else bool(
        agent_settings.get("use_model_for_roles", AgentConfig.use_model_for_roles)
    )
    config = AgentConfig(
        max_steps=max_steps,
        max_tool_calls_per_round=max_tool_calls_per_round,
        tool_timeout_seconds=tool_timeout_seconds,
        allow_paid_tools=allow_paid_tools,
        prefer_free_official_sources=bool(
            agent_settings.get("prefer_free_official_sources", AgentConfig.prefer_free_official_sources)
        ),
        use_model_for_roles=use_model_for_roles,
        final_answer_only_round=bool(agent_settings.get("final_answer_only_round", AgentConfig.final_answer_only_round)),
        model_name=args.model or agent_settings.get("model_name"),
        vision_model_name=agent_settings.get("vision_model_name"),
        benchmark_system_prompt=str(prompt_settings.get("system_constraints") or "").strip(),
        benchmark_planning_prompt=str(prompt_settings.get("planning_constraints") or "").strip(),
        tool_prompt_constraints={
            name: ([constraints] if isinstance(constraints, str) else list(constraints))
            for name, constraints in (prompt_settings.get("tool_constraints") or {}).items()
        },
        context_mode=context_mode,
        task_state_render_mode=task_state_render_mode,
        context_full_compatible_char_limit=int(
            agent_settings.get(
                "context_full_compatible_char_limit",
                AgentConfig.context_full_compatible_char_limit,
            )
        ),
        context_hybrid_char_threshold=int(
            agent_settings.get("context_hybrid_char_threshold", AgentConfig.context_hybrid_char_threshold)
        ),
        context_artifact_char_threshold=int(
            agent_settings.get("context_artifact_char_threshold", AgentConfig.context_artifact_char_threshold)
        ),
        context_artifact_preview_chars=int(
            agent_settings.get("context_artifact_preview_chars", AgentConfig.context_artifact_preview_chars)
        ),
        context_recent_steps=int(agent_settings.get("context_recent_steps", AgentConfig.context_recent_steps)),
        context_recent_observation_chars=int(
            agent_settings.get("context_recent_observation_chars", AgentConfig.context_recent_observation_chars)
        ),
        context_recent_observation_step_chars=int(
            agent_settings.get(
                "context_recent_observation_step_chars",
                AgentConfig.context_recent_observation_step_chars,
            )
        ),
        context_recent_tool_call_excerpt_chars=int(
            agent_settings.get(
                "context_recent_tool_call_excerpt_chars",
                AgentConfig.context_recent_tool_call_excerpt_chars,
            )
        ),
        context_action_ledger_limit=int(
            agent_settings.get("context_action_ledger_limit", AgentConfig.context_action_ledger_limit)
        ),
        context_duplicate_warning_limit=int(
            agent_settings.get("context_duplicate_warning_limit", AgentConfig.context_duplicate_warning_limit)
        ),
        context_pinned_evidence_limit=int(
            agent_settings.get("context_pinned_evidence_limit", AgentConfig.context_pinned_evidence_limit)
        ),
        context_pinned_evidence_chars=int(
            agent_settings.get("context_pinned_evidence_chars", AgentConfig.context_pinned_evidence_chars)
        ),
        context_answer_pin_limit=int(
            agent_settings.get("context_answer_pin_limit", AgentConfig.context_answer_pin_limit)
        ),
        context_answer_pin_chars=int(
            agent_settings.get("context_answer_pin_chars", AgentConfig.context_answer_pin_chars)
        ),
        context_pinned_raw_slice_limit=int(
            agent_settings.get("context_pinned_raw_slice_limit", AgentConfig.context_pinned_raw_slice_limit)
        ),
        context_pinned_raw_slice_chars=int(
            agent_settings.get("context_pinned_raw_slice_chars", AgentConfig.context_pinned_raw_slice_chars)
        ),
        context_pinned_raw_slice_step_chars=int(
            agent_settings.get(
                "context_pinned_raw_slice_step_chars",
                AgentConfig.context_pinned_raw_slice_step_chars,
            )
        ),
        context_artifact_slice_limit=int(
            agent_settings.get("context_artifact_slice_limit", AgentConfig.context_artifact_slice_limit)
        ),
        context_artifact_slice_chars=int(
            agent_settings.get("context_artifact_slice_chars", AgentConfig.context_artifact_slice_chars)
        ),
        context_artifact_slice_step_chars=int(
            agent_settings.get("context_artifact_slice_step_chars", AgentConfig.context_artifact_slice_step_chars)
        ),
        context_calc_ledger_limit=int(
            agent_settings.get("context_calc_ledger_limit", AgentConfig.context_calc_ledger_limit)
        ),
        context_artifact_ref_limit=int(
            agent_settings.get("context_artifact_ref_limit", AgentConfig.context_artifact_ref_limit)
        ),
        context_request_reserve_chars=int(
            agent_settings.get("context_request_reserve_chars", AgentConfig.context_request_reserve_chars)
        ),
        context_compression_mode=str(
            agent_settings.get(
                "context_compression_mode",
                os.getenv("FIRE_AGENT_CONTEXT_COMPRESSION_MODE", AgentConfig.context_compression_mode),
            )
        ),
        context_compression_retries=int(
            agent_settings.get(
                "context_compression_retries",
                os.getenv("FIRE_AGENT_CONTEXT_COMPRESSION_RETRIES", AgentConfig.context_compression_retries),
            )
        ),
        context_compression_max_source_chars=int(
            agent_settings.get(
                "context_compression_max_source_chars",
                os.getenv("FIRE_AGENT_CONTEXT_COMPRESSION_MAX_SOURCE_CHARS", AgentConfig.context_compression_max_source_chars),
            )
        ),
        context_compression_max_packet_chars=int(
            agent_settings.get(
                "context_compression_max_packet_chars",
                os.getenv("FIRE_AGENT_CONTEXT_COMPRESSION_MAX_PACKET_CHARS", AgentConfig.context_compression_max_packet_chars),
            )
        ),
        context_compression_old_observation_step_chars=int(
            agent_settings.get(
                "context_compression_old_observation_step_chars",
                os.getenv(
                    "FIRE_AGENT_CONTEXT_COMPRESSION_OLD_OBSERVATION_STEP_CHARS",
                    AgentConfig.context_compression_old_observation_step_chars,
                ),
            )
        ),
        context_compression_recent_observation_step_chars=int(
            agent_settings.get(
                "context_compression_recent_observation_step_chars",
                os.getenv(
                    "FIRE_AGENT_CONTEXT_COMPRESSION_RECENT_OBSERVATION_STEP_CHARS",
                    AgentConfig.context_compression_recent_observation_step_chars,
                ),
            )
        ),
        context_compression_include_draft_packet=_bool_setting(
            agent_settings.get(
                "context_compression_include_draft_packet",
                os.getenv(
                    "FIRE_AGENT_CONTEXT_COMPRESSION_INCLUDE_DRAFT_PACKET",
                    str(AgentConfig.context_compression_include_draft_packet),
                ),
            ),
            AgentConfig.context_compression_include_draft_packet,
        ),
    )

    if (
        args.parallel_backend == "process"
        and int(args.max_workers or 1) > 1
        and os.getenv("FIRE_AGENT_PARALLEL_CHILD") != "1"
    ):
        _run_process_backend(args, runner_cls)
        return

    def build_agent() -> FIREAgent:
        model = None if args.heuristic_only else build_model(
            args.model,
            reasoning_overrides=model_reasoning_settings,
        )
        auxiliary_model = None if args.heuristic_only else build_auxiliary_model()
        tools = build_default_tool_registry(enabled_tools, tool_settings=configured_tool_settings)
        return FIREAgent(model=model, auxiliary_model=auxiliary_model, config=config, tools=tools)

    agent = build_agent()
    runner = runner_cls(args.infile, args.outfile, agent, agent_factory=build_agent)
    runner.run(task_indices=args.task_indices, sample_num=args.sample_num, resume=args.resume, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
