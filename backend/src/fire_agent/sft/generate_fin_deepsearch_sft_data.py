from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .export_events import export_events


DEFAULT_INPUT = Path("sft/fin_deepsearch_sft.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/fin_deepsearch_sft_sft_generation")
DEFAULT_SAMPLE_TYPES = "agent_policy,context_compression,content_reader"


JsonDict = Dict[str, Any]


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for base in current.parents:
        if (base / "pyproject.toml").exists() and (base / "src" / "fire_agent").is_dir():
            return base
    return current.parents[3]


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root / expanded).resolve()


def _read_jsonl_head(path: Path, limit: int = 5) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if len(rows) >= limit:
                break
    return rows


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _validate_seed_file(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Seed QA JSONL not found: {path}")
    total = _count_jsonl(path)
    if total <= 0:
        raise ValueError(f"Seed QA JSONL is empty: {path}")
    bad: List[str] = []
    sample_ids: set[str] = set()
    duplicate_sample_ids = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                bad.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                bad.append(f"line {line_number}: row is not an object")
                continue
            for key in ("question", "answer", "sample_id"):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad.append(f"line {line_number}: missing non-empty {key}")
                    break
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                if sample_id in sample_ids:
                    duplicate_sample_ids += 1
                sample_ids.add(sample_id)
            if len(bad) >= 10:
                break
    if bad:
        raise ValueError("Seed QA validation failed:\n" + "\n".join(bad))
    preview = _read_jsonl_head(path, limit=3)
    return {
        "path": str(path),
        "rows": total,
        "unique_sample_ids": len(sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "preview": [
            {
                "sample_id": row.get("sample_id"),
                "question_chars": len(str(row.get("question") or "")),
                "answer": str(row.get("answer") or "")[:200],
            }
            for row in preview
        ],
    }


def _truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _build_env(args: argparse.Namespace, project_root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["FIRE_AGENT_KEEP_REASONING_CONTENT"] = "1"
    env["FIRE_AGENT_THINKING_ENABLED"] = "1" if args.thinking else "0"
    if args.respect_env:
        env.setdefault("FIRE_AGENT_REASONING_EFFORT", args.reasoning_effort)
        env.setdefault("FIRE_AGENT_CONTEXT_COMPRESSION_MODE", args.context_compression_mode)
    else:
        env["FIRE_AGENT_REASONING_EFFORT"] = args.reasoning_effort
        env["FIRE_AGENT_CONTEXT_COMPRESSION_MODE"] = args.context_compression_mode
    return env


def _run_command(command: Sequence[str], *, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[sft-generate] command: {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd), env=env, check=True)


def _agent_command(args: argparse.Namespace, traces_path: Path) -> List[str]:
    command = [
        args.python_executable,
        "-m",
        "fire_agent.cli",
        "--bench",
        "fin_deepsearch_sft",
        "--infile",
        str(args.input),
        "--outfile",
        str(traces_path),
        "--model",
        args.model,
        "--context_mode",
        args.context_mode,
        "--task_state_render_mode",
        args.task_state_render_mode,
        "--max_workers",
        str(args.max_workers),
        "--parallel_backend",
        args.parallel_backend,
    ]
    optional_pairs = [
        ("--bench_configs_dir", args.bench_configs_dir),
        ("--env_file", args.env_file),
        ("--sample_num", args.sample_num),
        ("--max_steps", args.max_steps),
        ("--max_tool_calls_per_round", args.max_tool_calls_per_round),
        ("--process_chunk_size", args.process_chunk_size),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    if args.task_indices:
        command.append("--task_indices")
        command.extend(str(index) for index in args.task_indices)
    if args.resume:
        command.append("--resume")
    if args.no_paid_tools:
        command.append("--no_paid_tools")
    return command


def _rescore_command(args: argparse.Namespace, traces_path: Path, judged_path: Path) -> List[str]:
    command = [
        args.python_executable,
        "-m",
        "fire_agent.rescore_fin_deepsearch_sft",
        "--infile",
        str(traces_path),
        "--outfile",
        str(judged_path),
        "--max_workers",
        str(args.judge_workers),
    ]
    if args.env_file is not None:
        command.extend(["--env_file", str(args.env_file)])
    return command


def _write_generation_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    seed_summary: JsonDict,
    traces_path: Path,
    judged_path: Optional[Path],
    sft_output_dir: Path,
    export_manifest: Optional[JsonDict],
    env: Dict[str, str],
) -> None:
    manifest = {
        "bench": "fin_deepsearch_sft",
        "seed_summary": seed_summary,
        "traces_path": str(traces_path),
        "judged_path": str(judged_path) if judged_path is not None else None,
        "sft_output_dir": str(sft_output_dir),
        "export_manifest": export_manifest,
        "run_options": {
            "model": args.model,
            "thinking": bool(args.thinking),
            "reasoning_effort": args.reasoning_effort,
            "context_mode": args.context_mode,
            "task_state_render_mode": args.task_state_render_mode,
            "context_compression_mode": args.context_compression_mode,
            "sample_num": args.sample_num,
            "task_indices": args.task_indices,
            "max_steps": args.max_steps,
            "max_workers": args.max_workers,
            "parallel_backend": args.parallel_backend,
            "process_chunk_size": args.process_chunk_size,
            "resume": args.resume,
            "rescore": args.rescore,
            "respect_env": args.respect_env,
            "require_correct": not args.no_require_correct,
            "sample_types": args.sample_types,
            "strip_reasoning_content": False,
        },
        "env_flags": {
            "FIRE_AGENT_KEEP_REASONING_CONTENT": env.get("FIRE_AGENT_KEEP_REASONING_CONTENT"),
            "FIRE_AGENT_THINKING_ENABLED": env.get("FIRE_AGENT_THINKING_ENABLED"),
            "FIRE_AGENT_REASONING_EFFORT": env.get("FIRE_AGENT_REASONING_EFFORT"),
            "FIRE_AGENT_CONTEXT_COMPRESSION_MODE": env.get("FIRE_AGENT_CONTEXT_COMPRESSION_MODE"),
            "FIRE_AGENT_GLOBAL_BALANCE_ERROR_THRESHOLD": env.get(
                "FIRE_AGENT_GLOBAL_BALANCE_ERROR_THRESHOLD"
            ),
            "FIRE_AGENT_GLOBAL_API_BREAKER_STATE_FILE": env.get(
                "FIRE_AGENT_GLOBAL_API_BREAKER_STATE_FILE"
            ),
            "FIRE_AGENT_GLOBAL_ABORT_FILE": env.get("FIRE_AGENT_GLOBAL_ABORT_FILE"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[sft-generate] manifest: {path}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FinDeepSearchSFT teacher transition traces with DeepSeek, "
            "filter evaluator-correct episodes, and export turn-level SFT rows."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="QA seed JSONL with question/answer/sample_id.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Generation workspace.")
    parser.add_argument("--traces_file", type=Path, default=None, help="Raw benchmark trace JSONL. Defaults under output_dir.")
    parser.add_argument("--sft_output_dir", type=Path, default=None, help="Directory for three SFT turn JSONL files.")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Teacher model used by the agent.")
    parser.add_argument("--thinking", action="store_true", help="Enable DeepSeek thinking mode while generating traces.")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--env_file", default=None, help="FIRE Agent env file for model/tool credentials.")
    parser.add_argument("--bench_configs_dir", default="configs/qa")
    parser.add_argument("--context_mode", default="auto", choices=["legacy", "auto", "packet", "hybrid", "full", "full_compatible"])
    parser.add_argument("--task_state_render_mode", default="auto", choices=["full", "view", "auto"])
    parser.add_argument("--context_compression_mode", default="llm", choices=["off", "heuristic", "llm"])
    parser.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high", "max"])
    parser.add_argument(
        "--respect_env",
        action="store_true",
        help="Let existing FIRE_AGENT_REASONING_EFFORT and FIRE_AGENT_CONTEXT_COMPRESSION_MODE override CLI defaults.",
    )
    parser.add_argument("--sample_num", type=int, default=None, help="Smoke-test row cap.")
    parser.add_argument("--task_indices", nargs="*", type=int, default=None, help="0-based task indices to run.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_tool_calls_per_round", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--parallel_backend", choices=["thread", "process"], default="thread")
    parser.add_argument("--process_chunk_size", type=int, default=None)
    parser.add_argument("--judge_workers", type=int, default=4, help="Workers for optional offline rescoring.")
    parser.add_argument("--resume", action="store_true", help="Resume existing trace file.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing trace file.")
    parser.add_argument("--skip_agent_run", action="store_true", help="Do not run the agent; export from an existing traces_file.")
    parser.add_argument("--skip_export", action="store_true", help="Only run/rescore traces; do not export SFT rows.")
    parser.add_argument("--rescore", action="store_true", help="Run offline FinDeepSearchSFT judge before exporting.")
    parser.add_argument("--no_paid_tools", action="store_true")
    parser.add_argument("--sample_types", default=DEFAULT_SAMPLE_TYPES)
    parser.add_argument("--allowed_status", default="success")
    parser.add_argument("--no_require_correct", action="store_true", help="Export all allowed-status episodes, not only score.is_correct=true.")
    parser.add_argument("--correct_key", default="is_correct")
    parser.add_argument("--score_key", default=None)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--include_rejected_compression", action="store_true")
    parser.add_argument("--allow_missing_turns", action="store_true", help="Allow accepted trace rows that produce no exportable turns.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print commands without running.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    project_root = _project_root()
    args.input = _resolve_project_path(args.input, project_root)
    output_dir = _resolve_project_path(args.output_dir, project_root)
    traces_path = (
        _resolve_project_path(args.traces_file, project_root)
        if args.traces_file is not None
        else output_dir / "fin_deepsearch_sft_deepseek_traces.jsonl"
    )
    judged_path = traces_path.with_suffix(".judged.jsonl") if args.rescore else None
    sft_output_dir = (
        _resolve_project_path(args.sft_output_dir, project_root)
        if args.sft_output_dir is not None
        else output_dir / "sft_events"
    )
    generation_manifest_path = output_dir / "sft_generation_manifest.json"

    seed_summary = _validate_seed_file(args.input)
    print(
        "[sft-generate] seed rows={rows} unique_sample_ids={unique}".format(
            rows=seed_summary["rows"],
            unique=seed_summary["unique_sample_ids"],
        ),
        flush=True,
    )

    if traces_path.exists() and not (args.resume or args.overwrite or args.skip_agent_run):
        raise SystemExit(
            f"Trace file already exists: {traces_path}. Use --resume, --overwrite, or --skip_agent_run."
        )
    if args.skip_agent_run and not traces_path.exists():
        raise SystemExit(f"--skip_agent_run requires an existing traces_file: {traces_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env(args, project_root)

    if not args.skip_agent_run:
        _run_command(_agent_command(args, traces_path), cwd=project_root, env=env, dry_run=args.dry_run)

    export_input = traces_path
    if args.rescore:
        assert judged_path is not None
        _run_command(_rescore_command(args, traces_path, judged_path), cwd=project_root, env=env, dry_run=args.dry_run)
        export_input = judged_path

    export_manifest: Optional[JsonDict] = None
    if not args.skip_export:
        if args.dry_run:
            print(
                "[sft-generate] export: input={input} output_dir={output_dir} require_correct={require_correct}".format(
                    input=export_input,
                    output_dir=sft_output_dir,
                    require_correct=not args.no_require_correct,
                ),
                flush=True,
            )
        else:
            export_manifest = export_events(
                [export_input],
                output_dir=sft_output_dir,
                sample_types=args.sample_types,
                allowed_status=args.allowed_status,
                score_key=args.score_key,
                min_score=args.min_score,
                require_correct=not args.no_require_correct,
                correct_key=args.correct_key,
                include_rejected_compression=bool(args.include_rejected_compression),
                fail_on_missing_turns=not args.allow_missing_turns,
                strip_reasoning_content=False,
            )
            print(
                "[sft-generate] exported rows={rows} module_counts={counts}".format(
                    rows=(export_manifest.get("counters") or {}).get("rows_written", 0),
                    counts=json.dumps(export_manifest.get("module_counts") or {}, ensure_ascii=False),
                ),
                flush=True,
            )

    if not args.dry_run:
        _write_generation_manifest(
            generation_manifest_path,
            args=args,
            seed_summary=seed_summary,
            traces_path=traces_path,
            judged_path=judged_path,
            sft_output_dir=sft_output_dir,
            export_manifest=export_manifest,
            env=env,
        )


if __name__ == "__main__":
    main()
