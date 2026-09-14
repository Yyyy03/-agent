from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregation import (
    aggregate_bizfinbench_v2_logs,
    aggregate_fin_deepsearch_sft_logs,
    aggregate_fingaia_logs,
    aggregate_rfc_task2_logs,
    aggregate_task_logs,
)
from .benchmarks.common import normalize_legacy_rows, read_jsonl
from .benchmarks.registry import canonical_bench_name


def project_root() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd(), *file_path.parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "official_benchmarks").exists():
            return base
        nested = base / "fire_agent"
        if (nested / "pyproject.toml").exists() and (nested / "official_benchmarks").exists():
            return nested
    return file_path.parents[2]


FIRE_AGENT_PROJECT_ROOT = project_root()


def default_python_executable() -> str:
    explicit = os.getenv("FIRE_AGENT_PYTHON_EXECUTABLE") or os.getenv("FIRE_AGENT_PYTHON")
    if explicit:
        return explicit
    if "/envs/gyz/" in sys.executable:
        return sys.executable
    for base in FIRE_AGENT_PROJECT_ROOT.parents:
        candidate = base / "conda" / "envs" / "gyz" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable

@dataclass(frozen=True)
class BenchRun:
    bench: str
    infile_name: str
    outfile_name: str
    required: bool = False
    bundled_infile: Optional[str] = None
    alternate_infile_names: tuple[str, ...] = ()


BENCHMARK_RUNS: List[BenchRun] = [
    BenchRun("fingaia", "fingaia.jsonl", "fire_fingaia_results.jsonl", required=True),
    BenchRun(
        "fin_deepsearch_sft",
        "fin_deepsearch_sft.jsonl",
        "fire_fin_deepsearch_sft_results.jsonl",
        required=False,
    ),
    BenchRun(
        "finsearchcomp",
        "finsearchcomp.jsonl",
        "fire_finsearchcomp_results.jsonl",
        required=True,
        bundled_infile="official_benchmarks/finsearchcomp/data/finsearchcomp.jsonl",
    ),
    BenchRun(
        "financeagentbench",
        "financeagentbench.jsonl",
        "fire_financeagentbench_results.jsonl",
        required=True,
        bundled_infile="official_benchmarks/financeagentbench/data/financeagent_public_full.jsonl",
    ),
    BenchRun(
        "financeagent_v2",
        "financeagent_v2_public_full.jsonl",
        "fire_financeagent_v2_results.jsonl",
        required=False,
        bundled_infile="official_benchmarks/financeagentbench/data/financeagent_v2_public_full.jsonl",
    ),
    BenchRun("finrpt", "finrpt.jsonl", "fire_finrpt_results.jsonl", required=True),
    BenchRun("findocresearch", "findocresearch.jsonl", "fire_findocresearch_results.jsonl", required=True),
    BenchRun(
        "rfc_bench",
        "rfc_bench.jsonl",
        "fire_rfc_bench_results.jsonl",
        required=False,
        bundled_infile="official_benchmarks/rfc_bench/data/rfc_bench.jsonl",
    ),
    BenchRun(
        "rfc_task2",
        "rfc_task2_final.jsonl",
        "fire_rfc_task2_results.jsonl",
        required=False,
    ),
    BenchRun(
        "bizfinbench_v2",
        "bizfinbench_eval700_no_ci_fra.jsonl",
        "fire_bizfinbench_v2_results.jsonl",
        required=False,
        bundled_infile="official_benchmarks/bizfinbench_v2/data/bizfinbench_v2_sample_500.jsonl",
        alternate_infile_names=("bizfinbench_v2_sample_500.jsonl",),
    ),
    BenchRun(
        "financebench_closed_book",
        "financebench.jsonl",
        "fire_financebench_closed_book_results.jsonl",
        required=False,
        bundled_infile="official_benchmarks/financebench/data/financebench.jsonl",
    ),
    BenchRun(
        "financebench",
        "financebench.jsonl",
        "fire_financebench_evidence_results.jsonl",
        required=False,
        bundled_infile="official_benchmarks/financebench/data/financebench.jsonl",
    ),
]


def run_command(command: List[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def maybe_run_official_report_eval(args: argparse.Namespace, bench_run: BenchRun, output_file: Path, cwd: Path) -> None:
    # FinRpt official ROUGE-L / BERTScore / number / trend / success scoring
    # is now triggered by FinRptRunner.finalize_after_run() automatically at the
    # end of every cli.py run, so the batch script no longer needs to invoke
    # exec_eval.py again. The `--run_official_report_eval` flag remains as an
    # explicit opt-in for the FinDocResearch OpenFinArena bridge, which still
    # requires the external competitions/ directory.
    if bench_run.bench == "finrpt":
        return

    if not args.run_official_report_eval:
        return

    if bench_run.bench == "findocresearch":
        if not args.openfinarena_competitions_dir:
            print("Skipping FinDocResearch official eval: --openfinarena_competitions_dir was not provided.", flush=True)
            return
        competitions_dir = Path(args.openfinarena_competitions_dir)
        prediction_folder = output_file.parent / output_file.stem / "markdown"
        if not (competitions_dir / "evaluation" / "run.py").exists():
            raise FileNotFoundError(f"OpenFinArena evaluation/run.py not found under: {competitions_dir}")
        if not any(prediction_folder.glob("*.md")):
            raise FileNotFoundError(f"FinDocResearch prediction folder has no markdown files: {prediction_folder}")
        run_command(
            [
                args.python_executable,
                "evaluation/run.py",
                "--track",
                "findocresearch",
                "--prediction_folder",
                str(prediction_folder),
                "--model",
                args.openfinarena_eval_model,
                "--overwrite",
            ],
            competitions_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FIRE Agent on supported projection-scored benchmarks.")
    parser.add_argument(
        "--python_executable",
        type=str,
        default=default_python_executable(),
        help="Python executable to use. Defaults to FIRE_AGENT_PYTHON_EXECUTABLE/FIRE_AGENT_PYTHON, then a local gyz conda env when present.",
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        default="benchmarks",
        help="Directory containing benchmark jsonl files; bundled full benchmark files are used as fallbacks when configured.",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for FIRE benchmark outputs.")
    parser.add_argument("--sample_num", type=int, default=None, help="Optional sample cap per benchmark.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum FIRE role rounds per sample. When omitted, use the benchmark YAML value.",
    )
    parser.add_argument(
        "--max_tool_calls_per_round",
        type=int,
        default=None,
        help="Maximum tool calls per FIRE round. When omitted, use the benchmark YAML value.",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional model name override.")
    parser.add_argument(
        "--context_mode",
        choices=["legacy", "auto", "packet", "hybrid", "full", "full_compatible"],
        default=None,
        help="Prompt context rendering mode passed through to fire_agent.cli.",
    )
    parser.add_argument(
        "--task_state_render_mode",
        choices=["full", "view", "auto"],
        default=None,
        help="TaskState rendering mode passed through to fire_agent.cli.",
    )
    parser.add_argument(
        "--heuristic_only",
        action="store_true",
        help="Disable FIRE agent role model calls; benchmark evaluators may still call their judge models.",
    )
    parser.add_argument("--no_paid_tools", action="store_true", help="Disable paid tools.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel sample threads per benchmark.")
    parser.add_argument(
        "--parallel_backend",
        choices=["thread", "process"],
        default=None,
        help="Task-level backend passed through to fire_agent.cli. Use process to bypass GIL-heavy local context work.",
    )
    parser.add_argument(
        "--process_chunk_size",
        type=int,
        default=None,
        help=(
            "Task indices per subprocess chunk for --parallel_backend process. "
            "Pass 1 for maximum long-tail balancing; omit for cli auto sizing."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed task_index rows in existing output files.")
    parser.add_argument(
        "--bench_configs_dir",
        default="configs/qa",
        help="Directory containing per-benchmark YAML configs passed through to fire_agent.cli.",
    )
    parser.add_argument(
        "--env_file",
        default=None,
        help="Path to a FIRE Agent .env file. Defaults to fire_agent/.env if it exists.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these bench names, e.g. finsearchcomp findocresearch.",
    )
    parser.add_argument(
        "--run_official_report_eval",
        action="store_true",
        help="After generation, run the external OpenFinArena evaluator for FinDocResearch when paths are provided.",
    )
    parser.add_argument(
        "--openfinarena_competitions_dir",
        default=None,
        help="Path to OpenFinArena competitions/ directory for FinDocResearch official evaluation.",
    )
    parser.add_argument(
        "--openfinarena_eval_model",
        default="gpt-4.1",
        help="LLM used by OpenFinArena's extractor/evaluator for report scoring.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    benchmarks_dir = Path(args.benchmarks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    known_benches = {bench_run.bench for bench_run in BENCHMARK_RUNS}
    selected = {canonical_bench_name(bench.lower()) for bench in args.only} if args.only else None
    if selected:
        unknown = selected - known_benches
        if unknown:
            raise ValueError(f"Unknown benchmark(s) for this benchmark run: {sorted(unknown)}")
    missing_required: List[str] = []

    for bench_run in BENCHMARK_RUNS:
        if selected and bench_run.bench.lower() not in selected:
            continue
        infile = benchmarks_dir / bench_run.infile_name
        if not infile.exists():
            for alternate_name in bench_run.alternate_infile_names:
                alternate = benchmarks_dir / alternate_name
                if alternate.exists():
                    infile = alternate
                    break
        if not infile.exists() and bench_run.bundled_infile:
            bundled_infile = FIRE_AGENT_PROJECT_ROOT / bench_run.bundled_infile
            if bundled_infile.exists():
                infile = bundled_infile
        if not infile.exists():
            message = f"Missing benchmark file for {bench_run.bench}: {infile}"
            if bench_run.required:
                missing_required.append(message)
            else:
                print(f"Skipping optional benchmark. {message}", flush=True)
            continue

        output_file = output_dir / bench_run.outfile_name
        command = [
            args.python_executable,
            "-m",
            "fire_agent.cli",
            "--bench",
            bench_run.bench,
            "--infile",
            str(infile),
            "--outfile",
            str(output_file),
        ]
        if args.max_steps is not None:
            command.extend(["--max_steps", str(args.max_steps)])
        if args.max_tool_calls_per_round is not None:
            command.extend(["--max_tool_calls_per_round", str(args.max_tool_calls_per_round)])
        if args.sample_num is not None:
            command.extend(["--sample_num", str(args.sample_num)])
        if args.model:
            command.extend(["--model", args.model])
        if args.context_mode:
            command.extend(["--context_mode", args.context_mode])
        if args.task_state_render_mode:
            command.extend(["--task_state_render_mode", args.task_state_render_mode])
        if args.heuristic_only:
            command.append("--heuristic_only")
        if args.no_paid_tools:
            command.append("--no_paid_tools")
        if args.resume:
            command.append("--resume")
        if args.max_workers > 1:
            command.extend(["--max_workers", str(args.max_workers)])
        if args.parallel_backend:
            command.extend(["--parallel_backend", args.parallel_backend])
        if args.process_chunk_size is not None:
            command.extend(["--process_chunk_size", str(args.process_chunk_size)])
        if args.bench_configs_dir:
            command.extend(["--bench_configs_dir", args.bench_configs_dir])
        if args.env_file:
            command.extend(["--env_file", args.env_file])
        run_command(command, cwd)
        maybe_run_official_report_eval(args, bench_run, output_file, cwd)

    if missing_required:
        raise FileNotFoundError("\n".join(missing_required))

    summarize_outputs(output_dir, selected)


def summarize_outputs(output_dir: Path, selected: Optional[set]) -> None:
    """After all benchmark runs finish, aggregate every bench's JSONL output
    into a single ``benchmarks_summary.json`` next to the per-bench files."""
    overall: Dict[str, Any] = {}
    print("\n========== Dataset-level summary ==========", flush=True)
    for bench_run in BENCHMARK_RUNS:
        if selected and bench_run.bench.lower() not in selected:
            continue
        output_file = output_dir / bench_run.outfile_name
        if not output_file.exists():
            continue
        try:
            rows = read_jsonl(output_file)
        except Exception as exc:
            print(f"[summary] Could not read {output_file}: {exc}", flush=True)
            continue
        rows = normalize_legacy_rows(rows)
        if bench_run.bench == "fingaia":
            summary = aggregate_fingaia_logs(rows)
        elif bench_run.bench == "fin_deepsearch_sft":
            summary = aggregate_fin_deepsearch_sft_logs(rows)
        elif bench_run.bench == "rfc_task2":
            summary = aggregate_rfc_task2_logs(rows)
        elif bench_run.bench == "bizfinbench_v2":
            summary = aggregate_bizfinbench_v2_logs(rows)
            summary["benchmark"] = bench_run.bench
        else:
            summary = aggregate_task_logs(rows)
            summary["benchmark"] = bench_run.bench
        summary["output_file"] = str(output_file)
        summary["rows_in_output"] = len(rows)
        overall[bench_run.bench] = summary

        line = f"[{bench_run.bench}] total={summary.get('total', len(rows))}"
        acc = summary.get("accuracy")
        if acc is not None:
            line += f" accuracy={acc:.4f}"
            if "accuracy_pct" in summary and summary["accuracy_pct"] is not None:
                line += f" ({summary['accuracy_pct']:.2f}%)"
        if summary.get("avg_judge_score") is not None:
            line += f" avg_judge_score={summary['avg_judge_score']:.4f}"
        token_stats = summary.get("token_stats") or {}
        if token_stats:
            line += (
                f" total_tokens={token_stats.get('total_tokens', 0)}"
                f" llm_calls={token_stats.get('total_llm_calls', 0)}"
            )
        print(line, flush=True)

    summary_path = output_dir / "benchmarks_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[summary] Wrote consolidated summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
