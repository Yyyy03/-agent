"""Offline LLM-as-judge rescoring for FinDeepSearchSFT result files.

Usage:

    PYTHONPATH=src python -m fire_agent.rescore_fin_deepsearch_sft \
        --infile output/fire_fin_deepsearch_sft_results.jsonl \
        --outfile output/fire_fin_deepsearch_sft_results.judged.jsonl

The script does not run the agent again. It reads existing result rows that
already contain ``agent_result`` and ``agent_trajectory``, runs the
FinDeepSearchSFT LLM judge, and writes a new JSONL plus summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .aggregation import aggregate_fin_deepsearch_sft_logs
    from .benchmarks.common import normalize_legacy_row, read_jsonl, write_jsonl
    from .config import load_fire_agent_env
    from .evaluators.deepsearch_sft import FinDeepSearchSFTEvaluator
except ImportError:  # pragma: no cover - allow direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fire_agent.aggregation import aggregate_fin_deepsearch_sft_logs  # type: ignore
    from fire_agent.benchmarks.common import normalize_legacy_row, read_jsonl, write_jsonl  # type: ignore
    from fire_agent.config import load_fire_agent_env  # type: ignore
    from fire_agent.evaluators.deepsearch_sft import FinDeepSearchSFTEvaluator  # type: ignore


PROGRESS_LOCK = threading.Lock()


def _rescore_row(evaluator: FinDeepSearchSFTEvaluator, row: Dict[str, Any]) -> Dict[str, Any]:
    new_row = normalize_legacy_row(row)
    judge_output = evaluator.evaluate(new_row)
    previous = new_row.get("score") if isinstance(new_row.get("score"), dict) else {}
    if previous:
        judge_output = {"previous_score": previous, **judge_output}
    new_row["score"] = judge_output
    return new_row


def _iter_rows_with_index(rows: List[Dict[str, Any]]) -> Iterable[tuple[int, Dict[str, Any]]]:
    for idx, row in enumerate(rows):
        yield idx, row


def rescore(
    infile: Path,
    outfile: Path,
    summary_path: Optional[Path],
    max_workers: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = read_jsonl(infile)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows to rescore in {infile}")

    evaluator = FinDeepSearchSFTEvaluator()
    rescored: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    total = len(rows)
    done = 0

    def _on_done(idx: int, result: Dict[str, Any]) -> None:
        nonlocal done
        rescored[idx] = result
        score = result.get("score") if isinstance(result.get("score"), dict) else {}
        with PROGRESS_LOCK:
            done += 1
            print(
                f"[rescore_fin_deepsearch_sft] {done}/{total} "
                f"task_index={result.get('task_index')} "
                f"is_correct={score.get('is_correct')} "
                f"answer_score={score.get('answer_score')} "
                f"trajectory_score={score.get('trajectory_score')} "
                f"mode={score.get('mode')}",
                flush=True,
            )

    if max_workers <= 1:
        for idx, row in _iter_rows_with_index(rows):
            _on_done(idx, _rescore_row(evaluator, row))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_rescore_row, evaluator, row): idx
                for idx, row in _iter_rows_with_index(rows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Rescoring failed at row index {idx}: {type(exc).__name__}: {exc}"
                    ) from exc
                _on_done(idx, result)

    final_rows = [row for row in rescored if row is not None]
    write_jsonl(outfile, final_rows)
    print(f"[rescore_fin_deepsearch_sft] Wrote {len(final_rows)} rows to {outfile}", flush=True)

    summary = aggregate_fin_deepsearch_sft_logs(final_rows)
    summary["output_file"] = str(outfile)
    summary["rows_in_output"] = len(final_rows)

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rescore_fin_deepsearch_sft] Wrote summary to {summary_path}", flush=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate an existing FinDeepSearchSFT result JSONL with the "
            "answer+trajectory LLM judge. Does not call the agent again."
        )
    )
    parser.add_argument("--infile", required=True, type=Path, help="Existing result JSONL.")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Where to write the rescored JSONL. Defaults to <infile_stem>.judged.jsonl.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Where to write the regenerated summary JSON. Defaults to <outfile>.summary.json.",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel judge calls.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument(
        "--env_file",
        default=None,
        help="Path to a FIRE Agent .env file. Defaults to the project .env if present.",
    )
    args = parser.parse_args()
    load_fire_agent_env(args.env_file)

    infile: Path = args.infile.expanduser().resolve()
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    outfile: Path = (
        args.outfile.expanduser().resolve()
        if args.outfile is not None
        else infile.with_suffix("").with_suffix(".judged.jsonl")
    )
    summary_path: Path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else outfile.with_suffix(outfile.suffix + ".summary.json")
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)

    summary = rescore(
        infile=infile,
        outfile=outfile,
        summary_path=summary_path,
        max_workers=max(1, int(args.max_workers)),
        limit=args.limit,
    )
    print(
        "[rescore_fin_deepsearch_sft] Done. strict_accuracy={strict_accuracy} "
        "answer_accuracy={answer_accuracy} trajectory_sound_rate={trajectory_sound_rate}".format(
            strict_accuracy=summary.get("strict_accuracy"),
            answer_accuracy=summary.get("answer_accuracy"),
            trajectory_sound_rate=summary.get("trajectory_sound_rate"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
