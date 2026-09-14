"""Offline LLM-as-judge rescoring for FinanceAgentBench result files.

Usage:

    python -m fire_agent.rescore_financeagent \
        --infile output/fire_financeagent_public_full_results.jsonl \
        --outfile output/fire_financeagent_public_full_results.llm.jsonl

The script reads existing FIRE Agent result rows (which already contain
``agent_result``, ``evaluator_metadata.rubric``, etc.), runs the new
``FinanceAgentBenchEvaluator`` on each row using the configured judge model,
and writes a new JSONL where ``metrics`` and ``judge_output`` are replaced by
the LLM judgements. A ``*.summary.json`` file is regenerated alongside the
output.

Requires ``FIRE_AGENT_*`` env vars (or ``FIRE_AGENT_FINANCEAGENT_JUDGE_MODEL``
override) so the judge model can be instantiated; if no judge model is
configured the evaluator leaves the score unset and tags rows with
``mode='judge_unavailable'``.
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
    from .aggregation import aggregate_task_logs
    from .benchmarks.common import normalize_legacy_row, read_jsonl, write_jsonl
    from .evaluators.qa import FinanceAgentBenchEvaluator
except ImportError:  # pragma: no cover - allow direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fire_agent.aggregation import aggregate_task_logs  # type: ignore
    from fire_agent.benchmarks.common import normalize_legacy_row, read_jsonl, write_jsonl  # type: ignore
    from fire_agent.evaluators.qa import FinanceAgentBenchEvaluator  # type: ignore


PROGRESS_LOCK = threading.Lock()


def _rescore_row(evaluator: FinanceAgentBenchEvaluator, row: Dict[str, Any]) -> Dict[str, Any]:
    new_row = normalize_legacy_row(row)
    judge_output = evaluator.evaluate(new_row)
    existing = dict(new_row.get("score") or {})
    existing.update(judge_output)
    new_row["score"] = existing
    return new_row


def _iter_rows_with_index(rows: List[Dict[str, Any]]) -> Iterable[tuple]:
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

    evaluator = FinanceAgentBenchEvaluator()
    rescored: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    total = len(rows)
    done = 0

    def _on_done(idx: int, result: Dict[str, Any]) -> None:
        nonlocal done
        rescored[idx] = result
        with PROGRESS_LOCK:
            done += 1
            print(
                f"[rescore] {done}/{total} task_index={result.get('task_index')} "
                f"is_correct={result.get('score', {}).get('is_correct')} "
                f"mode={result.get('score', {}).get('mode')}",
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
                except Exception as exc:  # pragma: no cover - propagate visibility
                    raise RuntimeError(
                        f"Rescoring failed at row index {idx}: {type(exc).__name__}: {exc}"
                    ) from exc
                _on_done(idx, result)

    final_rows = [row for row in rescored if row is not None]
    write_jsonl(outfile, final_rows)
    print(f"[rescore] Wrote {len(final_rows)} rows to {outfile}", flush=True)

    summary = aggregate_task_logs(final_rows)
    summary["benchmark"] = "financeagentbench"
    summary["output_file"] = str(outfile)
    summary["rows_in_output"] = len(final_rows)
    summary["protocol"] = FinanceAgentBenchEvaluator.PROTOCOL
    summary["judge_modes"] = _count_modes(final_rows)

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rescore] Wrote summary to {summary_path}", flush=True)

    return summary


def _count_modes(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        mode = str((row.get("score") or {}).get("mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate an existing FinanceAgentBench result JSONL with the "
            "LLM-as-judge evaluator. Does not call the agent again; only "
            "re-runs the judge."
        )
    )
    parser.add_argument("--infile", required=True, type=Path, help="Existing result JSONL.")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Where to write the rescored JSONL. Defaults to <infile_stem>.llm.jsonl alongside infile.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Where to write the regenerated summary JSON. Defaults to <outfile>.summary.json.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of judge calls to run in parallel (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to rescore (useful for smoke tests).",
    )
    args = parser.parse_args()

    infile: Path = args.infile.expanduser().resolve()
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    outfile: Path = (
        args.outfile.expanduser().resolve()
        if args.outfile is not None
        else infile.with_suffix("").with_suffix(".llm.jsonl")
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
        "[rescore] Done. accuracy={accuracy} avg_judge_score={avg_judge_score} "
        "judge_modes={judge_modes}".format(
            accuracy=summary.get("accuracy"),
            avg_judge_score=summary.get("avg_judge_score"),
            judge_modes=summary.get("judge_modes"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
