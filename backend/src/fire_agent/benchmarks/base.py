from __future__ import annotations

import csv
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..aggregation import aggregate_task_logs
from ..evaluators import get_evaluator
from ..runtime import FIREAgent
from ..schemas import BenchTaskContext, compact_text
from .common import (
    JsonRow,
    extract_urls,
    first_present,
    normalize_prompt_text,
    normalize_legacy_rows,
    normalize_text,
    parse_jsonish,
    read_jsonl,
    write_jsonl,
)


NO_SEED_URL = "<NO_SEED_URL>"


class BenchmarkRunner:
    """Lean Flash-Searcher-style runner.

    Each benchmark only fills:

    - ``question_keys`` / ``answer_keys`` / ``attachment_keys`` / ``url_keys`` —
      where to find the prompt + golden + side artefacts in a raw record.
    - ``bench_extra(record)`` — opaque per-bench metadata the bench-specific
      evaluator needs (rubric, judge templates, official response fields …).

    The runner emits the 9-key on-disk schema produced by
    :meth:`AgentResult.to_task_log` plus a ``score`` block from the evaluator.
    """

    bench_name = "generic"
    description = "Generic question-answer benchmark."
    task_family = "qa"
    evaluator = "generic"
    default_tools: Optional[Sequence[str]] = None
    question_keys = ("question", "Question", "prompt", "instruction")
    answer_keys = (
        "answer",
        "Final answer",
        "final_answer",
        "golden_answer",
        "ground_truth",
        "reward_model.ground_truth",
    )
    attachment_keys = ("file_name", "file_path", "attachment", "attachments")
    url_keys = ("root_url", "url", "source_url")
    seed_url_keys = ("root_url",)
    sample_id_keys = ("sample_id", "task_id", "id")

    def __init__(
        self,
        data_path: str | Path,
        output_path: str | Path,
        agent: FIREAgent,
        *,
        agent_factory: Optional[Callable[[], FIREAgent]] = None,
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.agent = agent
        self.agent_factory = agent_factory
        self._worker_agent_local = threading.local()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_records(self) -> List[JsonRow]:
        suffix = self.data_path.suffix.lower()
        if suffix == ".jsonl":
            return read_jsonl(self.data_path)
        if suffix == ".json":
            with self.data_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                data = data.get("data", data.get("rows", [data]))
            if not isinstance(data, list):
                raise ValueError(f"JSON benchmark must contain an object or list: {self.data_path}")
            return [dict(row) for row in data]
        if suffix == ".csv":
            with self.data_path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        raise ValueError(f"Unsupported data format: {self.data_path}")

    # ------------------------------------------------------------------
    # Context materialization
    # ------------------------------------------------------------------
    def materialize_context(self, record: JsonRow, task_index: int) -> BenchTaskContext:
        question = normalize_prompt_text(first_present(record, self.question_keys))
        if not question:
            raise ValueError(f"{self.bench_name} record {task_index} has no question/prompt field")
        prompt = self.build_prompt(record, question)
        urls = self.collect_urls(record, prompt)
        attachments = self.collect_attachments(record)
        golden = first_present(record, self.answer_keys)
        return BenchTaskContext(
            bench_name=self.bench_name,
            task_index=task_index,
            task_prompt=prompt,
            raw_question=question,
            raw_record=record,
            attachments=attachments,
            urls=urls,
            golden_answer=golden,
            bench_extra=self.bench_extra(record),
            task_family=self.task_family,
            evaluator_name=self.evaluator,
        )

    def build_prompt(self, record: JsonRow, question: str) -> str:
        return self.format_task_prompt(question, self.seed_url_for_record(record))

    def format_task_prompt(self, question: str, seed_url: str | None = None) -> str:
        seed_url = normalize_text(seed_url) or NO_SEED_URL
        return f"Question:\n{question}\n\nSeed URL:\n{seed_url}"

    def seed_url_for_record(self, record: JsonRow) -> str:
        for key in self.seed_url_keys:
            for item in self.iter_url_values(record.get(key)):
                url = normalize_text(item)
                if url.startswith(("http://", "https://")):
                    return url
        return NO_SEED_URL

    def bench_extra(self, record: JsonRow) -> JsonRow:
        """Per-bench evaluator/aggregator side-channel data.

        Override in each runner. The returned dict goes verbatim into
        ``task_log["bench_extra"]`` and replaces all of the previous
        ``eval_metadata`` / ``solver_metadata`` / ``evaluator_metadata`` /
        ``benchmark_scoring_input`` machinery.
        """
        return {}

    def collect_urls(self, record: JsonRow, prompt: str) -> List[str]:
        urls = extract_urls(prompt)
        for key in self.url_keys:
            for item in self.iter_url_values(record.get(key)):
                urls.append(normalize_text(item))
        return [url for url in dict.fromkeys(urls) if url.startswith(("http://", "https://"))]

    def iter_url_values(self, value: Any) -> Iterable[Any]:
        if value in (None, ""):
            return
        if isinstance(value, list):
            for item in value:
                yield from self.iter_url_values(item)
            return
        if isinstance(value, dict):
            for key in ["url", "source_url", "pdf_url", "download_url", "href"]:
                if value.get(key):
                    yield value[key]
            return
        parsed = parse_jsonish(value)
        if isinstance(parsed, (list, dict)) and parsed != value:
            yield from self.iter_url_values(parsed)
            return
        yield value

    def collect_attachments(self, record: JsonRow) -> List[str]:
        attachments: List[str] = []
        for key in self.attachment_keys:
            value = record.get(key)
            for item in self.iter_attachment_values(value):
                attachment = normalize_text(item)
                if attachment:
                    attachments.append(self.resolve_attachment_path(attachment))
        return list(dict.fromkeys(attachments))

    def iter_attachment_values(self, value: Any) -> Iterable[Any]:
        if value in (None, ""):
            return
        if isinstance(value, list):
            for item in value:
                yield from self.iter_attachment_values(item)
            return
        if isinstance(value, dict):
            for key in ["path", "file_path", "file_name", "filename", "attachment"]:
                if value.get(key):
                    yield value[key]
            return
        parsed = parse_jsonish(value)
        if isinstance(parsed, (list, dict)) and parsed != value:
            yield from self.iter_attachment_values(parsed)
            return
        yield value

    def resolve_attachment_path(self, attachment: str) -> str:
        if attachment.startswith(("http://", "https://")):
            return attachment
        path = Path(attachment).expanduser()
        if path.is_absolute():
            return str(path)
        candidates = [self.data_path.parent / path, Path.cwd() / path]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(self.data_path.parent / path)

    # ------------------------------------------------------------------
    # Judging + row assembly
    # ------------------------------------------------------------------
    def judge(self, task_log: JsonRow) -> JsonRow:
        if str(os.getenv("FIRE_AGENT_DISABLE_JUDGE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            return {
                "evaluator": self.evaluator,
                "judgement": None,
                "score": None,
                "is_correct": None,
                "mode": "judge_disabled",
                "note": "Judge disabled by FIRE_AGENT_DISABLE_JUDGE=1; trajectory generation only.",
            }
        return get_evaluator(self.evaluator).evaluate(task_log)

    def output_record(self, context: BenchTaskContext, task_log: JsonRow, judge_output: JsonRow) -> JsonRow:
        record = context.raw_record
        sample_id = first_present(record, list(self.sample_id_keys), context.task_index)
        row: JsonRow = dict(task_log)
        row["sample_id"] = sample_id
        row["score"] = judge_output if isinstance(judge_output, dict) else {}
        # Make sure benchmark identifying fields stay near the top of the record
        # for human readability when inspecting JSONL files.
        ordered_keys = [
            "episode_id",
            "bench_name",
            "task_index",
            "sample_id",
            "question",
            "task",
            "generation",
            "tools",
            "turns",
            "auxiliary_turns",
            "golden_answer",
            "agent_result",
            "bench_extra",
            "score",
            "status",
            "error",
            "error_traceback",
            "stats",
            "agent_trajectory",
        ]
        ordered: JsonRow = {key: row[key] for key in ordered_keys if key in row}
        for key, value in row.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    # ------------------------------------------------------------------
    # Resume / batching
    # ------------------------------------------------------------------
    def completed_indices(self) -> set[int]:
        if not self.output_path.exists():
            return set()
        terminal_statuses = {
            "success",
            "max_steps",
            "error",
            "runner_error",
            "empty_final_answer",
            "protocol_no_answer",
        }
        completed = set()
        for row in read_jsonl(self.output_path):
            if row.get("status") in terminal_statuses:
                try:
                    completed.add(int(row["task_index"]))
                except Exception:
                    continue
        return completed

    def _build_runner_error_log(
        self,
        context: BenchTaskContext,
        exc: Exception,
        *,
        traceback_text: str = "",
    ) -> JsonRow:
        return {
            "bench_name": context.bench_name,
            "task_index": context.task_index,
            "question": context.task_prompt,
            "task": {"question": context.task_prompt},
            "generation": {},
            "tools": [],
            "turns": [],
            "auxiliary_turns": [],
            "golden_answer": context.golden_answer,
            "agent_result": "",
            "agent_trajectory": [],
            "bench_extra": dict(context.bench_extra or {}),
            "status": "runner_error",
            "error": f"{type(exc).__name__}: {exc}",
            "error_traceback": compact_text(traceback_text or traceback.format_exc(), 12000),
            "stats": {
                "steps": 0,
                "tool_calls": 0,
                "tools_by_name": {},
                "tokens": {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0},
            },
        }

    def _agent_for_current_thread(self) -> FIREAgent:
        if self.agent_factory is None or threading.current_thread() is threading.main_thread():
            return self.agent
        agent = getattr(self._worker_agent_local, "agent", None)
        if agent is None:
            agent = self.agent_factory()
            self._worker_agent_local.agent = agent
        return agent

    def _run_single(self, records: List[JsonRow], idx: int) -> JsonRow:
        context = self.materialize_context(records[idx], idx)
        try:
            result = self._agent_for_current_thread().run(context)
            task_log = result.to_task_log()
        except Exception as exc:
            task_log = self._build_runner_error_log(context, exc, traceback_text=traceback.format_exc())
        judge_output = self.judge(task_log)
        return self.output_record(context, task_log, judge_output)

    def run(
        self,
        task_indices: Optional[List[int]] = None,
        sample_num: Optional[int] = None,
        resume: bool = False,
        max_workers: int = 1,
    ) -> List[JsonRow]:
        records = self.load_records()
        selected = list(range(len(records))) if task_indices is None else [idx for idx in task_indices if 0 <= idx < len(records)]
        if sample_num is not None:
            selected = selected[:sample_num]
        if resume:
            completed = self.completed_indices()
            selected = [idx for idx in selected if idx not in completed]

        if max_workers <= 1:
            logs: List[JsonRow] = []
            for idx in selected:
                output = self._run_single(records, idx)
                logs.append(output)
                write_jsonl(self.output_path, [output], mode="a")
            self.finalize_after_run(logs)
            return logs

        write_lock = threading.Lock()
        results_by_index: dict[int, JsonRow] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single, records, idx): idx
                for idx in selected
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    output = future.result()
                except Exception as exc:
                    output = {
                        "bench_name": self.bench_name,
                        "task_index": idx,
                        "status": "runner_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_traceback": compact_text(traceback.format_exc(), 12000),
                    }
                results_by_index[idx] = output
                with write_lock:
                    write_jsonl(self.output_path, [output], mode="a")

        ordered = [results_by_index[idx] for idx in selected if idx in results_by_index]
        self.finalize_after_run(ordered)
        return ordered

    # ------------------------------------------------------------------
    # Finalization & summary
    # ------------------------------------------------------------------
    def finalize_after_run(self, logs: List[JsonRow]) -> None:
        self.write_summary(logs)

    def aggregate(self, logs: List[JsonRow]) -> JsonRow:
        """Compute a benchmark-level summary dict from per-row logs."""
        return aggregate_task_logs(logs)

    def write_summary(self, logs: List[JsonRow]) -> Path:
        rows: List[JsonRow] = []
        if self.output_path.exists():
            try:
                rows = read_jsonl(self.output_path)
            except Exception:
                rows = list(logs)
        if not rows:
            rows = list(logs)
        rows = normalize_legacy_rows(rows)

        summary = self.aggregate(rows)
        if "benchmark" not in summary:
            summary = {"benchmark": self.bench_name, **summary}
        summary["data_path"] = str(self.data_path)
        summary["output_file"] = str(self.output_path)
        summary["rows_in_output"] = len(rows)

        summary_path = self.output_path.with_suffix(self.output_path.suffix + ".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{self.bench_name}] Summary written to {summary_path}", flush=True)
        try:
            self._print_summary_brief(summary)
        except Exception:
            pass
        return summary_path

    def _print_summary_brief(self, summary: JsonRow) -> None:
        bench = summary.get("benchmark", self.bench_name)
        total = summary.get("total")
        acc = summary.get("accuracy")
        acc_pct = summary.get("accuracy_pct")
        token_stats = summary.get("token_stats", {}) or {}
        total_tokens = token_stats.get("total_tokens")
        llm_calls = token_stats.get("total_llm_calls")
        line = f"[{bench}] total={total}"
        if acc is not None:
            line += f" accuracy={acc:.4f}"
        if acc_pct is not None:
            line += f" ({acc_pct:.2f}%)"
        if total_tokens is not None:
            line += f" total_tokens={total_tokens}"
        if llm_calls is not None:
            line += f" llm_calls={llm_calls}"
        print(line, flush=True)
