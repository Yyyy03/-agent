from __future__ import annotations

from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_text


class FinSearchCompRunner(BenchmarkRunner):
    bench_name = "finsearchcomp"
    description = "Chinese financial search competition tasks. Lean schema; official LLM judge."
    task_family = "qa"
    evaluator = "finsearchcomp"
    question_keys = ("question", "prompt")
    answer_keys = ("answer", "response_reference", "ground_truth")

    # Per-row judge templates the official scorer needs.
    # ``akshare_ticker`` / ``tags`` / ``method`` / ``wind_ticker`` /
    # ``yfinance_ticker`` / ``time`` are required by the T1 (Time-Sensitive)
    # realtime ground-truth refresher in :mod:`fire_agent.evaluators.qa`.
    _required_extra_keys = (
        "label",
        "ground_truth",
        "response_reference",
        "judge_prompt_template",
        "judge_system_prompt",
        "prompt_id",
        "akshare_ticker",
        "tags",
        "method",
        "wind_ticker",
        "yfinance_ticker",
        "time",
        "sample_id",
        "source_row_index",
        "source_dataset",
    )

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in self._required_extra_keys:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        # The official judge template references {prompt}; surface the raw
        # question under that key so the bench-side template renders.
        if record.get("prompt"):
            extra["prompt"] = record["prompt"]
        else:
            question = normalize_text(first_present(record, self.question_keys))
            if question:
                extra["prompt"] = question
        return extra
