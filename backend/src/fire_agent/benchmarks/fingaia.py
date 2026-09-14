from __future__ import annotations

from typing import List

from ..aggregation import aggregate_fingaia_logs
from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_text


class FinGAIARunner(BenchmarkRunner):
    bench_name = "fingaia"
    description = "FinGAIA QA tasks. Lean schema; LLM-as-judge with 1/0/-1 protocol."
    task_family = "qa"
    evaluator = "fingaia"
    question_keys = ("Question", "question", "prompt", "问题")
    answer_keys = ("Final answer", "final_answer", "answer", "答案", "standard_answer")
    attachment_keys = (
        "file_name",
        "file_path",
        "attachment",
        "attachments",
        "附件",
        "files",
        "supporting_files",
    )

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "Level",
            "level",
            "Scenario",
            "scenario",
            "Financial Scenario",
            "金融场景",
            "Scenario Depth",
            "场景深度",
            "Annotator Metadata",
            "as_of_date",
            "date",
            "trade_date",
            "report_date",
            "benchmark_date",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        question = normalize_text(first_present(record, self.question_keys))
        if question:
            extra["question"] = question
        return extra

    def aggregate(self, logs: List[JsonRow]) -> JsonRow:
        return aggregate_fingaia_logs(logs)
