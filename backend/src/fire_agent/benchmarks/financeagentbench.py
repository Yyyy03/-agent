from __future__ import annotations

from .base import BenchmarkRunner
from .common import JsonRow, first_present, parse_jsonish


class FinanceAgentBenchRunner(BenchmarkRunner):
    bench_name = "financeagentbench"
    description = "Finance-Agent public benchmark with rubric-guided LLM judging."
    task_family = "qa"
    evaluator = "financeagentbench"
    question_keys = ("question", "Question")
    answer_keys = ("answer", "Answer")

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        rubric = parse_jsonish(record.get("rubric") or record.get("Rubric"), default=[])
        if rubric:
            extra["rubric"] = rubric
        for key, source_keys in (
            ("question_type", ["question_type", "Question Type"]),
            ("expert_time_mins", ["expert_time_mins", "Expert time (mins)"]),
        ):
            value = first_present(record, source_keys)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        return extra


class FinanceAgentV2Runner(FinanceAgentBenchRunner):
    bench_name = "financeagent_v2"
    description = "Finance Agent v2 public benchmark with v2 prompt/tool contract."
