from __future__ import annotations

import re
from typing import Any, List

from ..aggregation import aggregate_bizfinbench_v2_logs
from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_prompt_text


class BizFinBenchV2Runner(BenchmarkRunner):
    bench_name = "bizfinbench_v2"
    description = "BizFinBench.v2 bilingual single-turn financial QA."
    task_family = "single_turn_qa"
    evaluator = "bizfinbench_v2"
    default_tools = ()
    question_keys = ("question", "prompt", "messages")
    answer_keys = ("answer",)
    attachment_keys = ()
    url_keys = ()
    sample_id_keys = ("sample_id",)

    def build_prompt(self, record: JsonRow, question: str) -> str:
        # BizFinBench.v2 rows already carry the official task prompt. Normalize
        # chat-message prompts and strip legacy TL direct-output instructions so
        # the harness remains aligned to native final_answer tool calls.
        raw_prompt = first_present(record, self.question_keys, question)
        prompt = normalize_prompt_text(raw_prompt or question)
        return self._align_native_final_answer_channel(prompt)

    @staticmethod
    def _align_native_final_answer_channel(prompt: Any) -> str:
        text = normalize_prompt_text(prompt)
        if not text:
            return ""

        text = re.sub(
            r"\n*输出格式要求[:：]\s*请以纯文本形式输出\s*JSON\s*，?"
            r"\s*不要包含任何代码块标记（如\s*```json\s*或\s*```）[:：]?\s*",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\n*请严格遵守题目中给定的输出格式，?只输出最终答案，?不要包含额外解释。?",
            "\n",
            text,
        )
        text = re.sub(
            r"\n*请以纯文本形式输出\s*JSON\s*，?\s*不要包含任何代码块标记（如\s*```json\s*或\s*```）[:：]?\s*",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if "final_answer.arguments.answer" not in text:
            text = (
                f"{text}\n\n"
                "Return the requested benchmark answer through the native final_answer tool call. "
                "Put the exact required answer payload text in final_answer.arguments.answer."
            ).strip()
        return text

    def collect_urls(self, record: JsonRow, prompt: str) -> list[str]:
        return []

    def collect_attachments(self, record: JsonRow) -> list[str]:
        return []

    def aggregate(self, logs: List[JsonRow]) -> JsonRow:
        return aggregate_bizfinbench_v2_logs(logs)

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "language",
            "task_type",
            "source_file",
            "source_row_index",
            "messages",
            "choices",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        return extra
