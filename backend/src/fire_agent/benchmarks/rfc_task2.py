from __future__ import annotations

from typing import List

from ..aggregation import aggregate_rfc_task2_logs
from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_text


class RFCTask2Runner(BenchmarkRunner):
    bench_name = "rfc_task2"
    description = "RFC-BENCH Task 2 source-grounded manipulation type classification."
    task_family = "single_turn_qa"
    evaluator = "rfc_task2"
    default_tools = ()
    question_keys = ("perturbed_text", "question")
    answer_keys = ("perturbation_type", "answer")
    attachment_keys = ()
    url_keys = ()
    sample_id_keys = ("sample_id",)
    LABEL_DESCRIPTIONS = (
        "numerical: salient quantitative information is altered, including but not limited to amounts, percentages, prices, counts, forecasts, ratios, or financial metrics.",
        "flipping: directional or polarity meaning is reversed, including but not limited to bullish vs. bearish, gain vs. loss, increase vs. decrease, or upgrade vs. downgrade.",
        "sentiment: evaluative tone, intensity, or stance is amplified or weakened toward a more bullish or bearish interpretation while core facts remain related.",
        "causal: the stated cause, driver, attribution, reason, or consequence is distorted, replaced, or made misleading.",
    )

    def build_prompt(self, record: JsonRow, question: str) -> str:
        source_page = normalize_text(first_present(record, ["source_page", "original_evidence", "evidence"]))
        perturbed_text = normalize_text(first_present(record, ["perturbed_text", "Perturbed", "perturbed", "question"]))
        labels = "\n".join(f"- {item}" for item in self.LABEL_DESCRIPTIONS)
        return (
            "RFC-BENCH Task 2: source-grounded counterfactual financial misinformation type classification.\n"
            "Given the source article/page content and a manipulated financial news paragraph, "
            "identify which manipulation type was applied.\n"
            "Use only the provided source page and manipulated paragraph. Do not browse, retrieve, or use outside evidence.\n\n"
            "Labels:\n"
            f"{labels}\n\n"
            "The descriptions above are guidelines and examples, not exhaustive rules. "
            "Choose the single label that best captures the main manipulation mechanism.\n"
            "The benchmark answer string must be exactly one label from: numerical, flipping, sentiment, causal. "
            "Output only one label; do not list or restate the labels. "
            "Place only that label inside final_answer.arguments.answer.\n\n"
            f"Source page:\n{source_page}\n\n"
            f"Manipulated paragraph:\n{perturbed_text or question}"
        )

    def collect_urls(self, record: JsonRow, prompt: str) -> List[str]:
        # Task 2 is source-grounded on the already materialized source_page field.
        return []

    def collect_attachments(self, record: JsonRow) -> List[str]:
        return []

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "split",
            "perturbation_type",
            "is_manipulated",
            "answer",
            "ticker",
            "Ticker",
            "date",
            "Date",
            "title",
            "Title",
            "link",
            "Link",
            "source_page_url",
            "source_page_source",
            "source_page_char_len",
            "source_file",
            "source_row_index",
            "perturbed_text",
            "Perturbed",
            "perturbed",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = normalize_text(first_present(record, self.answer_keys))
        if golden:
            extra["golden_answer"] = golden
        return extra

    def aggregate(self, logs: List[JsonRow]) -> JsonRow:
        return aggregate_rfc_task2_logs(logs)
