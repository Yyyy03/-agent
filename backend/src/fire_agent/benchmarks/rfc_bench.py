from __future__ import annotations

from typing import List

from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_text


class RFCBenchRunner(BenchmarkRunner):
    bench_name = "rfc_bench"
    description = "RFC-BENCH Task 1 reference-free counterfactual financial misinformation detection."
    task_family = "single_turn_qa"
    evaluator = "rfc_bench"
    default_tools = ()
    question_keys = ("question", "perturbed_text", "Perturbed", "perturbed")
    answer_keys = ("answer", "is_manipulated")
    attachment_keys = ()
    url_keys = ()
    sample_id_keys = ("sample_id",)

    def build_prompt(self, record: JsonRow, question: str) -> str:
        if (
            "RFC-BENCH Task 1" in question
            and "factual" in question
            and "manipulated" in question
        ):
            return question

        perturbed_text = normalize_text(first_present(record, ["perturbed_text", "Perturbed", "perturbed", "question"]))
        ticker = normalize_text(first_present(record, ["ticker", "Ticker"]))
        date = normalize_text(first_present(record, ["date", "Date"]))
        title = normalize_text(first_present(record, ["title", "Title"]))
        link = normalize_text(first_present(record, ["link", "Link", "url", "URL"]))

        metadata = []
        if ticker:
            metadata.append(f"Ticker: {ticker}")
        if date:
            metadata.append(f"Date: {date}")
        if title:
            metadata.append(f"Title: {title}")
        if link:
            metadata.append(f"Source link: {link}")
        metadata_block = "\n".join(metadata)
        if metadata_block:
            metadata_block += "\n\n"

        return (
            "RFC-BENCH Task 1: reference-free counterfactual financial misinformation detection.\n"
            "Given one financial news paragraph without the original article or external evidence, "
            "judge whether the paragraph is factual/original or manipulated/misinformation.\n"
            "Return only one label: factual or manipulated.\n\n"
            f"{metadata_block}"
            f"Paragraph:\n{perturbed_text or question}"
        )

    def collect_urls(self, record: JsonRow, prompt: str) -> List[str]:
        # RFC-BENCH is explicitly reference-free: links are provenance only,
        # not solver resources.
        return []

    def collect_attachments(self, record: JsonRow) -> List[str]:
        return []

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "split",
            "perturbation_type",
            "is_manipulated",
            "ticker",
            "Ticker",
            "date",
            "Date",
            "title",
            "Title",
            "link",
            "Link",
            "source_file",
            "source_row_index",
            "perturbed_text",
            "Perturbed",
            "perturbed",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        return extra
