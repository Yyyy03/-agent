from __future__ import annotations

from pathlib import Path
from typing import List

from ..schemas import BenchTaskContext
from .base import BenchmarkRunner
from .common import (
    JsonRow,
    first_present,
    normalize_text,
    sanitize_file_stem,
)


class FinDocResearchRunner(BenchmarkRunner):
    bench_name = "findocresearch"
    description = "OpenFinArena FinDocResearch annual-report-grounded markdown report-generation tasks."
    task_family = "report"
    evaluator = "findocresearch"
    question_keys = ("question", "Question", "prompt", "instruction")
    answer_keys = ("answer", "Answer", "golden_answer")
    attachment_keys = (
        "file_name",
        "file_path",
        "pdf_path",
        "document_path",
        "attachment",
        "attachments",
        "FY23 File Name",
        "FY24 File Name",
        "FY23 PDF Path",
        "FY24 PDF Path",
        "fy23_file_name",
        "fy24_file_name",
        "fy23_pdf_path",
        "fy24_pdf_path",
        "fy2023_pdf_path",
        "fy2024_pdf_path",
        "annual_report_paths",
    )
    url_keys = (
        "root_url",
        "url",
        "source_url",
        "pdf_url",
        "document_url",
        "FY23 PDF Download Site",
        "FY24 PDF Download Site",
        "fy23_pdf_download_site",
        "fy24_pdf_download_site",
        "annual_report_urls",
    )

    def materialize_context(self, record: JsonRow, task_index: int) -> BenchTaskContext:
        self._ensure_report_generation_record(record, task_index)
        if not first_present(record, self.question_keys):
            raise ValueError(
                f"FinDocResearch record {task_index} has no question/prompt/instruction field; "
                f"refusing to synthesize one from metadata."
            )
        return super().materialize_context(record, task_index)

    def _ensure_report_generation_record(self, record: JsonRow, task_index: int) -> None:
        note = normalize_text(record.get("benchmark_note")).lower()
        question = normalize_text(first_present(record, self.question_keys)).lower()
        synthetic_metadata_qa = "synthetic text-only qa" in note or "using only the metadata above, answer this question" in question
        if synthetic_metadata_qa:
            raise ValueError(
                "FinDocResearch runner expects official report-generation samples, not metadata QA rows. "
                f"Use findocresearch.jsonl with annual-report inputs for task index {task_index}."
            )

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "case_id",
            "Case ID",
            "company",
            "Company Name",
            "company_name",
            "market",
            "Market",
            "stock_market",
            "Stock Market",
            "location",
            "Output Report Language",
            "output_report_language",
            "language",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        return extra

    def collect_attachments(self, record: JsonRow) -> List[str]:
        attachments = super().collect_attachments(record)
        existing_or_remote: List[str] = []
        for attachment in attachments:
            if attachment.startswith(("http://", "https://")) or Path(attachment).exists():
                existing_or_remote.append(attachment)
        return existing_or_remote

    def output_record(self, context: BenchTaskContext, task_log: JsonRow, judge_output: JsonRow) -> JsonRow:
        output = super().output_record(context, task_log, judge_output)
        markdown_path = self._write_markdown_prediction(context, task_log)
        bench_extra = dict(output.get("bench_extra") or {})
        bench_extra["prediction_file"] = str(markdown_path)
        bench_extra["prediction_folder"] = str(markdown_path.parent)
        output["bench_extra"] = bench_extra
        return output

    def _case_id(self, record: JsonRow, task_index: int) -> str:
        value = first_present(record, ["case_id", "Case ID", "sample_id", "task_id", "id", "file_stem"])
        return sanitize_file_stem(value, f"task_{task_index:03d}")

    def _write_markdown_prediction(self, context: BenchTaskContext, task_log: JsonRow) -> Path:
        markdown_dir = self.output_path.parent / self.output_path.stem / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        case_id = self._case_id(context.raw_record, context.task_index)
        markdown_path = markdown_dir / f"{case_id}.md"
        markdown_path.write_text(str(task_log.get("agent_result") or ""), encoding="utf-8")
        return markdown_path
