from __future__ import annotations

from typing import Any, List

from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_prompt_text, normalize_text


class _FinanceBenchBaseRunner(BenchmarkRunner):
    task_family = "single_turn_qa"
    evaluator = "financebench"
    default_tools = ()
    question_keys = ("raw_question", "question", "prompt", "messages")
    answer_keys = ("answer", "reward_model.ground_truth")
    attachment_keys = ()
    url_keys = ()
    sample_id_keys = ("sample_id", "financebench_id")

    def collect_attachments(self, record: JsonRow) -> List[str]:
        return []

    def collect_urls(self, record: JsonRow, prompt: str) -> List[str]:
        return []

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "financebench_id",
            "company",
            "doc_name",
            "question_type",
            "question_reasoning",
            "domain_question_num",
            "raw_question",
            "answer",
            "justification",
            "evidence",
            "dataset_subset_label",
            "gics_sector",
            "doc_type",
            "doc_period",
            "doc_link",
            "attachment_path",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        extra["financebench_eval_mode"] = self.eval_mode
        return extra

    @staticmethod
    def _raw_question(record: JsonRow, question: str) -> str:
        return normalize_prompt_text(record.get("raw_question") or question)


class FinanceBenchClosedBookRunner(_FinanceBenchBaseRunner):
    bench_name = "financebench_closed_book"
    description = "FinanceBench closed-book QA: user question only, no filings or evidence."
    eval_mode = "closed_book"

    def build_prompt(self, record: JsonRow, question: str) -> str:
        raw_question = self._raw_question(record, question)
        return (
            "Answer the FinanceBench user question. No financial reports, PDFs, "
            "evidence pages, retrieval results, or external tools are provided.\n"
            "Give a concise final answer through the native final_answer tool call.\n\n"
            f"Question:\n{raw_question}"
        )


class FinanceBenchRunner(_FinanceBenchBaseRunner):
    bench_name = "financebench"
    description = "FinanceBench evidence-provided QA over gold evidence pages and fragments."
    eval_mode = "evidence_provided"

    def build_prompt(self, record: JsonRow, question: str) -> str:
        if record.get("prompt") not in (None, "") and not record.get("question"):
            return normalize_prompt_text(record.get("prompt"))
        raw_question = self._raw_question(record, question)
        evidence_block = self._format_evidence(record.get("evidence"))
        return (
            "Answer the FinanceBench user question using only the provided evidence "
            "pages, evidence fragments, and financial tables. Do not use external "
            "reports, PDFs, web search, SEC search, or retrieval tools.\n"
            "Provide a concise final answer and a brief justification grounded in the evidence. "
            "Place that content inside final_answer.arguments.answer.\n\n"
            f"Question:\n{raw_question}\n\n"
            f"Evidence:\n{evidence_block or '[No evidence provided]'}"
        )

    @classmethod
    def _format_evidence(cls, evidence: Any) -> str:
        if evidence in (None, ""):
            return ""
        if isinstance(evidence, list):
            parts = [cls._format_one_evidence(item, index + 1) for index, item in enumerate(evidence)]
            return "\n\n".join(part for part in parts if part).strip()
        return cls._format_one_evidence(evidence, 1)

    @staticmethod
    def _format_one_evidence(item: Any, index: int) -> str:
        if isinstance(item, dict):
            doc_name = normalize_text(item.get("doc_name"))
            page = normalize_text(item.get("evidence_page_num"))
            fragment = normalize_text(item.get("evidence_text"))
            full_page = normalize_text(item.get("evidence_text_full_page"))
            header = f"[Evidence {index}]"
            details = []
            if doc_name:
                details.append(f"Document: {doc_name}")
            if page:
                details.append(f"Page: {page}")
            if details:
                header = f"{header} " + "; ".join(details)
            body_parts = []
            if fragment:
                body_parts.append(f"Relevant fragment/table:\n{fragment}")
            if full_page and full_page != fragment:
                body_parts.append(f"Full evidence page:\n{full_page}")
            body = "\n\n".join(body_parts).strip()
            return f"{header}\n{body}".strip()
        text = normalize_text(item)
        return f"[Evidence {index}]\n{text}" if text else ""
