from __future__ import annotations

import re
from typing import Any, List

from .common import (
    FINDOCRESEARCH_REQUIRED_SECTIONS,
    FINDOCRESEARCH_REQUIRED_SUBSECTIONS,
    FINRPT_OFFICIAL_RESPONSE_FIELDS,
    JsonRow,
    normalize_text,
)
from .core import BaseEvaluator, _bench_extra
from .qa import ExactOrSubstringQAEvaluator


class ReportQualityEvaluator(BaseEvaluator):
    name = "report_quality"
    required_sections: List[str] = []

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        prediction = self._prediction(task_log)
        pred_norm = normalize_text(prediction)
        sections = self.required_sections or [
            "overview",
            "financial",
            "risk",
            "summary",
        ]
        section_hits = [section for section in sections if normalize_text(section) in pred_norm]
        section_score = len(section_hits) / len(sections) if sections else 0.0

        # The evidence board was retired with the lean schema; the agent's
        # tool-grounded numbers/sources now live inside the trajectory's
        # ``obs`` strings only. We approximate grounding via tool-usage stats.
        stats = task_log.get("stats") or {}
        tool_calls = int((stats.get("tool_calls") or 0) if isinstance(stats, dict) else 0)
        grounding_score = min(1.0, tool_calls / 8)
        length_score = 1.0 if len(prediction.split()) >= 250 else max(0.0, len(prediction.split()) / 250)
        score = round(0.45 * section_score + 0.4 * grounding_score + 0.15 * length_score, 4)
        return {
            "evaluator": self.name,
            "judgement": "pass" if score >= 0.7 else "needs_review",
            "score": score,
            "structure_score": section_score,
            "grounding_score": grounding_score,
            "length_score": length_score,
            "matched_sections": section_hits,
            "tool_calls": tool_calls,
        }


class FinRptEvaluator(ReportQualityEvaluator):
    name = "finrpt_official_artifact"

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        extra = _bench_extra(task_log)
        # The runner stores the parsed 5-field replay payload under
        # ``bench_extra.prediction_responses`` for the new schema. Old rows
        # kept it at the top level — fall back transparently.
        prediction_responses = extra.get("prediction_responses")
        if not isinstance(prediction_responses, dict):
            prediction_responses = {
                field: extra.get(field) for field in FINRPT_OFFICIAL_RESPONSE_FIELDS
            }
            if not any(prediction_responses.values()):
                # Legacy: fields lived at the top of the row pre-refactor.
                prediction_responses = {
                    field: task_log.get(field) for field in FINRPT_OFFICIAL_RESPONSE_FIELDS
                }
        present_fields = [
            field
            for field in FINRPT_OFFICIAL_RESPONSE_FIELDS
            if str(prediction_responses.get(field) or "").strip()
        ]
        trend_payload = prediction_responses.get("trend_write_response")
        trend_json = self._jsonish_object(trend_payload)
        trend_has_rating = isinstance(trend_json, dict) and bool(trend_json.get("评级"))
        risk_payload = prediction_responses.get("risk_response")
        risk_json_like = isinstance(self._jsonish_object(risk_payload), (dict, list))
        score = len(present_fields) / len(FINRPT_OFFICIAL_RESPONSE_FIELDS)
        return {
            "evaluator": self.name,
            "judgement": "artifact_ready" if score == 1.0 else "incomplete_artifact",
            "score": score,
            "present_fields": present_fields,
            "missing_fields": [field for field in FINRPT_OFFICIAL_RESPONSE_FIELDS if field not in present_fields],
            "trend_has_rating": trend_has_rating,
            "risk_json_like": risk_json_like,
            "official_eval_required": True,
            "note": "This checks FinRpt official JSONL artifact readiness only. Run FinRpt exec_eval.py for benchmark scores.",
        }

    def _jsonish_object(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        try:
            import json_repair

            return json_repair.loads(str(value or ""))
        except Exception:
            return None


class FinDocResearchEvaluator(ReportQualityEvaluator):
    name = "findocresearch_official_artifact"
    required_sections = FINDOCRESEARCH_REQUIRED_SECTIONS

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        prediction = self._prediction(task_log)
        pred_norm = normalize_text(prediction)
        section_hits = [section for section in FINDOCRESEARCH_REQUIRED_SECTIONS if normalize_text(section) in pred_norm]
        subsection_hits = [
            subsection
            for subsection in FINDOCRESEARCH_REQUIRED_SUBSECTIONS
            if normalize_text(subsection) in pred_norm
        ]
        table_count = len(re.findall(r"^\s*\|.*\|\s*$", str(prediction or ""), flags=re.MULTILINE))
        section_score = len(section_hits) / len(FINDOCRESEARCH_REQUIRED_SECTIONS)
        subsection_score = len(subsection_hits) / len(FINDOCRESEARCH_REQUIRED_SUBSECTIONS)
        table_score = min(1.0, table_count / len(FINDOCRESEARCH_REQUIRED_SUBSECTIONS))
        score = round(0.4 * section_score + 0.45 * subsection_score + 0.15 * table_score, 4)
        artifact_ready = section_score == 1.0 and subsection_score == 1.0 and table_score >= 0.5
        return {
            "evaluator": self.name,
            "judgement": "artifact_ready" if artifact_ready else "incomplete_artifact",
            "score": score,
            "section_score": round(section_score, 4),
            "subsection_score": round(subsection_score, 4),
            "table_score": round(table_score, 4),
            "matched_sections": section_hits,
            "matched_subsections": subsection_hits,
            "table_line_count": table_count,
            "official_eval_required": True,
            "note": "This checks OpenFinArena FinDocResearch markdown artifact readiness only. Run evaluation/run.py --track findocresearch for benchmark scores.",
        }


class FinDocBenchEvaluator(ReportQualityEvaluator):
    name = "findocbench_document_quality"
    required_sections = [
        "Document Scope",
        "Key Extracted Facts",
        "Financial Metrics",
        "Risks",
        "Final Summary",
    ]

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        question = normalize_text(task_log.get("question"))
        golden = self._golden(task_log)
        metadata_only = "no web search is needed" in question or "fully contained in the metadata" in question
        if metadata_only and golden:
            result = ExactOrSubstringQAEvaluator().evaluate(task_log)
            result["evaluator"] = self.name
            result["note"] = "Metadata-only document task; used QA-style exact/substr evaluation."
            return result
        return super().evaluate(task_log)


