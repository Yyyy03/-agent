from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..llm import parse_json_object
from ..schemas import compact_text
from .common import JsonRow
from .core import BaseEvaluator, _bench_extra


class FinDeepSearchSFTEvaluator(BaseEvaluator):
    """LLM-as-judge evaluator for financial deep-search SFT seeds.

    This evaluator scores two coupled questions:

    1. Is the final answer equivalent to the reference answer?
    2. Does the recorded trajectory actually support that final answer?

    A row is counted correct only when both are true. This intentionally
    rejects lucky guesses and answers that match the gold label but are not
    backed by the agent's evidence chain.
    """

    name = "fin_deepsearch_sft_llm_judge"

    PROTOCOL = (
        "FinDeepSearchSFT LLM-as-judge: dynamic answer criteria, answer equivalence, "
        "contradiction gate, and evidence-chain audit for financial search tasks. "
        "Overall correctness requires answer_score >= 0.8, trajectory_score >= 0.7, "
        "and no blocking contradiction."
    )

    SYSTEM_PROMPT = (
        "You are a strict evaluator for financial deep-search QA agents.\n"
        "You judge only from the supplied question, reference answer, model final answer, and recorded trajectory.\n"
        "Do not browse the web. Do not solve the task from outside knowledge.\n"
        "Your job is to decide whether the final answer is equivalent to the reference answer, and whether the trajectory supports the answer.\n"
        "Before scoring, derive concise task criteria from only the question and reference answer; then judge both answer coverage and trajectory support against those criteria.\n"
        "Apply financial-search audit lenses: task contract, source provenance, temporal/as-of basis, metric/unit basis, calculation path, and contradiction checks.\n"
        "A correct final answer with an unsupported, contradictory, or fabricated trajectory must NOT receive full credit.\n"
        "Return ONLY valid JSON. Do not wrap it in markdown."
    )

    USER_INSTRUCTIONS = (
        "Evaluate the model answer and trajectory.\n\n"
        "Answer-equivalence rules:\n"
        "- Judge the final answer against the reference answer for the original question.\n"
        "- Separate the leading answer atom from supporting explanation. For single-answer tasks, judge answer equivalence primarily on the leading entity/value/code/date; unsupported explanation should affect trajectory/support or format quality, not answer correctness, unless it introduces a competing answer or contradicts the atom.\n"
        "- Accept harmless wording, order, punctuation, abbreviation, and unit-format differences.\n"
        "- Accept a more specific subtype, official variant, or implementation name when it unambiguously belongs to the reference answer and the question does not explicitly require only the broader parent label. For example, CME Term SOFR can match SOFR for a SOFR benchmark question.\n"
        "- For hierarchical classification codes, identifiers, or prefixes, accept a prediction that extends the reference with a delimiter or subcode digits when the question asks for a prefix or does not specify an exact digit length. For example, 6205.20 can match 6205 for a code-prefix question.\n"
        "- For numeric answers, accept reasonable rounding or equivalent percentage/decimal forms only when the question/reference permits it.\n"
        "- Do not mark contradiction merely because the prediction is more specific than the reference; contradiction requires a factual conflict with the reference, the question contract, or the cited evidence.\n"
        "- Mark incorrect if required final answer entities, dates, values, units, or list items are missing, extra conflicting items appear, or the answer is ambiguous.\n\n"
        "Output-shape rules:\n"
        "- If task_metadata.dataset_output_constraint is present, use it to judge whether the final answer shape is too broad, too verbose, ambiguous, or contains extra candidates.\n"
        "- Output-shape problems should lower task_contract_alignment and may lower answer_score only when they obscure the answer atom, add conflicting candidates, omit required list items, or change the requested value.\n"
        "- For short-answer datasets, a correct first answer atom followed by unnecessary explanation is usually answer-equivalent but may have unsupported_claims or partial task_contract_alignment.\n\n"
        "Trajectory-support rules:\n"
        "- Judge whether the trajectory contains enough evidence, calculations, source choices, and reasoning to support the final answer.\n"
        "- Treat the question as a contract: required entities, dates/as-of windows, metric definitions, source basis, units, list filters, denominators, and requested output shape must be addressed.\n"
        "- Good trajectories use relevant and preferably authoritative sources. For exact financial numbers, filings, official tables, exchange/provider data, regulator/company materials, or reproducible structured data are stronger than generic snippets.\n"
        "- Do not require a single specific source family; judge whether the chosen sources are appropriate for this task. Avoid over-penalizing reputable secondary sources for stable, non-exact, or entity-identification facts.\n"
        "- Search snippets alone are weak evidence for exact numeric, tabular, filing, or time-bounded answers. They can support discovery or simple entity identification.\n"
        "- Penalize unsupported leaps, hallucinated citations, stale/current data used for historical questions, evidence that contradicts the final answer, missing calculations, and final answers copied without evidence.\n"
        "- Extra unsuccessful searches are acceptable if the final supporting path is clear and non-contradictory.\n"
        "- If the trajectory is empty, contains only the final answer, or has no usable evidence, trajectory_score should be 0 even if the answer matches.\n"
        "- Cite supporting trajectory step_index values in evidence_anchors. If no specific steps support the answer, evidence_anchors must be empty and trajectory_score should be low.\n\n"
        "FinanceAgentBench-style checklist rules:\n"
        "- First derive 2-8 atomic answer_criteria from the question and reference answer only. Do not derive criteria from the model answer or trajectory.\n"
        "- Criteria should distinguish final answer slots from support/clue/source-basis facts. Riddle clues and source-basis constraints can be criteria for trajectory support, but their wording does not need to appear in a short final answer.\n"
        "- Required final answer slots include entities, dates/periods, values, units, directionality, list membership, and requested calculation outputs. Source/as-of basis and clue facts should usually affect trajectory_supported, evidence_coverage, and task_contract_alignment rather than answer_matched when the answer atom is equivalent to the reference.\n"
        "- For each criterion, set answer_matched based on the final answer and trajectory_supported based on whether the trajectory has evidence/reasoning for that criterion.\n"
        "- A mismatch or missing item is not automatically a contradiction. Mark contradiction.detected=true only for a confident factual conflict with the reference or with the claimed supporting evidence.\n\n"
        "Use this JSON schema exactly:\n"
        "{\n"
        '  "answer_judgement": "correct | partial | incorrect | unassessable",\n'
        '  "answer_score": 1.0,\n'
        '  "trajectory_judgement": "sound | partially_sound | unsound | no_trajectory | unassessable",\n'
        '  "trajectory_score": 1.0,\n'
        '  "task_contract_alignment": "complete | partial | missed | unclear",\n'
        '  "evidence_coverage": "complete | partial | missing | contradictory",\n'
        '  "source_quality": "strong | adequate | weak | absent",\n'
        '  "retrieval_completeness": "sufficient | partial | insufficient | not_needed",\n'
        '  "derivation_quality": "well_derived | partially_derived | asserted | contradictory",\n'
        '  "reasoning_consistency": "consistent | minor_gap | major_gap | contradictory",\n'
        '  "calculation_unit_time_check": "passed | minor_issue | failed | not_applicable",\n'
        '  "answer_criteria": [{"index": 0, "criterion": "required answer slot", "answer_matched": true, "trajectory_supported": true, "supporting_step_indexes": [0], "rationale": "short"}],\n'
        '  "contradiction": {"detected": false, "scope": "answer | trajectory | both | unclear", "rationale": "short"},\n'
        '  "evidence_anchors": [{"step_index": 0, "supports": "short claim/value supported by this step"}],\n'
        '  "calculation_audit": {"required": true, "shown": true, "consistent": true, "issues": ["short issue strings"]},\n'
        '  "temporal_audit": {"required": true, "satisfied": true, "issues": ["short issue strings"]},\n'
        '  "unit_metric_audit": {"required": true, "satisfied": true, "issues": ["short issue strings"]},\n'
        '  "critical_issues": ["short issue strings"],\n'
        '  "unsupported_claims": ["short unsupported claim strings"],\n'
        '  "answer_rationale": "short explanation",\n'
        '  "trajectory_rationale": "short explanation",\n'
        '  "extracted_prediction": "normalized final answer",\n'
        '  "extracted_reference": "normalized reference answer"\n'
        "}\n\n"
        "Score guidance:\n"
        "- answer_score: 1.0 correct, 0.5 partial, 0.0 incorrect/unassessable.\n"
        "- trajectory_score: 1.0 fully supported, 0.75 mostly supported with minor gaps, 0.5 partial support, 0.25 weak/fragile support, 0.0 absent/contradictory/unsound.\n"
        "- answer_score should not exceed the fraction of answer_criteria with answer_matched=true unless a criterion is inapplicable and explained.\n"
        "- trajectory_score should not exceed the fraction of answer_criteria with trajectory_supported=true unless a criterion genuinely needs no trajectory evidence and is explained.\n"
        "- If answer_score is high but evidence_anchors are absent or irrelevant, trajectory_score must be low.\n"
        "- If the trajectory uses sources outside the requested/as-of period and this changes the answer, trajectory_score should be low.\n"
        "- Be strict about trajectory support: the evidence chain must justify the answer, not merely look busy."
    )

    _model_cache: Any = None
    _model_loaded: bool = False

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        prediction = self._recover_prediction(task_log)
        reference = self._golden(task_log)
        question = str(task_log.get("question") or _bench_extra(task_log).get("question") or "")

        if not str(reference or "").strip():
            return {
                "evaluator": self.name,
                "protocol": self.PROTOCOL,
                "judgement": None,
                "score": None,
                "is_correct": None,
                "mode": "missing_reference",
                "note": "Missing reference answer.",
            }

        if not str(prediction or "").strip():
            return self._empty_prediction_result(task_log, reference)

        model = self._ensure_model()
        if model is None:
            return {
                "evaluator": self.name,
                "protocol": self.PROTOCOL,
                "judgement": None,
                "score": None,
                "is_correct": None,
                "mode": "judge_unavailable",
                "note": (
                    "LLM judge unavailable: configure FIRE_AGENT_* or "
                    "FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_MODEL to enable scoring."
                ),
            }

        trajectory_payload, trajectory_meta = self._trajectory_payload(task_log)
        payload = {
            "instructions": self.USER_INSTRUCTIONS,
            "question": question,
            "reference_answer": reference,
            "model_final_answer": prediction,
            "trajectory": trajectory_payload,
            "trajectory_meta": trajectory_meta,
            "task_metadata": {
                key: value
                for key, value in (_bench_extra(task_log) or {}).items()
                if key
                in {
                    "source_dataset",
                    "difficulty",
                    "finance_subdomain",
                    "answer_type",
                    "num_required_hops",
                    "synthetic_method",
                    "dataset_output_constraint",
                }
            },
        }
        llm_result = self._llm_judge(model, payload)
        if llm_result is None:
            return {
                "evaluator": self.name,
                "protocol": self.PROTOCOL,
                "judgement": None,
                "score": None,
                "is_correct": None,
                "mode": "judge_failed",
                "note": "Judge model call failed or returned no usable response.",
                "trajectory_meta": trajectory_meta,
            }

        normalized = self._normalize_judge_result(
            llm_result,
            question=question,
            reference=str(reference or ""),
            prediction=prediction,
        )
        normalized["trajectory_meta"] = trajectory_meta
        return {
            "evaluator": self.name,
            "protocol": self.PROTOCOL,
            "mode": "llm",
            **normalized,
        }

    def _llm_judge(self, model: Any, payload: JsonRow) -> Optional[JsonRow]:
        try:
            max_retries = int(os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_RETRIES", "2"))
        except Exception:
            max_retries = 2

        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = model(
                    [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                    ]
                )
                text = getattr(response, "content", None) or str(response)
                parsed = parse_json_object(text)
                if parsed:
                    parsed["_judge_raw"] = text
                    return parsed
                return {"_judge_raw": text, "_parse_error": "Judge response did not contain a JSON object."}
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                time.sleep(min(2**attempt, 5))
        if last_error is not None:
            return {
                "_judge_error": f"{type(last_error).__name__}: {last_error}",
                "_judge_raw": "",
            }
        return None

    @classmethod
    def _ensure_model(cls) -> Any:
        if cls._model_loaded:
            return cls._model_cache
        cls._model_loaded = True
        try:
            from ..config import ModelSettings
            from ..llm import OpenAICompatibleChatModel
        except Exception:
            cls._model_cache = None
            return None

        override = (
            os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_MODEL")
            or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_MODEL")
            or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_MODEL")
            or os.getenv("FIRE_AGENT_JUDGE_MODEL")
        )
        base_settings = ModelSettings.from_env(model_override=override) if override else ModelSettings.from_env()
        settings = ModelSettings(
            model=override or base_settings.model,
            api_key=(
                os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_API_KEY")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_API_KEY")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_API_KEY")
                or os.getenv("FIRE_AGENT_JUDGE_API_KEY")
                or base_settings.api_key
            ),
            base_url=(
                os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_BASE_URL")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_BASE_URL")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_BASE_URL")
                or os.getenv("FIRE_AGENT_JUDGE_BASE_URL")
                or base_settings.base_url
            ),
            timeout_seconds=int(
                os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_TIMEOUT_SECONDS")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_TIMEOUT_SECONDS")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_TIMEOUT_SECONDS")
                or os.getenv("FIRE_AGENT_JUDGE_TIMEOUT_SECONDS")
                or base_settings.timeout_seconds
            ),
            max_tokens=int(
                os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_MAX_TOKENS")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_MAX_TOKENS")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_MAX_TOKENS")
                or os.getenv("FIRE_AGENT_JUDGE_MAX_TOKENS")
                or base_settings.max_tokens
            ),
            temperature=float(
                os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_JUDGE_TEMPERATURE")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_SFT_JUDGE_TEMPERATURE")
                or os.getenv("FIRE_AGENT_FINDEEPSEARCH_JUDGE_TEMPERATURE")
                or os.getenv("FIRE_AGENT_JUDGE_TEMPERATURE")
                or base_settings.temperature
            ),
            thinking_enabled=base_settings.thinking_enabled,
            reasoning_effort=base_settings.reasoning_effort,
        )
        if not settings.is_configured:
            cls._model_cache = None
            return None
        try:
            cls._model_cache = OpenAICompatibleChatModel(settings)
        except Exception:
            cls._model_cache = None
        return cls._model_cache

    def _normalize_judge_result(
        self,
        parsed: JsonRow,
        *,
        question: str = "",
        reference: str = "",
        prediction: str = "",
    ) -> JsonRow:
        answer_score = self._clamp_score(parsed.get("answer_score"))
        trajectory_score = self._clamp_score(parsed.get("trajectory_score"))

        answer_judgement = self._choice(
            parsed.get("answer_judgement"),
            {"correct", "partial", "incorrect", "unassessable"},
            default=self._judgement_from_answer_score(answer_score),
        )
        trajectory_judgement = self._choice(
            parsed.get("trajectory_judgement"),
            {"sound", "partially_sound", "unsound", "no_trajectory", "unassessable"},
            default=self._judgement_from_trajectory_score(trajectory_score),
        )
        answer_criteria = self._criteria_list(parsed.get("answer_criteria"))
        criteria_total = len(answer_criteria)
        criteria_answer_hits = sum(1 for item in answer_criteria if item.get("answer_matched"))
        criteria_trajectory_hits = sum(1 for item in answer_criteria if item.get("trajectory_supported"))
        contradiction = self._contradiction_block(parsed.get("contradiction"))
        contradiction_detected = bool(contradiction.get("detected"))
        evidence_anchors = self._anchor_list(parsed.get("evidence_anchors"))
        structural_issues: List[str] = []
        answer_atom_adjustment = self._answer_atom_equivalence_adjustment(
            question=question,
            reference=reference,
            prediction=prediction,
            parsed=parsed,
        )

        if criteria_total:
            answer_coverage = criteria_answer_hits / criteria_total
            trajectory_coverage = criteria_trajectory_hits / criteria_total
            if answer_score > answer_coverage:
                answer_score = answer_coverage
                structural_issues.append(
                    "Capped answer_score by derived answer_criteria coverage."
                )
            if trajectory_score > trajectory_coverage:
                trajectory_score = trajectory_coverage
                structural_issues.append(
                    "Capped trajectory_score by derived answer_criteria support coverage."
                )
        elif answer_score >= 0.8 or trajectory_score >= 0.7:
            answer_score = min(answer_score, 0.5)
            trajectory_score = min(trajectory_score, 0.5)
            structural_issues.append(
                "Judge did not return answer_criteria; capped high scores because checklist calibration was unavailable."
            )

        if trajectory_score > 0.5 and not evidence_anchors:
            trajectory_score = 0.5
            structural_issues.append(
                "Judge did not identify any supporting trajectory step_index values; capped trajectory_score at 0.5."
            )
            if trajectory_judgement == "sound":
                trajectory_judgement = "partially_sound"

        if answer_atom_adjustment:
            if answer_score < 0.8 or answer_judgement != "correct":
                answer_score = 1.0
                answer_judgement = "correct"
                structural_issues.append(
                    "Accepted leading answer atom by generic equivalence rule: "
                    + answer_atom_adjustment["reason"]
                )
            if contradiction_detected and self._is_answer_specificity_contradiction(
                contradiction, answer_atom_adjustment
            ):
                contradiction = {
                    "detected": False,
                    "scope": "unclear",
                    "rationale": (
                        "Cleared answer-scope contradiction because the leading answer atom is an "
                        "accepted equivalent; unsupported details remain trajectory/output issues."
                    ),
                }
                contradiction_detected = False
                structural_issues.append(
                    "Cleared specificity-only answer contradiction after answer-atom equivalence check."
                )

        if answer_score < 0.8 and answer_judgement == "correct":
            answer_judgement = self._judgement_from_answer_score(answer_score)
        if trajectory_score < 0.7 and trajectory_judgement in {"sound", "partially_sound"}:
            trajectory_judgement = self._judgement_from_trajectory_score(trajectory_score)

        contradiction_scope = str(contradiction.get("scope") or "unclear")
        answer_blocked = contradiction_detected and contradiction_scope in {"answer", "both", "unclear"}
        trajectory_blocked = contradiction_detected and contradiction_scope in {"trajectory", "both", "unclear"}
        answer_correct = answer_score >= 0.8 and answer_judgement == "correct" and not answer_blocked
        trajectory_sound = (
            trajectory_score >= 0.7
            and trajectory_judgement in {"sound", "partially_sound"}
            and not trajectory_blocked
        )
        is_correct = bool(answer_correct and trajectory_sound)
        overall_score = self._overall_score(answer_score, trajectory_score)

        note = str(parsed.get("_parse_error") or parsed.get("_judge_error") or "").strip()
        judge_raw = str(parsed.get("_judge_raw") or "")
        result: JsonRow = {
            "judgement": "correct" if is_correct else "incorrect",
            "score": overall_score,
            "is_correct": is_correct,
            "answer_judgement": answer_judgement,
            "answer_score": answer_score,
            "answer_correct": answer_correct,
            "trajectory_judgement": trajectory_judgement,
            "trajectory_score": trajectory_score,
            "trajectory_sound": trajectory_sound,
            "task_contract_alignment": self._choice(
                parsed.get("task_contract_alignment"),
                {"complete", "partial", "missed", "unclear"},
                default="unclear",
            ),
            "evidence_coverage": self._choice(
                parsed.get("evidence_coverage"),
                {"complete", "partial", "missing", "contradictory"},
                default="partial",
            ),
            "source_quality": self._choice(
                parsed.get("source_quality"),
                {"strong", "adequate", "weak", "absent"},
                default="weak",
            ),
            "retrieval_completeness": self._choice(
                parsed.get("retrieval_completeness"),
                {"sufficient", "partial", "insufficient", "not_needed"},
                default="partial",
            ),
            "derivation_quality": self._choice(
                parsed.get("derivation_quality"),
                {"well_derived", "partially_derived", "asserted", "contradictory"},
                default="partially_derived",
            ),
            "reasoning_consistency": self._choice(
                parsed.get("reasoning_consistency"),
                {"consistent", "minor_gap", "major_gap", "contradictory"},
                default="major_gap",
            ),
            "calculation_unit_time_check": self._choice(
                parsed.get("calculation_unit_time_check"),
                {"passed", "minor_issue", "failed", "not_applicable"},
                default="not_applicable",
            ),
            "answer_criteria": answer_criteria,
            "criteria_total": criteria_total,
            "criteria_answer_hits": criteria_answer_hits,
            "criteria_trajectory_hits": criteria_trajectory_hits,
            "criteria_answer_coverage": (
                criteria_answer_hits / criteria_total if criteria_total else None
            ),
            "criteria_trajectory_coverage": (
                criteria_trajectory_hits / criteria_total if criteria_total else None
            ),
            "contradiction": contradiction,
            "evidence_anchors": evidence_anchors,
            "calculation_audit": self._audit_block(parsed.get("calculation_audit")),
            "temporal_audit": self._audit_block(parsed.get("temporal_audit")),
            "unit_metric_audit": self._audit_block(parsed.get("unit_metric_audit")),
            "critical_issues": self._string_list(parsed.get("critical_issues")) + structural_issues,
            "unsupported_claims": self._string_list(parsed.get("unsupported_claims")),
            "answer_rationale": str(parsed.get("answer_rationale") or ""),
            "trajectory_rationale": str(parsed.get("trajectory_rationale") or ""),
            "extracted_prediction": str(parsed.get("extracted_prediction") or ""),
            "extracted_reference": str(parsed.get("extracted_reference") or ""),
            "judge_raw": compact_text(judge_raw, 6000),
        }
        if answer_atom_adjustment:
            result["answer_atom_equivalence"] = answer_atom_adjustment
        if note:
            result["note"] = note
            result["needs_official_review"] = True
        return result

    @staticmethod
    def _overall_score(answer_score: float, trajectory_score: float) -> float:
        # Weighted score is useful for ranking near-misses, but correctness is
        # still the strict two-gate decision above.
        return round((0.65 * answer_score) + (0.35 * trajectory_score), 4)

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return float(score)

    @staticmethod
    def _choice(value: Any, allowed: set[str], *, default: str) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return text if text in allowed else default

    @staticmethod
    def _judgement_from_answer_score(score: float) -> str:
        if score >= 0.8:
            return "correct"
        if score >= 0.25:
            return "partial"
        return "incorrect"

    @staticmethod
    def _judgement_from_trajectory_score(score: float) -> str:
        if score >= 0.9:
            return "sound"
        if score >= 0.5:
            return "partially_sound"
        if score <= 0:
            return "no_trajectory"
        return "unsound"

    @classmethod
    def _answer_atom_equivalence_adjustment(
        cls,
        *,
        question: str,
        reference: str,
        prediction: str,
        parsed: JsonRow,
    ) -> Optional[JsonRow]:
        ref_atom = str(parsed.get("extracted_reference") or reference or "").strip()
        raw_prediction_atom = str(parsed.get("extracted_prediction") or "").strip()
        prediction_atoms = [
            raw_prediction_atom,
            cls._leading_answer_atom(raw_prediction_atom),
            cls._leading_answer_atom(prediction),
        ]
        prediction_atoms = [atom for atom in dict.fromkeys(prediction_atoms) if atom]
        if not ref_atom or not prediction_atoms:
            return None

        ref_norm = cls._normalize_answer_atom(ref_atom)
        for pred_atom in prediction_atoms:
            pred_norm = cls._normalize_answer_atom(pred_atom)
            if ref_norm and pred_norm and ref_norm == pred_norm:
                return {
                    "kind": "normalized_atom_match",
                    "reference_atom": ref_atom,
                    "prediction_atom": pred_atom,
                    "reason": "normalized leading answer atom matches the reference",
                }

            if cls._hierarchical_code_prefix_match(question, ref_atom, pred_atom):
                return {
                    "kind": "hierarchical_code_prefix",
                    "reference_atom": ref_atom,
                    "prediction_atom": pred_atom,
                    "reason": "prediction extends the reference as a hierarchical code/subcode for a prefix-style question",
                }

            if cls._specific_variant_contains_reference(ref_atom, pred_atom):
                return {
                    "kind": "specific_variant_contains_reference",
                    "reference_atom": ref_atom,
                    "prediction_atom": pred_atom,
                    "reason": "prediction is a specific variant containing the reference acronym/atom",
                }

        return None

    @staticmethod
    def _leading_answer_atom(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        value = re.sub(r"(?is)^.*?(?:final answer|answer)\s*[:：]\s*", "", value, count=1)
        value = value.strip(" \t\r\n*-_#")
        first_line = next((line.strip() for line in value.splitlines() if line.strip()), value)
        first_line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", first_line).strip()
        first_line = re.split(r"\s+(?:--|-|\u2013|\u2014)\s+|\s+\(|[.;]\s+", first_line, maxsplit=1)[0]
        return first_line.strip(" \t\r\n*:：`\"'")

    @staticmethod
    def _normalize_answer_atom(text: str) -> str:
        value = str(text or "").strip().lower()
        value = re.sub(r"(?i)\b(?:final answer|answer|the answer is)\b", " ", value)
        value = re.sub(r"\*\*|__|`", "", value)
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" \t\r\n:：,.;")
        return value

    @classmethod
    def _hierarchical_code_prefix_match(cls, question: str, reference: str, prediction: str) -> bool:
        q = cls._question_contract_text(question).lower()
        if not re.search(r"\b(prefix|code prefix|classification code|tariff|hs code|heading)\b", q):
            return False
        if re.search(r"\b(exact|only|4-digit|four-digit|six-digit|6-digit|8-digit|eight-digit)\b", q):
            return False
        ref_code = cls._first_code_token(reference)
        pred_code = cls._first_code_token(prediction)
        if not ref_code or not pred_code:
            return False
        ref_digits = re.sub(r"\D", "", ref_code)
        pred_digits = re.sub(r"\D", "", pred_code)
        if len(ref_digits) < 3 or len(pred_digits) <= len(ref_digits):
            return False
        return pred_digits.startswith(ref_digits)

    @staticmethod
    def _question_contract_text(question: str) -> str:
        text = str(question or "")
        for marker in (
            "\n\nDataset Output Constraint:",
            "\n\nReference URLs:",
            "\n\nSeed URL:",
            "\nDataset Output Constraint:",
            "\nReference URLs:",
            "\nSeed URL:",
        ):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text

    @staticmethod
    def _first_code_token(text: str) -> str:
        match = re.search(r"\b\d{3,10}(?:[.\-]\d{1,8})*\b", str(text or ""))
        return match.group(0) if match else ""

    @staticmethod
    def _specific_variant_contains_reference(reference: str, prediction: str) -> bool:
        ref = str(reference or "").strip()
        pred = str(prediction or "").strip()
        ref_compact = re.sub(r"[^A-Za-z0-9]", "", ref)
        if not re.search(r"[A-Za-z]", ref_compact):
            return False
        if len(ref_compact) < 3 or ref_compact.upper() != ref_compact:
            return False
        if len(pred) > 120:
            return False
        if re.search(r"(?i)\b(?:not|except|excluding|without)\s+" + re.escape(ref) + r"\b", pred):
            return False
        return re.search(r"(?i)(?<![A-Za-z0-9])" + re.escape(ref) + r"(?![A-Za-z0-9])", pred) is not None

    @staticmethod
    def _is_answer_specificity_contradiction(contradiction: JsonRow, adjustment: JsonRow) -> bool:
        scope = str(contradiction.get("scope") or "unclear")
        if scope not in {"answer", "both", "unclear"}:
            return False
        rationale = str(contradiction.get("rationale") or "").lower()
        if adjustment.get("kind") == "hierarchical_code_prefix":
            return any(term in rationale for term in ["subheading", "prefix", "more specific", "digit"])
        if adjustment.get("kind") == "specific_variant_contains_reference":
            return any(term in rationale for term in ["more specific", "variant", "not present", "not supported"])
        return False

    @staticmethod
    def _string_list(value: Any, *, limit: int = 8) -> List[str]:
        if isinstance(value, list):
            return [compact_text(str(item), 300) for item in value[:limit] if str(item or "").strip()]
        if value in (None, ""):
            return []
        return [compact_text(str(value), 300)]

    @classmethod
    def _anchor_list(cls, value: Any, *, limit: int = 12) -> List[JsonRow]:
        anchors: List[JsonRow] = []
        if not isinstance(value, list):
            return anchors
        for item in value[:limit]:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("step_index")
            try:
                step_index: Optional[int] = int(raw_index)
            except Exception:
                step_index = None
            if step_index is None or step_index < 0:
                continue
            supports = compact_text(str(item.get("supports") or item.get("claim") or ""), 400)
            if not supports:
                continue
            anchors.append({"step_index": step_index, "supports": supports})
        return anchors

    @classmethod
    def _criteria_list(cls, value: Any, *, limit: int = 10) -> List[JsonRow]:
        criteria: List[JsonRow] = []
        if not isinstance(value, list):
            return criteria
        for pos, item in enumerate(value[:limit]):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except Exception:
                index = pos
            criterion = compact_text(str(item.get("criterion") or item.get("criteria") or ""), 500)
            if not criterion:
                continue
            raw_step_indexes = item.get("supporting_step_indexes")
            supporting_step_indexes: List[int] = []
            if isinstance(raw_step_indexes, list):
                for raw_index in raw_step_indexes[:12]:
                    try:
                        step_index = int(raw_index)
                    except Exception:
                        continue
                    if step_index >= 0 and step_index not in supporting_step_indexes:
                        supporting_step_indexes.append(step_index)
            criteria.append(
                {
                    "index": index,
                    "criterion": criterion,
                    "answer_matched": cls._bool_or_none(item.get("answer_matched")) is True,
                    "trajectory_supported": cls._bool_or_none(item.get("trajectory_supported")) is True,
                    "supporting_step_indexes": supporting_step_indexes,
                    "rationale": compact_text(str(item.get("rationale") or ""), 400),
                }
            )
        return criteria

    @classmethod
    def _contradiction_block(cls, value: Any) -> JsonRow:
        if not isinstance(value, dict):
            return {"detected": False, "scope": "unclear", "rationale": ""}
        scope = cls._choice(
            value.get("scope"),
            {"answer", "trajectory", "both", "unclear"},
            default="unclear",
        )
        return {
            "detected": cls._bool_or_none(value.get("detected")) is True,
            "scope": scope,
            "rationale": compact_text(str(value.get("rationale") or ""), 500),
        }

    @classmethod
    def _audit_block(cls, value: Any) -> JsonRow:
        if not isinstance(value, dict):
            return {"required": None, "satisfied": None, "shown": None, "consistent": None, "issues": []}
        return {
            "required": cls._bool_or_none(value.get("required")),
            "satisfied": cls._bool_or_none(value.get("satisfied")),
            "shown": cls._bool_or_none(value.get("shown")),
            "consistent": cls._bool_or_none(value.get("consistent")),
            "issues": cls._string_list(value.get("issues"), limit=6),
        }

    @staticmethod
    def _bool_or_none(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y", "passed", "satisfied", "shown", "consistent"}:
            return True
        if lowered in {"0", "false", "no", "n", "failed", "missing", "inconsistent"}:
            return False
        return None

    def _empty_prediction_result(self, task_log: JsonRow, reference: Any) -> JsonRow:
        trajectory_payload, trajectory_meta = self._trajectory_payload(task_log)
        has_trajectory = bool(trajectory_payload)
        return {
            "evaluator": self.name,
            "protocol": self.PROTOCOL,
            "mode": "deterministic_empty_prediction",
            "judgement": "incorrect",
            "score": 0.0,
            "is_correct": False,
            "answer_judgement": "incorrect",
            "answer_score": 0.0,
            "answer_correct": False,
            "trajectory_judgement": "unsound" if has_trajectory else "no_trajectory",
            "trajectory_score": 0.0,
            "trajectory_sound": False,
            "evidence_coverage": "missing",
            "source_quality": "absent",
            "task_contract_alignment": "missed",
            "retrieval_completeness": "insufficient",
            "derivation_quality": "asserted",
            "reasoning_consistency": "major_gap",
            "calculation_unit_time_check": "not_applicable",
            "answer_criteria": [],
            "criteria_total": 0,
            "criteria_answer_hits": 0,
            "criteria_trajectory_hits": 0,
            "criteria_answer_coverage": None,
            "criteria_trajectory_coverage": None,
            "contradiction": {"detected": False, "scope": "unclear", "rationale": ""},
            "evidence_anchors": [],
            "calculation_audit": {"required": None, "satisfied": None, "shown": None, "consistent": None, "issues": []},
            "temporal_audit": {"required": None, "satisfied": None, "shown": None, "consistent": None, "issues": []},
            "unit_metric_audit": {"required": None, "satisfied": None, "shown": None, "consistent": None, "issues": []},
            "critical_issues": ["Model final answer is empty."],
            "unsupported_claims": [],
            "answer_rationale": "模型没有给出最终答案。",
            "trajectory_rationale": "没有可评分的最终答案；轨迹即使存在也不能支撑空答案。",
            "extracted_prediction": "",
            "extracted_reference": str(reference or ""),
            "trajectory_meta": trajectory_meta,
        }

    def _recover_prediction(self, task_log: JsonRow) -> str:
        prediction = str(self._prediction(task_log) or "").strip()
        if prediction:
            return prediction
        for step in reversed(task_log.get("agent_trajectory") or []):
            if not isinstance(step, dict) or step.get("name") != "action":
                continue
            for call in step.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                if str(call.get("name") or "") != "final_answer":
                    continue
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                value = args.get("answer") or args.get("final_answer") or args.get("result") or args.get("text")
                if value not in (None, ""):
                    return str(value).strip()
        return ""

    def _trajectory_payload(self, task_log: JsonRow) -> Tuple[List[JsonRow], JsonRow]:
        raw_steps = task_log.get("agent_trajectory") or []
        if not isinstance(raw_steps, list):
            raw_steps = []
        try:
            max_steps = int(os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_TRAJECTORY_MAX_STEPS", "30"))
        except Exception:
            max_steps = 30
        try:
            total_limit = int(os.getenv("FIRE_AGENT_FIN_DEEPSEARCH_SFT_TRAJECTORY_CHAR_LIMIT", "36000"))
        except Exception:
            total_limit = 36000

        selected = list(self._select_steps(raw_steps, max_steps=max(1, max_steps)))
        compacted: List[JsonRow] = []
        used = 0
        truncated_for_budget = False
        for original_index, step in selected:
            item = self._compact_step(original_index, step)
            encoded = json.dumps(item, ensure_ascii=False, default=str)
            if used + len(encoded) > total_limit:
                item = self._compact_step(original_index, step, tight=True)
                encoded = json.dumps(item, ensure_ascii=False, default=str)
            if used + len(encoded) > total_limit:
                truncated_for_budget = True
                break
            compacted.append(item)
            used += len(encoded)

        stats = task_log.get("stats") if isinstance(task_log.get("stats"), dict) else {}
        return compacted, {
            "raw_step_count": len(raw_steps),
            "included_step_count": len(compacted),
            "step_selection_truncated": len(compacted) < len(raw_steps),
            "step_selection_strategy": "all" if len(raw_steps) <= max_steps else "head_tail",
            "char_budget": total_limit,
            "char_budget_truncated": truncated_for_budget,
            "status": task_log.get("status"),
            "error": task_log.get("error"),
            "stats": {
                "steps": stats.get("steps"),
                "tool_calls": stats.get("tool_calls"),
                "tools_by_name": stats.get("tools_by_name"),
            },
        }

    @staticmethod
    def _select_steps(steps: List[Any], *, max_steps: int) -> Iterable[Tuple[int, Any]]:
        if len(steps) <= max_steps:
            for idx, step in enumerate(steps):
                yield idx, step
            return
        head_count = min(len(steps), max(1, min(max_steps, max(4, max_steps // 3))))
        tail_count = max(0, max_steps - head_count)
        selected = list(range(head_count))
        selected.extend(range(max(0, len(steps) - tail_count), len(steps)))
        for idx in dict.fromkeys(index for index in selected if 0 <= index < len(steps)):
            yield idx, steps[idx]

    def _compact_step(self, original_index: int, step: Any, *, tight: bool = False) -> JsonRow:
        if not isinstance(step, dict):
            return {"step_index": original_index, "raw": compact_text(str(step), 400 if tight else 900)}

        if step.get("name") == "plan":
            return {
                "step_index": original_index,
                "name": "plan",
                "value": compact_text(str(step.get("value") or ""), 500 if tight else 1400),
                "think": compact_text(str(step.get("think") or ""), 250 if tight else 700),
                **({"error": compact_text(str(step.get("error")), 300)} if step.get("error") else {}),
            }

        tool_calls = []
        for call in (step.get("tool_calls") or [])[:8]:
            if not isinstance(call, dict):
                continue
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            tool_calls.append(
                {
                    "name": call.get("name"),
                    "action": call.get("action"),
                    "arguments": compact_text(json.dumps(args, ensure_ascii=False, default=str), 250 if tight else 700),
                }
            )
        return {
            "step_index": original_index,
            "name": str(step.get("name") or "action"),
            "tool_calls": tool_calls,
            "think": compact_text(str(step.get("think") or ""), 300 if tight else 900),
            "obs": compact_text(str(step.get("obs") or ""), 500 if tight else 1800),
            **({"error": compact_text(str(step.get("error")), 300 if tight else 600)} if step.get("error") else {}),
        }
