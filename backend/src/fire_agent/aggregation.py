from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def normalize_correctness(score_block: Dict[str, Any]) -> Optional[bool]:
    for key in ("is_correct", "correct"):
        if key in score_block and score_block[key] is not None:
            return bool(score_block[key])
    judgement = score_block.get("judgement")
    if isinstance(judgement, str):
        lowered = judgement.lower()
        if lowered in {"correct", "true", "yes"}:
            return True
        if lowered in {"incorrect", "false", "no", "error"}:
            return False
    score = score_block.get("score")
    if score is not None:
        try:
            return float(score) > 0
        except Exception:
            return None
    return None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _row_score(row: Dict[str, Any]) -> Dict[str, Any]:
    score = row.get("score")
    if isinstance(score, dict):
        return score
    # Legacy fallback for pre-refactor JSONLs.
    legacy = row.get("metrics") or row.get("judge_output") or {}
    return legacy if isinstance(legacy, dict) else {}


def _row_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    stats = row.get("stats")
    if isinstance(stats, dict):
        return stats
    tools_by_name: Dict[str, int] = {}
    tool_call_count = 0
    trajectory = row.get("agent_trajectory") or []
    if isinstance(trajectory, list):
        for step in trajectory:
            if not isinstance(step, dict) or step.get("name") != "action":
                continue
            for call in step.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                if name in {"", "final_answer"}:
                    continue
                tool_call_count += 1
                tools_by_name[name] = tools_by_name.get(name, 0) + 1
    else:
        trajectory = []
    token_stats = row.get("token_stats") or {}
    if not isinstance(token_stats, dict):
        token_stats = {}
    return {
        "steps": len(trajectory),
        "tool_calls": tool_call_count,
        "tools_by_name": tools_by_name,
        "tokens": {
            "prompt": token_stats.get("prompt_tokens") or token_stats.get("prompt"),
            "completion": token_stats.get("completion_tokens") or token_stats.get("completion"),
            "total": token_stats.get("total_tokens") or token_stats.get("total"),
            "llm_calls": token_stats.get("llm_calls"),
        },
    }


def aggregate_token_stats(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(logs)
    if not rows:
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_llm_calls": 0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "avg_llm_calls": 0.0,
        }
    prompt = 0
    completion = 0
    total = 0
    calls = 0
    for row in rows:
        tokens = (_row_stats(row).get("tokens") or {})
        prompt += _safe_int(tokens.get("prompt"))
        completion += _safe_int(tokens.get("completion"))
        total += _safe_int(tokens.get("total"))
        calls += _safe_int(tokens.get("llm_calls"))
    n = len(rows)
    return {
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_tokens": total,
        "total_llm_calls": calls,
        "avg_prompt_tokens": prompt / n,
        "avg_completion_tokens": completion / n,
        "avg_total_tokens": total / n,
        "avg_llm_calls": calls / n,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_task_logs(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = list(logs)
    correct_values = [normalize_correctness(_row_score(row)) for row in rows]
    judged = [value for value in correct_values if value is not None]
    total_tool_calls = sum(_safe_int(_row_stats(row).get("tool_calls")) for row in rows)
    total_steps = sum(_safe_int(_row_stats(row).get("steps")) for row in rows)
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    judge_scores: List[float] = []
    for row in rows:
        score_block = _row_score(row)
        score = _safe_float(score_block.get("score"))
        if score is not None:
            judge_scores.append(score)
    return {
        "total": len(rows),
        "judged": len(judged),
        "correct": sum(1 for value in judged if value),
        "accuracy": (sum(1 for value in judged if value) / len(judged)) if judged else None,
        "avg_judge_score": (sum(judge_scores) / len(judge_scores)) if judge_scores else None,
        "judge_score_samples": len(judge_scores),
        "avg_tool_calls": (total_tool_calls / len(rows)) if rows else 0,
        "avg_steps": (total_steps / len(rows)) if rows else 0,
        "status_counts": status_counts,
        "token_stats": aggregate_token_stats(rows),
    }


def aggregate_bizfinbench_v2_logs(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate BizFinBench.v2 using the official average-score accuracy.

    BizFinBench official evaluators can award partial credit, so the leaderboard
    style accuracy is the mean numeric score, not the strict exact-match rate.
    Keep the strict count/rate under explicit fields for debugging.
    """

    rows: List[Dict[str, Any]] = list(logs)
    summary = aggregate_task_logs(rows)
    strict_correct = summary.get("correct")
    strict_accuracy = summary.get("accuracy")
    judge_scores: List[float] = []
    for row in rows:
        score = _safe_float(_row_score(row).get("score"))
        if score is not None:
            judge_scores.append(score)
    if judge_scores:
        official_score_sum = sum(judge_scores)
        official_accuracy = official_score_sum / len(judge_scores)
        summary.update(
            {
                "accuracy": official_accuracy,
                "accuracy_pct": 100 * official_accuracy,
                "correct": official_score_sum,
                "strict_correct": strict_correct,
                "strict_accuracy": strict_accuracy,
                "strict_accuracy_pct": (100 * strict_accuracy) if strict_accuracy is not None else None,
                "official_score_sum": official_score_sum,
                "accuracy_source": "bizfinbench_v2_official_avg_score",
            }
        )
    return summary


RFC_TASK2_TYPES = ("flipping", "numerical", "sentiment", "causal")


def _metric_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _multiclass_mcc(confusion: Dict[str, Dict[str, int]], labels: Iterable[str]) -> Optional[float]:
    labels = list(labels)
    total = sum(confusion[gold][pred] for gold in labels for pred in labels)
    if total == 0:
        return None
    correct = sum(confusion[label][label] for label in labels)
    true_totals = {
        label: sum(confusion[label][pred] for pred in labels)
        for label in labels
    }
    pred_totals = {
        label: sum(confusion[gold][label] for gold in labels)
        for label in labels
    }
    covariance = correct * total - sum(true_totals[label] * pred_totals[label] for label in labels)
    true_variance = total * total - sum(value * value for value in true_totals.values())
    pred_variance = total * total - sum(value * value for value in pred_totals.values())
    denominator = math.sqrt(true_variance * pred_variance)
    return covariance / denominator if denominator else None


def aggregate_rfc_task2_logs(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate RFC-BENCH Task 2 rows using the paper's classification protocol.

    The paper computes classification metrics only over valid predictions that
    map unambiguously to the four Task 2 labels, while invalid outputs are
    counted separately as an output-format reliability signal.
    """

    rows: List[Dict[str, Any]] = list(logs)
    base = aggregate_task_logs(rows)
    labels = list(RFC_TASK2_TYPES)
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }
    total_support = {label: 0 for label in labels}
    valid_support = {label: 0 for label in labels}
    valid_predictions = 0
    invalid_predictions = 0
    missing_or_invalid_gold = 0

    for row in rows:
        score = _row_score(row)
        extra = row.get("bench_extra") if isinstance(row.get("bench_extra"), dict) else {}
        golden = str(
            score.get("golden_type")
            or score.get("golden_label")
            or row.get("golden_answer")
            or extra.get("golden_answer")
            or extra.get("perturbation_type")
            or ""
        ).strip().lower()
        predicted = str(score.get("predicted_type") or score.get("predicted_label") or "").strip().lower()

        if golden not in total_support:
            missing_or_invalid_gold += 1
            continue
        total_support[golden] += 1

        valid_flag = score.get("valid_prediction")
        ambiguous = bool(score.get("ambiguous_prediction"))
        if valid_flag is None:
            valid_flag = predicted in labels and not ambiguous
        if not valid_flag or predicted not in labels:
            invalid_predictions += 1
            continue

        valid_predictions += 1
        valid_support[golden] += 1
        confusion[golden][predicted] += 1

    correct = sum(confusion[label][label] for label in labels)
    per_type: Dict[str, Dict[str, Any]] = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = _metric_divide(tp, tp + fp)
        recall = _metric_divide(tp, tp + fn)
        f1 = _metric_divide(2 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_type[label] = {
            "support_total": total_support[label],
            "support_valid": valid_support[label],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    accuracy = correct / valid_predictions if valid_predictions else None
    macro_precision = sum(precision_values) / len(labels) if labels and valid_predictions else None
    macro_recall = sum(recall_values) / len(labels) if labels and valid_predictions else None
    macro_f1 = sum(f1_values) / len(labels) if labels and valid_predictions else None
    invalid_rate = invalid_predictions / len(rows) if rows else 0.0

    return {
        "benchmark": "rfc_task2",
        "total": len(rows),
        "judged": valid_predictions,
        "valid_predictions": valid_predictions,
        "invalid_predictions": invalid_predictions,
        "invalid_rate": invalid_rate,
        "invalid_rate_pct": invalid_rate * 100,
        "missing_or_invalid_gold": missing_or_invalid_gold,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100 if accuracy is not None else None,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro": macro_f1,
        "mcc": _multiclass_mcc(confusion, labels),
        "auroc": None,
        "auroc_note": "Not computed because this runtime records hard labels, not per-class scores or probabilities.",
        "per_type": per_type,
        "confusion_matrix": confusion,
        "status_counts": base.get("status_counts", {}),
        "avg_tool_calls": base.get("avg_tool_calls", 0),
        "avg_steps": base.get("avg_steps", 0),
        "token_stats": base.get("token_stats", {}),
        "protocol": (
            "RFC-BENCH Task 2 metrics: compute accuracy, macro precision, "
            "macro recall, macro-F1, and multiclass MCC over valid "
            "unambiguous predictions only; report invalid outputs separately."
        ),
    }


def _avg_optional(values: Iterable[Any]) -> Optional[float]:
    numeric: List[float] = []
    for value in values:
        score = _safe_float(value)
        if score is not None:
            numeric.append(score)
    return (sum(numeric) / len(numeric)) if numeric else None


def _truthy_score_flag(score_block: Dict[str, Any], key: str) -> Optional[bool]:
    if key not in score_block:
        return None
    value = score_block.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "correct", "sound", "1"}:
            return True
        if lowered in {"false", "no", "incorrect", "unsound", "0"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def aggregate_fin_deepsearch_sft_logs(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate FinDeepSearchSFT rows with separate answer and trajectory gates."""

    rows: List[Dict[str, Any]] = list(logs)
    base = aggregate_task_logs(rows)
    score_blocks = [_row_score(row) for row in rows]

    answer_judged = [score for score in score_blocks if _safe_float(score.get("answer_score")) is not None]
    trajectory_judged = [score for score in score_blocks if _safe_float(score.get("trajectory_score")) is not None]
    criteria_judged = [
        score for score in score_blocks if _safe_float(score.get("criteria_answer_coverage")) is not None
    ]

    def answer_correct(score: Dict[str, Any]) -> bool:
        explicit = _truthy_score_flag(score, "answer_correct")
        if explicit is not None:
            return explicit
        value = _safe_float(score.get("answer_score"))
        return bool(value is not None and value >= 0.8)

    def trajectory_sound(score: Dict[str, Any]) -> bool:
        explicit = _truthy_score_flag(score, "trajectory_sound")
        if explicit is not None:
            return explicit
        value = _safe_float(score.get("trajectory_score"))
        return bool(value is not None and value >= 0.7)

    modes: Dict[str, int] = {}
    task_contract_alignment: Dict[str, int] = {}
    evidence_coverage: Dict[str, int] = {}
    source_quality: Dict[str, int] = {}
    retrieval_completeness: Dict[str, int] = {}
    derivation_quality: Dict[str, int] = {}
    reasoning_consistency: Dict[str, int] = {}
    contradiction_scopes: Dict[str, int] = {}
    contradiction_detected = 0
    for score in score_blocks:
        for key, bucket in (
            ("mode", modes),
            ("task_contract_alignment", task_contract_alignment),
            ("evidence_coverage", evidence_coverage),
            ("source_quality", source_quality),
            ("retrieval_completeness", retrieval_completeness),
            ("derivation_quality", derivation_quality),
            ("reasoning_consistency", reasoning_consistency),
        ):
            value = str(score.get(key) or "unknown")
            bucket[value] = bucket.get(value, 0) + 1
        contradiction = score.get("contradiction") if isinstance(score.get("contradiction"), dict) else {}
        if contradiction.get("detected"):
            contradiction_detected += 1
        scope = str(contradiction.get("scope") or "unknown")
        contradiction_scopes[scope] = contradiction_scopes.get(scope, 0) + 1

    def summarize_group(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        group_scores = [_row_score(row) for row in group_rows]
        group_base = aggregate_task_logs(group_rows)
        group_answer = [score for score in group_scores if _safe_float(score.get("answer_score")) is not None]
        group_traj = [score for score in group_scores if _safe_float(score.get("trajectory_score")) is not None]
        return {
            "total": len(group_rows),
            "judged": group_base.get("judged"),
            "strict_correct": group_base.get("correct"),
            "strict_accuracy": group_base.get("accuracy"),
            "answer_judged": len(group_answer),
            "answer_correct": sum(1 for score in group_answer if answer_correct(score)),
            "answer_accuracy": (
                sum(1 for score in group_answer if answer_correct(score)) / len(group_answer)
                if group_answer
                else None
            ),
            "trajectory_judged": len(group_traj),
            "trajectory_sound": sum(1 for score in group_traj if trajectory_sound(score)),
            "trajectory_sound_rate": (
                sum(1 for score in group_traj if trajectory_sound(score)) / len(group_traj)
                if group_traj
                else None
            ),
            "avg_answer_score": _avg_optional(score.get("answer_score") for score in group_scores),
            "avg_trajectory_score": _avg_optional(score.get("trajectory_score") for score in group_scores),
            "avg_overall_score": _avg_optional(score.get("score") for score in group_scores),
            "avg_criteria_answer_coverage": _avg_optional(
                score.get("criteria_answer_coverage") for score in group_scores
            ),
            "avg_criteria_trajectory_coverage": _avg_optional(
                score.get("criteria_trajectory_coverage") for score in group_scores
            ),
            "contradiction_detected": sum(
                1
                for score in group_scores
                if isinstance(score.get("contradiction"), dict)
                and score.get("contradiction", {}).get("detected")
            ),
            "avg_evidence_anchors": _avg_optional(
                len(score.get("evidence_anchors") or [])
                if isinstance(score.get("evidence_anchors"), list)
                else 0
                for score in group_scores
            ),
        }

    by_source_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = _metadata_value(row, ["source_dataset"]) or "unknown"
        by_source_dataset[str(source)].append(row)

    return {
        "benchmark": "fin_deepsearch_sft",
        "protocol": (
            "Answer equivalence plus dynamic criteria, contradiction, and "
            "financial-search evidence-chain audit; strict correctness requires "
            "answer_correct, trajectory_sound, and no blocking contradiction."
        ),
        **base,
        "strict_accuracy": base.get("accuracy"),
        "answer_judged": len(answer_judged),
        "answer_correct": sum(1 for score in answer_judged if answer_correct(score)),
        "answer_accuracy": (
            sum(1 for score in answer_judged if answer_correct(score)) / len(answer_judged)
            if answer_judged
            else None
        ),
        "trajectory_judged": len(trajectory_judged),
        "trajectory_sound": sum(1 for score in trajectory_judged if trajectory_sound(score)),
        "trajectory_sound_rate": (
            sum(1 for score in trajectory_judged if trajectory_sound(score)) / len(trajectory_judged)
            if trajectory_judged
            else None
        ),
        "avg_answer_score": _avg_optional(score.get("answer_score") for score in score_blocks),
        "avg_trajectory_score": _avg_optional(score.get("trajectory_score") for score in score_blocks),
        "criteria_judged": len(criteria_judged),
        "avg_criteria_answer_coverage": _avg_optional(
            score.get("criteria_answer_coverage") for score in score_blocks
        ),
        "avg_criteria_trajectory_coverage": _avg_optional(
            score.get("criteria_trajectory_coverage") for score in score_blocks
        ),
        "contradiction_detected": contradiction_detected,
        "contradiction_scopes": contradiction_scopes,
        "avg_evidence_anchors": _avg_optional(
            len(score.get("evidence_anchors") or [])
            if isinstance(score.get("evidence_anchors"), list)
            else 0
            for score in score_blocks
        ),
        "judge_modes": modes,
        "task_contract_alignment": task_contract_alignment,
        "evidence_coverage": evidence_coverage,
        "source_quality": source_quality,
        "retrieval_completeness": retrieval_completeness,
        "derivation_quality": derivation_quality,
        "reasoning_consistency": reasoning_consistency,
        "by_source_dataset": {
            source: summarize_group(group)
            for source, group in sorted(by_source_dataset.items())
        },
    }


def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _metadata_value(row: Dict[str, Any], keys: List[str]) -> Any:
    """Read a metadata field, preferring the new ``bench_extra`` block but
    falling back to the legacy containers for old JSONLs."""
    for container_key in ("bench_extra", "evaluator_metadata", "eval_metadata", "solver_metadata"):
        metadata = row.get(container_key) or {}
        if isinstance(metadata, dict):
            value = _first_present(metadata, keys)
            if value not in (None, ""):
                return value
    return _first_present(row, keys)


def _official_score(row: Dict[str, Any]) -> Optional[int]:
    score_block = _row_score(row)
    for key in ("official_score", "score"):
        if key not in score_block or score_block[key] is None:
            continue
        try:
            value = int(float(score_block[key]))
        except Exception:
            continue
        if value in {-1, 0, 1}:
            return value
    correctness = normalize_correctness(score_block)
    if correctness is None:
        return None
    return 1 if correctness else 0


def aggregate_fingaia_logs(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate FinGAIA logs with the benchmark's official 1/0/-1 protocol."""

    rows: List[Dict[str, Any]] = list(logs)
    scores = [_official_score(row) for row in rows]
    judged_scores = [score for score in scores if score is not None]
    correct = sum(1 for score in judged_scores if score == 1)
    incorrect = sum(1 for score in judged_scores if score == 0)
    unassessable = sum(1 for score in judged_scores if score == -1)
    total = len(rows)
    assessable = correct + incorrect

    def summarize_group(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        group_scores = [_official_score(row) for row in group_rows]
        group_scores = [score for score in group_scores if score is not None]
        group_correct = sum(1 for score in group_scores if score == 1)
        group_incorrect = sum(1 for score in group_scores if score == 0)
        group_unassessable = sum(1 for score in group_scores if score == -1)
        group_total = len(group_rows)
        group_assessable = group_correct + group_incorrect
        return {
            "total": group_total,
            "judged": len(group_scores),
            "correct": group_correct,
            "incorrect": group_incorrect,
            "unassessable": group_unassessable,
            "accuracy": (group_correct / group_total) if group_total else None,
            "accuracy_pct": (100 * group_correct / group_total) if group_total else None,
            "assessable_accuracy": (group_correct / group_assessable) if group_assessable else None,
            "assessable_accuracy_pct": (100 * group_correct / group_assessable) if group_assessable else None,
        }

    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        level = _metadata_value(row, ["Level", "level", "Scenario Depth", "场景深度"]) or "unknown"
        scenario = _metadata_value(row, ["Scenario", "scenario", "Financial Scenario", "金融场景"]) or "unknown"
        by_level[str(level)].append(row)
        by_scenario[str(scenario)].append(row)

    return {
        "benchmark": "fingaia",
        "protocol": "FinGAIA official 1/0/-1 scoring",
        "total": total,
        "judged": len(judged_scores),
        "correct": correct,
        "incorrect": incorrect,
        "unassessable": unassessable,
        "accuracy": (correct / total) if total else None,
        "accuracy_pct": (100 * correct / total) if total else None,
        "assessable_accuracy": (correct / assessable) if assessable else None,
        "assessable_accuracy_pct": (100 * correct / assessable) if assessable else None,
        "wa": (100 * correct / total) if total else None,
        "by_level": {level: summarize_group(group) for level, group in sorted(by_level.items())},
        "by_scenario": {scenario: summarize_group(group) for scenario, group in sorted(by_scenario.items())},
        "token_stats": aggregate_token_stats(rows),
    }
