from __future__ import annotations

from typing import Any, Dict

from .common import JsonRow


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def _bench_extra(task_log: JsonRow) -> Dict[str, Any]:
    """Read the per-bench metadata block, transparently falling back to the
    legacy ``evaluator_metadata`` / ``eval_metadata`` containers so old result
    JSONLs can still be re-evaluated."""

    extra = task_log.get("bench_extra")
    if isinstance(extra, dict) and extra:
        return extra
    legacy = task_log.get("evaluator_metadata") or task_log.get("eval_metadata") or {}
    return legacy if isinstance(legacy, dict) else {}


class BaseEvaluator:
    name: str = "generic"

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        return {
            "evaluator": self.name,
            "judgement": None,
            "score": None,
            "note": "No benchmark-specific evaluator configured.",
        }

    def _prediction(self, task_log: JsonRow) -> str:
        return str(_first_non_missing(task_log.get("agent_result")))

    def _golden(self, task_log: JsonRow) -> str:
        extra = _bench_extra(task_log)
        return str(_first_non_missing(task_log.get("golden_answer"), extra.get("golden_answer")))
