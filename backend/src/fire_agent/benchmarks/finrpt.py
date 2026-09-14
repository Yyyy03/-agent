from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..finrpt_official import (
    OFFICIAL_PROMPT_FIELDS,
    FinRptOfficialResult,
    generate_official_responses,
)
from ..schemas import BenchTaskContext
from .base import BenchmarkRunner
from .common import (
    FINRPT_OFFICIAL_RESPONSE_FIELDS,
    JsonRow,
    first_present,
    jsonish_string,
    normalize_text,
    parse_jsonish,
    read_jsonl,
    write_jsonl,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class FinRptRunner(BenchmarkRunner):
    bench_name = "finrpt"
    description = "FinRpt official segmented report-generation tasks."
    task_family = "report"
    evaluator = "finrpt"
    question_keys = ("prompt", "question", "instruction")
    answer_keys = FINRPT_OFFICIAL_RESPONSE_FIELDS + ("answer", "report", "golden_answer")
    url_keys = ("root_url", "url", "source_url", "FY23 PDF Download Site", "FY24 PDF Download Site")

    use_official_generation = True
    official_max_rounds = 3
    official_required_prompt_keys = tuple(prompt_key for prompt_key, _ in OFFICIAL_PROMPT_FIELDS)

    official_eval_bundle_dir = Path(__file__).resolve().parents[3] / "official_benchmarks" / "finrpt"
    official_eval_results_dir = official_eval_bundle_dir / "data"
    official_eval_module_dir = official_eval_bundle_dir / "official_eval"

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------
    def materialize_context(self, record: JsonRow, task_index: int) -> BenchTaskContext:
        question = normalize_text(first_present(record, self.question_keys))
        if not question:
            raise ValueError(
                f"FinRpt record {task_index} has no question/prompt/instruction field; "
                f"refusing to synthesize one from metadata."
            )
        return super().materialize_context(record, task_index)

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in ["id", "stock_code", "date", "company_name", "stock_name"]:
            if record.get(key) not in (None, ""):
                extra[key] = record[key]
        # Carry the per-row reference responses so the evaluator can reach
        # them without going back to the raw dataset row.
        for key in FINRPT_OFFICIAL_RESPONSE_FIELDS:
            if record.get(key) not in (None, ""):
                extra[key] = record[key]
        return extra

    # ------------------------------------------------------------------
    # output_record adds the parsed 5-field replay payload to bench_extra so
    # the bundled exec_eval.py can be served from the JSONL alone.
    # ------------------------------------------------------------------
    def output_record(self, context: BenchTaskContext, task_log: JsonRow, judge_output: JsonRow) -> JsonRow:
        output = super().output_record(context, task_log, judge_output)
        official = self._official_prediction_record(context.raw_record, task_log)
        bench_extra = dict(output.get("bench_extra") or {})
        bench_extra.update(
            {
                "id": official["id"],
                "stock_code": official["stock_code"],
                "date": official["date"],
            }
        )
        prediction_payload = {
            field_name: official[field_name] for field_name in FINRPT_OFFICIAL_RESPONSE_FIELDS
        }
        bench_extra["prediction_responses"] = prediction_payload
        output["bench_extra"] = bench_extra
        if task_log.get("official_generation"):
            bench_extra["official_generation"] = task_log["official_generation"]
        return output

    # ------------------------------------------------------------------
    # Official-mode generation: replay upstream FinRpt-Gen prompts.
    # ------------------------------------------------------------------
    def _run_single(self, records: List[JsonRow], idx: int) -> JsonRow:
        record = records[idx]
        if self._should_use_official_generation(record):
            return self._run_single_official(records, idx)
        return super()._run_single(records, idx)

    def _should_use_official_generation(self, record: JsonRow) -> bool:
        if not _env_bool("FIRE_AGENT_FINRPT_USE_OFFICIAL_GENERATION", self.use_official_generation):
            return False
        if not self.agent.config.use_model_for_roles:
            return False
        if getattr(self.agent, "model", None) is None:
            return False
        return any(self._has_prompt(record, prompt_key) for prompt_key in self.official_required_prompt_keys)

    def _has_prompt(self, record: JsonRow, prompt_key: str) -> bool:
        value = record.get(prompt_key)
        if value in (None, ""):
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _run_single_official(self, records: List[JsonRow], idx: int) -> JsonRow:
        context = self.materialize_context(records[idx], idx)
        started = time.time()
        max_rounds = _env_int("FIRE_AGENT_FINRPT_OFFICIAL_MAX_ROUNDS", self.official_max_rounds)
        try:
            result = generate_official_responses(
                context.raw_record,
                self.agent.model,
                max_rounds=max_rounds,
            )
        except Exception as exc:
            task_log = self._build_official_error_task_log(context, exc, started)
        else:
            task_log = self._build_official_task_log(context, result, started)
        judge_output = self.judge(task_log)
        return self.output_record(context, task_log, judge_output)

    def _build_official_task_log(
        self,
        context: BenchTaskContext,
        result: FinRptOfficialResult,
        started: float,
    ) -> JsonRow:
        payload: Dict[str, str] = {field_name: "" for _, field_name in OFFICIAL_PROMPT_FIELDS}
        payload.update(result.responses)
        agent_result_str = json.dumps(payload, ensure_ascii=False)

        status: str
        error: Optional[str] = None
        if result.failed_prompts and not result.succeeded_any:
            status = "error"
            error = f"All official FinRpt prompts failed: {sorted(result.failed_prompts)}"
        elif result.failed_prompts or result.missing_prompts:
            status = "max_steps"
        else:
            status = "success"

        trajectory = self._build_official_trajectory(result)
        elapsed = time.time() - started
        official_summary = {
            "mode": "official_finrpt_gen_replay",
            "max_rounds": _env_int("FIRE_AGENT_FINRPT_OFFICIAL_MAX_ROUNDS", self.official_max_rounds),
            "missing_prompts": result.missing_prompts,
            "failed_prompts": result.failed_prompts,
            "elapsed_seconds": round(elapsed, 3),
            "fields": [
                {
                    "field": field_result.field,
                    "status": field_result.status,
                    "attempts": field_result.attempts,
                    "error": field_result.error,
                }
                for field_result in result.field_results
            ],
        }

        log: JsonRow = {
            "bench_name": context.bench_name,
            "task_index": context.task_index,
            "question": context.task_prompt,
            "golden_answer": context.golden_answer,
            "agent_result": agent_result_str,
            "agent_trajectory": trajectory,
            "bench_extra": dict(context.bench_extra or {}),
            "status": status,
            "stats": {
                "steps": len(result.field_results),
                "tool_calls": 0,
                "tools_by_name": {},
                "tokens": {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0},
            },
            "official_generation": official_summary,
        }
        if error:
            log["error"] = error
        return log

    def _build_official_error_task_log(self, context: BenchTaskContext, exc: Exception, started: float) -> JsonRow:
        payload = {field_name: "" for _, field_name in OFFICIAL_PROMPT_FIELDS}
        agent_result_str = json.dumps(payload, ensure_ascii=False)
        return {
            "bench_name": context.bench_name,
            "task_index": context.task_index,
            "question": context.task_prompt,
            "golden_answer": context.golden_answer,
            "agent_result": agent_result_str,
            "agent_trajectory": [],
            "bench_extra": dict(context.bench_extra or {}),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "stats": {
                "steps": 0,
                "tool_calls": 0,
                "tools_by_name": {},
                "tokens": {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0},
            },
            "official_generation": {
                "mode": "official_finrpt_gen_replay",
                "elapsed_seconds": round(time.time() - started, 3),
                "fatal_error": f"{type(exc).__name__}: {exc}",
            },
        }

    def _build_official_trajectory(self, result: FinRptOfficialResult) -> List[Dict[str, Any]]:
        trajectory: List[Dict[str, Any]] = []
        for field_result in result.field_results:
            trajectory.append(
                {
                    "name": "action",
                    "tool_calls": [
                        {
                            "name": "finrpt_official_writer",
                            "action": field_result.field,
                            "arguments": {},
                        }
                    ],
                    "obs": (
                        f"[finrpt_official_writer.{field_result.field}] "
                        f"status={field_result.status} attempts={field_result.attempts} "
                        f"prompt_chars={len(field_result.prompt or '')} "
                        f"response_chars={len(field_result.raw_response or '')}"
                        + (f" error={field_result.error}" if field_result.error else "")
                    ),
                    "think": "",
                }
            )
        return trajectory

    def _sample_id(self, record: JsonRow) -> str:
        explicit = first_present(record, ["id", "sample_id", "task_id"])
        if explicit:
            return str(explicit)
        stock_code = normalize_text(first_present(record, ["stock_code", "ticker", "symbol"]))
        date = normalize_text(first_present(record, ["date", "trade_date", "as_of_date"]))
        return f"{stock_code}_{date}".strip("_") or "unknown"

    def _official_prediction_record(self, record: JsonRow, task_log: JsonRow) -> JsonRow:
        official = self._official_response_fields(task_log)
        official.update(
            {
                "id": self._sample_id(record),
                "stock_code": first_present(record, ["stock_code", "ticker", "symbol"]),
                "date": first_present(record, ["date", "trade_date", "as_of_date"]),
            }
        )
        return official

    def _official_response_fields(self, task_log: JsonRow) -> JsonRow:
        payload = self._parse_agent_json_payload(task_log)
        official: JsonRow = {}
        for key in FINRPT_OFFICIAL_RESPONSE_FIELDS:
            official[key] = jsonish_string(payload.get(key, ""))
        if not any(official.values()):
            final_answer = str(task_log.get("agent_result") or "")
            official["report_write_response"] = final_answer
        return official

    def _parse_agent_json_payload(self, task_log: JsonRow) -> JsonRow:
        candidate = task_log.get("agent_result")
        parsed = parse_jsonish(candidate)
        if isinstance(parsed, dict):
            return parsed
        return {}

    # ------------------------------------------------------------------
    # Official corpus-level scoring (exec_eval.py ROUGE-L / BERTScore / etc.)
    # ------------------------------------------------------------------
    def finalize_after_run(self, logs: List[JsonRow]) -> None:
        if _env_bool("FIRE_AGENT_FINRPT_SKIP_OFFICIAL_EVAL", False):
            super().finalize_after_run(logs)
            return
        if not self.output_path.exists():
            super().finalize_after_run(logs)
            return
        rows = read_jsonl(self.output_path)
        if not rows:
            super().finalize_after_run(logs)
            return

        results_dir = Path(
            os.getenv("FIRE_AGENT_FINRPT_RESULTS_DIR")
            or self.official_eval_results_dir
        ).resolve()
        standard_file = results_dir / "standard.jsonl"
        if not standard_file.exists():
            print(
                f"[finrpt] Skipping official exec_eval: standard.jsonl not found at {standard_file}",
                flush=True,
            )
            super().finalize_after_run(logs)
            return

        try:
            scores = self._run_official_exec_eval(rows, results_dir)
        except Exception as exc:
            print(f"[finrpt] Official exec_eval failed: {type(exc).__name__}: {exc}", flush=True)
            super().finalize_after_run(logs)
            return

        if not scores:
            super().finalize_after_run(logs)
            return

        self._merge_official_scores_into_output(rows, scores)
        scores_path = self.output_path.with_name(self.output_path.stem + "_official_scores.json")
        scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[finrpt] Wrote official scores to {scores_path}", flush=True)
        super().finalize_after_run(logs)

    def _run_official_exec_eval(self, rows: List[JsonRow], results_dir: Path) -> Dict[str, Any]:
        results_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.output_path.stem
        model_jsonl = results_dir / f"{model_name}.jsonl"
        if model_jsonl.resolve() == self.output_path.resolve():
            model_name = f"{model_name}_official"
            model_jsonl = results_dir / f"{model_name}.jsonl"

        official_rows: List[JsonRow] = []
        for row in rows:
            extra = row.get("bench_extra") or {}
            sample_id = (
                extra.get("id") if isinstance(extra, dict) else None
            ) or row.get("sample_id") or row.get("id")
            if not sample_id:
                continue
            stock_code = (
                extra.get("stock_code") if isinstance(extra, dict) else None
            ) or row.get("stock_code")
            date = (
                extra.get("date") if isinstance(extra, dict) else None
            ) or row.get("date")
            prediction_responses: Dict[str, Any] = {}
            if isinstance(extra, dict):
                pr = extra.get("prediction_responses")
                if isinstance(pr, dict):
                    prediction_responses = pr
            if not prediction_responses:
                # Legacy compat path: read top-level fields too.
                prediction_responses = {
                    field_name: row.get(field_name) for field_name in FINRPT_OFFICIAL_RESPONSE_FIELDS
                }
            official_rows.append(
                {
                    "id": sample_id,
                    "stock_code": stock_code,
                    "date": date,
                    **{
                        field_name: (prediction_responses.get(field_name) or "") or ""
                        for field_name in FINRPT_OFFICIAL_RESPONSE_FIELDS
                    },
                }
            )

        if not official_rows:
            print("[finrpt] No usable rows for official exec_eval (missing sample_id/id).", flush=True)
            return {}

        write_jsonl(model_jsonl, official_rows, mode="w")
        print(
            f"[finrpt] Prepared official candidate file: {model_jsonl} ({len(official_rows)} rows)",
            flush=True,
        )

        scores_path = results_dir / f"{model_name}_scores.json"
        if scores_path.exists():
            scores_path.unlink()

        import sys

        module_dir = str(self.official_eval_module_dir.resolve())
        added_to_path = False
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            added_to_path = True
        try:
            from finrpt.benchmark.exec_eval import get_score_all  # type: ignore
            print("[finrpt] Running official exec_eval.get_score_all (this loads bert-base-chinese)...", flush=True)
            get_score_all(str(results_dir), model_name)
        finally:
            if added_to_path:
                try:
                    sys.path.remove(module_dir)
                except ValueError:
                    pass

        if not scores_path.exists():
            print(f"[finrpt] exec_eval did not produce {scores_path}; nothing to merge.", flush=True)
            return {}
        return json.loads(scores_path.read_text(encoding="utf-8"))

    def _merge_official_scores_into_output(self, rows: List[JsonRow], scores: Dict[str, Any]) -> None:
        summary = {
            "evaluator": "finrpt_official_exec_eval",
            "official_scores": scores,
            "rouge_l_all": _safe_get(scores, ["all", "rouge-l"]),
            "bert_all": _safe_get(scores, ["all", "bert"]),
            "number_all": _safe_get(scores, ["all", "number"]),
            "trend_accuracy": scores.get("trend"),
            "success_accuracy": scores.get("success"),
            "note": "Corpus-level scores from official FinRpt exec_eval.py (ROUGE-L / BERTScore / number / trend / success).",
        }
        for row in rows:
            score = row.get("score")
            if not isinstance(score, dict):
                score = {}
            score = dict(score)
            score.update(summary)
            score["official_eval_required"] = False
            row["score"] = score
        write_jsonl(self.output_path, rows, mode="w")


def _safe_get(data: Any, path: List[str]) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur
