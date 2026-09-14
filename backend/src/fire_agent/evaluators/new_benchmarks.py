from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import JsonRow, normalize_text
from .core import BaseEvaluator, _bench_extra


_DIRECT_ANSWER_KEYS = ("answer", "final_answer", "final_result", "result", "text")
_BIZFIN_OFFICIAL_ANSWER_KEYS = (
    "相关新闻序号",
    "相关内容序号",
    "无关内容序号",
    "Relevant Content Numbers",
    "Irrelevant Content Numbers",
    "排序结果",
)
_FINAL_ANSWER_MARKER_RE = re.compile(r"(?:final\s+answer|final\s+json|最终答案|答案)\s*[:：]", re.I)
_POLLUTED_REASONING_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:reasoning|analysis|thought|reason|思考|推理|理由)\s*[:：]|</?think\b|\"think\"\s*:|\"tools\"\s*:"
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "").strip()


def _iter_json_values(text: str):
    decoder = json.JSONDecoder()
    starts = sorted([match.start() for match in re.finditer(r"[\{\[]", str(text or ""))])
    for start in starts:
        try:
            value, _ = decoder.raw_decode(str(text)[start:])
        except Exception:
            continue
        yield value


def _final_tool_answer_from_value(value: Any) -> Any:
    if isinstance(value, dict):
        name = str(value.get("name") or value.get("tool") or value.get("tool_name") or "").strip()
        if name == "final_answer":
            args = value.get("arguments") or value.get("args") or {}
            if isinstance(args, dict):
                for key in _DIRECT_ANSWER_KEYS:
                    if key in args and args.get(key) not in (None, ""):
                        return args.get(key)
            elif args not in (None, ""):
                return args
        tools = value.get("tools")
        if isinstance(tools, dict):
            tools = [tools]
        if isinstance(tools, list):
            for item in reversed(tools):
                found = _final_tool_answer_from_value(item)
                if found not in (None, ""):
                    return found
    if isinstance(value, list):
        for item in reversed(value):
            found = _final_tool_answer_from_value(item)
            if found not in (None, ""):
                return found
    return None


def _extract_marked_final_answer(text: str) -> str:
    markers = list(_FINAL_ANSWER_MARKER_RE.finditer(str(text or "")))
    if not markers:
        return ""
    tail = str(text)[markers[-1].end() :].strip()
    if not tail:
        return ""
    values = list(_iter_json_values(tail))
    for value in reversed(values):
        found = _final_tool_answer_from_value(value)
        if found not in (None, ""):
            return _json_dumps(found)
    for value in reversed(values):
        if isinstance(value, (dict, list)):
            return _json_dumps(value)
    return tail


def _unwrap_runtime_final_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        marked = _extract_marked_final_answer(text)
        if marked:
            return _unwrap_runtime_final_answer(marked)
        for json_value in reversed(list(_iter_json_values(text))):
            found = _final_tool_answer_from_value(json_value)
            if found not in (None, ""):
                return _json_dumps(found)
        if _POLLUTED_REASONING_RE.search(text):
            return ""
        return text
    if "content" in parsed:
        return _unwrap_runtime_final_answer(parsed.get("content"))
    found = _final_tool_answer_from_value(parsed)
    if found not in (None, ""):
        return _json_dumps(found)
    for key in _DIRECT_ANSWER_KEYS:
        if key in parsed:
            return _unwrap_runtime_final_answer(parsed.get(key))
    if any(key in parsed for key in _BIZFIN_OFFICIAL_ANSWER_KEYS):
        return text
    tools = parsed.get("tools")
    if isinstance(tools, dict):
        tools = [tools]
    if isinstance(tools, list):
        for item in tools:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or item.get("tool") or "").strip() != "final_answer":
                continue
            args = item.get("arguments") or item.get("args") or {}
            if isinstance(args, dict):
                answer = args.get("answer", args.get("final_answer", args.get("result", args.get("text", ""))))
            else:
                answer = args
            if isinstance(answer, (dict, list)):
                return json.dumps(answer, ensure_ascii=False)
            return str(answer or "").strip()
    return ""


def _visible_result_content(task_log: JsonRow) -> str:
    """Return only the model's final visible content from the task result.

    Evaluators must not recover answers from reasoning-only fields such as
    ``agent_trajectory[*].think``, tool observations, or planning metadata. If a
    model emits a Qwen-style ``<think>...</think>`` block in the result content,
    only text after the reasoning block is considered answer content.
    """

    text = str(task_log.get("agent_result") or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)<think>.*$", "", text).strip()
    return text


class RFCBenchEvaluator(BaseEvaluator):
    """RFC-BENCH Task 1 binary reference-free misinformation scorer.

    The public release currently contains only perturbed/manipulated rewrites,
    so this adapter follows the paper's binary task interface but can only
    measure the released positive class until factual/original negatives are
    available.
    """

    name = "rfc_bench_task1_binary_exact"
    FACTUAL_LABELS = ("factual", "original")
    MANIPULATED_LABELS = ("manipulated", "misinformation", "counterfactual")
    LABELS = ("factual", "manipulated")

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        extra = _bench_extra(task_log)
        golden = normalize_text(
            task_log.get("golden_answer")
            or extra.get("golden_answer")
            or extra.get("answer")
            or ""
        )
        golden = self._normalize_label(golden, extra)
        prediction_content = _visible_result_content(task_log)
        predicted, ambiguous = self._extract_label(_unwrap_runtime_final_answer(prediction_content))
        if golden not in self.LABELS:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": "Missing or invalid RFC-BENCH binary golden label.",
                "golden_label": golden,
                "predicted_label": predicted,
            }

        is_correct = predicted == golden and not ambiguous
        return {
            "evaluator": self.name,
            "judgement": "correct" if is_correct else "incorrect",
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "golden_label": golden,
            "predicted_label": predicted,
            "ambiguous_prediction": ambiguous,
            "prediction_source": {"source": "agent_result_content"},
            "split": extra.get("split"),
            "perturbation_type": extra.get("perturbation_type"),
            "is_manipulated": golden == "manipulated",
            "protocol": (
                "RFC-BENCH Task 1 binary reference-free detection: factual/original vs "
                "manipulated/misinformation. The current public release contains only "
                "manipulated positives, so this score is positive-class accuracy/recall, "
                "not the full paper benchmark accuracy over both classes."
            ),
        }

    @classmethod
    def _extract_label(cls, value: Any) -> Tuple[str, bool]:
        text = normalize_text(value)
        normalized = cls._normalize_label(text, {})
        if normalized in cls.LABELS:
            return normalized, False
        matches = []
        for label in cls.FACTUAL_LABELS:
            if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", text):
                matches.append("factual")
        for label in cls.MANIPULATED_LABELS:
            if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", text):
                matches.append("manipulated")
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0], False
        if len(unique) > 1:
            return unique[0], True
        return "", False

    @classmethod
    def _normalize_label(cls, value: Any, extra: JsonRow) -> str:
        text = normalize_text(value)
        if text in cls.FACTUAL_LABELS:
            return "factual"
        if text in cls.MANIPULATED_LABELS:
            return "manipulated"
        if text in {"true", "1", "yes"}:
            return "manipulated"
        if text in {"false", "0", "no"}:
            return "factual"
        if extra.get("is_manipulated") is True:
            return "manipulated"
        if extra.get("is_manipulated") is False:
            return "factual"
        return text


class RFCTask2Evaluator(BaseEvaluator):
    """RFC-BENCH Task 2 four-way comparative manipulation diagnosis scorer."""

    name = "rfc_bench_task2_four_way_exact"
    LABELS = ("numerical", "flipping", "sentiment", "causal")
    LABEL_ALIASES = {
        "numeric": "numerical",
        "number": "numerical",
        "numbers": "numerical",
        "quantitative": "numerical",
        "numerical perturbation": "numerical",
        "numerical manipulation": "numerical",
        "numerical change": "numerical",
        "flip": "flipping",
        "flipped": "flipping",
        "polarity": "flipping",
        "directional": "flipping",
        "directional flipping": "flipping",
        "direction": "flipping",
        "sentiment amplification": "sentiment",
        "tone": "sentiment",
        "tonal": "sentiment",
        "stance": "sentiment",
        "causality": "causal",
        "cause": "causal",
        "causation": "causal",
        "causal distortion": "causal",
        "attribution": "causal",
    }

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        extra = _bench_extra(task_log)
        golden = self._normalize_label(task_log.get("golden_answer") or "")
        prediction_content = _visible_result_content(task_log)
        predicted, ambiguous = self._extract_label(_unwrap_runtime_final_answer(prediction_content))
        if golden not in self.LABELS:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": "Missing or invalid RFC-BENCH Task 2 golden label.",
                "golden_label": golden,
                "predicted_label": predicted,
                "valid_labels": list(self.LABELS),
            }

        is_correct = predicted == golden and not ambiguous
        return {
            "evaluator": self.name,
            "judgement": "correct" if is_correct else "incorrect",
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "golden_label": golden,
            "predicted_label": predicted,
            "ambiguous_prediction": ambiguous,
            "prediction_source": {"source": "agent_result_content"},
            "split": extra.get("split"),
            "perturbation_type": extra.get("perturbation_type"),
            "source_page_source": extra.get("source_page_source"),
            "source_page_char_len": extra.get("source_page_char_len"),
            "protocol": (
                "RFC-BENCH Task 2 comparative diagnosis: identify the manipulation "
                "type from numerical, flipping, sentiment, or causal using the "
                "provided source page and manipulated paragraph."
            ),
        }

    @classmethod
    def _extract_label(cls, value: Any) -> Tuple[str, bool]:
        text = normalize_text(value)
        normalized = cls._normalize_label(text)
        if normalized in cls.LABELS:
            return normalized, False

        matches = []
        for label in cls.LABELS:
            if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", text):
                matches.append(label)
        for alias, label in cls.LABEL_ALIASES.items():
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
                matches.append(label)
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0], False
        if len(unique) > 1:
            return unique[0], True
        return "", False

    @classmethod
    def _normalize_label(cls, value: Any) -> str:
        text = normalize_text(value)
        if text in cls.LABELS:
            return text
        return cls.LABEL_ALIASES.get(text, text)


class BizFinBenchV2OfficialEvaluator(BaseEvaluator):
    """Adapter around the official BizFinBench.v2 ``compare_func`` scripts."""

    name = "bizfinbench_v2_official"

    TASK_TO_SCRIPT: Dict[str, str] = {
        "anomaly_information_tracing": "eval_anomaly_information_tracing.py",
        "conterfactual": "eval_financial_quantitative_computation.py",
        "event_logic_reasoning": "eval_event_logic_reasoning.py",
        "financial_data_description": "eval_financial_data_description.py",
        "financial_multi-turn_perception": "eval_financial_multi-turn_perception.py",
        "financial_multiturn_perception": "eval_financial_multi-turn_perception.py",
        "financial_quantitative_computation": "eval_financial_quantitative_computation.py",
        "financial_report_analysis": "eval_financial_report_analysis.py",
        "stock_price_predict": "eval_stock_price_predict.py",
        "user_sentiment_analysis": "eval_user_sentiment_analysis.py",
    }

    _module_cache: Dict[str, Any] = {}
    _lock = threading.Lock()

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        extra = _bench_extra(task_log)
        task_type = normalize_text(extra.get("task_type") or "")
        script_name = self.TASK_TO_SCRIPT.get(task_type)
        if not script_name:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": f"Unsupported BizFinBench.v2 task_type: {task_type!r}.",
                "task_type": task_type,
            }

        module = self._load_official_module(script_name)
        if module is None or not hasattr(module, "evaluation"):
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": f"Official BizFinBench.v2 eval script is unavailable: {script_name}.",
                "task_type": task_type,
                "official_script": script_name,
            }

        try:
            evaluated = self._evaluate_official_prediction(module, task_log, extra, task_type, prediction=None)
        except Exception as exc:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "note": f"Official BizFinBench.v2 eval failed: {type(exc).__name__}: {exc}",
                "task_type": task_type,
                "official_script": script_name,
            }

        result = {
            "evaluator": self.name,
            "judgement": "correct" if evaluated["is_correct"] else "incorrect",
            "score": evaluated["score"],
            "is_correct": evaluated["is_correct"],
            "task_type": task_type,
            "language": extra.get("language"),
            "mode": "official_compare_func",
            "official_script": script_name,
            "official_summary": evaluated["official_summary"],
            "official_eval_result": evaluated["scored_row"].get("eval_result"),
            "prediction_adapter": evaluated["prediction_adapter"],
            "prediction_source": evaluated["prediction_source"],
            "protocol": "BizFinBench.v2 official compare_func adapter on one-row JSONL.",
        }
        return result

    @classmethod
    def _official_root(cls) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "official_benchmarks"
            / "bizfinbench_v2"
            / "official_eval"
        )

    @classmethod
    def _load_official_module(cls, script_name: str) -> Any:
        with cls._lock:
            if script_name in cls._module_cache:
                return cls._module_cache[script_name]
            root = cls._official_root()
            script = root / "benchmark_code" / "BizFinBench.v2" / script_name
            if not script.is_file():
                cls._module_cache[script_name] = None
                return None
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            module_name = f"_fire_agent_bizfinbench_v2_{script.stem}"
            spec = importlib.util.spec_from_file_location(module_name, script)
            if spec is None or spec.loader is None:
                cls._module_cache[script_name] = None
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._module_cache[script_name] = module
            return module

    def _official_row(self, task_log: JsonRow, extra: JsonRow, prediction: str) -> JsonRow:
        choices = extra.get("choices")
        if not choices:
            golden = task_log.get("golden_answer") or extra.get("golden_answer") or ""
            choices = [{"message": {"content": [{"text": str(golden)}]}}]
        row: JsonRow = {
            "messages": extra.get("messages") or [],
            "choices": choices,
            "predict_result": prediction,
            "sample_id": task_log.get("sample_id"),
            "task_type": extra.get("task_type"),
            "language": extra.get("language"),
            "source_file": extra.get("source_file"),
            "source_row_index": extra.get("source_row_index"),
        }
        return {key: value for key, value in row.items() if value not in (None, "")}

    @classmethod
    def _prediction_for_official(cls, task_log: JsonRow, task_type: str) -> Tuple[str, JsonRow]:
        visible = _visible_result_content(task_log)
        prediction = _unwrap_runtime_final_answer(visible)
        source: JsonRow = {"source": "agent_result_content"}
        # Keep final external mint-agent scoring aligned with the in-training
        # validation reward path: if the generic runtime unwrap returns empty
        # because the visible answer is polluted by reasoning, still let the
        # BizFin-specific adapter recover structured/boxed answers from that
        # visible content. This never reads hidden reason/log/trajectory fields.
        if not prediction and visible:
            prediction = visible
            source = {"source": "agent_result_content", "fallback": "visible_content_bizfin_adapter"}
        return prediction, source

    def _evaluate_official_prediction(
        self,
        module: Any,
        task_log: JsonRow,
        extra: JsonRow,
        task_type: str,
        prediction: Optional[str],
    ) -> JsonRow:
        if prediction is None:
            raw_prediction, prediction_source = self._prediction_for_official(task_log, task_type)
        else:
            raw_prediction = str(prediction or "").strip()
            prediction_source = {"source": "agent_result_content"}
        normalized_prediction, prediction_adapter = self._normalize_prediction_for_official(
            task_type, raw_prediction, extra
        )
        row = self._official_row(task_log, extra, normalized_prediction)
        official_summary, scored_row = self._run_single_row(module, row)
        score = self._score_from_scored_row(scored_row, official_summary)
        is_correct = bool(score is not None and float(score) >= 1.0)
        return {
            "raw_prediction": raw_prediction,
            "normalized_prediction": normalized_prediction,
            "prediction_source": prediction_source,
            "prediction_adapter": prediction_adapter,
            "official_summary": official_summary,
            "scored_row": scored_row,
            "score": score,
            "is_correct": is_correct,
        }

    @classmethod
    def _clean_polluted_biz_answer(cls, text: str, task_type: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        polluted = (
            len(value) > 800
            or "</think>" in value
            or "<think>" in value
            or '"tools"' in value
            or "Final Answer" in value
            or "Final JSON" in value
            or "最终答案" in value
        )
        if not polluted and cls._looks_like_biz_answer(value, task_type):
            return value

        tail = re.sub(r"(?is)^.*?</think>\s*", "", value).strip()
        for candidate in (tail, value):
            extracted = cls._extract_biz_answer_from_json_fragments(candidate, task_type)
            if extracted:
                return extracted
            extracted = cls._extract_boxed_answer(candidate)
            if extracted:
                return extracted
            extracted = cls._extract_biz_answer_from_markers(candidate, task_type)
            if extracted:
                return extracted
            extracted = cls._extract_simple_biz_tail(candidate, task_type)
            if extracted:
                return extracted
        return value

    @classmethod
    def _extract_final_answer_from_tools(cls, tools: List[Any]) -> str:
        for item in reversed(tools):
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or item.get("tool") or "").strip() != "final_answer":
                continue
            args = item.get("arguments") or item.get("args") or {}
            if isinstance(args, dict):
                answer = args.get("answer", args.get("final_answer", args.get("result", args.get("text", ""))))
            else:
                answer = args
            if isinstance(answer, (dict, list)):
                return json.dumps(answer, ensure_ascii=False)
            return str(answer or "").strip()
        return ""

    @classmethod
    def _extract_biz_answer_from_json_fragments(cls, text: str, task_type: str) -> str:
        for fragment in reversed(cls._balanced_json_spans(text, "{", "}")):
            try:
                parsed = json.loads(fragment)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            answer = cls._extract_final_answer_from_tools(parsed.get("tools") if isinstance(parsed.get("tools"), list) else [parsed.get("tools")] if isinstance(parsed.get("tools"), dict) else [])
            if answer:
                return cls._clean_polluted_biz_answer(answer, task_type)
            if task_type == "anomaly_information_tracing":
                keys = (
                    "相关新闻序号",
                    "相关内容序号",
                    "无关内容序号",
                    "Relevant Content Numbers",
                    "Irrelevant Content Numbers",
                )
            elif task_type == "financial_data_description":
                keys = ("answer",)
            else:
                keys = ("answer", "排序结果", "result")
            if any(key in parsed for key in keys):
                return json.dumps(parsed, ensure_ascii=False)

        if task_type == "financial_multi-turn_perception":
            for fragment in reversed(cls._balanced_json_spans(text, "[", "]")):
                try:
                    parsed = json.loads(fragment)
                except Exception:
                    continue
                if isinstance(parsed, list) and parsed and all(isinstance(item, dict) for item in parsed):
                    return json.dumps(parsed, ensure_ascii=False)
        return ""

    @staticmethod
    def _balanced_json_spans(text: str, open_ch: str, close_ch: str) -> List[str]:
        spans: List[str] = []
        stack = 0
        start = -1
        in_str = False
        escaped = False
        for idx, ch in enumerate(str(text or "")):
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                if stack == 0:
                    start = idx
                stack += 1
            elif ch == close_ch and stack:
                stack -= 1
                if stack == 0 and start >= 0:
                    spans.append(str(text)[start : idx + 1])
                    start = -1
        return spans

    @classmethod
    def _extract_biz_answer_from_markers(cls, text: str, task_type: str) -> str:
        markers = (
            "Final Answer:",
            "Final answer:",
            "Final JSON:",
            "Final output:",
            "最终答案：",
            "最终答案:",
            "答案：",
            "答案:",
        )
        for marker in markers:
            idx = str(text or "").rfind(marker)
            if idx < 0:
                continue
            tail = str(text)[idx + len(marker) :].strip()
            return (
                cls._extract_biz_answer_from_json_fragments(tail, task_type)
                or cls._extract_boxed_answer(tail)
                or cls._extract_simple_biz_tail(tail, task_type)
            )
        return ""

    @staticmethod
    def _extract_boxed_answer(text: str) -> str:
        answer_match = list(re.finditer(r"answer_boxed\s*\{\{?\s*[-+]?\d+(?:\.\d+)?\s*\}\}?", str(text or ""), re.I))
        interval_match = list(
            re.finditer(
                r"interval_boxed\s*\{\{?\s*\[\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\]\s*\}\}?",
                str(text or ""),
                re.I | re.S,
            )
        )
        if answer_match and interval_match:
            return answer_match[-1].group(0) + "\n" + interval_match[-1].group(0)
        boxed = list(re.finditer(r"boxed\s*\{[^{}]{1,300}\}", str(text or ""), re.I | re.S))
        if boxed:
            return boxed[-1].group(0)
        return ""

    @staticmethod
    def _extract_simple_biz_tail(text: str, task_type: str) -> str:
        tail = re.sub(r"(?is)^.*?</think>\s*", "", str(text or "")).strip().strip("`")
        if task_type in {"event_logic_reasoning", "financial_report_analysis"}:
            matches = re.findall(r"\b\d+(?:\s*[,，]\s*\d+){1,20}\b", tail)
            if matches:
                return matches[-1].replace("，", ",")
        if task_type in {"financial_quantitative_computation", "conterfactual"}:
            matches = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
            if matches:
                return matches[-1]
        if task_type in {"anomaly_information_tracing", "financial_data_description"}:
            bracketed = re.findall(r"\[[\s\d,，;；]+\]", tail)
            if bracketed:
                return bracketed[-1].replace("，", ",")
        return ""

    @classmethod
    def _normalize_prediction_for_official(
        cls, task_type: str, prediction: str, extra: JsonRow
    ) -> Tuple[str, JsonRow]:
        original = str(prediction or "").strip()
        adapter: JsonRow = {"applied": False, "type": "none"}
        if not original:
            return original, adapter
        cleaned = cls._clean_polluted_biz_answer(original, task_type)
        if cleaned and cleaned != original:
            original = cleaned
            adapter = {"applied": True, "type": "clean_polluted_visible_answer"}

        if task_type in {
            "conterfactual",
            "financial_quantitative_computation",
            "event_logic_reasoning",
        }:
            if cls._json_object_has_any_key(original, ("answer", "排序结果", "result")):
                return original, adapter
            return (
                json.dumps({"answer": original}, ensure_ascii=False),
                {"applied": True, "type": "wrap_answer_object"},
            )

        if task_type == "financial_data_description":
            if cls._json_object_has_any_key(original, ("answer",)):
                return original, adapter
            parsed_list = cls._parse_int_list_answer(original)
            if parsed_list is not None:
                return (
                    json.dumps({"answer": parsed_list}, ensure_ascii=False),
                    {"applied": True, "type": "wrap_answer_list"},
                )
            return original, adapter

        if task_type == "anomaly_information_tracing":
            if cls._json_object_has_any_key(
                original,
                (
                    "相关新闻序号",
                    "相关内容序号",
                    "无关内容序号",
                    "Relevant Content Numbers",
                    "Irrelevant Content Numbers",
                ),
            ):
                return original, adapter
            parsed_list = cls._parse_int_list_answer(original)
            if parsed_list is not None:
                language = normalize_text(extra.get("language") or "")
                key = "Relevant Content Numbers" if language == "en" else "相关内容序号"
                return (
                    json.dumps({key: parsed_list}, ensure_ascii=False),
                    {"applied": True, "type": "wrap_relevant_content_numbers", "key": key},
                )
            return original, adapter

        if task_type == "financial_multi-turn_perception":
            array_answer = cls._extract_json_array_answer(original)
            if array_answer:
                return array_answer, {"applied": True, "type": "extract_json_array_answer"}
            return original, adapter

        if task_type == "financial_report_analysis":
            if re.search(r"boxed\s*\{.*?\}", original, flags=re.IGNORECASE | re.DOTALL):
                return original, adapter
            compact = original.strip().strip("`")
            if compact:
                return f"boxed{{{compact}}}", {"applied": True, "type": "wrap_boxed"}
            return original, adapter

        if task_type == "stock_price_predict":
            normalized = cls._normalize_boxed_interval(original, lo_mul=0.99, hi_mul=1.01)
            if normalized != original:
                return (
                    normalized,
                    {
                        "applied": True,
                        "type": "normalize_boxed_interval",
                        "lo_mul": 0.99,
                        "hi_mul": 1.01,
                    },
                )
            return original, adapter

        if task_type == "user_sentiment_analysis":
            normalized = cls._normalize_boxed_interval(original, lo_mul=0.90, hi_mul=1.10)
            if normalized != original:
                return (
                    normalized,
                    {
                        "applied": True,
                        "type": "normalize_boxed_interval",
                        "lo_mul": 0.90,
                        "hi_mul": 1.10,
                    },
                )
            return original, adapter

        return original, adapter

    @classmethod
    def _looks_like_biz_answer(cls, text: str, task_type: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if re.search(r"(?:answer_boxed|interval_boxed|boxed)\s*\{", stripped, flags=re.IGNORECASE):
            return True
        if stripped.startswith(("{", "[")):
            return True
        if task_type in {"conterfactual", "financial_quantitative_computation"}:
            return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped))
        if task_type in {"event_logic_reasoning", "financial_report_analysis"}:
            return bool(re.fullmatch(r"[A-Za-z0-9,\s，]+", stripped)) and "," in stripped
        return False

    @staticmethod
    def _normalize_boxed_interval(text: str, lo_mul: float, hi_mul: float) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return stripped
        answer_match = re.search(
            r"answer_boxed\s*\{\{?\s*([-+]?\d+(?:\.\d+)?)\s*\}\}?",
            stripped,
            flags=re.IGNORECASE,
        )
        interval_match = re.search(
            r"interval_boxed\s*\{\{?\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*\}\}?",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not answer_match or not interval_match:
            return stripped
        try:
            answer = float(answer_match.group(1))
            lower = float(interval_match.group(1))
            upper = float(interval_match.group(2))
        except Exception:
            return stripped

        expected_lower = answer * lo_mul
        expected_upper = answer * hi_mul
        if expected_lower > expected_upper:
            expected_lower, expected_upper = expected_upper, expected_lower
        if lower > upper:
            lower, upper = upper, lower

        tolerance = max(0.05, abs(answer) * 0.0005)
        if abs(lower - expected_lower) > tolerance or abs(upper - expected_upper) > tolerance:
            return stripped
        return f"answer_boxed{{{answer}}}\ninterval_boxed{{[{expected_lower},{expected_upper}]}}"

    @staticmethod
    def _json_object_has_any_key(text: str, keys: Iterable[str]) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        candidates = [stripped]
        candidates.extend(item.strip() for item in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE))
        candidates.extend(item.strip() for item in re.findall(r"\{[\s\S]*\}", stripped))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict) and any(key in parsed for key in keys):
                return True
        return False

    @classmethod
    def _extract_json_array_answer(cls, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""
        candidates: List[str] = [stripped]
        candidates.extend(item.strip() for item in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE))
        tail = re.sub(r"(?is)^.*?</think>\s*", "", stripped).strip()
        if tail and tail != stripped:
            candidates.append(tail)
        candidates.extend(item.strip() for item in re.findall(r"\[\s*\{[\s\S]*?\}\s*\]", stripped))

        for candidate in reversed(candidates):
            variants = [
                candidate,
                candidate.strip().strip("`"),
                candidate.replace('\\"', '"'),
                candidate.strip().strip('"').replace('\\"', '"'),
            ]
            for variant in variants:
                parsed = cls._loads_json_array(variant)
                if parsed is None:
                    continue
                if cls._looks_like_multiturn_answer(parsed):
                    return json.dumps(parsed, ensure_ascii=False)
        return ""

    @staticmethod
    def _loads_json_array(text: str) -> Optional[List[Any]]:
        value = str(text or "").strip()
        if not value:
            return None
        start = value.find("[")
        end = value.rfind("]")
        if start >= 0 and end > start:
            value = value[start : end + 1]
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, list) else None

    @staticmethod
    def _looks_like_multiturn_answer(parsed: List[Any]) -> bool:
        if not parsed or not all(isinstance(item, dict) for item in parsed):
            return False
        for item in parsed:
            if not (("描述编号" in item) or ("description_id" in item)):
                return False
            if "answer" not in item or not isinstance(item.get("answer"), list):
                return False
        return True

    @staticmethod
    def _parse_int_list_answer(text: str) -> Optional[list[int]]:
        stripped = str(text or "").strip()
        if not stripped:
            return None
        candidates = [stripped]
        candidates.extend(item.strip() for item in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE))
        candidates.extend(item.strip() for item in re.findall(r"\[[\s\d,，;；]+\]", stripped))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate.replace("，", ","))
            except Exception:
                continue
            if isinstance(parsed, list):
                try:
                    return [int(item) for item in parsed]
                except Exception:
                    continue
        if re.fullmatch(r"\s*\d+(?:\s*[,，;；]\s*\d+)*\s*", stripped):
            return [int(item) for item in re.findall(r"\d+", stripped)]
        return None

    @staticmethod
    def _run_single_row(module: Any, row: JsonRow) -> Tuple[JsonRow, JsonRow]:
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            summary = module.evaluation(handle.name)
            handle.seek(0)
            scored_lines = [json.loads(line) for line in handle if line.strip()]
        scored = scored_lines[0] if scored_lines else {}
        return summary if isinstance(summary, dict) else {}, scored

    @staticmethod
    def _score_from_scored_row(scored_row: JsonRow, official_summary: JsonRow) -> Optional[float]:
        raw = scored_row.get("score")
        if raw is None and isinstance(scored_row.get("eval_result"), dict):
            raw = scored_row["eval_result"].get("score")
        if raw is None:
            raw = official_summary.get("acc")
        try:
            return float(raw)
        except Exception:
            return None


class FinanceBenchEvaluator(BaseEvaluator):
    """FinanceBench LLM-as-judge scorer for closed-book and evidence settings."""

    name = "financebench_llm_judge"

    SYSTEM_PROMPT = (
        "You are a fair and strict judge for FinanceBench financial QA tasks.\n"
        "Your job is to decide whether the model response correctly answers the question, "
        "using the reference answer, reference justification, and provided evidence.\n"
        "Do not solve the task again from outside knowledge. Do not use web search.\n"
        "Return ONLY valid JSON. Do not wrap it in markdown."
    )

    USER_TEMPLATE = (
        "Please judge the model response for this FinanceBench task.\n\n"
        "[Evaluation Mode]\n{eval_mode}\n\n"
        "[Question]\n{question}\n\n"
        "[Reference Answer]\n{reference_answer}\n\n"
        "[Reference Justification]\n{reference_justification}\n\n"
        "[Gold Evidence Pages / Fragments / Financial Tables]\n{evidence}\n\n"
        "[Model Response]\n{predicted_answer}\n\n"
        "Grading rules:\n"
        "1. Judge only whether the model response answers the question. Do not use outside knowledge.\n"
        "2. Use the reference answer as the primary target and the evidence/justification to resolve ambiguity.\n"
        "3. Numeric answers must match the value, unit, sign, scale, fiscal period, and company. "
        "Accept harmless formatting differences and reasonable rounding.\n"
        "4. Text answers may use different wording, but must preserve all required entities, periods, and conclusions.\n"
        "5. Penalize missing final answers, wrong units, wrong years, wrong companies, unsupported conclusions, "
        "or contradictions with the evidence.\n"
        "6. For evidence_provided runs, do not require a citation if the final answer is correct, but penalize "
        "claims that contradict the supplied evidence.\n"
        "7. For closed_book runs, remember the model did not receive evidence; still grade the final answer "
        "against the reference answer and evidence.\n"
        "8. Award partial credit only when the response is materially close but incomplete.\n\n"
        "Return strict JSON with this schema:\n"
        "{{\n"
        "  \"rationale\": \"brief explanation of the judgement\",\n"
        "  \"verdict\": \"correct | partial | incorrect\",\n"
        "  \"score\": 0.0,\n"
        "  \"max_score\": 1.0,\n"
        "  \"normalized_score\": 0.0,\n"
        "  \"extracted_prediction\": \"the final answer you identify in the model response, or empty string if none\",\n"
        "  \"extracted_reference\": \"the normalized reference answer\",\n"
        "  \"evidence_checks\": [\n"
        "    {{\n"
        "      \"criterion\": \"important expected fact, number, unit, period, or evidence support\",\n"
        "      \"status\": \"met | partially_met | not_met | contradicted | not_applicable\",\n"
        "      \"reason\": \"short reason\"\n"
        "    }}\n"
        "  ]\n"
        "}}"
    )

    _model_cache: Any = None
    _model_loaded: bool = False

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        extra = _bench_extra(task_log)
        prediction = _unwrap_runtime_final_answer(_visible_result_content(task_log))
        golden = str(task_log.get("golden_answer") or extra.get("golden_answer") or extra.get("answer") or "")
        if not golden.strip():
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": "Missing FinanceBench gold answer.",
            }
        if not prediction:
            return {
                "evaluator": self.name,
                "mode": "llm",
                "financebench_eval_mode": extra.get("financebench_eval_mode"),
                "financebench_id": extra.get("financebench_id"),
                "question_type": extra.get("question_type"),
                "question_reasoning": extra.get("question_reasoning"),
                "protocol": self._protocol(extra),
                "judgement": "incorrect",
                "verdict": "incorrect",
                "score": 0.0,
                "normalized_score": 0.0,
                "max_score": 1.0,
                "is_correct": False,
                "needs_official_review": False,
                "rationale": "Empty visible answer content.",
                "extracted_prediction": "",
                "extracted_reference": golden,
                "evidence_checks": [],
                "prediction_source": {"source": "agent_result_content", "empty": True},
            }

        llm_result = self._llm_judge(
            eval_mode=str(extra.get("financebench_eval_mode") or "unknown"),
            question=str(extra.get("raw_question") or task_log.get("question") or ""),
            reference_answer=golden,
            reference_justification=str(extra.get("justification") or ""),
            evidence=self._format_evidence(extra.get("evidence")),
            predicted_answer=prediction,
        )
        if llm_result is None:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "mode": "llm",
                "financebench_eval_mode": extra.get("financebench_eval_mode"),
                "note": (
                    "FinanceBench LLM judge unavailable: configure FIRE_AGENT_* "
                    "(or FIRE_AGENT_FINANCEBENCH_JUDGE_MODEL / FIRE_AGENT_JUDGE_MODEL)."
                ),
                "protocol": self._protocol(extra),
            }

        return {
            "evaluator": self.name,
            "mode": "llm",
            "financebench_eval_mode": extra.get("financebench_eval_mode"),
            "financebench_id": extra.get("financebench_id"),
            "question_type": extra.get("question_type"),
            "question_reasoning": extra.get("question_reasoning"),
            "prediction_source": {"source": "agent_result_content"},
            "protocol": self._protocol(extra),
            **llm_result,
        }

    def _llm_judge(
        self,
        eval_mode: str,
        question: str,
        reference_answer: str,
        reference_justification: str,
        evidence: str,
        predicted_answer: str,
    ) -> Optional[JsonRow]:
        try:
            user_input = self.USER_TEMPLATE.format(
                eval_mode=eval_mode,
                question=question,
                reference_answer=reference_answer,
                reference_justification=reference_justification,
                evidence=evidence,
                predicted_answer=predicted_answer,
            )
        except Exception as exc:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "note": f"Failed to render FinanceBench judge prompt: {type(exc).__name__}: {exc}",
            }

        model = self._ensure_model()
        if model is None:
            return None

        try:
            max_retries = int(os.getenv("FIRE_AGENT_FINANCEBENCH_JUDGE_RETRIES", "2"))
        except Exception:
            max_retries = 2

        judge_text = ""
        judge_finish_reason = ""
        judge_usage: JsonRow = {}
        judge_reasoning_content = ""
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = model(
                    [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_input},
                    ]
                )
                judge_text = getattr(response, "content", None) or str(response)
                judge_finish_reason = str(getattr(response, "finish_reason", "") or "")
                raw_usage = getattr(response, "raw_usage", None)
                usage = getattr(response, "usage", None)
                if isinstance(raw_usage, dict) and raw_usage:
                    judge_usage = dict(raw_usage)
                elif isinstance(usage, dict):
                    judge_usage = dict(usage)
                judge_reasoning_content = str(getattr(response, "reasoning_content", "") or "")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                try:
                    import time

                    time.sleep(min(2**attempt, 5))
                except Exception:
                    pass
        if last_error is not None:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "judge_raw": judge_text,
                "judge_finish_reason": judge_finish_reason,
                "judge_usage": judge_usage,
                "judge_reasoning_content_len": len(judge_reasoning_content),
                "note": (
                    f"FinanceBench judge model call failed after {max_retries + 1} attempts: "
                    f"{type(last_error).__name__}: {last_error}"
                ),
            }

        parsed = self._parse_judge_output(judge_text)
        if parsed is None:
            return {
                "judgement": "incorrect",
                "verdict": "incorrect",
                "score": 0.0,
                "normalized_score": 0.0,
                "is_correct": False,
                "needs_official_review": True,
                "judge_raw": judge_text,
                "judge_finish_reason": judge_finish_reason,
                "judge_usage": judge_usage,
                "judge_reasoning_content_len": len(judge_reasoning_content),
                "note": "Judge response did not contain valid JSON.",
            }

        verdict = str(parsed.get("verdict") or parsed.get("judgement") or "").strip().lower()
        raw_score = parsed.get("normalized_score", parsed.get("score"))
        try:
            score = float(raw_score)
        except Exception:
            score = 1.0 if verdict == "correct" else 0.5 if verdict == "partial" else 0.0
        score = max(0.0, min(1.0, score))
        if verdict not in {"correct", "partial", "incorrect"}:
            verdict = "correct" if score >= 0.85 else "partial" if score > 0 else "incorrect"
        judgement = "correct" if verdict == "correct" else "incorrect"
        return {
            "judgement": judgement,
            "verdict": verdict,
            "score": score,
            "normalized_score": score,
            "max_score": 1.0,
            "is_correct": verdict == "correct",
            "needs_official_review": False,
            "rationale": str(parsed.get("rationale") or ""),
            "extracted_prediction": str(parsed.get("extracted_prediction") or ""),
            "extracted_reference": str(parsed.get("extracted_reference") or ""),
            "evidence_checks": parsed.get("evidence_checks") if isinstance(parsed.get("evidence_checks"), list) else [],
            "judge_raw": judge_text,
            "judge_parsed": parsed,
            "judge_finish_reason": judge_finish_reason,
            "judge_usage": judge_usage,
            "judge_reasoning_content_len": len(judge_reasoning_content),
        }

    @staticmethod
    def _parse_judge_output(text: Any) -> Optional[JsonRow]:
        if not isinstance(text, str) or not text.strip():
            return None
        stripped = text.strip()
        candidates: List[JsonRow] = []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
        for match in re.finditer(r"\{", stripped):
            try:
                parsed, _ = decoder.raw_decode(stripped[match.start() :])
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
        if not candidates:
            return None

        def is_real_judgement(candidate: JsonRow) -> bool:
            verdict = str(candidate.get("verdict") or candidate.get("judgement") or "").strip().lower()
            if verdict in {"correct", "partial", "incorrect"}:
                return True
            score = candidate.get("normalized_score", candidate.get("score"))
            try:
                value = float(score)
            except Exception:
                return False
            return 0.0 <= value <= 1.0

        real_candidates = [candidate for candidate in candidates if is_real_judgement(candidate)]
        if real_candidates:
            return real_candidates[-1]
        return candidates[-1]

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
        settings = cls._judge_model_settings(ModelSettings)
        if not settings.is_configured:
            cls._model_cache = None
            return None
        try:
            cls._model_cache = OpenAICompatibleChatModel(settings)
        except Exception:
            cls._model_cache = None
        return cls._model_cache

    @staticmethod
    def _judge_model_settings(model_settings_cls: Any) -> Any:
        base_settings = model_settings_cls.from_env()

        def env_int(*names: str, default: int) -> int:
            for name in names:
                value = os.getenv(name)
                if value not in (None, ""):
                    return int(value)
            return default

        def env_float(*names: str, default: float) -> float:
            for name in names:
                value = os.getenv(name)
                if value not in (None, ""):
                    return float(value)
            return default

        def env_json(*names: str, default: Dict[str, Any]) -> Dict[str, Any]:
            for name in names:
                value = os.getenv(name)
                if value not in (None, ""):
                    parsed = json.loads(value)
                    if not isinstance(parsed, dict):
                        raise ValueError(f"{name} must be a JSON object.")
                    return parsed
            return default

        return model_settings_cls(
            model=os.getenv("FIRE_AGENT_FINANCEBENCH_JUDGE_MODEL")
            or os.getenv("FIRE_AGENT_JUDGE_MODEL")
            or base_settings.model,
            api_key=os.getenv("FIRE_AGENT_FINANCEBENCH_JUDGE_API_KEY")
            or os.getenv("FIRE_AGENT_JUDGE_API_KEY")
            or base_settings.api_key,
            base_url=os.getenv("FIRE_AGENT_FINANCEBENCH_JUDGE_BASE_URL")
            or os.getenv("FIRE_AGENT_JUDGE_BASE_URL")
            or base_settings.base_url,
            timeout_seconds=env_int(
                "FIRE_AGENT_FINANCEBENCH_JUDGE_TIMEOUT_SECONDS",
                "FIRE_AGENT_JUDGE_TIMEOUT_SECONDS",
                default=base_settings.timeout_seconds,
            ),
            max_tokens=env_int(
                "FIRE_AGENT_FINANCEBENCH_JUDGE_MAX_TOKENS",
                "FIRE_AGENT_JUDGE_MAX_TOKENS",
                default=base_settings.max_tokens,
            ),
            temperature=env_float(
                "FIRE_AGENT_FINANCEBENCH_JUDGE_TEMPERATURE",
                "FIRE_AGENT_JUDGE_TEMPERATURE",
                default=base_settings.temperature,
            ),
            extra_body=env_json(
                "FIRE_AGENT_FINANCEBENCH_JUDGE_EXTRA_BODY_JSON",
                "FIRE_AGENT_JUDGE_EXTRA_BODY_JSON",
                default=base_settings.extra_body or {},
            ),
        )

    @classmethod
    def _format_evidence(cls, evidence: Any) -> str:
        if evidence in (None, ""):
            return ""
        if isinstance(evidence, list):
            return "\n\n".join(cls._format_one_evidence(item, idx + 1) for idx, item in enumerate(evidence)).strip()
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
            body = fragment or full_page
            if fragment and full_page and full_page != fragment:
                body = f"{fragment}\n\nFull page:\n{full_page}"
            return f"{header}\n{body}".strip()
        text = normalize_text(item)
        return f"[Evidence {index}]\n{text}" if text else ""

    @staticmethod
    def _protocol(extra: JsonRow) -> str:
        mode = extra.get("financebench_eval_mode") or "unknown"
        return (
            "FinanceBench LLM-as-judge answer grading. Runs are separated into "
            "closed_book (question only; no reports/evidence/retrieval) and "
            "evidence_provided (question plus gold evidence pages/fragments/tables). "
            f"Current row mode: {mode}."
        )
