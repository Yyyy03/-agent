from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import JsonRow, normalize_text, numeric_overlap
from .core import BaseEvaluator, _bench_extra

_LOGGER = logging.getLogger(__name__)


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


def _env_int_first(*names: str, default: int) -> int:
    value = _env_first(*names)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _env_float_first(*names: str, default: float) -> float:
    value = _env_first(*names)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _prefixed_judge_model_settings(model_settings_cls: Any, prefix: str) -> Any:
    model_override = _env_first(f"FIRE_AGENT_{prefix}_JUDGE_MODEL", "FIRE_AGENT_JUDGE_MODEL")
    base_settings = model_settings_cls.from_env(model_override=model_override)
    model = model_override or base_settings.model
    return model_settings_cls(
        model=model,
        api_key=(
            _env_first(f"FIRE_AGENT_{prefix}_JUDGE_API_KEY", "FIRE_AGENT_JUDGE_API_KEY")
            or base_settings.api_key
        ),
        base_url=(
            _env_first(f"FIRE_AGENT_{prefix}_JUDGE_BASE_URL", "FIRE_AGENT_JUDGE_BASE_URL")
            or base_settings.base_url
        ),
        timeout_seconds=_env_int_first(
            f"FIRE_AGENT_{prefix}_JUDGE_TIMEOUT_SECONDS",
            "FIRE_AGENT_JUDGE_TIMEOUT_SECONDS",
            default=base_settings.timeout_seconds,
        ),
        max_tokens=_env_int_first(
            f"FIRE_AGENT_{prefix}_JUDGE_MAX_TOKENS",
            "FIRE_AGENT_JUDGE_MAX_TOKENS",
            default=base_settings.max_tokens,
        ),
        temperature=_env_float_first(
            f"FIRE_AGENT_{prefix}_JUDGE_TEMPERATURE",
            "FIRE_AGENT_JUDGE_TEMPERATURE",
            default=base_settings.temperature,
        ),
        thinking_enabled=base_settings.thinking_enabled,
        reasoning_effort=base_settings.reasoning_effort,
    )


class ExactOrSubstringQAEvaluator(BaseEvaluator):
    name = "qa_exact_or_substring"

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        prediction = normalize_text(self._prediction(task_log))
        golden = normalize_text(self._golden(task_log))
        if not golden:
            return {"evaluator": self.name, "judgement": None, "score": None, "note": "Missing golden answer."}
        exact = prediction == golden
        contains = bool(prediction and golden and (prediction in golden or golden in prediction))
        number_score = numeric_overlap(prediction, golden)
        is_correct = exact or contains or number_score >= 0.8
        return {
            "evaluator": self.name,
            "judgement": "correct" if is_correct else "incorrect",
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "exact_match": exact,
            "substring_match": contains,
            "numeric_overlap": number_score,
        }


class FinGAIAEvaluator(BaseEvaluator):
    """FinGAIA LLM-as-judge evaluator (answer-equivalence judging).

    Replaces the prior rule-based scorer. The judge is invoked through the same
    OpenAI-compatible chat client the agent uses (configured via FIRE_AGENT_*).
    No heuristic fallback: scores come from the LLM judge or are absent.

    Protocol stays FinGAIA 1/0/-1 (correct/incorrect/unassessable) where
    -1 is only reported when the standard answer itself is missing.
    """

    name = "fingaia_official"

    SYSTEM_PROMPT = (
        "You are a fair and strict judge for FinGAIA financial QA tasks.\n"
        "Your only job is to decide whether the predicted answer is equivalent to the labeled answer for the original question.\n"
        "Do not solve the question again. Do not use outside knowledge.\n"
        "Return ONLY valid JSON. Do not wrap it in markdown."
    )

    USER_TEMPLATE = (
        "请判断【模型答案】是否与【标准答案】等价。\n\n"
        "【题目】\n{question}\n\n"
        "【标准答案】\n{reference_answer}\n\n"
        "【模型答案】\n{predicted_answer}\n\n"
        "判分规则：\n"
        "1. 只判断最终答案是否等价，不重新解题，不使用外部知识。\n"
        "2. 如果模型答案为空、明显没有给出最终答案、只有搜索片段/网页摘录/过程但没有明确结论，判 incorrect。\n"
        "3. 数值题：\n"
        "   - 单位和数量级必须等价。\n"
        "   - 接受合理格式差异，例如\u201c130亿元\u201d和\u201c130 亿\u201d。\n"
        "   - 对比例/百分比题，若题意支持，接受\u201c0.7203\u201d和\u201c72.03%\u201d这类等价表达。\n"
        "   - 无特殊误差说明时，数值应基本一致；轻微四舍五入可以接受。\n"
        "4. 文本题：\n"
        "   - 接受同义表达、全称/简称、中文标点或空格差异。\n"
        "   - 如果漏掉必要实体、给出多个互相矛盾答案，或答案范围明显不一致，判 incorrect。\n"
        "5. 不要因为答案解释很长而扣分，只要其中有清晰、唯一且正确的最终答案。\n"
        "6. 不要因为缺少推理过程而扣分。\n\n"
        "请输出严格 JSON，字段如下：\n"
        "{{\n"
        "  \"rationale\": \"简短说明判断依据\",\n"
        "  \"judgement\": \"correct 或 incorrect\",\n"
        "  \"extracted_prediction\": \"你从模型答案中识别出的最终答案；如果没有明确最终答案，写空字符串\",\n"
        "  \"extracted_reference\": \"标准答案的规范化表达\"\n"
        "}}"
    )

    PROTOCOL = "FinGAIA 1/0/-1: correct/incorrect/unassessable; aggregate as correct divided by all questions."

    _model_cache: Any = None
    _model_loaded: bool = False

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        official = self._official_judgement(task_log)
        if official:
            return official

        prediction_raw = self._prediction(task_log)
        golden_raw = self._golden(task_log)
        question = self._extract_question(task_log)

        if not golden_raw.strip():
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "official_score": None,
                "note": "Missing FinGAIA standard answer.",
                "protocol": self.PROTOCOL,
            }

        llm_result = self._llm_judge(question, golden_raw, prediction_raw)
        if llm_result is None:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "note": (
                    "Official LLM judge unavailable: configure FIRE_AGENT_* "
                    "(or FIRE_AGENT_FINGAIA_JUDGE_MODEL) to enable scoring."
                ),
                "protocol": self.PROTOCOL,
            }

        return {
            "evaluator": self.name,
            "protocol": self.PROTOCOL,
            **llm_result,
        }

    def _llm_judge(self, question: str, reference: str, prediction: str) -> Optional[JsonRow]:
        try:
            user_input = self.USER_TEMPLATE.format(
                question=question or "",
                reference_answer=reference or "",
                predicted_answer=prediction or "",
            )
        except Exception as exc:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "note": f"Failed to render judge prompt: {type(exc).__name__}: {exc}",
            }

        model = self._ensure_model()
        if model is None:
            return None

        try:
            max_retries = int(os.getenv("FIRE_AGENT_FINGAIA_JUDGE_RETRIES", "2"))
        except Exception:
            max_retries = 2

        judge_text = ""
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = model([
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ])
                judge_text = getattr(response, "content", None) or str(response)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                try:
                    import time
                    time.sleep(min(2 ** attempt, 5))
                except Exception:
                    pass
        if last_error is not None:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "note": (
                    f"Judge model call failed after {max_retries + 1} attempts: "
                    f"{type(last_error).__name__}: {last_error}"
                ),
            }

        parsed = self._parse_judge_output(judge_text)
        if parsed is None:
            return {
                "judgement": "incorrect",
                "score": 0.0,
                "official_score": 0,
                "is_correct": False,
                "needs_official_review": True,
                "mode": "llm",
                "judge_raw": judge_text,
                "note": "Judge response did not contain valid JSON.",
            }

        judgement_value = str(parsed.get("judgement") or "").strip().lower()
        if judgement_value not in {"correct", "incorrect"}:
            return {
                "judgement": "incorrect",
                "score": 0.0,
                "official_score": 0,
                "is_correct": False,
                "needs_official_review": True,
                "mode": "llm",
                "judge_raw": judge_text,
                "judge_parsed": parsed,
                "note": f"Judge returned unrecognized judgement: {parsed.get('judgement')!r}.",
            }

        is_correct = judgement_value == "correct"
        return {
            "judgement": judgement_value,
            "score": 1.0 if is_correct else 0.0,
            "official_score": 1 if is_correct else 0,
            "is_correct": is_correct,
            "needs_official_review": False,
            "mode": "llm",
            "rationale": str(parsed.get("rationale") or ""),
            "extracted_prediction": str(parsed.get("extracted_prediction") or ""),
            "extracted_reference": str(parsed.get("extracted_reference") or ""),
            "judge_raw": judge_text,
        }

    @staticmethod
    def _parse_judge_output(text: Any) -> Optional[JsonRow]:
        if not isinstance(text, str) or not text.strip():
            return None
        stripped = text.strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        candidates = list(_iter_json_candidates(stripped))
        for candidate in candidates:
            if "judgement" in candidate:
                return candidate
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _extract_question(task_log: JsonRow) -> str:
        extra = _bench_extra(task_log)
        full_prompt = str(extra.get("question") or task_log.get("question") or "")
        if not full_prompt:
            return ""
        match = re.search(
            r"Question:\s*(.*?)(?:\n\s*(?:Scenario depth|Financial scenario)\s*[:：]|\Z)",
            full_prompt,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return full_prompt.strip()

    def _official_judgement(self, task_log: JsonRow) -> JsonRow | None:
        # Re-judge fast-path: if the row already carries a sealed official
        # 1/0/-1 verdict, just echo it. ``score`` is the new home; legacy
        # ``metrics`` / ``judge_output`` rows still work because the runtime
        # never injects them on a fresh run.
        sources = [
            task_log.get("score") or {},
            task_log.get("metrics") or {},
            task_log.get("judge_output") or {},
            _bench_extra(task_log),
        ]
        for source in sources:
            if not isinstance(source, dict):
                continue
            score = source.get("official_score")
            if score is None:
                continue
            try:
                official_score = int(float(score))
            except Exception:
                continue
            if official_score not in {-1, 0, 1}:
                continue
            judgement = source.get("judgement")
            if judgement not in {"correct", "incorrect", "unassessable"}:
                judgement = "correct" if official_score == 1 else "incorrect" if official_score == 0 else "unassessable"
            return {
                "evaluator": self.name,
                "judgement": judgement,
                "score": float(official_score),
                "official_score": official_score,
                "is_correct": True if official_score == 1 else False if official_score == 0 else None,
                "needs_official_review": False,
                "source": "provided_official_judgement",
                "protocol": self.PROTOCOL,
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
        settings = _prefixed_judge_model_settings(ModelSettings, "FINGAIA")
        if not settings.is_configured:
            cls._model_cache = None
            return None
        try:
            cls._model_cache = OpenAICompatibleChatModel(settings)
        except Exception:
            cls._model_cache = None
        return cls._model_cache


def _iter_json_candidates(text: str) -> Iterable[JsonRow]:
    """Yield JSON object candidates embedded in free-form judge output."""
    if not isinstance(text, str) or not text.strip():
        return
    decoder = json.JSONDecoder()
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                yield parsed
        except Exception:
            continue
    for match in re.finditer(r"\{", text):
        fragment = text[match.start():]
        try:
            parsed, _ = decoder.raw_decode(fragment)
        except Exception:
            continue
        if isinstance(parsed, dict):
            yield parsed


def _unwrap_score_value(value: Any) -> Optional[float]:
    while isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str):
        if value.strip().lower() in {"null", "none", "nan", ""}:
            return None
        value = value.strip()
    try:
        score = float(value)
    except Exception:
        return None
    if score in (0.0, 1.0):
        return score
    return score


_JUDGE_NULL_PATTERN = re.compile(
    r'"\s*(?:answer_score|score|final_score)\s*"\s*:\s*"?\s*null\s*"?',
    re.IGNORECASE,
)


def _judge_text_says_null(judge_output: Any) -> bool:
    """Detect the official ``{"score":"null"}`` verdict the FinSearchComp T1
    prompt asks for when realtime ground truth is missing."""

    if isinstance(judge_output, dict):
        for key in ("answer_score", "score", "final_score"):
            if key in judge_output:
                value = judge_output[key]
                while isinstance(value, list) and value:
                    value = value[0]
                if value is None:
                    return True
                if isinstance(value, str) and value.strip().lower() in {
                    "null",
                    "none",
                    "nan",
                }:
                    return True
        return False
    text = str(judge_output or "")
    return bool(_JUDGE_NULL_PATTERN.search(text))


def _parse_finsearchcomp_judge_score(judge_output: Any) -> Optional[float]:
    """Reimplements the official parse_judge_score logic for 0/1 scoring."""
    if isinstance(judge_output, dict):
        for key in ("answer_score", "score", "final_score"):
            if key in judge_output:
                return _unwrap_score_value(judge_output[key])
        return None
    text = str(judge_output or "")
    for obj in _iter_json_candidates(text):
        score = _parse_finsearchcomp_judge_score(obj)
        if score is not None:
            return score
    for pattern in (
        r'"answer_score"\s*:\s*([01])',
        r'"score"\s*:\s*([01])',
        r"final\s*score\s*[:：]\s*([01])",
        r"分数\s*[:：]?\s*([01])",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AKSHARE_VERSION_PATH = (
    _PROJECT_ROOT
    / "official_benchmarks"
    / "finsearchcomp"
    / "data"
    / "raw"
    / "finsearchcomp_akshare_version.json"
)
_DEFAULT_SNAPSHOT_DIR = (
    _PROJECT_ROOT
    / "official_benchmarks"
    / "finsearchcomp"
    / "data"
    / "market_snapshots"
)
_DEFAULT_LEGACY_MARKET_DATA = (
    _PROJECT_ROOT
    / "official_benchmarks"
    / "finsearchcomp"
    / "official_eval"
    / "legacy_market_data.py"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_finsearchcomp_date(value: Any) -> Optional[_dt.date]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _first_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_lookup_text(value: Any) -> str:
    scalar = _first_scalar(value)
    return str(scalar).strip() if scalar not in (None, "") else ""


def _make_finsearchcomp_row_key(metadata: JsonRow) -> str:
    sample_id = _normalize_lookup_text(metadata.get("sample_id"))
    if sample_id:
        return f"sample_id:{sample_id}"
    source_row_index = _normalize_lookup_text(metadata.get("source_row_index"))
    if source_row_index:
        return f"source_row_index:{source_row_index}"
    prompt_id = _normalize_lookup_text(metadata.get("prompt_id"))
    prompt = _normalize_lookup_text(metadata.get("prompt") or metadata.get("question"))
    return f"prompt_id:{prompt_id}|prompt:{prompt}"


class _FinSearchCompT1Refresher:
    """Resolve benchmark-bounded FinSearchComp T1 ground truth from snapshots.

    Behavior:
    - Loads the akshare-friendly raw release (`finsearchcomp_akshare_version.json`)
      once and indexes it by row-level identities (`sample_id` / `source_row_index`)
      before falling back to prompt text. This avoids collisions from duplicate
      ``prompt_id`` values in the benchmark release.
    - Uses the sample's benchmark ``time`` as an upper bound. For historical
      benchmark dates we must not refresh today's live market data, because that
      would exceed the benchmark cutoff.
    - Per-row quotes are cached for the lifetime of the process so repeated
      rescoring or N-way judging only resolves snapshot data once.
    - When compliant snapshot data cannot be obtained, returns ``None`` so the
      evaluator can keep the dataset's static ground truth instead of replacing
      it with out-of-window data.
    """

    _instance: Optional["_FinSearchCompT1Refresher"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._market_module: Any = None
        self._market_module_attempted = False
        self._refresh_attempted_for_dates: set[Optional[_dt.date]] = set()
        self._refresh_ok = False
        self._enrichment_by_row_key: Dict[str, JsonRow] = {}
        self._enrichment_by_prompt_id: Dict[str, List[JsonRow]] = {}
        self._enrichment_loaded = False
        self._quote_cache: Dict[str, Optional[str]] = {}

        self._enabled = _env_bool("FIRE_AGENT_FINSEARCHCOMP_REFRESH_T1", True)
        self._refresh_today = _env_bool(
            "FIRE_AGENT_FINSEARCHCOMP_REFRESH_SNAPSHOTS", True
        )
        self._snapshot_dir = Path(
            os.getenv(
                "FIRE_AGENT_FINSEARCHCOMP_SNAPSHOT_DIR",
                str(_DEFAULT_SNAPSHOT_DIR),
            )
        )
        self._akshare_version_path = Path(
            os.getenv(
                "FIRE_AGENT_FINSEARCHCOMP_AKSHARE_DATA",
                str(_DEFAULT_AKSHARE_VERSION_PATH),
            )
        )
        self._legacy_module_path = Path(
            os.getenv(
                "FIRE_AGENT_FINSEARCHCOMP_LEGACY_MARKET_DATA",
                str(_DEFAULT_LEGACY_MARKET_DATA),
            )
        )

    @classmethod
    def instance(cls) -> "_FinSearchCompT1Refresher":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load_legacy_module(self) -> Any:
        if self._market_module_attempted:
            return self._market_module
        self._market_module_attempted = True
        if not self._legacy_module_path.is_file():
            _LOGGER.warning(
                "FinSearchComp T1 refresher: legacy_market_data not found at %s",
                self._legacy_module_path,
            )
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "_finsearchcomp_legacy_market_data", str(self._legacy_module_path)
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._market_module = module
            try:
                module.set_snapshot_directory(str(self._snapshot_dir))
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - environment issue
            _LOGGER.warning(
                "FinSearchComp T1 refresher: failed to import legacy_market_data: %s",
                exc,
            )
            self._market_module = None
        return self._market_module

    def _maybe_refresh_for_date(self, target_date: Optional[_dt.date]) -> None:
        if not self._refresh_today:
            return
        if target_date in self._refresh_attempted_for_dates:
            return
        self._refresh_attempted_for_dates.add(target_date)
        module = self._load_legacy_module()
        if module is None:
            return
        today = _dt.datetime.utcnow().date()
        if target_date is not None and target_date < today:
            _LOGGER.info(
                "FinSearchComp T1 refresher: benchmark cutoff %s is historical; "
                "skipping live akshare refresh and using snapshots/static data only.",
                target_date.isoformat(),
            )
            return
        try:
            latest_date = self._latest_snapshot_date()
        except Exception:
            latest_date = None
        if latest_date == today:
            self._refresh_ok = True
            return
        try:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            module.save_universe_dataframes(
                str(self._snapshot_dir),
                ttl_sec=60,
                force_refresh=True,
                file_format="csv",
                timestamped=True,
            )
            self._refresh_ok = True
            _LOGGER.info(
                "FinSearchComp T1 refresher: refreshed market snapshots into %s",
                self._snapshot_dir,
            )
        except Exception as exc:
            _LOGGER.warning(
                "FinSearchComp T1 refresher: snapshot refresh failed (using stale "
                "snapshots in %s): %s",
                self._snapshot_dir,
                exc,
            )

    def _latest_snapshot_date(self) -> Optional[_dt.date]:
        if not self._snapshot_dir.is_dir():
            return None
        pattern = re.compile(r"_(\d{8})T\d+Z\.(csv|json)$", re.IGNORECASE)
        latest: Optional[_dt.date] = None
        for entry in self._snapshot_dir.iterdir():
            if not entry.is_file():
                continue
            match = pattern.search(entry.name)
            if not match:
                continue
            try:
                candidate = _dt.datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            if latest is None or candidate > latest:
                latest = candidate
        return latest

    def _load_enrichment(self) -> None:
        if self._enrichment_loaded:
            return
        self._enrichment_loaded = True
        if not self._akshare_version_path.is_file():
            _LOGGER.info(
                "FinSearchComp T1 refresher: akshare enrichment file not found at %s",
                self._akshare_version_path,
            )
            return
        try:
            with self._akshare_version_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            _LOGGER.warning(
                "FinSearchComp T1 refresher: failed to read enrichment file: %s",
                exc,
            )
            return
        if not isinstance(data, list):
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = _normalize_lookup_text(item.get("prompt_id"))
            if not pid or "T1" not in str(pid):
                continue
            row_key = _make_finsearchcomp_row_key(item)
            self._enrichment_by_row_key[row_key] = item
            self._enrichment_by_prompt_id.setdefault(pid, []).append(item)

    def _find_enrichment(self, metadata: JsonRow) -> Optional[JsonRow]:
        self._load_enrichment()

        direct = self._enrichment_by_row_key.get(_make_finsearchcomp_row_key(metadata))
        if direct:
            return direct

        prompt_id = _normalize_lookup_text(metadata.get("prompt_id"))
        if not prompt_id:
            return None
        candidates = self._enrichment_by_prompt_id.get(prompt_id, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        prompt = _normalize_lookup_text(metadata.get("prompt") or metadata.get("question"))
        if prompt:
            for item in candidates:
                candidate_prompt = _normalize_lookup_text(item.get("prompt") or item.get("question"))
                if candidate_prompt == prompt:
                    return item
        return None

    def _resolve_tickers(
        self, metadata: JsonRow
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (akshare_ticker, tags, method) by prefering metadata, then
        falling back to row-level enrichment."""

        akshare_ticker = metadata.get("akshare_ticker") or ""
        tags = metadata.get("tags") or metadata.get("tag") or ""
        method = metadata.get("method") or ""

        if (
            akshare_ticker
            and str(akshare_ticker).strip().upper() not in {"", "N/A", "NA", "NONE"}
            and tags
        ):
            return str(akshare_ticker), str(tags), (str(method) if method else None)

        enriched = self._find_enrichment(metadata)
        if not enriched:
            return None, None, None
        akshare_ticker = enriched.get("akshare_ticker") or akshare_ticker
        tags = enriched.get("tags") or enriched.get("tag") or tags
        method = enriched.get("method") or method
        if not akshare_ticker or str(akshare_ticker).strip().upper() in {"N/A", "NA", "NONE"}:
            return None, None, None
        if not tags:
            return None, None, None
        return str(akshare_ticker), str(tags), (str(method) if method else None)

    def refresh(self, metadata: JsonRow) -> Optional[str]:
        """Return a JSON-serialized realtime quote string for the T1 prompt,
        or ``None`` if realtime data cannot be obtained.

        Safe to call concurrently; results are cached by prompt_id.
        """

        if not self._enabled:
            return None
        prompt_id = _normalize_lookup_text(metadata.get("prompt_id"))
        if not prompt_id or "T1" not in prompt_id:
            return None
        row_key = _make_finsearchcomp_row_key(metadata)
        target_date = _parse_finsearchcomp_date(metadata.get("time"))
        cache_key = f"{row_key}|date:{target_date.isoformat() if target_date else 'latest'}"
        if cache_key in self._quote_cache:
            return self._quote_cache[cache_key]

        with self._lock:
            if cache_key in self._quote_cache:
                return self._quote_cache[cache_key]

            self._maybe_refresh_for_date(target_date)
            module = self._load_legacy_module()
            if module is None:
                self._quote_cache[cache_key] = None
                return None

            akshare_ticker, tags, method = self._resolve_tickers(metadata)
            if not akshare_ticker or not tags:
                _LOGGER.info(
                    "FinSearchComp T1 refresher: no akshare metadata for %s; "
                    "scoring will be marked unassessable.",
                    prompt_id,
                )
                self._quote_cache[cache_key] = None
                return None

            try:
                quote = module.fetch_single(
                    akshare_ticker,
                    tags,
                    method,
                    as_of_date=target_date,
                )
            except Exception as exc:
                _LOGGER.warning(
                    "FinSearchComp T1 refresher: fetch_single failed for %s "
                    "(ticker=%s tags=%s method=%s): %s",
                    prompt_id,
                    akshare_ticker,
                    tags,
                    method,
                    exc,
                )
                self._quote_cache[cache_key] = None
                return None

            if quote in (None, {}, []):
                self._quote_cache[cache_key] = None
                return None

            try:
                rendered = json.dumps(quote, ensure_ascii=False, default=str)
            except Exception:
                rendered = str(quote)
            self._quote_cache[cache_key] = rendered
            return rendered


class FinSearchCompEvaluator(BaseEvaluator):
    """Official FinSearchComp LLM-as-Judge evaluator.

    Uses the per-row ``judge_prompt_template`` and ``judge_system_prompt`` from the
    ByteSeedXpert/FinSearchComp release. The judge is invoked through the same
    OpenAI-compatible chat client the agent uses (configured via FIRE_AGENT_* env).

    Behavior:
    - If judge prompts are present and the judge model is configured, build the
      official user input, call the judge, and parse a 0/1 score.
    - For T1 (Time-Sensitive) rows we only substitute ``ground_truth`` when we
      can resolve a quote that does not exceed the benchmark's own ``time``
      field. Historical benchmark rows must not be refreshed with today's data.
      If compliant snapshot data is unavailable, we keep the dataset's static
      ground truth instead of replacing it with out-of-window data.
    - Otherwise return a structured ``judge_unavailable`` record so downstream
      aggregation can flag the row for re-evaluation. There is no heuristic
      fallback: scores either come from the official judge or are absent.
    """

    name = "finsearchcomp_official"

    _model_cache: Any = None
    _model_loaded: bool = False

    @staticmethod
    def _is_t1(metadata: JsonRow) -> bool:
        prompt_id = metadata.get("prompt_id")
        if isinstance(prompt_id, list):
            prompt_id = prompt_id[0] if prompt_id else ""
        if prompt_id and "T1" in str(prompt_id):
            return True
        label = str(metadata.get("label") or "")
        return ("Time-Sensitive" in label) or ("Time_Sensitive" in label)

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        prediction = self._prediction(task_log)
        golden = self._golden(task_log)
        metadata = _bench_extra(task_log)

        if not golden and not metadata.get("response_reference"):
            return {"evaluator": self.name, "judgement": None, "score": None, "note": "Missing reference answer."}

        judge_template = metadata.get("judge_prompt_template")
        judge_system = metadata.get("judge_system_prompt")
        if not (judge_template and judge_system):
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": "Missing judge_prompt_template or judge_system_prompt; cannot run official LLM judge.",
            }

        is_t1 = self._is_t1(metadata)
        ground_truth_source = "static_dataset"
        if is_t1:
            refresher = _FinSearchCompT1Refresher.instance()
            refreshed = refresher.refresh(metadata) if refresher.enabled else None
            metadata = dict(metadata)
            if refreshed:
                metadata["ground_truth"] = refreshed
                ground_truth_source = "akshare_bounded_snapshot"
            elif not metadata.get("ground_truth"):
                ground_truth_source = "unavailable"

        llm_result = self._llm_judge(prediction, metadata, judge_template, judge_system)
        if llm_result is None:
            return {
                "evaluator": self.name,
                "judgement": None,
                "score": None,
                "note": (
                    "Official LLM judge unavailable: configure FIRE_AGENT_* (or "
                    "FIRE_AGENT_FINSEARCHCOMP_JUDGE_MODEL) to enable scoring."
                ),
            }

        if is_t1:
            llm_result.setdefault("task_type", "T1")
            llm_result["ground_truth_source"] = ground_truth_source

        return {"evaluator": self.name, **llm_result}

    def _llm_judge(
        self,
        prediction: str,
        metadata: JsonRow,
        judge_template: str,
        judge_system: str,
    ) -> Optional[JsonRow]:
        try:
            user_input = self._build_user_input(judge_template, prediction, metadata)
        except Exception as exc:
            return {
                "judgement": None,
                "score": None,
                "note": f"Failed to render judge prompt: {type(exc).__name__}: {exc}",
            }

        model = self._ensure_model()
        if model is None:
            return None

        try:
            max_retries = int(os.getenv("FIRE_AGENT_FINSEARCHCOMP_JUDGE_RETRIES", "2"))
        except Exception:
            max_retries = 2

        judge_text = ""
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = model([
                    {"role": "system", "content": str(judge_system)},
                    {"role": "user", "content": user_input},
                ])
                judge_text = getattr(response, "content", None) or str(response)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                try:
                    import time
                    time.sleep(min(2 ** attempt, 5))
                except Exception:
                    pass
        if last_error is not None:
            return {
                "judgement": None,
                "score": None,
                "note": f"Judge model call failed after {max_retries + 1} attempts: "
                f"{type(last_error).__name__}: {last_error}",
            }

        score = _parse_finsearchcomp_judge_score(judge_text)
        if score is None:
            # Distinguish "judge explicitly said null" (data was missing — per
            # the official T1 prompt rule) from "we couldn't parse anything"
            # (judge output malformed). The former is unassessable; the latter
            # is conservatively treated as incorrect to preserve old behavior.
            if _judge_text_says_null(judge_text):
                return {
                    "judgement": "unassessable",
                    "score": None,
                    "is_correct": None,
                    "mode": "llm",
                    "judge_raw": judge_text,
                    "note": (
                        "Judge returned {\"score\":\"null\"} (real-time ground "
                        "truth unavailable for this T1 sample); excluded from "
                        "accuracy."
                    ),
                }
            return {
                "judgement": "incorrect",
                "score": 0.0,
                "is_correct": False,
                "mode": "llm",
                "judge_raw": judge_text,
                "note": "Judge response did not contain a parseable 0/1 score.",
            }
        is_correct = score >= 1.0
        return {
            "judgement": "correct" if is_correct else "incorrect",
            "score": float(score),
            "is_correct": is_correct,
            "mode": "llm",
            "judge_raw": judge_text,
        }

    @staticmethod
    def _build_user_input(template: str, response: str, metadata: JsonRow) -> str:
        values = {
            "prompt": str(metadata.get("prompt") or metadata.get("question") or ""),
            "response_reference": str(metadata.get("response_reference") or metadata.get("golden_answer") or ""),
            "ground_truth": str(metadata.get("ground_truth") or ""),
            "response": str(response or ""),
        }
        try:
            return str(template).format(**values)
        except KeyError as exc:
            values[exc.args[0]] = ""
            return str(template).format(**values)

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
        settings = _prefixed_judge_model_settings(ModelSettings, "FINSEARCHCOMP")
        if not settings.is_configured:
            cls._model_cache = None
            return None
        try:
            cls._model_cache = OpenAICompatibleChatModel(settings)
        except Exception:
            cls._model_cache = None
        return cls._model_cache


class FinanceAgentBenchEvaluator(BaseEvaluator):
    """FinanceAgentBench LLM-as-judge evaluator (rubric + contradiction).

    Aligns with the upstream Finance Agent Benchmark protocol described in
    Vals AI's paper (arXiv:2508.00828):

    - Each question carries a list of ``correctness`` rubric criteria written
      by domain experts (initially extracted via GPT-4o, then peer-reviewed).
    - For every criterion the judge LLM decides whether the agent's final
      answer satisfies it semantically (accepting rounded numbers, paraphrase,
      format / writing-style differences).
    - A separate contradiction check looks for factual conflicts between the
      agent's answer and the expert reference.
    - The conjunction rule: ``is_correct`` iff every rubric item matched AND
      no contradiction was detected.

    Behavior:

    - When a judge model is configured (``FIRE_AGENT_FINANCEAGENT_JUDGE_*``
      / ``FIRE_AGENT_JUDGE_*`` / default ``FIRE_AGENT_*``) the LLM judge is the
      source of truth. Judge-specific API settings can point at a different
      OpenAI-compatible endpoint than the agent's main model.
    - When no judge is configured, the evaluator returns ``judge_unavailable``
      without assigning a score.
    """

    name = "financeagentbench_llm_judge"

    SYSTEM_PROMPT = (
        "You are a strict but fair judge for the Finance Agent Benchmark.\n"
        "You will be given a financial question, an expert reference answer, "
        "the agent's final answer, and a list of rubric criteria that the "
        "expert answer is expected to satisfy.\n"
        "Decide, for EACH rubric criterion, whether the agent's final answer "
        "satisfies that criterion. Also decide whether the agent's answer "
        "contradicts the expert reference on any specific fact.\n"
        "Do NOT solve the question yourself. Do NOT use outside knowledge.\n"
        "Return ONLY valid JSON. No prose. No markdown fences."
    )

    USER_TEMPLATE = (
        "[Question]\n{question}\n\n"
        "[Expert reference answer]\n{reference}\n\n"
        "[Agent's final answer]\n{prediction}\n\n"
        "[Rubric criteria]\n{rubric_block}\n\n"
        "Judging rules:\n"
        "1. A criterion is matched if the agent answer conveys the same "
        "substantive content (numbers, names, dates, relationships) as the "
        "criterion. Format, writing style, ordering, and length must not "
        "affect this judgment.\n"
        "2. Numerical figures: accept reasonable rounding (e.g., $5.23B vs "
        "$5,234,183,000; 20.3% vs ~20%). Units and order of magnitude must "
        "match. If the criterion implies a precision, respect it.\n"
        "3. Names / tickers / dates: substance must match (full name vs "
        "ticker / abbreviation is fine; wrong company or wrong period is not).\n"
        "4. Missing information for a criterion -> matched=false.\n"
        "5. Empty answer, search snippet only, or no clear final answer "
        "-> every criterion matched=false.\n"
        "6. Contradiction: flag detected=true only when the agent makes a "
        "confident factual claim that directly disagrees with the expert "
        "reference (different precise numbers, opposite direction, wrong "
        "entity, wrong period). A criterion mismatch alone is NOT a "
        "contradiction.\n\n"
        "Output JSON schema (no extra keys, no markdown):\n"
        "{{\n"
        '  "criteria_results": [\n'
        '    {{"index": <int starting at 0>, "matched": true|false, "rationale": "<short>"}},\n'
        "    ... one entry per rubric criterion, same order as input ...\n"
        "  ],\n"
        '  "contradiction": {{"detected": true|false, "rationale": "<short>"}},\n'
        '  "overall_rationale": "<short>"\n'
        "}}"
    )

    PROTOCOL = (
        "FinanceAgentBench LLM-as-judge: per-criterion satisfies_statement + "
        "contradiction check; conjunction rule for final correctness."
    )

    _model_cache: Any = None
    _model_loaded: bool = False

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        metadata = _bench_extra(task_log)
        rubric = metadata.get("rubric") or []
        correctness_items = [
            item for item in rubric if isinstance(item, dict) and item.get("operator") == "correctness"
        ]
        prediction = str(self._prediction(task_log) or "")
        reference = str(self._golden(task_log) or metadata.get("golden_answer") or "")
        question = str(task_log.get("question") or metadata.get("question") or "")

        if not correctness_items:
            fallback = ExactOrSubstringQAEvaluator()
            result = fallback.evaluate(task_log)
            result["evaluator"] = self.name
            result["note"] = "No rubric correctness items found; used QA fallback."
            result["protocol"] = self.PROTOCOL
            return result

        llm_result = self._llm_judge(question, reference, prediction, correctness_items)
        if llm_result is not None:
            return {"evaluator": self.name, "protocol": self.PROTOCOL, **llm_result}

        return {
            "evaluator": self.name,
            "protocol": self.PROTOCOL,
            "judgement": None,
            "score": None,
            "is_correct": None,
            "mode": "judge_unavailable",
            "note": (
                "LLM judge unavailable: configure FIRE_AGENT_* or "
                "FIRE_AGENT_FINANCEAGENT_JUDGE_* to enable rubric judging."
            ),
        }

    def _llm_judge(
        self,
        question: str,
        reference: str,
        prediction: str,
        correctness_items: List[Dict[str, Any]],
    ) -> Optional[JsonRow]:
        model = self._ensure_model()
        if model is None:
            return None

        rubric_lines = []
        for idx, item in enumerate(correctness_items):
            criteria = str(item.get("criteria") or "").strip()
            rubric_lines.append(f"{idx}. {criteria}")
        rubric_block = "\n".join(rubric_lines)

        try:
            user_input = self.USER_TEMPLATE.format(
                question=question or "",
                reference=reference or "",
                prediction=prediction or "",
                rubric_block=rubric_block or "",
            )
        except Exception as exc:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "rubric_total": len(correctness_items),
                "note": f"Failed to render judge prompt: {type(exc).__name__}: {exc}",
            }

        try:
            max_retries = int(os.getenv("FIRE_AGENT_FINANCEAGENT_JUDGE_RETRIES", "2"))
        except Exception:
            max_retries = 2

        judge_text = ""
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = model([
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ])
                judge_text = getattr(response, "content", None) or str(response)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                try:
                    import time

                    time.sleep(min(2 ** attempt, 5))
                except Exception:
                    pass
        if last_error is not None:
            return {
                "judgement": None,
                "score": None,
                "needs_official_review": True,
                "rubric_total": len(correctness_items),
                "note": (
                    f"Judge model call failed after {max_retries + 1} attempts: "
                    f"{type(last_error).__name__}: {last_error}"
                ),
            }

        parsed = self._parse_judge_output(judge_text)
        if not isinstance(parsed, dict):
            return {
                "judgement": "incorrect",
                "score": 0.0,
                "rubric_score": 0.0,
                "is_correct": False,
                "mode": "llm",
                "rubric_total": len(correctness_items),
                "rubric_hits": 0,
                "judge_raw": judge_text,
                "note": "Judge response did not contain valid JSON.",
            }

        criteria_results = self._merge_criteria_results(
            correctness_items, parsed.get("criteria_results")
        )
        hits = sum(1 for c in criteria_results if c.get("matched"))
        total = len(correctness_items)
        rubric_score = hits / total if total else 0.0
        contradiction_raw = parsed.get("contradiction") or {}
        contradiction_detected = bool(contradiction_raw.get("detected"))
        all_matched = total > 0 and hits == total
        is_correct = all_matched and not contradiction_detected

        return {
            "judgement": "correct" if is_correct else "incorrect",
            "score": 1.0 if is_correct else 0.0,
            "rubric_score": rubric_score,
            "is_correct": is_correct,
            "mode": "llm",
            "rubric_hits": hits,
            "rubric_total": total,
            "criteria_results": criteria_results,
            "contradiction": {
                "detected": contradiction_detected,
                "rationale": str(contradiction_raw.get("rationale") or ""),
            },
            "rationale": str(parsed.get("overall_rationale") or ""),
            "judge_raw": judge_text,
        }

    @staticmethod
    def _merge_criteria_results(
        correctness_items: List[Dict[str, Any]],
        raw_results: Any,
    ) -> List[Dict[str, Any]]:
        """Align judge output entries with the requested rubric order."""
        items_by_index: Dict[int, Dict[str, Any]] = {}
        if isinstance(raw_results, list):
            for pos, entry in enumerate(raw_results):
                if not isinstance(entry, dict):
                    continue
                try:
                    idx = int(entry.get("index"))
                except Exception:
                    idx = pos
                if idx < 0 or idx >= len(correctness_items):
                    idx = pos
                if 0 <= idx < len(correctness_items) and idx not in items_by_index:
                    items_by_index[idx] = entry
        merged: List[Dict[str, Any]] = []
        for idx, rubric_item in enumerate(correctness_items):
            entry = items_by_index.get(idx) or {}
            merged.append(
                {
                    "index": idx,
                    "criteria": rubric_item.get("criteria"),
                    "matched": bool(entry.get("matched", False)),
                    "rationale": str(entry.get("rationale") or ""),
                }
            )
        return merged

    @staticmethod
    def _parse_judge_output(text: Any) -> Optional[JsonRow]:
        if not isinstance(text, str) or not text.strip():
            return None
        stripped = text.strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        candidates = list(_iter_json_candidates(stripped))
        for candidate in candidates:
            if "criteria_results" in candidate or "contradiction" in candidate:
                return candidate
        if candidates:
            return candidates[0]
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
        model = (
            os.getenv("FIRE_AGENT_FINANCEAGENT_JUDGE_MODEL")
            or os.getenv("FIRE_AGENT_JUDGE_MODEL")
            or base_settings.model
        )
        api_key = (
            os.getenv("FIRE_AGENT_FINANCEAGENT_JUDGE_API_KEY")
            or os.getenv("FIRE_AGENT_JUDGE_API_KEY")
            or base_settings.api_key
        )
        base_url = (
            os.getenv("FIRE_AGENT_FINANCEAGENT_JUDGE_BASE_URL")
            or os.getenv("FIRE_AGENT_JUDGE_BASE_URL")
            or base_settings.base_url
        )
        return model_settings_cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=FinanceAgentBenchEvaluator._env_int(
                "FIRE_AGENT_FINANCEAGENT_JUDGE_TIMEOUT_SECONDS",
                "FIRE_AGENT_JUDGE_TIMEOUT_SECONDS",
                default=base_settings.timeout_seconds,
            ),
            max_tokens=FinanceAgentBenchEvaluator._env_int(
                "FIRE_AGENT_FINANCEAGENT_JUDGE_MAX_TOKENS",
                "FIRE_AGENT_JUDGE_MAX_TOKENS",
                default=base_settings.max_tokens,
            ),
            temperature=FinanceAgentBenchEvaluator._env_float(
                "FIRE_AGENT_FINANCEAGENT_JUDGE_TEMPERATURE",
                "FIRE_AGENT_JUDGE_TEMPERATURE",
                default=base_settings.temperature,
            ),
        )

    @staticmethod
    def _env_int(primary: str, fallback: str, default: int) -> int:
        raw = os.getenv(primary) or os.getenv(fallback)
        if raw is None or str(raw).strip() == "":
            return default
        return int(raw)

    @staticmethod
    def _env_float(primary: str, fallback: str, default: float) -> float:
        raw = os.getenv(primary) or os.getenv(fallback)
        if raw is None or str(raw).strip() == "":
            return default
        return float(raw)
