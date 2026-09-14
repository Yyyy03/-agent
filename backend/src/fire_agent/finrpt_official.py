"""Official FinRpt-Gen replay utilities.

This module mirrors the upstream ``FinRpt/finrpt/module/{Advisor, Predictor,
RiskAssessor}`` generation behaviour at the LLM-call level. The upstream
pipeline assembles five prompts (``finance_write_prompt``,
``news_write_prompt``, ``report_write_prompt``, ``trend_write_prompt``,
``risk_prompt``) and feeds them to ``OpenAIModel.json_prompt`` one by one to
produce the five official response fields.

The ``jinsong8/FinRpt`` HuggingFace dataset already contains those prebuilt
prompts (they are the exact strings used by the reference gpt-4o run), so
faithfully replaying them through the configured chat model reproduces the
upstream generation logic without requiring the upstream SQLite cache or the
live akshare/sina/eastmoney data pulls.

The JSON parsing is a faithful translation of
``FinRpt/finrpt/utils/data_processing.robust_load_json`` plus the surrounding
``json_prompt`` retry loop in ``FinRpt/finrpt/module/OpenAI.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


OFFICIAL_PROMPT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("finance_write_prompt", "finance_write_response"),
    ("news_write_prompt", "news_write_response"),
    ("report_write_prompt", "report_write_response"),
    ("trend_write_prompt", "trend_write_response"),
    ("risk_prompt", "risk_response"),
)


def _extract_outer_braces(text: str) -> Optional[str]:
    """Mirror of ``FinRpt.utils.data_processing.extract_outer_braces``."""
    stack: List[str] = []
    start: Optional[int] = None
    for i, char in enumerate(text):
        if char == "{":
            if start is None:
                start = i
            stack.append(char)
        elif char == "}":
            if not stack:
                continue
            stack.pop()
            if not stack and start is not None:
                return text[start : i + 1]
    return None


def robust_load_json(text: str) -> Any:
    """Best-effort JSON loader mirroring upstream's robust_load_json.

    Tries the same fallback sequence as ``FinRpt.utils.data_processing``: strip
    a leading ```json fence (length 7 or 8), parse the raw text, extract the
    outermost ``{...}`` block, and finally match a fenced ```json``` block.
    """

    if not isinstance(text, str):
        raise ValueError(f"robust_load_json expected str, got {type(text).__name__}")

    candidates: List[str] = []
    if len(text) >= 10:
        candidates.append(text[7:-3].strip())
        candidates.append(text[8:-3].strip())
    candidates.append(text)

    stripped = text.strip()
    if stripped.startswith("```"):
        no_fences = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        no_fences = re.sub(r"```\s*$", "", no_fences)
        candidates.append(no_fences.strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue

    extracted = _extract_outer_braces(text)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except Exception:
            pass

    fenced_pattern = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
    match = fenced_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    raise ValueError("No valid JSON object found")


def _call_model(model: Any, prompt_text: str) -> str:
    """Invoke the FIRE Agent chat model with a single user-role message."""
    if model is None:
        raise RuntimeError("FinRpt official generation requires a configured chat model.")
    response = model([{"role": "user", "content": prompt_text}])
    if hasattr(response, "content"):
        return response.content or ""
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    return str(response)


def json_prompt(model: Any, prompt_text: str, max_rounds: int = 3) -> Tuple[str, Any]:
    """Mirror of ``OpenAIModel.json_prompt``.

    Repeatedly calls ``model`` until ``robust_load_json`` succeeds or
    ``max_rounds`` is exhausted. Returns ``(raw_text, parsed_obj)`` exactly like
    upstream so downstream code can serialise with the same
    ``json.dumps(parsed, ensure_ascii=False)`` semantics.
    """

    last_text = ""
    last_error: Optional[Exception] = None
    for attempt in range(max_rounds):
        raw = _call_model(model, prompt_text)
        last_text = raw
        try:
            return raw, robust_load_json(raw)
        except Exception as exc:
            last_error = exc
            continue
    snippet = last_text[:200].replace("\n", " ")
    raise RuntimeError(
        f"FinRpt official json_prompt exhausted {max_rounds} retries: {last_error}; last response head: {snippet!r}"
    )


@dataclass
class OfficialFieldResult:
    field: str
    prompt: str = ""
    raw_response: str = ""
    parsed: Any = None
    response_text: str = ""
    status: str = "skipped"
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class FinRptOfficialResult:
    responses: Dict[str, str] = field(default_factory=dict)
    field_results: List[OfficialFieldResult] = field(default_factory=list)
    missing_prompts: List[str] = field(default_factory=list)
    failed_prompts: List[str] = field(default_factory=list)

    @property
    def succeeded_any(self) -> bool:
        return any(result.status == "success" for result in self.field_results)


def generate_official_responses(
    record: Dict[str, Any],
    model: Any,
    *,
    max_rounds: int = 3,
    prompt_fields: Sequence[Tuple[str, str]] = OFFICIAL_PROMPT_FIELDS,
    on_field_done: Optional[Callable[[OfficialFieldResult], None]] = None,
) -> FinRptOfficialResult:
    """Run the upstream 5-call FinRpt-Gen generation against a single record.

    Each ``(*_prompt, *_response)`` pair from ``prompt_fields`` is passed
    through ``json_prompt``. Missing or empty prompts are recorded but do not
    abort the run, mirroring upstream's robustness when an analyser failed to
    produce intermediate text.
    """

    aggregate = FinRptOfficialResult()
    for prompt_field, response_field in prompt_fields:
        raw_prompt = record.get(prompt_field)
        prompt_text = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt or "")
        prompt_text = prompt_text.strip()
        field_result = OfficialFieldResult(field=response_field, prompt=prompt_text)
        if not prompt_text:
            aggregate.missing_prompts.append(prompt_field)
            field_result.status = "missing_prompt"
            aggregate.responses[response_field] = ""
            aggregate.field_results.append(field_result)
            if on_field_done:
                on_field_done(field_result)
            continue

        try:
            raw_text, parsed = json_prompt(model, prompt_text, max_rounds=max_rounds)
        except Exception as exc:
            aggregate.failed_prompts.append(prompt_field)
            field_result.status = "error"
            field_result.error = f"{type(exc).__name__}: {exc}"
            field_result.attempts = max_rounds
            aggregate.responses[response_field] = ""
            aggregate.field_results.append(field_result)
            if on_field_done:
                on_field_done(field_result)
            continue

        if isinstance(parsed, (dict, list)):
            response_text = json.dumps(parsed, ensure_ascii=False)
        else:
            response_text = str(parsed)
        field_result.raw_response = raw_text
        field_result.parsed = parsed
        field_result.response_text = response_text
        field_result.status = "success"
        field_result.attempts = 1
        aggregate.responses[response_field] = response_text
        aggregate.field_results.append(field_result)
        if on_field_done:
            on_field_done(field_result)

    return aggregate
