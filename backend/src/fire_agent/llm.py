from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json_repair

from .config import ModelSettings
from .http_client import http_post
from .api_balance_circuit_breaker import (
    GlobalExperimentAbort,
    raise_if_abort_requested,
    record_api_error,
    record_api_success,
)


def parse_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        parsed = json_repair.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json_repair.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def split_thinking_tags(text: Any) -> tuple[str, str]:
    """Split Qwen-style ``<think>...</think>`` from generated content.

    Some OpenAI-compatible Qwen deployments return thinking and final content
    as one text stream instead of a separate ``reasoning_content`` field. The
    FIRE harness keeps model-native thinking separate from visible assistant content.
    """

    if not isinstance(text, str):
        return "", str(text or "")
    stripped = text.lstrip()
    leading_ws = len(text) - len(stripped)
    if not stripped.startswith("<think>"):
        close = stripped.find("</think>")
        open_pos = stripped.find("<think>")
        if close >= 0 and (open_pos < 0 or close < open_pos):
            reasoning = stripped[:close].strip()
            content = stripped[close + len("</think>") :].lstrip()
            return reasoning, content
        return "", text
    think_start = leading_ws + len("<think>")
    close = text.find("</think>", think_start)
    if close < 0:
        return "", text
    reasoning = text[think_start:close].strip()
    content = text[close + len("</think>") :].lstrip()
    return reasoning, content


@dataclass
class ChatResponse:
    content: str
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    raw_message: Dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleChatModel:
    """Small OpenAI-compatible chat/completions client owned by FIRE Agent."""

    def __init__(self, settings: ModelSettings):
        if not settings.is_configured:
            raise ValueError("FIRE Agent model settings are incomplete.")
        self.settings = settings

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> ChatResponse:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if self.settings.thinking_enabled is None:
            payload["temperature"] = self.settings.temperature
        else:
            payload["thinking"] = {
                "type": "enabled" if self.settings.thinking_enabled else "disabled"
            }
            if self.settings.thinking_enabled:
                payload["reasoning_effort"] = self.settings.reasoning_effort or "high"
            else:
                payload["temperature"] = self.settings.temperature
        # SGLang/Qwen uses chat_template_kwargs for the native thinking switch,
        # while other OpenAI-compatible providers may use the generic thinking
        # field above. Keep this opt-in to avoid changing existing providers.
        raw_template_kwargs = os.getenv("FIRE_AGENT_CHAT_TEMPLATE_KWARGS", "").strip()
        if raw_template_kwargs:
            try:
                template_kwargs = json.loads(raw_template_kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError("FIRE_AGENT_CHAT_TEMPLATE_KWARGS must be a JSON object") from exc
            if not isinstance(template_kwargs, dict):
                raise ValueError("FIRE_AGENT_CHAT_TEMPLATE_KWARGS must be a JSON object")
            payload["chat_template_kwargs"] = template_kwargs
        if self.settings.extra_body:
            payload.update(self.settings.extra_body)
        base_url = str(self.settings.base_url or "").rstrip("/")
        api_key = str(self.settings.api_key or "")
        model = str(self.settings.model or "")
        raise_if_abort_requested()
        try:
            response = http_post(
                base_url + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json_body=payload,
                timeout=self.settings.timeout_seconds,
            )
        except GlobalExperimentAbort:
            raise
        except Exception as exc:
            if record_api_error(base_url=base_url, api_key=api_key, model=model, error=exc):
                raise GlobalExperimentAbort(
                    f"Global experiment abort: {model} returned five consecutive balance errors"
                ) from exc
            raise
        record_api_success(base_url=base_url, api_key=api_key, model=model)
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content or "")
        raw_tool_calls = message.get("tool_calls")
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        parsed_reasoning, parsed_content = split_thinking_tags(content)
        if parsed_reasoning or parsed_content != content:
            if not reasoning_content:
                reasoning_content = parsed_reasoning
            content = parsed_content
        raw_usage = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(
                raw_usage.get("total_tokens",
                              (raw_usage.get("prompt_tokens", 0) or 0)
                              + (raw_usage.get("completion_tokens", 0) or 0))
                or 0
            ),
        }
        return ChatResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
            model=str(data.get("model") or self.settings.model),
            finish_reason=str(choice.get("finish_reason") or ""),
            raw_message=message if isinstance(message, dict) else {},
        )


def call_chat_model(model: Any, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[str]:
    """Call a FIRE Agent chat model or a compatible callable."""
    if model is None:
        return None

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    response = model(messages)
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    return str(response)
