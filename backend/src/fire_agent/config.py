from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_TEXT_MODEL = "deepseek-v4-flash"
DEFAULT_TEXT_BASE_URL = "https://api.deepseek.com"
DEFAULT_VISION_MODEL = "qwen/qwen3.6-plus"
DEFAULT_VISION_BASE_URL = "https://openrouter.ai/api/v1"


def load_fire_agent_env(env_file: Optional[str] = None) -> None:
    """Load only FIRE Agent's own environment file.

    FIRE Agent intentionally does not load the repository-level `.env`; this
    keeps its model, tool, and benchmark configuration independent.
    """

    path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if path.exists():
        load_dotenv(path, override=True)


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(f"FIRE_AGENT_{name}", default)


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = env(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_json(name: str) -> Dict[str, Any]:
    value = env(name)
    if value is None or value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"FIRE_AGENT_{name} must be a JSON object.")
    return parsed


def _looks_like_deepseek(model: Optional[str], base_url: Optional[str]) -> bool:
    text = f"{model or ''} {base_url or ''}".lower()
    return "deepseek" in text


@dataclass(frozen=True)
class ModelSettings:
    model: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    timeout_seconds: int = 120
    max_tokens: int = 4096
    temperature: float = 0.0
    # ``None`` means do not send a provider-specific thinking parameter.
    # DeepSeek-compatible providers may default to thinking mode, so when the
    # env var is unset we explicitly disable thinking for those providers.
    thinking_enabled: Optional[bool] = None
    reasoning_effort: str = "high"
    extra_body: Dict[str, Any] = None

    @classmethod
    def from_env(cls, model_override: Optional[str] = None) -> "ModelSettings":
        model = model_override or env("MODEL", DEFAULT_TEXT_MODEL)
        base_url = env("BASE_URL", DEFAULT_TEXT_BASE_URL)
        raw_thinking = env("THINKING_ENABLED")
        if raw_thinking is None or raw_thinking.strip() == "":
            thinking_enabled = False if _looks_like_deepseek(model, base_url) else None
        else:
            thinking_enabled = raw_thinking.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
        reasoning_effort = (env("REASONING_EFFORT", "high") or "high").strip().lower()
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            reasoning_effort = "high"
        return cls(
            model=model,
            api_key=env("API_KEY"),
            base_url=base_url,
            timeout_seconds=env_int("MODEL_TIMEOUT_SECONDS", 120),
            max_tokens=env_int("MODEL_MAX_TOKENS", 4096),
            temperature=float(env("MODEL_TEMPERATURE", "0")),
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            extra_body=env_json("EXTRA_BODY_JSON"),
        )

    @classmethod
    def from_role_env(cls, role: str) -> "ModelSettings":
        """Load a role-specific model, falling back field-by-field to the primary model."""

        prefix = str(role or "").strip().upper()
        primary = cls.from_env()

        def role_env(name: str, fallback: Optional[str]) -> Optional[str]:
            value = env(f"{prefix}_{name}")
            return value if value not in (None, "") else fallback

        model = role_env("MODEL", primary.model)
        base_url = role_env("BASE_URL", primary.base_url)
        raw_thinking = env(f"{prefix}_THINKING_ENABLED")
        if raw_thinking is None or raw_thinking.strip() == "":
            thinking_enabled = False if _looks_like_deepseek(model, base_url) else primary.thinking_enabled
        else:
            thinking_enabled = raw_thinking.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
        reasoning_effort = (role_env("REASONING_EFFORT", primary.reasoning_effort) or "high").strip().lower()
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            reasoning_effort = "high"
        return cls(
            model=model,
            api_key=role_env("API_KEY", primary.api_key),
            base_url=base_url,
            timeout_seconds=int(role_env("TIMEOUT_SECONDS", str(primary.timeout_seconds)) or primary.timeout_seconds),
            max_tokens=int(role_env("MAX_TOKENS", str(primary.max_tokens)) or primary.max_tokens),
            temperature=float(role_env("TEMPERATURE", str(primary.temperature)) or primary.temperature),
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            extra_body=env_json(f"{prefix}_EXTRA_BODY_JSON") or dict(primary.extra_body or {}),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.model and self.api_key and self.base_url)


@dataclass(frozen=True)
class ToolSettings:
    serper_api_key: Optional[str]
    jina_api_key: Optional[str]
    sec_api_key: Optional[str]
    sec_user_agent: str
    market_data_provider: str
    vision_model: Optional[str]
    vision_api_key: Optional[str]
    vision_base_url: Optional[str]

    @classmethod
    def from_env(cls) -> "ToolSettings":
        return cls(
            serper_api_key=env("SERPER_API_KEY"),
            jina_api_key=env("JINA_API_KEY"),
            sec_api_key=env("SEC_API_KEY") or env("SEC_EDGAR_API_KEY") or os.getenv("SEC_EDGAR_API_KEY"),
            sec_user_agent=env("SEC_USER_AGENT", "fire-agent contact@example.com") or "fire-agent contact@example.com",
            market_data_provider=env("MARKET_DATA_PROVIDER", "akshare") or "akshare",
            vision_model=env("VISION_MODEL", DEFAULT_VISION_MODEL),
            vision_api_key=env("VISION_API_KEY"),
            vision_base_url=env("VISION_BASE_URL", DEFAULT_VISION_BASE_URL),
        )
