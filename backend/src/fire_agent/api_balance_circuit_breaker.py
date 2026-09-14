from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:  # fcntl is Unix-only; advisory file locking is a no-op elsewhere (e.g. Windows).
    import fcntl
except ImportError:  # pragma: no cover - platform-dependent
    fcntl = None


JsonDict = Dict[str, Any]


class GlobalExperimentAbort(RuntimeError):
    """Raised when a shared API breaker has stopped the experiment."""


_BALANCE_ERROR_MARKERS = (
    "insufficient balance",
    "insufficient_balance",
    "insufficient quota",
    "insufficient_quota",
    "not enough balance",
    "not enough credits",
    "balance is insufficient",
    "account balance",
    "账户余额不足",
    "余额不足",
    "欠费",
)


def is_balance_error(error: Any) -> bool:
    text = re.sub(r"\s+", " ", str(error or "")).strip().lower()
    return any(marker in text for marker in _BALANCE_ERROR_MARKERS)


def breaker_state_path() -> Optional[Path]:
    raw = (os.getenv("FIRE_AGENT_GLOBAL_API_BREAKER_STATE_FILE", "") or "").strip()
    return Path(raw) if raw else None


def breaker_abort_path() -> Optional[Path]:
    raw = (os.getenv("FIRE_AGENT_GLOBAL_ABORT_FILE", "") or "").strip()
    if raw:
        return Path(raw)
    state_path = breaker_state_path()
    return state_path.with_suffix(state_path.suffix + ".abort.json") if state_path else None


def breaker_threshold() -> int:
    raw = os.getenv("FIRE_AGENT_GLOBAL_BALANCE_ERROR_THRESHOLD", "5")
    try:
        return max(0, int(raw))
    except Exception:
        return 5


def api_identity(*, base_url: str, api_key: str, model: str) -> str:
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    payload = f"{base_url.rstrip('/')}\0{key_hash}\0{model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def abort_requested() -> bool:
    path = breaker_abort_path()
    return bool(path and path.exists())


def abort_details() -> JsonDict:
    path = breaker_abort_path()
    if not path or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def raise_if_abort_requested() -> None:
    if not abort_requested():
        return
    details = abort_details()
    reason = str(details.get("reason") or "global API balance circuit breaker triggered")
    raise GlobalExperimentAbort(reason)


def record_api_success(*, base_url: str, api_key: str, model: str) -> None:
    _update_state(base_url=base_url, api_key=api_key, model=model, balance_error=None)


def record_api_error(*, base_url: str, api_key: str, model: str, error: Any) -> bool:
    """Record one raw API response error and return True if global abort triggered."""

    return _update_state(
        base_url=base_url,
        api_key=api_key,
        model=model,
        balance_error=str(error) if is_balance_error(error) else None,
    )


def _update_state(
    *, base_url: str, api_key: str, model: str, balance_error: Optional[str]
) -> bool:
    state_path = breaker_state_path()
    threshold = breaker_threshold()
    if state_path is None or threshold <= 0:
        return False

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    identity = api_identity(base_url=base_url, api_key=api_key, model=model)
    now = datetime.now(timezone.utc).isoformat()

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = _read_json(state_path)
        apis = state.setdefault("apis", {})
        entry = apis.setdefault(identity, {})
        entry.update(
            {
                "identity": identity,
                "base_url": base_url.rstrip("/"),
                "model": model,
                "threshold": threshold,
                "updated_at_utc": now,
            }
        )

        if balance_error is None:
            entry["consecutive_balance_errors"] = 0
            entry.pop("last_error", None)
            _atomic_write_json(state_path, state)
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            return False

        count = int(entry.get("consecutive_balance_errors", 0) or 0) + 1
        entry["consecutive_balance_errors"] = count
        entry["last_error"] = _sanitize_error(balance_error)
        triggered = count >= threshold
        if triggered:
            abort_payload = {
                "triggered": True,
                "reason": (
                    f"API balance insufficient for model {model}: "
                    f"{count} consecutive balance errors (threshold={threshold})"
                ),
                "api_identity": identity,
                "base_url": base_url.rstrip("/"),
                "model": model,
                "count": count,
                "threshold": threshold,
                "last_error": entry["last_error"],
                "triggered_at_utc": now,
            }
            state["abort"] = abort_payload
            abort_path = breaker_abort_path()
            if abort_path is not None:
                _atomic_write_json(abort_path, abort_payload)
        _atomic_write_json(state_path, state)
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return triggered


def _sanitize_error(error: str) -> str:
    text = re.sub(r"\s+", " ", str(error or "")).strip()
    text = re.sub(r"(?i)bearer\s+[a-z0-9._~+\-/]+=*", "Bearer [REDACTED]", text)
    return text[:1000]


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
