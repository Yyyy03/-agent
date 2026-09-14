from __future__ import annotations

import csv
import io
import re
import time
from typing import Any, Dict, List, Optional

from ..http_client import HTTPRequestError, http_post


FINANCE_AGENT_MAX_END_DATE = "2025-04-07"
DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"


def finance_agent_validate_date(
    value: Any,
    field_name: str,
    max_end_date: str = FINANCE_AGENT_MAX_END_DATE,
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} '{text}' is not in yyyy-mm-dd format")
    if not re.match(DATE_REGEX, text):
        raise ValueError(f"{field_name} '{text}' is not in yyyy-mm-dd format")
    if text > max_end_date:
        return max_end_date
    return text


def finance_agent_records_to_csv(records: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not records:
        return ""
    if columns is None:
        seen: Dict[str, None] = {}
        for record in records:
            for key, value in record.items():
                if value is not None:
                    seen.setdefault(str(key), None)
        columns = list(seen)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({key: "N/A" if record.get(key) is None else record.get(key) for key in columns})
    return output.getvalue().strip()


def finance_agent_retryable_post(
    url: str,
    *,
    headers: Dict[str, str],
    json_body: Any,
    timeout: int,
    retry_statuses: tuple[int, ...],
) -> Any:
    last_error: Optional[HTTPRequestError] = None
    for attempt in range(3):
        try:
            return http_post(url, headers=headers, json_body=json_body, timeout=timeout)
        except HTTPRequestError as exc:
            last_error = exc
            message = str(exc)
            if not any(f"HTTP {status}" in message for status in retry_statuses) or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if last_error:
        raise last_error
    raise RuntimeError("retryable POST failed unexpectedly")
