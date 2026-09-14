from __future__ import annotations

import calendar
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TUSHARE_TOKEN_ENV_NAMES = (
    "FIRE_AGENT_TUSHARE_TOKEN",
    "TUSHARE_TOKEN",
    # Legacy names are accepted so existing local .env files keep working, but
    # the runtime path is the official tushare client against the configured HTTP backend.
    "FIRE_AGENT_TINYSHARE_TOKEN",
    "TINYSHARE_TOKEN",
)
TUSHARE_HTTP_URL_ENV_NAMES = (
    "FIRE_AGENT_TUSHARE_HTTP_URL",
    "TUSHARE_HTTP_URL",
    # Legacy aliases for older deployments.
    "FIRE_AGENT_TINYSHARE_HTTP_URL",
    "TINYSHARE_HTTP_URL",
)
DEFAULT_TUSHARE_HTTP_URL = "http://8.148.76.181:8686/"
DEFAULT_DATE_COLUMNS = (
    "trade_date",
    "date",
    "ann_date",
    "end_date",
    "f_ann_date",
    "nav_date",
    "period",
    "pub_date",
    "cal_date",
    "m",
    "month",
    "q",
    "quarter",
    "stat_month",
    "list_date",
    "delist_date",
    "exercise_date",
    "last_ddate",
)


class TushareProviderError(RuntimeError):
    pass


@dataclass
class TushareFrame:
    endpoint: str
    records: List[Dict[str, Any]]
    columns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TushareProvider:
    """Small Tushare provider for typed China financial data actions."""

    provider_name = "tushare_http"

    def __init__(self, token: Optional[str] = None, backend: str = "tushare_http") -> None:
        self._backend = self._normalize_backend(backend)
        self._token = (token or self.token()).strip()
        self._http_url = self.http_url() or DEFAULT_TUSHARE_HTTP_URL
        self._client: Any = None

    @classmethod
    def token(cls) -> str:
        return cls._value_from_env_or_env_file(TUSHARE_TOKEN_ENV_NAMES)

    @classmethod
    def http_url(cls) -> str:
        return cls._value_from_env_or_env_file(TUSHARE_HTTP_URL_ENV_NAMES)

    @classmethod
    def has_token(cls) -> bool:
        return bool(cls.token())

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        # Provider hints such as "tushare" and legacy "tinyshare" all map to
        # the same runtime path: official tushare client + configured HTTP URL.
        return "tushare_http"

    @staticmethod
    def _value_from_env_or_env_file(names: Sequence[str]) -> str:
        for name in names:
            value = os.getenv(name)
            if value:
                return value.strip().strip("\"'")
        return TushareProvider._value_from_env_file(names)

    @staticmethod
    def _token_from_env_file() -> str:
        return TushareProvider._value_from_env_file(TUSHARE_TOKEN_ENV_NAMES)

    @staticmethod
    def _value_from_env_file(names: Sequence[str]) -> str:
        current = Path(__file__).resolve()
        for parent in current.parents:
            env_path = parent / ".env"
            if not env_path.exists():
                continue
            try:
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    if name.strip() in names:
                        cleaned = value.strip().strip("\"'")
                        if cleaned:
                            return cleaned
            except OSError:
                continue
        return ""

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._token:
            raise TushareProviderError(
                "Tushare provider requires FIRE_AGENT_TUSHARE_TOKEN, FIRE_AGENT_TINYSHARE_TOKEN, "
                "TUSHARE_TOKEN, or TINYSHARE_TOKEN."
            )
        self._client = self._official_tushare_client()
        return self._client

    @property
    def backend_name(self) -> str:
        return self._backend

    def _official_tushare_client(self) -> Any:
        try:
            import tushare as ts  # type: ignore
        except ImportError as exc:
            raise TushareProviderError(f"Tushare dependency is not available: {exc}") from exc
        client = ts.pro_api(self._token)
        try:
            client._DataApi__token = self._token
            client._DataApi__http_url = self._http_url
        except Exception as exc:
            raise TushareProviderError(f"Failed to configure Tushare HTTP URL: {exc}") from exc
        return client

    def call(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        start_date: Any = None,
        end_date: Any = None,
        date_columns: Sequence[str] = DEFAULT_DATE_COLUMNS,
        sort_by: str = "",
        limit: Optional[int] = None,
        limit_mode: str = "tail",
        query: str = "",
        query_fields: Optional[Iterable[str]] = None,
    ) -> TushareFrame:
        endpoint = str(endpoint or "").strip()
        if not re.fullmatch(r"[A-Za-z]\w*", endpoint):
            raise TushareProviderError(f"Invalid Tushare endpoint name: {endpoint!r}")
        client = self.client()
        fn = getattr(client, endpoint, None)
        if not callable(fn):
            raise TushareProviderError(f"Tushare endpoint is not available: {endpoint}")

        clean_params = self._clean_params(params or {})
        try:
            df = fn(**clean_params)
        except Exception as exc:
            raise TushareProviderError(f"{type(exc).__name__}: {exc}") from exc

        records, columns = self._records_from_dataframe(df)
        requested_start = format_tushare_date(start_date, is_end=False)
        requested_end = format_tushare_date(end_date, is_end=True)
        records, date_filter_applied = self._filter_records_by_date(
            records,
            requested_start,
            requested_end,
            date_columns,
        )
        if query:
            records = self._filter_records_by_query(records, query, query_fields)
        if sort_by:
            records = self._sort_records(records, sort_by)
        if limit is not None and limit > 0:
            records = records[-limit:] if limit_mode == "tail" else records[:limit]

        return TushareFrame(
            endpoint=endpoint,
            records=records,
            columns=columns,
            metadata={
                "provider": self.backend_name,
                "endpoint": endpoint,
                "params": clean_params,
                "date_filter_applied": date_filter_applied,
                "result_count": len(records),
            },
        )

    @staticmethod
    def _clean_params(params: Mapping[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in params.items():
            if str(key).startswith("_fire_"):
                continue
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                value = ",".join(str(item) for item in value if item not in (None, ""))
                if not value:
                    continue
            clean[str(key)] = value
        return clean

    @classmethod
    def _records_from_dataframe(cls, df: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        if df is None:
            return [], []
        if not hasattr(df, "to_dict") or not hasattr(df, "columns"):
            raise TushareProviderError("Tushare endpoint did not return a DataFrame-like table.")
        columns = [str(column) for column in list(df.columns)]
        records: List[Dict[str, Any]] = []
        for raw_row in df.to_dict("records"):
            row = {str(key): cls._to_python_value(value) for key, value in raw_row.items()}
            records.append(row)
        return records, columns

    @classmethod
    def _to_python_value(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if hasattr(value, "item"):
            try:
                return cls._to_python_value(value.item())
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return value

    @classmethod
    def _filter_records_by_date(
        cls,
        records: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        date_columns: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not records or (not start_date and not end_date):
            return records, False
        start_key = _date_key(start_date, is_end=False) if start_date else ""
        end_key = _date_key(end_date, is_end=True) if end_date else ""
        keyed: List[Tuple[Dict[str, Any], str]] = []
        for record in records:
            key = cls._record_date_key(record, date_columns)
            if key:
                keyed.append((record, key))
        if not keyed:
            return records, False
        filtered = [
            record
            for record, key in keyed
            if (not start_key or key >= start_key) and (not end_key or key <= end_key)
        ]
        return filtered, True

    @staticmethod
    def _record_date_key(record: Mapping[str, Any], date_columns: Sequence[str]) -> str:
        lower_to_key = {str(key).lower(): key for key in record}
        for column in date_columns:
            key = lower_to_key.get(str(column).lower())
            if key is None:
                continue
            date_key = _date_key(record.get(key), is_end=True)
            if date_key:
                return date_key
        return ""

    @staticmethod
    def _filter_records_by_query(
        records: List[Dict[str, Any]],
        query: str,
        query_fields: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        needle = str(query or "").strip().upper()
        if not needle:
            return records
        fields = {str(field) for field in query_fields or [] if str(field)}
        filtered: List[Dict[str, Any]] = []
        for record in records:
            values = (
                [record.get(field) for field in fields if field in record]
                if fields
                else list(record.values())
            )
            if any(needle in str(value).upper() for value in values if value is not None):
                filtered.append(record)
        return filtered

    @classmethod
    def _sort_records(cls, records: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        key_name = str(sort_by or "")
        if not key_name:
            return records

        def sort_key(record: Mapping[str, Any]) -> Tuple[int, str]:
            value = record.get(key_name)
            date_key = _date_key(value, is_end=True)
            return (0, date_key) if date_key else (1, str(value or ""))

        return sorted(records, key=sort_key)


def format_tushare_date(value: Any, *, is_end: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{8}", text):
        return text
    if re.fullmatch(r"\d{6}", text):
        return f"{text}{_month_day(text[:4], text[4:6], is_end):02d}"
    if re.fullmatch(r"\d{4}", text):
        return f"{text}1231" if is_end else f"{text}0101"
    quarter = _quarter_match(text)
    if quarter:
        year, q = quarter
        month_day = {"1": "0331", "2": "0630", "3": "0930", "4": "1231"}[q]
        return f"{year}{month_day}"
    match = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}{int(match.group(2)):02d}{int(match.group(3)):02d}"
    match = re.search(r"(\d{4})[-/.年](\d{1,2})", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return f"{year:04d}{month:02d}{_month_day(year, month, is_end):02d}"
    digits = re.sub(r"\D", "", text)
    if re.fullmatch(r"\d{8}", digits):
        return digits
    if re.fullmatch(r"\d{6}", digits):
        return f"{digits}{_month_day(digits[:4], digits[4:6], is_end):02d}"
    return ""


def format_tushare_period(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{8}", text):
        return text
    if re.fullmatch(r"\d{4}", text):
        return f"{text}1231"
    quarter = _quarter_match(text)
    if quarter:
        year, q = quarter
        month_day = {"1": "0331", "2": "0630", "3": "0930", "4": "1231"}[q]
        return f"{year}{month_day}"
    return format_tushare_date(text, is_end=True)


def normalize_tushare_ts_code(ticker: Any) -> str:
    value = str(ticker or "").strip().upper()
    if not value:
        return ""
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("6", "9")):
            return f"{value}.SH"
        if value.startswith(("0", "2", "3")):
            return f"{value}.SZ"
        if value.startswith(("4", "8")):
            return f"{value}.BJ"
        return ""
    match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ|SS)", value)
    if match:
        suffix = "SH" if match.group(2) == "SS" else match.group(2)
        return f"{match.group(1)}.{suffix}"
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", value)
    if match:
        return f"{match.group(2)}.{match.group(1)}"
    return ""


def normalize_tushare_index_code(code: Any) -> str:
    value = str(code or "").strip().upper()
    if not value:
        return ""
    match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ|SS)", value)
    if match:
        suffix = "SH" if match.group(2) == "SS" else match.group(2)
        return f"{match.group(1)}.{suffix}"
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", value)
    if match:
        return f"{match.group(2)}.{match.group(1)}"
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("399", "159", "16")):
            return f"{value}.SZ"
        if value.startswith(("000", "50", "51", "88", "93")):
            return f"{value}.SH"
    return normalize_tushare_ts_code(value)


def normalize_tushare_fund_code(code: Any) -> str:
    value = str(code or "").strip().upper()
    if not value:
        return ""
    match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ|SS)", value)
    if match:
        suffix = "SH" if match.group(2) == "SS" else match.group(2)
        return f"{match.group(1)}.{suffix}"
    match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", value)
    if match:
        return f"{match.group(2)}.{match.group(1)}"
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("5", "6")):
            return f"{value}.SH"
        if value.startswith(("0", "1", "2", "3")):
            return f"{value}.SZ"
    return ""


def normalize_tushare_contract(contract: Any, exchange: Any = None) -> str:
    value = str(contract or "").strip().upper()
    if not value:
        return ""
    if "." in value:
        code, suffix = value.rsplit(".", 1)
        suffix = _normalize_future_exchange_suffix(suffix)
        return f"{code}.{suffix}" if suffix else value
    suffix = _normalize_future_exchange_suffix(exchange)
    return f"{value}.{suffix}" if suffix else value


def _normalize_future_exchange_suffix(exchange: Any) -> str:
    value = str(exchange or "").strip().upper()
    aliases = {
        "SHFE": "SHF",
        "SHF": "SHF",
        "DCE": "DCE",
        "CZCE": "CZC",
        "CZC": "CZC",
        "ZCE": "CZC",
        "CFFEX": "CFX",
        "CFX": "CFX",
        "INE": "INE",
        "GFEX": "GFE",
        "GFE": "GFE",
    }
    return aliases.get(value, "")


def _date_key(value: Any, *, is_end: bool) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{8}", text):
        return text
    if re.fullmatch(r"\d{6}", text):
        return f"{text}{_month_day(text[:4], text[4:6], is_end):02d}"
    if re.fullmatch(r"\d{4}", text):
        return f"{text}1231" if is_end else f"{text}0101"
    quarter = _quarter_match(text)
    if quarter:
        year, q = quarter
        month_day = {"1": "0331", "2": "0630", "3": "0930", "4": "1231"}[q]
        return f"{year}{month_day}"
    return format_tushare_date(text, is_end=is_end)


def _month_day(year: Any, month: Any, is_end: bool) -> int:
    if not is_end:
        return 1
    try:
        return calendar.monthrange(int(year), int(month))[1]
    except Exception:
        return 31


def _quarter_match(value: str) -> Optional[Tuple[str, str]]:
    text = str(value or "").strip().upper()
    patterns = (
        r"(\d{4})\s*Q([1-4])",
        r"(\d{4})[-_/]([1-4])Q",
        r"(\d{4})\s*第?([1-4])季度",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return match.group(1), match.group(2)
    return None
