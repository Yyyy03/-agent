from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..schemas import ToolResult
from ..tools import ToolActionSpec, ToolFamily


class AkShareTool(ToolFamily):
    name = "ak"
    description = (
        "Single AkShare finance data tool. Choose the AkShare API through the "
        "`function` argument; supports bundled market snapshots and live "
        "AkShare historical/financial/macro functions."
    )
    paid = False
    default_action = "call"
    actions = {
        "call": ToolActionSpec(
            "Call one supported AkShare snapshot or live historical data function and optionally filter rows.",
            required=["function"],
            optional=[
                "query",
                "symbol",
                "ticker",
                "code",
                "start_date",
                "end_date",
                "date",
                "period",
                "adjust",
                "indicator",
                "filter",
                "limit",
                "category",
                "columns",
                "source",
            ],
        )
    }

    # These functions have bundled FinSearchComp snapshot fallbacks.
    supported_functions = {
        "stock_zh_a_spot_em",
        "stock_us_spot_em",
        "stock_hk_spot_em",
        "stock_zh_index_spot_em",
        "stock_hk_index_spot_em",
        "index_global_spot_em",
        "forex_spot_em",
        # Common historical market data functions.
        "stock_zh_a_hist",
        "stock_zh_a_daily",
        "stock_us_hist",
        "stock_us_daily",
        "stock_hk_hist",
        "stock_zh_index_daily_em",
        "index_global_hist_em",
        "forex_hist_em",
        "currency_boc_sina",
        "currency_boc_safe",
        # Futures, options, bonds, funds.
        "futures_main_sina",
        "futures_zh_daily_sina",
        "futures_hist_em",
        "futures_foreign_hist",
        "futures_dce_daily",
        "futures_settle_shfe",
        "option_sse_daily_sina",
        "option_sse_codes_sina",
        "option_finance_board",
        "bond_china_yield",
        "fund_etf_hist_em",
        "fund_portfolio_hold_em",
        "index_hist_sw",
        "index_component_sw",
        "sw_index_first_info",
        "sw_index_second_info",
        "sw_index_third_info",
        "index_stock_cons_csindex",
        "index_stock_cons_weight_csindex",
        # A-share financial statements and fundamentals.
        "stock_financial_abstract_ths",
        "stock_financial_abstract",
        "stock_yjbb_em",
        "stock_financial_report_sina",
        "stock_profit_sheet_by_report_em",
        "stock_balance_sheet_by_report_em",
        "stock_cash_flow_sheet_by_report_em",
        "stock_zh_a_financial_analysis_indicator",
        "stock_info_a_code_name",
        "stock_zyjs_em",
        "stock_zh_a_financial_abstract",
        "stock_business_revenue_em",
        "stock_fhy_main_body_report_em",
        "stock_zh_a_dividend",
        # Macro series.
        "macro_china_money_supply",
        "macro_japan_gdp",
        "macro_germany_gdp_yearly",
        "macro_germany_ifo",
        "macro_cpi_global",
        "macro_us_business_inventories",
        "us_treasury_yield",
    }
    function_aliases = {
        # Names older agents tended to invent from AkShare docs/examples.
        "stock_zh_a_hist_em": "stock_zh_a_hist",
        "stock_us_hist_em": "stock_us_hist",
        "stock_us_daily_em": "stock_us_daily",
        "stock_hk_hist_em": "stock_hk_hist",
        "stock_hk_daily": "stock_hk_hist",
        "stock_us_index_daily": "index_global_hist_em",
        "index_global_hist": "index_global_hist_em",
        "fx_spot_quote": "forex_spot_em",
        "futures_main": "futures_main_sina",
        "main_futures_sina": "futures_main_sina",
        "index_stock_cons_weight_csi": "index_stock_cons_weight_csindex",
        "index_stock_cons_weight_cs_index": "index_stock_cons_weight_csindex",
        "index_stock_cons_cs_index": "index_stock_cons_csindex",
        "bond_zh_yield_curve": "bond_china_yield",
        "bond_china_yield_curve": "bond_china_yield",
        "option_sse_daily": "option_sse_daily_sina",
        "option_sse_hist": "option_sse_daily_sina",
        "option_sse_hist_sina": "option_sse_daily_sina",
        "option_etf_daily": "option_sse_daily_sina",
    }
    function_guidance = {
        "sw_index_daily": {
            "recommended_tool": "market_data",
            "recommended_action": "index_history",
            "argument_mapping": {
                "index_code": "symbol/query such as 801125.SI",
                "start_date": "start_date",
                "end_date": "end_date",
            },
            "retry_same_plan_allowed": False,
            "reason": "Shenwan index history is exposed through the typed index_history namespace route.",
        },
        "index_daily_sw": {
            "recommended_tool": "market_data",
            "recommended_action": "index_history",
            "argument_mapping": {"index_code": "symbol/query", "start_date": "start_date", "end_date": "end_date"},
            "retry_same_plan_allowed": False,
        },
        "stock_zh_index_daily_sw": {
            "recommended_tool": "market_data",
            "recommended_action": "index_history",
            "argument_mapping": {"index_code": "symbol/query", "start_date": "start_date", "end_date": "end_date"},
            "retry_same_plan_allowed": False,
        },
        "sw_index_cons": {
            "recommended_tool": "market_data",
            "recommended_action": "index_constituents",
            "argument_mapping": {"index_code": "symbol/query such as 801125.SI"},
            "retry_same_plan_allowed": False,
        },
        "index_industry_sw": {
            "recommended_tool": "market_data",
            "recommended_action": "index_catalog or index_history",
            "argument_mapping": {"query": "query", "index_code": "symbol when a concrete code is known"},
            "retry_same_plan_allowed": False,
        },
        "stock_board_industry_sw_daily": {
            "recommended_tool": "market_data",
            "recommended_action": "index_history",
            "argument_mapping": {"index_code": "symbol/query", "start_date": "start_date", "end_date": "end_date"},
            "retry_same_plan_allowed": False,
        },
        "bond_zh_cn": {
            "recommended_tool": "market_data",
            "recommended_action": "market_table",
            "argument_mapping": {"function": "bond_china_yield", "start_date": "start_date", "end_date": "end_date"},
            "retry_same_plan_allowed": False,
        },
        "option_sse_spot_price": {
            "recommended_tool": "market_data",
            "recommended_action": "cn_options_basic or market_table(function='option_finance_board')",
            "argument_mapping": {"query": "contract or underlying"},
            "retry_same_plan_allowed": False,
        },
        "option_sse_listed_etf_options": {
            "recommended_tool": "market_data",
            "recommended_action": "cn_options_basic or market_table(function='option_sse_codes_sina')",
            "argument_mapping": {"query": "underlying/month/option type"},
            "retry_same_plan_allowed": False,
        },
        "fund_new_found": {
            "recommended_tool": "market_data",
            "recommended_action": "fund_basic",
            "argument_mapping": {"fields": "ts_code,name,fund_type,custodian,issue_date,establishment_date"},
            "retry_same_plan_allowed": False,
        },
        "fund_new_found_em": {
            "recommended_tool": "market_data",
            "recommended_action": "fund_basic",
            "argument_mapping": {"fields": "ts_code,name,fund_type,custodian,issue_date,establishment_date"},
            "retry_same_plan_allowed": False,
        },
    }
    no_symbol_functions = {
        "stock_yjbb_em",
        "stock_info_a_code_name",
        "bond_china_yield",
        "currency_boc_safe",
        "macro_china_money_supply",
        "macro_japan_gdp",
        "macro_germany_gdp_yearly",
        "macro_germany_ifo",
        "macro_cpi_global",
        "macro_us_business_inventories",
        "us_treasury_yield",
    }
    snapshot_files = {
        "stock_zh_a_spot_em": "cn_stock",
        "stock_us_spot_em": "us_stock",
        "stock_hk_spot_em": "hk_stock",
        "stock_zh_index_spot_em": "cn_index",
        "stock_hk_index_spot_em": "h_index",
        "index_global_spot_em": "global_idx",
        "forex_spot_em": "fx",
    }

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        return "call" if candidate in {"default", "call", "query"} else candidate

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        function = str(
            arguments.get("function")
            or arguments.get("api")
            or arguments.get("method")
            or ""
        ).strip()
        if function.startswith("ak."):
            function = function[3:]
        requested_function = function
        function = self.function_aliases.get(function, function)
        if not re.fullmatch(r"[A-Za-z]\w*", function or "") or function.startswith("_"):
            return ToolResult(
                self.name,
                "akshare",
                "error",
                action=action,
                error=f"Invalid AkShare function name: {requested_function}",
                paid=False,
                confidence=0.1,
            )

        try:
            import pandas as pd
        except ImportError as exc:
            return ToolResult(
                self.name,
                "akshare",
                "error",
                action=action,
                error=f"Pandas dependency is not available: {exc}",
                paid=False,
                confidence=0.1,
            )

        source = str(arguments.get("source") or "auto").strip().lower()
        live_error = ""
        try:
            import akshare as ak
        except ImportError as exc:
            ak = None
            live_error = f"AkShare dependency is not available: {exc}"

        has_live_function = bool(ak is not None and hasattr(ak, function))
        if ak is None and function not in self.snapshot_files:
            return self._akshare_dependency_error(action, requested_function, function, live_error)
        if function not in self.snapshot_files and not has_live_function:
            supported = ", ".join(sorted(self.supported_functions))
            guidance = self._function_guidance(requested_function, function)
            guidance_text = ""
            if guidance:
                recommended = guidance.get("recommended_action") or guidance.get("replacement_function")
                guidance_text = f" Recommended next step: {recommended}. "
            return ToolResult(
                self.name,
                "akshare",
                "error",
                action=action,
                error=(
                    f"Unsupported AkShare function '{requested_function}'"
                    + (f" (resolved to '{function}')" if requested_function != function else "")
                    + f". It is not exposed by the installed akshare package. "
                    + guidance_text
                    + f"Known high-value functions: {supported}"
                ),
                paid=False,
                confidence=0.1,
                metadata={
                    "requested_function": requested_function,
                    "resolved_function": function,
                    "python_executable": sys.executable,
                    "akshare_dependency_available": ak is not None,
                    "known_high_value_functions": sorted(self.supported_functions),
                    "function_guidance": guidance,
                    "retry_same_plan_allowed": False if guidance else None,
                },
            )

        # Snapshot files only exist for FinSearchComp T1 spot universes. For
        # historical/financial/macro functions, a config default of
        # source=snapshot should not block the live AkShare call.
        if source == "snapshot" and function not in self.snapshot_files:
            source = "live"

        live_error = ""
        try:
            if source == "snapshot":
                df = self._load_snapshot(function, pd)
            else:
                df = self._call_akshare(ak, function, arguments)
        except Exception as exc:
            live_error = f"{type(exc).__name__}: {exc}"
            if source == "live" or function not in self.snapshot_files:
                return ToolResult(
                    self.name,
                    "akshare",
                    "error",
                    action=action,
                    error=live_error,
                    paid=False,
                    confidence=0.1,
                )
            try:
                df = self._load_snapshot(function, pd)
                source = "snapshot"
            except Exception as snapshot_exc:
                return ToolResult(
                    self.name,
                    "akshare",
                    "error",
                    action=action,
                    error=(
                        f"Live AkShare failed: {live_error}; "
                        f"snapshot fallback failed: {type(snapshot_exc).__name__}: {snapshot_exc}"
                    ),
                    paid=False,
                    confidence=0.1,
                )

        if not isinstance(df, pd.DataFrame):
            return ToolResult(
                self.name,
                "akshare",
                "error",
                action=action,
                error=f"AkShare function '{function}' did not return a DataFrame.",
                paid=False,
                confidence=0.1,
            )

        df = self._filter_date_range(df, arguments)
        query = str(arguments.get("query") or arguments.get("查询内容") or "").strip()
        filter_query = self._filter_query(function, arguments)
        limit = self._optional_positive_int(arguments.get("limit") or arguments.get("max_results"))
        default_limit = self._default_row_limit(action)
        filtered = self._filter_dataframe(df, filter_query)
        if limit is not None:
            filtered = filtered.head(limit)
        elif default_limit is not None:
            filtered = filtered.head(default_limit)
        rows = filtered.to_dict(orient="records")
        return ToolResult(
            self.name,
            "akshare",
            "success",
            action=action,
            observation=filtered.to_csv(index=False).strip(),
            tables=[{"columns": list(filtered.columns), "rows": rows}],
            confidence=0.82 if rows else 0.35,
            paid=False,
            metadata={
                "provider": "akshare",
                "function": function,
                "requested_function": requested_function,
                "query": query,
                "filter": filter_query,
                "source": source,
                "matched_rows": len(filtered),
                "total_rows": len(df),
                "live_error": live_error,
            },
        )

    def _akshare_dependency_error(
        self,
        action: str,
        requested_function: str,
        resolved_function: str,
        live_error: str,
    ) -> ToolResult:
        guidance = self._function_guidance(requested_function, resolved_function)
        return ToolResult(
            self.name,
            "akshare",
            "error",
            action=action,
            error=(
                f"{live_error}. Current Python executable is {sys.executable}. "
                "Run the benchmark with an environment where akshare is installed, or use a non-AkShare typed provider when available."
            ),
            paid=False,
            confidence=0.1,
            metadata={
                "requested_function": requested_function,
                "resolved_function": resolved_function,
                "python_executable": sys.executable,
                "akshare_dependency_available": False,
                "function_guidance": guidance,
                "retry_same_plan_allowed": False,
            },
        )

    @classmethod
    def _function_guidance(cls, requested_function: str, resolved_function: str = "") -> Dict[str, Any]:
        requested = str(requested_function or "").strip()
        resolved = str(resolved_function or "").strip()
        guidance = cls.function_guidance.get(requested) or cls.function_guidance.get(resolved) or {}
        if guidance:
            return dict(guidance)
        if resolved and resolved != requested and resolved in cls.supported_functions:
            return {
                "replacement_function": resolved,
                "recommended_tool": "market_data",
                "recommended_action": "market_table",
                "argument_mapping": {"function": resolved},
                "retry_same_plan_allowed": True,
            }
        return {}

    def _call_akshare(self, ak: Any, function: str, arguments: Dict[str, Any]) -> Any:
        if ak is None:
            raise ImportError("AkShare dependency is not available")
        if function == "stock_zh_index_spot_em":
            category = str(arguments.get("category") or arguments.get("market") or "沪深重要指数")
            return ak.stock_zh_index_spot_em(category)
        fn = getattr(ak, function)
        kwargs = self._build_akshare_kwargs(function, fn, arguments)
        return fn(**kwargs)

    def _build_akshare_kwargs(self, function: str, fn: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        signature = inspect.signature(fn)
        params = signature.parameters
        kwargs: Dict[str, Any] = {}

        symbol = self._symbol_argument(arguments)
        if function == "stock_financial_us_report_em":
            stock = str(arguments.get("stock") or arguments.get("ticker") or arguments.get("query") or "").strip()
            if stock and "stock" in params:
                kwargs["stock"] = stock.upper()
            report_type = arguments.get("report_type") or arguments.get("statement_symbol") or arguments.get("symbol")
            if report_type not in (None, "") and "symbol" in params:
                kwargs["symbol"] = str(report_type)
            indicator = arguments.get("report_indicator") or arguments.get("period") or arguments.get("indicator")
            if indicator not in (None, "") and "indicator" in params:
                kwargs["indicator"] = str(indicator)
            return kwargs

        if function == "stock_financial_us_analysis_indicator_em":
            stock = str(arguments.get("stock") or arguments.get("ticker") or arguments.get("query") or "").strip()
            if stock and "symbol" in params:
                kwargs["symbol"] = stock.upper()
            indicator = arguments.get("report_indicator") or arguments.get("period") or arguments.get("indicator")
            if indicator not in (None, "") and "indicator" in params:
                kwargs["indicator"] = str(indicator)
            return kwargs

        if function == "stock_financial_report_sina":
            if symbol and "stock" in params:
                kwargs["stock"] = self._normalize_sina_stock(symbol)
            report_type = (
                arguments.get("report_type")
                or arguments.get("statement_type")
                or arguments.get("indicator")
                or arguments.get("symbol")
            )
            if report_type not in (None, "") and "symbol" in params:
                kwargs["symbol"] = str(report_type)
            return kwargs

        if "stock" in params and symbol:
            kwargs["stock"] = self._normalize_sina_stock(symbol)
        if "symbol" in params and symbol:
            kwargs["symbol"] = self._normalize_symbol_for_function(function, symbol, arguments)
        if "code" in params and symbol:
            kwargs["code"] = symbol

        for param_name in ("start_date", "end_date"):
            if param_name in params:
                value = self._date_argument(arguments, param_name)
                if value:
                    kwargs[param_name] = value

        if "date" in params:
            value = self._date_argument(arguments, "date") or arguments.get("year")
            if value:
                kwargs["date"] = str(value)

        for param_name in ("period", "adjust", "indicator", "market", "category"):
            if param_name in params and arguments.get(param_name) not in (None, ""):
                kwargs[param_name] = str(arguments[param_name])

        if "timeout" in params and arguments.get("timeout") not in (None, ""):
            kwargs["timeout"] = float(arguments["timeout"])
        return kwargs

    def _symbol_argument(self, arguments: Dict[str, Any]) -> str:
        return str(
            arguments.get("symbol")
            or arguments.get("ticker")
            or arguments.get("code")
            or arguments.get("query")
            or arguments.get("name")
            or ""
        ).strip()

    def _date_argument(self, arguments: Dict[str, Any], name: str) -> str:
        aliases = {
            "start_date": ("start_date", "start", "from", "begin_date", "begin", "start_time"),
            "end_date": ("end_date", "end", "to", "finish_date", "finish", "end_time"),
            "date": ("date", "report_date", "year"),
        }[name]
        for key in aliases:
            value = arguments.get(key)
            if value not in (None, ""):
                return self._normalize_akshare_date(str(value))
        return ""

    @staticmethod
    def _normalize_akshare_date(value: str) -> str:
        cleaned = value.strip()
        if re.fullmatch(r"\d{8}", cleaned):
            return cleaned
        match = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", cleaned)
        if match:
            return f"{int(match.group(1)):04d}{int(match.group(2)):02d}{int(match.group(3)):02d}"
        if re.fullmatch(r"\d{4}", cleaned):
            return cleaned
        return re.sub(r"\D", "", cleaned) or cleaned

    def _normalize_symbol_for_function(self, function: str, symbol: str, arguments: Dict[str, Any]) -> str:
        raw = symbol.strip()
        upper = raw.upper()
        if function == "stock_zh_a_daily":
            if re.fullmatch(r"\d{6}", raw):
                prefix = "sh" if raw.startswith(("5", "6", "9")) else "bj" if raw.startswith(("4", "8")) else "sz"
                return prefix + raw
            return raw.lower()
        if function in {
            "stock_profit_sheet_by_report_em",
            "stock_balance_sheet_by_report_em",
            "stock_cash_flow_sheet_by_report_em",
        }:
            if re.fullmatch(r"\d{6}", raw):
                prefix = "SH" if raw.startswith(("5", "6", "9")) else "BJ" if raw.startswith(("4", "8")) else "SZ"
                return prefix + raw
            if upper.endswith((".SH", ".SZ", ".BJ")):
                code, suffix = upper.split(".", 1)
                return suffix + code
        if function in {"stock_zh_a_hist", "stock_financial_abstract_ths", "stock_financial_abstract"}:
            return re.sub(r"\.(SH|SZ|BJ)$", "", upper)
        if function in {"stock_hk_hist", "fund_etf_hist_em"} and re.fullmatch(r"\d{1,5}", raw):
            return raw.zfill(5) if function == "stock_hk_hist" else raw
        if function == "stock_us_hist" and re.fullmatch(r"[A-Z.]{1,8}", upper) and not re.match(r"10[567]\.", upper):
            exchange = str(arguments.get("exchange") or arguments.get("market") or "").strip().upper()
            prefix = {"NASDAQ": "105", "NAS": "105", "NYSE": "106", "AMEX": "107"}.get(exchange, "105")
            return f"{prefix}.{upper}"
        if function in {"index_hist_sw", "index_component_sw"}:
            return re.sub(r"\.SI$", "", upper)
        if function in {"index_stock_cons_csindex", "index_stock_cons_weight_csindex"}:
            return re.sub(r"\.(SH|SZ|BJ|SS|CSI|GI)$", "", upper)
        if function == "currency_boc_sina":
            currency_aliases = {
                "USD": "美元",
                "USDCNY": "美元",
                "CNYUSD": "美元",
                "HKD": "港币",
                "HKDCNY": "港币",
                "EUR": "欧元",
                "EURCNY": "欧元",
                "JPY": "日元",
                "GBP": "英镑",
            }
            return currency_aliases.get(upper, raw)
        return raw

    def _normalize_sina_stock(self, symbol: str) -> str:
        raw = symbol.strip()
        upper = raw.upper()
        if re.fullmatch(r"\d{6}", raw):
            prefix = "sh" if raw.startswith(("5", "6", "9")) else "bj" if raw.startswith(("4", "8")) else "sz"
            return prefix + raw
        if upper.endswith((".SH", ".SZ", ".BJ")):
            code, suffix = upper.split(".", 1)
            return suffix.lower() + code
        return raw.lower()

    def _filter_query(self, function: str, arguments: Dict[str, Any]) -> str:
        explicit = arguments.get("filter") or arguments.get("row_filter")
        if explicit not in (None, ""):
            return str(explicit).strip()
        query = str(arguments.get("query") or arguments.get("查询内容") or "").strip()
        if function in self.snapshot_files:
            return query
        if function in self.no_symbol_functions:
            return query
        # For live historical/statement functions, query is normally the symbol
        # already passed into AkShare. Filtering the returned time series by the
        # same ticker would remove all rows because K-line rows usually do not
        # repeat the symbol.
        return "" if self._symbol_argument(arguments) else query

    def _filter_date_range(self, df: Any, arguments: Dict[str, Any]) -> Any:
        start = self._date_argument(arguments, "start_date")
        end = self._date_argument(arguments, "end_date")
        if not start and not end:
            return df
        date_columns = [
            "日期",
            "date",
            "交易日期",
            "trade_date",
            "trading_date",
            "时间",
            "统计时间",
            "报告期",
            "公告日期",
        ]
        for column in date_columns:
            if column not in df.columns:
                continue
            key = df[column].astype(str).map(self._normalize_akshare_date)
            mask = key.astype(bool)
            if start:
                mask = mask & (key >= start)
            if end:
                mask = mask & (key <= end)
            return df[mask]
        return df

    def _load_snapshot(self, function: str, pd: Any) -> Any:
        base_name = self.snapshot_files[function]
        snapshot_dir = Path(__file__).resolve().parents[3] / "official_benchmarks" / "finsearchcomp" / "data" / "market_snapshots"
        candidates = sorted(snapshot_dir.glob(f"{base_name}_*.csv"))
        if not candidates:
            candidates = sorted(snapshot_dir.glob(f"{base_name}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No bundled AkShare snapshot found for {function} in {snapshot_dir}")
        return pd.read_csv(candidates[-1], dtype=str)

    def _filter_dataframe(self, df: Any, query: str) -> Any:
        if not query:
            return df
        query_upper = query.upper()
        mask = None
        for column in df.columns:
            series = df[column].astype(str)
            column_mask = series.str.upper().str.contains(query_upper, na=False, regex=False)
            mask = column_mask if mask is None else (mask | column_mask)
        if mask is None:
            return df.head(0)
        return df[mask]

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _optional_positive_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _default_row_limit(action: str) -> Optional[int]:
        if action == "price_history":
            return None
        return 10
