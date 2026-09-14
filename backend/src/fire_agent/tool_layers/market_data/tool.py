from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, quote, urljoin, urlparse

from ...http_client import HTTPRequestError, http_get, http_post
from ...llm import parse_json_object
from ...providers import (
    TushareProvider,
    TushareProviderError,
    format_tushare_date,
    format_tushare_period,
    normalize_tushare_contract,
    normalize_tushare_fund_code,
    normalize_tushare_index_code,
    normalize_tushare_ts_code,
)
from ...schemas import ToolResult, compact_text
from ...tools import ActionDispatcher
from ..akshare_tool import AkShareTool
from .specs import MARKET_DATA_ACTIONS, MARKET_DATA_DESCRIPTION, TUSHARE_TYPED_ACTIONS
from ..finance_utils import (
    DATE_REGEX,
    FINANCE_AGENT_MAX_END_DATE,
    finance_agent_records_to_csv as _finance_agent_records_to_csv,
    finance_agent_retryable_post as _finance_agent_retryable_post,
    finance_agent_validate_date as _finance_agent_validate_date,
)


class MarketDataTool(AkShareTool):
    name = "market_data"
    description = MARKET_DATA_DESCRIPTION
    paid = False
    default_action = "capabilities"
    actions = MARKET_DATA_ACTIONS

    yahoo_symbol_aliases = {
        "DJI": "^DJI",
        "^DJI": "^DJI",
        "DOW": "^DJI",
        "DOW JONES": "^DJI",
        "FTSE": "^FTSE",
        "^FTSE": "^FTSE",
        "FTSE100": "^FTSE",
        "FTSE 100": "^FTSE",
        "UKX": "^FTSE",
        "N225": "^N225",
        "^N225": "^N225",
        "NI225": "^N225",
        "JP225": "^N225",
        "NIKKEI": "^N225",
        "NIKKEI 225": "^N225",
        "DAX": "^GDAXI",
        "GDAXI": "^GDAXI",
        "^GDAXI": "^GDAXI",
        "GERMAN DAX": "^GDAXI",
        "VIX": "^VIX",
        "^VIX": "^VIX",
        "VIXCLS": "^VIX",
        "DXY": "DX-Y.NYB",
        "USDX": "DX-Y.NYB",
        "USDOLLAR": "DX-Y.NYB",
        "DX-Y.NYB": "DX-Y.NYB",
        "CL": "CL=F",
        "WTI": "CL=F",
        "CL=F": "CL=F",
        "GC": "GC=F",
        "GOLD": "GC=F",
        "GC=F": "GC=F",
        "ZC": "ZC=F",
        "CORN": "ZC=F",
        "ZC=F": "ZC=F",
        "ZS": "ZS=F",
        "SOYBEAN": "ZS=F",
        "ZS=F": "ZS=F",
    }
    fred_series_aliases = {
        "FEDERAL FUNDS RATE": "FEDFUNDS",
        "US_INTEREST_RATE": "FEDFUNDS",
        "US INTEREST RATE": "FEDFUNDS",
        "US10Y": "DGS10",
        "10Y": "DGS10",
        "10 YEAR TREASURY": "DGS10",
        "10 YEAR TREASURY CONSTANT MATURITY": "DGS10",
        "DXY": "DTWEXBGS",
        "VIX": "VIXCLS",
        "VIXCLS": "VIXCLS",
        "M2SL": "M2SL",
        "M2V": "M2V",
        "DTB6": "DTB6",
        "6 MONTH TREASURY BILL SECONDARY MARKET RATE": "DTB6",
        "6 MONTH TREASURY BILL SECONDARY MARKET": "DTB6",
        "6 MONTH T BILL SECONDARY MARKET RATE": "DTB6",
        "6 MONTH T BILL SECONDARY MARKET": "DTB6",
        "6 MONTH SECONDARY MARKET RATE": "DTB6",
        "DGS6MO": "DGS6MO",
        "6 MONTH TREASURY CONSTANT MATURITY": "DGS6MO",
        "6 MONTH CONSTANT MATURITY": "DGS6MO",
        "TOTRESNS": "TOTRESNS",
        "BORROW": "BORROW",
        "TOTBKCR": "TOTBKCR",
        "A001RY2A225NXBE": "A001RY2A225NXBE",
        "HSN1FSA": "HSN1FSA",
        "NHSDCUSSA": "NHSDCUSSA",
        "RSFS": "RSFS",
    }
    fred_series_catalog = {
        "DTB6": {
            "series_id": "DTB6",
            "series_title": "6-Month Treasury Bill Secondary Market Rate, Discount Basis",
            "frequency": "Daily",
            "units": "Percent",
            "notes": "Use for 6-month Treasury bill Secondary Market Rate questions.",
        },
        "DGS6MO": {
            "series_id": "DGS6MO",
            "series_title": "Market Yield on U.S. Treasury Securities at 6-Month Constant Maturity, Quoted on an Investment Basis",
            "frequency": "Daily",
            "units": "Percent",
            "notes": "Use only when the question asks for 6-month Treasury constant maturity yield/rate.",
        },
        "DGS10": {
            "series_id": "DGS10",
            "series_title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis",
            "frequency": "Daily",
            "units": "Percent",
        },
        "FEDFUNDS": {
            "series_id": "FEDFUNDS",
            "series_title": "Federal Funds Effective Rate",
            "frequency": "Monthly",
            "units": "Percent",
        },
        "VIXCLS": {
            "series_id": "VIXCLS",
            "series_title": "CBOE Volatility Index: VIX",
            "frequency": "Daily",
            "units": "Index",
        },
        "M2SL": {
            "series_id": "M2SL",
            "series_title": "M2",
            "frequency": "Weekly",
            "units": "Billions of Dollars",
        },
        "M2V": {
            "series_id": "M2V",
            "series_title": "Velocity of M2 Money Stock",
            "frequency": "Quarterly",
            "units": "Ratio",
        },
    }
    world_bank_indicator_aliases = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "world_bank_gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "real_gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "gdp_current_usd": "NY.GDP.MKTP.CD",
        "nominal_gdp": "NY.GDP.MKTP.CD",
        "gdp_per_capita": "NY.GDP.PCAP.CD",
        "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
        "gdp_per_capita_current_usd": "NY.GDP.PCAP.CD",
        "gdp_per_capita_current_us": "NY.GDP.PCAP.CD",
        "gdp_per_capita_current_dollars": "NY.GDP.PCAP.CD",
        "unemployment": "SL.UEM.TOTL.ZS",
        "unemployment_rate": "SL.UEM.TOTL.ZS",
        "inflation": "FP.CPI.TOTL.ZG",
        "cpi": "FP.CPI.TOTL.ZG",
        "claims_on_central_government": "FM.AST.CGOV.ZG.M3",
        "claims_on_central_government_annual_growth": "FM.AST.CGOV.ZG.M3",
        "claims_on_central_government_annual_growth_as_of_broad_money": "FM.AST.CGOV.ZG.M3",
    }
    world_bank_indicator_catalog = {
        "NY.GDP.MKTP.KD.ZG": {
            "indicator": "NY.GDP.MKTP.KD.ZG",
            "indicator_name": "GDP growth (annual %)",
            "aliases": ["gdp growth", "real gdp growth", "real GDP growth rate", "annual GDP growth"],
        },
        "NY.GDP.MKTP.CD": {
            "indicator": "NY.GDP.MKTP.CD",
            "indicator_name": "GDP (current US$)",
            "aliases": ["nominal gdp", "gdp current usd", "GDP current US dollars"],
        },
        "NY.GDP.PCAP.CD": {
            "indicator": "NY.GDP.PCAP.CD",
            "indicator_name": "GDP per capita (current US$)",
            "aliases": [
                "gdp per capita",
                "gdp per capita current usd",
                "GDP per capita current US dollars",
                "per capita GDP",
            ],
        },
        "SL.UEM.TOTL.ZS": {
            "indicator": "SL.UEM.TOTL.ZS",
            "indicator_name": "Unemployment, total (% of total labor force) (modeled ILO estimate)",
            "aliases": ["unemployment", "unemployment rate"],
        },
        "FP.CPI.TOTL.ZG": {
            "indicator": "FP.CPI.TOTL.ZG",
            "indicator_name": "Inflation, consumer prices (annual %)",
            "aliases": ["inflation", "annual inflation", "consumer price inflation", "cpi annual percentage"],
        },
        "FM.AST.CGOV.ZG.M3": {
            "indicator": "FM.AST.CGOV.ZG.M3",
            "indicator_name": "Claims on central government (annual growth as % of broad money)",
            "aliases": [
                "claims on central government",
                "claims on central government annual growth",
                "claims on central government annual growth as % of broad money",
                "central government claims broad money",
            ],
        },
    }
    europe_country_iso3 = {
        "ALB",
        "AND",
        "AUT",
        "BEL",
        "BGR",
        "BIH",
        "BLR",
        "CHE",
        "CYP",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "HRV",
        "HUN",
        "IRL",
        "ISL",
        "ITA",
        "LIE",
        "LTU",
        "LUX",
        "LVA",
        "MCO",
        "MDA",
        "MKD",
        "MLT",
        "MNE",
        "NLD",
        "NOR",
        "POL",
        "PRT",
        "ROU",
        "RUS",
        "SMR",
        "SRB",
        "SVK",
        "SVN",
        "SWE",
        "TUR",
        "UKR",
        "XKX",
    }
    country_aliases = {
        "AU": "AUS",
        "AUSTRALIA": "AUS",
        "AUS": "AUS",
        "BRAZIL": "BRA",
        "CANADA": "CAN",
        "CN": "CHN",
        "CHINA": "CHN",
        "EGYPT": "EGY",
        "DE": "DEU",
        "GERMANY": "DEU",
        "ICELAND": "ISL",
        "ID": "IDN",
        "IDN": "IDN",
        "INDONESIA": "IDN",
        "ISRAEL": "ISR",
        "JP": "JPN",
        "JAPAN": "JPN",
        "JPN": "JPN",
        "KOREA": "KOR",
        "SOUTH KOREA": "KOR",
        "MEXICO": "MEX",
        "NEW ZEALAND": "NZL",
        "NZL": "NZL",
        "POLAND": "POL",
        "RUSSIA": "RUS",
        "SWEDEN": "SWE",
        "SOUTH AFRICA": "ZAF",
        "ZAF": "ZAF",
        "TUNISIA": "TUN",
        "UNITED KINGDOM": "GBR",
        "UK": "GBR",
        "UNITED STATES": "USA",
        "US": "USA",
        "USA": "USA",
        "ZIMBABWE": "ZWE",
    }
    comtrade_commodity_cache: List[Dict[str, Any]] = []
    CN_FINANCIAL_FIELD_CATALOG: Dict[str, Dict[str, Any]] = {
        "LONG_EQUITY_INVEST": {
            "label_zh": "长期股权投资",
            "aliases": ["长期股权投资", "long-term equity investment", "long term equity investment", "long equity invest"],
            "statement": "balance",
        },
        "DEFER_TAX_ASSET": {
            "label_zh": "递延所得税资产",
            "aliases": ["递延所得税资产", "deferred tax assets", "deferred tax asset"],
            "statement": "balance",
        },
        "INVENTORY": {
            "label_zh": "存货",
            "aliases": ["存货", "inventory", "inventories"],
            "statement": "balance",
        },
        "TOTAL_ASSETS": {
            "label_zh": "资产总计",
            "aliases": ["资产总计", "总资产", "total assets"],
            "statement": "balance",
        },
        "TOTAL_PARENT_EQUITY": {
            "label_zh": "归属于母公司股东权益合计",
            "aliases": ["归属于母公司股东权益", "归属于上市公司股东的净资产", "归母净资产", "parent equity", "stockholders equity attributable to parent"],
            "statement": "balance",
        },
        "PARENT_EQUITY_BALANCE": {
            "label_zh": "归属于母公司股东权益",
            "aliases": ["归属于母公司股东权益", "归属于上市公司股东的净资产", "归母净资产", "parent equity balance"],
            "statement": "balance",
        },
        "TOTAL_EQUITY": {
            "label_zh": "所有者权益合计",
            "aliases": ["所有者权益合计", "股东权益合计", "total equity"],
            "statement": "balance",
        },
        "PARENT_NETPROFIT": {
            "label_zh": "归属于母公司股东的净利润",
            "aliases": ["归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润", "net profit attributable to parent"],
            "statement": "income",
        },
        "RESEARCH_EXPENSE": {
            "label_zh": "研发费用",
            "aliases": ["研发费用", "研发投入", "research expense", "r&d expense", "research and development expense"],
            "statement": "income",
            "note": "利润表字段通常是研发费用；年报披露的研发投入可能包含资本化投入，需核对题目口径。",
        },
        "ME_RESEARCH_EXPENSE": {
            "label_zh": "管理费用中的研发费用",
            "aliases": ["管理费用研发费用", "管理费用中的研发费用", "研发费用", "研发投入"],
            "statement": "income",
            "note": "该字段是管理费用相关研发字段；研发投入口径可能不同，需核对年报披露。",
        },
        "OPERATE_INCOME": {
            "label_zh": "营业收入",
            "aliases": ["营业收入", "operating revenue", "operating income"],
            "statement": "income",
        },
        "TOTAL_OPERATE_INCOME": {
            "label_zh": "营业总收入",
            "aliases": ["营业总收入", "total operating income", "total operating revenue"],
            "statement": "income",
        },
    }
    A_SHARE_STANDARD_AKSHARE_FUNCTIONS = {
        "stock_profit_sheet_by_report_em",
        "stock_balance_sheet_by_report_em",
        "stock_cash_flow_sheet_by_report_em",
        "stock_financial_abstract",
        "stock_financial_abstract_ths",
    }
    TUSHARE_FINANCIAL_FIELD_SCHEMA: Dict[str, Dict[str, Dict[str, Any]]] = {
        "fina_indicator": {
            "roe": {
                "label_zh": "加权净资产收益率",
                "aliases": ["加权净资产收益率", "加权平均净资产收益率", "weighted roe", "weighted return on equity"],
                "unit": "%",
            },
            "roe_dt": {
                "label_zh": "摊薄净资产收益率",
                "aliases": ["摊薄净资产收益率", "摊薄roe", "diluted roe", "roe diluted"],
                "unit": "%",
            },
            "grossprofit_margin": {
                "label_zh": "销售毛利率",
                "aliases": ["销售毛利率", "毛利率", "gross margin", "gross profit margin"],
                "unit": "%",
            },
            "netprofit_margin": {
                "label_zh": "销售净利率",
                "aliases": ["销售净利率", "净利率", "net profit margin"],
                "unit": "%",
            },
            "current_ratio": {
                "label_zh": "流动比率",
                "aliases": ["流动比率", "current ratio"],
                "unit": "ratio",
            },
            "quick_ratio": {
                "label_zh": "速动比率",
                "aliases": ["速动比率", "quick ratio", "acid test ratio"],
                "unit": "ratio",
            },
            "debt_to_assets": {
                "label_zh": "资产负债率",
                "aliases": ["资产负债率", "debt to assets", "debt-to-assets"],
                "unit": "%",
            },
        },
        "income": {
            "INT_INCOME": {
                "label_zh": "利息收入",
                "aliases": ["利息收入", "interest income"],
                "unit": "元",
            },
            "INT_EXP": {
                "label_zh": "利息支出",
                "aliases": ["利息支出", "interest expense"],
                "unit": "元",
            },
            "__NET_INTEREST_INCOME": {
                "label_zh": "净利息收入",
                "aliases": ["净利息收入", "利息净收入", "net interest income"],
                "unit": "元",
                "expression": "INT_INCOME - INT_EXP",
                "source_fields": ["INT_INCOME", "INT_EXP"],
                "note": "银行常用净利息收入可由利息收入减利息支出得到；若源表有同名直接字段，应优先核对直接字段。",
            },
            "N_INCOME_ATTR_P": {
                "label_zh": "归属于母公司股东的净利润",
                "aliases": ["归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"],
                "unit": "元",
            },
            "N_INCOME": {
                "label_zh": "净利润",
                "aliases": ["净利润", "net income"],
                "unit": "元",
            },
            "TOTAL_REVENUE": {
                "label_zh": "营业总收入",
                "aliases": ["营业总收入", "total revenue", "total operating revenue"],
                "unit": "元",
            },
            "REVENUE": {
                "label_zh": "营业收入",
                "aliases": ["营业收入", "revenue", "operating revenue"],
                "unit": "元",
            },
            "RD_EXP": {
                "label_zh": "研发费用",
                "aliases": ["研发费用", "research expense", "r&d expense"],
                "unit": "元",
                "note": "研发费用不一定等同年报披露的研发投入合计；若题目问研发投入，需核对资本化研发投入。",
            },
        },
        "balancesheet": {
            "MONEY_CAP": {
                "label_zh": "货币资金",
                "aliases": ["货币资金", "cash and bank balances", "cash"],
                "unit": "元",
            },
            "INVENTORIES": {
                "label_zh": "存货",
                "aliases": ["存货", "inventory", "inventories"],
                "unit": "元",
            },
            "TOTAL_ASSETS": {
                "label_zh": "资产总计",
                "aliases": ["资产总计", "总资产", "total assets"],
                "unit": "元",
            },
            "MINORITY_INT": {
                "label_zh": "少数股东权益",
                "aliases": ["少数股东权益", "minority interest", "non-controlling interests"],
                "unit": "元",
            },
            "TOTAL_Hldr_EQY_EXCL_MIN_INT": {
                "label_zh": "归属于母公司股东权益合计",
                "aliases": ["归属于母公司股东权益", "归属于上市公司股东的净资产", "parent equity"],
                "unit": "元",
            },
            "TOTAL_HLDR_EQY_INC_MIN_INT": {
                "label_zh": "所有者权益合计",
                "aliases": ["所有者权益合计", "股东权益合计", "total equity"],
                "unit": "元",
            },
            "LT_BORR": {
                "label_zh": "长期借款",
                "aliases": ["长期借款", "long-term borrowing", "long term borrowings"],
                "unit": "元",
            },
            "ST_BORR": {
                "label_zh": "短期借款",
                "aliases": ["短期借款", "short-term borrowing", "short term borrowings"],
                "unit": "元",
            },
            "LT_EQT_INVEST": {
                "label_zh": "长期股权投资",
                "aliases": ["长期股权投资", "long-term equity investment"],
                "unit": "元",
            },
            "DEFER_TAX_ASSETS": {
                "label_zh": "递延所得税资产",
                "aliases": ["递延所得税资产", "deferred tax assets"],
                "unit": "元",
            },
        },
        "cashflow": {
            "N_CASHFLOW_ACT": {
                "label_zh": "经营活动产生的现金流量净额",
                "aliases": ["经营活动现金流净额", "经营活动产生的现金流量净额", "cash flow from operating activities"],
                "unit": "元",
            },
            "N_CASHFLOW_INV_ACT": {
                "label_zh": "投资活动产生的现金流量净额",
                "aliases": ["投资活动现金流净额", "投资活动产生的现金流量净额"],
                "unit": "元",
            },
            "N_CASH_FLOWS_FNC_ACT": {
                "label_zh": "筹资活动产生的现金流量净额",
                "aliases": ["筹资活动现金流净额", "筹资活动产生的现金流量净额"],
                "unit": "元",
            },
        },
    }
    TUSHARE_DIVIDEND_FIELD_SCHEMA: Dict[str, Dict[str, Any]] = {
        "CASH_DVD_DIV": {
            "label_zh": "每股派息",
            "aliases": ["每股派息", "每股股息", "cash dividend per share"],
            "unit": "元/股",
        },
        "PER_SHARE_BEFORE_TAX": {
            "label_zh": "税前每股派息",
            "aliases": ["税前每股派息", "税前每股股息", "per share before tax"],
            "unit": "元/股",
        },
        "PER_SHARE_AFTER_TAX": {
            "label_zh": "税后每股派息",
            "aliases": ["税后每股派息", "税后每股股息", "per share after tax"],
            "unit": "元/股",
        },
        "DIV_PROC": {
            "label_zh": "分红实施进度",
            "aliases": ["分红进度", "实施", "预案", "dividend process"],
            "unit": "text",
        },
        "END_DATE": {
            "label_zh": "分红对应报告期",
            "aliases": ["报告期", "end date", "period"],
            "unit": "date",
        },
        "EX_DATE": {
            "label_zh": "除权除息日",
            "aliases": ["除权除息日", "ex date"],
            "unit": "date",
        },
    }
    akshare_macro_aliases = {
        "社会融资规模存量": "macro_china_shrzgm",
        "社会融资规模存量同比": "macro_china_shrzgm",
        "macro_china_tax_revenue": "macro_china_national_tax_receipts",
        "中国国家财政收入": "macro_china_national_tax_receipts",
        "macro_china_insurance_premium": "macro_china_insurance_income",
        "中国外汇储备": "macro_china_fx_reserves_yearly",
        "japan_unemployment_rate": "macro_japan_unemployment_rate",
        "日本失业率": "macro_japan_unemployment_rate",
        "Australia CPI": "macro_australia_cpi_yearly",
        "CPI Australia": "macro_australia_cpi_yearly",
        "ifo business climate germany": "macro_germany_ifo",
        "US Retail Sales Total NSA": "macro_usa_retail_sales",
    }

    tushare_typed_actions = TUSHARE_TYPED_ACTIONS

    comtrade_reporter_cache: Dict[str, str] = {}
    comtrade_partner_cache: Dict[str, str] = {}

    def _structured_error(
        self,
        provider: str,
        action: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.1,
    ) -> ToolResult:
        return self.results.error(
            provider,
            action,
            message,
            metadata=metadata,
            confidence=confidence,
        )

    def reset_for_task(self) -> None:
        self._dataframe_failure_counts: Dict[str, int] = {}

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        return self.default_action if candidate == "default" else candidate

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        action = self.normalize_action(action)
        clean_arguments = arguments or {}
        if action == "capabilities":
            return self._capabilities_result(clean_arguments)
        if action in self.tushare_typed_actions:
            return self._forward_tushare_typed_action(action, clean_arguments, timeout=timeout)
        if action == "market_table":
            blocked = self._standard_a_share_market_table_error(clean_arguments)
            if blocked is not None:
                return blocked
        dispatched = self._dispatch_explicit_action(action, clean_arguments, timeout)
        if dispatched is not None:
            return dispatched
        if action == "financial_statement":
            sec_result = self._maybe_sec_fundamentals(dict(clean_arguments), timeout=timeout)
            if sec_result is not None:
                return sec_result
            tushare_result = self._maybe_tushare_a_share_financials(dict(clean_arguments), timeout=timeout)
            if tushare_result is not None:
                return tushare_result
            blocked = self._standard_a_share_financial_statement_error(clean_arguments)
            if blocked is not None:
                return blocked
        mapped = self._map_market_action(action, dict(clean_arguments))
        result = super().forward("call", mapped, timeout=timeout)
        result.tool_family = self.name
        result.action = action
        if action == "financial_statement" and result.status == "success":
            self._augment_cn_financial_statement_result(result, clean_arguments, mapped)
        if (
            action == "financial_statement"
            and result.status == "error"
            and str(mapped.get("function") or "").startswith("stock_financial_us_")
        ):
            ticker = str(mapped.get("ticker") or "").strip().upper()
            result.error = (
                f"AkShare US financial statement lookup failed for ticker {ticker}: {result.error}. "
                "This data source may not cover the ticker or statement reliably; for US SEC filings, "
                "use sec_search/sec_reader as the authoritative fallback."
            )
            result.metadata = {
                **(result.metadata or {}),
                "function": mapped.get("function"),
                "ticker": ticker,
                "statement_symbol": mapped.get("statement_symbol"),
                "period": mapped.get("period"),
                "recommended_fallback": "sec_search/sec_reader",
            }
        return result

    def _dispatch_explicit_action(
        self,
        action: str,
        arguments: Dict[str, Any],
        timeout: int,
    ) -> Optional[ToolResult]:
        dispatcher = getattr(self, "_explicit_action_dispatcher", None)
        if dispatcher is None:
            dispatcher = ActionDispatcher(
                {
                    "price_history": self._forward_price_history,
                    "option_chain_metrics": self._forward_option_chain_metrics,
                    "macro_series": self._forward_macro_series,
                    "trade_series": self._forward_trade_series,
                    "futures_contract_history": self._forward_futures_contract_history,
                    "metals_price": self._forward_metals_price,
                    "official_attachment_table": self._forward_official_attachment_table,
                    "source_catalog": self._forward_source_catalog,
                }
            )
            self._explicit_action_dispatcher = dispatcher
        return dispatcher.dispatch(action, arguments, timeout)

    def _capabilities_result(self, arguments: Dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        rows = [
            {
                "evidence_need": "China A-share standard company data",
                "preferred_actions": "equity_universe -> equity_financials / equity_dividend / equity_price_history / equity_daily_basic / equity_factor_history",
                "source_family": "Tushare typed schema",
                "avoid": "web snippets, financial_statement(provider='akshare'), market_table statement functions",
            },
            {
                "evidence_need": "US listed company historical GAAP fundamentals",
                "preferred_actions": "financial_statement(provider='sec') or financial_statement(auto)",
                "source_family": "SEC XBRL companyfacts",
                "avoid": "AkShare US financial_statement rough path when SEC data is available",
            },
            {
                "evidence_need": "global prices / indexes / FX / simple market time series",
                "preferred_actions": "price_history(provider='yahoo') or quote_snapshot",
                "source_family": "Yahoo Finance",
                "avoid": "web snippets for exact close prices",
            },
            {
                "evidence_need": "China macro series and interbank rates",
                "preferred_actions": "cn_macro_series / shibor_series",
                "source_family": "Tushare China macro/rate schema",
                "avoid": "generic macro_series(auto) for China-specific official Tushare tables",
            },
            {
                "evidence_need": "US macro / vintage macro",
                "preferred_actions": "macro_series(provider='fred') or macro_series(provider='alfred')",
                "source_family": "FRED/ALFRED",
                "avoid": "auto current-data substitution for vintage questions",
            },
            {
                "evidence_need": "country annual macro",
                "preferred_actions": "macro_series(provider='world_bank')",
                "source_family": "World Bank",
                "avoid": "nearby search snippets",
            },
            {
                "evidence_need": "China index constituents/history/catalog, including Shenwan .SI and CSIndex/Zhongzheng indexes",
                "preferred_actions": "index_basic/index_catalog -> index_constituents or index_weight; index_history for OHLCV",
                "source_family": "Namespace-routed Tushare + AkShare SW/CSIndex",
                "avoid": "generic web_search or market_table for supported index codes such as 801125.SI, 851246.SI, 000300, 000905",
            },
            {
                "evidence_need": "China fund and ETF catalog, NAV, holdings, and OHLCV",
                "preferred_actions": "fund_basic -> fund_nav / fund_history / fund_portfolio",
                "source_family": "Tushare fund schema",
                "avoid": "web snippets for exact NAV/holding/date rows",
            },
            {
                "evidence_need": "China futures catalog, mappings, daily prices, warehouse receipts, and settlement rows",
                "preferred_actions": "cn_futures_basic -> cn_futures_mapping / cn_futures_daily / cn_futures_warehouse / cn_futures_settle; futures_contract_history",
                "source_family": "Tushare futures schema plus explicit contract history",
                "avoid": "continuous-contract substitution unless the question asks for continuous/main contract basis",
            },
            {
                "evidence_need": "China commodity option-chain analytics",
                "preferred_actions": "cn_options_basic -> cn_options_daily; option_chain_metrics(exchange='SHFE', trade_date=..., metric='time_value'/'intrinsic_value'/'moneyness')",
                "source_family": "AkShare exchange option-chain tables plus exchange futures settlement tables",
                "avoid": "web snippets for ranked chain metrics that require contract parsing and underlying settlement alignment",
            },
            {
                "evidence_need": "China bond yield curves and SHIBOR/interbank rates",
                "preferred_actions": "cn_bond_yield_curve / shibor_series",
                "source_family": "Tushare ChinaBond and SHIBOR schema",
                "avoid": "web snippets for exact curve/rate rows",
            },
            {
                "evidence_need": "official precious-metals benchmark prices",
                "preferred_actions": "metals_price(provider='lbma')",
                "source_family": "official LBMA connector",
                "avoid": "search snippets for exact benchmark session/date prices",
            },
            {
                "evidence_need": "OECD/IEA/trade/SAFE/NFRA official tables",
                "preferred_actions": "source_catalog + macro_series(provider='oecd'/'iea'), trade_series, official_attachment_table",
                "source_family": "explicit official provider",
                "avoid": "macro_series(auto) or AkShare fallback",
            },
            {
                "evidence_need": "uncovered China-market table",
                "preferred_actions": "market_table(function='<explicit AkShare API>')",
                "source_family": "AkShare escape hatch",
                "avoid": "standard A-share statements/indicators/dividends already covered by typed Tushare",
            },
        ]
        filtered = False
        if query:
            routed_rows = self._capability_rows_for_query(rows, query)
            if routed_rows:
                rows = routed_rows
                filtered = True
        metadata: Dict[str, Any] = {
            "query": query,
            "filtered": filtered,
            "note": "This action is guidance only and does not fetch data.",
        }
        if filtered:
            recommended_actions = self._capability_recommended_actions(rows)
            if recommended_actions:
                metadata["recommended_actions"] = recommended_actions
        return self._records_result(
            "market_router",
            "capabilities",
            rows,
            metadata=metadata,
            confidence=0.95,
        )

    def _capability_rows_for_query(self, rows: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        rules: Dict[str, Tuple[str, ...]] = {
            "China A-share standard company data": (
                "china a-share",
                "a-share",
                "a-share company",
                "listed company",
                "a股公司",
                "上市公司",
                "证券代码",
                "股票代码",
                "股票简称",
                "分红",
                "净利润",
                "财报",
            ),
            "US listed company historical GAAP fundamentals": (
                "sec xbrl",
                "gaap",
                "10-k",
                "10-q",
                "revenue",
                "net income",
                "eps",
                "income statement",
                "balance sheet",
                "cash flow",
            ),
            "global prices / indexes / FX / simple market time series": (
                "share price",
                "stock price",
                "close price",
                "closing price",
                "adjusted close",
                "price return",
                "stock return",
                "total return",
                "trading date",
                "historical price",
                "ohlc",
                "yahoo",
                "fx",
                "股价",
                "股票收盘",
                "汇率",
                "复权",
            ),
            "China macro series and interbank rates": (
                "china macro",
                "cn macro",
                "chinese macro",
                "china gdp",
                "china cpi",
                "china ppi",
                "china pmi",
                "cpi",
                "ppi",
                "pmi",
                "shibor",
                "interbank rate",
                "宏观",
                "居民消费价格",
                "工业生产者",
                "采购经理",
                "社融",
                "货币供应",
                "利率",
            ),
            "US macro / vintage macro": (
                "fred",
                "alfred",
                "vintage",
                "us cpi",
                "us ppi",
                "us pmi",
                "us gdp",
                "unemployment",
                "federal reserve",
            ),
            "country annual macro": (
                "world bank",
                "country annual",
                "gdp per capita",
                "population",
                "poverty",
                "life expectancy",
            ),
            "China index constituents/history/catalog, including Shenwan .SI and CSIndex/Zhongzheng indexes": (
                "china index",
                "index constituent",
                "constituent weight",
                "index weight",
                "index history",
                "csi",
                "csindex",
                "shenwan",
                "zhongzheng",
                "指数",
                "成分",
                "权重",
                "申万",
                "中证",
            ),
            "China fund and ETF catalog, NAV, holdings, and OHLCV": (
                "fund",
                "etf",
                "nav",
                "portfolio",
                "holding",
                "fund holding",
                "fund daily",
                "基金",
                "净值",
                "基金持仓",
                "重仓股",
            ),
            "China futures catalog, mappings, daily prices, warehouse receipts, and settlement rows": (
                "future",
                "futures",
                "futures contract",
                "settlement",
                "warehouse receipt",
                "continuous contract",
                "main contract",
                "期货",
                "合约",
                "结算",
                "仓单",
                "主力",
            ),
            "China commodity option-chain analytics": (
                "option chain",
                "moneyness",
                "intrinsic value",
                "time value",
                "commodity option",
                "期权",
            ),
            "China bond yield curves and SHIBOR/interbank rates": (
                "bond yield",
                "yield curve",
                "chinabond",
                "china bond",
                "shibor",
                "interbank",
                "债券",
                "收益率曲线",
                "国债",
                "同业",
            ),
            "official precious-metals benchmark prices": (
                "gold",
                "silver",
                "platinum",
                "palladium",
                "precious metal",
                "lbma",
                "金",
                "银",
                "贵金属",
            ),
            "OECD/IEA/trade/SAFE/NFRA official tables": (
                "oecd",
                "iea",
                "comtrade",
                "wits",
                "safe",
                "nfra",
                "official attachment",
                "official table",
                "trade series",
                "import",
                "export",
                "外汇局",
                "进出口",
                "贸易",
            ),
            "uncovered China-market table": (
                "akshare",
                "uncovered",
                "unsupported",
                "escape hatch",
            ),
        }
        selected: List[Dict[str, Any]] = []
        for row in rows:
            need = str(row.get("evidence_need") or "")
            terms = rules.get(need)
            if terms and self._capability_query_matches(query, terms):
                selected.append(row)
        return selected[:4]

    @staticmethod
    def _capability_query_matches(query: str, terms: Tuple[str, ...]) -> bool:
        text = str(query or "").lower()
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        for term in terms:
            term_text = str(term or "").lower().strip()
            if not term_text:
                continue
            if re.search(r"[\u4e00-\u9fff]", term_text):
                if re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", term_text) in compact:
                    return True
                continue
            if " " in term_text or "-" in term_text:
                if term_text in text:
                    return True
                continue
            pattern = rf"(?<![0-9a-z]){re.escape(term_text)}(?![0-9a-z])"
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def _capability_recommended_actions(rows: List[Dict[str, Any]]) -> List[str]:
        selected: List[str] = []
        for row in rows:
            text = str(row.get("preferred_actions") or "").lower()
            for action_name in MARKET_DATA_ACTIONS:
                if action_name == "capabilities":
                    continue
                pattern = rf"(?<![0-9a-z_]){re.escape(action_name.lower())}(?![0-9a-z_])"
                if re.search(pattern, text) and action_name not in selected:
                    selected.append(action_name)
        return selected

    def _standard_a_share_market_table_error(self, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        function = str(arguments.get("function") or "").strip()
        if function not in self.A_SHARE_STANDARD_AKSHARE_FUNCTIONS:
            return None
        return self._structured_error(
            "akshare",
            "market_table",
            (
                f"AkShare market_table function {function!r} is a rough generic path for standard China A-share "
                "financial statements. Use market_data.equity_universe to resolve a Chinese company name to ts_code, "
                "then market_data.equity_financials/equity_dividend for standard financial statements, indicators, "
                "and dividends. If the exact field is not covered by typed Tushare, read the official "
                "company announcement or non-standard table with structured_table_reader instead."
            ),
            metadata={
                "requested_function": function,
                "recommended_typed_actions": ["equity_universe", "equity_financials", "equity_dividend"],
            },
            confidence=0.2,
        )

    def _standard_a_share_financial_statement_error(self, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        provider = self._provider_hint(arguments)
        if provider != "akshare":
            return None
        ticker = self._market_symbol(arguments)
        if not normalize_tushare_ts_code(ticker) and not self._looks_like_cn_a_share_standard_financial_request(arguments):
            return None
        return self._structured_error(
            "akshare",
            "financial_statement",
            (
                "China A-share standard financial statements, indicators, and dividends should use typed "
                "Tushare actions rather than AkShare financial_statement. Use equity_financials "
                "or equity_dividend; if the typed schema does not cover the exact field, use the official "
                "announcement/table with structured_table_reader."
            ),
            metadata={
                "requested_provider": "akshare",
                "requested_ticker": ticker,
                "recommended_typed_actions": ["equity_financials", "equity_dividend"],
            },
            confidence=0.2,
        )

    def _looks_like_cn_a_share_standard_financial_request(self, arguments: Dict[str, Any]) -> bool:
        request_text = self._normalize_lookup_text(
            " ".join(
                str(arguments.get(key) or "")
                for key in ("ticker", "symbol", "query", "name", "statement_type", "indicator", "metric", "fields")
            )
        )
        if not request_text:
            return False
        if re.search(r"[\u4e00-\u9fff]", str(arguments.get("ticker") or arguments.get("query") or arguments.get("name") or "")):
            return True
        for schema in self.TUSHARE_FINANCIAL_FIELD_SCHEMA.values():
            for field, meta in schema.items():
                candidates = [field, meta.get("label_zh") or "", *(meta.get("aliases") or [])]
                for candidate in candidates:
                    normalized = self._normalize_lookup_text(candidate)
                    if normalized and normalized in request_text:
                        return True
        for field, meta in self.TUSHARE_DIVIDEND_FIELD_SCHEMA.items():
            candidates = [field, meta.get("label_zh") or "", *(meta.get("aliases") or [])]
            for candidate in candidates:
                normalized = self._normalize_lookup_text(candidate)
                if normalized and normalized in request_text:
                    return True
        return False

    def _maybe_tushare_a_share_financials(self, arguments: Dict[str, Any], timeout: int = 30) -> Optional[ToolResult]:
        provider = self._provider_hint(arguments)
        if provider not in {"auto", "", "tushare_http"}:
            return None
        ticker = self._market_symbol(arguments)
        if not normalize_tushare_ts_code(ticker):
            return None
        routed = dict(arguments)
        if not routed.get("statement_type"):
            inferred = self._infer_tushare_financial_endpoint(routed)
            if inferred:
                routed["statement_type"] = inferred
        return self._forward_tushare_typed_action("equity_financials", routed, timeout=timeout)

    def _map_market_action(self, action: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if action == "market_table":
            arguments.setdefault("source", "auto")
            return arguments
        if action == "quote_snapshot":
            function = self._snapshot_function(arguments)
            return {"function": function, "query": self._market_symbol(arguments), "limit": arguments.get("limit", 10), "source": arguments.get("source", "snapshot"), "category": arguments.get("category")}
        if action == "price_history":
            function = self._history_function(arguments)
            max_end_date = str(arguments.get("max_end_date") or "9999-12-31").strip()
            if not re.match(DATE_REGEX, max_end_date):
                raise ValueError(f"max_end_date {max_end_date!r} is not in yyyy-mm-dd format")
            start_date = _finance_agent_validate_date(arguments.get("start_date"), "start_date", max_end_date)
            end_date = _finance_agent_validate_date(arguments.get("end_date"), "end_date", max_end_date)
            if start_date > end_date:
                raise ValueError(f"start_date '{start_date}' is later than end_date '{end_date}'.")
            return {
                "function": function,
                "ticker": self._market_symbol(arguments),
                "query": self._market_symbol(arguments),
                "start_date": start_date,
                "end_date": end_date,
                "adjust": arguments.get("adjust", ""),
                "limit": arguments.get("limit"),
                "source": "live",
            }
        if action == "financial_statement":
            statement_type = str(arguments.get("statement_type") or arguments.get("indicator") or "abstract").lower()
            ticker = self._market_symbol(arguments)
            if self._looks_like_us_equity_ticker(ticker):
                function, statement_symbol = self._us_financial_statement_function(statement_type)
                mapped: Dict[str, Any] = {
                    "function": function,
                    "ticker": ticker.upper(),
                    "query": ticker.upper(),
                    "period": self._us_financial_statement_period(arguments),
                    "limit": arguments.get("limit", 20),
                    "source": "live",
                }
                if statement_symbol:
                    mapped["statement_symbol"] = statement_symbol
                return mapped
            function = {
                "income": "stock_profit_sheet_by_report_em",
                "income_statement": "stock_profit_sheet_by_report_em",
                "profit": "stock_profit_sheet_by_report_em",
                "balance": "stock_balance_sheet_by_report_em",
                "balance_sheet": "stock_balance_sheet_by_report_em",
                "cash": "stock_cash_flow_sheet_by_report_em",
                "cashflow": "stock_cash_flow_sheet_by_report_em",
                "cash_flow": "stock_cash_flow_sheet_by_report_em",
                "cash_flow_statement": "stock_cash_flow_sheet_by_report_em",
                "abstract": "stock_financial_abstract_ths",
            }.get(statement_type, statement_type if statement_type.startswith("stock_") else "stock_financial_abstract_ths")
            return {"function": function, "ticker": ticker, "query": ticker, "date": arguments.get("date") or arguments.get("period"), "indicator": arguments.get("indicator"), "limit": arguments.get("limit", 20), "source": "live"}
        if action == "macro_series":
            return {"function": arguments.get("series"), "query": arguments.get("query", ""), "start_date": arguments.get("start_date"), "end_date": arguments.get("end_date"), "limit": arguments.get("limit", 20), "source": "live"}
        return arguments

    def _forward_tushare_typed_action(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        index_routed_actions = {"index_history", "index_weight", "index_constituents", "index_basic", "index_catalog"}
        index_provider_allowed = action in index_routed_actions and provider in {"akshare", "sw", "shenwan", "csindex", "csi"}
        if provider not in {"auto", "", "tushare_http"} and not index_provider_allowed:
            return self._structured_error(
                provider,
                action,
                f"{self.name}.{action} is served by Tushare-compatible providers only; use another market_data action for provider={provider!r}.",
                metadata={"requested_provider": provider},
            )

        if action == "equity_price_history":
            return self._fetch_tushare_equity_price_history(action, arguments, timeout=timeout)

        if action == "equity_daily_basic":
            if self._equity_daily_basic_needs_price_history(arguments):
                routed_args = dict(arguments)
                result = self._fetch_tushare_equity_price_history("equity_price_history", routed_args, timeout=timeout)
                result.action = action
                result.metadata = {
                    **(result.metadata or {}),
                    "routed_from": "equity_daily_basic",
                    "routed_to": "equity_price_history",
                    "route_reason": "requested daily OHLCV/return fields are served by Tushare daily, not daily_basic",
                }
                return result
            ts_code = normalize_tushare_ts_code(self._market_symbol(arguments))
            if not ts_code:
                return self._tushare_symbol_error(action, arguments)
            params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=ts_code)
            return self._call_tushare_endpoint(
                action,
                "daily_basic",
                params,
                start_date=start_date,
                end_date=end_date,
                sort_by="trade_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={
                    "ts_code": ts_code,
                    "unit_schema": {
                        "turnover_rate": "percent",
                        "turnover_rate_f": "percent",
                        "volume_ratio": "ratio",
                        "pe": "ratio",
                        "pe_ttm": "ratio",
                        "pb": "ratio",
                        "ps": "ratio",
                        "ps_ttm": "ratio",
                        "dv_ratio": "percent",
                        "dv_ttm": "percent",
                        "total_share": "10k shares",
                        "float_share": "10k shares",
                        "free_share": "10k shares",
                        "total_mv": "10k CNY",
                        "circ_mv": "10k CNY",
                    },
                },
            )

        if action == "equity_factor_history":
            ts_code = normalize_tushare_ts_code(self._market_symbol(arguments))
            if not ts_code:
                return self._tushare_symbol_error(action, arguments)
            kind = str(arguments.get("kind") or arguments.get("endpoint") or "adj_factor").strip().lower()
            endpoint = "stk_factor" if kind in {"stk_factor", "stock_factor", "factor"} else "adj_factor"
            params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=ts_code)
            return self._call_tushare_endpoint(
                action,
                endpoint,
                params,
                start_date=start_date,
                end_date=end_date,
                sort_by="trade_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"ts_code": ts_code, "kind": endpoint},
            )

        if action == "equity_financials":
            ts_code = normalize_tushare_ts_code(self._market_symbol(arguments))
            if not ts_code:
                return self._tushare_symbol_error(action, arguments)
            endpoint = self._tushare_financial_endpoint(arguments)
            if not endpoint:
                return self._structured_error(
                    "tushare_http",
                    action,
                    "Unsupported statement_type for equity_financials. Use income, balance, cashflow, or indicator.",
                    metadata={"statement_type": arguments.get("statement_type")},
                )
            params = {
                "ts_code": ts_code,
                "ann_date": format_tushare_date(arguments.get("ann_date") or arguments.get("date"), is_end=True),
                "start_date": format_tushare_date(arguments.get("start_date"), is_end=False),
                "end_date": format_tushare_date(arguments.get("end_date"), is_end=True),
                "period": format_tushare_period(arguments.get("period")),
                "report_type": arguments.get("report_type"),
                "fields": self._tushare_fields(arguments),
            }
            result = self._call_tushare_endpoint(
                action,
                endpoint,
                params,
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                sort_by="ann_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"ts_code": ts_code, "statement_endpoint": endpoint},
            )
            self._augment_tushare_financial_result(result, endpoint, arguments)
            return result

        if action == "equity_dividend":
            ts_code = normalize_tushare_ts_code(self._market_symbol(arguments))
            if not ts_code:
                return self._tushare_symbol_error(action, arguments)
            params = {
                "ts_code": ts_code,
                "ann_date": format_tushare_date(arguments.get("ann_date") or arguments.get("date"), is_end=True),
                "record_date": format_tushare_date(arguments.get("record_date"), is_end=True),
                "ex_date": format_tushare_date(arguments.get("ex_date"), is_end=True),
                "imp_ann_date": format_tushare_date(arguments.get("imp_ann_date"), is_end=True),
                "fields": self._tushare_fields(arguments),
            }
            result = self._call_tushare_endpoint(
                action,
                "dividend",
                params,
                sort_by="ann_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"ts_code": ts_code, "requested_period": arguments.get("period")},
            )
            self._augment_tushare_dividend_result(result, arguments)
            return result

        if action == "equity_universe":
            params = {
                "exchange": arguments.get("exchange"),
                "market": arguments.get("market"),
                "list_status": arguments.get("list_status") or "L",
                "is_hs": arguments.get("is_hs"),
                "fields": self._tushare_fields(arguments),
            }
            query = str(arguments.get("query") or arguments.get("ticker") or arguments.get("name") or "").strip()
            return self._call_tushare_endpoint(
                action,
                "stock_basic",
                params,
                query=query,
                query_fields=["ts_code", "symbol", "name"],
                limit=self._optional_positive_int(arguments.get("limit")),
                limit_mode="head",
                metadata={"query": query},
            )

        if action == "index_history":
            batch_codes = self._index_code_batch_values(arguments)
            if len(batch_codes) > 1:
                return self._forward_index_batch(action, batch_codes, arguments, timeout=timeout)
            namespace = self._china_index_namespace(arguments)
            if namespace == "sw":
                return self._forward_sw_index_history(action, arguments, timeout=timeout)
            if namespace == "csindex":
                return self._forward_csindex_index_history(action, arguments, timeout=timeout)
            index_code = normalize_tushare_index_code(arguments.get("index_code") or self._market_symbol(arguments))
            if not index_code:
                return self._structured_error(
                    "tushare_http",
                    action,
                    "index_history requires an index_code such as 000300.SH or 399001.SZ; use provider='sw' for .SI/Shenwan indexes or provider='csindex' for CSI/Zhongzheng indexes.",
                    metadata={"requested_index_code": arguments.get("index_code") or self._market_symbol(arguments)},
                )
            params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=index_code)
            return self._call_tushare_endpoint(
                action,
                "index_daily",
                params,
                start_date=start_date,
                end_date=end_date,
                sort_by="trade_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={
                    "index_code": index_code,
                    "adjust": "none",
                    "volume_unit": "hand",
                    "amount_unit": "thousand CNY",
                },
            )

        if action in {"index_weight", "index_constituents"}:
            batch_codes = self._index_code_batch_values(arguments)
            if len(batch_codes) > 1:
                return self._forward_index_batch(action, batch_codes, arguments, timeout=timeout)
            namespace = self._china_index_namespace(arguments)
            if namespace == "sw":
                return self._forward_sw_index_constituents(action, arguments, timeout=timeout)
            if namespace == "csindex":
                return self._forward_csindex_index_constituents(action, arguments, timeout=timeout)
            index_code = normalize_tushare_index_code(arguments.get("index_code") or self._market_symbol(arguments))
            if not index_code:
                return self._structured_error(
                    "tushare_http",
                    action,
                    "index constituents/weights require an index_code such as 000300.SH or 399001.SZ; use provider='sw' for .SI/Shenwan indexes or provider='csindex' for CSI/Zhongzheng indexes.",
                    metadata={"requested_index_code": arguments.get("index_code") or self._market_symbol(arguments)},
                )
            params, start_date, end_date = self._tushare_dated_params(arguments)
            params["index_code"] = index_code
            return self._call_tushare_endpoint(
                action,
                "index_weight",
                params,
                start_date=start_date,
                end_date=end_date,
                sort_by="trade_date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"index_code": index_code},
            )

        if action in {"index_basic", "index_catalog"}:
            namespace = self._china_index_namespace(arguments, catalog=True)
            if namespace == "sw":
                return self._forward_sw_index_catalog(action, arguments, timeout=timeout)
            if namespace == "csindex":
                return self._forward_csindex_index_catalog(action, arguments, timeout=timeout)
            params = {
                "market": arguments.get("market"),
                "publisher": arguments.get("publisher"),
                "category": arguments.get("category"),
                "fields": self._tushare_fields(arguments),
            }
            query = str(arguments.get("query") or arguments.get("index_code") or "").strip()
            return self._call_tushare_endpoint(
                action,
                "index_basic",
                params,
                query=query,
                query_fields=["ts_code", "name", "market", "publisher"],
                limit=self._optional_positive_int(arguments.get("limit")),
                limit_mode="head",
                metadata={"query": query},
            )

        if action in {"fund_history", "fund_nav", "fund_portfolio"}:
            return self._forward_tushare_fund_action(action, arguments, timeout=timeout)

        if action == "fund_basic":
            params = {
                "market": arguments.get("market"),
                "status": arguments.get("status"),
                "fields": self._tushare_fields(arguments),
            }
            query = str(arguments.get("query") or arguments.get("fund_code") or "").strip()
            result = self._call_tushare_endpoint(
                action,
                "fund_basic",
                params,
                query=query,
                query_fields=["ts_code", "name", "fund_type", "market"],
                limit=self._optional_positive_int(arguments.get("limit")),
                limit_mode="head",
                metadata={"query": query},
            )
            if result.status == "success":
                return result
            fallback = self._fund_basic_fallback_from_series(query, arguments, result, timeout=timeout)
            return fallback or result

        if action.startswith("cn_futures_"):
            return self._forward_tushare_futures_action(action, arguments, timeout=timeout)

        if action.startswith("cn_options_"):
            return self._forward_tushare_options_action(action, arguments, timeout=timeout)

        if action == "cn_bond_yield_curve":
            params, start_date, end_date = self._tushare_dated_params(arguments)
            params.update(
                {
                    "curve_type": arguments.get("curve_type"),
                    "curve_name": arguments.get("curve_name"),
                    "ts_code": arguments.get("ts_code"),
                }
            )
            query = str(arguments.get("query") or arguments.get("curve_name") or "").strip()
            return self._call_tushare_endpoint(
                action,
                "yc_cb",
                params,
                start_date=start_date,
                end_date=end_date,
                query=query,
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"query": query},
            )

        if action == "cn_macro_series":
            return self._forward_tushare_macro_action(arguments, timeout=timeout)

        if action == "shibor_series":
            kind = str(arguments.get("kind") or "fixing").strip().lower()
            endpoint = "shibor_quote" if kind in {"quote", "quotes", "market_quote"} else "shibor"
            params, start_date, end_date = self._tushare_dated_params(arguments)
            return self._call_tushare_endpoint(
                action,
                endpoint,
                params,
                start_date=start_date,
                end_date=end_date,
                sort_by="date",
                limit=self._optional_positive_int(arguments.get("limit")),
                metadata={"kind": endpoint},
            )

        return self._structured_error("tushare_http", action, f"Unsupported Tushare typed action: {action}")

    def _index_code_batch_values(self, arguments: Dict[str, Any]) -> List[str]:
        raw = arguments.get("index_code")
        if raw in (None, ""):
            raw = arguments.get("index_codes")
        values: List[str] = []
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw]
        elif isinstance(raw, tuple):
            values = [str(item).strip() for item in raw]
        elif isinstance(raw, str) and re.search(r"[,，;；]", raw):
            values = [part.strip() for part in re.split(r"[,，;；]+", raw)]
        else:
            return []
        output: List[str] = []
        seen = set()
        for value in values:
            if not value or value in seen:
                continue
            output.append(value)
            seen.add(value)
        return output

    def _forward_index_batch(
        self,
        action: str,
        index_codes: List[str],
        arguments: Dict[str, Any],
        *,
        timeout: int = 30,
    ) -> ToolResult:
        records: List[Dict[str, Any]] = []
        coverage: List[Dict[str, Any]] = []
        errors: List[str] = []
        for index_code in index_codes:
            single_arguments = dict(arguments)
            single_arguments["index_code"] = index_code
            single_arguments.pop("index_codes", None)
            result = self._forward_tushare_typed_action(action, single_arguments, timeout=timeout)
            row_count = 0
            if result.status == "success":
                for table in result.tables or []:
                    if not isinstance(table, dict) or not isinstance(table.get("rows"), list):
                        continue
                    for row in table["rows"]:
                        if not isinstance(row, dict):
                            continue
                        item = dict(row)
                        item.setdefault("index_code", index_code)
                        item.setdefault("batch_index_code", index_code)
                        records.append(item)
                        row_count += 1
            else:
                errors.append(f"{index_code}: {result.error}")
            coverage.append(
                {
                    "index_code": index_code,
                    "status": result.status,
                    "row_count": row_count,
                    "provider": result.provider,
                    "error": result.error,
                }
            )

        columns = self._record_columns(
            records,
            [
                "batch_index_code",
                "index_code",
                "date",
                "trade_date",
                "constituent_code",
                "constituent_name",
                "weight",
                "open",
                "high",
                "low",
                "close",
            ],
        )
        success_count = sum(1 for item in coverage if item.get("status") == "success")
        return self._records_result(
            "market_data_batch",
            action,
            records,
            columns=columns,
            metadata={
                "provider": "market_data_batch",
                "batch_action": action,
                "requested_index_codes": index_codes,
                "coverage": coverage,
                "success_count": success_count,
                "error_count": len(index_codes) - success_count,
                "errors": errors[:20],
                "limit_mode": "per_index_code",
            },
            error=f"Batch {action} returned no rows for requested index_code values.",
            confidence=0.82 if records else 0.2,
        )

    def _fetch_tushare_equity_price_history(
        self,
        action: str,
        arguments: Dict[str, Any],
        *,
        mapped: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> ToolResult:
        if not self._tushare_adjust_is_supported(arguments):
            return self._structured_error(
                "tushare_http",
                action,
                "Tushare daily returns unadjusted China A-share OHLCV only; use equity_factor_history for adjustment factors.",
                metadata={"requested_ticker": self._market_symbol(arguments), "adjust": arguments.get("adjust")},
            )
        ts_code = normalize_tushare_ts_code(self._market_symbol(arguments))
        if not ts_code:
            return self._tushare_symbol_error(action, arguments)
        date_source = {**(mapped or {}), **arguments}
        start_date = format_tushare_date(date_source.get("start_date"), is_end=False)
        end_date = format_tushare_date(date_source.get("end_date"), is_end=True)
        if not start_date or not end_date:
            return self._structured_error(
                "tushare_http",
                action,
                "Tushare daily requires start_date and end_date.",
                metadata={"ts_code": ts_code},
            )
        return self._call_tushare_endpoint(
            action,
            "daily",
            {
                "_fire_tushare_backend": self._tushare_backend_from_arguments(arguments),
                "ts_code": ts_code,
                "start_date": start_date,
                "end_date": end_date,
                "fields": self._tushare_fields(arguments),
            },
            start_date=start_date,
            end_date=end_date,
            sort_by="trade_date",
            limit=self._optional_positive_int(arguments.get("limit") or (mapped or {}).get("limit")),
            metadata={
                "ts_code": ts_code,
                "adjust": "none",
                "adjust_basis": "raw_unadjusted",
                "raw_unadjusted_available": True,
                "volume_unit": "hand",
                "amount_unit": "thousand CNY",
                "auto_selected": self._provider_hint(arguments) in {"auto", ""},
                "unit_schema": {
                    "raw_unadjusted_open": "CNY/share",
                    "raw_unadjusted_close": "CNY/share",
                    "open": "CNY/share",
                    "high": "CNY/share",
                    "low": "CNY/share",
                    "close": "CNY/share",
                    "volume": "hand",
                    "amount": "thousand CNY",
                },
            },
            record_transform=self._normalize_tushare_daily_record,
            preferred_columns=[
                "date",
                "trade_date",
                "symbol",
                "ts_code",
                "adjust_basis",
                "raw_unadjusted_available",
                "raw_unadjusted_open",
                "open",
                "high",
                "low",
                "raw_unadjusted_close",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "volume",
                "vol",
                "amount",
            ],
        )

    def _forward_tushare_fund_action(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        fund_code = normalize_tushare_fund_code(arguments.get("fund_code") or self._market_symbol(arguments))
        if not fund_code:
            return self._structured_error(
                "tushare_http",
                action,
                "Fund actions require a China fund/ETF code such as 510300.SH or 159915.SZ.",
                metadata={"requested_code": arguments.get("fund_code") or self._market_symbol(arguments)},
            )
        params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=fund_code)
        if action == "fund_history":
            endpoint = "fund_daily"
            sort_by = "trade_date"
            extra_metadata = {
                "adjust": "none",
                "volume_unit": "hand",
                "amount_unit": "thousand CNY",
            }
        elif action == "fund_nav":
            endpoint = "fund_nav"
            sort_by = "end_date"
            params["market"] = arguments.get("market")
            extra_metadata = {
                "unit_schema": {
                    "unit_nav": "CNY/unit",
                    "accum_nav": "CNY/unit",
                    "adj_nav": "CNY/unit",
                }
            }
        else:
            endpoint = "fund_portfolio"
            sort_by = "ann_date"
            params["ann_date"] = format_tushare_date(arguments.get("ann_date") or arguments.get("date"), is_end=True)
            params["period"] = format_tushare_period(arguments.get("period"))
            extra_metadata = {
                "unit_schema": {
                    "mkv": "CNY",
                    "amount": "shares",
                    "stk_mkv_ratio": "percent",
                    "stk_float_ratio": "percent",
                }
            }
        return self._call_tushare_endpoint(
            action,
            endpoint,
            params,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            limit=self._optional_positive_int(arguments.get("limit")),
            metadata={"fund_code": fund_code, **extra_metadata},
        )

    def _fund_basic_fallback_from_series(
        self,
        query: str,
        arguments: Dict[str, Any],
        primary_result: ToolResult,
        *,
        timeout: int = 30,
    ) -> Optional[ToolResult]:
        fund_code = normalize_tushare_fund_code(arguments.get("fund_code") or query or self._market_symbol(arguments))
        if not fund_code:
            return None

        probe_args = {
            **arguments,
            "fund_code": fund_code,
            "limit": 1,
        }
        source_actions: List[str] = []
        observed_dates: List[str] = []
        for source_action in ("fund_nav", "fund_history"):
            result = self._forward_tushare_fund_action(source_action, probe_args, timeout=timeout)
            if result.status != "success":
                continue
            rows: List[Dict[str, Any]] = []
            for table in result.tables or []:
                if isinstance(table, dict) and isinstance(table.get("rows"), list):
                    rows.extend(row for row in table["rows"] if isinstance(row, dict))
            if not rows:
                continue
            source_actions.append(source_action)
            for row in rows:
                for key in ("end_date", "trade_date", "date"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        observed_dates.append(value)
                        break

        if not source_actions:
            return None

        market = "SH" if fund_code.upper().endswith(".SH") else "SZ" if fund_code.upper().endswith(".SZ") else ""
        record = {
            "ts_code": fund_code,
            "name": "",
            "fund_type": "",
            "market": market,
            "observed_start_date": min(observed_dates) if observed_dates else "",
            "observed_end_date": max(observed_dates) if observed_dates else "",
            "fallback_source": "+".join(source_actions),
        }
        return self._records_result(
            "tushare_http",
            "fund_basic",
            [record],
            metadata={
                "query": query,
                "fund_code": fund_code,
                "fallback_from": "fund_basic",
                "fallback_source_actions": source_actions,
                "primary_error": primary_result.error,
                "data_source": "tushare_api_via_custom_http",
            },
            columns=[
                "ts_code",
                "name",
                "fund_type",
                "market",
                "observed_start_date",
                "observed_end_date",
                "fallback_source",
            ],
            confidence=0.55,
        )

    def _forward_tushare_futures_action(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        if action == "cn_futures_basic":
            params = {
                "exchange": str(arguments.get("exchange") or "").strip().upper(),
                "fut_type": arguments.get("fut_type"),
                "fields": self._tushare_fields(arguments),
            }
            query = str(arguments.get("query") or arguments.get("contract") or "").strip()
            return self._call_tushare_endpoint(
                action,
                "fut_basic",
                params,
                query=query,
                query_fields=["ts_code", "symbol", "name", "exchange"],
                limit=self._optional_positive_int(arguments.get("limit")),
                limit_mode="head",
                metadata={"query": query},
            )

        contract = normalize_tushare_contract(arguments.get("contract") or self._market_symbol(arguments), arguments.get("exchange"))
        if not contract and action not in {"cn_futures_warehouse"}:
            return self._structured_error(
                "tushare_http",
                action,
                "Futures actions require a contract such as M2501.DCE, CU2405.SHF, or IF2406.CFX.",
            )
        params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=contract)
        if action == "cn_futures_mapping":
            endpoint = "fut_mapping"
            sort_by = "trade_date"
        elif action == "cn_futures_daily":
            endpoint = "fut_daily"
            sort_by = "trade_date"
        elif action == "cn_futures_warehouse":
            endpoint = "fut_wsr"
            sort_by = "trade_date"
            params = {
                "trade_date": format_tushare_date(arguments.get("trade_date") or arguments.get("date"), is_end=True),
                "symbol": arguments.get("symbol") or arguments.get("commodity") or arguments.get("query"),
                "fields": self._tushare_fields(arguments),
            }
        else:
            endpoint = "fut_settle"
            sort_by = "trade_date"
        query = str(arguments.get("query") or arguments.get("symbol") or "").strip()
        return self._call_tushare_endpoint(
            action,
            endpoint,
            params,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            query=query,
            limit=self._optional_positive_int(arguments.get("limit")),
            metadata={"contract": contract, "query": query},
        )

    def _forward_tushare_options_action(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        if action == "cn_options_basic":
            params = {
                "exchange": str(arguments.get("exchange") or "").strip().upper(),
                "opt_code": arguments.get("opt_code") or arguments.get("contract"),
                "call_put": arguments.get("call_put"),
                "fields": self._tushare_fields(arguments),
            }
            query = str(arguments.get("query") or arguments.get("contract") or "").strip()
            return self._call_tushare_endpoint(
                action,
                "opt_basic",
                params,
                query=query,
                query_fields=["ts_code", "name", "opt_code", "exchange"],
                limit=self._optional_positive_int(arguments.get("limit")),
                limit_mode="head",
                metadata={"query": query},
            )
        contract = normalize_tushare_contract(arguments.get("contract") or self._market_symbol(arguments), arguments.get("exchange"))
        if not contract:
            return self._structured_error("tushare_http", action, "cn_options_daily requires a contract option ts_code.")
        params, start_date, end_date = self._tushare_dated_params(arguments, ts_code=contract)
        return self._call_tushare_endpoint(
            action,
            "opt_daily",
            params,
            start_date=start_date,
            end_date=end_date,
            sort_by="trade_date",
            limit=self._optional_positive_int(arguments.get("limit")),
            metadata={"contract": contract},
        )

    def _forward_tushare_macro_action(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        raw_series = str(arguments.get("series") or "").strip()
        key = re.sub(r"[^a-z0-9]+", "_", raw_series.lower()).strip("_")
        macro_endpoints = {
            "gdp": "cn_gdp",
            "china_gdp": "cn_gdp",
            "cn_gdp": "cn_gdp",
            "cpi": "cn_cpi",
            "china_cpi": "cn_cpi",
            "cn_cpi": "cn_cpi",
            "ppi": "cn_ppi",
            "china_ppi": "cn_ppi",
            "cn_ppi": "cn_ppi",
            "m": "cn_m",
            "money": "cn_m",
            "money_supply": "cn_m",
            "m2": "cn_m",
            "cn_m": "cn_m",
            "pmi": "cn_pmi",
            "china_pmi": "cn_pmi",
            "cn_pmi": "cn_pmi",
        }
        endpoint = macro_endpoints.get(key)
        if not endpoint:
            return self._structured_error(
                "tushare_http",
                "cn_macro_series",
                "Unsupported cn_macro_series. Use gdp, cpi, ppi, money_supply/m2, or pmi.",
                metadata={"series": raw_series},
            )
        params = {"fields": self._tushare_fields(arguments)}
        if endpoint == "cn_gdp":
            params["q"] = self._tushare_quarter(arguments.get("quarter") or arguments.get("date") or arguments.get("period"))
            sort_by = "quarter"
        else:
            params["m"] = self._tushare_month(arguments.get("month") or arguments.get("date") or arguments.get("period"))
            sort_by = "month"
        return self._call_tushare_endpoint(
            "cn_macro_series",
            endpoint,
            params,
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            sort_by=sort_by,
            limit=self._optional_positive_int(arguments.get("limit")),
            metadata={"series": raw_series, "macro_endpoint": endpoint},
        )

    def _call_tushare_endpoint(
        self,
        action: str,
        endpoint: str,
        params: Dict[str, Any],
        *,
        start_date: Any = None,
        end_date: Any = None,
        sort_by: str = "",
        limit: Optional[int] = None,
        limit_mode: str = "tail",
        query: str = "",
        query_fields: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_transform: Optional[Any] = None,
        preferred_columns: Optional[List[str]] = None,
        confidence: float = 0.9,
    ) -> ToolResult:
        clean_params = dict(params or {})
        backend = str(clean_params.pop("_fire_tushare_backend", "") or "tushare_http")
        provider = TushareProvider(backend=backend)
        provider_name = provider.backend_name
        try:
            frame = provider.call(
                endpoint,
                clean_params,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                limit=limit,
                limit_mode=limit_mode,
                query=query,
                query_fields=query_fields,
            )
        except TushareProviderError as exc:
            return self._structured_error(
                provider_name,
                action,
                str(exc),
                metadata={
                    **(metadata or {}),
                    "endpoint": endpoint,
                    "params": TushareProvider._clean_params(clean_params),
                },
                confidence=0.15,
            )

        records = frame.records
        if record_transform is not None:
            records = [record_transform(row) for row in records]
        columns = self._record_columns(records, preferred_columns or frame.columns)
        return self._records_result(
            provider_name,
            action,
            records,
            metadata={
                **(metadata or {}),
                "provider": provider_name,
                "data_source": "tushare_api_via_custom_http",
                "endpoint": endpoint,
                "params": TushareProvider._clean_params(clean_params),
                **frame.metadata,
            },
            columns=columns,
            error=f"Tushare endpoint {endpoint} returned no rows for the requested query.",
            confidence=confidence,
        )

    def _forward_option_chain_metrics(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        exchange = str(arguments.get("exchange") or "").strip().upper()
        if exchange in {"SHF"}:
            exchange = "SHFE"
        if exchange != "SHFE":
            return self._structured_error(
                "option_chain_metrics",
                "option_chain_metrics",
                "option_chain_metrics currently supports exchange='SHFE' for China commodity option-chain metrics.",
                metadata={"requested_exchange": exchange},
            )
        trade_date = format_tushare_date(arguments.get("trade_date") or arguments.get("date"), is_end=True)
        if not trade_date:
            return self._structured_error(
                "option_chain_metrics",
                "option_chain_metrics",
                "option_chain_metrics requires trade_date in YYYY-MM-DD or YYYYMMDD format.",
            )
        option_type = self._normalize_option_type(arguments.get("option_type") or arguments.get("call_put") or "all")
        metric = self._normalize_option_metric(arguments.get("metric") or "time_value")
        price_basis = self._normalize_option_price_basis(arguments.get("price_basis") or "settlement")
        top_k = self._positive_int(arguments.get("top_k") or arguments.get("limit"), 10)
        varieties = self._shfe_option_varieties(arguments)
        if not varieties:
            return self._structured_error(
                "akshare_shfe",
                "option_chain_metrics",
                "Could not resolve SHFE option variety. Provide variety such as cu, al, au, ag, rb, or use variety='all'.",
                metadata={"requested_variety": arguments.get("variety") or arguments.get("commodity") or arguments.get("symbol")},
            )

        settle_frame = self._akshare_dataframe("futures_settle_shfe", {"date": trade_date}, timeout=timeout)
        if isinstance(settle_frame, ToolResult):
            settle_frame.action = "option_chain_metrics"
            return settle_frame
        settle_records = self._dataframe_records(settle_frame)
        settle_by_contract: Dict[str, float] = {}
        for row in settle_records:
            symbol = str(row.get("symbol") or "").strip().lower()
            settle = self._to_float(row.get("settle_price"))
            if symbol and settle is not None:
                settle_by_contract[symbol] = settle

        records: List[Dict[str, Any]] = []
        fetch_errors: List[str] = []
        contracts_seen = 0
        contracts_parsed = 0
        contracts_with_underlying = 0
        for root, ak_symbol in varieties:
            frame = self._akshare_dataframe(
                "option_hist_shfe",
                {"symbol": ak_symbol, "trade_date": trade_date},
                timeout=timeout,
            )
            if isinstance(frame, ToolResult):
                fetch_errors.append(f"{ak_symbol}: {frame.error}")
                continue
            for raw_row in self._dataframe_records(frame):
                contracts_seen += 1
                contract = str(raw_row.get("合约代码") or raw_row.get("contract") or "").strip()
                parsed = self._parse_commodity_option_contract(contract)
                if not parsed:
                    continue
                contracts_parsed += 1
                parsed_root, underlying_contract, parsed_type, strike = parsed
                if root and parsed_root.lower() != root.lower():
                    # SHFE pages sometimes include only one variety, but keep this guard for catalog drift.
                    continue
                if option_type != "all" and parsed_type != option_type:
                    continue
                underlying_settle = settle_by_contract.get(underlying_contract.lower())
                if underlying_settle is not None:
                    contracts_with_underlying += 1
                option_settle = self._to_float(raw_row.get("结算价"))
                option_close = self._to_float(raw_row.get("收盘价"))
                option_price = option_settle if price_basis == "settlement" else option_close
                if option_price is None:
                    option_price = option_close if price_basis == "settlement" else option_settle
                intrinsic = None
                time_value = None
                moneyness = None
                underlying_to_strike = None
                if underlying_settle is not None and strike is not None:
                    if strike != 0:
                        underlying_to_strike = underlying_settle / strike
                    if parsed_type == "call":
                        intrinsic = max(underlying_settle - strike, 0.0)
                        moneyness = underlying_settle / strike if strike else None
                    elif parsed_type == "put":
                        intrinsic = max(strike - underlying_settle, 0.0)
                        moneyness = strike / underlying_settle if underlying_settle else None
                if option_price is not None and intrinsic is not None:
                    time_value = option_price - intrinsic
                metric_value = self._option_metric_value(
                    metric,
                    option_price=option_price,
                    option_settle=option_settle,
                    option_close=option_close,
                    intrinsic=intrinsic,
                    time_value=time_value,
                    moneyness=moneyness,
                    underlying_to_strike=underlying_to_strike,
                    strike=strike,
                    underlying_settle=underlying_settle,
                )
                record = {
                    "exchange": exchange,
                    "trade_date": trade_date,
                    "variety": root,
                    "option_variety_name": ak_symbol,
                    "contract": contract,
                    "underlying_contract": underlying_contract,
                    "option_type": parsed_type,
                    "strike": strike,
                    "option_settle": option_settle,
                    "option_close": option_close,
                    "option_price_basis": price_basis,
                    "option_price": option_price,
                    "underlying_settle": underlying_settle,
                    "intrinsic_value": intrinsic,
                    "time_value": time_value,
                    "moneyness": moneyness,
                    "underlying_to_strike": underlying_to_strike,
                    "metric": metric,
                    "metric_value": metric_value,
                    "volume": self._to_float(raw_row.get("成交量")),
                    "open_interest": self._to_float(raw_row.get("持仓量")),
                    "delta": self._to_float(raw_row.get("德尔塔")),
                    "source_function": "option_hist_shfe+futures_settle_shfe",
                }
                records.append(record)

        records = [row for row in records if row.get("metric_value") is not None]
        records.sort(key=lambda row: float(row.get("metric_value") or 0.0), reverse=True)
        ranked = []
        for rank, row in enumerate(records[:top_k], start=1):
            ranked.append({"rank": rank, **row})
        coverage = {
            "exchange": exchange,
            "trade_date": trade_date,
            "varieties_requested": [name for _root, name in varieties],
            "contracts_seen": contracts_seen,
            "contracts_parsed": contracts_parsed,
            "metric_rows": len(records),
            "contracts_with_underlying_settle": contracts_with_underlying,
            "settlement_contracts_available": len(settle_by_contract),
            "fetch_errors": fetch_errors[:20],
        }
        return self._records_result(
            "akshare_shfe",
            "option_chain_metrics",
            ranked,
            columns=self._record_columns(
                ranked,
                [
                    "rank",
                    "exchange",
                    "trade_date",
                    "contract",
                    "underlying_contract",
                    "option_type",
                    "strike",
                    "underlying_settle",
                    "option_price",
                    "intrinsic_value",
                    "time_value",
                    "moneyness",
                    "metric",
                    "metric_value",
                ],
            ),
            metadata={
                "provider": "akshare",
                "metric": metric,
                "option_type": option_type,
                "price_basis": price_basis,
                "coverage": coverage,
                "calculation": "call intrinsic=max(S-K,0); put intrinsic=max(K-S,0); time_value=option_price-intrinsic; S uses futures settlement.",
            },
            error="option_chain_metrics produced no ranked rows for the requested option chain.",
            confidence=0.88 if ranked else 0.25,
        )

    def _forward_dataframe_query(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        plan = self._dataframe_plan(arguments)
        if not isinstance(plan, dict) or not plan:
            return self._structured_error(
                "dataframe_query",
                "dataframe_query",
                "dataframe_query requires a JSON object plan with universe, metrics, and optional operations.",
            )
        plan = self._normalize_dataframe_plan(plan)
        max_entities = self._positive_int(arguments.get("max_entities") or plan.get("max_entities"), 120)
        top_k = self._positive_int(arguments.get("top_k") or plan.get("top_k") or arguments.get("limit"), 10)
        trace: List[Dict[str, Any]] = []
        universe_spec = plan.get("universe") or {}
        metrics = self._dataframe_metric_specs(plan)
        operations = self._dataframe_operation_specs(plan, default_top_k=top_k)
        compile_failure = self._dataframe_compile_failure_for_plan(plan, universe_spec, metrics)
        if compile_failure is not None:
            return compile_failure
        rows_result = self._dataframe_universe_rows(universe_spec, timeout=timeout)
        if isinstance(rows_result, ToolResult):
            return rows_result
        rows = rows_result
        original_count = len(rows)
        symbols = self._dataframe_symbols(rows)
        if max_entities and len(symbols) > max_entities:
            symbols = symbols[:max_entities]
            allowed = set(symbols)
            rows = [row for row in rows if self._dataframe_symbol_key(row) in allowed]
        trace.append({"op": "universe", "rows": original_count, "symbols": len(symbols), "spec": universe_spec})

        for metric_spec in metrics:
            if not isinstance(metric_spec, dict):
                continue
            source = self._dataframe_metric_source(metric_spec)
            if source == "equity_daily_basic":
                long_rows = self._dataframe_metric_long_mode(plan, metric_spec, operations)
                if self._dataframe_metric_requires_price_history(metric_spec, operations):
                    fetched, fetch_meta = self._dataframe_fetch_price_panel(
                        rows,
                        metric_spec,
                        metric_source="price_panel",
                        timeout=timeout,
                        long_rows=True,
                    )
                    rows = self._dataframe_expand_join(
                        rows,
                        fetched,
                        left_keys=["symbol_key", "ts_code", "constituent_code", "symbol", "ticker"],
                        right_key="symbol_key",
                    )
                    trace.append({"op": "batch_metric", "source": "equity_price_history_for_daily_fields", **fetch_meta})
                else:
                    fetched, fetch_meta = self._dataframe_fetch_daily_basic(
                        symbols,
                        metric_spec,
                        timeout=timeout,
                        long_rows=long_rows,
                    )
                    if long_rows:
                        rows = self._dataframe_expand_join(
                            rows,
                            fetched,
                            left_keys=["ts_code", "constituent_code", "symbol_key"],
                            right_key="ts_code",
                        )
                    else:
                        rows = self._dataframe_left_join(rows, fetched, left_keys=["ts_code", "constituent_code"], right_key="ts_code")
                    trace.append({"op": "batch_metric", "source": "equity_daily_basic", **fetch_meta})
                continue
            if source in {"price_panel", "return_panel", "volume_panel"}:
                long_rows = self._dataframe_metric_long_mode(plan, metric_spec, operations)
                fetched, fetch_meta = self._dataframe_fetch_price_panel(
                    rows,
                    metric_spec,
                    metric_source=source,
                    timeout=timeout,
                    long_rows=long_rows,
                )
                if long_rows:
                    rows = self._dataframe_expand_join(
                        rows,
                        fetched,
                        left_keys=["symbol_key", "ts_code", "constituent_code", "symbol", "ticker"],
                        right_key="symbol_key",
                    )
                else:
                    rows = self._dataframe_left_join(
                        rows,
                        fetched,
                        left_keys=["symbol_key", "ts_code", "constituent_code", "symbol", "ticker"],
                        right_key="symbol_key",
                    )
                trace.append({"op": "batch_metric", "source": source, **fetch_meta})
                continue
            failure = self._dataframe_compile_failure(
                failure_class="unsupported_metric",
                message=f"Unsupported dataframe_query metric source {source!r}.",
                plan=plan,
                unsupported_value=source,
                trace=trace,
            )
            return failure

        rows, op_trace = self._apply_dataframe_operations(rows, operations, default_top_k=top_k)
        trace.extend(op_trace)
        has_long_panel = any(isinstance(item, dict) and item.get("shape") == "long" for item in trace)
        top_k_ops = {"rank", "max_by", "min_by", "list"}
        has_top_k_op = any(isinstance(item, dict) and str(item.get("op") or "") in top_k_ops for item in operations)
        default_limit = 200 if has_long_panel and not has_top_k_op else top_k
        rows = self._limit_records(rows, {"limit": arguments.get("limit") or plan.get("limit") or default_limit}, limit_mode="head")
        coverage = {
            "universe_rows": original_count,
            "entities_requested": len(self._dataframe_symbols(rows_result)),
            "entities_processed": len(symbols),
            "result_rows": len(rows),
            "max_entities": max_entities,
        }
        execution_card = self._dataframe_execution_card(plan, rows, trace, coverage)
        result = self._records_result(
            "dataframe_query",
            "dataframe_query",
            rows,
            columns=self._record_columns(rows, None),
            metadata={
                "provider": "market_data_dataframe",
                "coverage": coverage,
                "trace": trace,
                "execution_card": execution_card,
                "supported_operations": ["filter", "sort", "rank", "aggregate", "count", "list", "max_by", "min_by", "pct_change"],
            },
            error="dataframe_query produced no rows for the requested plan.",
            confidence=0.86 if rows else 0.25,
        )
        card_text = json.dumps(execution_card, ensure_ascii=False, default=str, separators=(",", ":"))
        if result.status == "success":
            result.observation = f"EXECUTION_CARD {card_text}\n{result.observation_text()}".strip()
        elif self._dataframe_no_matching_rows(trace):
            result.status = "success"
            result.error = None
            result.confidence = 0.72
            result.observation = f"EXECUTION_CARD {card_text}\nNO_MATCHING_ROWS_AFTER_OPERATIONS"
        return result

    @staticmethod
    def _normalize_option_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        mapping = {
            "": "all",
            "all": "all",
            "both": "all",
            "c": "call",
            "call": "call",
            "calls": "call",
            "看涨": "call",
            "认购": "call",
            "p": "put",
            "put": "put",
            "puts": "put",
            "看跌": "put",
            "认沽": "put",
        }
        return mapping.get(text, text if text in {"all", "call", "put"} else "all")

    @staticmethod
    def _normalize_option_metric(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "": "time_value",
            "time": "time_value",
            "timevalue": "time_value",
            "time_value": "time_value",
            "时间价值": "time_value",
            "intrinsic": "intrinsic_value",
            "intrinsic_value": "intrinsic_value",
            "内在价值": "intrinsic_value",
            "moneyness": "moneyness",
            "实值程度": "moneyness",
            "premium": "option_price",
            "option_price": "option_price",
            "settlement": "option_settle",
            "settle": "option_settle",
            "close": "option_close",
            "strike": "strike",
            "underlying_settle": "underlying_settle",
        }
        return mapping.get(text, text if text in {"time_value", "intrinsic_value", "moneyness", "option_price", "option_settle", "option_close", "strike", "underlying_settle"} else "time_value")

    @staticmethod
    def _normalize_option_price_basis(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text in {"close", "closing", "收盘", "收盘价"}:
            return "close"
        return "settlement"

    def _shfe_option_varieties(self, arguments: Dict[str, Any]) -> List[Tuple[str, str]]:
        varieties = {
            "cu": "铜期权",
            "al": "铝期权",
            "zn": "锌期权",
            "pb": "铅期权",
            "ni": "镍期权",
            "sn": "锡期权",
            "au": "黄金期权",
            "ag": "白银期权",
            "rb": "螺纹钢期权",
            "ru": "天然橡胶期权",
            "fu": "燃料油期权",
            "bu": "石油沥青期权",
            "sp": "纸浆期权",
            "ao": "氧化铝期权",
            "br": "合成橡胶期权",
        }
        raw = str(arguments.get("variety") or arguments.get("commodity") or arguments.get("symbol") or "").strip()
        if not raw or raw.lower() in {"all", "*", "全部", "全市场"}:
            return list(varieties.items())
        normalized = raw.lower().replace("期权", "").replace(" ", "")
        if normalized in varieties:
            return [(normalized, varieties[normalized])]
        for root, name in varieties.items():
            if raw in name or normalized in name.replace("期权", "").lower():
                return [(root, name)]
        return []

    @staticmethod
    def _parse_commodity_option_contract(contract: str) -> Optional[Tuple[str, str, str, float]]:
        text = str(contract or "").strip()
        match = re.fullmatch(r"([A-Za-z]+)(\d{4})([CPcp])([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return None
        root = match.group(1).lower()
        underlying = f"{root}{match.group(2)}"
        option_type = "call" if match.group(3).upper() == "C" else "put"
        try:
            strike = float(match.group(4))
        except ValueError:
            return None
        return root, underlying, option_type, strike

    @staticmethod
    def _option_metric_value(metric: str, **values: Any) -> Optional[float]:
        value = values.get(metric)
        if value is None and metric == "option_price":
            value = values.get("option_price")
        if value is None and metric == "option_settle":
            value = values.get("option_settle")
        if value is None and metric == "option_close":
            value = values.get("option_close")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dataframe_plan(arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("plan")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            parsed = parse_json_object(raw)
            return parsed if isinstance(parsed, dict) else {}
        plan = {
            key: arguments.get(key)
            for key in (
                "universe",
                "metrics",
                "metric",
                "operations",
                "rank_by",
                "order_by",
                "order",
                "top_k",
                "limit",
                "max_entities",
                "start_date",
                "end_date",
                "date",
                "trade_date",
                "fields",
                "output",
                "output_fields",
                "shape",
                "mode",
            )
            if arguments.get(key) not in (None, "")
        }
        return plan

    def _normalize_dataframe_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(plan)
        universe = normalized.get("universe")
        top_level_symbols = normalized.get("symbols") or normalized.get("tickers") or normalized.get("symbol_list")
        if isinstance(universe, str):
            source = re.sub(r"[\s-]+", "_", universe.strip().lower())
            if source in {"symbols", "symbol_list", "tickers"} and top_level_symbols not in (None, "", [], {}):
                normalized["universe"] = {"type": "symbols", "symbols": top_level_symbols}
            else:
                universe_args = {
                    key: normalized.get(key)
                    for key in (
                        "index_code",
                        "ts_code",
                        "trade_date",
                        "date",
                        "start_date",
                        "end_date",
                        "market",
                        "limit",
                    )
                    if normalized.get(key) not in (None, "", [], {})
                }
                normalized["universe"] = {"type": source, **universe_args}
            universe = normalized.get("universe")
        if universe in (None, "", [], {}) and top_level_symbols not in (None, "", [], {}):
            normalized["universe"] = {"type": "symbols", "symbols": top_level_symbols}
            universe = normalized.get("universe")
        if isinstance(universe, dict) and not any(universe.get(key) for key in ("type", "source", "action")):
            if (
                universe.get("symbols") not in (None, "", [], {})
                or universe.get("tickers") not in (None, "", [], {})
                or universe.get("symbol_list") not in (None, "", [], {})
            ):
                normalized["universe"] = {**universe, "type": "symbols"}
        return normalized

    def _dataframe_metric_specs(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        metrics = plan.get("metrics") or []
        if not metrics and plan.get("metric") not in (None, ""):
            metrics = [{"metric": plan.get("metric")}]
        if not metrics:
            expanded = []
            for key in self._dataframe_supported_metrics():
                value = plan.get(key)
                if value in (None, "", [], {}):
                    continue
                spec = dict(value) if isinstance(value, dict) else {}
                spec.setdefault("source", key)
                expanded.append(spec)
            metrics = expanded
        if isinstance(metrics, dict):
            source_keys = set(self._dataframe_supported_metrics())
            normalized_keys = {
                self._normalize_dataframe_source_key(key)
                for key in metrics.keys()
            }
            if normalized_keys & source_keys and not any(key in metrics for key in ("source", "action", "type", "metric")):
                expanded = []
                for key, value in metrics.items():
                    source = self._normalize_dataframe_source_key(key)
                    if source not in source_keys:
                        continue
                    spec = dict(value) if isinstance(value, dict) else {}
                    spec.setdefault("source", source)
                    expanded.append(spec)
                metrics = expanded
            else:
                metrics = [metrics]
        if not isinstance(metrics, list):
            return []
        specs = []
        source_keys = set(self._dataframe_supported_metrics())
        for metric in metrics:
            if isinstance(metric, dict):
                if not any(metric.get(key) not in (None, "", [], {}) for key in ("source", "action", "type", "metric")):
                    expanded = False
                    for key, value in metric.items():
                        source = self._normalize_dataframe_source_key(key)
                        if source not in source_keys:
                            continue
                        spec = dict(value) if isinstance(value, dict) else {}
                        spec.setdefault("source", source)
                        specs.append(spec)
                        expanded = True
                    if expanded:
                        continue
                specs.append(dict(metric))
            elif isinstance(metric, str) and metric.strip():
                specs.append({"metric": metric.strip()})
        inherited_keys = (
            "start_date",
            "end_date",
            "date",
            "trade_date",
            "fields",
            "market",
            "provider",
            "source_provider",
            "adjust",
            "shape",
            "mode",
            "output_fields",
        )
        output = plan.get("output") if isinstance(plan.get("output"), dict) else {}
        output_fields = plan.get("output_fields") or output.get("columns") or output.get("fields")
        for spec in specs:
            for key in inherited_keys:
                if key == "output_fields":
                    value = output_fields
                else:
                    value = plan.get(key)
                if spec.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    spec[key] = value
            if spec.get("shape") in (None, "") and output.get("shape") not in (None, ""):
                spec["shape"] = output.get("shape")
            if spec.get("mode") in (None, "") and output.get("mode") not in (None, ""):
                spec["mode"] = output.get("mode")
        return specs

    @staticmethod
    def _dataframe_universe_source(universe_spec: Any) -> str:
        if not isinstance(universe_spec, dict):
            return ""
        if (
            universe_spec.get("symbols") not in (None, "", [], {})
            or universe_spec.get("tickers") not in (None, "", [], {})
            or universe_spec.get("symbol_list") not in (None, "", [], {})
        ):
            if not any(universe_spec.get(key) for key in ("type", "source", "action")):
                return "symbols"
        source = str(universe_spec.get("type") or universe_spec.get("source") or universe_spec.get("action") or "").strip().lower()
        return re.sub(r"[\s-]+", "_", source)

    @staticmethod
    def _normalize_dataframe_source_key(value: Any) -> str:
        source = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
        aliases = {
            "daily_basic": "equity_daily_basic",
            "equity_daily_basic": "equity_daily_basic",
            "price": "price_panel",
            "prices": "price_panel",
            "price_history": "price_panel",
            "price_history_panel": "price_panel",
            "ohlcv": "price_panel",
            "ohlcv_panel": "price_panel",
            "price_panel": "price_panel",
            "return": "return_panel",
            "returns": "return_panel",
            "pct_return": "return_panel",
            "price_return": "return_panel",
            "return_panel": "return_panel",
            "volume": "volume_panel",
            "volumes": "volume_panel",
            "volume_panel": "volume_panel",
        }
        return aliases.get(source, source)

    @staticmethod
    def _dataframe_metric_source(metric_spec: Dict[str, Any]) -> str:
        raw = metric_spec.get("source") or metric_spec.get("action") or metric_spec.get("type") or metric_spec.get("metric") or metric_spec.get("name") or ""
        return MarketDataTool._normalize_dataframe_source_key(raw)

    @staticmethod
    def _dataframe_supported_universes() -> List[str]:
        return ["symbols", "symbol_list", "tickers", "index_constituents", "index_weight", "index"]

    @staticmethod
    def _dataframe_supported_metrics() -> List[str]:
        return ["equity_daily_basic", "daily_basic", "price_panel", "return_panel", "volume_panel"]

    def _dataframe_compile_failure_for_plan(
        self,
        plan: Dict[str, Any],
        universe_spec: Any,
        metrics: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        universe_source = self._dataframe_universe_source(universe_spec)
        if universe_source not in {"symbols", "symbol_list", "tickers", "index_constituents", "index_weight", "index"}:
            return self._dataframe_compile_failure(
                failure_class="unsupported_universe",
                message=f"Unsupported dataframe_query universe source {universe_source!r}.",
                plan=plan,
                unsupported_value=universe_source,
            )
        for metric_spec in metrics:
            source = self._dataframe_metric_source(metric_spec)
            if source not in {"equity_daily_basic", "price_panel", "return_panel", "volume_panel"}:
                return self._dataframe_compile_failure(
                    failure_class="unsupported_metric",
                    message=f"Unsupported dataframe_query metric source {source!r}.",
                    plan=plan,
                    unsupported_value=source,
                )
            if source in {"price_panel", "return_panel", "volume_panel"} and universe_source not in {"symbols", "symbol_list", "tickers"}:
                return self._dataframe_compile_failure(
                    failure_class="unsupported_panel_universe",
                    message=f"dataframe_query {source} currently requires an explicit symbol universe.",
                    plan=plan,
                    unsupported_value=universe_source,
                )
            if source in {"price_panel", "return_panel", "volume_panel"} and not any(
                metric_spec.get(key) not in (None, "")
                for key in ("date", "trade_date", "start_date", "end_date", "start", "end", "from", "to")
            ):
                return self._dataframe_compile_failure(
                    failure_class="missing_panel_dates",
                    message=f"dataframe_query {source} requires date or start_date/end_date.",
                    plan=plan,
                    unsupported_value=source,
                )
        return None

    def _dataframe_compile_failure(
        self,
        *,
        failure_class: str,
        message: str,
        plan: Dict[str, Any],
        unsupported_value: str = "",
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolResult:
        if not hasattr(self, "_dataframe_failure_counts"):
            self._dataframe_failure_counts = {}
        fingerprint_payload = {
            "failure_class": failure_class,
            "unsupported_value": unsupported_value,
            "universe": plan.get("universe"),
            "metrics": plan.get("metrics"),
        }
        fingerprint = compact_text(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False, default=str), 600)
        count = int(self._dataframe_failure_counts.get(fingerprint, 0)) + 1
        self._dataframe_failure_counts[fingerprint] = count
        suggested_recovery = {
            "unsupported_universe": "Use an explicit symbol list or a supported China index constituent/weight universe; otherwise choose a resolver/catalog or an atomic structured tool.",
            "unsupported_metric": "Use one of equity_daily_basic, price_panel, return_panel, or volume_panel; otherwise choose a typed market_data action or a source/catalog step.",
            "unsupported_panel_universe": "Use a bounded explicit symbol list for price_panel, return_panel, or volume_panel; use index constituents with equity_daily_basic only, or an atomic structured lookup for other needs.",
            "missing_panel_dates": "Add date or start_date/end_date to the panel metric, or use an atomic resolver/catalog step if the date is not known yet.",
        }.get(failure_class, "Revise the dataframe plan to a supported structured universe and metric.")
        failure_card = {
            "failure_class": failure_class,
            "failed_source": "dataframe_query",
            "unsupported_value": unsupported_value,
            "retry_same_plan_allowed": False,
            "suggested_recovery": suggested_recovery,
            "supported_universes": self._dataframe_supported_universes(),
            "supported_metrics": self._dataframe_supported_metrics(),
        }
        return self._structured_error(
            "dataframe_query",
            "dataframe_query",
            (
                f"{message} retry_same_plan_allowed=false; "
                f"supported_universes={', '.join(self._dataframe_supported_universes())}; "
                f"supported_metrics={', '.join(self._dataframe_supported_metrics())}."
            ),
            metadata={
                "failure_card": failure_card,
                "failure_class": failure_class,
                "failure_fingerprint": fingerprint,
                "failure_repeat_count": count,
                "blocked_by_dataframe_fuse": count > 1,
                "trace": trace or [],
            },
            confidence=0.05,
        )

    def _dataframe_operation_specs(self, plan: Dict[str, Any], *, default_top_k: int) -> List[Dict[str, Any]]:
        operations = plan.get("operations") or []
        if isinstance(operations, dict):
            operations = [operations]
        if not isinstance(operations, list):
            operations = []
        normalized: List[Dict[str, Any]] = []
        for operation in operations:
            normalized.extend(self._normalize_dataframe_operation(operation, default_top_k=default_top_k))
        output = plan.get("output") if isinstance(plan.get("output"), dict) else {}
        output_sort = output.get("sort") or output.get("order_by")
        if output_sort not in (None, "", [], {}):
            if isinstance(output_sort, dict):
                field, order = next(iter(output_sort.items()))
                field, order = self._normalize_dataframe_sort_spec(field, order or output.get("order") or "asc")
                normalized.append({"op": "sort", "field": field, "order": order})
            else:
                field, order = self._normalize_dataframe_sort_spec(output_sort, output.get("order") or "asc")
                normalized.append({"op": "sort", "field": field, "order": order})
        if not normalized and (plan.get("rank_by") or plan.get("order_by")):
            normalized.append(
                {
                    "op": "rank",
                    "by": plan.get("rank_by") or plan.get("order_by"),
                    "order": plan.get("order", "desc"),
                    "top_k": default_top_k,
                }
            )
        return normalized

    def _normalize_dataframe_operation(self, operation: Any, *, default_top_k: int) -> List[Dict[str, Any]]:
        if not isinstance(operation, dict):
            return []
        op = str(operation.get("op") or operation.get("type") or operation.get("operation") or operation.get("action") or "").strip().lower()
        if op:
            normalized = dict(operation)
            normalized["op"] = op
            if op == "filter" and not normalized.get("field"):
                if normalized.get("column") not in (None, ""):
                    normalized["field"] = normalized.get("column")
                elif normalized.get("by") not in (None, ""):
                    normalized["field"] = normalized.get("by")
                parsed = self._parse_dataframe_filter_condition(normalized.get("condition"))
                if parsed:
                    normalized.update(parsed)
                if normalized.get("operator") in (None, ""):
                    normalized["operator"] = normalized.get("comparison") or normalized.get("comparator") or normalized.get("condition") or "eq"
                if normalized.get("value") in (None, ""):
                    for key in ("threshold", "target", "equals"):
                        if normalized.get(key) not in (None, ""):
                            normalized["value"] = normalized.get(key)
                            break
            if op == "sort" and not normalized.get("field"):
                normalized["field"] = normalized.get("by") or normalized.get("column")
                if normalized.get("direction") not in (None, "") and normalized.get("order") in (None, ""):
                    normalized["order"] = normalized.get("direction")
            if op == "sort":
                field, order = self._normalize_dataframe_sort_spec(normalized.get("field"), normalized.get("order") or normalized.get("direction") or "")
                normalized["field"] = field
                if order:
                    normalized["order"] = order
            return [normalized]

        if isinstance(operation.get("compute"), dict):
            compute_spec = operation.get("compute") or {}
            output: List[Dict[str, Any]] = []
            for field, expression in compute_spec.items():
                text = str(expression or "").strip()
                if "pct_change" in text.lower():
                    match = re.search(r"pct_change\(([^)]+)\)", text, flags=re.I)
                    column = match.group(1).strip() if match else "close"
                    output.append({"op": "pct_change", "column": column, "output_field": field, "as_ratio": True})
                else:
                    output.append({"op": "add_column", "column": field, "expression": text})
            return output

        if isinstance(operation.get("filter"), dict):
            filter_spec = operation["filter"]
            if (
                filter_spec.get("field") not in (None, "")
                or filter_spec.get("column") not in (None, "")
                or filter_spec.get("by") not in (None, "")
            ):
                return [
                    {
                        "op": "filter",
                        "field": filter_spec.get("field") or filter_spec.get("column") or filter_spec.get("by"),
                        "operator": filter_spec.get("operator") or filter_spec.get("comparison") or filter_spec.get("comparator") or filter_spec.get("condition") or "eq",
                        "value": filter_spec.get("value") if filter_spec.get("value") not in (None, "") else filter_spec.get("threshold"),
                    }
                ]
            filters = []
            for field, condition in filter_spec.items():
                if isinstance(condition, dict) and condition:
                    for operator, value in condition.items():
                        filters.append({"op": "filter", "field": field, "operator": operator, "value": value})
                else:
                    filters.append({"op": "filter", "field": field, "operator": "eq", "value": condition})
            return filters

        sort_spec = operation.get("sort") or operation.get("order_by")
        if sort_spec not in (None, "", [], {}):
            if isinstance(sort_spec, list):
                output = []
                for item in sort_spec:
                    if isinstance(item, dict):
                        output.append(
                            self._dataframe_sort_operation(
                                item.get("field") or item.get("by") or item.get("column"),
                                item.get("order") or item.get("direction") or ("asc" if item.get("ascending") is not False else "desc"),
                            )
                        )
                return output
            if isinstance(sort_spec, dict):
                field, order = next(iter(sort_spec.items()))
                if field in {"field", "by"}:
                    return [self._dataframe_sort_operation(order, sort_spec.get("order") or sort_spec.get("direction") or "asc")]
                return [self._dataframe_sort_operation(field, order or operation.get("order") or "asc")]
            return [self._dataframe_sort_operation(sort_spec, operation.get("order") or "asc")]

        for key in ("max_by", "min_by"):
            if operation.get(key) not in (None, ""):
                value = operation.get(key)
                if isinstance(value, dict):
                    field = value.get("field") or value.get("by")
                    top_k = self._positive_int(value.get("top_k") or operation.get("top_k"), default_top_k)
                else:
                    field = value
                    top_k = self._positive_int(operation.get("top_k"), default_top_k)
                return [{"op": key, "field": field, "top_k": top_k}]

        if operation.get("count") is not None:
            return [{"op": "count", "field": operation.get("field") or operation.get("count")}]
        if operation.get("list") is not None:
            value = operation.get("list")
            fields = value.get("fields") if isinstance(value, dict) else value
            return [{"op": "list", "fields": fields, "limit": operation.get("limit") or operation.get("top_k") or default_top_k}]
        if operation.get("aggregate") is not None:
            value = operation.get("aggregate")
            if isinstance(value, dict):
                return [{"op": "aggregate", **value}]
            return [{"op": "aggregate", "function": value, "field": operation.get("field")}]
        return []

    @staticmethod
    def _parse_dataframe_filter_condition(condition: Any) -> Dict[str, Any]:
        text = str(condition or "").strip()
        if not text:
            return {}
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.\-]*)\s*(>=|<=|==|=|!=|>|<)\s*(-?\d+(?:\.\d+)?|'.*?'|\".*?\"|[A-Za-z0-9_.%-]+)", text)
        if not match:
            return {}
        op_map = {
            ">": "gt",
            ">=": "ge",
            "<": "lt",
            "<=": "le",
            "=": "eq",
            "==": "eq",
            "!=": "ne",
        }
        value: Any = match.group(3).strip()
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        return {"field": match.group(1), "operator": op_map.get(match.group(2), match.group(2)), "value": value}

    @staticmethod
    def _normalize_dataframe_sort_spec(field: Any, order: Any = "") -> Tuple[str, str]:
        text = str(field or "").strip()
        order_text = str(order or "").strip().lower()
        lower = text.lower()
        if lower.endswith("_asc") and order_text in {"", "asc"}:
            return text[:-4], "asc"
        if lower.endswith("_desc") and order_text in {"", "asc", "desc"}:
            return text[:-5], "desc"
        if order_text not in {"asc", "desc"}:
            order_text = "asc"
        return text, order_text

    @classmethod
    def _dataframe_sort_operation(cls, field: Any, order: Any = "asc") -> Dict[str, Any]:
        field_text, order_text = cls._normalize_dataframe_sort_spec(field, order)
        return {"op": "sort", "field": field_text, "order": order_text}

    def _dataframe_metric_long_mode(
        self,
        plan: Dict[str, Any],
        metric_spec: Dict[str, Any],
        operations: List[Dict[str, Any]],
    ) -> bool:
        shape = str(
            metric_spec.get("shape")
            or metric_spec.get("mode")
            or plan.get("shape")
            or plan.get("mode")
            or ""
        ).strip().lower()
        if shape in {"long", "rows", "row", "daily", "daily_rows", "panel_rows", "long_panel"}:
            return True
        output = plan.get("output") if isinstance(plan.get("output"), dict) else {}
        output_fields = metric_spec.get("output_fields") or plan.get("output_fields") or output.get("columns") or output.get("fields")
        output_names = self._dataframe_field_names(output_fields)
        row_level_fields = {"date", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "change", "volume", "vol", "amount"}
        if output_names & row_level_fields and self._metric_has_date_range(metric_spec):
            return True
        for operation in operations:
            op = str(operation.get("op") or "").lower()
            field = str(operation.get("field") or operation.get("by") or "").strip()
            if op in {"filter", "sort", "list", "count", "max_by", "min_by"} and field in row_level_fields:
                return True
        return False

    def _dataframe_metric_requires_price_history(self, metric_spec: Dict[str, Any], operations: List[Dict[str, Any]]) -> bool:
        names = self._dataframe_field_names(metric_spec.get("fields"))
        names |= self._dataframe_field_names(metric_spec.get("output_fields"))
        for operation in operations:
            names.add(str(operation.get("field") or operation.get("by") or "").strip())
        price_fields = {"open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "vol", "amount"}
        return bool(names & price_fields)

    @staticmethod
    def _dataframe_field_names(value: Any) -> set:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value]
        elif isinstance(value, tuple):
            items = [str(item).strip() for item in value]
        else:
            items = []
        return {item for item in items if item}

    @staticmethod
    def _metric_has_date_range(metric_spec: Dict[str, Any]) -> bool:
        return any(metric_spec.get(key) not in (None, "") for key in ("start_date", "end_date", "start", "end", "from", "to"))

    def _dataframe_universe_rows(self, universe_spec: Any, *, timeout: int = 30) -> Any:
        if not isinstance(universe_spec, dict):
            return self._structured_error(
                "dataframe_query",
                "dataframe_query",
                "dataframe_query universe must be an object such as {'type':'index_constituents','index_code':'000300'}.",
                metadata={"universe": universe_spec},
            )
        source = str(universe_spec.get("type") or universe_spec.get("source") or universe_spec.get("action") or "").strip().lower()
        if source in {"symbols", "symbol_list", "tickers"}:
            symbols = universe_spec.get("symbols") or universe_spec.get("tickers") or universe_spec.get("symbol_list") or []
            if isinstance(symbols, str):
                symbols = [item.strip() for item in symbols.split(",") if item.strip()]
            market_hint = str(universe_spec.get("market") or universe_spec.get("asset_class") or "").strip().lower()
            rows = []
            for symbol in symbols if isinstance(symbols, list) else []:
                normalized = self._normalize_dataframe_symbol(symbol, market_hint=market_hint)
                if normalized.get("symbol_key"):
                    rows.append(normalized)
            return rows
        if source in {"index_constituents", "index_weight", "index"}:
            args = {key: value for key, value in universe_spec.items() if key not in {"type", "source", "action"}}
            args.setdefault("limit", universe_spec.get("universe_limit") or universe_spec.get("limit"))
            result = self.forward("index_weight" if source == "index_weight" else "index_constituents", args, timeout=timeout)
            if result.status != "success":
                return self._structured_error(
                    result.provider or "dataframe_query",
                    "dataframe_query",
                    result.error or "index universe construction failed.",
                    metadata={"universe": universe_spec, "source_result": result.metadata},
                )
            table = result.tables[0] if result.tables else {}
            rows = table.get("rows") if isinstance(table, dict) else []
            normalized_rows = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                code = normalize_tushare_ts_code(row.get("constituent_code") or row.get("证券代码") or row.get("成分券代码") or row.get("ts_code"))
                normalized = dict(row)
                if code:
                    normalized["ts_code"] = code
                    normalized.setdefault("constituent_code", code)
                    normalized.setdefault("symbol", code)
                    normalized.setdefault("ticker", code)
                    normalized["symbol_key"] = code
                    normalized["provider_symbol"] = code
                    normalized["market"] = "cn"
                normalized_rows.append(normalized)
            return normalized_rows
        return self._structured_error(
            "dataframe_query",
            "dataframe_query",
            f"Unsupported dataframe_query universe source {source!r}. Supported universe sources include index_constituents, index_weight, and symbols.",
            metadata={"universe": universe_spec},
        )

    def _normalize_dataframe_symbol(self, symbol: Any, *, market_hint: str = "") -> Dict[str, Any]:
        raw = str(symbol or "").strip()
        if not raw:
            return {}
        hint = market_hint.lower().replace("-", "_")
        upper = raw.upper().strip()
        cn_code = normalize_tushare_ts_code(upper)
        if cn_code and hint not in {"us", "usa", "hk", "hkg", "hongkong"}:
            market = "etf_cn" if hint in {"fund", "etf_cn", "cn_etf", "china_etf"} else "cn"
            return {
                "symbol": raw,
                "ticker": cn_code,
                "ts_code": cn_code,
                "constituent_code": cn_code,
                "symbol_key": cn_code,
                "provider_symbol": cn_code,
                "market": market,
            }

        hk_match = re.fullmatch(r"(?:HK)?0*(\d{1,5})(?:\.HK)?", upper)
        if hk_match and (hint in {"hk", "hkg", "hongkong", "hk_stock"} or upper.endswith(".HK") or upper.startswith("HK")):
            number = int(hk_match.group(1))
            yahoo_symbol = f"{number:04d}.HK"
            hk_code = f"{number:05d}"
            return {
                "symbol": raw,
                "ticker": yahoo_symbol,
                "symbol_key": yahoo_symbol,
                "provider_symbol": yahoo_symbol,
                "hk_code": hk_code,
                "market": "hk",
                "provider": "yahoo",
            }

        if re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", upper):
            market = "us" if hint in {"", "us", "usa", "equity_us", "stock_us", "etf", "us_etf"} else hint
            return {
                "symbol": raw,
                "ticker": upper,
                "symbol_key": upper,
                "provider_symbol": upper,
                "market": market or "us",
                "provider": "yahoo" if market in {"us", "usa", "global", "yahoo", "etf", "us_etf"} else "",
            }

        return {
            "symbol": raw,
            "ticker": upper or raw,
            "symbol_key": upper or raw,
            "provider_symbol": upper or raw,
            "market": hint or "auto",
        }

    @staticmethod
    def _dataframe_symbol_key(row: Dict[str, Any]) -> str:
        return str(
            row.get("symbol_key")
            or row.get("ts_code")
            or row.get("constituent_code")
            or row.get("ticker")
            or row.get("symbol")
            or ""
        ).strip()

    @staticmethod
    def _dataframe_symbols(rows: List[Dict[str, Any]]) -> List[str]:
        symbols: List[str] = []
        seen: set = set()
        for row in rows:
            symbol = MarketDataTool._dataframe_symbol_key(row)
            if not symbol:
                continue
            normalized = normalize_tushare_ts_code(symbol) or symbol
            if normalized and normalized not in seen:
                seen.add(normalized)
                symbols.append(normalized)
        return symbols

    def _dataframe_fetch_daily_basic(
        self,
        symbols: List[str],
        metric_spec: Dict[str, Any],
        *,
        timeout: int = 30,
        long_rows: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        fields = metric_spec.get("fields")
        if isinstance(fields, list):
            field_set = {str(field).strip() for field in fields if str(field).strip()}
        else:
            field_set = {field.strip() for field in str(fields or "").split(",") if field.strip()}
        field_set.update({"ts_code", "trade_date"})
        trade_date = format_tushare_date(metric_spec.get("trade_date") or metric_spec.get("date"), is_end=True)
        start_date = format_tushare_date(metric_spec.get("start_date"), is_end=False)
        end_date = format_tushare_date(metric_spec.get("end_date"), is_end=True)
        base_params = {
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
            "fields": ",".join(sorted(field_set)) if field_set else None,
        }
        provider = TushareProvider()
        symbol_set = set(symbols)
        all_market_rows = 0
        batch_error = ""
        try:
            if trade_date:
                frame = provider.call("daily_basic", base_params, sort_by="trade_date")
                all_market_rows = len(frame.records)
                records = [
                    row for row in frame.records
                    if str(row.get("ts_code") or "").strip() in symbol_set
                ]
                mode = "trade_date_all_market_filter"
            else:
                records = []
                mode = "trade_date_missing"
        except TushareProviderError as exc:
            records = []
            mode = "trade_date_all_market_failed"
            batch_error = f"{type(exc).__name__}: {exc}"
        if not records:
            batch_params = {**base_params, "ts_code": ",".join(symbols)}
            try:
                frame = provider.call("daily_basic", batch_params, sort_by="trade_date")
                records = frame.records
                mode = "ts_code_batch"
            except TushareProviderError as exc:
                records = []
                mode = "ts_code_batch_failed"
                batch_error = f"{batch_error}; {type(exc).__name__}: {exc}" if batch_error else f"{type(exc).__name__}: {exc}"
        if not records and len(symbols) > 1:
            fallback_records: List[Dict[str, Any]] = []
            fallback_errors: List[str] = []
            for symbol in symbols:
                single_params = dict(base_params)
                single_params["ts_code"] = symbol
                try:
                    frame = provider.call("daily_basic", single_params, sort_by="trade_date")
                    fallback_records.extend(frame.records if long_rows else frame.records[-1:])
                except TushareProviderError as exc:
                    fallback_errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            records = fallback_records
            mode = "per_symbol_fallback"
        else:
            fallback_errors = []
        normalized_records: List[Dict[str, Any]] = []
        for row in records:
            ts_code = str(row.get("ts_code") or "").strip()
            if ts_code:
                normalized = dict(row)
                normalized["symbol_key"] = ts_code
                normalized_records.append(normalized)
        if long_rows:
            normalized_records.sort(key=lambda row: (str(row.get("ts_code") or ""), str(row.get("trade_date") or "")))
            output_records = normalized_records
            returned_count = len({str(row.get("ts_code") or "") for row in normalized_records if row.get("ts_code")})
        else:
            latest_by_symbol: Dict[str, Dict[str, Any]] = {}
            for row in normalized_records:
                ts_code = str(row.get("ts_code") or "").strip()
                if ts_code:
                    latest_by_symbol[ts_code] = row
            output_records = list(latest_by_symbol.values())
            returned_count = len(latest_by_symbol)
        return output_records, {
            "requested": len(symbols),
            "returned": returned_count,
            "returned_rows": len(output_records),
            "endpoint": "daily_basic",
            "mode": mode,
            "shape": "long" if long_rows else "latest_by_symbol",
            "trade_date": trade_date,
            "all_market_rows": all_market_rows,
            "batch_error": batch_error,
            "fallback_errors": fallback_errors[:20],
        }

    def _dataframe_fetch_price_panel(
        self,
        universe_rows: List[Dict[str, Any]],
        metric_spec: Dict[str, Any],
        *,
        metric_source: str,
        timeout: int = 30,
        long_rows: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        requested_start = self._coerce_date(
            metric_spec.get("start_date")
            or metric_spec.get("start")
            or metric_spec.get("from")
            or metric_spec.get("date")
            or metric_spec.get("trade_date"),
            default="",
            is_end=False,
        )
        requested_end = self._coerce_date(
            metric_spec.get("end_date")
            or metric_spec.get("end")
            or metric_spec.get("to")
            or metric_spec.get("date")
            or metric_spec.get("trade_date")
            or requested_start,
            default=requested_start,
            is_end=True,
        )
        if not requested_start or not requested_end:
            return [], {
                "requested": len(universe_rows),
                "returned": 0,
                "mode": metric_source,
                "error": "price/return/volume panels require start_date/end_date or date.",
            }
        if requested_start > requested_end:
            requested_start, requested_end = requested_end, requested_start
        fetch_start = self._shift_iso_date(requested_start, -10)
        fetch_end = self._shift_iso_date(requested_end, 10)
        panel_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        seen: set = set()
        for universe_row in universe_rows:
            symbol_key = self._dataframe_symbol_key(universe_row)
            if not symbol_key or symbol_key in seen:
                continue
            seen.add(symbol_key)
            price_args = self._dataframe_price_history_args(universe_row, metric_spec, fetch_start, fetch_end)
            result = self.forward("price_history", price_args, timeout=timeout)
            if result.status != "success" or not result.tables:
                errors.append({"symbol_key": symbol_key, "error": result.error, "provider": result.provider})
                continue
            raw_rows = result.tables[0].get("rows") if isinstance(result.tables[0], dict) else []
            canonical_rows = [
                canonical
                for canonical in (self._canonical_price_record(row) for row in raw_rows if isinstance(row, dict))
                if canonical
            ]
            if long_rows:
                daily_rows = self._dataframe_price_panel_daily_rows(
                    universe_row,
                    canonical_rows,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    metric_source=metric_source,
                    metric_spec=metric_spec,
                    provider=result.provider,
                )
                if daily_rows:
                    panel_rows.extend(daily_rows)
                else:
                    errors.append({"symbol_key": symbol_key, "error": "no aligned trading rows", "provider": result.provider})
            else:
                summary = self._dataframe_price_panel_summary(
                    universe_row,
                    canonical_rows,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    metric_source=metric_source,
                    metric_spec=metric_spec,
                    provider=result.provider,
                )
                if summary:
                    panel_rows.append(summary)
                else:
                    errors.append({"symbol_key": symbol_key, "error": "no aligned trading rows", "provider": result.provider})
        returned_entities = len({str(row.get("symbol_key") or "") for row in panel_rows if row.get("symbol_key")})
        return panel_rows, {
            "requested": len(seen),
            "returned": returned_entities if long_rows else len(panel_rows),
            "returned_rows": len(panel_rows),
            "mode": metric_source,
            "shape": "long" if long_rows else "summary",
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "fetch_start_date": fetch_start,
            "fetch_end_date": fetch_end,
            "trading_day_align": "start_on_or_after,end_on_or_before; single_date defaults to previous",
            "errors": errors[:20],
        }

    def _dataframe_price_history_args(
        self,
        row: Dict[str, Any],
        metric_spec: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        market = str(metric_spec.get("market") or row.get("market") or "").strip().lower()
        provider = str(metric_spec.get("provider") or row.get("provider") or "").strip().lower()
        ticker = str(row.get("provider_symbol") or row.get("ticker") or row.get("symbol_key") or "").strip()
        if market in {"hk", "hkg", "hongkong", "hk_stock"}:
            provider = provider or "yahoo"
            market = "hk"
        elif market in {"us", "usa", "equity_us", "stock_us", "global", "yahoo", "etf", "us_etf"}:
            provider = provider or "yahoo"
            market = "us" if market in {"etf", "us_etf", "equity_us", "stock_us", "usa"} else market
        elif market in {"etf_cn", "cn_etf", "fund"}:
            market = "etf_cn"
        args: Dict[str, Any] = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "market": market,
            "provider": provider or "auto",
            "adjust": metric_spec.get("adjust", ""),
        }
        if metric_spec.get("asset_class") not in (None, ""):
            args["asset_class"] = metric_spec.get("asset_class")
        return args

    @staticmethod
    def _shift_iso_date(value: str, days: int) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except Exception:
            return value
        return (parsed + timedelta(days=days)).isoformat()

    def _canonical_price_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        date = self._canonical_price_date(
            row.get("date")
            or row.get("trade_date")
            or row.get("日期")
            or row.get("交易日期")
            or row.get("时间")
        )
        if not date:
            return {}
        return {
            "date": date,
            "open": self._first_float(row, ["raw_unadjusted_open", "open", "开盘", "开盘价"]),
            "high": self._first_float(row, ["high", "最高", "最高价"]),
            "low": self._first_float(row, ["low", "最低", "最低价"]),
            "close": self._first_float(row, ["raw_unadjusted_close", "close", "收盘", "收盘价", "最新价"]),
            "adj_close": self._first_float(row, ["adj_close", "Adj Close", "复权收盘价"]),
            "volume": self._first_float(row, ["volume", "vol", "成交量"]),
            "amount": self._first_float(row, ["amount", "成交额"]),
        }

    def _canonical_price_date(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if re.fullmatch(DATE_REGEX, text[:10]):
            return text[:10]
        digits = re.sub(r"\D", "", text)
        if re.fullmatch(r"\d{8}", digits):
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        return self._coerce_date(text, default="", is_end=False)

    def _first_float(self, row: Dict[str, Any], fields: List[str]) -> Optional[float]:
        for field in fields:
            if field in row:
                value = self._to_float(row.get(field))
                if value is not None:
                    return value
        return None

    def _dataframe_price_panel_daily_rows(
        self,
        universe_row: Dict[str, Any],
        records: List[Dict[str, Any]],
        *,
        requested_start: str,
        requested_end: str,
        metric_source: str,
        metric_spec: Dict[str, Any],
        provider: str,
    ) -> List[Dict[str, Any]]:
        records = sorted((row for row in records if row.get("date")), key=lambda row: str(row.get("date")))
        if not records:
            return []
        single_date = requested_start == requested_end
        align = str(metric_spec.get("align") or metric_spec.get("date_align") or "").strip().lower()
        if single_date:
            mode = align if align in {"next", "on_or_after", "exact"} else "previous"
            start_record = self._select_price_record(records, requested_start, mode)
            end_record = start_record
        else:
            start_record = self._select_price_record(records, requested_start, "on_or_after")
            end_record = self._select_price_record(records, requested_end, "previous")
        if not start_record or not end_record:
            return []
        start_actual = str(start_record.get("date"))
        end_actual = str(end_record.get("date"))
        previous_close_by_date: Dict[str, Optional[float]] = {}
        last_close: Optional[float] = None
        for row in records:
            date = str(row.get("date") or "")
            previous_close_by_date[date] = last_close
            close = self._to_float(row.get("close"))
            if close is not None:
                last_close = close

        output: List[Dict[str, Any]] = []
        for row in records:
            date = str(row.get("date") or "")
            if date < start_actual or date > end_actual:
                continue
            close = self._to_float(row.get("close"))
            pre_close = self._to_float(row.get("pre_close"))
            if pre_close is None:
                pre_close = previous_close_by_date.get(date)
            pct_chg = None
            if pre_close not in (None, 0) and close is not None:
                pct_chg = ((close / pre_close) - 1) * 100
            output.append(
                {
                    "symbol_key": self._dataframe_symbol_key(universe_row),
                    "ticker": universe_row.get("ticker") or universe_row.get("symbol_key"),
                    "symbol": universe_row.get("symbol") or universe_row.get("ticker") or universe_row.get("symbol_key"),
                    "market": universe_row.get("market"),
                    "provider": provider,
                    "metric_source": metric_source,
                    "requested_start_date": requested_start,
                    "requested_end_date": requested_end,
                    "actual_start_date": start_actual,
                    "actual_end_date": end_actual,
                    "date": date,
                    "trade_date": date.replace("-", ""),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": close,
                    "pre_close": pre_close,
                    "pct_chg": pct_chg,
                    "change": close - pre_close if close is not None and pre_close is not None else None,
                    "adj_close": row.get("adj_close"),
                    "volume": row.get("volume"),
                    "vol": row.get("volume"),
                    "amount": row.get("amount"),
                    "trading_day_aligned": start_actual != requested_start or end_actual != requested_end,
                }
            )
        return output

    def _dataframe_price_panel_summary(
        self,
        universe_row: Dict[str, Any],
        records: List[Dict[str, Any]],
        *,
        requested_start: str,
        requested_end: str,
        metric_source: str,
        metric_spec: Dict[str, Any],
        provider: str,
    ) -> Dict[str, Any]:
        records = sorted((row for row in records if row.get("date")), key=lambda row: str(row.get("date")))
        if not records:
            return {}
        single_date = requested_start == requested_end
        align = str(metric_spec.get("align") or metric_spec.get("date_align") or "").strip().lower()
        if single_date:
            mode = align if align in {"next", "on_or_after", "exact"} else "previous"
            start_record = self._select_price_record(records, requested_start, mode)
            end_record = start_record
        else:
            start_record = self._select_price_record(records, requested_start, "on_or_after")
            end_record = self._select_price_record(records, requested_end, "previous")
        if not start_record or not end_record:
            return {}
        start_actual = str(start_record.get("date"))
        end_actual = str(end_record.get("date"))
        window_rows = [
            row for row in records
            if start_actual <= str(row.get("date")) <= end_actual
        ] or [start_record, end_record]
        start_close = self._to_float(start_record.get("close"))
        end_close = self._to_float(end_record.get("close"))
        return_pct = ((end_close / start_close) - 1) * 100 if start_close not in (None, 0) and end_close is not None else None
        summary = {
            "symbol_key": self._dataframe_symbol_key(universe_row),
            "ticker": universe_row.get("ticker") or universe_row.get("symbol_key"),
            "symbol": universe_row.get("symbol") or universe_row.get("ticker") or universe_row.get("symbol_key"),
            "market": universe_row.get("market"),
            "provider": provider,
            "metric_source": metric_source,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "actual_start_date": start_actual,
            "actual_end_date": end_actual,
            "start_close": start_close,
            "end_close": end_close,
            "return_pct": return_pct,
            "return_ratio": return_pct / 100 if return_pct is not None else None,
            "panel_row_count": len(window_rows),
            "trading_day_aligned": start_actual != requested_start or end_actual != requested_end,
        }
        summary.update(self._window_extrema(window_rows, "high", "max_high", find_max=True))
        summary.update(self._window_extrema(window_rows, "low", "min_low", find_max=False))
        summary.update(self._window_extrema(window_rows, "close", "max_close", find_max=True))
        summary.update(self._window_extrema(window_rows, "close", "min_close", find_max=False))
        summary.update(self._window_extrema(window_rows, "volume", "max_volume", find_max=True))
        volumes = [self._to_float(row.get("volume")) for row in window_rows]
        volumes = [value for value in volumes if value is not None]
        summary["volume_sum"] = sum(volumes) if volumes else None
        summary["avg_volume"] = sum(volumes) / len(volumes) if volumes else None
        if metric_source == "return_panel":
            summary["metric_value"] = return_pct
        elif metric_source == "volume_panel":
            summary["metric_value"] = summary.get("volume_sum")
        elif metric_source == "price_panel":
            summary["metric_value"] = end_close
        return summary

    def _select_price_record(self, records: List[Dict[str, Any]], target: str, mode: str) -> Optional[Dict[str, Any]]:
        normalized_mode = mode.lower().replace("-", "_")
        if normalized_mode == "exact":
            return next((row for row in records if str(row.get("date")) == target), None)
        if normalized_mode in {"next", "on_or_after", "after"}:
            candidates = [row for row in records if str(row.get("date")) >= target]
            return candidates[0] if candidates else None
        candidates = [row for row in records if str(row.get("date")) <= target]
        return candidates[-1] if candidates else None

    def _window_extrema(
        self,
        rows: List[Dict[str, Any]],
        field: str,
        output_field: str,
        *,
        find_max: bool,
    ) -> Dict[str, Any]:
        candidates = [
            (self._to_float(row.get(field)), str(row.get("date")))
            for row in rows
            if self._to_float(row.get(field)) is not None
        ]
        if not candidates:
            return {output_field: None, f"{output_field}_date": None}
        value, date = (max if find_max else min)(candidates, key=lambda item: item[0])
        return {output_field: value, f"{output_field}_date": date}

    @staticmethod
    def _dataframe_left_join(
        left_rows: List[Dict[str, Any]],
        right_rows: List[Dict[str, Any]],
        *,
        left_keys: List[str],
        right_key: str,
    ) -> List[Dict[str, Any]]:
        right_by_key = {str(row.get(right_key) or "").strip(): row for row in right_rows}
        joined = []
        for left in left_rows:
            key = ""
            for left_key in left_keys:
                candidate = str(left.get(left_key) or "").strip()
                if candidate:
                    key = candidate
                    break
            merged = dict(left)
            if key in right_by_key:
                for field, value in right_by_key[key].items():
                    merged.setdefault(field, value)
                    if field not in left:
                        merged[field] = value
            joined.append(merged)
        return joined

    @staticmethod
    def _dataframe_expand_join(
        left_rows: List[Dict[str, Any]],
        right_rows: List[Dict[str, Any]],
        *,
        left_keys: List[str],
        right_key: str,
    ) -> List[Dict[str, Any]]:
        left_by_key: Dict[str, Dict[str, Any]] = {}
        for left in left_rows:
            for left_key in left_keys:
                candidate = str(left.get(left_key) or "").strip()
                if candidate:
                    left_by_key.setdefault(candidate, left)
        expanded: List[Dict[str, Any]] = []
        for right in right_rows:
            key = str(right.get(right_key) or "").strip()
            base = left_by_key.get(key, {})
            merged = dict(base)
            merged.update(right)
            expanded.append(merged)
        return expanded

    def _dataframe_execution_card(
        self,
        plan: Dict[str, Any],
        rows: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        missing_slots: List[Dict[str, Any]] = []
        for item in trace:
            errors = item.get("errors")
            if not isinstance(errors, list):
                continue
            for error in errors[:12]:
                if not isinstance(error, dict):
                    continue
                missing_slots.append(
                    {
                        "entity": error.get("symbol_key") or error.get("ticker") or error.get("symbol"),
                        "reason": compact_text(str(error.get("error") or "fetch_failed"), 180),
                        "provider": error.get("provider"),
                    }
                )
        filled_slots = []
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            filled_slots.append(
                self._dataframe_drop_empty(
                    {
                        "entity": row.get("symbol_key") or row.get("ticker") or row.get("ts_code") or row.get("symbol"),
                        "date": row.get("date") or row.get("trade_date") or row.get("actual_end_date"),
                        "metric_source": row.get("metric_source"),
                        "value": row.get("metric_value")
                        if row.get("metric_value") is not None
                        else row.get("value")
                        if row.get("value") is not None
                        else row.get("close")
                        if row.get("close") is not None
                        else row.get("return_pct"),
                        "basis": {
                            "provider": row.get("provider"),
                            "actual_start_date": row.get("actual_start_date"),
                            "actual_end_date": row.get("actual_end_date"),
                        },
                    }
                )
            )
        result_rows = int(coverage.get("result_rows", 0) or 0)
        requested = int(coverage.get("entities_requested", 0) or 0)
        processed = int(coverage.get("entities_processed", 0) or 0)
        no_matching_rows = self._dataframe_no_matching_rows(trace)
        status = "success"
        if missing_slots and result_rows:
            status = "partial_success"
        elif not result_rows and no_matching_rows:
            status = "success_no_matching_rows"
        elif not result_rows:
            status = "non_retryable_failure"
        return self._dataframe_drop_empty(
            {
                "card_type": "dataframe_execution_card",
                "card_version": "1",
                "status": status,
                "coverage_type": ",".join(
                    sorted(
                        {
                            str(item.get("source") or "")
                            for item in trace
                            if isinstance(item, dict) and item.get("op") == "batch_metric"
                        }
                    )
                ),
                "coverage": coverage,
                "filled_slots": filled_slots,
                "missing_slots": missing_slots,
                "retry_same_plan_allowed": False if status == "non_retryable_failure" else None,
                "can_answer_now": bool((result_rows or no_matching_rows) and not missing_slots and (not requested or processed >= requested)),
                "next_react_state_update": {
                    "answer_state": "add filled_slots that match the task contract",
                    "open_slots": "keep only requested cells not covered by filled_slots",
                    "next_focus": "use missing_slots or compute from result rows",
                },
            }
        )

    @staticmethod
    def _dataframe_drop_empty(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                compacted = MarketDataTool._dataframe_drop_empty(item)
                if compacted in (None, "", [], {}):
                    continue
                cleaned[key] = compacted
            return cleaned
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                compacted = MarketDataTool._dataframe_drop_empty(item)
                if compacted in (None, "", [], {}):
                    continue
                cleaned_list.append(compacted)
            return cleaned_list
        return value

    @staticmethod
    def _dataframe_no_matching_rows(trace: List[Dict[str, Any]]) -> bool:
        fetched_rows = any(
            isinstance(item, dict)
            and item.get("op") == "batch_metric"
            and int(item.get("returned_rows", 0) or 0) > 0
            for item in trace
        )
        filtered_to_zero = any(
            isinstance(item, dict)
            and item.get("op") == "filter"
            and int(item.get("before", 0) or 0) > 0
            and int(item.get("after", 0) or 0) == 0
            for item in trace
        )
        return fetched_rows and filtered_to_zero

    def _apply_dataframe_operations(
        self,
        rows: List[Dict[str, Any]],
        operations: Any,
        *,
        default_top_k: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        if not isinstance(operations, list):
            return rows, trace
        current = list(rows)
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            op = str(operation.get("op") or operation.get("type") or "").strip().lower()
            if op == "filter":
                before = len(current)
                field = str(operation.get("field") or operation.get("column") or operation.get("by") or "").strip()
                operator = str(operation.get("operator") or operation.get("comparison") or operation.get("comparator") or operation.get("condition") or "eq").strip().lower()
                value = operation.get("value") if operation.get("value") not in (None, "") else operation.get("threshold")
                if not field:
                    trace.append({"op": "filter", "field": field, "operator": operator, "before": before, "after": len(current), "skipped": "missing_field"})
                    continue
                current = [row for row in current if self._dataframe_filter_match(row.get(field), operator, value)]
                trace.append({"op": "filter", "field": field, "operator": operator, "before": before, "after": len(current)})
            elif op == "add_column":
                output_field = str(operation.get("column") or operation.get("as") or operation.get("output_field") or "").strip()
                expression = str(operation.get("expression") or "").strip()
                before = len(current)
                if output_field and self._is_close_pre_close_return_expression(expression):
                    updated = []
                    non_null = 0
                    for row in current:
                        close = self._to_float(row.get("close"))
                        pre_close = self._to_float(row.get("pre_close"))
                        value = ((close / pre_close) - 1) * 100 if pre_close not in (None, 0) and close is not None else None
                        if value is not None:
                            non_null += 1
                        updated.append({**row, output_field: value})
                    current = updated
                trace.append({"op": "add_column", "field": output_field, "expression": compact_text(expression, 120), "before": before, "non_null_rows": sum(1 for row in current if row.get(output_field) is not None) if output_field else 0})
            elif op == "sort":
                field = str(operation.get("field") or operation.get("by") or "").strip()
                ascending = operation.get("ascending")
                order = str(operation.get("order") or ("asc" if ascending is not False else "desc")).strip().lower()
                reverse = order == "desc"
                current.sort(
                    key=lambda row: self._dataframe_sort_key(row.get(field), reverse=reverse),
                    reverse=reverse,
                )
                trace.append({"op": "sort", "field": field, "order": order, "after": len(current)})
            elif op == "rank":
                field = str(operation.get("by") or operation.get("field") or "").strip()
                order = str(operation.get("order") or "desc").strip().lower()
                top_k = self._positive_int(operation.get("top_k"), default_top_k)
                reverse = order != "asc"
                def rank_value(row: Dict[str, Any]) -> float:
                    value = self._to_float(row.get(field))
                    if value is None:
                        return float("-inf") if reverse else float("inf")
                    return value
                current.sort(
                    key=rank_value,
                    reverse=reverse,
                )
                current = [{**row, "rank": idx + 1} for idx, row in enumerate(current[:top_k])]
                trace.append({"op": "rank", "field": field, "order": order, "top_k": top_k, "after": len(current)})
            elif op in {"max_by", "min_by"}:
                field = str(operation.get("by") or operation.get("field") or "").strip()
                top_k = self._positive_int(operation.get("top_k"), default_top_k)
                reverse = op == "max_by"
                candidates = [row for row in current if self._to_float(row.get(field)) is not None]
                candidates.sort(key=lambda row: self._to_float(row.get(field)) or 0.0, reverse=reverse)
                current = [
                    {**row, "rank": idx + 1, "selected_by": field, "selection_op": op}
                    for idx, row in enumerate(candidates[:top_k])
                ]
                trace.append({"op": op, "field": field, "top_k": top_k, "after": len(current)})
            elif op in {"pct_change", "return", "returns"}:
                if operation.get("column") not in (None, ""):
                    column = str(operation.get("column") or "").strip()
                    output_field = str(operation.get("as") or operation.get("output_field") or f"pct_change_{column}").strip()
                    as_ratio = bool(operation.get("as_ratio"))
                    current = self._dataframe_row_pct_change(current, column=column, output_field=output_field, as_ratio=as_ratio)
                    trace.append({"op": "pct_change", "column": column, "output_field": output_field, "mode": "rowwise", "as_ratio": as_ratio})
                    continue
                start_field = str(operation.get("start_field") or operation.get("from_field") or "start_close").strip()
                end_field = str(operation.get("end_field") or operation.get("to_field") or "end_close").strip()
                output_field = str(operation.get("as") or operation.get("output_field") or "return_pct").strip()
                updated = []
                non_null = 0
                for row in current:
                    start_value = self._to_float(row.get(start_field))
                    end_value = self._to_float(row.get(end_field))
                    value = ((end_value / start_value) - 1) * 100 if start_value not in (None, 0) and end_value is not None else None
                    if value is not None:
                        non_null += 1
                    updated.append({**row, output_field: value})
                current = updated
                trace.append({"op": "pct_change", "start_field": start_field, "end_field": end_field, "output_field": output_field, "non_null_rows": non_null})
            elif op in {"aggregate", "count"}:
                field = str(operation.get("field") or "").strip()
                func = str(operation.get("function") or operation.get("func") or ("count" if op == "count" else "count")).strip().lower()
                values = [self._to_float(row.get(field)) for row in current]
                numeric = [value for value in values if value is not None]
                if func == "sum":
                    aggregate_value: Any = sum(numeric)
                elif func in {"mean", "avg", "average"}:
                    aggregate_value = sum(numeric) / len(numeric) if numeric else None
                elif func in {"min", "minimum"}:
                    aggregate_value = min(numeric) if numeric else None
                elif func in {"max", "maximum"}:
                    aggregate_value = max(numeric) if numeric else None
                else:
                    aggregate_value = len(current)
                current = [{"aggregate_function": func, "field": field, "value": aggregate_value, "input_rows": len(rows), "non_null_rows": len(numeric)}]
                trace.append({"op": "aggregate", "field": field, "function": func, "after": 1})
            elif op in {"list", "select_fields"}:
                raw_fields = operation.get("fields") or operation.get("columns") or []
                fields = [str(item).strip() for item in raw_fields] if isinstance(raw_fields, list) else [item.strip() for item in str(raw_fields).split(",") if item.strip()]
                limit = self._positive_int(operation.get("limit") or operation.get("top_k"), len(current) if op == "select_fields" else default_top_k)
                if fields:
                    current = [{field: row.get(field) for field in fields if field in row} for row in current[:limit]]
                else:
                    current = current[:limit]
                trace.append({"op": op, "fields": fields, "limit": limit, "after": len(current)})
        return current, trace

    @staticmethod
    def _is_close_pre_close_return_expression(expression: str) -> bool:
        compact = re.sub(r"\s+", "", expression.lower())
        return "close/pre_close" in compact and any(token in compact for token in ("-1", "1-")) and ("*100" in compact or "100*" in compact)

    def _dataframe_row_pct_change(
        self,
        rows: List[Dict[str, Any]],
        *,
        column: str,
        output_field: str,
        as_ratio: bool = False,
    ) -> List[Dict[str, Any]]:
        sorted_rows = sorted(rows, key=lambda row: (str(row.get("symbol_key") or row.get("ticker") or row.get("ts_code") or ""), str(row.get("date") or row.get("trade_date") or "")))
        previous_by_symbol: Dict[str, Optional[float]] = {}
        updated: List[Dict[str, Any]] = []
        for row in sorted_rows:
            symbol = str(row.get("symbol_key") or row.get("ticker") or row.get("ts_code") or "")
            current_value = self._to_float(row.get(column))
            previous_value = previous_by_symbol.get(symbol)
            pct = (current_value / previous_value) - 1 if previous_value not in (None, 0) and current_value is not None else None
            next_row = dict(row)
            next_row[output_field] = pct if as_ratio or pct is None else pct * 100
            updated.append(next_row)
            if current_value is not None:
                previous_by_symbol[symbol] = current_value
        return updated

    def _dataframe_sort_key(self, value: Any, *, reverse: bool) -> Any:
        numeric = self._to_float(value)
        if numeric is not None:
            return (0, numeric)
        text = str(value or "")
        return (1, text if text else ("\uffff" if not reverse else ""))

    def _dataframe_filter_match(self, raw_value: Any, operator: str, expected: Any) -> bool:
        actual_num = self._to_float(raw_value)
        expected_num = self._to_float(expected)
        if operator in {"gt", ">", "greater_than"}:
            return actual_num is not None and expected_num is not None and actual_num > expected_num
        if operator in {"ge", ">=", "gte"}:
            return actual_num is not None and expected_num is not None and actual_num >= expected_num
        if operator in {"lt", "<", "less_than"}:
            return actual_num is not None and expected_num is not None and actual_num < expected_num
        if operator in {"le", "<=", "lte"}:
            return actual_num is not None and expected_num is not None and actual_num <= expected_num
        if operator in {"ne", "!=", "not_equal"}:
            return str(raw_value) != str(expected)
        if operator in {"contains", "like"}:
            return str(expected).lower() in str(raw_value).lower()
        return str(raw_value) == str(expected)

    def _china_index_namespace(self, arguments: Dict[str, Any], *, catalog: bool = False) -> str:
        provider = self._provider_hint(arguments)
        if provider in {"sw", "shenwan"}:
            return "sw"
        if provider in {"csindex", "csi"}:
            return "csindex"
        raw_code = self._raw_index_code(arguments)
        upper_code = raw_code.upper()
        if re.search(r"\d{6}\.SI\b", upper_code):
            return "sw"
        compact_code = re.sub(r"\D", "", upper_code)
        if re.fullmatch(r"(?:80[0-9]|85[0-9])\d{3}", compact_code):
            return "sw"
        text = " ".join(
            str(arguments.get(key) or "")
            for key in ("query", "index_code", "name", "market", "publisher", "category", "provider", "source")
        )
        lowered = text.lower()
        if "申万" in text or "shenwan" in lowered or "sw index" in lowered:
            return "sw"
        if "中证" in text or "csindex" in lowered or "csi index" in lowered or "csi指数" in lowered:
            return "csindex"
        if catalog and provider == "akshare":
            if "sw" in lowered:
                return "sw"
            if "csi" in lowered:
                return "csindex"
        return ""

    def _raw_index_code(self, arguments: Dict[str, Any]) -> str:
        raw = str(
            arguments.get("index_code")
            or arguments.get("code")
            or arguments.get("ticker")
            or arguments.get("symbol")
            or arguments.get("query")
            or ""
        ).strip()
        if re.fullmatch(r"\d{6}(?:\.[A-Za-z]{2,3})?", raw.strip(), flags=re.I):
            return raw.strip()
        match = re.search(r"\b(\d{6})(?:\.(SI|SH|SZ|BJ|SS))?\b", raw.upper())
        if match:
            return f"{match.group(1)}.{match.group(2)}" if match.group(2) else match.group(1)
        return raw

    @staticmethod
    def _strip_china_index_suffix(code: Any) -> str:
        value = str(code or "").strip().upper()
        return re.sub(r"\.(SI|SH|SZ|BJ|SS)$", "", value)

    def _resolve_sw_index_code(self, arguments: Dict[str, Any], timeout: int = 30) -> str:
        raw = self._raw_index_code(arguments)
        compact = self._strip_china_index_suffix(raw)
        if re.fullmatch(r"\d{6}", compact):
            return compact
        query = str(arguments.get("query") or arguments.get("name") or raw or "").strip()
        if not query:
            return ""
        result = self._forward_sw_index_catalog("index_catalog", {"query": query, "limit": 5}, timeout=timeout)
        if result.status != "success" or not result.tables:
            return ""
        rows = result.tables[0].get("rows") if isinstance(result.tables[0], dict) else []
        if not isinstance(rows, list) or not rows:
            return ""
        code = rows[0].get("index_code") or rows[0].get("行业代码") or rows[0].get("代码")
        return self._strip_china_index_suffix(code)

    def _resolve_csindex_code(self, arguments: Dict[str, Any], timeout: int = 30) -> str:
        raw = self._raw_index_code(arguments)
        compact = self._strip_china_index_suffix(raw)
        if re.fullmatch(r"\d{6}", compact):
            return compact
        query = str(arguments.get("query") or arguments.get("name") or raw or "").strip()
        if not query:
            return ""
        result = self._forward_csindex_index_catalog("index_catalog", {"query": query, "limit": 5}, timeout=timeout)
        if result.status != "success" or not result.tables:
            return ""
        rows = result.tables[0].get("rows") if isinstance(result.tables[0], dict) else []
        if not isinstance(rows, list) or not rows:
            return ""
        code = rows[0].get("index_code") or rows[0].get("指数代码") or rows[0].get("代码")
        return self._strip_china_index_suffix(code)

    def _forward_sw_index_catalog(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        level = str(arguments.get("level") or arguments.get("category") or "").strip().lower()
        if level in {"1", "first", "一级", "申万一级"}:
            functions = [("sw_index_first_info", "1")]
        elif level in {"2", "second", "二级", "申万二级"}:
            functions = [("sw_index_second_info", "2")]
        elif level in {"3", "third", "三级", "申万三级"}:
            functions = [("sw_index_third_info", "3")]
        else:
            functions = [
                ("sw_index_first_info", "1"),
                ("sw_index_second_info", "2"),
                ("sw_index_third_info", "3"),
            ]
        records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for function, level_value in functions:
            frame = self._akshare_dataframe(function, {}, timeout=timeout)
            if isinstance(frame, ToolResult):
                errors.append(f"{function}: {frame.error}")
                continue
            for row in self._dataframe_records(frame):
                code = str(row.get("行业代码") or row.get("代码") or "").strip().upper()
                row.setdefault("index_code", code)
                row.setdefault("index_name", row.get("行业名称") or row.get("名称"))
                row["index_namespace"] = "SW_INDEX"
                row["sw_level"] = level_value
                row["source_function"] = function
                records.append(row)
        query = str(arguments.get("query") or arguments.get("index_code") or "").strip()
        records = self._filter_records_by_text(records, query, ["index_code", "index_name", "行业代码", "行业名称"])
        records = self._limit_records(records, arguments, limit_mode="head")
        return self._records_result(
            "akshare_sw",
            action,
            records,
            columns=self._record_columns(records, ["index_code", "index_name", "sw_level", "成份个数", "静态市盈率", "TTM(滚动)市盈率", "市净率", "静态股息率"]),
            metadata={
                "provider": "akshare",
                "namespace": "SW_INDEX",
                "functions": [name for name, _level in functions],
                "query": query,
                "errors": errors,
            },
            error="AkShare SW index catalog returned no matching rows.",
        )

    def _forward_sw_index_constituents(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        index_code = self._resolve_sw_index_code(arguments, timeout=timeout)
        if not index_code:
            return self._structured_error(
                "akshare_sw",
                action,
                "Shenwan/SW index constituents require a .SI code such as 801125.SI or a resolvable Shenwan index name.",
                metadata={"requested_index_code": self._raw_index_code(arguments)},
            )
        frame = self._akshare_dataframe("index_component_sw", {"symbol": index_code}, timeout=timeout)
        if isinstance(frame, ToolResult):
            frame.action = action
            return frame
        records = self._dataframe_records(frame)
        for row in records:
            raw_code = row.get("证券代码") or row.get("代码")
            row.setdefault("index_code", f"{index_code}.SI")
            row.setdefault("constituent_code", normalize_tushare_ts_code(raw_code) or str(raw_code or ""))
            row.setdefault("constituent_name", row.get("证券名称") or row.get("名称"))
            row.setdefault("weight", self._to_float(row.get("最新权重") or row.get("权重")))
            row["index_namespace"] = "SW_INDEX"
            row["source_function"] = "index_component_sw"
        records = self._limit_records(records, arguments, limit_mode="head")
        return self._records_result(
            "akshare_sw",
            action,
            records,
            columns=self._record_columns(records, ["index_code", "constituent_code", "constituent_name", "weight", "计入日期"]),
            metadata={
                "provider": "akshare",
                "namespace": "SW_INDEX",
                "function": "index_component_sw",
                "index_code": f"{index_code}.SI",
            },
            error=f"AkShare SW index_component_sw returned no rows for {index_code}.",
        )

    def _forward_sw_index_history(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        index_code = self._resolve_sw_index_code(arguments, timeout=timeout)
        if not index_code:
            return self._structured_error(
                "akshare_sw",
                action,
                "Shenwan/SW index history requires a .SI code such as 851246.SI or a resolvable Shenwan index name.",
                metadata={"requested_index_code": self._raw_index_code(arguments)},
            )
        period = self._akshare_sw_period(arguments.get("period"))
        frame = self._akshare_dataframe("index_hist_sw", {"symbol": index_code, "period": period}, timeout=timeout)
        if isinstance(frame, ToolResult):
            frame.action = action
            return frame
        frame = self._filter_date_range(frame, arguments)
        records = self._dataframe_records(frame)
        for row in records:
            row.setdefault("index_code", f"{index_code}.SI")
            row.setdefault("date", row.get("日期"))
            row.setdefault("close", self._to_float(row.get("收盘")))
            row.setdefault("open", self._to_float(row.get("开盘")))
            row.setdefault("high", self._to_float(row.get("最高")))
            row.setdefault("low", self._to_float(row.get("最低")))
            row["index_namespace"] = "SW_INDEX"
            row["source_function"] = "index_hist_sw"
        records = self._limit_records(records, arguments, limit_mode="tail")
        return self._records_result(
            "akshare_sw",
            action,
            records,
            columns=self._record_columns(records, ["date", "index_code", "open", "high", "low", "close", "成交量", "成交额"]),
            metadata={
                "provider": "akshare",
                "namespace": "SW_INDEX",
                "function": "index_hist_sw",
                "index_code": f"{index_code}.SI",
                "period": period,
            },
            error=f"AkShare SW index_hist_sw returned no rows for {index_code}.",
        )

    def _forward_csindex_index_catalog(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        frame = self._akshare_dataframe("index_csindex_all", {}, timeout=timeout)
        if isinstance(frame, ToolResult):
            frame.action = action
            return frame
        records = self._dataframe_records(frame)
        for row in records:
            code = row.get("指数代码") or row.get("代码") or row.get("index_code")
            name = row.get("指数简称") or row.get("指数名称") or row.get("名称") or row.get("index_name")
            row.setdefault("index_code", self._strip_china_index_suffix(code))
            row.setdefault("index_name", name)
            row["index_namespace"] = "CSINDEX"
            row["source_function"] = "index_csindex_all"
        query = str(arguments.get("query") or arguments.get("index_code") or "").strip()
        records = self._filter_records_by_text(records, query, ["index_code", "index_name", "指数代码", "指数简称", "指数名称"])
        records = self._limit_records(records, arguments, limit_mode="head")
        return self._records_result(
            "akshare_csindex",
            action,
            records,
            columns=self._record_columns(records, ["index_code", "index_name", "指数英文名称", "发布日期", "样本数量"]),
            metadata={
                "provider": "akshare",
                "namespace": "CSINDEX",
                "function": "index_csindex_all",
                "query": query,
            },
            error="AkShare CSIndex catalog returned no matching rows.",
        )

    def _forward_csindex_index_constituents(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        index_code = self._resolve_csindex_code(arguments, timeout=timeout)
        if not index_code:
            return self._structured_error(
                "akshare_csindex",
                action,
                "CSIndex/Zhongzheng constituents require an index code such as 000300 or a resolvable CSI index name.",
                metadata={"requested_index_code": self._raw_index_code(arguments)},
            )
        function = "index_stock_cons_weight_csindex" if action == "index_weight" else "index_stock_cons_csindex"
        frame = self._akshare_dataframe(function, {"symbol": index_code}, timeout=timeout)
        if isinstance(frame, ToolResult):
            frame.action = action
            return frame
        records = self._dataframe_records(frame)
        for row in records:
            raw_code = row.get("成分券代码") or row.get("成份券代码") or row.get("证券代码") or row.get("品种代码") or row.get("代码")
            raw_name = row.get("成分券名称") or row.get("成份券名称") or row.get("证券简称") or row.get("证券名称") or row.get("名称")
            raw_weight = row.get("权重") or row.get("权重(%)") or row.get("最新权重")
            row.setdefault("index_code", index_code)
            row.setdefault("constituent_code", normalize_tushare_ts_code(raw_code) or str(raw_code or ""))
            row.setdefault("constituent_name", raw_name)
            row.setdefault("weight", self._to_float(raw_weight))
            row["index_namespace"] = "CSINDEX"
            row["source_function"] = function
        records = self._limit_records(records, arguments, limit_mode="head")
        return self._records_result(
            "akshare_csindex",
            action,
            records,
            columns=self._record_columns(records, ["index_code", "constituent_code", "constituent_name", "weight", "日期"]),
            metadata={
                "provider": "akshare",
                "namespace": "CSINDEX",
                "function": function,
                "index_code": index_code,
            },
            error=f"AkShare {function} returned no rows for {index_code}.",
        )

    def _forward_csindex_index_history(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        index_code = self._resolve_csindex_code(arguments, timeout=timeout)
        if not index_code:
            return self._structured_error(
                "akshare_csindex",
                action,
                "CSIndex/Zhongzheng history requires an index code such as 000300 or a resolvable CSI index name.",
                metadata={"requested_index_code": self._raw_index_code(arguments)},
            )
        kwargs = {
            "symbol": index_code,
            "period": self._akshare_csindex_period(arguments.get("period")),
        }
        start = self._date_argument(arguments, "start_date")
        end = self._date_argument(arguments, "end_date")
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        frame = self._akshare_dataframe("index_zh_a_hist", kwargs, timeout=timeout)
        if isinstance(frame, ToolResult):
            frame.action = action
            return frame
        frame = self._filter_date_range(frame, arguments)
        records = self._dataframe_records(frame)
        for row in records:
            row.setdefault("index_code", index_code)
            row.setdefault("date", row.get("日期"))
            row.setdefault("close", self._to_float(row.get("收盘")))
            row.setdefault("open", self._to_float(row.get("开盘")))
            row.setdefault("high", self._to_float(row.get("最高")))
            row.setdefault("low", self._to_float(row.get("最低")))
            row["index_namespace"] = "CSINDEX"
            row["source_function"] = "index_zh_a_hist"
        records = self._limit_records(records, arguments, limit_mode="tail")
        return self._records_result(
            "akshare_csindex",
            action,
            records,
            columns=self._record_columns(records, ["date", "index_code", "open", "high", "low", "close", "成交量", "成交额"]),
            metadata={
                "provider": "akshare",
                "namespace": "CSINDEX",
                "function": "index_zh_a_hist",
                "index_code": index_code,
            },
            error=f"AkShare index_zh_a_hist returned no rows for {index_code}.",
        )

    def _akshare_dataframe(self, function: str, kwargs: Dict[str, Any], *, timeout: int = 30) -> Any:
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            return self._structured_error(
                "akshare",
                function,
                f"AkShare dependency is not available: {exc}",
                metadata={
                    "function": function,
                    "kwargs": kwargs,
                    "python_executable": sys.executable,
                    "akshare_dependency_available": False,
                    "function_guidance": self._function_guidance(function, function),
                    "retry_same_plan_allowed": False,
                },
            )
        fn = getattr(ak, function, None)
        if not callable(fn):
            guidance = self._function_guidance(function, function)
            guidance_text = ""
            if guidance:
                guidance_text = f" Recommended next step: {guidance.get('recommended_action') or guidance.get('replacement_function')}."
            return self._structured_error(
                "akshare",
                function,
                f"AkShare function {function!r} is not available in the installed package.{guidance_text}",
                metadata={
                    "function": function,
                    "kwargs": kwargs,
                    "python_executable": sys.executable,
                    "akshare_dependency_available": True,
                    "function_guidance": guidance,
                    "retry_same_plan_allowed": False if guidance else None,
                },
            )
        try:
            return fn(**kwargs)
        except Exception as exc:
            return self._structured_error(
                "akshare",
                function,
                f"{type(exc).__name__}: {exc}",
                metadata={"function": function, "kwargs": kwargs, "timeout": timeout},
            )

    @staticmethod
    def _dataframe_records(df: Any) -> List[Dict[str, Any]]:
        if df is None or not hasattr(df, "to_dict"):
            return []
        records: List[Dict[str, Any]] = []
        for raw_row in df.to_dict("records"):
            records.append({str(key): TushareProvider._to_python_value(value) for key, value in raw_row.items()})
        return records

    @staticmethod
    def _filter_records_by_text(records: List[Dict[str, Any]], query: str, fields: List[str]) -> List[Dict[str, Any]]:
        needle = str(query or "").strip().upper()
        if not needle:
            return records
        return [
            row
            for row in records
            if any(needle in str(row.get(field) or "").upper() for field in fields)
        ]

    def _limit_records(self, records: List[Dict[str, Any]], arguments: Dict[str, Any], *, limit_mode: str) -> List[Dict[str, Any]]:
        limit = self._optional_positive_int(arguments.get("limit"))
        if limit is None or limit <= 0:
            return records
        return records[-limit:] if limit_mode == "tail" else records[:limit]

    @staticmethod
    def _akshare_sw_period(value: Any) -> str:
        text = str(value or "").strip().lower()
        mapping = {
            "": "day",
            "d": "day",
            "day": "day",
            "daily": "day",
            "week": "week",
            "weekly": "week",
            "month": "month",
            "monthly": "month",
        }
        return mapping.get(text, text or "day")

    @staticmethod
    def _akshare_csindex_period(value: Any) -> str:
        text = str(value or "").strip().lower()
        mapping = {
            "": "daily",
            "d": "daily",
            "day": "daily",
            "daily": "daily",
            "week": "weekly",
            "weekly": "weekly",
            "month": "monthly",
            "monthly": "monthly",
        }
        return mapping.get(text, text or "daily")

    def _augment_tushare_financial_result(self, result: ToolResult, endpoint: str, arguments: Dict[str, Any]) -> None:
        if result.status != "success":
            return
        table = result.tables[0] if result.tables else {}
        rows = table.get("rows") if isinstance(table, dict) else []
        columns = table.get("columns") if isinstance(table, dict) else []
        if not isinstance(rows, list) or not rows:
            return

        schema = self.TUSHARE_FINANCIAL_FIELD_SCHEMA.get(endpoint, {})
        if not schema:
            return
        available_fields = self._available_schema_fields(schema, rows, columns if isinstance(columns, list) else [])
        field_schema = [self._compact_schema_entry(field, schema[field]) for field in available_fields if field in schema]

        request_text = self._tushare_financial_request_text(arguments)
        selected_fields = self._tushare_schema_field_scores(schema, request_text, available_fields)
        if selected_fields:
            top_score = selected_fields[0][1]
            selected_fields = [item for item in selected_fields if item[1] >= top_score - 0.5]
        want_period = format_tushare_period(arguments.get("period")) or format_tushare_date(arguments.get("date"), is_end=True)
        candidate_rows = [row for row in rows if isinstance(row, dict)]
        ranked_rows = sorted(candidate_rows, key=lambda row: self._tushare_row_rank(row, want_period), reverse=True)

        semantic_candidates: List[Dict[str, Any]] = []
        for field, score in selected_fields:
            if field.startswith("__"):
                for row in ranked_rows[: max(1, min(3, len(ranked_rows)))]:
                    derived = self._derived_tushare_financial_value(field, row)
                    if derived is None:
                        continue
                    semantic_candidates.append(
                        self._semantic_candidate_from_value(
                            field,
                            schema[field],
                            derived,
                            row,
                            endpoint,
                            match_score=score,
                            source_fields=schema[field].get("source_fields"),
                        )
                    )
                    break
                continue
            for row in ranked_rows[: max(1, min(3, len(ranked_rows)))]:
                raw_value = self._get_record_value(row, field)
                numeric = self._coerce_financial_number(raw_value)
                if numeric is None:
                    continue
                semantic_candidates.append(
                    self._semantic_candidate_from_value(
                        field,
                        schema[field],
                        numeric,
                        row,
                        endpoint,
                        match_score=score,
                    )
                )
                break

        if not field_schema and not semantic_candidates:
            return
        payload = {
            "field_schema": field_schema[:16],
            "semantic_candidates": semantic_candidates[:12],
            "usage_note": (
                "Tushare typed schema: use semantic_candidates only after matching entity, period, "
                "metric label, unit, and source fields to the question. Raw CSV follows."
            ),
        }
        raw_observation = result.observation_text()
        result.observation = (
            json.dumps(payload, ensure_ascii=False, default=str)
            + ("\nraw_csv:\n" + raw_observation if raw_observation else "")
        )
        result.metadata = {
            **(result.metadata or {}),
            "field_schema": field_schema[:16],
            "semantic_candidates": semantic_candidates[:12],
            "typed_schema": "tushare_financials",
        }

    def _augment_tushare_dividend_result(self, result: ToolResult, arguments: Dict[str, Any]) -> None:
        if result.status != "success":
            return
        table = result.tables[0] if result.tables else {}
        rows = table.get("rows") if isinstance(table, dict) else []
        columns = table.get("columns") if isinstance(table, dict) else []
        if not isinstance(rows, list) or not rows:
            return
        available_fields = self._available_schema_fields(
            self.TUSHARE_DIVIDEND_FIELD_SCHEMA,
            rows,
            columns if isinstance(columns, list) else [],
        )
        field_schema = [
            self._compact_schema_entry(field, self.TUSHARE_DIVIDEND_FIELD_SCHEMA[field])
            for field in available_fields
            if field in self.TUSHARE_DIVIDEND_FIELD_SCHEMA
        ]
        if not field_schema:
            return
        payload = {
            "field_schema": field_schema[:12],
            "usage_note": (
                "Dividend schema: match end_date/report period, div_proc implementation status, ex/record/pay dates, "
                "and per-share tax basis before using a dividend value. Raw CSV follows."
            ),
        }
        raw_observation = result.observation_text()
        result.observation = (
            json.dumps(payload, ensure_ascii=False, default=str)
            + ("\nraw_csv:\n" + raw_observation if raw_observation else "")
        )
        result.metadata = {
            **(result.metadata or {}),
            "field_schema": field_schema[:12],
            "typed_schema": "tushare_dividend",
        }

    def _available_schema_fields(
        self,
        schema: Dict[str, Dict[str, Any]],
        rows: List[Any],
        columns: List[Any],
    ) -> List[str]:
        available_keys = {self._normalize_field_key(column) for column in columns}
        for row in rows:
            if isinstance(row, dict):
                available_keys.update(self._normalize_field_key(key) for key in row.keys())
        fields: List[str] = []
        for field, meta in schema.items():
            if field.startswith("__"):
                source_fields = meta.get("source_fields") or []
                if all(self._normalize_field_key(source) in available_keys for source in source_fields):
                    fields.append(field)
                continue
            if self._normalize_field_key(field) in available_keys:
                fields.append(field)
        return fields

    @staticmethod
    def _compact_schema_entry(field: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "field": field,
            "label_zh": meta.get("label_zh") or field,
            "unit": meta.get("unit") or "",
        }
        if meta.get("expression"):
            entry["expression"] = meta["expression"]
        if meta.get("source_fields"):
            entry["source_fields"] = meta["source_fields"]
        if meta.get("note"):
            entry["note"] = meta["note"]
        return entry

    def _tushare_financial_request_text(self, arguments: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("indicator", "metric", "query", "statement_type", "fields"):
            value = arguments.get(key)
            if isinstance(value, (list, tuple, set)):
                parts.extend(str(item) for item in value)
            else:
                parts.append(str(value or ""))
        return " ".join(part for part in parts if part.strip())

    def _tushare_schema_field_scores(
        self,
        schema: Dict[str, Dict[str, Any]],
        request_text: str,
        available_fields: List[str],
    ) -> List[Tuple[str, float]]:
        normalized_request = self._normalize_lookup_text(request_text)
        if not normalized_request:
            return []
        available = set(available_fields)
        scores: List[Tuple[str, float]] = []
        for field, meta in schema.items():
            if field not in available:
                continue
            score = 0.0
            field_norm = self._normalize_lookup_text(field)
            label_norm = self._normalize_lookup_text(meta.get("label_zh") or "")
            if label_norm and label_norm == normalized_request:
                score = max(score, 12.0)
            if field_norm and field_norm in normalized_request:
                score = max(score, 10.0)
            if label_norm and label_norm in normalized_request:
                score = max(score, 9.0)
            for alias in meta.get("aliases") or []:
                alias_norm = self._normalize_lookup_text(alias)
                if not alias_norm:
                    continue
                if alias_norm == normalized_request:
                    score = max(score, 12.0)
                elif alias_norm in normalized_request:
                    score = max(score, 8.0)
                else:
                    overlap = self._token_overlap(alias_norm, normalized_request)
                    if overlap >= 0.75:
                        score = max(score, 5.0 + overlap)
            if score > 0:
                scores.append((field, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores

    @staticmethod
    def _normalize_field_key(value: Any) -> str:
        return re.sub(r"[^0-9a-z]+", "", str(value or "").lower())

    @classmethod
    def _get_record_value(cls, row: Dict[str, Any], field: str) -> Any:
        wanted = cls._normalize_field_key(field)
        for key, value in row.items():
            if cls._normalize_field_key(key) == wanted:
                return value
        return None

    def _derived_tushare_financial_value(self, field: str, row: Dict[str, Any]) -> Optional[float]:
        if field == "__NET_INTEREST_INCOME":
            int_income = self._coerce_financial_number(self._get_record_value(row, "INT_INCOME"))
            int_exp = self._coerce_financial_number(self._get_record_value(row, "INT_EXP"))
            if int_income is None or int_exp is None:
                return None
            return int_income - int_exp
        return None

    def _semantic_candidate_from_value(
        self,
        field: str,
        meta: Dict[str, Any],
        value: float,
        row: Dict[str, Any],
        endpoint: str,
        *,
        match_score: float,
        source_fields: Any = None,
    ) -> Dict[str, Any]:
        unit = str(meta.get("unit") or "")
        candidate: Dict[str, Any] = {
            "field": field,
            "label_zh": meta.get("label_zh") or field,
            "period": str(self._get_record_value(row, "end_date") or self._get_record_value(row, "END_DATE") or ""),
            "ann_date": str(self._get_record_value(row, "ann_date") or self._get_record_value(row, "ANN_DATE") or ""),
            "value_raw": value,
            "unit_raw": unit,
            "source_endpoint": endpoint,
            "match_score": round(match_score, 3),
        }
        if unit == "元":
            candidate["converted_values"] = {
                "元": round(value, 6),
                "万元": round(value / 10_000.0, 6),
                "亿元": round(value / 100_000_000.0, 6),
            }
        if source_fields:
            candidate["source_fields"] = source_fields
        if meta.get("expression"):
            candidate["expression"] = meta["expression"]
        if meta.get("note"):
            candidate["note"] = meta["note"]
        return candidate

    def _tushare_row_rank(self, row: Dict[str, Any], want_period: str) -> Tuple[int, int, str, str]:
        end_date = str(self._get_record_value(row, "end_date") or self._get_record_value(row, "END_DATE") or "")
        ann_date = str(self._get_record_value(row, "ann_date") or self._get_record_value(row, "ANN_DATE") or "")
        period_match = bool(want_period and (end_date == want_period or end_date.startswith(want_period[:4])))
        annual = end_date.endswith("1231") or end_date.endswith("12-31")
        return (1 if period_match else 0, 1 if annual else 0, end_date, ann_date)

    def _tushare_dated_params(self, arguments: Dict[str, Any], *, ts_code: str = "") -> Tuple[Dict[str, Any], str, str]:
        start_date = format_tushare_date(arguments.get("start_date"), is_end=False)
        end_date = format_tushare_date(arguments.get("end_date"), is_end=True)
        trade_date = format_tushare_date(arguments.get("trade_date") or arguments.get("date"), is_end=True)
        params = {
            "_fire_tushare_backend": self._tushare_backend_from_arguments(arguments),
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "trade_date": trade_date,
            "fields": self._tushare_fields(arguments),
        }
        return params, start_date, end_date

    def _tushare_backend_from_arguments(self, arguments: Dict[str, Any]) -> str:
        return "tushare_http"

    @staticmethod
    def _tushare_fields(arguments: Dict[str, Any]) -> Any:
        fields = arguments.get("fields")
        if isinstance(fields, (list, tuple, set)):
            return ",".join(str(field).strip() for field in fields if str(field).strip())
        return fields

    @staticmethod
    def _tushare_financial_endpoint(arguments: Dict[str, Any]) -> str:
        statement = str(arguments.get("statement_type") or arguments.get("indicator") or "indicator").strip().lower()
        key = re.sub(r"[^a-z]+", "_", statement).strip("_")
        mapping = {
            "income": "income",
            "income_statement": "income",
            "profit": "income",
            "profit_statement": "income",
            "balance": "balancesheet",
            "balance_sheet": "balancesheet",
            "balancesheet": "balancesheet",
            "cash": "cashflow",
            "cashflow": "cashflow",
            "cash_flow": "cashflow",
            "cash_flow_statement": "cashflow",
            "indicator": "fina_indicator",
            "indicators": "fina_indicator",
            "financial_indicator": "fina_indicator",
            "fina_indicator": "fina_indicator",
        }
        return mapping.get(key, MarketDataTool._infer_tushare_financial_endpoint(arguments))

    @staticmethod
    def _infer_tushare_financial_endpoint(arguments: Dict[str, Any]) -> str:
        request_text = MarketDataTool._normalize_lookup_text(
            " ".join(
                str(arguments.get(key) or "")
                for key in ("indicator", "metric", "query", "fields", "statement_type")
            )
        )
        if not request_text:
            return ""
        for endpoint, schema in MarketDataTool.TUSHARE_FINANCIAL_FIELD_SCHEMA.items():
            for field, meta in schema.items():
                candidates = [field, meta.get("label_zh") or "", *(meta.get("aliases") or [])]
                for candidate in candidates:
                    normalized = MarketDataTool._normalize_lookup_text(candidate)
                    if normalized and normalized in request_text:
                        return endpoint
        return ""

    @staticmethod
    def _tushare_month(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if re.fullmatch(r"\d{6}", text):
            return text
        formatted = format_tushare_date(text, is_end=True)
        return formatted[:6] if formatted else ""

    @staticmethod
    def _tushare_quarter(value: Any) -> str:
        text = str(value or "").strip().upper()
        if re.fullmatch(r"\d{4}Q[1-4]", text):
            return text
        match = re.fullmatch(r"(\d{4})[-_/ ]?Q?([1-4])", text)
        if match and "Q" in text:
            return f"{match.group(1)}Q{match.group(2)}"
        formatted = format_tushare_period(text)
        if formatted:
            month = formatted[4:6]
            quarter = {"03": "1", "06": "2", "09": "3", "12": "4"}.get(month)
            if quarter:
                return f"{formatted[:4]}Q{quarter}"
        return ""

    @staticmethod
    def _normalize_tushare_daily_record(row: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(row)
        trade_date = str(record.get("trade_date") or "").strip()
        if trade_date and re.fullmatch(r"\d{8}", trade_date):
            record.setdefault("date", f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}")
        if record.get("ts_code") is not None:
            record.setdefault("symbol", record.get("ts_code"))
        if "vol" in record and "volume" not in record:
            record["volume"] = record.get("vol")
        if "open" in record:
            record.setdefault("raw_unadjusted_open", record.get("open"))
        if "close" in record:
            record.setdefault("raw_unadjusted_close", record.get("close"))
        record.setdefault("adjust_basis", "raw_unadjusted")
        record.setdefault("raw_unadjusted_available", True)
        return record

    @staticmethod
    def _record_columns(records: List[Dict[str, Any]], preferred: Optional[List[str]]) -> List[str]:
        seen: Dict[str, None] = {}
        for column in preferred or []:
            if any(column in record and record.get(column) is not None for record in records):
                seen.setdefault(str(column), None)
        for record in records:
            for key, value in record.items():
                if value is not None:
                    seen.setdefault(str(key), None)
        return list(seen)

    def _tushare_symbol_error(self, action: str, arguments: Dict[str, Any]) -> ToolResult:
        return self._structured_error(
            "tushare_http",
            action,
            "Tushare China A-share actions require a ticker like 600519.SH, 000001.SZ, BJ430047, or 600519.",
            metadata={"requested_ticker": self._market_symbol(arguments)},
        )

    def _equity_daily_basic_needs_price_history(self, arguments: Dict[str, Any]) -> bool:
        fields = self._dataframe_field_names(arguments.get("fields"))
        request_text = " ".join(
            str(arguments.get(key) or "")
            for key in ("query", "metric", "indicator")
        ).lower()
        price_history_fields = {
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "pct_change",
            "daily_return",
            "return",
            "vol",
            "volume",
            "amount",
        }
        if fields & price_history_fields:
            return True
        return any(token in request_text for token in ("pct_chg", "pct change", "涨跌幅", "日涨幅", "成交量", "成交额"))

    def _forward_price_history(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        mapped = self._map_market_action("price_history", dict(arguments or {}))
        provider = self._provider_hint(arguments)
        yahoo_symbol = self._resolve_yahoo_symbol(arguments)
        unsupported = self._unsupported_price_request(arguments, provider, yahoo_symbol)
        if unsupported:
            return unsupported
        should_try_tushare_first = provider == "tushare_http" or (
            provider in {"auto", ""}
            and self._looks_like_cn_a_share_price_request(arguments)
        )
        if should_try_tushare_first:
            tushare_result = self._fetch_tushare_equity_price_history("price_history", arguments, mapped=mapped, timeout=timeout)
            return tushare_result
        should_try_yahoo_first = provider in {"yahoo", "yfinance"} or (
            provider in {"auto", ""}
            and yahoo_symbol
            and self._is_global_price_request(arguments)
        )

        akshare_error = ""
        if should_try_yahoo_first:
            yahoo = self._fetch_yahoo_history(yahoo_symbol, mapped, arguments, timeout=timeout)
            if yahoo.status == "success" and yahoo.observation_text():
                return yahoo
            akshare_error = yahoo.error or ""

        akshare_result = super().forward("price_history", mapped, timeout=timeout)
        akshare_result.tool_family = self.name
        akshare_result.action = "price_history"
        if akshare_result.status == "success" and akshare_result.observation_text():
            return akshare_result

        if not should_try_yahoo_first and yahoo_symbol:
            yahoo = self._fetch_yahoo_history(yahoo_symbol, mapped, arguments, timeout=timeout)
            if yahoo.status == "success" and yahoo.observation_text():
                yahoo.metadata["fallback_from"] = "akshare"
                yahoo.metadata["akshare_error"] = akshare_result.error
                return yahoo
            akshare_result.metadata = {**(akshare_result.metadata or {}), "yahoo_error": yahoo.error}
        if akshare_error:
            akshare_result.metadata = {**(akshare_result.metadata or {}), "first_yahoo_error": akshare_error}
        return akshare_result

    def _looks_like_cn_a_share_price_request(self, arguments: Dict[str, Any]) -> bool:
        asset = str(arguments.get("asset_class") or "").strip().lower()
        market = str(arguments.get("market") or "").strip().lower()
        if asset in {"index", "global_index", "fund", "etf", "etf_cn", "futures", "future", "commodity", "commodities", "option", "options"}:
            return False
        if market in {"index", "cn_index", "china_index", "fund", "etf", "etf_cn", "futures", "commodity", "commodities", "option", "options"}:
            return False
        if not self._tushare_adjust_is_supported(arguments):
            return False
        ticker = self._market_symbol(arguments).upper().strip()
        return bool(normalize_tushare_ts_code(ticker))

    @staticmethod
    def _tushare_adjust_is_supported(arguments: Dict[str, Any]) -> bool:
        adjust = str(arguments.get("adjust") or "").strip().lower().replace("-", "_")
        return adjust in {"", "none", "no", "false", "unadjusted", "non_adjusted", "不复权"}

    def _forward_macro_series(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        series = str(arguments.get("series") or arguments.get("function") or "").strip()
        query = str(arguments.get("query") or arguments.get("country") or "").strip()
        limit = self._positive_int(arguments.get("limit"), 20)

        if provider == "iea":
            return self._fetch_iea_series(arguments, limit=limit, timeout=timeout)

        if provider == "oecd":
            return self._fetch_oecd_series(arguments, limit=limit, timeout=timeout)

        if provider == "alfred":
            fred_id = self._resolve_fred_series(series)
            if not fred_id:
                return self._structured_error("alfred", "macro_series", f"Could not resolve ALFRED/FRED series {series!r}.")
            if not self._has_vintage_request(arguments):
                return self._structured_error(
                    "alfred",
                    "macro_series",
                    "ALFRED provider requires vintage_date or realtime_start/realtime_end; it will not fall back to current FRED data.",
                    metadata={"series": fred_id},
                )
            return self._fetch_fred_series(fred_id, arguments, limit=limit, timeout=timeout)

        if provider in {"fred", "auto", ""}:
            fred_id = self._resolve_fred_series(series)
            if fred_id:
                fred_result = self._fetch_fred_series(fred_id, arguments, limit=limit, timeout=timeout)
                if fred_result.status == "success" or provider == "fred":
                    return fred_result
                yahoo_symbol = self._resolve_yahoo_symbol({"ticker": series, **arguments})
                if yahoo_symbol and self._looks_like_market_series(series):
                    mapped = {
                        "ticker": series,
                        "start_date": self._coerce_date(arguments.get("start_date"), default="1900-01-01", is_end=False),
                        "end_date": self._coerce_date(arguments.get("end_date"), default=datetime.now(timezone.utc).date().isoformat(), is_end=True),
                        "limit": limit,
                    }
                    yahoo_result = self._fetch_yahoo_history(yahoo_symbol, mapped, arguments, timeout=timeout, action="macro_series")
                    if yahoo_result.status == "success":
                        yahoo_result.metadata["fallback_from"] = "fred"
                        yahoo_result.metadata["fred_error"] = fred_result.error
                        return yahoo_result
                return fred_result

        if provider in {"world_bank", "worldbank", "wb", "auto", ""}:
            wb = self._resolve_world_bank_series(series, query, arguments)
            if wb:
                country, indicator = wb
                return self._fetch_world_bank_series(country, indicator, arguments, limit=limit, timeout=timeout)

        if provider in {"yahoo", "yfinance", "auto", ""}:
            yahoo_symbol = self._resolve_yahoo_symbol({"ticker": series, **arguments})
            if yahoo_symbol and self._looks_like_market_series(series):
                mapped = {
                    "ticker": series,
                    "start_date": self._coerce_date(arguments.get("start_date"), default="1900-01-01", is_end=False),
                    "end_date": self._coerce_date(arguments.get("end_date"), default=datetime.now(timezone.utc).date().isoformat(), is_end=True),
                    "limit": limit,
                }
                return self._fetch_yahoo_history(yahoo_symbol, mapped, arguments, timeout=timeout, action="macro_series")

        if provider == "tushare_http":
            return self._structured_error(
                provider,
                "macro_series",
                "Use market_data.cn_macro_series for supported Tushare China macro endpoints; macro_series does not route to China macro endpoints implicitly.",
                metadata={"series": series, "recommended_action": "cn_macro_series"},
            )

        if provider == "akshare":
            function = self.akshare_macro_aliases.get(series, series)
            mapped = {
                "function": function,
                "query": query,
                "start_date": arguments.get("start_date"),
                "end_date": arguments.get("end_date"),
                "limit": limit,
                "source": "live",
            }
            result = super().forward("call", mapped, timeout=timeout)
            result.tool_family = self.name
            result.action = "macro_series"
            if function != series:
                result.metadata = {**(result.metadata or {}), "resolved_series": function, "requested_series": series}
            return result

        return self._structured_error(
            "market_router",
            "macro_series",
            (
                "macro_series(auto) could not resolve the request to a precise FRED, World Bank, or Yahoo market series. "
                "No AkShare or unofficial macro fallback is used in auto mode. Provide an explicit provider/action: "
                "provider='fred'/'alfred' for US macro, provider='world_bank' for country annual macro, "
                "provider='oecd' or provider='iea' with official dataset/key, trade_series for Comtrade/WITS, "
                "official_attachment_table for SAFE/NFRA attachments, cn_macro_series for supported Tushare China macro, "
                "or provider='akshare' only when a named AkShare macro function is intentionally required."
            ),
            metadata={
                "series": series,
                "query": query,
                "recommended_actions": [
                    "macro_series(provider='fred'|'alfred'|'world_bank'|'oecd'|'iea')",
                    "cn_macro_series",
                    "trade_series",
                    "official_attachment_table",
                    "macro_series(provider='akshare')",
                ],
            },
            confidence=0.2,
        )

    def _records_result(
        self,
        provider: str,
        action: str,
        records: List[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        error: Optional[str] = None,
        confidence: float = 0.86,
    ) -> ToolResult:
        result_metadata: Dict[str, Any] = {**(metadata or {}), "result_count": len(records)}
        if "execution_card" not in result_metadata:
            result_metadata["execution_card"] = self._records_execution_card(
                provider,
                action,
                records,
                metadata=result_metadata,
                error=error,
            )
        observation = _finance_agent_records_to_csv(records, columns=columns)
        if provider != "dataframe_query":
            card = result_metadata.get("execution_card")
            if isinstance(card, dict):
                card_text = json.dumps(card, ensure_ascii=False, default=str, separators=(",", ":"))
                if observation:
                    observation = f"EXECUTION_CARD {card_text}\n{observation}"
                elif not records:
                    observation = f"EXECUTION_CARD {card_text}"
        return ToolResult(
            self.name,
            provider,
            "success" if records else "error",
            action=action,
            observation=observation,
            tables=[{"columns": columns or list(records[0].keys()), "rows": records}] if records else [],
            error=error if not records else None,
            confidence=confidence if records else 0.25,
            paid=False,
            metadata=result_metadata,
        )

    def _records_execution_card(
        self,
        provider: str,
        action: str,
        records: List[Dict[str, Any]],
        *,
        metadata: Dict[str, Any],
        error: Optional[str],
    ) -> Dict[str, Any]:
        result_rows = len(records)
        status = "success" if result_rows else "no_rows_or_source_gap"
        guidance_only = action in {"capabilities", "source_catalog"}
        return self._dataframe_drop_empty(
            {
                "card_type": "market_data_execution_card",
                "card_version": "1",
                "tool": self.name,
                "provider": provider,
                "action": action,
                "status": status,
                "requested": self._execution_card_requested(metadata),
                "result_rows": result_rows,
                "filled_candidates": self._execution_card_filled_candidates(records),
                "missing_slots": self._execution_card_missing_slots(provider, records, metadata, error),
                "diagnostics": {
                    "error": compact_text(str(error or ""), 240) if not records and error else None,
                    "result_count": result_rows,
                    "function": metadata.get("function") or metadata.get("requested_function") or metadata.get("endpoint"),
                },
                "retry_same_args_allowed": False if not records else None,
                "can_answer_now": bool(result_rows) and not guidance_only,
                "next_action": (
                    "call_recommended_market_data_action"
                    if result_rows and guidance_only
                    else "use_rows_or_calculate"
                    if result_rows
                    else "resolve_parameters_or_use_alternate_source"
                ),
            }
        )

    @staticmethod
    def _execution_card_requested(metadata: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "ts_code",
            "index_code",
            "fund_code",
            "contract",
            "series",
            "series_id",
            "requested_ticker",
            "symbol",
            "query",
            "start_date",
            "end_date",
            "trade_date",
            "period",
            "fields",
            "params",
            "function",
            "endpoint",
        )
        return {
            key: metadata.get(key)
            for key in keys
            if metadata.get(key) not in (None, "", [], {})
        }

    def _execution_card_filled_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row in records[:12]:
            if not isinstance(row, dict):
                continue
            candidates.append(
                self._dataframe_drop_empty(
                    {
                        "entity": self._execution_card_first(row, "ts_code", "symbol", "ticker", "code", "index_code", "fund_code", "contract"),
                        "date": self._execution_card_first(row, "trade_date", "date", "period", "ann_date", "end_date", "actual_end_date"),
                        "source_fields": self._execution_card_value_fields(row),
                        "basis": {
                            "provider": row.get("provider"),
                            "currency": row.get("currency"),
                            "unit": row.get("unit"),
                        },
                    }
                )
            )
        return candidates

    @staticmethod
    def _execution_card_missing_slots(
        provider: str,
        records: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        error: Optional[str],
    ) -> List[Dict[str, Any]]:
        if records:
            return []
        requested = MarketDataTool._execution_card_requested(metadata)
        return [
            MarketDataTool._dataframe_drop_empty(
                {
                    "requested": requested,
                    "reason": compact_text(str(error or "no rows returned"), 240),
                    "provider": provider,
                }
            )
        ]

    @staticmethod
    def _execution_card_first(row: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = row.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    @staticmethod
    def _execution_card_value_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        identity_keys = {
            "ts_code",
            "symbol",
            "ticker",
            "code",
            "index_code",
            "fund_code",
            "contract",
            "trade_date",
            "date",
            "period",
            "ann_date",
            "end_date",
            "actual_end_date",
            "provider",
            "currency",
            "unit",
        }
        preferred = (
            "metric_value",
            "value",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "return_pct",
            "vol",
            "volume",
            "amount",
            "turnover_rate",
            "pe",
            "pb",
            "dv_ratio",
        )
        fields: Dict[str, Any] = {}
        for key in preferred:
            value = row.get(key)
            if value not in (None, "", [], {}):
                fields[key] = value
            if len(fields) >= 6:
                return fields
        for key, value in row.items():
            if key in identity_keys or value in (None, "", [], {}):
                continue
            fields[key] = value
            if len(fields) >= 6:
                break
        return fields

    def _augment_cn_financial_statement_result(
        self,
        result: ToolResult,
        arguments: Dict[str, Any],
        mapped: Dict[str, Any],
    ) -> None:
        """Add candidate evidence for China A-share financial statement rows.

        The raw AkShare table is preserved. This only makes common statement
        fields easier to audit; it does not choose a final answer.
        """

        function = str((result.metadata or {}).get("function") or mapped.get("function") or "")
        if function not in {
            "stock_profit_sheet_by_report_em",
            "stock_balance_sheet_by_report_em",
            "stock_cash_flow_sheet_by_report_em",
            "stock_financial_abstract_ths",
            "stock_financial_abstract",
        }:
            return
        if result.provider == "sec_xbrl":
            return
        table = result.tables[0] if result.tables else {}
        rows = table.get("rows") if isinstance(table, dict) else []
        columns = table.get("columns") if isinstance(table, dict) else []
        if not isinstance(rows, list) or not rows:
            return

        request_text = " ".join(
            str(arguments.get(key) or "")
            for key in ("indicator", "query", "statement_type", "period", "date", "year")
        )
        field_scores = self._cn_financial_field_scores(request_text, columns if isinstance(columns, list) else [])
        if not field_scores:
            return

        want_year = self._period_year(arguments)
        candidate_rows = [row for row in rows if isinstance(row, dict)]
        if want_year is not None:
            candidate_rows = [row for row in candidate_rows if self._report_period(row).startswith(str(want_year))]
        if not candidate_rows:
            result.metadata = {
                **(result.metadata or {}),
                "candidate_note": f"No candidate financial cells were emitted because the returned rows do not include requested_year={want_year}.",
            }
            return
        ranked_rows = sorted(candidate_rows, key=lambda row: self._cn_financial_row_rank(row, want_year), reverse=True)
        candidate_cells: List[Dict[str, Any]] = []
        nearby_candidates: List[Dict[str, Any]] = []
        for field, score in field_scores:
            meta = self.CN_FINANCIAL_FIELD_CATALOG.get(field, {})
            for row in ranked_rows[: max(3, len(ranked_rows))]:
                if field not in row:
                    continue
                raw_value = row.get(field)
                numeric = self._coerce_financial_number(raw_value)
                if numeric is None:
                    continue
                cell = {
                    "field": field,
                    "label_zh": meta.get("label_zh") or field,
                    "period": self._report_period(row),
                    "report_type": row.get("REPORT_TYPE"),
                    "report_date_name": row.get("REPORT_DATE_NAME"),
                    "value_raw": raw_value,
                    "unit_raw": self._cn_financial_unit(function, field),
                    "converted_values": self._converted_financial_values(numeric, function, field),
                    "source_function": function,
                    "match_score": round(score, 3),
                }
                if meta.get("note"):
                    cell["note"] = meta["note"]
                if score >= 5.0:
                    candidate_cells.append(cell)
                else:
                    nearby_candidates.append(cell)
                break

        if not candidate_cells and not nearby_candidates:
            return
        evidence = {
            "candidate_cells": candidate_cells[:12],
            "nearby_candidates": nearby_candidates[:8],
            "usage_note": (
                "Use a candidate only after checking field/label, period, report_type, "
                "unit_raw, and converted_values against the question. Raw AkShare CSV follows."
            ),
        }
        raw_observation = result.observation_text()
        result.observation = (
            json.dumps(evidence, ensure_ascii=False, default=str)
            + ("\nraw_csv:\n" + raw_observation if raw_observation else "")
        )
        result.metadata = {
            **(result.metadata or {}),
            "candidate_cells": candidate_cells[:12],
            "nearby_candidates": nearby_candidates[:8],
            "candidate_note": "Candidate cells are generic financial-statement evidence, not final answers.",
        }

    def _cn_financial_field_scores(self, request_text: str, columns: List[Any]) -> List[Tuple[str, float]]:
        available = {str(column) for column in columns}
        normalized_request = self._normalize_lookup_text(request_text)
        if not normalized_request:
            return []
        scores: List[Tuple[str, float]] = []
        for field, meta in self.CN_FINANCIAL_FIELD_CATALOG.items():
            if field not in available:
                continue
            score = 0.0
            field_norm = self._normalize_lookup_text(field)
            label_norm = self._normalize_lookup_text(meta.get("label_zh") or "")
            if field_norm and field_norm in normalized_request:
                score = max(score, 10.0)
            if label_norm and label_norm in normalized_request:
                score = max(score, 9.0)
            for alias in meta.get("aliases") or []:
                alias_norm = self._normalize_lookup_text(alias)
                if not alias_norm:
                    continue
                if alias_norm in normalized_request:
                    score = max(score, 8.0)
                else:
                    overlap = self._token_overlap(alias_norm, normalized_request)
                    if overlap >= 0.75:
                        score = max(score, 5.0 + overlap)
            if score > 0:
                scores.append((field, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores

    @staticmethod
    def _normalize_lookup_text(value: Any) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _token_overlap(needle: str, haystack: str) -> float:
        tokens = [token for token in re.split(r"[^0-9a-z\u4e00-\u9fff]+", needle.lower()) if token]
        if not tokens:
            return 0.0
        return sum(1 for token in tokens if token in haystack) / len(tokens)

    @staticmethod
    def _cn_financial_row_rank(row: Dict[str, Any], want_year: Optional[int]) -> Tuple[int, int, str]:
        report_date = str(row.get("REPORT_DATE") or row.get("报告期") or "")
        report_type = str(row.get("REPORT_TYPE") or row.get("REPORT_DATE_NAME") or "")
        year_match = bool(want_year and report_date.startswith(str(want_year)))
        annual = bool("年报" in report_type or "-12-31" in report_date or report_date.endswith("1231"))
        return (1 if year_match else 0, 1 if annual else 0, report_date)

    @staticmethod
    def _report_period(row: Dict[str, Any]) -> str:
        report_date = str(row.get("REPORT_DATE") or row.get("报告期") or "")
        if re.match(DATE_REGEX, report_date[:10]):
            return report_date[:10]
        return report_date

    @staticmethod
    def _cn_financial_unit(function: str, field: str) -> str:
        if function in {
            "stock_profit_sheet_by_report_em",
            "stock_balance_sheet_by_report_em",
            "stock_cash_flow_sheet_by_report_em",
        } and not field.endswith("_YOY"):
            return "元"
        return "source_scale"

    @classmethod
    def _coerce_financial_number(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
            if math.isnan(parsed) or math.isinf(parsed):
                return None
            return parsed
        except Exception:
            pass
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"nan", "none", "false"}:
            return None
        multiplier = 1.0
        if "万亿" in text:
            multiplier = 1_000_000_000_000.0
        elif "亿" in text:
            multiplier = 100_000_000.0
        elif "万" in text:
            multiplier = 10_000.0
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0)) * multiplier

    @staticmethod
    def _converted_financial_values(value: float, function: str, field: str) -> Dict[str, float]:
        if function not in {
            "stock_profit_sheet_by_report_em",
            "stock_balance_sheet_by_report_em",
            "stock_cash_flow_sheet_by_report_em",
        } or field.endswith("_YOY"):
            return {}
        return {
            "元": round(value, 6),
            "万元": round(value / 10_000.0, 6),
            "亿元": round(value / 100_000_000.0, 6),
        }

    def _forward_source_catalog(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        limit = self._positive_int(arguments.get("limit"), 50)
        if provider == "fred":
            return self._fetch_fred_catalog(arguments, limit=limit, timeout=timeout)
        if provider in {"world_bank", "worldbank", "wb"}:
            return self._fetch_world_bank_catalog(arguments, limit=limit, timeout=timeout)
        if provider == "iea":
            return self._fetch_iea_catalog(arguments, limit=limit, timeout=timeout)
        if provider == "oecd":
            return self._fetch_oecd_catalog(arguments, limit=limit, timeout=timeout)
        if provider == "comtrade":
            return self._fetch_comtrade_catalog(arguments, limit=limit, timeout=timeout)
        return self._structured_error(
            provider or "source_catalog",
            "source_catalog",
            "source_catalog currently supports provider='fred', 'world_bank', 'iea', 'oecd', or 'comtrade'.",
            metadata={
                "provider": provider,
                "supported_providers": ["fred", "world_bank", "iea", "oecd", "comtrade"],
                "recommended_typed_actions": [
                    "capabilities",
                    "equity_universe",
                    "index_basic",
                    "index_history",
                    "fund_basic",
                    "fund_history",
                    "cn_futures_basic",
                    "cn_futures_daily",
                    "cn_macro_series",
                ],
            },
        )

    def _fetch_fred_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        query = str(arguments.get("query") or arguments.get("series") or "").strip()
        query_key = self._fred_alias_key(query)
        records: List[Dict[str, Any]] = []
        seen: set = set()

        resolved = self.fred_series_aliases.get(query_key) if query_key else None
        if resolved and resolved in self.fred_series_catalog:
            records.append(self._fred_catalog_record(resolved, arguments))
            seen.add(resolved)

        for series_id, meta in self.fred_series_catalog.items():
            if series_id in seen:
                continue
            haystack = " ".join(str(meta.get(key) or "") for key in ("series_id", "series_title", "frequency", "units", "notes"))
            if query and not self._catalog_query_matches(query, haystack):
                continue
            records.append(self._fred_catalog_record(series_id, arguments))
            seen.add(series_id)
            if len(records) >= limit:
                break

        api_key = self._fred_api_key()
        if api_key and len(records) < limit and query:
            url = (
                "https://api.stlouisfed.org/fred/series/search"
                f"?search_text={quote(query, safe='')}&file_type=json&api_key={quote(api_key, safe='')}"
                f"&limit={max(1, min(limit, 100))}"
            )
            try:
                payload = self._market_http_json(url, timeout=timeout)
                for item in payload.get("seriess") or []:
                    series_id = str(item.get("id") or "").strip()
                    if not series_id or series_id in seen:
                        continue
                    call_example = {
                        "action": "macro_series",
                        "provider": "fred",
                        "series": series_id,
                    }
                    for key in ("date", "start_date", "end_date", "limit"):
                        if arguments.get(key) not in (None, ""):
                            call_example[key] = arguments.get(key)
                    records.append(
                        {
                            "provider": "fred",
                            "series_id": series_id,
                            "series_title": item.get("title"),
                            "frequency": item.get("frequency"),
                            "units": item.get("units"),
                            "seasonal_adjustment": item.get("seasonal_adjustment"),
                            "observation_start": item.get("observation_start"),
                            "observation_end": item.get("observation_end"),
                            "popularity": item.get("popularity"),
                            "call_example": json.dumps(call_example, ensure_ascii=False),
                        }
                    )
                    seen.add(series_id)
                    if len(records) >= limit:
                        break
            except Exception:
                pass

        return self._records_result(
            "fred",
            "source_catalog",
            records[:limit],
            metadata={"provider": "fred", "query": query, "catalog": "fred_series"},
            error="FRED catalog returned no matching series.",
        )

    def _fred_catalog_record(self, series_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        meta = self.fred_series_catalog.get(series_id, {})
        call_example = {
            "action": "macro_series",
            "provider": "fred",
            "series": series_id,
        }
        for key in ("date", "start_date", "end_date", "limit"):
            if arguments.get(key) not in (None, ""):
                call_example[key] = arguments.get(key)
        return {
            "provider": "fred",
            "series_id": series_id,
            "series_title": meta.get("series_title") or series_id,
            "frequency": meta.get("frequency"),
            "units": meta.get("units"),
            "notes": meta.get("notes"),
            "call_example": json.dumps(call_example, ensure_ascii=False),
        }

    def _fetch_world_bank_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        query = str(arguments.get("query") or arguments.get("series") or arguments.get("indicator") or "").strip()
        country = self._country_iso3(arguments.get("country"))
        records: List[Dict[str, Any]] = []
        seen: set = set()

        resolved = self._resolve_world_bank_indicator(query)
        if resolved and resolved in self.world_bank_indicator_catalog:
            records.append(self._world_bank_catalog_record(resolved, arguments, country))
            seen.add(resolved)

        for indicator, meta in self.world_bank_indicator_catalog.items():
            if indicator in seen:
                continue
            haystack = " ".join(
                [
                    str(meta.get("indicator") or ""),
                    str(meta.get("indicator_name") or ""),
                    " ".join(str(alias) for alias in meta.get("aliases") or []),
                ]
            )
            if query and not self._catalog_query_matches(query, haystack):
                continue
            records.append(self._world_bank_catalog_record(indicator, arguments, country))
            seen.add(indicator)
            if len(records) >= limit:
                break

        return self._records_result(
            "world_bank",
            "source_catalog",
            records[:limit],
            metadata={"provider": "world_bank", "query": query, "country": country, "catalog": "world_bank_indicators"},
            error="World Bank catalog returned no matching indicators.",
        )

    def _world_bank_catalog_record(self, indicator: str, arguments: Dict[str, Any], country: str = "") -> Dict[str, Any]:
        meta = self.world_bank_indicator_catalog.get(indicator, {})
        call_example = {
            "action": "macro_series",
            "provider": "world_bank",
            "series": indicator,
            "indicator": indicator,
        }
        if country:
            call_example["country"] = country
        elif arguments.get("country") not in (None, ""):
            call_example["country"] = arguments.get("country")
        for key in ("year", "start_date", "end_date", "limit"):
            if arguments.get(key) not in (None, ""):
                call_example[key] = arguments.get(key)
        return {
            "provider": "world_bank",
            "indicator": indicator,
            "indicator_name": meta.get("indicator_name") or indicator,
            "aliases": "; ".join(str(alias) for alias in meta.get("aliases") or []),
            "country": country or arguments.get("country"),
            "call_example": json.dumps(call_example, ensure_ascii=False),
        }

    def _fetch_iea_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        country = self._country_iso3(arguments.get("country") or arguments.get("query"))
        if not country:
            return self._structured_error("iea", "source_catalog", "IEA catalog requires a country name or ISO3 country code.")
        url = f"https://api.iea.org/stats/indicators?countries={quote(country, safe='')}"
        try:
            payload = self._market_http_json(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("iea", "source_catalog", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        query = str(arguments.get("query") or "").strip().lower()
        records: List[Dict[str, Any]] = []
        for item in payload if isinstance(payload, list) else []:
            name = str(item.get("name") or "")
            indicator_id = str(item.get("id") or "")
            haystack = f"{indicator_id} {name} {' '.join(item.get('categories') or [])}".lower()
            if query and not self._catalog_query_matches(query, haystack):
                continue
            source = item.get("source") or {}
            call_example = {
                "action": "macro_series",
                "provider": "iea",
                "country": country,
                "indicator": indicator_id,
            }
            for key in ("product", "flow", "unit", "year", "start_date", "end_date"):
                if arguments.get(key) not in (None, ""):
                    call_example[key] = arguments.get(key)
            records.append(
                {
                    "provider": "iea",
                    "country": country,
                    "indicator": indicator_id,
                    "name": name,
                    "categories": "; ".join(item.get("categories") or []),
                    "source_text": source.get("text") if isinstance(source, dict) else None,
                    "source_href": source.get("href") if isinstance(source, dict) else None,
                    "call_example": json.dumps(call_example, ensure_ascii=False),
                }
            )
            if len(records) >= limit:
                break
        return self._records_result(
            "iea",
            "source_catalog",
            records,
            metadata={"url": url, "country": country, "query": query},
            error="IEA catalog returned no matching indicators for the requested country/query.",
        )

    def _fetch_iea_series(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        indicator = str(arguments.get("indicator") or arguments.get("series") or "").strip()
        country = self._country_iso3(arguments.get("country") or arguments.get("query"))
        if not indicator:
            return self._structured_error("iea", "macro_series", "IEA provider requires indicator, e.g. TESbySource.")
        if not country:
            return self._structured_error("iea", "macro_series", "IEA provider requires country as a name or ISO3 code.")
        url = f"https://api.iea.org/stats/indicator/{quote(indicator, safe='')}?countries={quote(country, safe='')}"
        try:
            payload = self._market_http_json(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("iea", "macro_series", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        if not isinstance(payload, list):
            return self._structured_error("iea", "macro_series", "IEA returned a non-list payload.", metadata={"url": url})

        start_year, end_year = self._year_bounds(arguments)
        product_filter = str(arguments.get("product") or "").strip().upper()
        flow_filter = str(arguments.get("flow") or "").strip().upper()
        unit_filter = str(arguments.get("unit") or "").strip().lower()
        records: List[Dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            year_text = str(row.get("year") or "")
            year_num = int(year_text) if year_text.isdigit() else None
            if start_year and (year_num is None or year_num < start_year):
                continue
            if end_year and (year_num is None or year_num > end_year):
                continue
            if product_filter and not self._dimension_filter_matches(
                product_filter,
                row.get("product"),
                row.get("productLabel"),
                row.get("seriesLabel"),
            ):
                continue
            if flow_filter and not self._dimension_filter_matches(flow_filter, row.get("flow"), row.get("flowLabel")):
                continue
            if unit_filter and unit_filter not in str(row.get("units") or "").lower():
                continue
            records.append(
                {
                    "year": row.get("year"),
                    "country": row.get("country"),
                    "short": row.get("short"),
                    "indicator": indicator,
                    "flow": row.get("flow"),
                    "flow_label": row.get("flowLabel"),
                    "product": row.get("product"),
                    "product_label": row.get("productLabel"),
                    "series_label": row.get("seriesLabel"),
                    "value": row.get("value"),
                    "units": row.get("units"),
                }
            )
        records.sort(key=lambda item: (str(item.get("year") or ""), str(item.get("flow") or ""), str(item.get("product") or "")))
        records = records[-limit:] if limit else records
        return self._records_result(
            "iea",
            "macro_series",
            records,
            metadata={
                "provider": "iea",
                "url": url,
                "country": country,
                "indicator": indicator,
                "filters": {"product": product_filter or None, "flow": flow_filter or None, "unit": unit_filter or None},
            },
            error="IEA returned no rows for the requested filters.",
        )

    def _fetch_oecd_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        url = "https://sdmx.oecd.org/public/rest/dataflow/all/all/latest"
        try:
            text = self._market_http_text(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("oecd", "source_catalog", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        query = str(arguments.get("query") or "").strip().lower()
        records: List[Dict[str, Any]] = []
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(text.encode("utf-8"))
            for elem in root.iter():
                if not str(elem.tag).endswith("Dataflow"):
                    continue
                dataflow_id = str(elem.attrib.get("id") or "")
                agency = str(elem.attrib.get("agencyID") or "")
                names = []
                for child in list(elem):
                    if str(child.tag).endswith("Name") and child.text:
                        names.append(child.text.strip())
                name = names[0] if names else ""
                haystack = f"{agency} {dataflow_id} {name}".lower()
                if query and not self._catalog_query_matches(query, haystack):
                    continue
                call_example = {
                    "action": "macro_series",
                    "provider": "oecd",
                    "dataflow": dataflow_id,
                    "key": "<SDMX dimension key required>",
                }
                if arguments.get("year") not in (None, ""):
                    call_example["year"] = arguments.get("year")
                records.append(
                    {
                        "provider": "oecd",
                        "agency": agency,
                        "dataflow": dataflow_id,
                        "name": name,
                        "requires_sdmx_key": True,
                        "call_example": json.dumps(call_example, ensure_ascii=False),
                    }
                )
                if len(records) >= limit:
                    break
        except Exception as exc:
            return self._structured_error("oecd", "source_catalog", f"Failed to parse OECD dataflow catalog: {type(exc).__name__}: {exc}", metadata={"url": url})
        return self._records_result(
            "oecd",
            "source_catalog",
            records,
            metadata={"url": url, "query": query},
            error="OECD dataflow catalog returned no matching dataflows for the requested query.",
        )

    def _fetch_oecd_series(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        dataflow = str(arguments.get("dataflow") or arguments.get("dataset") or arguments.get("series") or "").strip()
        key = str(arguments.get("key") or arguments.get("dimension_key") or arguments.get("query") or "").strip()
        if not dataflow:
            return self._structured_error("oecd", "macro_series", "OECD provider requires dataflow/dataset/series.")
        if not key:
            return self._structured_error(
                "oecd",
                "macro_series",
                "OECD provider requires a SDMX dimension key. Use source_catalog(provider='oecd', query=...) to find dataflows, then provide the official key; no automatic current-data fallback is used.",
                metadata={"dataflow": dataflow, "call_example": {"provider": "oecd", "dataflow": dataflow, "key": "<SDMX dimension key required>"}},
            )
        start_period = self._oecd_period(arguments.get("start_period") or arguments.get("start_date") or arguments.get("year"))
        end_period = self._oecd_period(arguments.get("end_period") or arguments.get("end_date") or arguments.get("year"))
        params = ["dimensionAtObservation=AllDimensions", "format=csvfilewithlabels"]
        if start_period:
            params.append(f"startPeriod={quote(start_period, safe='-')}")
        if end_period:
            params.append(f"endPeriod={quote(end_period, safe='-')}")
        url = (
            "https://sdmx.oecd.org/public/rest/data/"
            f"{quote(dataflow, safe=',@._-')}/{quote(key, safe='.,:@+_~-')}"
            f"?{'&'.join(params)}"
        )
        try:
            text = self._market_http_text(url, timeout=timeout)
            if text.lstrip().startswith("<"):
                raise ValueError("OECD returned XML/error content instead of CSV data.")
            import pandas as pd

            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            return self._structured_error("oecd", "macro_series", f"{type(exc).__name__}: {exc}", metadata={"url": url, "dataflow": dataflow, "key": key})
        query = str(arguments.get("filter") or "").strip()
        if query:
            try:
                mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False, na=False)).any(axis=1)
                df = df[mask]
            except Exception:
                pass
        if limit:
            df = df.tail(limit)
        records = df.where(df.notna(), None).to_dict(orient="records")
        return self._records_result(
            "oecd",
            "macro_series",
            records,
            metadata={"provider": "oecd", "url": url, "dataflow": dataflow, "key": key, "start_period": start_period, "end_period": end_period},
            error="OECD returned no rows for the requested SDMX key.",
        )

    def _fetch_comtrade_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        kind = str(arguments.get("kind") or arguments.get("catalog") or "reporter").strip().lower()
        if kind in {"commodity", "commodities", "product", "products", "hs", "cmd", "cmd_code"}:
            return self._fetch_comtrade_commodity_catalog(arguments, limit=limit, timeout=timeout)
        if kind in {"partner", "partners"}:
            url = "https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json"
            field_code, field_desc, field_iso3 = "PartnerCode", "PartnerDesc", "PartnerCodeIsoAlpha3"
        else:
            url = "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"
            field_code, field_desc, field_iso3 = "reporterCode", "reporterDesc", "reporterCodeIsoAlpha3"
        try:
            payload = self._market_http_json(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("comtrade", "source_catalog", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        query = str(arguments.get("query") or "").strip().lower()
        records: List[Dict[str, Any]] = []
        for row in (payload.get("results") if isinstance(payload, dict) else []) or []:
            haystack = f"{row.get(field_code)} {row.get(field_desc)} {row.get(field_iso3)} {row.get('text')}".lower()
            if query and not self._catalog_query_matches(query, haystack):
                continue
            records.append(
                {
                    "provider": "comtrade",
                    "catalog": kind,
                    "code": row.get(field_code),
                    "description": row.get(field_desc),
                    "iso3": row.get(field_iso3),
                    "is_group": row.get("isGroup"),
                    "call_field": "partner_code" if kind in {"partner", "partners"} else "reporter_code",
                }
            )
            if len(records) >= limit:
                break
        return self._records_result(
            "comtrade",
            "source_catalog",
            records,
            metadata={"url": url, "catalog": kind, "query": query},
            error="Comtrade catalog returned no matching area codes for the requested query.",
        )

    def _fetch_comtrade_commodity_catalog(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        query = str(arguments.get("query") or arguments.get("product") or arguments.get("cmd_code") or arguments.get("commodity_code") or "").strip()
        try:
            rows = self._comtrade_hs_rows(timeout=timeout)
        except Exception as exc:
            return self._structured_error("comtrade", "source_catalog", f"{type(exc).__name__}: {exc}", metadata={"catalog": "HS"})
        candidates = self._rank_comtrade_commodity_candidates(query, rows)
        records = []
        for row, score in candidates[:limit]:
            code = str(row.get("id") or "")
            call_example = {
                "action": "trade_series",
                "provider": "comtrade",
                "cmd_code": code,
                "reporter": arguments.get("reporter") or "<reporter>",
                "partner": arguments.get("partner") or "World",
                "flow": arguments.get("flow") or "<Export|Import>",
                "year": arguments.get("year") or arguments.get("period") or "<year>",
            }
            records.append(
                {
                    "provider": "comtrade",
                    "catalog": "HS",
                    "cmd_code": code,
                    "description": row.get("text"),
                    "parent": row.get("parent"),
                    "is_leaf": row.get("isLeaf"),
                    "aggregate_level": row.get("aggrLevel"),
                    "standard_unit": row.get("standardUnitAbbr"),
                    "match_score": round(score, 3),
                    "call_example": json.dumps(call_example, ensure_ascii=False),
                }
            )
        return self._records_result(
            "comtrade",
            "source_catalog",
            records,
            metadata={"url": "https://comtradeapi.un.org/files/v1/app/reference/HS.json", "catalog": "HS", "query": query},
            error="Comtrade HS catalog returned no matching commodity codes.",
        )

    def _forward_trade_series(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        limit = self._positive_int(arguments.get("limit"), 100)
        if provider == "comtrade":
            return self._fetch_comtrade_series(arguments, limit=limit, timeout=timeout)
        if provider == "wits":
            return self._fetch_wits_series(arguments, limit=limit, timeout=timeout)
        return self._structured_error(provider or "trade_series", "trade_series", "trade_series requires provider='comtrade' or provider='wits'.")

    def _fetch_comtrade_series(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        reporter = self._comtrade_area_code(arguments.get("reporter_code") or arguments.get("reporter"), "reporter", timeout=timeout)
        partner = self._comtrade_area_code(arguments.get("partner_code") or arguments.get("partner") or "0", "partner", timeout=timeout)
        period = str(arguments.get("period") or arguments.get("year") or "").strip()
        cmd_raw = arguments.get("cmd_code") or arguments.get("commodity_code") or arguments.get("product") or ""
        cmd_code, cmd_candidates = self._comtrade_commodity_code(cmd_raw, timeout=timeout)
        flow_code = self._comtrade_flow_code(arguments.get("flow_code") or arguments.get("flow"))
        if not reporter:
            return self._structured_error("comtrade", "trade_series", "Comtrade requires reporter/reporter_code.")
        if not period:
            return self._structured_error("comtrade", "trade_series", "Comtrade requires period/year.")
        if not cmd_code:
            preview = "; ".join(f"{item.get('id')}: {item.get('text')}" for item in cmd_candidates[:5])
            return self._structured_error(
                "comtrade",
                "trade_series",
                "Comtrade requires an unambiguous cmd_code/commodity_code/product. "
                + (f"Candidate HS codes: {preview}" if preview else "Use source_catalog(provider='comtrade', kind='commodity', query=...) to find official HS codes."),
                metadata={"commodity_query": str(cmd_raw), "candidates": cmd_candidates[:10]},
            )
        classification = str(arguments.get("classification") or "HS").strip().upper()
        type_code = str(arguments.get("type_code") or "C").strip().upper()
        freq_code = str(arguments.get("freq_code") or "A").strip().upper()
        endpoint = "preview"
        api_key = os.getenv("COMTRADE_API_KEY") or os.getenv("FIRE_AGENT_COMTRADE_API_KEY")
        if api_key:
            endpoint = "get"
        url = (
            f"https://comtradeapi.un.org/{'data' if api_key else 'public'}/v1/{endpoint}/"
            f"{quote(type_code, safe='')}/{quote(freq_code, safe='')}/{quote(classification, safe='')}"
            f"?reporterCode={quote(reporter, safe='')}&period={quote(period, safe='')}"
            f"&cmdCode={quote(cmd_code, safe='')}&flowCode={quote(flow_code, safe='')}"
            f"&partnerCode={quote(partner, safe='')}&includeDesc=true"
        )
        try:
            payload = self._market_http_json_with_headers(url, timeout=timeout, headers=self._comtrade_headers(api_key))
        except Exception as exc:
            return self._structured_error("comtrade", "trade_series", f"{type(exc).__name__}: {exc}", metadata={"url": self._redact_subscription_key(url), "endpoint": endpoint})
        rows = payload.get("data") if isinstance(payload, dict) else []
        records: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            records.append(
                {
                    "typeCode": row.get("typeCode"),
                    "freqCode": row.get("freqCode"),
                    "period": row.get("period"),
                    "refYear": row.get("refYear"),
                    "reporterCode": row.get("reporterCode"),
                    "reporterISO": row.get("reporterISO"),
                    "reporterDesc": row.get("reporterDesc"),
                    "flowCode": row.get("flowCode"),
                    "flowDesc": row.get("flowDesc"),
                    "partnerCode": row.get("partnerCode"),
                    "partnerISO": row.get("partnerISO"),
                    "partnerDesc": row.get("partnerDesc"),
                    "cmdCode": row.get("cmdCode"),
                    "cmdDesc": row.get("cmdDesc"),
                    "qtyUnitAbbr": row.get("qtyUnitAbbr"),
                    "qty": row.get("qty"),
                    "netWgt": row.get("netWgt"),
                    "grossWgt": row.get("grossWgt"),
                    "primaryValue": row.get("primaryValue"),
                    "isAggregate": row.get("isAggregate"),
                    "customsDesc": row.get("customsDesc"),
                    "motDesc": row.get("motDesc"),
                }
            )
            if len(records) >= limit:
                break
        return self._records_result(
            "comtrade",
            "trade_series",
            records,
            metadata={
                "provider": "comtrade",
                "url": self._redact_subscription_key(url),
                "endpoint": endpoint,
                "count": payload.get("count") if isinstance(payload, dict) else None,
                "reporterCode": reporter,
                "partnerCode": partner,
                "flowCode": flow_code,
                "cmdCode": cmd_code,
            },
            error="Comtrade returned no rows for the requested parameters.",
        )

    def _fetch_wits_series(self, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        reporter = self._country_iso3(arguments.get("reporter") or arguments.get("reporter_code"))
        partner = self._country_iso3(arguments.get("partner") or arguments.get("partner_code") or "WLD") or "WLD"
        year = str(arguments.get("year") or arguments.get("period") or "all").strip()
        product_raw = arguments.get("product") or arguments.get("cmd_code") or arguments.get("commodity_code") or ""
        product, product_candidates = self._comtrade_commodity_code(product_raw, timeout=timeout)
        flow = str(arguments.get("flow") or "Export").strip().title()
        if not reporter or not product:
            preview = "; ".join(f"{item.get('id')}: {item.get('text')}" for item in product_candidates[:5])
            return self._structured_error(
                "wits",
                "trade_series",
                "WITS requires reporter and an unambiguous product/cmd_code. "
                + (f"Candidate HS codes: {preview}" if preview else "Use source_catalog(provider='comtrade', kind='commodity', query=...) to find official HS codes."),
                metadata={"product_query": str(product_raw), "candidates": product_candidates[:10]},
            )
        url = (
            "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/trn/"
            f"reporter/{quote(reporter, safe='')}/year/{quote(year, safe='')}/"
            f"tradeflow/{quote(flow, safe='')}/partner/{quote(partner, safe='')}/product/{quote(product, safe='')}"
        )
        try:
            text = self._market_http_text(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("wits", "trade_series", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        records = [{"provider": "wits", "url": url, "raw_xml_preview": text[:4000]}]
        return self._records_result("wits", "trade_series", records[:limit], metadata={"provider": "wits", "url": url, "note": "WITS SDMX XML preview; use Comtrade for HS quantity/value rows when possible."}, confidence=0.55)

    def _forward_futures_contract_history(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        if provider not in {"auto", "", "cme", "cbot", "yahoo", "yfinance", "yahoo_contract"}:
            return self._structured_error(provider, "futures_contract_history", "Only Yahoo-backed specific futures contract history is implemented; continuous contract fallback is disabled.")
        symbol = self._resolve_specific_futures_symbol(arguments)
        if not symbol:
            return self._structured_error(
                "yahoo_contract",
                "futures_contract_history",
                "Could not resolve a specific futures contract symbol. Provide contract like ZSN25.CBT or contract=ZSN25 with exchange=CBOT.",
            )
        if symbol.endswith("=F"):
            return self._structured_error("yahoo_contract", "futures_contract_history", f"Continuous futures symbol {symbol!r} is not allowed for a specific contract request.")
        start_date = self._coerce_date(arguments.get("start_date") or arguments.get("date"), default="", is_end=False)
        end_date = self._coerce_date(arguments.get("end_date") or arguments.get("date"), default=start_date, is_end=True)
        if not start_date or not end_date:
            return self._structured_error("yahoo_contract", "futures_contract_history", "start_date/end_date or date is required in yyyy-mm-dd format.")
        mapped = {"ticker": symbol, "start_date": start_date, "end_date": end_date, "limit": arguments.get("limit")}
        result = self._fetch_yahoo_history(symbol, mapped, arguments, timeout=timeout, action="futures_contract_history")
        result.metadata = {
            **(result.metadata or {}),
            "provider": "yahoo_contract",
            "contract_symbol": symbol,
            "continuous_contract_fallback": False,
        }
        return result

    def _forward_metals_price(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        if provider == "lbma":
            return self._fetch_lbma_price(arguments, timeout=timeout)
        if provider == "lme":
            return self._structured_error(
                "lme",
                "metals_price",
                "LME historical official price data is not available through a stable public endpoint in this environment; no search-snippet fallback is used.",
            )
        return self._structured_error(provider or "metals_price", "metals_price", "metals_price requires provider='lbma' or provider='lme'.")

    def _fetch_lbma_price(self, arguments: Dict[str, Any], timeout: int) -> ToolResult:
        metal = re.sub(r"[^a-z]", "", str(arguments.get("metal") or "").lower())
        session = re.sub(r"[^a-z]", "", str(arguments.get("session") or "").lower())
        endpoint = ""
        if metal == "gold":
            endpoint = f"gold_{session if session in {'am', 'pm'} else 'pm'}"
        elif metal == "silver":
            endpoint = "silver"
        elif metal in {"platinum", "palladium"}:
            endpoint = f"{metal}_{session if session in {'am', 'pm'} else 'pm'}"
        if not endpoint:
            return self._structured_error("lbma", "metals_price", "LBMA metal must be gold, silver, platinum, or palladium.")
        url = f"https://prices.lbma.org.uk/json/{endpoint}.json"
        try:
            payload = self._market_http_json(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("lbma", "metals_price", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        start_date = self._coerce_date(arguments.get("start_date") or arguments.get("date"), default="", is_end=False)
        end_date = self._coerce_date(arguments.get("end_date") or arguments.get("date"), default=start_date, is_end=True)
        limit = self._positive_int(arguments.get("limit"), 20)
        records: List[Dict[str, Any]] = []
        for item in payload if isinstance(payload, list) else []:
            date = str(item.get("d") or "")
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            values = item.get("v") if isinstance(item.get("v"), list) else []
            records.append(
                {
                    "date": date,
                    "metal": metal,
                    "session": endpoint.replace(f"{metal}_", "") if "_" in endpoint else None,
                    "price_usd": self._list_get(values, 0),
                    "price_gbp": self._list_get(values, 1),
                    "price_eur": self._list_get(values, 2),
                    "is_cms_locked": item.get("is_cms_locked"),
                }
            )
        records = records[-limit:] if limit else records
        return self._records_result("lbma", "metals_price", records, metadata={"provider": "lbma", "url": url, "endpoint": endpoint}, error="LBMA returned no rows for the requested date range.")

    def _forward_official_attachment_table(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        provider = self._provider_hint(arguments)
        limit = self._positive_int(arguments.get("limit"), 200)
        header_rows = self._positive_int(arguments.get("header_rows"), 6)
        if provider == "safe":
            return self._fetch_safe_attachment_table(arguments, limit=limit, header_rows=header_rows, timeout=timeout)
        if provider == "nfra":
            return self._fetch_nfra_attachment_table(arguments, limit=limit, header_rows=header_rows, timeout=timeout)
        return self._structured_error(provider or "official_attachment_table", "official_attachment_table", "official_attachment_table requires provider='safe' or provider='nfra'.")

    def _fetch_safe_attachment_table(self, arguments: Dict[str, Any], limit: int, header_rows: int, timeout: int) -> ToolResult:
        url = str(arguments.get("url") or "").strip()
        if not url:
            return self._structured_error("safe", "official_attachment_table", "SAFE attachment table requires an official page URL.")
        try:
            html = self._market_http_text(url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("safe", "official_attachment_table", f"{type(exc).__name__}: {exc}", metadata={"url": url})
        attachment_urls = self._extract_excel_links(html, url)
        if not attachment_urls:
            return self._structured_error("safe", "official_attachment_table", "No Excel attachment link found on SAFE page.", metadata={"url": url})
        records, metadata = self._read_excel_attachments(attachment_urls, arguments, limit=limit, header_rows=header_rows, timeout=timeout)
        metadata.update({"provider": "safe", "page_url": url, "attachment_count": len(attachment_urls)})
        return self._records_result("safe", "official_attachment_table", records, metadata=metadata, error="SAFE attachment tables produced no rows.")

    def _fetch_nfra_attachment_table(self, arguments: Dict[str, Any], limit: int, header_rows: int, timeout: int) -> ToolResult:
        doc_id = str(arguments.get("doc_id") or "").strip()
        page_url = str(arguments.get("url") or "").strip()
        if not doc_id and page_url:
            parsed = urlparse(page_url)
            doc_id = (parse_qs(parsed.query).get("docId") or parse_qs(parsed.query).get("doc_id") or [""])[0]
        if not doc_id:
            return self._structured_error("nfra", "official_attachment_table", "NFRA attachment table requires doc_id or an ItemDetail URL containing docId.")
        base = "https://www.nfra.gov.cn"
        api_url = f"{base}/cbircweb/DocInfo/SelectByDocId?docId={quote(doc_id, safe='')}"
        try:
            payload = self._market_http_json(api_url, timeout=timeout)
        except Exception as exc:
            return self._structured_error("nfra", "official_attachment_table", f"{type(exc).__name__}: {exc}", metadata={"url": api_url, "doc_id": doc_id})
        data = payload.get("data") if isinstance(payload, dict) else {}
        attachments = data.get("attachmentInfoVOList") if isinstance(data, dict) else []
        attachment_urls: List[str] = []
        for item in attachments or []:
            if not isinstance(item, dict):
                continue
            href = str(item.get("urlOtherName") or item.get("fileUrl") or "").strip()
            if href.lower().endswith((".xls", ".xlsx")):
                attachment_urls.append(urljoin(base, href))
        records: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {
            "provider": "nfra",
            "doc_id": doc_id,
            "api_url": api_url,
            "doc_title": data.get("docTitle") if isinstance(data, dict) else None,
            "publish_date": data.get("publishDate") if isinstance(data, dict) else None,
            "doc_source": data.get("docSource") if isinstance(data, dict) else None,
            "attachment_count": len(attachment_urls),
        }
        if attachment_urls:
            records, excel_meta = self._read_excel_attachments(attachment_urls, arguments, limit=limit, header_rows=header_rows, timeout=timeout)
            metadata.update(excel_meta)
        elif isinstance(data, dict) and data.get("docClob"):
            records = self._read_html_tables(str(data.get("docClob") or ""), source_url=api_url, arguments=arguments, limit=limit, header_rows=header_rows)
        return self._records_result("nfra", "official_attachment_table", records, metadata=metadata, error="NFRA document produced no attachment or embedded table rows.")

    def _country_iso3(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        upper = re.sub(r"\s+", " ", text.upper().replace("_", " ")).strip()
        if re.fullmatch(r"[A-Z]{3}", upper):
            return upper
        alias = self.country_aliases.get(upper)
        if alias:
            return alias
        try:
            import pycountry  # type: ignore

            country = None
            if re.fullmatch(r"[A-Z]{2}", upper):
                country = pycountry.countries.get(alpha_2=upper)
            if country is None:
                country = pycountry.countries.lookup(text)
            alpha_3 = getattr(country, "alpha_3", "") if country is not None else ""
            return str(alpha_3 or "").upper()
        except Exception:
            return ""

    def _year_bounds(self, arguments: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        year = self._year_from_date(arguments.get("year") or arguments.get("date") or arguments.get("period"))
        start = self._year_from_date(arguments.get("start_date") or arguments.get("start_year")) or year
        end = self._year_from_date(arguments.get("end_date") or arguments.get("end_year")) or year or start
        return (int(start) if start else None, int(end) if end else None)

    @staticmethod
    def _oecd_period(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if re.fullmatch(r"\d{4}", text):
            return text
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return text
        if re.fullmatch(DATE_REGEX, text):
            return text[:7]
        match = re.search(r"((?:19|20)\d{2})(?:[-/.年](\d{1,2}))?", text)
        if not match:
            return ""
        if match.group(2):
            return f"{match.group(1)}-{int(match.group(2)):02d}"
        return match.group(1)

    @staticmethod
    def _market_http_text_with_headers(url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> str:
        request_headers = {"User-Agent": "Mozilla/5.0", **(headers or {})}
        try:
            import requests

            response = requests.get(url, timeout=timeout, headers=request_headers)
            response.raise_for_status()
            return response.text
        except ImportError:
            return http_get(url, headers=request_headers, timeout=timeout).text

    @classmethod
    def _market_http_json_with_headers(cls, url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> Any:
        return json.loads(cls._market_http_text_with_headers(url, timeout=timeout, headers=headers))

    @staticmethod
    def _market_http_bytes(url: str, timeout: int = 30) -> Tuple[bytes, str, str]:
        try:
            import requests

            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return response.content, str(response.headers.get("content-type") or ""), str(response.url)
        except ImportError:
            response = http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            content = response.text.encode("utf-8")
            return content, "", url

    @staticmethod
    def _comtrade_headers(api_key: str) -> Dict[str, str]:
        if not api_key:
            return {}
        return {"Ocp-Apim-Subscription-Key": api_key, "subscription-key": api_key}

    @staticmethod
    def _redact_subscription_key(url: str) -> str:
        return re.sub(r"(subscription[-_]key=)[^&]+", r"\1***", url, flags=re.I)

    @staticmethod
    def _catalog_query_matches(query: str, haystack: str) -> bool:
        query_text = str(query or "").strip().lower()
        if not query_text:
            return True
        haystack_text = str(haystack or "").lower()
        if query_text in haystack_text:
            return True
        tokens = [token for token in re.split(r"[^0-9a-z\u4e00-\u9fff]+", query_text) if len(token) >= 2]
        if not tokens:
            return False
        return sum(1 for token in tokens if token in haystack_text) >= max(1, min(len(tokens), 3))

    @staticmethod
    def _dimension_filter_matches(filter_text: str, *values: Any) -> bool:
        requested = str(filter_text or "").strip().upper()
        if not requested:
            return True
        normalized_requested = re.sub(r"[^0-9A-Z]+", "", requested)
        for value in values:
            candidate = str(value or "").strip().upper()
            if not candidate:
                continue
            if requested == candidate or requested in candidate or candidate in requested:
                return True
            normalized_candidate = re.sub(r"[^0-9A-Z]+", "", candidate)
            if normalized_requested and (
                normalized_requested == normalized_candidate
                or normalized_requested in normalized_candidate
                or normalized_candidate in normalized_requested
            ):
                return True
        return False

    def _comtrade_commodity_code(self, value: Any, timeout: int = 30) -> Tuple[str, List[Dict[str, Any]]]:
        text = str(value or "").strip()
        if not text:
            return "", []
        upper = text.upper()
        if upper in {"TOTAL", "ALL", "ALL COMMODITIES"}:
            return "TOTAL", []
        compact = re.sub(r"[^0-9A-Za-z]+", "", text)
        if compact.isdigit():
            return compact, []
        try:
            rows = self._comtrade_hs_rows(timeout=timeout)
        except Exception:
            return "", []
        candidates = self._rank_comtrade_commodity_candidates(text, rows)
        if not candidates:
            return "", []
        top_score = candidates[0][1]
        top_row = candidates[0][0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0
        if top_score >= 8.0 and (top_score - second_score >= 1.5 or second_score < 6.0):
            return str(top_row.get("id") or ""), []
        return "", [row for row, _score in candidates[:10]]

    @classmethod
    def _comtrade_hs_rows(cls, timeout: int = 30) -> List[Dict[str, Any]]:
        if cls.comtrade_commodity_cache:
            return cls.comtrade_commodity_cache
        url = "https://comtradeapi.un.org/files/v1/app/reference/HS.json"
        payload = cls._market_http_json(url, timeout=timeout)
        rows = payload.get("results") if isinstance(payload, dict) else []
        cls.comtrade_commodity_cache = [row for row in rows or [] if isinstance(row, dict)]
        return cls.comtrade_commodity_cache

    def _rank_comtrade_commodity_candidates(self, query: str, rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        query_text = str(query or "").strip()
        if not query_text:
            return [(row, 1.0) for row in rows[:50]]
        query_norm = re.sub(r"[^0-9a-z]+", " ", query_text.lower()).strip()
        query_tokens = [token for token in query_norm.split() if len(token) >= 2]
        numeric = re.sub(r"[^0-9]+", "", query_text)
        scored: List[Tuple[Dict[str, Any], float]] = []
        for row in rows:
            code = str(row.get("id") or "")
            text = str(row.get("text") or "")
            haystack = re.sub(r"[^0-9a-z]+", " ", f"{code} {text}".lower()).strip()
            score = 0.0
            if numeric and code == numeric:
                score += 12.0
            elif numeric and code.startswith(numeric):
                score += 5.0
            if query_norm and query_norm in haystack:
                score += 9.0
            if query_tokens:
                hits = sum(1 for token in query_tokens if token in haystack)
                score += (hits / len(query_tokens)) * 7.0
                if hits == len(query_tokens):
                    score += 2.0
            if str(row.get("isLeaf") or "") == "1":
                score += 0.5
            if score > 0:
                scored.append((row, score))
        scored.sort(key=lambda item: (-item[1], str(item[0].get("id") or "")))
        return scored

    def _comtrade_area_code(self, value: Any, kind: str, timeout: int = 30) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.isdigit():
            return str(int(text))
        if text.upper() in {"WORLD", "WLD"} and kind == "partner":
            return "0"
        cache = self.comtrade_partner_cache if kind == "partner" else self.comtrade_reporter_cache
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if normalized in cache:
            return cache[normalized]
        url = (
            "https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json"
            if kind == "partner"
            else "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"
        )
        try:
            payload = self._market_http_json(url, timeout=timeout)
        except Exception:
            return ""
        code_field = "PartnerCode" if kind == "partner" else "reporterCode"
        desc_field = "PartnerDesc" if kind == "partner" else "reporterDesc"
        iso2_field = "PartnerCodeIsoAlpha2" if kind == "partner" else "reporterCodeIsoAlpha2"
        iso3_field = "PartnerCodeIsoAlpha3" if kind == "partner" else "reporterCodeIsoAlpha3"
        for row in (payload.get("results") if isinstance(payload, dict) else []) or []:
            code = row.get(code_field)
            if code is None:
                continue
            keys = {
                str(row.get(desc_field) or ""),
                str(row.get("text") or ""),
                str(row.get(iso2_field) or ""),
                str(row.get(iso3_field) or ""),
            }
            for key in keys:
                norm_key = re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()
                if norm_key:
                    cache.setdefault(norm_key, str(code))
        return cache.get(normalized, "")

    @staticmethod
    def _comtrade_flow_code(value: Any) -> str:
        text = str(value or "X").strip()
        upper = text.upper()
        mapping = {
            "EXPORT": "X",
            "EXPORTS": "X",
            "X": "X",
            "IMPORT": "M",
            "IMPORTS": "M",
            "M": "M",
            "RE-EXPORT": "RX",
            "REEXPORT": "RX",
            "RE-EXPORTS": "RX",
            "RX": "RX",
            "RE-IMPORT": "RM",
            "REIMPORT": "RM",
            "RE-IMPORTS": "RM",
            "RM": "RM",
        }
        return mapping.get(upper, upper or "X")

    def _resolve_specific_futures_symbol(self, arguments: Dict[str, Any]) -> str:
        raw = str(arguments.get("contract") or arguments.get("ticker") or arguments.get("symbol") or "").strip().upper()
        if raw.endswith(("=F", ".CBT", ".NYM", ".CMX", ".CME")):
            return raw
        if raw and re.fullmatch(r"[A-Z]+?[FGHJKMNQUVXZ](?:\d{2}|\d{4})", raw):
            return raw + self._futures_exchange_suffix(raw)

        commodity = str(arguments.get("commodity") or "").strip().lower()
        root_map = {
            "soybean": "ZS",
            "soybeans": "ZS",
            "corn": "ZC",
            "wheat": "ZW",
            "wti": "CL",
            "crude": "CL",
            "crude oil": "CL",
            "brent": "BZ",
            "natural gas": "NG",
            "gold": "GC",
            "silver": "SI",
            "copper": "HG",
        }
        root = str(arguments.get("root") or root_map.get(commodity) or "").strip().upper()
        month_code = self._futures_month_code(arguments.get("contract_month") or arguments.get("month"))
        year_text = str(arguments.get("contract_year") or arguments.get("year") or "").strip()
        year_match = re.search(r"(?:19|20)?(\d{2})$", year_text)
        if root and month_code and year_match:
            symbol = f"{root}{month_code}{year_match.group(1)}"
            return symbol + self._futures_exchange_suffix(symbol)
        return ""

    @staticmethod
    def _futures_exchange_suffix(symbol: str) -> str:
        root_match = re.fullmatch(r"([A-Z]+?)[FGHJKMNQUVXZ](?:\d{2}|\d{4})", symbol)
        if not root_match:
            root_match = re.match(r"([A-Z]+)", symbol)
        root = root_match.group(1) if root_match else ""
        if root in {"ZS", "ZC", "ZW", "KE", "ZM", "ZL"}:
            return ".CBT"
        if root in {"CL", "BZ", "NG", "RB", "HO"}:
            return ".NYM"
        if root in {"GC", "SI", "HG", "PL", "PA"}:
            return ".CMX"
        return ""

    @staticmethod
    def _futures_month_code(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        if re.fullmatch(r"[fghjkmnquvxz]", text):
            return text.upper()
        mapping = {
            "1": "F", "jan": "F", "january": "F",
            "2": "G", "feb": "G", "february": "G",
            "3": "H", "mar": "H", "march": "H",
            "4": "J", "apr": "J", "april": "J",
            "5": "K", "may": "K",
            "6": "M", "jun": "M", "june": "M",
            "7": "N", "jul": "N", "july": "N",
            "8": "Q", "aug": "Q", "august": "Q",
            "9": "U", "sep": "U", "sept": "U", "september": "U",
            "10": "V", "oct": "V", "october": "V",
            "11": "X", "nov": "X", "november": "X",
            "12": "Z", "dec": "Z", "december": "Z",
        }
        return mapping.get(text, "")

    @staticmethod
    def _extract_excel_links(html: str, page_url: str) -> List[str]:
        links: List[str] = []
        for match in re.finditer(r"""href\s*=\s*["']([^"']+\.(?:xls|xlsx)(?:\?[^"']*)?)["']""", html or "", flags=re.I):
            links.append(urljoin(page_url, match.group(1)))
        seen: Dict[str, None] = {}
        return [link for link in links if not (link in seen or seen.setdefault(link, None))]

    def _read_excel_attachments(
        self,
        attachment_urls: List[str],
        arguments: Dict[str, Any],
        *,
        limit: int,
        header_rows: int,
        timeout: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for attachment_url in attachment_urls:
            try:
                content, content_type, final_url = self._market_http_bytes(attachment_url, timeout=timeout)
                rows = self._read_excel_bytes(
                    content,
                    source_url=final_url or attachment_url,
                    arguments=arguments,
                    limit=max(0, limit - len(records)),
                    header_rows=header_rows,
                )
                records.extend(rows)
            except Exception as exc:
                errors.append(f"{attachment_url}: {type(exc).__name__}: {exc}")
            if len(records) >= limit:
                break
        return records[:limit], {"attachment_urls": attachment_urls, "attachment_errors": errors}

    def _read_excel_bytes(
        self,
        content: bytes,
        *,
        source_url: str,
        arguments: Dict[str, Any],
        limit: int,
        header_rows: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        import pandas as pd

        query = str(arguments.get("query") or "").strip()
        excel = pd.ExcelFile(io.BytesIO(content))
        records: List[Dict[str, Any]] = []
        for sheet_name in excel.sheet_names:
            df = pd.read_excel(excel, sheet_name=sheet_name, header=None, dtype=object)
            records.extend(
                self._dataframe_grid_records(
                    df,
                    source_url=source_url,
                    sheet_name=str(sheet_name),
                    query=query,
                    limit=max(0, limit - len(records)),
                    header_rows=header_rows,
                )
            )
            if len(records) >= limit:
                break
        return records[:limit]

    def _read_html_tables(
        self,
        html: str,
        *,
        source_url: str,
        arguments: Dict[str, Any],
        limit: int,
        header_rows: int,
    ) -> List[Dict[str, Any]]:
        import pandas as pd

        query = str(arguments.get("query") or "").strip()
        tables = pd.read_html(io.StringIO(html))
        records: List[Dict[str, Any]] = []
        for idx, df in enumerate(tables):
            records.extend(
                self._dataframe_grid_records(
                    df,
                    source_url=source_url,
                    sheet_name=f"html_table_{idx}",
                    query=query,
                    limit=max(0, limit - len(records)),
                    header_rows=header_rows,
                )
            )
            if len(records) >= limit:
                break
        return records[:limit]

    def _dataframe_grid_records(
        self,
        df: Any,
        *,
        source_url: str,
        sheet_name: str,
        query: str,
        limit: int,
        header_rows: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        row_indices: List[int] = []
        header_limit = min(header_rows, len(df))
        row_indices.extend(range(header_limit))
        if query:
            query_lower = query.lower()
            for idx, row in df.iterrows():
                row_text = " | ".join(self._stringify_cell(value) for value in row.tolist()).lower()
                if query_lower in row_text and idx not in row_indices:
                    row_indices.append(int(idx))
        else:
            for idx in range(header_limit, len(df)):
                row_indices.append(int(idx))
                if len(row_indices) >= limit:
                    break
        records: List[Dict[str, Any]] = []
        for idx in row_indices:
            if len(records) >= limit:
                break
            row = df.iloc[idx].tolist()
            values = [self._stringify_cell(value) for value in row]
            while values and values[-1] == "":
                values.pop()
            record: Dict[str, Any] = {
                "source_url": source_url,
                "sheet": sheet_name,
                "row_index": idx,
                "row_text": " | ".join(value for value in values if value != ""),
            }
            for col_idx, value in enumerate(values):
                record[f"cell_{col_idx}"] = value
            records.append(record)
        return records

    @staticmethod
    def _stringify_cell(value: Any) -> str:
        if value is None:
            return ""
        try:
            import pandas as pd

            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    # Curated us-gaap concepts -> human label. Order matters: for each label the
    # first concept that has data for the requested period wins, so synonymous
    # tags (e.g. Revenues vs RevenueFromContractWithCustomerExcludingAssessedTax)
    # collapse to one standardized line item.
    SEC_FUNDAMENTAL_CONCEPTS: List[Tuple[str, str]] = [
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
        ("Revenues", "Revenue"),
        ("CostOfGoodsAndServicesSold", "Cost of revenue"),
        ("CostOfRevenue", "Cost of revenue"),
        ("GrossProfit", "Gross profit"),
        ("ResearchAndDevelopmentExpense", "R&D expense"),
        ("SellingGeneralAndAdministrativeExpense", "SG&A expense"),
        ("OperatingIncomeLoss", "Operating income"),
        ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "Pre-tax income"),
        ("NetIncomeLoss", "Net income"),
        ("EarningsPerShareBasic", "EPS (basic)"),
        ("EarningsPerShareDiluted", "EPS (diluted)"),
        ("WeightedAverageNumberOfSharesOutstandingBasic", "Weighted avg shares (basic)"),
        ("WeightedAverageNumberOfDilutedSharesOutstanding", "Weighted avg shares (diluted)"),
        ("Assets", "Total assets"),
        ("Liabilities", "Total liabilities"),
        ("StockholdersEquity", "Stockholders equity"),
        ("CashAndCashEquivalentsAtCarryingValue", "Cash and equivalents"),
        ("NetCashProvidedByUsedInOperatingActivities", "Cash flow from operations"),
        ("NetCashProvidedByUsedInInvestingActivities", "Cash flow from investing"),
        ("NetCashProvidedByUsedInFinancingActivities", "Cash flow from financing"),
        ("PaymentsToAcquirePropertyPlantAndEquipment", "Capital expenditures"),
        ("PaymentsOfDividendsCommonStock", "Dividends paid"),
        ("PaymentsOfDividends", "Dividends paid"),
        ("CommonStockDividendsPerShareDeclared", "Dividends per share"),
        ("CommonStockDividendsPerShareCashPaid", "Dividends per share"),
        ("LongTermDebtNoncurrent", "Long-term debt (noncurrent)"),
    ]
    # Friendly indicator aliases -> us-gaap concept, for when the model asks for a
    # specific line item via the ``indicator`` argument.
    SEC_INDICATOR_ALIASES: Dict[str, str] = {
        "netincome": "NetIncomeLoss",
        "netincomeloss": "NetIncomeLoss",
        "revenue": "Revenues",
        "revenues": "Revenues",
        "totalrevenue": "Revenues",
        "eps": "EarningsPerShareDiluted",
        "epsdiluted": "EarningsPerShareDiluted",
        "epsbasic": "EarningsPerShareBasic",
        "grossprofit": "GrossProfit",
        "operatingincome": "OperatingIncomeLoss",
        "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
        "capitalexpenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
        "cfo": "NetCashProvidedByUsedInOperatingActivities",
        "operatingcashflow": "NetCashProvidedByUsedInOperatingActivities",
        "dividends": "PaymentsOfDividendsCommonStock",
        "dividendspaid": "PaymentsOfDividendsCommonStock",
        "paymentsofdividends": "PaymentsOfDividends",
        "commonstockdividends": "PaymentsOfDividendsCommonStock",
        "dividendpershare": "CommonStockDividendsPerShareDeclared",
        "dps": "CommonStockDividendsPerShareDeclared",
        "assets": "Assets",
        "totalassets": "Assets",
        "liabilities": "Liabilities",
        "stockholdersequity": "StockholdersEquity",
        "equity": "StockholdersEquity",
        "longtermdebt": "LongTermDebtNoncurrent",
    }
    _sec_ticker_cik_cache: Dict[str, str] = {}

    def _maybe_sec_fundamentals(
        self,
        arguments: Dict[str, Any],
        timeout: int = 30,
    ) -> Optional[ToolResult]:
        """Standardized US fundamentals from SEC XBRL companyfacts.

        Returns a ToolResult when SEC should serve the request (explicit
        provider hint, or a US equity ticker under auto-routing) and the call
        succeeds. Returns ``None`` to fall through to the existing AkShare path
        when SEC is not applicable, or when an auto-routed SEC attempt fails (so
        behavior never regresses relative to before this source existed). An
        explicit ``provider=sec`` failure is surfaced as an error instead.
        """
        provider = str(arguments.get("provider") or arguments.get("source") or "").strip().lower().replace("-", "_")
        sec_requested = provider in {"sec", "sec_xbrl", "edgar", "companyfacts", "xbrl"}
        ticker = self._market_symbol(arguments)
        is_us = self._looks_like_us_equity_ticker(ticker)
        if not sec_requested and not is_us:
            return None
        if not ticker:
            return None

        result = self._fetch_sec_companyfacts(ticker, arguments, timeout=timeout)
        if result.status == "success":
            return result
        if sec_requested:
            return result
        # Auto-routed attempt failed: let AkShare path try (preserves old behavior).
        return None

    def _fetch_sec_companyfacts(
        self,
        ticker: str,
        arguments: Dict[str, Any],
        timeout: int = 30,
    ) -> ToolResult:
        ticker_up = ticker.strip().upper()
        try:
            cik = self._sec_ticker_to_cik(ticker_up, timeout=timeout)
        except Exception as exc:
            return self._sec_error(ticker_up, f"ticker->CIK lookup failed: {type(exc).__name__}: {exc}")
        if not cik:
            return self._sec_error(ticker_up, f"No SEC CIK mapping found for ticker {ticker_up!r}.")

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            payload = self._sec_http_json(url, timeout=timeout)
        except Exception as exc:
            return self._sec_error(ticker_up, f"companyfacts fetch failed: {type(exc).__name__}: {exc}", url=url, cik=cik)

        facts = ((payload or {}).get("facts") or {}).get("us-gaap") or {}
        if not facts:
            return self._sec_error(ticker_up, "companyfacts returned no us-gaap facts.", url=url, cik=cik)

        want_year = self._period_year(arguments)
        statement_type = str(arguments.get("statement_type") or "").lower()
        # ``indicator`` is a request for a SPECIFIC line item and must NOT be
        # folded into statement_type (doing so made e.g. indicator="DividendsPaid"
        # match no statement bucket and silently return nothing).
        indicator_raw = str(arguments.get("indicator") or "").strip()
        freq = "quarterly" if self._wants_quarterly(arguments) else "annual"
        forms = {"10-Q"} if freq == "quarterly" else {"10-K", "20-F", "40-F"}
        periods_per_label = self._positive_int(arguments.get("limit"), 0) or (1 if want_year else 4)

        # Resolve which us-gaap concepts to return. Honors a specific indicator
        # (curated label OR any concept the company actually reports), otherwise
        # the statement-type bundle. Keeping arbitrary-concept passthrough avoids
        # a hard-coded whitelist ceiling (e.g. InterestExpense, Inventory).
        concept_plan = self._build_concept_plan(indicator_raw, statement_type, facts)

        records: List[Dict[str, Any]] = []
        seen_labels: set = set()
        for concept, label in concept_plan:
            if label in seen_labels:
                continue
            concept_data = facts.get(concept)
            if not isinstance(concept_data, dict):
                continue
            units = concept_data.get("units") or {}
            entries = self._sec_select_entries(units, forms, want_year, periods_per_label)
            if not entries:
                continue
            seen_labels.add(label)
            for entry in entries:
                end = str(entry.get("end") or "")
                records.append(
                    {
                        "line_item": label,
                        # Authoritative period = calendar year of the data period_end.
                        # (SEC's raw `fy` can equal the filing year for calendar-year
                        # filers, so it is kept only as a secondary trace field.)
                        "period": end[:4],
                        "period_end": end,
                        "period_start": entry.get("start"),
                        "fiscal_period": entry.get("fp"),
                        "form": entry.get("form"),
                        "value": entry.get("val"),
                        "unit": entry.get("unit"),
                        "us_gaap_concept": concept,
                        "sec_fiscal_year": entry.get("fy"),
                    }
                )

        if not records:
            return self._sec_error(
                ticker_up,
                f"companyfacts had no curated line items for the requested period "
                f"(year={want_year}, freq={freq}).",
                url=url,
                cik=cik,
            )

        records.sort(key=lambda r: (str(r.get("line_item")), str(r.get("period_end") or "")))
        return ToolResult(
            self.name,
            "sec_xbrl",
            "success",
            action="financial_statement",
            observation=_finance_agent_records_to_csv(records),
            tables=[{"columns": list(records[0].keys()), "rows": records}],
            confidence=0.9,
            paid=False,
            metadata={
                "provider": "sec_xbrl",
                "source": "sec_companyfacts",
                "ticker": ticker_up,
                "cik": cik,
                "entity_name": (payload or {}).get("entityName"),
                "url": url,
                "frequency": freq,
                "requested_year": want_year,
                "requested_indicator": str(arguments.get("indicator") or "") or None,
                "line_items": sorted(seen_labels),
                "result_count": len(records),
                "note": (
                    "Standardized XBRL line items as reported to the SEC. 'value' is in the "
                    "stated 'unit' (USD, USD/shares, or shares) at full scale (not thousands/millions)."
                ),
            },
        )

    def _sec_error(self, ticker: str, message: str, url: str = "", cik: str = "") -> ToolResult:
        return ToolResult(
            self.name,
            "sec_xbrl",
            "error",
            action="financial_statement",
            error=(
                f"SEC XBRL companyfacts lookup failed for {ticker}: {message} "
                "For US filings you can also use sec_search/sec_reader on the filing text/tables."
            ),
            paid=False,
            confidence=0.15,
            metadata={"provider": "sec_xbrl", "ticker": ticker, "cik": cik, "url": url},
        )

    @classmethod
    def _sec_ticker_to_cik(cls, ticker: str, timeout: int = 30) -> str:
        if ticker in cls._sec_ticker_cik_cache:
            return cls._sec_ticker_cik_cache[ticker]
        payload = cls._sec_http_json("https://www.sec.gov/files/company_tickers.json", timeout=timeout)
        if isinstance(payload, dict):
            for entry in payload.values():
                if not isinstance(entry, dict):
                    continue
                sym = str(entry.get("ticker") or "").strip().upper()
                cik_raw = entry.get("cik_str")
                if sym and cik_raw is not None:
                    cls._sec_ticker_cik_cache[sym] = f"{int(cik_raw):010d}"
        return cls._sec_ticker_cik_cache.get(ticker, "")

    @staticmethod
    def _sec_select_entries(
        units: Dict[str, Any],
        forms: set,
        want_year: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        # Prefer USD, then USD/shares, then shares, then any other unit.
        unit_order = sorted(
            units.keys(),
            key=lambda u: (0 if u == "USD" else 1 if u == "USD/shares" else 2 if u == "shares" else 3, u),
        )

        def _fy_int(item: Dict[str, Any]) -> int:
            raw = str(item.get("fy") or "")
            return int(raw) if raw.isdigit() else 0

        for unit_name in unit_order:
            entries = units.get(unit_name) or []
            # Collapse to one row per DATA period (period_end, period_start). The
            # same period appears in multiple filings as a comparative; keep the
            # most recently filed value. CRITICAL: match the requested year on the
            # data period_end, NOT on `fy` (the filing's fiscal year), because a
            # 10-K's prior-year comparatives all carry the filing's fy and would
            # otherwise be mislabeled as the requested year.
            by_period: Dict[Tuple[str, str], Tuple[Tuple[str, int], Dict[str, Any]]] = {}
            for item in entries:
                if not isinstance(item, dict):
                    continue
                form = str(item.get("form") or "")
                if forms and form not in forms:
                    continue
                end = str(item.get("end") or "")
                if not end:
                    continue
                if want_year is not None and end[:4] != str(want_year):
                    continue
                key = (end, str(item.get("start") or ""))
                rank = (str(item.get("filed") or ""), _fy_int(item))
                prev = by_period.get(key)
                if prev is None or rank > prev[0]:
                    by_period[key] = (rank, {**item, "unit": unit_name})
            candidates = [value[1] for value in by_period.values()]
            candidates.sort(key=lambda it: str(it.get("end") or ""), reverse=True)
            if candidates:
                return candidates[: max(1, limit)]
        return []

    @classmethod
    def _indicator_concepts(cls, indicator: str) -> Optional[set]:
        """Map a requested ``indicator`` to the curated concepts to keep.

        Returns ``None`` when the indicator is empty or matches nothing curated,
        in which case the caller falls back to the statement-type bundle rather
        than returning an empty result.
        """
        norm = re.sub(r"[^a-z0-9]", "", str(indicator or "").lower())
        if not norm:
            return None
        hits: set = set()
        for concept, label in cls.SEC_FUNDAMENTAL_CONCEPTS:
            cnorm = concept.lower()
            lnorm = re.sub(r"[^a-z0-9]", "", label.lower())
            if norm in cnorm or norm in lnorm or lnorm in norm:
                hits.add(concept)
        for alias, concept in cls.SEC_INDICATOR_ALIASES.items():
            if norm == alias or norm in alias or alias in norm:
                hits.add(concept)
        return hits or None

    def _build_concept_plan(
        self,
        indicator_raw: str,
        statement_type: str,
        facts: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """Decide which (concept, label) pairs to return.

        Priority: a specific ``indicator`` (first via curated labels, then via
        any us-gaap concept the company actually reports) overrides the
        statement bundle. If the indicator matches nothing, fall back to the
        statement-type bundle rather than returning empty.
        """
        if indicator_raw:
            curated = self._indicator_concepts(indicator_raw)
            if curated:
                return [(c, l) for (c, l) in self.SEC_FUNDAMENTAL_CONCEPTS if c in curated]
            generic = self._resolve_facts_concepts(indicator_raw, facts)
            if generic:
                return [
                    (c, str((facts.get(c) or {}).get("label") or self._humanize_concept(c)))
                    for c in generic
                ]
        return [
            (c, l)
            for (c, l) in self.SEC_FUNDAMENTAL_CONCEPTS
            if not statement_type or self._concept_matches_statement(l, statement_type)
        ]

    @classmethod
    def _resolve_facts_concepts(cls, indicator: str, facts: Dict[str, Any]) -> List[str]:
        """Match an indicator to actual us-gaap tags present in this filer's facts."""
        norm = re.sub(r"[^a-z0-9]", "", str(indicator or "").lower())
        if not norm or not isinstance(facts, dict) or not facts:
            return []
        keys = list(facts.keys())
        norm_keys = {k: re.sub(r"[^a-z0-9]", "", k.lower()) for k in keys}
        alias = cls.SEC_INDICATOR_ALIASES.get(norm)
        if alias and alias in facts:
            return [alias]
        exact = [k for k in keys if norm_keys[k] == norm]
        if exact:
            return exact[:3]
        if len(norm) >= 4:
            subs = [k for k in keys if norm in norm_keys[k] or norm_keys[k] in norm]
            subs.sort(key=lambda k: len(norm_keys[k]))
            if subs:
                return subs[:5]
        return []

    @staticmethod
    def _humanize_concept(concept: str) -> str:
        spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", str(concept or ""))
        return spaced.strip() or str(concept or "")

    @staticmethod
    def _period_year(arguments: Dict[str, Any]) -> Optional[int]:
        for key in ("period", "date", "fiscal_year", "year", "query"):
            text = str(arguments.get(key) or "")
            match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)
            if match:
                return int(match.group(0))
        return None

    @staticmethod
    def _wants_quarterly(arguments: Dict[str, Any]) -> bool:
        text = " ".join(
            str(arguments.get(key) or "")
            for key in ("period", "statement_type", "indicator", "query", "frequency")
        ).lower()
        return bool(re.search(r"\bq[1-4]\b|quarter|10-?q", text))

    @staticmethod
    def _concept_matches_statement(label: str, statement_type: str) -> bool:
        income = {"Revenue", "Cost of revenue", "Gross profit", "R&D expense", "SG&A expense",
                  "Operating income", "Pre-tax income", "Net income", "EPS (basic)", "EPS (diluted)",
                  "Weighted avg shares (basic)", "Weighted avg shares (diluted)", "Dividends per share"}
        balance = {"Total assets", "Total liabilities", "Stockholders equity",
                   "Cash and equivalents", "Long-term debt (noncurrent)"}
        cash = {"Cash flow from operations", "Cash flow from investing",
                "Cash flow from financing", "Capital expenditures", "Dividends paid"}
        if any(k in statement_type for k in ("income", "profit", "operations")):
            return label in income
        if "balance" in statement_type:
            return label in balance
        if "cash" in statement_type:
            return label in cash
        return True

    @classmethod
    def _sec_http_text(cls, url: str, timeout: int = 30) -> str:
        # SEC's EDGAR/data endpoints require a descriptive User-Agent identifying
        # the requester; a bare browser UA is frequently rejected with 403.
        ua = (
            os.getenv("SEC_API_USER_AGENT")
            or os.getenv("FIRE_AGENT_SEC_USER_AGENT")
            or "fire-agent research contact@example.com"
        )
        try:
            import requests

            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            )
            response.raise_for_status()
            return response.text
        except ImportError:
            return http_get(url, headers={"User-Agent": ua}, timeout=timeout).text

    @classmethod
    def _sec_http_json(cls, url: str, timeout: int = 30) -> Any:
        return json.loads(cls._sec_http_text(url, timeout=timeout))

    @staticmethod
    def _provider_hint(arguments: Dict[str, Any]) -> str:
        valid_providers = {
            "auto",
            "akshare",
            "tushare_http",
            "yahoo",
            "yfinance",
            "fred",
            "alfred",
            "fed",
            "world_bank",
            "worldbank",
            "wb",
            "oecd",
            "iea",
            "comtrade",
            "wits",
            "lbma",
            "lme",
            "safe",
            "nfra",
            "cme",
            "cbot",
            "yahoo_contract",
            "sw",
            "shenwan",
            "csindex",
            "csi",
        }
        provider = str(arguments.get("provider") or "").strip().lower().replace("-", "_")
        if provider == "fed":
            return "fred"
        if provider in {"tushare", "tinyshare", "tushare_http", "tushare_private", "private_tushare"}:
            return "tushare_http"
        if provider in valid_providers:
            return provider
        source = str(arguments.get("source") or "").strip().lower().replace("-", "_")
        if source == "fed":
            return "fred"
        if source in {"tushare", "tinyshare", "tushare_http", "tushare_private", "private_tushare"}:
            return "tushare_http"
        return source if source in valid_providers else "auto"

    def _unsupported_price_request(
        self,
        arguments: Dict[str, Any],
        provider: str,
        yahoo_symbol: str,
    ) -> Optional[ToolResult]:
        asset = str(arguments.get("asset_class") or "").strip().lower()
        market = str(arguments.get("market") or "").strip().lower()
        if asset not in {"futures", "future", "commodity", "commodities"} and market not in {"futures", "commodity", "commodities"}:
            return None
        if yahoo_symbol:
            return None
        ticker = self._market_symbol(arguments)
        return ToolResult(
            self.name,
            provider if provider != "auto" else "market_router",
            "error",
            action="price_history",
            error=(
                f"Unsupported futures/commodity price_history symbol {ticker!r}. "
                "The router only maps supported continuous Yahoo symbols such as CL=F, GC=F, ZC=F, "
                "or explicit aliases such as CL/WTI/GOLD/CORN. Specific contract settlement symbols "
                "need a dedicated futures data mapping before they can be queried safely."
            ),
            paid=False,
            confidence=0.2,
            metadata={
                "requested_ticker": ticker,
                "asset_class": asset or market,
                "provider_hint": provider,
                "reason": "avoid_falling_back_to_stock_zh_a_hist",
            },
        )

    def _fetch_yahoo_history(
        self,
        yahoo_symbol: str,
        mapped: Dict[str, Any],
        original_arguments: Dict[str, Any],
        timeout: int = 30,
        action: str = "price_history",
    ) -> ToolResult:
        start_date = self._coerce_date(mapped.get("start_date"), default="1900-01-01", is_end=False)
        end_date = self._coerce_date(mapped.get("end_date"), default=datetime.now(timezone.utc).date().isoformat(), is_end=True)
        limit = self._optional_positive_int(mapped.get("limit"))
        if action != "price_history" and limit is None:
            limit = self._optional_positive_int(original_arguments.get("limit"))
        period1 = self._date_to_epoch(start_date)
        period2_date = datetime.strptime(end_date, "%Y-%m-%d").date() + timedelta(days=1)
        period2 = int(datetime(period2_date.year, period2_date.month, period2_date.day, tzinfo=timezone.utc).timestamp())
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(yahoo_symbol, safe='')}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history"
        )
        try:
            payload = self._market_http_json(url, timeout=timeout)
            chart = payload.get("chart") or {}
            if chart.get("error"):
                raise ValueError(str(chart["error"]))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise ValueError("Yahoo Finance returned no chart result")
            timestamps = result.get("timestamp") or []
            indicators = result.get("indicators") or {}
            quote_rows = (indicators.get("quote") or [{}])[0]
            adj_rows = (indicators.get("adjclose") or [{}])[0]
            meta = result.get("meta") or {}
        except Exception as exc:
            return ToolResult(
                self.name,
                "yahoo_finance",
                "error",
                action=action,
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
                metadata={"symbol": yahoo_symbol, "url": url},
            )

        records: List[Dict[str, Any]] = []
        requested_unadjusted = self._requested_unadjusted_price(original_arguments)
        yahoo_adjust_basis = (
            "split_adjusted_ohlc; adj_close is adjusted for splits and dividends when Yahoo provides it; "
            "raw pre-split/unadjusted OHLC is not available from Yahoo chart"
        )
        if requested_unadjusted:
            return ToolResult(
                self.name,
                "yahoo_finance",
                "error",
                action=action,
                error=(
                    "Requested unadjusted/raw OHLC, but Yahoo Finance chart returns split-adjusted OHLC "
                    "and does not expose raw pre-split/unadjusted open/close rows through this endpoint."
                ),
                paid=False,
                confidence=0.2,
                metadata={
                    "provider": "yahoo_finance",
                    "url": url,
                    "requested_ticker": original_arguments.get("ticker") or original_arguments.get("series"),
                    "symbol": yahoo_symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "endpoint": "chart",
                    "requested_adjust": original_arguments.get("adjust"),
                    "adjust_basis": yahoo_adjust_basis,
                    "raw_unadjusted_available": False,
                    "price_basis_warning": "requested_unadjusted_but_yahoo_chart_returns_split_adjusted_ohlc",
                    "retry_same_args_allowed": False,
                },
            )
        for idx, ts in enumerate(timestamps):
            date = datetime.fromtimestamp(int(ts), timezone.utc).date().isoformat()
            record = {
                "date": date,
                "symbol": yahoo_symbol,
                "adjust_basis": yahoo_adjust_basis,
                "raw_unadjusted_available": False,
                "open": self._list_get(quote_rows.get("open"), idx),
                "high": self._list_get(quote_rows.get("high"), idx),
                "low": self._list_get(quote_rows.get("low"), idx),
                "close": self._list_get(quote_rows.get("close"), idx),
                "adj_close": self._list_get(adj_rows.get("adjclose"), idx),
                "volume": self._list_get(quote_rows.get("volume"), idx),
                "currency": meta.get("currency"),
            }
            if any(record.get(field) is not None for field in ("open", "high", "low", "close", "adj_close")):
                records.append(record)
        if limit is not None:
            records = records[-limit:]
        return ToolResult(
            self.name,
            "yahoo_finance",
            "success" if records else "error",
            action=action,
            observation=_finance_agent_records_to_csv(records),
            tables=[{"columns": list(records[0].keys()) if records else [], "rows": records}] if records else [],
            error=None if records else "Yahoo Finance returned no rows in the requested date range",
            confidence=0.82 if records else 0.25,
            paid=False,
            metadata={
                "provider": "yahoo_finance",
                "url": url,
                "requested_ticker": original_arguments.get("ticker") or original_arguments.get("series"),
                "symbol": yahoo_symbol,
                "start_date": start_date,
                "end_date": end_date,
                "result_count": len(records),
                "endpoint": "chart",
                "params": {
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "history",
                },
                "currency": meta.get("currency"),
                "adjust": yahoo_adjust_basis,
                "adjust_basis": yahoo_adjust_basis,
                "requested_adjust": original_arguments.get("adjust"),
                "raw_unadjusted_available": False,
                "unit_schema": {
                    "raw_unadjusted_open": "not available from Yahoo chart",
                    "raw_unadjusted_close": "not available from Yahoo chart",
                    "open": meta.get("currency") or "source currency",
                    "high": meta.get("currency") or "source currency",
                    "low": meta.get("currency") or "source currency",
                    "close": meta.get("currency") or "source currency",
                    "adj_close": meta.get("currency") or "source currency",
                    "volume": "shares",
                },
            },
        )

    @staticmethod
    def _requested_unadjusted_price(arguments: Dict[str, Any]) -> bool:
        text = " ".join(
            str(arguments.get(key) or "")
            for key in ("adjust", "basis", "price_basis", "query", "field", "fields")
        ).lower().replace("-", "_")
        return any(
            marker in text
            for marker in (
                "unadjusted",
                "raw",
                "not_adjusted",
                "non_adjusted",
                "no_adjust",
                "不复权",
                "未复权",
            )
        )

    @staticmethod
    def _has_vintage_request(arguments: Dict[str, Any]) -> bool:
        return any(str(arguments.get(key) or "").strip() for key in ("vintage_date", "realtime_start", "realtime_end"))

    def _fetch_fred_series(self, fred_id: str, arguments: Dict[str, Any], limit: int, timeout: int) -> ToolResult:
        start_date = self._coerce_date(arguments.get("start_date"), default="", is_end=False)
        end_date = self._coerce_date(arguments.get("end_date"), default="", is_end=True)
        single_date = self._coerce_date(arguments.get("date"), default="", is_end=False)
        if single_date and not start_date and not end_date:
            start_date = single_date
            end_date = single_date
        vintage_date = self._coerce_date(arguments.get("vintage_date"), default="", is_end=True)
        realtime_start = self._coerce_date(arguments.get("realtime_start"), default=vintage_date, is_end=False)
        realtime_end = self._coerce_date(arguments.get("realtime_end"), default=vintage_date or realtime_start, is_end=True)
        wants_vintage = bool(vintage_date or realtime_start or realtime_end)
        api_key = self._fred_api_key()
        series_title = self._fred_series_title(fred_id)
        if wants_vintage and not api_key:
            return self._structured_error(
                "alfred",
                "macro_series",
                "FRED/ALFRED realtime vintage observations require FRED_API_KEY or FIRE_AGENT_FRED_API_KEY; current-value CSV fallback is disabled for vintage requests.",
                metadata={"series_id": fred_id, "series_title": series_title, "realtime_start": realtime_start, "realtime_end": realtime_end},
            )
        if api_key:
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={quote(fred_id, safe='')}&file_type=json&api_key={quote(api_key, safe='')}"
                f"{'&observation_start=' + start_date if start_date else ''}"
                f"{'&observation_end=' + end_date if end_date else ''}"
                f"{'&realtime_start=' + realtime_start if realtime_start else ''}"
                f"{'&realtime_end=' + realtime_end if realtime_end else ''}"
            )
            return self._fetch_fred_series_api(
                fred_id,
                url,
                limit,
                timeout,
                provider="alfred" if wants_vintage else "fred",
                series_title=series_title,
                requested_start_date=start_date,
                requested_end_date=end_date,
            )

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote(fred_id, safe='')}"
        try:
            text = self._market_http_text(url, timeout=timeout)
            rows = list(csv.DictReader(io.StringIO(text)))
        except Exception as exc:
            return ToolResult(
                self.name,
                "fred",
                "error",
                action="macro_series",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
                metadata={"series_id": fred_id, "url": url},
            )

        records: List[Dict[str, Any]] = []
        for row in rows:
            date = str(row.get("observation_date") or "").strip()
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            raw_value = row.get(fred_id)
            record = {"date": date, "series": fred_id, "value": self._to_float(raw_value), "raw_value": raw_value}
            if series_title:
                record["series_title"] = series_title
            records.append(record)
        records = records[-limit:]
        actual_dates = [str(row.get("date") or "") for row in records if row.get("date")]
        return ToolResult(
            self.name,
            "fred",
            "success" if records else "error",
            action="macro_series",
            observation=_finance_agent_records_to_csv(records),
            tables=[{"columns": list(records[0].keys()) if records else [], "rows": records}] if records else [],
            error=None if records else "FRED returned no rows in the requested date range",
            confidence=0.86 if records else 0.25,
            paid=False,
            metadata={
                "provider": "fred",
                "url": url,
                "series_id": fred_id,
                "series": fred_id,
                "series_title": series_title,
                "requested_start_date": start_date,
                "requested_end_date": end_date,
                "actual_start_date": min(actual_dates) if actual_dates else None,
                "actual_end_date": max(actual_dates) if actual_dates else None,
                "result_count": len(records),
            },
        )

    def _fetch_fred_series_api(
        self,
        fred_id: str,
        url: str,
        limit: int,
        timeout: int,
        *,
        provider: str = "fred",
        series_title: str = "",
        requested_start_date: str = "",
        requested_end_date: str = "",
    ) -> ToolResult:
        include_realtime = provider == "alfred" or "realtime_start=" in url or "realtime_end=" in url
        try:
            payload = self._market_http_json(url, timeout=timeout)
            rows = payload.get("observations") or []
        except Exception as exc:
            return ToolResult(
                self.name,
                provider,
                "error",
                action="macro_series",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
                metadata={
                    "provider": provider,
                    "series_id": fred_id,
                    "series": fred_id,
                    "series_title": series_title,
                    "requested_start_date": requested_start_date,
                    "requested_end_date": requested_end_date,
                    "url": re.sub(r"api_key=[^&]+", "api_key=***", url),
                    "current_fred_fallback": False if include_realtime else None,
                },
            )

        records = []
        for row in rows:
            if include_realtime:
                realtime_start = row.get("realtime_start")
                realtime_end = row.get("realtime_end")
                record = {
                    "observation_date": row.get("date"),
                    "series": fred_id,
                    "value": self._to_float(row.get("value")),
                    "raw_value": row.get("value"),
                    "vintage_date": realtime_start if realtime_start == realtime_end else None,
                    "realtime_start": realtime_start,
                    "realtime_end": realtime_end,
                }
            else:
                record = {
                    "date": row.get("date"),
                    "series": fred_id,
                    "value": self._to_float(row.get("value")),
                    "raw_value": row.get("value"),
                }
            if series_title:
                record["series_title"] = series_title
            if include_realtime:
                record["data_version"] = "vintage"
            records.append(record)
        records = records[-limit:]
        actual_dates = [
            str(row.get("observation_date") or row.get("date") or "")
            for row in records
            if row.get("observation_date") or row.get("date")
        ]
        return ToolResult(
            self.name,
            provider,
            "success" if records else "error",
            action="macro_series",
            observation=_finance_agent_records_to_csv(records),
            tables=[{"columns": list(records[0].keys()) if records else [], "rows": records}] if records else [],
            error=None if records else f"{provider.upper()} returned no rows in the requested date range",
            confidence=0.88 if records else 0.25,
            paid=False,
            metadata={
                "provider": provider,
                "url": re.sub(r"api_key=[^&]+", "api_key=***", url),
                "series_id": fred_id,
                "series": fred_id,
                "series_title": series_title,
                "requested_start_date": requested_start_date,
                "requested_end_date": requested_end_date,
                "actual_start_date": min(actual_dates) if actual_dates else None,
                "actual_end_date": max(actual_dates) if actual_dates else None,
                "result_count": len(records),
                "api": "fred_api",
                "data_version": "vintage" if include_realtime else "current",
                "current_fred_fallback": False if include_realtime else None,
            },
        )

    @staticmethod
    def _fred_api_key() -> str:
        key = os.getenv("FRED_API_KEY") or os.getenv("FIRE_AGENT_FRED_API_KEY")
        if key:
            return key
        env_path = Path(__file__).resolve().parents[4] / ".env"
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                name, value = stripped.split("=", 1)
                if name.strip() in {"FRED_API_KEY", "FIRE_AGENT_FRED_API_KEY"}:
                    return value.strip().strip("\"'")
        except OSError:
            return ""
        return ""

    @staticmethod
    def _world_bank_all_countries_requested(country: str, query: str, combined: str) -> bool:
        country_text = str(country or "").strip().lower()
        if country_text in {"all", "all countries", "countries", "world", "worldwide", "global", "europe", "european"}:
            return True
        text = f"{country} {query} {combined}".lower()
        if re.fullmatch(r"[a-z]{3}", country_text):
            return bool(re.search(r"\b(all|countries|among|which|rank|ranking|lowest|highest|top|bottom)\b", text))
        return bool(
            re.search(r"\b(all|countries|worldwide|global|europe|european)\b", text)
            and not re.search(r"\b(country code|iso3|iso 3)\b", text)
        )

    @staticmethod
    def _world_bank_region_filter(arguments: Dict[str, Any], requested_country: str) -> str:
        text = " ".join(
            str(arguments.get(key) or "")
            for key in ("region", "scope", "country", "query", "series", "indicator")
        )
        text = f"{requested_country} {text}".lower()
        if "europe" in text or "european" in text:
            return "europe"
        return ""

    @staticmethod
    def _world_bank_rank_order(arguments: Dict[str, Any], requested_country: str) -> str:
        sort = str(arguments.get("sort") or arguments.get("order") or "").strip().lower()
        if sort in {"value_asc", "asc", "ascending", "lowest", "min", "minimum", "bottom"}:
            return "asc"
        if sort in {"value_desc", "desc", "descending", "highest", "max", "maximum", "top"}:
            return "desc"
        text = " ".join(str(arguments.get(key) or "") for key in ("query", "series", "indicator", "rank", "metric"))
        text = f"{requested_country} {text}".lower()
        if re.search(r"\b(lowest|smallest|minimum|min|bottom)\b", text):
            return "asc"
        if re.search(r"\b(highest|largest|maximum|max|top)\b", text):
            return "desc"
        return ""

    def _fetch_world_bank_series(
        self,
        country: str,
        indicator: str,
        arguments: Dict[str, Any],
        limit: int,
        timeout: int,
    ) -> ToolResult:
        requested_country = str(arguments.get("country") or arguments.get("query") or country or "").strip()
        combined_request = " ".join(
            str(arguments.get(key) or "")
            for key in ("country", "query", "series", "indicator", "region", "scope")
        )
        all_countries = self._world_bank_all_countries_requested(country, requested_country, combined_request)
        country = self._country_iso3(country) or country
        api_country = "all" if all_countries else country
        start_year, end_year = self._year_bounds(arguments)
        date_part = f"&date={start_year}:{end_year}" if start_year and end_year else ""
        url = (
            "https://api.worldbank.org/v2/country/"
            f"{quote(api_country, safe='')}/indicator/{quote(indicator, safe='.')}"
            f"?format=json&per_page=20000{date_part}"
        )
        try:
            payload = self._market_http_json(url, timeout=timeout)
            rows = payload[1] if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list) else []
        except Exception as exc:
            return ToolResult(
                self.name,
                "world_bank",
                "error",
                action="macro_series",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
                metadata={
                    "country": api_country,
                    "requested_country": requested_country,
                    "indicator": indicator,
                    "requested_start_year": start_year,
                    "requested_end_year": end_year,
                    "url": url,
                },
            )

        records: List[Dict[str, Any]] = []
        region_filter = self._world_bank_region_filter(arguments, requested_country)
        for row in rows:
            row_country = ((row.get("country") or {}).get("value") if isinstance(row.get("country"), dict) else country)
            row_country_code = str(row.get("countryiso3code") or "").strip().upper()
            if all_countries:
                if not re.fullmatch(r"[A-Z]{3}", row_country_code):
                    continue
                if region_filter == "europe" and row_country_code not in self.europe_country_iso3:
                    continue
            else:
                row_country_code = row_country_code or country
            records.append(
                {
                    "date": row.get("date"),
                    "country": row_country,
                    "country_code": row_country_code,
                    "indicator": indicator,
                    "indicator_name": ((row.get("indicator") or {}).get("value") if isinstance(row.get("indicator"), dict) else indicator),
                    "value": row.get("value"),
                }
            )
        rank_order = self._world_bank_rank_order(arguments, requested_country)
        if rank_order:
            if all_countries:
                latest_by_country: Dict[str, Dict[str, Any]] = {}
                for row in records:
                    code = str(row.get("country_code") or row.get("country") or "").strip()
                    if not code:
                        continue
                    existing = latest_by_country.get(code)
                    if existing is None or str(row.get("date") or "") > str(existing.get("date") or ""):
                        latest_by_country[code] = row
                records = list(latest_by_country.values())
            records = [row for row in records if self._to_float(row.get("value")) is not None]
            records.sort(
                key=lambda item: (
                    self._to_float(item.get("value")) or 0.0,
                    str(item.get("country") or ""),
                ),
                reverse=rank_order == "desc",
            )
            records = records[:limit]
        elif all_countries:
            records.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("country") or "")), reverse=True)
            records = records[:limit]
            records.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("country") or "")))
        else:
            records.sort(key=lambda item: str(item.get("date") or ""))
            records = records[-limit:]
        actual_dates = [str(row.get("date") or "") for row in records if row.get("date")]
        return ToolResult(
            self.name,
            "world_bank",
            "success" if records else "error",
            action="macro_series",
            observation=_finance_agent_records_to_csv(records),
            tables=[{"columns": list(records[0].keys()) if records else [], "rows": records}] if records else [],
            error=None if records else "World Bank returned no rows in the requested date range",
            confidence=0.86 if records else 0.25,
            paid=False,
            metadata={
                "provider": "world_bank",
                "url": url,
                "country": api_country,
                "requested_country": requested_country,
                "all_countries": all_countries,
                "region_filter": region_filter,
                "rank_order": rank_order,
                "indicator": indicator,
                "requested_start_year": start_year,
                "requested_end_year": end_year,
                "actual_start_date": min(actual_dates) if actual_dates else None,
                "actual_end_date": max(actual_dates) if actual_dates else None,
                "result_count": len(records),
            },
        )

    def _resolve_yahoo_symbol(self, arguments: Dict[str, Any]) -> str:
        raw = str(arguments.get("ticker") or arguments.get("symbol") or arguments.get("series") or arguments.get("query") or "").strip()
        if not raw:
            return ""
        upper = raw.upper().replace("_", " ").strip()
        compact = re.sub(r"\s+", " ", upper)
        if compact in self.yahoo_symbol_aliases:
            return self.yahoo_symbol_aliases[compact]
        if raw.startswith("^") or raw.upper().endswith(("=F", ".NYB")):
            return raw.upper()
        hk_match = re.fullmatch(r"(?:HK)?0*(\d{1,5})(?:\.HK)?", upper)
        if hk_match and (upper.endswith(".HK") or upper.startswith("HK") or str(arguments.get("market") or "").lower() in {"hk", "hkg", "hongkong", "hk_stock"}):
            return f"{int(hk_match.group(1)):04d}.HK"
        provider = self._provider_hint(arguments)
        market = str(arguments.get("market") or "").lower()
        if re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", upper) and (provider in {"yahoo", "yfinance"} or market in {"us", "usa", "yahoo", "global"}):
            return upper
        return ""

    @staticmethod
    def _market_http_text(url: str, timeout: int = 30) -> str:
        try:
            import requests

            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return response.text
        except ImportError:
            return http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout).text

    @classmethod
    def _market_http_json(cls, url: str, timeout: int = 30) -> Any:
        return json.loads(cls._market_http_text(url, timeout=timeout))

    def _resolve_fred_series(self, series: str) -> str:
        value = str(series or "").strip()
        upper = value.upper().replace("-", "_")
        normalized = re.sub(r"\s+", " ", upper)
        if normalized in self.fred_series_aliases:
            return self.fred_series_aliases[normalized]
        alias_key = self._fred_alias_key(value)
        if alias_key in self.fred_series_aliases:
            return self.fred_series_aliases[alias_key]
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,24}", value) and not value.lower().startswith("macro"):
            return value
        return ""

    @staticmethod
    def _fred_alias_key(value: Any) -> str:
        return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()

    def _fred_series_title(self, fred_id: str) -> str:
        meta = self.fred_series_catalog.get(str(fred_id or "").upper(), {})
        return str(meta.get("series_title") or "").strip()

    def _resolve_world_bank_indicator(self, text: Any) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        match = re.search(r"[A-Z]{2}\.[A-Z0-9.]{5,}", value)
        if match:
            return match.group(0)
        key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if key in self.world_bank_indicator_aliases:
            return self.world_bank_indicator_aliases[key]
        lowered = value.lower()
        best_indicator = ""
        best_score = 0
        for indicator, meta in self.world_bank_indicator_catalog.items():
            candidates = [str(meta.get("indicator_name") or ""), *(str(alias) for alias in meta.get("aliases") or [])]
            for candidate in candidates:
                candidate_l = candidate.lower()
                if not candidate_l:
                    continue
                if candidate_l in lowered or lowered in candidate_l:
                    score = len(candidate_l)
                else:
                    query_tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if len(token) >= 3]
                    candidate_tokens = set(token for token in re.split(r"[^a-z0-9]+", candidate_l) if len(token) >= 3)
                    score = sum(1 for token in query_tokens if token in candidate_tokens)
                if score > best_score:
                    best_score = score
                    best_indicator = indicator
        return best_indicator if best_score >= 2 else ""

    def _resolve_world_bank_series(self, series: str, query: str, arguments: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        indicator = str(arguments.get("indicator") or "").strip()
        country = str(arguments.get("country") or query or "").strip()
        series_text = str(series or "").strip()
        combined = f"{series_text} {query}".strip()
        indicator_match = re.search(r"[A-Z]{2}\.[A-Z0-9.]{5,}", combined)
        if indicator_match:
            indicator = indicator_match.group(0)
        elif indicator:
            indicator = self._resolve_world_bank_indicator(indicator)
        country_indicator_match = re.search(r"\b([A-Z]{3})[_: -]+[A-Z]{2}\.[A-Z0-9.]{5,}", combined.upper())
        if country_indicator_match:
            country = country_indicator_match.group(1)
        if not indicator:
            indicator = self._resolve_world_bank_indicator(series_text) or self._resolve_world_bank_indicator(combined)
        if not indicator:
            lowered = combined.lower()
            if "unemployment" in lowered:
                indicator = "SL.UEM.TOTL.ZS"
            elif "inflation" in lowered or "cpi" in lowered:
                indicator = "FP.CPI.TOTL.ZG"
            elif "gdp" in lowered and ("per capita" in lowered or "pcap" in lowered):
                indicator = "NY.GDP.PCAP.CD"
            elif "gdp" in lowered and ("growth" in lowered or "real" in lowered):
                indicator = "NY.GDP.MKTP.KD.ZG"
            elif "gdp" in lowered:
                indicator = "NY.GDP.MKTP.CD"
        if not country:
            for name, code in self.country_aliases.items():
                if re.search(rf"\b{re.escape(name.lower())}\b", combined.lower()):
                    country = code
                    break
        if indicator and self._world_bank_all_countries_requested(country, query, combined):
            return "all", indicator
        country_code = self._country_iso3(country)
        if country_code and indicator and re.fullmatch(r"[A-Z]{3}", country_code):
            return country_code, indicator
        return None

    def _is_global_price_request(self, arguments: Dict[str, Any]) -> bool:
        market = str(arguments.get("market") or "").lower()
        asset = str(arguments.get("asset_class") or "").lower()
        ticker = str(arguments.get("ticker") or arguments.get("symbol") or "").strip().upper()
        if market in {"global", "yahoo", "us", "usa", "ger", "cbot", "otc"}:
            return True
        if asset in {"index", "global_index", "commodity", "futures", "fx", "forex"}:
            return True
        return ticker in self.yahoo_symbol_aliases or ticker.startswith("^") or ticker.endswith(("=F", ".NYB"))

    def _looks_like_market_series(self, series: str) -> bool:
        value = str(series or "").strip().upper()
        return value in self.yahoo_symbol_aliases or value.startswith("^") or value.endswith(("=F", ".NYB"))

    @staticmethod
    def _date_to_epoch(value: str) -> int:
        parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())

    @staticmethod
    def _coerce_date(value: Any, default: str = "", is_end: bool = False) -> str:
        text = str(value or "").strip()
        if not text:
            return default
        if re.fullmatch(r"\d{4}", text):
            return f"{text}-12-31" if is_end else f"{text}-01-01"
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return f"{text}-28" if is_end else f"{text}-01"
        if re.fullmatch(DATE_REGEX, text):
            return text
        digits = re.sub(r"\D", "", text)
        if re.fullmatch(r"\d{8}", digits):
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        return default

    @staticmethod
    def _year_from_date(value: Any) -> str:
        text = str(value or "").strip()
        match = re.search(r"\d{4}", text)
        return match.group(0) if match else ""

    @staticmethod
    def _list_get(values: Any, idx: int) -> Any:
        if not isinstance(values, list) or idx >= len(values):
            return None
        value = values[idx]
        return value if value is not None else None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, "", "."):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _market_symbol(self, arguments: Dict[str, Any]) -> str:
        return str(arguments.get("ticker") or arguments.get("symbol") or arguments.get("query") or "").strip()

    @staticmethod
    def _looks_like_us_equity_ticker(ticker: str) -> bool:
        value = (ticker or "").strip().upper()
        if not value or value.endswith((".SH", ".SZ", ".BJ")):
            return False
        if re.fullmatch(r"\d{6}", value) or re.fullmatch(r"(SH|SZ|BJ)\d{6}", value):
            return False
        return bool(re.fullmatch(r"[A-Z][A-Z.]{0,7}", value))

    @staticmethod
    def _us_financial_statement_function(statement_type: str) -> Tuple[str, Optional[str]]:
        normalized = re.sub(r"[^a-z]+", "_", (statement_type or "").lower()).strip("_")
        statement_map = {
            "income": "综合损益表",
            "income_statement": "综合损益表",
            "profit": "综合损益表",
            "profit_statement": "综合损益表",
            "balance": "资产负债表",
            "balance_sheet": "资产负债表",
            "cash": "现金流量表",
            "cashflow": "现金流量表",
            "cash_flow": "现金流量表",
            "cash_flow_statement": "现金流量表",
        }
        if normalized in {"abstract", "indicator", "indicators", "analysis", "financial_indicator"}:
            return "stock_financial_us_analysis_indicator_em", None
        return "stock_financial_us_report_em", statement_map.get(normalized, "综合损益表")

    @staticmethod
    def _us_financial_statement_period(arguments: Dict[str, Any]) -> str:
        value = str(arguments.get("period") or arguments.get("date") or arguments.get("indicator") or "annual").lower()
        if any(token in value for token in ("single", "quarter", "季度", "单季", "10-q")):
            return "单季报"
        if any(token in value for token in ("cumulative", "ytd", "累计")):
            return "累计季报"
        return "年报"

    def _snapshot_function(self, arguments: Dict[str, Any]) -> str:
        market = str(arguments.get("market") or arguments.get("asset_class") or "").lower()
        if market in {"us", "usa", "us_stock", "equity_us"}:
            return "stock_us_spot_em"
        if market in {"hk", "hkg", "hongkong", "hk_stock"}:
            return "stock_hk_spot_em"
        if market in {"index", "global_index", "global_idx"}:
            return "index_global_spot_em"
        if market in {"cn_index", "china_index"}:
            return "stock_zh_index_spot_em"
        if market in {"hk_index"}:
            return "stock_hk_index_spot_em"
        if market in {"fx", "forex"}:
            return "forex_spot_em"
        return "stock_zh_a_spot_em"

    def _history_function(self, arguments: Dict[str, Any]) -> str:
        market = str(arguments.get("market") or arguments.get("asset_class") or "").lower()
        if market in {"us", "usa", "us_stock", "equity_us", "equity", "etf"}:
            return "stock_us_hist"
        if market in {"hk", "hkg", "hongkong", "hk_stock"}:
            return "stock_hk_hist"
        if market in {"index", "global_index", "global_idx"}:
            return "index_global_hist_em"
        if market in {"fx", "forex"}:
            return "forex_hist_em"
        if market in {"fund", "etf_cn"}:
            return "fund_etf_hist_em"
        return "stock_zh_a_hist"
