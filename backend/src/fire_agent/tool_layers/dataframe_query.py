from __future__ import annotations

from typing import Any, Dict

from ..schemas import ToolResult
from ..tools import ToolActionSpec, ToolFamily
from .market_data import MarketDataTool


class DataFrameQueryTool(ToolFamily):
    name = "dataframe_query"
    description = (
        "Execute constrained dataframe plans over supported structured financial datasets. "
        "Write the action arguments only as this canonical JSON schema: "
        "{\"plan\":{\"universe\":{\"type\":\"symbols\",\"symbols\":[\"600900.SH\"]},"
        "\"metrics\":[{\"source\":\"price_panel\",\"start_date\":\"YYYYMMDD\","
        "\"end_date\":\"YYYYMMDD\",\"fields\":[\"ts_code\",\"trade_date\",\"close\","
        "\"pre_close\",\"pct_chg\"],\"adjust\":\"none\",\"shape\":\"long\"}],"
        "\"operations\":[{\"op\":\"filter\",\"field\":\"pct_chg\",\"operator\":\"gt\","
        "\"value\":3},{\"op\":\"sort\",\"field\":\"trade_date\",\"order\":\"asc\"},"
        "{\"op\":\"select_fields\",\"fields\":[\"ts_code\",\"trade_date\",\"pct_chg\"]}],"
        "\"limit\":200}}. Supported universe.type values are symbols, index_constituents, "
        "and index_weight. Supported metric source values are price_panel, return_panel, "
        "volume_panel, and equity_daily_basic. Supported operations are filter, sort, "
        "select_fields, rank, max_by, min_by, count, list, aggregate, and pct_change. "
        "Do not use alternate DSL shapes such as operation/column/compute, metrics keyed by "
        "source name, or universe as a string. The tool returns result rows with execution_card, "
        "coverage, trace, filled/missing slots, and failure diagnostics. It does not read SEC "
        "filings, parse narrative documents, browse the web, or execute arbitrary code."
    )
    paid = False
    parallel_safe = True
    default_action = "run"
    actions = {
        "run": ToolActionSpec(
            "Run one canonical dataframe plan. Required argument shape is "
            "{\"plan\":{\"universe\":{\"type\":\"symbols\",\"symbols\":[...]},"
            "\"metrics\":[{\"source\":\"price_panel|return_panel|volume_panel|equity_daily_basic\","
            "\"start_date\":\"YYYYMMDD\",\"end_date\":\"YYYYMMDD\",\"fields\":[...],"
            "\"shape\":\"long\"}],\"operations\":[{\"op\":\"filter|sort|select_fields|rank|"
            "max_by|min_by|count|list|aggregate|pct_change\",...}],\"limit\":200}}. "
            "Use only source/field/operator/value keys for filters and field/order keys for sorts.",
            required=["plan"],
            optional=["top_k", "limit", "max_entities"],
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "description": "Canonical dataframe plan object with universe, metrics, operations, and output/limit fields.",
                    },
                    "top_k": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1},
                    "max_entities": {"type": "integer", "minimum": 1},
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        ),
    }

    def __init__(self):
        self._market_data = MarketDataTool()

    def reset_for_task(self) -> None:
        self._market_data.reset_for_task()

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip().lower() or self.default_action
        if candidate in {"default", "query", "execute", "dataframe_query"}:
            return "run"
        return candidate

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        result = self._market_data._forward_dataframe_query(arguments or {}, timeout=timeout)
        result.tool_family = self.name
        result.action = "run"
        result.metadata = {
            **(result.metadata or {}),
            "public_tool": self.name,
            "public_action": "run",
            "executor": "structured_dataframe_executor",
        }
        return result
