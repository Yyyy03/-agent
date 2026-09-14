from __future__ import annotations

import json
from typing import Any, Dict

from ..config import ToolSettings
from ..http_client import http_post
from ..schemas import ToolResult
from ..tools import ToolActionSpec, ToolFamily


class SearchDiscoveryTool(ToolFamily):
    name = "search_discovery"
    description = "Discover candidate web sources using Serper. Use for source discovery, not final evidence."
    paid = True
    default_action = "search"
    actions = {
        "search": ToolActionSpec(
            description="Search the web for candidate evidence sources.",
            required=["query"],
            optional=["provider", "max_results"],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "provider": {"type": "string", "enum": ["auto", "serper"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
    }

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        settings = ToolSettings.from_env()
        query = str(arguments.get("query", "")).strip()
        provider = str(arguments.get("provider", "auto")).lower()
        max_results = int(arguments.get("max_results", 5))
        if not query:
            return self._error("auto", action, "Missing query")

        if provider in {"serper", "auto"} and settings.serper_api_key:
            return self._serper(query, max_results, timeout, settings.serper_api_key)
        return self._error(provider, action, "Missing FIRE_AGENT_SERPER_API_KEY")

    def _serper(self, query: str, max_results: int, timeout: int, api_key: str) -> ToolResult:
        response = http_post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            data=json.dumps({"q": query, "num": max_results}),
            timeout=timeout,
        )
        payload = response.json()
        organic = payload.get("organic", [])[:max_results]
        results = []
        for item in organic:
            result = {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "date": item.get("date"),
                "provider": "serper",
            }
            results.append(result)
        return ToolResult(
            tool_family=self.name,
            provider="serper",
            status="success",
            action="search",
            observation=json.dumps(results, ensure_ascii=False, default=str),
            confidence=0.55,
            paid=True,
            metadata={"result_count": len(results), "payload_keys": list(payload.keys())[:20]},
        )

    def _error(self, provider: str, action: str, message: str) -> ToolResult:
        return ToolResult(
            tool_family=self.name,
            provider=provider,
            status="error",
            action=action,
            error=message,
            paid=True,
            confidence=0.1,
        )

class WebSearchTool(SearchDiscoveryTool):
    name = "web_search"
    description = (
        "Search public non-SEC web results through Serper. Returns snippets and URLs only; "
        "does not read full page content or SEC filings."
    )
    default_action = "search"
    actions = {
        "search": ToolActionSpec(
            description="Search public non-SEC web results and return snippets and URLs.",
            required=["query"],
            optional=["max_results"],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Public non-SEC web search query."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            examples=[{"query": "2024 social financing stock PBOC 408.34 trillion", "max_results": 10}],
        )
    }

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        clean_args = dict(arguments or {})
        clean_args["provider"] = "serper"
        return super().forward(action, clean_args, timeout=timeout)
