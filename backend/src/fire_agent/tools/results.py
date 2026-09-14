from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import ToolResult


class ToolResultFactory:
    """Small helper for constructing ToolResult objects consistently.

    The factory is intentionally thin: it does not change ToolFamily.run()
    semantics and does not infer provider/action names. Tools opt into it where
    repeated result construction would otherwise obscure their domain logic.
    """

    def __init__(self, tool_name: str, *, paid: bool = False) -> None:
        self.tool_name = tool_name
        self.paid = paid

    def error(
        self,
        provider: str,
        action: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.1,
    ) -> ToolResult:
        return ToolResult(
            self.tool_name,
            provider,
            "error",
            action=action,
            error=message,
            paid=self.paid,
            confidence=confidence,
            metadata=metadata or {},
        )

    def success(
        self,
        provider: str,
        action: str,
        observation: Any = "",
        *,
        tables: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
    ) -> ToolResult:
        return ToolResult(
            self.tool_name,
            provider,
            "success",
            action=action,
            observation=observation,
            tables=tables or [],
            confidence=confidence,
            paid=self.paid,
            metadata=metadata or {},
        )
