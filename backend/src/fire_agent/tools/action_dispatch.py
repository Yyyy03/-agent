from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from ..schemas import ToolResult

ActionHandler = Callable[[Dict[str, Any], int], ToolResult]


class ActionDispatcher:
    """Name-to-handler dispatch table for ToolFamily implementations.

    This keeps routing data explicit without changing the public ToolFamily
    contract. Handlers receive normalized arguments and timeout seconds.
    """

    def __init__(self, handlers: Optional[Mapping[str, ActionHandler]] = None) -> None:
        self._handlers: Dict[str, ActionHandler] = dict(handlers or {})

    def register(self, action: str, handler: ActionHandler) -> None:
        self._handlers[str(action)] = handler

    def get(self, action: str) -> Optional[ActionHandler]:
        return self._handlers.get(str(action))

    def dispatch(self, action: str, arguments: Dict[str, Any], timeout: int) -> Optional[ToolResult]:
        handler = self.get(action)
        if handler is None:
            return None
        return handler(arguments, timeout)
