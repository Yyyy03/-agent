from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..schemas import ToolCall, ToolResult
from .results import ToolResultFactory


@dataclass(frozen=True)
class ToolActionSpec:
    description: str
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    # Optional JSON-Schema object for the action's direct native function
    # parameters. When omitted, runtime infers a conservative schema from
    # required/optional names so older tool definitions keep working.
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "description": self.description,
            "required": self.required,
            "optional": self.optional,
        }
        if self.parameters:
            data["parameters"] = self.parameters
        if self.examples:
            data["examples"] = self.examples
        return data


class ToolFamily:
    name = "base"
    description = ""
    paid = False
    parallel_safe = True
    default_action = "default"
    actions: Mapping[str, ToolActionSpec] = {}

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "paid": self.paid,
            "parallel_safe": self.parallel_safe,
            "default_action": self.default_action,
            "actions": {name: spec.to_dict() for name, spec in self.actions.items()},
        }

    @property
    def results(self) -> ToolResultFactory:
        return ToolResultFactory(self.name, paid=self.paid)

    def run(self, call: ToolCall, timeout: int = 30) -> ToolResult:
        action = self.normalize_action(call.action)
        try:
            validation_error = self.validate_arguments(action, call.arguments)
            if validation_error:
                return ToolResult(
                    tool_family=self.name,
                    provider=self.name,
                    status="error",
                    action=action,
                    error=validation_error,
                    paid=self.paid,
                    confidence=0.1,
                )
            return self.forward(action, call.arguments, timeout=timeout)
        except Exception as exc:
            return ToolResult(
                tool_family=self.name,
                provider=self.name,
                status="error",
                action=action,
                error=f"{type(exc).__name__}: {exc}",
                paid=self.paid,
                confidence=0.1,
            )

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        if candidate == "default":
            candidate = self.default_action
        return candidate

    def validate_arguments(self, action: str, arguments: Dict[str, Any]) -> Optional[str]:
        if self.actions and action not in self.actions:
            supported = ", ".join(sorted(self.actions))
            return f"Unsupported action '{action}'. Supported actions: {supported}"
        action_spec = self.actions.get(action)
        if not action_spec:
            return None
        missing = [key for key in action_spec.required if key not in arguments or arguments.get(key) in (None, "")]
        if missing:
            return f"Missing required argument(s) for {self.name}.{action}: {', '.join(missing)}"
        return None

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        raise NotImplementedError

    def reset_for_task(self) -> None:
        """Clear per-task state for stateful tools. Stateless tools do nothing."""
        return None
