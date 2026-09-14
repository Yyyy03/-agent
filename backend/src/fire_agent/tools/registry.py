from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from ..schemas import ToolCall, ToolResult
from .base import ToolFamily


class ToolRegistry(dict):
    """Name-addressed tool registry with duplicate protection and specs."""

    def register(self, tool: ToolFamily, *, replace: bool = False) -> None:
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        if tool.name in self and not replace:
            raise ValueError(f"Tool already registered: {tool.name}")
        self[tool.name] = tool

    def specs(self) -> Dict[str, Dict[str, Any]]:
        return {name: tool.spec() for name, tool in self.items()}

    def filter(self, names: Optional[Iterable[str]]) -> "ToolRegistry":
        if names is None:
            return self
        wanted = list(dict.fromkeys(names))
        filtered = ToolRegistry()
        for name in wanted:
            if name not in self:
                raise KeyError(f"Unknown tool requested by runner: {name}")
            filtered.register(self[name])
        return filtered


class ConcreteToolAlias(ToolFamily):
    """Expose a concrete backend/tool name while reusing an existing family."""

    def __init__(
        self,
        name: str,
        target: ToolFamily,
        description: str,
        default_arguments: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self._target = target
        self.description = description
        self.paid = target.paid
        self.parallel_safe = target.parallel_safe
        self.default_action = target.default_action
        self.actions = target.actions
        self.default_arguments = default_arguments or {}

    def normalize_action(self, action: str) -> str:
        return self._target.normalize_action(action)

    def validate_arguments(self, action: str, arguments: Dict[str, Any]) -> Optional[str]:
        merged = {**self.default_arguments, **arguments}
        return self._target.validate_arguments(action, merged)

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        result = self._target.forward(action, {**self.default_arguments, **arguments}, timeout=timeout)
        result.tool_family = self.name
        return result


class ConfiguredToolWrapper(ToolFamily):
    """Apply per-benchmark YAML-facing schema and argument mapping to a tool."""

    def __init__(self, target: ToolFamily, settings: Mapping[str, Any]):
        self.name = target.name
        self._target = target
        self._settings = dict(settings or {})
        self.description = str(self._settings.get("description") or target.description)
        self.paid = target.paid
        self.parallel_safe = target.parallel_safe
        self.default_action = str(self._settings.get("default_action") or target.default_action)
        self.actions = target.actions
        self.default_arguments = self._dict_setting("defaults")
        self.argument_aliases = self._dict_setting("argument_aliases")
        self.action_aliases = self._dict_setting("action_aliases")
        raw_hidden_actions = self._settings.get("hidden_actions") or []
        if isinstance(raw_hidden_actions, str):
            raw_hidden_actions = [raw_hidden_actions]
        self.hidden_actions = {str(action) for action in raw_hidden_actions if str(action).strip()}

    def _dict_setting(self, key: str) -> Dict[str, Any]:
        value = self._settings.get(key) or {}
        return dict(value) if isinstance(value, Mapping) else {}

    def spec(self) -> Dict[str, Any]:
        spec = self._target.spec()
        spec["name"] = self.name
        spec["description"] = self.description
        spec["default_action"] = self.default_action
        if self._settings.get("canonical_tool"):
            spec["canonical_tool"] = self._settings["canonical_tool"]
        if self._settings.get("backend"):
            spec["backend"] = self._settings["backend"]
        configured_actions = self._settings.get("actions")
        if isinstance(configured_actions, Mapping):
            spec["default_action"] = self.default_action
            spec["actions"] = self._configured_action_specs(configured_actions)
        if self.hidden_actions:
            spec["actions"] = {
                name: action_spec
                for name, action_spec in (spec.get("actions") or {}).items()
                if name not in self.hidden_actions
            }
        return spec

    def _configured_action_specs(self, configured_actions: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        base_actions = self._target.spec().get("actions", {})
        for action_name, raw_action in configured_actions.items():
            if not isinstance(raw_action, Mapping):
                continue
            action_name = str(action_name)
            base_action = base_actions.get(self._target_action(action_name), {})
            action_spec: Dict[str, Any] = {
                "description": raw_action.get("description", base_action.get("description", "")),
                "required": list(raw_action.get("required", base_action.get("required", [])) or []),
                "optional": list(raw_action.get("optional", base_action.get("optional", [])) or []),
            }
            if isinstance(raw_action.get("parameters"), Mapping):
                action_spec["parameters"] = dict(raw_action["parameters"])
            elif isinstance(base_action.get("parameters"), Mapping):
                action_spec["parameters"] = dict(base_action["parameters"])
            if isinstance(raw_action.get("examples"), list):
                action_spec["examples"] = list(raw_action["examples"])
            elif isinstance(base_action.get("examples"), list):
                action_spec["examples"] = list(base_action["examples"])
            if isinstance(raw_action.get("defaults"), Mapping):
                action_spec["defaults"] = dict(raw_action["defaults"])
            specs[action_name] = action_spec
        return specs

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        if candidate == "default":
            candidate = self.default_action
        return self._target_action(candidate)

    def validate_arguments(self, action: str, arguments: Dict[str, Any]) -> Optional[str]:
        return self._target.validate_arguments(
            self._target_action(action),
            self._target_arguments(arguments),
        )

    def run(self, call: ToolCall, timeout: int = 30) -> ToolResult:
        visible_action = self.normalize_action(call.action)
        target_action = self._target_action(visible_action)
        target_arguments = self._target_arguments(call.arguments)
        target_call = ToolCall(
            name=self._target.name,
            action=target_action,
            arguments=target_arguments,
            id=call.id,
            rationale=call.rationale,
        )
        result = self._target.run(target_call, timeout=timeout)
        result.tool_family = self.name
        result.action = visible_action
        return result

    def reset_for_task(self) -> None:
        self._target.reset_for_task()

    def _target_action(self, visible_action: str) -> str:
        aliased = self.action_aliases.get(visible_action, visible_action)
        return self._target.normalize_action(aliased)

    def _target_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        mapped = dict(self.default_arguments)
        for key, value in (arguments or {}).items():
            target_key = str(self.argument_aliases.get(key, key))
            mapped[target_key] = value
        return mapped
