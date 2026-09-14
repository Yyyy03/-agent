"""Linear chat-style memory for the Flash-Searcher-aligned ReAct runtime.

This intentionally mirrors the spirit of ``FlashOAgents.memory`` (a single
``AgentMemory`` of ordered steps that can be rendered into OpenAI-style chat
messages), but without the DAG/summary infrastructure.

Each ``MemoryStep`` knows how to project itself into ``{role, content}``
messages that the LLM consumes. ``runtime.py`` keeps these alongside the
lean ``AgentState`` trajectory so the benchmark scoring path keeps working.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _message_chars(messages: List[Message]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


def _legacy_step_chars(step: MemoryStep) -> int:
    if isinstance(step, TaskStep):
        return len("New task:\n") + len(step.task or "")
    if isinstance(step, PlanningStep):
        return (
            len("Now write a short initial plan for the task above. Do not call tools yet.")
            + len("[PLAN]\n")
            + len((step.plan or "").strip())
        )
    if isinstance(step, ActionStep):
        assistant_payload: Dict[str, Any] = {
            "think": step.think or "",
            "tools": step.tool_calls or [],
        }
        total = len("Calling tools:\n") + len(_safe_json(assistant_payload))
        if step.observations:
            total += len(f"Tool calling observation (step {step.step_number}):\n")
            total += len(step.observations)
        if step.error:
            total += len(f"Error in step {step.step_number}: ")
            total += len(step.error)
            total += len("\nTake a different approach in your next step.")
        return total
    return _message_chars(step.to_messages())


@dataclass
class MemoryStep:
    def to_messages(self) -> List[Message]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = type(self).__name__
        return data


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self) -> List[Message]:
        return [{"role": "system", "content": self.system_prompt}]


@dataclass
class TaskStep(MemoryStep):
    task: str

    def to_messages(self) -> List[Message]:
        return [{"role": "user", "content": f"New task:\n{self.task}"}]


@dataclass
class PlanningStep(MemoryStep):
    plan: str

    def to_messages(self) -> List[Message]:
        return [
            {
                "role": "user",
                "content": "Now write a short initial plan for the task above. Do not call tools yet.",
            },
            {
                "role": "assistant",
                "content": f"[PLAN]\n{self.plan.strip()}",
            },
        ]


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    think: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: str = ""
    error: Optional[str] = None

    def to_messages(self) -> List[Message]:
        messages: List[Message] = []
        assistant_payload: Dict[str, Any] = {
            "think": self.think or "",
            "tools": self.tool_calls or [],
        }
        messages.append(
            {
                "role": "assistant",
                "content": "Calling tools:\n" + _safe_json(assistant_payload),
            }
        )
        if self.observations:
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool calling observation (step {self.step_number}):\n{self.observations}",
                }
            )
        if self.error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error in step {self.step_number}: {self.error}\n"
                        "Take a different approach in your next step."
                    ),
                }
            )
        return messages


class AgentMemory:
    """Ordered linear memory of steps that renders to a chat-message list."""

    def __init__(self, system_prompt: str):
        self.original_system_prompt = system_prompt
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.original_task = ""
        self.steps: List[MemoryStep] = []
        self.action_steps: List[ActionStep] = []
        self._legacy_chars_without_instruction = len(system_prompt or "")

    def append(self, step: MemoryStep) -> None:
        if isinstance(step, TaskStep) and step.task:
            self.original_task = step.task
        if isinstance(step, ActionStep):
            self.action_steps.append(step)
        self._legacy_chars_without_instruction += _legacy_step_chars(step)
        self.steps.append(step)

    def reset(self) -> None:
        self.steps = []
        self.action_steps = []
        self._legacy_chars_without_instruction = len(self.original_system_prompt or "")

    def estimated_prompt_chars(self, instruction: str = "") -> int:
        return self._legacy_chars_without_instruction + len(instruction or "")

    def to_messages(self) -> List[Message]:
        messages = self.system_prompt.to_messages()
        for step in self.steps:
            messages.extend(step.to_messages())
        return messages

    def dict_list(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.steps]
