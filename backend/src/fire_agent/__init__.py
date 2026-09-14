"""FIRE single-agent framework.

FIRE = Financial Integrated Reasoning & Evidence.

A Flash-Searcher-aligned single-agent ReAct runtime (no DAG planning, no
periodic summary) called by benchmark-specific runners. The on-disk JSONL
schema mirrors Flash-Searcher's lean trajectory format (``{name, tool_calls,
obs, think}`` per step) plus a per-bench ``bench_extra`` block for whatever
metadata the bench-specific evaluator and aggregator need.
"""

from .config import ModelSettings, ToolSettings, load_fire_agent_env
from .llm import OpenAICompatibleChatModel
from .memory import ActionStep, AgentMemory, MemoryStep, PlanningStep, SystemPromptStep, TaskStep
from .renderers import (
    CellBBoxRenderer,
    LayoutJSONRenderer,
    TableHTMLRenderer,
    TOCTreeRenderer,
    render_document_parse_outputs,
)
from .runtime import FIREAgent
from .schemas import (
    AgentConfig,
    AgentResult,
    AgentState,
    AgentTurn,
    BenchTaskContext,
    ToolCall,
    ToolResult,
    TrajectoryStep,
)
from .tool_families import ToolActionSpec, ToolFamily, ToolRegistry, build_default_tool_registry

__all__ = [
    "ActionStep",
    "AgentConfig",
    "AgentMemory",
    "AgentResult",
    "AgentState",
    "AgentTurn",
    "BenchTaskContext",
    "CellBBoxRenderer",
    "FIREAgent",
    "LayoutJSONRenderer",
    "MemoryStep",
    "ModelSettings",
    "OpenAICompatibleChatModel",
    "PlanningStep",
    "SystemPromptStep",
    "TOCTreeRenderer",
    "TableHTMLRenderer",
    "TaskStep",
    "ToolActionSpec",
    "ToolCall",
    "ToolFamily",
    "ToolRegistry",
    "ToolResult",
    "ToolSettings",
    "TrajectoryStep",
    "build_default_tool_registry",
    "load_fire_agent_env",
    "render_document_parse_outputs",
]
