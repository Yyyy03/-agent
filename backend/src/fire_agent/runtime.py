"""Flash-Searcher-aligned ReAct single-agent runtime.

Single ReAct loop modeled after Flash-Searcher's ``ToolCallingAgent``, with
two explicit divergences:

- No DAG planning. There is one initial plan; no re-planning.
- No periodic summary. Memory is kept linear and unsummarized.

Each trajectory step is stored in the lean Flash-Searcher shape:
``{"name": "plan"|"action", "tool_calls": [...], "obs": <str>, "think": <str>}``.
Structured tool results are formatted into a single LLM-facing ``obs`` string
and only kept on disk when ``FIRE_AGENT_KEEP_TOOL_RESULTS=1``.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
import traceback
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .api_balance_circuit_breaker import GlobalExperimentAbort
from .context import (
    TASK_STATE_SCHEMA_INSTRUCTION,
    TASK_STATE_VIEW_KEYS,
    ContextAssembler,
    build_artifact_preview,
    canonicalize_task_state,
    complete_task_state_snapshot,
    json_compact,
    messages_char_count,
    merge_task_state,
    record_answer_pins,
    record_evidence_items,
    record_action_ledger,
    should_render_task_state_view,
    should_use_context_system,
    task_state_support_audit,
    task_state_snapshot_issues,
    task_state_prompt_view,
    tool_observation_schema,
)
from .context_compression import (
    build_context_compression_messages,
    context_packet_object_from_text,
    is_direct_memory_sections_json,
    parse_context_compression_packet_with_reason,
    should_attempt_context_compression,
)
from .context_cache import ContextCompressionCacheManager
from .llm import parse_json_object
from .memory import ActionStep, AgentMemory, PlanningStep, TaskStep
from .prompts import (
    CANONICAL_STEP_INSTRUCTION,
    CANONICAL_SYSTEM_PROMPT,
    FINANCE_AGENT_BENCH_PLANNING_CONSTRAINTS,
    FINANCE_AGENT_BENCH_SYSTEM_CONSTRAINTS,
    INITIAL_PLANNING_PROMPT,
)
from .schemas import (
    AgentConfig,
    AgentResult,
    AgentState,
    AgentTurn,
    AuxiliaryTurn,
    BenchTaskContext,
    ToolCall,
    ToolResult,
    TrajectoryStep,
    compact_text,
    keep_reasoning_content,
    keep_step_task_state,
    keep_tool_results,
)
from .tool_families import ToolFamily, build_default_tool_registry


FINAL_ANSWER_TOOL_NAME = "final_answer"
FINAL_ANSWER_TOOL_ALIASES = {FINAL_ANSWER_TOOL_NAME}
_TASK_STATE_DISCARD_RECORDED_KEY = "__fire_runtime_task_state_discard_recorded__"
MARKET_DATA_TOOL_NAME = "market_data"
MARKET_DATA_ALWAYS_NATIVE_TOOLS = (
    "market_data_capabilities",
)
MARKET_DATA_TOOL_UNLOCKS_KEY = "market_data_tool_unlocks"
MARKET_DATA_DYNAMIC_TOP_K = 8
MARKET_DATA_SOURCE_CATALOG_NATIVE_TOOL = "market_data_source_catalog"
MARKET_DATA_MARKET_TABLE_NATIVE_TOOL = "market_data_market_table"
FINANCE_AGENT_BENCH_NAME = "financeagentbench"


class FIREAgent:
    """Flash-Searcher-style ReAct single-agent runtime (no DAG, no summary)."""

    def __init__(
        self,
        model: Optional[Any] = None,
        auxiliary_model: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, ToolFamily]] = None,
    ):
        self.model = model
        self.auxiliary_model = auxiliary_model
        self.config = config or AgentConfig()
        self.tools = tools if tools is not None else build_default_tool_registry()
        self.context_assembler = ContextAssembler(self.config)
        self.context_cache = ContextCompressionCacheManager(
            self.config,
            self.context_assembler,
            task_state_fingerprint_fn=self._task_state_progress_fingerprint,
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self, task_context: BenchTaskContext) -> AgentResult:
        self._reset_tools_for_task()
        state = AgentState(task_context=task_context, config=self.config)
        state.generation = self._trace_generation_config()
        state.tools = self._trace_tool_schemas()
        state.context_stats["runtime_stage"] = "initializing"

        if not (self.config.use_model_for_roles and self.model is not None):
            return AgentResult(
                "",
                "error",
                state,
                error="FIRE runtime requires a configured chat model.",
            )

        memory = AgentMemory(system_prompt=self._build_system_prompt(state))
        memory.append(TaskStep(task=state.task_context.task_prompt))

        try:
            if self._should_skip_initial_planning(state):
                state.context_stats["initial_planning_skipped"] = "final_answer_only_round"
            else:
                state.context_stats["runtime_stage"] = "initial_planning"
                self._initial_planning_step(state, memory)

            for step_number in range(1, self.config.max_steps + 1):
                state.context_stats["runtime_stage"] = "action_step"
                state.context_stats["last_action_step_number"] = step_number
                final_answer = self._action_step(state, memory, step_number)
                if final_answer is not None:
                    state.context_stats["runtime_stage"] = "completed"
                    state.context_stats["finish_type"] = "natural_final"
                    state.context_stats["natural_final"] = True
                    state.context_stats["terminal_fallback"] = False
                    state.context_stats["max_steps_reached"] = False
                    return AgentResult(final_answer, "success", state)
                breaker = state.context_stats.get("model_error_circuit_breaker")
                if isinstance(breaker, dict) and breaker.get("triggered"):
                    state.context_stats["runtime_stage"] = "model_error_aborted"
                    error_msg = str(
                        breaker.get("last_error")
                        or breaker.get("signature")
                        or "Repeated model error circuit breaker triggered."
                    )
                    return AgentResult("", "error", state, error=error_msg)

            if self._is_final_answer_only_round(state):
                state.context_stats["runtime_stage"] = "single_turn_protocol_retry"
                state.context_stats["max_steps_reached"] = True
                retry_answer, retry_status = self._single_turn_protocol_retry(state, memory)
                if retry_answer:
                    state.context_stats["runtime_stage"] = "completed"
                    state.context_stats["finish_type"] = "protocol_retry"
                    state.context_stats["natural_final"] = False
                    state.context_stats["terminal_fallback"] = False
                    state.context_stats["protocol_retry_success"] = True
                    return AgentResult(retry_answer, "success", state)
                state.context_stats["runtime_stage"] = retry_status
                state.context_stats["finish_type"] = retry_status
                state.context_stats["natural_final"] = False
                state.context_stats["terminal_fallback"] = False
                state.context_stats["protocol_retry_success"] = False
                return AgentResult("", retry_status, state)

            state.context_stats["runtime_stage"] = "terminal_best_effort"
            state.context_stats["max_steps_reached"] = True
            fallback_answer = self._terminal_best_effort_answer(state, memory)
            if fallback_answer:
                state.context_stats["runtime_stage"] = "completed"
                state.context_stats["finish_type"] = "terminal_fallback"
                state.context_stats["natural_final"] = False
                state.context_stats["terminal_fallback"] = True
                return AgentResult(fallback_answer, "success", state)
            state.context_stats["runtime_stage"] = "max_steps"
            state.context_stats["finish_type"] = "max_steps_no_answer"
            state.context_stats["natural_final"] = False
            state.context_stats["terminal_fallback"] = False
            return AgentResult("", "max_steps", state)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            stage = str(state.context_stats.get("runtime_stage") or "unknown")
            trace = traceback.format_exc()
            state.context_stats["runtime_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": stage,
                "last_action_step_number": state.context_stats.get("last_action_step_number"),
                "traceback": compact_text(trace, 12000),
            }
            if not state.trajectory or state.trajectory[-1].error != error_msg:
                state.add_step(
                    TrajectoryStep(
                        name="action",
                        error=f"{error_msg} | stage={stage}",
                    )
                )
            return AgentResult("", "error", state, error=error_msg)

    # ------------------------------------------------------------------
    # ReAct loop pieces
    # ------------------------------------------------------------------
    def _reset_tools_for_task(self) -> None:
        for tool in self.tools.values():
            tool.reset_for_task()

    def _should_skip_initial_planning(self, state: AgentState) -> bool:
        return self._is_final_answer_only_round(state)

    def _is_final_answer_only_round(self, state: AgentState) -> bool:
        family = str(state.task_context.task_family or "").strip().lower()
        return not self.tools and (
            bool(getattr(state.config, "final_answer_only_round", False))
            or family == "single_turn_qa"
            or int(getattr(state.config, "max_steps", 0) or 0) <= 1
        )

    def _is_single_turn_reasoning_pass(self, state: AgentState, step_number: int) -> bool:
        return (
            step_number == 1
            and self._is_final_answer_only_round(state)
            and bool(getattr(state.config, "single_turn_reasoning_pass", False))
            and self._model_thinking_enabled()
        )

    def _single_turn_reasoning_answer(
        self,
        parsed: Dict[str, Any],
        content: Any,
    ) -> str:
        if parsed:
            for key in ("answer", "final_answer", "final_result", "result", "text"):
                value = parsed.get(key)
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value).strip()
            return ""
        answer = str(content or "").strip()
        return "" if self._is_terminal_non_answer_text(answer) else answer

    def _initial_planning_step(self, state: AgentState, memory: AgentMemory) -> str:
        messages = memory.to_messages() + [{"role": "user", "content": self._initial_planning_prompt(state)}]
        try:
            response = self._call_model_with_retries(
                state,
                "planning",
                messages,
                step_number=0,
                render_mode="planning",
                tools=[],
                tool_choice=None,
            )
            content = getattr(response, "content", None)
            if content is None:
                content = str(response or "")
            reasoning_content = self._response_reasoning_content(response)
            plan_text = str(content).strip()
        except Exception as exc:
            state.add_step(
                TrajectoryStep(name="plan", value="", error=f"{type(exc).__name__}: {exc}")
            )
            return ""

        if plan_text:
            memory.append(PlanningStep(plan=plan_text))
        self._record_agent_turn(
            state,
            turn_type="plan",
            turn_id=self._turn_id("plan", 0),
            input_messages=messages,
            output_message=self._trace_assistant_message(
                content=plan_text,
                reasoning_content=reasoning_content,
            ),
            observations=[],
            tools=[],
        )
        state.add_step(TrajectoryStep(name="plan", value=plan_text, reasoning_content=reasoning_content))
        return plan_text

    @staticmethod
    def _trace_task_state(state: AgentState) -> Optional[Dict[str, Any]]:
        return state.current_task_state if keep_step_task_state() else None

    def _action_step(
        self, state: AgentState, memory: AgentMemory, step_number: int
    ) -> Optional[str]:
        context_mode = str(state.config.context_mode or "").strip().lower()
        if context_mode == "legacy":
            render_hint = "legacy"
        elif context_mode in {"full", "full_compatible", "auto", "hybrid", ""}:
            render_hint = "full_compatible"
        else:
            render_hint = "packet"
        task_state_block = self._task_state_block(
            state,
            view=should_render_task_state_view(state.config, render_hint),
            inline=render_hint in {"legacy", "full_compatible"},
        )
        instruction = self._step_instruction(state, step_number, task_state_block=task_state_block)
        reasoning_pass = self._is_single_turn_reasoning_pass(state, step_number)
        reasoning_pass_instruction = (
            "\n\nNative thinking pass: reason through the current request before the final "
            "answer protocol runs. Do not call final_answer in this pass. Preserve the model's "
            "native reasoning content; the runtime will normalize the answer after this pass and "
            "request the final_answer tool separately only if needed."
        )
        if reasoning_pass:
            instruction += reasoning_pass_instruction
            state.context_stats["single_turn_reasoning_pass"] = {
                "enabled": True,
                "step_number": step_number,
            }
        step_tools = self._select_step_tools(state, memory, instruction)
        render_mode, legacy_chars = self.context_assembler.select_render_mode(
            state,
            memory,
            instruction,
            tools=step_tools,
        )
        if render_mode != render_hint:
            task_state_block = self._task_state_block(
                state,
                view=should_render_task_state_view(state.config, render_mode),
                inline=render_mode in {"legacy", "full_compatible"},
            )
            instruction = self._step_instruction(state, step_number, task_state_block=task_state_block)
            if reasoning_pass:
                instruction += reasoning_pass_instruction
            step_tools = self._select_step_tools(state, memory, instruction)
            if not (render_hint == "full_compatible" and render_mode in {"lean_packet", "compact_packet"}):
                render_mode, legacy_chars = self.context_assembler.select_render_mode(
                    state,
                    memory,
                    instruction,
                    tools=step_tools,
                )
        packet_override = self._maybe_compress_context_packet(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            step_number=step_number,
            tools=step_tools,
        )
        raw_messages = self.context_assembler.action_messages(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            packet_override=packet_override,
            tools=step_tools,
        )
        messages = self._model_input_messages(self._canonicalize_turn_input_messages(state, raw_messages))
        if reasoning_pass:
            tool_choice: Optional[Any] = "none"
        elif self._is_final_answer_only_round(state):
            tool_choice = self._final_answer_tool_choice_for_model()
        else:
            tool_choice = None

        try:
            response = self._call_model_with_retries(
                state,
                "action",
                messages,
                step_number=step_number,
                render_mode=render_mode,
                tools=step_tools,
                tool_choice=tool_choice,
            )
            content = getattr(response, "content", None)
            if content is None:
                content = str(response or "")
            reasoning_content = self._response_reasoning_content(response)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            state.add_step(TrajectoryStep(name="action", error=error_msg))
            memory.append(
                ActionStep(
                    step_number=step_number,
                    think="",
                    tool_calls=[],
                    observations="",
                    error=error_msg,
                )
            )
            self._record_action_model_error(state, error_msg, step_number)
            if self._is_model_protocol_error(error_msg):
                self._trigger_model_protocol_error(state, error_msg, step_number)
            return None

        self._clear_action_model_error_streak(state)

        state_task_state_before_response = deepcopy(state.current_task_state)
        native_tool_calls = self._response_tool_calls(response)
        parsed, think, canonical_content = self._canonical_action_content(
            state,
            content,
            step_number=step_number,
        )
        proposed = self._tool_calls_to_proposed(native_tool_calls, state=state) if native_tool_calls else []
        self._record_task_state_progress_signal(state, step_number)

        final_call = self._extract_final_answer(proposed)
        reasoning_pass_answer = False
        if reasoning_pass and final_call is None:
            answer_from_reasoning = self._single_turn_reasoning_answer(parsed, content)
            if answer_from_reasoning:
                final_call = {
                    "name": FINAL_ANSWER_TOOL_NAME,
                    "action": "default",
                    "arguments": {"answer": answer_from_reasoning},
                }
                reasoning_pass_answer = True
                state.context_stats["single_turn_reasoning_pass"].update(
                    {
                        "answer_reused": True,
                        "answer_chars": len(answer_from_reasoning),
                    }
                )
        if final_call is not None:
            answer_text = self._coerce_final_answer_value(final_call)
            final_tool_call = self._final_answer_tool_call(answer_text, final_call)
            final_call_dict = {
                "name": FINAL_ANSWER_TOOL_NAME,
                "action": "default",
                "arguments": {"answer": answer_text},
            }
            if self._is_empty_final_answer_text(answer_text):
                observation = "[policy.final_answer_empty] final_answer was called with an empty answer. Continue gathering evidence or provide a non-empty final answer."
                policy_result = self._policy_tool_result(final_tool_call, observation, code="policy.final_answer_empty")
                observation_text, prompt_observation_text, observation_messages = self._format_observation_pair(
                    state,
                    [final_tool_call],
                    [policy_result],
                    step_number=step_number,
                    think=think,
                )
                policy_feedback_messages = [
                    self._trace_policy_feedback(
                        final_tool_call,
                        policy_result,
                        content=prompt_observation_text,
                    )
                ]
                state.add_step(
                    TrajectoryStep(
                        name="action",
                        tool_calls=[final_call_dict],
                        obs=observation_text,
                        think=think,
                        task_state=self._trace_task_state(state),
                        reasoning_content=reasoning_content,
                    )
                )
                self._record_agent_turn(
                    state,
                    turn_type="action",
                    turn_id=self._turn_id("action", step_number),
                    input_messages=messages,
                    output_message=self._trace_assistant_message(
                        content=canonical_content,
                        reasoning_content=reasoning_content,
                        step_number=step_number,
                    ),
                    observations=policy_feedback_messages,
                    tools=step_tools,
                    include_in_sft=False,
                )
                memory.append(
                    ActionStep(
                        step_number=step_number,
                        think=think,
                        tool_calls=[final_tool_call.to_dict()],
                        observations=prompt_observation_text,
                    )
                )
                return None
            self._record_packet_final_soft_audit(state, render_mode)
            if not self._is_final_answer_only_round(state):
                self._discard_terminal_content_task_state(
                    state,
                    parsed,
                    prior_task_state=state_task_state_before_response,
                    reason="final_answer_tool_call",
                )
            canonical_content = self._json_compact({"think": think, "task_state": {}})
            state.add_step(
                TrajectoryStep(
                    name="action",
                    tool_calls=[final_call_dict],
                    obs=f"[final_answer] {answer_text[:200]}",
                    think=think,
                    task_state=self._trace_task_state(state),
                    reasoning_content=reasoning_content,
                )
            )
            self._record_agent_turn(
                state,
                turn_type="final",
                turn_id=self._turn_id("final", step_number),
                input_messages=messages,
                output_message=self._trace_assistant_message(
                    content=canonical_content,
                    reasoning_content=reasoning_content,
                    tool_calls=[final_tool_call],
                    step_number=step_number,
                ),
                observations=[],
                tools=step_tools,
                include_in_sft=(
                    not reasoning_pass_answer
                    and self._final_answer_uses_canonical_answer_key(final_call)
                ),
            )
            memory.append(
                ActionStep(
                    step_number=step_number,
                    think=think,
                    tool_calls=[final_call_dict],
                    observations=f"Final answer received: {answer_text[:200]}",
                )
            )
            return answer_text

        non_final_proposed = [
            item
            for item in proposed
            if not (
                isinstance(item, dict)
                and str(item.get("name", "")).strip()
                in FINAL_ANSWER_TOOL_ALIASES
            )
        ]
        round_calls, executable_indexes, policy_results = self._validate_tool_calls(
            state,
            non_final_proposed,
            step_number,
            tools=step_tools,
        )
        model_round_call_count = len(round_calls)
        executable_calls = [round_calls[index] for index in executable_indexes]
        executed_results: List[ToolResult] = []
        if executable_calls:
            executed_results = self._execute_tool_round(state, executable_calls)
            self._record_tool_progress_signals(state, executable_calls, executed_results, step_number)
            self._record_market_data_tool_unlocks(state, executable_calls, executed_results, step_number)
            self._record_web_search_soft_fuse_results(state, executable_calls, executed_results, step_number)
            self._record_tool_failure_fuses(state, executable_calls, executed_results, step_number)

        result_by_index: Dict[int, ToolResult] = dict(policy_results)
        for index, result in zip(executable_indexes, executed_results):
            result_by_index[index] = result
        round_results = [
            result_by_index.get(index)
            or self._policy_tool_result(
                call,
                f"[policy.missing_tool_result] no result was produced for {self._tool_call_display_name(call)}.",
                code="policy.missing_tool_result",
            )
            for index, call in enumerate(round_calls)
        ]
        self._append_structured_table_auto_follow(
            state,
            round_calls,
            round_results,
            executed_count=len(executable_calls),
            step_number=step_number,
        )

        observation_text, prompt_observation_text, observation_messages = self._format_observation_pair(
            state,
            round_calls,
            round_results,
            step_number=step_number,
            think=think,
        )
        protocol_calls, protocol_observations, policy_feedback = self._protocol_messages_for_results(
            round_calls[:model_round_call_count],
            round_results[:model_round_call_count],
            observation_messages[:model_round_call_count],
        )
        runtime_feedback = self._runtime_feedback_messages(
            round_calls[model_round_call_count:],
            round_results[model_round_call_count:],
            observation_messages[model_round_call_count:],
        )
        protocol_observations.extend(policy_feedback)
        protocol_observations.extend(runtime_feedback)
        step = TrajectoryStep(
            name="action",
            tool_calls=[call.to_lean_dict() for call in executable_calls],
            obs=observation_text,
            think=think,
            task_state=self._trace_task_state(state),
            reasoning_content=reasoning_content,
        )
        if keep_tool_results():
            step.raw_tool_results = [result.to_dict() for result in executed_results]
        state.add_step(step)
        self._record_agent_turn(
            state,
            turn_type="action",
            turn_id=self._turn_id("action", step_number),
            input_messages=messages,
            output_message=self._trace_assistant_message(
                content=canonical_content,
                reasoning_content=reasoning_content,
                tool_calls=protocol_calls,
                step_number=step_number,
            ),
            observations=protocol_observations,
            tools=step_tools,
            include_in_sft=not self._is_final_answer_only_round(state),
        )
        memory.append(
            ActionStep(
            step_number=step_number,
            think=think,
            tool_calls=[call.to_dict() for call in round_calls[:model_round_call_count]],
            observations=prompt_observation_text,
        )
        )
        return None

    @staticmethod
    def _native_final_answer_protocol_instruction(answer_descriptor: str) -> str:
        return (
            "Emit exactly one OpenAI-native final_answer tool-call now. "
            f"Put only the {answer_descriptor} in final_answer.arguments.answer. "
            "Do not write the final answer in assistant.content. "
            "Do not output the final answer as plain text or markdown. "
            "Do not put the answer only in reasoning_content."
        )

    def _single_turn_protocol_retry(self, state: AgentState, memory: AgentMemory) -> Tuple[str, str]:
        retry_stats: Dict[str, Any] = {
            "attempted": True,
            "max_retries": 1,
            "step_number": state.config.max_steps + 1,
        }
        state.context_stats["protocol_retry"] = retry_stats
        instruction = (
            "Protocol retry: the previous single-turn response did not produce a usable final_answer. "
            f"{self._native_final_answer_protocol_instruction('benchmark answer atom/value')} "
            "Use only the original task prompt and any already visible context; do not use other tools. "
            "Do not include chain-of-thought or markdown inside the answer string."
        )
        final_answer_tools = self._final_answer_tool_schemas()
        render_mode, legacy_chars = self.context_assembler.select_render_mode(
            state,
            memory,
            instruction,
            tools=final_answer_tools,
        )
        raw_messages = self.context_assembler.action_messages(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            packet_override=None,
            tools=final_answer_tools,
        )
        messages = self._model_input_messages_without_tool_protocol(
            self._canonicalize_turn_input_messages(state, raw_messages)
        )
        try:
            response = self._call_model_with_retries(
                state,
                "protocol_retry",
                messages,
                step_number=state.config.max_steps + 1,
                tools=final_answer_tools,
                tool_choice=self._final_answer_tool_choice_for_model(),
            )
            content = getattr(response, "content", None)
            if content is None:
                content = str(response or "")
            reasoning_content = self._response_reasoning_content(response)
            if reasoning_content.strip():
                retry_stats["reasoning_chars"] = len(reasoning_content)
                retry_stats["reasoning_excerpt"] = compact_text(reasoning_content, 1000)
        except Exception as exc:
            retry_stats.update(
                {
                    "used": False,
                    "error": compact_text(f"{type(exc).__name__}: {exc}", 1000),
                }
            )
            state.add_step(
                TrajectoryStep(name="action", error=f"protocol_retry {type(exc).__name__}: {exc}")
            )
            return "", "protocol_no_answer"

        answer_text, final_call, retry_think, canonical_content = self._terminal_answer_payload(
            state,
            response,
            content,
            step_number=state.config.max_steps + 1,
        )
        if final_call is None:
            retry_stats.update({"used": False, "failure": "missing_final_answer_tool_call"})
            return "", "protocol_no_answer"
        if self._is_empty_final_answer_text(answer_text):
            retry_stats.update({"used": False, "failure": "empty_final_answer"})
            return "", "empty_final_answer"
        if self._is_terminal_non_answer_text(answer_text):
            retry_stats.update({"used": False, "failure": "non_answer_final_answer"})
            return "", "protocol_no_answer"

        retry_stats.update({"used": True, "answer_chars": len(answer_text)})
        final_call_dict = {
            "name": FINAL_ANSWER_TOOL_NAME,
            "action": "default",
            "arguments": {"answer": answer_text},
        }
        final_tool_call = self._final_answer_tool_call(answer_text, final_call)
        self._record_agent_turn(
            state,
            turn_type="final",
            turn_id=self._turn_id("protocol_retry", state.config.max_steps + 1),
            input_messages=messages,
            output_message=self._trace_assistant_message(
                content=canonical_content,
                reasoning_content=reasoning_content,
                tool_calls=[final_tool_call],
                step_number=state.config.max_steps + 1,
            ),
            observations=[],
            tools=final_answer_tools,
            include_in_sft=self._final_answer_uses_canonical_answer_key(final_call),
        )
        retry_stats["tool_calls"] = [final_call_dict]
        return answer_text, "success"

    def _terminal_best_effort_answer(self, state: AgentState, memory: AgentMemory) -> str:
        terminal_stats: Dict[str, Any] = {
            "attempted": True,
            "max_steps": state.config.max_steps,
            "step_number": state.config.max_steps + 1,
        }
        state.context_stats["terminal_summary"] = terminal_stats
        instruction = (
            "The step limit has been reached. "
            f"{self._native_final_answer_protocol_instruction('final answer')} "
            "final_answer is not an evidence tool. Do not search, read, fetch, or calculate further. "
            "Use the task prompt, observed evidence, and tool observations already in context. "
            "For each requested value, answer from the strongest observed candidate: supported/candidate slots, "
            "calculator results, table rows, snippets, answer-critical pins, or the task-provided information. "
            "If a slot is marked missing or unresolved, still choose the best observed candidate first; "
            "state a source gap only when no observed candidate exists for that slot. "
            "Do not invent facts or arbitrary numbers. Keep any caveat after the candidate answer."
        )
        final_answer_tools = self._final_answer_tool_schemas()
        render_mode, legacy_chars = self.context_assembler.select_render_mode(
            state,
            memory,
            instruction,
            tools=final_answer_tools,
        )
        packet_override = self._maybe_compress_context_packet(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            step_number=state.config.max_steps + 1,
            tools=final_answer_tools,
        )
        raw_messages = self.context_assembler.action_messages(
            state,
            memory,
            instruction,
            render_mode=render_mode,
            legacy_chars=legacy_chars,
            packet_override=packet_override,
            tools=final_answer_tools,
        )
        messages = self._model_input_messages_without_tool_protocol(
            self._canonicalize_turn_input_messages(state, raw_messages)
        )
        try:
            response = self._call_model_with_retries(
                state,
                "terminal_summary",
                messages,
                step_number=state.config.max_steps + 1,
                tools=final_answer_tools,
                # Use generic required rather than exact named-tool forcing:
                # Qwen/vLLM can still emit private reasoning before the native
                # final_answer call, while the response must end as tool_calls.
                tool_choice=self._final_answer_tool_choice_for_model(),
            )
            content = getattr(response, "content", None)
            if content is None:
                content = str(response or "")
            reasoning_content = self._response_reasoning_content(response)
            if reasoning_content.strip():
                terminal_stats["reasoning_chars"] = len(reasoning_content)
                terminal_stats["reasoning_excerpt"] = compact_text(reasoning_content, 1000)
        except Exception as exc:
            terminal_stats.update(
                {
                    "used": False,
                    "error": compact_text(f"{type(exc).__name__}: {exc}", 1000),
                }
            )
            state.add_step(
                TrajectoryStep(name="action", error=f"terminal_summary {type(exc).__name__}: {exc}")
            )
            return ""

        answer_text, final_call, terminal_think, canonical_content = self._terminal_answer_payload(
            state,
            response,
            content,
            step_number=state.config.max_steps + 1,
        )
        repair_reason = self._terminal_answer_repair_reason(
            answer_text,
            final_call,
            require_native_final_answer=False,
        )
        if repair_reason:
            terminal_stats.update({"repair_attempted": True, "repair_reason": repair_reason})
            repair_instruction = (
                "The previous terminal turn did not produce a usable final_answer. "
                f"{self._native_final_answer_protocol_instruction('final answer')} "
                "Use only the task prompt, observed evidence, and tool observations already in context; "
                "do not search, read, fetch, or calculate further. "
                "Prefer the strongest observed candidate from slots, calculator results, table rows, snippets, pins, "
                "or task-provided information. "
                "If the context says missing or unresolved, answer from those candidates when any exist; "
                "use a source-gap answer only when the trajectory contains no candidate for the requested slot. "
                "Do not invent facts or arbitrary numbers. A supported numeric 0 is valid; a blank answer is not."
            )
            repair_render_mode, repair_legacy_chars = self.context_assembler.select_render_mode(
                state,
                memory,
                repair_instruction,
                tools=final_answer_tools,
            )
            repair_packet_override = self._maybe_compress_context_packet(
                state,
                memory,
                repair_instruction,
                render_mode=repair_render_mode,
                legacy_chars=repair_legacy_chars,
                step_number=state.config.max_steps + 1,
                tools=final_answer_tools,
            )
            repair_raw_messages = self.context_assembler.action_messages(
                state,
                memory,
                repair_instruction,
                render_mode=repair_render_mode,
                legacy_chars=repair_legacy_chars,
                packet_override=repair_packet_override,
                tools=final_answer_tools,
            )
            repair_messages = self._model_input_messages_without_tool_protocol(
                self._canonicalize_turn_input_messages(state, repair_raw_messages)
            )
            try:
                repair_response = self._call_model_with_retries(
                    state,
                    "terminal_summary_repair",
                    repair_messages,
                    step_number=state.config.max_steps + 1,
                    tools=final_answer_tools,
                    tool_choice=self._final_answer_tool_choice_for_model(),
                )
                repair_content = getattr(repair_response, "content", None)
                if repair_content is None:
                    repair_content = str(repair_response or "")
                response = repair_response
                content = repair_content
                reasoning_content = self._response_reasoning_content(repair_response)
                if reasoning_content.strip():
                    terminal_stats["repair_reasoning_chars"] = len(reasoning_content)
                    terminal_stats["repair_reasoning_excerpt"] = compact_text(reasoning_content, 1000)
                messages = repair_messages
                answer_text, final_call, terminal_think, canonical_content = self._terminal_answer_payload(
                    state,
                    repair_response,
                    repair_content,
                    step_number=state.config.max_steps + 1,
                )
            except Exception as exc:
                terminal_stats.update(
                    {
                        "repair_error": compact_text(f"{type(exc).__name__}: {exc}", 1000),
                    }
                )

        final_repair_reason = self._terminal_answer_repair_reason(
            answer_text,
            final_call,
            require_native_final_answer=False,
        )
        if final_repair_reason:
            terminal_stats.update(
                {
                    "used": False,
                    final_repair_reason: True,
                }
            )
            return ""

        self._record_packet_final_soft_audit(
            state,
            str(state.context_stats.get("last_render_mode") or ""),
        )

        if final_call is None:
            plain_fallback_think = "Step limit reached; used terminal plain-text answer fallback."
            terminal_stats.update(
                {
                    "used": True,
                    "answer_chars": len(answer_text),
                    "plain_text_fallback": True,
                    "sft_excluded": True,
                }
            )
            state.add_step(
                TrajectoryStep(
                    name="action",
                    tool_calls=[],
                    obs=f"[terminal_summary.plain_answer] {answer_text[:200]}",
                    think=plain_fallback_think,
                    task_state=self._trace_task_state(state),
                    reasoning_content=reasoning_content,
                )
            )
            memory.append(
                ActionStep(
                    step_number=state.config.max_steps + 1,
                    think=plain_fallback_think,
                    tool_calls=[],
                    observations=f"Terminal plain-text answer accepted: {answer_text[:200]}",
                )
            )
            return answer_text

        terminal_stats.update(
            {
                "used": True,
                "answer_chars": len(answer_text),
            }
        )
        final_call_dict = {
            "name": FINAL_ANSWER_TOOL_NAME,
            "action": "default",
            "arguments": {"answer": answer_text},
        }
        final_tool_call = self._final_answer_tool_call(answer_text, final_call)
        state.add_step(
            TrajectoryStep(
                name="action",
                tool_calls=[final_call_dict],
                obs=f"[terminal_summary.final_answer] {answer_text[:200]}",
                think=terminal_think or "Step limit reached; generated terminal final_answer.",
                task_state=self._trace_task_state(state),
                reasoning_content=reasoning_content,
            )
        )
        self._record_agent_turn(
            state,
            turn_type="final",
            turn_id=self._turn_id("terminal", state.config.max_steps + 1),
            input_messages=messages,
            output_message=self._trace_assistant_message(
                content=canonical_content,
                reasoning_content=reasoning_content,
                tool_calls=[final_tool_call],
                step_number=state.config.max_steps + 1,
            ),
            observations=[],
            tools=final_answer_tools,
            include_in_sft=self._final_answer_uses_canonical_answer_key(final_call),
        )
        memory.append(
            ActionStep(
                step_number=state.config.max_steps + 1,
                think=terminal_think or "Step limit reached; generated terminal final_answer.",
                tool_calls=[final_call_dict],
                observations=f"Final answer received: {answer_text[:200]}",
            )
        )
        return answer_text

    def _terminal_answer_payload(
        self,
        state: AgentState,
        response: Any,
        content: Any,
        *,
        step_number: int,
    ) -> Tuple[str, Optional[Dict[str, Any]], str, str]:
        native_tool_calls = self._response_tool_calls(response)
        proposed = self._tool_calls_to_proposed(native_tool_calls, state=state) if native_tool_calls else []
        final_call = self._extract_final_answer(proposed)
        state_task_state_before_response = deepcopy(state.current_task_state)
        parsed, terminal_think, canonical_content = self._canonical_action_content(
            state,
            content,
            step_number=step_number,
        )
        answer_text = ""
        if final_call is not None:
            answer_text = self._coerce_final_answer_value(final_call)
            if not self._is_final_answer_only_round(state):
                self._discard_terminal_content_task_state(
                    state,
                    parsed,
                    prior_task_state=state_task_state_before_response,
                    reason="terminal_final_answer_tool_call",
                )
            canonical_content = self._json_compact({"think": terminal_think, "task_state": {}})
        elif not self._is_final_answer_only_round(state) and not parsed:
            answer_text = str(content or "").strip()
        return answer_text, final_call, terminal_think, canonical_content

    def _terminal_answer_repair_reason(
        self,
        answer_text: Any,
        final_call: Optional[Dict[str, Any]] = None,
        *,
        require_native_final_answer: bool = True,
    ) -> str:
        if require_native_final_answer and final_call is None:
            return "missing_final_answer_tool_call"
        if self._is_empty_final_answer_text(answer_text):
            return "empty_answer"
        if self._is_terminal_non_answer_text(answer_text):
            return "non_answer"
        return ""

    def _call_model_with_retries(
        self,
        state: AgentState,
        role: str,
        messages: List[Dict[str, Any]],
        *,
        step_number: Optional[int] = None,
        render_mode: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Any:
        selected_model = self.auxiliary_model if role == "context_compression" and self.auxiliary_model is not None else self.model
        response = self._invoke_model_with_retries(
            messages, tools=tools, tool_choice=tool_choice, model=selected_model
        )
        state.record_usage(role, self._extract_usage(response))
        return response

    @staticmethod
    def _response_reasoning_content(response: Any) -> str:
        value = getattr(response, "reasoning_content", "")
        return value if isinstance(value, str) else str(value or "")

    @classmethod
    def _response_tool_calls(cls, response: Any) -> List[Dict[str, Any]]:
        value = getattr(response, "tool_calls", None)
        if isinstance(value, list):
            return [item for item in (cls._coerce_tool_call_dict(item) for item in value) if item]
        raw_message = getattr(response, "raw_message", None)
        if isinstance(raw_message, dict) and isinstance(raw_message.get("tool_calls"), list):
            return [item for item in (cls._coerce_tool_call_dict(item) for item in raw_message["tool_calls"]) if item]
        return []

    @staticmethod
    def _coerce_tool_call_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        for method_name in ("model_dump", "dict"):
            method = getattr(item, method_name, None)
            if callable(method):
                try:
                    dumped = method()
                except TypeError:
                    try:
                        dumped = method(exclude_none=True)
                    except Exception:
                        dumped = None
                except Exception:
                    dumped = None
                if isinstance(dumped, dict):
                    return dumped
        function = getattr(item, "function", None)
        function_payload: Dict[str, Any] = {}
        if isinstance(function, dict):
            function_payload = function
        elif function is not None:
            function_payload = {
                "name": getattr(function, "name", ""),
                "arguments": getattr(function, "arguments", {}),
            }
        name = getattr(item, "name", "")
        arguments = getattr(item, "arguments", {})
        if not function_payload and (name or arguments):
            function_payload = {"name": name, "arguments": arguments}
        if not function_payload:
            return {}
        return {
            "id": str(getattr(item, "id", "") or ""),
            "type": str(getattr(item, "type", "function") or "function"),
            "function": function_payload,
        }

    def _tool_calls_to_proposed(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        state: Optional[AgentState] = None,
    ) -> List[Dict[str, Any]]:
        proposed: List[Dict[str, Any]] = []
        native_map = self._native_tool_mappings()
        for call in tool_calls:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            raw_name = str(function.get("name") or call.get("name") or "").strip()
            if not raw_name:
                continue
            raw_arguments = function.get("arguments", call.get("arguments", {}))
            arguments_obj = self._decode_tool_arguments(raw_arguments)
            name = raw_name
            action = "default"
            arguments: Dict[str, Any] = {}
            if name == FINAL_ANSWER_TOOL_NAME:
                arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
                trace_arguments = arguments if isinstance(arguments, dict) else {}
            elif name in native_map:
                name, action = native_map[name]
                trace_arguments = self._direct_native_arguments(arguments_obj)
                arguments = self._repair_tool_arguments(name, action, trace_arguments, state=state)
            elif name in self.tools:
                if isinstance(arguments_obj, dict) and (
                    "action" in arguments_obj or isinstance(arguments_obj.get("arguments"), dict)
                ):
                    action = str(arguments_obj.get("action") or "default")
                    nested = arguments_obj.get("arguments", {})
                    arguments = nested if isinstance(nested, dict) else {}
                else:
                    action = "default"
                    arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
                action = self.tools[name].normalize_action(action)
                trace_arguments = arguments if isinstance(arguments, dict) else {}
                arguments = self._repair_tool_arguments(name, action, arguments, state=state)
            elif "." in name:
                family, maybe_action = name.split(".", 1)
                if family in self.tools:
                    name = family
                    action = self.tools[name].normalize_action(maybe_action or "default")
                    arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
                    trace_arguments = arguments if isinstance(arguments, dict) else {}
                    arguments = self._repair_tool_arguments(name, action, arguments, state=state)
                else:
                    trace_arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
            else:
                trace_arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
            if name == FINAL_ANSWER_TOOL_NAME:
                native_name = FINAL_ANSWER_TOOL_NAME
                native_arguments = trace_arguments if isinstance(trace_arguments, dict) else {}
            elif name in self.tools:
                native_name = self._native_function_name(name, action, self._tool_action_count(name))
                native_arguments = self._trace_native_arguments(
                    name,
                    action,
                    trace_arguments,
                    arguments,
                )
            else:
                native_name = raw_name
                native_arguments = trace_arguments if isinstance(trace_arguments, dict) else {}
            proposed.append(
                {
                    "id": str(call.get("id") or ""),
                    "name": name,
                    "action": action,
                    "arguments": arguments,
                    "native_name": native_name,
                    "native_arguments": native_arguments,
                }
            )
        return proposed

    @staticmethod
    def _direct_native_arguments(arguments_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(arguments_obj, dict):
            return {}
        nested = arguments_obj.get("arguments")
        if isinstance(nested, dict) and ("action" in arguments_obj or len(arguments_obj) == 1):
            return nested
        return dict(arguments_obj)

    def _trace_native_arguments(
        self,
        name: str,
        action: str,
        raw_arguments: Dict[str, Any],
        repaired_arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        if not isinstance(repaired_arguments, dict):
            repaired_arguments = raw_arguments
        if name == "sec_search" and action == "full_text_search":
            schema = self._action_parameters_schema(self._tool_action_spec(name, action))
            properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            canonical: Dict[str, Any] = {}
            for key in properties:
                if key in repaired_arguments:
                    canonical[key] = repaired_arguments[key]
            return canonical
        return dict(raw_arguments)

    def _repair_tool_arguments(
        self,
        name: str,
        action: str,
        arguments: Dict[str, Any],
        *,
        state: Optional[AgentState] = None,
    ) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            return {}
        repaired = dict(arguments)
        if name == "sec_search" and action == "full_text_search":
            if "search_query" not in repaired and repaired.get("query") not in (None, ""):
                repaired["search_query"] = repaired.get("query")
        if name == MARKET_DATA_TOOL_NAME and action == "capabilities":
            if not str(repaired.get("query") or "").strip() and state is not None:
                query = self._market_data_task_query(state)
                if query:
                    repaired["query"] = query
        schema = self._action_parameters_schema(self._tool_action_spec(name, action))
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, prop in properties.items():
            if key not in repaired:
                continue
            repaired[key] = self._coerce_argument_value(repaired[key], prop)
        return repaired

    def _tool_action_spec(self, name: str, action: str) -> Dict[str, Any]:
        spec = self._tool_specs().get(name)
        if not isinstance(spec, dict):
            return {}
        actions = spec.get("actions") if isinstance(spec.get("actions"), dict) else {}
        action_spec = actions.get(action) if isinstance(actions, dict) else {}
        return action_spec if isinstance(action_spec, dict) else {}

    @classmethod
    def _coerce_argument_value(cls, value: Any, schema: Any) -> Any:
        if not isinstance(schema, dict):
            return value
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
        if schema_type == "array":
            return cls._coerce_array_argument(value, schema.get("items") if isinstance(schema.get("items"), dict) else {})
        if schema_type == "integer":
            return cls._coerce_integer_argument(value)
        if schema_type == "number":
            return cls._coerce_number_argument(value)
        if schema_type == "object":
            return cls._coerce_object_argument(value)
        if schema_type == "boolean":
            return cls._coerce_boolean_argument(value)
        return value

    @classmethod
    def _coerce_array_argument(cls, value: Any, item_schema: Dict[str, Any]) -> Any:
        if isinstance(value, list):
            items = value
        elif isinstance(value, tuple):
            items = list(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            parsed = parse_json_object(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                items = parsed.get("items") or []
            elif text.startswith("[") and text.endswith("]"):
                try:
                    loaded = json.loads(text)
                except Exception:
                    loaded = None
                items = loaded if isinstance(loaded, list) else [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
            else:
                items = [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
        elif value in (None, ""):
            return []
        else:
            items = [value]
        if item_schema:
            return [cls._coerce_argument_value(item, item_schema) for item in items]
        return items

    @staticmethod
    def _coerce_integer_argument(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[-+]?\d+", text):
                try:
                    return int(text)
                except Exception:
                    return value
        return value

    @staticmethod
    def _coerce_number_argument(value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", text):
                try:
                    return float(text)
                except Exception:
                    return value
        return value

    @staticmethod
    def _coerce_object_argument(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = parse_json_object(value)
            if isinstance(parsed, dict):
                return parsed
        return value

    @staticmethod
    def _coerce_boolean_argument(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return value

    @staticmethod
    def _decode_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            parsed = parse_json_object(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _canonical_action_content(
        self,
        state: AgentState,
        content: Any,
        *,
        step_number: int,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Normalize the model step note to the FIRE think/task_state content."""

        content_text = content if isinstance(content, str) else str(content or "")
        parsed = parse_json_object(content_text) or {}
        if isinstance(parsed, dict) and parsed:
            if self._is_final_answer_only_round(state):
                self._record_discarded_content_task_state(
                    state,
                    parsed,
                    reason="final_answer_only_round",
                )
            else:
                self._capture_task_state(state, parsed, step_number=step_number)
            think = parsed.get("think")
            if not isinstance(think, str):
                think = "" if think in (None, "", [], {}) else str(think)
            task_state = (
                {}
                if self._is_final_answer_only_round(state)
                else canonicalize_task_state(state.current_task_state)
            )
            payload = {
                "think": think,
                "task_state": task_state,
            }
            return parsed, think.strip(), self._json_compact(payload)

        think = content_text.strip()
        payload = {
            "think": think,
            "task_state": (
                {}
                if self._is_final_answer_only_round(state)
                else canonicalize_task_state(state.current_task_state)
            ),
        }
        return {}, think, self._json_compact(payload)

    @staticmethod
    def _parsed_has_nonempty_task_state(parsed: Dict[str, Any]) -> bool:
        raw_task_state = parsed.get("task_state") if isinstance(parsed, dict) else None
        return isinstance(raw_task_state, dict) and bool(raw_task_state)

    def _record_discarded_content_task_state(
        self,
        state: AgentState,
        parsed: Dict[str, Any],
        *,
        reason: str,
    ) -> None:
        if not self._parsed_has_nonempty_task_state(parsed):
            return
        if parsed.get(_TASK_STATE_DISCARD_RECORDED_KEY) is True:
            return
        parsed[_TASK_STATE_DISCARD_RECORDED_KEY] = True
        raw_task_state = parsed.get("task_state")
        stats = state.context_stats
        stats["content_task_state_discard_count"] = (
            int(stats.get("content_task_state_discard_count") or 0) + 1
        )
        stats["content_task_state_last_discard_reason"] = reason
        if isinstance(raw_task_state, dict):
            stats["content_task_state_last_discard_keys"] = sorted(map(str, raw_task_state))[:24]

    def _discard_terminal_content_task_state(
        self,
        state: AgentState,
        parsed: Dict[str, Any],
        *,
        prior_task_state: Optional[Dict[str, Any]],
        reason: str,
    ) -> None:
        self._record_discarded_content_task_state(state, parsed, reason=reason)
        if self._parsed_has_nonempty_task_state(parsed):
            state.current_task_state = deepcopy(prior_task_state)

    @staticmethod
    def _json_compact(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        except Exception:
            return str(value)

    @staticmethod
    def _llm_target_type(role: str) -> str:
        return {
            "planning": "plan",
            "action": "action",
            "terminal_summary": "final_answer",
            "context_compression": "context_compression_sections",
            "content_reader_summary": "content_reader_summary",
        }.get(role, "auxiliary")

    @staticmethod
    def _llm_supervision_contract(target_type: str) -> Dict[str, Any]:
        output_format = {
            "plan": "plan_text",
            "action": "assistant_content_plus_native_tool_calls",
            "final_answer": "assistant_content_plus_native_tool_calls",
            "context_compression_sections": "memory_sections_json",
            "content_reader_summary": "reader_summary_cards_text",
        }.get(target_type, "model_text")
        target_fields = ["output_message.reasoning_content", "output_message.content"]
        if target_type in {"action", "final_answer"}:
            target_fields.append("output_message.tool_calls")
        contract: Dict[str, Any] = {
            "input_field": "messages",
            "target_fields": target_fields,
            "target_type": target_type,
            "target_format": output_format,
        }
        if target_type == "context_compression_sections":
            contract.update(
                {
                    "model_output_artifact": "memory_sections",
                    "model_output_is_final_harness_artifact": False,
                    "harness_postprocess": "merge_memory_sections_into_deterministic_packet_base",
                    "postprocessed_artifact": "WorkingContextPacket",
                }
            )
        return contract

    @staticmethod
    def _turn_id(turn_type: str, step_number: Optional[int]) -> str:
        return f"turn_{str(turn_type or 'unknown').strip() or 'unknown'}_{int(step_number or 0):04d}"

    @staticmethod
    def _auxiliary_turn_id(state: AgentState, module: str) -> str:
        return f"aux_{str(module or 'auxiliary').strip() or 'auxiliary'}_{len(state.auxiliary_turns) + 1:04d}"

    def _response_output_message(self, response: Any) -> Dict[str, Any]:
        message = {
            "role": "assistant",
            "content": str(getattr(response, "content", "") or ""),
        }
        tool_calls = self._response_tool_calls(response)
        if tool_calls:
            message["tool_calls"] = tool_calls
        reasoning = self._response_reasoning_content(response)
        if keep_reasoning_content() and reasoning.strip():
            message["reasoning_content"] = reasoning
        return message

    @staticmethod
    def _response_output_message_from_payload(response: Dict[str, Any]) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content") if isinstance(response.get("content"), str) else str(response.get("content") or ""),
        }
        tool_calls = response.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            message["tool_calls"] = deepcopy(tool_calls)
        reasoning = response.get("reasoning_content")
        if keep_reasoning_content() and isinstance(reasoning, str) and reasoning.strip():
            message["reasoning_content"] = reasoning
        return message

    def _record_auxiliary_turn(
        self,
        state: AgentState,
        *,
        module: str,
        turn_type: str,
        target_type: str,
        input_messages: List[Dict[str, Any]],
        output_message: Dict[str, Any],
        source: Dict[str, Any],
        step_number: Optional[int] = None,
        render_mode: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        include_in_sft: bool = True,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        state.add_auxiliary_turn(
            AuxiliaryTurn(
                id=self._auxiliary_turn_id(state, module),
                type=turn_type,
                module=module,
                target_type=target_type,
                step_number=step_number,
                render_mode=str(render_mode or state.context_stats.get("last_render_mode") or ""),
                input_messages=self._model_input_messages(input_messages),
                output_message=self._trace_message(output_message),
                observations=[self._trace_message(message) for message in (observations or [])],
                tools=deepcopy(tools) if isinstance(tools, list) else [],
                source=deepcopy(source),
                include_in_sft=bool(include_in_sft),
            )
        )

    def _record_auxiliary_turn_from_response(
        self,
        state: AgentState,
        *,
        module: str,
        turn_type: str,
        target_type: str,
        messages: List[Dict[str, Any]],
        response: Any,
        source: Dict[str, Any],
        step_number: Optional[int] = None,
        render_mode: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        include_in_sft: bool = True,
    ) -> None:
        self._record_auxiliary_turn(
            state,
            module=module,
            turn_type=turn_type,
            target_type=target_type,
            input_messages=messages,
            output_message=self._response_output_message(response),
            source=source,
            step_number=step_number,
            render_mode=render_mode,
            tools=tools,
            include_in_sft=include_in_sft,
        )

    def _record_tool_auxiliary_turns(
        self,
        state: AgentState,
        call: ToolCall,
        result: ToolResult,
        *,
        step_number: int,
        call_index: int,
    ) -> None:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        raw_turns = metadata.pop("_auxiliary_turns", None)
        if not raw_turns:
            return
        turns = raw_turns if isinstance(raw_turns, list) else [raw_turns]
        for raw_turn in turns:
            if not isinstance(raw_turn, dict):
                continue
            messages = raw_turn.get("messages")
            response = raw_turn.get("response")
            if not isinstance(messages, list) or not isinstance(response, dict):
                continue
            usage = response.get("usage") if isinstance(response, dict) else {}
            if isinstance(usage, dict):
                state.record_usage("content_reader_summary", usage)
            source = raw_turn.get("source") if isinstance(raw_turn.get("source"), dict) else {}
            source = {
                **source,
                "parent_turn_id": self._turn_id("action", step_number),
                "parent_step_number": step_number,
                "parent_tool_call_id": self._trace_tool_call_id(call, step_number=step_number, call_index=call_index),
                "tool_name": self._tool_call_display_name(call),
                "tool_family": call.name,
                "tool_action": call.action,
                "call_index": call_index,
            }
            self._record_auxiliary_turn(
                state,
                module=str(raw_turn.get("module") or "content_reader"),
                turn_type=str(raw_turn.get("type") or "content_reader_summary"),
                target_type=str(raw_turn.get("target_type") or "content_reader_summary"),
                input_messages=messages,
                output_message=self._response_output_message_from_payload(response),
                source=source,
                step_number=step_number,
                render_mode=str(raw_turn.get("render_mode") or "tool_internal"),
                tools=[],
                include_in_sft=bool(raw_turn.get("include_in_sft", True)),
            )

    def _record_agent_turn(
        self,
        state: AgentState,
        *,
        turn_type: str,
        turn_id: Optional[str] = None,
        input_messages: List[Dict[str, Any]],
        output_message: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        include_in_sft: bool = True,
    ) -> None:
        canonical_input_messages = self._model_input_messages(input_messages)
        state.add_turn(
            AgentTurn(
                id=turn_id or self._turn_id(turn_type, len(state.turns) + 1),
                type=turn_type,
                input_messages=canonical_input_messages,
                output_message=self._trace_message(output_message),
                observations=[self._trace_message(message) for message in observations],
                tools=deepcopy(tools if isinstance(tools, list) else state.tools),
                include_in_sft=bool(include_in_sft),
            )
        )

    @classmethod
    def _model_input_messages(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Project stored messages to standard chat-completions input shape.

        Provider-native reasoning traces, synthetic/debug markers, and tool
        observation metadata are useful in the trace, but they are not valid
        input fields for most OpenAI-compatible ChatCompletions APIs. SFT uses
        the same projection so training sees the same prompt shape as runtime.
        """

        projected: List[Dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if not role:
                continue
            content = item.get("content", "")
            content_text = content if isinstance(content, str) else str(content or "")
            if role == "assistant":
                message: Dict[str, Any] = {"role": "assistant", "content": content_text}
                tool_calls = item.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    message["tool_calls"] = tool_calls
                projected.append(message)
                continue
            if role == "tool":
                tool_call_id = str(item.get("tool_call_id") or "").strip()
                if tool_call_id:
                    projected.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content_text,
                        }
                    )
                else:
                    projected.append(
                        {
                            "role": "user",
                            "content": "[tool_observation_without_call_id]\n" + content_text,
                        }
                    )
                continue
            if role in {"system", "user"}:
                projected.append({"role": role, "content": content_text})
                continue
            projected.append({"role": "user", "content": f"[{role}]\n{content_text}"})
        return cls._repair_tool_call_message_sequence(projected)

    @classmethod
    def _repair_tool_call_message_sequence(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        repaired: List[Dict[str, Any]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            role = str(message.get("role") or "").strip()
            if role == "assistant" and isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
                tool_calls = message.get("tool_calls") or []
                expected_ids = [
                    str(call.get("id") or "").strip()
                    for call in tool_calls
                    if isinstance(call, dict) and str(call.get("id") or "").strip()
                ]
                next_index = index + 1
                following_tools: List[Dict[str, Any]] = []
                while next_index < len(messages) and str(messages[next_index].get("role") or "") == "tool":
                    following_tools.append(messages[next_index])
                    next_index += 1
                got_ids = [
                    str(item.get("tool_call_id") or "").strip()
                    for item in following_tools
                    if str(item.get("tool_call_id") or "").strip()
                ]
                if (
                    expected_ids
                    and len(got_ids) >= len(expected_ids)
                    and set(got_ids[: len(expected_ids)]) == set(expected_ids)
                ):
                    repaired.append(deepcopy(message))
                    repaired.extend(deepcopy(item) for item in following_tools[: len(expected_ids)])
                    for extra_tool in following_tools[len(expected_ids) :]:
                        repaired.append(cls._tool_message_as_user(extra_tool))
                    index = next_index
                    continue
                repaired.append(cls._assistant_tool_calls_as_text(message))
                for tool_message in following_tools:
                    repaired.append(cls._tool_message_as_user(tool_message))
                index = next_index
                continue
            if role == "tool":
                repaired.append(cls._tool_message_as_user(message))
                index += 1
                continue
            repaired.append(deepcopy(message))
            index += 1
        return repaired

    @staticmethod
    def _assistant_tool_calls_as_text(message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content", "")
        content_text = content if isinstance(content, str) else str(content or "")
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        try:
            rendered_calls = json.dumps(tool_calls, ensure_ascii=False, default=str, separators=(",", ":"))
        except Exception:
            rendered_calls = str(tool_calls)
        parts = [content_text.strip()] if content_text.strip() else []
        parts.append("Historical assistant tool calls:\n" + rendered_calls)
        return {
            "role": "assistant",
            "content": "\n".join(parts),
        }

    @staticmethod
    def _tool_message_as_user(message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content", "")
        content_text = content if isinstance(content, str) else str(content or "")
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        prefix = f"[tool_observation:{tool_call_id}]" if tool_call_id else "[tool_observation]"
        return {
            "role": "user",
            "content": f"{prefix}\n{content_text}",
        }

    @staticmethod
    def _model_input_messages_without_tool_protocol(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Project messages to plain chat input without native tool-call protocol.

        Terminal best-effort summaries are prompted after a long trace that may
        contain compacted or historical tool calls. Rehydrating those into
        native assistant tool_calls can violate provider sequencing rules when
        the paired tool messages were intentionally omitted from the prompt.
        This projection keeps the evidence text while removing protocol state.
        """

        projected: List[Dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if not role:
                continue
            content = item.get("content", "")
            content_text = content if isinstance(content, str) else str(content or "")
            if role == "system":
                projected.append({"role": "system", "content": content_text})
                continue
            if role == "user":
                projected.append({"role": "user", "content": content_text})
                continue
            if role == "assistant":
                if not content_text and isinstance(item.get("tool_calls"), list):
                    content_text = "Assistant proposed tool calls:\n" + json.dumps(
                        item.get("tool_calls") or [],
                        ensure_ascii=False,
                        default=str,
                    )
                projected.append({"role": "assistant", "content": content_text})
                continue
            if role == "tool":
                tool_call_id = str(item.get("tool_call_id") or "").strip()
                prefix = f"[tool_observation:{tool_call_id}]" if tool_call_id else "[tool_observation]"
                projected.append({"role": "user", "content": f"{prefix}\n{content_text}"})
                continue
            projected.append({"role": "user", "content": f"[{role}]\n{content_text}"})
        return projected

    def _canonicalize_turn_input_messages(
        self,
        state: AgentState,
        input_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        action_turns = [turn for turn in state.turns if getattr(turn, "type", "") == "action"]
        action_cursor = 0
        pending_observation_turn: Optional[AgentTurn] = None
        canonical: List[Dict[str, Any]] = []

        for message in input_messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if role == "assistant" and content.startswith("Calling tools:\n"):
                payload = parse_json_object(content[len("Calling tools:\n") :])
                turn = self._match_action_turn(action_turns, payload, action_cursor)
                if turn is not None:
                    canonical.append(self._trace_message(turn.output_message))
                    pending_observation_turn = turn
                    action_cursor = max(action_cursor, action_turns.index(turn) + 1)
                    continue
            if role == "user" and content.startswith("Tool calling observation"):
                turn = pending_observation_turn or self._match_observation_turn(action_turns, content)
                if turn is not None:
                    canonical.extend(self._trace_message(item) for item in turn.observations)
                    pending_observation_turn = None
                    continue
            if role == "user" and self._is_working_context_packet(content):
                canonical.append(
                    {
                        "role": "assistant",
                        "type": "memory_summary",
                        "synthetic": True,
                        "content": content,
                    }
                )
                continue
            canonical.append(self._trace_message(message))
        return canonical

    @staticmethod
    def _is_working_context_packet(content: str) -> bool:
        stripped = str(content or "").lstrip()
        return stripped.startswith("WorkingContextPacket:") or stripped.startswith("<working_context_packet")

    def _match_action_turn(
        self,
        action_turns: List[AgentTurn],
        payload: Dict[str, Any],
        cursor: int,
    ) -> Optional[AgentTurn]:
        step = payload.get("step")
        if step not in (None, ""):
            try:
                step_number = int(step)
            except Exception:
                step_number = 0
            if step_number > 0:
                prefix = f"call_{step_number:04d}_"
                for turn in action_turns:
                    if self._turn_has_tool_call_prefix(turn, prefix):
                        return turn
        raw_tools = payload.get("tools")
        tools = raw_tools if isinstance(raw_tools, list) else []
        ids = {str(item.get("id") or "") for item in tools if isinstance(item, dict) and item.get("id")}
        if ids:
            for turn in action_turns:
                if ids & self._turn_tool_call_ids(turn):
                    return turn
        if cursor < len(action_turns):
            return action_turns[cursor]
        return None

    def _match_observation_turn(self, action_turns: List[AgentTurn], content: str) -> Optional[AgentTurn]:
        match = re.search(r"Tool calling observation \(step\s+(\d+)\)", str(content or ""))
        if not match:
            return None
        step_number = int(match.group(1))
        prefix = f"call_{step_number:04d}_"
        for turn in action_turns:
            if self._turn_has_tool_call_prefix(turn, prefix):
                return turn
        return None

    @staticmethod
    def _turn_tool_call_ids(turn: AgentTurn) -> set[str]:
        output = turn.output_message if isinstance(turn.output_message, dict) else {}
        tool_calls = output.get("tool_calls") if isinstance(output.get("tool_calls"), list) else []
        return {str(call.get("id") or "") for call in tool_calls if isinstance(call, dict) and call.get("id")}

    def _turn_has_tool_call_prefix(self, turn: AgentTurn, prefix: str) -> bool:
        return any(call_id.startswith(prefix) for call_id in self._turn_tool_call_ids(turn))

    def _trace_generation_config(self) -> Dict[str, Any]:
        settings = getattr(self.model, "settings", None)
        thinking_enabled = getattr(settings, "thinking_enabled", False)
        return {"thinking_enabled": bool(thinking_enabled)}

    def _trace_tool_schemas(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for name, spec in self._tool_specs().items():
            if not isinstance(spec, dict):
                continue
            actions = spec.get("actions") if isinstance(spec.get("actions"), dict) else {}
            action_names = list(actions) or [str(spec.get("default_action") or "default")]
            for action_name in action_names:
                action_spec = actions.get(action_name) if isinstance(actions, dict) else {}
                if not isinstance(action_spec, dict):
                    action_spec = {}
                full_description = self._native_tool_description(
                    tool_name=str(name),
                    action_name=str(action_name),
                    action_count=len(action_names),
                    tool_spec=spec,
                    action_spec=action_spec,
                )
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": self._native_function_name(str(name), str(action_name), len(action_names)),
                            "description": full_description,
                            "parameters": self._native_action_parameters_schema(str(name), action_spec),
                        },
                    }
                )
        tools.extend(self._final_answer_tool_schemas())
        return tools

    def _select_step_tools(
        self,
        state: AgentState,
        memory: AgentMemory,
        _instruction: str,
    ) -> List[Dict[str, Any]]:
        tools = state.tools or self._trace_tool_schemas()
        market_names = {
            self._tool_schema_name(tool)
            for tool in tools
            if self._tool_schema_name(tool).startswith("market_data_")
        }
        if not market_names:
            return tools

        selected = {
            self._tool_schema_name(tool)
            for tool in tools
            if not self._tool_schema_name(tool).startswith("market_data_")
        }
        selected.update(name for name in MARKET_DATA_ALWAYS_NATIVE_TOOLS if name in market_names)
        selected.update(self._retrieved_market_data_tools(state, memory, market_names))
        selected.update(self._unlocked_market_data_tools(state, market_names))
        selected.update(self._recent_market_data_tools(state))
        return [
            tool
            for tool in tools
            if self._tool_schema_name(tool) in selected
        ]

    @staticmethod
    def _tool_schema_name(tool: Dict[str, Any]) -> str:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        return str(function.get("name") or "").strip()

    @staticmethod
    def _unlocked_market_data_tools(state: AgentState, market_names: set[str]) -> set[str]:
        raw_unlocks = state.context_stats.get(MARKET_DATA_TOOL_UNLOCKS_KEY)
        if isinstance(raw_unlocks, dict):
            candidates = raw_unlocks.keys()
        elif isinstance(raw_unlocks, (list, tuple, set)):
            candidates = raw_unlocks
        else:
            return set()
        return {str(name) for name in candidates if str(name) in market_names}

    def _unlock_market_data_tool_names(
        self,
        state: AgentState,
        tool_names: set[str],
        *,
        step_number: int,
        reason: str = "",
    ) -> None:
        if not tool_names:
            return
        market_names = {
            self._tool_schema_name(tool)
            for tool in (state.tools or self._trace_tool_schemas())
            if self._tool_schema_name(tool).startswith("market_data_")
        }
        discovered = {name for name in tool_names if name in market_names}
        if not discovered:
            return
        raw_unlocks = state.context_stats.setdefault(MARKET_DATA_TOOL_UNLOCKS_KEY, {})
        if not isinstance(raw_unlocks, dict):
            raw_unlocks = {}
            state.context_stats[MARKET_DATA_TOOL_UNLOCKS_KEY] = raw_unlocks
        for name in sorted(discovered):
            payload = {"step": step_number}
            if reason:
                payload["reason"] = reason
            raw_unlocks[name] = payload

    def _retrieved_market_data_tools(
        self,
        state: AgentState,
        memory: AgentMemory,
        market_names: set[str],
    ) -> set[str]:
        query = self._market_data_retrieval_query(state, memory)
        if not query.strip():
            return set()

        scored: List[Tuple[int, str, str]] = []
        for action_name in self._market_data_action_names():
            if action_name == "capabilities":
                continue
            native_name = self._market_data_native_name(action_name)
            if native_name not in market_names:
                continue
            if native_name in {MARKET_DATA_SOURCE_CATALOG_NATIVE_TOOL, MARKET_DATA_MARKET_TABLE_NATIVE_TOOL}:
                continue
            score = self._market_data_action_score(action_name, query)
            if score >= 3:
                scored.append((score, action_name, native_name))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = {native_name for _, _, native_name in scored[:MARKET_DATA_DYNAMIC_TOP_K]}

        if (
            MARKET_DATA_SOURCE_CATALOG_NATIVE_TOOL in market_names
            and self._market_data_source_catalog_intent(query)
        ):
            selected.add(MARKET_DATA_SOURCE_CATALOG_NATIVE_TOOL)
        if (
            MARKET_DATA_MARKET_TABLE_NATIVE_TOOL in market_names
            and self._market_data_market_table_intent(query)
        ):
            selected.add(MARKET_DATA_MARKET_TABLE_NATIVE_TOOL)
        return selected

    def _market_data_retrieval_query(self, state: AgentState, memory: AgentMemory) -> str:
        parts: List[str] = []
        task_query = self._market_data_task_query(state)
        if task_query:
            parts.append(task_query)
        if state.current_task_state:
            parts.append(compact_text(json_compact(state.current_task_state), 2400))
        for step in memory.action_steps[-2:]:
            if step.think:
                parts.append(compact_text(step.think, 1200))
            if step.observations and not self._action_step_has_market_data_capabilities(step):
                parts.append(compact_text(step.observations, 2400))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _action_step_has_market_data_capabilities(step: ActionStep) -> bool:
        calls = getattr(step, "tool_calls", None)
        if not isinstance(calls, list):
            return False
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            action = str(call.get("action") or "").strip()
            native_name = str(call.get("native_name") or "").strip()
            if name == MARKET_DATA_TOOL_NAME and action == "capabilities":
                return True
            if native_name == "market_data_capabilities":
                return True
        return False

    @staticmethod
    def _market_data_task_query(state: AgentState) -> str:
        ctx = state.task_context
        parts: List[str] = []
        for value in (ctx.raw_question, ctx.task_prompt):
            text = str(value or "").strip()
            if text and text not in parts:
                parts.append(text)
        raw_record = ctx.raw_record if isinstance(ctx.raw_record, dict) else {}
        for key in ("question", "query", "task", "problem", "prompt"):
            text = str(raw_record.get(key) or "").strip()
            if text and text not in parts:
                parts.append(text)
        return compact_text("\n".join(parts), 5000)

    def _market_data_action_score(self, action_name: str, query: str) -> int:
        query_text = self._market_data_match_text(query)
        if not query_text:
            return 0
        score = 0
        action_token = action_name.lower()
        if re.search(rf"(?<![0-9a-z_]){re.escape(action_token)}(?![0-9a-z_])", query_text):
            score += 12
        spaced_action = action_token.replace("_", " ")
        if spaced_action != action_token and spaced_action in query_text:
            score += 8

        for term in self._market_data_internal_action_terms().get(action_name, ()):
            term_text = self._market_data_match_text(term)
            if not term_text:
                continue
            if term_text in query_text:
                score += 5 if len(term_text) >= 4 else 3

        action_spec = self._tool_action_spec(MARKET_DATA_TOOL_NAME, action_name)
        doc_text = " ".join(
            str(item or "")
            for item in (
                action_name.replace("_", " "),
                action_spec.get("description", ""),
                " ".join(str(key) for key in action_spec.get("required") or []),
                " ".join(str(key) for key in action_spec.get("optional") or []),
            )
        )
        doc_tokens = self._market_data_match_tokens(doc_text)
        query_tokens = self._market_data_match_tokens(query)
        overlap = doc_tokens.intersection(query_tokens)
        score += min(6, len(overlap))
        return score

    @staticmethod
    def _market_data_match_text(text: Any) -> str:
        lowered = str(text or "").lower()
        lowered = lowered.replace("-", "_")
        return re.sub(r"\s+", " ", lowered).strip()

    @classmethod
    def _market_data_match_tokens(cls, text: Any) -> set[str]:
        raw = cls._market_data_match_text(text)
        tokens = set(re.findall(r"[a-z0-9_./]{2,}", raw))
        tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", raw))
        stopwords = {
            "market",
            "data",
            "market_data",
            "provider",
            "query",
            "date",
            "start_date",
            "end_date",
            "limit",
            "fields",
            "fetch",
            "rows",
            "through",
            "returns",
            "source",
            "family",
            "preferred",
            "actions",
            "action",
            "typed",
            "schema",
            "use",
            "optional",
            "required",
            "the",
            "to",
            "an",
            "and",
            "then",
            "find",
        }
        return {token for token in tokens if token not in stopwords}

    @staticmethod
    def _market_data_internal_action_terms() -> Dict[str, Tuple[str, ...]]:
        return {
            "quote_snapshot": ("quote", "snapshot", "spot", "latest price", "realtime", "实时", "现价", "行情快照"),
            "price_history": ("historical price", "close price", "ohlcv", "return", "yahoo", "收盘价", "开盘价", "涨跌幅", "成交量", "成交额", "历史行情"),
            "equity_price_history": ("a share", "stock daily", "tushare daily", "股票日线", "A股", "股价", "股票收盘", "证券代码"),
            "equity_daily_basic": ("market cap", "pe", "pb", "turnover", "daily_basic", "总市值", "流通市值", "市盈率", "市净率", "换手率"),
            "equity_factor_history": ("adjust factor", "adj_factor", "复权", "复权因子", "前复权", "后复权"),
            "equity_financials": ("income statement", "balance sheet", "cashflow", "financial indicator", "营收", "净利润", "财报", "利润表", "资产负债表", "现金流量表", "财务指标"),
            "equity_dividend": ("dividend", "ex date", "record date", "分红", "派息", "股息", "除权", "除息"),
            "equity_universe": ("stock basic", "stock universe", "listed company", "股票代码", "股票简称", "上市公司", "上市日期", "证券简称"),
            "index_history": ("index history", "index daily", "ohlcv", "指数", "指数行情", "指数日线", "收盘点位", "上证", "深证", "中证", "沪深", "申万", ".si"),
            "index_weight": ("index weight", "constituent weight", "成分权重", "权重", "样本权重", "指数权重"),
            "index_constituents": ("index constituents", "constituent", "成分股", "样本股", "指数成分", "成份股"),
            "index_basic": ("index basic", "index catalog", "指数代码", "指数目录", "指数列表", "发布方", "中证指数", "申万指数"),
            "index_catalog": ("index catalog", "index basic", "指数代码", "指数目录", "指数列表", "发布方", "中证指数", "申万指数"),
            "fund_history": ("fund daily", "fund history", "etf daily", "基金行情", "基金收盘", "ETF", "etf", "场内基金"),
            "fund_nav": ("fund nav", "net asset value", "NAV", "nav", "净值", "单位净值", "累计净值"),
            "fund_portfolio": ("fund portfolio", "holding", "基金持仓", "重仓股", "投资组合"),
            "fund_basic": ("fund basic", "fund catalog", "ETF", "etf", "基金代码", "基金简称", "基金列表", "成立日期"),
            "cn_futures_basic": ("futures basic", "contract catalog", "期货合约", "合约代码", "期货品种", "交易所"),
            "cn_futures_mapping": ("dominant contract", "continuous contract", "contract mapping", "主力合约", "连续合约", "合约映射"),
            "cn_futures_daily": ("futures daily", "futures ohlc", "open interest", "期货日线", "期货收盘", "结算价", "持仓量", "成交量"),
            "cn_futures_warehouse": ("warehouse receipt", "warehouse", "仓单", "注册仓单", "库存"),
            "cn_futures_settle": ("settlement", "delivery", "结算", "交割", "交割结算"),
            "cn_options_basic": ("options basic", "option contract", "期权合约", "期权代码", "期权列表"),
            "cn_options_daily": ("options daily", "option ohlc", "期权日线", "期权行情", "期权收盘"),
            "option_chain_metrics": ("option chain", "intrinsic value", "time value", "moneyness", "期权链", "内在价值", "时间价值", "虚实值"),
            "cn_bond_yield_curve": ("bond yield curve", "chinabond", "国债收益率", "收益率曲线", "中债"),
            "cn_macro_series": ("china macro", "cpi", "ppi", "pmi", "gdp", "m0", "m1", "m2", "中国宏观", "居民消费价格", "工业生产者", "采购经理"),
            "shibor_series": ("shibor", "interbank", "上海银行间", "同业拆借", "银行间利率"),
            "financial_statement": ("sec xbrl", "gaap", "revenue", "net income", "eps", "income statement", "balance sheet", "cash flow", "10-k", "10-q", "财报"),
            "macro_series": ("fred", "alfred", "world bank", "oecd", "iea", "dataflow", "indicator code", "macro", "inflation", "unemployment", "interest rate", "gdp", "pce", "宏观", "通胀", "失业率"),
            "trade_series": ("comtrade", "wits", "trade", "import", "export", "hs code", "贸易", "进口", "出口", "商品编码"),
            "futures_contract_history": ("futures contract history", "contract ohlc", "期货合约行情", "期货收盘", "持仓量", "合约日线"),
            "metals_price": ("lbma", "gold", "silver", "precious metal", "黄金", "白银", "贵金属"),
            "official_attachment_table": ("official attachment", "safe", "nfra", "excel", "spreadsheet", "附件", "附表", "外汇局", "金融监管总局", "官方表格"),
            "source_catalog": ("source_catalog", "fred", "world bank", "oecd", "iea", "comtrade", "wits", "sdmx", "dataflow", "indicator code", "official provider catalog", "官方目录", "指标代码"),
            "market_table": ("market_table", "akshare function", "explicit function", "function_guidance", "requested_function", "unsupported akshare function", "函数名"),
        }

    def _market_data_source_catalog_intent(self, query: str) -> bool:
        query_text = self._market_data_match_text(query)
        if not query_text:
            return False
        china_market_terms = (
            "china",
            "tushare",
            "akshare",
            "a_share",
            "a股",
            "沪深",
            "中证",
            "申万",
            "指数",
            "基金",
            "期货",
        )
        official_terms = (
            "fred",
            "world bank",
            "world_bank",
            "oecd",
            "iea",
            "comtrade",
            "wits",
            "sdmx",
            "dataflow",
            "data flow",
            "indicator code",
            "series catalog",
            "official provider catalog",
        )
        if any(self._market_data_match_text(term) in query_text for term in official_terms):
            return True
        if any(self._market_data_match_text(term) in query_text for term in china_market_terms):
            return False
        return "source_catalog" in query_text

    def _market_data_market_table_intent(self, query: str) -> bool:
        query_text = self._market_data_match_text(query)
        if not query_text:
            return False
        positive_terms = (
            "market_table(function",
            "preferred_actions: market_table",
            "recommended_action\": \"market_table",
            "recommended_action': 'market_table",
            "recommended_typed_actions\": [\"market_table",
            "function_guidance",
            "requested_function",
            "unsupported akshare function",
            "akshare function",
            "explicit function",
            "显式函数",
        )
        if any(self._market_data_match_text(term) in query_text for term in positive_terms):
            return True
        return "akshare" in query_text and "函数" in query_text

    def _record_market_data_tool_unlocks(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        step_number: int,
    ) -> None:
        if not calls or not results:
            return
        market_names = {
            self._tool_schema_name(tool)
            for tool in (state.tools or self._trace_tool_schemas())
            if self._tool_schema_name(tool).startswith("market_data_")
        }
        if not market_names:
            return

        discovered: set[str] = set()
        for call, result in zip(calls, results):
            if call.name != MARKET_DATA_TOOL_NAME:
                continue
            current_name = self._market_data_native_name(call.action)
            if current_name in market_names:
                discovered.add(current_name)
            discovered.update(self._market_data_tools_recommended_by_result(result, market_names))
        self._unlock_market_data_tool_names(
            state,
            discovered,
            step_number=step_number,
            reason="tool_result",
        )

    def _market_data_tools_recommended_by_result(self, result: ToolResult, market_names: set[str]) -> set[str]:
        if result.tool_family != MARKET_DATA_TOOL_NAME:
            return set()
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if result.action == "capabilities":
            result_count = int(metadata.get("result_count", 0) or 0)
            if result_count > 3 and not bool(metadata.get("filtered")):
                return set()

        payloads: List[Any] = []
        for key in (
            "recommended_action",
            "recommended_actions",
            "recommended_typed_actions",
            "recommended_next",
            "state_recommended_next",
            "recommended_fallback",
            "function_guidance",
            "argument_mapping",
            "requested_function",
            "replacement_function",
        ):
            if key in metadata:
                payloads.append(metadata.get(key))
        for table in result.tables or []:
            rows = table.get("rows") if isinstance(table, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows[:24]:
                if not isinstance(row, dict):
                    continue
                for key in (
                    "action",
                    "preferred_actions",
                    "recommended_action",
                    "recommended_actions",
                    "recommended_typed_actions",
                    "call_example",
                    "function_guidance",
                    "recommended_next",
                ):
                    if key in row:
                        payloads.append(row.get(key))

        selected: set[str] = set()
        for payload in payloads:
            selected.update(self._market_data_tools_from_payload(payload, market_names))
        return selected

    def _market_data_tools_from_payload(self, payload: Any, market_names: set[str]) -> set[str]:
        if payload in (None, "", [], {}):
            return set()
        if isinstance(payload, dict):
            nested: set[str] = set()
            if "action" in payload:
                nested.update(self._market_data_tools_from_payload(payload.get("action"), market_names))
            for key in (
                "preferred_actions",
                "recommended_action",
                "recommended_actions",
                "recommended_typed_actions",
                "call_example",
                "function_guidance",
                "recommended_next",
            ):
                if key in payload:
                    nested.update(self._market_data_tools_from_payload(payload.get(key), market_names))
            if str(payload.get("recommended_tool") or "").strip() == MARKET_DATA_TOOL_NAME:
                nested.update(self._market_data_tools_from_payload(payload.get("recommended_action"), market_names))
            return nested
        if isinstance(payload, (list, tuple, set)):
            selected: set[str] = set()
            for item in payload:
                selected.update(self._market_data_tools_from_payload(item, market_names))
            return selected

        text = str(payload or "").strip()
        if not text:
            return set()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return self._market_data_tools_from_payload(parsed, market_names)

        selected: set[str] = set()
        lowered = text.lower()
        for action_name in self._market_data_action_names():
            native_name = self._market_data_native_name(action_name)
            if native_name not in market_names:
                continue
            pattern = rf"(?<![0-9a-z_]){re.escape(action_name.lower())}(?![0-9a-z_])"
            if re.search(pattern, lowered):
                selected.add(native_name)
        return selected

    def _market_data_action_names(self) -> List[str]:
        spec = self._tool_specs().get(MARKET_DATA_TOOL_NAME)
        actions = spec.get("actions") if isinstance(spec, dict) and isinstance(spec.get("actions"), dict) else {}
        return [str(name) for name in actions]

    def _market_data_native_name(self, action_name: str) -> str:
        return self._native_function_name(
            MARKET_DATA_TOOL_NAME,
            str(action_name or "default").strip() or "default",
            self._tool_action_count(MARKET_DATA_TOOL_NAME),
        )

    def _recent_market_data_tools(self, state: AgentState) -> set[str]:
        selected: set[str] = set()
        action_count = self._tool_action_count("market_data")
        for step in state.trajectory[-4:]:
            calls = step.tool_calls if isinstance(step.tool_calls, list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                action = str(call.get("action") or "default").strip() or "default"
                if name == "market_data":
                    selected.add(self._native_function_name("market_data", action, action_count))
                elif name.startswith("market_data_"):
                    selected.add(name)
        return selected

    def _native_tool_description(
        self,
        *,
        tool_name: str,
        action_name: str,
        action_count: int,
        tool_spec: Dict[str, Any],
        action_spec: Dict[str, Any],
    ) -> str:
        """Return the prompt-facing native function description.

        Native tool descriptions are part of the model prompt. Keep the
        always-on layer to a short action contract; detailed routing guidance
        should come from capability tools, validation hints, or observations.
        """

        action_description = str(action_spec.get("description") or "").strip()
        tool_description = str(tool_spec.get("description") or "").strip()
        description = action_description or tool_description
        if not description:
            native_name = self._native_function_name(tool_name, action_name, action_count)
            description = f"Call {native_name}."

        limit = 120 if tool_name == "market_data" else 220
        description = self._compact_schema_text(description, limit=limit)
        if tool_name == "market_data":
            optional = [str(key) for key in action_spec.get("optional") or [] if str(key).strip()]
            if optional:
                description = f"{description} Optional args: {','.join(optional)}."
        return description

    @staticmethod
    def _compact_schema_text(text: str, *, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if limit <= 0 or len(text) <= limit:
            return text
        cutoff = max(1, limit - 1)
        candidate = text[:cutoff]
        word_boundary = candidate.rfind(" ")
        if word_boundary >= max(24, int(cutoff * 0.65)):
            candidate = candidate[:word_boundary]
        return candidate.rstrip(" ,;:.-") + "."

    def _native_tool_mappings(self) -> Dict[str, Tuple[str, str]]:
        mappings: Dict[str, Tuple[str, str]] = {}
        for tool_name, spec in self._tool_specs().items():
            if not isinstance(spec, dict):
                continue
            actions = spec.get("actions") if isinstance(spec.get("actions"), dict) else {}
            action_names = list(actions) or [str(spec.get("default_action") or "default")]
            for action_name in action_names:
                native_name = self._native_function_name(str(tool_name), str(action_name), len(action_names))
                mappings[native_name] = (str(tool_name), str(action_name))
        return mappings

    @staticmethod
    def _native_function_name(tool_name: str, action_name: str, action_count: int = 1) -> str:
        tool_name = str(tool_name or "").strip()
        action_name = str(action_name or "default").strip() or "default"
        if tool_name == FINAL_ANSWER_TOOL_NAME:
            return FINAL_ANSWER_TOOL_NAME
        if action_name == "default":
            return tool_name
        if action_count == 1 and action_name in {"search", "run", "calculate"}:
            return tool_name
        return f"{tool_name}_{action_name}"

    def _tool_call_display_name(self, call: ToolCall) -> str:
        return str(call.native_name or "").strip() or self._native_function_name(
            call.name,
            call.action,
            self._tool_action_count(call.name),
        )

    def _tool_action_count(self, tool_name: str) -> int:
        spec = self._tool_specs().get(tool_name)
        if not isinstance(spec, dict):
            return 1
        actions = spec.get("actions") if isinstance(spec.get("actions"), dict) else {}
        return max(1, len(actions))

    def _action_parameters_schema(self, action_spec: Dict[str, Any]) -> Dict[str, Any]:
        raw_parameters = action_spec.get("parameters") if isinstance(action_spec, dict) else None
        if isinstance(raw_parameters, dict) and raw_parameters.get("type") == "object":
            schema = deepcopy(raw_parameters)
        elif isinstance(raw_parameters, dict):
            schema = {"type": "object", "properties": deepcopy(raw_parameters)}
        else:
            schema = {"type": "object", "properties": {}}
        properties = schema.setdefault("properties", {})
        if not isinstance(properties, dict):
            properties = {}
            schema["properties"] = properties
        explicit_required = schema.get("required") if isinstance(schema.get("required"), list) else []
        required = [
            str(key)
            for key in (action_spec.get("required") or explicit_required or [])
            if str(key).strip()
        ]
        optional = [str(key) for key in action_spec.get("optional") or [] if str(key).strip()]
        for key in required + optional:
            properties.setdefault(key, self._inferred_argument_schema(key))
        if required:
            schema["required"] = required
        else:
            schema.pop("required", None)
        schema.setdefault("additionalProperties", False)
        return self._compact_parameter_schema(schema)

    def _native_action_parameters_schema(self, tool_name: str, action_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return the compact prompt-facing parameter schema for native tools.

        Internal repair/validation keeps using the full action schema.  For
        market_data the native schema should expose the required calling
        contract without spending prompt budget on every advanced optional
        argument.  Optional arguments remain valid because this schema allows
        additional properties and their names are listed in the short function
        description.
        """

        schema = self._action_parameters_schema(action_spec)
        if str(tool_name) != "market_data":
            return schema
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [str(key) for key in schema.get("required") or [] if str(key).strip()]
        slim_properties = {
            key: deepcopy(properties[key])
            for key in required
            if key in properties
        }
        slim_schema: Dict[str, Any] = {
            "type": "object",
            "properties": slim_properties,
            "additionalProperties": True,
        }
        if required:
            slim_schema["required"] = required
        return slim_schema

    def _compact_parameter_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Trim verbose JSON-Schema descriptions without changing structure."""

        def compact(value: Any) -> Any:
            if isinstance(value, dict):
                cleaned: Dict[str, Any] = {}
                for key, item in value.items():
                    if key == "description" and isinstance(item, str):
                        cleaned[key] = self._compact_schema_text(item, limit=120)
                    else:
                        cleaned[key] = compact(item)
                return cleaned
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value

        compacted = compact(schema)
        return compacted if isinstance(compacted, dict) else schema

    @staticmethod
    def _inferred_argument_schema(key: str) -> Dict[str, Any]:
        lowered = str(key or "").strip().lower()
        array_keys = {
            "form_types",
            "ciks",
            "fields",
            "columns",
            "symbols",
            "table_indices",
            "attachment_urls",
        }
        object_keys = {
            "plan",
            "state_params",
            "filter",
            "filters",
            "params",
            "defaults",
            "metadata",
        }
        integer_keys = {
            "limit",
            "max_results",
            "top_n_results",
            "top_k",
            "page",
            "precision",
            "max_tokens",
            "max_pages",
            "preview_rows",
            "table_index",
            "row_start",
            "row_end",
            "start",
            "end",
            "header_rows",
            "max_entities",
        }
        boolean_keys = {"require_state_match", "allow_partial", "include_metadata"}
        if lowered in array_keys:
            return {"type": "array", "items": {"type": "string"}}
        if lowered == "index_code":
            return {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
        if lowered in object_keys:
            return {"type": "object"}
        if lowered in integer_keys or lowered.endswith("_limit") or lowered.endswith("_rows"):
            return {"type": "integer"}
        if lowered in boolean_keys or lowered.startswith("include_") or lowered.startswith("use_"):
            return {"type": "boolean"}
        return {"type": "string"}

    @staticmethod
    def _final_answer_tool_schemas() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                 "function": {
                     "name": FINAL_ANSWER_TOOL_NAME,
                     "description": (
                         "Finish the task with the final answer string. Use this only when the answer is ready. "
                         "In the same assistant response, assistant.content must be compact JSON with a non-empty "
                         "public think sentence and task_state; never emit an empty think string, and never put "
                         "the answer value/text in assistant.content."
                     ),
                     "parameters": {
                         "type": "object",
                         "properties": {
                             "answer": {
                                 "type": "string",
                                 "description": (
                                     "The final answer to return to the user. Put the answer only here; "
                                     "assistant.content.think must contain only a short public basis/check note."
                                 ),
                             }
                         },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @staticmethod
    def _final_answer_tool_choice() -> Dict[str, Any]:
        return {"type": "function", "function": {"name": FINAL_ANSWER_TOOL_NAME}}

    def _model_thinking_enabled(self) -> bool:
        settings = getattr(self.model, "settings", None)
        return bool(getattr(settings, "thinking_enabled", False))

    def _model_is_deepseek(self) -> bool:
        settings = getattr(self.model, "settings", None)
        text = f"{getattr(settings, 'model', '') or ''} {getattr(settings, 'base_url', '') or ''}".lower()
        return "deepseek" in text

    def _final_answer_tool_choice_for_model(self) -> Any:
        # DeepSeek V4 thinking mode accepts native tools, but rejects
        # tool_choice="required". tool_choice="auto" still allows the model to
        # emit final_answer while preserving native reasoning_content.
        if self._model_is_deepseek() and self._model_thinking_enabled():
            return "auto"
        return "required"

    @staticmethod
    def _trace_message(message: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "role",
            "type",
            "synthetic",
            "content",
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
            "name",
            "status",
        }
        cleaned: Dict[str, Any] = {}
        for key in allowed:
            if key not in message:
                continue
            value = message.get(key)
            if value is None:
                cleaned[key] = None
            elif key == "content":
                cleaned[key] = value if isinstance(value, str) else str(value)
            else:
                cleaned[key] = deepcopy(value)
        return cleaned

    def _trace_assistant_message(
        self,
        *,
        content: str,
        reasoning_content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        step_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": str(content or ""),
        }
        if str(reasoning_content or "").strip():
            message["reasoning_content"] = str(reasoning_content or "")
        if tool_calls:
            message["tool_calls"] = [
                self._trace_tool_call(call, step_number=step_number, call_index=index)
                for index, call in enumerate(tool_calls)
            ]
        return message

    @staticmethod
    def _trace_tool_call_id(call: ToolCall, *, step_number: Optional[int], call_index: int) -> str:
        if call.id:
            return str(call.id)
        step = int(step_number or 0)
        return f"call_{step:04d}_{call_index + 1:02d}"

    def _trace_tool_call(self, call: ToolCall, *, step_number: Optional[int], call_index: int) -> Dict[str, Any]:
        function_name = str(call.native_name or "").strip()
        arguments = call.native_arguments if function_name and isinstance(call.native_arguments, dict) else {}
        if not function_name:
            function_name = self._tool_call_display_name(call)
        if not arguments and not call.native_name:
            arguments = call.arguments or {}
        return {
            "id": self._trace_tool_call_id(call, step_number=step_number, call_index=call_index),
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }

    def _trace_tool_observation(
        self,
        call: ToolCall,
        result: Optional[ToolResult],
        *,
        content: str,
        step_number: Optional[int],
        call_index: int,
    ) -> Dict[str, Any]:
        name = self._tool_call_display_name(call)
        return {
            "role": "tool",
            "tool_call_id": self._trace_tool_call_id(call, step_number=step_number, call_index=call_index),
            "name": name,
            "status": str(getattr(result, "status", "") or "missing_result"),
            "content": str(content or ""),
        }

    @staticmethod
    def _is_policy_result(result: Optional[ToolResult]) -> bool:
        if result is None:
            return False
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        return bool(metadata.get("policy_error")) or str(result.provider or "") == "policy"

    def _trace_policy_feedback(
        self,
        call: ToolCall,
        result: Optional[ToolResult],
        *,
        content: str,
    ) -> Dict[str, Any]:
        metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
        code = str(metadata.get("policy_code") or "policy.feedback")
        display_name = self._tool_call_display_name(call)
        return {
            "role": "user",
            "type": "policy_feedback",
            "synthetic": True,
            "content": f"[{code}] {display_name}\n{str(content or '').strip()}",
        }

    def _runtime_feedback_messages(
        self,
        calls: List[ToolCall],
        results: List[ToolResult],
        observation_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for index, call in enumerate(calls):
            observation = observation_messages[index] if index < len(observation_messages) else {}
            content = str(observation.get("content") or "")
            display_name = self._tool_call_display_name(call)
            messages.append(
                {
                    "role": "user",
                    "type": "runtime_feedback",
                    "synthetic": True,
                    "content": f"[runtime.auto_follow] {display_name}\n{content.strip()}",
                }
            )
        return messages

    def _protocol_messages_for_results(
        self,
        calls: List[ToolCall],
        results: List[ToolResult],
        observation_messages: List[Dict[str, Any]],
    ) -> Tuple[List[ToolCall], List[Dict[str, Any]], List[Dict[str, Any]]]:
        protocol_calls: List[ToolCall] = []
        tool_observations: List[Dict[str, Any]] = []
        policy_feedback: List[Dict[str, Any]] = []
        for index, call in enumerate(calls):
            result = results[index] if index < len(results) else None
            observation = observation_messages[index] if index < len(observation_messages) else {}
            content = str(observation.get("content") or "")
            if self._is_policy_result(result):
                policy_feedback.append(self._trace_policy_feedback(call, result, content=content))
                continue
            protocol_calls.append(call)
            if isinstance(observation, dict) and observation:
                tool_observations.append(observation)
        if not calls and observation_messages:
            tool_observations.extend(observation_messages)
        return protocol_calls, tool_observations, policy_feedback

    def _invoke_model_with_retries(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> Any:
        """Call the model with retries WITHOUT mutating shared state.

        Kept state-free so it can be invoked concurrently from a thread pool;
        usage accounting is recorded by the caller on the main thread.
        """

        selected_model = model if model is not None else self.model
        max_retries = self._model_retry_count()
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                if tools:
                    try:
                        return selected_model(messages, tools=tools, tool_choice=tool_choice)
                    except TypeError:
                        try:
                            return selected_model(messages, tools=tools)
                        except TypeError as tools_exc:
                            raise TypeError(
                                "Configured FIREAgent model callable must accept native tools; "
                                "running without tools would break the training/execution protocol."
                            ) from tools_exc
                return selected_model(messages)
            except Exception as exc:
                if isinstance(exc, GlobalExperimentAbort):
                    raise
                last_error = exc
                if attempt >= max_retries:
                    break
                time.sleep(min(2**attempt, 5))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _model_retry_count() -> int:
        raw = os.getenv("FIRE_AGENT_MODEL_RETRIES", "2")
        try:
            return max(0, int(raw))
        except Exception:
            return 2

    def _record_action_model_error(
        self, state: AgentState, error_msg: str, step_number: int
    ) -> None:
        threshold = self._model_error_circuit_breaker_threshold()
        if threshold <= 0:
            return
        signature = self._model_error_signature(error_msg)
        previous = state.context_stats.get("model_error_streak")
        previous_signature = previous.get("signature") if isinstance(previous, dict) else None
        previous_count = int(previous.get("count", 0) or 0) if isinstance(previous, dict) else 0
        count = previous_count + 1 if previous_signature == signature else 1
        streak = {
            "signature": signature,
            "count": count,
            "threshold": threshold,
            "last_error": compact_text(error_msg, 1000),
            "last_step_number": step_number,
        }
        state.context_stats["model_error_streak"] = streak
        if count < threshold:
            return
        state.context_stats["model_error_circuit_breaker"] = {
            "triggered": True,
            "reason": "repeated_action_model_error",
            "signature": signature,
            "count": count,
            "threshold": threshold,
            "last_error": compact_text(error_msg, 1000),
            "last_step_number": step_number,
            "policy": (
                "Abort only after consecutive identical model-call failures in action steps; "
                "ordinary tool errors such as 403, 429, timeouts, and no rows do not trigger this breaker."
            ),
        }

    @staticmethod
    def _clear_action_model_error_streak(state: AgentState) -> None:
        state.context_stats.pop("model_error_streak", None)

    @staticmethod
    def _is_model_protocol_error(error_msg: str) -> bool:
        lowered = str(error_msg or "").lower()
        return "must accept native tools" in lowered

    @staticmethod
    def _trigger_model_protocol_error(state: AgentState, error_msg: str, step_number: int) -> None:
        state.context_stats["model_error_circuit_breaker"] = {
            "triggered": True,
            "reason": "model_protocol_error",
            "last_error": compact_text(error_msg, 1000),
            "last_step_number": step_number,
            "policy": "Abort immediately when model invocation would drop native tools.",
        }

    @staticmethod
    def _model_error_circuit_breaker_threshold() -> int:
        raw = os.getenv("FIRE_AGENT_MODEL_ERROR_CIRCUIT_BREAKER_THRESHOLD", "3")
        try:
            return max(0, int(raw))
        except Exception:
            return 3

    @staticmethod
    def _model_error_signature(error_msg: str) -> str:
        text = re.sub(r"\s+", " ", str(error_msg or "")).strip()
        lowered = text.lower()
        if "content exists risk" in lowered:
            return "model_http_400_content_exists_risk"
        if "invalid_request_error" in lowered and "http 400" in lowered:
            return "model_http_400_invalid_request_error"
        if "http 429" in lowered or "rate limit" in lowered:
            return "model_http_429_rate_limit"
        if "context_length" in lowered or "maximum context" in lowered:
            return "model_context_length"
        if "timeout" in lowered or "timed out" in lowered:
            return "model_timeout"
        return compact_text(text, 300)

    def _extract_final_answer(self, proposed: List[Any]) -> Optional[Dict[str, Any]]:
        for item in proposed:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() in FINAL_ANSWER_TOOL_ALIASES:
                return item
        return None

    def _coerce_final_answer_value(self, final_call: Dict[str, Any]) -> str:
        arguments = final_call.get("arguments")
        if isinstance(arguments, dict):
            nested = arguments.get("arguments")
            if isinstance(nested, dict) and "answer" not in arguments:
                arguments = nested
            answer = None
            for key in ("answer", "final_result", "final_answer", "result", "text"):
                if key in arguments:
                    answer = arguments.get(key)
                    break
            if isinstance(answer, (dict, list)):
                return json.dumps(answer, ensure_ascii=False)
            if answer is None:
                return ""
            return str(answer).strip()
        if isinstance(arguments, str):
            return arguments.strip()
        if arguments is not None:
            return str(arguments).strip()
        return ""

    @staticmethod
    def _final_answer_argument_key(final_call: Dict[str, Any]) -> str:
        arguments = final_call.get("arguments") if isinstance(final_call, dict) else None
        if isinstance(arguments, dict):
            nested = arguments.get("arguments")
            if isinstance(nested, dict) and "answer" not in arguments:
                arguments = nested
            for key in ("answer", "final_result", "final_answer", "result", "text"):
                if key in arguments and arguments.get(key) is not None:
                    return key
            return ""
        if isinstance(arguments, str):
            return "__string_arguments__"
        if arguments is not None:
            return "__non_dict_arguments__"
        return ""

    def _final_answer_uses_canonical_answer_key(self, final_call: Optional[Dict[str, Any]]) -> bool:
        return self._final_answer_argument_key(final_call or {}) in {"", "answer"}

    @staticmethod
    def _is_empty_final_answer_text(answer_text: Any) -> bool:
        if answer_text is None:
            return True
        return str(answer_text).strip() == ""

    @staticmethod
    def _is_terminal_non_answer_text(answer_text: Any) -> bool:
        text = "" if answer_text is None else str(answer_text).strip()
        if not text:
            return True
        lowered = text.lower()
        number_with_optional_unit_pattern = (
            r"(?<![a-zA-Z])([-+]?\d[\d,]*(?:\.\d+)?)"
            r"(\s*(?:%|bps?|pp|percentage points?|x|times|倍|个基点|个百分点|元|美元|港元|新台币|nt\$|\$|"
            r"cny|rmb|usd|hkd|twd|million|billion|万|亿))?"
        )
        numeric_candidates = re.findall(
            number_with_optional_unit_pattern,
            text,
            flags=re.IGNORECASE,
        )
        has_numeric_candidate = False
        for value, unit in numeric_candidates:
            clean_value = value.replace(",", "")
            try:
                numeric_value = float(clean_value)
            except ValueError:
                numeric_value = None
            if unit.strip() or "." in clean_value or "," in value:
                has_numeric_candidate = True
                break
            if numeric_value is not None and not (1900 <= numeric_value <= 2099 and numeric_value.is_integer()):
                has_numeric_candidate = True
                break
        has_structured_candidate = bool(
            re.search(
                r"(best[- ]?effort|estimate|estimated|candidate|result\s*[:=]|答案\s*[:：=]|估计|候选|结果\s*[:：=]|约为|大约|约)",
                lowered,
                flags=re.IGNORECASE,
            )
        )
        if has_numeric_candidate and has_structured_candidate:
            return False
        strong_patterns = [
            r"^\s*(i\s+)?(cannot|can't|can not|could not|unable to|not able to)\b",
            r"\b(insufficient|not enough|lack of)\s+(evidence|data|information)\b",
            r"\b(no|without)\s+(reliable|definitive|supported|available)\s+(answer|value|number|data|evidence)\b",
            r"\b(answer|value|number)\s+(cannot|can't|can not)\s+be\s+(determined|computed|calculated)\b",
            r"^\s*(无法|不能|未能|无法准确|无法可靠).{0,16}(确定|计算|获取|得出|回答|判断)",
            r"(证据|数据|信息).{0,8}(不足|不完整|缺失).{0,12}(无法|不能|未能)",
        ]
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in strong_patterns)

    @staticmethod
    def _final_answer_tool_call(answer_text: str, final_call: Optional[Dict[str, Any]] = None) -> ToolCall:
        source = final_call if isinstance(final_call, dict) else {}
        arguments = {"answer": "" if answer_text is None else str(answer_text)}
        return ToolCall(
            name=FINAL_ANSWER_TOOL_NAME,
            action="default",
            arguments=arguments,
            id=str(source.get("id") or ""),
            native_name=FINAL_ANSWER_TOOL_NAME,
            native_arguments=arguments,
        )

    def _record_packet_final_soft_audit(self, state: AgentState, render_mode: str) -> None:
        state.context_stats.pop("packet_final_soft_audit", None)
        mode = str(render_mode or state.context_stats.get("last_render_mode") or "").strip().lower()
        if mode not in {"lean_packet", "compact_packet"}:
            return
        audit = task_state_support_audit(state.current_task_state)
        state.context_stats["task_state_support_audit"] = audit
        if audit.get("blocking_reasons"):
            state.context_stats["packet_final_soft_audit"] = {
                "reasons": list(audit.get("blocking_reasons") or [])[:8],
                "open_slots_count": audit.get("open_slots_count"),
                "supported_count": audit.get("supported_count"),
                "proven_supported_count": audit.get("proven_supported_count"),
                "unresolved_count": audit.get("unresolved_count"),
                "unresolved_slots": audit.get("unresolved_slots"),
            }

    def _packet_tail_budget_chars(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        _, full_limit = self.context_assembler._effective_context_budgets(state, tools=tools)
        fixed_messages = self.context_assembler._base_messages(memory)
        fixed_chars = messages_char_count(fixed_messages) + len(instruction or "")
        budget = max(1, full_limit - fixed_chars)
        state.context_stats["context_packet_fixed_chars"] = fixed_chars
        state.context_stats["context_packet_tail_budget_chars"] = budget
        return budget

    def _maybe_compress_context_packet(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        render_mode: str,
        legacy_chars: int,
        step_number: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        setattr(state, "_context_compression_raw_tail_start_action_index", None)
        setattr(state, "_context_recent_steps_override", None)
        if not should_attempt_context_compression(state.config, render_mode):
            return None

        packet_mode = "compact" if str(render_mode or "").strip().lower() == "compact_packet" else "lean"
        stats = state.context_stats.setdefault("llm_context_compression", {})
        max_packet_chars = max(4000, int(getattr(state.config, "context_compression_max_packet_chars", 0) or 0))
        action_steps_count = len(getattr(memory, "action_steps", []) or [])
        packet_tail_budget = self._packet_tail_budget_chars(state, memory, instruction, tools=tools)
        default_keep_steps = min(max(0, int(state.config.context_recent_steps or 0)), action_steps_count)
        raw_tail_start_action_index = max(0, action_steps_count - default_keep_steps)
        effective_keep_steps = max(0, action_steps_count - raw_tail_start_action_index)
        setattr(state, "_context_recent_steps_override", effective_keep_steps)
        state.context_stats["last_recent_raw_effective_steps"] = effective_keep_steps
        state.context_stats["last_recent_raw_target_start_action_index"] = raw_tail_start_action_index
        cache_signature = self.context_cache.signature(
            state,
            render_mode=render_mode,
            packet_mode=packet_mode,
            raw_tail_start_action_index=raw_tail_start_action_index,
        )
        if isinstance(stats, dict):
            stats["last_render_mode"] = render_mode
        cached_packet = self.context_cache.reuse(
            state,
            memory,
            stats if isinstance(stats, dict) else {},
            cache_signature,
            max_packet_chars=max_packet_chars,
            budget_chars=packet_tail_budget,
        )
        if cached_packet:
            return cached_packet

        max_attempts = max(1, int(getattr(state.config, "context_compression_retries", 0) or 0) + 1)

        def compress_once(keep_steps: int) -> Tuple[Optional[str], Dict[str, Any], int, str]:
            raw_tail_start = max(0, action_steps_count - max(0, int(keep_steps or 0)))
            effective_keep = max(0, action_steps_count - raw_tail_start)
            setattr(state, "_context_recent_steps_override", effective_keep)
            state.context_stats["last_recent_raw_effective_steps"] = effective_keep
            state.context_stats["last_recent_raw_target_start_action_index"] = raw_tail_start
            signature = self.context_cache.signature(
                state,
                render_mode=render_mode,
                packet_mode=packet_mode,
                raw_tail_start_action_index=raw_tail_start,
            )
            draft_packet = self.context_assembler.build_working_context_packet(
                state,
                memory,
                legacy_chars,
                packet_mode=packet_mode,
            )
            if isinstance(stats, dict):
                stats["last_draft_packet_chars"] = len(draft_packet or "")
            try:
                source_payload = self._context_compression_source_payload(
                    state,
                    memory,
                    instruction,
                    render_mode=render_mode,
                    legacy_chars=legacy_chars,
                    draft_packet=draft_packet,
                    raw_tail_start_action_index=raw_tail_start,
                )
            except Exception as exc:
                if isinstance(stats, dict):
                    stats["errors"] = int(stats.get("errors", 0) or 0) + 1
                    stats["last_error"] = compact_text(f"{type(exc).__name__}: {exc}", 1000)
                return None, signature, raw_tail_start, f"{type(exc).__name__}: {exc}"

            rejected_feedback: List[Dict[str, Any]] = []
            last_reason = ""
            packet: Optional[str] = None
            for attempt in range(1, max_attempts + 1):
                if isinstance(stats, dict):
                    stats["attempted_calls"] = int(stats.get("attempted_calls", 0) or 0) + 1
                    if attempt > 1:
                        stats["retry_calls"] = int(stats.get("retry_calls", 0) or 0) + 1
                try:
                    messages = build_context_compression_messages(
                        state,
                        source_payload=source_payload,
                        attempt=attempt,
                        rejected_feedback=rejected_feedback or None,
                    )
                    response = self._call_model_with_retries(
                        state,
                        "context_compression",
                        messages,
                        step_number=step_number,
                        render_mode=render_mode,
                    )
                    content = getattr(response, "content", None)
                    if content is None:
                        content = str(response or "")
                    packet, reason = parse_context_compression_packet_with_reason(
                        content,
                        packet_mode=packet_mode,
                        max_packet_chars=max_packet_chars,
                        base_packet=draft_packet,
                    )
                    direct_sections, direct_sections_reason = is_direct_memory_sections_json(content)
                    accepted = bool(packet)
                    accepted_for_sft = accepted
                    rejected_reason = "" if accepted else (reason or "empty_or_invalid")
                    source_actions = source_payload.get("raw_action_trace")
                    covered_steps = [
                        int(item.get("step"))
                        for item in (source_actions if isinstance(source_actions, list) else [])
                        if isinstance(item, dict) and item.get("step") not in (None, "")
                    ]
                    render_payload = source_payload.get("render") if isinstance(source_payload, dict) else {}
                    render_payload = render_payload if isinstance(render_payload, dict) else {}
                    self._record_auxiliary_turn_from_response(
                        state,
                        module="context_compression",
                        turn_type="context_compression",
                        target_type="context_compression_sections",
                        messages=messages,
                        response=response,
                        step_number=step_number,
                        render_mode=render_mode,
                        include_in_sft=accepted_for_sft,
                        source={
                            "target_turn_id": self._turn_id("action", step_number),
                            "target_action_step_number": step_number,
                            "injected_into": "action_input_working_context_packet",
                            "covered_action_step_numbers": covered_steps,
                            "covered_turn_ids": [
                                self._turn_id("action", covered_step)
                                for covered_step in covered_steps
                                if covered_step > 0
                            ],
                            "attempt": attempt,
                            "accepted_by_harness": accepted,
                            "accepted_for_sft": accepted_for_sft,
                            "harness_reject_reason": "" if accepted else rejected_reason,
                            "sft_reject_reason": "" if accepted_for_sft else (reason or "empty_or_invalid"),
                            "packet_mode": packet_mode,
                            "teacher_output_contract": "memory_sections_json",
                            "teacher_output_is_direct_memory_sections_json": direct_sections,
                            "base_packet_source": "deterministic_packet_base_in_messages",
                            "harness_postprocess": "merge_memory_sections_into_deterministic_packet_base",
                            "postprocessed_artifact": "WorkingContextPacket",
                            "base_packet_chars": len(draft_packet or ""),
                            "postprocessed_packet_chars": len(packet or ""),
                            "max_packet_chars": max_packet_chars,
                            "raw_tail_start_action_index": raw_tail_start,
                            "recent_raw_keep_steps": effective_keep,
                            "rolling_raw_tail_start_action_index": render_payload.get("rolling_raw_tail_start_action_index"),
                            "compression_end_action_index": render_payload.get("compression_end_action_index"),
                        },
                    )
                    if packet:
                        if isinstance(stats, dict):
                            stats["successful_calls"] = int(stats.get("successful_calls", 0) or 0) + 1
                            stats["last_packet_chars"] = len(packet)
                            stats["last_success_attempt"] = attempt
                            stats["last_packet_source"] = "llm_sections"
                        return packet, signature, raw_tail_start, ""
                    last_reason = rejected_reason or "empty_or_invalid"
                    rejected_feedback = [
                        {
                            "attempt": attempt,
                            "reason": last_reason,
                            "max_packet_chars": max_packet_chars,
                            "instruction": (
                                "Return one valid memory-sections JSON object under the size limit. Do not copy "
                                "deterministic_packet_base or root frame fields. Use object/array section types from "
                                "the compression schema. Keep only answer-critical facts, provenance, gaps, "
                                "calculation lineage, old call outcomes, and replay handles."
                            ),
                        }
                    ]
                    if isinstance(stats, dict):
                        if str(last_reason).startswith("oversize:"):
                            stats["oversize"] = int(stats.get("oversize", 0) or 0) + 1
                        else:
                            stats["empty_or_invalid"] = int(stats.get("empty_or_invalid", 0) or 0) + 1
                        stats["last_reject_reason"] = compact_text(last_reason, 300)
                except Exception as exc:
                    if isinstance(stats, dict):
                        stats["errors"] = int(stats.get("errors", 0) or 0) + 1
                        stats["last_error"] = compact_text(f"{type(exc).__name__}: {exc}", 1000)
                    return None, signature, raw_tail_start, f"{type(exc).__name__}: {exc}"
            return None, signature, raw_tail_start, last_reason

        first_packet, first_signature, first_raw_tail_start, last_reason = compress_once(default_keep_steps)
        if not first_packet:
            if isinstance(stats, dict) and last_reason:
                stats["fallback_reason"] = compact_text(last_reason, 300)
            cached_fallback = self.context_cache.fallback(
                state,
                memory,
                stats if isinstance(stats, dict) else {},
                max_packet_chars=max_packet_chars,
                budget_chars=packet_tail_budget,
            )
            if cached_fallback:
                return cached_fallback
            return None

        def packet_fits_raw_tail(packet_text: str, keep_steps: int) -> bool:
            return len(packet_text or "") + self.context_cache.recent_raw_tail_chars(memory, keep_steps) <= packet_tail_budget

        actual_keep_steps = self.context_cache.choose_recent_raw_keep_steps(
            state,
            memory,
            packet_chars=len(first_packet),
            budget_chars=packet_tail_budget,
        )
        if actual_keep_steps < default_keep_steps:
            if isinstance(stats, dict):
                stats["adaptive_recompression"] = {
                    "from_recent_raw_steps": default_keep_steps,
                    "to_recent_raw_steps": actual_keep_steps,
                    "reason": state.context_stats.get("last_recent_raw_adaptive_reason"),
                    "first_packet_chars": len(first_packet),
                }
            adaptive_candidates: List[int] = []
            for keep_steps in (actual_keep_steps, 3, 1, 0):
                keep_steps = min(default_keep_steps, max(0, int(keep_steps or 0)))
                if keep_steps < default_keep_steps and keep_steps not in adaptive_candidates:
                    adaptive_candidates.append(keep_steps)
            adaptive_failures: List[Dict[str, Any]] = []
            for keep_steps in adaptive_candidates:
                next_packet, next_signature, next_raw_tail_start, next_reason = compress_once(keep_steps)
                if next_packet and packet_fits_raw_tail(next_packet, keep_steps):
                    self.context_cache.store(
                        state,
                        next_packet,
                        next_signature,
                        max_packet_chars=max_packet_chars,
                    )
                    setattr(state, "_context_compression_raw_tail_start_action_index", next_raw_tail_start)
                    if isinstance(stats, dict):
                        stats["adaptive_recompression_success"] = {
                            "recent_raw_steps": keep_steps,
                            "packet_chars": len(next_packet),
                            "raw_tail_start_action_index": next_raw_tail_start,
                        }
                    return next_packet
                adaptive_failures.append(
                    {
                        "recent_raw_steps": keep_steps,
                        "reason": next_reason or ("packet_plus_raw_tail_exceeds_budget" if next_packet else "empty_or_invalid"),
                        "packet_chars": len(next_packet or ""),
                    }
                )
            if isinstance(stats, dict):
                stats["adaptive_recompression_failures"] = adaptive_failures
                stats["adaptive_recompression_fallback_reason"] = (
                    "all_adaptive_candidates_failed; deterministic_packet_fallback"
                )
            return None

        if not packet_fits_raw_tail(first_packet, default_keep_steps):
            if isinstance(stats, dict):
                stats["first_packet_rejected_reason"] = "packet_plus_default_raw_tail_exceeds_budget"
            return None
        self.context_cache.store(
            state,
            first_packet,
            first_signature,
            max_packet_chars=max_packet_chars,
        )
        setattr(state, "_context_compression_raw_tail_start_action_index", first_raw_tail_start)
        return first_packet
        return None

    def _context_compression_source_payload(
        self,
        state: AgentState,
        memory: AgentMemory,
        instruction: str,
        *,
        render_mode: str,
        legacy_chars: int,
        draft_packet: str,
        raw_tail_start_action_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        recent_numbers = set(self.context_assembler._recent_action_numbers(state))
        old_obs_limit = max(800, int(state.config.context_compression_old_observation_step_chars or 0))
        recent_obs_limit = max(1200, int(state.config.context_compression_recent_observation_step_chars or 0))
        action_count = len(getattr(memory, "action_steps", []) or [])
        compression_end_index = max(0, int(raw_tail_start_action_index if raw_tail_start_action_index is not None else action_count))
        compression_end_index = min(compression_end_index, action_count)
        prior_cache = getattr(state, "_context_compression_cache", None)
        prior_packet = ""
        prior_signature: Dict[str, Any] = {}
        rolling_start_index = 0
        if isinstance(prior_cache, dict):
            cached_packet = prior_cache.get("packet")
            cached_signature = prior_cache.get("signature")
            if isinstance(cached_packet, str) and cached_packet.strip() and isinstance(cached_signature, dict):
                cached_raw_tail_start = self.context_cache.raw_tail_start(cached_signature)
                if cached_raw_tail_start <= compression_end_index:
                    prior_packet = cached_packet
                    prior_signature = cached_signature
                    rolling_start_index = cached_raw_tail_start

        source_action_steps = (
            memory.action_steps[rolling_start_index:compression_end_index]
            if rolling_start_index <= compression_end_index
            else []
        )
        actions: List[Dict[str, Any]] = []
        for step in source_action_steps:
            is_recent = int(step.step_number or 0) in recent_numbers
            obs_limit = recent_obs_limit if is_recent else old_obs_limit
            actions.append(
                {
                    "step": step.step_number,
                    "is_after_previous_compression": bool(rolling_start_index > 0),
                    "is_recent_raw_window": is_recent,
                    "assistant_action": {
                        "think": compact_text(step.think or "", 800),
                        "tools": self._compression_compact_value(step.tool_calls or [], 2400),
                    },
                    "observation_excerpt": compact_text(step.observations or "", obs_limit),
                    "error": compact_text(str(step.error or ""), 800) if step.error else "",
                }
            )

        include_draft = bool(getattr(state.config, "context_compression_include_draft_packet", True))
        max_source = max(8000, int(state.config.context_compression_max_source_chars or 0))
        packet_mode = "compact" if str(render_mode or "").strip().lower() == "compact_packet" else "lean"
        deterministic_base = context_packet_object_from_text(draft_packet, packet_mode=packet_mode)
        payload: Dict[str, Any] = {
            "task": {
                "bench_name": state.task_context.bench_name,
                "task_index": state.task_context.task_index,
                "task_prompt": compact_text(state.task_context.task_prompt, 2200),
            },
            "render": {
                "render_mode": render_mode,
                "legacy_prompt_chars_if_full": legacy_chars,
                "recent_raw_steps": sorted(recent_numbers),
                "rolling_raw_tail_start_action_index": rolling_start_index,
                "compression_end_action_index": compression_end_index,
                "rolling_raw_tail_action_count": len(source_action_steps),
                "current_step_instruction_excerpt": compact_text(instruction, 5000),
            },
            "raw_action_trace": actions,
            "rolling_compression": {
                "has_previous_compressed_packet": bool(prior_packet),
                "previous_packet_action_steps": prior_signature.get("action_steps"),
                "previous_packet_raw_tail_start_action_index": prior_signature.get("raw_tail_start_action_index"),
                "previous_packet_progress_seq": prior_signature.get("progress_seq"),
                "previous_packet_task_state_fingerprint": prior_signature.get("task_state_fingerprint"),
                "raw_tail_replaces_full_history_scan": bool(prior_packet),
            },
            "runtime_ledgers": {
                "current_task_state": self._compression_compact_value(state.current_task_state or {}, 6000),
                "action_ledger": self._compression_tail_records(state.action_ledger, 80, 12000),
                "evidence_ledger": self._compression_tail_records(state.evidence_ledger, 80, 16000),
                "answer_critical_pins": self._compression_tail_records(
                    self.context_assembler._answer_critical_pins(state),
                    80,
                    14000,
                ),
                "calc_ledger": self._compression_tail_records(state.calc_ledger, 64, 8000),
                "artifact_refs": self._compression_tail_records(
                    [{key: value for key, value in record.items() if key != "content"} for record in state.artifact_store.values()],
                    40,
                    8000,
                ),
            },
            "compression_contract": {
                "output": "Return one memory-sections JSON object only. The runtime merges it into deterministic_packet_base.",
                "format_rules": [
                    "The first non-whitespace character must be { and the last non-whitespace character must be }.",
                    "Do not output markdown fences, labels, prose, comments, apologies, or analysis outside JSON.",
                    "Do not copy deterministic_packet_base or emit root frame fields such as render_mode/task_card/recent_raw_window.",
                    "Use JSON objects for memory object sections and JSON arrays for memory list sections.",
                    "old_action_ledger.items and evidence_ledger.items must be arrays.",
                ],
                "do_not_modify_memory": True,
                "packet_replaces": "older action/observation history and durable working state outside the selected recent raw turns",
                "older_tool_calls_are_compacted_in_packet": True,
                "recent_raw_turns_are_appended_after_packet": True,
                "must_preserve": [
                    "answer-critical facts and units",
                    "source and locator provenance",
                    "calculation lineage",
                    "open gaps and blocked paths",
                    "old tool-call outcome chronology by step/call_id, without duplicating full arguments",
                    "retry/replay constraints",
                ],
            },
        }
        if prior_packet:
            payload["previous_compressed_packet"] = self._compression_compact_value(
                context_packet_object_from_text(prior_packet, packet_mode=packet_mode),
                max(12000, min(36000, max_source // 2)),
            )
        if include_draft:
            draft_limit = 12000 if prior_packet else max(12000, max_source // 2)
            payload["deterministic_packet_base"] = self._compression_compact_value(deterministic_base, draft_limit)
        return payload

    @staticmethod
    def _compression_tail_records(records: List[Dict[str, Any]], limit: int, char_limit: int) -> Any:
        selected = records[-limit:] if limit and len(records) > limit else records
        output = [
            FIREAgent._compression_compact_value(record, max(1200, min(3200, char_limit // max(1, min(len(selected), limit or len(selected) or 1)))))
            for record in selected
        ]
        while len(output) > 1 and len(json_compact(output)) > char_limit:
            output = output[-max(1, len(output) // 2):]
        if len(json_compact(output)) > char_limit:
            return [{"truncated": True, "json_excerpt": compact_text(json_compact(output), char_limit)}]
        return output

    @staticmethod
    def _compression_compact_value(value: Any, char_limit: int) -> Any:
        clipped = FIREAgent._compression_clip_value(value)
        if len(json_compact(clipped)) <= char_limit:
            return clipped
        if isinstance(clipped, dict):
            return {"truncated": True, "json_excerpt": compact_text(json_compact(clipped), char_limit)}
        if isinstance(clipped, list):
            return [{"truncated": True, "json_excerpt": compact_text(json_compact(clipped), char_limit)}]
        return compact_text(str(clipped), char_limit)

    @staticmethod
    def _compression_clip_value(value: Any, *, depth: int = 0, key: str = "") -> Any:
        if depth > 5:
            return compact_text(json_compact(value), 600)
        if isinstance(value, str):
            lowered_key = key.lower()
            if lowered_key in {"content", "raw_content", "html", "raw_html", "full_text"}:
                return compact_text(value, 400)
            if lowered_key in {"result_summary", "observation_excerpt", "quote_or_row_excerpt", "quote", "snippet"}:
                return compact_text(value, 900)
            if lowered_key in {"arguments", "metadata", "schema"}:
                return compact_text(value, 1200)
            return compact_text(value, 1600)
        if isinstance(value, dict):
            output: Dict[str, Any] = {}
            for item_key, item_value in value.items():
                if item_key in {"content", "raw_content", "html", "raw_html", "full_text"}:
                    continue
                output[str(item_key)] = FIREAgent._compression_clip_value(item_value, depth=depth + 1, key=str(item_key))
            return output
        if isinstance(value, list):
            selected = value[-50:] if len(value) > 50 else value
            return [FIREAgent._compression_clip_value(item, depth=depth + 1, key=key) for item in selected]
        return value

    def _format_observations(
        self,
        calls: List[ToolCall],
        results: List[ToolResult],
        state: Optional[AgentState] = None,
    ) -> str:
        if not calls and not results:
            return "No tool calls were executed in this step."
        chunks: List[str] = []
        for call, result in zip(calls, results):
            chunks.append(self._format_single_observation(state, call, result))
        for call in calls[len(results):]:
            chunks.append(f"[{self._tool_call_display_name(call)}] (no result)")
        return "\n\n".join(chunks)

    def _format_observation_pair(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        *,
        step_number: int,
        think: str = "",
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        if not calls and not results:
            text = "No tool calls were executed in this step."
            return text, text, [{"role": "user", "content": text}]

        full_chunks: List[str] = []
        prompt_chunks: List[str] = []
        observation_messages: List[Dict[str, Any]] = []
        for call_index, (call, result) in enumerate(zip(calls, results)):
            self._record_tool_auxiliary_turns(
                state,
                call,
                result,
                step_number=step_number,
                call_index=call_index,
            )
            full_single = self._format_single_observation(state, call, result)
            prompt_single, artifact_ids = self._format_prompt_single_observation(state, call, result, full_single)
            full_chunks.append(full_single)
            prompt_chunks.append(prompt_single)
            observation_messages.append(
                self._trace_tool_observation(
                    call,
                    result,
                    content=prompt_single,
                    step_number=step_number,
                    call_index=call_index,
                )
            )
            ledger_observation = (
                prompt_single
                if self._should_store_ledger_observation_excerpt(state, call, result, prompt_single, artifact_ids)
                else ""
            )
            record_action_ledger(
                state,
                step_number=step_number,
                call=call,
                result=result,
                full_chars=len(full_single),
                prompt_chars=len(prompt_single),
                call_index=call_index,
                think=think,
                artifact_ids=artifact_ids,
                prompt_observation=ledger_observation,
            )

        for missing_index, call in enumerate(calls[len(results):], start=len(results)):
            text = f"[{self._tool_call_display_name(call)}] (no result)"
            full_chunks.append(text)
            prompt_chunks.append(text)
            observation_messages.append(
                self._trace_tool_observation(
                    call,
                    None,
                    content=text,
                    step_number=step_number,
                    call_index=missing_index,
                )
            )
            record_action_ledger(
                state,
                step_number=step_number,
                call=call,
                result=None,
                full_chars=len(text),
                prompt_chars=len(text),
                call_index=missing_index,
                think=think,
                prompt_observation=text,
            )
        return "\n\n".join(full_chunks), "\n\n".join(prompt_chunks), observation_messages

    def _should_store_ledger_observation_excerpt(
        self,
        state: AgentState,
        call: ToolCall,
        result: ToolResult,
        prompt_observation: str,
        artifact_ids: List[str],
    ) -> bool:
        if not prompt_observation:
            return False
        if not should_use_context_system(state.config):
            return True
        if artifact_ids:
            return True
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if metadata.get("failure_card"):
            return True
        if result.status != "success":
            return True
        last_render = str(state.context_stats.get("last_render_mode") or "").strip().lower()
        if last_render in {"lean_packet", "compact_packet"}:
            return True
        if call.name == "calculator":
            return True
        return len(prompt_observation) <= max(800, int(state.config.context_recent_tool_call_excerpt_chars or 0))

    def _format_prompt_single_observation(
        self,
        state: AgentState,
        call: ToolCall,
        result: ToolResult,
        full_observation: str,
    ) -> Tuple[str, List[str]]:
        if (
            should_use_context_system(state.config)
            and len(full_observation) > state.config.context_artifact_char_threshold
        ):
            preview, artifact_id = build_artifact_preview(state, call, result, full_observation)
            if len(preview) >= len(full_observation):
                preview = compact_text(full_observation, state.config.context_artifact_preview_chars)
            return preview, [artifact_id]
        return full_observation, []

    def _format_single_observation(self, state: Optional[AgentState], call: ToolCall, result: ToolResult) -> str:
        header = f"[{self._tool_call_display_name(call)}] status={result.status}"
        card_lines = self._metadata_card_lines(result)
        if result.status != "success" and result.error:
            card_text = "\n".join(card_lines)
            parts = [header]
            if card_text:
                parts.append(card_text)
            parts.append(f"error: {self._compact_error(result.error)}")
            hint = self._tool_validation_hint(call, result.error)
            if hint:
                parts.append(hint)
            return "\n".join(parts)
        body = result.observation_text()
        if card_lines and not any(line.split(" ", 1)[0] in body for line in card_lines):
            body = "\n".join(card_lines + ([body] if body else []))
        if state is not None and result.status == "success":
            schema = tool_observation_schema(call, result)
            if schema:
                schema_text = json.dumps(schema, ensure_ascii=False, default=str, indent=2)
                body = f"OBSERVATION_SCHEMA\n{schema_text}\nDATA\n{body}" if body else f"OBSERVATION_SCHEMA\n{schema_text}"
        if not body and result.tables:
            body = self._format_json(result.tables)
        document_line = self._format_reader_document_line(call, result)
        if document_line:
            body = f"{document_line}\n{body}" if body else document_line
        return f"{header}\n{body}" if body else header

    def _format_reader_document_line(self, call: ToolCall, result: ToolResult) -> str:
        document_key = str((result.metadata or {}).get("document_key") or "").strip()
        if not document_key or result.status != "success":
            return ""
        payload = {
            "document_key": document_key,
            "stored_chars": result.metadata.get("stored_chars"),
            "cached": bool(result.metadata.get("cached")),
            "source": result.metadata.get("url") or result.metadata.get("source"),
        }
        if result.metadata.get("document_kind") == "sec_tables":
            return (
                f"document: {self._format_json(payload)}\n"
                f"note: complete SEC tables are saved for this task; re-read with {self._tool_call_display_name(call)} "
                "using document_key, table_index, and either query or row_start/row_end for precise rows."
            )
        return (
            f"document: {self._format_json(payload)}\n"
            f"note: full source text is saved for this task; re-read with {self._tool_call_display_name(call)} "
            f"using document_key, and optionally start/end character indices (end-exclusive) "
            f"to scope a large filing if more specific evidence is needed."
        )

    @staticmethod
    def _metadata_card_lines(result: ToolResult) -> List[str]:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        lines: List[str] = []
        for metadata_key, label in (
            ("execution_card", "EXECUTION_CARD"),
            ("failure_card", "FAILURE_CARD"),
        ):
            card = metadata.get(metadata_key)
            if not isinstance(card, dict):
                continue
            try:
                encoded = json.dumps(card, ensure_ascii=False, default=str, separators=(",", ":"))
            except Exception:
                encoded = str(card)
            lines.append(f"{label} {encoded}")
        return lines

    @staticmethod
    def _compact_error(text: str) -> str:
        text = str(text or "")
        # Strip giant HTML / cloudflare blobs that periodically come back from
        # blocked endpoints; the on-disk trace shouldn't carry kilobytes of them.
        if len(text) > 256:
            return text[:256] + "...(truncated)"
        return text

    def _format_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _tool_validation_hint(self, call: ToolCall, error: str) -> str:
        lowered = str(error or "").lower()
        if not self._is_non_retryable_validation_error(lowered):
            return ""
        schema = self._action_parameters_schema(self._tool_action_spec(call.name, call.action))
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        lines = [
            "validation_hint:",
            f"- function: {self._native_function_name(call.name, call.action, self._tool_action_count(call.name))}",
        ]
        if required:
            lines.append("- required: " + ", ".join(str(item) for item in required))
        typed_parts = []
        for key in list(required) + [key for key in properties if key not in set(required)]:
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            prop_type = prop.get("type")
            if prop_type:
                typed_parts.append(f"{key}:{prop_type}")
            if len(typed_parts) >= 10:
                break
        if typed_parts:
            lines.append("- argument_types: " + ", ".join(typed_parts))
        examples = self._tool_action_spec(call.name, call.action).get("examples")
        if isinstance(examples, list) and examples:
            try:
                example = json.dumps(examples[0], ensure_ascii=False, separators=(",", ":"))
            except Exception:
                example = str(examples[0])
            lines.append("- valid_example: " + compact_text(example, 500))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def _build_system_prompt(self, state: AgentState) -> str:
        return (
            self._canonical_system_prompt_template(state)
            .replace("{max_tool_calls}", str(self.config.max_tool_calls_per_round))
            .replace("{tool_operating_rules}", self._system_tool_operating_rules(state))
            .replace("{benchmark_constraints}", self._canonical_benchmark_system_constraints(state))
        )

    def _canonical_system_prompt_template(self, state: AgentState) -> str:
        if not self._is_final_answer_only_round(state):
            return CANONICAL_SYSTEM_PROMPT
        return CANONICAL_SYSTEM_PROMPT.replace(
            "You solve finance, market data, filings, and document-grounded tasks by reasoning about evidence and using the available tools.",
            "You solve finance, market data, filings, and document-grounded tasks by reasoning from the task context and answering through the native final_answer channel.",
        )

    def _initial_planning_prompt(self, state: AgentState) -> str:
        return INITIAL_PLANNING_PROMPT.replace(
            "{benchmark_planning_constraints}",
            self._benchmark_planning_constraints(state),
        )

    def _benchmark_system_constraints(self, state: AgentState) -> str:
        if state.config.benchmark_system_prompt:
            return "\n\nBenchmark-specific constraints:\n" + state.config.benchmark_system_prompt.strip() + "\n"
        if self._is_finance_agent_bench(state):
            return FINANCE_AGENT_BENCH_SYSTEM_CONSTRAINTS
        return ""

    def _benchmark_planning_constraints(self, state: AgentState) -> str:
        if state.config.benchmark_planning_prompt:
            return "\n" + state.config.benchmark_planning_prompt.strip() + "\n"
        if self._is_finance_agent_bench(state):
            return FINANCE_AGENT_BENCH_PLANNING_CONSTRAINTS
        return ""

    def _is_finance_agent_bench(self, state: AgentState) -> bool:
        return str(state.task_context.bench_name or "").strip().lower() == FINANCE_AGENT_BENCH_NAME

    def _step_instruction(self, state: AgentState, step_number: int, *, task_state_block: str) -> str:
        profile_block = self._profile_block(state)
        if profile_block:
            profile_block = "\nAvailable resources:\n" + profile_block + "\n"
        tool_policy_block = self._tool_policy_block(state)
        if not self.tools:
            tool_policy_block = (
                "Only the final_answer tool is available for this benchmark. "
                "No external evidence, retrieval, browsing, filing, market-data, calculation, search, PDF, web, or SEC tools are available. "
                "Finish now with final_answer; put only the benchmark answer atom/value in final_answer.arguments.answer.\n"
                + tool_policy_block
            )
        if tool_policy_block:
            tool_policy_block = "\nBenchmark tool-use constraints:\n" + tool_policy_block

        return (
            CANONICAL_STEP_INSTRUCTION
            .replace("{profile_block}", profile_block)
            .replace("{tool_policy_block}", tool_policy_block)
            .replace("{task_state_block}", task_state_block)
            .replace("{max_tool_calls}", str(self.config.max_tool_calls_per_round))
            .replace("{step_tool_action_rules}", self._step_tool_action_rules(state))
        )

    def _system_tool_operating_rules(self, state: AgentState) -> str:
        if self._is_final_answer_only_round(state):
            return (
                "- No external evidence, retrieval, browsing, filing, market-data, calculation, search, PDF, web, or SEC tools are available.\n"
                 "- The only available tool is final_answer; use it as the native answer channel.\n"
                 "- In the same assistant response, assistant.content must be a compact JSON object with keys think and task_state.\n"
                 "- content.think is required and must be non-empty; {\"think\":\"\",\"task_state\":{}} is invalid.\n"
                 "- content.think must be one short public sentence describing the evidence/check/calculation basis in abstract terms, without answer values, answer text, option labels, or final answer atoms.\n"
                "- For single-turn QA, content.task_state must be exactly an empty object {}; do not put answer values, evidence snippets, or working memory in task_state.\n"
                 "- Emit the benchmark answer only through the OpenAI-native final_answer tool-call, with the answer in final_answer.arguments.answer.\n"
                 "- Do not put the benchmark answer in assistant.content, reasoning_content, plain text, or markdown."
             )
        return (
            "- When evidence or calculation is needed, use the relevant available tool immediately; do not merely describe the intended tool in the note.\n"
            "- Do not use final_answer together with other tools in the same step.\n"
            "- When finishing, emit final_answer as an OpenAI-native tool-call; do not put the final answer in assistant.content.\n"
            "- Use calculator for arithmetic instead of mental math."
        )

    def _step_tool_action_rules(self, state: AgentState) -> str:
        if self._is_final_answer_only_round(state):
            return (
                "Only the final_answer tool is available in this step. "
                "Finish with exactly one OpenAI-native final_answer tool-call now. "
                "Use two separate output channels:\n"
                "1. assistant.content must be exactly one compact JSON object with keys think and task_state. "
                "content.think is mandatory and must be a non-empty one-sentence public brief reasoning process. "
                 "Never emit an empty think string; {\"think\":\"\",\"task_state\":{}} is invalid. "
                 "Use an abstract note such as "
                 "{\"think\":\"checked the provided evidence against the requested metric and output format\",\"task_state\":{}}. "
                 "Do not include the answer value, answer text, option label, or any final answer atom in content.think.\n"
                "content.task_state must be exactly {}; single-turn QA has no next step, so do not store answer values, evidence snippets, or working memory there.\n"
                "2. final_answer.arguments must be exactly {\"answer\":\"<benchmark answer>\"}.\n"
                 "Do not put the assistant.content JSON inside final_answer.arguments.answer. "
                 "The think field is a short public reasoning summary, not private chain-of-thought; mention only the evidence/check/calculation basis. "
                 "Do not write the answer in assistant.content, plain text, or markdown."
             )
        return (
            f"Use at most {self.config.max_tool_calls_per_round} evidence/calculation tools in this step.\n"
            "If the answer is ready, finish with exactly one OpenAI-native final_answer tool-call and do not use other tools.\n"
            "Do not write the final answer in assistant.content, plain text, or markdown.\n"
            "If more evidence or calculation is needed, use the specific available tool needed for the current gap immediately; "
            "do not merely state the intended tool in assistant content."
        )

    def _canonical_benchmark_system_constraints(self, state: AgentState) -> str:
        text = self._benchmark_system_constraints(state)
        return text

    @staticmethod
    def _capture_task_state(state: AgentState, parsed: Dict[str, Any], *, step_number: int) -> None:
        """Persist the model-owned working memory across action steps.

        Compact evidence and stale paths stay visible in TaskState while
        fact-like updates also move into EvidenceLedger for packet indexing.
        """
        raw_candidate = parsed.get("task_state")
        candidate = canonicalize_task_state(raw_candidate)
        stats = state.context_stats
        if candidate:
            stats["task_state_update_count"] = int(stats.get("task_state_update_count") or 0) + 1
            if isinstance(raw_candidate, dict):
                free_keys = sorted(
                    str(key)
                    for key in raw_candidate
                    if str(key) not in TASK_STATE_VIEW_KEYS
                )
                if free_keys:
                    stats["task_state_normalized_update_count"] = (
                        int(stats.get("task_state_normalized_update_count") or 0) + 1
                    )
                    stats["task_state_last_normalized_keys"] = free_keys[:24]
            record_evidence_items(
                state,
                candidate.get("evidence"),
                origin="task_state.evidence",
                step_number=step_number,
            )
            record_answer_pins(
                state,
                candidate.get("answer_pins"),
                origin="task_state.answer_pins",
                step_number=step_number,
            )
        else:
            stats["task_state_empty_update_count"] = (
                int(stats.get("task_state_empty_update_count") or 0) + 1
            )
            stats["task_state_runtime_scaffold_count"] = (
                int(stats.get("task_state_runtime_scaffold_count") or 0) + 1
            )
        state.current_task_state = complete_task_state_snapshot(
            state.current_task_state,
            candidate,
            task=state.task_context.task_prompt,
            revision=step_number,
        )
        stats["task_state_last_schema_keys"] = list(state.current_task_state or {})
        snapshot_issues = task_state_snapshot_issues(state.current_task_state)
        if snapshot_issues:
            stats["task_state_snapshot_issue_count"] = (
                int(stats.get("task_state_snapshot_issue_count") or 0) + 1
            )
            stats["task_state_last_snapshot_issues"] = snapshot_issues
        evidence_updates = parsed.get("evidence_updates")
        record_evidence_items(
            state,
            evidence_updates,
            origin="evidence_updates",
            step_number=step_number,
        )

    def _task_state_block(self, state: AgentState, *, view: bool, inline: bool = True) -> str:
        ts = state.current_task_state
        if view and not inline:
            state.context_stats["task_state_render_mode"] = "view"
            return (
                "\nTaskStateView is included in WorkingContextPacket. Use it as compact working memory. "
                "Keep the full updated task_state in your working note, and use available tools when needed.\n"
                + TASK_STATE_SCHEMA_INSTRUCTION
                + "\n"
            )
        if not ts:
            if self._is_final_answer_only_round(state):
                state.context_stats["task_state_render_mode"] = "view" if view else "full"
                return (
                    "\nCurrent TaskStateView: (empty). This is single-turn QA, so content.task_state must remain exactly {}. "
                    "Use content.think only for one short public evidence/check basis, and put the answer only in final_answer.arguments.answer.\n"
                )
            if not view:
                return (
                    "\nCurrent TaskState: (empty). Start the answer contract from the task and observations, "
                    "then keep the updated task_state in your working note.\n"
                    + TASK_STATE_SCHEMA_INSTRUCTION
                    + "\n"
                )
            return (
                "\nCurrent TaskStateView: (empty). Start the answer contract from the task and observations, "
                "then keep the updated task_state in your working note.\n"
                + TASK_STATE_SCHEMA_INSTRUCTION
                + "\n"
            )
        if view:
            rendered_obj = task_state_prompt_view(ts)
            try:
                rendered = json.dumps(rendered_obj, ensure_ascii=False, indent=2)
            except Exception:
                rendered = str(rendered_obj)
            block = (
                "\nCurrent TaskStateView (compact working memory; older tool history, calculations, "
                "deterministic ledgers, and any pinned facts/slices are in the selected context view):\n"
                + rendered
                + "\nUse this compact working memory to choose the next available tool or final_answer. "
                "Keep the full updated task_state in your working note.\n"
                + TASK_STATE_SCHEMA_INSTRUCTION
                + "\n"
            )
            state.context_stats["task_state_render_mode"] = "view"
            return block
        try:
            rendered = json.dumps(ts, ensure_ascii=False, indent=2)
        except Exception:
            rendered = str(ts)
        block = (
            "\nCurrent TaskState (working memory carried forward by the harness; keep the full updated task_state in your working note):\n"
            + rendered
            + "\n"
            + TASK_STATE_SCHEMA_INSTRUCTION
            + "\n"
        )
        state.context_stats["task_state_render_mode"] = "full"
        return block

    def _profile_block(self, state: AgentState) -> str:
        ctx = state.task_context
        lines: List[str] = []
        if ctx.attachments:
            lines.append("- attachments: " + json.dumps(ctx.attachments[:8], ensure_ascii=False))
        if ctx.urls:
            lines.append("- urls: " + json.dumps(ctx.urls[:8], ensure_ascii=False))
        return "\n".join(lines)

    def _tool_policy_block(self, state: AgentState) -> str:
        cleaned: List[str] = []
        for tool_name in self.tools:
            if tool_name == MARKET_DATA_TOOL_NAME:
                continue
            constraints = state.config.tool_prompt_constraints.get(tool_name, [])
            for item in constraints:
                text = str(item).strip()
                if text:
                    cleaned.append(f"{tool_name}: {text}")
        extra = state.task_context.bench_extra or {}
        if not isinstance(extra, dict):
            return "\n".join(f"- {item}" for item in cleaned) + ("\n" if cleaned else "")
        constraints = extra.get("tool_prompt_constraints") or []
        if isinstance(constraints, list):
            cleaned.extend(
                text
                for text in (str(item).strip() for item in constraints)
                if text and not text.lower().startswith(f"{MARKET_DATA_TOOL_NAME}:")
            )
        if not cleaned:
            return ""
        return "\n".join(f"- {item}" for item in cleaned) + "\n"

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _execute_tool_round(self, state: AgentState, calls: List[ToolCall]) -> List[ToolResult]:
        if not calls:
            return []
        results_by_index: Dict[int, ToolResult] = {}
        if any(not self.tools[call.name].parallel_safe for call in calls):
            for idx, call in enumerate(calls):
                try:
                    result = self.tools[call.name].run(call, state.config.tool_timeout_seconds)
                except Exception as exc:
                    result = ToolResult(
                        call.name,
                        call.name,
                        "error",
                        action=call.action,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results_by_index[idx] = result
            return [results_by_index[idx] for idx in sorted(results_by_index)]

        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            futures = {
                executor.submit(self.tools[call.name].run, call, state.config.tool_timeout_seconds): (idx, call)
                for idx, call in enumerate(calls)
            }
            for future in as_completed(futures):
                idx, call = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ToolResult(
                        call.name,
                        call.name,
                        "error",
                        action=call.action,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results_by_index[idx] = result

        return [results_by_index[idx] for idx in sorted(results_by_index)]

    def _append_structured_table_auto_follow(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        *,
        executed_count: int,
        step_number: int,
    ) -> None:
        stats = state.context_stats.setdefault("structured_table_auto_follow", {})
        if not isinstance(stats, dict):
            return
        if executed_count >= state.config.max_tool_calls_per_round:
            self._record_structured_table_auto_follow_skip(stats, "max_tool_calls_reached")
            return
        seen = {self._tool_call_signature(call) for call in calls}
        for call, result in zip(list(calls), list(results)):
            follow_call = self._structured_table_auto_follow_call(
                call,
                result,
                step_number=step_number,
                call_index=len(calls),
            )
            if follow_call is None:
                continue
            signature = self._tool_call_signature(follow_call)
            if signature in seen or self._has_equivalent_structured_table_read(calls, follow_call):
                self._record_structured_table_auto_follow_skip(stats, "duplicate_recommended_read_tables")
                continue
            blocked_observation = self._blocked_tool_call_observation(state, follow_call)
            if blocked_observation:
                calls.append(follow_call)
                results.append(
                    self._policy_tool_result(
                        follow_call,
                        blocked_observation,
                        code="policy.blocked_tool_call",
                    )
                )
                self._record_structured_table_auto_follow_skip(stats, "blocked_tool_call")
                return
            auto_results = self._execute_tool_round(state, [follow_call])
            auto_result = auto_results[0] if auto_results else self._policy_tool_result(
                follow_call,
                f"[policy.missing_tool_result] no result was produced for {self._tool_call_display_name(follow_call)}.",
                code="policy.missing_tool_result",
            )
            calls.append(follow_call)
            results.append(auto_result)
            stats["executed"] = int(stats.get("executed", 0) or 0) + 1
            stats["last_step_number"] = step_number
            stats["last_call"] = follow_call.to_lean_dict()
            self._record_tool_progress_signals(state, [follow_call], [auto_result], step_number)
            self._record_tool_failure_fuses(state, [follow_call], [auto_result], step_number)
            return

    def _structured_table_auto_follow_call(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        step_number: int,
        call_index: int,
    ) -> Optional[ToolCall]:
        if call.name != "structured_table_reader" or call.action != "discover_tables":
            return None
        if result.status != "success":
            return None
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        recommended = metadata.get("recommended_next")
        if not isinstance(recommended, dict):
            payload = metadata.get("discovery_payload")
            if isinstance(payload, dict):
                recommended = payload.get("recommended_next")
        if not isinstance(recommended, dict) or not recommended:
            return None
        tool_name = str(recommended.get("tool") or call.name).strip()
        if tool_name != "structured_table_reader" or tool_name not in self.tools:
            return None
        action = self.tools[tool_name].normalize_action(str(recommended.get("action") or "").strip())
        if action != "read_tables":
            return None
        raw_arguments = recommended.get("arguments")
        arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
        if not (arguments.get("url") or arguments.get("file_path")):
            return None
        return ToolCall(
            name=tool_name,
            action=action,
            arguments=arguments,
            id=f"call_{step_number:04d}_{call_index + 1:02d}_auto",
            rationale="auto_follow structured_table_reader.discover_tables recommended_next",
            native_name=self._native_function_name(tool_name, action, self._tool_action_count(tool_name)),
            native_arguments=arguments,
        )

    @staticmethod
    def _tool_call_signature(call: ToolCall) -> str:
        try:
            arguments = json.dumps(call.arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            arguments = str(call.arguments or {})
        return f"{call.name}\n{call.action}\n{arguments}"

    @staticmethod
    def _has_equivalent_structured_table_read(calls: List[ToolCall], candidate: ToolCall) -> bool:
        if candidate.name != "structured_table_reader" or candidate.action != "read_tables":
            return False
        candidate_source = str(
            (candidate.arguments or {}).get("url") or (candidate.arguments or {}).get("file_path") or ""
        ).strip()
        if not candidate_source:
            return False
        for call in calls:
            if call.name != "structured_table_reader" or call.action != "read_tables":
                continue
            source = str((call.arguments or {}).get("url") or (call.arguments or {}).get("file_path") or "").strip()
            if source and source == candidate_source:
                return True
        return False

    @staticmethod
    def _record_structured_table_auto_follow_skip(stats: Dict[str, Any], reason: str) -> None:
        stats["skipped"] = int(stats.get("skipped", 0) or 0) + 1
        stats["last_skip_reason"] = reason

    def _record_task_state_progress_signal(self, state: AgentState, step_number: int) -> None:
        fingerprint = self._task_state_progress_fingerprint(state.current_task_state)
        if not fingerprint:
            return
        progress = state.context_stats.setdefault("progress_signals", {})
        if not isinstance(progress, dict):
            return
        previous = str(progress.get("task_state_fingerprint") or "")
        if previous == fingerprint:
            return
        progress["task_state_fingerprint"] = fingerprint
        self._bump_progress_signal(state, "task_state_or_batch_coverage_delta", step_number)

    def _record_tool_progress_signals(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        step_number: int,
    ) -> None:
        for call, result in zip(calls, results):
            if result.status != "success":
                continue
            if call.name == "web_reader":
                self._bump_progress_signal(state, f"{call.name}_success", step_number)
                continue
            if self._structured_tool_result_has_progress(call, result):
                self._bump_progress_signal(state, f"{call.name}.{call.action}_structured_progress", step_number)

    def _record_web_search_soft_fuse_results(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        step_number: int,
    ) -> None:
        threshold = self._web_search_soft_fuse_threshold()
        if threshold <= 0:
            return
        fuse_state = state.context_stats.setdefault("web_search_soft_fuse", {})
        if not isinstance(fuse_state, dict):
            return
        clusters = fuse_state.setdefault("clusters", {})
        if not isinstance(clusters, dict):
            return

        for call, result in zip(calls, results):
            if call.name != "web_search" or call.action != "search":
                continue
            query = str(call.arguments.get("query") or call.arguments.get("search_query") or "").strip()
            signature = self._web_search_query_signature(query)
            if not signature.get("terms"):
                continue
            cluster_key = self._find_matching_web_search_cluster(clusters, signature) or str(signature["key"])
            record = clusters.get(cluster_key) if isinstance(clusters.get(cluster_key), dict) else {}
            source_keys = self._web_search_result_source_keys(result)
            progress_seq = self._progress_seq(state)
            novelty = self._web_search_has_new_source_keys(record, source_keys)
            progress_since_last = progress_seq > int(record.get("last_progress_seq", 0) or 0)
            no_gain = result.status == "success" and not novelty and not progress_since_last

            search_count = int(record.get("search_count", 0) or 0) + 1
            no_gain_count = int(record.get("no_gain_count", 0) or 0) + 1 if no_gain else 0
            record.update(
                {
                    "cluster_key": cluster_key,
                    "terms": sorted(set(record.get("terms") or []) | set(signature.get("terms") or []))[:40],
                    "anchors": sorted(set(record.get("anchors") or []) | set(signature.get("anchors") or []))[:24],
                    "search_count": search_count,
                    "no_gain_count": no_gain_count,
                    "threshold": threshold,
                    "last_query": query,
                    "last_step_number": step_number,
                    "last_result_status": result.status,
                    "last_progress_seq": progress_seq,
                    "last_novelty": bool(novelty),
                    "last_progress_since_previous_search": bool(progress_since_last),
                    "blocked": bool(no_gain_count >= threshold and search_count >= threshold + 1),
                }
            )
            self._append_unique_limited(record, "queries", [query], limit=12)
            self._append_unique_limited(record, "seen_urls", source_keys.get("urls") or [], limit=160)
            self._append_unique_limited(record, "seen_domains", source_keys.get("domains") or [], limit=80)
            self._append_unique_limited(record, "seen_dates", source_keys.get("dates") or [], limit=80)
            self._append_unique_limited(record, "seen_source_keys", source_keys.get("source_keys") or [], limit=220)
            if no_gain:
                record["last_no_gain_reason"] = (
                    "same semantic query cluster returned no new URL/domain/date/source key, "
                    "and no reader, structured-tool, task_state, or batch_coverage progress occurred since the prior search."
                )
            clusters[cluster_key] = record

    def _bump_progress_signal(self, state: AgentState, reason: str, step_number: int) -> None:
        progress = state.context_stats.setdefault("progress_signals", {})
        if not isinstance(progress, dict):
            return
        seq = int(progress.get("seq", 0) or 0) + 1
        progress["seq"] = seq
        progress["last_reason"] = reason
        progress["last_step_number"] = step_number

    @staticmethod
    def _progress_seq(state: AgentState) -> int:
        progress = state.context_stats.get("progress_signals")
        if not isinstance(progress, dict):
            return 0
        return int(progress.get("seq", 0) or 0)

    @staticmethod
    def _task_state_progress_fingerprint(task_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(task_state, dict) or not task_state:
            return ""
        projected = {
            key: task_state.get(key)
            for key in (
                "contract",
                "evidence",
                "gaps",
                "answer_state",
                "answer_pins",
                "batch_coverage",
                "open_slots",
                "blocked_slots",
                "checks",
                "basis_notes",
                "stale_paths",
                "next_focus",
            )
            if task_state.get(key) not in (None, "", [], {})
        }
        if not projected:
            return ""
        encoded = json.dumps(projected, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8", errors="replace")).hexdigest()[:16]

    @staticmethod
    def _structured_tool_result_has_progress(call: ToolCall, result: ToolResult) -> bool:
        if call.name == "web_search":
            return False
        if call.name in {"market_data", "dataframe_query", "structured_table_reader", "sec_search", "sec_reader"}:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            execution_card = metadata.get("execution_card")
            if isinstance(execution_card, dict):
                try:
                    if int(execution_card.get("result_rows", 0) or 0) > 0:
                        return True
                except Exception:
                    pass
                if execution_card.get("can_answer_now") is True:
                    return True
            for key in ("result_count", "returned_rows", "matched_rows", "total_rows", "table_count", "asset_count", "state_control_count", "ajax_candidate_count"):
                try:
                    if int(metadata.get(key, 0) or 0) > 0:
                        return True
                except Exception:
                    continue
            for table in result.tables or []:
                if isinstance(table, dict):
                    rows = table.get("rows")
                    if isinstance(rows, list) and rows:
                        return True
            if metadata.get("document_key") not in (None, "", [], {}):
                return True
        return False

    def _blocked_tool_call_observation(self, state: AgentState, call: ToolCall) -> str:
        breakers = state.context_stats.get("tool_call_circuit_breakers")
        if not isinstance(breakers, dict):
            breakers = {}
        for key in self._tool_call_fuse_lookup_keys(call):
            entry = breakers.get(key)
            if not isinstance(entry, dict) or not entry.get("blocked"):
                continue
            skipped = int(entry.get("skipped", 0) or 0) + 1
            entry["skipped"] = skipped
            entry["last_skipped_tool"] = call.name
            entry["last_skipped_action"] = call.action
            display_name = self._tool_call_display_name(call)
            return (
                f"[policy.tool_call_fuse] skipped {display_name}: an equivalent call already returned "
                f"a non-retryable failure ({entry.get('reason') or 'non_retryable_failure'}; "
                f"failure_class={entry.get('failure_class') or 'unknown'}). "
                "Use different arguments/source, a resolver/catalog step, another tool, or answer with the source gap; "
                "do not repeat the same call."
            )
        return self._blocked_web_search_soft_fuse_observation(state, call)

    def _blocked_web_search_soft_fuse_observation(self, state: AgentState, call: ToolCall) -> str:
        if call.name != "web_search" or call.action != "search":
            return ""
        threshold = self._web_search_soft_fuse_threshold()
        if threshold <= 0:
            return ""
        fuse_state = state.context_stats.get("web_search_soft_fuse")
        if not isinstance(fuse_state, dict):
            return ""
        clusters = fuse_state.get("clusters")
        if not isinstance(clusters, dict):
            return ""
        query = str(call.arguments.get("query") or call.arguments.get("search_query") or "").strip()
        signature = self._web_search_query_signature(query)
        cluster_key = self._find_matching_web_search_cluster(clusters, signature)
        if not cluster_key:
            return ""
        record = clusters.get(cluster_key)
        if not isinstance(record, dict) or not record.get("blocked"):
            return ""
        skipped = int(record.get("skipped", 0) or 0) + 1
        record["skipped"] = skipped
        display_name = self._tool_call_display_name(call)
        return (
            f"[policy.web_search_soft_fuse] skipped {display_name}: this query is in a semantic search cluster "
            f"with {record.get('no_gain_count')} consecutive no-gain searches "
            f"({record.get('last_no_gain_reason') or 'no new source coverage or progress'}). "
            f"last_query={compact_text(str(record.get('last_query') or ''), 220)}; "
            f"seen_domains={', '.join((record.get('seen_domains') or [])[:8])}. "
            "Use a new source already discovered with web_reader, switch to resolver/catalog or a structured tool when keys are known, "
            "or answer with the remaining source gap. Only run web_search again with materially different entity/date/metric/source intent."
        )

    @staticmethod
    def _web_search_soft_fuse_threshold() -> int:
        raw = os.getenv("FIRE_AGENT_WEB_SEARCH_SOFT_FUSE_THRESHOLD", "2")
        try:
            return max(0, int(raw))
        except Exception:
            return 2

    def _find_matching_web_search_cluster(
        self,
        clusters: Dict[str, Any],
        signature: Dict[str, Any],
    ) -> str:
        terms = set(signature.get("terms") or [])
        anchors = set(signature.get("anchors") or [])
        if not terms:
            return ""
        exact_key = str(signature.get("key") or "")
        if exact_key and isinstance(clusters.get(exact_key), dict):
            return exact_key

        best_key = ""
        best_score = 0.0
        for key, record in clusters.items():
            if not isinstance(record, dict):
                continue
            record_terms = set(record.get("terms") or [])
            if not record_terms:
                continue
            union = terms | record_terms
            if not union:
                continue
            score = len(terms & record_terms) / max(1, len(union))
            record_anchors = set(record.get("anchors") or [])
            anchor_overlap = bool(anchors and record_anchors and (anchors & record_anchors))
            if score >= 0.82 or (anchor_overlap and score >= 0.62):
                if score > best_score:
                    best_score = score
                    best_key = str(key)
        return best_key

    @staticmethod
    def _web_search_query_signature(query: str) -> Dict[str, Any]:
        raw = str(query or "").strip()
        lowered = raw.lower()
        lowered = re.sub(r"https?://\S+", " ", lowered)
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9][a-z0-9._$%-]{1,}", lowered)
        stopwords = {
            "the",
            "and",
            "for",
            "from",
            "with",
            "that",
            "this",
            "what",
            "when",
            "where",
            "which",
            "official",
            "source",
            "sources",
            "website",
            "page",
            "web",
            "search",
            "google",
            "serper",
            "latest",
            "data",
        }
        terms = sorted({token.strip("._-%$") for token in tokens if token.strip("._-%$") and token not in stopwords})[:48]
        anchors = set()
        for match in re.findall(r"\b[A-Z]{1,8}(?:[.\-][A-Z0-9]{1,5})?\b", raw):
            if match.lower() not in stopwords:
                anchors.add(match.lower())
        for match in re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b", raw):
            anchors.add(match.lower())
        for token in terms:
            if any(ch.isdigit() for ch in token):
                anchors.add(token)
        key_material = " ".join(terms)
        digest = hashlib.sha1(key_material.encode("utf-8", errors="replace")).hexdigest()[:16] if key_material else ""
        return {"key": f"q:{digest}" if digest else "", "terms": terms, "anchors": sorted(anchors)[:24]}

    def _web_search_result_source_keys(self, result: ToolResult) -> Dict[str, List[str]]:
        urls: List[str] = []
        domains: List[str] = []
        dates: List[str] = []
        source_keys: List[str] = []
        try:
            parsed = json.loads(result.observation_text() or "[]")
        except Exception:
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        for item in parsed[:20]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("link") or "").strip()
            normalized_url = self._normalize_search_result_url(url)
            domain = self._normalize_domain(url)
            date = compact_text(str(item.get("date") or "").strip(), 80)
            title = self._normalize_source_title(str(item.get("title") or "").strip())
            if normalized_url:
                urls.append(normalized_url)
            if domain:
                domains.append(domain)
            if date:
                dates.append(date)
            source_key = normalized_url or "|".join(part for part in (domain, title, date) if part)
            if source_key:
                source_keys.append(compact_text(source_key, 500))
        return {
            "urls": self._dedupe_list(urls),
            "domains": self._dedupe_list(domains),
            "dates": self._dedupe_list(dates),
            "source_keys": self._dedupe_list(source_keys),
        }

    @staticmethod
    def _web_search_has_new_source_keys(record: Dict[str, Any], source_keys: Dict[str, List[str]]) -> bool:
        for key, existing_key in (
            ("urls", "seen_urls"),
            ("domains", "seen_domains"),
            ("dates", "seen_dates"),
            ("source_keys", "seen_source_keys"),
        ):
            existing = set(record.get(existing_key) or [])
            current = set(source_keys.get(key) or [])
            if current - existing:
                return True
        return False

    @staticmethod
    def _append_unique_limited(record: Dict[str, Any], key: str, values: List[str], *, limit: int) -> None:
        existing = list(record.get(key) or []) if isinstance(record.get(key), list) else []
        seen = set(existing)
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            existing.append(text)
            seen.add(text)
        if limit > 0 and len(existing) > limit:
            existing = existing[-limit:]
        record[key] = existing

    @staticmethod
    def _dedupe_list(values: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    @staticmethod
    def _normalize_search_result_url(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
        except Exception:
            return compact_text(url.strip().lower(), 500)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
        normalized = parsed._replace(scheme=parsed.scheme.lower() or "https", netloc=netloc, path=path, params="", query="", fragment="")
        return compact_text(normalized.geturl(), 500)

    @staticmethod
    def _normalize_domain(url: str) -> str:
        if not url:
            return ""
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return ""
        return domain[4:] if domain.startswith("www.") else domain

    @staticmethod
    def _normalize_source_title(title: str) -> str:
        lowered = re.sub(r"\s+", " ", str(title or "").strip().lower())
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff .:_-]+", "", lowered)
        return compact_text(lowered, 180)

    def _record_tool_failure_fuses(
        self,
        state: AgentState,
        calls: List[ToolCall],
        results: List[ToolResult],
        step_number: int,
    ) -> None:
        for call, result in zip(calls, results):
            entry = self._non_retryable_tool_failure_entry(call, result, step_number)
            if not entry:
                continue
            breakers = state.context_stats.setdefault("tool_call_circuit_breakers", {})
            if not isinstance(breakers, dict):
                continue
            key = str(entry.get("key") or "")
            if not key:
                continue
            previous = breakers.get(key) if isinstance(breakers.get(key), dict) else {}
            entry["count"] = int(previous.get("count", 0) or 0) + 1 if isinstance(previous, dict) else 1
            entry["blocked"] = True
            breakers[key] = entry

    def _non_retryable_tool_failure_entry(
        self,
        call: ToolCall,
        result: ToolResult,
        step_number: int,
    ) -> Optional[Dict[str, Any]]:
        if result.status == "success":
            return None
        error = str(result.error or "")
        metadata = result.metadata or {}
        failure_card = metadata.get("failure_card") if isinstance(metadata, dict) else None
        if isinstance(failure_card, dict) and failure_card.get("retry_same_plan_allowed") is False:
            failure_fingerprint = str(metadata.get("failure_fingerprint") or "").strip()
            key = f"{call.name}.{call.action}:non_retryable:{self._tool_call_fingerprint(call)}"
            return {
                "key": key,
                "reason": "tool_reported_non_retryable_failure",
                "failure_class": str(failure_card.get("failure_class") or metadata.get("failure_class") or ""),
                "failure_fingerprint": failure_fingerprint,
                "tool": call.name,
                "action": call.action,
                "first_step_number": step_number,
                "last_error": compact_text(error, 1000),
            }

        lowered = error.lower()
        if call.name == "web_reader" and self._is_non_retryable_reader_error(lowered):
            url = str(call.arguments.get("url") or "").strip()
            if url:
                return {
                    "key": f"{call.name}.{call.action}:blocked_url:{self._normalize_url_key(url)}",
                    "reason": "reader_access_blocked_same_url",
                    "failure_class": "blocked",
                    "tool": call.name,
                    "action": call.action,
                    "first_step_number": step_number,
                    "last_error": compact_text(error, 1000),
                }

        if self._is_non_retryable_validation_error(lowered):
            return {
                "key": f"{call.name}.{call.action}:validation:{self._tool_call_fingerprint(call)}",
                "reason": "deterministic_validation_or_unsupported_error",
                "failure_class": "unsupported_or_invalid_arguments",
                "tool": call.name,
                "action": call.action,
                "first_step_number": step_number,
                "last_error": compact_text(error, 1000),
            }
        return None

    def _tool_call_fuse_lookup_keys(self, call: ToolCall) -> List[str]:
        keys = [
            f"{call.name}.{call.action}:non_retryable:{self._tool_call_fingerprint(call)}",
            f"{call.name}.{call.action}:validation:{self._tool_call_fingerprint(call)}",
        ]
        url = str(call.arguments.get("url") or "").strip()
        if call.name == "web_reader" and url:
            keys.append(f"{call.name}.{call.action}:blocked_url:{self._normalize_url_key(url)}")
        return keys

    @staticmethod
    def _is_non_retryable_reader_error(lowered_error: str) -> bool:
        return any(
            marker in lowered_error
            for marker in (
                "http 403",
                "error code 1010",
                "cloudflare",
                "akamai",
                "challenge page",
                "access denied",
                "forbidden",
            )
        )

    @staticmethod
    def _is_non_retryable_validation_error(lowered_error: str) -> bool:
        retryable_markers = ("timeout", "timed out", "http 429", "too many requests", "rate limit", "remote disconnected")
        if any(marker in lowered_error for marker in retryable_markers):
            return False
        return any(
            marker in lowered_error
            for marker in (
                "unsupported action",
                "missing required argument",
                "invalid akshare function",
                "unsupported akshare function",
                "invalid json",
                "requires a json object plan",
                "must be a list",
                "must be an array",
                "not in yyyy-mm-dd format",
                "later than end_date",
                "retry_same_plan_allowed=false",
            )
        )

    @staticmethod
    def _normalize_url_key(url: str) -> str:
        try:
            parsed = urlparse(url)
        except Exception:
            return compact_text(url.strip(), 600)
        normalized = parsed._replace(fragment="").geturl()
        return compact_text(normalized.strip(), 600)

    @staticmethod
    def _tool_call_fingerprint(call: ToolCall) -> str:
        payload = {
            "name": call.name,
            "action": call.action,
            "arguments": call.arguments,
        }
        return compact_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str), 1200)

    @staticmethod
    def _policy_tool_result(call: ToolCall, observation: str, *, code: str) -> ToolResult:
        return ToolResult(
            call.name or "unknown_tool",
            "policy",
            "error",
            action=call.action or "default",
            observation=observation,
            confidence=0.0,
            error=observation,
            metadata={"policy_error": True, "policy_code": code},
        )

    def _validate_tool_calls(
        self,
        state: AgentState,
        proposed: List[Any],
        step_number: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[ToolCall], List[int], Dict[int, ToolResult]]:
        round_calls: List[ToolCall] = []
        executable_indexes: List[int] = []
        policy_results: Dict[int, ToolResult] = {}
        seen_call_ids: set[str] = set()
        available_native_names = (
            {self._tool_schema_name(tool) for tool in tools if self._tool_schema_name(tool)}
            if isinstance(tools, list)
            else set()
        )
        all_market_native_names = {
            self._tool_schema_name(tool)
            for tool in (state.tools or self._trace_tool_schemas())
            if self._tool_schema_name(tool).startswith("market_data_")
        }
        for item in proposed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool") or "").strip()
            if name == FINAL_ANSWER_TOOL_NAME:
                continue
            action = str(item.get("action", "")).strip() or "default"
            raw_arguments = item.get("arguments", item.get("args", {}))
            arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
            if name in self.tools:
                action = self.tools[name].normalize_action(action)
            raw_call_id = str(item.get("id") or "").strip()
            call_id = raw_call_id or f"call_{step_number:04d}_{len(round_calls) + 1:02d}"
            if call_id in seen_call_ids:
                call_id = f"call_{step_number:04d}_{len(round_calls) + 1:02d}"
            while call_id in seen_call_ids:
                call_id = f"{call_id}_dedup"
            seen_call_ids.add(call_id)
            call = ToolCall(
                name=name or "unknown_tool",
                action=action,
                arguments=arguments,
                id=call_id,
                rationale=str(item.get("rationale", "")),
                native_name=str(item.get("native_name") or ""),
                native_arguments=item.get("native_arguments") if isinstance(item.get("native_arguments"), dict) else {},
            )
            round_calls.append(call)
            call_index = len(round_calls) - 1
            if name not in self.tools:
                observation = (
                    f"[policy.unknown_tool] skipped unknown tool '{name or '<empty>'}'. "
                    "Use only tool names from the provided API tool schema."
                )
                policy_results[call_index] = self._policy_tool_result(
                    call,
                    observation,
                    code="policy.unknown_tool",
                )
                continue
            if available_native_names and call.native_name and call.native_name not in available_native_names:
                observation = (
                    f"[policy.tool_not_exposed] skipped {call.native_name}: this function was not "
                    "available in the current step tool schema."
                )
                if call.name == MARKET_DATA_TOOL_NAME:
                    hidden_recommendations = {call.native_name} if call.native_name in all_market_native_names else set()
                    hidden_recommendations.update(
                        self._market_data_tools_from_payload(call.action, all_market_native_names)
                    )
                    self._unlock_market_data_tool_names(
                        state,
                        hidden_recommendations,
                        step_number=step_number,
                        reason="model_requested_hidden_action",
                    )
                    observation += (
                        " Use only the market_data functions exposed in this step. If the needed typed action "
                        "is absent, call market_data_capabilities with a concise query or adapt on the next step "
                        "after tool diagnostics expose the relevant typed action."
                    )
                policy_results[call_index] = self._policy_tool_result(
                    call,
                    observation,
                    code="policy.tool_not_exposed",
                )
                continue
            if len(executable_indexes) >= state.config.max_tool_calls_per_round:
                display_name = self._tool_call_display_name(call)
                observation = (
                    f"[policy.max_tool_calls] skipped {display_name}: this step already reached "
                    f"the limit of {state.config.max_tool_calls_per_round} executable tool calls."
                )
                policy_results[call_index] = self._policy_tool_result(
                    call,
                    observation,
                    code="policy.max_tool_calls",
                )
                continue
            blocked_observation = self._blocked_tool_call_observation(state, call)
            if blocked_observation:
                policy_results[call_index] = self._policy_tool_result(
                    call,
                    blocked_observation,
                    code="policy.blocked_tool_call",
                )
                continue
            executable_indexes.append(call_index)
        return round_calls, executable_indexes, policy_results

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        raw = getattr(response, "usage", None)
        if not isinstance(raw, dict):
            return {}
        try:
            return {
                "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
                "total_tokens": int(raw.get("total_tokens", 0) or 0),
            }
        except (TypeError, ValueError):
            return {}

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def _tool_specs(self) -> Dict[str, Dict[str, Any]]:
        if hasattr(self.tools, "specs"):
            specs = self.tools.specs()
        else:
            specs = {name: tool.spec() for name, tool in self.tools.items()}
        return specs
