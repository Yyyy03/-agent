from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_LENGTH_TRUNCATE_CONTENT = 20000


def compact_text(text: str, limit: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    """Head+tail truncation aligned with Flash-Searcher's ``truncate_content``.

    Keeps the first and last halves of the content and drops the middle so that
    both the leading context and the trailing figures/summary survive, instead
    of a head-only cut.
    """

    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n..._This content has been truncated to stay below {limit} characters_...\n"
        + text[-half:]
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def keep_tool_results() -> bool:
    """Whether action steps should also dump structured tool_results to the
    on-disk trajectory. Default off (only the formatted ``obs`` string is
    kept, matching Flash-Searcher's lean schema). Set
    ``FIRE_AGENT_KEEP_TOOL_RESULTS=1`` to opt into debug traces."""

    return _env_bool("FIRE_AGENT_KEEP_TOOL_RESULTS", False)


def keep_reasoning_content() -> bool:
    """Whether to keep model-native reasoning traces on trajectory rows.

    DeepSeek thinking mode returns ``reasoning_content`` separately from
    ``content``. Default on because this project uses DeepSeek as a teacher for
    SFT; set ``FIRE_AGENT_KEEP_REASONING_CONTENT=0`` for lean benchmark runs.
    """

    return _env_bool("FIRE_AGENT_KEEP_REASONING_CONTENT", True)


def keep_step_task_state() -> bool:
    """Whether to dump the current TaskState snapshot into every trajectory
    action row. Default off because online reasoning reads
    ``AgentState.current_task_state`` directly; repeated per-step copies only
    inflate JSONL traces and serialization cost. Set
    ``FIRE_AGENT_KEEP_STEP_TASK_STATE=1`` for detailed debugging."""

    return _env_bool("FIRE_AGENT_KEEP_STEP_TASK_STATE", False)


def keep_full_context_stats() -> bool:
    """Whether to dump the full runtime context diagnostics tree into
    ``stats.context``. Default off keeps the official/evaluation row lean while
    retaining the high-signal counters used for diagnostics."""

    return _env_bool("FIRE_AGENT_KEEP_FULL_CONTEXT_STATS", False)


def _compact_mapping(value: Any, *, string_limit: int = 1200, list_limit: int = 12) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_mapping(item, string_limit=string_limit, list_limit=list_limit)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            _compact_mapping(item, string_limit=string_limit, list_limit=list_limit)
            for item in value[:list_limit]
        ]
    if isinstance(value, str):
        return compact_text(value, string_limit)
    return value


def compact_context_stats(context_stats: Dict[str, Any]) -> Dict[str, Any]:
    if keep_full_context_stats():
        return dict(context_stats)

    scalar_keys = {
        "runtime_stage",
        "finish_type",
        "natural_final",
        "terminal_fallback",
        "max_steps_reached",
        "last_action_step_number",
        "task_state_render_mode",
        "last_render_mode",
        "last_packet_mode",
        "last_legacy_prompt_chars",
        "last_packet_prompt_chars",
        "last_recent_raw_turns",
        "last_recent_raw_turn_chars",
        "last_recent_raw_default_steps",
        "last_recent_raw_default_chars",
        "last_recent_raw_effective_steps",
        "last_recent_raw_adaptive_budget",
        "last_recent_raw_adaptive_reason",
        "last_recent_raw_target_start_action_index",
        "last_compression_raw_tail_steps",
        "last_compression_raw_tail_chars",
        "last_compression_raw_tail_start_action_index",
        "context_hybrid_char_threshold",
        "context_full_compatible_char_limit",
        "tool_schema_chars",
        "context_request_reserve_chars",
        "effective_context_hybrid_char_threshold",
        "effective_context_full_compatible_char_limit",
        "artifact_count",
        "artifact_chars",
        "max_artifact_chars",
        "total_full_observation_chars",
        "total_prompt_observation_chars",
        "max_full_observation_chars",
        "last_total_observation_chars",
        "last_max_observation_chars",
        "last_visible_answer_critical_pins",
        "last_aggregated_answer_critical_pin_groups",
        "answer_critical_pin_count",
        "task_state_update_count",
        "task_state_empty_update_count",
        "task_state_normalized_update_count",
        "task_state_runtime_scaffold_count",
        "task_state_snapshot_issue_count",
        "task_state_last_snapshot_issues",
        "task_state_last_normalized_keys",
        "task_state_last_schema_keys",
        "packet_deferred_reason",
        "packet_forced_compact_reason",
    }
    nested_keys = {
        "render_mode_counts",
        "terminal_summary",
        "protocol_retry",
        "model_error_streak",
        "model_error_circuit_breaker",
        "runtime_error",
        "llm_context_compression",
        "task_state_support_audit",
        "packet_final_soft_audit",
    }
    compacted: Dict[str, Any] = {}
    for key in scalar_keys:
        value = context_stats.get(key)
        if value not in (None, "", [], {}):
            compacted[key] = _compact_mapping(value, string_limit=1200, list_limit=8)
    for key in nested_keys:
        value = context_stats.get(key)
        if value not in (None, "", [], {}):
            compacted[key] = _compact_mapping(value, string_limit=1600, list_limit=12)

    progress = context_stats.get("progress_signals")
    if isinstance(progress, dict):
        compacted["progress_signals"] = {
            key: _compact_mapping(progress.get(key), string_limit=800, list_limit=6)
            for key in ("seq", "last_reason", "last_step_number", "task_state_fingerprint")
            if progress.get(key) not in (None, "", [], {})
        }

    web_fuse = context_stats.get("web_search_soft_fuse")
    if isinstance(web_fuse, dict):
        clusters = web_fuse.get("clusters")
        cluster_summary = []
        if isinstance(clusters, dict):
            sorted_clusters = sorted(
                (item for item in clusters.values() if isinstance(item, dict)),
                key=lambda item: (int(item.get("blocked", False)), int(item.get("no_gain_count", 0) or 0), int(item.get("search_count", 0) or 0)),
                reverse=True,
            )
            for item in sorted_clusters[:8]:
                cluster_summary.append(
                    {
                        "cluster_key": item.get("cluster_key"),
                        "search_count": item.get("search_count"),
                        "no_gain_count": item.get("no_gain_count"),
                        "blocked": item.get("blocked"),
                        "skipped": item.get("skipped"),
                        "last_step_number": item.get("last_step_number"),
                        "last_result_status": item.get("last_result_status"),
                        "last_query": compact_text(str(item.get("last_query") or ""), 300),
                        "seen_domains": list(item.get("seen_domains") or [])[:8],
                    }
                )
        if cluster_summary:
            compacted["web_search_soft_fuse"] = {"cluster_summary": cluster_summary}

    breakers = context_stats.get("tool_call_circuit_breakers")
    if isinstance(breakers, dict):
        compacted["tool_call_circuit_breakers"] = {
            key: _compact_mapping(value, string_limit=800, list_limit=8)
            for key, value in list(breakers.items())[:24]
            if isinstance(value, dict)
        }

    readiness = context_stats.get("packet_semantic_readiness")
    if isinstance(readiness, dict):
        compacted["packet_semantic_readiness"] = {
            key: _compact_mapping(readiness.get(key), string_limit=1000, list_limit=8)
            for key in (
                "ready",
                "artifact_sensitive_task",
                "older_tool_calls",
                "answer_covered_old_tool_calls",
                "recoverable_index_old_tool_calls",
                "critical_recoverable_index_old_tool_calls",
                "thin_summary_covered_old_tool_calls",
                "blocking_gap_old_tool_calls",
                "critical_recovery_indexes",
                "blocking_gaps",
                "missing_action_steps",
                "uncovered_steps",
                "reason",
            )
            if readiness.get(key) not in (None, "", [], {})
        }

    return compacted


@dataclass
class BenchTaskContext:
    """Runner-provided task context.

    Lightweight container shared between runners and the runtime. Each
    benchmark fills:

    - ``task_prompt`` — the formatted user prompt built from the dataset
      question and optional seed URL.
    - ``attachments`` / ``urls`` — runtime injects them into per-step
      ``Available resources``.
    - ``golden_answer`` — passed through to the evaluator only; never sent to
      the agent LLM.
    - ``bench_extra`` — opaque per-bench metadata required by the bench-
      specific evaluator / aggregator (rubric, judge templates, official
      response fields, …). This replaces the legacy
      ``eval_metadata`` / ``solver_metadata`` / ``evaluator_metadata`` trio.
    """

    bench_name: str
    task_index: int
    task_prompt: str
    raw_question: str = ""
    raw_record: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    golden_answer: Optional[Any] = None
    bench_extra: Dict[str, Any] = field(default_factory=dict)
    # The following two are kept only so the runtime can stamp them on
    # per-step LLM-side debug logs; they never reach the on-disk row schema.
    task_family: str = "qa"
    evaluator_name: str = "generic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfig:
    max_steps: int = 40
    max_tool_calls_per_round: int = 3
    model_name: Optional[str] = None
    vision_model_name: Optional[str] = None
    allow_paid_tools: bool = True
    prefer_free_official_sources: bool = True
    tool_timeout_seconds: int = 90
    use_model_for_roles: bool = True
    final_answer_only_round: bool = True
    # Run one tool-free native-thinking pass before the protocol retry asks
    # for the final_answer tool. This keeps fast answers machine-parseable
    # while exposing provider-native reasoning to streaming clients.
    single_turn_reasoning_pass: bool = False
    benchmark_system_prompt: str = ""
    benchmark_planning_prompt: str = ""
    tool_prompt_constraints: Dict[str, List[str]] = field(default_factory=dict)
    # Context rendering. ``auto`` keeps short trajectories full-compatible and
    # switches longer trajectories to lean/compact WorkingContextPacket views.
    # ``full_compatible`` and ``legacy`` are retained for A/B and fallback runs.
    context_mode: str = "auto"
    task_state_render_mode: str = "view"
    context_full_compatible_char_limit: int = 128000
    context_hybrid_char_threshold: int = 96000
    context_artifact_char_threshold: int = 96000
    context_artifact_preview_chars: int = 20000
    # In packet mode, keep the newest action steps after WorkingContextPacket as
    # normal assistant/user ReAct turns. Old history remains in packet ledgers.
    context_recent_steps: int = 5
    context_recent_observation_chars: int = 30000
    context_recent_observation_step_chars: int = 6000
    context_recent_tool_call_excerpt_chars: int = 1200
    context_action_ledger_limit: int = 1000
    context_duplicate_warning_limit: int = 12
    context_pinned_evidence_limit: int = 48
    context_pinned_evidence_chars: int = 24000
    context_answer_pin_limit: int = 64
    context_answer_pin_chars: int = 24000
    context_pinned_raw_slice_limit: int = 8
    context_pinned_raw_slice_chars: int = 24000
    context_pinned_raw_slice_step_chars: int = 3000
    context_artifact_slice_limit: int = 4
    context_artifact_slice_chars: int = 12000
    context_artifact_slice_step_chars: int = 3000
    context_calc_ledger_limit: int = 80
    context_artifact_ref_limit: int = 40
    # Tool schemas are passed through ChatCompletions `tools`, not as message
    # text, but they still consume prompt/context budget after provider or
    # tokenizer rendering. Reserve room for tool schemas and generation
    # overhead when deciding whether full history can stay in-context.
    context_request_reserve_chars: int = 8000
    # Optional LLM context compression. This does not write runtime memory; it
    # rewrites the prompt-facing WorkingContextPacket at render time. The
    # deterministic packet remains the fallback and is supplied as a draft.
    context_compression_mode: str = "llm"
    context_compression_retries: int = 1
    context_compression_max_source_chars: int = 96000
    context_compression_max_packet_chars: int = 32000
    context_compression_old_observation_step_chars: int = 3000
    context_compression_recent_observation_step_chars: int = 6000
    context_compression_include_draft_packet: bool = True


@dataclass
class ToolCall:
    name: str
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    rationale: str = ""
    native_name: str = ""
    native_arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "action": self.action,
            "arguments": self.arguments,
            "id": self.id,
            "rationale": self.rationale,
        }
        if self.native_name:
            data["native_name"] = self.native_name
        if self.native_arguments:
            data["native_arguments"] = self.native_arguments
        return data

    def to_lean_dict(self) -> Dict[str, Any]:
        """Trajectory-facing projection (drops id / rationale)."""
        return {"name": self.name, "action": self.action, "arguments": self.arguments}


@dataclass
class ToolResult:
    tool_family: str
    provider: str
    status: str
    action: str = ""
    observation: Any = ""
    tables: List[Any] = field(default_factory=list)
    confidence: float = 0.5
    error: Optional[str] = None
    paid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def observation_text(self) -> str:
        if isinstance(self.observation, str):
            return self.observation.strip()
        if self.observation in (None, ""):
            return ""
        try:
            return json.dumps(self.observation, ensure_ascii=False, default=str)
        except Exception:
            return str(self.observation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    """Flash-Searcher-aligned slim trajectory step.

    Two shapes:

    - ``name="plan"``: ``{"name": "plan", "value": <plan_text>, "think": ""}``
    - ``name="action"``: ``{"name": "action", "tool_calls": [...], "obs": <str>, "think": <str>}``

    With optional ``"error"`` and (only when ``FIRE_AGENT_KEEP_TOOL_RESULTS=1``)
    ``"tool_results"`` debug dump.
    """

    name: str
    value: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    obs: str = ""
    think: str = ""
    error: Optional[str] = None
    raw_tool_results: Optional[List[Dict[str, Any]]] = None
    task_state: Optional[Dict[str, Any]] = None
    reasoning_content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        if self.name == "plan":
            data: Dict[str, Any] = {"name": "plan", "value": self.value, "think": self.think or ""}
        else:
            data = {
                "name": self.name,
                "tool_calls": self.tool_calls,
                "obs": self.obs,
                "think": self.think or "",
            }
        if self.task_state is not None:
            data["task_state"] = self.task_state
        if self.reasoning_content and keep_reasoning_content():
            data["reasoning_content"] = self.reasoning_content
        if self.error:
            data["error"] = self.error
        if self.raw_tool_results is not None:
            data["tool_results"] = self.raw_tool_results
        return data


@dataclass
class AgentTurn:
    """Training-facing agent transition.

    A turn is the unit used by SFT/RL exporters: the exact prompt messages for
    one policy call, the assistant message produced by the model, and the tool
    observations created after executing that assistant action.
    """

    type: str
    input_messages: List[Dict[str, Any]]
    output_message: Dict[str, Any]
    id: str = ""
    observations: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    include_in_sft: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "input_messages": self.input_messages,
            "output_message": self.output_message,
            "observations": self.observations,
            "tools": self.tools,
            "include_in_sft": self.include_in_sft,
        }


@dataclass
class AuxiliaryTurn:
    """Training-facing transition for internal LLM skills.

    These calls are not part of the main agent policy chronology, but they use
    the same prompt/target shape so context-compression and content-reader
    skills can be SFT-trained with the same dataset builder.
    """

    id: str
    type: str
    module: str
    input_messages: List[Dict[str, Any]]
    output_message: Dict[str, Any]
    observations: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    target_type: str = ""
    render_mode: str = ""
    step_number: Optional[int] = None
    include_in_sft: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "module": self.module,
            "target_type": self.target_type,
            "step_number": self.step_number,
            "render_mode": self.render_mode,
            "input_messages": self.input_messages,
            "output_message": self.output_message,
            "observations": self.observations,
            "tools": self.tools,
            "source": self.source,
            "include_in_sft": self.include_in_sft,
        }


def _zero_token_stats() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }


@dataclass
class AgentState:
    task_context: BenchTaskContext
    config: AgentConfig
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    turns: List[AgentTurn] = field(default_factory=list)
    auxiliary_turns: List[AuxiliaryTurn] = field(default_factory=list)
    generation: Dict[str, Any] = field(default_factory=dict)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    token_stats: Dict[str, Any] = field(default_factory=_zero_token_stats)
    # Model-owned persistent working state (the "TaskState" memory). The model
    # writes/updates it every action step; the runtime pins the latest copy at
    # the top of each step instruction so facts/gaps/checks are never lost as
    # raw history accrues. The harness only carries it forward; it does
    # not judge or fill it (that reasoning belongs to the model / training).
    current_task_state: Optional[Dict[str, Any]] = None
    # Runtime-owned shadow memory. These ledgers are append-only and are used by
    # the context assembler once full linear history becomes too large.
    action_ledger: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ledger: List[Dict[str, Any]] = field(default_factory=list)
    answer_critical_pins: List[Dict[str, Any]] = field(default_factory=list)
    answer_pin_index: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    evidence_fingerprint_index: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    evidence_ref_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    evidence_schema_items: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    calc_ledger: List[Dict[str, Any]] = field(default_factory=list)
    artifact_store: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trajectory_action_steps: List[Tuple[int, TrajectoryStep]] = field(default_factory=list, repr=False)
    tool_call_count: int = 0
    tools_by_name: Dict[str, int] = field(default_factory=dict)
    action_duplicate_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    action_artifact_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    context_stats: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: AgentTurn) -> None:
        self.turns.append(turn)

    def add_auxiliary_turn(self, turn: AuxiliaryTurn) -> None:
        self.auxiliary_turns.append(turn)

    def add_step(self, step: TrajectoryStep) -> None:
        self.trajectory.append(step)
        if step.name != "action":
            return
        self.trajectory_action_steps.append((len(self.trajectory_action_steps) + 1, step))
        for call in step.tool_calls or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            if name in {"", "final_answer"}:
                continue
            self.tool_call_count += 1
            self.tools_by_name[name] = int(self.tools_by_name.get(name, 0) or 0) + 1

    def record_usage(self, role: str, usage: Dict[str, int]) -> None:
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or (prompt + completion))
        self.token_stats["prompt_tokens"] += prompt
        self.token_stats["completion_tokens"] += completion
        self.token_stats["total_tokens"] += total
        self.token_stats["llm_calls"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_context": self.task_context.to_dict(),
            "config": asdict(self.config),
            "generation": self.generation,
            "tools": self.tools,
            "turns": [turn.to_dict() for turn in self.turns],
            "auxiliary_turns": [turn.to_dict() for turn in self.auxiliary_turns],
            "trajectory": [step.to_dict() for step in self.trajectory],
            "token_stats": self.token_stats,
            "current_task_state": self.current_task_state,
            "answer_critical_pins": self.answer_critical_pins,
            "action_ledger": self.action_ledger,
            "evidence_ledger": self.evidence_ledger,
            "calc_ledger": self.calc_ledger,
            "artifact_refs": [
                {key: value for key, value in record.items() if key != "content"}
                for record in self.artifact_store.values()
            ],
            "context_stats": self.context_stats,
        }


@dataclass
class AgentResult:
    agent_result: Any
    status: str
    state: AgentState
    error: Optional[str] = None

    def to_task_log(self) -> Dict[str, Any]:
        """Project runtime state into the lean Flash-Searcher-style log.

        Top-level keys:

        - ``bench_name`` / ``task_index`` / ``question`` / ``golden_answer``
        - ``agent_result`` / ``agent_trajectory``
        - ``status`` / ``error``
        - ``stats`` (steps + tool_calls + tools_by_name + tokens)

        Plus passthrough of ``bench_extra`` from the BenchTaskContext so each
        benchmark runner can let its evaluator read whatever side-channel
        data it needs (rubric / judge templates / official response fields).
        Nothing else is exposed; ``output_profile`` / ``prompt_profile`` /
        ``evaluator_metadata`` / ``solver_metadata`` / ``benchmark_scoring_input``
        / ``evidence_board`` / ``agent_output`` have all been dropped.
        """

        context = self.state.task_context
        tools_by_name: Dict[str, int] = dict(self.state.tools_by_name)
        tool_call_count = int(self.state.tool_call_count or 0)

        stats = {
            "steps": len(self.state.trajectory),
            "tool_calls": tool_call_count,
            "tools_by_name": tools_by_name,
            "tokens": {
                "prompt": int(self.state.token_stats.get("prompt_tokens", 0) or 0),
                "completion": int(self.state.token_stats.get("completion_tokens", 0) or 0),
                "total": int(self.state.token_stats.get("total_tokens", 0) or 0),
                "llm_calls": int(self.state.token_stats.get("llm_calls", 0) or 0),
            },
        }
        if self.state.context_stats:
            stats["context"] = compact_context_stats(self.state.context_stats)

        log: Dict[str, Any] = {
            "episode_id": "/".join(
                [
                    str(context.bench_name or "unknown"),
                    str(context.task_index if context.task_index is not None else "unknown"),
                ]
            ),
            "bench_name": context.bench_name,
            "task_index": context.task_index,
            "question": context.task_prompt,
            "task": {"question": context.task_prompt},
            "generation": dict(self.state.generation or {}),
            "tools": list(self.state.tools or []),
            "turns": [turn.to_dict() for turn in self.state.turns],
            "auxiliary_turns": [turn.to_dict() for turn in self.state.auxiliary_turns],
            "golden_answer": context.golden_answer,
            "agent_result": self.agent_result,
            "agent_trajectory": [step.to_dict() for step in self.state.trajectory],
            "bench_extra": dict(context.bench_extra or {}),
            "status": self.status,
            "stats": stats,
        }
        if self.error:
            log["error"] = self.error
        return log
