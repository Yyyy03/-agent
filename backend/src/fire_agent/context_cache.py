from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context import ContextAssembler
from .memory import ActionStep, AgentMemory
from .schemas import AgentConfig, AgentState


class ContextCompressionCacheManager:
    """Owns prompt-facing context-compression cache policy.

    FIREAgent drives the ReAct loop. This component owns the lower-level cache
    mechanics: raw-tail boundaries, cache signatures, deterministic delta
    overlays, and compression-failure fallback.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_assembler: ContextAssembler,
        *,
        task_state_fingerprint_fn: Callable[[Optional[Dict[str, Any]]], str],
    ) -> None:
        self.config = config
        self.context_assembler = context_assembler
        self._task_state_fingerprint = task_state_fingerprint_fn

    def signature(
        self,
        state: AgentState,
        *,
        render_mode: str,
        packet_mode: str,
        raw_tail_start_action_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        progress = state.context_stats.get("progress_signals")
        progress_seq = int(progress.get("seq", 0) or 0) if isinstance(progress, dict) else 0
        task_state_fingerprint = (
            str(progress.get("task_state_fingerprint") or "")
            if isinstance(progress, dict)
            else self._task_state_fingerprint(state.current_task_state)
        )
        return {
            "render_mode": str(render_mode or ""),
            "packet_mode": str(packet_mode or ""),
            "recent_action_steps": sorted(self.context_assembler._recent_action_numbers(state)),
            "raw_tail_start_action_index": max(0, int(raw_tail_start_action_index or 0)),
            "action_steps": len(getattr(state, "trajectory_action_steps", []) or []),
            "action_ledger": len(state.action_ledger),
            "evidence_ledger": len(state.evidence_ledger),
            "answer_pins": len(getattr(state, "answer_critical_pins", []) or []),
            "calc_ledger": len(state.calc_ledger),
            "artifact_store": len(state.artifact_store),
            "total_full_observation_chars": int(state.context_stats.get("total_full_observation_chars", 0) or 0),
            "total_prompt_observation_chars": int(state.context_stats.get("total_prompt_observation_chars", 0) or 0),
            "progress_seq": progress_seq,
            "task_state_fingerprint": task_state_fingerprint,
        }

    def choose_recent_raw_keep_steps(
        self,
        state: AgentState,
        memory: AgentMemory,
        *,
        packet_chars: int,
        budget_chars: Optional[int] = None,
    ) -> int:
        default_keep = max(0, int(state.config.context_recent_steps or 0))
        action_steps = self._action_steps(memory)
        if not action_steps:
            return 0
        configured_budget = int(budget_chars or 0)
        if configured_budget <= 0:
            configured_budget = int(state.context_stats.get("context_packet_tail_budget_chars", 0) or 0)
        budget = max(
            1,
            configured_budget
            or int(
                getattr(state.config, "context_full_compatible_char_limit", 0)
                or getattr(state.config, "context_hybrid_char_threshold", 0)
                or 128000
            ),
        )
        default_tail_chars = self.recent_raw_tail_chars(memory, default_keep)
        state.context_stats["last_recent_raw_default_steps"] = min(default_keep, len(action_steps))
        state.context_stats["last_recent_raw_default_chars"] = default_tail_chars
        state.context_stats["last_recent_raw_adaptive_budget"] = budget
        candidates = [min(default_keep, len(action_steps))]
        for keep in (3, 1, 0):
            keep = min(keep, len(action_steps))
            if keep not in candidates:
                candidates.append(keep)
        for keep in candidates:
            tail_chars = self.recent_raw_tail_chars(memory, keep)
            if int(packet_chars or 0) + tail_chars <= budget:
                if keep < candidates[0]:
                    state.context_stats["last_recent_raw_adaptive_reason"] = "packet_plus_recent_default_exceeds_budget"
                    state.context_stats["last_recent_raw_adaptive_selected_steps"] = keep
                    state.context_stats["last_recent_raw_adaptive_selected_chars"] = tail_chars
                else:
                    state.context_stats.pop("last_recent_raw_adaptive_reason", None)
                    state.context_stats.pop("last_recent_raw_adaptive_selected_steps", None)
                    state.context_stats.pop("last_recent_raw_adaptive_selected_chars", None)
                return keep
        state.context_stats["last_recent_raw_adaptive_reason"] = "packet_plus_recent_zero_exceeds_budget"
        state.context_stats["last_recent_raw_adaptive_selected_steps"] = 0
        state.context_stats["last_recent_raw_adaptive_selected_chars"] = 0
        return 0

    def reuse(
        self,
        state: AgentState,
        memory: AgentMemory,
        stats: Dict[str, Any],
        signature: Dict[str, Any],
        *,
        max_packet_chars: int,
        budget_chars: Optional[int] = None,
    ) -> Optional[str]:
        cached = self._load(state, stats, max_packet_chars=max_packet_chars, record_miss=True)
        if cached is None:
            return None
        cached_packet, cached_signature = cached

        plan = self._tail_plan(state, memory, cached_packet, cached_signature, budget_chars=budget_chars)
        cached_raw_tail_start = plan["cached_raw_tail_start"]
        target_raw_tail_start = plan["target_raw_tail_start"]
        if target_raw_tail_start != cached_raw_tail_start:
            miss_reasons = self._miss_reasons(signature, cached_signature)
            if target_raw_tail_start > cached_raw_tail_start and not miss_reasons:
                delta_packet = self._delta_packet(
                    state,
                    memory,
                    cached_packet,
                    cached_raw_tail_start=cached_raw_tail_start,
                    target_raw_tail_start=target_raw_tail_start,
                    max_packet_chars=max_packet_chars,
                )
                if delta_packet is not None:
                    combined_packet, overlay_chars = delta_packet
                    packet = self._accept(
                        state,
                        stats,
                        combined_packet,
                        source="cache_delta",
                        plan=plan,
                        raw_tail_start=target_raw_tail_start,
                        overlay_chars=overlay_chars,
                        count_cache_hit=True,
                    )
                    stats["last_cache_delta"] = self._delta_stats(
                        signature,
                        cached_signature,
                        cached_raw_tail_start=cached_raw_tail_start,
                        target_raw_tail_start=target_raw_tail_start,
                        keep_steps=plan["keep_steps"],
                        delta_overlay=True,
                    )
                    return packet

            stats["cache_misses"] = int(stats.get("cache_misses", 0) or 0) + 1
            reason = "raw_tail_boundary_changed"
            if miss_reasons:
                reason += ":" + ",".join(miss_reasons[:3])
            elif target_raw_tail_start > cached_raw_tail_start:
                reason += ":delta_overlay_unavailable"
            stats["last_cache_miss_reason"] = reason
            stats["last_cache_delta"] = self._delta_stats(
                signature,
                cached_signature,
                cached_raw_tail_start=cached_raw_tail_start,
                target_raw_tail_start=target_raw_tail_start,
                keep_steps=plan["keep_steps"],
            )
            return None

        miss_reasons = self._miss_reasons(signature, cached_signature)
        if miss_reasons:
            stats["cache_misses"] = int(stats.get("cache_misses", 0) or 0) + 1
            stats["last_cache_miss_reason"] = ",".join(miss_reasons[:4])
            stats["last_cache_delta"] = self._delta_stats(
                signature,
                cached_signature,
                cached_raw_tail_start=cached_raw_tail_start,
                target_raw_tail_start=target_raw_tail_start,
                keep_steps=plan["keep_steps"],
            )
            return None

        packet = self._accept(
            state,
            stats,
            cached_packet,
            source="cache",
            plan=plan,
            raw_tail_start=cached_raw_tail_start,
            count_cache_hit=True,
        )
        stats["last_cache_delta"] = self._delta_stats(
            signature,
            cached_signature,
            cached_raw_tail_start=cached_raw_tail_start,
            target_raw_tail_start=cached_raw_tail_start,
            keep_steps=plan["keep_steps"],
        )
        return packet

    @staticmethod
    def store(
        state: AgentState,
        packet: str,
        signature: Dict[str, Any],
        *,
        max_packet_chars: int,
    ) -> None:
        if not isinstance(packet, str) or not packet.strip() or len(packet) > max_packet_chars:
            return
        setattr(
            state,
            "_context_compression_cache",
            {
                "packet": packet,
                "signature": dict(signature),
                "stored_at": time.time(),
            },
        )

    def fallback(
        self,
        state: AgentState,
        memory: AgentMemory,
        stats: Dict[str, Any],
        *,
        max_packet_chars: int,
        budget_chars: Optional[int] = None,
    ) -> Optional[str]:
        cached = self._load(state, stats, max_packet_chars=max_packet_chars, record_miss=False)
        if cached is None:
            return None
        cached_packet, cached_signature = cached

        plan = self._tail_plan(state, memory, cached_packet, cached_signature, budget_chars=budget_chars)
        current_action_steps = plan["current_action_steps"]
        cached_raw_tail_start = plan["cached_raw_tail_start"]
        target_raw_tail_start = plan["target_raw_tail_start"]

        delta_packet = self._delta_packet(
            state,
            memory,
            cached_packet,
            cached_raw_tail_start=cached_raw_tail_start,
            target_raw_tail_start=target_raw_tail_start,
            max_packet_chars=max_packet_chars,
            allow_empty_same_boundary=True,
        )
        if delta_packet is not None:
            combined_packet, overlay_chars = delta_packet
            return self._accept(
                state,
                stats,
                combined_packet,
                source="cache_delta_fallback" if overlay_chars else "cache_fallback",
                plan=plan,
                raw_tail_start=target_raw_tail_start,
                overlay_chars=overlay_chars,
                fallback_source="cache_delta_after_compression_failure",
            )

        raw_tail_steps = max(0, current_action_steps - cached_raw_tail_start)
        fallback_tail_chars = self.recent_raw_tail_chars(memory, raw_tail_steps)
        configured_budget = int(budget_chars or 0)
        budget = max(
            1,
            configured_budget
            or int(
                getattr(state.config, "context_full_compatible_char_limit", 0)
                or getattr(state.config, "context_hybrid_char_threshold", 0)
                or 128000
            ),
        )
        if len(cached_packet) + fallback_tail_chars > budget:
            stats["fallback_cache_rejected_reason"] = "cached_packet_plus_raw_tail_exceeds_budget"
            return None
        return self._accept(
            state,
            stats,
            cached_packet,
            source="cache_raw_tail_fallback",
            plan=plan,
            raw_tail_start=cached_raw_tail_start,
            fallback_source="cache_raw_tail_after_compression_failure",
        )

    @staticmethod
    def raw_tail_start(cached_signature: Dict[str, Any]) -> int:
        value = (
            cached_signature.get("raw_tail_start_action_index")
            if cached_signature.get("raw_tail_start_action_index") not in (None, "")
            else cached_signature.get("action_steps", 0)
        )
        try:
            return max(0, int(value or 0))
        except Exception:
            return 0

    @staticmethod
    def _action_steps(memory: AgentMemory) -> List[ActionStep]:
        action_steps = getattr(memory, "action_steps", None)
        if isinstance(action_steps, list):
            return action_steps
        return [step for step in memory.steps if isinstance(step, ActionStep)]

    def recent_raw_tail_chars(self, memory: AgentMemory, keep_steps: int) -> int:
        keep = max(0, int(keep_steps or 0))
        if keep <= 0:
            return 0
        total = 0
        for step in self._action_steps(memory)[-keep:]:
            for message in step.to_messages():
                total += len(str(message.get("content") or ""))
        return total

    @staticmethod
    def _delta(current: Dict[str, Any], previous: Dict[str, Any], key: str) -> int:
        try:
            return int(current.get(key, 0) or 0) - int(previous.get(key, 0) or 0)
        except Exception:
            return 0

    def _miss_reasons(self, signature: Dict[str, Any], cached_signature: Dict[str, Any]) -> List[str]:
        if str(signature.get("packet_mode") or "") != str(cached_signature.get("packet_mode") or ""):
            return ["packet_mode_changed"]
        if self._delta(signature, cached_signature, "action_steps") < 0:
            return ["action_history_rewound"]

        reasons: List[str] = []
        action_delta = self._delta(signature, cached_signature, "action_steps")
        full_obs_delta = self._delta(signature, cached_signature, "total_full_observation_chars")
        prompt_obs_delta = self._delta(signature, cached_signature, "total_prompt_observation_chars")
        action_ledger_delta = self._delta(signature, cached_signature, "action_ledger")
        evidence_delta = self._delta(signature, cached_signature, "evidence_ledger")
        calc_delta = self._delta(signature, cached_signature, "calc_ledger")
        artifact_delta = self._delta(signature, cached_signature, "artifact_store")
        progress_delta = self._delta(signature, cached_signature, "progress_seq")

        if action_delta > 6:
            reasons.append("action_delta_gt_6")
        if full_obs_delta > 96000 or prompt_obs_delta > 48000:
            reasons.append("observation_delta_gt_threshold")
        if action_ledger_delta > 24:
            reasons.append("action_ledger_delta_gt_24")
        if evidence_delta > 16:
            reasons.append("evidence_delta_gt_16")
        if calc_delta > 10:
            reasons.append("calc_delta_gt_10")
        if artifact_delta > 2:
            reasons.append("artifact_delta_gt_2")
        if progress_delta > 6:
            reasons.append("coverage_progress_delta_gt_6")
        return reasons

    def _load(
        self,
        state: AgentState,
        stats: Dict[str, Any],
        *,
        max_packet_chars: int,
        record_miss: bool,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        def miss(reason: str) -> None:
            if not record_miss:
                return
            stats["cache_misses"] = int(stats.get("cache_misses", 0) or 0) + 1
            stats["last_cache_miss_reason"] = reason

        cache = getattr(state, "_context_compression_cache", None)
        if not isinstance(cache, dict):
            miss("empty_cache")
            return None
        cached_packet = cache.get("packet")
        cached_signature = cache.get("signature")
        if not isinstance(cached_packet, str) or not cached_packet.strip() or not isinstance(cached_signature, dict):
            miss("invalid_cache")
            return None
        if len(cached_packet) > max_packet_chars:
            miss("cached_packet_oversize")
            return None
        return cached_packet, cached_signature

    def _tail_plan(
        self,
        state: AgentState,
        memory: AgentMemory,
        cached_packet: str,
        cached_signature: Dict[str, Any],
        *,
        budget_chars: Optional[int] = None,
    ) -> Dict[str, int]:
        action_steps = self._action_steps(memory)
        current_action_steps = len(action_steps)
        cached_raw_tail_start = self.raw_tail_start(cached_signature)
        keep_steps = self.choose_recent_raw_keep_steps(
            state,
            memory,
            packet_chars=len(cached_packet),
            budget_chars=budget_chars,
        )
        return {
            "current_action_steps": current_action_steps,
            "cached_raw_tail_start": cached_raw_tail_start,
            "keep_steps": keep_steps,
            "target_raw_tail_start": max(0, current_action_steps - keep_steps),
        }

    @staticmethod
    def _apply_tail_window(
        state: AgentState,
        *,
        current_action_steps: int,
        raw_tail_start: int,
    ) -> None:
        recent_steps = max(0, current_action_steps - raw_tail_start)
        setattr(state, "_context_compression_raw_tail_start_action_index", raw_tail_start)
        setattr(state, "_context_recent_steps_override", recent_steps)
        state.context_stats["last_recent_raw_effective_steps"] = recent_steps
        state.context_stats["last_recent_raw_target_start_action_index"] = raw_tail_start

    def _delta_stats(
        self,
        signature: Dict[str, Any],
        cached_signature: Dict[str, Any],
        *,
        cached_raw_tail_start: int,
        target_raw_tail_start: int,
        keep_steps: int,
        delta_overlay: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action_steps": self._delta(signature, cached_signature, "action_steps"),
            "observation_chars": self._delta(signature, cached_signature, "total_full_observation_chars"),
            "progress_seq": self._delta(signature, cached_signature, "progress_seq"),
            "raw_tail_start_action_index": cached_raw_tail_start,
            "target_raw_tail_start_action_index": target_raw_tail_start,
            "target_recent_raw_steps": keep_steps,
        }
        if delta_overlay:
            payload["delta_overlay"] = True
        return payload

    def _delta_packet(
        self,
        state: AgentState,
        memory: AgentMemory,
        cached_packet: str,
        *,
        cached_raw_tail_start: int,
        target_raw_tail_start: int,
        max_packet_chars: int,
        allow_empty_same_boundary: bool = False,
    ) -> Optional[Tuple[str, int]]:
        if target_raw_tail_start < cached_raw_tail_start:
            return None
        if target_raw_tail_start == cached_raw_tail_start:
            return (cached_packet, 0) if allow_empty_same_boundary else None

        overlay_limit = max_packet_chars - len(cached_packet) - 64
        if overlay_limit < 1200:
            return None
        overlay = self.context_assembler.build_rolling_compression_delta_packet(
            state,
            memory,
            start_action_index=cached_raw_tail_start,
            end_action_index=target_raw_tail_start,
            char_limit=min(12000, overlay_limit),
        )
        if not overlay:
            return None
        combined_packet = cached_packet + "\n" + overlay
        if len(combined_packet) > max_packet_chars:
            return None
        return combined_packet, len(overlay)

    def _accept(
        self,
        state: AgentState,
        stats: Dict[str, Any],
        packet: str,
        *,
        source: str,
        plan: Dict[str, int],
        raw_tail_start: int,
        overlay_chars: int = 0,
        count_cache_hit: bool = False,
        fallback_source: str = "",
    ) -> str:
        if count_cache_hit:
            stats["cache_hits"] = int(stats.get("cache_hits", 0) or 0) + 1
            stats["skipped_calls_by_cache"] = int(stats.get("skipped_calls_by_cache", 0) or 0) + 1
        stats["last_packet_chars"] = len(packet)
        stats["last_packet_source"] = source
        if overlay_chars:
            stats["last_delta_overlay_chars"] = overlay_chars
        if fallback_source:
            stats["fallback_packet_source"] = fallback_source
        self._apply_tail_window(
            state,
            current_action_steps=plan["current_action_steps"],
            raw_tail_start=raw_tail_start,
        )
        return packet
