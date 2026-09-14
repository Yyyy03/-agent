from __future__ import annotations

import copy
import logging
import os
from typing import Any

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample


__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _episode_rows(sample: Sample) -> list[dict[str, Any]]:
    rows = sample.prompt
    if not isinstance(rows, list):
        raise ValueError(f"episode rollout expects sample.prompt to be a list of rows, got {type(rows).__name__}")
    if not rows:
        raise ValueError("episode rollout received an empty episode")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"episode row {index} must be a JSON object, got {type(row).__name__}")
        if not isinstance(row.get("messages"), list):
            raise ValueError(f"episode row {index} is missing list field 'messages'")
    return rows


def _episode_id(episode_sample: Sample, rows: list[dict[str, Any]]) -> str:
    metadata = episode_sample.metadata if isinstance(episode_sample.metadata, dict) else {}
    value = metadata.get("episode_id")
    if value is None:
        row_metadata = rows[0].get("metadata") if isinstance(rows[0].get("metadata"), dict) else {}
        value = row_metadata.get("episode_id")
    return str(value if value is not None else episode_sample.index)


def _rollout_id(episode_sample: Sample) -> int:
    if episode_sample.index is None:
        raise ValueError("episode sample index is required for stable episode rollout_id assignment")
    return int(episode_sample.index)


def _sample_index(episode_sample: Sample, step_index: int) -> int:
    base = 0 if episode_sample.index is None else int(episode_sample.index)
    return base * 100000 + step_index


def _data_parallel_size(args) -> int:
    value = getattr(args, "data_parallel_size", None)
    if isinstance(value, int) and value > 0:
        return value
    world_size = int(getattr(args, "world_size", 1) or 1)
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    denom = max(1, tp_size * pp_size * cp_size)
    return max(1, world_size // denom)


def _sequence_token_limit(args) -> int:
    configured = os.environ.get("SLIME_EPISODE_MAX_TOKENS")
    value = configured if configured not in {None, ""} else getattr(args, "seq_length", 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid SLIME_EPISODE_MAX_TOKENS/seq_length: {value!r}") from exc


def _apply_overlength_policy(
    *,
    args,
    token_ids: list[int],
    loss_mask: list[int],
    episode_id: str,
    step_index: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    max_tokens = _sequence_token_limit(args)
    original_tokens = len(token_ids)
    if max_tokens <= 0 or original_tokens <= max_tokens:
        return token_ids, loss_mask, {}

    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        raise ValueError(f"episode {episode_id} step {step_index} produced empty response_length")
    if response_length > max_tokens:
        raise ValueError(
            f"episode {episode_id} step {step_index} target region is longer than the sequence limit: "
            f"response_length={response_length} max_tokens={max_tokens}"
        )

    policy = os.environ.get("SLIME_EPISODE_OVERLENGTH_POLICY", "error").strip().lower()
    if policy == "error":
        raise ValueError(
            f"episode {episode_id} step {step_index} exceeds the sequence limit: "
            f"tokens={original_tokens} max_tokens={max_tokens}. Set "
            "SLIME_EPISODE_OVERLENGTH_POLICY=truncate_left to retain the complete target "
            "and remove only the oldest context tokens."
        )
    if policy != "truncate_left":
        raise ValueError(f"unsupported SLIME_EPISODE_OVERLENGTH_POLICY={policy!r}")

    truncated_tokens = original_tokens - max_tokens
    token_ids = token_ids[truncated_tokens:]
    loss_mask = loss_mask[truncated_tokens:]
    if MASK_GENERATOR.get_response_lengths([loss_mask])[0] != response_length:
        raise ValueError(
            f"episode {episode_id} step {step_index} left truncation changed the target region"
        )
    return token_ids, loss_mask, {
        "original_token_count": original_tokens,
        "left_truncated_tokens": truncated_tokens,
        "sequence_token_limit": max_tokens,
        "overlength_policy": policy,
    }


def _build_step_sample(
    *,
    args,
    episode_sample: Sample,
    episode_rows: list[dict[str, Any]],
    row: dict[str, Any],
    step_index: int,
    episode_id: str,
    episode_rollout_id: int,
) -> Sample:
    messages = row["messages"]
    tools = row.get("tools", None)
    if tools is None:
        metadata_tools = row.get("metadata", {}).get("tools") if isinstance(row.get("metadata"), dict) else None
        tools = metadata_tools

    token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise ValueError(
            f"episode SFT rollout produced mismatched token_ids/loss_mask lengths: "
            f"{len(token_ids)=}, {len(loss_mask)=}"
        )
    token_ids, loss_mask, truncation_metadata = _apply_overlength_policy(
        args=args,
        token_ids=token_ids,
        loss_mask=loss_mask,
        episode_id=episode_id,
        step_index=step_index,
    )
    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        raise ValueError(f"episode {episode_id} step {step_index} produced empty response_length")

    step_sample = copy.deepcopy(episode_sample)
    row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    step_sample.prompt = messages
    step_sample.metadata = {
        **row_metadata,
        "episode_id": episode_id,
        "episode_step_index": step_index,
        "episode_num_steps": len(episode_rows),
        "episode_rollout_id": episode_rollout_id,
        **truncation_metadata,
    }
    step_sample.train_metadata = dict(step_sample.metadata)
    step_sample.index = _sample_index(episode_sample, step_index)
    step_sample.rollout_id = episode_rollout_id
    step_sample.tokens = token_ids
    step_sample.response_length = response_length
    step_sample.reward = 0
    step_sample.loss_mask = loss_mask[-response_length:]
    return step_sample


def _build_padding_sample(
    *,
    episode_sample: Sample,
    step_index: int,
    episode_id: str,
    episode_rollout_id: int,
) -> Sample:
    dummy_messages = [
        {"role": "user", "content": "padding", "step_loss_mask": 0},
        {"role": "assistant", "content": "padding", "step_loss_mask": 1},
    ]
    token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(dummy_messages, tools=None)
    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        raise ValueError("episode padding sample produced empty response_length")

    step_sample = copy.deepcopy(episode_sample)
    step_sample.prompt = dummy_messages
    step_sample.metadata = {
        "episode_id": episode_id,
        "episode_step_index": step_index,
        "episode_rollout_id": episode_rollout_id,
        "episode_padding": True,
    }
    step_sample.train_metadata = dict(step_sample.metadata)
    step_sample.index = _sample_index(episode_sample, step_index)
    step_sample.rollout_id = episode_rollout_id
    step_sample.tokens = token_ids
    step_sample.response_length = response_length
    step_sample.reward = 0
    step_sample.loss_mask = [0] * response_length
    step_sample.remove_sample = True
    return step_sample


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate SFT train samples while keeping all rows from one episode as one rollout.

    The supervision row format is unchanged: every step still renders its own
    messages and trains only the assistant target with step_loss_mask=1. The
    only scheduling change is that all step samples from the same episode share
    a rollout_id, so Slime accumulates them as one rollout/update unit.
    """

    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    episode_groups = data_buffer.get_samples(args.rollout_batch_size)
    compact_groups: list[list[list[Sample]]] = []
    episode_count = 0
    step_count = 0
    max_episode_steps = 0
    padding_count = 0
    dp_size = _data_parallel_size(args)
    pad_to_dp = os.environ.get("SLIME_EPISODE_PAD_TO_DP", "1") != "0"

    for group in episode_groups:
        if len(group) != 1:
            raise ValueError(
                f"episode rollout requires n_samples_per_prompt=1; got group size {len(group)}. "
                "Increase rollout_batch_size for more episodes instead."
            )
        (episode_sample,) = group
        rows = _episode_rows(episode_sample)
        episode_id = _episode_id(episode_sample, rows)
        episode_rollout_id = _rollout_id(episode_sample)
        episode_metadata = episode_sample.metadata if isinstance(episode_sample.metadata, dict) else {}
        if episode_metadata.get("dataset_padding") is True:
            step_samples = [
                _build_padding_sample(
                    episode_sample=episode_sample,
                    step_index=0,
                    episode_id=episode_id,
                    episode_rollout_id=episode_rollout_id,
                )
            ]
            padding_count += 1
        else:
            step_samples = [
                _build_step_sample(
                    args=args,
                    episode_sample=episode_sample,
                    episode_rows=rows,
                    row=row,
                    step_index=step_index,
                    episode_id=episode_id,
                    episode_rollout_id=episode_rollout_id,
                )
                for step_index, row in enumerate(rows)
            ]
        if pad_to_dp and dp_size > 1:
            pad_count = (-len(step_samples)) % dp_size
            for pad_index in range(pad_count):
                step_samples.append(
                    _build_padding_sample(
                        episode_sample=episode_sample,
                        step_index=len(rows) + pad_index,
                        episode_id=episode_id,
                        episode_rollout_id=episode_rollout_id,
                    )
                )
            padding_count += pad_count
        compact_groups.append([step_samples])
        episode_count += 1
        step_count += len(step_samples)
        max_episode_steps = max(max_episode_steps, len(step_samples))

        if not SAMPLE_PRINTED:
            logger.info(
                "episode_sft_rollout example: "
                f"{episode_id=} {len(step_samples)=} {dp_size=} {padding_count=} first_step={step_samples[0]} "
                f"(raw_messages){step_samples[0].prompt}"
            )
            SAMPLE_PRINTED = True

    return compact_groups
