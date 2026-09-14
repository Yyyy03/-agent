from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch
from megatron.core import mpu
from slime.backends.megatron_utils.loss import get_responses
from slime.utils.ppo_utils import compute_log_probs
from slime.utils.types import RolloutBatch

__all__ = ["response_only_sft_loss_function"]


def _compute_response_log_probs(
    logits_chunk: torch.Tensor,
    tokens_chunk: torch.Tensor,
    *,
    chunk_size: int,
    tp_group,
) -> torch.Tensor:
    if logits_chunk.numel() == 0:
        return logits_chunk.new_empty((0,))

    pieces: list[torch.Tensor] = []
    if chunk_size > 0 and logits_chunk.size(0) > chunk_size:
        for start in range(0, logits_chunk.size(0), chunk_size):
            end = min(start + chunk_size, logits_chunk.size(0))
            piece = compute_log_probs(
                logits_chunk[start:end].contiguous().clone(),
                tokens_chunk[start:end],
                tp_group,
            ).squeeze(-1)
            pieces.append(piece)
    else:
        pieces.append(
            compute_log_probs(
                logits_chunk.contiguous().clone(),
                tokens_chunk,
                tp_group,
            ).squeeze(-1)
        )
    return torch.cat(pieces, dim=0)


def response_only_sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SFT loss that runs vocab CE only on response spans.

    This is an experimental equivalence test for the current Slime SFT path.
    It still receives full-sequence logits from Megatron, so it does not remove
    the full lm_head output allocation. It does avoid full-sequence vocab CE,
    which is the OOM site observed in smoke testing.
    """

    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = int(getattr(args, "log_probs_chunk_size", -1) or -1)

    log_prob_parts: list[torch.Tensor] = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=True,
    ):
        log_prob_parts.append(
            _compute_response_log_probs(
                logits_chunk,
                tokens_chunk,
                chunk_size=chunk_size,
                tp_group=tp_group,
            )
        )

    if log_prob_parts:
        log_probs = torch.cat(log_prob_parts, dim=0)
    else:
        log_probs = logits.new_empty((0,))

    loss = -sum_of_sample_mean(log_probs)

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    return loss, {"loss": loss.clone().detach()}
