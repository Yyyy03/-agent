from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .dataset import (
    BuildStats,
    CompletionOnlyCollator,
    build_features,
    convert_slime_messages_row,
    read_event_rows,
)
from .train_qwen_sft import (
    _filter_rows,
    _length_summary,
    _load_model,
    _load_tokenizer,
    _metadata,
    _padding_feature,
    _str_to_bool,
)


@dataclass
class EpisodeFeatures:
    episode_id: str
    features: list[dict[str, Any]]
    max_tokens: int
    supervised_tokens: int

    @property
    def steps(self) -> int:
        return len(self.features)


def _paths(values: Optional[Sequence[str]]) -> list[Path]:
    if not values:
        return []
    return [Path(path).expanduser().resolve() for path in values]


def _merge_build_stats(total: BuildStats, stats: BuildStats) -> None:
    total.examples_seen += stats.examples_seen
    total.examples_kept += stats.examples_kept
    total.dropped_bad_shape += stats.dropped_bad_shape
    total.dropped_missing_reasoning += stats.dropped_missing_reasoning
    total.dropped_empty_completion += stats.dropped_empty_completion
    total.dropped_overlength += stats.dropped_overlength
    total.truncated_prompt += stats.truncated_prompt
    total.max_prompt_tokens = max(total.max_prompt_tokens, stats.max_prompt_tokens)
    total.max_completion_tokens = max(total.max_completion_tokens, stats.max_completion_tokens)
    total.max_total_tokens = max(total.max_total_tokens, stats.max_total_tokens)


def _label_token_count(feature: dict[str, Any]) -> int:
    labels = feature.get("labels") or []
    if len(labels) <= 1:
        return 0
    return sum(1 for value in labels[1:] if int(value) != -100)


def _build_episode_dataset(
    paths: Sequence[Path],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    name: str,
) -> list[EpisodeFeatures]:
    rows = read_event_rows(paths)
    if name == "train" and args.episode_subset == "largest_tokens":
        rows.sort(key=lambda row: int(_metadata(row).get("max_token_count") or 0), reverse=True)
    elif name == "train" and args.episode_subset == "largest_steps":
        rows.sort(key=lambda row: int(_metadata(row).get("num_steps") or 0), reverse=True)
    if name == "train" and args.max_train_episodes > 0:
        rows = rows[: args.max_train_episodes]
    if name == "eval" and args.max_eval_episodes > 0:
        rows = rows[: args.max_eval_episodes]

    total_stats = BuildStats()
    episodes: list[EpisodeFeatures] = []
    raw_step_counts: list[int] = []
    kept_step_counts: list[int] = []
    max_token_counts: list[int] = []
    dropped = 0

    for episode in rows:
        raw_rows = episode.get("rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            dropped += 1
            continue
        episode_id = str(episode.get("episode_id") or _metadata(episode).get("episode_id") or f"{name}-{len(episodes)}")
        converted: list[dict[str, Any]] = []
        for step_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                continue
            item = convert_slime_messages_row(row, default_module=args.slime_default_module)
            if item is None:
                continue
            item.setdefault("episode_id", episode_id)
            item.setdefault("episode_step_index", step_index)
            converted.append(item)
        converted = _filter_rows(converted, args)
        raw_step_counts.append(len(raw_rows))
        if not converted:
            dropped += 1
            continue
        features, stats = build_features(
            converted,
            tokenizer,
            max_seq_length=args.max_seq_length,
            train_profile=args.train_profile,
            reasoning_format=args.reasoning_format,
            require_reasoning=args.require_reasoning,
            on_overlength=args.on_overlength,
            add_eos=not args.no_add_eos,
            enable_thinking_template=args.enable_thinking_template,
            require_chat_template=args.require_chat_template,
        )
        _merge_build_stats(total_stats, stats)
        if not features:
            dropped += 1
            continue
        for step_index, feature in enumerate(features):
            feature["episode_id"] = feature.get("episode_id") or episode_id
            feature["episode_step_index"] = step_index
            feature["episode_num_steps"] = len(raw_rows)
        max_tokens = max(len(feature["input_ids"]) for feature in features)
        supervised_tokens = sum(_label_token_count(feature) for feature in features)
        episodes.append(
            EpisodeFeatures(
                episode_id=episode_id,
                features=features,
                max_tokens=max_tokens,
                supervised_tokens=supervised_tokens,
            )
        )
        kept_step_counts.append(len(features))
        max_token_counts.append(max_tokens)

    if not episodes:
        raise ValueError(f"No usable {name} episodes after filtering.")
    if getattr(args, "log_dataset_stats", True):
        print(
            f"[episode-lora] {name} episodes: raw={len(rows)} kept={len(episodes)} "
            f"dropped={dropped} features={sum(item.steps for item in episodes)}"
        )
        print(f"[episode-lora] {name} build stats: {json.dumps(total_stats.to_dict(), ensure_ascii=False)}")
        print(f"[episode-lora] {name} raw step summary: {json.dumps(_length_summary(raw_step_counts), ensure_ascii=False)}")
        print(f"[episode-lora] {name} kept step summary: {json.dumps(_length_summary(kept_step_counts), ensure_ascii=False)}")
        print(f"[episode-lora] {name} max token summary: {json.dumps(_length_summary(max_token_counts), ensure_ascii=False)}")
    return episodes


def _ordered_indices(
    episodes: Sequence[EpisodeFeatures],
    *,
    epoch: int,
    seed: int,
    order: str,
    bucket_size: int,
) -> list[int]:
    indices = list(range(len(episodes)))
    rng = random.Random(seed + epoch)
    if order == "sequential":
        return indices
    if order == "shuffle":
        rng.shuffle(indices)
        return indices

    if order == "bucket_tokens":
        indices.sort(key=lambda index: (episodes[index].max_tokens, episodes[index].steps, episodes[index].episode_id))
    elif order == "bucket_steps":
        indices.sort(key=lambda index: (episodes[index].steps, episodes[index].max_tokens, episodes[index].episode_id))
    else:
        raise ValueError(f"Unsupported episode order: {order}")

    bucket_size = max(1, int(bucket_size))
    buckets = [indices[start : start + bucket_size] for start in range(0, len(indices), bucket_size)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    return [index for bucket in buckets for index in bucket]


def _init_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=args.ddp_timeout),
            device_id=torch.device("cuda", local_rank),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _all_reduce_sum_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_sum_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _local_update_features(
    episodes: Sequence[EpisodeFeatures],
    update_indices: Sequence[int],
    *,
    rank: int,
    episodes_per_rank: int,
) -> list[dict[str, Any]]:
    start = rank * episodes_per_rank
    end = start + episodes_per_rank
    local_features: list[dict[str, Any]] = []
    for index in update_indices[start:end]:
        local_features.extend(episodes[index].features)
    return local_features


def _run_update(
    *,
    model: DistributedDataParallel,
    collator: CompletionOnlyCollator,
    padding_feature: dict[str, Any],
    local_features: Sequence[dict[str, Any]],
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 0.0,
) -> tuple[float, int, int]:
    local_tokens = sum(_label_token_count(feature) for feature in local_features)
    global_tokens = _all_reduce_sum_int(local_tokens, device)
    local_steps = len(local_features)
    global_steps = _all_reduce_sum_int(local_steps, device)
    max_local_steps = _all_reduce_max_int(local_steps, device)
    if global_steps == 0 or max_local_steps == 0:
        return 0.0, 0, 0

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    for step_index in range(max_local_steps):
        feature = local_features[step_index] if step_index < local_steps else padding_feature
        batch = _move_to_device(collator([feature]), device)
        num_items = torch.tensor(max(1, global_tokens), device=device, dtype=torch.float32)
        sync_context = (
            model.no_sync()
            if train and step_index < max_local_steps - 1
            else contextlib.nullcontext()
        )
        with sync_context:
            with torch.set_grad_enabled(train):
                outputs = model(**batch, num_items_in_batch=num_items)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                if train:
                    loss.backward()
        if step_index < local_steps:
            total_loss += float(loss.detach().float().item())

    if train and optimizer is not None:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
        optimizer.step()

    global_loss = _all_reduce_sum_float(total_loss, device)
    return global_loss, global_steps, global_tokens


@torch.no_grad()
def _evaluate(
    *,
    model: DistributedDataParallel,
    episodes: Sequence[EpisodeFeatures],
    collator: CompletionOnlyCollator,
    padding_feature: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    episodes_per_rank: int,
    limit_updates: int,
) -> dict[str, float]:
    model.eval()
    global_episode_batch = world_size * episodes_per_rank
    update_count = math.ceil(len(episodes) / global_episode_batch)
    if limit_updates > 0:
        update_count = min(update_count, limit_updates)
    loss_sum = 0.0
    token_sum = 0
    step_sum = 0
    for update_index in range(update_count):
        start = update_index * global_episode_batch
        update_indices = list(range(start, min(start + global_episode_batch, len(episodes))))
        local_features = _local_update_features(
            episodes,
            update_indices,
            rank=rank,
            episodes_per_rank=episodes_per_rank,
        )
        loss, steps, tokens = _run_update(
            model=model,
            collator=collator,
            padding_feature=padding_feature,
            local_features=local_features,
            device=device,
            train=False,
        )
        loss_sum += loss
        token_sum += tokens
        step_sum += steps
    model.train()
    return {
        "eval_loss": loss_sum / max(1, update_count),
        "eval_updates": float(update_count),
        "eval_steps": float(step_sum),
        "eval_tokens": float(token_sum),
    }


def _build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optim == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError("--optim paged_adamw_8bit requires bitsandbytes.") from exc
        return bnb.optim.PagedAdamW8bit(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    if args.optim != "adamw_torch":
        raise ValueError(f"Unsupported optimizer for episode LoRA loop: {args.optim}")
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )


def _build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, total_updates: int) -> Any:
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    warmup_updates = int(total_updates * args.warmup_ratio)
    if args.lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    if args.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    raise ValueError(f"Unsupported scheduler for episode LoRA loop: {args.lr_scheduler_type}")


def _save_adapter(
    *,
    model: DistributedDataParallel,
    tokenizer: Any,
    output_dir: Path,
    name: str,
    rank: int,
) -> None:
    dist.barrier()
    if rank == 0:
        save_dir = output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"[episode-lora] saved adapter: {save_dir}")
    dist.barrier()


def _load_lora_adapter_state(model: torch.nn.Module, adapter_path: str, *, rank: int) -> None:
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"LoRA adapter checkpoint does not exist: {path}")
    weights_path = path / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"LoRA adapter weights not found: {weights_path}")
    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError("Resuming LoRA adapters requires peft and safetensors.") from exc

    state_dict = load_file(str(weights_path), device="cpu")
    result = set_peft_model_state_dict(model, state_dict, adapter_name="default")
    if rank == 0:
        missing = len(getattr(result, "missing_keys", []) or [])
        unexpected = len(getattr(result, "unexpected_keys", []) or [])
        print(
            "[episode-lora] loaded resume adapter: "
            f"{path} missing_keys={missing} unexpected_keys={unexpected}"
        )


def _prune_checkpoints(output_dir: Path, limit: int, *, rank: int) -> None:
    if limit <= 0:
        return
    dist.barrier()
    if rank == 0:
        checkpoints: list[tuple[int, Path]] = []
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            checkpoints.append((step, path))
        checkpoints.sort()
        for _, path in checkpoints[:-limit]:
            import shutil

            shutil.rmtree(path, ignore_errors=True)
            print(f"[episode-lora] pruned old checkpoint: {path}")
    dist.barrier()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episode-batch LoRA SFT for Qwen-style FIRE Agent data.")
    parser.add_argument("--train_file", nargs="+", required=True)
    parser.add_argument("--eval_file", nargs="*", default=None)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=81920)
    parser.add_argument("--on_overlength", default="truncate_left_prompt", choices=["drop", "error", "truncate_left_prompt"])
    parser.add_argument("--slime_default_module", default="agent_policy")
    parser.add_argument("--sample_types", default=None)
    parser.add_argument("--target_types", default=None)
    parser.add_argument("--max_train_episodes", type=int, default=0)
    parser.add_argument("--max_eval_episodes", type=int, default=0)
    parser.add_argument(
        "--episode_subset",
        default="first",
        choices=["first", "largest_tokens", "largest_steps"],
        help="Which training episodes to keep when --max_train_episodes is set; useful for worst-case smoke tests.",
    )
    parser.add_argument("--train_profile", default="agent_policy_reasoning")
    parser.add_argument("--require_reasoning", action="store_true")
    parser.add_argument("--reasoning_format", default="qwen_auto_think_tags")
    parser.add_argument("--enable_thinking_template", type=lambda value: _str_to_bool(value, True), default=True)
    parser.add_argument("--require_chat_template", type=lambda value: _str_to_bool(value, True), default=True)
    parser.add_argument("--no_add_eos", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--auto_model_class", default="multimodal_lm")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--freeze_vision_model", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--loss_impl", default="cut_cross_entropy", choices=["cut_cross_entropy", "model_default"])
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.02)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--episodes_per_rank", type=int, default=2)
    parser.add_argument("--episode_order", default="bucket_tokens", choices=["sequential", "shuffle", "bucket_tokens", "bucket_steps"])
    parser.add_argument("--bucket_size", type=int, default=96)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_updates", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--optim", default="adamw_torch", choices=["adamw_torch", "paged_adamw_8bit"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--eval_updates", type=int, default=20)
    parser.add_argument("--resume_lora_adapter", default=None)
    parser.add_argument("--resume_from_update", type=int, default=0)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp_timeout", type=int, default=7200)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_save_model", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args(argv)
    args.data_format = "slime_episode_rows"
    args.deepspeed = None
    args.device_map = None
    args.bf16 = True
    args.fp16 = False
    args.tf32 = True
    args.gradient_checkpointing_reentrant = False
    args.ddp_find_unused_parameters = False
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    rank, local_rank, world_size, device = _init_distributed(args)
    args.log_dataset_stats = rank == 0
    if rank == 0:
        print(f"[episode-lora] rank setup: world_size={world_size} local_rank={local_rank}")
        print(f"[episode-lora] args: {json.dumps(vars(args), ensure_ascii=False, default=str)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = _load_tokenizer(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if args.eval_only:
        if not args.eval_file:
            raise ValueError("--eval_only requires --eval_file.")
        eval_episodes = _build_episode_dataset(_paths(args.eval_file), tokenizer, args, name="eval")
        if args.dry_run:
            if rank == 0:
                print("[episode-lora] eval-only dry run complete; model was not loaded.")
            dist.barrier()
            dist.destroy_process_group()
            return
        model = _load_model(args).to(device)
        if args.resume_lora_adapter:
            _load_lora_adapter_state(model, args.resume_lora_adapter, rank=rank)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        collator = CompletionOnlyCollator(tokenizer, pad_to_multiple_of=args.pad_to_multiple_of)
        padding_feature = _padding_feature(tokenizer, episode_id="ddp_padding", pad_index=0)
        metrics = _evaluate(
            model=model,
            episodes=eval_episodes,
            collator=collator,
            padding_feature=padding_feature,
            device=device,
            rank=rank,
            world_size=world_size,
            episodes_per_rank=args.episodes_per_rank,
            limit_updates=args.eval_updates,
        )
        if rank == 0:
            print(f"[episode-lora] eval_only: {json.dumps(metrics, ensure_ascii=False)}")
        dist.barrier()
        dist.destroy_process_group()
        return

    train_episodes = _build_episode_dataset(_paths(args.train_file), tokenizer, args, name="train")
    eval_episodes = (
        _build_episode_dataset(_paths(args.eval_file), tokenizer, args, name="eval")
        if args.eval_file
        else []
    )
    if args.dry_run:
        if rank == 0:
            print("[episode-lora] dry run complete; model was not loaded.")
        dist.barrier()
        dist.destroy_process_group()
        return

    model = _load_model(args).to(device)
    if args.resume_lora_adapter:
        _load_lora_adapter_state(model, args.resume_lora_adapter, rank=rank)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    model.train()

    optimizer = _build_optimizer(args, model)
    global_episode_batch = world_size * args.episodes_per_rank
    updates_per_epoch = math.ceil(len(train_episodes) / global_episode_batch)
    total_updates = updates_per_epoch * args.num_train_epochs
    if args.max_updates > 0:
        total_updates = min(total_updates, args.max_updates)
    if args.resume_from_update < 0:
        raise ValueError("--resume_from_update must be >= 0")
    if args.resume_from_update > total_updates:
        raise ValueError(
            f"--resume_from_update={args.resume_from_update} exceeds total_updates={total_updates}"
        )
    scheduler = _build_scheduler(args, optimizer, max(1, total_updates))
    for _ in range(args.resume_from_update):
        scheduler.step()
    collator = CompletionOnlyCollator(tokenizer, pad_to_multiple_of=args.pad_to_multiple_of)
    padding_feature = _padding_feature(tokenizer, episode_id="ddp_padding", pad_index=0)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            "[episode-lora] training setup: "
            f"episodes={len(train_episodes)} world_size={world_size} "
            f"episodes_per_rank={args.episodes_per_rank} global_episode_batch={global_episode_batch} "
            f"updates_per_epoch={updates_per_epoch} total_updates={total_updates} "
            f"resume_from_update={args.resume_from_update}"
        )

    global_update = 0
    stop = False
    for epoch in range(args.num_train_epochs):
        order = _ordered_indices(
            train_episodes,
            epoch=epoch,
            seed=args.seed,
            order=args.episode_order,
            bucket_size=args.bucket_size,
        )
        for update_in_epoch in range(updates_per_epoch):
            if global_update < args.resume_from_update:
                global_update += 1
                continue
            if args.max_updates > 0 and global_update >= args.max_updates:
                stop = True
                break
            start = update_in_epoch * global_episode_batch
            update_indices = order[start : start + global_episode_batch]
            local_features = _local_update_features(
                train_episodes,
                update_indices,
                rank=rank,
                episodes_per_rank=args.episodes_per_rank,
            )
            loss, global_steps, global_tokens = _run_update(
                model=model,
                collator=collator,
                padding_feature=padding_feature,
                local_features=local_features,
                device=device,
                train=True,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
            )
            scheduler.step()
            global_update += 1

            if rank == 0 and (global_update == 1 or global_update % args.logging_steps == 0):
                lr = scheduler.get_last_lr()[0]
                memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(
                    "[episode-lora] "
                    f"update={global_update}/{total_updates} epoch={epoch + 1} "
                    f"loss={loss:.6f} lr={lr:.6g} steps={global_steps} "
                    f"tokens={global_tokens} max_mem_gb={memory_gb:.2f}"
                )

            if eval_episodes and args.eval_steps > 0 and global_update % args.eval_steps == 0:
                metrics = _evaluate(
                    model=model,
                    episodes=eval_episodes,
                    collator=collator,
                    padding_feature=padding_feature,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    episodes_per_rank=args.episodes_per_rank,
                    limit_updates=args.eval_updates,
                )
                if rank == 0:
                    print(f"[episode-lora] eval update={global_update}: {json.dumps(metrics, ensure_ascii=False)}")

            if not args.skip_save_model and args.save_steps > 0 and global_update % args.save_steps == 0:
                _save_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    name=f"checkpoint-{global_update}",
                    rank=rank,
                )
                _prune_checkpoints(output_dir, args.save_total_limit, rank=rank)
        if stop:
            break

    if not args.skip_save_model:
        _save_adapter(model=model, tokenizer=tokenizer, output_dir=output_dir, name="final_adapter", rank=rank)
    if rank == 0:
        print("[episode-lora] training complete")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
