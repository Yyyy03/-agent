from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional, Sequence

from .dataset import (
    BuildStats,
    CompletionOnlyCollator,
    EventSFTDataset,
    build_features,
    convert_slime_messages_row,
    read_event_rows,
    split_by_episode,
)


MODULES = ("agent_policy", "context_compression", "content_reader")
DEFAULT_MIXTURE = "agent_policy=0.65,context_compression=0.25,content_reader=0.10"


def _split_csv(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    parsed = {part.strip() for part in value.split(",") if part.strip()}
    return parsed or None


def _filter_rows(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    sample_types = _split_csv(args.sample_types)
    target_types = _split_csv(args.target_types)
    filtered: list[Dict[str, Any]] = []
    for row in rows:
        module = str(row.get("module") or "")
        if sample_types is not None and module not in sample_types:
            continue
        if target_types is not None and str(row.get("target_type") or "") not in target_types:
            continue
        filtered.append(dict(row))
    return filtered


def _paths(values: Optional[Sequence[str]]) -> list[Path]:
    if not values:
        return []
    return [Path(path).expanduser().resolve() for path in values]


def _read_module_rows(paths: Sequence[Path], *, module: Optional[str] = None) -> list[Dict[str, Any]]:
    rows = [dict(row) for row in read_event_rows(paths)]
    if module:
        for row in rows:
            row.setdefault("module", module)
    return rows


def _convert_slime_rows(rows: Sequence[Dict[str, Any]], args: argparse.Namespace, *, split: str) -> list[Dict[str, Any]]:
    converted: list[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        item = convert_slime_messages_row(row, default_module=args.slime_default_module)
        if item is None:
            dropped += 1
            continue
        converted.append(item)
    print(
        f"[fire-agent-sft-train] {split} slime message rows converted: "
        f"kept={len(converted)} dropped={dropped}"
    )
    return converted


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


def _padding_feature(tokenizer: Any, *, episode_id: Any, pad_index: int) -> Dict[str, Any]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    token_id = int(pad_token_id if pad_token_id is not None else eos_token_id if eos_token_id is not None else 0)
    return {
        "input_ids": [token_id],
        "attention_mask": [1],
        "labels": [-100],
        "episode_id": episode_id,
        "turn_id": None,
        "module": "agent_policy",
        "target_type": "episode_padding",
        "step_number": pad_index,
        "target_mode": "padding",
        "episode_padding": True,
    }


def _build_episode_features(
    episode_rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    name: str,
) -> list[Dict[str, Any]]:
    total_stats = BuildStats()
    features: list[Dict[str, Any]] = []
    module_counts: Counter[str] = Counter()
    episode_count = 0
    kept_episode_count = 0
    dropped_episode_count = 0
    padding_count = 0
    episode_step_counts: list[int] = []
    episode_feature_counts: list[int] = []
    for episode in episode_rows:
        rows = episode.get("rows")
        if not isinstance(rows, list) or not rows:
            dropped_episode_count += 1
            continue
        episode_id = episode.get("episode_id") or _metadata(episode).get("episode_id")
        converted: list[Dict[str, Any]] = []
        for step_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            item = convert_slime_messages_row(row, default_module=args.slime_default_module)
            if item is None:
                continue
            item.setdefault("episode_id", episode_id)
            item.setdefault("episode_step_index", step_index)
            converted.append(item)
        converted = _filter_rows(converted, args)
        episode_count += 1
        episode_step_counts.append(len(rows))
        if not converted:
            dropped_episode_count += 1
            continue
        step_features, stats = build_features(
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
        if not step_features:
            dropped_episode_count += 1
            continue
        for step_index, feature in enumerate(step_features):
            feature["episode_id"] = feature.get("episode_id") or episode_id
            feature["episode_step_index"] = step_index
            feature["episode_num_steps"] = len(rows)
        kept_episode_count += 1
        episode_feature_counts.append(len(step_features))
        module_counts.update(str(item.get("module") or "unknown") for item in step_features)
        features.extend(step_features)
        if name == "train" and args.episode_pad_to_multiple > 0:
            pad_to = int(args.episode_pad_to_multiple)
            pad_count = (-len(step_features)) % pad_to
            for pad_index in range(pad_count):
                features.append(
                    _padding_feature(
                        tokenizer,
                        episode_id=episode_id,
                        pad_index=len(step_features) + pad_index,
                    )
                )
            padding_count += pad_count
    print(
        f"[fire-agent-sft-train] {name} episode rows: "
        f"episodes={episode_count} kept={kept_episode_count} dropped={dropped_episode_count} "
        f"features={len(features)} padding={padding_count}"
    )
    print(f"[fire-agent-sft-train] {name} build stats: {json.dumps(total_stats.to_dict(), ensure_ascii=False)}")
    if episode_step_counts:
        print(
            f"[fire-agent-sft-train] {name} episode step summary: "
            f"{json.dumps(_length_summary(episode_step_counts), ensure_ascii=False)}"
        )
    if episode_feature_counts:
        print(
            f"[fire-agent-sft-train] {name} kept feature summary: "
            f"{json.dumps(_length_summary(episode_feature_counts), ensure_ascii=False)}"
        )
    if not features:
        raise ValueError(f"No usable {name} features after episode filtering.")
    print(f"[fire-agent-sft-train] {name} module counts: {json.dumps(dict(module_counts), ensure_ascii=False)}")
    return features


def _length_summary(values: Sequence[int]) -> Dict[str, int]:
    if not values:
        return {}
    ordered = sorted(int(value) for value in values)

    def q(pct: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * pct) - 1))]

    return {
        "min": ordered[0],
        "p50": q(0.50),
        "p80": q(0.80),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def _load_rows(args: argparse.Namespace, *, split: str) -> list[Dict[str, Any]]:
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split: {split}")
    prefix = "" if split == "train" else "eval_"
    combined_values = args.train_file if split == "train" else args.eval_file
    rows: list[Dict[str, Any]] = []
    rows.extend(_read_module_rows(_paths(combined_values)))
    rows.extend(_read_module_rows(_paths(getattr(args, f"{prefix}agent_policy_file")), module="agent_policy"))
    rows.extend(
        _read_module_rows(
            _paths(getattr(args, f"{prefix}context_compression_file")),
            module="context_compression",
        )
    )
    rows.extend(_read_module_rows(_paths(getattr(args, f"{prefix}content_reader_file")), module="content_reader"))
    if args.data_format == "slime_episode_rows":
        return rows
    if args.data_format == "slime_messages":
        rows = _convert_slime_rows(rows, args, split=split)
    return _filter_rows(rows, args)


def _str_to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _torch_dtype(name: str) -> Any:
    import torch

    value = str(name or "auto").strip().lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _parse_mixture(value: Optional[str]) -> Dict[str, float]:
    if value is None:
        value = DEFAULT_MIXTURE
    text = str(value).strip()
    if not text:
        return {}
    mixture: Dict[str, float] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, raw_weight = item.split("=", 1)
        elif ":" in item:
            key, raw_weight = item.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid mixture item {item!r}; expected module=weight."
            )
        module = key.strip()
        if module not in MODULES:
            raise argparse.ArgumentTypeError(f"Unknown mixture module: {module}")
        weight = float(raw_weight)
        if weight < 0:
            raise argparse.ArgumentTypeError(f"Negative mixture weight for {module}: {weight}")
        mixture[module] = weight
    return mixture


def _module_counts(features: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(item.get("module") or "unknown") for item in features))


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _allocate_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    if total <= 0:
        return {}
    weight_sum = sum(float(value) for value in weights.values() if value > 0)
    if weight_sum <= 0:
        return {}
    raw = {module: total * (float(weight) / weight_sum) for module, weight in weights.items() if weight > 0}
    counts = {module: int(value) for module, value in raw.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        ordered = sorted(raw, key=lambda module: (raw[module] - int(raw[module]), module), reverse=True)
        for module in ordered[:remainder]:
            counts[module] += 1
    return counts


def _sample_features(
    features: Sequence[Dict[str, Any]],
    *,
    mixture: Dict[str, float],
    total_examples: int,
    seed: int,
) -> list[Dict[str, Any]]:
    if not features or not mixture:
        return list(features)
    by_module: Dict[str, list[Dict[str, Any]]] = {}
    for feature in features:
        by_module.setdefault(str(feature.get("module") or "unknown"), []).append(dict(feature))
    active_weights = {
        module: weight
        for module, weight in mixture.items()
        if weight > 0 and by_module.get(module)
    }
    if not active_weights:
        return list(features)
    total = int(total_examples or 0) if total_examples else len(features)
    counts = _allocate_counts(active_weights, total)
    rng = random.Random(seed)
    mixed: list[Dict[str, Any]] = []
    for module, count in counts.items():
        pool = by_module.get(module) or []
        if not pool or count <= 0:
            continue
        if count <= len(pool):
            mixed.extend(rng.sample(pool, count))
        else:
            mixed.extend(pool)
            mixed.extend(rng.choice(pool) for _ in range(count - len(pool)))
    rng.shuffle(mixed)
    return mixed


def _load_tokenizer(model_name_or_path: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(args: argparse.Namespace) -> Any:
    if args.deepspeed and args.device_map:
        raise ValueError("--device_map cannot be used with --deepspeed distributed training.")
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": _torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError("--load_in_4bit requires bitsandbytes to be installed.") from exc
        kwargs["load_in_4bit"] = True
    model = _load_model_with_auto_class(args.model_name_or_path, args.auto_model_class, kwargs)
    if args.freeze_vision_model:
        _freeze_vision_model(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if args.loss_impl == "cut_cross_entropy":
        _patch_cut_cross_entropy_loss(model)
    if args.use_lora:
        model = _wrap_lora(model, args)
        if args.freeze_vision_model:
            _freeze_vision_model(model)
    _print_trainable_parameters(model)
    return model


def _load_model_with_auto_class(model_name_or_path: str, auto_model_class: str, kwargs: Dict[str, Any]) -> Any:
    import transformers

    class_names = {
        "causal_lm": ["AutoModelForCausalLM"],
        "multimodal_lm": ["AutoModelForMultimodalLM"],
        "image_text_to_text": ["AutoModelForImageTextToText"],
        "auto": ["AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"],
    }[auto_model_class]
    errors: list[str] = []
    for class_name in class_names:
        cls = getattr(transformers, class_name, None)
        if cls is None:
            errors.append(f"{class_name}: unavailable in transformers {transformers.__version__}")
            continue
        try:
            print(f"[fire-agent-sft-train] loading model with {class_name}")
            return cls.from_pretrained(model_name_or_path, **kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
            if auto_model_class != "auto":
                break
    raise RuntimeError(
        "Could not load model. Tried:\n  - " + "\n  - ".join(errors)
    )


def _freeze_vision_model(model: Any) -> None:
    frozen = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "vision" in lowered or "visual" in lowered:
            parameter.requires_grad = False
            frozen += parameter.numel()
    print(f"[fire-agent-sft-train] froze vision/visual parameters: {frozen}")


def _print_trainable_parameters(model: Any) -> None:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    pct = (100.0 * trainable / total) if total else 0.0
    print(
        "[fire-agent-sft-train] trainable parameters: "
        f"{trainable}/{total} ({pct:.4f}%)"
    )


def _patch_cut_cross_entropy_loss(model: Any) -> None:
    try:
        from cut_cross_entropy import linear_cross_entropy
    except Exception as exc:
        raise RuntimeError("--loss_impl cut_cross_entropy requires cut_cross_entropy to be installed.") from exc

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast
    except Exception:
        Qwen3_5CausalLMOutputWithPast = None

    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise RuntimeError("--loss_impl cut_cross_entropy expects a Qwen-style model with .model and .lm_head.")

    original_forward = model.forward

    def forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        labels: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        mm_token_type_ids: Any = None,
        logits_to_keep: Any = 0,
        num_items_in_batch: Any = None,
        **kwargs: Any,
    ) -> Any:
        if labels is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        shifted_labels = labels[..., 1:] if getattr(labels, "ndim", 0) >= 2 and labels.shape[-1] > 1 else labels
        if not bool(shifted_labels.ne(-100).any().item()):
            lm_head_zero = self.lm_head(hidden_states[:, -1:, :]).sum()
            loss = (hidden_states.sum() + lm_head_zero) * 0.0
        elif num_items_in_batch is None:
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                ignore_index=-100,
                reduction="mean",
                shift=True,
            )
        else:
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                ignore_index=-100,
                reduction="sum",
                shift=True,
            )
            loss = loss / num_items_in_batch.to(loss.device)

        if Qwen3_5CausalLMOutputWithPast is None:
            return {"loss": loss}
        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    model.forward = MethodType(forward, model)
    print("[fire-agent-sft-train] patched model forward for cut_cross_entropy loss")


def _wrap_lora(model: Any, args: argparse.Namespace) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError("--use_lora requires peft to be installed.") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    elif args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_modules: Any
    if args.lora_target_modules.strip() == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
        if not target_modules:
            raise ValueError("--lora_target_modules produced an empty target list.")
        _validate_lora_target_modules(model, target_modules)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def _validate_lora_target_modules(model: Any, target_modules: Sequence[str]) -> None:
    import torch

    counts: Dict[str, int] = {target: 0 for target in target_modules}
    matched: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for target in target_modules:
            if name == target or name.endswith(f".{target}"):
                counts[target] += 1
                matched.append(name)
                break
    missing = [target for target, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            "--lora_target_modules contains names that do not match any Linear "
            f"module: {','.join(missing)}"
        )
    visual_matches = [
        name
        for name in matched
        if "vision" in name.lower() or "visual" in name.lower()
    ]
    if visual_matches:
        preview = ", ".join(visual_matches[:8])
        raise ValueError(
            "--lora_target_modules matched visual/vision modules in text-only SFT: "
            f"{preview}"
        )
    print(
        "[fire-agent-sft-train] LoRA target module matches: "
        f"{json.dumps(counts, ensure_ascii=False)} total={len(matched)}"
    )


def _build_features_for_rows(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    name: str,
) -> list[Dict[str, Any]]:
    features, stats = build_features(
        rows,
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
    print(f"[fire-agent-sft-train] {name} build stats: {json.dumps(stats.to_dict(), ensure_ascii=False)}")
    if not features:
        raise ValueError(f"No usable {name} features after filtering.")
    print(f"[fire-agent-sft-train] {name} module counts: {json.dumps(_module_counts(features), ensure_ascii=False)}")
    return features


def _trainer_class(base_trainer: Any, args: argparse.Namespace) -> Any:
    if args.train_sampler != "sequential":
        return base_trainer

    class SequentialTrainer(base_trainer):
        def _get_train_sampler(self, dataset: Any = None) -> Any:
            target_dataset = dataset if dataset is not None else self.train_dataset
            if target_dataset is None:
                return None
            try:
                len(target_dataset)
            except TypeError:
                return None
            from torch.utils.data import SequentialSampler

            return SequentialSampler(target_dataset)

    return SequentialTrainer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Qwen-style causal LMs on FIRE Agent turn-level SFT rows with "
            "completion-only loss. Historical messages are prompt context only."
        )
    )
    parser.add_argument(
        "--train_file",
        nargs="*",
        default=None,
        help="Optional combined turn-level SFT JSONL files from export_events.",
    )
    parser.add_argument("--agent_policy_file", nargs="*", default=None, help="Agent policy turn JSONL files.")
    parser.add_argument(
        "--context_compression_file",
        nargs="*",
        default=None,
        help="Context compression turn JSONL files.",
    )
    parser.add_argument("--content_reader_file", nargs="*", default=None, help="Content reader turn JSONL files.")
    parser.add_argument("--eval_file", nargs="*", default=None, help="Optional turn-level eval JSONL files.")
    parser.add_argument("--eval_agent_policy_file", nargs="*", default=None, help="Eval agent policy turn JSONL files.")
    parser.add_argument(
        "--eval_context_compression_file",
        nargs="*",
        default=None,
        help="Eval context compression turn JSONL files.",
    )
    parser.add_argument("--eval_content_reader_file", nargs="*", default=None, help="Eval content reader turn JSONL files.")
    parser.add_argument(
        "--data_format",
        default="events",
        choices=["events", "slime_messages", "slime_episode_rows"],
        help=(
            "Input row shape. Use slime_messages for JSONL exported by "
            "fire_agent.sft.export_slime_qwen35_sft, or slime_episode_rows "
            "for train.episodes.jsonl/eval.episodes.jsonl."
        ),
    )
    parser.add_argument(
        "--slime_default_module",
        default="agent_policy",
        choices=sorted(MODULES),
        help="Module label assigned to Slime message rows before feature building.",
    )
    parser.add_argument("--validation_split_ratio", type=float, default=0.0, help="Episode-level split ratio if eval_file is absent.")
    parser.add_argument("--sample_types", default=None, help="Optional comma-separated filter, e.g. agent_policy.")
    parser.add_argument("--target_types", default=None, help="Optional comma-separated filter, e.g. action,final_answer.")
    parser.add_argument("--max_train_examples", type=int, default=0, help="Optional row cap after filtering; useful for smoke tests.")
    parser.add_argument("--max_eval_examples", type=int, default=0, help="Optional eval row cap after filtering; useful for smoke tests.")
    parser.add_argument(
        "--episode_pad_to_multiple",
        type=int,
        default=0,
        help=(
            "For slime_episode_rows train data, insert no-loss padding features at "
            "episode boundaries so sequential distributed microbatches do not cross episodes. "
            "Use world_size * per_device_train_batch_size."
        ),
    )
    parser.add_argument(
        "--mixture",
        default=DEFAULT_MIXTURE,
        help=(
            "Comma-separated module weights, e.g. "
            "agent_policy=0.65,context_compression=0.25,content_reader=0.10. "
            "Use empty string to keep natural data proportions."
        ),
    )
    parser.add_argument(
        "--mixture_total_examples",
        type=int,
        default=0,
        help="Optional train examples after mixture sampling. Defaults to usable train feature count.",
    )
    parser.add_argument("--model_name_or_path", required=True, help="Base Qwen model path or HF id.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints/final model.")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument(
        "--train_profile",
        default="content_only",
        choices=["content_only", "agent_policy_reasoning"],
        help=(
            "content_only trains output_message.content/tool_calls for all modules; "
            "agent_policy_reasoning trains reasoning+content only for agent_policy."
        ),
    )
    parser.add_argument("--require_reasoning", action="store_true", help="Drop examples missing output_message.reasoning_content.")
    parser.add_argument(
        "--reasoning_format",
        default="qwen_auto_think_tags",
        choices=["qwen_auto_think_tags", "qwen_think_tags", "qwen_prefilled_think", "deepseek_reasoning_content"],
        help="How to serialize output_message.reasoning_content for reasoning_and_content targets.",
    )
    parser.add_argument(
        "--enable_thinking_template",
        type=lambda v: _str_to_bool(v, True),
        default=None,
        help="Optional Qwen chat-template enable_thinking kwarg for prompt rendering.",
    )
    parser.add_argument(
        "--require_chat_template",
        type=lambda v: _str_to_bool(v, True),
        default=True,
        help="Require the target tokenizer chat_template. Keep true for real Qwen SFT.",
    )
    parser.add_argument(
        "--on_overlength",
        default="drop",
        choices=["drop", "error", "truncate_left_prompt"],
        help="Exact SFT should usually use drop or regenerate with a larger context budget.",
    )
    parser.add_argument("--no_add_eos", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument(
        "--loss_impl",
        default="model_default",
        choices=["model_default", "cut_cross_entropy"],
        help=(
            "Loss implementation. cut_cross_entropy avoids materializing full "
            "sequence x vocab logits during long-context full-parameter SFT."
        ),
    )
    parser.add_argument(
        "--auto_model_class",
        default="auto",
        choices=["auto", "causal_lm", "multimodal_lm", "image_text_to_text"],
        help="Transformers AutoModel class to use. Qwen3.5-9B usually needs multimodal_lm.",
    )
    parser.add_argument("--attn_implementation", default=None, help="Optional transformers attention backend, e.g. sdpa/eager/flash_attention_2.")
    parser.add_argument("--device_map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--freeze_vision_model", action="store_true", help="Freeze parameters whose names contain vision/visual.")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj",
        help=(
            "Comma-separated module suffixes for LoRA. The default covers Qwen full "
            "attention, Qwen3.5 Gated DeltaNet projections, and MLP projections. "
            "Use all-linear to let PEFT target all linear layers."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument(
        "--lr_scheduler_kwargs",
        default=None,
        help='Optional JSON dict forwarded to TrainingArguments.lr_scheduler_kwargs, e.g. {"min_lr": 1e-6}.',
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", default="steps", choices=["no", "steps", "epoch", "best"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing_reentrant",
        type=lambda v: _str_to_bool(v, False),
        default=False,
        help="Passed to TrainingArguments.gradient_checkpointing_kwargs as use_reentrant.",
    )
    parser.add_argument("--bf16", type=lambda v: _str_to_bool(v, True), default=True)
    parser.add_argument("--fp16", type=lambda v: _str_to_bool(v, False), default=False)
    parser.add_argument("--tf32", type=lambda v: _str_to_bool(v, True), default=None)
    parser.add_argument(
        "--deepspeed",
        default=None,
        help="Optional DeepSpeed config JSON path for distributed full-parameter training.",
    )
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ddp_find_unused_parameters", type=lambda v: _str_to_bool(v, False), default=None)
    parser.add_argument("--ddp_timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--train_sampler", default="random", choices=["random", "sequential"])
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--skip_save_model", action="store_true", help="Skip final model/tokenizer save; useful for smoke tests.")
    parser.add_argument("--dry_run", action="store_true", help="Build datasets and exit before loading the model.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.train_profile == "agent_policy_reasoning" and args.enable_thinking_template is False:
        raise SystemExit(
            "--enable_thinking_template false is incompatible with "
            "--train_profile agent_policy_reasoning. Use --train_profile content_only "
            "to disable thinking supervision."
        )
    if not any(
        [
            args.train_file,
            args.agent_policy_file,
            args.context_compression_file,
            args.content_reader_file,
        ]
    ):
        raise SystemExit(
            "Provide --train_file or at least one module file: "
            "--agent_policy_file/--context_compression_file/--content_reader_file."
        )
    train_rows = _load_rows(args, split="train")
    eval_rows = _load_rows(args, split="eval")
    if args.max_train_examples > 0:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_eval_examples > 0:
        eval_rows = eval_rows[: args.max_eval_examples]
    if not eval_rows and args.validation_split_ratio > 0:
        train_rows, eval_rows = split_by_episode(
            train_rows,
            eval_ratio=args.validation_split_ratio,
            seed=args.seed,
        )
    print(f"[fire-agent-sft-train] raw rows: train={len(train_rows)} eval={len(eval_rows)}")

    tokenizer = _load_tokenizer(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if args.data_format == "slime_episode_rows":
        if args.train_sampler != "sequential":
            print(
                "[fire-agent-sft-train] warning: slime_episode_rows should normally use "
                "--train_sampler sequential to preserve episode scheduling."
            )
        train_features = _build_episode_features(train_rows, tokenizer, args, name="train")
        mixture = _parse_mixture(args.mixture)
        if mixture:
            print(
                "[fire-agent-sft-train] warning: ignoring --mixture for slime_episode_rows "
                "because mixture sampling would break episode ordering."
            )
        eval_dataset = (
            EventSFTDataset(_build_episode_features(eval_rows, tokenizer, args, name="eval"))
            if eval_rows
            else None
        )
    else:
        train_features = _build_features_for_rows(train_rows, tokenizer, args, name="train")
        mixture = _parse_mixture(args.mixture)
        if mixture:
            print(f"[fire-agent-sft-train] requested mixture: {json.dumps(mixture, ensure_ascii=False)}")
            train_features = _sample_features(
                train_features,
                mixture=mixture,
                total_examples=args.mixture_total_examples,
                seed=args.seed,
            )
            print(
                "[fire-agent-sft-train] train module counts after mixture: "
                f"{json.dumps(_module_counts(train_features), ensure_ascii=False)}"
            )
        eval_dataset = (
            EventSFTDataset(_build_features_for_rows(eval_rows, tokenizer, args, name="eval"))
            if eval_rows
            else None
        )
    train_dataset = EventSFTDataset(train_features)

    if args.dry_run:
        print("[fire-agent-sft-train] dry run complete; model was not loaded.")
        return

    model = _load_model(args)

    from transformers import Trainer, TrainingArguments
    TrainerClass = _trainer_class(Trainer, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_scheduler_kwargs = None
    if args.lr_scheduler_kwargs:
        try:
            lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--lr_scheduler_kwargs must be valid JSON: {exc}") from exc
        if not isinstance(lr_scheduler_kwargs, dict):
            raise SystemExit("--lr_scheduler_kwargs must decode to a JSON object.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": args.gradient_checkpointing_reentrant}
            if args.gradient_checkpointing
            else None
        ),
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        deepspeed=args.deepspeed,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_timeout=args.ddp_timeout,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
    )
    collator = CompletionOnlyCollator(tokenizer, pad_to_multiple_of=args.pad_to_multiple_of)
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    if args.skip_save_model:
        print(f"[fire-agent-sft-train] skip_save_model=true; not saving model/tokenizer to {output_dir}")
    else:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"[fire-agent-sft-train] saved model/tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
