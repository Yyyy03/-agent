from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]
MODULES = {"agent_policy", "context_compression", "content_reader"}


@dataclass
class BuildStats:
    examples_seen: int = 0
    examples_kept: int = 0
    dropped_bad_shape: int = 0
    dropped_missing_reasoning: int = 0
    dropped_empty_completion: int = 0
    dropped_overlength: int = 0
    truncated_prompt: int = 0
    max_prompt_tokens: int = 0
    max_completion_tokens: int = 0
    max_total_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "examples_seen": self.examples_seen,
            "examples_kept": self.examples_kept,
            "dropped_bad_shape": self.dropped_bad_shape,
            "dropped_missing_reasoning": self.dropped_missing_reasoning,
            "dropped_empty_completion": self.dropped_empty_completion,
            "dropped_overlength": self.dropped_overlength,
            "truncated_prompt": self.truncated_prompt,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_total_tokens": self.max_total_tokens,
        }


class EventSFTDataset:
    """Map-style dataset for completion-only SFT over exported FIRE turn rows."""

    def __init__(self, features: List[JsonDict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> JsonDict:
        return self.features[index]


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                row.setdefault("_source_path", str(path))
                row.setdefault("_source_line", line_number)
                yield row


def read_event_rows(paths: Sequence[Path]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _metadata(row: JsonDict) -> JsonDict:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _without_step_loss_mask(message: JsonDict, *, keep_reasoning: bool) -> JsonDict:
    cleaned = dict(message)
    cleaned.pop("step_loss_mask", None)
    if not keep_reasoning:
        cleaned.pop("reasoning_content", None)
    return cleaned


def convert_slime_messages_row(
    row: JsonDict,
    *,
    default_module: str = "agent_policy",
) -> Optional[JsonDict]:
    """Convert one Slime SFT row into the event-row shape used by this trainer.

    Slime rows exported by ``export_slime_qwen35_sft`` store prompt and target in
    one ``messages`` list. Historical messages have ``step_loss_mask=0`` and the
    current assistant target has ``step_loss_mask=1``. The HF/PEFT trainer builds
    completion-only labels from an explicit ``messages`` prompt plus
    ``output_message`` target, so this adapter performs that split.
    """

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    supervised_indices: List[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        try:
            step_loss_mask = int(message.get("step_loss_mask", 0) or 0)
        except Exception:
            step_loss_mask = 0
        if step_loss_mask == 1:
            supervised_indices.append(index)

    if len(supervised_indices) != 1:
        return None
    target_index = supervised_indices[0]
    if target_index >= len(messages) - 1:
        trailing_messages: List[JsonDict] = []
    else:
        trailing_messages = messages[target_index + 1 :]
    if trailing_messages:
        return None

    target = messages[target_index]
    if str(target.get("role") or "") != "assistant":
        return None

    metadata = _metadata(row)
    converted: JsonDict = {
        "messages": [
            _without_step_loss_mask(message, keep_reasoning=False)
            for message in messages[:target_index]
        ],
        "output_message": _without_step_loss_mask(target, keep_reasoning=True),
        "tools": row.get("tools") if isinstance(row.get("tools"), list) else [],
        "module": str(row.get("module") or metadata.get("module") or default_module),
        "target_type": str(row.get("target_type") or metadata.get("target_type") or ""),
        "episode_id": row.get("episode_id") or metadata.get("episode_id"),
        "turn_id": row.get("turn_id") or metadata.get("turn_id"),
        "step_number": row.get("step_number") or metadata.get("step_number"),
    }
    if row.get("_source_path") is not None:
        converted["_source_path"] = row["_source_path"]
    if row.get("_source_line") is not None:
        converted["_source_line"] = row["_source_line"]
    return converted


def split_by_episode(
    rows: Sequence[JsonDict],
    *,
    eval_ratio: float,
    seed: int,
) -> Tuple[List[JsonDict], List[JsonDict]]:
    if eval_ratio <= 0:
        return list(rows), []
    by_episode: Dict[str, List[JsonDict]] = {}
    for row in rows:
        episode = str(row.get("episode_id") or "unknown")
        by_episode.setdefault(episode, []).append(row)
    episodes = list(by_episode)
    rng = random.Random(seed)
    rng.shuffle(episodes)
    if len(episodes) <= 1:
        return list(rows), []
    eval_count = max(1, int(round(len(episodes) * eval_ratio)))
    eval_count = min(eval_count, len(episodes) - 1)
    eval_episodes = set(episodes[:eval_count])
    train_rows: List[JsonDict] = []
    eval_rows: List[JsonDict] = []
    for episode in episodes:
        target = eval_rows if episode in eval_episodes else train_rows
        target.extend(by_episode[episode])
    return train_rows, eval_rows


def normalize_messages(value: Any) -> Optional[List[JsonDict]]:
    if not isinstance(value, list) or not value:
        return None
    messages: List[JsonDict] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").strip()
        has_tool_calls = isinstance(item.get("tool_calls"), list) and bool(item.get("tool_calls"))
        content = item.get("content", "" if has_tool_calls else None)
        if not role or content is None:
            return None
        content_text = content if isinstance(content, str) else str(content)
        if role == "assistant":
            message = {"role": "assistant", "content": content_text}
            if has_tool_calls:
                message["tool_calls"] = item["tool_calls"]
        elif role == "tool":
            tool_call_id = str(item.get("tool_call_id") or "").strip()
            if tool_call_id:
                message = {"role": "tool", "tool_call_id": tool_call_id, "content": content_text}
            else:
                message = {"role": "user", "content": "[tool_observation_without_call_id]\n" + content_text}
        elif role in {"system", "user"}:
            message = {"role": role, "content": content_text}
        else:
            message = {"role": "user", "content": f"[{role}]\n{content_text}"}
        messages.append(message)
    return messages


def row_module(row: JsonDict) -> str:
    return str(row.get("module") or "").strip()


def target_output_message(row: JsonDict) -> Optional[JsonDict]:
    output_message = row.get("output_message")
    return output_message if isinstance(output_message, dict) else None


def get_target_mode(module: str, train_profile: str) -> str:
    if train_profile == "content_only":
        return "content_only"
    if train_profile == "agent_policy_reasoning":
        if module == "agent_policy":
            return "reasoning_and_content"
        return "content_only"
    raise ValueError(f"Unknown train_profile: {train_profile}")


def _reasoning_text(response: JsonDict) -> str:
    reasoning = response.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else str(reasoning or "")


def _content_text(response: JsonDict) -> str:
    content = response.get("content")
    return content if isinstance(content, str) else str(content or "")


def _qwen_reasoning_content(
    *,
    reasoning_text: str,
    content_text: str,
    reasoning_format: str,
    prompt_text: str,
) -> str:
    reasoning = reasoning_text.strip()
    if not reasoning:
        return content_text
    prompt_prefills_open_think = prompt_text.rstrip().endswith("<think>")
    if reasoning_format == "qwen_auto_think_tags":
        if prompt_prefills_open_think:
            return f"{reasoning}\n</think>\n{content_text}"
        return f"<think>\n{reasoning}\n</think>\n{content_text}"
    if reasoning_format == "qwen_think_tags":
        return f"<think>\n{reasoning}\n</think>\n{content_text}"
    if reasoning_format == "qwen_prefilled_think":
        return f"{reasoning}\n</think>\n{content_text}"
    return content_text


def prepare_output_message_for_template(
    output_message: JsonDict,
    *,
    target_mode: str,
    reasoning_format: str,
    require_reasoning: bool,
    prompt_text: str,
) -> Tuple[Optional[JsonDict], str]:
    reasoning = _reasoning_text(output_message)
    content_text = _content_text(output_message)
    output: JsonDict = {
        "role": str(output_message.get("role") or "assistant"),
        "content": content_text,
    }
    tool_calls = output_message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        output["tool_calls"] = tool_calls
    if reasoning.strip():
        output["reasoning_content"] = reasoning
    if target_mode == "reasoning_and_content":
        if require_reasoning and not reasoning.strip():
            return None, "missing_reasoning"
        if reasoning.strip() and reasoning_format.startswith("qwen_"):
            output["content"] = _qwen_reasoning_content(
                reasoning_text=reasoning,
                content_text=content_text,
                reasoning_format=reasoning_format,
                prompt_text=prompt_text,
            )
        elif reasoning.strip() and require_reasoning:
            return None, "bad_shape"
    output.pop("reasoning_content", None)
    return output, ""


def apply_chat_template_text(
    tokenizer: Any,
    messages: List[JsonDict],
    *,
    tools: Optional[List[JsonDict]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
    require_chat_template: bool = True,
) -> str:
    if require_chat_template and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat_template. Use the target Qwen tokenizer or pass "
            "--require_chat_template false only for debugging."
        )
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        message = str(exc)
        can_retry_without_tools = bool(tools) and (
            "tools" in message
            or "unexpected keyword" in message
            or "unexpected argument" in message
        )
        if can_retry_without_tools:
            if require_chat_template:
                raise ValueError(
                    "tokenizer.apply_chat_template does not accept tools, but this sample has "
                    "tool schemas. Fix the target Qwen chat_template instead of silently "
                    "dropping tools."
                ) from exc
            kwargs.pop("tools", None)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as retry_exc:
                if require_chat_template:
                    raise ValueError(
                        "tokenizer.apply_chat_template failed after retrying without "
                        "tools. Fix the tokenizer/chat_template first."
                    ) from retry_exc
        can_retry_without_thinking = (
            enable_thinking is not None
            and (
                "enable_thinking" in message
                or "unexpected keyword" in message
                or "unexpected argument" in message
            )
        )
        if can_retry_without_thinking:
            kwargs.pop("enable_thinking", None)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as retry_exc:
                if require_chat_template:
                    raise ValueError(
                        "tokenizer.apply_chat_template failed after retrying without "
                        "enable_thinking. Fix the tokenizer/chat_template first."
                    ) from retry_exc
        elif require_chat_template:
            raise ValueError(
                "tokenizer.apply_chat_template failed. Do not silently train with a non-Qwen "
                "fallback template; fix the tokenizer/chat_template first."
            ) from exc
    except Exception as exc:
        if require_chat_template:
            raise ValueError(
                "tokenizer.apply_chat_template failed. Do not silently train with a non-Qwen "
                "fallback template; fix the tokenizer/chat_template first."
            ) from exc
    if require_chat_template:
        raise ValueError(
            "tokenizer.apply_chat_template failed. Do not silently train with a non-Qwen "
            "fallback template; fix the tokenizer/chat_template first."
        )
    chunks: List[str] = []
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def render_output_message_completion(
    tokenizer: Any,
    *,
    prompt_text: str,
    input_messages: List[JsonDict],
    output_message: JsonDict,
    tools: Optional[List[JsonDict]],
    enable_thinking: Optional[bool],
    target_mode: str,
    reasoning_format: str,
    require_reasoning: bool,
    require_chat_template: bool,
) -> Tuple[Optional[str], str]:
    if not isinstance(output_message, dict):
        return None, "bad_shape"
    output, reason = prepare_output_message_for_template(
        output_message,
        target_mode=target_mode,
        reasoning_format=reasoning_format,
        require_reasoning=require_reasoning,
        prompt_text=prompt_text,
    )
    if output is None:
        return None, reason
    full_text = apply_chat_template_text(
        tokenizer,
        input_messages + [output],
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        require_chat_template=require_chat_template,
    )
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :], ""
    return None, "bad_shape"


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(token_id) for token_id in ids]


def build_features(
    rows: Sequence[JsonDict],
    tokenizer: Any,
    *,
    max_seq_length: int,
    train_profile: str,
    reasoning_format: str,
    require_reasoning: bool,
    on_overlength: str,
    add_eos: bool,
    enable_thinking_template: Optional[bool],
    require_chat_template: bool,
) -> Tuple[List[JsonDict], BuildStats]:
    stats = BuildStats()
    features: List[JsonDict] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for row in rows:
        stats.examples_seen += 1
        messages = normalize_messages(row.get("messages"))
        output_message = target_output_message(row)
        if messages is None or output_message is None:
            stats.dropped_bad_shape += 1
            continue
        module = row_module(row)
        if module not in MODULES:
            stats.dropped_bad_shape += 1
            continue
        target_mode = get_target_mode(module, train_profile)
        tools = row.get("tools") if isinstance(row.get("tools"), list) else None
        template_enable_thinking = (
            enable_thinking_template
            if enable_thinking_template is not None
            else target_mode == "reasoning_and_content"
        )
        prompt_text = apply_chat_template_text(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=template_enable_thinking,
            require_chat_template=require_chat_template,
        )
        completion_text, reason = render_output_message_completion(
            tokenizer,
            prompt_text=prompt_text,
            input_messages=messages,
            output_message=output_message,
            tools=tools,
            enable_thinking=template_enable_thinking,
            target_mode=target_mode,
            reasoning_format=reasoning_format,
            require_reasoning=require_reasoning,
            require_chat_template=require_chat_template,
        )
        if completion_text is None:
            if reason == "missing_reasoning":
                stats.dropped_missing_reasoning += 1
            elif reason == "empty_completion":
                stats.dropped_empty_completion += 1
            else:
                stats.dropped_bad_shape += 1
            continue
        prompt_ids = _tokenize_text(tokenizer, prompt_text)
        full_ids = _tokenize_text(tokenizer, prompt_text + completion_text)
        if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
            input_ids = full_ids
            completion_ids = full_ids[len(prompt_ids) :]
        else:
            # Some tokenizers can merge across the prompt/completion boundary.
            # FIRE prompts normally end at a chat-template generation boundary; if
            # that invariant is broken, keep the supervised target exact instead.
            completion_ids = _tokenize_text(tokenizer, completion_text)
            input_ids = prompt_ids + completion_ids
        if add_eos and eos_token_id is not None:
            completion_ids = completion_ids + [int(eos_token_id)]
            input_ids = input_ids + [int(eos_token_id)]
        total_len = len(input_ids)
        stats.max_prompt_tokens = max(stats.max_prompt_tokens, len(prompt_ids))
        stats.max_completion_tokens = max(stats.max_completion_tokens, len(completion_ids))
        stats.max_total_tokens = max(stats.max_total_tokens, total_len)
        if total_len > max_seq_length:
            if on_overlength == "drop":
                stats.dropped_overlength += 1
                continue
            if on_overlength == "error":
                raise ValueError(
                    f"Overlength sample {row.get('episode_id')}#{row.get('turn_id')}: "
                    f"{total_len} > {max_seq_length}"
                )
            if on_overlength == "truncate_left_prompt":
                keep_prompt = max_seq_length - len(completion_ids)
                if keep_prompt <= 0:
                    stats.dropped_overlength += 1
                    continue
                prompt_ids = prompt_ids[-keep_prompt:]
                input_ids = prompt_ids + completion_ids
                stats.truncated_prompt += 1
            else:
                raise ValueError(f"Unsupported on_overlength policy: {on_overlength}")
        labels = [-100] * len(prompt_ids) + completion_ids
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "episode_id": row.get("episode_id"),
                "turn_id": row.get("turn_id"),
                "module": module,
                "target_type": row.get("target_type"),
                "step_number": row.get("step_number"),
                "target_mode": target_mode,
            }
        )
        stats.examples_kept += 1
    return features, stats


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any, *, pad_to_multiple_of: Optional[int] = None):
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else eos_token_id if eos_token_id is not None else 0)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: Sequence[JsonDict]) -> JsonDict:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if max_len % multiple:
                max_len = ((max_len // multiple) + 1) * multiple
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for item in batch:
            ids = list(item["input_ids"])
            mask = list(item["attention_mask"])
            item_labels = list(item["labels"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(item_labels + [-100] * pad_len)

        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
