from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence


JsonDict = dict[str, Any]

DATA_LAYER_REFERENCE = [
    {
        "name": "A_full_sound_agent",
        "training_action": "record_only",
        "description": "Answer and agent trajectory are both sound; use as normal full agent SFT data.",
    },
    {
        "name": "B_partial_supported_agent",
        "training_action": "record_only",
        "description": "Trajectory is mostly useful but only partially supports the final answer.",
    },
    {
        "name": "C_recovery_agent",
        "training_action": "record_only",
        "description": "The trajectory includes a wrong direction or failure that is explicitly corrected later.",
    },
    {
        "name": "D_local_prior_step_agent",
        "training_action": "record_only",
        "description": "A local step uses model prior knowledge or planning without requiring a fresh tool lookup.",
    },
    {
        "name": "E_unsound_or_contradictory",
        "training_action": "record_only",
        "description": "The answer may be correct, but the trajectory is unsupported or contradictory.",
    },
]


def _read_jsonl(path: Path) -> Iterable[tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, row


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _metadata(row: JsonDict) -> JsonDict:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _episode_id(row: JsonDict, *, source_path: Path, line_number: int) -> str:
    metadata = _metadata(row)
    value = metadata.get("episode_id") or row.get("episode_id")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{source_path}:{line_number}: missing metadata.episode_id")
    return str(value)


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return default


def _step_sort_key(row: JsonDict, source_position: int) -> tuple[int, int, int]:
    metadata = _metadata(row)
    turn_index = _as_int(metadata.get("turn_index"), source_position)
    step_number = _as_int(metadata.get("step_number"), turn_index)
    return (turn_index, step_number, source_position)


def _clean_row(row: JsonDict) -> JsonDict:
    cleaned = dict(row)
    cleaned.pop("_episode_view_source_position", None)
    return cleaned


def _summarize_episode(rows: Sequence[JsonDict]) -> JsonDict:
    target_counts: Counter[str] = Counter()
    token_counts: list[int] = []
    loss_token_counts: list[int] = []
    qualities: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in rows:
        metadata = _metadata(row)
        target_counts[str(metadata.get("target_type") or "unknown")] += 1
        if metadata.get("token_count") is not None:
            token_counts.append(_as_int(metadata.get("token_count"), 0))
        if metadata.get("loss_token_count") is not None:
            loss_token_counts.append(_as_int(metadata.get("loss_token_count"), 0))
        if metadata.get("sft_quality") is not None:
            qualities[str(metadata.get("sft_quality"))] += 1
        if metadata.get("sft_source_run") is not None:
            sources[str(metadata.get("sft_source_run"))] += 1
    return {
        "num_steps": len(rows),
        "target_type_counts": dict(target_counts),
        "max_token_count": max(token_counts) if token_counts else None,
        "total_loss_token_count": sum(loss_token_counts) if loss_token_counts else None,
        "sft_quality_counts": dict(qualities),
        "sft_source_run_counts": dict(sources),
    }


def _quantiles(values: Sequence[int]) -> JsonDict:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * p) - 1))]

    return {
        "min": ordered[0],
        "p50": q(0.5),
        "p80": q(0.8),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def build_episode_rows(
    source_path: Path,
    *,
    shuffle_episodes: bool,
    seed: int,
    sort_within_episode: bool,
) -> tuple[list[JsonDict], JsonDict]:
    episodes: OrderedDict[str, list[JsonDict]] = OrderedDict()
    row_count = 0
    for line_number, row in _read_jsonl(source_path):
        episode_id = _episode_id(row, source_path=source_path, line_number=line_number)
        copied = dict(row)
        copied["_episode_view_source_position"] = row_count
        episodes.setdefault(episode_id, []).append(copied)
        row_count += 1

    episode_ids = list(episodes)
    if shuffle_episodes:
        rng = random.Random(seed)
        rng.shuffle(episode_ids)

    output_rows: list[JsonDict] = []
    episode_lengths: list[int] = []
    episode_max_tokens: list[int] = []
    target_counts: Counter[str] = Counter()
    for ordinal, episode_id in enumerate(episode_ids):
        rows = episodes[episode_id]
        if sort_within_episode:
            rows = sorted(
                rows,
                key=lambda item: _step_sort_key(item, _as_int(item.get("_episode_view_source_position"), 0)),
            )
        cleaned_rows = [_clean_row(row) for row in rows]
        summary = _summarize_episode(cleaned_rows)
        episode_lengths.append(int(summary["num_steps"]))
        if summary.get("max_token_count") is not None:
            episode_max_tokens.append(int(summary["max_token_count"]))
        target_counts.update(summary["target_type_counts"])
        output_rows.append(
            {
                "episode_view_schema": "fire_agent.slime_episode_view.v1",
                "episode_id": episode_id,
                "rows": cleaned_rows,
                "metadata": {
                    "episode_id": episode_id,
                    "episode_ordinal": ordinal,
                    "source_path": str(source_path),
                    **summary,
                },
            }
        )

    stats = {
        "source": str(source_path),
        "rows": row_count,
        "episodes": len(output_rows),
        "target_type_counts": dict(target_counts),
        "episode_length_summary": _quantiles(episode_lengths),
        "episode_max_token_summary": _quantiles(episode_max_tokens),
        "episodes_with_fewer_than_3_steps": sum(1 for value in episode_lengths if value < 3),
        "shuffle_episodes": shuffle_episodes,
        "sort_within_episode": sort_within_episode,
        "seed": seed,
    }
    return output_rows, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an episode-level Slime SFT view from step-level Qwen3.5 Agent SFT rows. "
            "This preserves step rows and only changes scheduling granularity."
        )
    )
    parser.add_argument("--train", required=True, help="Step-level train.jsonl.")
    parser.add_argument("--eval", default=None, help="Optional step-level eval.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Directory for train/eval episode views and manifest.")
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_episodes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sort_within_episode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = Path(args.train).expanduser().resolve()
    eval_path = Path(args.eval).expanduser().resolve() if args.eval else None
    train_output = Path(args.train_output).expanduser().resolve() if args.train_output else output_dir / "train.episodes.jsonl"
    eval_output = Path(args.eval_output).expanduser().resolve() if args.eval_output else output_dir / "eval.episodes.jsonl"
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "episode_view_manifest.json"

    train_rows, train_stats = build_episode_rows(
        train_path,
        shuffle_episodes=args.shuffle_episodes,
        seed=args.seed,
        sort_within_episode=args.sort_within_episode,
    )
    written_train = _write_jsonl(train_output, train_rows)

    outputs: JsonDict = {"train": str(train_output)}
    stats: JsonDict = {"train": train_stats}
    written: JsonDict = {"train": written_train}
    if eval_path is not None and eval_path.exists():
        eval_rows, eval_stats = build_episode_rows(
            eval_path,
            shuffle_episodes=False,
            seed=args.seed,
            sort_within_episode=args.sort_within_episode,
        )
        written_eval = _write_jsonl(eval_output, eval_rows)
        outputs["eval"] = str(eval_output)
        stats["eval"] = eval_stats
        written["eval"] = written_eval

    manifest = {
        "episode_view_schema": "fire_agent.slime_episode_view.v1",
        "purpose": "episode-aware scheduling; row supervision content is unchanged",
        "outputs": outputs,
        "written": written,
        "stats": stats,
        "data_layers_reference": DATA_LAYER_REFERENCE,
        "settings": {
            "shuffle_episodes": args.shuffle_episodes,
            "sort_within_episode": args.sort_within_episode,
            "seed": args.seed,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
