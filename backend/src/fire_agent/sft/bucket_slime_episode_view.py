from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


JsonDict = dict[str, Any]


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


def _as_int(value: Any, default: int = 0) -> int:
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


def _metadata(row: JsonDict) -> JsonDict:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _episode_max_token_count(episode: JsonDict) -> int:
    metadata = _metadata(episode)
    if metadata.get("max_token_count") is not None:
        return _as_int(metadata.get("max_token_count"))
    values: list[int] = []
    for row in episode.get("rows") or []:
        row_metadata = _metadata(row)
        if row_metadata.get("token_count") is not None:
            values.append(_as_int(row_metadata.get("token_count")))
    return max(values) if values else 0


def _episode_loss_token_count(episode: JsonDict) -> int:
    metadata = _metadata(episode)
    if metadata.get("total_loss_token_count") is not None:
        return _as_int(metadata.get("total_loss_token_count"))
    total = 0
    for row in episode.get("rows") or []:
        row_metadata = _metadata(row)
        total += _as_int(row_metadata.get("loss_token_count"))
    return total


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


def _bucket_name(max_tokens: int, *, main_max_tokens: int, tail_min_tokens: int) -> str:
    if max_tokens <= main_max_tokens:
        return f"main_le_{main_max_tokens}"
    if max_tokens <= tail_min_tokens:
        return f"mid_{main_max_tokens + 1}_{tail_min_tokens}"
    return f"tail_gt_{tail_min_tokens}"


def _summarize(rows: Sequence[JsonDict]) -> JsonDict:
    max_tokens = [_episode_max_token_count(row) for row in rows]
    loss_tokens = [_episode_loss_token_count(row) for row in rows]
    step_counts = [len(row.get("rows") or []) for row in rows]
    target_counts: Counter[str] = Counter()
    for episode in rows:
        for row in episode.get("rows") or []:
            target_counts[str(_metadata(row).get("target_type") or "unknown")] += 1
    return {
        "episodes": len(rows),
        "rows": sum(step_counts),
        "max_token_summary": _quantiles(max_tokens),
        "loss_token_summary": _quantiles(loss_tokens),
        "step_count_summary": _quantiles(step_counts),
        "target_type_counts": dict(target_counts),
    }


def split_episode_file(
    source: Path,
    output_dir: Path,
    *,
    split_name: str,
    main_max_tokens: int,
    tail_min_tokens: int,
) -> JsonDict:
    buckets: dict[str, list[JsonDict]] = {
        f"main_le_{main_max_tokens}": [],
        f"mid_{main_max_tokens + 1}_{tail_min_tokens}": [],
        f"tail_gt_{tail_min_tokens}": [],
    }
    for _, episode in _read_jsonl(source):
        max_tokens = _episode_max_token_count(episode)
        buckets[_bucket_name(max_tokens, main_max_tokens=main_max_tokens, tail_min_tokens=tail_min_tokens)].append(
            episode
        )

    split_manifest: JsonDict = {"source": str(source), "buckets": {}}
    for bucket, rows in buckets.items():
        bucket_path = output_dir / bucket / f"{split_name}.episodes.jsonl"
        written = _write_jsonl(bucket_path, rows)
        split_manifest["buckets"][bucket] = {
            "path": str(bucket_path),
            "written": written,
            "stats": _summarize(rows),
        }
    return split_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Slime episode-view SFT data into length buckets.")
    parser.add_argument("--train", required=True, help="Input train.episodes.jsonl.")
    parser.add_argument("--eval", default=None, help="Optional input eval.episodes.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Output directory containing per-bucket subdirectories.")
    parser.add_argument("--main_max_tokens", type=int, default=32768, help="Max episode step length for main bucket.")
    parser.add_argument("--tail_min_tokens", type=int, default=49152, help="Max episode step length for mid bucket.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_path = Path(args.train).expanduser().resolve()
    eval_path = Path(args.eval).expanduser().resolve() if args.eval else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.main_max_tokens >= args.tail_min_tokens:
        raise ValueError("--main_max_tokens must be smaller than --tail_min_tokens")

    manifest: JsonDict = {
        "schema": "fire_agent.slime_episode_length_buckets.v1",
        "purpose": "length-bucketed episode-aware SFT scheduling; row supervision content is unchanged",
        "settings": {
            "main_max_tokens": args.main_max_tokens,
            "tail_min_tokens": args.tail_min_tokens,
        },
        "splits": {
            "train": split_episode_file(
                train_path,
                output_dir,
                split_name="train",
                main_max_tokens=args.main_max_tokens,
                tail_min_tokens=args.tail_min_tokens,
            )
        },
    }
    if eval_path is not None and eval_path.exists():
        manifest["splits"]["eval"] = split_episode_file(
            eval_path,
            output_dir,
            split_name="eval",
            main_max_tokens=args.main_max_tokens,
            tail_min_tokens=args.tail_min_tokens,
        )

    manifest_path = output_dir / "bucket_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
