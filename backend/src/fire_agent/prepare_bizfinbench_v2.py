from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


JsonRow = Dict[str, Any]


def project_root() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd(), *file_path.parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "official_benchmarks").exists():
            return base
        nested = base / "fire_agent"
        if (nested / "pyproject.toml").exists() and (nested / "official_benchmarks").exists():
            return nested
    return file_path.parents[2]


def default_data_dir() -> Path:
    return project_root() / "official_benchmarks" / "bizfinbench_v2" / "data"


def write_jsonl(path: Path, rows: Iterable[JsonRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[JsonRow]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(content_to_text(item.get("text")))
                elif "content" in item:
                    parts.append(content_to_text(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(content_to_text(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return content_to_text(content.get("text"))
        if "content" in content:
            return content_to_text(content.get("content"))
    return json.dumps(content, ensure_ascii=False)


def messages_to_question(messages: Any) -> str:
    if not isinstance(messages, list):
        return content_to_text(messages)
    rendered: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            text = content_to_text(message)
            if text:
                rendered.append(text)
            continue
        role = str(message.get("role") or "").strip()
        text = content_to_text(message.get("content"))
        if not text:
            continue
        if len(messages) == 1 and role in {"", "user"}:
            rendered.append(text)
        else:
            rendered.append(f"{role or 'message'}:\n{text}")
    return "\n\n".join(rendered).strip()


def choices_to_answer(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            return content_to_text(message.get("content"))
        if "content" in first:
            return content_to_text(first.get("content"))
    return content_to_text(first)


def task_type_from_file(path: Path, language: str) -> str:
    stem = path.stem
    suffix = f"_{language}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def source_files(data_dir: Path, languages: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for language in languages:
        language_dir = data_dir / language
        if not language_dir.is_dir():
            raise FileNotFoundError(f"BizFinBench.v2 language directory not found: {language_dir}")
        files.extend(sorted(language_dir.glob("*.jsonl")))
    return files


def build_record(data_dir: Path, path: Path, source_row_index: int, row: JsonRow) -> JsonRow:
    language = path.parent.name
    task_type = task_type_from_file(path, language)
    question = messages_to_question(row.get("messages"))
    answer = choices_to_answer(row.get("choices"))
    if not question:
        raise ValueError(f"Missing messages/question at {path}:{source_row_index + 1}")
    sample_id = f"{language}_{task_type}_{source_row_index:06d}"
    return {
        "source_dataset": "BizFinBench.v2",
        "source_file": str(path.relative_to(data_dir)),
        "source_row_index": source_row_index,
        "sample_id": sample_id,
        "language": language,
        "task_type": task_type,
        "question": question,
        "answer": answer,
        "messages": row.get("messages"),
        "choices": row.get("choices"),
    }


def sample_rows(rows: List[JsonRow], sample_size: Optional[int], strategy: str, seed: int) -> List[JsonRow]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    if sample_size <= 0:
        return []
    if strategy == "first":
        return rows[:sample_size]
    if strategy != "stratified":
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[JsonRow]] = {}
    for row in rows:
        key = (str(row.get("language") or ""), str(row.get("task_type") or ""))
        buckets.setdefault(key, []).append(row)

    keys = sorted(buckets)
    base = sample_size // len(keys)
    remainder = sample_size % len(keys)
    selected: List[JsonRow] = []
    overflow: List[JsonRow] = []
    for index, key in enumerate(keys):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        quota = base + (1 if index < remainder else 0)
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        overflow.extend(bucket[take:])

    if len(selected) < sample_size and overflow:
        rng.shuffle(overflow)
        selected.extend(overflow[: sample_size - len(selected)])

    order = {str(row.get("sample_id")): index for index, row in enumerate(rows)}
    return sorted(selected, key=lambda row: order.get(str(row.get("sample_id")), 10**12))


def prepare(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    rows: List[JsonRow] = []
    for path in source_files(data_dir, languages):
        for index, row in enumerate(read_jsonl(path)):
            rows.append(build_record(data_dir, path, index, row))
    rows = sample_rows(rows, args.sample_size, args.sample_strategy, args.seed)

    output_path = data_dir / args.output_name
    count = write_jsonl(output_path, rows)
    print(f"Wrote {count} BizFinBench.v2 records: {output_path}")

    if args.benchmarks_dir:
        benchmark_path = Path(args.benchmarks_dir) / args.output_name
        write_jsonl(benchmark_path, rows)
        print(f"Wrote benchmark copy: {benchmark_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BizFinBench.v2 chat JSONL files for FIRE Agent.")
    parser.add_argument("--data_dir", default=None, help="Local BizFinBench.v2 data directory.")
    parser.add_argument("--output_name", default="bizfinbench_v2.jsonl", help="Prepared FIRE input filename.")
    parser.add_argument("--benchmarks_dir", default=None, help="Optionally copy prepared JSONL into shared benchmarks dir.")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional first-N sample cap for smoke tests.")
    parser.add_argument(
        "--sample_strategy",
        choices=["first", "stratified"],
        default="first",
        help="Sampling strategy when --sample_size is set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling.")
    parser.add_argument("--languages", default="cn,en", help="Comma-separated language directories to include.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
