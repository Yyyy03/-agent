from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
    return project_root() / "official_benchmarks" / "rfc_bench" / "data"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_present(row: JsonRow, keys: Sequence[str], default: str = "") -> str:
    normalized_lookup = {key.strip().lower(): key for key in row}
    for key in keys:
        if key in row and normalize_text(row[key]):
            return normalize_text(row[key])
        normalized_key = normalized_lookup.get(key.strip().lower())
        if normalized_key and normalize_text(row[normalized_key]):
            return normalize_text(row[normalized_key])
    return default


def read_csv_rows(path: Path) -> List[JsonRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_jsonl(path: Path, rows: Iterable[JsonRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def perturbation_type_from_file(path: Path) -> str:
    name = path.name.lower()
    for label in ("causal", "flipping", "numerical", "sentiment"):
        if name.startswith(label):
            return label
    return path.stem.split("_", 1)[0].lower()


def split_from_file(path: Path) -> str:
    parent = path.parent.name.lower()
    if "hard" in parent:
        return "hard"
    if "gold" in parent:
        return "gold"
    return parent


def build_question(perturbed_text: str, ticker: str, date: str, link: str, title: str) -> str:
    metadata = []
    if ticker:
        metadata.append(f"Ticker: {ticker}")
    if date:
        metadata.append(f"Date: {date}")
    if title:
        metadata.append(f"Title: {title}")
    if link:
        metadata.append(f"Source link: {link}")
    metadata_block = "\n".join(metadata)
    if metadata_block:
        metadata_block += "\n\n"
    return (
        "RFC-BENCH Task 1: reference-free counterfactual financial misinformation detection.\n"
        "Given one financial news paragraph without the original article or external evidence, "
        "judge whether the paragraph is factual/original or manipulated/misinformation.\n"
        "Return only one label: factual or manipulated.\n\n"
        f"{metadata_block}"
        f"Paragraph:\n{perturbed_text}"
    )


def build_record(path: Path, source_row_index: int, row: JsonRow) -> JsonRow:
    split = split_from_file(path)
    perturbation_type = perturbation_type_from_file(path)
    perturbed_text = first_present(row, ["Perturbed", "perturbed"])
    if not perturbed_text:
        raise ValueError(f"Missing Perturbed text in {path}:{source_row_index + 2}")
    ticker = first_present(row, ["Ticker", "ticker"])
    date = first_present(row, ["Date", "date"])
    link = first_present(row, ["Link", "link", "URL", "url"])
    title = first_present(row, ["Title", "title"])
    sample_id = f"{split}_{perturbation_type}_{source_row_index:04d}"
    return {
        "source_dataset": "RFC-BENCH",
        "source_file": str(path.relative_to(default_data_dir()) if path.is_relative_to(default_data_dir()) else path),
        "source_row_index": source_row_index,
        "sample_id": sample_id,
        "split": split,
        "perturbation_type": perturbation_type,
        "is_manipulated": True,
        "question": build_question(perturbed_text, ticker=ticker, date=date, link=link, title=title),
        "answer": "manipulated",
        "ticker": ticker,
        "date": date,
        "title": title,
        "link": link,
        "perturbed_text": perturbed_text,
    }


def source_files(data_dir: Path, splits: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    split_dirs = {
        "gold": data_dir / "gold-label-set",
        "hard": data_dir / "hard-case-set",
    }
    for split in splits:
        split_dir = split_dirs[split]
        if not split_dir.is_dir():
            raise FileNotFoundError(f"RFC-BENCH split directory not found: {split_dir}")
        files.extend(sorted(path for path in split_dir.glob("*.csv") if path.name.lower() != "readme.md"))
    return files


def prepare(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    splits = ["gold"] if args.gold_only else ["gold", "hard"]
    rows: List[JsonRow] = []
    for path in source_files(data_dir, splits):
        for index, row in enumerate(read_csv_rows(path)):
            rows.append(build_record(path, index, row))
    if args.sample_size is not None:
        rows = rows[: args.sample_size]

    output_path = data_dir / args.output_name
    count = write_jsonl(output_path, rows)
    print(f"Wrote {count} RFC-BENCH records: {output_path}")

    if args.benchmarks_dir:
        benchmark_path = Path(args.benchmarks_dir) / args.output_name
        write_jsonl(benchmark_path, rows)
        print(f"Wrote benchmark copy: {benchmark_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RFC-BENCH CSV files for FIRE Agent.")
    parser.add_argument("--data_dir", default=None, help="Local RFC-BENCH data directory.")
    parser.add_argument("--output_name", default="rfc_bench.jsonl", help="Prepared FIRE input filename.")
    parser.add_argument("--benchmarks_dir", default=None, help="Optionally copy prepared JSONL into shared benchmarks dir.")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional first-N sample cap for smoke tests.")
    parser.add_argument("--gold_only", action="store_true", help="Only include gold-label-set rows, excluding hard-case-set.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
