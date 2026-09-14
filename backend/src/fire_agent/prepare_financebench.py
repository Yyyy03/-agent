from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


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
    return project_root() / "official_benchmarks" / "financebench" / "data"


def read_jsonl(path: Path) -> Iterator[JsonRow]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def write_jsonl(path: Path, rows: Iterable[JsonRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_pdf_index(pdfs_dir: Path) -> Dict[str, Path]:
    if not pdfs_dir.is_dir():
        raise FileNotFoundError(f"FinanceBench PDF directory not found: {pdfs_dir}")
    return {path.stem: path for path in pdfs_dir.glob("*.pdf") if path.is_file()}


def build_question(row: JsonRow) -> str:
    return str(row.get("question") or "").strip()


def build_record(row: JsonRow, source_row_index: int, pdf_index: Dict[str, Path]) -> JsonRow:
    doc_name = str(row.get("doc_name") or "").strip()
    pdf_path = pdf_index.get(doc_name)
    if pdf_path is None:
        raise FileNotFoundError(f"No local PDF found for FinanceBench doc_name={doc_name!r}")
    sample_id = str(row.get("financebench_id") or f"financebench_{source_row_index:04d}")
    return {
        "source_dataset": "FinanceBench",
        "source_row_index": source_row_index,
        "sample_id": sample_id,
        "financebench_id": row.get("financebench_id"),
        "company": row.get("company"),
        "doc_name": doc_name,
        "question_type": row.get("question_type"),
        "question_reasoning": row.get("question_reasoning"),
        "domain_question_num": row.get("domain_question_num"),
        "question": build_question(row),
        "raw_question": row.get("question"),
        "answer": row.get("answer"),
        "justification": row.get("justification"),
        "evidence": row.get("evidence"),
        "dataset_subset_label": row.get("dataset_subset_label"),
        "gics_sector": row.get("gics_sector"),
        "doc_type": row.get("doc_type"),
        "doc_period": row.get("doc_period"),
        "doc_link": row.get("doc_link"),
        "attachment_path": str(pdf_path),
    }


def prepare(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    input_path = Path(args.input_file) if args.input_file else data_dir / "financebench_merged.jsonl"
    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else data_dir / "pdfs"
    pdf_index = build_pdf_index(pdfs_dir)
    rows = [build_record(row, index, pdf_index) for index, row in enumerate(read_jsonl(input_path))]
    if args.sample_size is not None:
        rows = rows[: args.sample_size]

    output_path = data_dir / args.output_name
    count = write_jsonl(output_path, rows)
    print(f"Wrote {count} FinanceBench records: {output_path}")

    if args.benchmarks_dir:
        benchmark_path = Path(args.benchmarks_dir) / args.output_name
        write_jsonl(benchmark_path, rows)
        print(f"Wrote benchmark copy: {benchmark_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FinanceBench JSONL and local PDFs for FIRE Agent.")
    parser.add_argument("--data_dir", default=None, help="Local FinanceBench data directory.")
    parser.add_argument("--input_file", default=None, help="Source FinanceBench merged JSONL file.")
    parser.add_argument("--pdfs_dir", default=None, help="Directory containing local FinanceBench PDFs.")
    parser.add_argument("--output_name", default="financebench.jsonl", help="Prepared FIRE input filename.")
    parser.add_argument("--benchmarks_dir", default=None, help="Optionally copy prepared JSONL into shared benchmarks dir.")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional first-N sample cap for smoke tests.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
