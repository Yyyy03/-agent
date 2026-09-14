from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests


DEFAULT_DATASET_URL = "https://huggingface.co/datasets/OpenFinArena/FinDocResearch/resolve/main/dataset.csv"

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
    return project_root() / "official_benchmarks" / "findocresearch" / "data"


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


def download_dataset_csv(url: str, output_path: Path, overwrite: bool = False) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


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


def build_report_index(reports_dir: Optional[Path]) -> Dict[str, str]:
    if not reports_dir:
        return {}
    reports_dir = reports_dir.expanduser()
    if not reports_dir.exists():
        raise FileNotFoundError(f"FinDocResearch reports directory does not exist: {reports_dir}")
    index: Dict[str, str] = {}
    for path in reports_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, str(path))
    return index


def local_report_path(file_name: str, report_index: Dict[str, str]) -> str:
    if not file_name:
        return ""
    path = Path(file_name).expanduser()
    if path.exists() and path.is_file():
        return str(path)
    return report_index.get(Path(file_name).name, "")


def build_record(row: JsonRow, source_index: int, report_index: Optional[Dict[str, str]] = None) -> JsonRow:
    report_index = report_index or {}
    case_id = first_present(row, ["Case ID", "case_id", "id"], f"case_{source_index:03d}")
    company = first_present(row, ["Company Name", "company", "company_name"])
    market = first_present(row, ["Market", "market", "stock_market"])
    language = first_present(row, ["Output Report Language", "output_report_language", "language"])
    fy23_file = first_present(row, ["FY23 File Name", "FY2023 File Name", "fy23_file_name", "fy2023_file_name"])
    fy23_path = first_present(row, ["FY23 PDF Path", "FY2023 PDF Path", "fy23_pdf_path", "fy2023_pdf_path"])
    fy23_url = first_present(
        row,
        [
            "FY23 PDF Download Site",
            "FY2023 PDF Download Site",
            "FY23 annual report URL",
            "FY2023 annual report URL",
            "fy23_pdf_download_site",
            "fy2023_pdf_download_site",
            "fy23_url",
            "fy2023_url",
        ],
    )
    fy24_file = first_present(row, ["FY24 File Name", "FY2024 File Name", "fy24_file_name", "fy2024_file_name"])
    fy24_path = first_present(row, ["FY24 PDF Path", "FY2024 PDF Path", "fy24_pdf_path", "fy2024_pdf_path"])
    fy24_url = first_present(
        row,
        [
            "FY24 PDF Download Site",
            "FY2024 PDF Download Site",
            "FY24 annual report URL",
            "FY2024 annual report URL",
            "fy24_pdf_download_site",
            "fy2024_pdf_download_site",
            "fy24_url",
            "fy2024_url",
        ],
    )
    fy23_path = fy23_path or local_report_path(fy23_file, report_index)
    fy24_path = fy24_path or local_report_path(fy24_file, report_index)

    missing = [
        name
        for name, value in {
            "case_id": case_id,
            "company": company,
            "market": market,
            "output_report_language": language,
            "fy23_report": fy23_path or fy23_url or fy23_file,
            "fy24_report": fy24_path or fy24_url or fy24_file,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"FinDocResearch CSV row {source_index} is missing required fields: {missing}")

    return {
        "source_dataset": "OpenFinArena/FinDocResearch",
        "source_row_index": source_index,
        "sample_id": case_id,
        "case_id": case_id,
        "company": company,
        "company_name": company,
        "market": market,
        "Output Report Language": language,
        "output_report_language": language,
        "FY23 File Name": fy23_file,
        "FY23 PDF Path": fy23_path,
        "FY23 PDF Download Site": fy23_url,
        "FY24 File Name": fy24_file,
        "FY24 PDF Path": fy24_path,
        "FY24 PDF Download Site": fy24_url,
        "question": (
            "Generate a rigorous structured FinDocResearch report from the provided FY2024 and FY2023 annual "
            f"reports for {company} in {market}. Write in {language}."
        ),
    }


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else default_data_dir()
    dataset_csv = Path(args.dataset_csv) if args.dataset_csv else output_dir / "dataset.csv"
    if not args.dataset_csv:
        dataset_csv = download_dataset_csv(args.dataset_url, dataset_csv, overwrite=args.overwrite_csv)

    raw_rows = read_csv_rows(dataset_csv)
    if args.sample_size is not None:
        raw_rows = raw_rows[: args.sample_size]

    report_index = build_report_index(Path(args.reports_dir)) if args.reports_dir else {}
    rows = [build_record(row, idx, report_index=report_index) for idx, row in enumerate(raw_rows)]
    output_file = output_dir / args.output_name
    count = write_jsonl(output_file, rows)

    if args.benchmarks_dir:
        benchmark_path = Path(args.benchmarks_dir) / args.output_name
        write_jsonl(benchmark_path, rows)
        print(f"Wrote benchmark copy: {benchmark_path}")

    print(f"Wrote {count} FinDocResearch records: {output_file}")
    print(f"Source CSV: {dataset_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official OpenFinArena FinDocResearch dataset.csv for FIRE Agent.")
    parser.add_argument("--dataset_csv", default=None, help="Local OpenFinArena FinDocResearch dataset.csv.")
    parser.add_argument("--dataset_url", default=DEFAULT_DATASET_URL, help="Dataset CSV URL used when --dataset_csv is absent.")
    parser.add_argument("--overwrite_csv", action="store_true", help="Redownload dataset.csv when using --dataset_url.")
    parser.add_argument("--output_dir", default=None, help="Directory for dataset.csv and findocresearch.jsonl.")
    parser.add_argument("--output_name", default="findocresearch.jsonl", help="Prepared FIRE input filename.")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional first-N sample cap for smoke tests.")
    parser.add_argument("--benchmarks_dir", default=None, help="Optionally copy prepared JSONL into shared benchmarks dir.")
    parser.add_argument(
        "--reports_dir",
        default=None,
        help="Optional local directory containing FY2023/FY2024 annual-report PDFs. Matching paths are embedded in the JSONL.",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
