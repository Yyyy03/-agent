from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

try:
    from .http_client import http_get
    from .constants import FINRPT_OFFICIAL_RESPONSE_FIELDS
except ImportError:  # pragma: no cover - keeps direct script execution usable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fire_agent.constants import FINRPT_OFFICIAL_RESPONSE_FIELDS
    from fire_agent.http_client import http_get


HF_DATASET = "jinsong8/FinRpt"
HF_FILES = {
    "zh": "FinRpt.jsonl",
    "en": "FinRpt_en.jsonl",
}

DEFAULT_SOURCE_FIELDS = (
    "company_info",
    "financials",
    "news",
    "report",
    "finance_write_prompt",
    "news_write_prompt",
    "report_write_prompt",
    "risk_prompt",
    "trend_write_prompt",
)

INTERMEDIATE_FIELDS = (
    "news_anlyzer_prompt",
    "news_anlyzer_response",
    "income_prompt",
    "income_response",
    "balance_prompt",
    "balance_response",
    "cash_prompt",
    "cash_response",
)


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
    return project_root() / "official_benchmarks" / "finrpt" / "data"


def hf_resolve_url(dataset: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset}/resolve/main/{filename}?download=true"


def download_file(url: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"Using existing raw file: {output_path}", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"Downloading {url} -> {output_path}", flush=True)
    response = http_get(url, timeout=60, headers=headers)
    tmp_path.write_bytes(response.content)
    tmp_path.replace(output_path)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def read_remote_jsonl(url: str) -> Iterator[Dict[str, Any]]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"Streaming {url}", flush=True)
    response = http_get(url, timeout=60, headers=headers)
    for line_number, line in enumerate(response.text.splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object from {url}:{line_number}")
        yield row


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def first_present(row: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_date(value: Any) -> str:
    text = str(value or "")
    return text[:10] if len(text) >= 10 else text


def sample_id(row: Dict[str, Any]) -> str:
    explicit = first_present(row, ("id", "sample_id", "task_id"))
    if explicit:
        return str(explicit)
    stock_code = str(first_present(row, ("stock_code", "ticker", "symbol")))
    date = normalize_date(first_present(row, ("date", "trade_date", "as_of_date")))
    return f"{stock_code}_{date}".strip("_")


def jsonish_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def truncate_text(value: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return value
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "\n[TRUNCATED]"
    if isinstance(value, list):
        return [truncate_text(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_text(item, max_chars) for key, item in value.items()}
    return value


def should_take(index: int, row_id: str, sample_ids: Optional[Set[str]], sample_size: Optional[int], taken: int) -> bool:
    if sample_ids is not None:
        return row_id in sample_ids
    if sample_size is None:
        return True
    return taken < sample_size


def load_sample_ids(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def source_rows(
    args: argparse.Namespace,
    raw_file: Path,
    filename: str,
) -> Iterator[Dict[str, Any]]:
    if args.source_file or args.skip_download:
        yield from read_jsonl(raw_file)
        return
    if args.cache_raw:
        download_file(hf_resolve_url(args.dataset, filename), raw_file, overwrite=args.overwrite_raw)
        yield from read_jsonl(raw_file)
        return
    yield from read_remote_jsonl(hf_resolve_url(args.dataset, filename))


def has_official_gold(row: Dict[str, Any]) -> bool:
    """Whether a HuggingFace row carries any of the five gold response fields.

    The official ``exec_eval.py`` scorer requires non-empty reference strings
    for the five ``*_write_response`` / ``risk_response`` fields. Some rows in
    the ``jinsong8/FinRpt`` dataset only contain prompts and intermediate
    analyser outputs, so we use this helper to filter or warn during prep.
    """

    for field in FINRPT_OFFICIAL_RESPONSE_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        return True
    return False


def rows_for_prepare(
    rows_iter: Iterable[Dict[str, Any]],
    sample_size: Optional[int],
    sample_ids: Optional[Set[str]],
    shuffle: bool,
    seed: int,
    require_gold: bool = False,
    drop_log: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    filtered_iter = _filter_iter(rows_iter, require_gold=require_gold, drop_log=drop_log)
    if sample_ids is not None:
        remaining = set(sample_ids)
        selected_by_id: List[Dict[str, Any]] = []
        for row in filtered_iter:
            row_id = sample_id(row)
            if row_id not in remaining:
                continue
            selected_by_id.append(row)
            remaining.remove(row_id)
            if not remaining:
                break
        if shuffle:
            rng.shuffle(selected_by_id)
        return selected_by_id

    if not shuffle:
        selected: List[Dict[str, Any]] = []
        for row in filtered_iter:
            selected.append(row)
            if sample_size is not None and len(selected) >= sample_size:
                break
        return selected

    if shuffle and sample_ids is None and sample_size is not None:
        reservoir: List[Dict[str, Any]] = []
        seen = 0
        for row in filtered_iter:
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
                continue
            replacement_index = rng.randint(0, seen - 1)
            if replacement_index < sample_size:
                reservoir[replacement_index] = row
        return reservoir

    rows = list(filtered_iter)
    if shuffle:
        rng.shuffle(rows)

    selected: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_id = sample_id(row)
        if should_take(index, row_id, sample_ids, sample_size, len(selected)):
            selected.append(row)
        if sample_ids is None and sample_size is not None and len(selected) >= sample_size and not shuffle:
            break
    return selected


def _filter_iter(
    rows_iter: Iterable[Dict[str, Any]],
    *,
    require_gold: bool,
    drop_log: Optional[List[str]],
) -> Iterator[Dict[str, Any]]:
    for row in rows_iter:
        if require_gold and not has_official_gold(row):
            if drop_log is not None:
                drop_log.append(sample_id(row))
            continue
        yield row


def standard_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample_id(row),
        "stock_code": first_present(row, ("stock_code", "ticker", "symbol")),
        "date": normalize_date(first_present(row, ("date", "trade_date", "as_of_date"))),
        **{field: jsonish_string(row.get(field, "")) for field in FINRPT_OFFICIAL_RESPONSE_FIELDS},
    }


def solver_record(row: Dict[str, Any], context_mode: str, max_context_chars: int) -> Dict[str, Any]:
    stock_code = first_present(row, ("stock_code", "ticker", "symbol"))
    date = normalize_date(first_present(row, ("date", "trade_date", "as_of_date")))
    company = first_present(row, ("company_name", "company", "stock_name"))
    record: Dict[str, Any] = {
        "id": sample_id(row),
        "sample_id": sample_id(row),
        "source_dataset": HF_DATASET,
        "stock_code": stock_code,
        "date": date,
        "company_name": company,
        "question": (
            "Generate FinRpt official benchmark responses for the supplied frozen financial context. "
            f"Stock code: {stock_code or 'unknown'}; company: {company or 'unknown'}; date: {date or 'unknown'}."
        ),
    }

    include_fields: Set[str]
    if context_mode == "prompts_only":
        include_fields = {key for key in row if key.endswith("_prompt")} | set(DEFAULT_SOURCE_FIELDS)
    elif context_mode == "prompts_and_intermediate":
        include_fields = {key for key in row if key.endswith("_prompt")} | set(DEFAULT_SOURCE_FIELDS) | set(INTERMEDIATE_FIELDS)
    elif context_mode == "full_no_targets":
        include_fields = set(row) - set(FINRPT_OFFICIAL_RESPONSE_FIELDS) - {
            "answer",
            "golden_answer",
            "ground_truth",
            "standard_answer",
            "reference",
            "label",
        }
    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")

    for key in sorted(include_fields):
        if key in record or key in FINRPT_OFFICIAL_RESPONSE_FIELDS:
            continue
        value = row.get(key)
        if value in (None, ""):
            continue
        record[key] = truncate_text(value, max_context_chars)
    return record


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = HF_FILES[args.language]
    raw_file = Path(args.raw_file).expanduser().resolve() if args.raw_file else output_dir / filename

    if args.source_file:
        raw_file = Path(args.source_file).expanduser().resolve()

    sample_ids = load_sample_ids(Path(args.sample_ids).expanduser().resolve() if args.sample_ids else None)
    drop_log: List[str] = []
    rows = rows_for_prepare(
        source_rows(args, raw_file, filename),
        args.sample_size,
        sample_ids,
        args.shuffle,
        args.seed,
        require_gold=args.require_gold_responses,
        drop_log=drop_log,
    )
    if not rows:
        raise ValueError("No rows selected from FinRpt raw file.")

    if drop_log:
        print(
            f"Dropped {len(drop_log)} rows that lack all five official response fields; "
            "remaining rows have at least one non-empty gold response.",
            flush=True,
        )

    empty_gold = [sample_id(row) for row in rows if not has_official_gold(row)]
    if empty_gold:
        message = (
            f"WARNING: {len(empty_gold)}/{len(rows)} selected rows have empty gold responses. "
            "The official exec_eval.py scorer will compute ROUGE/BERT against empty references for those ids. "
            "Re-run with --require_gold_responses to filter them out automatically."
        )
        print(message, flush=True)

    input_path = output_dir / args.input_name
    standard_path = output_dir / args.standard_name
    selected_ids_path = output_dir / args.ids_name

    write_jsonl(
        input_path,
        (solver_record(row, args.context_mode, args.max_context_chars) for row in rows),
    )
    write_jsonl(standard_path, (standard_record(row) for row in rows))
    selected_ids_path.write_text("\n".join(sample_id(row) for row in rows) + "\n", encoding="utf-8")

    if args.benchmarks_dir:
        benchmark_path = Path(args.benchmarks_dir).expanduser().resolve() / args.input_name
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote FIRE benchmark copy: {benchmark_path}", flush=True)

    print(f"Selected rows: {len(rows)}", flush=True)
    print(f"Wrote FIRE input: {input_path}", flush=True)
    print(f"Wrote official reference: {standard_path}", flush=True)
    print(f"Wrote id list: {selected_ids_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare FinRpt HuggingFace data for FIRE Agent.")
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset repo id.")
    parser.add_argument("--language", choices=sorted(HF_FILES), default="zh", help="FinRpt language split to use.")
    parser.add_argument("--output_dir", default=str(default_data_dir()), help="Directory for finrpt.jsonl and standard.jsonl.")
    parser.add_argument("--source_file", default=None, help="Use an existing raw FinRpt JSONL instead of downloading.")
    parser.add_argument("--raw_file", default=None, help="Raw downloaded JSONL path. Defaults under output_dir.")
    parser.add_argument("--skip_download", action="store_true", help="Do not contact HuggingFace; expects raw_file/source_file to exist.")
    parser.add_argument("--cache_raw", action="store_true", help="Download and cache the full raw HF JSONL before preparing outputs.")
    parser.add_argument("--overwrite_raw", action="store_true", help="Redownload raw HF file when using --cache_raw.")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional number of rows to export.")
    parser.add_argument("--sample_ids", default=None, help="Optional file containing one FinRpt id per line.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying sample_size.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument(
        "--context_mode",
        choices=("prompts_only", "prompts_and_intermediate", "full_no_targets"),
        default="prompts_only",
        help="Which non-answer fields are visible to the solver input.",
    )
    parser.add_argument("--max_context_chars", type=int, default=0, help="Truncate long string fields; 0 keeps full text.")
    parser.add_argument("--input_name", default="finrpt.jsonl", help="FIRE input filename.")
    parser.add_argument("--standard_name", default="standard.jsonl", help="Official reference filename.")
    parser.add_argument("--ids_name", default="standard.txt", help="Selected id list filename.")
    parser.add_argument("--benchmarks_dir", default=None, help="Optionally copy finrpt.jsonl into a shared benchmarks directory.")
    parser.add_argument(
        "--require_gold_responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop HF rows that have all five official response fields empty. Enabled by default because "
            "official_benchmarks/finrpt/official_eval/.../exec_eval.py (and FinRptRunner.finalize_after_run) "
            "need a non-empty reference in standard.jsonl to compute ROUGE-L/BERTScore. "
            "Use --no-require_gold_responses to retain rows with empty gold responses."
        ),
    )
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
