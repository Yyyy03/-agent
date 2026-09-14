from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable


JsonRow = Dict[str, Any]


def _present(value: Any) -> bool:
    return value not in (None, "")


def _first_present(*values: Any) -> Any:
    for value in values:
        if _present(value):
            return value
    return None


def _prompt_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
            elif str(item).strip():
                parts.append(str(item).strip())
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content.strip()
    return str(value).strip()


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _nested(row: JsonRow, *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_full_rfc_task2_prompt(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return (
        "RFC-BENCH Task 2" in value
        and "Source page:" in value
        and "Manipulated paragraph:" in value
    )


def _extract_between(text: str, start_marker: str, end_markers: Iterable[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    ends = [text.find(marker, start) for marker in end_markers]
    ends = [end for end in ends if end >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end].strip()


def _prepare_rfc_task2_row(row: JsonRow) -> JsonRow:
    prepared = dict(row)
    extra = prepared.get("extra_info") if isinstance(prepared.get("extra_info"), dict) else {}
    prompt_text = _prompt_text(_first_present(prepared.get("question"), prepared.get("prompt")))

    if not _present(prepared.get("source_page")):
        prepared["source_page"] = _first_present(
            prepared.get("original_evidence"),
            prepared.get("evidence"),
            extra.get("original_evidence"),
            extra.get("evidence"),
        )
    if not _present(prepared.get("source_page")) and _is_full_rfc_task2_prompt(prompt_text):
        prepared["source_page"] = _extract_between(
            prompt_text,
            "Source page:",
            ("Manipulated paragraph:", "Paragraph to verify:"),
        )
    if not _present(prepared.get("source_page")) and "Original evidence paragraph:" in prompt_text:
        prepared["source_page"] = _extract_between(
            prompt_text,
            "Original evidence paragraph:",
            ("Manipulated paragraph:", "Paragraph to verify:"),
        )
    if not _present(prepared.get("perturbed_text")) and _is_full_rfc_task2_prompt(prompt_text):
        prepared["perturbed_text"] = _extract_between(
            prompt_text,
            "Manipulated paragraph:",
            ("Final answer", "Return exactly", "\n\nAnswer:", "\n\nOutput"),
        )
    if not _present(prepared.get("perturbation_type")):
        prepared["perturbation_type"] = _first_present(extra.get("perturbation_type"), prepared.get("answer"))
    if not _present(prepared.get("answer")):
        prepared["answer"] = _first_present(extra.get("answer"), _nested(prepared, "reward_model", "ground_truth"))
    for key in [
        "split",
        "sample_id",
        "source_page_url",
        "source_page_source",
        "source_page_char_len",
        "ticker",
        "date",
        "source_row_index",
    ]:
        if not _present(prepared.get(key)) and _present(extra.get(key)):
            prepared[key] = extra[key]
    return prepared


def _prepare_financebench_row(row: JsonRow) -> JsonRow:
    prepared = dict(row)
    extra = prepared.get("extra_info") if isinstance(prepared.get("extra_info"), dict) else {}
    for key in [
        "raw_question",
        "question",
        "answer",
        "financebench_id",
        "company",
        "doc_name",
        "question_type",
        "question_reasoning",
        "domain_question_num",
        "justification",
        "dataset_subset_label",
        "gics_sector",
        "doc_type",
        "doc_period",
        "doc_link",
        "attachment_path",
    ]:
        if not _present(prepared.get(key)) and _present(extra.get(key)):
            prepared[key] = extra[key]
    if not _present(prepared.get("answer")):
        prepared["answer"] = _nested(prepared, "reward_model", "ground_truth")
    evidence = _first_present(prepared.get("evidence"), extra.get("evidence"))
    if _present(evidence):
        prepared["evidence"] = _parse_jsonish(evidence)
    return prepared


def _prepare_bizfinbench_row(row: JsonRow) -> JsonRow:
    prepared = dict(row)
    extra = prepared.get("extra_info") if isinstance(prepared.get("extra_info"), dict) else {}
    if not _present(prepared.get("question")):
        prepared["question"] = _prompt_text(prepared.get("prompt"))
    if not _present(prepared.get("answer")):
        prepared["answer"] = _first_present(extra.get("answer"), _nested(prepared, "reward_model", "ground_truth"))
    for key in ["sample_id", "language", "task_type", "source_file", "source_row_index", "source_name"]:
        if not _present(prepared.get(key)) and _present(extra.get(key)):
            prepared[key] = extra[key]
    return prepared


def _validate_prepared_row(input_path: Path, row: JsonRow, line_number: int) -> None:
    if input_path.name == "rfc_task2_final.jsonl":
        source_page = str(row.get("source_page") or "").strip()
        perturbed = str(row.get("perturbed_text") or "").strip()
        source_page_char_len = str(row.get("source_page_char_len") or "").strip()
        expected_full_page = source_page_char_len.isdigit() and int(source_page_char_len) >= 1000
        min_source_len = 1000 if expected_full_page else 40
        if len(source_page) < min_source_len or not perturbed:
            raise ValueError(
                f"{input_path}:{line_number} invalid RFC Task2 row after prepare: "
                f"source_page_len={len(source_page)} perturbed_text_len={len(perturbed)}"
            )
    if input_path.name == "financebench.jsonl" and isinstance(row.get("evidence"), str):
        stripped = row["evidence"].strip()
        if stripped.startswith(("[", "{")):
            raise ValueError(
                f"{input_path}:{line_number} FinanceBench evidence is still a JSON string after prepare"
            )


def _copy_jsonl(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if input_path.name == "rfc_task2_final.jsonl":
                row = _prepare_rfc_task2_row(row)
            elif input_path.name == "financebench.jsonl":
                row = _prepare_financebench_row(row)
            elif input_path.name in {"bizfinbench_v2_sample_500.jsonl", "bizfinbench_eval700_no_ci_fra.jsonl"}:
                row = _prepare_bizfinbench_row(row)
            _validate_prepared_row(input_path, row, line_number)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_inputs(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.iterdir()):
        target = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.suffix == ".jsonl":
            _copy_jsonl(path, target)
        else:
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TL mint-agent 1.0-compatible single-turn QA inputs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    prepare_inputs(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
