from __future__ import annotations

import re
from typing import Any, Dict, List

from ..constants import (
    FINDOCRESEARCH_REQUIRED_SECTIONS,
    FINDOCRESEARCH_REQUIRED_SUBSECTIONS,
    FINRPT_OFFICIAL_RESPONSE_FIELDS,
)


JsonRow = Dict[str, Any]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def extract_numbers(text: Any) -> List[str]:
    return re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", str(text or ""))


def numeric_overlap(prediction: Any, reference: Any) -> float:
    pred_numbers = set(extract_numbers(prediction))
    ref_numbers = set(extract_numbers(reference))
    if not ref_numbers:
        return 0.0
    return len(pred_numbers & ref_numbers) / len(ref_numbers)


def token_overlap(prediction: Any, reference: Any) -> float:
    pred_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(prediction)))
    ref_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(reference)))
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def count_similarity(pred_count: int, ref_count: int) -> float:
    if ref_count <= 0:
        return 0.0
    if pred_count <= 0:
        return 0.0
    return min(pred_count, ref_count) / max(pred_count, ref_count)


def bbox_iou(pred_bbox: Any, ref_bbox: Any) -> float:
    if not isinstance(pred_bbox, list) or not isinstance(ref_bbox, list) or len(pred_bbox) != 4 or len(ref_bbox) != 4:
        return 0.0
    try:
        px0, py0, px1, py1 = [float(value) for value in pred_bbox]
        rx0, ry0, rx1, ry1 = [float(value) for value in ref_bbox]
    except Exception:
        return 0.0
    inter_x0 = max(min(px0, px1), min(rx0, rx1))
    inter_y0 = max(min(py0, py1), min(ry0, ry1))
    inter_x1 = min(max(px0, px1), max(rx0, rx1))
    inter_y1 = min(max(py0, py1), max(ry0, ry1))
    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    pred_area = abs((px1 - px0) * (py1 - py0))
    ref_area = abs((rx1 - rx0) * (ry1 - ry0))
    union = pred_area + ref_area - inter_area
    return inter_area / union if union > 0 else 0.0


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: List[str] = []
        for key in ["title", "text", "label", "heading", "html"]:
            if value.get(key):
                parts.append(str(value.get(key)))
        for child_key in ["children", "items", "nodes"]:
            if child_key in value:
                parts.append(flatten_text(value[child_key]))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value or "")


def best_bbox_alignment(pred_items: List[JsonRow], ref_items: List[JsonRow]) -> float:
    ref_bboxes = [item for item in ref_items if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4]
    if not ref_bboxes:
        return 0.0
    scores: List[float] = []
    for ref_item in ref_bboxes:
        candidates = pred_items
        same_page = [item for item in pred_items if item.get("page_id") == ref_item.get("page_id")]
        if same_page:
            candidates = same_page
        keyed = [
            item
            for item in candidates
            if item.get("table_id") == ref_item.get("table_id")
            and item.get("row") == ref_item.get("row")
            and item.get("col") == ref_item.get("col")
        ]
        if keyed:
            candidates = keyed
        scores.append(max((bbox_iou(item.get("bbox"), ref_item.get("bbox")) for item in candidates), default=0.0))
    return sum(scores) / len(scores)

