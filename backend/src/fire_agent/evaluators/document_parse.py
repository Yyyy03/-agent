from __future__ import annotations

from typing import List

from ..renderers import render_document_parse_outputs
from .common import (
    JsonRow,
    best_bbox_alignment,
    count_similarity,
    flatten_text,
    normalize_text,
    token_overlap,
)
from .core import BaseEvaluator


class FinDocBenchParseEvaluator(BaseEvaluator):
    name = "findocbench_parse"

    def evaluate(self, task_log: JsonRow) -> JsonRow:
        rendered = task_log.get("rendered_outputs") or render_document_parse_outputs(task_log.get("agent_result"))
        artifact = rendered.get("artifact") or {}
        layout_json = rendered.get("layout_json") or {}
        toc_tree = rendered.get("toc_tree") or []
        table_html = rendered.get("table_html") or []
        cell_bbox = rendered.get("cell_bbox") or []
        gold_artifact = self._gold_artifact(task_log)

        required_keys = ["pages", "layout_blocks", "toc_tree", "tables", "cells", "bbox_space", "provenance"]
        schema_valid = all(key in artifact for key in required_keys)
        layout_blocks = layout_json.get("layout_blocks") or []
        bbox_values = [block.get("bbox") for block in layout_blocks if isinstance(block, dict)]
        bbox_valid_count = sum(1 for bbox in bbox_values if isinstance(bbox, list) and len(bbox) == 4)
        bbox_score = (bbox_valid_count / len(bbox_values)) if bbox_values else 0.0
        layout_score = min(1.0, len(layout_blocks) / 10) * (0.5 + 0.5 * bbox_score) if layout_blocks else 0.0

        table_score = 0.0
        if table_html:
            valid_html = sum(1 for table in table_html if "<table" in str(table.get("html", "")).lower())
            table_score = valid_html / len(table_html)

        toc_score = 1.0 if toc_tree else 0.0
        cell_bbox_values = [cell.get("bbox") for cell in cell_bbox if isinstance(cell, dict)]
        cell_bbox_score = (
            sum(1 for bbox in cell_bbox_values if isinstance(bbox, list) and len(bbox) == 4) / len(cell_bbox_values)
            if cell_bbox_values
            else 0.0
        )
        schema_score = 1.0 if schema_valid else 0.0
        baseline_score = round(
            0.2 * schema_score + 0.25 * layout_score + 0.2 * table_score + 0.15 * toc_score + 0.2 * cell_bbox_score,
            4,
        )
        gold_scores = self._gold_scores(rendered, gold_artifact)
        score = gold_scores.get("gold_weighted_score")
        if score is None:
            score = baseline_score
        return {
            "evaluator": self.name,
            "judgement": "pass" if score >= 0.7 else "needs_review",
            "score": score,
            "baseline_schema_render_score": baseline_score,
            "schema_valid": schema_valid,
            "layout_score": round(layout_score, 4),
            "table_score": round(table_score, 4),
            "toc_score": round(toc_score, 4),
            "cell_bbox_score": round(cell_bbox_score, 4),
            "layout_block_count": len(layout_blocks),
            "table_count": len(table_html),
            "cell_count": len(cell_bbox),
            "bbox_space": artifact.get("bbox_space"),
            **gold_scores,
            "note": (
                "Gold parse annotations used when available; otherwise this is a schema/rendering baseline. "
                "Replace with official IoU/TEDS/tree metrics when the benchmark scorer is available."
            ),
        }

    def _gold_artifact(self, task_log: JsonRow) -> JsonRow:
        from .core import _bench_extra

        metadata = _bench_extra(task_log)
        combined: JsonRow = {}
        for key in ["ground_truth", "answer", "golden_answer"]:
            value = metadata.get(key)
            if not value:
                continue
            artifact = render_document_parse_outputs(value).get("artifact") or {}
            if self._has_parse_payload(artifact):
                combined.update({k: v for k, v in artifact.items() if v not in (None, "", [], {})})
        if task_log.get("golden_answer"):
            artifact = render_document_parse_outputs(task_log.get("golden_answer")).get("artifact") or {}
            if self._has_parse_payload(artifact):
                combined.update({k: v for k, v in artifact.items() if v not in (None, "", [], {})})
        field_map = {
            "layout_gt": "layout_blocks",
            "toc_gt": "toc_tree",
            "tables_gt": "tables",
            "cells_gt": "cells",
        }
        for source_key, target_key in field_map.items():
            if source_key in metadata and metadata[source_key] not in (None, "", [], {}):
                combined[target_key] = metadata[source_key]
        if not combined:
            return {}
        return render_document_parse_outputs(combined).get("artifact") or {}

    def _has_parse_payload(self, artifact: JsonRow) -> bool:
        return any(artifact.get(key) for key in ["layout_blocks", "toc_tree", "tables", "cells"])

    def _gold_scores(self, rendered: JsonRow, gold_artifact: JsonRow) -> JsonRow:
        if not self._has_parse_payload(gold_artifact):
            return {"gold_available": False, "gold_weighted_score": None}

        pred_artifact = rendered.get("artifact") or {}
        pred_layout = rendered.get("layout_json", {}).get("layout_blocks") or []
        pred_tables = rendered.get("table_html") or []
        pred_toc = rendered.get("toc_tree") or []
        pred_cells = rendered.get("cell_bbox") or []
        gold_rendered = render_document_parse_outputs(gold_artifact)
        gold_layout = gold_rendered.get("layout_json", {}).get("layout_blocks") or []
        gold_tables = gold_rendered.get("table_html") or []
        gold_toc = gold_rendered.get("toc_tree") or []
        gold_cells = gold_rendered.get("cell_bbox") or []

        component_scores: List[tuple[float, float]] = []
        layout_gold_score = None
        if gold_layout:
            count_score = count_similarity(len(pred_layout), len(gold_layout))
            text_score = token_overlap(flatten_text(pred_layout), flatten_text(gold_layout))
            bbox_score = best_bbox_alignment(pred_layout, gold_layout)
            layout_gold_score = round(0.35 * count_score + 0.35 * text_score + 0.3 * bbox_score, 4)
            component_scores.append((layout_gold_score, 0.3))

        table_gold_score = None
        if gold_tables:
            count_score = count_similarity(len(pred_tables), len(gold_tables))
            text_score = token_overlap(flatten_text(pred_tables), flatten_text(gold_tables))
            table_gold_score = round(0.4 * count_score + 0.6 * text_score, 4)
            component_scores.append((table_gold_score, 0.25))

        toc_gold_score = None
        if gold_toc:
            toc_gold_score = round(token_overlap(flatten_text(pred_toc), flatten_text(gold_toc)), 4)
            component_scores.append((toc_gold_score, 0.2))

        cell_bbox_gold_score = None
        if gold_cells:
            count_score = count_similarity(len(pred_cells), len(gold_cells))
            text_score = token_overlap(flatten_text(pred_cells), flatten_text(gold_cells))
            bbox_score = best_bbox_alignment(pred_cells, gold_cells)
            cell_bbox_gold_score = round(0.25 * count_score + 0.25 * text_score + 0.5 * bbox_score, 4)
            component_scores.append((cell_bbox_gold_score, 0.25))

        schema_score = 1.0 if all(key in pred_artifact for key in ["pages", "layout_blocks", "toc_tree", "tables", "cells", "bbox_space", "provenance"]) else 0.0
        component_scores.append((schema_score, 0.1))
        total_weight = sum(weight for _, weight in component_scores)
        weighted = sum(score * weight for score, weight in component_scores) / total_weight if total_weight else None

        return {
            "gold_available": True,
            "gold_weighted_score": round(weighted, 4) if weighted is not None else None,
            "layout_gold_score": layout_gold_score,
            "table_gold_score": table_gold_score,
            "toc_gold_score": toc_gold_score,
            "cell_bbox_gold_score": cell_bbox_gold_score,
            "gold_layout_block_count": len(gold_layout),
            "gold_table_count": len(gold_tables),
            "gold_cell_count": len(gold_cells),
        }
