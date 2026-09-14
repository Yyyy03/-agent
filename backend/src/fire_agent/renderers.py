from __future__ import annotations

import html
import json
from typing import Any, Dict, List

import json_repair


JsonRow = Dict[str, Any]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_bbox(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    bbox: List[float] = []
    for item in value:
        try:
            bbox.append(float(item))
        except Exception:
            return []
    return bbox


def _dedupe_cells(cells: List[JsonRow]) -> List[JsonRow]:
    deduped: List[JsonRow] = []
    seen = set()
    for cell in cells:
        key = (
            cell.get("table_id"),
            cell.get("page_id"),
            cell.get("row"),
            cell.get("col"),
            tuple(cell.get("bbox") or []),
            cell.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cell)
    return deduped


def parse_document_artifact(value: Any) -> JsonRow:
    """Normalize a document-parse artifact into the internal schema.

    The renderer layer is deliberately tolerant: model output, tool output, or
    an already-structured dict can all flow through the same normalizer.
    """

    if isinstance(value, dict):
        payload = value
    else:
        text = str(value or "").strip()
        if not text:
            payload = {}
        else:
            try:
                parsed = json_repair.loads(text)
                payload = parsed if isinstance(parsed, dict) else {}
            except Exception:
                payload = {"raw_text": text}

    pages = _as_list(payload.get("pages"))
    layout_blocks = _as_list(payload.get("layout_blocks") or payload.get("blocks"))
    toc_tree = payload.get("toc_tree") or payload.get("toc") or []
    tables = _as_list(payload.get("tables"))
    cells = _as_list(payload.get("cells"))

    normalized_cells: List[JsonRow] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        normalized_cells.append(
            {
                "table_id": cell.get("table_id"),
                "page_id": cell.get("page_id"),
                "row": cell.get("row"),
                "col": cell.get("col"),
                "rowspan": cell.get("rowspan", 1),
                "colspan": cell.get("colspan", 1),
                "bbox": _coerce_bbox(cell.get("bbox")),
                "text": str(cell.get("text") or ""),
            }
        )

    normalized_tables: List[JsonRow] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_cells = _as_list(table.get("cells"))
        normalized_table_cells = []
        for cell in table_cells:
            if isinstance(cell, dict):
                normalized_table_cells.append(
                    {
                        "table_id": cell.get("table_id") or table.get("table_id") or table.get("id"),
                        "page_id": cell.get("page_id") or table.get("page_id"),
                        "row": cell.get("row"),
                        "col": cell.get("col"),
                        "rowspan": cell.get("rowspan", 1),
                        "colspan": cell.get("colspan", 1),
                        "bbox": _coerce_bbox(cell.get("bbox")),
                        "text": str(cell.get("text") or ""),
                    }
                )
        normalized_tables.append(
            {
                "table_id": table.get("table_id") or table.get("id"),
                "page_id": table.get("page_id"),
                "bbox": _coerce_bbox(table.get("bbox")),
                "html": str(table.get("html") or ""),
                "cells": normalized_table_cells,
            }
        )
        normalized_cells.extend(normalized_table_cells)

    return {
        "pages": [page for page in pages if isinstance(page, dict)],
        "layout_blocks": [block for block in layout_blocks if isinstance(block, dict)],
        "toc_tree": toc_tree if isinstance(toc_tree, (dict, list)) else [],
        "tables": normalized_tables,
        "cells": _dedupe_cells(normalized_cells),
        "bbox_space": payload.get("bbox_space") or "unknown",
        "provenance": payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
        "raw_text": payload.get("raw_text", ""),
    }


class LayoutJSONRenderer:
    name = "layout_json"

    def render(self, artifact: JsonRow) -> JsonRow:
        blocks = []
        for index, block in enumerate(_as_list(artifact.get("layout_blocks"))):
            if not isinstance(block, dict):
                continue
            blocks.append(
                {
                    "block_id": block.get("block_id") or block.get("id") or f"block-{index}",
                    "page_id": block.get("page_id"),
                    "type": block.get("type") or block.get("label") or "unknown",
                    "bbox": _coerce_bbox(block.get("bbox")),
                    "text": str(block.get("text") or ""),
                    "reading_order": block.get("reading_order", index),
                }
            )
        return {
            "pages": artifact.get("pages") or [],
            "layout_blocks": blocks,
            "bbox_space": artifact.get("bbox_space") or "unknown",
        }


class TOCTreeRenderer:
    name = "toc_tree"

    def render(self, artifact: JsonRow) -> Any:
        return artifact.get("toc_tree") or []


class TableHTMLRenderer:
    name = "table_html"

    def render(self, artifact: JsonRow) -> List[JsonRow]:
        rendered: List[JsonRow] = []
        for table in _as_list(artifact.get("tables")):
            if not isinstance(table, dict):
                continue
            html_text = table.get("html")
            if not html_text:
                html_text = self._cells_to_html(_as_list(table.get("cells")))
            rendered.append(
                {
                    "table_id": table.get("table_id"),
                    "page_id": table.get("page_id"),
                    "bbox": table.get("bbox") or [],
                    "html": html_text,
                }
            )
        return rendered

    def _cells_to_html(self, cells: List[Any]) -> str:
        grid: Dict[int, Dict[int, JsonRow]] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            try:
                row = int(cell.get("row", 0))
                col = int(cell.get("col", 0))
            except Exception:
                continue
            grid.setdefault(row, {})[col] = cell
        if not grid:
            return "<table></table>"
        rows = []
        for row_idx in sorted(grid):
            cols = []
            for col_idx in sorted(grid[row_idx]):
                cell = grid[row_idx][col_idx]
                attrs = []
                if int(cell.get("rowspan") or 1) > 1:
                    attrs.append(f' rowspan="{int(cell.get("rowspan"))}"')
                if int(cell.get("colspan") or 1) > 1:
                    attrs.append(f' colspan="{int(cell.get("colspan"))}"')
                cols.append(f"<td{''.join(attrs)}>{html.escape(str(cell.get('text') or ''))}</td>")
            rows.append(f"<tr>{''.join(cols)}</tr>")
        return f"<table>{''.join(rows)}</table>"


class CellBBoxRenderer:
    name = "cell_bbox"

    def render(self, artifact: JsonRow) -> List[JsonRow]:
        rendered: List[JsonRow] = []
        for index, cell in enumerate(_as_list(artifact.get("cells"))):
            if not isinstance(cell, dict):
                continue
            rendered.append(
                {
                    "cell_id": cell.get("cell_id") or f"cell-{index}",
                    "table_id": cell.get("table_id"),
                    "page_id": cell.get("page_id"),
                    "row": cell.get("row"),
                    "col": cell.get("col"),
                    "rowspan": cell.get("rowspan", 1),
                    "colspan": cell.get("colspan", 1),
                    "bbox": _coerce_bbox(cell.get("bbox")),
                    "text": str(cell.get("text") or ""),
                }
            )
        return rendered


def render_document_parse_outputs(value: Any) -> JsonRow:
    artifact = parse_document_artifact(value)
    return {
        "artifact": artifact,
        "layout_json": LayoutJSONRenderer().render(artifact),
        "toc_tree": TOCTreeRenderer().render(artifact),
        "table_html": TableHTMLRenderer().render(artifact),
        "cell_bbox": CellBBoxRenderer().render(artifact),
    }
