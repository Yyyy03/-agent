from __future__ import annotations

import importlib.metadata
import csv
import json
import math
import mimetypes
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

from ..http_client import HTTPRequestError, http_get
from ..schemas import ToolResult, compact_text
from .base import ToolActionSpec, ToolFamily


SEC_STRUCTURED_TABLE_READER_ERROR = (
    "SEC filing URLs must be read with sec_reader.read_tables, not structured_table_reader."
)


def _env_timeout_seconds(name: str, requested: int, default: int) -> int:
    try:
        configured = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        configured = default
    try:
        requested_int = int(requested)
    except (TypeError, ValueError):
        requested_int = default
    return max(1, min(requested_int, configured))


def _is_slow_dynamic_market_table_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or ""))
    except Exception:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host in {"finance.yahoo.com", "hk.finance.yahoo.com"} and "/history" in path:
        return True
    if host == "indexes.nasdaqomx.com" and "/index/history/" in path:
        return True
    if host == "cn.investing.com" and "historical-data" in path:
        return True
    if host.endswith("nasdaq.com") and "historical" in path:
        return True
    if host == "www.sse.com.cn" and "/market/sseindex/quotation" in path:
        return True
    if host == "quote.eastmoney.com":
        return True
    return False


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml,application/pdf,text/csv,"
        "application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/xhtml+xml": ".html",
    "text/html": ".html",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": ".xlsm",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12": ".xlsb",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
}

_DOCLING_SUPPORTED_SUFFIXES = {
    # ODS is handled by the lightweight OpenDocument parser below before Docling.
    ".ods",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".pdf",
    ".asciidoc",
    ".adoc",
    ".md",
    ".csv",
    ".xlsx",
    ".xml",
    ".json",
    ".mp3",
    ".wav",
    ".m4a",
    ".vtt",
    ".tex",
    ".latex",
    ".eml",
    ".email",
}

_SERVER_ENDPOINT_SUFFIXES = {
    ".action",
    ".asp",
    ".aspx",
    ".cgi",
    ".do",
    ".jsp",
    ".php",
}

_TABULAR_ASSET_SUFFIXES = {
    ".csv",
    ".ods",
    ".pdf",
    ".tsv",
    ".xls",
    ".xlsb",
    ".xlsm",
    ".xlsx",
}

_DOWNLOAD_ASSET_SUFFIXES = _TABULAR_ASSET_SUFFIXES | {".html", ".htm"}

_STATE_PARAM_KEYS = {
    "as_of",
    "date",
    "date_display",
    "effective_date",
    "period",
    "product",
    "region",
    "state",
    "version",
    "year",
}

_ODS_NAMESPACES = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

_ODS_TABLE_ATTR = f"{{{_ODS_NAMESPACES['table']}}}"
_ODS_OFFICE_ATTR = f"{{{_ODS_NAMESPACES['office']}}}"
_ODS_MAX_COLUMNS = 500


def _is_sec_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    host = parsed.netloc.lower()
    return host.endswith("sec.gov") or host.endswith("data.sec.gov")


@dataclass
class _InvalidDownloadedDocumentError(RuntimeError):
    message: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return self.message


class StructuredTableReaderTool(ToolFamily):
    """Non-SEC table source reader.

    ``discover_tables`` discovers stateful HTML table sources and downloadable
    assets. ``read_tables`` extracts document/table structure. The tool still
    intentionally avoids final semantic cell binding; callers must match cells
    against the question's entity, period, metric, unit, and source basis.
    """

    name = "structured_table_reader"
    description = (
        "Discover and read non-SEC HTML, PDF, CSV, ODS, and Excel table sources. "
        "Use discover_tables for dynamic/stateful HTML pages, date/version controls, "
        "forms, downloadable assets, and likely AJAX endpoints; use read_tables for "
        "exact table rows, cells, headers, and provenance. SEC filing tables must use sec_reader."
    )
    paid = False
    default_action = "read_tables"
    actions = {
        "discover_tables": ToolActionSpec(
            (
                "Lightly scan a non-SEC URL or local HTML/table source for table assets, "
                "HTML tables, date/version controls, forms, and likely AJAX endpoints. "
                "This action discovers how to read a stateful table source; it does not "
                "produce final table-cell evidence."
            ),
            optional=["url", "file_path", "query", "state_params", "limit"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "file_path": {"type": "string"},
                    "query": {"type": "string"},
                    "state_params": {"type": "object", "description": "Optional state/form/query parameters for dynamic tables."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "additionalProperties": False,
            },
        ),
        "read_tables": ToolActionSpec(
            "Read tables from a URL or local file with Docling.",
            optional=[
                "url",
                "file_path",
                "query",
                "table_indices",
                "limit",
                "preview_rows",
                "max_pages",
                "state_params",
                "require_state_match",
            ],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "file_path": {"type": "string"},
                    "query": {"type": "string"},
                    "table_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional zero-based table indices to read precisely.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    "preview_rows": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_pages": {"type": "integer", "minimum": 1},
                    "state_params": {"type": "object"},
                    "require_state_match": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        ),
    }

    def __init__(self) -> None:
        self._converter: Optional[Any] = None

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        aliases = {
            "default": self.default_action,
            "discover": "discover_tables",
            "inspect": "discover_tables",
            "inspect_tables": "discover_tables",
            "table_discovery": "discover_tables",
            "tables": "read_tables",
            "read_table": "read_tables",
            "html": "read_tables",
            "read_html_tables": "read_tables",
            "csv": "read_tables",
            "tsv": "read_tables",
            "read_csv": "read_tables",
            "excel": "read_tables",
            "xls": "read_tables",
            "xlsx": "read_tables",
            "read_excel": "read_tables",
            "pdf": "read_tables",
            "read_pdf_tables": "read_tables",
        }
        return aliases.get(candidate, candidate)

    def validate_arguments(self, action: str, arguments: Dict[str, Any]) -> Optional[str]:
        base = super().validate_arguments(action, arguments)
        if base:
            return base
        if not (arguments.get("url") or arguments.get("file_path")):
            return f"Missing required argument for {self.name}.{action}: url or file_path"
        return None

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        timeout = _env_timeout_seconds("FIRE_AGENT_STRUCTURED_TABLE_TIMEOUT_SECONDS", timeout, 8)
        url = str(arguments.get("url") or "").strip()
        if url and _is_sec_url(url):
            return self._sec_reader_error(action)
        if url and _is_slow_dynamic_market_table_url(url):
            return ToolResult(
                self.name,
                "fast_fail_dynamic_market_table",
                "error",
                action=action,
                error="Skipped slow/dynamic market table page; use market_data, search snippets, or a static CSV/XLS source instead.",
                paid=False,
                confidence=0.1,
                metadata={"url": url, "reason": "slow_dynamic_market_table"},
            )
        if action == "discover_tables":
            return self._discover_tables(arguments, timeout=timeout)

        try:
            source_label, source_path, cleanup_path, acquisition_metadata = self._prepare_source(arguments, timeout=timeout)
        except _InvalidDownloadedDocumentError as exc:
            return self._invalid_download_error(action, exc)
        except Exception as exc:
            return self._source_acquisition_error(action, arguments, exc)

        if _is_sec_url(source_label):
            if cleanup_path:
                self._safe_unlink(cleanup_path)
            return self._sec_reader_error(action)

        query = str(arguments.get("query") or "").strip()
        limit = self._non_negative_int(arguments.get("limit"), 5)
        preview_rows = self._positive_int(arguments.get("preview_rows"), 50)
        max_cells = self._cell_limit_for_preview(preview_rows)
        max_pages = self._positive_int(arguments.get("max_pages"), 40)

        try:
            if self._requires_state_match(arguments, acquisition_metadata):
                return self._state_mismatch_error(
                    action=action,
                    source=source_label,
                    arguments=arguments,
                    acquisition_metadata=acquisition_metadata,
                )
            suffix_lower = source_path.suffix.lower()
            if suffix_lower in {".csv", ".tsv"}:
                tables = self._read_delimited_tables(
                    source_path=source_path,
                    source=source_label,
                    query=query,
                    preview_rows=preview_rows,
                    max_cells=max_cells,
                    delimiter="\t" if suffix_lower == ".tsv" else ",",
                    parser=suffix_lower.lstrip("."),
                )
                selected = self._select_tables(tables, arguments=arguments, limit=limit)
                warnings = self._native_warnings(tables=tables, selected=selected, parser=suffix_lower.lstrip("."))
                return self._native_tables_result(
                    action=action,
                    source=source_label,
                    query=query,
                    tables=tables,
                    selected=selected,
                    warnings=warnings,
                    parser=suffix_lower.lstrip("."),
                    preview_rows=preview_rows,
                    acquisition_metadata=acquisition_metadata,
                    usage_note=(
                        "Delimited rows were extracted with the native CSV/TSV parser. "
                        "Use row_index, column labels, markdown preview, and source provenance directly."
                    ),
                )
            if suffix_lower in {".html", ".htm"}:
                tables = self._read_html_dom_tables(
                    source_path=source_path,
                    source=source_label,
                    query=query,
                    preview_rows=preview_rows,
                    max_cells=max_cells,
                )
                if tables:
                    selected = self._select_tables(tables, arguments=arguments, limit=limit)
                    warnings = self._native_warnings(tables=tables, selected=selected, parser="html_dom")
                    return self._native_tables_result(
                        action=action,
                        source=source_label,
                        query=query,
                        tables=tables,
                        selected=selected,
                        warnings=warnings,
                        parser="html_dom",
                        preview_rows=preview_rows,
                        acquisition_metadata=acquisition_metadata,
                        usage_note=(
                            "HTML tables were extracted with the native DOM parser. "
                            "Use row_index, column labels, markdown preview, and source provenance directly."
                        ),
                    )
            if suffix_lower == ".ods":
                tables = self._read_ods_tables(
                    source_path=source_path,
                    source=source_label,
                    query=query,
                    preview_rows=preview_rows,
                    max_cells=max_cells,
                )
                selected = self._select_tables(tables, arguments=arguments, limit=limit)
                warnings = self._ods_warnings(tables=tables, selected=selected, arguments=arguments)
                observation = self._format_observation(
                    source=source_label,
                    query=query,
                    tables=tables,
                    selected=selected,
                    warnings=warnings,
                    parser="ods",
                    usage_note=(
                        "OpenDocument spreadsheet rows were extracted directly from content.xml. "
                        "Use row_index, column_index, labels, markdown preview, and sheet provenance directly; "
                        "no candidate-cell ranking or LLM semantic binding is performed by this tool."
                    ),
                )
                return ToolResult(
                    self.name,
                    "ods_structured_tables",
                    "success",
                    action=action,
                    observation=observation,
                    tables=selected,
                    confidence=0.82 if selected else 0.25,
                    paid=False,
                    metadata={
                        "source": source_label,
                        "query": query,
                        "parser": "ods",
                        **acquisition_metadata,
                        "table_count": len(tables),
                        "returned": len(selected),
                        "preview_rows": preview_rows,
                        "structured_tables": selected,
                        "warnings": warnings,
                    },
                )

            conversion = self._convert(source_path, max_pages=max_pages)
            document = getattr(conversion, "document", None)
            doc_tables = list(getattr(document, "tables", []) or []) if document is not None else []
            tables = [
                self._serialize_table(
                    table=table,
                    document=document,
                    source=source_label,
                    table_index=index,
                    preview_rows=preview_rows,
                    max_cells=max_cells,
                )
                for index, table in enumerate(doc_tables)
            ]
            selected = self._select_tables(tables, arguments=arguments, limit=limit)
            warnings = self._warnings(conversion, tables=tables, selected=selected, arguments=arguments)
            if acquisition_metadata.get("state_warning"):
                warnings.append(str(acquisition_metadata["state_warning"]))
            observation = self._format_observation(
                source=source_label,
                query=query,
                tables=tables,
                selected=selected,
                warnings=warnings,
                parser="docling",
            )
            return ToolResult(
                self.name,
                "docling_structured_tables",
                "success",
                action=action,
                observation=observation,
                tables=selected,
                confidence=0.82 if selected else 0.25,
                paid=False,
                metadata={
                    "source": source_label,
                    "query": query,
                    "parser": "docling",
                    "docling_version": self._docling_version(),
                    **acquisition_metadata,
                    "table_count": len(tables),
                    "returned": len(selected),
                    "preview_rows": preview_rows,
                    "structured_tables": selected,
                    "warnings": warnings,
                },
            )
        except Exception as exc:
            return self._parser_error(action, source_label, exc, acquisition_metadata)
        finally:
            if cleanup_path:
                self._safe_unlink(cleanup_path)

    def _sec_reader_error(self, action: str) -> ToolResult:
        return ToolResult(
            self.name,
            "docling_structured_tables",
            "error",
            action=action,
            error=SEC_STRUCTURED_TABLE_READER_ERROR,
            paid=False,
            confidence=0.1,
        )

    def _invalid_download_error(self, action: str, exc: _InvalidDownloadedDocumentError) -> ToolResult:
        metadata = {"error_code": "invalid_downloaded_document", **(exc.metadata or {})}
        return ToolResult(
            self.name,
            "docling_structured_tables",
            "error",
            action=action,
            error=exc.message,
            paid=False,
            confidence=0.1,
            metadata=metadata,
        )

    def _discover_tables(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        try:
            source_label, headers, content = self._load_discovery_source(arguments, timeout=timeout)
        except Exception as exc:
            return self._source_acquisition_error("discover_tables", arguments, exc)

        suffix = self._suffix_for_download(source_label, headers, content)
        content_type = self._header_value(headers, "content-type")
        state_params = self._state_params(arguments)
        limit = self._positive_int(arguments.get("limit"), 20)
        if not self._looks_like_html_download(headers, content, suffix):
            asset = self._asset_record(
                url=source_label,
                text=Path(urlparse(source_label).path).name or source_label,
                suffix=suffix,
                state_terms=[],
            )
            payload = {
                "source": source_label,
                "source_status": "non_html_table_asset" if suffix in _TABULAR_ASSET_SUFFIXES else "non_html_source",
                "parser": "asset_discovery",
                "content_type": content_type,
                "suffix": suffix,
                "downloaded_bytes": len(content or b""),
                "table_count": 0,
                "html_tables": [],
                "download_assets": [asset] if asset else [],
                "state_controls": [],
                "forms": [],
                "ajax_candidates": [],
                "current_state": {},
                "recommended_next": {
                    "tool": self.name,
                    "action": "read_tables",
                    "arguments": {"url": source_label, "query": query} if source_label.startswith(("http://", "https://")) else {"file_path": source_label, "query": query},
                    "reason": "Source is already a direct table-like asset; read_tables can parse it directly.",
                },
                "warnings": [] if suffix in _TABULAR_ASSET_SUFFIXES else ["Source did not look like HTML or a recognized table asset."],
            }
        else:
            html = content.decode("utf-8", errors="replace")
            payload = self._discover_html_source(
                source=source_label,
                html=html,
                query=query,
                state_params=state_params,
                limit=limit,
            )
            payload["content_type"] = content_type
            payload["suffix"] = suffix
            payload["downloaded_bytes"] = len(content or b"")

        observation = compact_text(json.dumps(payload, ensure_ascii=False, default=str), 20000)
        return ToolResult(
            self.name,
            "table_source_discovery",
            "success",
            action="discover_tables",
            observation=observation,
            confidence=self._discovery_confidence(payload),
            paid=False,
            metadata={
                "source": source_label,
                "query": query,
                "parser": payload.get("parser") or "table_source_discovery",
                "source_status": payload.get("source_status"),
                "table_count": payload.get("table_count"),
                "asset_count": len(payload.get("download_assets") or []),
                "state_control_count": len(payload.get("state_controls") or []),
                "ajax_candidate_count": len(payload.get("ajax_candidates") or []),
                "recommended_next": payload.get("recommended_next"),
                "failure_class": payload.get("failure_class"),
                "discovery_payload": payload,
            },
        )

    def _load_discovery_source(self, arguments: Dict[str, Any], timeout: int) -> Tuple[str, Dict[str, str], bytes]:
        url = str(arguments.get("url") or "").strip()
        file_path = str(arguments.get("file_path") or "").strip()
        if url:
            response = http_get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            return str(response.url or url), dict(response.headers or {}), response.content or b""
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return str(path), {}, path.read_bytes()

    def _prepare_source(self, arguments: Dict[str, Any], timeout: int) -> Tuple[str, Path, Optional[Path], Dict[str, Any]]:
        url = str(arguments.get("url") or "").strip()
        file_path = str(arguments.get("file_path") or "").strip()
        acquisition_metadata: Dict[str, Any] = {}
        if url:
            state_params = self._state_params(arguments)
            explicit_state_url = self._explicit_state_url(url, state_params)
            if explicit_state_url:
                url = explicit_state_url
                acquisition_metadata["state_applied"] = True
                acquisition_metadata["state_resolution"] = "explicit_state_url"
            response = http_get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            source_label = str(response.url or url)
            if state_params and self._looks_like_html_download(response.headers, response.content, ""):
                html = response.content.decode("utf-8", errors="replace")
                discovery = self._discover_html_source(
                    source=source_label,
                    html=html,
                    query=str(arguments.get("query") or ""),
                    state_params=state_params,
                    limit=25,
                )
                acquisition_metadata.update(self._state_metadata(discovery, state_params))
                state_url = self._resolved_state_url(discovery)
                if state_url and state_url != source_label:
                    response = http_get(state_url, headers=_BROWSER_HEADERS, timeout=timeout)
                    source_label = str(response.url or state_url)
                    acquisition_metadata["state_applied"] = True
                    acquisition_metadata["state_resolution"] = "matched_link_or_form"
                    acquisition_metadata["state_source_url"] = state_url
            suffix = self._suffix_for_download(source_label, response.headers, response.content)
            self._validate_downloaded_content(
                source=source_label,
                suffix=suffix,
                headers=response.headers,
                content=response.content,
            )
            handle = tempfile.NamedTemporaryFile(prefix="fire_docling_", suffix=suffix, delete=False)
            try:
                handle.write(response.content)
                temp_path = Path(handle.name)
            finally:
                handle.close()
            return source_label, temp_path, temp_path, acquisition_metadata

        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        self._validate_local_file(path)
        return str(path), path, None, acquisition_metadata

    def _discover_html_source(
        self,
        *,
        source: str,
        html: str,
        query: str,
        state_params: Dict[str, Any],
        limit: int,
    ) -> Dict[str, Any]:
        state_terms = self._state_terms(state_params, query)
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html or "", "html.parser")
        except Exception as exc:
            return self._discover_html_regex(
                source=source,
                html=html,
                query=query,
                state_params=state_params,
                state_terms=state_terms,
                error=exc,
            )

        title = self._node_text(soup.title) if getattr(soup, "title", None) else ""
        html_tables = self._discover_dom_tables(soup, limit=limit)
        download_assets = self._discover_dom_assets(soup, source, state_terms=state_terms, limit=max(limit * 2, 40))
        state_controls = self._discover_dom_state_controls(soup, state_terms=state_terms, limit=limit)
        forms = self._discover_dom_forms(soup, source, state_terms=state_terms, limit=limit)
        ajax_candidates = self._discover_dom_ajax_candidates(soup, source, state_terms=state_terms, limit=max(limit, 20))
        visible_text = self._node_text(soup)
        detected_dates = self._date_candidates(visible_text)
        matching_assets = [asset for asset in download_assets if asset.get("state_match")]
        matching_forms = [form for form in forms if form.get("state_match") or form.get("resolved_url")]
        matching_controls = [control for control in state_controls if control.get("state_match")]
        recommended_next = self._recommended_next(
            source=source,
            query=query,
            state_params=state_params,
            html_tables=html_tables,
            download_assets=download_assets,
            state_controls=state_controls,
            forms=forms,
        )
        warnings: List[str] = []
        if state_params and not (matching_assets or matching_forms or matching_controls or self._matches_terms(visible_text, state_terms)):
            warnings.append("Requested state/date/version was not visibly matched in the HTML discovery pass.")
        if not html_tables and not download_assets:
            warnings.append("No HTML tables or downloadable table assets were discovered.")

        return {
            "source": source,
            "source_status": self._html_source_status(html),
            "parser": "dom_discovery",
            "title": title,
            "query": query,
            "requested_state": state_params,
            "state_match": bool(matching_assets or matching_forms or matching_controls or self._matches_terms(visible_text, state_terms)),
            "table_count": len(html_tables),
            "html_tables": html_tables,
            "download_assets": download_assets,
            "state_controls": state_controls,
            "forms": forms,
            "ajax_candidates": ajax_candidates,
            "current_state": {
                "selected_controls": [
                    {
                        "name": control.get("name"),
                        "id": control.get("id"),
                        "kind": control.get("kind"),
                        "selected": control.get("selected"),
                    }
                    for control in state_controls
                    if control.get("selected") not in (None, "", [], {})
                ][:20],
                "detected_dates": detected_dates[:20],
            },
            "recommended_next": recommended_next,
            "warnings": warnings,
        }

    def _discover_html_regex(
        self,
        *,
        source: str,
        html: str,
        query: str,
        state_params: Dict[str, Any],
        state_terms: List[str],
        error: Exception,
    ) -> Dict[str, Any]:
        table_count = len(re.findall(r"(?is)<table\b", html or ""))
        assets: List[Dict[str, Any]] = []
        for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", html or "", flags=re.I):
            href = match.group(1)
            suffix = self._normalize_supported_suffix(Path(urlparse(href).path).suffix.lower())
            if suffix in _DOWNLOAD_ASSET_SUFFIXES:
                asset = self._asset_record(url=urljoin(source, href), text=href, suffix=suffix, state_terms=state_terms)
                if asset:
                    assets.append(asset)
            if len(assets) >= 40:
                break
        return {
            "source": source,
            "source_status": self._html_source_status(html),
            "parser": "html_regex_discovery",
            "query": query,
            "requested_state": state_params,
            "state_match": self._matches_terms(html, state_terms),
            "table_count": table_count,
            "html_tables": [
                {
                    "table_index": index,
                    "caption": "",
                    "headers": [],
                    "row_count_estimate": None,
                    "column_count_estimate": None,
                }
                for index in range(min(table_count, 20))
            ],
            "download_assets": assets,
            "state_controls": [],
            "forms": [],
            "ajax_candidates": [],
            "current_state": {"detected_dates": self._date_candidates(html)[:20]},
            "recommended_next": {
                "tool": self.name,
                "action": "read_tables",
                "arguments": {"url": source, "query": query},
                "reason": "Fallback regex discovery found table-like HTML; parse with read_tables.",
            } if table_count or assets else {},
            "warnings": [f"BeautifulSoup HTML discovery failed; used regex fallback: {type(error).__name__}: {error}"],
        }

    def _discover_dom_tables(self, soup: Any, *, limit: int) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        for index, table in enumerate(soup.find_all("table")):
            if len(tables) >= limit:
                break
            caption_node = table.find("caption")
            headers = [self._node_text(th) for th in table.find_all("th")[:40]]
            rows = table.find_all("tr")
            first_data_row = None
            for row in rows:
                cells = row.find_all(["td", "th"])
                if cells:
                    first_data_row = [self._node_text(cell) for cell in cells[:12]]
                    break
            column_count = len(first_data_row or headers)
            tables.append(
                {
                    "table_index": index,
                    "caption": self._node_text(caption_node),
                    "headers": [header for header in headers if header][:40],
                    "row_count_estimate": len(rows),
                    "column_count_estimate": column_count or None,
                    "sample_first_row": first_data_row or [],
                    "id": str(table.get("id") or ""),
                    "classes": self._class_list(table)[:8],
                }
            )
        return tables

    def _discover_dom_assets(self, soup: Any, source: str, *, state_terms: List[str], limit: int) -> List[Dict[str, Any]]:
        assets: List[Dict[str, Any]] = []
        seen = set()
        for link in soup.find_all("a", href=True):
            href = str(link.get("href") or "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            absolute = self._join_source_ref(source, href)
            if absolute in seen:
                continue
            text = self._node_text(link)
            suffix = self._normalize_supported_suffix(Path(urlparse(absolute).path).suffix.lower())
            is_table_asset = suffix in _DOWNLOAD_ASSET_SUFFIXES
            looks_download = bool(re.search(r"\b(download|csv|excel|xlsx|xls|ods|pdf|table|spreadsheet|biểu|bảng|tải)\b", text, flags=re.I))
            state_match = self._matches_terms(" ".join([text, href]), state_terms)
            if not (is_table_asset or looks_download or state_match):
                continue
            asset = self._asset_record(url=absolute, text=text or href, suffix=suffix, state_terms=state_terms)
            if asset:
                assets.append(asset)
                seen.add(absolute)
            if len(assets) >= limit:
                break
        return assets

    def _discover_dom_state_controls(self, soup: Any, *, state_terms: List[str], limit: int) -> List[Dict[str, Any]]:
        controls: List[Dict[str, Any]] = []
        for select in soup.find_all("select"):
            if len(controls) >= limit:
                break
            options: List[Dict[str, Any]] = []
            matches: List[Dict[str, Any]] = []
            selected_text = ""
            for option in select.find_all("option"):
                item = {
                    "value": str(option.get("value") or "").strip(),
                    "text": self._node_text(option),
                    "selected": bool(option.get("selected")),
                }
                if item["selected"]:
                    selected_text = item["text"] or item["value"]
                if self._matches_terms(" ".join([item["value"], item["text"]]), state_terms):
                    item["state_match"] = True
                    matches.append(item)
                if len(options) < 25:
                    options.append(item)
            for item in matches:
                if item not in options:
                    options.append(item)
            label = " ".join(str(select.get(key) or "") for key in ("name", "id", "aria-label"))
            control = {
                "type": "select",
                "name": str(select.get("name") or ""),
                "id": str(select.get("id") or ""),
                "kind": self._control_kind(label + " " + " ".join(item.get("text", "") for item in options[:8])),
                "selected": selected_text,
                "option_count": len(select.find_all("option")),
                "options_sample": options[:30],
                "state_match": bool(matches),
                "matched_options": matches[:10],
            }
            controls.append(control)
        for element in soup.find_all("input"):
            if len(controls) >= limit:
                break
            input_type = str(element.get("type") or "").lower()
            label = " ".join(str(element.get(key) or "") for key in ("name", "id", "placeholder", "value"))
            if input_type not in {"date", "month", "hidden", "text", "search"} and not self._control_kind(label):
                continue
            controls.append(
                {
                    "type": "input",
                    "input_type": input_type,
                    "name": str(element.get("name") or ""),
                    "id": str(element.get("id") or ""),
                    "kind": self._control_kind(label),
                    "value": str(element.get("value") or ""),
                    "placeholder": str(element.get("placeholder") or ""),
                    "state_match": self._matches_terms(label, state_terms),
                }
            )
        return controls

    def _discover_dom_forms(self, soup: Any, source: str, *, state_terms: List[str], limit: int) -> List[Dict[str, Any]]:
        forms: List[Dict[str, Any]] = []
        for form_index, form in enumerate(soup.find_all("form")):
            if len(forms) >= limit:
                break
            action = str(form.get("action") or "").strip()
            method = str(form.get("method") or "get").strip().lower()
            fields: List[Dict[str, Any]] = []
            params: Dict[str, Any] = {}
            state_match = False
            for element in form.find_all(["input", "select", "button"])[:60]:
                name = str(element.get("name") or "").strip()
                value = str(element.get("value") or "").strip()
                text = self._node_text(element)
                kind = self._control_kind(" ".join([name, str(element.get("id") or ""), text, value]))
                if name and value:
                    params.setdefault(name, value)
                match = self._matches_terms(" ".join([name, value, text]), state_terms)
                state_match = state_match or match
                fields.append(
                    {
                        "tag": str(getattr(element, "name", "") or ""),
                        "name": name,
                        "value": value,
                        "text": text,
                        "kind": kind,
                        "state_match": match,
                    }
                )
            resolved_url = ""
            if method in {"", "get"} and action and state_match:
                resolved_url = self._url_with_params(self._join_source_ref(source, action), params)
            forms.append(
                {
                    "form_index": form_index,
                    "method": method or "get",
                    "action": self._join_source_ref(source, action) if action else source,
                    "fields": fields[:40],
                    "state_match": state_match,
                    "resolved_url": resolved_url,
                }
            )
        return forms

    def _discover_dom_ajax_candidates(self, soup: Any, source: str, *, state_terms: List[str], limit: int) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        pattern = re.compile(
            r"""["']([^"']*(?:ajax|api|json|table|list|price|data|gia|xang|Get|Load|Search)[^"']*)["']""",
            flags=re.I,
        )
        for script in soup.find_all("script"):
            text = script.string or script.get_text(" ", strip=False) or ""
            for match in pattern.finditer(text):
                raw = match.group(1).strip()
                if not raw or raw.startswith(("data:", "#")):
                    continue
                if len(raw) > 260:
                    continue
                url = self._join_source_ref(source, raw)
                if url in seen:
                    continue
                seen.add(url)
                candidates.append(
                    {
                        "url": url,
                        "raw": raw,
                        "state_match": self._matches_terms(raw, state_terms),
                    }
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _recommended_next(
        self,
        *,
        source: str,
        query: str,
        state_params: Dict[str, Any],
        html_tables: List[Dict[str, Any]],
        download_assets: List[Dict[str, Any]],
        state_controls: List[Dict[str, Any]],
        forms: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        matched_assets = [asset for asset in download_assets if asset.get("state_match")]
        table_assets = [asset for asset in download_assets if asset.get("asset_type") in {"csv", "ods", "pdf", "tsv", "xls", "xlsb", "xlsm", "xlsx"}]
        matched_forms = [form for form in forms if form.get("resolved_url")]
        if matched_assets:
            return {
                "tool": self.name,
                "action": "read_tables",
                "arguments": {"url": matched_assets[0].get("url"), "query": query},
                "reason": "A discovered downloadable table asset matched the requested state/date/version.",
            }
        if matched_forms:
            return {
                "tool": self.name,
                "action": "read_tables",
                "arguments": {"url": matched_forms[0].get("resolved_url"), "query": query},
                "reason": "A GET form candidate matched the requested state/date/version.",
            }
        if table_assets:
            return {
                "tool": self.name,
                "action": "read_tables",
                "arguments": {"url": table_assets[0].get("url"), "query": query},
                "reason": "A downloadable table asset was discovered.",
            }
        arguments: Dict[str, Any] = {"url": source, "query": query}
        if state_params:
            arguments["state_params"] = state_params
            if state_controls:
                arguments["require_state_match"] = True
        if html_tables or state_controls:
            return {
                "tool": self.name,
                "action": "read_tables",
                "arguments": arguments,
                "reason": "HTML tables or state controls were discovered on the source page.",
            }
        return {}

    def _resolved_state_url(self, discovery: Dict[str, Any]) -> str:
        recommended = discovery.get("recommended_next") if isinstance(discovery, dict) else {}
        arguments = recommended.get("arguments") if isinstance(recommended, dict) else {}
        url = arguments.get("url") if isinstance(arguments, dict) else ""
        return str(url or "").strip()

    def _state_metadata(self, discovery: Dict[str, Any], state_params: Dict[str, Any]) -> Dict[str, Any]:
        state_applicable = bool(discovery.get("state_controls") or discovery.get("forms") or discovery.get("download_assets"))
        state_match = bool(discovery.get("state_match"))
        metadata: Dict[str, Any] = {
            "requested_state": state_params,
            "state_applicable": state_applicable,
            "state_match": state_match,
            "state_applied": False,
        }
        if discovery.get("current_state") not in (None, "", [], {}):
            metadata["current_state"] = discovery.get("current_state")
        if discovery.get("recommended_next") not in (None, "", [], {}):
            metadata["state_recommended_next"] = discovery.get("recommended_next")
        if state_params and state_applicable and not state_match:
            metadata["state_warning"] = "Requested state/date/version was not visibly matched before table parsing."
        return metadata

    def _state_params(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("state_params")
        if not isinstance(raw, dict):
            raw = {}
        output: Dict[str, Any] = {
            str(key): value
            for key, value in raw.items()
            if value not in (None, "", [], {})
        }
        for key in _STATE_PARAM_KEYS:
            if arguments.get(key) not in (None, "", [], {}) and key not in output:
                output[key] = arguments.get(key)
        return output

    def _explicit_state_url(self, base_url: str, state_params: Dict[str, Any]) -> str:
        for key in ("url", "source_url", "asset_url", "download_url", "href", "resolved_url"):
            value = str(state_params.get(key) or "").strip()
            if value:
                return urljoin(base_url, value)
        return ""

    def _requires_state_match(self, arguments: Dict[str, Any], acquisition_metadata: Dict[str, Any]) -> bool:
        if not self._truthy(arguments.get("require_state_match")):
            return False
        if not self._state_params(arguments):
            return False
        if acquisition_metadata.get("state_applicable") is not True:
            return False
        return not bool(acquisition_metadata.get("state_applied") or acquisition_metadata.get("state_match"))

    def _state_mismatch_error(
        self,
        *,
        action: str,
        source: str,
        arguments: Dict[str, Any],
        acquisition_metadata: Dict[str, Any],
    ) -> ToolResult:
        metadata = {
            "source": source,
            "failure_class": "state_mismatch",
            "requested_state": self._state_params(arguments),
            **acquisition_metadata,
            "retry_same_source_allowed": True,
            "suggested_recovery": (
                "Call structured_table_reader.discover_tables to inspect available date/version controls, "
                "then pass a discovered asset URL or matching state_params to read_tables."
            ),
        }
        return ToolResult(
            self.name,
            "table_state_resolution",
            "error",
            action=action,
            error="Requested table state/date/version was not visibly matched or applied; refusing to parse the default page as final evidence.",
            paid=False,
            confidence=0.2,
            metadata=metadata,
        )

    def _source_acquisition_error(self, action: str, arguments: Dict[str, Any], exc: Exception) -> ToolResult:
        source = str(arguments.get("url") or arguments.get("file_path") or "").strip()
        metadata = self._failure_metadata(source=source, exc=exc)
        return ToolResult(
            self.name,
            "source_acquisition",
            "error",
            action=action,
            error=f"{type(exc).__name__}: {exc}",
            paid=False,
            confidence=0.1,
            metadata=metadata,
        )

    def _parser_error(self, action: str, source: str, exc: Exception, acquisition_metadata: Dict[str, Any]) -> ToolResult:
        metadata = {
            "source": source,
            "failure_class": "parser_error",
            "retry_same_source_allowed": False,
            "suggested_recovery": "Use discover_tables to find a different table asset, or use multimodal_reader if the table is image-like.",
            **(acquisition_metadata or {}),
        }
        return ToolResult(
            self.name,
            "table_parser",
            "error",
            action=action,
            error=f"{type(exc).__name__}: {exc}",
            paid=False,
            confidence=0.1,
            metadata=metadata,
        )

    def _failure_metadata(self, *, source: str, exc: Exception) -> Dict[str, Any]:
        message = str(exc or "")
        lowered = message.lower()
        status_match = re.search(r"\bHTTP\s+(\d{3})\b", message)
        status_code = int(status_match.group(1)) if status_match else None
        if status_code in {401, 403} or any(token in lowered for token in ("cloudflare", "akamai", "just a moment", "forbidden", "challenge")):
            failure_class = "blocked"
            challenge_kind = "cloudflare" if "cloudflare" in lowered or "just a moment" in lowered else "access_challenge"
            retry_allowed = False
            source_exists = "likely"
        elif status_code == 404:
            failure_class = "not_found"
            challenge_kind = ""
            retry_allowed = False
            source_exists = "unknown"
        elif status_code == 429 or "rate" in lowered:
            failure_class = "rate_limited"
            challenge_kind = ""
            retry_allowed = False
            source_exists = "likely"
        elif "ssl" in lowered or "certificate" in lowered or "eof" in lowered:
            failure_class = "ssl_or_network"
            challenge_kind = ""
            retry_allowed = True
            source_exists = "unknown"
        elif isinstance(exc, FileNotFoundError):
            failure_class = "local_file_missing"
            challenge_kind = ""
            retry_allowed = False
            source_exists = "no"
        elif isinstance(exc, HTTPRequestError):
            failure_class = "provider_error"
            challenge_kind = ""
            retry_allowed = True
            source_exists = "unknown"
        else:
            failure_class = "provider_error"
            challenge_kind = ""
            retry_allowed = True
            source_exists = "unknown"
        return {
            "source": source,
            "failure_class": failure_class,
            "http_status": status_code,
            "challenge_kind": challenge_kind,
            "source_exists": source_exists,
            "not_found_allowed": failure_class == "not_found",
            "retry_same_source_allowed": retry_allowed,
            "content_preview": compact_text(" ".join(message.split()), 900),
            "suggested_recovery": self._suggested_recovery(failure_class),
        }

    @staticmethod
    def _suggested_recovery(failure_class: str) -> str:
        if failure_class == "blocked":
            return "Use an official downloadable asset, SEC/filing mirror if applicable, browser-rendered snapshot, archive, or reputable mirror; do not conclude the source lacks data."
        if failure_class == "not_found":
            return "Verify the exact URL via web_search or discover a current source page before concluding a source gap."
        if failure_class == "ssl_or_network":
            return "Retry once later or switch to a mirror/source-discovery path; do not treat as source absence."
        if failure_class == "rate_limited":
            return "Switch provider or defer; this is a tool/provider limit, not source absence."
        return "Try discover_tables, a direct downloadable asset, or another authoritative source."

    def _looks_like_html_download(self, headers: Dict[str, str], content: bytes, suffix: str) -> bool:
        normalized = self._header_value(headers, "content-type").split(";", 1)[0].strip().lower()
        if normalized in {"text/html", "application/xhtml+xml"}:
            return True
        if suffix.lower() in {".html", ".htm"}:
            return True
        head = (content or b"")[:4096].lstrip().lower()
        return head.startswith((b"<!doctype html", b"<html")) or b"<table" in head or b"<body" in head

    def _asset_record(self, *, url: str, text: str, suffix: str, state_terms: List[str]) -> Dict[str, Any]:
        parsed_suffix = Path(urlparse(url).path).suffix.lower()
        raw_suffix = suffix or parsed_suffix
        asset_type = raw_suffix.lstrip(".") if raw_suffix else ""
        if asset_type == "htm":
            asset_type = "html"
        if not asset_type and "download" not in (text or "").lower():
            return {}
        return {
            "url": url,
            "asset_type": asset_type or "unknown",
            "suffix": raw_suffix,
            "anchor_text": compact_text(" ".join(str(text or "").split()), 240),
            "state_match": self._matches_terms(" ".join([url, text or ""]), state_terms),
            "recommended_action": "read_tables" if raw_suffix in _TABULAR_ASSET_SUFFIXES or asset_type in {"csv", "html", "ods", "pdf", "tsv", "xls", "xlsb", "xlsm", "xlsx"} else "discover_tables",
        }

    @staticmethod
    def _discovery_confidence(payload: Dict[str, Any]) -> float:
        if not isinstance(payload, dict):
            return 0.2
        if payload.get("source_status") in {"blocked", "challenge", "login"}:
            return 0.2
        if payload.get("table_count") or payload.get("download_assets") or payload.get("state_controls"):
            return 0.78
        return 0.45

    @classmethod
    def _node_text(cls, node: Any, limit: int = 2000) -> str:
        if node is None:
            return ""
        try:
            text = node.get_text(" ", strip=True)
        except Exception:
            text = str(node or "")
        return compact_text(" ".join(str(text or "").split()), limit)

    @staticmethod
    def _class_list(node: Any) -> List[str]:
        value = []
        try:
            raw = node.get("class") or []
            value = raw if isinstance(raw, list) else str(raw).split()
        except Exception:
            value = []
        return [str(item) for item in value if str(item).strip()]

    def _state_terms(self, state_params: Dict[str, Any], query: str = "") -> List[str]:
        terms: List[str] = []
        for value in (state_params or {}).values():
            text = str(value or "").strip()
            if not text:
                continue
            terms.append(text)
            terms.extend(self._date_variants(text))
        # Query terms are weaker than explicit state_params, but useful when
        # the caller asks discover_tables before constructing state_params.
        for term in re.findall(r"[A-Za-z0-9\u4e00-\u9fff./_-]+", query or ""):
            if len(term) >= 3 and any(ch.isdigit() for ch in term):
                terms.append(term)
                terms.extend(self._date_variants(term))
        normalized: List[str] = []
        for term in terms:
            clean = " ".join(str(term or "").strip().split())
            if clean and clean.casefold() not in {item.casefold() for item in normalized}:
                normalized.append(clean)
        return normalized[:40]

    @staticmethod
    def _date_variants(text: str) -> List[str]:
        stripped = str(text or "").strip()
        variants: List[str] = []
        match = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", stripped)
        if match:
            year, month, day = match.groups()
            month2 = month.zfill(2)
            day2 = day.zfill(2)
            variants.extend(
                [
                    f"{year}-{month2}-{day2}",
                    f"{day2}-{month2}-{year}",
                    f"{day2}/{month2}/{year}",
                    f"{day2}.{month2}.{year}",
                    f"{year}{month2}{day2}",
                ]
            )
        match = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", stripped)
        if match:
            day, month, year = match.groups()
            month2 = month.zfill(2)
            day2 = day.zfill(2)
            variants.extend(
                [
                    f"{year}-{month2}-{day2}",
                    f"{day2}-{month2}-{year}",
                    f"{day2}/{month2}/{year}",
                    f"{day2}.{month2}.{year}",
                    f"{year}{month2}{day2}",
                ]
            )
        return variants

    @staticmethod
    def _matches_terms(text: Any, terms: List[str]) -> bool:
        if not terms:
            return False
        lowered = str(text or "").casefold()
        if not lowered:
            return False
        compact_lowered = re.sub(r"[\s_\-/.]+", "", lowered)
        for term in terms:
            clean = str(term or "").strip().casefold()
            if not clean:
                continue
            if clean in lowered:
                return True
            if re.sub(r"[\s_\-/.]+", "", clean) in compact_lowered:
                return True
        return False

    @staticmethod
    def _date_candidates(text: str) -> List[str]:
        candidates: List[str] = []
        patterns = (
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b",
            r"\b\d{4}年\d{1,2}月\d{1,2}日\b",
        )
        for pattern in patterns:
            for match in re.findall(pattern, text or ""):
                if match not in candidates:
                    candidates.append(match)
                if len(candidates) >= 40:
                    return candidates
        return candidates

    @staticmethod
    def _html_source_status(html: str) -> str:
        lowered = (html or "")[:20000].lower()
        if any(token in lowered for token in ("just a moment", "cloudflare", "cf-chl", "challenge-platform")):
            return "challenge"
        if any(token in lowered for token in ("login", "sign in", "password")) and "table" not in lowered:
            return "login"
        return "ok"

    @staticmethod
    def _control_kind(text: str) -> str:
        lowered = str(text or "").lower()
        if any(token in lowered for token in ("date", "ngày", "ngay", "effective", "asof", "as_of", "time")):
            return "date"
        if any(token in lowered for token in ("period", "quarter", "month", "year", "kỳ", "nam", "năm")):
            return "period"
        if any(token in lowered for token in ("product", "fuel", "grade", "loại", "xăng", "dầu")):
            return "product"
        if any(token in lowered for token in ("region", "province", "city", "market", "area")):
            return "region"
        if any(token in lowered for token in ("version", "revision", "release")):
            return "version"
        return ""

    @staticmethod
    def _url_with_params(url: str, params: Dict[str, Any]) -> str:
        if not params:
            return url
        parsed = urlparse(url)
        existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for key, value in params.items():
            if key and value not in (None, ""):
                existing[str(key)] = str(value)
        return urlunparse(parsed._replace(query=urlencode(existing)))

    @staticmethod
    def _join_source_ref(source: str, ref: str) -> str:
        ref = str(ref or "").strip()
        if not ref:
            return str(source or "")
        if ref.startswith(("http://", "https://")):
            return ref
        if str(source or "").startswith(("http://", "https://")):
            return urljoin(source, ref)
        try:
            base = Path(str(source or "")).expanduser()
            parent = base.parent if base.suffix else base
            return str((parent / ref.lstrip("/")).resolve())
        except Exception:
            return ref

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    def _convert(self, source_path: Path, *, max_pages: int) -> Any:
        converter = self._get_converter()
        conversion = converter.convert(
            source_path,
            raises_on_error=False,
            max_num_pages=max_pages,
        )
        if getattr(conversion, "document", None) is None:
            status = str(getattr(conversion, "status", "") or "unknown")
            errors = self._json_safe(getattr(conversion, "errors", []) or [])
            raise RuntimeError(f"Docling conversion produced no document: status={status}, errors={errors}")
        return conversion

    def _validate_downloaded_content(
        self,
        *,
        source: str,
        suffix: str,
        headers: Dict[str, str],
        content: bytes,
    ) -> None:
        normalized_suffix = self._normalize_supported_suffix(suffix)
        if not normalized_suffix:
            content_type = self._header_value(headers, "content-type")
            raise _InvalidDownloadedDocumentError(
                (
                    "Downloaded content format is not supported by structured_table_reader/Docling: "
                    f"inferred suffix={suffix!r}. Use web_reader for narrative pages, market_data for "
                    "supported structured sources, or provide a direct PDF/HTML/CSV/XLSX document URL."
                ),
                {
                    "source": source,
                    "suffix": suffix,
                    "content_type": content_type,
                    "downloaded_bytes": len(content or b""),
                    "content_preview": self._download_preview(content),
                },
            )
        if suffix.lower() == ".ods":
            if self._has_ods_magic(content):
                return
            content_type = self._header_value(headers, "content-type")
            raise _InvalidDownloadedDocumentError(
                (
                    "Downloaded content is not a valid ODS spreadsheet: URL or response indicates .ods, "
                    "but the file does not contain an OpenDocument spreadsheet payload."
                ),
                {
                    "source": source,
                    "suffix": suffix,
                    "content_type": content_type,
                    "downloaded_bytes": len(content or b""),
                    "content_preview": self._download_preview(content),
                },
            )
        if suffix.lower() != ".pdf":
            return
        if self._has_pdf_magic(content):
            return
        content_type = self._header_value(headers, "content-type")
        raise _InvalidDownloadedDocumentError(
            (
                "Downloaded content is not a valid PDF: URL or response indicates .pdf, "
                "but the file does not start with %PDF. This is likely an HTML landing, "
                "download, login, or anti-bot challenge page; structured_table_reader will not "
                "send it to Docling/OCR."
            ),
            {
                "source": source,
                "suffix": suffix,
                "content_type": content_type,
                "downloaded_bytes": len(content or b""),
                "content_preview": self._download_preview(content),
            },
        )

    def _validate_local_file(self, path: Path) -> None:
        suffix = self._normalize_supported_suffix(path.suffix.lower())
        if not suffix:
            raise _InvalidDownloadedDocumentError(
                (
                    "Input file format is not supported by structured_table_reader/Docling: "
                    f"suffix={path.suffix.lower()!r}. Provide a PDF/HTML/CSV/XLSX or another Docling-supported file."
                ),
                {
                    "source": str(path),
                    "suffix": path.suffix.lower(),
                    "downloaded_bytes": path.stat().st_size if path.exists() else None,
                },
            )
        if suffix == ".ods":
            try:
                with path.open("rb") as handle:
                    head = handle.read(4096)
                if self._has_ods_magic(head, path=path):
                    return
            except OSError:
                return
            raise _InvalidDownloadedDocumentError(
                (
                    "Input file is not a valid ODS spreadsheet: file_path ends with .ods, "
                    "but the file does not contain an OpenDocument spreadsheet payload."
                ),
                {
                    "source": str(path),
                    "suffix": path.suffix.lower(),
                    "downloaded_bytes": path.stat().st_size if path.exists() else None,
                    "content_preview": self._download_preview(head),
                },
            )
        if suffix != ".pdf":
            return
        try:
            with path.open("rb") as handle:
                head = handle.read(2048)
        except OSError:
            return
        if self._has_pdf_magic(head):
            return
        raise _InvalidDownloadedDocumentError(
            (
                "Input file is not a valid PDF: file_path ends with .pdf, but the file does "
                "not start with %PDF; structured_table_reader will not send it to Docling/OCR."
            ),
            {
                "source": str(path),
                "suffix": path.suffix.lower(),
                "downloaded_bytes": path.stat().st_size if path.exists() else None,
                "content_preview": self._download_preview(head),
            },
        )

    @staticmethod
    def _has_pdf_magic(content: bytes) -> bool:
        return bool((content or b"").lstrip().startswith(b"%PDF"))

    @staticmethod
    def _has_ods_magic(content: bytes, path: Optional[Path] = None) -> bool:
        if not (content or b"").startswith(b"PK"):
            return False
        try:
            if path is not None:
                with zipfile.ZipFile(path) as archive:
                    mimetype = archive.read("mimetype").decode("utf-8", errors="replace")
            else:
                import io

                with zipfile.ZipFile(io.BytesIO(content or b"")) as archive:
                    mimetype = archive.read("mimetype").decode("utf-8", errors="replace")
        except Exception:
            return False
        return "application/vnd.oasis.opendocument.spreadsheet" in mimetype

    @staticmethod
    def _download_preview(content: bytes, limit: int = 240) -> str:
        raw = (content or b"")[:limit]
        text = raw.decode("utf-8", errors="replace")
        return compact_text(" ".join(text.split()), limit)

    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Docling is required for structured_table_reader. Install it with `pip install docling`."
                ) from exc
            self._converter = DocumentConverter()
        return self._converter

    def _serialize_table(
        self,
        *,
        table: Any,
        document: Any,
        source: str,
        table_index: int,
        preview_rows: int,
        max_cells: int,
    ) -> Dict[str, Any]:
        dumped = self._model_dump(table)
        dataframe = self._export_dataframe(table, document)
        columns, rows, row_count, truncated_rows = self._rows_from_dataframe(dataframe, max_rows=preview_rows)
        cells, cell_count, truncated_cells = self._cells_from_dump(dumped, max_rows=preview_rows, max_cells=max_cells)
        provenance = self._extract_provenance(dumped, table)
        markdown = self._rows_to_markdown(columns, rows)
        if not markdown:
            markdown = self._export_markdown(table, document)
        html = self._export_html(table, document)
        caption = self._caption_text(table, document, dumped)
        return {
            "table_id": f"docling_table_{table_index}",
            "table_index": table_index,
            "docling_ref": str(dumped.get("self_ref") or getattr(table, "self_ref", "") or ""),
            "source": source,
            "page_or_sheet": self._page_or_sheet(provenance),
            "caption": caption,
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "returned_rows": len(rows),
            "truncated_rows": truncated_rows,
            "cells": cells,
            "cell_count": cell_count,
            "returned_cells": len(cells),
            "truncated_cells": truncated_cells,
            "preview_rows": preview_rows,
            "markdown": compact_text(markdown, 8000) if markdown else "",
            "html": compact_text(html, 8000) if html else "",
            "provenance": provenance,
        }

    def _read_delimited_tables(
        self,
        *,
        source_path: Path,
        source: str,
        query: str,
        preview_rows: int,
        max_cells: int,
        delimiter: str,
        parser: str,
    ) -> List[Dict[str, Any]]:
        with source_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            raw_rows = [list(row) for row in reader if any(self._stringify(value) for value in row)]
        if not raw_rows:
            return []
        header = [self._stringify(value) or f"column_{index}" for index, value in enumerate(raw_rows[0])]
        data_rows = raw_rows[1:] if len(raw_rows) > 1 else []
        columns = [{"index": index, "name": name or f"column_{index}"} for index, name in enumerate(header)]
        selected = self._select_matrix_rows(data_rows, query=query, preview_rows=preview_rows, row_offset=1)
        rows = self._matrix_structured_rows(selected, columns)
        cells = self._ods_cells(rows, max_cells=max_cells)
        markdown = self._rows_to_markdown(columns, rows)
        return [
            {
                "table_id": f"{parser}_table_0",
                "table_index": 0,
                "docling_ref": "",
                "source": source,
                "page_or_sheet": Path(source_path).name,
                "caption": Path(source_path).name,
                "columns": columns,
                "rows": rows,
                "row_count": len(data_rows),
                "returned_rows": len(rows),
                "truncated_rows": max(0, len(data_rows) - len({row.get("row_index") for row in rows})),
                "cells": cells,
                "cell_count": sum(len(row) for row in raw_rows),
                "returned_cells": len(cells),
                "truncated_cells": max(0, sum(len(row) for row in raw_rows) - len(cells)),
                "preview_rows": preview_rows,
                "markdown": compact_text(markdown, 8000) if markdown else "",
                "query_markdown": compact_text(markdown, 8000) if markdown else "",
                "html": "",
                "provenance": [{"source": source, "parser": parser}],
                "query_match_count": sum(1 for _, row in selected if self._ods_row_match_score(row, self._ods_query_terms(query)) > 0),
            }
        ]

    def _read_html_dom_tables(
        self,
        *,
        source_path: Path,
        source: str,
        query: str,
        preview_rows: int,
        max_cells: int,
    ) -> List[Dict[str, Any]]:
        html = source_path.read_text(encoding="utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return []
        soup = BeautifulSoup(html or "", "html.parser")
        tables: List[Dict[str, Any]] = []
        query_terms = self._ods_query_terms(query)
        for table_index, table in enumerate(soup.find_all("table")):
            matrix: List[List[str]] = []
            header_from_th = False
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                if tr.find_all("th"):
                    header_from_th = True if not matrix else header_from_th
                matrix.append([self._node_text(cell, limit=500) for cell in cells])
            if not matrix:
                continue
            if header_from_th:
                header = matrix[0]
                data_rows = matrix[1:]
                row_offset = 1
            else:
                max_cols = max(len(row) for row in matrix)
                header = [f"column_{index}" for index in range(max_cols)]
                data_rows = matrix
                row_offset = 0
            max_cols = max(len(header), max((len(row) for row in data_rows), default=0))
            columns = [
                {"index": index, "name": self._stringify(header[index]) if index < len(header) and self._stringify(header[index]) else f"column_{index}"}
                for index in range(max_cols)
            ]
            selected = self._select_matrix_rows(data_rows, query=query, preview_rows=preview_rows, row_offset=row_offset)
            rows = self._matrix_structured_rows(selected, columns)
            cells = self._ods_cells(rows, max_cells=max_cells)
            markdown = self._rows_to_markdown(columns, rows)
            caption_node = table.find("caption")
            tables.append(
                {
                    "table_id": f"html_table_{table_index}",
                    "table_index": table_index,
                    "docling_ref": "",
                    "source": source,
                    "page_or_sheet": None,
                    "caption": self._node_text(caption_node),
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(data_rows),
                    "returned_rows": len(rows),
                    "truncated_rows": max(0, len(data_rows) - len({row.get("row_index") for row in rows})),
                    "cells": cells,
                    "cell_count": sum(len(row) for row in matrix),
                    "returned_cells": len(cells),
                    "truncated_cells": max(0, sum(len(row) for row in matrix) - len(cells)),
                    "preview_rows": preview_rows,
                    "markdown": compact_text(markdown, 8000) if markdown else "",
                    "query_markdown": compact_text(markdown, 8000) if markdown else "",
                    "html": compact_text(str(table), 8000),
                    "provenance": [{"source": source, "parser": "html_dom", "table_index": table_index}],
                    "query_match_count": sum(1 for _, row in selected if self._ods_row_match_score(row, query_terms) > 0),
                }
            )
        if query_terms:
            tables.sort(
                key=lambda table: (
                    int(table.get("query_match_count") or 0) > 0,
                    int(table.get("query_match_count") or 0),
                    int(table.get("row_count") or 0),
                ),
                reverse=True,
            )
        return tables

    def _select_matrix_rows(
        self,
        rows: List[List[Any]],
        *,
        query: str,
        preview_rows: int,
        row_offset: int,
    ) -> List[Tuple[int, List[Any]]]:
        query_terms = self._ods_query_terms(query)
        preview: List[Tuple[int, List[Any]]] = [
            (row_offset + index, row)
            for index, row in enumerate(rows[:preview_rows])
        ]
        matches: List[Tuple[int, int, List[Any]]] = []
        if query_terms:
            for index, row in enumerate(rows):
                score = self._ods_row_match_score(row, query_terms)
                if score > 0:
                    matches.append((score, row_offset + index, row))
        selected: List[Tuple[int, List[Any]]] = []
        seen = set()
        for _, row_index, row in sorted(matches, key=lambda item: (-item[0], item[1]))[:preview_rows]:
            selected.append((row_index, row))
            seen.add(row_index)
        for item in preview:
            if item[0] in seen:
                continue
            selected.append(item)
            seen.add(item[0])
        return selected[:preview_rows]

    def _matrix_structured_rows(
        self,
        selected_rows: List[Tuple[int, List[Any]]],
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        column_names = [self._stringify(column.get("name")) or f"column_{index}" for index, column in enumerate(columns)]
        for row_index, values in selected_rows:
            cells = [
                {
                    "column_index": column_index,
                    "column": column_names[column_index] if column_index < len(column_names) else f"column_{column_index}",
                    "value": self._clean_value(values[column_index]) if column_index < len(values) else "",
                }
                for column_index in range(len(columns))
            ]
            rows.append({"row_index": row_index, "cells": cells})
        return rows

    def _native_warnings(self, *, tables: List[Dict[str, Any]], selected: List[Dict[str, Any]], parser: str) -> List[str]:
        if not tables:
            return [f"No machine-readable tables were detected by the native {parser} parser."]
        if not selected:
            return [f"Native {parser} tables were detected, but the requested table_indices selected none."]
        return []

    def _native_tables_result(
        self,
        *,
        action: str,
        source: str,
        query: str,
        tables: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        warnings: List[str],
        parser: str,
        preview_rows: int,
        acquisition_metadata: Dict[str, Any],
        usage_note: str,
    ) -> ToolResult:
        observation = self._format_observation(
            source=source,
            query=query,
            tables=tables,
            selected=selected,
            warnings=warnings,
            parser=parser,
            usage_note=usage_note,
        )
        return ToolResult(
            self.name,
            "native_structured_tables",
            "success",
            action=action,
            observation=observation,
            tables=selected,
            confidence=0.84 if selected else 0.25,
            paid=False,
            metadata={
                "source": source,
                "query": query,
                "parser": parser,
                **(acquisition_metadata or {}),
                "table_count": len(tables),
                "returned": len(selected),
                "preview_rows": preview_rows,
                "structured_tables": selected,
                "warnings": warnings,
            },
        )

    def _read_ods_tables(
        self,
        *,
        source_path: Path,
        source: str,
        query: str,
        preview_rows: int,
        max_cells: int,
    ) -> List[Dict[str, Any]]:
        with zipfile.ZipFile(source_path) as archive:
            content_xml = archive.read("content.xml")
        root = ET.fromstring(content_xml)
        spreadsheet = root.find("office:body/office:spreadsheet", _ODS_NAMESPACES)
        if spreadsheet is None:
            return []

        query_terms = self._ods_query_terms(query)
        tables: List[Dict[str, Any]] = []
        for table_index, sheet in enumerate(spreadsheet.findall("table:table", _ODS_NAMESPACES)):
            sheet_name = str(sheet.attrib.get(f"{_ODS_TABLE_ATTR}name") or f"sheet_{table_index}")
            preview: List[Tuple[int, List[Any]]] = []
            match_candidates: List[Tuple[int, int, List[Any]]] = []
            row_count = 0
            max_columns = 0
            total_cells = 0
            caption = ""

            for raw_row in sheet.findall("table:table-row", _ODS_NAMESPACES):
                row = self._ods_row_values(raw_row)
                if not any(self._stringify(value) for value in row):
                    continue
                repeats = self._positive_int(raw_row.attrib.get(f"{_ODS_TABLE_ATTR}number-rows-repeated"), 1)
                max_columns = max(max_columns, len(row))
                total_cells += len(row) * repeats
                if not caption:
                    caption = self._stringify(row[0]) if row else ""
                for _ in range(repeats):
                    row_index = row_count
                    row_count += 1
                    if len(preview) < preview_rows:
                        preview.append((row_index, row))
                    if query_terms:
                        score = self._ods_row_match_score(row, query_terms)
                        if score > 0:
                            match_candidates.append((score, row_index, row))

            matches = [
                (row_index, row)
                for _, row_index, row in sorted(match_candidates, key=lambda item: (-item[0], item[1]))[:preview_rows]
            ]
            selected_rows: List[Tuple[int, List[Any]]] = []
            seen_rows = set()
            for item in [*matches, *preview]:
                if item[0] in seen_rows:
                    continue
                selected_rows.append(item)
                seen_rows.add(item[0])

            column_count = max(max_columns, max((len(row) for _, row in selected_rows), default=0))
            columns = [{"index": index, "name": f"column_{index}"} for index in range(column_count)]
            rows = self._ods_structured_rows(selected_rows, columns)
            cells = self._ods_cells(rows, max_cells=max_cells)
            markdown = self._rows_to_markdown(columns, rows)
            query_markdown = self._ods_query_markdown(columns, rows, query_terms)
            tables.append(
                {
                    "table_id": f"ods_sheet_{table_index}",
                    "table_index": table_index,
                    "docling_ref": "",
                    "source": source,
                    "page_or_sheet": sheet_name,
                    "caption": caption,
                    "columns": columns,
                    "rows": rows,
                    "row_count": row_count,
                    "returned_rows": len(rows),
                    "truncated_rows": max(0, row_count - len(seen_rows)),
                    "cells": cells,
                    "cell_count": total_cells,
                    "returned_cells": len(cells),
                    "truncated_cells": max(0, total_cells - len(cells)),
                    "preview_rows": preview_rows,
                    "markdown": compact_text(markdown, 8000) if markdown else "",
                    "query_markdown": compact_text(query_markdown, 8000) if query_markdown else "",
                    "html": "",
                    "provenance": [{"sheet_name": sheet_name, "sheet_index": table_index}],
                    "query_match_count": len(match_candidates),
                }
            )
        if query_terms:
            tables.sort(
                key=lambda table: (
                    int(table.get("query_match_count") or 0) > 0,
                    int(table.get("query_match_count") or 0),
                    int(table.get("row_count") or 0),
                ),
                reverse=True,
            )
        return tables

    def _ods_row_values(self, raw_row: Any) -> List[Any]:
        row: List[Any] = []
        for cell in list(raw_row):
            if not (
                str(cell.tag).endswith("table-cell")
                or str(cell.tag).endswith("covered-table-cell")
            ):
                continue
            repeats = self._positive_int(cell.attrib.get(f"{_ODS_TABLE_ATTR}number-columns-repeated"), 1)
            repeats = min(repeats, max(0, _ODS_MAX_COLUMNS - len(row)))
            value = self._ods_cell_value(cell)
            row.extend([value] * repeats)
            if len(row) >= _ODS_MAX_COLUMNS:
                break
        while row and not self._stringify(row[-1]):
            row.pop()
        return row

    def _ods_cell_value(self, cell: Any) -> Any:
        for attr in ("value", "date-value", "time-value", "string-value", "boolean-value"):
            key = f"{_ODS_OFFICE_ATTR}{attr}"
            if key in cell.attrib:
                return cell.attrib.get(key)
        texts: List[str] = []
        for paragraph in cell.findall(".//text:p", _ODS_NAMESPACES):
            text = "".join(paragraph.itertext()).strip()
            if text:
                texts.append(text)
        return "\n".join(texts).strip()

    def _ods_structured_rows(
        self,
        selected_rows: List[Tuple[int, List[Any]]],
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        column_names = [self._stringify(column.get("name")) for column in columns]
        for row_index, values in selected_rows:
            cells = [
                {
                    "column_index": column_index,
                    "column": column_names[column_index] if column_index < len(column_names) else f"column_{column_index}",
                    "value": self._clean_value(values[column_index]) if column_index < len(values) else "",
                }
                for column_index in range(len(columns))
            ]
            rows.append({"row_index": row_index, "cells": cells})
        return rows

    def _ods_query_markdown(
        self,
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
        query_terms: List[str],
    ) -> str:
        if not columns or not rows or not query_terms:
            return ""
        structural_terms = {
            "apr",
            "april",
            "aug",
            "august",
            "dec",
            "december",
            "feb",
            "february",
            "jan",
            "january",
            "jul",
            "july",
            "jun",
            "june",
            "mar",
            "march",
            "may",
            "nov",
            "november",
            "oct",
            "october",
            "period",
            "sep",
            "september",
        }
        column_terms = [
            term
            for term in query_terms
            if not term.isdigit() and len(term) > 2 and term not in structural_terms
        ]
        selected_indexes = {0, 1}
        for column in columns:
            try:
                column_index = int(column.get("index"))
            except (TypeError, ValueError):
                continue
            text_parts: List[str] = []
            for row in rows:
                for cell in row.get("cells") or []:
                    if cell.get("column_index") == column_index:
                        text_parts.append(self._stringify(cell.get("value")))
            column_text = " ".join(text_parts).casefold()
            if column_terms and any(term in column_text for term in column_terms):
                selected_indexes.add(column_index)
        if len(selected_indexes) <= 2:
            selected_indexes.update(range(min(8, len(columns))))
        selected = sorted(index for index in selected_indexes if 0 <= index < len(columns))
        slim_columns = [{"index": i, "name": f"column_{i}"} for i in selected]
        slim_rows: List[Dict[str, Any]] = []
        for row in rows:
            values_by_index = {
                int(cell.get("column_index")): cell.get("value")
                for cell in row.get("cells") or []
                if isinstance(cell.get("column_index"), int)
            }
            slim_rows.append(
                {
                    "row_index": row.get("row_index"),
                    "cells": [
                        {
                            "column_index": position,
                            "column": f"column_{original_index}",
                            "value": values_by_index.get(original_index, ""),
                        }
                        for position, original_index in enumerate(selected)
                    ],
                }
            )
        return self._rows_to_markdown(slim_columns, slim_rows)

    def _ods_cells(self, rows: List[Dict[str, Any]], *, max_cells: int) -> List[Dict[str, Any]]:
        cells: List[Dict[str, Any]] = []
        for row in rows:
            row_index = row.get("row_index")
            for cell in row.get("cells") or []:
                text = self._stringify(cell.get("value"))
                if not text:
                    continue
                column_index = cell.get("column_index")
                cells.append(
                    {
                        "row_start": row_index,
                        "row_end": row_index,
                        "col_start": column_index,
                        "col_end": column_index,
                        "row_span": 1,
                        "col_span": 1,
                        "text": text,
                        "column_header": False,
                        "row_header": column_index == 0,
                        "row_section": False,
                        "bbox": None,
                    }
                )
                if len(cells) >= max_cells:
                    return cells
        return cells

    def _ods_query_terms(self, query: str) -> List[str]:
        terms = [term.casefold() for term in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", query or "")]
        stopwords = {"and", "for", "from", "in", "of", "the", "to", "with"}
        month_aliases = {
            "january": "jan",
            "february": "feb",
            "march": "mar",
            "april": "apr",
            "june": "jun",
            "july": "jul",
            "august": "aug",
            "september": "sep",
            "october": "oct",
            "november": "nov",
            "december": "dec",
        }
        expanded: List[str] = []
        for term in terms:
            if len(term) < 2 or term in stopwords:
                continue
            expanded.append(term)
            alias = month_aliases.get(term)
            if alias:
                expanded.append(alias)
        return list(dict.fromkeys(expanded))

    def _ods_row_match_score(self, row: List[Any], query_terms: List[str]) -> int:
        text = " ".join(self._stringify(value) for value in row).casefold()
        if not text:
            return 0
        tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text))
        hits = sum(
            1
            for term in query_terms
            if (term in tokens if term.isdigit() else term in text)
        )
        return hits if hits >= min(2, len(query_terms)) else 0

    def _ods_warnings(
        self,
        *,
        tables: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        arguments: Dict[str, Any],
    ) -> List[str]:
        warnings: List[str] = []
        if not tables:
            warnings.append("No worksheets with non-empty rows were detected in this ODS source.")
        elif not selected:
            warnings.append("ODS worksheets were detected, but the requested table_indices selected none.")
        return warnings

    def _select_tables(self, tables: List[Dict[str, Any]], *, arguments: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        requested_indices = self._requested_indices(arguments)
        if requested_indices:
            selected = [table for table in tables if int(table.get("table_index", -1)) in requested_indices]
        else:
            selected = tables
        return selected if limit == 0 else selected[:limit]

    def _requested_indices(self, arguments: Dict[str, Any]) -> List[int]:
        raw = arguments.get("table_indices", None)
        if raw in (None, ""):
            return []
        if isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = str(raw).replace(";", ",").split(",")
        indices: List[int] = []
        for value in values:
            try:
                index = int(str(value).strip())
            except ValueError:
                continue
            if index >= 0 and index not in indices:
                indices.append(index)
        return indices

    def _warnings(
        self,
        conversion: Any,
        *,
        tables: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        arguments: Dict[str, Any],
    ) -> List[str]:
        warnings: List[str] = []
        status = str(getattr(conversion, "status", "") or "").strip()
        if status and not status.upper().endswith("SUCCESS"):
            warnings.append(f"Docling conversion status: {status}")
        errors = getattr(conversion, "errors", []) or []
        if errors:
            warnings.append(f"Docling reported conversion issues: {compact_text(json.dumps(self._json_safe(errors), ensure_ascii=False, default=str), 1000)}")
        if not tables:
            warnings.append("Docling did not detect machine-readable tables in this source.")
        elif not selected:
            warnings.append("Tables were detected, but the requested table_indices selected none.")
        return warnings

    def _format_observation(
        self,
        *,
        source: str,
        query: str,
        tables: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        warnings: List[str],
        parser: str = "docling",
        usage_note: Optional[str] = None,
    ) -> str:
        payload = {
            "source": source,
            "parser": parser,
            "query": query,
            "tables_found": len(tables),
            "returned": len(selected),
            "tables": [self._observation_table(table) for table in selected],
            "warnings": warnings,
            "usage_note": usage_note or (
                "Docling has provided document/table structure only. Use the returned row, column, "
                "cell coordinate/header flags, markdown preview, and provenance directly; no candidate-cell "
                "ranking or LLM semantic binding is performed by this tool."
            ),
        }
        return compact_text(json.dumps(payload, ensure_ascii=False, default=str), 20000)

    def _observation_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "table_id": table.get("table_id"),
            "docling_ref": table.get("docling_ref"),
            "page_or_sheet": table.get("page_or_sheet"),
            "caption": table.get("caption"),
            "columns": table.get("columns"),
            "row_count": table.get("row_count"),
            "returned_rows": table.get("returned_rows"),
            "truncated_rows": table.get("truncated_rows"),
            "preview_rows": table.get("preview_rows"),
            "cell_count": table.get("cell_count"),
            "returned_cells": table.get("returned_cells"),
            "truncated_cells": table.get("truncated_cells"),
            "markdown_preview": compact_text(str(table.get("markdown") or ""), 3000),
            "query_markdown_preview": compact_text(str(table.get("query_markdown") or ""), 6000),
            "provenance": table.get("provenance"),
            "query_match_count": table.get("query_match_count"),
        }

    def _rows_from_dataframe(self, dataframe: Any, *, max_rows: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        if dataframe is None:
            return [], [], 0, 0
        try:
            row_count = int(len(dataframe))
            raw_columns = list(dataframe.columns)
        except Exception:
            return [], [], 0, 0

        column_names = [self._stringify(column) or f"column_{index}" for index, column in enumerate(raw_columns)]
        columns = [{"index": index, "name": name} for index, name in enumerate(column_names)]
        rows: List[Dict[str, Any]] = []
        try:
            iterator = dataframe.head(max_rows).itertuples(index=False, name=None)
            for row_index, values in enumerate(iterator):
                cells = [
                    {
                        "column_index": column_index,
                        "column": column_names[column_index] if column_index < len(column_names) else f"column_{column_index}",
                        "value": self._clean_value(value),
                    }
                    for column_index, value in enumerate(values)
                ]
                rows.append({"row_index": row_index, "cells": cells})
        except Exception:
            return columns, [], row_count, row_count
        truncated = max(0, row_count - len(rows))
        return columns, rows, row_count, truncated

    def _rows_to_markdown(self, columns: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
        column_names = [self._stringify(column.get("name")) or f"column_{index}" for index, column in enumerate(columns or [])]
        if not column_names or not rows:
            return ""
        lines = [
            "| " + " | ".join(self._markdown_cell(name) for name in column_names) + " |",
            "| " + " | ".join("---" for _ in column_names) + " |",
        ]
        for row in rows:
            cell_values = ["" for _ in column_names]
            for cell in row.get("cells") or []:
                try:
                    column_index = int(cell.get("column_index"))
                except (TypeError, ValueError):
                    continue
                if 0 <= column_index < len(cell_values):
                    cell_values[column_index] = self._stringify(cell.get("value"))
            lines.append("| " + " | ".join(self._markdown_cell(value) for value in cell_values) + " |")
        return "\n".join(lines)

    @staticmethod
    def _markdown_cell(value: str) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
        return text.replace("|", "\\|")

    def _cells_from_dump(self, dumped: Dict[str, Any], *, max_rows: int, max_cells: int) -> Tuple[List[Dict[str, Any]], int, int]:
        data = dumped.get("data") if isinstance(dumped.get("data"), dict) else {}
        raw_cells = data.get("table_cells") if isinstance(data.get("table_cells"), list) else []
        cells: List[Dict[str, Any]] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict):
                continue
            row = raw_cell.get("start_row_offset_idx")
            is_header = bool(raw_cell.get("column_header") or raw_cell.get("row_header") or raw_cell.get("row_section"))
            if isinstance(row, int) and row >= max_rows and not is_header:
                continue
            cells.append(
                {
                    "row_start": raw_cell.get("start_row_offset_idx"),
                    "row_end": raw_cell.get("end_row_offset_idx"),
                    "col_start": raw_cell.get("start_col_offset_idx"),
                    "col_end": raw_cell.get("end_col_offset_idx"),
                    "row_span": raw_cell.get("row_span", 1),
                    "col_span": raw_cell.get("col_span", 1),
                    "text": self._stringify(raw_cell.get("text")),
                    "column_header": bool(raw_cell.get("column_header")),
                    "row_header": bool(raw_cell.get("row_header")),
                    "row_section": bool(raw_cell.get("row_section")),
                    "bbox": self._json_safe(raw_cell.get("bbox")),
                }
            )
            if len(cells) >= max_cells:
                break
        return cells, len(raw_cells), max(0, len(raw_cells) - len(cells))

    def _extract_provenance(self, dumped: Dict[str, Any], table: Any) -> List[Dict[str, Any]]:
        raw_provenance = dumped.get("prov")
        if not raw_provenance:
            raw_provenance = self._json_safe(getattr(table, "prov", []) or [])
        if not isinstance(raw_provenance, list):
            return []
        provenance: List[Dict[str, Any]] = []
        for item in raw_provenance:
            if not isinstance(item, dict):
                item = self._json_safe(item)
            if not isinstance(item, dict):
                continue
            provenance.append(
                {
                    "page_no": item.get("page_no"),
                    "bbox": self._json_safe(item.get("bbox")),
                    "charspan": self._json_safe(item.get("charspan")),
                }
            )
        return provenance

    def _page_or_sheet(self, provenance: List[Dict[str, Any]]) -> Optional[int]:
        for item in provenance:
            page_no = item.get("page_no")
            if isinstance(page_no, int):
                return page_no
            try:
                return int(str(page_no))
            except (TypeError, ValueError):
                continue
        return None

    def _export_dataframe(self, table: Any, document: Any) -> Any:
        try:
            return table.export_to_dataframe(doc=document)
        except TypeError:
            return table.export_to_dataframe()
        except Exception:
            return None

    def _export_markdown(self, table: Any, document: Any) -> str:
        try:
            return str(table.export_to_markdown(doc=document) or "").strip()
        except TypeError:
            return str(table.export_to_markdown() or "").strip()
        except Exception:
            return ""

    def _export_html(self, table: Any, document: Any) -> str:
        try:
            return str(table.export_to_html(doc=document) or "").strip()
        except TypeError:
            return str(table.export_to_html() or "").strip()
        except Exception:
            return ""

    def _caption_text(self, table: Any, document: Any, dumped: Dict[str, Any]) -> str:
        if hasattr(table, "caption_text"):
            try:
                caption = str(table.caption_text(document) or "").strip()
                if caption:
                    return caption
            except Exception:
                pass
        captions = dumped.get("captions")
        if isinstance(captions, list) and captions:
            return self._stringify(captions[0])
        return ""

    def _suffix_for_download(self, source: str, headers: Dict[str, str], content: bytes) -> str:
        raw_suffix = Path(urlparse(source).path).suffix.lower()
        suffix = self._normalize_supported_suffix(raw_suffix)
        if suffix and raw_suffix not in _SERVER_ENDPOINT_SUFFIXES:
            return suffix
        suffix = self._suffix_from_content_disposition(headers)
        if suffix:
            return suffix
        content_type = self._header_value(headers, "content-type").split(";", 1)[0].strip().lower()
        if content_type in _CONTENT_TYPE_SUFFIXES:
            suffix = self._normalize_supported_suffix(_CONTENT_TYPE_SUFFIXES[content_type])
            if suffix:
                return suffix
        guessed = mimetypes.guess_extension(content_type) if content_type else ""
        suffix = self._normalize_supported_suffix(guessed)
        if suffix:
            return suffix
        head = content[:1000].lstrip().lower()
        if head.startswith(b"%pdf"):
            return ".pdf"
        if head.startswith(b"pk") and b"mimetypeapplication/vnd.oasis.opendocument.spreadsheet" in head:
            return ".ods"
        if b"<html" in head or b"<table" in head:
            return ".html"
        if self._looks_like_csv_or_tsv(head):
            return ".csv"
        return ".bin"

    @classmethod
    def _normalize_supported_suffix(cls, suffix: Any) -> str:
        text = str(suffix or "").strip().lower()
        if not text:
            return ""
        if not text.startswith("."):
            text = "." + text
        aliases = {
            ".htm": ".html",
            ".jpeg": ".jpg",
            ".adoc": ".asciidoc",
            ".latex": ".tex",
            ".email": ".eml",
        }
        normalized = aliases.get(text, text)
        return normalized if normalized in _DOCLING_SUPPORTED_SUFFIXES else ""

    def _suffix_from_content_disposition(self, headers: Dict[str, str]) -> str:
        disposition = self._header_value(headers, "content-disposition")
        if not disposition:
            return ""
        match = re.search(r"filename\*\s*=\s*(?:UTF-8''|\"?)([^\";]+)", disposition, flags=re.I)
        if not match:
            match = re.search(r"filename\s*=\s*\"?([^\";]+)", disposition, flags=re.I)
        if not match:
            return ""
        filename = unquote(match.group(1).strip().strip('"'))
        return self._normalize_supported_suffix(Path(filename).suffix)

    @staticmethod
    def _looks_like_csv_or_tsv(head: bytes) -> bool:
        try:
            text = head.decode("utf-8", errors="ignore")
        except Exception:
            return False
        lines = [line for line in text.splitlines()[:5] if line.strip()]
        if len(lines) < 2:
            return False
        comma_counts = [line.count(",") for line in lines]
        tab_counts = [line.count("\t") for line in lines]
        return max(comma_counts or [0]) >= 2 or max(tab_counts or [0]) >= 2

    @staticmethod
    def _header_value(headers: Dict[str, str], name: str) -> str:
        target = name.lower()
        for key, value in (headers or {}).items():
            if str(key).lower() == target:
                return str(value or "")
        return ""

    @staticmethod
    def _model_dump(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}
        if hasattr(obj, "dict"):
            try:
                dumped = obj.dict()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return cls._json_safe(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return cls._json_safe(value.dict())
            except Exception:
                pass
        if hasattr(value, "value"):
            return cls._json_safe(value.value)
        return str(value)

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    @classmethod
    def _clean_value(cls, value: Any) -> Any:
        if value is None:
            return ""
        try:
            if isinstance(value, float) and math.isnan(value):
                return ""
        except TypeError:
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return cls._stringify(value)

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _non_negative_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    @staticmethod
    def _cell_limit_for_preview(preview_rows: int) -> int:
        return max(500, min(5000, preview_rows * 80))

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _docling_version() -> str:
        try:
            return importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
