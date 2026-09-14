from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from ..config import ToolSettings
from ..http_client import HTTPRequestError, http_get, http_post
from ..schemas import ToolResult, compact_text
from ..tools import ToolActionSpec, ToolFamily
from .evidence_extractor import EvidenceExtractor
from .finance_utils import (
    DATE_REGEX,
    finance_agent_retryable_post as _finance_agent_retryable_post,
    finance_agent_validate_date as _finance_agent_validate_date,
)
from .reader_store import READER_DOCUMENT_STORE as _READER_DOCUMENT_STORE
from .reader_tools import ContentReaderTool
from .sec_utils import (
    is_sec_accession_landing_url as _is_sec_accession_landing_url,
    is_sec_url as _is_sec_url,
    sec_accession_index_url as _sec_accession_index_url,
    sec_primary_document_url as _sec_primary_document_url,
)

SEC_DEFAULT_RESULT_LIMIT = 100
SEC_TABLE_DISCOVERY_LIMIT = 10
SEC_TABLE_ROW_LIMIT = 100
SEC_TABLE_PREVIEW_ROWS = 5


class SECSearchTool(ToolFamily):
    name = "sec_search"
    description = (
        "Search the SEC EDGAR database through the SEC API. Returns filing metadata and URLs only; "
        "does not read filing values."
    )
    paid = True
    default_action = "full_text_search"
    actions = {
        "full_text_search": ToolActionSpec(
            "Search EDGAR filings and return metadata, not filing body.",
            required=["search_query"],
            optional=["form_types", "ciks", "start_date", "end_date", "page", "top_n_results"],
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": (
                            "Non-empty SEC full-text query. Include the company/ticker, year or period, "
                            "filing type/topic, and the requested fact."
                        ),
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional SEC form filters, for example [\"DEF 14A\"], [\"10-K\"], or [\"8-K\"].",
                    },
                    "ciks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional SEC CIK filters as strings without punctuation.",
                    },
                    "start_date": {"type": "string", "description": "Optional start date in YYYY-MM-DD format."},
                    "end_date": {"type": "string", "description": "Optional end date in YYYY-MM-DD format."},
                    "page": {"type": "integer", "minimum": 1},
                    "top_n_results": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["search_query"],
                "additionalProperties": False,
            },
            examples=[
                {
                    "search_query": "BBSI 2024 DEF 14A board nominees",
                    "form_types": ["DEF 14A"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "top_n_results": 10,
                }
            ],
        ),
    }
    sec_api_url = "https://api.sec-api.io/full-text-search"

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        if action != "full_text_search":
            return ToolResult(
                self.name,
                "sec_api",
                "error",
                action=action,
                error=f"Unsupported action: {action}",
                paid=True,
                confidence=0.1,
            )
        return self._full_text_search(arguments, timeout)

    def _full_text_search(self, arguments: Dict[str, Any], timeout: int) -> ToolResult:
        search_query = str(arguments.get("search_query") or "").strip()
        if not search_query:
            return ToolResult(
                self.name,
                "sec_api",
                "error",
                action="full_text_search",
                error="Missing search_query",
                paid=True,
                confidence=0.1,
            )
        settings = ToolSettings.from_env()
        api_key = settings.sec_api_key
        if not api_key:
            return ToolResult(
                self.name,
                "sec_api",
                "error",
                action="full_text_search",
                error="Missing FIRE_AGENT_SEC_API_KEY, FIRE_AGENT_SEC_EDGAR_API_KEY, or SEC_EDGAR_API_KEY",
                paid=True,
                confidence=0.1,
            )
        form_types = arguments.get("form_types")
        if form_types is not None and not isinstance(form_types, list):
            raise ValueError(f"The parameter form_types must be a list if provided. Was of type {type(form_types)}")
        ciks = arguments.get("ciks")
        if ciks is not None and not isinstance(ciks, list):
            raise ValueError(f"The parameter ciks must be a list if provided. Was of type {type(ciks)}")
        max_end_date = str(arguments.get("max_end_date") or "9999-12-31").strip()
        if not re.match(DATE_REGEX, max_end_date):
            raise ValueError(f"max_end_date {max_end_date!r} is not in yyyy-mm-dd format")
        start_date = _finance_agent_validate_date(arguments.get("start_date") or "1900-01-01", "start_date", max_end_date)
        end_date = _finance_agent_validate_date(arguments.get("end_date") or max_end_date, "end_date", max_end_date)
        if start_date > end_date:
            raise ValueError(f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'")
        payload: Dict[str, Any] = {"query": search_query, "startDate": start_date, "endDate": end_date}
        page = int(arguments.get("page", 1) or 1)
        if page:
            payload["page"] = page
        if form_types:
            payload["formTypes"] = form_types
        if ciks:
            payload["ciks"] = ciks
        response = _finance_agent_retryable_post(
            self.sec_api_url,
            headers={"Content-Type": "application/json", "Authorization": api_key},
            json_body=payload,
            timeout=timeout,
            retry_statuses=(429, 503),
        )
        result_payload = response.json()
        filings = result_payload.get("filings", [])
        top_n_results = int(arguments.get("top_n_results") or 100)
        filings = filings[: min(top_n_results, 100)]
        return ToolResult(
            self.name,
            "sec_api",
            "success",
            action="full_text_search",
            observation=json.dumps(filings, ensure_ascii=False, default=str),
            confidence=0.75,
            paid=True,
            metadata={"endpoint": self.sec_api_url, "payload": payload, "result_count": len(filings)},
        )

class SECReaderTool(ToolFamily):
    name = "sec_reader"
    description = (
        "Read known SEC filing evidence directly from sec.gov/data.sec.gov, cache full source text internally for the task, return a document_key for filing text, and return full-document evidence (optionally scoped by start/end character range). Supports filing text, "
        "HTML tables, and exhibits. Does not search and never uses Jina. "
        "Use sec_reader for every SEC URL and for narrative/qualitative evidence (risk factors, MD&A, footnotes, business descriptions, exhibits) and for any HTML table or line item. "
        "For standardized HISTORICAL GAAP fundamentals by period (revenue, net income, EPS, shares, assets/liabilities/equity, operating/investing/financing cash flow, capex, dividends), prefer market_data.financial_statement, which returns them already labeled by period and unit; fall back to sec_reader tables for any line item it does not cover. "
        "For earnings releases, guidance, shareholder letters, and investor presentations, prefer read_exhibits to locate 8-K EX-99.1/EX-99 exhibits when available."
    )
    paid = False
    parallel_safe = False
    default_action = "read_filing"
    actions = {
        "read_filing": ToolActionSpec(
            "Read a known SEC filing URL/cik/accession/primary_document, or re-read a previously saved SEC document_key. By default the full document is analyzed; optionally pass start/end character indices (end-exclusive) to scope a large filing.",
            optional=["url", "document_key", "cik", "accession", "primary_document", "query", "start", "end"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "SEC filing URL from sec.gov or data.sec.gov."},
                    "document_key": {"type": "string", "description": "Previously returned SEC document_key, for example sec:1."},
                    "cik": {"type": "string", "description": "Company CIK when resolving by accession."},
                    "accession": {"type": "string", "description": "SEC accession number when resolving by CIK/accession."},
                    "primary_document": {"type": "string", "description": "Primary document filename when known."},
                    "query": {"type": "string", "description": "Fact or section to extract from the filing."},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
        ),
        "read_tables": ToolActionSpec(
            "Read and rank HTML tables from a known SEC filing. Caches complete tables losslessly and lists EVERY table "
            "(with its dimensions and columns), showing row previews only for the most relevant ones; every table_index "
            "stays readable. Re-read with document_key, table_index, and query or row_start/row_end for precise rows.",
            optional=["url", "document_key", "cik", "accession", "primary_document", "query", "limit", "table_index", "row_start", "row_end"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "document_key": {"type": "string"},
                    "cik": {"type": "string"},
                    "accession": {"type": "string"},
                    "primary_document": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "table_index": {"type": "integer", "minimum": 0},
                    "row_start": {"type": "integer", "minimum": 0},
                    "row_end": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
        ),
        "read_exhibits": ToolActionSpec(
            f"List and optionally read exhibits from a known SEC filing. Default limit is {SEC_DEFAULT_RESULT_LIMIT}.",
            optional=["url", "cik", "accession", "query", "exhibit_type", "limit"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "cik": {"type": "string"},
                    "accession": {"type": "string"},
                    "query": {"type": "string"},
                    "exhibit_type": {"type": "string", "description": "Optional exhibit type filter, for example EX-99.1."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
        ),
    }

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or ToolSettings.from_env().sec_user_agent
        self._extractor = EvidenceExtractor()
        self._document_store = _READER_DOCUMENT_STORE

    def reset_for_task(self) -> None:
        self._document_store.clear()

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        aliases = {
            "default": self.default_action,
            "read_table": "read_tables",
            "tables": "read_tables",
            "exhibits": "read_exhibits",
        }
        return aliases.get(candidate, candidate)

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        if action == "read_tables":
            return self._read_tables(arguments, timeout)
        if action == "read_exhibits":
            return self._read_exhibits(arguments, timeout)
        return self._read_filing(arguments, timeout)

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate", "Accept": "text/html,text/plain,application/json,application/pdf,*/*"}

    def _sec_get(self, url: str, timeout: int, retries: int = 2) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return http_get(url, headers=self._headers(), timeout=timeout)
            except Exception as exc:
                last_exc = exc
                message = str(exc)
                retryable = (
                    "HTTP 429" in message
                    or "HTTP 503" in message
                    or "HTTP 502" in message
                    or "TimeoutError" in message
                    or "UNEXPECTED_EOF" in message
                    or "EOF occurred" in message
                )
                if not retryable or attempt >= retries:
                    raise
                time.sleep(2 ** attempt)
        raise last_exc or RuntimeError("SEC GET failed")

    def _resolve_url(self, arguments: Dict[str, Any], *, allow_index: bool = True) -> str:
        url = str(arguments.get("url") or "").strip()
        if url:
            if not _is_sec_url(url):
                raise ValueError("sec_reader only reads sec.gov or data.sec.gov URLs")
            return url
        if arguments.get("primary_document") not in (None, ""):
            return _sec_primary_document_url(arguments.get("cik"), arguments.get("accession"), arguments.get("primary_document"))
        if allow_index and arguments.get("cik") not in (None, "") and arguments.get("accession") not in (None, ""):
            return _sec_accession_index_url(arguments.get("cik"), arguments.get("accession"))
        raise ValueError("Missing url or cik+accession(+primary_document)")

    def _read_filing(self, arguments: Dict[str, Any], timeout: int) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        document_key = str(arguments.get("document_key") or "").strip()
        if document_key:
            cached = self._document_store.get_key(document_key)
            if not cached or cached.get("namespace") != "sec":
                return ToolResult(self.name, "reader_cache", "error", action="read_filing", error=f"Unknown SEC document_key: {document_key}", paid=False, confidence=0.1)
            url = str(cached.get("source") or "").strip()
            text = str(cached.get("content") or "")
            cached_hit = True
        else:
            url = self._resolve_url(arguments, allow_index=True)
            if _is_sec_accession_landing_url(url):
                url = self._best_sec_document_url(url, query, timeout) or url
            cached = self._document_store.get("sec", url)
            if cached:
                text = str(cached.get("content") or "")
                cached_hit = True
            else:
                try:
                    text = self._fetch_sec_text(url, timeout)
                except HTTPRequestError as exc:
                    fallback_url = self._fallback_document_url(arguments, query, timeout)
                    if not fallback_url:
                        raise
                    url = fallback_url
                    cached = self._document_store.get("sec", url)
                    if cached:
                        text = str(cached.get("content") or "")
                        cached_hit = True
                    else:
                        text = self._fetch_sec_text(url, timeout)
                        cached_hit = False
                else:
                    cached_hit = False
        record = self._document_store.put(
            "sec",
            url,
            text,
            {"provider": "sec_direct", "action": "read_filing"},
        )
        evidence = self._extractor.extract_text_evidence(
            text,
            query,
            url,
            start=arguments.get("start"),
            end=arguments.get("end"),
        )
        observation = evidence
        enumeration_metadata: Dict[str, Any] = {}
        # For enumeration-style questions (share classes, segments, reconciliation
        # adjustments, multi-line breakdowns, ...), the query-focused text extraction
        # can silently drop minor/zero members. Surface the complete rows of the most
        # relevant HTML table as a supplement so no member is lost. Skip when an
        # explicit character range was requested (the caller is scoping deliberately).
        if (
            self._query_implies_enumeration(query)
            and arguments.get("start") in (None, "")
            and arguments.get("end") in (None, "")
        ):
            try:
                supplement, enumeration_metadata = self._enumeration_table_supplement(url, query, timeout)
            except Exception:
                supplement, enumeration_metadata = "", {}
            if supplement:
                observation = f"{evidence}\n\n{supplement}"
        metadata = {"document_key": record["key"], "stored_chars": record["chars"], "cached": cached_hit, "url": url}
        if enumeration_metadata:
            metadata["enumeration_supplement"] = enumeration_metadata
        return ToolResult(
            self.name,
            "sec_direct",
            "success",
            action="read_filing",
            observation=observation,
            confidence=0.82,
            paid=False,
            metadata=metadata,
        )

    @staticmethod
    def _query_implies_enumeration(query: str) -> bool:
        """Detect queries that require a COMPLETE set of members rather than a single value.

        Cues are generic financial-document concepts (share classes, segment/category
        breakdowns, reconciliation adjustments, multi-item lists, "each"/"all"
        quantifiers), so this applies to any filing-grounded financial task, not a
        specific benchmark.
        """

        text = str(query or "").lower()
        if not text:
            return False
        enumeration_cues = (
            "class",
            "classes",
            "each",
            " all ",
            "every",
            "list",
            "list of",
            "segment",
            "segments",
            "category",
            "categories",
            "breakdown",
            "broken down",
            "by segment",
            "by region",
            "by product",
            "components",
            "line items",
            "adjustment",
            "adjustments",
            "reconcil",  # reconciliation / reconcile
            "outstanding",  # shares outstanding (multi-class)
            "metrics",
            "guide on",
            "guides on",
        )
        return any(cue in f" {text} " for cue in enumeration_cues)

    def _enumeration_table_supplement(self, url: str, query: str, timeout: int) -> Tuple[str, Dict[str, Any]]:
        """Return the complete rows of the most relevant HTML table for enumeration queries.

        Cache-aware: reuses any previously parsed ``sec_tables`` bundle for the URL,
        otherwise parses and stores it (so a later read_tables call hits the cache).
        Bounded by a character budget to avoid flooding the context.
        """

        cached = self._document_store.get("sec_tables", url)
        if cached:
            bundle = json.loads(str(cached.get("content") or "{}"))
            tables = list(bundle.get("tables") or [])
        else:
            tables = self._parse_sec_tables(url, timeout)
            self._document_store.put(
                "sec_tables",
                url,
                json.dumps({"url": url, "tables": tables}, ensure_ascii=False, default=str),
                {"provider": "sec_direct", "action": "read_tables"},
            )
        if not tables:
            return "", {}
        best = max(tables, key=lambda table: self._table_score(table.get("rows") or [], query))
        best_score = self._table_score(best.get("rows") or [], query)
        if best_score <= 0:
            return "", {}
        rows = list(best.get("rows") or [])
        all_indices = list(range(len(rows)))
        formatted = self._format_table_rows(rows, all_indices)
        payload = {
            "table_index": best.get("table_index"),
            "columns": best.get("columns", []),
            "row_count": best.get("row_count", len(rows)),
            "rows": formatted,
        }
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        truncated = False
        if len(rendered) > 6000:
            # Keep header + query-matching rows complete when the full table is large.
            matched = self._matching_row_indices(rows, query)
            keep = self._unique_indices([0, *matched])
            payload["rows"] = self._format_table_rows(rows, keep)
            payload["note"] = (
                "table truncated to header + query-matching rows; re-read the full table with "
                f"sec_reader.read_tables table_index={best.get('table_index')} for any remaining rows"
            )
            rendered = json.dumps(payload, ensure_ascii=False, default=str)
            truncated = True
        supplement = (
            "Enumeration completeness supplement (complete rows of the most relevant table; "
            "use these to ensure no member of the requested set is omitted):\n" + rendered
        )
        return supplement, {
            "table_index": best.get("table_index"),
            "table_score": best_score,
            "row_count": best.get("row_count", len(rows)),
            "truncated": truncated,
        }

    def _fallback_document_url(self, arguments: Dict[str, Any], query: str, timeout: int) -> str:
        if arguments.get("cik") in (None, "") or arguments.get("accession") in (None, ""):
            return ""
        try:
            index_url = _sec_accession_index_url(arguments.get("cik"), arguments.get("accession"))
            candidates = self._list_sec_links(
                index_url,
                timeout,
                query=query or str(arguments.get("primary_document") or ""),
                limit=None,
            )
        except Exception:
            return ""
        requested = str(arguments.get("primary_document") or "").lower()
        if requested:
            requested_stem = re.sub(r"[^a-z0-9]+", "", requested.rsplit(".", 1)[0])
            for item in candidates:
                filename_stem = re.sub(r"[^a-z0-9]+", "", str(item.get("filename") or "").lower().rsplit(".", 1)[0])
                if requested_stem and (requested_stem in filename_stem or filename_stem in requested_stem):
                    return str(item.get("url") or "")
        return str((candidates[0] or {}).get("url") or "") if candidates else ""

    def _best_sec_document_url(self, url: str, query: str, timeout: int) -> str:
        try:
            candidates = self._list_sec_links(url, timeout, query=query, limit=None)
        except Exception:
            return ""
        for item in candidates:
            candidate_url = str(item.get("url") or "")
            clean_path = urlparse(candidate_url).path.lower()
            if clean_path.endswith((".htm", ".html", ".txt", ".pdf")):
                return candidate_url
        return str((candidates[0] or {}).get("url") or "") if candidates else ""

    def _fetch_sec_text(self, url: str, timeout: int) -> str:
        response = self._sec_get(url, timeout)
        content_type = response.headers.get("Content-Type") or response.headers.get("content-type") or ""
        clean_url = url.split("?", 1)[0].split("#", 1)[0].lower()
        if "pdf" in content_type.lower() or clean_url.endswith(".pdf") or response.content.startswith(b"%PDF"):
            return ContentReaderTool()._extract_pdf_text_from_bytes(response.content)
        text = response.text
        if "html" in content_type.lower() or "<html" in text[:1000].lower():
            try:
                from bs4 import BeautifulSoup
            except ImportError as exc:
                raise RuntimeError("SEC HTML reading requires beautifulsoup4") from exc
            soup = BeautifulSoup(text, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            lines = (line.strip() for line in soup.get_text("\n").splitlines())
            return "\n".join(line for line in lines if line)
        return text

    def _read_tables(self, arguments: Dict[str, Any], timeout: int) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        is_row_read = arguments.get("table_index") not in (None, "")
        default_limit = SEC_TABLE_ROW_LIMIT if is_row_read else SEC_TABLE_DISCOVERY_LIMIT
        limit = int(arguments.get("limit", default_limit) or default_limit)
        record, tables, url, cached_hit = self._load_sec_tables(arguments, query, timeout)
        if is_row_read:
            return self._read_table_rows(arguments, tables, record, url, query, cached_hit, limit)

        # Score every table (cheap) and keep ALL of them visible so the model
        # never loses awareness of a table on a filing with many tables. The
        # ``limit`` now only gates how many tables get the heavier row preview;
        # the long tail is still listed with its dimensions/columns and remains
        # fully readable by table_index + row_start/row_end. This keeps
        # read_tables consistent with read_filing's lossless-by-default design.
        scored = [
            (self._table_score(table["rows"], query), table) for table in tables
        ]
        scored.sort(key=lambda item: (-item[0], item[1]["table_index"]))

        catalogue: List[Dict[str, Any]] = []
        for rank, (score, table) in enumerate(scored):
            entry: Dict[str, Any] = {
                "table_index": table["table_index"],
                "score": score,
                "row_count": table["row_count"],
                "column_count": table["column_count"],
                "columns": table["columns"],
            }
            if table.get("unit_hint"):
                entry["unit_hint"] = table["unit_hint"]
            if table.get("period_hint"):
                entry["period_hint"] = table["period_hint"]
            if rank < limit:
                preview = self._table_preview(table, query=query)
                entry["matching_row_numbers"] = preview["matching_row_numbers"]
                entry["preview_rows"] = preview["rows"]
            catalogue.append(entry)

        previewed = sum(1 for entry in catalogue if "preview_rows" in entry)
        observation = {
            "message": (
                f"Found {len(tables)} tables; all are listed below with their dimensions and columns. "
                f"Row previews are shown for the top {previewed} by relevance, but every table_index is readable. "
                "Use document_key + table_index with query or row_start/row_end to read precise rows from any table."
            ),
            "document_key": record["key"],
            "url": url,
            "tables": catalogue,
        }
        return ToolResult(
            self.name,
            "sec_direct",
            "success",
            action="read_tables",
            observation=json.dumps(observation, ensure_ascii=False, default=str),
            tables=catalogue,
            confidence=0.78 if catalogue else 0.35,
            paid=False,
            metadata={
                "document_key": record["key"],
                "document_kind": "sec_tables",
                "stored_chars": record["chars"],
                "cached": cached_hit,
                "url": url,
                "table_count": len(tables),
                "returned": len(catalogue),
                "previewed": previewed,
                "query": query,
            },
        )

    def _load_sec_tables(
        self,
        arguments: Dict[str, Any],
        query: str,
        timeout: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, bool]:
        document_key = str(arguments.get("document_key") or "").strip()
        if document_key:
            record = self._document_store.get_key(document_key)
            if not record or record.get("namespace") != "sec_tables":
                raise ValueError(f"Unknown SEC tables document_key: {document_key}")
            bundle = json.loads(str(record.get("content") or "{}"))
            return record, list(bundle.get("tables") or []), str(bundle.get("url") or record.get("source") or ""), True

        url = self._resolve_url(arguments)
        if _is_sec_accession_landing_url(url):
            url = self._best_sec_document_url(url, query, timeout) or url
        cached = self._document_store.get("sec_tables", url)
        if cached:
            bundle = json.loads(str(cached.get("content") or "{}"))
            return cached, list(bundle.get("tables") or []), url, True

        tables = self._parse_sec_tables(url, timeout)
        bundle = {"url": url, "tables": tables}
        record = self._document_store.put(
            "sec_tables",
            url,
            json.dumps(bundle, ensure_ascii=False, default=str),
            {"provider": "sec_direct", "action": "read_tables"},
        )
        return record, tables, url, False

    def _parse_sec_tables(self, url: str, timeout: int) -> List[Dict[str, Any]]:
        response = self._sec_get(url, timeout)
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("SEC table reading requires beautifulsoup4") from exc
        soup = BeautifulSoup(response.text, "html.parser")
        tables = []
        for index, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            normalized = self._normalize_table_rows(rows)
            columns = self._table_columns(normalized)
            caption_el = table.find("caption")
            caption = caption_el.get_text(" ", strip=True) if caption_el else ""
            unit_hint, period_hint = self._extract_table_labels(caption, normalized)
            tables.append(
                {
                    "table_index": index,
                    "row_count": len(normalized),
                    "column_count": max((len(row) for row in normalized), default=0),
                    "columns": columns,
                    "unit_hint": unit_hint,
                    "period_hint": period_hint,
                    "rows": normalized,
                }
            )
        return tables

    @staticmethod
    def _extract_table_labels(
        caption: str,
        normalized: List[List[str]],
    ) -> Tuple[Optional[str], List[str]]:
        """Surface the scale/currency note and period columns of an SEC table.

        SEC financial tables almost always state their scale ("in millions"),
        currency, and the periods they cover in the caption or the first header
        rows, but that context is easy to lose once the model reads individual
        numeric rows. We extract it explicitly so a value is never silently read
        at the wrong order of magnitude or attributed to the wrong period. This
        is a faithfulness aid only — it adds labels, it does not interpret them.
        """
        header_text = " ".join(
            " ".join(cell for cell in row if cell) for row in normalized[:4]
        )
        scan = f"{caption} {header_text}".strip()
        lowered = scan.lower()

        unit_parts: List[str] = []
        scale_match = re.search(r"in\s+(thousand|million|billion)s?", lowered)
        if scale_match:
            unit_parts.append(f"in {scale_match.group(1)}s")
        if "except per share" in lowered or "per share" in lowered:
            unit_parts.append("per-share amounts unscaled")
        if "$" in scan or re.search(r"\b(usd|us dollars?)\b", lowered):
            unit_parts.append("USD")
        elif "€" in scan or re.search(r"\beur\b", lowered):
            unit_parts.append("EUR")
        elif "£" in scan or re.search(r"\bgbp\b", lowered):
            unit_parts.append("GBP")
        elif re.search(r"%", scan):
            unit_parts.append("percent")
        unit_hint = "; ".join(dict.fromkeys(unit_parts)) or None

        periods: List[str] = []
        period_context = re.search(
            r"((?:year|quarter|three|six|nine|twelve)[\w ]*?ended[\w ,]*?\d{4})",
            lowered,
        )
        if period_context:
            periods.append(period_context.group(1).strip())
        for row in normalized[:4]:
            for cell in row:
                for year in re.findall(r"\b(?:19|20)\d{2}\b", cell):
                    if year not in periods:
                        periods.append(year)
        # Keep period hints compact and ordered (caption phrase first, then years).
        deduped: List[str] = []
        for item in periods:
            if item not in deduped:
                deduped.append(item)
        return unit_hint, deduped[:8]

    def _read_table_rows(
        self,
        arguments: Dict[str, Any],
        tables: List[Dict[str, Any]],
        record: Dict[str, Any],
        url: str,
        query: str,
        cached_hit: bool,
        limit: int,
    ) -> ToolResult:
        table_index = int(arguments.get("table_index"))
        table = next((item for item in tables if int(item.get("table_index", -1)) == table_index), None)
        if table is None:
            available = [item.get("table_index") for item in tables]
            return ToolResult(
                self.name,
                "sec_direct",
                "error",
                action="read_tables",
                error=f"Unknown table_index {table_index}. Available table_index values: {available}",
                paid=False,
                confidence=0.1,
            )

        rows = list(table.get("rows") or [])
        row_count = len(rows)
        row_start = arguments.get("row_start")
        row_end = arguments.get("row_end")
        if row_start not in (None, "") or row_end not in (None, ""):
            start = max(1, int(row_start or 1))
            end = min(row_count, int(row_end)) if row_end not in (None, "") else min(row_count, start + max(1, limit) - 1)
            if end < start:
                return ToolResult(self.name, "sec_direct", "error", action="read_tables", error="row_end must be greater than or equal to row_start", paid=False, confidence=0.1)
            selected_indices = list(range(start - 1, end))
            selection = {"mode": "row_range", "row_start": start, "row_end": end}
        elif query:
            matched = self._matching_row_indices(rows, query)
            selected_indices = matched[: max(1, limit)]
            selection = {"mode": "query", "query": query, "matched_row_count": len(matched), "returned_limit": max(1, limit)}
        else:
            end = min(row_count, max(1, limit or SEC_TABLE_ROW_LIMIT))
            selected_indices = list(range(0, end))
            selection = {"mode": "head", "row_start": 1 if row_count else 0, "row_end": end, "returned_limit": end}

        returned_rows = self._format_table_rows(rows, selected_indices)
        observation = {
            "message": "Rows are 1-based and row_end is inclusive. Use row_start/row_end to continue reading this table.",
            "document_key": record["key"],
            "url": url,
            "table_index": table_index,
            "row_count": row_count,
            "column_count": table.get("column_count", 0),
            "columns": table.get("columns", []),
            "unit_hint": table.get("unit_hint"),
            "period_hint": table.get("period_hint") or [],
            "selection": selection,
            "returned_rows": returned_rows,
        }
        return ToolResult(
            self.name,
            "sec_direct",
            "success",
            action="read_tables",
            observation=json.dumps(observation, ensure_ascii=False, default=str),
            tables=[observation],
            confidence=0.82 if returned_rows else 0.35,
            paid=False,
            metadata={
                "document_key": record["key"],
                "document_kind": "sec_tables",
                "stored_chars": record["chars"],
                "cached": cached_hit,
                "url": url,
                "table_count": len(tables),
                "table_index": table_index,
                "returned_rows": len(returned_rows),
                "query": query,
            },
        )

    def _normalize_table_rows(self, rows: List[List[str]]) -> List[List[str]]:
        max_cols = max(len(row) for row in rows)
        return [row + [""] * (max_cols - len(row)) for row in rows]

    def _table_columns(self, normalized: List[List[str]]) -> List[str]:
        if not normalized:
            return []
        header = normalized[0]
        has_header = any(cell and not re.fullmatch(r"[-+]?[$%(),.\d\s]+", cell) for cell in header)
        if has_header:
            return [cell or f"col_{i + 1}" for i, cell in enumerate(header)]
        return [f"col_{i + 1}" for i in range(len(header))]

    def _table_preview(self, table: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        rows = list(table.get("rows") or [])
        matched = self._matching_row_indices(rows, query)
        if matched:
            preview_indices = self._unique_indices([0, *matched])[:SEC_TABLE_PREVIEW_ROWS]
        else:
            preview_indices = list(range(min(len(rows), SEC_TABLE_PREVIEW_ROWS)))
        return {
            "matching_row_numbers": [index + 1 for index in matched[:20]],
            "rows": self._format_table_rows(rows, preview_indices),
        }

    def _matching_row_indices(self, rows: List[List[str]], query: str) -> List[int]:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9$%.,_-]*", query) if len(term) > 2]
        if not terms:
            return []
        matched = []
        for index, row in enumerate(rows):
            text = " ".join(row).lower()
            if any(term in text for term in terms):
                matched.append(index)
        return matched

    @staticmethod
    def _unique_indices(indices: List[int]) -> List[int]:
        seen = set()
        unique = []
        for index in indices:
            if index in seen:
                continue
            seen.add(index)
            unique.append(index)
        return unique

    @staticmethod
    def _format_table_rows(rows: List[List[str]], indices: List[int]) -> List[Dict[str, Any]]:
        return [{"row": index + 1, "cells": rows[index]} for index in indices if 0 <= index < len(rows)]

    def _table_score(self, rows: List[List[str]], query: str) -> float:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9$%.,_-]*", query) if len(term) > 2]
        if not terms:
            return float(sum(1 for row in rows for cell in row if re.search(r"\d", cell))) / max(1, len(rows))
        score = 0.0
        header = " ".join(rows[0]).lower() if rows else ""
        first_col = " ".join(row[0] for row in rows if row).lower()
        all_text = " ".join(" ".join(row) for row in rows).lower()
        for term in terms:
            if term in header:
                score += 4
            if term in first_col:
                score += 3
            if term in all_text:
                score += 1
        numeric_cells = sum(1 for row in rows for cell in row if re.search(r"\d", cell))
        score += min(numeric_cells / max(1, len(rows)), 5)
        return score

    def _read_exhibits(self, arguments: Dict[str, Any], timeout: int) -> ToolResult:
        url = self._resolve_url(arguments, allow_index=True)
        query = str(arguments.get("query") or "").strip()
        exhibit_type = str(arguments.get("exhibit_type") or "").strip().lower()
        limit = int(arguments.get("limit", SEC_DEFAULT_RESULT_LIMIT) or SEC_DEFAULT_RESULT_LIMIT)
        exhibits = self._list_sec_links(url, timeout, query=query, exhibit_type=exhibit_type, limit=limit)
        return ToolResult(
            self.name,
            "sec_direct",
            "success",
            action="read_exhibits",
            observation=json.dumps(exhibits, ensure_ascii=False, default=str),
            confidence=0.72 if exhibits else 0.3,
            paid=False,
            metadata={"url": url, "result_count": len(exhibits)},
        )

    def _list_sec_links(
        self,
        url: str,
        timeout: int,
        *,
        query: str = "",
        exhibit_type: str = "",
        limit: Optional[int] = SEC_DEFAULT_RESULT_LIMIT,
    ) -> List[Dict[str, str]]:
        response = self._sec_get(url, timeout)
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("SEC exhibit reading requires beautifulsoup4") from exc
        soup = BeautifulSoup(response.text, "html.parser")
        rows: List[Dict[str, str]] = []
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query) if len(term) > 2]
        for link in soup.find_all("a"):
            href = str(link.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            row = link.find_parent("tr")
            label = row.get_text(" ", strip=True) if row else link.get_text(" ", strip=True)
            filename = href.rsplit("/", 1)[-1]
            lower = (href + " " + label).lower()
            looks_relevant = any(token in lower for token in ("ex-", "exhibit", "press", "release", "earnings", ".htm", ".html", ".txt", ".pdf", ".xlsx"))
            if not looks_relevant:
                continue
            if exhibit_type and exhibit_type not in lower:
                continue
            score = self._link_score(lower, terms)
            exhibit_url = urljoin(url, href)
            if _is_sec_url(exhibit_url):
                rows.append({"url": exhibit_url, "description": label, "filename": filename, "score": str(score)})
        rows.sort(key=lambda item: (-float(item.get("score") or 0), item.get("filename") or ""))
        return rows if limit is None else rows[:limit]

    def _link_score(self, lower: str, terms: List[str]) -> float:
        score = 0.0
        for term in terms:
            if term in lower:
                score += 3.0
        if "ex-99" in lower or "ex99" in lower or "99.1" in lower:
            score += 8.0
        if "press" in lower or "release" in lower or "earnings" in lower:
            score += 5.0
        if lower.endswith(('.htm', '.html')):
            score += 1.0
        return score
