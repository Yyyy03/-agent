from __future__ import annotations

import io
import json
import os
import re
import time
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from ..config import ModelSettings, ToolSettings
from ..http_client import HTTPRequestError, http_get
from ..llm import OpenAICompatibleChatModel
from ..schemas import ToolResult, compact_text
from ..tools import ToolActionSpec, ToolFamily
from .reader_store import READER_DOCUMENT_STORE as _READER_DOCUMENT_STORE
from .sec_utils import is_sec_url as _is_sec_url


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


def _is_slow_dynamic_market_reader_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or ""))
    except Exception:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return host in {"finance.yahoo.com", "hk.finance.yahoo.com"} and "/history" in path


class ContentReaderTool(ToolFamily):
    name = "content_reader"
    description = "Read a known URL or local document file, cache the full text internally for the task, return a document_key, and return query-focused evidence."
    paid = True
    parallel_safe = False
    default_action = "read_url"
    actions = {
        "read_url": ToolActionSpec(
            description="Read a known HTTP(S) URL, or re-read a previously saved URL document_key, and summarize content matching the query.",
            optional=["url", "document_key", "query"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Known non-SEC HTTP(S) URL to read."},
                    "document_key": {"type": "string", "description": "Previously returned URL document_key."},
                    "query": {"type": "string", "description": "Fact, table, or section to extract."},
                },
                "additionalProperties": False,
            },
        ),
        "read_file": ToolActionSpec(
            description="Read a local text or PDF document file, or re-read a previously saved file document_key, and summarize content matching the query.",
            optional=["file_path", "document_key", "query"],
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Local text or PDF path from task attachments."},
                    "document_key": {"type": "string", "description": "Previously returned file document_key."},
                    "query": {"type": "string", "description": "Fact, table, or section to extract."},
                },
                "additionalProperties": False,
            },
        ),
    }

    def __init__(self):
        self._summary_model: Optional[OpenAICompatibleChatModel] = None
        self._document_store = _READER_DOCUMENT_STORE

    def reset_for_task(self) -> None:
        self._document_store.clear()

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        timeout = _env_timeout_seconds("FIRE_AGENT_WEB_READER_TIMEOUT_SECONDS", timeout, 8)
        if action == "read_file":
            return self._read_file(arguments)
        return self._read_url(arguments, timeout=timeout)

    def _read_url(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        settings = ToolSettings.from_env()
        url = str(arguments.get("url", "")).strip()
        query = str(arguments.get("query", "")).strip()
        document_key = str(arguments.get("document_key") or "").strip()
        if document_key:
            cached = self._document_store.get_key(document_key)
            if not cached or cached.get("namespace") != "url":
                return ToolResult(self.name, "reader_cache", "error", action="read_url", error=f"Unknown URL document_key: {document_key}", paid=False, confidence=0.1)
            url = str(cached.get("source") or url).strip()
            return self._summarize_result(
                content=str(cached.get("content") or ""),
                query=query,
                url=url,
                provider=str((cached.get("metadata") or {}).get("provider") or "reader_cache"),
                action="read_url",
                cached=True,
            )
        if not url.startswith(("http://", "https://")):
            return ToolResult(self.name, "jina", "error", action="read_url", error="URL must start with http:// or https://", paid=True)

        if _is_slow_dynamic_market_reader_url(url):
            return ToolResult(
                self.name,
                "reader_fast_fail_dynamic_market_page",
                "error",
                action="read_url",
                error="Skipped slow/dynamic market history page; use market_data, search snippets, or a static data source instead.",
                paid=False,
                confidence=0.1,
                metadata={"url": url, "reason": "slow_dynamic_market_history"},
            )

        cached = self._document_store.get("url", url)
        if cached:
            return self._summarize_result(
                content=str(cached.get("content") or ""),
                query=query,
                url=url,
                provider=str((cached.get("metadata") or {}).get("provider") or "reader_cache"),
                action="read_url",
                cached=True,
            )

        direct_pdf_error = ""
        direct_pdf_attempted = False
        if self._looks_like_pdf_url(url):
            direct_pdf_attempted = True
            direct_result = self._read_remote_pdf_url(url, query, arguments, timeout=timeout)
            if direct_result.status == "success":
                return direct_result
            direct_pdf_error = direct_result.error or ""

        headers = {
            "X-Engine": "browser",
            "X-Return-Format": "markdown",
            "X-Retain-Images": "none",
            "X-Token-Budget": "200000",
            "X-Timeout": "10",
        }
        if settings.jina_api_key:
            headers["Authorization"] = f"Bearer {settings.jina_api_key}"
        try:
            response = http_get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
        except Exception as exc:
            direct_html_error = ""
            if not direct_pdf_attempted:
                direct_html_result = self._read_remote_html_url(url, query, arguments, timeout=timeout)
                if direct_html_result.status == "success":
                    return direct_html_result
                direct_html_error = direct_html_result.error or ""
                direct_result = self._read_remote_pdf_url(url, query, arguments, timeout=timeout)
                if direct_result.status == "success":
                    return direct_result
                direct_pdf_error = direct_result.error or ""
            jina_error = f"{type(exc).__name__}: {exc}"
            fallback_errors = []
            if direct_html_error:
                fallback_errors.append(f"Direct HTML fetch failed: {direct_html_error}")
            if direct_pdf_error:
                fallback_errors.append(f"Direct PDF fetch failed: {direct_pdf_error}")
            error = "; ".join([*fallback_errors, f"Jina reader failed: {jina_error}"])
            return ToolResult(
                self.name,
                "jina",
                "error",
                action="read_url",
                error=error,
                paid=bool(settings.jina_api_key),
                confidence=0.1,
            )
        return self._summarize_result(
            content=response.text,
            query=query,
            url=url,
            provider="jina",
            action="read_url",
        )

    def _read_remote_pdf_url(
        self,
        url: str,
        query: str,
        arguments: Dict[str, Any],
        timeout: int = 30,
    ) -> ToolResult:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,text/html,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = http_get(url, headers=headers, timeout=timeout)
            content_type = response.headers.get("Content-Type") or response.headers.get("content-type") or ""
            is_pdf = "pdf" in content_type.lower() or response.content.startswith(b"%PDF")
            if not is_pdf:
                return ToolResult(
                    self.name,
                    "direct_pdf",
                    "error",
                    action="read_url",
                    error=f"URL did not return PDF content; content-type={content_type or 'unknown'}",
                    paid=False,
                    confidence=0.1,
                )
            text = self._extract_pdf_text_from_bytes(response.content)
        except Exception as exc:
            return ToolResult(
                self.name,
                "direct_pdf",
                "error",
                action="read_url",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
            )

        return self._summarize_result(
            content=text,
            query=query,
            url=url,
            provider="direct_pdf",
            action="read_url",
        )

    def _read_remote_html_url(
        self,
        url: str,
        query: str,
        arguments: Dict[str, Any],
        timeout: int = 30,
    ) -> ToolResult:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }
        try:
            response = http_get(url, headers=headers, timeout=timeout)
            content_type = response.headers.get("Content-Type") or response.headers.get("content-type") or ""
            content = response.text
            if not self._looks_like_html_response(content_type, response.content):
                return ToolResult(
                    self.name,
                    "direct_html",
                    "error",
                    action="read_url",
                    error=f"URL did not return HTML content; content-type={content_type or 'unknown'}",
                    paid=False,
                    confidence=0.1,
                )
            text = self._html_to_text(content)
        except Exception as exc:
            return ToolResult(
                self.name,
                "direct_html",
                "error",
                action="read_url",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
            )

        return self._summarize_result(
            content=text,
            query=query,
            url=url,
            provider="direct_html",
            action="read_url",
        )

    def _read_file(self, arguments: Dict[str, Any]) -> ToolResult:
        file_path = str(arguments.get("file_path", "")).strip()
        query = str(arguments.get("query", "")).strip()
        document_key = str(arguments.get("document_key") or "").strip()
        if document_key:
            cached = self._document_store.get_key(document_key)
            if not cached or cached.get("namespace") != "file":
                return ToolResult(self.name, "reader_cache", "error", action="read_file", error=f"Unknown file document_key: {document_key}", paid=False, confidence=0.1)
            return self._summarize_result(
                content=str(cached.get("content") or ""),
                query=query,
                url=str(cached.get("source") or file_path),
                provider="local_file",
                action="read_file",
                cached=True,
            )
        if not file_path:
            return ToolResult(self.name, "local_file", "error", action="read_file", error="Missing file_path", paid=False)
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            return ToolResult(self.name, "local_file", "error", action="read_file", error=f"File not found: {file_path}", paid=False)

        cached = self._document_store.get("file", str(path))
        if cached:
            return self._summarize_result(
                content=str(cached.get("content") or ""),
                query=query,
                url=str(path),
                provider="local_file",
                action="read_file",
                cached=True,
            )

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = self._extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(
                self.name,
                "local_file",
                "error",
                action="read_file",
                error=f"{type(exc).__name__}: {exc}",
                paid=False,
                confidence=0.1,
            )

        return self._summarize_result(
            content=text,
            query=query,
            url=str(path),
            provider="local_file",
            action="read_file",
        )

    def _summarize_result(
        self,
        content: str,
        query: str,
        url: str,
        provider: str,
        action: str,
        cached: bool = False,
    ) -> ToolResult:
        namespace = "file" if provider == "local_file" else "url"
        record = self._document_store.put(
            namespace,
            url,
            content,
            {"provider": provider, "action": action},
        )
        truncated_content = self.truncate_text(content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        summary, auxiliary_turn = self.retry_predict_with_event(prompt)
        metadata = {
            "document_key": record["key"],
            "stored_chars": record["chars"],
            "cached": cached,
            "url": url,
            "query": query,
            "provider": provider,
            "action": action,
            "namespace": namespace,
        }
        if auxiliary_turn:
            auxiliary_turn["source"] = {
                "document_key": record["key"],
                "document_url": url,
                "document_namespace": namespace,
                "document_provider": provider,
                "document_action": action,
                "document_cached": cached,
                "document_stored_chars": record["chars"],
                "query": query,
            }
            metadata["_auxiliary_turns"] = [auxiliary_turn]
        return ToolResult(
            tool_family=self.name,
            provider=provider,
            status="success",
            action=action,
            observation=summary,
            confidence=0.72,
            paid=True,
            metadata=metadata,
        )

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        """Truncate text to specified length."""
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract query-focused evidence and decision cards from the source for the query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Produce exactly these labeled sections, in this order:\n\n"
            "SOURCE_STATUS:\n"
            "- source_role: evidence | event_date | source_discovery | structured_key_discovery | "
            "blocked_or_unusable | no_relevant_info.\n"
            "- State whether the page is usable source content, a search/landing/challenge/login/error page, "
            "or only a pointer to another source.\n"
            "- If blocked or unusable, name the visible reason and do not invent values from snippets or page chrome.\n\n"
            "ANSWER_SLOT_CARD:\n"
            "- Compact rows in the form slot_hint | status(supported/candidate/missing/blocked) | entity | "
            "period/date | metric/field | value | unit | source locator.\n"
            "- Include only values directly present in the source. If the page does not answer any requested slot, write 'none'.\n\n"
            "ENTITY_RESOLUTION_CARD:\n"
            "- Compact rows in the form raw_entity | resolved_id/ticker/code | asset_type | market/exchange | "
            "confidence(high/medium/low) | source locator.\n"
            "- Use this for explicit tickers, company names, fund codes, index codes, series IDs, event dates, "
            "table URLs, or official-source links found in the source. If none, write 'none'.\n\n"
            "WEB_SOURCE_CARD:\n"
            "- source_role | official_or_primary_url | extracted_structured_keys | should_switch_to_structured_tool(yes/no) | "
            "blocked_domain_or_url(yes/no) | reason.\n"
            "- Set should_switch_to_structured_tool=yes when the page reveals a ticker/code/date/source key but the requested "
            "answer is an exact price, return, volume, market cap, ranking, or structured table value.\n\n"
            "STRUCTURED_NEXT:\n"
            "- recommended_tool.action | arguments_hint | reason.\n"
            "- Recommend a structured next step only when the source explicitly provides enough keys, such as ticker/code, "
            "date range, table URL, filing URL, series ID, or official dataset name. Otherwise write 'none'.\n\n"
            "FAILURE_CARD:\n"
            "- failure_class: none | id_mismatch | ambiguous_id | non_trading_date | field_missing | no_rows | "
            "unsupported | provider_error | blocked | rate_limited | no_relevant_info.\n"
            "- failed_source | visible_reason | retry_same_source_allowed(yes/no) | suggested_recovery.\n"
            "- Use retry_same_source_allowed=no for access blocks, challenge pages, repeated empty pages, or pages that "
            "only expose source-discovery keys but not the requested numeric evidence.\n\n"
            "EVIDENCE:\n"
            "- Reproduce only source text, table rows, lists, definitions, caveats, and labels that are relevant to the query.\n"
            "- Preserve exact numbers, dates, units, currencies, table labels, ranges, definitions, and source wording; do not round or reformat source values.\n"
            "- Enumeration completeness: when the query or page implies a set of members (share classes, "
            "segments, categories, line items, reconciliation adjustments, components, competitors, or a "
            "multi-metric list), reproduce EVERY member, including zero/nil/empty/immaterial ones; never drop a "
            "member for looking small or less relevant.\n"
            "- When the source contains a structured block or table that answers the query, reproduce the "
            "complete set of labeled rows/values verbatim (with units), including any source-stated total/subtotal.\n"
            "- Scope anchor: if the query uses a term whose scope is defined by the source itself (for "
            "example the company's own stated competitor/peer set, or the composition of a named aggregate "
            "such as 'total debt' / 'all debt'), reproduce that source-stated definition or membership "
            "verbatim so the caller anchors the scope to the document rather than guessing.\n"
            "- If no relevant information exists, output 'No relevant information' in this section.\n\n"
            "KEY_VALUES:\n"
            "- Compact rows in the form entity | period/date | metric/field | value | unit | source locator.\n"
            "- Include table names, row labels, and any unit/scale note needed to audit the value.\n"
            "- If the source has no relevant values, write 'none'.\n\n"
            "COVERAGE:\n"
            "- State which requested slots are answered by this source, which remain missing or ambiguous, and whether "
            "the next action should be final_answer, calculator, structured data, narrower document read, or source gap.\n\n"
            "SUMMARY:\n"
            "- One short synthesis directly answering the query from the evidence above.\n"
            "- Do not introduce any value, member, code, URL, or figure that is not present in the cards, EVIDENCE, or KEY_VALUES."
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        summary, _ = self.retry_predict_with_event(prompt, max_retries=max_retries)
        return summary

    def retry_predict_with_event(
        self,
        prompt: str,
        max_retries: int = 3,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Retry model prediction with exponential backoff."""
        model = self._get_summary_model()
        if model is None:
            return (
                "Content extraction failed: missing FIRE_AGENT_API_KEY, FIRE_AGENT_BASE_URL, or FIRE_AGENT_MODEL",
                None,
            )
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                response = model(messages)
                content = getattr(response, "content", None)
                if content is None:
                    content = str(response)
                summary = content.strip() if isinstance(content, str) else str(content)
                return summary, self._build_content_reader_auxiliary_turn(messages, response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Content extraction failed: {str(e)}", None
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        return "Content extraction failed after multiple attempts", None

    @staticmethod
    def _build_content_reader_auxiliary_turn(
        messages: List[Dict[str, Any]],
        response: Any,
    ) -> Optional[Dict[str, Any]]:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        usage = getattr(response, "usage", {}) or {}
        if not isinstance(usage, dict):
            usage = {}
        return {
            "type": "content_reader_summary",
            "module": "content_reader",
            "target_type": "content_reader_summary",
            "render_mode": "tool_internal",
            "messages": messages,
            "response": {
                "reasoning_content": str(getattr(response, "reasoning_content", "") or ""),
                "content": content if isinstance(content, str) else str(content),
                "usage": {
                    "prompt_tokens": ContentReaderTool._usage_int(usage.get("prompt_tokens")),
                    "completion_tokens": ContentReaderTool._usage_int(usage.get("completion_tokens")),
                    "total_tokens": ContentReaderTool._usage_int(usage.get("total_tokens")),
                },
            },
            "include_in_sft": True,
            "source": {},
        }

    @staticmethod
    def _usage_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _get_summary_model(self) -> Optional[OpenAICompatibleChatModel]:
        if self._summary_model is not None:
            return self._summary_model
        model_settings = ModelSettings.from_role_env("CONTENT_SUMMARY")
        if not model_settings.is_configured:
            return None
        self._summary_model = OpenAICompatibleChatModel(model_settings)
        return self._summary_model

    def _looks_like_pdf_url(self, url: str) -> bool:
        clean_url = url.split("?", 1)[0].split("#", 1)[0].lower()
        return clean_url.endswith(".pdf")

    def _looks_like_html_response(self, content_type: str, content: bytes) -> bool:
        normalized = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized in {"text/html", "application/xhtml+xml"}:
            return True
        head = (content or b"")[:2048].lstrip().lower()
        return b"<html" in head or b"<!doctype html" in head or b"<table" in head or b"<body" in head

    def _html_to_text(self, content: str) -> str:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content or "", "html.parser")
            for element in soup(["script", "style", "noscript"]):
                element.decompose()
            return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        except Exception:
            text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", content or "")
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            return " ".join(unescape(text).split())

    def _extract_pdf_text(self, path: Path) -> str:
        with path.open("rb") as handle:
            return self._extract_pdf_text_from_bytes(handle.read())

    def _extract_pdf_text_from_bytes(self, content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading local PDFs requires pypdf. Install it with `pip install pypdf`.") from exc

        reader = PdfReader(io.BytesIO(content))
        page_texts: List[str] = []
        page_count = len(reader.pages)
        for page_index in range(page_count):
            page = reader.pages[page_index]
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(f"\n\n--- Page {page_index + 1} ---\n{page_text.strip()}")
        if not page_texts:
            return ""
        return "".join(page_texts).strip()

class WebReaderTool(ContentReaderTool):
    name = "web_reader"
    description = (
        "Read known non-SEC URLs or local PDF/text files using Jina/local readers, cache full source text internally for the task, return a document_key, and return query-focused evidence. "
        "SEC filing URLs from sec.gov, data.sec.gov, or SEC Archives must use sec_reader. "
        "If a direct PDF read reports text/html instead of PDF, treat the URL as a landing/download/challenge page and find a true PDF, SEC exhibit, BusinessWire/PRNewswire mirror, or HTML evidence. "
        "If Jina returns HTTP 403, Cloudflare/Akamai challenge, or error code 1010 for a domain, switch sources instead of retrying that domain."
    )

    def _read_url(self, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        url = str(arguments.get("url", "")).strip()
        document_key = str(arguments.get("document_key") or "").strip()
        if document_key:
            cached = self._document_store.get_key(document_key)
            url = str((cached or {}).get("source") or url).strip()
        if _is_sec_url(url):
            return ToolResult(
                self.name,
                "jina",
                "error",
                action="read_url",
                error="SEC filing URLs must be read with sec_reader, not web_reader.",
                paid=True,
                confidence=0.1,
            )
        return super()._read_url(arguments, timeout=timeout)
