from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urljoin, urlparse

from ..config import ModelSettings, ToolSettings
from ..http_client import HTTPRequestError, http_get, http_post
from ..llm import OpenAICompatibleChatModel
from ..schemas import ToolResult, compact_text
from ..tools import ToolActionSpec, ToolFamily


class MultimodalUnderstandingTool(ToolFamily):
    name = "multimodal_understanding"
    description = "Inspect image, video, and audio attachments with a multimodal model API."
    paid = True
    default_action = "inspect"
    actions = {
        "inspect": ToolActionSpec(
            description="Inspect an image, video, or audio file/URL.",
            optional=["file_path", "url", "question", "max_tokens"],
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "url": {"type": "string"},
                    "question": {"type": "string"},
                    "max_tokens": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        ),
        "inspect_image_table": ToolActionSpec(
            description="Inspect an image-like table and return row/column/cell evidence for a finance question.",
            optional=["file_path", "url", "question", "query", "max_tokens"],
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "url": {"type": "string"},
                    "question": {"type": "string"},
                    "query": {"type": "string"},
                    "max_tokens": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        ),
    }
    mime_overrides = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/mov",
        ".webm": "video/webm",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    audio_format_by_suffix = {
        ".mp3": "mp3",
        ".wav": "wav",
        ".m4a": "mp4",
        ".aac": "aac",
        ".aiff": "aiff",
        ".aif": "aiff",
        ".flac": "flac",
        ".ogg": "ogg",
    }
    image_asset_extensions = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    image_table_hint_terms = (
        "table", "statistical", "statistics", "data table", "chart", "spreadsheet",
        "表", "统计表", "情况表", "数据表", "指标表", "截图", "图片", "图表", "统计信息", "批准预售",
    )
    visual_asset_noise_terms = (
        "logo", "icon", "qrcode", "qr-code", "share", "banner", "wechat", "weibo",
        "jiucuo", "finderror", "red.png", "ga.png", "pe-m-logo",
        "二维码", "扫一扫", "分享", "微信", "微博", "版权", "导航", "找错", "纠错", "备案", "公安",
    )

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        aliases = {
            "default": self.default_action,
            "image": "inspect",
            "url": "inspect",
            "inspect_url": "inspect",
            "image_table": "inspect_image_table",
            "table_image": "inspect_image_table",
            "inspect_table": "inspect_image_table",
        }
        return aliases.get(candidate, candidate)

    def validate_arguments(self, action: str, arguments: Dict[str, Any]) -> Optional[str]:
        base = super().validate_arguments(action, arguments)
        if base:
            return base
        if not (arguments.get("file_path") or arguments.get("url")):
            return f"Missing required argument for {self.name}.{action}: file_path or url"
        return None

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        settings = ToolSettings.from_env()
        api_key = settings.vision_api_key
        base_url = settings.vision_base_url
        model = settings.vision_model
        if not api_key or not base_url or not model:
            return ToolResult(
                self.name,
                "multimodal_model",
                "error",
                action=action,
                error="Missing FIRE_AGENT_VISION_API_KEY, FIRE_AGENT_VISION_BASE_URL, or FIRE_AGENT_VISION_MODEL",
                paid=True,
            )
        source_label, media_bytes, mime_type = self._load_media(action, arguments, timeout=timeout)
        if not mime_type:
            return ToolResult(
                self.name,
                "multimodal_model",
                "error",
                action=action,
                error=f"Unsupported attachment type for multimodal inspection: {source_label or 'unknown source'}",
                paid=True,
            )
        content_part, modality = self._build_media_content_part_from_bytes(media_bytes, mime_type)
        max_tokens = int(arguments.get("max_tokens", 3000 if action == "inspect_image_table" else 2000))
        question = self._question_for_action(action, arguments)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        content_part,
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        response = http_post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json_body=payload,
            timeout=timeout,
        )
        content = response.json()["choices"][0]["message"].get("content", "")
        parsed = self._parse_json_object(content)
        metadata: Dict[str, Any] = {
            "source": source_label,
            "url": str(arguments.get("url") or "").strip(),
            "file_path": str(arguments.get("file_path") or "").strip(),
            "modality": modality,
            "mime_type": mime_type,
            "file_size_bytes": len(media_bytes),
            "model": model,
            "max_tokens": max_tokens,
        }
        if isinstance(parsed, dict):
            metadata["parsed"] = parsed
            candidate_cells = parsed.get("candidate_cells")
            if isinstance(candidate_cells, list):
                metadata["candidate_cells"] = candidate_cells
        return ToolResult(
            self.name,
            "multimodal_model",
            "success",
            action=action,
            observation=compact_text(content, 50000),
            confidence=0.74 if action == "inspect_image_table" else 0.65,
            paid=True,
            metadata=metadata,
        )

    def _load_media(self, action: str, arguments: Dict[str, Any], timeout: int) -> Tuple[str, bytes, str]:
        url = str(arguments.get("url") or "").strip()
        file_path = str(arguments.get("file_path") or "").strip()
        query = str(arguments.get("query") or arguments.get("question") or "").strip()
        if url:
            response = http_get(url, headers=self._media_fetch_headers(), timeout=timeout)
            source = response.url or url
            mime_type = self._guess_mime_type_from_source(source, response.headers)
            if mime_type:
                return source, response.content, mime_type
            if action == "inspect_image_table" and self._looks_like_html(response.content, response.headers, source):
                asset_source = self._select_html_visual_asset(response.content, source, query)
                if asset_source:
                    asset_response = http_get(asset_source, headers=self._media_fetch_headers(), timeout=timeout)
                    resolved_asset = asset_response.url or asset_source
                    return resolved_asset, asset_response.content, self._guess_mime_type_from_source(resolved_asset, asset_response.headers)
            return source, response.content, mime_type
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = path.read_bytes()
        mime_type = self._guess_mime_type(path)
        if mime_type:
            return str(path), content, mime_type
        if action == "inspect_image_table" and self._looks_like_html(content, {}, str(path)):
            asset_source = self._select_html_visual_asset(content, str(path), query)
            if asset_source:
                asset_path = Path(asset_source).expanduser()
                if asset_path.exists():
                    return str(asset_path), asset_path.read_bytes(), self._guess_mime_type(asset_path)
        return str(path), content, mime_type

    @staticmethod
    def _media_fetch_headers() -> Dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def _looks_like_html(self, content: bytes, headers: Mapping[str, Any], source: str) -> bool:
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "").lower()
        if "html" in content_type:
            return True
        suffix = Path(urlparse(source).path).suffix.lower() if source.startswith(("http://", "https://")) else Path(source).suffix.lower()
        if suffix in {".html", ".htm"}:
            return True
        return bytes(content or b"").lstrip()[:64].lower().startswith((b"<!doctype html", b"<html"))

    def _select_html_visual_asset(self, content: bytes, source: str, query: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return ""
        soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
        best_source = ""
        best_score = float("-inf")
        for node in soup.find_all(["img", "a", "source"]):
            label = self._visual_asset_label(node)
            context = self._visual_asset_context(node)
            for candidate in self._visual_asset_candidates(node):
                resolved = self._resolve_visual_asset_source(source, candidate)
                suffix = self._visual_asset_suffix(resolved)
                if suffix not in self.image_asset_extensions:
                    continue
                score = self._visual_asset_score(resolved, label, context, query)
                if score > best_score:
                    best_source = resolved
                    best_score = score
        return best_source if best_score > 0 else ""

    @staticmethod
    def _visual_asset_candidates(node: Any) -> List[str]:
        candidates: List[str] = []
        for attr in ("src", "data-src", "data-original", "data-url", "data-lazy-src", "href"):
            value = str(node.get(attr) or "").strip()
            if value:
                candidates.append(value)
        srcset = str(node.get("srcset") or node.get("data-srcset") or "").strip()
        if srcset:
            for item in srcset.split(","):
                url = item.strip().split(" ", 1)[0].strip()
                if url:
                    candidates.append(url)
        deduped: List[str] = []
        for value in candidates:
            if value and value not in deduped and not value.startswith(("data:", "javascript:", "#")):
                deduped.append(value)
        return deduped

    @staticmethod
    def _visual_asset_label(node: Any) -> str:
        parts = [
            str(node.get("alt") or ""),
            str(node.get("title") or ""),
            str(node.get("aria-label") or ""),
            node.get_text(" ", strip=True) if hasattr(node, "get_text") else "",
        ]
        return compact_text(re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip(), 240)

    @staticmethod
    def _visual_asset_context(node: Any) -> str:
        parent = node.find_parent(["figure", "p", "td", "li", "div"]) if hasattr(node, "find_parent") else None
        if not parent:
            return ""
        return compact_text(re.sub(r"\s+", " ", parent.get_text(" ", strip=True)).strip(), 360)

    def _visual_asset_score(self, source: str, label: str, context: str, query: str) -> float:
        text = f"{source} {label} {context}"
        normalized = text.lower()
        score = 0.0
        has_table_hint = any(term.lower() in normalized for term in self.image_table_hint_terms)
        overlap = self._text_overlap_score(text, query)
        if has_table_hint:
            score += 4.0
        score += overlap * 1.5
        if any(term in normalized for term in self.visual_asset_noise_terms):
            score -= 12.0
        if not has_table_hint and overlap < 2.0:
            score -= 4.0
        return score

    @staticmethod
    def _text_overlap_score(text: str, query: str) -> float:
        normalized = re.sub(r"\s+", " ", str(text or "")).lower()
        terms: List[str] = []
        for term in re.findall(r"[a-z0-9$%._-]+", str(query or "").lower()):
            if len(term) > 2 and term not in terms:
                terms.append(term)
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", str(query or "")):
            if chunk not in terms:
                terms.append(chunk)
            for size in (8, 6, 4, 3, 2):
                if len(chunk) < size:
                    continue
                for idx in range(0, len(chunk) - size + 1):
                    term = chunk[idx : idx + size]
                    if term not in terms:
                        terms.append(term)
                if len(terms) >= 60:
                    break
        score = 0.0
        seen = set()
        for term in terms[:60]:
            term_norm = re.sub(r"\s+", " ", term).lower()
            if term_norm and term_norm not in seen and term_norm in normalized:
                seen.add(term_norm)
                score += min(max(len(term_norm), 2), 8) / 2.0
        return score

    @staticmethod
    def _resolve_visual_asset_source(source: str, href: str) -> str:
        if source.startswith(("http://", "https://")):
            return urljoin(source, href)
        path = Path(source).expanduser()
        return str((path.parent / href).resolve())

    @staticmethod
    def _visual_asset_suffix(source: str) -> str:
        if source.startswith(("http://", "https://")):
            return Path(urlparse(source).path).suffix.lower()
        return Path(source).suffix.lower()

    def _question_for_action(self, action: str, arguments: Dict[str, Any]) -> str:
        question = str(arguments.get("question") or arguments.get("query") or "").strip()
        if action != "inspect_image_table":
            return question or "Describe finance-relevant content in this file."
        return (
            "You are reading an image or screenshot that may contain a financial, regulatory, or statistical table. "
            "Extract only table evidence needed to answer the user question. Return compact JSON with keys: "
            "source_description, table_title, candidate_cells, warnings. Each candidate cell must include "
            "row_label, column_label, value, unit, period, and evidence_text. Do not infer values not visible in the image. "
            f"User question: {question or 'Identify finance-relevant table cells.'}"
        )

    def _guess_mime_type_from_source(self, source: str, headers: Mapping[str, Any]) -> str:
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if content_type and self._is_supported_mime_type(content_type):
            return content_type
        suffix = Path(urlparse(source).path).suffix.lower()
        guessed, _ = mimetypes.guess_type(source)
        for mime_type in (self.mime_overrides.get(suffix), guessed):
            if mime_type and self._is_supported_mime_type(mime_type):
                return mime_type
        return ""

    @staticmethod
    def _parse_json_object(text: str) -> Any:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _guess_mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        guessed, _ = mimetypes.guess_type(str(path))
        for mime_type in (self.mime_overrides.get(suffix), guessed):
            if mime_type and self._is_supported_mime_type(mime_type):
                return mime_type
        return ""

    def _is_supported_mime_type(self, mime_type: str) -> bool:
        return mime_type.startswith(("image/", "video/", "audio/"))

    def _build_media_content_part(self, path: Path, mime_type: str) -> tuple[Dict[str, Any], str]:
        return self._build_media_content_part_from_bytes(path.read_bytes(), mime_type, suffix=path.suffix)

    def _build_media_content_part_from_bytes(self, media: bytes, mime_type: str, suffix: str = "") -> tuple[Dict[str, Any], str]:
        media_b64 = base64.b64encode(media).decode("utf-8")
        data_url = f"data:{mime_type};base64,{media_b64}"
        if mime_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": data_url}}, "image"
        if mime_type.startswith("video/"):
            return {"type": "video_url", "video_url": {"url": data_url}}, "video"
        if mime_type.startswith("audio/"):
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": media_b64,
                    "format": self.audio_format_by_suffix.get(str(suffix).lower(), "mp3"),
                },
            }, "audio"
        raise ValueError(f"Unsupported multimodal MIME type: {mime_type}")

    def _audio_format(self, path: Path) -> str:
        return self.audio_format_by_suffix.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "mp3")

class MultimodalReaderTool(MultimodalUnderstandingTool):
    name = "multimodal_reader"
    description = "Inspect image, video, and audio attachments with the configured multimodal model."
