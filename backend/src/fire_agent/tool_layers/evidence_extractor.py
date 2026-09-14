from __future__ import annotations

from typing import Optional

from ..config import ModelSettings
from ..llm import OpenAICompatibleChatModel
from ..schemas import compact_text


class EvidenceExtractor:
    """Lossless, full-document evidence extraction used by reader tools.

    Aligned with finance-agent's ``RetrieveInformation``: by default the entire
    stored document is handed to the LLM (no query-focused pre-filtering, no
    lexical chunk dropping). The agent may optionally scope a single explicit
    character range (``start``/``end``, end-exclusive) to avoid token limits on
    very large filings, mirroring finance-agent's ``input_character_ranges``.
    """

    def __init__(self) -> None:
        self._model: Optional[OpenAICompatibleChatModel] = None

    def extract_text_evidence(
        self,
        content: str,
        query: str,
        source: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> str:
        content = content or ""
        query = str(query or "").strip()
        scoped = self._slice_content(content, start, end)
        model = self._get_model()
        if model is None:
            return self._fallback_text(scoped)
        prompt = (
            "Task: Extract evidence from the document for the query, in TWO labeled sections.\n"
            f"Query: {query or '(no specific query)'}\n"
            f"Source: {source}\n\n"
            f"Document content:\n{scoped}\n\n"
            "Produce your output as exactly these two sections, in this order:\n\n"
            "EVIDENCE (verbatim, exhaustive):\n"
            "- Reproduce the full original context relevant to the query; never miss any important information.\n"
            "- Cover every requested entity, period, metric, class, definition, range endpoint, and comparison point visible in the document.\n"
            "- Preserve exact numbers, dates, units, currencies, percentages, basis points, signs, table labels, and source context; keep table-like relationships as labeled rows/values verbatim rather than condensing them into prose.\n"
            "- Enumeration completeness: when the query or the document implies a set of members "
            "(for example share classes, segments, business units, reportable categories, line items, "
            "reconciliation adjustments, components, competitors, or a multi-metric guidance list), reproduce "
            "EVERY member of that set, including members whose value is zero, nil, empty, immaterial, or "
            "seemingly minor. Never drop a member because it looks small, zero, or less relevant; a missing "
            "member makes the answer wrong.\n"
            "- Scope anchor: when the query uses a term whose scope the document itself defines (for example "
            "the filer's own stated competitor/peer set, the composition of a named aggregate such as 'total "
            "debt'/'all debt', or which fiscal periods the filing actually reports), reproduce that "
            "source-stated definition/membership/period coverage verbatim, including any document-stated "
            "total/subtotal, so the caller anchors ambiguous scope to the source rather than to a prior assumption.\n"
            "- If a requested field is not present in the supplied content, explicitly mark it as 'not found' instead of omitting it.\n\n"
            "SUMMARY (concise):\n"
            "- A short synthesis that directly answers the query, fully consistent with the EVIDENCE above.\n"
            "- Do not introduce any value, member, or figure that is not present in the EVIDENCE section."
        )
        try:
            response = model([{"role": "user", "content": prompt}])
            content_out = response.content if hasattr(response, "content") else str(response)
            return str(content_out).strip()
        except Exception as exc:
            return f"Content extraction failed: {exc}; fallback evidence:\n{self._fallback_text(scoped)}"

    @staticmethod
    def _slice_content(content: str, start: Optional[int], end: Optional[int]) -> str:
        """Apply an optional character range (start inclusive, end exclusive).

        Mirrors finance-agent semantics: when neither bound is provided the full
        document is returned unchanged (lossless).
        """

        if start is None and end is None:
            return content
        length = len(content)
        try:
            start_idx = 0 if start is None else max(0, int(start))
        except (TypeError, ValueError):
            start_idx = 0
        try:
            end_idx = length if end is None else int(end)
        except (TypeError, ValueError):
            end_idx = length
        if end_idx < 0:
            end_idx = length
        end_idx = min(end_idx, length)
        if end_idx <= start_idx:
            return ""
        return content[start_idx:end_idx]

    def _get_model(self) -> Optional[OpenAICompatibleChatModel]:
        if self._model is not None:
            return self._model
        settings = ModelSettings.from_role_env("EVIDENCE")
        if not settings.is_configured:
            return None
        self._model = OpenAICompatibleChatModel(settings)
        return self._model

    @staticmethod
    def _fallback_text(content: str) -> str:
        """Resilience fallback when the extraction LLM is unavailable/failing.

        Returns the (already range-scoped) raw text bounded by the shared
        head+tail ``compact_text`` budget. No query-focused filtering is applied,
        so no requested slot is silently dropped on lexical grounds.
        """

        text = compact_text(content)
        return text if text.strip() else "No content available"
