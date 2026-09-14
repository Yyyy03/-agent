from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .tool_layers import (
    AkShareTool,
    CalculatorTool,
    ContentReaderTool,
    DataFrameQueryTool,
    EvidenceExtractor,
    LocalCalculatorTool,
    MarketDataTool,
    MultimodalReaderTool,
    MultimodalUnderstandingTool,
    ReaderDocumentStore,
    READER_DOCUMENT_STORE as _READER_DOCUMENT_STORE,
    SECReaderTool,
    SECSearchTool,
    SearchDiscoveryTool,
    WebReaderTool,
    WebSearchTool,
)
from .tool_layers.finance_utils import (
    DATE_REGEX,
    FINANCE_AGENT_MAX_END_DATE,
    finance_agent_records_to_csv as _finance_agent_records_to_csv,
    finance_agent_retryable_post as _finance_agent_retryable_post,
    finance_agent_validate_date as _finance_agent_validate_date,
)
from .tool_layers.sec_tools import (
    SEC_DEFAULT_RESULT_LIMIT,
    SEC_TABLE_DISCOVERY_LIMIT,
    SEC_TABLE_PREVIEW_ROWS,
    SEC_TABLE_ROW_LIMIT,
)
from .tool_layers.sec_utils import (
    is_sec_accession_landing_url as _is_sec_accession_landing_url,
    is_sec_url as _is_sec_url,
    sec_accession_index_url as _sec_accession_index_url,
    sec_primary_document_url as _sec_primary_document_url,
)
from .tools import ConcreteToolAlias, ConfiguredToolWrapper, StructuredTableReaderTool, ToolActionSpec, ToolFamily, ToolRegistry


DEFAULT_TOOL_ORDER = [
    WebSearchTool,
    WebReaderTool,
    SECSearchTool,
    SECReaderTool,
    StructuredTableReaderTool,
    MarketDataTool,
    DataFrameQueryTool,
    MultimodalReaderTool,
    CalculatorTool,
]


def build_default_tool_registry(
    enabled_tools: Optional[Iterable[str]] = None,
    tool_settings: Optional[Mapping[str, Any]] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_cls in DEFAULT_TOOL_ORDER:
        registry.register(tool_cls())
    if tool_settings:
        for name, settings in tool_settings.items():
            if name not in registry:
                continue
            if not isinstance(settings, Mapping):
                continue
            registry.register(ConfiguredToolWrapper(registry[name], settings), replace=True)
    if enabled_tools is not None:
        requested = set(enabled_tools)
        if "multimodal_understanding" in requested and "multimodal_understanding" not in registry:
            registry.register(
                ConcreteToolAlias(
                    "multimodal_understanding",
                    registry["multimodal_reader"],
                    registry["multimodal_reader"].description,
                )
            )
    return registry.filter(enabled_tools)
