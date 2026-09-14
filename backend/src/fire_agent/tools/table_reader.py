from __future__ import annotations

from .structured_table_reader import (
    SEC_STRUCTURED_TABLE_READER_ERROR,
    StructuredTableReaderTool,
)


SEC_TABLE_READER_ERROR = SEC_STRUCTURED_TABLE_READER_ERROR


class TableReaderTool(StructuredTableReaderTool):
    """Compatibility shim for the former rule-based table_reader module.

    The inherited tool name remains ``structured_table_reader`` so importing
    this legacy symbol cannot re-register the old rule-based tool family.
    """

