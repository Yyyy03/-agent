from __future__ import annotations

# Compatibility facade: concrete evaluators are split by concern.
from .core import BaseEvaluator
from .document_parse import FinDocBenchParseEvaluator
from .qa import (
    ExactOrSubstringQAEvaluator,
    FinGAIAEvaluator,
    FinanceAgentBenchEvaluator,
    FinSearchCompEvaluator,
)
from .reports import FinDocBenchEvaluator, FinDocResearchEvaluator, FinRptEvaluator, ReportQualityEvaluator

__all__ = [
    "BaseEvaluator",
    "ExactOrSubstringQAEvaluator",
    "FinGAIAEvaluator",
    "FinSearchCompEvaluator",
    "FinanceAgentBenchEvaluator",
    "ReportQualityEvaluator",
    "FinRptEvaluator",
    "FinDocResearchEvaluator",
    "FinDocBenchEvaluator",
    "FinDocBenchParseEvaluator",
]
