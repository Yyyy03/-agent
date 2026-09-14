from __future__ import annotations

from typing import Dict

from .core import BaseEvaluator
from .deepsearch_sft import FinDeepSearchSFTEvaluator
from .document_parse import FinDocBenchParseEvaluator
from .new_benchmarks import BizFinBenchV2OfficialEvaluator, FinanceBenchEvaluator, RFCBenchEvaluator, RFCTask2Evaluator
from .qa import ExactOrSubstringQAEvaluator, FinGAIAEvaluator, FinanceAgentBenchEvaluator, FinSearchCompEvaluator
from .reports import FinDocBenchEvaluator, FinDocResearchEvaluator, FinRptEvaluator, ReportQualityEvaluator


EVALUATOR_REGISTRY: Dict[str, BaseEvaluator] = {
    "generic": BaseEvaluator(),
    "fingaia": FinGAIAEvaluator(),
    "finsearchcomp": FinSearchCompEvaluator(),
    "fin_deepsearch_sft": FinDeepSearchSFTEvaluator(),
    "financeagentbench": FinanceAgentBenchEvaluator(),
    "rfc_bench": RFCBenchEvaluator(),
    "rfc_task2": RFCTask2Evaluator(),
    "bizfinbench_v2": BizFinBenchV2OfficialEvaluator(),
    "financebench": FinanceBenchEvaluator(),
    "finrpt": FinRptEvaluator(),
    "findocresearch": FinDocResearchEvaluator(),
}


def get_evaluator(name: str) -> BaseEvaluator:
    return EVALUATOR_REGISTRY.get(name, EVALUATOR_REGISTRY["generic"])


__all__ = [
    "BaseEvaluator",
    "BizFinBenchV2OfficialEvaluator",
    "EVALUATOR_REGISTRY",
    "ExactOrSubstringQAEvaluator",
    "FinDeepSearchSFTEvaluator",
    "FinanceBenchEvaluator",
    "FinGAIAEvaluator",
    "FinDocBenchEvaluator",
    "FinDocBenchParseEvaluator",
    "FinDocResearchEvaluator",
    "FinanceAgentBenchEvaluator",
    "FinRptEvaluator",
    "FinSearchCompEvaluator",
    "RFCBenchEvaluator",
    "RFCTask2Evaluator",
    "ReportQualityEvaluator",
    "get_evaluator",
]
