from __future__ import annotations

from typing import Dict, Type

from .base import BenchmarkRunner
from .bizfinbench_v2 import BizFinBenchV2Runner
from .financeagentbench import FinanceAgentBenchRunner, FinanceAgentV2Runner
from .financebench import FinanceBenchClosedBookRunner, FinanceBenchRunner
from .fin_deepsearch_sft import FinDeepSearchSFTRunner
from .findocresearch import FinDocResearchRunner
from .fingaia import FinGAIARunner
from .finrpt import FinRptRunner
from .finsearchcomp import FinSearchCompRunner
from .rfc_bench import RFCBenchRunner
from .rfc_task2 import RFCTask2Runner


CANONICAL_RUNNERS: Dict[str, Type[BenchmarkRunner]] = {
    "fingaia": FinGAIARunner,
    "fin_deepsearch_sft": FinDeepSearchSFTRunner,
    "finsearchcomp": FinSearchCompRunner,
    "financeagentbench": FinanceAgentBenchRunner,
    "financeagent_v2": FinanceAgentV2Runner,
    "finrpt": FinRptRunner,
    "findocresearch": FinDocResearchRunner,
    "rfc_bench": RFCBenchRunner,
    "rfc_task2": RFCTask2Runner,
    "bizfinbench_v2": BizFinBenchV2Runner,
    "financebench": FinanceBenchRunner,
    "financebench_closed_book": FinanceBenchClosedBookRunner,
}


BENCH_ALIASES: Dict[str, str] = {
    "financeagent": "financeagentbench",
    "finance-agent": "financeagentbench",
    "finance_agent": "financeagentbench",
    "finance-agent-v2": "financeagent_v2",
    "finance_agent_v2": "financeagent_v2",
    "financeagentv2": "financeagent_v2",
    "fin-gaia": "fingaia",
    "fin_gaia": "fingaia",
    "fin-deepsearch-sft": "fin_deepsearch_sft",
    "findeepsearch-sft": "fin_deepsearch_sft",
    "findeepsearch_sft": "fin_deepsearch_sft",
    "fin-deep-search-sft": "fin_deepsearch_sft",
    "fin_deep_search_sft": "fin_deepsearch_sft",
    "deepsearch_sft": "fin_deepsearch_sft",
    "deepsearch-sft": "fin_deepsearch_sft",
    "fin-search-comp": "finsearchcomp",
    "fin_search_comp": "finsearchcomp",
    "finreport": "finrpt",
    "fin-rpt": "finrpt",
    "findocresearch": "findocresearch",
    "findoc-research": "findocresearch",
    "findoc_research": "findocresearch",
    "openfinarena-findocresearch": "findocresearch",
    "rfc-bench": "rfc_bench",
    "rfcbench": "rfc_bench",
    "rfc-task2": "rfc_task2",
    "rfctask2": "rfc_task2",
    "rfc_task_2": "rfc_task2",
    "rfc-task-2": "rfc_task2",
    "bizfinbench-v2": "bizfinbench_v2",
    "bizfinbench.v2": "bizfinbench_v2",
    "bizfinbenchv2": "bizfinbench_v2",
    "finance-bench": "financebench",
    "finance_bench": "financebench",
    "financebench-evidence": "financebench",
    "financebench_evidence": "financebench",
    "financebench-closed-book": "financebench_closed_book",
    "financebench_closedbook": "financebench_closed_book",
    "finance-bench-closed-book": "financebench_closed_book",
}


def canonical_bench_name(name: str) -> str:
    return BENCH_ALIASES.get(name, name)


RUNNER_ALIASES: Dict[str, str] = dict(BENCH_ALIASES)

RUNNER_REGISTRY: Dict[str, Type[BenchmarkRunner]] = {}
for name, runner_cls in CANONICAL_RUNNERS.items():
    RUNNER_REGISTRY[name] = runner_cls
for alias, canonical in RUNNER_ALIASES.items():
    if canonical in CANONICAL_RUNNERS:
        RUNNER_REGISTRY[alias] = CANONICAL_RUNNERS[canonical]

BENCHMARK_SPECS = {
    name: {
        "bench": name,
        "description": runner_cls.description,
        "task_family": runner_cls.task_family,
        "evaluator": runner_cls.evaluator,
        "default_tools": list(runner_cls.default_tools or ()),
    }
    for name, runner_cls in CANONICAL_RUNNERS.items()
}
