from __future__ import annotations

from typing import List

from ..aggregation import aggregate_fin_deepsearch_sft_logs
from .base import BenchmarkRunner
from .common import JsonRow, first_present, normalize_text


DATASET_OUTPUT_CONSTRAINTS = {
    "IGPO/train.parquet": (
        "Most answers are short entities or exact numeric values. The first line of final_answer must contain only "
        "the requested final atom: one entity, product type, place, founder, date, or number. Do not answer with an "
        "intermediate entity from the reasoning chain, do not list multiple candidate values, and do not add extra "
        "context before the atom. Put any necessary evidence note after the first line."
    ),
    "PolarSeeker/OpenSeeker-v1-Data": (
        "Most answers are a single entity. The first line of final_answer must be only the entity or short phrase "
        "requested by the question. Do not include side-clue entities, succession chains, biographies, alternate "
        "candidates, or uncertainty wording in the answer line. Put any necessary caveat after the atom."
    ),
    "YqjMartin/AgenticRAGTracer": (
        "Render only the final target requested after the multi-hop comparison or inference: one entity, one value, "
        "or the requested list. Do not list intermediate compared places, companies, people, or multiple candidates "
        "unless the question explicitly asks for them."
    ),
    "Zchu/REDSearcher_SFT_10K": (
        "These are riddle-style deep-search tasks but the final answer is still one gradable atom. The first line of "
        "final_answer must contain only the unique firm, person, instrument, concept, year, title, or value. Do not "
        "end with candidate discussion, ambiguity, 'not fully verified', uncertainty wording, or a reasoning loop. "
        "Choose the best supported answer; use not-found only for a specific missing cell after targeted source attempts."
    ),
    "allenai/MoNaCo_Benchmark": (
        "Match the reference-style numeric or categorical output exactly. For numeric differences, use calculator and "
        "state the final number first, preserving requested decimal style such as 32.0 when applicable. For city/person/"
        "decade answers, output the exact short label first."
    ),
    "callanwu/WebWalkerQA": (
        "Use a concise Chinese answer matching the question scope. For dates, amounts, and counts, preserve Chinese "
        "units and wording. For project/service lists, output only the requested items from the specified page/date/"
        "event, not every related item found during search."
    ),
    "dongguanting/ARPO-RL-DeepSearch-1K": (
        "If the question is binary, final_answer must start with exactly Yes or No. If it asks for an entity/year/value, "
        "start with only that atom. Do not answer Cannot determine unless targeted source attempts show the required "
        "articles or facts are genuinely unavailable."
    ),
    "dongguanting/ARPO-SFT-54K": (
        "For multiple-choice tasks, final_answer must start with the option letter only, such as A, B, C, D, or E. "
        "For non-choice tasks, output every requested field in order, especially paired dates/events, and do not add "
        "unsupported extra entities."
    ),
    "google/deepsearchqa": (
        "Many answers are ordered lists from official data. The final answer must be exactly the requested list, in "
        "the requested order, with no missing or extra members. For single-entity table lookups, the first line must "
        "be only that entity. Do not answer None or not found unless the official source evidence explicitly supports "
        "an empty set or the required table cell is unavailable after targeted reads."
    ),
    "groundhogLLM/ACC-dataset/search_agent": (
        "Start final_answer with the single candidate entity or short answer requested. If the task has multiple "
        "criteria, do not include alternate candidates; keep any criterion caveats after the answer atom."
    ),
    "orbit-ai/orbit-20k": (
        "The first line of final_answer must be the shortest sufficient single riddle answer: one entity, concept, "
        "rate, date, year, code prefix, or value. Do not include multiple candidates or unresolved alternatives. For "
        "parent/subtype answers, lead with the parent concept/code/rate requested by the question and put the subtype "
        "in parentheses only if helpful. For hierarchical codes, output the requested prefix first and mention subcodes "
        "only after the prefix. If Reference URLs conflict with the clue chain, use the clue contract and state the "
        "corrected authoritative basis after the answer atom."
    ),
    "vtllms/sealqa": (
        "Never leave final_answer empty. Preserve strict answer formatting for prices, dates, and entities: keep the "
        "original unit/currency and decimal precision, and use the exact requested date style when the question implies it."
    ),
    "zai-org/DeepDive": (
        "Output the exact final word, name, title, year, or value requested by the deep chain. Do not substitute a "
        "near-synonym for exact-term questions, and do not include speculative chain reconstruction before the answer atom."
    ),
}


class FinDeepSearchSFTRunner(BenchmarkRunner):
    bench_name = "fin_deepsearch_sft"
    description = "Financial deep-search SFT seed QA with answer and trajectory LLM judging."
    task_family = "deep_search_qa"
    evaluator = "fin_deepsearch_sft"
    question_keys = ("question", "prompt")
    answer_keys = ("answer", "final_answer", "golden_answer", "ground_truth")
    sample_id_keys = ("sample_id", "id", "task_id")
    reference_url_keys = (
        "reference_urls",
        "clean_source_urls",
        "clean_reference_urls",
        "source_url_slots",
        "question_url_slots",
        "manual_replacement_urls",
    )

    def build_prompt(self, record: JsonRow, question: str) -> str:
        return self.format_task_prompt(
            question,
            self.seed_url_for_record(record),
            self.reference_urls_for_record(record),
            self.output_constraint_for_record(record),
            self.temporal_constraint_for_record(record),
        )

    def format_task_prompt(
        self,
        question: str,
        seed_url: str | None = None,
        reference_urls: List[str] | None = None,
        output_constraint: str | None = None,
        temporal_constraint: str | None = None,
    ) -> str:
        prompt = super().format_task_prompt(question, seed_url)
        temporal = normalize_text(temporal_constraint)
        if temporal:
            prompt += "\n\nTemporal Constraint:\n" + temporal
        urls = [url for url in dict.fromkeys(reference_urls or []) if url.startswith(("http://", "https://"))]
        if urls:
            prompt += "\n\nReference URLs:\n" + "\n".join(f"- {url}" for url in urls)
        constraint = normalize_text(output_constraint)
        if constraint:
            prompt += "\n\nDataset Output Constraint:\n" + constraint
        return prompt

    @staticmethod
    def temporal_constraint_for_record(record: JsonRow) -> str:
        return normalize_text(record.get("temporal_constraint"))

    def reference_urls_for_record(self, record: JsonRow) -> List[str]:
        urls: List[str] = []
        for key in self.reference_url_keys:
            for item in self.iter_url_values(record.get(key)):
                url = normalize_text(item)
                if url.startswith(("http://", "https://")):
                    urls.append(url)
        url_cleaning = record.get("url_cleaning")
        if isinstance(url_cleaning, dict):
            for key in self.reference_url_keys:
                for item in self.iter_url_values(url_cleaning.get(key)):
                    url = normalize_text(item)
                    if url.startswith(("http://", "https://")):
                        urls.append(url)
        return list(dict.fromkeys(urls))

    def output_constraint_for_record(self, record: JsonRow) -> str:
        source_dataset = normalize_text(record.get("source_dataset"))
        return DATASET_OUTPUT_CONSTRAINTS.get(source_dataset, "")

    def bench_extra(self, record: JsonRow) -> JsonRow:
        extra: JsonRow = {}
        for key in [
            "source_dataset",
            "source_row_index",
            "difficulty",
            "finance_subdomain",
            "answer_type",
            "priority",
            "synthetic_method",
            "num_required_hops",
            "root_url",
            "reference_urls",
            "clean_source_urls",
            "clean_reference_urls",
            "source_url_slots",
            "temporal_constraint",
        ]:
            value = record.get(key)
            if value not in (None, ""):
                extra[key] = value
        reference_urls = self.reference_urls_for_record(record)
        if reference_urls and "reference_urls" not in extra:
            extra["reference_urls"] = reference_urls
        output_constraint = self.output_constraint_for_record(record)
        if output_constraint:
            extra["dataset_output_constraint"] = output_constraint
        golden = first_present(record, self.answer_keys)
        if golden not in (None, ""):
            extra["golden_answer"] = golden
        return extra

    def aggregate(self, logs: List[JsonRow]) -> JsonRow:
        return aggregate_fin_deepsearch_sft_logs(logs)
