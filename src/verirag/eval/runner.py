"""Evaluation runner.

``validate_dataset`` runs first and is not optional: a gold phrase that no
longer occurs in the corpus would silently depress every retrieval metric and
make the whole report meaningless. Better to fail loudly on a broken label than
to publish a number nobody can trust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..engine import VeriRAG
from .dataset import EvalCase, all_cases
from .metrics import CaseResult, EvalReport, contains_phrase


def validate_dataset(engine: VeriRAG, cases: Sequence[EvalCase]) -> list[str]:
    """Return a list of gold phrases that cannot be found in any chunk."""
    corpus = engine.indexer.vectors.all_chunks()
    broken: list[str] = []
    for case in cases:
        if case.is_out_of_corpus or not case.gold_phrase:
            continue
        if not any(contains_phrase(chunk.text, case.gold_phrase) for chunk in corpus):
            broken.append(f"{case.question}  ->  {case.gold_phrase!r}")
    return broken


def run_eval(
    engine: VeriRAG,
    cases: Sequence[EvalCase] | None = None,
    *,
    k: int | None = None,
    strict: bool = True,
) -> EvalReport:
    """Grade *engine* against the golden set."""
    cases = list(cases if cases is not None else all_cases())
    k = k or engine.settings.top_k_final

    broken = validate_dataset(engine, cases)
    if broken and strict:
        raise AssertionError(
            "Gold phrases missing from the indexed corpus — regenerate the sample "
            "PDFs and re-ingest, or fix the labels:\n  " + "\n  ".join(broken)
        )

    results: list[CaseResult] = []
    for case in cases:
        outcome = engine.ask(case.question, render_proof=False, persist=False, top_k=k)
        answer = outcome.answer

        relevance = [
            contains_phrase(item.chunk.text, case.gold_phrase) if case.gold_phrase else False
            for item in outcome.retrieval.chunks
        ]

        answer_hit = bool(case.answer_must_contain) and all(
            contains_phrase(answer.text, needle) for needle in case.answer_must_contain
        )

        used = answer.used_citations
        citation_correct = bool(used) and contains_phrase(used[0].quote, case.gold_phrase)

        results.append(
            CaseResult(
                question=case.question,
                category=case.category,
                relevance=relevance,
                answer_hit=answer_hit,
                citation_correct=citation_correct,
                groundedness=answer.groundedness,
                refused=answer.refused,
                flagged=answer.refused or answer.weak_evidence,
                latency_ms=answer.latency_ms,
                best_score=outcome.retrieval.best_score,
                top_locator=outcome.retrieval.chunks[0].chunk.locator if outcome.retrieval.chunks else "",
                top_doc=outcome.retrieval.chunks[0].chunk.doc_name if outcome.retrieval.chunks else "",
            )
        )

    return EvalReport(
        results=results,
        k=k,
        provider=str(engine.stats().get("llm", "")),
        config={
            "embedder": engine.indexer.embedder.name,
            "reranker": engine.retriever.reranker.name,
            "bm25": engine.indexer.bm25.impl,
            "top_k_final": k,
            "multi_query": engine.settings.multi_query,
            "min_retrieval_score": engine.settings.min_retrieval_score,
            "grounding_threshold": engine.settings.grounding_threshold,
        },
    )


def save_report(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return target
