"""Evaluation harness: golden QA set, ranking metrics and refusal calibration."""

from __future__ import annotations

from .dataset import EvalCase, all_cases, in_corpus_cases
from .metrics import CaseResult, EvalReport, contains_phrase, ndcg_at_k, reciprocal_rank
from .runner import run_eval, save_report, validate_dataset

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "all_cases",
    "contains_phrase",
    "in_corpus_cases",
    "ndcg_at_k",
    "reciprocal_rank",
    "run_eval",
    "save_report",
    "validate_dataset",
]
