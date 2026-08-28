"""Retrieval and answer-quality metrics.

Implemented from the definitions rather than pulled from a library, both to keep
the dependency list minimal and because the exact conventions matter:

* ``Recall@k`` here is *hit rate* — the fraction of questions for which at least
  one relevant chunk appears in the top k. With one gold passage per question
  that is the meaningful quantity, and it is what a generator actually needs.
* ``MRR`` uses the reciprocal rank of the *first* relevant chunk.
* ``nDCG@k`` uses binary gains, so the IDCG of a single relevant document is 1
  and the metric reduces to ``1 / log2(rank + 1)``.
* Refusal is scored as a classification problem over in-corpus vs out-of-corpus
  questions, giving precision, recall and a false-refusal rate.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from ..textnorm import contains_phrase, normalise_for_compare


def normalise(text: str) -> str:
    """Lowercase, fold typographic punctuation, collapse whitespace.

    Punctuation folding is essential here: a model that answers "P\u201142"
    (non-breaking hyphen) is correct, and a literal comparison against the PDF's
    "P-42" would score it as wrong.
    """
    return normalise_for_compare(text)


__all__ = [
    "CaseResult",
    "EvalReport",
    "contains_phrase",
    "first_relevant_rank",
    "hit_at_k",
    "ndcg_at_k",
    "normalise",
    "precision_at_k",
    "reciprocal_rank",
]


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------
def first_relevant_rank(relevance: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant item, or ``None``."""
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return index
    return None


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    rank = first_relevant_rank(relevance)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(relevance: Sequence[bool], k: int) -> float:
    """Binary-gain nDCG@k. IDCG is 1.0 for a single relevant passage."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(relevance[:k], start=1) if hit)
    ideal = 1.0  # best case: the single relevant passage sits at rank 1
    return min(dcg / ideal, 1.0) if ideal else 0.0


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    window = relevance[:k]
    return (sum(1 for hit in window if hit) / len(window)) if window else 0.0


# ---------------------------------------------------------------------------
# aggregate report
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CaseResult:
    """Per-question outcome."""

    question: str
    category: str
    relevance: list[bool] = field(default_factory=list)
    answer_hit: bool = False
    citation_correct: bool = False
    groundedness: float = 0.0
    refused: bool = False
    flagged: bool = False
    latency_ms: int = 0
    best_score: float = 0.0
    top_locator: str = ""
    top_doc: str = ""

    @property
    def is_out_of_corpus(self) -> bool:
        return self.category == "out_of_corpus"


@dataclass(slots=True)
class EvalReport:
    """Aggregated metrics across a run."""

    results: list[CaseResult]
    k: int = 5
    provider: str = ""
    config: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def in_corpus(self) -> list[CaseResult]:
        return [r for r in self.results if not r.is_out_of_corpus]

    @property
    def out_of_corpus(self) -> list[CaseResult]:
        return [r for r in self.results if r.is_out_of_corpus]

    # --------------------------------------------------------------- retrieval
    def retrieval_metrics(self) -> dict[str, float]:
        answered = [r for r in self.in_corpus if r.relevance]
        if not answered:
            return {}
        return {
            "recall@1": _mean(hit_at_k(r.relevance, 1) for r in answered),
            "recall@3": _mean(hit_at_k(r.relevance, 3) for r in answered),
            f"recall@{self.k}": _mean(hit_at_k(r.relevance, self.k) for r in answered),
            "mrr": _mean(reciprocal_rank(r.relevance) for r in answered),
            f"ndcg@{self.k}": _mean(ndcg_at_k(r.relevance, self.k) for r in answered),
            f"precision@{self.k}": _mean(precision_at_k(r.relevance, self.k) for r in answered),
        }

    # ----------------------------------------------------------------- answers
    def answer_metrics(self) -> dict[str, float]:
        cases = self.in_corpus
        if not cases:
            return {}
        return {
            "answer_accuracy": _mean(1.0 if r.answer_hit else 0.0 for r in cases),
            "citation_precision@1": _mean(1.0 if r.citation_correct else 0.0 for r in cases),
            "mean_groundedness": _mean(r.groundedness for r in cases if not r.refused),
        }

    # ---------------------------------------------------------------- refusals
    def refusal_metrics(self) -> dict[str, float]:
        """Refusal scored as binary classification, plus the softer guard.

        ``*_refusal_*`` covers hard refusals (the system declined to answer).
        ``*_flagged_*`` also counts answers returned with a low-confidence
        warning, which is how VeriRAG handles the band where evidence is weak
        but not absent — the honest middle ground between answering
        confidently and refusing outright.
        """
        ood = self.out_of_corpus
        in_corpus = self.in_corpus
        if not ood and not in_corpus:
            return {}
        true_positive = sum(1 for r in ood if r.refused)
        false_positive = sum(1 for r in in_corpus if r.refused)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        return {
            "refusal_recall": (true_positive / len(ood)) if ood else 0.0,
            "refusal_precision": precision,
            "false_refusal_rate": (false_positive / len(in_corpus)) if in_corpus else 0.0,
            "ood_flagged_rate": (sum(1 for r in ood if r.flagged) / len(ood)) if ood else 0.0,
            "in_corpus_flagged_rate": (
                sum(1 for r in in_corpus if r.flagged) / len(in_corpus)
            ) if in_corpus else 0.0,
        }

    # ----------------------------------------------------------------- latency
    def latency_metrics(self) -> dict[str, float]:
        values = sorted(r.latency_ms for r in self.results)
        if not values:
            return {}
        return {
            "latency_p50_ms": float(statistics.median(values)),
            "latency_p95_ms": float(values[max(int(len(values) * 0.95) - 1, 0)]),
            "latency_max_ms": float(values[-1]),
        }

    # --------------------------------------------------------------- threshold
    def suggest_refusal_threshold(self) -> dict[str, float]:
        """Best separating threshold between in-corpus and out-of-corpus scores.

        Reports the score distributions and the midpoint of the widest gap so the
        configured ``min_retrieval_score`` is a measured value rather than a
        guess. ``separation`` below zero means the classes overlap and refusal
        cannot be made reliable by thresholding alone at this configuration.
        """
        in_scores = [r.best_score for r in self.in_corpus]
        ood_scores = [r.best_score for r in self.out_of_corpus]
        if not in_scores or not ood_scores:
            return {}
        lowest_in = min(in_scores)
        highest_ood = max(ood_scores)
        return {
            "in_corpus_min_score": round(lowest_in, 4),
            "in_corpus_median_score": round(float(statistics.median(in_scores)), 4),
            "out_of_corpus_max_score": round(highest_ood, 4),
            "separation": round(lowest_in - highest_ood, 4),
            "suggested_threshold": round((lowest_in + highest_ood) / 2.0, 4),
        }

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "config": self.config,
            "n_questions": len(self.results),
            "n_in_corpus": len(self.in_corpus),
            "n_out_of_corpus": len(self.out_of_corpus),
            "retrieval": {k: round(v, 4) for k, v in self.retrieval_metrics().items()},
            "answers": {k: round(v, 4) for k, v in self.answer_metrics().items()},
            "refusal": {k: round(v, 4) for k, v in self.refusal_metrics().items()},
            "latency": {k: round(v, 1) for k, v in self.latency_metrics().items()},
            "threshold_calibration": self.suggest_refusal_threshold(),
        }

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        lines = [
            "=" * 74,
            f"VeriRAG evaluation - {len(self.results)} questions "
            f"({len(self.in_corpus)} in-corpus, {len(self.out_of_corpus)} out-of-corpus)",
            f"provider: {self.provider}",
            "=" * 74,
        ]

        def block(title: str, payload: dict[str, float], fmt: str = "{:.3f}") -> None:
            if not payload:
                return
            lines.append(f"\n{title}")
            for key, value in payload.items():
                lines.append(f"  {key:<26} {fmt.format(value)}")

        block("RETRIEVAL", self.retrieval_metrics())
        block("ANSWER QUALITY", self.answer_metrics())
        block("REFUSAL (hallucination guard)", self.refusal_metrics())
        block("LATENCY", self.latency_metrics(), "{:.0f}")
        block("REFUSAL THRESHOLD CALIBRATION", self.suggest_refusal_threshold(), "{:.4f}")

        calibration = self.suggest_refusal_threshold()
        if calibration and calibration.get("separation", 0.0) < 0:
            lines.append(
                "\n  note: separation < 0 means in-corpus and out-of-corpus score ranges\n"
                "  overlap, so no single threshold can cleanly refuse out-of-domain\n"
                "  questions at this configuration. VeriRAG therefore keeps the hard gate\n"
                "  low and flags the weak band instead (see ood_flagged_rate). Enabling\n"
                "  the neural embedder and cross-encoder widens the separation."
            )

        failures = [r for r in self.in_corpus if not r.answer_hit]
        if failures:
            lines.append(f"\nFAILED QUESTIONS ({len(failures)})")
            for result in failures:
                flag = "refused" if result.refused else f"top={result.top_doc} {result.top_locator}"
                lines.append(f"  - {result.question[:62]:<62} [{flag}]")

        lines.append("=" * 74)
        return "\n".join(lines)


def _mean(values) -> float:
    collected = list(values)
    return float(sum(collected) / len(collected)) if collected else 0.0
