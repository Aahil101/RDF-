"""Groundedness verification — the hallucination guard.

Prompting a model to cite its sources is not evidence that it did.  This module
independently re-checks the finished answer against the actual PDF lines:

For every sentence of the answer it

1. collects the sources the sentence cites,
2. slides a 1-4 line window across those sources' lines,
3. scores each window with a blend of IDF-weighted term coverage, fuzzy token
   similarity and **numeric recall**, and
4. keeps the best window as the *proven span* — the narrowest page + line range
   that actually supports the sentence.

Sentences scoring below ``grounding_threshold`` are returned as unsupported so
the UI can strike them through, and the answer's overall groundedness score is
the length-weighted mean.  Numeric recall is treated as a multiplier rather than
a term because a wrong amount, date or clause number is the most damaging error
a document-QA system can make.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from rapidfuzz import fuzz

from ..config import Settings, get_settings
from ..index.embedder import tokenize
from ..index.indexer import LineStore
from ..models import Answer, Citation, ClaimVerdict, EvidenceSpan, PdfLine, merge_bboxes
from ..generation.citations import sentences_with_markers, strip_markers
from ..textnorm import normalise_for_compare

_STOP = frozenset(
    """a an and are as at be been by for from has have in is it its of on or shall that the
    their there this to was were with would which who what when where why how also such any
    each other than then these those into upon under over per may must can will not no""".split()
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
MAX_WINDOW_LINES = 14
"""Upper bound on a proven span. Greedy expansion makes this cheap, and a claim
wrapped across a dozen PDF lines is common in dense legal typesetting."""


@dataclass(slots=True)
class GroundingReport:
    """Aggregate verification outcome for one answer."""

    verdicts: list[ClaimVerdict]
    groundedness: float
    supported: int
    total: int

    @property
    def unsupported(self) -> int:
        return self.total - self.supported

    def to_dict(self) -> dict[str, object]:
        return {
            "groundedness": self.groundedness,
            "supported": self.supported,
            "total": self.total,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------
def _content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 1]


def _numbers(text: str) -> set[str]:
    """Numeric tokens, normalised so 1,00,000 and 100000 compare equal."""
    return {match.group(0).replace(",", "").rstrip(".") for match in _NUMBER_RE.finditer(text)}


def support_score(claim: str, evidence: str) -> float:
    """Score in [0, 1] for how well *evidence* supports *claim*."""
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = set(_content_terms(evidence))
    if not evidence_terms:
        return 0.0

    # 1) Term coverage, with rarer (longer) terms weighted higher.
    weights = {term: 1.0 + math.log1p(len(term)) for term in set(claim_terms)}
    total_weight = sum(weights.values())
    hit_weight = sum(weight for term, weight in weights.items() if term in evidence_terms)
    coverage = hit_weight / total_weight if total_weight else 0.0

    # 2) Fuzzy similarity: robust to paraphrase, word order and inflection.
    #    Punctuation is folded first, because a model writing "months\u2019" or
    #    "lock\u2011in" against a PDF containing "months'" and "lock-in" is
    #    quoting verbatim and must not be penalised for typography.
    claim_folded = normalise_for_compare(claim)
    evidence_folded = normalise_for_compare(evidence)
    fuzzy = max(
        fuzz.token_set_ratio(claim_folded, evidence_folded),
        fuzz.partial_ratio(claim_folded, evidence_folded),
    ) / 100.0

    base = 0.60 * coverage + 0.40 * fuzzy

    # 3) Numeric recall as a multiplier: an unsupported figure is disqualifying.
    claim_numbers = _numbers(claim)
    if claim_numbers:
        evidence_numbers = _numbers(evidence)
        recall = len(claim_numbers & evidence_numbers) / len(claim_numbers)
        base *= 0.45 + 0.55 * recall

    return round(min(base, 1.0), 4)


def best_window(
    claim: str,
    lines: Sequence[PdfLine],
    *,
    max_window: int = MAX_WINDOW_LINES,
    patience: int = 2,
) -> tuple[float, list[PdfLine]]:
    """Find the tightest run of lines that best supports *claim*.

    A claim frequently spans an arbitrary number of wrapped PDF lines, so a
    fixed-size window systematically under-reports support.  Instead we seed on
    the single best-matching line and grow the span greedily towards whichever
    neighbour helps more, keeping the best span seen.  ``patience`` allows the
    search to step over an intervening line (a heading, a page artefact) that
    momentarily lowers the score, which plain hill-climbing would stop at.

    Cost is O(n) scoring calls versus O(n * window) for exhaustive search.
    """
    if not lines:
        return 0.0, []

    n = len(lines)
    cap = min(max(max_window, 1), n)

    single = [support_score(claim, line.text) for line in lines]
    seed = max(range(n), key=single.__getitem__)

    lo = hi = seed
    best_score = single[seed]
    best_lo, best_hi = lo, hi
    strikes = 0

    while (hi - lo + 1) < cap:
        options: list[tuple[float, int, int]] = []
        if lo > 0:
            text = " ".join(ln.text for ln in lines[lo - 1 : hi + 1])
            options.append((support_score(claim, text), lo - 1, hi))
        if hi < n - 1:
            text = " ".join(ln.text for ln in lines[lo : hi + 2])
            options.append((support_score(claim, text), lo, hi + 1))
        if not options:
            break

        score, lo, hi = max(options, key=lambda option: option[0])
        if score > best_score + 1e-9:
            best_score, best_lo, best_hi, strikes = score, lo, hi, 0
        else:
            strikes += 1
            if strikes > patience:
                break

    return round(best_score, 4), list(lines[best_lo : best_hi + 1])


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------
class GroundingVerifier:
    """Verifies an :class:`Answer` against the physical lines of the PDFs."""

    def __init__(self, line_store: LineStore, settings: Settings | None = None) -> None:
        self.lines = line_store
        self.settings = settings or get_settings()

    # ----------------------------------------------------------------- public
    def verify(self, answer: Answer) -> GroundingReport:
        """Populate per-sentence verdicts and the overall groundedness score."""
        if answer.refused or not answer.citations:
            report = GroundingReport(verdicts=[], groundedness=0.0, supported=0, total=0)
            answer.verdicts, answer.groundedness = [], 0.0
            return report

        by_marker = {c.marker: c for c in answer.citations}
        verdicts: list[ClaimVerdict] = []

        for sentence, marker_numbers in sentences_with_markers(answer.text):
            claim = strip_markers(sentence)
            if len(_content_terms(claim)) < 2:  # connective fragment, not a claim
                continue

            cited = [by_marker[f"S{n}"] for n in marker_numbers if f"S{n}" in by_marker]
            candidates = cited or answer.citations  # uncited sentence: search everything

            score, span = self._best_evidence(claim, candidates)
            supported = score >= self.settings.grounding_threshold and bool(cited)
            verdicts.append(
                ClaimVerdict(
                    sentence=sentence,
                    supported=supported,
                    score=score,
                    markers=[f"S{n}" for n in marker_numbers],
                    evidence=span,
                )
            )

        groundedness = _aggregate(verdicts)
        answer.verdicts = verdicts
        answer.groundedness = groundedness
        self._narrow_citations(answer)

        return GroundingReport(
            verdicts=verdicts,
            groundedness=groundedness,
            supported=sum(1 for v in verdicts if v.supported),
            total=len(verdicts),
        )

    # ---------------------------------------------------------------- helpers
    def _best_evidence(self, claim: str, citations: Sequence[Citation]) -> tuple[float, EvidenceSpan | None]:
        best_score = 0.0
        best: EvidenceSpan | None = None

        for citation in citations:
            lines = self.lines.get_range(
                citation.doc_id, citation.page_no, citation.line_start, citation.line_end
            )
            if not lines:
                # Index built without a line store (e.g. unit test) — fall back
                # to comparing against the citation's own quoted text.
                score = support_score(claim, citation.quote)
                if score > best_score:
                    best_score = score
                    best = EvidenceSpan(
                        doc_id=citation.doc_id,
                        doc_name=citation.doc_name,
                        page_no=citation.page_no,
                        line_start=citation.line_start,
                        line_end=citation.line_end,
                        text=citation.quote,
                        bboxes=list(citation.bboxes),
                        similarity=score,
                    )
                continue

            score, window = best_window(claim, lines)
            if score > best_score and window:
                best_score = score
                best = EvidenceSpan(
                    doc_id=citation.doc_id,
                    doc_name=citation.doc_name,
                    page_no=citation.page_no,
                    line_start=window[0].line_no,
                    line_end=window[-1].line_no,
                    text=" ".join(ln.text for ln in window),
                    bboxes=[ln.bbox for ln in window],
                    similarity=score,
                )

        return best_score, best

    def _narrow_citations(self, answer: Answer) -> None:
        """Attach proven spans back onto citations for precise highlighting."""
        proven: dict[str, list[EvidenceSpan]] = {}
        for verdict in answer.verdicts:
            if not verdict.evidence:
                continue
            for marker in verdict.markers:
                proven.setdefault(marker, []).append(verdict.evidence)

        for citation in answer.citations:
            spans = proven.get(citation.marker)
            if not spans:
                continue
            tight = max(spans, key=lambda s: s.similarity)
            citation.bboxes = list(tight.bboxes) or citation.bboxes
            citation.line_start = tight.line_start
            citation.line_end = tight.line_end
            citation.quote = tight.text or citation.quote


def _aggregate(verdicts: Sequence[ClaimVerdict]) -> float:
    """Length-weighted mean score; longer claims carry more risk and weight."""
    if not verdicts:
        return 0.0
    total_weight = 0.0
    accumulated = 0.0
    for verdict in verdicts:
        weight = max(len(_content_terms(verdict.sentence)), 1)
        accumulated += verdict.score * weight
        total_weight += weight
    return round(accumulated / total_weight, 4) if total_weight else 0.0


def evidence_bbox(span: EvidenceSpan | None):
    """Union bbox of a proven span, for callers that need one rectangle."""
    return merge_bboxes(span.bboxes) if span and span.bboxes else None
