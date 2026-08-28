"""Reranking — the cheapest large win in a RAG pipeline.

Fusion optimises recall; reranking optimises the *precision of the few chunks
that actually enter the prompt*.  Two implementations:

``LexicalReranker`` (default)
    Free, instant, no downloads.  Blends four signals: IDF-weighted query-term
    coverage, proximity of matched terms, exact phrase presence, and a mild
    length prior.  Deliberately favours coverage so numeric/legal identifiers
    are not diluted by long chunks.

``CrossEncoderReranker`` (opt-in)
    ``ms-marco-MiniLM-L-6-v2`` scoring every (query, chunk) pair jointly.
    Best quality; needs ``requirements-ml.txt``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

from ..index.embedder import tokenize
from ..models import RetrievedChunk

_STOP = frozenset(
    """a an and are as at be by can did do does for from has have how in is it its of on or shall
    that the their there these this to was were what when where which who why will with would""".split()
)


@runtime_checkable
class Reranker(Protocol):
    name: str
    refusal_threshold: float
    low_confidence_threshold: float

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class LexicalReranker:
    """Signal-blended lexical reranker (no model weights required)."""

    name = "lexical"

    # Thresholds belong to whatever produces the scores. This reranker emits a
    # roughly continuous score in [0, ~1.1]; measured on the sample corpus,
    # in-corpus questions bottom out near 0.17 and out-of-domain ones top out
    # near 0.34, so the classes overlap and the gate must stay permissive.
    refusal_threshold = 0.10
    low_confidence_threshold = 0.35

    def __init__(self, phrase_bonus: float = 0.35, proximity_weight: float = 0.20) -> None:
        self.phrase_bonus = phrase_bonus
        self.proximity_weight = proximity_weight

    # ------------------------------------------------------------------ score
    def score(
        self,
        query: str,
        text: str,
        *,
        idf: dict[str, float] | None = None,
        default_weight: float | None = None,
    ) -> float:
        q_terms = [t for t in tokenize(query) if t not in _STOP]
        if not q_terms:
            return 0.0
        d_tokens = tokenize(text)
        if not d_tokens:
            return 0.0
        q_set = set(q_terms)
        d_set = set(d_tokens)
        weights = idf or {}

        # A term absent from the whole candidate corpus is maximally rare, so it
        # must carry the *highest* weight. Defaulting it to 1.0 (below the idf of
        # terms that do occur) made a question about an unrelated topic look
        # well-covered because its one incidental match dominated the total.
        missing_weight = default_weight if default_weight is not None else max(weights.values(), default=1.0)

        def weight_of(term: str) -> float:
            return weights.get(term, missing_weight)

        # 1) IDF-weighted coverage: which query terms appear at all.
        total_w = sum(weight_of(t) for t in q_set)
        hit_w = sum(weight_of(t) for t in q_set if t in d_set)
        coverage = hit_w / total_w if total_w else 0.0

        # 2) Proximity via the *smallest window* containing every matched term,
        #    scaled by how much of the query was matched at all.
        proximity = _min_cover_proximity(d_tokens, q_set)

        # 3) Repetition: a clause that says "rent" five times is *about* rent,
        #    whereas a passage mentioning it once is merely adjacent to the topic.
        counts = Counter(d_tokens)
        repeats = sum(min(counts[t] - 1, 3) for t in q_set if counts[t] > 1)
        repetition = min(repeats / (2.0 * len(q_set)), 1.0)

        # 4) Exact contiguous phrase match (strong signal for clause lookups).
        phrase = 0.0
        joined_doc = " ".join(d_tokens)
        for size in (4, 3, 2):
            grams = [" ".join(q_terms[i : i + size]) for i in range(max(len(q_terms) - size + 1, 0))]
            if any(g in joined_doc for g in grams if g):
                phrase = self.phrase_bonus * (size / 4.0)
                break

        # 5) Sufficiency, not brevity. An earlier version rewarded short chunks,
        #    which let 3-line page tails outrank the clause that answers the
        #    question. A chunk needs enough words to actually contain an answer.
        sufficiency = 0.85 + 0.15 * min(len(d_tokens) / 60.0, 1.0)

        return (
            0.55 * coverage + self.proximity_weight * proximity + 0.12 * repetition + phrase
        ) * sufficiency

    # ----------------------------------------------------------------- rerank
    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        idf = _corpus_idf([c.chunk.text for c in candidates])
        missing_weight = max(idf.values(), default=1.0)
        for item in candidates:
            body = f"{item.chunk.section} {item.chunk.text}" if item.chunk.section else item.chunk.text
            lexical = self.score(query, body, idf=idf, default_weight=missing_weight)
            # Blend in the dense signal so a semantically-matched, lexically
            # disjoint chunk still scores. Deliberately *not* blended with the
            # RRF prior, which is rank-based and would put a floor under every
            # candidate — destroying the score's usefulness as a refusal signal.
            dense = max(item.dense_score or 0.0, 0.0)
            item.rerank_score = round(0.85 * lexical + 0.15 * dense, 6)
            item.score = item.rerank_score
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


class CrossEncoderReranker:
    """Transformer cross-encoder reranker (optional, local, free)."""

    name = "cross-encoder"

    # A cross-encoder's sigmoid output is strongly *bimodal*, not continuous:
    # measured on the sample corpus the in-corpus median is 0.978 while
    # out-of-domain questions land near 0.0002. Reusing the lexical reranker's
    # 0.10 gate here wrongly refused 11% of real questions. Score scales across
    # rerankers are not comparable, so the threshold has to travel with the
    # component that produces the score.
    refusal_threshold = 0.001
    low_confidence_threshold = 0.30

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]
        raw = self._model.predict(pairs, show_progress_bar=False)
        for item, logit in zip(candidates, raw):
            item.rerank_score = 1.0 / (1.0 + math.exp(-float(logit)))  # logit -> [0,1]
            item.score = item.rerank_score
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


def _min_cover_proximity(d_tokens: Sequence[str], q_set: set[str]) -> float:
    """Proximity as ``matched_terms / smallest_window_containing_them``.

    The obvious implementation — distance from the first match to the last —
    is actively harmful: repeating a query term later in the passage widens the
    span and *lowers* the score, penalising the very chunks that are most on
    topic.  A minimum covering window (classic two-pointer sweep, O(n)) measures
    what we actually care about: do the query terms occur close together
    *somewhere* in this chunk.

    The result is scaled by the fraction of the query that was matched, so a
    chunk containing one incidental term out of five cannot claim perfect
    proximity.
    """
    if not q_set:
        return 0.0
    hits = [(index, token) for index, token in enumerate(d_tokens) if token in q_set]
    if not hits:
        return 0.0
    present = {token for _, token in hits}
    needed = len(present)
    matched_fraction = needed / len(q_set)
    if needed == 1:
        return matched_fraction

    best_width: int | None = None
    counts: dict[str, int] = {}
    have = 0
    left = 0

    for right, (_pos, term) in enumerate(hits):
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            have += 1
        while have == needed:
            width = hits[right][0] - hits[left][0] + 1
            if best_width is None or width < best_width:
                best_width = width
            left_term = hits[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                have -= 1
            left += 1

    if not best_width:
        return 0.0
    return min(needed / best_width, 1.0) * matched_fraction


def _corpus_idf(texts: Sequence[str]) -> dict[str, float]:
    n = max(len(texts), 1)
    df: dict[str, int] = {}
    for text in texts:
        for term in set(tokenize(text)):
            df[term] = df.get(term, 0) + 1
    return {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}


def get_reranker(kind: str = "lexical", *, model_name: str = "") -> Reranker:
    """Factory that silently degrades to the lexical reranker."""
    if (kind or "lexical").strip().lower() in {"cross-encoder", "crossencoder", "ce", "neural"}:
        try:
            return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:  # noqa: BLE001
            pass
    return LexicalReranker()
