"""Lexical (sparse) retrieval — BM25 Okapi.

Dense embeddings are weak exactly where document QA needs precision: clause
numbers, statute citations, party names, dates and amounts.  BM25 nails those,
so VeriRAG always runs both and fuses the rankings.

The scorer is implemented in NumPy so the project has no hard dependency on
``rank_bm25``; when that package *is* installed it is used instead (identical
formula, battle-tested implementation).  ``VERIRAG_BM25_IMPL=internal`` forces
the built-in path.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from ..models import Chunk
from .embedder import tokenize

try:  # optional
    from rank_bm25 import BM25Okapi as _RankBM25

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    _RankBM25 = None  # type: ignore[assignment]
    _HAS_RANK_BM25 = False


class InternalBM25:
    """Vectorised BM25 Okapi (k1=1.5, b=0.75) over a tokenised corpus."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if self.n_docs else 0.0

        # term -> {doc_index: term_frequency}
        self.postings: dict[str, dict[int, int]] = {}
        for index, tokens in enumerate(corpus_tokens):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, {})[index] = freq

        self.idf: dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)
            # Robertson/Sparck-Jones IDF, floored so common terms never go negative.
            self.idf[term] = max(math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)), 1e-6)

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avg_len == 0.0:
            return scores
        length_norm = self.k1 * (1.0 - self.b + self.b * self.doc_lens / self.avg_len)
        for term in query_tokens:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            indices = np.fromiter(posting.keys(), dtype=np.int64, count=len(posting))
            freqs = np.fromiter(posting.values(), dtype=np.float32, count=len(posting))
            scores[indices] += idf * (freqs * (self.k1 + 1.0)) / (freqs + length_norm[indices])
        return scores


class BM25Store:
    """Persistent BM25 index aligned to the same :class:`Chunk` objects."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._model: InternalBM25 | object | None = None
        self._impl = "none"

    @property
    def path(self) -> Path:
        return self.directory / "bm25.json"

    # -------------------------------------------------------------- mutation
    def build(self, chunks: Sequence[Chunk]) -> "BM25Store":
        self._chunks = list(chunks)
        self._tokens = [self._doc_tokens(c) for c in self._chunks]
        self._fit()
        return self

    def _doc_tokens(self, chunk: Chunk) -> list[str]:
        # Section headings are folded in: they carry high-signal terms.
        return tokenize(f"{chunk.section} {chunk.text}")

    def _fit(self) -> None:
        if not self._tokens:
            self._model, self._impl = None, "none"
            return
        force_internal = os.getenv("VERIRAG_BM25_IMPL", "").lower() == "internal"
        if _HAS_RANK_BM25 and not force_internal:
            self._model = _RankBM25(self._tokens)
            self._impl = "rank_bm25"
        else:
            self._model = InternalBM25(self._tokens)
            self._impl = "internal"

    def clear(self) -> None:
        self._chunks, self._tokens, self._model, self._impl = [], [], None, "none"
        self.path.unlink(missing_ok=True)

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._model is None or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._model.get_scores(tokens), dtype=np.float32)  # type: ignore[union-attr]
        if scores.size == 0:
            return []
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0.0]

    @property
    def impl(self) -> str:
        return self._impl

    def __len__(self) -> int:
        return len(self._chunks)

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        payload = {"chunks": [c.to_dict() for c in self._chunks]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        if not self.path.exists():
            return False
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        self._tokens = [self._doc_tokens(c) for c in self._chunks]
        self._fit()
        return bool(self._chunks)
