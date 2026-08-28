"""The hybrid retriever: multi-query -> dense + BM25 -> RRF -> rerank.

Everything is observable: :meth:`HybridRetriever.retrieve` returns
:class:`RetrievedChunk` objects that still carry their dense rank, BM25 rank,
fused score and rerank score, so the UI can show *why* each cited chunk was
selected.  That auditability is the whole point of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..config import Settings, get_settings
from ..index.embedder import tokenize
from ..index.indexer import Indexer
from ..models import RetrievedChunk
from .fusion import merge_multi_query, reciprocal_rank_fusion
from .query_expansion import expand_query
from .reranker import Reranker, get_reranker


@dataclass(slots=True)
class RetrievalResult:
    """Final ranked evidence plus the trace that produced it."""

    query: str
    variants: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    n_candidates: int = 0

    @property
    def best_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)

    def trace(self) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": item.chunk_id,
                "doc": item.chunk.doc_name,
                "locator": item.chunk.locator,
                "section": item.chunk.section,
                "score": round(item.score, 5),
                "dense_rank": item.dense_rank,
                "lexical_rank": item.lexical_rank,
                "fused_score": None if item.fused_score is None else round(item.fused_score, 5),
                "rerank_score": None if item.rerank_score is None else round(item.rerank_score, 5),
                "why": item.explain(),
            }
            for item in self.chunks
        ]


class HybridRetriever:
    """Dense + lexical retrieval with rank fusion and reranking."""

    def __init__(
        self,
        indexer: Indexer,
        settings: Settings | None = None,
        reranker: Reranker | None = None,
        llm_expand: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.indexer = indexer
        self.settings = settings or get_settings()
        self.reranker = reranker or get_reranker(
            self.settings.reranker, model_name=self.settings.rerank_model
        )
        self.llm_expand = llm_expand
        self._vocab_cache: list[str] | None = None
        self._vocab_size: int = -1

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve the best evidence chunks for *query*."""
        top_k = top_k or self.settings.top_k_final
        variants = expand_query(
            query,
            enabled=self.settings.multi_query,
            llm_expand=self.llm_expand,
            vocabulary=self._vocabulary(),
        )
        if not variants:
            return RetrievalResult(query=query, variants=[], chunks=[], n_candidates=0)

        runs: list[list[RetrievedChunk]] = []
        for variant in variants:
            dense = self._dense(variant)
            lexical = self.indexer.bm25.search(variant, self.settings.top_k_lexical)
            runs.append(
                reciprocal_rank_fusion(dense, lexical, k=self.settings.rrf_k)
            )

        fused = runs[0] if len(runs) == 1 else merge_multi_query(runs, k=self.settings.rrf_k)

        if doc_ids:
            allowed = set(doc_ids)
            fused = [item for item in fused if item.chunk.doc_id in allowed]

        candidate_pool = fused[: max(top_k * 6, 24)]
        n_candidates = len(fused)
        ranked = self.reranker.rerank(query, candidate_pool, top_k)
        ranked = _drop_overlapping(ranked)
        return RetrievalResult(query=query, variants=variants, chunks=ranked, n_candidates=n_candidates)

    def _dense(self, query: str) -> list[tuple]:
        vector = self.indexer.embedder.encode([query])[0]
        return self.indexer.vectors.search(vector, self.settings.top_k_dense)

    def _vocabulary(self) -> list[str]:
        """Distinct terms in the indexed corpus, for typo repair.

        Cached against the chunk count so it is rebuilt after ingestion but not on
        every query. Correcting against the corpus's own vocabulary beats a general
        dictionary here: it fixes half-remembered technical terms as well as typos,
        and never "corrects" a word into something the documents do not contain.
        """
        size = len(self.indexer.vectors)
        if self._vocab_cache is None or self._vocab_size != size:
            terms: set[str] = set()
            for chunk in self.indexer.vectors.all_chunks():
                terms.update(t for t in tokenize(f"{chunk.section} {chunk.text}") if len(t) >= 4)
            self._vocab_cache = sorted(terms)
            self._vocab_size = size
        return self._vocab_cache

    # ------------------------------------------------------------------- info
    def describe(self) -> dict[str, object]:
        return {
            "embedder": self.indexer.embedder.name,
            "reranker": self.reranker.name,
            "bm25": self.indexer.bm25.impl,
            "top_k_dense": self.settings.top_k_dense,
            "top_k_lexical": self.settings.top_k_lexical,
            "top_k_final": self.settings.top_k_final,
            "multi_query": self.settings.multi_query,
        }


def _drop_overlapping(items: Sequence[RetrievedChunk], overlap_ratio: float = 0.75) -> list[RetrievedChunk]:
    """Remove near-duplicate spans so the prompt is not filled with repeats.

    Overlapping chunk windows on the same page can otherwise consume the whole
    context budget with the same sentences.
    """
    kept: list[RetrievedChunk] = []
    for item in items:
        duplicate = False
        for existing in kept:
            same_page = (
                item.chunk.doc_id == existing.chunk.doc_id and item.chunk.page_no == existing.chunk.page_no
            )
            if not same_page:
                continue
            lo = max(item.chunk.line_start, existing.chunk.line_start)
            hi = min(item.chunk.line_end, existing.chunk.line_end)
            shared = max(hi - lo + 1, 0)
            smallest = min(item.chunk.n_lines, existing.chunk.n_lines)
            if smallest and shared / smallest >= overlap_ratio:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept
