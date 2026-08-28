"""Retrieval layer: query expansion, hybrid search, rank fusion, reranking."""

from __future__ import annotations

from .fusion import merge_multi_query, reciprocal_rank_fusion
from .query_expansion import expand_query, keyword_form, normalise_query
from .reranker import CrossEncoderReranker, LexicalReranker, get_reranker
from .retriever import HybridRetriever, RetrievalResult

__all__ = [
    "CrossEncoderReranker",
    "HybridRetriever",
    "LexicalReranker",
    "RetrievalResult",
    "expand_query",
    "get_reranker",
    "keyword_form",
    "merge_multi_query",
    "normalise_query",
    "reciprocal_rank_fusion",
]
