"""Rank fusion.

Dense and lexical retrievers produce scores on incomparable scales, so VeriRAG
fuses by *rank* using Reciprocal Rank Fusion (Cormack et al., 2009):

.. math:: RRF(d) = \\sum_{r \\in retrievers} \\frac{1}{k + rank_r(d)}

RRF needs no score normalisation, no tuning beyond ``k`` (60 is the standard
default) and is robust when one retriever returns garbage — which is precisely
why it is preferred here over a weighted score blend.
"""

from __future__ import annotations

from typing import Sequence

from ..models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense: Sequence[tuple[Chunk, float]],
    lexical: Sequence[tuple[Chunk, float]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists into one, preserving each source's evidence."""
    merged: dict[str, RetrievedChunk] = {}

    def slot(chunk: Chunk) -> RetrievedChunk:
        if chunk.chunk_id not in merged:
            merged[chunk.chunk_id] = RetrievedChunk(chunk=chunk, fused_score=0.0)
        return merged[chunk.chunk_id]

    for rank, (chunk, score) in enumerate(dense, start=1):
        item = slot(chunk)
        item.dense_score = score
        item.dense_rank = rank
        item.fused_score = (item.fused_score or 0.0) + dense_weight / (k + rank)

    for rank, (chunk, score) in enumerate(lexical, start=1):
        item = slot(chunk)
        item.lexical_score = score
        item.lexical_rank = rank
        item.fused_score = (item.fused_score or 0.0) + lexical_weight / (k + rank)

    for item in merged.values():
        item.score = item.fused_score or 0.0

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def merge_multi_query(runs: Sequence[Sequence[RetrievedChunk]], *, k: int = 60) -> list[RetrievedChunk]:
    """Fuse the fused results of several query variants into a single ranking."""
    merged: dict[str, RetrievedChunk] = {}
    for run in runs:
        for rank, item in enumerate(run, start=1):
            existing = merged.get(item.chunk_id)
            if existing is None:
                existing = RetrievedChunk(
                    chunk=item.chunk,
                    dense_score=item.dense_score,
                    lexical_score=item.lexical_score,
                    dense_rank=item.dense_rank,
                    lexical_rank=item.lexical_rank,
                    fused_score=0.0,
                )
                merged[item.chunk_id] = existing
            existing.fused_score = (existing.fused_score or 0.0) + 1.0 / (k + rank)
            # Keep the strongest per-retriever evidence seen across variants.
            if item.dense_score is not None and (
                existing.dense_score is None or item.dense_score > existing.dense_score
            ):
                existing.dense_score, existing.dense_rank = item.dense_score, item.dense_rank
            if item.lexical_score is not None and (
                existing.lexical_score is None or item.lexical_score > existing.lexical_score
            ):
                existing.lexical_score, existing.lexical_rank = item.lexical_score, item.lexical_rank

    for item in merged.values():
        item.score = item.fused_score or 0.0
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)
