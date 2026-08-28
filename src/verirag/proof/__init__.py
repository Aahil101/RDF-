"""Proof layer: render source pages with cited lines visually highlighted."""

from __future__ import annotations

from .highlighter import HighlightSpan, ProofImage, ProofRenderer, spans_for_answer

__all__ = ["HighlightSpan", "ProofImage", "ProofRenderer", "spans_for_answer"]
