"""Verification layer: independent groundedness checking of generated answers."""

from __future__ import annotations

from .grounding import GroundingReport, GroundingVerifier, best_window, support_score

__all__ = ["GroundingReport", "GroundingVerifier", "best_window", "support_score"]
