"""PDF ingestion: geometry-preserving parsing and line-aware chunking."""

from __future__ import annotations

from .chunker import chunk_lines
from .pdf_parser import extract_lines, line_index, lines_by_page, parse_pdf

__all__ = ["chunk_lines", "extract_lines", "line_index", "lines_by_page", "parse_pdf"]
