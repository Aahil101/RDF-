"""Line-aware chunking.

Design rules, all driven by the citation requirement:

1. **A chunk never spans two pages.** A citation must resolve to exactly one
   renderable page so the highlight overlay is unambiguous.
2. **Chunks are contiguous line runs**, so ``line_start``/``line_end`` are
   always meaningful and every line's bbox is retained.
3. **Overlap is expressed in lines, not characters**, keeping boundaries
   aligned with the coordinates we later draw.
4. **Heading lines start a new chunk** and label it, which materially improves
   both retrieval and the readability of citations.
"""

from __future__ import annotations

from ..models import Chunk, PdfLine, stable_id
from .pdf_parser import is_heading, lines_by_page


def _flush(
    buffer: list[PdfLine],
    *,
    doc_id: str,
    doc_name: str,
    section: str,
) -> Chunk | None:
    if not buffer:
        return None
    text = " ".join(ln.text for ln in buffer).strip()
    if not text:
        return None
    page_no = buffer[0].page_no
    line_start = buffer[0].line_no
    line_end = buffer[-1].line_no
    return Chunk(
        chunk_id=stable_id(doc_id, page_no, line_start, line_end),
        doc_id=doc_id,
        doc_name=doc_name,
        page_no=page_no,
        line_start=line_start,
        line_end=line_end,
        text=text,
        line_bboxes=[ln.bbox for ln in buffer],
        section=section,
    )


def chunk_lines(
    lines: list[PdfLine],
    *,
    doc_id: str,
    doc_name: str,
    target_words: int = 140,
    overlap_lines: int = 2,
    min_words: int = 12,
) -> list[Chunk]:
    """Group *lines* into overlapping, page-local, line-tracked chunks."""
    if target_words < 10:
        raise ValueError("target_words must be >= 10")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be >= 0")

    chunks: list[Chunk] = []

    for _page_no, page_lines in sorted(lines_by_page(lines).items()):
        buffer: list[PdfLine] = []
        words = 0
        section = ""

        for line in page_lines:
            # A cell of a multi-column row is never a heading, however
            # heading-like it looks in isolation ("5NF", "BCNF", "1NF").
            heading_here = not line.in_table and is_heading(line.text)

            # A heading closes the previous chunk so sections stay separable.
            if heading_here and words >= min_words:
                chunk = _flush(buffer, doc_id=doc_id, doc_name=doc_name, section=section)
                if chunk:
                    chunks.append(chunk)
                buffer, words = [], 0

            if heading_here:
                section = line.text.strip()

            buffer.append(line)
            words += line.word_count

            if words >= target_words:
                chunk = _flush(buffer, doc_id=doc_id, doc_name=doc_name, section=section)
                if chunk:
                    chunks.append(chunk)
                tail = buffer[-overlap_lines:] if overlap_lines else []
                buffer = list(tail)
                words = sum(ln.word_count for ln in buffer)

        # Trailing remainder of the page.
        if buffer:
            leftover_words = sum(ln.word_count for ln in buffer)
            chunk = _flush(buffer, doc_id=doc_id, doc_name=doc_name, section=section)
            if chunk is None:
                continue
            # Tiny tails are merged back into the previous chunk of the same page
            # instead of becoming near-useless retrieval units.
            if (
                leftover_words < min_words
                and chunks
                and chunks[-1].page_no == chunk.page_no
                and chunks[-1].line_end >= chunk.line_start - 1
            ):
                previous = chunks[-1]
                merged_lines = {ln.line_no: ln for ln in buffer}
                chunks[-1] = Chunk(
                    chunk_id=previous.chunk_id,
                    doc_id=previous.doc_id,
                    doc_name=previous.doc_name,
                    page_no=previous.page_no,
                    line_start=previous.line_start,
                    line_end=max(previous.line_end, chunk.line_end),
                    text=f"{previous.text} {chunk.text}".strip(),
                    line_bboxes=previous.line_bboxes
                    + [ln.bbox for no, ln in sorted(merged_lines.items()) if no > previous.line_end],
                    section=previous.section or chunk.section,
                )
            else:
                chunks.append(chunk)

    return _dedupe(chunks)


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact duplicate spans that overlap windows can produce."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique
