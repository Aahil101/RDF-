"""Ingestion tests: geometry-preserving parsing and line-aware chunking.

These are the highest-value tests in the suite. If a line number or a bounding
box is wrong here, every citation and every highlight downstream is wrong too,
and the failure would be silent — the answer would still look plausible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verirag.ingest.chunker import chunk_lines
from verirag.ingest.pdf_parser import (
    extract_lines,
    is_heading,
    line_index,
    lines_by_page,
    normalise_line,
    parse_pdf,
)


class TestNormalisation:
    def test_collapses_internal_whitespace(self):
        assert normalise_line("The   monthly\trent  is") == "The monthly rent is"

    def test_strips_soft_hyphens_and_newlines(self):
        assert normalise_line("de\u00admised\npremises") == "demised premises"

    def test_blank_line_becomes_empty(self):
        assert normalise_line("   \t  ") == ""


class TestHeadingDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "CLAUSE 1 - RENT AND DEPOSIT",
            "SCHEDULE A - DESCRIPTION",
            "3.1 FUNCTIONAL DEPENDENCY",
            "7. ORDER",
            "UNIT 3 - NORMALIZATION",
        ],
    )
    def test_recognises_headings(self, text):
        assert is_heading(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The monthly rent shall be Rs. 48,500 payable in advance.",
            # A numbered clause body is NOT a heading, even though it starts with
            # a number — this is the case that previously corrupted section labels.
            "1.1 The monthly rent for the demised premises shall be Rs. 48,500 payable in",
            "2.4 The occupation certificate was granted only on 19 November 2022.",
            "(iv) What relief the Appellant is entitled to by way of compensation",
            "",
            "ab",
        ],
    )
    def test_rejects_body_text(self, text):
        assert not is_heading(text)


class TestExtractLines:
    def test_line_numbers_are_one_based_and_per_page(self, sample_pdf: Path):
        lines = extract_lines(sample_pdf)
        pages = lines_by_page(lines)
        assert set(pages) == {1, 2}
        for page_lines in pages.values():
            assert [ln.line_no for ln in page_lines] == list(range(1, len(page_lines) + 1))

    def test_text_content_matches_source(self, sample_pdf: Path):
        lines = extract_lines(sample_pdf)
        first = lines[0]
        assert first.page_no == 1
        assert first.line_no == 1
        assert first.text == "CLAUSE 1 - RENT AND DEPOSIT"

    def test_lines_are_in_reading_order_top_to_bottom(self, sample_pdf: Path):
        for page_lines in lines_by_page(extract_lines(sample_pdf)).values():
            tops = [ln.bbox[1] for ln in page_lines]
            assert tops == sorted(tops)

    def test_every_line_has_a_positive_area_bbox(self, sample_pdf: Path):
        for line in extract_lines(sample_pdf):
            x0, y0, x1, y1 = line.bbox
            assert x1 > x0 and y1 > y0

    def test_line_index_lookup(self, sample_pdf: Path):
        lines = extract_lines(sample_pdf)
        index = line_index(lines)
        assert index[(1, 1)].text == "CLAUSE 1 - RENT AND DEPOSIT"
        assert (2, 1) in index

    def test_parse_pdf_reports_document_metadata(self, sample_pdf: Path):
        document, lines = parse_pdf(sample_pdf)
        assert document.n_pages == 2
        assert document.n_lines == len(lines)
        assert len(document.sha256) == 64
        assert document.doc_id

    def test_empty_pdf_raises_with_actionable_message(self, tmp_path: Path):
        import fitz

        blank = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(blank))
        doc.close()
        with pytest.raises(ValueError, match="OCR"):
            parse_pdf(blank)


class TestChunker:
    def _chunks(self, sample_pdf: Path, **kwargs):
        lines = extract_lines(sample_pdf)
        return chunk_lines(lines, doc_id="doc", doc_name="lease_test.pdf", **kwargs)

    def test_no_chunk_spans_two_pages(self, sample_pdf: Path):
        lines = extract_lines(sample_pdf)
        by_id = {(ln.page_no, ln.line_no): ln for ln in lines}
        for chunk in self._chunks(sample_pdf, target_words=25):
            for line_no in range(chunk.line_start, chunk.line_end + 1):
                assert (chunk.page_no, line_no) in by_id

    def test_line_range_is_ordered_and_consistent(self, sample_pdf: Path):
        for chunk in self._chunks(sample_pdf, target_words=25):
            assert chunk.line_start <= chunk.line_end
            assert chunk.n_lines == chunk.line_end - chunk.line_start + 1
            assert len(chunk.line_bboxes) == chunk.n_lines

    def test_chunk_text_is_reconstructible_from_its_lines(self, sample_pdf: Path):
        lookup = {(ln.page_no, ln.line_no): ln.text for ln in extract_lines(sample_pdf)}
        for chunk in self._chunks(sample_pdf, target_words=25):
            expected = " ".join(
                lookup[(chunk.page_no, n)] for n in range(chunk.line_start, chunk.line_end + 1)
            )
            assert chunk.text == expected

    def test_section_heading_is_attached(self, sample_pdf: Path):
        chunks = self._chunks(sample_pdf, target_words=25)
        assert any(c.section.startswith("CLAUSE 1") for c in chunks)
        assert any(c.section.startswith("CLAUSE 2") for c in chunks)

    def test_body_text_strips_the_heading(self, sample_pdf: Path):
        chunk = next(c for c in self._chunks(sample_pdf, target_words=25) if c.section)
        assert chunk.text.startswith(chunk.section)
        assert not chunk.body_text.startswith(chunk.section)

    def test_locator_format(self, sample_pdf: Path):
        chunk = self._chunks(sample_pdf, target_words=25)[0]
        assert chunk.locator.startswith(f"p.{chunk.page_no} L{chunk.line_start}")

    def test_chunk_ids_are_unique_and_deterministic(self, sample_pdf: Path):
        first = self._chunks(sample_pdf, target_words=25)
        second = self._chunks(sample_pdf, target_words=25)
        ids = [c.chunk_id for c in first]
        assert len(ids) == len(set(ids))
        assert ids == [c.chunk_id for c in second]

    def test_union_bbox_covers_all_line_boxes(self, sample_pdf: Path):
        chunk = self._chunks(sample_pdf, target_words=25)[0]
        x0, y0, x1, y1 = chunk.union_bbox()
        for bx0, by0, bx1, by1 in chunk.line_bboxes:
            assert x0 <= bx0 and y0 <= by0 and x1 >= bx1 and y1 >= by1

    def test_rejects_invalid_parameters(self, sample_pdf: Path):
        lines = extract_lines(sample_pdf)
        with pytest.raises(ValueError):
            chunk_lines(lines, doc_id="d", doc_name="n", target_words=3)
        with pytest.raises(ValueError):
            chunk_lines(lines, doc_id="d", doc_name="n", overlap_lines=-1)

    def test_empty_input_yields_no_chunks(self):
        assert chunk_lines([], doc_id="d", doc_name="n") == []
