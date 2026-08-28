"""Proof layer tests: highlighted page rendering and evidence cropping."""

from __future__ import annotations

import struct
from pathlib import Path

from verirag.proof.highlighter import HighlightSpan, ProofRenderer, spans_for_answer
from verirag.models import Citation

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """Read width/height straight out of the PNG IHDR chunk."""
    assert data.startswith(PNG_MAGIC)
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def citation(page: int = 1, start: int = 2, end: int = 3) -> Citation:
    return Citation(
        marker="S1",
        doc_id="doc",
        doc_name="lease_test.pdf",
        page_no=page,
        line_start=start,
        line_end=end,
        quote="The monthly rent shall be Rs. 48,500.",
        used_in_answer=True,
        bboxes=[(64.0, 100.0, 500.0, 114.0), (64.0, 118.0, 500.0, 132.0)],
    )


class TestProofRenderer:
    def test_renders_a_valid_png(self, settings, sample_pdf: Path):
        renderer = ProofRenderer(settings)
        image = renderer.render(sample_pdf, 1, [HighlightSpan(bboxes=[(64.0, 96.0, 500.0, 112.0)], line_start=2)])
        assert image is not None
        assert image.png.startswith(PNG_MAGIC)
        width, height = png_size(image.png)
        assert width > 400 and height > 600

    def test_highlighting_changes_the_rendered_pixels(self, settings, sample_pdf: Path):
        """A highlight that produced an identical image would be no proof at all."""
        renderer = ProofRenderer(settings)
        plain = renderer.render(sample_pdf, 1, [], use_cache=False)
        marked = renderer.render(
            sample_pdf,
            1,
            [HighlightSpan(bboxes=[(64.0, 96.0, 500.0, 112.0)], line_start=2)],
            use_cache=False,
        )
        assert plain is not None and marked is not None
        assert plain.png != marked.png

    def test_render_citation_convenience(self, settings, sample_pdf: Path):
        image = ProofRenderer(settings).render_citation(sample_pdf, citation())
        assert image is not None
        assert image.page_no == 1
        assert image.labels

    def test_second_call_is_served_from_cache(self, settings, sample_pdf: Path):
        renderer = ProofRenderer(settings)
        spans = [HighlightSpan(bboxes=[(64.0, 96.0, 500.0, 112.0)], line_start=2)]
        first = renderer.render(sample_pdf, 1, spans)
        assert first is not None and first.path is not None and first.path.exists()
        second = renderer.render(sample_pdf, 1, spans)
        assert second is not None
        assert second.png == first.png

    def test_context_and_proven_spans_render_differently(self, settings, sample_pdf: Path):
        renderer = ProofRenderer(settings)
        box = [(64.0, 96.0, 500.0, 112.0)]
        proven = renderer.render(sample_pdf, 1, [HighlightSpan(bboxes=box, proven=True, line_start=2)], use_cache=False)
        context = renderer.render(sample_pdf, 1, [HighlightSpan(bboxes=box, proven=False, line_start=2)], use_cache=False)
        assert proven is not None and context is not None
        assert proven.png != context.png

    def test_out_of_range_page_returns_none(self, settings, sample_pdf: Path):
        assert ProofRenderer(settings).render(sample_pdf, 99, []) is None

    def test_missing_file_returns_none(self, settings, tmp_path: Path):
        assert ProofRenderer(settings).render(tmp_path / "nope.pdf", 1, []) is None

    def test_crop_is_smaller_than_the_full_page(self, settings, sample_pdf: Path):
        renderer = ProofRenderer(settings)
        full = renderer.render(sample_pdf, 1, [], use_cache=False)
        crop = renderer.crop(sample_pdf, 1, [(64.0, 96.0, 500.0, 112.0)])
        assert full is not None and crop is not None
        assert png_size(crop)[1] < png_size(full.png)[1]

    def test_crop_without_boxes_returns_none(self, settings, sample_pdf: Path):
        assert ProofRenderer(settings).crop(sample_pdf, 1, []) is None


class TestSpanGrouping:
    def test_groups_by_document_and_page(self):
        grouped = spans_for_answer([citation(page=1), citation(page=2)])
        assert set(grouped) == {("doc", 1), ("doc", 2)}

    def test_skips_unused_citations_by_default(self):
        unused = citation()
        unused.used_in_answer = False
        assert spans_for_answer([unused]) == {}

    def test_includes_unused_when_requested(self):
        unused = citation()
        unused.used_in_answer = False
        assert spans_for_answer([unused], only_used=False)

    def test_skips_citations_without_geometry(self):
        bare = citation()
        bare.bboxes = []
        assert spans_for_answer([bare]) == {}


class TestEngineProofIntegration:
    def test_ask_produces_a_proof_image_per_cited_page(self, engine):
        result = engine.ask("What is the monthly rent?", render_proof=True, persist=False)
        assert result.proofs
        for image in result.proofs:
            assert image.png.startswith(PNG_MAGIC)
            assert image.page_no >= 1

    def test_render_citation_through_the_engine(self, engine):
        result = engine.ask("What is the security deposit?", render_proof=False, persist=False)
        image = engine.render_citation(result.answer.used_citations[0])
        assert image is not None and image.png.startswith(PNG_MAGIC)

    def test_crop_citation_through_the_engine(self, engine):
        result = engine.ask("What is the security deposit?", render_proof=False, persist=False)
        data = engine.crop_citation(result.answer.used_citations[0])
        assert data is not None and data.startswith(PNG_MAGIC)
