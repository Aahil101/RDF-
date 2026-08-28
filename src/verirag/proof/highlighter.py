"""Visual proof: render the source page with the cited lines highlighted.

This is the feature that turns "trust me" into "look at it".  Because chunks
kept their line bounding boxes through ingestion, a citation can be drawn back
onto the original page:

* **amber block** — the retrieved chunk that entered the prompt (context),
* **green block + border** — the exact lines the verifier proved support the
  claim,
* **left gutter labels** — the line numbers used in the textual citation, so the
  rendered image and the string ``p.3 L12-18`` can be checked against each other.

Rendered PNGs are cached on disk under ``data/proof_cache`` keyed by a hash of
the request, so re-opening a chat turn is instant.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from ..config import Settings, get_settings
from ..models import BBox, Citation, merge_bboxes

CONTEXT_COLOR: tuple[float, float, float] = (1.00, 0.84, 0.25)  # amber
PROVEN_COLOR: tuple[float, float, float] = (0.30, 0.82, 0.45)  # green
PROVEN_BORDER: tuple[float, float, float] = (0.05, 0.45, 0.20)


@dataclass(slots=True)
class HighlightSpan:
    """One highlightable region: a set of line boxes plus how to style it."""

    bboxes: list[BBox]
    label: str = ""
    proven: bool = True
    line_start: int | None = None
    line_end: int | None = None

    def union(self) -> BBox | None:
        return merge_bboxes(self.bboxes)


@dataclass(slots=True)
class ProofImage:
    """A rendered proof image plus the metadata the UI shows beside it."""

    png: bytes
    path: Path | None
    page_no: int
    doc_name: str
    width: int
    height: int
    labels: list[str] = field(default_factory=list)


def _pad(bbox: BBox, dx: float = 1.5, dy: float = 1.0) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def _cache_key(pdf_path: Path, page_no: int, spans: list[HighlightSpan], dpi: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(pdf_path.resolve()).encode("utf-8"))
    digest.update(f"|{page_no}|{dpi}".encode("utf-8"))
    for span in spans:
        digest.update(f"|{span.proven}|{span.line_start}|{span.line_end}".encode("utf-8"))
        for bbox in span.bboxes:
            digest.update(("|" + ",".join(f"{v:.2f}" for v in bbox)).encode("utf-8"))
    return digest.hexdigest()[:20]


class ProofRenderer:
    """Renders and caches highlighted page images."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

    # ----------------------------------------------------------------- public
    def render(
        self,
        pdf_path: str | Path,
        page_no: int,
        spans: list[HighlightSpan],
        *,
        dpi: int | None = None,
        gutter_labels: bool = True,
        use_cache: bool = True,
    ) -> ProofImage | None:
        """Render page *page_no* (1-based) with *spans* highlighted."""
        path = Path(pdf_path)
        if not path.exists():
            return None
        dpi = dpi or self.settings.highlight_dpi

        cache_path = self.settings.proof_dir / f"{_cache_key(path, page_no, spans, dpi)}.png"
        if use_cache and cache_path.exists():
            png = cache_path.read_bytes()
            with fitz.open(path) as doc:
                page = doc[page_no - 1]
                width, height = int(page.rect.width), int(page.rect.height)
            return ProofImage(
                png=png,
                path=cache_path,
                page_no=page_no,
                doc_name=path.name,
                width=width,
                height=height,
                labels=[s.label for s in spans if s.label],
            )

        with fitz.open(path) as doc:
            if not 1 <= page_no <= doc.page_count:
                return None
            page = doc[page_no - 1]
            self._draw(page, spans, gutter_labels=gutter_labels)
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            png = pixmap.tobytes("png")
            width, height = pixmap.width, pixmap.height

        try:
            cache_path.write_bytes(png)
        except OSError:
            cache_path = None  # type: ignore[assignment]

        return ProofImage(
            png=png,
            path=cache_path,
            page_no=page_no,
            doc_name=path.name,
            width=width,
            height=height,
            labels=[s.label for s in spans if s.label],
        )

    def crop(
        self,
        pdf_path: str | Path,
        page_no: int,
        bboxes: list[BBox],
        *,
        dpi: int | None = None,
        margin: float = 10.0,
    ) -> bytes | None:
        """Tight crop around *bboxes* — a compact thumbnail for citation cards."""
        path = Path(pdf_path)
        union = merge_bboxes(bboxes)
        if not path.exists() or union is None:
            return None
        dpi = dpi or self.settings.highlight_dpi

        with fitz.open(path) as doc:
            if not 1 <= page_no <= doc.page_count:
                return None
            page = doc[page_no - 1]
            for bbox in bboxes:
                page.draw_rect(
                    _pad(bbox),
                    color=None,
                    fill=PROVEN_COLOR,
                    fill_opacity=0.32,
                    overlay=True,
                )
            clip = fitz.Rect(
                max(union[0] - margin, page.rect.x0),
                max(union[1] - margin, page.rect.y0),
                min(union[2] + margin, page.rect.x1),
                min(union[3] + margin, page.rect.y1),
            )
            return page.get_pixmap(dpi=dpi, clip=clip, alpha=False).tobytes("png")

    def render_citation(self, pdf_path: str | Path, citation: Citation, **kwargs) -> ProofImage | None:
        """Convenience wrapper: highlight one citation's proven lines."""
        span = HighlightSpan(
            bboxes=list(citation.bboxes),
            label=f"[{citation.marker}] {citation.locator}",
            proven=True,
            line_start=citation.line_start,
            line_end=citation.line_end,
        )
        return self.render(pdf_path, citation.page_no, [span], **kwargs)

    # ---------------------------------------------------------------- drawing
    def _draw(self, page: "fitz.Page", spans: list[HighlightSpan], *, gutter_labels: bool) -> None:
        # Context spans first so proven spans render on top of them.
        for span in sorted(spans, key=lambda s: s.proven):
            colour = PROVEN_COLOR if span.proven else CONTEXT_COLOR
            opacity = 0.34 if span.proven else 0.22
            for bbox in span.bboxes:
                page.draw_rect(_pad(bbox), color=None, fill=colour, fill_opacity=opacity, overlay=True)

            union = span.union()
            if span.proven and union is not None:
                page.draw_rect(
                    _pad(union, 2.5, 2.0),
                    color=PROVEN_BORDER,
                    width=0.9,
                    fill=None,
                    stroke_opacity=0.85,
                    overlay=True,
                )

            if gutter_labels:
                self._draw_gutter(page, span)

    @staticmethod
    def _draw_gutter(page: "fitz.Page", span: HighlightSpan) -> None:
        """Write line numbers beside the highlighted region.

        Two subtleties, both learned from the rendered output:

        * Labels share one x-column derived from the *leftmost* box in the span.
          Positioning each label relative to its own line makes them collide with
          indented text, because a clause number like "1.3" sits further left
          than its body.
        * A "line" in PyMuPDF is a text object, not a visual row. With a hanging
          indent the clause number and its first line of prose are two separate
          objects sharing one baseline, so labelling every object overstrikes
          two numbers on top of each other. Rows are therefore grouped by
          baseline and labelled once.
        """
        if span.line_start is None or not span.bboxes:
            return

        column_x = max(min(box[0] for box in span.bboxes) - 30.0, page.rect.x0 + 2.0)

        rows: dict[int, tuple[int, float]] = {}
        for offset, bbox in enumerate(span.bboxes):
            line_no = span.line_start + offset
            if span.line_end is not None and line_no > span.line_end:
                break
            key = int(round(bbox[3] / 4.0))  # group boxes sharing a baseline
            existing = rows.get(key)
            if existing is None or line_no < existing[0]:
                rows[key] = (line_no, bbox[3])

        for line_no, baseline in sorted(rows.values(), key=lambda row: row[1]):
            try:
                page.insert_text(
                    fitz.Point(column_x, baseline - 1.0),
                    f"L{line_no}",
                    fontsize=6.2,
                    color=(0.35, 0.35, 0.40),
                    overlay=True,
                )
            except Exception:  # noqa: BLE001 - never fail a render over a label
                return


# ---------------------------------------------------------------------------
def spans_for_answer(citations: list[Citation], *, only_used: bool = True) -> dict[tuple[str, int], list[HighlightSpan]]:
    """Group an answer's citations into ``(doc_id, page) -> spans`` for rendering."""
    grouped: dict[tuple[str, int], list[HighlightSpan]] = {}
    for citation in citations:
        if only_used and not citation.used_in_answer:
            continue
        if not citation.bboxes:
            continue
        key = (citation.doc_id, citation.page_no)
        grouped.setdefault(key, []).append(
            HighlightSpan(
                bboxes=list(citation.bboxes),
                label=f"[{citation.marker}] {citation.locator}",
                proven=True,
                line_start=citation.line_start,
                line_end=citation.line_end,
            )
        )
    return grouped
