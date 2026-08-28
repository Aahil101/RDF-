"""PDF text extraction that *keeps the geometry*.

Most RAG tutorials call ``page.get_text()`` and throw away every coordinate,
which makes real citations impossible.  VeriRAG instead walks PyMuPDF's
structured output and emits one :class:`~verirag.models.PdfLine` per visual
line, carrying:

* the 1-based page number,
* a 1-based line number assigned in reading order within that page,
* the normalised line text,
* the exact bounding box, later used to draw highlight rectangles.

Line numbers are deterministic for a given PDF, so a citation such as
``p.3 L12-18`` is reproducible and auditable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

from ..models import BBox, Document, PdfLine, sha256_file, stable_id

_WS_RE = re.compile(r"[ \t\u00a0]+")
_SOFT_HYPHEN = "\u00ad"


def normalise_line(text: str) -> str:
    """Collapse intra-line whitespace and drop soft hyphens."""
    cleaned = text.replace(_SOFT_HYPHEN, "").replace("\r", " ").replace("\n", " ")
    return _WS_RE.sub(" ", cleaned).strip()


# Words that make a line a heading regardless of casing.
_HEADING_KEYWORDS = (
    "SCHEDULE",
    "ANNEXURE",
    "CLAUSE",
    "SECTION",
    "CHAPTER",
    "PART",
    "ARTICLE",
    "UNIT",
    "MODULE",
    "APPENDIX",
    "EXHIBIT",
)

# A leading enumerator such as "3.1", "IV.", "2)" — stripped before judging.
_ENUMERATOR_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+")

_HEADING_MAX_LEN = 90
_HEADING_CAPS_RATIO = 0.6


def _caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def is_heading(text: str) -> bool:
    """Heuristic heading detector used to label chunks with a section.

    The tempting rule — "starts with a number, therefore a heading" — is wrong
    for exactly the documents this project targets: every clause of a contract
    and every numbered paragraph of a judgment begins that way. Treating those
    as headings corrupts the section label and truncates quoted text.

    A heading is therefore a *short, shouted, non-sentence* line: strip any
    leading enumerator, then require either a structural keyword (CLAUSE,
    SCHEDULE, ...) or predominantly upper-case text.
    """
    stripped = text.strip()
    if not (3 <= len(stripped) <= 120):
        return False

    body = _ENUMERATOR_RE.sub("", stripped).strip()
    if not body or len(body) > _HEADING_MAX_LEN:
        return False
    if body.endswith((".", ";", ",", ":")) and not body.isupper():
        return False

    if any(body.upper().startswith(keyword) for keyword in _HEADING_KEYWORDS):
        return True
    return _caps_ratio(body) >= _HEADING_CAPS_RATIO


def _reading_order_key(bbox: BBox, y_tolerance: float = 2.5) -> tuple[float, float]:
    """Sort key that tolerates sub-pixel baseline jitter within a text row."""
    x0, y0, _x1, _y1 = bbox
    return (round(y0 / y_tolerance), x0)


def _row_key(bbox: BBox, y_tolerance: float = 4.0) -> int:
    """Bucket a bbox by baseline so cells of one table row group together."""
    return int(round(bbox[3] / y_tolerance))


def extract_lines(pdf_path: str | Path) -> list[PdfLine]:
    """Return every non-empty text line of *pdf_path* with page/line/bbox."""
    path = Path(pdf_path)
    lines: list[PdfLine] = []

    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            raw: list[tuple[str, BBox]] = []
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 == text block; 1 == image
                    continue
                for line in block.get("lines", []):
                    text = normalise_line("".join(span.get("text", "") for span in line.get("spans", [])))
                    if not text:
                        continue
                    bbox = tuple(float(v) for v in line["bbox"])  # type: ignore[assignment]
                    raw.append((text, bbox))

            raw.sort(key=lambda item: _reading_order_key(item[1]))

            # Count how many text objects share each baseline. A row with several
            # objects is a table row, and its cells must not be read as headings.
            row_counts: dict[int, int] = {}
            for _text, bbox in raw:
                row_counts[_row_key(bbox)] = row_counts.get(_row_key(bbox), 0) + 1

            for line_no, (text, bbox) in enumerate(raw, start=1):
                lines.append(
                    PdfLine(
                        page_no=page_index,
                        line_no=line_no,
                        text=text,
                        bbox=bbox,
                        row_span=row_counts.get(_row_key(bbox), 1),
                    )
                )

    return lines


def page_count(pdf_path: str | Path) -> int:
    with fitz.open(Path(pdf_path)) as doc:
        return doc.page_count


def lines_by_page(lines: list[PdfLine]) -> dict[int, list[PdfLine]]:
    """Group lines per page, preserving reading order."""
    grouped: dict[int, list[PdfLine]] = {}
    for line in lines:
        grouped.setdefault(line.page_no, []).append(line)
    for page_lines in grouped.values():
        page_lines.sort(key=lambda ln: ln.line_no)
    return grouped


def line_index(lines: list[PdfLine]) -> dict[tuple[int, int], PdfLine]:
    """``(page_no, line_no) -> PdfLine`` lookup used by the proof layer."""
    return {(ln.page_no, ln.line_no): ln for ln in lines}


def build_document(pdf_path: str | Path, n_chunks: int = 0, n_lines: int | None = None) -> Document:
    """Create the :class:`Document` record for an ingested PDF."""
    path = Path(pdf_path).resolve()
    digest = sha256_file(str(path))
    total_lines = n_lines if n_lines is not None else len(extract_lines(path))
    return Document(
        doc_id=stable_id(path.name, digest),
        name=path.name,
        path=str(path),
        n_pages=page_count(path),
        n_lines=total_lines,
        n_chunks=n_chunks,
        sha256=digest,
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def parse_pdf(pdf_path: str | Path) -> tuple[Document, list[PdfLine]]:
    """Parse *pdf_path* once, returning its metadata and all located lines."""
    lines = extract_lines(pdf_path)
    if not lines:
        raise ValueError(
            f"No extractable text found in {Path(pdf_path).name}. "
            "It is probably a scanned PDF — run OCR (e.g. ocrmypdf) first."
        )
    document = build_document(pdf_path, n_lines=len(lines))
    return document, lines
