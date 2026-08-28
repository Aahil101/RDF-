"""Domain models shared by every VeriRAG stage.

The single most important idea in this project lives here: a retrievable unit
of text never loses its *physical location* in the source PDF.  Every
:class:`Chunk` carries the page number, the 1-based line range and the exact
bounding boxes of the lines it was built from, which is what makes
page + line citations and on-page visual highlighting possible.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

BBox = tuple[float, float, float, float]
"""PDF rectangle in PyMuPDF coordinates: ``(x0, y0, x1, y1)``, origin top-left."""


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PdfLine:
    """One visual line of text on one page, with its geometry preserved."""

    page_no: int  # 1-based
    line_no: int  # 1-based, per page, in reading order
    text: str
    bbox: BBox
    row_span: int = 1
    """How many text objects share this line's baseline.

    ``> 1`` means the line is one cell of a multi-column row — a table. Knowing
    this from *layout* rather than from string heuristics is what stops a cell
    like ``5NF`` being mistaken for a section heading.
    """

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def in_table(self) -> bool:
        return self.row_span > 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """A contiguous run of lines from a single page — the retrieval unit."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_no: int
    line_start: int
    line_end: int
    text: str
    line_bboxes: list[BBox] = field(default_factory=list)
    section: str = ""

    # ------------------------------------------------------------------ utils
    @property
    def locator(self) -> str:
        """Human-readable citation locator, e.g. ``p.3 L12-18``."""
        if self.line_start == self.line_end:
            return f"p.{self.page_no} L{self.line_start}"
        return f"p.{self.page_no} L{self.line_start}-{self.line_end}"

    @property
    def n_lines(self) -> int:
        return self.line_end - self.line_start + 1

    @property
    def body_text(self) -> str:
        """Chunk text without its leading section heading.

        The heading stays in :attr:`text` because it is high-signal for
        retrieval and because dropping it would break the line-range contract.
        Quoting prose back to the user, however, reads badly with a shouted
        heading glued to the front of the first sentence.
        """
        if self.section and self.text.startswith(self.section):
            return self.text[len(self.section) :].lstrip(" -–—:.")
        return self.text

    def union_bbox(self) -> BBox | None:
        if not self.line_bboxes:
            return None
        x0 = min(b[0] for b in self.line_bboxes)
        y0 = min(b[1] for b in self.line_bboxes)
        x1 = max(b[2] for b in self.line_bboxes)
        y1 = max(b[3] for b in self.line_bboxes)
        return (x0, y0, x1, y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        data = dict(payload)
        data["line_bboxes"] = [tuple(b) for b in data.get("line_bboxes", [])]
        return cls(**data)


@dataclass(slots=True)
class Document:
    """An ingested PDF plus its provenance."""

    doc_id: str
    name: str
    path: str
    n_pages: int
    n_lines: int
    n_chunks: int
    sha256: str
    ingested_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RetrievedChunk:
    """A chunk with the scores that got it into the context window."""

    chunk: Chunk
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def explain(self) -> str:
        """Short, human-auditable trace of why this chunk was selected."""
        bits: list[str] = []
        if self.dense_rank is not None:
            bits.append(f"dense#{self.dense_rank}")
        if self.lexical_rank is not None:
            bits.append(f"bm25#{self.lexical_rank}")
        if self.fused_score is not None:
            bits.append(f"rrf={self.fused_score:.4f}")
        if self.rerank_score is not None:
            bits.append(f"rerank={self.rerank_score:.3f}")
        return " | ".join(bits) or "n/a"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["locator"] = self.chunk.locator
        return payload


# ---------------------------------------------------------------------------
# answering + proof
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EvidenceSpan:
    """The narrowest proven region of the PDF supporting a claim."""

    doc_id: str
    doc_name: str
    page_no: int
    line_start: int
    line_end: int
    text: str
    bboxes: list[BBox] = field(default_factory=list)
    similarity: float = 0.0

    @property
    def locator(self) -> str:
        if self.line_start == self.line_end:
            return f"p.{self.page_no} L{self.line_start}"
        return f"p.{self.page_no} L{self.line_start}-{self.line_end}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    """A source the model was given, and which the answer referenced."""

    marker: str  # "S1", "S2", ...
    doc_id: str
    doc_name: str
    page_no: int
    line_start: int
    line_end: int
    quote: str
    retrieval_score: float = 0.0
    used_in_answer: bool = False
    bboxes: list[BBox] = field(default_factory=list)

    @property
    def locator(self) -> str:
        if self.line_start == self.line_end:
            return f"p.{self.page_no} L{self.line_start}"
        return f"p.{self.page_no} L{self.line_start}-{self.line_end}"

    @property
    def label(self) -> str:
        return f"[{self.marker}] {self.doc_name} — {self.locator}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimVerdict:
    """Per-sentence groundedness verdict — the hallucination guard's output."""

    sentence: str
    supported: bool
    score: float
    markers: list[str] = field(default_factory=list)
    evidence: EvidenceSpan | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = self.evidence.to_dict() if self.evidence else None
        return payload


@dataclass(slots=True)
class Answer:
    """Everything the UI/CLI needs to render a verifiable answer."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    groundedness: float = 0.0
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    refused: bool = False
    weak_evidence: bool = False
    retrieval_score: float = 0.0
    provider_error: str = ""
    """Why the requested LLM was not used, when a fallback happened.

    A silent fallback is worse than a failure: the answer still looks fine, so a
    misconfigured model name or an exhausted quota would go unnoticed.
    """
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    @property
    def unsupported_claims(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if not v.supported]

    @property
    def used_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.used_in_answer]

    def confidence_band(self) -> str:
        """Confidence combines *how well* the answer is grounded with *how
        strongly* the evidence was retrieved.

        The two signals answer different questions and neither is sufficient:

        * Groundedness asks "is this claim actually in the cited lines?" It
          cannot detect an answer that is perfectly supported by a passage which
          does not address the question.
        * Retrieval score asks "did the passage match the question?" It drops
          naturally for long, multi-part questions, because coverage is divided
          over more query terms — so treating a weak score as decisive would
          label correct, verified answers as low confidence.

        Verified support therefore *upgrades* a weak retrieval score to medium
        rather than being overridden by it.
        """
        if self.refused:
            return "refused"
        if self.groundedness >= 0.75:
            return "medium" if self.weak_evidence else "high"
        if self.groundedness >= 0.5:
            return "low" if self.weak_evidence else "medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "groundedness": self.groundedness,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "refused": self.refused,
            "weak_evidence": self.weak_evidence,
            "retrieval_score": self.retrieval_score,
            "provider_error": self.provider_error,
            "confidence": self.confidence_band(),
            "retrieval_trace": self.retrieval_trace,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stable_id(*parts: Any, length: int = 12) -> str:
    """Deterministic short id — keeps re-ingestion idempotent."""
    joined = "\u241f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_bboxes(bboxes: Iterable[BBox]) -> BBox | None:
    boxes = list(bboxes)
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
