"""Shared pytest fixtures.

Every test runs against a temporary ``data_dir`` and a purpose-built PDF, so the
suite never reads or writes the developer's real index, chat database or sample
corpus. The test PDF is generated with known text at known line positions, which
lets the citation assertions check exact page and line numbers rather than just
"something was returned".
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verirag.config import Settings  # noqa: E402
from verirag.engine import VeriRAG  # noqa: E402

PAGE_ONE = [
    "CLAUSE 1 - RENT AND DEPOSIT",
    "1.1 The monthly rent for the demised premises shall be Rs. 48,500 payable in",
    "advance on or before the fifth day of every calendar month.",
    "1.2 The Lessee has paid an interest-free security deposit of Rs. 2,91,000 being",
    "equivalent to six months rent, the receipt whereof is acknowledged.",
    "1.3 The rent shall stand escalated by six per cent every eleven months.",
]

PAGE_TWO = [
    "CLAUSE 2 - TERMINATION",
    "2.1 Either party may terminate this lease by giving three months prior notice",
    "in writing to the other party.",
    "2.2 The Lessor may terminate forthwith upon default in payment of rent for two",
    "consecutive months.",
    "2.3 The Lessee shall pay Rs. 22,000 towards repainting charges on vacating.",
]


def _write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a PDF with one text line per supplied string, at fixed positions."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 90.0
        for line in lines:
            page.insert_text(fitz.Point(64.0, y), line, fontname="helv", fontsize=11)
            y += 18.0
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path), deflate=True)
    doc.close()
    return path


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated, fast, deterministic settings.

    Every value that the developer's ``.env`` could change is pinned here. Without
    this the suite silently inherits local configuration — loading a 90 MB neural
    model per fixture, and passing or failing depending on whose machine it runs
    on. Tests assert behaviour, so they fix the configuration that behaviour
    depends on.
    """
    config = Settings()
    config.data_dir = tmp_path / "data"
    config.llm_provider = "extractive"
    config.embedder = "hashing"      # no model download, deterministic vectors
    config.reranker = "lexical"      # no cross-encoder load
    config.vector_backend = "numpy"
    config.embed_dim = 384
    config.chunk_target_words = 40
    config.chunk_overlap_lines = 1
    config.min_retrieval_score = 0.10
    config.low_confidence_score = 0.35
    return config.ensure_dirs()


@pytest.fixture()
def sample_pdf(settings: Settings) -> Path:
    return _write_pdf(settings.raw_dir / "lease_test.pdf", [PAGE_ONE, PAGE_TWO])


@pytest.fixture()
def second_pdf(settings: Settings) -> Path:
    return _write_pdf(
        settings.raw_dir / "notes_test.pdf",
        [
            [
                "UNIT 3 - NORMALIZATION",
                "A relation is in Boyce-Codd normal form if for every non-trivial",
                "functional dependency the determinant is a superkey of the relation.",
                "Repeatable Read prevents dirty reads but still permits phantom reads.",
            ]
        ],
    )


@pytest.fixture()
def engine(settings: Settings, sample_pdf: Path) -> VeriRAG:
    instance = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
    reports = instance.ingest(sample_pdf)
    assert reports[0].ok, reports[0].error
    return instance
