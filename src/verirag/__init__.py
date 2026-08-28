"""VeriRAG — verifiable retrieval-augmented QA over PDFs.

Every answer is returned with span-level proof: the document, the page, the
line range, the quoted evidence and a rendered image of the page with the
cited lines highlighted.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
