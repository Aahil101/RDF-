"""Index orchestration: PDF -> lines -> chunks -> dense + lexical indexes.

Also owns the *line store*: a compact JSON of every parsed line with its bbox.
The proof layer reads it to narrow a chunk-level citation down to the specific
lines that support a sentence, and to draw highlight rectangles without
re-parsing the PDF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..config import Settings, get_settings
from ..ingest.chunker import chunk_lines
from ..ingest.pdf_parser import parse_pdf
from ..models import Chunk, Document, PdfLine
from .bm25_store import BM25Store
from .embedder import Embedder, get_embedder
from .vector_store import VectorStore, get_vector_store


@dataclass(slots=True)
class IngestReport:
    """Per-file outcome of an ingestion run."""

    document: Document | None
    chunks: int
    lines: int
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.document is not None and not self.error


class LineStore:
    """``doc_id -> page -> line_no -> (text, bbox)`` persisted as JSON."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, dict[str, object]]] = {}

    @property
    def path(self) -> Path:
        return self.directory / "lines.json"

    def put(self, doc_id: str, lines: Iterable[PdfLine]) -> None:
        bucket: dict[str, dict[str, object]] = {}
        for line in lines:
            bucket.setdefault(str(line.page_no), {})[str(line.line_no)] = {
                "t": line.text,
                "b": list(line.bbox),
                "r": line.row_span,
            }
        self._data[doc_id] = bucket

    def drop(self, doc_id: str) -> None:
        self._data.pop(doc_id, None)

    def get_lines(self, doc_id: str, page_no: int) -> list[PdfLine]:
        page = self._data.get(doc_id, {}).get(str(page_no), {})
        out = [
            PdfLine(
                page_no=page_no,
                line_no=int(line_no),
                text=str(payload["t"]),  # type: ignore[index]
                bbox=tuple(float(v) for v in payload["b"]),  # type: ignore[index,arg-type]
                row_span=int(payload.get("r", 1)),  # type: ignore[union-attr]
            )
            for line_no, payload in page.items()
        ]
        out.sort(key=lambda ln: ln.line_no)
        return out

    def get_range(self, doc_id: str, page_no: int, line_start: int, line_end: int) -> list[PdfLine]:
        return [ln for ln in self.get_lines(doc_id, page_no) if line_start <= ln.line_no <= line_end]

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        if not self.path.exists():
            return False
        self._data = json.loads(self.path.read_text(encoding="utf-8"))
        return True

    def clear(self) -> None:
        self._data = {}
        self.path.unlink(missing_ok=True)


class DocumentRegistry:
    """Tracks ingested documents so re-ingestion is idempotent by content hash."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, Document] = {}

    @property
    def path(self) -> Path:
        return self.directory / "documents.json"

    def put(self, document: Document) -> None:
        self._docs[document.doc_id] = document

    def drop(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def by_hash(self, sha256: str) -> Document | None:
        return next((d for d in self._docs.values() if d.sha256 == sha256), None)

    def all(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.name.lower())

    def save(self) -> None:
        self.path.write_text(
            json.dumps({k: v.to_dict() for k, v in self._docs.items()}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.path.exists():
            return False
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._docs = {k: Document(**v) for k, v in raw.items()}
        return True

    def clear(self) -> None:
        self._docs = {}
        self.path.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._docs)


class Indexer:
    """Owns every persistent artefact of the knowledge base."""

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.embedder: Embedder = embedder or get_embedder(
            self.settings.embedder,
            model_name=self.settings.embed_model,
            dim=self.settings.embed_dim,
        )
        self.vectors: VectorStore = get_vector_store(
            self.settings.vector_backend, self.settings.index_dir, self.embedder.dim
        )
        self.bm25 = BM25Store(self.settings.index_dir)
        self.lines = LineStore(self.settings.index_dir)
        self.registry = DocumentRegistry(self.settings.index_dir)

    # ---------------------------------------------------------------- ingest
    def ingest_pdf(self, pdf_path: str | Path, *, force: bool = False) -> IngestReport:
        """Parse, chunk, embed and index a single PDF."""
        path = Path(pdf_path)
        try:
            document, lines = parse_pdf(path)
        except Exception as exc:  # noqa: BLE001 - report, never crash a batch
            return IngestReport(document=None, chunks=0, lines=0, error=str(exc))

        existing = self.registry.by_hash(document.sha256)
        if existing and not force:
            return IngestReport(document=existing, chunks=existing.n_chunks, lines=existing.n_lines, skipped=True)

        chunks = chunk_lines(
            lines,
            doc_id=document.doc_id,
            doc_name=document.name,
            target_words=self.settings.chunk_target_words,
            overlap_lines=self.settings.chunk_overlap_lines,
            min_words=self.settings.min_chunk_words,
        )
        if not chunks:
            return IngestReport(document=None, chunks=0, lines=len(lines), error="no chunks produced")

        if existing and force and hasattr(self.vectors, "remove_document"):
            self.vectors.remove_document(existing.doc_id)  # type: ignore[attr-defined]

        document.n_chunks = len(chunks)
        self.registry.put(document)
        self.lines.put(document.doc_id, lines)
        self._embed_and_add(chunks)
        self._rebuild_lexical()
        return IngestReport(document=document, chunks=len(chunks), lines=len(lines))

    def ingest_directory(self, directory: str | Path, *, force: bool = False) -> list[IngestReport]:
        pdfs = sorted(Path(directory).glob("*.pdf"))
        return [self.ingest_pdf(pdf, force=force) for pdf in pdfs]

    def _embed_and_add(self, chunks: Sequence[Chunk]) -> None:
        corpus = [self._embed_text(c) for c in chunks]
        # The hashing embedder learns IDF from the corpus, so refit over
        # everything currently indexed to keep vectors comparable.
        if self.embedder.name == "hashing":
            existing = [self._embed_text(c) for c in self.vectors.all_chunks()]
            self.embedder.fit(existing + corpus)
            if existing:
                stale = self.vectors.all_chunks()
                self.vectors.add(stale, self.embedder.encode(existing))
        vectors = self.embedder.encode(corpus)
        self.vectors.add(chunks, vectors)

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        prefix = f"{chunk.doc_name} | {chunk.section} | " if chunk.section else f"{chunk.doc_name} | "
        return f"{prefix}{chunk.text}"

    def _rebuild_lexical(self) -> None:
        self.bm25.build(self.vectors.all_chunks())

    # ------------------------------------------------------------ maintenance
    def delete_document(self, doc_id: str) -> bool:
        document = self.registry.get(doc_id)
        if document is None:
            return False
        if hasattr(self.vectors, "remove_document"):
            self.vectors.remove_document(doc_id)  # type: ignore[attr-defined]
        self.registry.drop(doc_id)
        self.lines.drop(doc_id)
        self._rebuild_lexical()
        self.save()
        return True

    def reset(self) -> None:
        self.vectors.clear()
        self.bm25.clear()
        self.lines.clear()
        self.registry.clear()

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        self.vectors.save()
        self.bm25.save()
        self.lines.save()
        self.registry.save()
        self.embedder.save(self.settings.index_dir)

    def load(self) -> bool:
        self.embedder.load(self.settings.index_dir)
        loaded = self.vectors.load()
        self.registry.load()
        self.lines.load()
        if loaded and not self.bm25.load():
            self._rebuild_lexical()
        return loaded and len(self.vectors) > 0

    # ------------------------------------------------------------------ info
    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self.registry),
            "chunks": len(self.vectors),
            "embedder": self.embedder.name,
            "embed_dim": self.embedder.dim,
            "vector_backend": type(self.vectors).__name__,
            "bm25_impl": self.bm25.impl,
            "index_dir": str(self.settings.index_dir),
        }
