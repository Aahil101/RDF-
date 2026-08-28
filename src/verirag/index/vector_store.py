"""Persistent dense vector store.

``NumpyVectorStore`` is the default: an exact (brute-force) cosine index backed
by a single ``.npz`` for vectors and a JSON sidecar for chunk metadata.  For
the corpus sizes this project targets (10k-100k chunks) exact search is both
faster to build and more accurate than an approximate index, and it keeps the
dependency footprint at zero.

``ChromaVectorStore`` is a thin adapter kept behind the same protocol to show
the abstraction is real — switch with ``VERIRAG_VECTOR_BACKEND=chroma``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..models import Chunk
from .embedder import cosine_scores


@runtime_checkable
class VectorStore(Protocol):
    """Interface the retrieval layer depends on."""

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def save(self) -> None: ...

    def load(self) -> bool: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
class NumpyVectorStore:
    """Exact cosine-similarity store persisted to ``vectors.npz`` + JSON."""

    def __init__(self, directory: Path, dim: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}

    # ------------------------------------------------------------------ paths
    @property
    def vectors_path(self) -> Path:
        return self.directory / "vectors.npz"

    @property
    def chunks_path(self) -> Path:
        return self.directory / "chunks.json"

    # -------------------------------------------------------------- mutation
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunk/vector count mismatch: {len(chunks)} vs {vectors.shape[0]}")
        if vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")

        fresh_chunks: list[Chunk] = []
        fresh_rows: list[np.ndarray] = []
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in self._by_id:  # idempotent re-ingestion
                self._matrix[self._by_id[chunk.chunk_id]] = vector
                self._chunks[self._by_id[chunk.chunk_id]] = chunk
                continue
            fresh_chunks.append(chunk)
            fresh_rows.append(np.asarray(vector, dtype=np.float32))

        if fresh_chunks:
            block = np.vstack(fresh_rows).astype(np.float32)
            self._matrix = block if self._matrix.size == 0 else np.vstack([self._matrix, block])
            for chunk in fresh_chunks:
                self._by_id[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)

    def remove_document(self, doc_id: str) -> int:
        """Drop every chunk of *doc_id*; returns how many were removed."""
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self._by_id = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        return removed

    def clear(self) -> None:
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._chunks = []
        self._by_id = {}
        for path in (self.vectors_path, self.chunks_path):
            path.unlink(missing_ok=True)

    # ---------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self._chunks or k <= 0:
            return []
        scores = cosine_scores(np.asarray(query_vector, dtype=np.float32), self._matrix)
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in top]

    # ----------------------------------------------------------------- reads
    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        index = self._by_id.get(chunk_id)
        return self._chunks[index] if index is not None else None

    def doc_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for chunk in self._chunks:
            seen.setdefault(chunk.doc_id, None)
        return list(seen)

    def __len__(self) -> int:
        return len(self._chunks)

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        np.savez_compressed(self.vectors_path, matrix=self._matrix)
        payload = {"dim": self.dim, "chunks": [c.to_dict() for c in self._chunks]}
        self.chunks_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        if not (self.vectors_path.exists() and self.chunks_path.exists()):
            return False
        payload = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if int(payload.get("dim", -1)) != self.dim:
            return False  # embedder changed -> stale index, force a rebuild
        self._chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        with np.load(self.vectors_path) as data:
            self._matrix = data["matrix"].astype(np.float32)
        if self._matrix.shape[0] != len(self._chunks):
            self.clear()
            return False
        self._by_id = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        return True


# ---------------------------------------------------------------------------
class ChromaVectorStore:
    """Optional ChromaDB adapter (persistent client, cosine space)."""

    def __init__(self, directory: Path, dim: int, collection: str = "verirag") -> None:
        import chromadb  # noqa: PLC0415

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._client = chromadb.PersistentClient(path=str(self.directory / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[{"payload": json.dumps(c.to_dict(), ensure_ascii=False)} for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: list[tuple[Chunk, float]] = []
        for metadata, distance in zip(metadatas, distances):
            chunk = Chunk.from_dict(json.loads(metadata["payload"]))
            out.append((chunk, 1.0 - float(distance)))  # cosine distance -> similarity
        return out

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["metadatas"])
        return [Chunk.from_dict(json.loads(m["payload"])) for m in result.get("metadatas") or []]

    def get(self, chunk_id: str) -> Chunk | None:
        result = self._collection.get(ids=[chunk_id], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return Chunk.from_dict(json.loads(metadatas[0]["payload"])) if metadatas else None

    def save(self) -> None:
        return None  # Chroma persists on write

    def load(self) -> bool:
        return self._collection.count() > 0

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return int(self._collection.count())


def get_vector_store(backend: str, directory: Path, dim: int) -> VectorStore:
    """Factory that falls back to NumPy if Chroma is unavailable."""
    if (backend or "numpy").strip().lower() == "chroma":
        try:
            return ChromaVectorStore(directory, dim)
        except Exception:  # noqa: BLE001
            pass
    return NumpyVectorStore(directory, dim)
