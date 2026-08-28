"""Text embedding with a zero-dependency default and a neural upgrade path.

Two interchangeable implementations sit behind :func:`get_embedder`:

``HashingEmbedder`` (default)
    A deterministic, corpus-fitted hashed n-gram vectoriser with IDF weighting
    and signed hashing.  Needs nothing beyond NumPy, downloads nothing, runs
    instantly, and is fully reproducible — which is why it is the default for
    a project that must be free and offline-friendly.

``SentenceTransformerEmbedder`` (opt-in)
    ``all-MiniLM-L6-v2`` running locally on CPU.  Still free, but ~2 GB of
    wheels; enable with ``pip install -r requirements-ml.txt`` and
    ``VERIRAG_EMBEDDER=sentence-transformers``.

Both return L2-normalised ``float32`` matrices so cosine similarity is a plain
dot product.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be been being by for from has have had he her his i in is it its of on or
    that the their there they this to was were what when where which who will with would you your""".split()
)


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------
def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens; digits kept (clause numbers, dates, amounts)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def ngram_features(text: str) -> list[str]:
    """Word unigrams + bigrams + char 4-grams of long words."""
    words = tokenize(text)
    features: list[str] = list(words)
    features += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    for word in words:
        if len(word) >= 6:
            features += [f"#{word[i:i + 4]}" for i in range(len(word) - 3)]
    return features


def _hash_bucket(token: str, dim: int) -> tuple[int, float]:
    """Signed hashing: bucket index plus a +/-1 sign to cancel collisions."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used by the index and retrieval layers."""

    name: str
    dim: int

    def fit(self, corpus: Sequence[str]) -> "Embedder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> bool: ...


# ---------------------------------------------------------------------------
# default implementation
# ---------------------------------------------------------------------------
class HashingEmbedder:
    """IDF-weighted signed hashing vectoriser (no downloads, deterministic)."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.name = "hashing"
        self.dim = dim
        self._idf: np.ndarray = np.ones(dim, dtype=np.float32)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, corpus: Sequence[str]) -> "HashingEmbedder":
        n_docs = max(len(corpus), 1)
        doc_freq = np.zeros(self.dim, dtype=np.float32)
        for text in corpus:
            buckets = {_hash_bucket(f, self.dim)[0] for f in ngram_features(text)}
            for bucket in buckets:
                doc_freq[bucket] += 1.0
        self._idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
        self._fitted = True
        return self

    # --------------------------------------------------------------- encode
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            signs: dict[int, float] = {}
            for feature in ngram_features(text):
                bucket, sign = _hash_bucket(feature, self.dim)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
                signs[bucket] = sign
            for bucket, count in counts.items():
                tf = 1.0 + math.log(count)
                matrix[row, bucket] = tf * self._idf[bucket] * signs[bucket]
        return _l2_normalise(matrix)

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "hashing_idf.npy", self._idf)
        (directory / "hashing_meta.json").write_text(
            json.dumps({"dim": self.dim, "fitted": self._fitted}), encoding="utf-8"
        )

    def load(self, directory: Path) -> bool:
        idf_path = directory / "hashing_idf.npy"
        meta_path = directory / "hashing_meta.json"
        if not (idf_path.exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("dim", -1)) != self.dim:
            return False
        self._idf = np.load(idf_path).astype(np.float32)
        self._fitted = bool(meta.get("fitted", False))
        return True


# ---------------------------------------------------------------------------
# optional neural implementation
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Local CPU sentence-transformers encoder (free, downloads once)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.name = "sentence-transformers"
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def fit(self, corpus: Sequence[str]) -> "SentenceTransformerEmbedder":  # noqa: ARG002
        return self  # pretrained; nothing to fit

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "st_meta.json").write_text(
            json.dumps({"model": self.model_name, "dim": self.dim}), encoding="utf-8"
        )

    def load(self, directory: Path) -> bool:  # noqa: ARG002
        return True


# ---------------------------------------------------------------------------
# helpers / factory
# ---------------------------------------------------------------------------
def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalised query against a normalised matrix."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query.reshape(-1).astype(np.float32)


def get_embedder(kind: str = "hashing", *, model_name: str = "", dim: int = 384) -> Embedder:
    """Build an embedder, degrading to :class:`HashingEmbedder` on any failure."""
    normalised = (kind or "hashing").strip().lower()
    if normalised in {"sentence-transformers", "st", "minilm", "neural"}:
        try:
            return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        except Exception:  # noqa: BLE001 - missing wheels, no network, etc.
            pass
    return HashingEmbedder(dim=dim)


def iter_batches(items: Sequence[str], size: int = 256) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
