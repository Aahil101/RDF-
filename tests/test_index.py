"""Indexing tests: embeddings, dense store persistence and BM25 parity."""

from __future__ import annotations

import numpy as np
import pytest

from verirag.index.bm25_store import BM25Store, InternalBM25
from verirag.index.embedder import HashingEmbedder, cosine_scores, get_embedder, tokenize
from verirag.index.vector_store import NumpyVectorStore
from verirag.models import Chunk

CORPUS = [
    "The monthly rent for the demised premises shall be Rs. 48,500 payable in advance.",
    "The Lessee has paid an interest-free security deposit of Rs. 2,91,000.",
    "Either party may terminate this lease by giving three months prior notice.",
    "A relation is in Boyce-Codd normal form if every determinant is a superkey.",
]


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        doc_id="doc",
        doc_name="test.pdf",
        page_no=1,
        line_start=index * 2 + 1,
        line_end=index * 2 + 2,
        text=text,
        line_bboxes=[(10.0, 10.0 + index * 20, 500.0, 24.0 + index * 20)],
    )


class TestTokenizer:
    def test_keeps_digits_because_clause_numbers_matter(self):
        assert "48" in tokenize("Rs. 48,500")
        assert "500" in tokenize("Rs. 48,500")

    def test_lowercases(self):
        assert tokenize("Monthly RENT") == ["monthly", "rent"]

    def test_optional_stopword_removal(self):
        assert "the" not in tokenize("the rent", drop_stopwords=True)


class TestHashingEmbedder:
    def test_output_shape_and_normalisation(self):
        embedder = HashingEmbedder(dim=128).fit(CORPUS)
        vectors = embedder.encode(CORPUS)
        assert vectors.shape == (4, 128)
        assert vectors.dtype == np.float32
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_across_instances(self):
        first = HashingEmbedder(dim=128).fit(CORPUS).encode(CORPUS)
        second = HashingEmbedder(dim=128).fit(CORPUS).encode(CORPUS)
        assert np.allclose(first, second)

    def test_similar_text_scores_higher_than_unrelated(self):
        embedder = HashingEmbedder(dim=256).fit(CORPUS)
        matrix = embedder.encode(CORPUS)
        query = embedder.encode(["what is the monthly rent"])[0]
        scores = cosine_scores(query, matrix)
        assert int(np.argmax(scores)) == 0

    def test_empty_text_is_handled(self):
        embedder = HashingEmbedder(dim=64).fit(CORPUS)
        vector = embedder.encode([""])
        assert vector.shape == (1, 64)
        assert not np.isnan(vector).any()

    def test_rejects_tiny_dimension(self):
        with pytest.raises(ValueError):
            HashingEmbedder(dim=8)

    def test_persistence_round_trip(self, tmp_path):
        embedder = HashingEmbedder(dim=128).fit(CORPUS)
        embedder.save(tmp_path)
        restored = HashingEmbedder(dim=128)
        assert restored.load(tmp_path)
        assert np.allclose(embedder.encode(CORPUS), restored.encode(CORPUS))

    def test_load_rejects_dimension_mismatch(self, tmp_path):
        HashingEmbedder(dim=128).fit(CORPUS).save(tmp_path)
        assert not HashingEmbedder(dim=64).load(tmp_path)

    def test_factory_falls_back_when_neural_unavailable(self):
        embedder = get_embedder("sentence-transformers", model_name="does/not/exist", dim=96)
        assert embedder.dim in {96, 384}  # fell back to hashing, or a real model loaded


class TestNumpyVectorStore:
    def _store(self, tmp_path, dim=128):
        embedder = HashingEmbedder(dim=dim).fit(CORPUS)
        chunks = [make_chunk(i, text) for i, text in enumerate(CORPUS)]
        store = NumpyVectorStore(tmp_path, dim)
        store.add(chunks, embedder.encode(CORPUS))
        return store, embedder

    def test_add_and_len(self, tmp_path):
        store, _ = self._store(tmp_path)
        assert len(store) == 4

    def test_search_returns_ranked_results(self, tmp_path):
        store, embedder = self._store(tmp_path)
        results = store.search(embedder.encode(["security deposit amount"])[0], k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]
        assert "deposit" in results[0][0].text.lower()

    def test_search_on_empty_store_is_safe(self, tmp_path):
        assert NumpyVectorStore(tmp_path, 64).search(np.zeros(64, dtype=np.float32), k=3) == []

    def test_k_larger_than_corpus(self, tmp_path):
        store, embedder = self._store(tmp_path)
        assert len(store.search(embedder.encode(["rent"])[0], k=99)) == 4

    def test_readding_same_id_updates_instead_of_duplicating(self, tmp_path):
        store, embedder = self._store(tmp_path)
        store.add([make_chunk(0, "updated text")], embedder.encode(["updated text"]))
        assert len(store) == 4
        assert store.get("c0").text == "updated text"

    def test_dimension_mismatch_raises(self, tmp_path):
        store = NumpyVectorStore(tmp_path, 128)
        with pytest.raises(ValueError):
            store.add([make_chunk(0, "x")], np.zeros((1, 64), dtype=np.float32))

    def test_count_mismatch_raises(self, tmp_path):
        store = NumpyVectorStore(tmp_path, 128)
        with pytest.raises(ValueError):
            store.add([make_chunk(0, "x")], np.zeros((2, 128), dtype=np.float32))

    def test_persistence_round_trip(self, tmp_path):
        store, embedder = self._store(tmp_path)
        store.save()
        restored = NumpyVectorStore(tmp_path, 128)
        assert restored.load()
        assert len(restored) == 4
        expected = store.search(embedder.encode(["rent"])[0], k=1)[0][0].chunk_id
        assert restored.search(embedder.encode(["rent"])[0], k=1)[0][0].chunk_id == expected

    def test_load_detects_embedder_change(self, tmp_path):
        store, _ = self._store(tmp_path)
        store.save()
        assert not NumpyVectorStore(tmp_path, 64).load()

    def test_remove_document(self, tmp_path):
        store, _ = self._store(tmp_path)
        assert store.remove_document("doc") == 4
        assert len(store) == 0

    def test_clear_deletes_files(self, tmp_path):
        store, _ = self._store(tmp_path)
        store.save()
        store.clear()
        assert len(store) == 0
        assert not store.vectors_path.exists()

    def test_bboxes_survive_persistence(self, tmp_path):
        store, _ = self._store(tmp_path)
        store.save()
        restored = NumpyVectorStore(tmp_path, 128)
        restored.load()
        chunk = restored.get("c1")
        assert chunk.line_bboxes and len(chunk.line_bboxes[0]) == 4


class TestBM25:
    def test_internal_scores_exact_term_match_highest(self):
        tokens = [tokenize(text) for text in CORPUS]
        model = InternalBM25(tokens)
        scores = model.get_scores(tokenize("security deposit"))
        assert int(np.argmax(scores)) == 1

    def test_internal_unknown_term_scores_zero(self):
        model = InternalBM25([tokenize(text) for text in CORPUS])
        assert float(np.max(model.get_scores(tokenize("zzzzqqq")))) == 0.0

    def test_internal_empty_corpus(self):
        assert InternalBM25([]).get_scores(["rent"]).size == 0

    def test_store_search_and_persistence(self, tmp_path):
        chunks = [make_chunk(i, text) for i, text in enumerate(CORPUS)]
        store = BM25Store(tmp_path).build(chunks)
        results = store.search("prior notice to terminate", k=2)
        assert results
        assert "terminate" in results[0][0].text.lower()

        store.save()
        restored = BM25Store(tmp_path)
        assert restored.load()
        assert len(restored) == 4
        assert restored.search("prior notice", k=1)[0][0].chunk_id == results[0][0].chunk_id

    def test_store_handles_empty_query(self, tmp_path):
        store = BM25Store(tmp_path).build([make_chunk(0, CORPUS[0])])
        assert store.search("", k=3) == []

    def test_internal_matches_rank_bm25_ordering(self, monkeypatch, tmp_path):
        """The bundled scorer must rank identically to the reference library."""
        pytest.importorskip("rank_bm25")
        chunks = [make_chunk(i, text) for i, text in enumerate(CORPUS)]

        reference = BM25Store(tmp_path).build(chunks)
        assert reference.impl == "rank_bm25"
        reference_order = [c.chunk_id for c, _ in reference.search("monthly rent deposit", k=4)]

        monkeypatch.setenv("VERIRAG_BM25_IMPL", "internal")
        internal = BM25Store(tmp_path).build(chunks)
        assert internal.impl == "internal"
        internal_order = [c.chunk_id for c, _ in internal.search("monthly rent deposit", k=4)]

        assert internal_order == reference_order
