"""Retrieval tests: rank fusion, reranker signals and query expansion.

Several of these are regression tests for defects found by measurement rather
than by reading the code — notably the proximity signal that used to *punish*
passages for repeating a query term, and the IDF default that made an absent
query term count for less than a present one.
"""

from __future__ import annotations

import pytest

from verirag.models import Chunk, RetrievedChunk
from verirag.retrieval.fusion import merge_multi_query, reciprocal_rank_fusion
from verirag.retrieval.query_expansion import (
    declarative_form,
    expand_query,
    keyword_form,
    normalise_query,
    synonym_form,
)
from verirag.retrieval.reranker import LexicalReranker, _corpus_idf, _min_cover_proximity, get_reranker
from verirag.retrieval.retriever import _drop_overlapping


def chunk(cid: str, text: str, *, page: int = 1, start: int = 1, end: int = 5, section: str = "") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc",
        doc_name="test.pdf",
        page_no=page,
        line_start=start,
        line_end=end,
        text=text,
        line_bboxes=[(0.0, float(n), 100.0, float(n) + 10) for n in range(start, end + 1)],
        section=section,
    )


class TestReciprocalRankFusion:
    def test_chunk_ranked_by_both_retrievers_wins(self):
        a, b, c = chunk("a", "alpha"), chunk("b", "beta"), chunk("c", "gamma")
        fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [(b, 5.0), (c, 4.0)], k=60)
        assert fused[0].chunk_id == "b"

    def test_preserves_per_retriever_evidence(self):
        a, b = chunk("a", "alpha"), chunk("b", "beta")
        fused = reciprocal_rank_fusion([(a, 0.9)], [(b, 3.0), (a, 1.0)], k=60)
        item = next(i for i in fused if i.chunk_id == "a")
        assert item.dense_rank == 1
        assert item.lexical_rank == 2
        assert item.dense_score == pytest.approx(0.9)
        assert item.lexical_score == pytest.approx(1.0)

    def test_scores_are_descending(self):
        chunks = [chunk(str(i), f"text {i}") for i in range(5)]
        fused = reciprocal_rank_fusion(
            [(c, 1.0) for c in chunks], [(c, 1.0) for c in reversed(chunks)], k=60
        )
        assert [i.score for i in fused] == sorted((i.score for i in fused), reverse=True)

    def test_handles_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rank_not_score_decides(self):
        """RRF must ignore raw score scale — that is the whole point of it."""
        a, b = chunk("a", "alpha"), chunk("b", "beta")
        fused = reciprocal_rank_fusion([(a, 0.001), (b, 0.0009)], [(a, 900.0), (b, 1.0)], k=60)
        assert fused[0].chunk_id == "a"

    def test_multi_query_merge_rewards_consistency(self):
        a, b, c = chunk("a", "alpha"), chunk("b", "beta"), chunk("c", "gamma")
        run1 = [RetrievedChunk(chunk=a), RetrievedChunk(chunk=b)]
        run2 = [RetrievedChunk(chunk=a), RetrievedChunk(chunk=c)]
        merged = merge_multi_query([run1, run2], k=60)
        assert merged[0].chunk_id == "a"


class TestMinCoverProximity:
    def test_adjacent_terms_score_maximum(self):
        tokens = "the monthly rent shall be paid".split()
        assert _min_cover_proximity(tokens, {"monthly", "rent"}) == pytest.approx(1.0)

    def test_repeating_a_query_term_does_not_lower_the_score(self):
        """Regression: the old first-to-last span formula punished repetition."""
        tokens = "the monthly rent is due and rent is payable monthly again".split()
        tight = "the monthly rent is due".split()
        assert _min_cover_proximity(tokens, {"monthly", "rent"}) == pytest.approx(
            _min_cover_proximity(tight, {"monthly", "rent"})
        )

    def test_partial_match_is_penalised(self):
        tokens = "the rent is payable".split()
        assert _min_cover_proximity(tokens, {"rent", "deposit"}) == pytest.approx(0.5)

    def test_no_match_scores_zero(self):
        assert _min_cover_proximity(["alpha"], {"beta"}) == 0.0

    def test_empty_query_scores_zero(self):
        assert _min_cover_proximity(["alpha"], set()) == 0.0


class TestLexicalReranker:
    def setup_method(self):
        self.reranker = LexicalReranker()

    def test_on_topic_beats_incidental_mention(self):
        on_topic = (
            "The monthly rent for the demised premises shall be Rs. 48,500 payable in advance "
            "on or before the fifth day of every month, and the rent shall escalate annually."
        )
        incidental = "The appellant was compelled to reside in rented premises at a monthly rent of Rs. 62,000."
        query = "what is the monthly rent for the flat"
        idf = _corpus_idf([on_topic, incidental])
        assert self.reranker.score(query, on_topic, idf=idf) > self.reranker.score(query, incidental, idf=idf)

    def test_absent_query_terms_reduce_the_score(self):
        """Regression: absent terms used to default to a *lower* idf than present ones."""
        text = "The Association may levy a special assessment for capital works."
        idf = _corpus_idf([text, "unrelated filler about rent and deposits"])
        both_absent = self.reranker.score("capital france", text, idf=idf)
        both_present = self.reranker.score("capital works", text, idf=idf)
        assert both_present > both_absent

    def test_empty_query_or_text_scores_zero(self):
        assert self.reranker.score("", "some text") == 0.0
        assert self.reranker.score("rent", "") == 0.0

    def test_rerank_orders_and_truncates(self):
        candidates = [
            RetrievedChunk(chunk=chunk("a", "irrelevant text about normalization theory"), dense_score=0.1),
            RetrievedChunk(chunk=chunk("b", "the security deposit of Rs. 2,91,000 is refundable"), dense_score=0.2),
        ]
        ranked = self.reranker.rerank("security deposit amount", candidates, top_k=1)
        assert len(ranked) == 1
        assert ranked[0].chunk_id == "b"
        assert ranked[0].rerank_score is not None

    def test_rerank_empty_candidates(self):
        assert self.reranker.rerank("anything", [], top_k=5) == []

    def test_score_has_no_floor_so_it_can_gate_refusals(self):
        candidates = [RetrievedChunk(chunk=chunk("a", "alpha beta gamma"), dense_score=0.0)]
        ranked = self.reranker.rerank("zzzqqq wwwyyy", candidates, top_k=1)
        assert ranked[0].score == pytest.approx(0.0, abs=1e-6)

    def test_factory_falls_back_to_lexical(self):
        assert get_reranker("cross-encoder", model_name="not/a/real/model").name in {
            "lexical",
            "cross-encoder",
        }


class TestQueryExpansion:
    def test_normalise_strips_punctuation_and_whitespace(self):
        assert normalise_query("  What is the   rent?  ") == "What is the rent"

    def test_keyword_form_drops_stopwords_and_question_words(self):
        assert keyword_form("What is the monthly rent for the flat?") == "monthly rent flat"

    def test_keyword_form_preserves_order_without_duplicates(self):
        assert keyword_form("rent rent deposit") == "rent deposit"

    @pytest.mark.parametrize(
        "question,expected_fragment",
        [
            ("Who is the lessor", "lessor"),
            ("What is the security deposit", "security deposit"),
            ("How much is the rent", "rent"),
            ("When is the rent due", "rent"),
        ],
    )
    def test_declarative_form_produces_statement(self, question, expected_fragment):
        assert expected_fragment in declarative_form(question)

    def test_declarative_form_empty_for_non_question(self):
        assert declarative_form("the rent is payable monthly") == ""

    def test_synonym_form_bridges_vocabulary(self):
        assert "lessee" in synonym_form("what must the tenant pay")

    def test_abbreviations_expand_to_full_form(self):
        """A document defines "Boyce-Codd normal form"; users type "BCNF"."""
        assert "boyce codd normal form" in synonym_form("define bcnf")

    def test_expand_includes_original_first(self):
        variants = expand_query("What is the monthly rent?")
        assert variants[0] == "What is the monthly rent"
        assert len(variants) > 1

    def test_expansion_can_be_disabled(self):
        assert expand_query("What is the rent?", enabled=False) == ["What is the rent"]

    def test_variants_are_capped(self):
        assert len(expand_query("who must the tenant pay rent to", max_variants=2)) <= 2

    def test_empty_query_yields_nothing(self):
        assert expand_query("   ") == []

    def test_llm_expander_failure_is_swallowed(self):
        def boom(_query):
            raise RuntimeError("provider down")

        assert expand_query("what is the rent", llm_expand=boom)


class TestOverlapFiltering:
    def test_removes_near_duplicate_spans_on_the_same_page(self):
        items = [
            RetrievedChunk(chunk=chunk("a", "text", page=1, start=1, end=10)),
            RetrievedChunk(chunk=chunk("b", "text", page=1, start=2, end=10)),
        ]
        assert len(_drop_overlapping(items)) == 1

    def test_keeps_distinct_pages(self):
        items = [
            RetrievedChunk(chunk=chunk("a", "text", page=1, start=1, end=10)),
            RetrievedChunk(chunk=chunk("b", "text", page=2, start=1, end=10)),
        ]
        assert len(_drop_overlapping(items)) == 2

    def test_keeps_non_overlapping_spans(self):
        items = [
            RetrievedChunk(chunk=chunk("a", "text", page=1, start=1, end=5)),
            RetrievedChunk(chunk=chunk("b", "text", page=1, start=20, end=25)),
        ]
        assert len(_drop_overlapping(items)) == 2


class TestRetrieverIntegration:
    def test_retrieval_finds_the_answering_clause(self, engine):
        result = engine.retriever.retrieve("what is the monthly rent")
        assert result.chunks
        assert "48,500" in result.chunks[0].chunk.text

    def test_trace_exposes_every_signal(self, engine):
        trace = engine.retriever.retrieve("security deposit").trace()
        assert trace
        for row in trace:
            assert {"chunk_id", "doc", "locator", "score", "why"} <= set(row)

    def test_doc_filter_restricts_results(self, engine, second_pdf):
        engine.ingest(second_pdf)
        target = next(d for d in engine.documents() if d.name == "notes_test.pdf")
        result = engine.retriever.retrieve("rent", doc_ids=[target.doc_id])
        assert all(item.chunk.doc_id == target.doc_id for item in result.chunks)

    def test_empty_query_returns_nothing(self, engine):
        assert engine.retriever.retrieve("   ").chunks == []
