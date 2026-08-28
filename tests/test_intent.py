"""Intent routing tests.

These exist because of a real, embarrassing failure: a user uploaded a PDF, typed
"explain abput this document", and the system refused. Two causes — a global
request has no passage that resembles it, so similarity retrieval finds nothing;
and a single typo removed the last usable term. Both were fatal, on the very first
question a new user asks.
"""

from __future__ import annotations

import pytest

from verirag.engine import VeriRAG
from verirag.generation.answerer import compose_overview
from verirag.models import RetrievedChunk
from verirag.retrieval.intent import Intent, classify, topic_of_interest
from verirag.retrieval.query_expansion import expand_query, spelling_form


class TestClassify:
    @pytest.mark.parametrize(
        "query",
        [
            "summarise this",
            "summarize this document",
            "explain this document",
            "explain this",
            "describe the document",
            "what is this pdf about",
            "give me an overview",
            "key points",
            "main takeaways",
            "tl;dr",
            "tldr",
            "this document",
            "walk me through this",
            "brief me on this file",
            "explain the whole pdf",
        ],
    )
    def test_global_requests_route_to_summary(self, query):
        assert classify(query) is Intent.SUMMARY

    @pytest.mark.parametrize(
        "query",
        [
            "explain abput this document",   # the query that failed in production
            "sumarize this documnt",
            "summarise this docment",
            "explan this pdf",
        ],
    )
    def test_typos_still_route_to_summary(self, query):
        assert classify(query) is Intent.SUMMARY

    @pytest.mark.parametrize(
        "query",
        [
            "what topics are covered",
            "which sections are in this",
            "list the topics",
            "what can i ask",
            "what does this cover",
        ],
    )
    def test_structure_questions_route_to_topics(self, query):
        assert classify(query) is Intent.TOPICS

    @pytest.mark.parametrize(
        "query",
        [
            "what is the monthly rent",
            "explain BCNF",
            "explain bcnf in this document",
            "what is the rent in this document",
            "define superkey",
            "who is the lessor",
            "explain the write-ahead logging rule",
            "what is the carpet area of the flat",
            "how much is the security deposit",
        ],
    )
    def test_specific_questions_stay_on_the_lookup_path(self, query):
        """A query carrying its own subject is a lookup, however it is phrased."""
        assert classify(query) is Intent.LOOKUP

    def test_empty_query(self):
        assert classify("") is Intent.LOOKUP
        assert classify("   ") is Intent.LOOKUP


class TestTopicOfInterest:
    def test_scoped_summary_keeps_its_subject(self):
        assert "clause 3" in topic_of_interest("summarise clause 3").lower()

    def test_whole_document_request_has_no_subject(self):
        assert topic_of_interest("explain this document") == ""
        assert topic_of_interest("summarise this") == ""

    def test_empty_input(self):
        assert topic_of_interest("") == ""


class TestSpellingRepair:
    def test_corrects_against_corpus_vocabulary(self):
        vocab = ["security", "deposit", "refundable", "lessee"]
        assert "deposit" in spelling_form("depsit amount", vocab)

    def test_leaves_known_words_alone(self):
        vocab = ["security", "deposit"]
        assert spelling_form("security deposit", vocab) == ""

    def test_no_vocabulary_is_a_no_op(self):
        assert spelling_form("anything", None) == ""

    def test_short_words_are_not_corrected(self):
        """Three-letter words are too easy to "correct" into the wrong term."""
        assert spelling_form("abc", ["abd", "abe"]) == ""

    def test_unrecognisable_word_is_left_intact(self):
        assert spelling_form("zzzqqqwww", ["security", "deposit"]) == ""

    def test_expansion_includes_the_repaired_form(self):
        variants = expand_query("depsit amount", vocabulary=["deposit", "amount", "security"])
        assert any("deposit" in variant for variant in variants)


class TestSummaryPath:
    def test_summary_request_is_answered_not_refused(self, engine):
        result = engine.ask("explain this document", render_proof=False, persist=False)
        assert not result.answer.refused
        assert result.answer.text.strip()

    def test_typo_summary_request_is_answered(self, engine):
        """The exact production failure."""
        result = engine.ask("explain abput this document", render_proof=False, persist=False)
        assert not result.answer.refused

    def test_summary_carries_citations(self, engine):
        result = engine.ask("summarise this", render_proof=False, persist=False)
        assert result.answer.used_citations
        for citation in result.answer.used_citations:
            assert citation.page_no >= 1
            assert citation.line_start <= citation.line_end

    def test_summary_samples_across_the_document(self, engine):
        """An overview must not come from one page of a multi-page file."""
        result = engine.ask("summarise this", render_proof=False, persist=False)
        pages = {c.chunk.page_no for c in result.retrieval.chunks}
        assert len(pages) >= 2

    def test_summary_is_not_flagged_weak(self, engine):
        """A spread of the document is the right evidence, not weak evidence."""
        result = engine.ask("summarise this", render_proof=False, persist=False)
        assert not result.answer.weak_evidence

    def test_scoped_summary_uses_retrieval(self, engine):
        result = engine.ask("summarise the termination clause", render_proof=False, persist=False)
        assert not result.answer.refused

    def test_summary_persists_like_any_turn(self, engine):
        session_id = engine.new_session()
        engine.ask("summarise this", render_proof=False, session_id=session_id)
        messages = engine.chat.get_messages(session_id)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].citations


class TestTopicsPath:
    def test_lists_sections_with_page_ranges(self, engine):
        result = engine.ask("what topics are covered", render_proof=False, persist=False)
        assert not result.answer.refused
        assert "p." in result.answer.text
        assert result.answer.provider == "structure"

    def test_every_topic_is_cited(self, engine):
        result = engine.ask("what can i ask", render_proof=False, persist=False)
        assert result.answer.used_citations


class TestNeverADeadEnd:
    def test_refusal_offers_answerable_prompts(self, engine):
        result = engine.ask("who won the 2018 world cup", render_proof=False, persist=False)
        assert result.answer.refused
        assert result.suggestions, "a refusal must not trap the user"

    def test_suggestions_reference_real_documents(self, engine):
        names = {d.name for d in engine.documents()}
        joined = " ".join(engine.suggestions())
        assert any(name in joined for name in names)

    def test_suggestions_are_capped(self, engine):
        assert len(engine.suggestions(limit=3)) <= 3

    def test_empty_index_explains_what_to_do(self, settings):
        engine = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        result = engine.ask("summarise this", render_proof=False, persist=False)
        assert result.answer.refused
        assert "upload" in result.answer.text.lower()


class TestOverviewComposer:
    def test_works_without_an_llm(self, engine):
        chunks = engine.indexer.vectors.all_chunks()[:4]
        items = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]
        text, used = compose_overview(items, "test.pdf")
        assert used == [1, 2, 3, 4]
        assert text.count("[S") == 4
        assert "test.pdf" in text

    def test_marks_itself_as_extractive(self, engine):
        items = [RetrievedChunk(chunk=engine.indexer.vectors.all_chunks()[0], score=1.0)]
        text, _ = compose_overview(items, "test.pdf")
        assert "excerpt" in text.lower()
