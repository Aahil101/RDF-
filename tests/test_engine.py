"""End-to-end tests: engine orchestration, LLM abstraction, eval harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from verirag.engine import VeriRAG
from verirag.eval import EvalCase, run_eval, validate_dataset
from verirag.eval.metrics import EvalReport, hit_at_k, ndcg_at_k, precision_at_k, reciprocal_rank
from verirag.generation.answerer import NO_EVIDENCE_MESSAGE, Answerer, compose_extractive
from verirag.generation.llm import LLMResponse, describe_provider, get_llm
from verirag.generation.prompts import REFUSAL_TOKEN, build_answer_prompt


# ---------------------------------------------------------------------------
class FakeLLM:
    """Deterministic stand-in so generation is testable without a network."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self.reply, provider=self.provider, model=self.model)


class BrokenLLM(FakeLLM):
    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text="", provider=self.provider, model=self.model, error="HTTP 429: rate limited")


# ---------------------------------------------------------------------------
class TestIngestion:
    def test_ingest_reports_pages_lines_chunks(self, settings, sample_pdf: Path):
        engine = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        report = engine.ingest(sample_pdf)[0]
        assert report.ok
        assert report.chunks > 0
        assert report.lines == 12  # 6 lines on each of 2 pages
        assert report.document.n_pages == 2

    def test_reingesting_identical_content_is_skipped(self, engine, sample_pdf: Path):
        report = engine.ingest(sample_pdf)[0]
        assert report.skipped
        assert len(engine.documents()) == 1

    def test_force_reingest_does_not_duplicate_chunks(self, engine, sample_pdf: Path):
        before = len(engine.indexer.vectors)
        engine.ingest(sample_pdf, force=True)
        assert len(engine.indexer.vectors) == before

    def test_directory_ingest_picks_up_every_pdf(self, settings, sample_pdf, second_pdf):
        engine = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        reports = engine.ingest(settings.raw_dir)
        assert len(reports) == 2
        assert all(r.ok for r in reports)

    def test_index_persists_across_instances(self, engine, settings):
        engine.indexer.save()
        reloaded = VeriRAG(settings, llm=None, probe_llm=False)
        assert not reloaded.is_empty()
        assert len(reloaded.documents()) == 1

    def test_delete_document_empties_the_index(self, engine):
        doc_id = engine.documents()[0].doc_id
        assert engine.delete_document(doc_id)
        assert engine.is_empty()

    def test_corrupt_file_is_reported_not_raised(self, settings):
        broken = settings.raw_dir / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")
        engine = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        report = engine.ingest(broken)[0]
        assert not report.ok
        assert report.error


class TestAsking:
    def test_answer_carries_citation_with_page_and_lines(self, engine):
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        citation = result.answer.used_citations[0]
        assert citation.doc_name == "lease_test.pdf"
        assert citation.page_no in {1, 2}
        assert citation.line_start >= 1
        assert citation.locator.startswith("p.")

    def test_answer_quotes_the_document_value(self, engine):
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert "48,500" in result.answer.text

    def test_out_of_domain_question_is_refused(self, engine):
        result = engine.ask("How do I train a neural network on ImageNet?", render_proof=False, persist=False)
        assert result.answer.refused
        assert result.answer.text == NO_EVIDENCE_MESSAGE
        assert result.answer.confidence_band() == "refused"

    def test_empty_question_raises(self, engine):
        with pytest.raises(ValueError):
            engine.ask("   ")

    def test_turn_is_persisted_with_evidence(self, engine):
        session_id = engine.new_session("test session")
        result = engine.ask("What is the security deposit?", render_proof=False, session_id=session_id)
        assert result.message_id > 0
        messages = engine.chat.get_messages(session_id)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].citations

    def test_history_is_used_for_follow_ups(self, engine):
        session_id = engine.new_session()
        engine.ask("What is the monthly rent?", render_proof=False, session_id=session_id)
        second = engine.ask("And the deposit?", render_proof=False, session_id=session_id)
        assert len(engine.chat.get_messages(session_id)) == 4
        assert second.answer.text

    def test_stats_expose_the_configuration(self, engine):
        stats = engine.stats()
        assert stats["documents"] == 1
        assert stats["chunks"] > 0
        assert "llm" in stats and "history" in stats


class TestGenerationWithLLM:
    def test_llm_answer_is_used_and_citations_marked(self, engine):
        engine.answerer = Answerer(FakeLLM("The monthly rent is Rs. 48,500. [S1]"), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.provider == "fake"
        assert result.answer.used_citations
        assert result.answer.used_citations[0].marker == "S1"

    def test_hallucinated_marker_is_stripped(self, engine):
        engine.answerer = Answerer(FakeLLM("Rent is Rs. 48,500. [S1] Invented fact. [S99]"), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert "S99" not in result.answer.text

    def test_model_refusal_token_is_honoured(self, engine):
        engine.answerer = Answerer(
            FakeLLM(f"{REFUSAL_TOKEN}: the sources do not mention the parking allocation"),
            engine.settings,
        )
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.refused
        assert "parking allocation" in result.answer.text

    def test_uncited_answer_gets_a_citation_attached(self, engine):
        engine.answerer = Answerer(FakeLLM("The rent is Rs. 48,500 with no marker at all"), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.used_citations

    def test_provider_failure_falls_back_to_extractive(self, engine):
        engine.answerer = Answerer(BrokenLLM(""), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.provider == "extractive-fallback"
        assert "48,500" in result.answer.text

    def test_prompt_contains_locators_for_every_source(self, engine):
        retrieval = engine.retriever.retrieve("What is the monthly rent?")
        prompt = build_answer_prompt("What is the monthly rent?", retrieval.chunks)
        for index, item in enumerate(retrieval.chunks, start=1):
            assert f"[S{index}]" in prompt
            assert f"page: {item.chunk.page_no}" in prompt
            assert f"lines: {item.chunk.line_start}-{item.chunk.line_end}" in prompt


class TestExtractiveComposer:
    def test_every_sentence_gets_a_marker(self, engine):
        retrieval = engine.retriever.retrieve("What is the monthly rent?")
        text, used = compose_extractive("What is the monthly rent?", retrieval.chunks)
        assert used
        assert text.count("[S") >= 1

    def test_headings_are_not_quoted_back(self, engine):
        retrieval = engine.retriever.retrieve("What is the monthly rent?")
        text, _ = compose_extractive("What is the monthly rent?", retrieval.chunks)
        assert "CLAUSE 1 - RENT AND DEPOSIT" not in text

    def test_variants_help_match_abbreviations(self, settings, second_pdf):
        engine = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        engine.ingest(second_pdf)
        retrieval = engine.retriever.retrieve("define BCNF")
        text, _ = compose_extractive("define BCNF", retrieval.chunks, variants=retrieval.variants)
        assert "superkey" in text.lower()


class TestProviderResolution:
    def test_extractive_returns_no_client(self, settings):
        settings.llm_provider = "extractive"
        assert get_llm(settings) is None

    def test_missing_key_yields_no_client(self, settings):
        settings.llm_provider = "groq"
        settings.groq_api_key = ""
        assert get_llm(settings) is None

    def test_probe_can_be_skipped(self, settings):
        settings.llm_provider = "groq"
        settings.groq_api_key = ""
        client = get_llm(settings, probe=False)
        assert client is not None and client.provider == "groq"

    def test_describe_provider_labels_extractive_mode(self):
        assert describe_provider(None) == "extractive:no-llm"


class TestRankingMetrics:
    def test_hit_at_k(self):
        assert hit_at_k([False, True, False], 2) == 1.0
        assert hit_at_k([False, True, False], 1) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank([False, True]) == pytest.approx(0.5)
        assert reciprocal_rank([False, False]) == 0.0

    def test_ndcg_rewards_higher_positions(self):
        assert ndcg_at_k([True, False], 2) > ndcg_at_k([False, True], 2)

    def test_ndcg_is_bounded(self):
        assert ndcg_at_k([True, True, True], 3) <= 1.0

    def test_precision_at_k(self):
        assert precision_at_k([True, False, True, False], 4) == pytest.approx(0.5)

    def test_empty_relevance(self):
        assert reciprocal_rank([]) == 0.0
        assert precision_at_k([], 3) == 0.0


class TestEvalHarness:
    def test_broken_gold_label_is_detected(self, engine):
        cases = [EvalCase("q", gold_phrase="this phrase is definitely not in the corpus")]
        assert validate_dataset(engine, cases)

    def test_valid_gold_label_passes(self, engine):
        cases = [EvalCase("What is the monthly rent?", gold_phrase="Rs. 48,500", answer_must_contain=["48,500"])]
        assert validate_dataset(engine, cases) == []

    def test_strict_mode_raises_on_broken_labels(self, engine):
        with pytest.raises(AssertionError, match="Gold phrases missing"):
            run_eval(engine, [EvalCase("q", gold_phrase="absent phrase xyzzy")], strict=True)

    def test_run_produces_metrics(self, engine):
        cases = [
            EvalCase("What is the monthly rent?", gold_phrase="Rs. 48,500", answer_must_contain=["48,500"]),
            EvalCase("What is the security deposit?", gold_phrase="Rs. 2,91,000", answer_must_contain=["2,91,000"]),
            EvalCase("Who won the 2018 World Cup?", category="out_of_corpus"),
        ]
        report = run_eval(engine, cases, strict=True)
        assert isinstance(report, EvalReport)
        assert report.retrieval_metrics()["recall@1"] == 1.0
        assert report.answer_metrics()["answer_accuracy"] == 1.0
        assert report.refusal_metrics()["false_refusal_rate"] == 0.0

    def test_report_renders_and_serialises(self, engine):
        cases = [EvalCase("What is the monthly rent?", gold_phrase="Rs. 48,500", answer_must_contain=["48,500"])]
        report = run_eval(engine, cases)
        text = report.render()
        assert "RETRIEVAL" in text and "ANSWER QUALITY" in text
        payload = report.to_dict()
        assert payload["n_questions"] == 1
        assert "threshold_calibration" in payload


class TestSampleCorpus:
    """Guards the shipped corpus and its golden labels, when present."""

    def test_bundled_dataset_labels_are_valid(self):
        from verirag.config import get_settings
        from verirag.eval import all_cases

        settings = get_settings()
        if not any(settings.raw_dir.glob("*.pdf")):
            pytest.skip("sample corpus not generated")
        engine = VeriRAG(settings, llm=None, probe_llm=False, enable_history=False)
        if engine.is_empty():
            pytest.skip("sample corpus not indexed")
        assert validate_dataset(engine, all_cases()) == []
