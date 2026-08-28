"""Study-mode tests.

The critical property is that an answer key is never shown unless it can be
traced back to the PDF. A quiz with an unverifiable answer teaches a student the
wrong thing and gives them no way to notice, so the rejection path is tested as
carefully as the happy path.
"""

from __future__ import annotations

import json

import pytest

from verirag.generation.llm import LLMResponse
from verirag.study import MCQ, StudyGenerator, extract_key_terms, extract_topics
from verirag.study.generator import _find_salient, _parse_json_array


# ---------------------------------------------------------------------------
class ScriptedLLM:
    """Returns a fixed reply, so generation is testable without a network."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> LLMResponse:  # noqa: ARG002
        self.calls += 1
        return LLMResponse(text=self.reply, provider=self.provider, model=self.model)


def mcq_reply(question: str, options: list[str], correct: int, evidence: str) -> str:
    return json.dumps(
        [
            {
                "question": question,
                "options": options,
                "correct_index": correct,
                "evidence": evidence,
                "explanation": "because the passage says so",
                "difficulty": "medium",
            }
        ]
    )


# ---------------------------------------------------------------------------
class TestJsonParsing:
    def test_plain_array(self):
        assert _parse_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_tolerates_code_fences(self):
        assert _parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_tolerates_surrounding_prose(self):
        assert _parse_json_array('Here you go:\n[{"a": 1}]\nHope that helps!') == [{"a": 1}]

    def test_wraps_a_single_object(self):
        assert _parse_json_array('{"question": "q"}') == [{"question": "q"}]

    def test_unwraps_a_questions_key(self):
        assert _parse_json_array('{"questions": [{"a": 1}]}') == [{"a": 1}]

    def test_invalid_json_returns_empty(self):
        assert _parse_json_array("not json at all") == []

    def test_empty_input(self):
        assert _parse_json_array("") == []

    def test_drops_non_dict_elements(self):
        assert _parse_json_array('[{"a": 1}, "junk", 5]') == [{"a": 1}]


class TestSalientFactDetection:
    @pytest.mark.parametrize(
        "sentence,kind",
        [
            ("The rent shall be Rs. 48,500 per month.", "money"),
            ("The rent escalates by six per cent every year.", "percent"),
            ("The lease commences on 1 May 2024.", "date"),
            ("The lock-in period is eleven (11) months.", "duration"),
        ],
    )
    def test_detects_each_kind(self, sentence, kind):
        found = _find_salient(sentence)
        assert found is not None
        assert found[1] == kind

    def test_returns_none_without_a_fact(self):
        assert _find_salient("The parties agree to act in good faith.") is None


class TestTopicExtraction:
    def test_recovers_sections_as_topics(self, engine):
        chunks = engine.indexer.vectors.all_chunks()
        topics = extract_topics(chunks)
        assert topics
        names = " ".join(t.name.lower() for t in topics)
        assert "rent" in names or "termination" in names

    def test_topics_carry_a_real_page_range(self, engine):
        for topic in extract_topics(engine.indexer.vectors.all_chunks()):
            assert topic.page_start >= 1
            assert topic.page_end >= topic.page_start
            assert topic.page_range.startswith("p.")
            assert topic.chunk_ids

    def test_key_terms_exclude_stopwords_and_generic_words(self):
        terms = extract_key_terms(
            "The section states that the monthly rent shall be payable and the deposit is refundable."
        )
        assert "the" not in terms and "section" not in terms
        assert any(t in {"monthly", "payable", "deposit", "refundable"} for t in terms)

    def test_empty_input(self):
        assert extract_topics([]) == []
        assert extract_key_terms("") == []

    def test_table_cells_are_not_treated_as_headings(self, engine):
        """A cell like "5NF" must not become a topic name."""
        names = [t.name.strip().lower() for t in extract_topics(engine.indexer.vectors.all_chunks())]
        assert all(len(name) > 4 for name in names), names


class TestMCQStructure:
    def _mcq(self, **overrides) -> MCQ:
        payload = {
            "question": "What is the monthly rent?",
            "options": ["Rs. 48,500", "Rs. 62,000", "Rs. 22,000", "Rs. 29,100"],
            "correct_index": 0,
        }
        payload.update(overrides)
        return MCQ(**payload)

    def test_wellformed_accepts_a_good_question(self):
        assert self._mcq().is_wellformed()

    def test_rejects_duplicate_options(self):
        assert not self._mcq(options=["a", "a", "b", "c"]).is_wellformed()

    def test_rejects_too_few_options(self):
        assert not self._mcq(options=["a", "b"]).is_wellformed()

    def test_rejects_out_of_range_answer_index(self):
        assert not self._mcq(correct_index=9).is_wellformed()

    def test_rejects_blank_question(self):
        assert not self._mcq(question="   ").is_wellformed()

    def test_correct_letter_and_option(self):
        question = self._mcq(correct_index=2)
        assert question.correct_letter == "C"
        assert question.correct_option == "Rs. 22,000"


class TestVerificationGate:
    def test_supported_answer_is_kept_and_cited(self, engine):
        reply = mcq_reply(
            "What is the monthly rent for the demised premises?",
            ["Rs. 48,500", "Rs. 62,000", "Rs. 22,000", "Rs. 29,100"],
            0,
            "The monthly rent for the demised premises shall be Rs. 48,500",
        )
        generator = StudyGenerator(ScriptedLLM(reply), engine.indexer.lines, engine.settings, seed=1)
        questions, rejected = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=1)
        assert questions, f"nothing generated (rejected={rejected})"
        question = questions[0]
        assert question.verified
        assert question.verification_score >= engine.settings.grounding_threshold
        assert question.citation is not None
        assert question.citation.page_no >= 1
        assert question.citation.line_start <= question.citation.line_end

    def test_fabricated_answer_is_rejected(self, engine):
        """The correct option names a figure that appears nowhere in the PDF."""
        reply = mcq_reply(
            "What is the monthly rent for the demised premises?",
            ["Rs. 9,99,999 payable to the municipal corporation of Atlantis", "b", "c", "d"],
            0,
            "invented evidence",
        )
        generator = StudyGenerator(ScriptedLLM(reply), engine.indexer.lines, engine.settings, seed=1)
        questions, rejected = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=1)
        assert questions == []
        assert rejected >= 1

    def test_malformed_question_is_rejected(self, engine):
        reply = mcq_reply("dup options", ["same", "same", "same", "same"], 0, "whatever")
        generator = StudyGenerator(ScriptedLLM(reply), engine.indexer.lines, engine.settings, seed=1)
        questions, rejected = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=1)
        assert questions == []
        assert rejected >= 1

    def test_provider_failure_falls_back_to_cloze(self, engine):
        class DeadLLM(ScriptedLLM):
            def complete(self, system: str, user: str) -> LLMResponse:  # noqa: ARG002
                return LLMResponse(text="", provider="x", model="y", error="boom")

        generator = StudyGenerator(DeadLLM(""), engine.indexer.lines, engine.settings, seed=3)
        questions, _rejected = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=2)
        assert questions
        assert all(q.generator == "cloze" for q in questions)


class TestClozeGeneration:
    def test_works_without_any_llm(self, engine):
        generator = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=11)
        questions, _rejected = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=3)
        assert questions
        for question in questions:
            assert question.is_wellformed()
            assert question.verified
            assert "______" in question.question
            assert question.citation is not None

    def test_distractors_are_distinct_from_the_answer(self, engine):
        generator = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=5)
        questions, _ = generator.generate_mcqs(engine.indexer.vectors.all_chunks(), count=4)
        for question in questions:
            lowered = [o.strip().lower() for o in question.options]
            assert len(set(lowered)) == len(lowered)

    def test_generation_is_reproducible_with_a_seed(self, engine):
        chunks = engine.indexer.vectors.all_chunks()
        first, _ = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=42).generate_mcqs(chunks, 3)
        second, _ = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=42).generate_mcqs(chunks, 3)
        assert [q.question for q in first] == [q.question for q in second]
        assert [q.options for q in first] == [q.options for q in second]


class TestStudyPack:
    def test_pack_reports_stats(self, engine):
        generator = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=2)
        pack = generator.build_pack(engine.indexer.vectors.all_chunks(), n_mcqs=3, n_short=1, n_cards=2)
        stats = pack.stats()
        assert stats["generator"] == "cloze"
        assert stats["topics"] >= 1
        assert stats["mcqs"] == stats["mcqs_verified"]

    def test_pack_serialises(self, engine):
        generator = StudyGenerator(None, engine.indexer.lines, engine.settings, seed=2)
        pack = generator.build_pack(engine.indexer.vectors.all_chunks(), n_mcqs=2, n_short=1, n_cards=1)
        payload = pack.to_dict()
        assert payload["doc_name"]
        assert json.dumps(payload)  # must be JSON-serialisable end to end

    def test_empty_chunks_raise(self, engine):
        generator = StudyGenerator(None, engine.indexer.lines, engine.settings)
        with pytest.raises(ValueError):
            generator.build_pack([])


class TestEngineStudyApi:
    def test_topics_via_engine(self, engine):
        doc_id = engine.documents()[0].doc_id
        assert engine.topics(doc_id)

    def test_study_pack_via_engine(self, engine):
        doc_id = engine.documents()[0].doc_id
        pack = engine.study_pack(doc_id, n_mcqs=2, n_short=1, n_cards=1, seed=9)
        assert pack.doc_id == doc_id
        assert pack.mcqs

    def test_unknown_document_raises(self, engine):
        with pytest.raises(KeyError):
            engine.study_pack("not-a-real-doc-id")

    def test_explain_returns_a_cited_answer(self, engine):
        result = engine.explain("security deposit", level="beginner")
        assert result.answer.text
        if not result.answer.refused:
            assert result.answer.citations


class TestThresholdAutoCalibration:
    def test_reranker_supplies_its_own_thresholds(self, engine):
        reranker = engine.retriever.reranker
        assert hasattr(reranker, "refusal_threshold")
        assert hasattr(reranker, "low_confidence_threshold")
        assert engine.settings.min_retrieval_score == reranker.refusal_threshold

    def test_explicit_env_override_wins(self, settings, sample_pdf, monkeypatch):
        from verirag.engine import VeriRAG

        monkeypatch.setenv("VERIRAG_MIN_RETRIEVAL_SCORE", "0.77")
        instance = VeriRAG(settings, llm=None, autoload=False, probe_llm=False)
        instance.settings.min_retrieval_score = 0.77
        instance._calibrate_thresholds()
        assert instance.settings.min_retrieval_score == 0.77
