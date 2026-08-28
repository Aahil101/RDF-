"""Text normalisation tests, and the observability of provider fallbacks.

Both cover failures that are dangerous because they are *invisible*: a correct
answer scored as wrong because of a typographic hyphen, and an LLM that was
never actually called.
"""

from __future__ import annotations

import pytest

from verirag.generation.answerer import Answerer
from verirag.generation.llm import LLMResponse
from verirag.textnorm import contains_phrase, fold_punctuation, normalise_for_compare
from verirag.verify.grounding import support_score


class TestFoldPunctuation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("P\u201142", "P-42"),          # non-breaking hyphen
            ("lock\u2010in", "lock-in"),    # hyphen
            ("2024\u20132027", "2024-2027"),  # en dash
            ("cost\u2014plus", "cost-plus"),  # em dash
            ("months\u2019", "months'"),    # right single quote
            ("\u2018quoted\u2019", "'quoted'"),
            ("\u201cquoted\u201d", '"quoted"'),
            ("a\u00a0b", "a b"),            # non-breaking space
            ("soft\u00adhyphen", "softhyphen"),
            ("zero\u200bwidth", "zerowidth"),
        ],
    )
    def test_folds_typographic_variants(self, raw, expected):
        assert fold_punctuation(raw) == expected

    def test_ascii_text_is_unchanged(self):
        assert fold_punctuation("Rs. 48,500 payable in advance.") == "Rs. 48,500 payable in advance."

    def test_empty_input(self):
        assert fold_punctuation("") == ""

    def test_normalise_lowercases_and_collapses(self):
        assert normalise_for_compare("  The   MONTHLY\trent  ") == "the monthly rent"


class TestContainsPhrase:
    def test_matches_across_hyphen_styles(self):
        """Regression: an LLM writing "P‑42" must not be scored as wrong."""
        assert contains_phrase("The spaces are P\u201142 and P\u201143.", "P-42")

    def test_matches_across_apostrophe_styles(self):
        assert contains_phrase("three (3) months\u2019 prior notice", "months' prior notice")

    def test_case_and_whitespace_insensitive(self):
        assert contains_phrase("RS.  48,500   PAYABLE", "Rs. 48,500 payable")

    def test_rejects_absent_phrase(self):
        assert not contains_phrase("The rent is Rs. 48,500.", "Rs. 99,999")

    def test_empty_needle_is_false(self):
        assert not contains_phrase("anything", "")


class TestGroundingIsPunctuationInsensitive:
    def test_typographic_quote_does_not_reduce_support(self):
        evidence = "either party may terminate this lease by giving three (3) months' prior notice"
        ascii_claim = "Either party may terminate by giving three (3) months' prior notice."
        fancy_claim = "Either party may terminate by giving three (3) months\u2019 prior notice."
        assert support_score(fancy_claim, evidence) == pytest.approx(
            support_score(ascii_claim, evidence), abs=1e-6
        )

    def test_non_breaking_hyphen_does_not_reduce_support(self):
        evidence = "the lock-in period expires on 31 March 2025"
        assert support_score("The lock\u2011in period expires on 31 March 2025.", evidence) > 0.85


class TestGeneratedTextIsDisplayable:
    """Regression: models emit Unicode spaces that vanish in legacy consoles."""

    def test_narrow_no_break_space_becomes_a_real_space(self):
        from verirag.generation.citations import sanitize_answer

        raw = "paid within\u202fninety\u202fdays of the judgment. [S1]"
        text, used = sanitize_answer(raw, n_sources=2)
        assert "within ninety days" in text
        assert "\u202f" not in text
        assert used == [1]

    def test_non_breaking_hyphen_in_generated_text_is_folded(self):
        from verirag.generation.citations import sanitize_answer

        text, _ = sanitize_answer("the lock\u2011in period ends [S1]", n_sources=1)
        assert "lock-in" in text

    def test_markers_still_parse_after_folding(self):
        from verirag.generation.citations import sanitize_answer

        text, used = sanitize_answer("Fact one\u202f[S1] and fact two [S2].", n_sources=2)
        assert used == [1, 2]
        assert "[S1]" in text and "[S2]" in text

    def test_source_quotes_are_not_altered(self, engine):
        """Citation quotes must stay faithful to the PDF, unlike model prose."""
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        quote = result.answer.used_citations[0].quote
        assert quote  # taken from the line store verbatim, not folded


class TestProviderErrorVisibility:
    class FailingLLM:
        provider = "groq"
        model = "does-not-exist"

        def available(self) -> bool:
            return True

        def complete(self, system: str, user: str) -> LLMResponse:  # noqa: ARG002
            return LLMResponse(
                text="",
                provider=self.provider,
                model=self.model,
                error='HTTP 404: {"error":{"message":"The model does not exist"}}',
            )

    def test_fallback_records_why_the_llm_was_not_used(self, engine):
        engine.answerer = Answerer(self.FailingLLM(), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.provider == "extractive-fallback"
        assert "404" in result.answer.provider_error
        assert "does not exist" in result.answer.provider_error

    def test_successful_call_records_no_error(self, engine):
        class WorkingLLM(TestProviderErrorVisibility.FailingLLM):
            def complete(self, system: str, user: str) -> LLMResponse:  # noqa: ARG002
                return LLMResponse(text="The rent is Rs. 48,500. [S1]", provider="groq", model="ok")

        engine.answerer = Answerer(WorkingLLM(), engine.settings)
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.provider_error == ""

    def test_error_is_serialised(self, engine):
        engine.answerer = Answerer(self.FailingLLM(), engine.settings)
        payload = engine.ask("What is the monthly rent?", render_proof=False, persist=False).answer.to_dict()
        assert payload["provider_error"]


class TestConfidenceBand:
    def test_verified_support_upgrades_weak_retrieval_to_medium(self):
        from verirag.models import Answer

        answer = Answer(question="q", text="a", groundedness=0.85, weak_evidence=True)
        assert answer.confidence_band() == "medium"

    def test_strong_retrieval_and_support_is_high(self):
        from verirag.models import Answer

        answer = Answer(question="q", text="a", groundedness=0.85, weak_evidence=False)
        assert answer.confidence_band() == "high"

    def test_weak_support_and_weak_retrieval_is_low(self):
        from verirag.models import Answer

        answer = Answer(question="q", text="a", groundedness=0.55, weak_evidence=True)
        assert answer.confidence_band() == "low"

    def test_refusal_dominates(self):
        from verirag.models import Answer

        answer = Answer(question="q", text="a", groundedness=0.99, refused=True)
        assert answer.confidence_band() == "refused"
