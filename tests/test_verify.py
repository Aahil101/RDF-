"""Verification tests: citation parsing, sentence segmentation, groundedness.

The marker-attachment test below is the most important regression guard in the
project. Models (and the extractive composer) write ``"... Rs. 48,500. [S2]"``
with the marker *after* the full stop. Naive splitting attributed that marker to
the next sentence, so the verifier compared every claim against the wrong
source and reported a groundedness of ~0.31 for text copied verbatim out of the
PDF. If this test ever fails, the proof layer is lying.
"""

from __future__ import annotations

import pytest

from verirag.generation.citations import (
    attach_trailing_markers,
    parse_markers,
    sanitize_answer,
    sentences_with_markers,
    split_sentences,
    strip_markers,
)
from verirag.models import Answer, Citation, PdfLine
from verirag.verify.grounding import GroundingVerifier, best_window, support_score


class TestMarkerParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The rent is Rs. 48,500 [S2].", [2]),
            ("Both parties must sign [S1][S4].", [1, 4]),
            ("Combined form [S1, S3].", [1, 3]),
            ("Semicolons too [S2; S5].", [2, 5]),
            ("Bare numbers [3].", [3]),
            ("Spaced out [ S 7 ].", [7]),
            ("No citation here.", []),
        ],
    )
    def test_parses_every_marker_style(self, text, expected):
        assert parse_markers(text) == expected

    def test_first_use_order_without_duplicates(self):
        assert parse_markers("[S3] then [S1] then [S3] again") == [3, 1]

    def test_strip_markers_leaves_clean_prose(self):
        assert strip_markers("The rent is Rs. 48,500 [S2].") == "The rent is Rs. 48,500 ."


class TestSanitizeAnswer:
    def test_drops_markers_beyond_the_supplied_sources(self):
        text, used = sanitize_answer("Grounded [S2]. Invented [S9].", n_sources=3)
        assert used == [2]
        assert "S9" not in text

    def test_keeps_valid_markers(self):
        text, used = sanitize_answer("A [S1] and B [S3].", n_sources=3)
        assert used == [1, 3]
        assert "[S1]" in text and "[S3]" in text

    def test_normalises_mixed_syntax(self):
        text, _ = sanitize_answer("Combined [S1, S2].", n_sources=2)
        assert "[S1][S2]" in text

    def test_handles_empty_text(self):
        assert sanitize_answer("", 3) == ("", [])


class TestSentenceSplitting:
    def test_does_not_split_on_legal_abbreviations(self):
        text = "The rent is Rs. 48,500 per month. Clause No. 4 applies."
        assert len(split_sentences(text)) == 2

    def test_does_not_split_on_initials(self):
        assert len(split_sentences("Mr. R. Sharma filed the appeal in 2021.")) == 1

    def test_does_not_split_on_case_citations(self):
        assert len(split_sentences("See Sharma v. Metro Realty for the holding.")) == 1

    def test_splits_multiple_sentences(self):
        assert len(split_sentences("First point here. Second point here. Third one.")) == 3

    def test_bullets_become_separate_claims(self):
        assert len(split_sentences("Items: \u2022 first item here \u2022 second item here")) >= 2

    def test_empty_input(self):
        assert split_sentences("") == []


class TestMarkerAttachment:
    def test_trailing_marker_binds_to_its_own_sentence(self):
        """Regression: markers after the full stop must not migrate forward."""
        text = "The rent is Rs. 48,500. [S1] The deposit is Rs. 2,91,000. [S2]"
        pairs = sentences_with_markers(text)
        assert len(pairs) == 2
        assert "48,500" in pairs[0][0] and pairs[0][1] == [1]
        assert "2,91,000" in pairs[1][0] and pairs[1][1] == [2]

    def test_marker_before_the_period_still_works(self):
        pairs = sentences_with_markers("The rent is Rs. 48,500 [S1]. The term is 33 months [S2].")
        assert pairs[0][1] == [1]
        assert pairs[1][1] == [2]

    def test_normalisation_moves_the_marker(self):
        assert "[S1]." in attach_trailing_markers("Fact stated. [S1]")

    def test_multiple_markers_stay_together(self):
        pairs = sentences_with_markers("Both apply. [S1][S2] Next fact. [S3]")
        assert pairs[0][1] == [1, 2]
        assert pairs[1][1] == [3]

    def test_text_without_markers_is_untouched(self):
        assert attach_trailing_markers("Plain sentence.") == "Plain sentence."


class TestSupportScore:
    def test_verbatim_quote_scores_near_one(self):
        text = "The monthly rent shall be Rs. 48,500 payable in advance."
        assert support_score(text, text) > 0.95

    def test_unrelated_text_scores_low(self):
        assert support_score("The rent is Rs. 48,500.", "Deadlock is detected with a wait-for graph.") < 0.3

    def test_paraphrase_scores_moderately(self):
        score = support_score(
            "The tenant pays 48,500 rupees each month.",
            "The monthly rent for the demised premises shall be Rs. 48,500 payable in advance.",
        )
        assert 0.3 < score < 1.0

    def test_wrong_number_is_penalised_hard(self):
        evidence = "The monthly rent shall be Rs. 48,500 payable in advance."
        correct = support_score("The monthly rent shall be Rs. 48,500.", evidence)
        wrong = support_score("The monthly rent shall be Rs. 99,999.", evidence)
        assert wrong < correct

    def test_thousand_separators_normalise(self):
        assert support_score("amount 100000 due", "the amount of 1,00,000 is due") > 0.5

    def test_empty_inputs_score_zero(self):
        assert support_score("", "anything") == 0.0
        assert support_score("anything", "") == 0.0


class TestBestWindow:
    @staticmethod
    def _lines(texts: list[str]) -> list[PdfLine]:
        return [
            PdfLine(page_no=1, line_no=i, text=t, bbox=(0.0, float(i * 12), 400.0, float(i * 12 + 10)))
            for i, t in enumerate(texts, start=1)
        ]

    def test_finds_the_single_supporting_line(self):
        lines = self._lines(
            [
                "CLAUSE 2 - RENT",
                "The monthly rent shall be Rs. 48,500 payable in advance.",
                "The Lessee shall bear electricity charges.",
            ]
        )
        score, window = best_window("The monthly rent shall be Rs. 48,500.", lines)
        assert score > 0.7
        assert [ln.line_no for ln in window] == [2]

    def test_expands_beyond_four_lines_when_the_claim_is_long(self):
        """Regression: a fixed 4-line window under-reported long claims."""
        sentence_parts = [
            "The Respondent shall refund to the Appellant the sum of",
            "Rs. 1,28,25,000 together with simple interest at the rate of",
            "9.5 per cent per annum from 1 January 2020 until the date of",
            "actual payment, and shall additionally pay compensation of",
            "Rs. 5,00,000 for mental agony and harassment suffered.",
        ]
        lines = self._lines(sentence_parts)
        claim = " ".join(sentence_parts)
        score, window = best_window(claim, lines)
        assert len(window) == 5
        assert score > 0.9

    def test_returns_tightest_span_not_the_whole_chunk(self):
        lines = self._lines(
            ["Unrelated preamble text here."] * 4
            + ["The security deposit is Rs. 2,91,000 and is interest free."]
            + ["More unrelated trailing text."] * 4
        )
        _score, window = best_window("The security deposit is Rs. 2,91,000.", lines)
        assert len(window) <= 3
        assert any("2,91,000" in ln.text for ln in window)

    def test_empty_lines(self):
        assert best_window("anything", []) == (0.0, [])


class TestGroundingVerifier:
    def test_verbatim_extractive_answer_verifies_high(self, engine):
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        assert result.answer.groundedness > 0.9
        assert result.grounding.total > 0
        assert result.grounding.supported == result.grounding.total

    def test_fabricated_claim_is_flagged(self, engine):
        """An invented figure attached to a real source must not verify."""
        retrieval = engine.retriever.retrieve("What is the monthly rent?")
        answer = engine.answerer.answer(retrieval)
        answer.text = "The monthly rent is Rs. 9,99,999 and the landlord lives in Antarctica [S1]."
        answer.verdicts = []
        report = engine.verifier.verify(answer)
        assert report.total >= 1
        assert report.groundedness < 0.6
        assert answer.unsupported_claims

    def test_citation_narrows_to_the_proven_span(self, engine):
        result = engine.ask("What is the monthly rent?", render_proof=False, persist=False)
        citation = result.answer.used_citations[0]
        source = next(
            c for c in engine.indexer.vectors.all_chunks() if c.chunk_id and c.page_no == citation.page_no
        )
        assert citation.line_start >= source.line_start or citation.line_end <= source.line_end
        assert citation.line_start <= citation.line_end

    def test_refused_answer_has_no_verdicts(self, engine):
        answer = Answer(question="q", text="no evidence", refused=True)
        report = engine.verifier.verify(answer)
        assert report.total == 0
        assert answer.groundedness == 0.0

    def test_verifier_falls_back_to_quote_without_a_line_store(self, settings):
        from verirag.index.indexer import LineStore

        verifier = GroundingVerifier(LineStore(settings.index_dir), settings)
        quote = "The monthly rent shall be Rs. 48,500 payable in advance."
        answer = Answer(
            question="rent",
            text=f"The monthly rent shall be Rs. 48,500 [S1].",
            citations=[
                Citation(
                    marker="S1",
                    doc_id="missing",
                    doc_name="x.pdf",
                    page_no=1,
                    line_start=1,
                    line_end=2,
                    quote=quote,
                    used_in_answer=True,
                )
            ],
        )
        report = verifier.verify(answer)
        assert report.total == 1
        assert report.verdicts[0].supported
