"""Chat persistence tests.

Storing the answer text is easy; the point of this schema is that the *evidence*
survives too. If citations, line ranges or per-sentence verdicts did not round
trip, reopening a conversation would show an answer nobody could re-verify.
"""

from __future__ import annotations

import sqlite3

import pytest

from verirag.models import Answer, Citation, ClaimVerdict, EvidenceSpan
from verirag.store.chat_db import ChatDB


@pytest.fixture()
def db(tmp_path) -> ChatDB:
    return ChatDB(tmp_path / "chat.sqlite3")


def make_answer(text: str = "The monthly rent is Rs. 48,500 [S1].") -> Answer:
    citation = Citation(
        marker="S1",
        doc_id="doc-1",
        doc_name="lease.pdf",
        page_no=2,
        line_start=14,
        line_end=17,
        quote="The monthly rent shall be Rs. 48,500 payable in advance.",
        retrieval_score=0.8123,
        used_in_answer=True,
        bboxes=[(64.0, 100.0, 500.0, 114.0)],
    )
    evidence = EvidenceSpan(
        doc_id="doc-1",
        doc_name="lease.pdf",
        page_no=2,
        line_start=14,
        line_end=15,
        text="The monthly rent shall be Rs. 48,500",
        bboxes=[(64.0, 100.0, 500.0, 114.0)],
        similarity=0.97,
    )
    return Answer(
        question="what is the rent",
        text=text,
        citations=[citation],
        verdicts=[ClaimVerdict(sentence=text, supported=True, score=0.97, markers=["S1"], evidence=evidence)],
        groundedness=0.97,
        provider="extractive",
        model="",
        latency_ms=42,
        retrieval_score=0.8123,
        retrieval_trace=[{"chunk_id": "abc", "score": 0.81}],
    )


class TestSessions:
    def test_create_and_list(self, db: ChatDB):
        session_id = db.create_session("Lease review")
        sessions = db.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].id == session_id
        assert sessions[0].title == "Lease review"

    def test_blank_title_defaults(self, db: ChatDB):
        db.create_session("   ")
        assert db.list_sessions()[0].title == "New chat"

    def test_first_question_becomes_the_title(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "What is the monthly rent for the flat?")
        assert db.list_sessions()[0].title.startswith("What is the monthly rent")

    def test_long_title_is_truncated(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "x" * 200)
        assert len(db.list_sessions()[0].title) <= 61

    def test_rename(self, db: ChatDB):
        session_id = db.create_session()
        db.rename_session(session_id, "Renamed")
        assert db.list_sessions()[0].title == "Renamed"

    def test_delete_returns_flag(self, db: ChatDB):
        session_id = db.create_session()
        assert db.delete_session(session_id) is True
        assert db.delete_session("nope") is False

    def test_ordered_most_recent_first(self, db: ChatDB):
        first = db.create_session("first")
        second = db.create_session("second")
        # Newest session leads until the older one is touched again.
        assert [s.id for s in db.list_sessions()][0] == second
        db.add_user_message(first, "later activity")
        ordering = [s.id for s in db.list_sessions()]
        assert ordering[0] == first
        assert ordering[1] == second


class TestMessagesAndEvidence:
    def test_round_trip_preserves_citation_geometry(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "what is the rent")
        db.add_answer(session_id, make_answer())

        messages = db.get_messages(session_id)
        assert [m.role for m in messages] == ["user", "assistant"]
        stored = messages[1].citations[0]
        assert stored.page_no == 2
        assert (stored.line_start, stored.line_end) == (14, 17)
        assert stored.used_in_answer is True
        assert stored.retrieval_score == pytest.approx(0.8123)
        assert stored.bboxes == [(64.0, 100.0, 500.0, 114.0)]

    def test_round_trip_preserves_verdicts_and_evidence(self, db: ChatDB):
        session_id = db.create_session()
        db.add_answer(session_id, make_answer())
        verdict = db.get_messages(session_id)[0].verdicts[0]
        assert verdict.supported is True
        assert verdict.markers == ["S1"]
        assert verdict.evidence is not None
        assert verdict.evidence.line_start == 14
        assert verdict.evidence.similarity == pytest.approx(0.97)

    def test_metadata_columns_round_trip(self, db: ChatDB):
        session_id = db.create_session()
        answer = make_answer()
        answer.weak_evidence = True
        db.add_answer(session_id, answer)
        message = db.get_messages(session_id)[0]
        assert message.provider == "extractive"
        assert message.latency_ms == 42
        assert message.weak_evidence is True
        assert message.retrieval_score == pytest.approx(0.8123)
        assert message.retrieval_trace[0]["chunk_id"] == "abc"

    def test_confidence_band_is_stored(self, db: ChatDB):
        session_id = db.create_session()
        db.add_answer(session_id, make_answer())
        assert db.get_messages(session_id)[0].confidence == "high"

    def test_history_pairs_for_follow_up_context(self, db: ChatDB):
        session_id = db.create_session()
        for index in range(3):
            db.add_user_message(session_id, f"question {index}")
            db.add_answer(session_id, make_answer(f"answer {index} [S1]."))
        pairs = db.history_pairs(session_id, limit=2)
        assert len(pairs) == 2
        assert pairs[-1][0] == "question 2"
        assert pairs[-1][1].startswith("answer 2")

    def test_cascade_delete_removes_evidence(self, db: ChatDB):
        session_id = db.create_session()
        db.add_answer(session_id, make_answer())
        db.delete_session(session_id)
        with sqlite3.connect(db.db_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM citations").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 0

    def test_role_constraint_is_enforced(self, db: ChatDB):
        session_id = db.create_session()
        with sqlite3.connect(db.db_path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                    (session_id, "system", "x", "now"),
                )

    def test_messages_of_unknown_session(self, db: ChatDB):
        assert db.get_messages("does-not-exist") == []


class TestFeedback:
    def test_set_and_get(self, db: ChatDB):
        session_id = db.create_session()
        message_id = db.add_answer(session_id, make_answer())
        db.set_feedback(message_id, 1, "accurate")
        assert db.get_feedback(message_id)["rating"] == 1

    def test_upsert_overwrites(self, db: ChatDB):
        session_id = db.create_session()
        message_id = db.add_answer(session_id, make_answer())
        db.set_feedback(message_id, 1)
        db.set_feedback(message_id, -1, "wrong clause")
        feedback = db.get_feedback(message_id)
        assert feedback["rating"] == -1
        assert feedback["note"] == "wrong clause"

    def test_missing_feedback_is_none(self, db: ChatDB):
        assert db.get_feedback(999) is None


class TestSearch:
    def test_finds_stored_turns(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "What is the security deposit?")
        db.add_answer(session_id, make_answer())
        assert any("deposit" in hit["content"].lower() for hit in db.search("deposit"))

    def test_punctuation_does_not_break_the_query(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "What is the rent?")
        assert db.search('rent? "quoted" (parens)') is not None

    def test_empty_query_returns_nothing(self, db: ChatDB):
        assert db.search("   ") == []

    def test_no_match_returns_empty(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "rent")
        assert db.search("zzzqqqwww") == []


class TestExport:
    def test_markdown_includes_locators(self, db: ChatDB):
        session_id = db.create_session("Lease review")
        db.add_user_message(session_id, "what is the rent")
        db.add_answer(session_id, make_answer())
        markdown = db.export_markdown(session_id)
        assert "Lease review" in markdown
        assert "p.2 L14-17" in markdown
        assert "lease.pdf" in markdown

    def test_json_export_structure(self, db: ChatDB):
        session_id = db.create_session()
        db.add_answer(session_id, make_answer())
        payload = db.export_session(session_id)
        assert payload["session"]["id"] == session_id
        assert payload["messages"][0]["citations"][0]["page_no"] == 2

    def test_unknown_session_raises(self, db: ChatDB):
        with pytest.raises(KeyError):
            db.export_session("nope")


class TestStats:
    def test_reports_counts_and_averages(self, db: ChatDB):
        session_id = db.create_session()
        db.add_user_message(session_id, "q")
        db.add_answer(session_id, make_answer())
        stats = db.stats()
        assert stats["sessions"] == 1
        assert stats["messages"] == 2
        assert stats["citations"] == 1
        assert stats["avg_groundedness"] == pytest.approx(0.97, abs=0.01)

    def test_empty_database_is_safe(self, db: ChatDB):
        stats = db.stats()
        assert stats["sessions"] == 0
        assert stats["avg_groundedness"] == 0.0
