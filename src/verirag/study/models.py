"""Study-mode domain models.

Everything generated here keeps the project's central promise: a question, an
answer key, or a flashcard is only useful if you can check it against the source.
So every item carries a :class:`~verirag.models.Citation` — document, page and
line range — and a verification score produced by the same grounding verifier
used for chat answers.

An MCQ whose "correct" option cannot be traced back to the PDF is worse than no
MCQ at all, because a student would learn the wrong thing and have no way to
notice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import Citation


@dataclass(slots=True)
class StudyTopic:
    """A coherent unit of study material discovered in a document."""

    name: str
    doc_id: str
    doc_name: str
    chunk_ids: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    key_terms: list[str] = field(default_factory=list)
    word_count: int = 0

    @property
    def page_range(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MCQ:
    """A multiple-choice question with a source-verified answer key."""

    question: str
    options: list[str]
    correct_index: int
    explanation: str = ""
    topic: str = ""
    difficulty: str = "medium"
    citation: Citation | None = None
    verified: bool = False
    verification_score: float = 0.0
    generator: str = ""

    @property
    def correct_option(self) -> str:
        if 0 <= self.correct_index < len(self.options):
            return self.options[self.correct_index]
        return ""

    @property
    def correct_letter(self) -> str:
        return "ABCDEFGH"[self.correct_index] if 0 <= self.correct_index < 8 else "?"

    def is_wellformed(self) -> bool:
        """Structural sanity: 3+ distinct options and a valid answer index."""
        if not self.question.strip() or len(self.options) < 3:
            return False
        if not 0 <= self.correct_index < len(self.options):
            return False
        normalised = [o.strip().lower() for o in self.options if o.strip()]
        return len(normalised) == len(self.options) and len(set(normalised)) == len(normalised)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citation"] = self.citation.to_dict() if self.citation else None
        payload["correct_letter"] = self.correct_letter
        return payload


@dataclass(slots=True)
class ShortAnswerQuestion:
    """An open question with a model answer taken from the source."""

    question: str
    answer: str
    topic: str = ""
    marks: int = 2
    citation: Citation | None = None
    verified: bool = False
    verification_score: float = 0.0
    generator: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citation"] = self.citation.to_dict() if self.citation else None
        return payload


@dataclass(slots=True)
class Flashcard:
    """A term/definition pair for spaced repetition."""

    front: str
    back: str
    topic: str = ""
    citation: Citation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citation"] = self.citation.to_dict() if self.citation else None
        return payload


@dataclass(slots=True)
class StudyPack:
    """Everything generated for one document."""

    doc_id: str
    doc_name: str
    topics: list[StudyTopic] = field(default_factory=list)
    mcqs: list[MCQ] = field(default_factory=list)
    short_answers: list[ShortAnswerQuestion] = field(default_factory=list)
    flashcards: list[Flashcard] = field(default_factory=list)
    generator: str = ""
    rejected: int = 0
    """Items discarded because their answer could not be verified against the PDF."""

    @property
    def verified_mcqs(self) -> list[MCQ]:
        return [q for q in self.mcqs if q.verified]

    def stats(self) -> dict[str, Any]:
        return {
            "document": self.doc_name,
            "topics": len(self.topics),
            "mcqs": len(self.mcqs),
            "mcqs_verified": len(self.verified_mcqs),
            "short_answers": len(self.short_answers),
            "flashcards": len(self.flashcards),
            "rejected_unverifiable": self.rejected,
            "generator": self.generator,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "generator": self.generator,
            "rejected": self.rejected,
            "topics": [t.to_dict() for t in self.topics],
            "mcqs": [q.to_dict() for q in self.mcqs],
            "short_answers": [q.to_dict() for q in self.short_answers],
            "flashcards": [c.to_dict() for c in self.flashcards],
        }
