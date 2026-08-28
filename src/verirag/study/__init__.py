"""Study mode: topic discovery, MCQ/short-answer/flashcard generation, tutoring.

Every generated item carries a citation and is verified against the source PDF
lines before it is shown, so an answer key can always be checked.
"""

from __future__ import annotations

from .generator import StudyGenerator
from .models import MCQ, Flashcard, ShortAnswerQuestion, StudyPack, StudyTopic
from .topics import extract_key_terms, extract_topics

__all__ = [
    "Flashcard",
    "MCQ",
    "ShortAnswerQuestion",
    "StudyGenerator",
    "StudyPack",
    "StudyTopic",
    "extract_key_terms",
    "extract_topics",
]
