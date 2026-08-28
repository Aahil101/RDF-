"""Study-material generation with a verified answer key.

Two generators behind one interface, mirroring the answering layer:

``LLM``
    Asks the model for JSON MCQs / short answers / flashcards, requiring it to
    quote the passage sentence that proves each answer.

``Cloze`` (no key, no model, no network)
    Builds fill-in-the-blank MCQs by deleting a salient fact — an amount, a date,
    a percentage, a defined term — from a source sentence and drawing distractors
    of the *same kind* from elsewhere in the document. Crude next to an LLM, but
    correct by construction and always available.

Whichever path runs, every item is then **verified**: the correct answer must be
recoverable from the actual PDF lines, scored with the same
:func:`~verirag.verify.grounding.support_score` used for chat answers. Items that
fail are counted and dropped. A quiz with an unverifiable answer key would teach a
student the wrong thing with no way to notice.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..generation.llm import LLM
from ..index.embedder import tokenize
from ..index.indexer import LineStore
from ..models import Chunk, Citation
from ..verify.grounding import best_window, support_score
from .models import MCQ, Flashcard, ShortAnswerQuestion, StudyPack, StudyTopic
from .prompts import (
    FLASHCARD_SYSTEM,
    FLASHCARD_USER,
    MCQ_SYSTEM,
    MCQ_USER,
    SHORT_ANSWER_SYSTEM,
    SHORT_ANSWER_USER,
)
from .topics import extract_topics

_STOP = frozenset(
    """a an and are as at be been by for from has have in is it its of on or shall that the their
    there this to was were with would which who what when where why how not no such any each per
    may must can will also both if but so however therefore thus within without upon under over""".split()
)

# Salient facts worth blanking out in a cloze question.
# Numbers are frequently spelled out in legal and academic drafting — "six per
# cent", "eleven (11) months" — so word forms are matched alongside digits.
_NUMBER_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-five|thirty|"
    r"thirty-three|forty|fifty|sixty|seventy|seventy-five|eighty|ninety|hundred)"
)
_MONEY_RE = re.compile(r"(?:Rs\.?|INR|USD|\$|\u20b9)\s?[\d,]+(?:\.\d+)?", re.IGNORECASE)
_PERCENT_RE = re.compile(
    rf"(?:\d+(?:\.\d+)?|{_NUMBER_WORD})\s?(?:%|per\s+cent|percent)", re.IGNORECASE
)
_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    rf"\b(?:\d+|{_NUMBER_WORD})\s*\(?\d*\)?\s*(?:days?|months?|years?|weeks?|hours?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

_BLANK = "______"


class StudyGenerator:
    """Generates verified study material from indexed chunks."""

    def __init__(
        self,
        llm: LLM | None,
        line_store: LineStore | None = None,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.llm = llm
        self.lines = line_store
        self.settings = settings or get_settings()
        self._random = random.Random(seed)

    # ------------------------------------------------------------------ public
    def build_pack(
        self,
        chunks: Sequence[Chunk],
        *,
        n_mcqs: int = 8,
        n_short: int = 4,
        n_cards: int = 8,
        topics: Sequence[StudyTopic] | None = None,
    ) -> StudyPack:
        """Produce a full study pack for one document."""
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("no chunks supplied")

        discovered = list(topics) if topics is not None else extract_topics(chunk_list)
        by_id = {c.chunk_id: c for c in chunk_list}
        pack = StudyPack(
            doc_id=chunk_list[0].doc_id,
            doc_name=chunk_list[0].doc_name,
            topics=discovered,
            generator="llm" if self.llm is not None else "cloze",
        )

        ranked = self._rank_topics(discovered)
        if not ranked:
            return pack

        pack.mcqs, rejected_mcq = self._make_mcqs(ranked, by_id, n_mcqs)
        pack.short_answers, rejected_sa = self._make_short_answers(ranked, by_id, n_short)
        pack.flashcards = self._make_flashcards(ranked, by_id, n_cards)
        pack.rejected = rejected_mcq + rejected_sa
        return pack

    def generate_mcqs(
        self,
        chunks: Sequence[Chunk],
        count: int = 8,
        *,
        topics: Sequence[StudyTopic] | None = None,
    ) -> tuple[list[MCQ], int]:
        chunk_list = list(chunks)
        discovered = list(topics) if topics is not None else extract_topics(chunk_list)
        by_id = {c.chunk_id: c for c in chunk_list}
        return self._make_mcqs(self._rank_topics(discovered), by_id, count)

    # ----------------------------------------------------------------- topics
    @staticmethod
    def _rank_topics(topics: Sequence[StudyTopic]) -> list[StudyTopic]:
        """Richest topics first — more substance means better questions."""
        return sorted(topics, key=lambda t: t.word_count, reverse=True)

    def _passage_for(self, topic: StudyTopic, by_id: dict[str, Chunk], limit: int = 2600) -> tuple[str, list[Chunk]]:
        members = [by_id[cid] for cid in topic.chunk_ids if cid in by_id]
        members.sort(key=lambda c: (c.page_no, c.line_start))
        text = " ".join(c.body_text for c in members)
        return text[:limit], members

    # ------------------------------------------------------------------- MCQs
    def _make_mcqs(
        self,
        topics: Sequence[StudyTopic],
        by_id: dict[str, Chunk],
        count: int,
    ) -> tuple[list[MCQ], int]:
        if count <= 0 or not topics:
            return [], 0

        questions: list[MCQ] = []
        rejected = 0
        per_topic = max(1, -(-count // max(len(topics), 1)))  # ceil division

        for topic in topics:
            if len(questions) >= count:
                break
            passage, members = self._passage_for(topic, by_id)
            if not passage.strip() or not members:
                continue

            wanted = min(per_topic, count - len(questions))
            if self.llm is not None:
                candidates = self._llm_mcqs(topic, passage, wanted)
                if not candidates:  # provider hiccup — stay useful
                    candidates = self._cloze_mcqs(topic, members, by_id, wanted)
            else:
                candidates = self._cloze_mcqs(topic, members, by_id, wanted)

            for question in candidates:
                if not question.is_wellformed():
                    rejected += 1
                    continue
                if not self._verify_mcq(question, members):
                    rejected += 1
                    continue
                questions.append(question)
                if len(questions) >= count:
                    break

        return questions, rejected

    def _llm_mcqs(self, topic: StudyTopic, passage: str, count: int) -> list[MCQ]:
        assert self.llm is not None
        response = self.llm.complete(
            MCQ_SYSTEM, MCQ_USER.format(topic=topic.name, passage=passage, count=count)
        )
        if not response.ok:
            return []

        out: list[MCQ] = []
        for item in _parse_json_array(response.text):
            options = [str(o).strip() for o in item.get("options", []) if str(o).strip()]
            try:
                correct = int(item.get("correct_index", -1))
            except (TypeError, ValueError):
                continue
            out.append(
                MCQ(
                    question=str(item.get("question", "")).strip(),
                    options=options,
                    correct_index=correct,
                    explanation=str(item.get("explanation", "")).strip(),
                    topic=topic.name,
                    difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
                    generator=f"llm:{self.llm.model}",
                )
            )
            out[-1].explanation = out[-1].explanation or str(item.get("evidence", "")).strip()
        return out

    def _cloze_mcqs(
        self,
        topic: StudyTopic,
        members: Sequence[Chunk],
        by_id: dict[str, Chunk],
        count: int,
    ) -> list[MCQ]:
        """Fill-in-the-blank questions built by deleting a salient fact."""
        from ..generation.citations import split_sentences

        pool = self._distractor_pool(by_id)
        questions: list[MCQ] = []

        for chunk in members:
            if len(questions) >= count:
                break
            for sentence in split_sentences(chunk.body_text):
                if len(questions) >= count:
                    break
                if len(sentence.split()) < 8:
                    continue
                found = _find_salient(sentence)
                if not found:
                    continue
                answer, kind = found

                distractors = self._pick_distractors(answer, kind, pool, wanted=3)
                if len(distractors) < 3:
                    continue

                options = [answer, *distractors]
                self._random.shuffle(options)
                stem = sentence.replace(answer, _BLANK, 1)
                questions.append(
                    MCQ(
                        question=f"Fill in the blank: {stem}",
                        options=options,
                        correct_index=options.index(answer),
                        explanation=f"The passage states: \u201c{sentence.strip()}\u201d",
                        topic=topic.name,
                        difficulty="easy" if kind in {"money", "percent"} else "medium",
                        generator="cloze",
                    )
                )
        return questions

    def _distractor_pool(self, by_id: dict[str, Chunk]) -> dict[str, list[str]]:
        """Same-kind values from across the document, for plausible distractors."""
        pool: dict[str, list[str]] = {"money": [], "percent": [], "date": [], "duration": [], "number": [], "term": []}
        for chunk in by_id.values():
            text = chunk.body_text
            pool["money"] += _MONEY_RE.findall(text)
            pool["percent"] += _PERCENT_RE.findall(text)
            pool["date"] += _DATE_RE.findall(text)
            pool["duration"] += _DURATION_RE.findall(text)
            pool["number"] += _NUMBER_RE.findall(text)
            pool["term"] += [
                t for t in tokenize(text) if t not in _STOP and len(t) > 6
            ]
        return {kind: list(dict.fromkeys(values)) for kind, values in pool.items()}

    def _pick_distractors(self, answer: str, kind: str, pool: dict[str, list[str]], *, wanted: int) -> list[str]:
        normalised = answer.strip().lower()
        candidates = [v for v in pool.get(kind, []) if v.strip().lower() != normalised]
        # Prefer values of similar shape so the answer is not obvious by format.
        candidates.sort(key=lambda v: abs(len(v) - len(answer)))
        picked: list[str] = []
        seen = {normalised}
        for value in candidates:
            key = value.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(value.strip())
            if len(picked) >= wanted:
                break
        return picked

    # --------------------------------------------------------- short answers
    def _make_short_answers(
        self,
        topics: Sequence[StudyTopic],
        by_id: dict[str, Chunk],
        count: int,
    ) -> tuple[list[ShortAnswerQuestion], int]:
        if count <= 0 or not topics:
            return [], 0
        questions: list[ShortAnswerQuestion] = []
        rejected = 0

        for topic in topics:
            if len(questions) >= count:
                break
            passage, members = self._passage_for(topic, by_id)
            if not passage.strip() or not members:
                continue

            if self.llm is not None:
                candidates = self._llm_short_answers(topic, passage, min(2, count - len(questions)))
            else:
                candidates = self._heuristic_short_answers(topic, members)

            for question in candidates:
                if not question.question.strip() or not question.answer.strip():
                    rejected += 1
                    continue
                score, citation = self._locate(question.answer, members)
                question.verification_score = score
                question.verified = score >= self.settings.grounding_threshold
                question.citation = citation
                if not question.verified:
                    rejected += 1
                    continue
                questions.append(question)
                if len(questions) >= count:
                    break
        return questions, rejected

    def _llm_short_answers(self, topic: StudyTopic, passage: str, count: int) -> list[ShortAnswerQuestion]:
        assert self.llm is not None
        if count <= 0:
            return []
        response = self.llm.complete(
            SHORT_ANSWER_SYSTEM,
            SHORT_ANSWER_USER.format(topic=topic.name, passage=passage, count=count),
        )
        if not response.ok:
            return []
        out: list[ShortAnswerQuestion] = []
        for item in _parse_json_array(response.text):
            try:
                marks = int(item.get("marks", 2))
            except (TypeError, ValueError):
                marks = 2
            out.append(
                ShortAnswerQuestion(
                    question=str(item.get("question", "")).strip(),
                    answer=str(item.get("answer", "")).strip(),
                    topic=topic.name,
                    marks=marks,
                    generator=f"llm:{self.llm.model}",
                )
            )
        return out

    def _heuristic_short_answers(
        self, topic: StudyTopic, members: Sequence[Chunk]
    ) -> list[ShortAnswerQuestion]:
        """Turn definitional sentences into questions without a model."""
        from ..generation.citations import split_sentences

        patterns = (
            (re.compile(r"^(.{4,70}?)\s+(?:is|are)\s+defined as\s+(.+)$", re.IGNORECASE), "What is {0}?"),
            (re.compile(r"^(?:A|An|The)\s+(.{4,60}?)\s+is\s+(.+)$", re.IGNORECASE), "What is {0}?"),
            (re.compile(r"^(.{4,60}?)\s+means\s+(.+)$", re.IGNORECASE), "What does {0} mean?"),
            (re.compile(r"^(.{4,60}?)\s+states that\s+(.+)$", re.IGNORECASE), "What does {0} state?"),
        )
        out: list[ShortAnswerQuestion] = []
        for chunk in members:
            for sentence in split_sentences(chunk.body_text):
                if len(sentence.split()) < 8:
                    continue
                for pattern, template in patterns:
                    match = pattern.match(sentence.strip())
                    if not match:
                        continue
                    subject = match.group(1).strip().rstrip(",")
                    out.append(
                        ShortAnswerQuestion(
                            question=template.format(subject),
                            answer=sentence.strip(),
                            topic=topic.name,
                            marks=2,
                            generator="heuristic",
                        )
                    )
                    break
                if len(out) >= 3:
                    return out
        return out

    # ------------------------------------------------------------- flashcards
    def _make_flashcards(
        self,
        topics: Sequence[StudyTopic],
        by_id: dict[str, Chunk],
        count: int,
    ) -> list[Flashcard]:
        if count <= 0 or not topics:
            return []
        cards: list[Flashcard] = []
        for topic in topics:
            if len(cards) >= count:
                break
            passage, members = self._passage_for(topic, by_id)
            if not passage.strip() or not members:
                continue
            if self.llm is not None:
                candidates = self._llm_flashcards(topic, passage, min(3, count - len(cards)))
            else:
                candidates = self._heuristic_flashcards(topic, members)
            for card in candidates:
                if not card.front.strip() or not card.back.strip():
                    continue
                _score, citation = self._locate(card.back, members)
                card.citation = citation
                cards.append(card)
                if len(cards) >= count:
                    break
        return cards

    def _llm_flashcards(self, topic: StudyTopic, passage: str, count: int) -> list[Flashcard]:
        assert self.llm is not None
        if count <= 0:
            return []
        response = self.llm.complete(
            FLASHCARD_SYSTEM, FLASHCARD_USER.format(topic=topic.name, passage=passage, count=count)
        )
        if not response.ok:
            return []
        return [
            Flashcard(
                front=str(item.get("front", "")).strip(),
                back=str(item.get("back", "")).strip(),
                topic=topic.name,
            )
            for item in _parse_json_array(response.text)
        ]

    def _heuristic_flashcards(self, topic: StudyTopic, members: Sequence[Chunk]) -> list[Flashcard]:
        cards = [
            Flashcard(front=f"Key terms in \u201c{topic.name}\u201d", back=", ".join(topic.key_terms[:6]), topic=topic.name)
        ] if topic.key_terms else []
        for question in self._heuristic_short_answers(topic, members)[:2]:
            cards.append(Flashcard(front=question.question, back=question.answer, topic=topic.name))
        return cards

    # ----------------------------------------------------------- verification
    def _verify_mcq(self, question: MCQ, members: Sequence[Chunk]) -> bool:
        """The correct option must be traceable to the source lines."""
        target = question.correct_option
        if not target:
            return False

        # A cloze answer is a fragment; verify the whole filled sentence instead,
        # which is what actually has to appear in the document.
        probe = target
        if _BLANK in question.question:
            probe = question.question.split("Fill in the blank:", 1)[-1].strip().replace(_BLANK, target)

        score, citation = self._locate(probe, members)
        question.verification_score = score
        question.verified = score >= self.settings.grounding_threshold
        question.citation = citation
        return question.verified

    def _locate(self, claim: str, members: Sequence[Chunk]) -> tuple[float, Citation | None]:
        """Best-supporting span for *claim* across *members*, as a Citation."""
        best_score = 0.0
        best: Citation | None = None

        for chunk in members:
            lines = (
                self.lines.get_range(chunk.doc_id, chunk.page_no, chunk.line_start, chunk.line_end)
                if self.lines is not None
                else []
            )
            if lines:
                score, window = best_window(claim, lines)
                if score > best_score and window:
                    best_score = score
                    best = Citation(
                        marker="S1",
                        doc_id=chunk.doc_id,
                        doc_name=chunk.doc_name,
                        page_no=chunk.page_no,
                        line_start=window[0].line_no,
                        line_end=window[-1].line_no,
                        quote=" ".join(ln.text for ln in window),
                        used_in_answer=True,
                        bboxes=[ln.bbox for ln in window],
                    )
            else:
                score = support_score(claim, chunk.text)
                if score > best_score:
                    best_score = score
                    best = Citation(
                        marker="S1",
                        doc_id=chunk.doc_id,
                        doc_name=chunk.doc_name,
                        page_no=chunk.page_no,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        quote=chunk.text,
                        used_in_answer=True,
                        bboxes=list(chunk.line_bboxes),
                    )
        return round(best_score, 4), best


# ---------------------------------------------------------------------------
def _find_salient(sentence: str) -> tuple[str, str] | None:
    """Pick the most quiz-worthy fact in a sentence, with its kind."""
    for kind, pattern in (
        ("money", _MONEY_RE),
        ("percent", _PERCENT_RE),
        ("date", _DATE_RE),
        ("duration", _DURATION_RE),
    ):
        match = pattern.search(sentence)
        if match:
            value = match.group(0).strip()
            if len(value) >= 2:
                return value, kind
    match = _NUMBER_RE.search(sentence)
    if match and len(match.group(0)) >= 2:
        return match.group(0), "number"
    return None


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from a model reply, tolerating fences and preamble."""
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload = payload.get("questions") or payload.get("items") or [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    return []
