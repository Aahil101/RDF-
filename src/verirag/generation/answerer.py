"""Grounded answer generation.

Two composition strategies share one output contract:

``LLM mode``
    The retrieved sources are numbered, injected into a citation-mandating
    prompt, and the reply is sanitised so only real source markers survive.

``Extractive mode`` (no API key, no model, no network)
    The answer is stitched from the highest-scoring *lines* of the top chunks,
    each carrying its own ``[S#]`` marker.  Quality is lower than an LLM, but
    it is 100% grounded by construction and it means the project always runs.

Either way the caller receives citations bound to a document, page and line
range, ready for the verification and highlighting layers.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

from ..config import Settings, get_settings
from ..index.embedder import tokenize
from ..models import Answer, Citation, RetrievedChunk
from ..retrieval.retriever import RetrievalResult
from .citations import sanitize_answer, split_sentences
from .llm import LLM, describe_provider
from .prompts import (
    REFUSAL_TOKEN,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_history_preamble,
    build_sources_text,
)

_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or shall that the their
    there this to was were what which who will with would does do did how when where why""".split()
)

NO_EVIDENCE_MESSAGE = (
    "I could not find this in the indexed documents, so I will not guess. "
    "Try rephrasing with wording closer to the document, or ingest the PDF that covers it."
)


class Answerer:
    """Turns retrieved evidence into a cited answer."""

    def __init__(self, llm: LLM | None, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ public
    def answer(
        self,
        retrieval: RetrievalResult,
        *,
        history: Sequence[tuple[str, str]] = (),
    ) -> Answer:
        started = time.perf_counter()
        question = retrieval.query
        items = retrieval.chunks

        if not items or retrieval.best_score < self.settings.min_retrieval_score or self._no_lexical_footprint(
            question, items
        ):
            return Answer(
                question=question,
                text=NO_EVIDENCE_MESSAGE,
                citations=[],
                groundedness=0.0,
                provider=describe_provider(self.llm),
                model=getattr(self.llm, "model", ""),
                latency_ms=int((time.perf_counter() - started) * 1000),
                refused=True,
                weak_evidence=True,
                retrieval_score=round(retrieval.best_score, 5),
                retrieval_trace=retrieval.trace(),
            )

        citations = self._build_citations(items)
        provider_error = ""

        if self.llm is not None:
            text, used, refused, provider_error = self._generate_with_llm(question, items, history)
            provider = self.llm.provider
            model = self.llm.model
            if not text:  # provider failed at runtime -> stay useful, stay grounded
                text, used = compose_extractive(question, items, variants=retrieval.variants)
                provider, model, refused = "extractive-fallback", "", False
        else:
            text, used = compose_extractive(question, items, variants=retrieval.variants)
            provider, model, refused = "extractive", "", False

        used_set = set(used)
        for index, citation in enumerate(citations, start=1):
            citation.used_in_answer = index in used_set

        return Answer(
            question=question,
            text=text,
            citations=citations,
            groundedness=0.0,  # filled in by the verification layer
            provider=provider,
            model=model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            refused=refused,
            weak_evidence=retrieval.best_score < self.settings.low_confidence_score,
            retrieval_score=round(retrieval.best_score, 5),
            provider_error=provider_error,
            retrieval_trace=retrieval.trace(),
        )

    # ----------------------------------------------------------------- helpers
    def summarise(self, question: str, retrieval: RetrievalResult, doc_name: str) -> Answer:
        """Summarise a document from a representative spread of its chunks.

        Separate from :meth:`answer` because the task is different: the excerpts
        are a *sample* of the whole file rather than the passages most similar to a
        question, and the prompt has to say so or the model will imply
        completeness it cannot have.
        """
        started = time.perf_counter()
        items = retrieval.chunks
        if not items:
            return Answer(
                question=question,
                text=NO_EVIDENCE_MESSAGE,
                refused=True,
                provider=describe_provider(self.llm),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        citations = self._build_citations(items)
        provider_error = ""

        if self.llm is not None:
            prompt = SUMMARY_USER.format(doc_name=doc_name, sources=build_sources_text(items, max_chars=900))
            response = self.llm.complete(SUMMARY_SYSTEM, prompt)
            if response.ok:
                text, used = sanitize_answer(response.text.strip(), len(items))
                provider, model = self.llm.provider, self.llm.model
                if not used:
                    text, used = f"{text} [S1]".strip(), [1]
            else:
                provider_error = response.error or "empty response from provider"
                text, used = compose_overview(items, doc_name)
                provider, model = "extractive-fallback", ""
        else:
            text, used = compose_overview(items, doc_name)
            provider, model = "extractive", ""

        used_set = set(used)
        for index, citation in enumerate(citations, start=1):
            citation.used_in_answer = index in used_set

        return Answer(
            question=question,
            text=text,
            citations=citations,
            provider=provider,
            model=model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            weak_evidence=False,  # a spread of the document is the right evidence here
            retrieval_score=1.0,
            provider_error=provider_error,
            retrieval_trace=retrieval.trace(),
        )

    @staticmethod
    def _no_lexical_footprint(question: str, items: Sequence[RetrievedChunk]) -> bool:
        """True when none of the question's content words occur in the evidence.

        A relevance threshold alone can be fooled by an unrelated question that
        happens to share one incidental word with the corpus. If not a single
        content term of the question appears in the best chunk, there is nothing
        to ground an answer on and refusing is the only honest response.
        """
        terms = {t for t in tokenize(question) if t not in _STOP and len(t) > 2}
        if not terms:
            return False
        best = items[0].chunk
        evidence = set(tokenize(f"{best.section} {best.text}"))
        return not (terms & evidence)

    @staticmethod
    def _build_citations(items: Sequence[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for index, item in enumerate(items, start=1):
            chunk = item.chunk
            citations.append(
                Citation(
                    marker=f"S{index}",
                    doc_id=chunk.doc_id,
                    doc_name=chunk.doc_name,
                    page_no=chunk.page_no,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    quote=chunk.text,
                    retrieval_score=round(item.score, 5),
                    bboxes=list(chunk.line_bboxes),
                )
            )
        return citations

    def _generate_with_llm(
        self,
        question: str,
        items: Sequence[RetrievedChunk],
        history: Sequence[tuple[str, str]],
    ) -> tuple[str, list[int], bool, str]:
        """Returns ``(text, used_markers, refused, provider_error)``."""
        assert self.llm is not None
        prompt = build_history_preamble(list(history)) + build_answer_prompt(question, items)
        response = self.llm.complete(SYSTEM_PROMPT, prompt)
        if not response.ok:
            return "", [], False, response.error or "empty response from provider"

        raw = response.text.strip()
        if REFUSAL_TOKEN in raw.upper():
            reason = raw.split(":", 1)[1].strip() if ":" in raw else ""
            message = NO_EVIDENCE_MESSAGE if not reason else f"{NO_EVIDENCE_MESSAGE} Missing: {reason}"
            return message, [], True, ""

        text, used = sanitize_answer(raw, len(items))
        if not used:
            # The model answered but forgot to cite: attach the strongest source
            # rather than surfacing an uncited claim as if it were verified.
            text = f"{text} [S1]".strip()
            used = [1]
        return text, used, False, ""


def compose_overview(items: Sequence[RetrievedChunk], doc_name: str) -> tuple[str, list[int]]:
    """Build a document overview with no LLM.

    One line per excerpt: its section heading where the document has one, plus the
    most substantive sentence from it. Crude beside a generated summary, but every
    line is quoted from the file and carries its own citation — and crucially it
    answers the question instead of refusing.
    """
    lines = [f"{doc_name} contains the following, based on a spread of {len(items)} excerpts:"]
    used: list[int] = []

    for index, item in enumerate(items, start=1):
        chunk = item.chunk
        sentences = [s for s in split_sentences(chunk.body_text) if len(s.split()) >= 5]
        if not sentences:
            sentences = [chunk.body_text.strip()]
        # The longest sentence is usually the substantive one; headings and
        # fragments are short.
        best = max(sentences, key=lambda s: len(s.split()))[:260].strip()
        label = chunk.section.strip().rstrip(".") if chunk.section else f"page {chunk.page_no}"
        lines.append(f"- {label}: {best.rstrip('.')}. [S{index}]")
        used.append(index)

    lines.append(
        "This is assembled from excerpts rather than written as prose. "
        "Add a free Groq or Gemini key for a fluent summary."
    )
    return "\n".join(lines), used


# ---------------------------------------------------------------------------
# extractive composer (no LLM required)
# ---------------------------------------------------------------------------
_WANTS_VALUE_RE = re.compile(
    r"\b(how much|how many|how long|how far|what rate|what percentage|what amount|when|by when|"
    r"which date|what date|what is the (?:rent|fee|amount|deposit|rate|penalty|interest|area|"
    r"duration|term|period|notice|price|cost|charge|escalation|tax))\b"
)
_WANTS_PERSON_RE = re.compile(r"\b(who|whom|whose|which party|by whom)\b")
_INTERROGATIVE_RE = re.compile(r"^\s*(?:\(?[ivxlc]+\)?[.)]\s*)?whether\b", re.IGNORECASE)
_ISSUE_MARKER_RE = re.compile(r"^\s*\(?(?:i{1,3}|iv|v|vi{1,3}|ix|x)\)", re.IGNORECASE)
# Sentences that *ask* rather than *tell*. Study material is full of these
# ("Explain why Repeatable Read permits phantom reads. (5 marks)") and they
# overlap a user's question almost perfectly while answering nothing.
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:explain|define|describe|differentiate|distinguish|compare|state|list|discuss|"
    r"derive|prove|compute|calculate|show|write|give|justify|outline|summarise|summarize)\b",
    re.IGNORECASE,
)
_MARKS_RE = re.compile(r"\(\s*\d+\s*marks?\s*\)", re.IGNORECASE)
_LEADING_NOISE_RE = re.compile(
    r"^(?:\s*(?:\(\s*\d+\s*marks?\s*\)|[\u2022\u00b7\u2023\u25aa\u25cf\u2043\u2219]|"
    r"\d+(?:\.\d+)*[.)]|\(\s*[a-z0-9ivxlc]+\s*\)))+\s*",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
_OBLIGATION_RE = re.compile(r"\b(shall|must|is liable|is responsible|shall bear|shall pay|agrees to)\b", re.IGNORECASE)


def _caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def compose_extractive(
    question: str,
    items: Sequence[RetrievedChunk],
    max_sentences: int = 4,
    *,
    variants: Sequence[str] = (),
) -> tuple[str, list[int]]:
    """Build an answer from the best-matching sentences of the top chunks.

    Scoring blends lexical overlap with question-type priors, because the
    highest-overlap sentence is very often the one that *restates* the question
    (a framed issue in a judgment, an exam question in study notes) rather than
    the one that answers it.

    ``variants`` are the expanded queries from the retrieval stage. Including
    their terms lets the composer match a passage that spells an abbreviation
    out in full — "Boyce-Codd normal form" for a question asking about "BCNF".
    """
    lowered = question.lower()
    wants_value = bool(_WANTS_VALUE_RE.search(lowered))
    wants_person = bool(_WANTS_PERSON_RE.search(lowered))

    q_terms = {t for t in tokenize(question) if t not in _STOP and len(t) > 1}
    expanded_terms = set(q_terms)
    for variant in variants:
        expanded_terms |= {t for t in tokenize(variant) if t not in _STOP and len(t) > 1}

    scored: list[tuple[float, int, str]] = []

    for index, item in enumerate(items, start=1):
        for sentence in split_sentences(item.chunk.body_text):
            cleaned = sentence.strip()
            terms = set(tokenize(cleaned))
            if not terms:
                continue

            direct = len(q_terms & terms) / max(len(q_terms), 1)
            widened = len(expanded_terms & terms) / max(len(expanded_terms), 1)
            overlap = max(direct, 0.85 * widened)  # expansion counts, but slightly less
            density = len(expanded_terms & terms) / (len(terms) ** 0.5)
            rank_prior = 1.0 / index
            score = 0.55 * overlap + 0.28 * density + 0.10 * rank_prior

            # Answer-bearing signals.
            if wants_value and _DIGIT_RE.search(cleaned):
                score += 0.16
            if wants_person and _OBLIGATION_RE.search(cleaned):
                score += 0.10

            # Anti-signals are *multiplicative*. A sentence that asks rather than
            # tells is categorically not an answer, however well its wording
            # matches the question — and a fixed subtraction cannot express that,
            # because an exam prompt matches the user's question almost perfectly
            # and so starts from the highest possible overlap.
            #
            # They are tested against the sentence with leading noise (bullet
            # glyphs, a preceding item's "(6 marks)" tail, enumerators) removed,
            # otherwise a prompt escapes detection simply because the extractor
            # glued something in front of it.
            probe = _LEADING_NOISE_RE.sub("", cleaned).strip()
            if _INSTRUCTION_RE.match(probe):  # "Explain why Repeatable Read ..."
                score *= 0.35
            if _INTERROGATIVE_RE.search(probe) or probe.rstrip().endswith("?"):
                score *= 0.55
            if _ISSUE_MARKER_RE.match(probe):  # "(iii) Whether the Appellant ..."
                score *= 0.70
            if _MARKS_RE.search(cleaned):  # "... (6 marks)" — an exam prompt
                score *= 0.65
            if _caps_ratio(cleaned) > 0.6:  # a shouted heading
                score *= 0.70
            if len(terms) < 5:
                score *= 0.85

            if overlap > 0 or index == 1:
                scored.append((score, index, cleaned))

    scored.sort(key=lambda row: row[0], reverse=True)

    # Keep only sentences competitive with the best one: padding an extractive
    # answer with weakly-related prose is what drags groundedness down.
    picked: list[tuple[int, str]] = []
    seen: set[str] = set()
    cutoff = scored[0][0] * 0.55 if scored else 0.0
    for score, index, sentence in scored:
        if picked and score < cutoff:
            break
        key = sentence.lower()[:90]
        if key in seen:
            continue
        seen.add(key)
        picked.append((index, sentence))
        if len(picked) >= max_sentences:
            break

    if not picked:
        first = items[0].chunk.body_text.strip()
        picked = [(1, first[:400])]

    parts = [f"{sentence.rstrip('.')}. [S{index}]" for index, sentence in picked]
    used = list(dict.fromkeys(index for index, _ in picked))
    return " ".join(parts), used
