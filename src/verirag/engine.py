"""VeriRAG engine — the single entry point the CLI, UI and tests all use.

Pipeline per question::

    question
      -> query expansion (rule-based variants)
      -> dense retrieval (cosine)  +  BM25 retrieval
      -> reciprocal rank fusion
      -> reranking
      -> citation-strict prompt  ->  LLM   (or extractive composer)
      -> independent groundedness verification per sentence
      -> proven span narrowing (page + line range)
      -> highlighted page render
      -> persist turn + evidence to SQLite

Keeping this orchestration in one class means every surface behaves identically,
and it is the natural place to point an interviewer at first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import Settings, get_settings
from .generation.answerer import Answerer
from .generation.llm import LLM, describe_provider, get_llm
from .index.indexer import Indexer, IngestReport
from .models import Answer, Citation, Document, RetrievedChunk
from .proof.highlighter import HighlightSpan, ProofImage, ProofRenderer
from .retrieval.intent import Intent, classify, topic_of_interest
from .retrieval.retriever import HybridRetriever, RetrievalResult
from .store.chat_db import ChatDB
from .verify.grounding import GroundingReport, GroundingVerifier


@dataclass(slots=True)
class AskResult:
    """Answer plus everything needed to display and audit it."""

    answer: Answer
    retrieval: RetrievalResult
    grounding: GroundingReport
    session_id: str = ""
    message_id: int = 0
    proofs: list[ProofImage] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    """Answerable prompts offered when the system declines — never a dead end."""

    @property
    def text(self) -> str:
        return self.answer.text

    def summary(self) -> str:
        used = ", ".join(c.label for c in self.answer.used_citations) or "none"
        return (
            f"{self.answer.text}\n\n"
            f"evidence: {used}\n"
            f"groundedness: {self.answer.groundedness:.2f} ({self.answer.confidence_band()})"
        )


class VeriRAG:
    """Facade over ingestion, retrieval, generation, verification and storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLM | None = None,
        *,
        autoload: bool = True,
        enable_history: bool = True,
        probe_llm: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.indexer = Indexer(self.settings)
        if autoload:
            self.indexer.load()

        self.llm = llm if llm is not None else get_llm(self.settings, probe=probe_llm)
        self.retriever = HybridRetriever(self.indexer, self.settings)
        self._calibrate_thresholds()
        self.answerer = Answerer(self.llm, self.settings)
        self.verifier = GroundingVerifier(self.indexer.lines, self.settings)
        self.renderer = ProofRenderer(self.settings)
        self.chat: ChatDB | None = ChatDB(self.settings.chat_db_path) if enable_history else None

    def _calibrate_thresholds(self) -> None:
        """Adopt the active reranker's score thresholds.

        Reranker scores are not on a comparable scale — the lexical reranker is
        roughly continuous while a cross-encoder's sigmoid output is bimodal — so
        a single hardcoded refusal gate is wrong for whichever one it was not
        tuned against. The thresholds therefore travel with the scorer, and an
        explicit environment override always wins.
        """
        reranker = self.retriever.reranker
        if not self.settings.is_explicitly_set("VERIRAG_MIN_RETRIEVAL_SCORE"):
            self.settings.min_retrieval_score = getattr(
                reranker, "refusal_threshold", self.settings.min_retrieval_score
            )
        if not self.settings.is_explicitly_set("VERIRAG_LOW_CONFIDENCE_SCORE"):
            self.settings.low_confidence_score = getattr(
                reranker, "low_confidence_threshold", self.settings.low_confidence_score
            )

    # -------------------------------------------------------------- ingestion
    def ingest(self, target: str | Path, *, force: bool = False) -> list[IngestReport]:
        """Ingest a PDF file or every PDF in a directory, then persist indexes."""
        path = Path(target)
        if path.is_dir():
            reports = self.indexer.ingest_directory(path, force=force)
        else:
            reports = [self.indexer.ingest_pdf(path, force=force)]
        if any(report.ok and not report.skipped for report in reports):
            self.indexer.save()
        return reports

    def documents(self) -> list[Document]:
        return self.indexer.registry.all()

    def document_path(self, doc_id: str) -> Path | None:
        document = self.indexer.registry.get(doc_id)
        if document is None:
            return None
        path = Path(document.path)
        if path.exists():
            return path
        # PDFs may have been moved after ingestion; look in the known folders.
        for directory in (self.settings.raw_dir, self.settings.upload_dir):
            candidate = directory / document.name
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------ asking
    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        use_history: bool = True,
        render_proof: bool = True,
        persist: bool = True,
    ) -> AskResult:
        """Answer *question* with verified, span-level evidence.

        The question is first classified. "Explain this document" and "what topics
        are covered" are *global* requests: no single passage resembles them, so
        sending them down the similarity path guarantees a refusal on the most
        common question a new user asks. They get a representative spread of the
        document instead.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        intent = classify(question)
        if intent is Intent.TOPICS:
            return self._answer_topics(question, doc_ids=doc_ids, session_id=session_id, persist=persist)
        if intent is Intent.SUMMARY:
            return self._answer_summary(
                question,
                doc_ids=doc_ids,
                session_id=session_id,
                persist=persist,
                render_proof=render_proof,
            )

        history: list[tuple[str, str]] = []
        if self.chat is not None and session_id and use_history:
            history = self.chat.history_pairs(session_id, limit=3)

        retrieval = self.retriever.retrieve(question, top_k=top_k, doc_ids=doc_ids)
        answer = self.answerer.answer(retrieval, history=history)
        grounding = self.verifier.verify(answer)

        proofs: list[ProofImage] = []
        if render_proof and not answer.refused:
            proofs = self.render_proofs(answer)

        message_id = 0
        if persist and self.chat is not None and session_id:
            self.chat.add_user_message(session_id, question)
            message_id = self.chat.add_answer(session_id, answer)

        return AskResult(
            answer=answer,
            retrieval=retrieval,
            grounding=grounding,
            session_id=session_id or "",
            message_id=message_id,
            proofs=proofs,
            suggestions=self.suggestions(doc_ids) if answer.refused else [],
        )

    # ------------------------------------------------------- global requests
    def _target_documents(self, doc_ids: Sequence[str] | None) -> list[Document]:
        """Which documents a whole-document request refers to.

        With an explicit filter, that. With one document indexed, that one. With
        several and no filter, the most recently ingested — "this document" almost
        always means the one just uploaded — and the answer says which it chose so
        the guess is visible rather than silent.
        """
        documents = self.documents()
        if not documents:
            return []
        if doc_ids:
            allowed = set(doc_ids)
            chosen = [d for d in documents if d.doc_id in allowed]
            if chosen:
                return chosen
        if len(documents) == 1:
            return documents
        return [max(documents, key=lambda d: d.ingested_at)]

    def _representative_chunks(self, doc_id: str, budget: int = 14) -> list:
        """A spread across the document, not the top-k most similar to anything.

        One chunk per topic in document order, so a summary covers the whole file
        rather than over-sampling whichever section happens to be longest.
        """
        chunks = [c for c in self.indexer.vectors.all_chunks() if c.doc_id == doc_id]
        if not chunks:
            return []
        chunks.sort(key=lambda c: (c.page_no, c.line_start))
        if len(chunks) <= budget:
            return chunks

        from .study import extract_topics

        picked: list = []
        by_id = {c.chunk_id: c for c in chunks}
        for topic in extract_topics(chunks):
            for chunk_id in topic.chunk_ids[:1]:
                if chunk_id in by_id:
                    picked.append(by_id[chunk_id])
        # Top up with an even stride so long documents are still covered.
        if len(picked) < budget:
            stride = max(len(chunks) // budget, 1)
            for chunk in chunks[::stride]:
                if chunk not in picked:
                    picked.append(chunk)
                if len(picked) >= budget:
                    break
        picked.sort(key=lambda c: (c.page_no, c.line_start))
        return picked[:budget]

    def _answer_summary(
        self,
        question: str,
        *,
        doc_ids: Sequence[str] | None,
        session_id: str | None,
        persist: bool,
        render_proof: bool,
    ) -> AskResult:
        targets = self._target_documents(doc_ids)
        if not targets:
            return self._empty_result(question, session_id, persist)

        topic = topic_of_interest(question)
        document = targets[0]

        if topic:
            # "summarise clause 3" is a scoped request: retrieve for the topic.
            retrieval = self.retriever.retrieve(
                topic, top_k=max(self.settings.top_k_final, 8), doc_ids=[document.doc_id]
            )
            items = retrieval.chunks
        else:
            chunks = self._representative_chunks(document.doc_id)
            items = [RetrievedChunk(chunk=chunk, score=1.0) for chunk in chunks]
            retrieval = RetrievalResult(query=question, variants=[question], chunks=items)

        if not items:
            return self._empty_result(question, session_id, persist)

        answer = self.answerer.summarise(question, retrieval, document.name)
        if len(targets) == 1 and len(self.documents()) > 1 and not doc_ids:
            answer.text = f"Summarising **{document.name}**, the most recently added document.\n\n{answer.text}"

        grounding = self.verifier.verify(answer)
        proofs = self.render_proofs(answer) if render_proof and not answer.refused else []

        message_id = 0
        if persist and self.chat is not None and session_id:
            self.chat.add_user_message(session_id, question)
            message_id = self.chat.add_answer(session_id, answer)

        return AskResult(
            answer=answer,
            retrieval=retrieval,
            grounding=grounding,
            session_id=session_id or "",
            message_id=message_id,
            proofs=proofs,
        )

    def _answer_topics(
        self,
        question: str,
        *,
        doc_ids: Sequence[str] | None,
        session_id: str | None,
        persist: bool,
    ) -> AskResult:
        """Answer "what's in here?" from document structure, not retrieval."""
        targets = self._target_documents(doc_ids)
        if not targets:
            return self._empty_result(question, session_id, persist)

        document = targets[0]
        topics = self.topics(document.doc_id)
        if not topics:
            return self._empty_result(question, session_id, persist)

        chunk_by_id = {c.chunk_id: c for c in self.indexer.vectors.all_chunks()}
        lines = [f"{document.name} covers {len(topics)} sections across {document.n_pages} page(s):"]
        citations: list[Citation] = []
        for index, entry in enumerate(topics, start=1):
            marker = f"S{index}"
            lines.append(f"- {entry.name} ({entry.page_range}) [{marker}]")
            source = next((chunk_by_id[c] for c in entry.chunk_ids if c in chunk_by_id), None)
            if source is None:
                continue
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=source.doc_id,
                    doc_name=source.doc_name,
                    page_no=source.page_no,
                    line_start=source.line_start,
                    line_end=source.line_end,
                    quote=source.text,
                    retrieval_score=1.0,
                    used_in_answer=True,
                    bboxes=list(source.line_bboxes),
                )
            )

        answer = Answer(
            question=question,
            text="\n".join(lines),
            citations=citations,
            groundedness=1.0,
            provider="structure",
            retrieval_score=1.0,
        )
        retrieval = RetrievalResult(query=question, variants=[question], chunks=[])

        message_id = 0
        if persist and self.chat is not None and session_id:
            self.chat.add_user_message(session_id, question)
            message_id = self.chat.add_answer(session_id, answer)

        return AskResult(
            answer=answer,
            retrieval=retrieval,
            grounding=GroundingReport(verdicts=[], groundedness=1.0, supported=0, total=0),
            session_id=session_id or "",
            message_id=message_id,
        )

    def _empty_result(self, question: str, session_id: str | None, persist: bool) -> AskResult:
        answer = Answer(
            question=question,
            text=(
                "There is nothing indexed to summarise yet. Upload a PDF in the Documents "
                "tab and index it, then ask again."
            ),
            refused=True,
            provider="structure",
        )
        message_id = 0
        if persist and self.chat is not None and session_id:
            self.chat.add_user_message(session_id, question)
            message_id = self.chat.add_answer(session_id, answer)
        return AskResult(
            answer=answer,
            retrieval=RetrievalResult(query=question, variants=[], chunks=[]),
            grounding=GroundingReport(verdicts=[], groundedness=0.0, supported=0, total=0),
            session_id=session_id or "",
            message_id=message_id,
            suggestions=self.suggestions(None),
        )

    # -------------------------------------------------------------- wayfinding
    def suggestions(self, doc_ids: Sequence[str] | None = None, limit: int = 5) -> list[str]:
        """Questions this corpus can actually answer.

        A bare refusal is a dead end. Every screen should tell the user where they
        can go next, so a refusal ships with concrete, answerable prompts drawn
        from the indexed material's own section headings.
        """
        targets = self._target_documents(doc_ids) or self.documents()
        prompts: list[str] = []
        for document in targets[:2]:
            for entry in self.topics(document.doc_id):
                if len(prompts) >= limit:
                    break
                name = entry.name.strip().rstrip(".")
                if len(name) < 4:
                    continue
                prompts.append(f"What does the section on {name} say?")
        if targets:
            prompts.insert(0, f"Summarise {targets[0].name}")
        return prompts[:limit]

    # -------------------------------------------------------------------- proof
    def render_proofs(self, answer: Answer, *, only_used: bool = True, dpi: int | None = None) -> list[ProofImage]:
        """Render one highlighted page image per cited (document, page)."""
        grouped: dict[tuple[str, int], list[HighlightSpan]] = {}
        for citation in answer.citations:
            if only_used and not citation.used_in_answer:
                continue
            if not citation.bboxes:
                continue
            grouped.setdefault((citation.doc_id, citation.page_no), []).append(
                HighlightSpan(
                    bboxes=list(citation.bboxes),
                    label=f"[{citation.marker}] {citation.doc_name} {citation.locator}",
                    proven=True,
                    line_start=citation.line_start,
                    line_end=citation.line_end,
                )
            )

        images: list[ProofImage] = []
        for (doc_id, page_no), spans in grouped.items():
            pdf_path = self.document_path(doc_id)
            if pdf_path is None:
                continue
            image = self.renderer.render(pdf_path, page_no, spans, dpi=dpi)
            if image is not None:
                images.append(image)
        return images

    def render_citation(self, citation, *, dpi: int | None = None) -> ProofImage | None:
        """Render the page of a single citation with its proven lines highlighted."""
        pdf_path = self.document_path(citation.doc_id)
        if pdf_path is None:
            return None
        return self.renderer.render_citation(pdf_path, citation, dpi=dpi)

    def crop_citation(self, citation, *, dpi: int | None = None) -> bytes | None:
        """Tight PNG crop of a single citation's proven lines."""
        pdf_path = self.document_path(citation.doc_id)
        if pdf_path is None or not citation.bboxes:
            return None
        return self.renderer.crop(pdf_path, citation.page_no, list(citation.bboxes), dpi=dpi)

    # ----------------------------------------------------------------- documents
    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all of its chunks from the index."""
        return self.indexer.delete_document(doc_id)

    # ----------------------------------------------------------------- sessions
    def new_session(self, title: str = "New chat", doc_filter: Sequence[str] = ()) -> str:
        if self.chat is None:
            raise RuntimeError("history is disabled on this engine instance")
        return self.chat.create_session(title, doc_filter)

    # -------------------------------------------------------------- study mode
    def _chunks_for(self, doc_id: str) -> list:
        return [c for c in self.indexer.vectors.all_chunks() if c.doc_id == doc_id]

    def topics(self, doc_id: str) -> list:
        """Discover the study topics of an indexed document."""
        from .study import extract_topics

        return extract_topics(self._chunks_for(doc_id))

    def study_pack(
        self,
        doc_id: str,
        *,
        n_mcqs: int = 8,
        n_short: int = 4,
        n_cards: int = 8,
        seed: int | None = None,
    ):
        """Generate a verified study pack (topics, MCQs, questions, flashcards)."""
        from .study import StudyGenerator

        chunks = self._chunks_for(doc_id)
        if not chunks:
            raise KeyError(f"no indexed chunks for doc_id {doc_id!r}")
        generator = StudyGenerator(self.llm, self.indexer.lines, self.settings, seed=seed)
        return generator.build_pack(chunks, n_mcqs=n_mcqs, n_short=n_short, n_cards=n_cards)

    def generate_mcqs(self, doc_id: str, count: int = 8, *, seed: int | None = None):
        """Generate only MCQs, each with a verified answer key."""
        from .study import StudyGenerator

        chunks = self._chunks_for(doc_id)
        if not chunks:
            raise KeyError(f"no indexed chunks for doc_id {doc_id!r}")
        generator = StudyGenerator(self.llm, self.indexer.lines, self.settings, seed=seed)
        return generator.generate_mcqs(chunks, count)

    def explain(
        self,
        topic: str,
        *,
        doc_ids: Sequence[str] | None = None,
        level: str = "exam",
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> AskResult:
        """Tutor-style explanation of a topic, grounded and cited.

        Uses the same retrieve -> generate -> verify path as :meth:`ask`, so an
        explanation is as auditable as any other answer.
        """
        level = level if level in {"beginner", "exam"} else "exam"
        question = (
            f"Explain the topic '{topic}' for a student preparing an exam."
            if level == "exam"
            else f"Explain the topic '{topic}' simply, for a beginner, defining any jargon."
        )
        return self.ask(
            question,
            session_id=session_id,
            doc_ids=doc_ids,
            top_k=top_k or max(self.settings.top_k_final, 6),
            use_history=False,
            render_proof=False,
        )

    # -------------------------------------------------------------------- info
    def stats(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.indexer.stats(),
            "llm": describe_provider(self.llm),
            "reranker": self.retriever.reranker.name,
            "grounding_threshold": self.settings.grounding_threshold,
        }
        if self.chat is not None:
            payload["history"] = self.chat.stats()
        return payload

    def is_empty(self) -> bool:
        return len(self.indexer.vectors) == 0
