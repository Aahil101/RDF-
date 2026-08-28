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
from .models import Answer, Document
from .proof.highlighter import HighlightSpan, ProofImage, ProofRenderer
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
        """Answer *question* with verified, span-level evidence."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

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
        )

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
