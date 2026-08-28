"""VeriRAG Streamlit UI.

    streamlit run app.py

Four tabs:

* **Chat** — ask questions, see the answer with per-sentence groundedness,
  expandable citation cards, and the source page rendered with the cited lines
  highlighted. History is loaded from and written to SQLite, so refreshing the
  browser or restarting the app does not lose anything.
* **Documents** — upload PDFs, index them, inspect page/line/chunk counts.
* **History** — full-text search across every past conversation, export a
  session as Markdown or JSON.
* **Evaluation** — run the golden-set harness in-app and read the metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from verirag.config import get_settings  # noqa: E402
from verirag.engine import VeriRAG  # noqa: E402
from verirag.models import Answer, Citation  # noqa: E402
from verirag.store.chat_db import StoredMessage  # noqa: E402
from verirag.ui import theme  # noqa: E402

st.set_page_config(
    page_title="VeriRAG",
    page_icon="\u25c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The stylesheet goes in before anything else renders, so no element paints
# unstyled first.
theme.inject(st)


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent / "src" / "verirag"


def source_fingerprint() -> str:
    """Fingerprint of the package source, used as part of the cache key.

    Streamlit caches the engine object with ``st.cache_resource``, and it does not
    re-import an already-loaded package. Editing ``src/verirag`` while the server
    runs therefore leaves a cached engine built from the *previous* class
    definition, which surfaces as a baffling ``AttributeError`` for a method you
    can plainly see in the file. Folding the newest source mtime into the cache
    key makes the engine rebuild itself instead.
    """
    try:
        newest = max(path.stat().st_mtime_ns for path in _SRC_ROOT.rglob("*.py"))
    except ValueError:  # pragma: no cover - source tree missing (installed wheel)
        return "static"
    return str(newest)


@st.cache_resource(show_spinner="Loading index…")
def load_engine(provider: str, _fingerprint: str) -> VeriRAG:
    settings = get_settings(refresh=True)
    if provider:
        settings.llm_provider = provider
    return VeriRAG(settings)


def reset_engine() -> None:
    load_engine.clear()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_citation_card(engine: VeriRAG, citation: Citation, index: int) -> None:
    header = f"[{citation.marker}]  {citation.doc_name}  ·  page {citation.page_no}, lines {citation.line_start}-{citation.line_end}"
    with st.expander(header, expanded=index == 0):
        st.markdown(theme.quote_block(citation.quote, proven=True), unsafe_allow_html=True)
        st.markdown(
            f"{theme.locator_chip(citation.locator)} "
            f'<span style="color:var(--text-faint);font-size:0.78rem;">'
            f"retrieval score {citation.retrieval_score:.4f}</span>",
            unsafe_allow_html=True,
        )

        image = engine.render_citation(citation)
        if image is not None:
            st.image(
                image.png,
                caption=f"{citation.doc_name} — page {citation.page_no}, cited lines highlighted",
                use_column_width=True,
            )
        else:
            st.info("Source PDF is no longer at its indexed path, so the page cannot be rendered.")


def render_answer(engine: VeriRAG, answer: Answer, *, show_trace: bool) -> None:
    st.markdown(answer.text)
    st.markdown(
        theme.confidence_pill(answer.confidence_band(), answer.groundedness, answer.retrieval_score),
        unsafe_allow_html=True,
    )
    st.caption(f"{answer.provider or 'extractive'} · {answer.latency_ms} ms")

    if answer.provider_error:
        st.error(
            f"**The configured LLM was not used.** {answer.provider_error[:300]}\n\n"
            "The answer above came from the extractive composer instead. "
            "Check the model name and API key in `.env`."
        )

    if answer.refused:
        return

    if answer.weak_evidence:
        if answer.groundedness >= 0.75:
            st.info(
                f"The answer is well supported by the cited lines, but the best passage only scored "
                f"{answer.retrieval_score:.2f} against your question — confirm it is the clause you meant."
            )
        else:
            st.warning(
                f"Weak evidence: the best retrieved passage scored {answer.retrieval_score:.2f}. "
                "The quoted text may not actually address your question — check the source below."
            )

    unsupported = answer.unsupported_claims
    if unsupported:
        with st.expander(f"{len(unsupported)} sentence(s) could not be verified", expanded=False):
            for verdict in unsupported:
                st.markdown(f"- {verdict.sentence}  \n  _support score {verdict.score:.2f}_")

    used = answer.used_citations or answer.citations
    if used:
        st.markdown("**Proof — where this came from**")
        for index, citation in enumerate(used):
            render_citation_card(engine, citation, index)

    if show_trace and answer.retrieval_trace:
        with st.expander("Retrieval trace (why these passages were chosen)"):
            st.dataframe(answer.retrieval_trace, use_container_width=True, hide_index=True)


def message_to_answer(message: StoredMessage) -> Answer:
    """Rebuild an Answer from a stored row so history renders identically."""
    return Answer(
        question="",
        text=message.content,
        citations=message.citations,
        verdicts=message.verdicts,
        groundedness=message.groundedness,
        provider=message.provider,
        model=message.model,
        latency_ms=message.latency_ms,
        refused=message.refused,
        weak_evidence=message.weak_evidence,
        retrieval_score=message.retrieval_score,
        retrieval_trace=message.retrieval_trace,
    )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def sidebar() -> tuple[VeriRAG, str, bool]:
    st.sidebar.title("VeriRAG")
    st.sidebar.caption("Answers with page + line level proof")

    provider = st.sidebar.selectbox(
        "LLM provider",
        ["auto", "extractive", "groq", "gemini", "ollama"],
        index=0,
        help=(
            "auto tries Groq → Gemini → Ollama → extractive. "
            "extractive needs no API key at all and composes the answer from retrieved lines."
        ),
    )
    engine = load_engine(provider, source_fingerprint())

    stats = engine.stats()
    st.sidebar.markdown(
        f"**Index**  \n{stats['documents']} documents · {stats['chunks']} chunks  \n"
        f"embedder `{stats['embedder']}` · reranker `{stats['reranker']}`  \n"
        f"LLM `{stats['llm']}`"
    )
    if st.sidebar.button("Reload engine", help="Rebuild after changing code or configuration",
                         use_container_width=True):
        reset_engine()
        st.rerun()

    if engine.is_empty():
        st.sidebar.warning("No documents indexed yet.")
        if st.sidebar.button("Index the bundled sample PDFs", use_container_width=True):
            with st.spinner("Indexing sample corpus…"):
                engine.ingest(engine.settings.raw_dir)
            reset_engine()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Sessions")
    assert engine.chat is not None
    sessions = engine.chat.list_sessions(limit=50)
    labels = {s.id: f"{s.title[:34]}  ({s.n_messages})" for s in sessions}

    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state["session_id"] = engine.new_session()
        st.rerun()

    if sessions:
        ids = [s.id for s in sessions]
        current = st.session_state.get("session_id")
        default = ids.index(current) if current in ids else 0
        chosen = st.sidebar.radio(
            "Open a session",
            ids,
            index=default,
            format_func=lambda i: labels.get(i, i),
            label_visibility="collapsed",
        )
        st.session_state["session_id"] = chosen
    elif "session_id" not in st.session_state:
        st.session_state["session_id"] = engine.new_session()

    session_id = st.session_state["session_id"]

    st.sidebar.divider()
    show_trace = st.sidebar.toggle("Show retrieval trace", value=False)

    documents = engine.documents()
    if documents:
        st.sidebar.subheader("Restrict to documents")
        selected = st.sidebar.multiselect(
            "Search only these",
            [d.doc_id for d in documents],
            format_func=lambda i: next((d.name for d in documents if d.doc_id == i), i),
            label_visibility="collapsed",
        )
        st.session_state["doc_filter"] = selected

    if sessions and st.sidebar.button("Delete this session", use_container_width=True):
        engine.chat.delete_session(session_id)
        st.session_state.pop("session_id", None)
        st.rerun()

    return engine, session_id, show_trace


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
def tab_chat(engine: VeriRAG, session_id: str, show_trace: bool) -> None:
    assert engine.chat is not None

    if engine.is_empty():
        st.info("Index some documents first — use the sidebar button or the Documents tab.")
        return

    for message in engine.chat.get_messages(session_id):
        with st.chat_message("user" if message.role == "user" else "assistant"):
            if message.role == "user":
                st.markdown(message.content)
            else:
                render_answer(engine, message_to_answer(message), show_trace=show_trace)

    question = st.chat_input("Ask about the indexed documents…")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Retrieving, answering and verifying…"):
        result = engine.ask(
            question,
            session_id=session_id,
            doc_ids=st.session_state.get("doc_filter") or None,
            render_proof=False,  # rendered lazily per citation card
        )
        render_answer(engine, result.answer, show_trace=show_trace)


def tab_documents(engine: VeriRAG) -> None:
    st.subheader("Indexed documents")
    documents = engine.documents()
    if documents:
        st.dataframe(
            [
                {
                    "document": d.name,
                    "pages": d.n_pages,
                    "lines": d.n_lines,
                    "chunks": d.n_chunks,
                    "ingested": d.ingested_at,
                    "doc_id": d.doc_id,
                }
                for d in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Nothing indexed yet.")

    st.divider()
    st.subheader("Add PDFs")
    uploads = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary"):
        engine.settings.ensure_dirs()
        progress = st.progress(0.0)
        for position, upload in enumerate(uploads, start=1):
            target = engine.settings.upload_dir / upload.name
            target.write_bytes(upload.getbuffer())
            report = engine.ingest(target)[0]
            if report.error:
                st.error(f"{upload.name}: {report.error}")
            elif report.skipped:
                st.info(f"{upload.name}: already indexed ({report.chunks} chunks)")
            else:
                st.success(f"{upload.name}: {report.chunks} chunks from {report.lines} lines")
            progress.progress(position / len(uploads))
        reset_engine()

    if documents:
        st.divider()
        victim = st.selectbox(
            "Remove a document from the index",
            [d.doc_id for d in documents],
            format_func=lambda i: next((d.name for d in documents if d.doc_id == i), i),
        )
        if st.button("Remove", type="secondary"):
            engine.delete_document(victim)
            reset_engine()
            st.rerun()


def tab_study(engine: VeriRAG) -> None:
    st.subheader("Study mode")
    st.caption(
        "Turn study material into practice questions. Every answer key is checked against "
        "the source PDF before it is shown — items whose answer cannot be traced back are discarded."
    )

    documents = engine.documents()
    if not documents:
        st.info("Upload and index a PDF in the Documents tab first.")
        return

    doc_id = st.selectbox(
        "Study material",
        [d.doc_id for d in documents],
        format_func=lambda i: next((d.name for d in documents if d.doc_id == i), i),
        key="study_doc",
    )

    topics = engine.topics(doc_id)
    if topics:
        with st.expander(f"{len(topics)} topics found in this document", expanded=False):
            st.dataframe(
                [
                    {"pages": t.page_range, "topic": t.name, "words": t.word_count,
                     "key terms": ", ".join(t.key_terms[:5])}
                    for t in topics
                ],
                use_container_width=True,
                hide_index=True,
            )

    quiz_tab, cards_tab, tutor_tab = st.tabs(["Practice quiz", "Flashcards", "Ask a doubt"])

    # ------------------------------------------------------------------ quiz
    with quiz_tab:
        left, middle, right = st.columns([1, 1, 2])
        n_mcqs = left.number_input("Questions", min_value=1, max_value=20, value=5, key="study_n")
        n_short = middle.number_input("Short answer", min_value=0, max_value=10, value=2, key="study_s")
        right.write("")
        if right.button("Generate quiz", type="primary", use_container_width=True):
            with st.spinner("Generating and verifying against the source…"):
                pack = engine.study_pack(doc_id, n_mcqs=int(n_mcqs), n_short=int(n_short), n_cards=6)
            st.session_state["study_pack"] = pack
            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_answers"] = {}

        pack = st.session_state.get("study_pack")
        if pack is None or pack.doc_id != doc_id:
            st.info("Choose how many questions you want and press **Generate quiz**.")
            return

        stats = pack.stats()
        st.caption(
            f"generator: `{stats['generator']}` · {stats['mcqs_verified']}/{stats['mcqs']} MCQs verified "
            f"against the PDF · {stats['rejected_unverifiable']} discarded as unverifiable"
        )

        if not pack.mcqs:
            st.warning(
                "No verifiable questions could be generated from this document. It may be too short, "
                "or mostly tables and figures rather than prose."
            )
            return

        submitted = st.session_state.get("quiz_submitted", False)

        with st.form("quiz_form"):
            for index, question in enumerate(pack.mcqs):
                st.markdown(f"**Q{index + 1}.** {question.question}")
                st.radio(
                    f"q{index}",
                    options=list(range(len(question.options))),
                    format_func=lambda position, q=question: f"{chr(65 + position)}. {q.options[position]}",
                    key=f"quiz_choice_{index}",
                    index=None,
                    label_visibility="collapsed",
                )
                st.divider()
            do_submit = st.form_submit_button("Submit answers", type="primary")

        if do_submit:
            st.session_state["quiz_submitted"] = True
            submitted = True

        if not submitted:
            return

        correct = 0
        for index, question in enumerate(pack.mcqs):
            choice = st.session_state.get(f"quiz_choice_{index}")
            is_right = choice == question.correct_index
            correct += int(is_right)

            if choice is None:
                st.warning(f"**Q{index + 1}** — not answered. "
                           f"Correct answer: **{question.correct_letter}. {question.correct_option}**")
            elif is_right:
                st.success(f"**Q{index + 1}** — correct: **{question.correct_letter}. {question.correct_option}**")
            else:
                st.error(
                    f"**Q{index + 1}** — you chose **{chr(65 + choice)}**. "
                    f"Correct answer: **{question.correct_letter}. {question.correct_option}**"
                )

            if question.explanation:
                st.caption(question.explanation)
            if question.citation is not None:
                with st.expander(f"Proof — {question.citation.doc_name} {question.citation.locator}"):
                    st.markdown(
                        theme.quote_block(question.citation.quote, proven=True, limit=900),
                        unsafe_allow_html=True,
                    )
                    image = engine.render_citation(question.citation)
                    if image is not None:
                        st.image(image.png, use_column_width=True,
                                 caption=f"page {question.citation.page_no}, answer lines highlighted")

        total = len(pack.mcqs)
        percentage = correct / total if total else 0.0
        st.metric("Score", f"{correct} / {total}", f"{percentage:.0%}")
        if percentage == 1.0:
            st.balloons()

        if pack.short_answers:
            st.divider()
            st.markdown("### Short-answer practice")
            for index, question in enumerate(pack.short_answers, start=1):
                with st.expander(f"{index}. ({question.marks} marks) {question.question}"):
                    st.write(question.answer)
                    if question.citation is not None:
                        st.caption(f"source: {question.citation.doc_name} {question.citation.locator}")

    # ------------------------------------------------------------- flashcards
    with cards_tab:
        pack = st.session_state.get("study_pack")
        if pack is None or not pack.flashcards:
            st.info("Generate a quiz first — flashcards are produced alongside it.")
        else:
            for card in pack.flashcards:
                with st.expander(card.front):
                    st.write(card.back)
                    if card.citation is not None:
                        st.caption(f"source: {card.citation.doc_name} {card.citation.locator}")

    # ------------------------------------------------------------------ tutor
    with tutor_tab:
        st.write("Stuck on a topic? Get an explanation built only from this document.")
        suggestions = [t.name for t in topics[:12]]
        chosen = st.selectbox("Pick a topic", ["(type my own)"] + suggestions, key="tutor_topic")
        custom = st.text_input("Or describe your doubt", key="tutor_custom")
        level = st.radio("Explain for", ["beginner", "exam"], horizontal=True, key="tutor_level")

        topic = custom.strip() or (chosen if chosen != "(type my own)" else "")
        if st.button("Explain", type="primary") and topic:
            with st.spinner("Reading the document…"):
                result = engine.explain(topic, doc_ids=[doc_id], level=level)
            render_answer(engine, result.answer, show_trace=False)
        elif st.button("Explain", key="explain_noop", disabled=True):
            pass


def tab_history(engine: VeriRAG, session_id: str) -> None:
    assert engine.chat is not None
    st.subheader("Search all conversations")
    query = st.text_input("Search stored questions and answers", placeholder="e.g. security deposit")
    if query:
        hits = engine.chat.search(query, limit=40)
        if hits:
            st.dataframe(
                [
                    {
                        "session": h["title"][:40],
                        "role": h["role"],
                        "content": " ".join(h["content"].split())[:160],
                        "when": h["created_at"][:19],
                    }
                    for h in hits
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No matches.")

    st.divider()
    st.subheader("Export current session")
    left, right = st.columns(2)
    try:
        markdown = engine.chat.export_markdown(session_id)
        payload = json.dumps(engine.chat.export_session(session_id), indent=2)
    except KeyError:
        st.info("This session has no messages yet.")
        return
    left.download_button("Download Markdown", markdown, file_name=f"verirag_{session_id}.md", use_container_width=True)
    right.download_button("Download JSON", payload, file_name=f"verirag_{session_id}.json", use_container_width=True)

    with st.expander("Storage statistics"):
        st.json(engine.chat.stats())


def tab_eval(engine: VeriRAG) -> None:
    st.subheader("Evaluation harness")
    st.caption(
        "Grades retrieval, answer accuracy, citation correctness and refusal behaviour against a "
        "golden question set whose relevance labels are verbatim phrases from the sample PDFs."
    )
    if engine.is_empty():
        st.info("Index the sample corpus first.")
        return
    if not st.button("Run evaluation", type="primary"):
        return

    from verirag.eval import run_eval

    with st.spinner("Running the golden set…"):
        report = run_eval(engine, strict=False)
    payload = report.to_dict()

    columns = st.columns(4)
    retrieval = payload["retrieval"]
    answers = payload["answers"]
    columns[0].metric("Recall@5", f"{retrieval.get('recall@5', 0):.1%}")
    columns[1].metric("MRR", f"{retrieval.get('mrr', 0):.3f}")
    columns[2].metric("Answer accuracy", f"{answers.get('answer_accuracy', 0):.1%}")
    columns[3].metric("Groundedness", f"{answers.get('mean_groundedness', 0):.1%}")

    st.code(report.render(), language="text")
    st.download_button("Download report JSON", json.dumps(payload, indent=2), file_name="verirag_eval.json")


# ---------------------------------------------------------------------------
def main() -> None:
    engine, session_id, show_trace = sidebar()

    st.markdown(
        theme.hero(
            "Answers you can check.",
            "Ask questions about your PDFs and get answers that cite the exact document, page "
            "and line range \u2014 with the source page rendered and the cited lines highlighted.",
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    chat, documents, study, history, evaluation = st.tabs(
        ["Chat", "Documents", "Study mode", "History", "Evaluation"]
    )
    with chat:
        tab_chat(engine, session_id, show_trace)
    with documents:
        tab_documents(engine)
    with study:
        tab_study(engine)
    with history:
        tab_history(engine, session_id)
    with evaluation:
        tab_eval(engine)


if __name__ == "__main__":
    main()
