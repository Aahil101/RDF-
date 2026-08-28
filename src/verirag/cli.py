"""Command line interface.

    verirag demo                     one command: build corpus, index, ask, prove
    verirag ingest data/raw          index a PDF or a folder of PDFs
    verirag ask "what is the rent?"  answer with page + line proof
    verirag chat                     interactive session, history persisted
    verirag topics --doc notes       list the topics found in study material
    verirag quiz --doc notes -n 10   generate MCQs with a source-verified key
    verirag explain "BCNF"           tutor-style explanation, cited
    verirag sessions                 list stored chat sessions
    verirag history <id> --markdown  export a session with its evidence
    verirag search "deposit"         full-text search across all chat history
    verirag eval                     grade retrieval, answers and refusals
    verirag stats                    index and history statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .engine import VeriRAG
from .models import Answer

# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------
_RULE = "-" * 78


def _configure_streams() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page (cp1252), where printing a
    single em-dash raises UnicodeEncodeError and kills the command mid-report.
    Replacing unencodable characters is far better than losing the output.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-TTY streams
            pass


def _print_answer(answer: Answer, *, show_trace: bool = False) -> None:
    print(f"\n{answer.text}\n")

    if answer.provider_error:
        print(f"  PROVIDER FELL BACK: {answer.provider_error[:200]}")
        print("  The answer above came from the extractive composer, not the LLM.\n")

    if answer.refused:
        print("  no grounded evidence found - refused rather than guessing")
        return

    print("EVIDENCE")
    for citation in answer.citations:
        flag = "*" if citation.used_in_answer else " "
        quote = " ".join(citation.quote.split())[:110]
        print(f" {flag}[{citation.marker}] {citation.doc_name}  {citation.locator}  (score {citation.retrieval_score:.3f})")
        print(f'      "{quote}"')

    band = answer.confidence_band()
    print(f"\nGROUNDEDNESS  {answer.groundedness:.2f}  ({band})   provider: {answer.provider}"
          f"   {answer.latency_ms} ms")

    if answer.weak_evidence:
        if answer.groundedness >= 0.75:
            print(f"  NOTE: the answer is well supported by the cited lines, but the best passage "
                  f"only scored {answer.retrieval_score:.2f} against your question - confirm it is "
                  "the clause you meant.")
        else:
            print(f"  WEAK EVIDENCE: best retrieval score {answer.retrieval_score:.3f} is low. "
                  "The quoted text may not address the question - verify against the source.")

    unsupported = answer.unsupported_claims
    if unsupported:
        print(f"\n  {len(unsupported)} sentence(s) not verifiably supported:")
        for verdict in unsupported:
            print(f"   - \"{verdict.sentence[:90]}\"  (score {verdict.score:.2f})")

    if show_trace:
        print("\nRETRIEVAL TRACE")
        for row in answer.retrieval_trace:
            print(f"   {row['doc'][:34]:<34} {row['locator']:<13} {row['score']:.4f}  {row['why']}")


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    if not rows:
        print("  (none)")
        return
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) for i in range(len(headers))]
    print("  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def _engine(args: argparse.Namespace) -> VeriRAG:
    settings = get_settings()
    if getattr(args, "provider", None):
        settings.llm_provider = args.provider
    if getattr(args, "k", None):
        settings.top_k_final = args.k
    return VeriRAG(settings)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if args.reset:
        engine.indexer.reset()
        print("index reset")

    target = Path(args.path) if args.path else engine.settings.raw_dir
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2

    reports = engine.ingest(target, force=args.force)
    rows = []
    for report in reports:
        if report.error:
            rows.append([target.name, "ERROR", "-", report.error[:44]])
        elif report.skipped:
            rows.append([report.document.name, "skipped", str(report.chunks), "already indexed"])
        else:
            rows.append([report.document.name, "indexed", str(report.chunks), f"{report.lines} lines"])
    _print_table(rows, ["document", "status", "chunks", "detail"])

    stats = engine.stats()
    print(f"\n  {stats['documents']} documents · {stats['chunks']} chunks · "
          f"embedder={stats['embedder']} · bm25={stats['bm25_impl']}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if engine.is_empty():
        print("index is empty — run `verirag ingest` first", file=sys.stderr)
        return 2

    session_id = args.session
    result = engine.ask(
        args.question,
        session_id=session_id,
        doc_ids=args.doc or None,
        render_proof=not args.no_proof,
        persist=bool(session_id),
    )

    if args.json:
        print(json.dumps(result.answer.to_dict(), indent=2))
        return 0

    _print_answer(result.answer, show_trace=args.trace)

    if result.proofs and not args.no_proof:
        out_dir = Path(args.save_proof) if args.save_proof else engine.settings.proof_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nVISUAL PROOF (highlighted pages)")
        for image in result.proofs:
            name = f"{Path(image.doc_name).stem}_p{image.page_no}.png"
            path = out_dir / name
            path.write_bytes(image.png)
            print(f"   {path}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if engine.is_empty():
        print("index is empty — run `verirag ingest` first", file=sys.stderr)
        return 2

    session_id = args.session or engine.new_session("CLI chat")
    print(_RULE)
    print(f"VeriRAG chat · session {session_id} · provider {engine.stats()['llm']}")
    print("commands: /exit  /history  /docs  /new  /trace")
    print(_RULE)

    show_trace = False
    while True:
        try:
            question = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            print(f"session saved: {session_id}")
            return 0
        if question == "/new":
            session_id = engine.new_session("CLI chat")
            print(f"new session: {session_id}")
            continue
        if question == "/trace":
            show_trace = not show_trace
            print(f"trace {'on' if show_trace else 'off'}")
            continue
        if question == "/docs":
            _print_table(
                [[d.name, str(d.n_pages), str(d.n_chunks)] for d in engine.documents()],
                ["document", "pages", "chunks"],
            )
            continue
        if question == "/history":
            assert engine.chat is not None
            for message in engine.chat.get_messages(session_id, with_evidence=False):
                who = "you" if message.role == "user" else "verirag"
                print(f"  {who:>8}: {message.content[:120]}")
            continue

        result = engine.ask(question, session_id=session_id, render_proof=False)
        _print_answer(result.answer, show_trace=show_trace)


def cmd_sessions(args: argparse.Namespace) -> int:
    engine = _engine(args)
    assert engine.chat is not None
    sessions = engine.chat.list_sessions()
    _print_table(
        [[s.id, s.title[:46], str(s.n_messages), s.updated_at[:19]] for s in sessions],
        ["id", "title", "msgs", "updated"],
    )
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    engine = _engine(args)
    assert engine.chat is not None
    try:
        if args.markdown:
            print(engine.chat.export_markdown(args.session_id))
        else:
            print(json.dumps(engine.chat.export_session(args.session_id), indent=2))
    except KeyError:
        print(f"unknown session: {args.session_id}", file=sys.stderr)
        return 2
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = _engine(args)
    assert engine.chat is not None
    hits = engine.chat.search(args.query, limit=args.limit)
    _print_table(
        [[h["session_id"], h["role"], " ".join(h["content"].split())[:70]] for h in hits],
        ["session", "role", "content"],
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    engine = _engine(args)
    print(json.dumps(engine.stats(), indent=2, default=str))
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    engine = _engine(args)
    _print_table(
        [[d.name, str(d.n_pages), str(d.n_lines), str(d.n_chunks), d.doc_id] for d in engine.documents()],
        ["document", "pages", "lines", "chunks", "doc_id"],
    )
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    engine = _engine(args)
    doc = _resolve_doc(engine, args.doc)
    if doc is None:
        return 2
    topics = engine.topics(doc.doc_id)
    _print_table(
        [[t.page_range, str(t.word_count), t.name[:46], ", ".join(t.key_terms[:4])] for t in topics],
        ["pages", "words", "topic", "key terms"],
    )
    print(f"\n  {len(topics)} topics in {doc.name}")
    return 0


def cmd_quiz(args: argparse.Namespace) -> int:
    """Generate a quiz from study material, with a source-verified answer key."""
    engine = _engine(args)
    doc = _resolve_doc(engine, args.doc)
    if doc is None:
        return 2

    print(f"generating from {doc.name} ...")
    pack = engine.study_pack(
        doc.doc_id, n_mcqs=args.n, n_short=args.short, n_cards=args.cards, seed=args.seed
    )

    if args.json:
        print(json.dumps(pack.to_dict(), indent=2, default=str))
        return 0

    stats = pack.stats()
    print(f"\n{_RULE}")
    print(f"STUDY PACK - {doc.name}")
    print(f"generator: {stats['generator']}   topics: {stats['topics']}   "
          f"discarded as unverifiable: {stats['rejected_unverifiable']}")
    print(_RULE)

    if pack.mcqs:
        print("\nMULTIPLE CHOICE")
        for index, question in enumerate(pack.mcqs, start=1):
            print(f"\n{index}. [{question.difficulty}] {question.question}")
            for position, option in enumerate(question.options):
                print(f"     {chr(65 + position)}. {option}")
            if not args.hide_answers:
                citation = question.citation
                where = f"{citation.doc_name} {citation.locator}" if citation else "-"
                print(f"     -> {question.correct_letter}. {question.correct_option}")
                print(f"        source: {where}  (verified {question.verification_score:.2f})")
                if question.explanation:
                    print(f"        why: {question.explanation[:180]}")

    if pack.short_answers:
        print(f"\n{_RULE}\nSHORT ANSWER")
        for index, question in enumerate(pack.short_answers, start=1):
            print(f"\n{index}. ({question.marks} marks) {question.question}")
            if not args.hide_answers:
                citation = question.citation
                print(f"     answer: {question.answer[:300]}")
                print(f"     source: {citation.doc_name} {citation.locator}" if citation else "     source: -")

    if pack.flashcards:
        print(f"\n{_RULE}\nFLASHCARDS")
        for card in pack.flashcards:
            print(f"\n  Q: {card.front}")
            if not args.hide_answers:
                print(f"  A: {card.back[:240]}")

    if args.hide_answers:
        print(f"\n{_RULE}\nAnswers hidden. Re-run without --hide-answers to reveal them.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Tutor-style explanation of a topic, grounded in the documents."""
    engine = _engine(args)
    if engine.is_empty():
        print("index is empty - run `verirag ingest` first", file=sys.stderr)
        return 2
    doc_ids = None
    if args.doc:
        doc = _resolve_doc(engine, args.doc)
        if doc is None:
            return 2
        doc_ids = [doc.doc_id]

    result = engine.explain(args.topic, doc_ids=doc_ids, level=args.level)
    _print_answer(result.answer, show_trace=args.trace)
    return 0


def _resolve_doc(engine: VeriRAG, needle: str | None):
    """Find a document by doc_id or by a case-insensitive name fragment."""
    documents = engine.documents()
    if not documents:
        print("no documents indexed - run `verirag ingest` first", file=sys.stderr)
        return None
    if not needle:
        if len(documents) == 1:
            return documents[0]
        print("several documents indexed; pass --doc <name-or-id>:", file=sys.stderr)
        for document in documents:
            print(f"  {document.doc_id}  {document.name}", file=sys.stderr)
        return None

    lowered = needle.lower()
    exact = next((d for d in documents if d.doc_id == needle), None)
    if exact:
        return exact
    matches = [d for d in documents if lowered in d.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"no document matches {needle!r}", file=sys.stderr)
    else:
        print(f"{needle!r} matches several documents:", file=sys.stderr)
        for document in matches:
            print(f"  {document.doc_id}  {document.name}", file=sys.stderr)
    return None


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import run_eval, save_report  # local import keeps startup fast

    engine = _engine(args)
    if engine.is_empty():
        print("index is empty — run `verirag ingest` first", file=sys.stderr)
        return 2

    report = run_eval(engine, k=args.k, strict=not args.lenient)
    print(report.render())

    if args.json:
        path = save_report(report, args.json)
        print(f"\nreport written to {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end demonstration — the fastest way to review this project."""
    settings = get_settings()
    if not any(settings.raw_dir.glob("*.pdf")):
        print("generating sample PDFs …")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.make_sample_pdfs import main as make_pdfs

        make_pdfs()

    engine = _engine(args)
    if engine.is_empty():
        print("\nindexing …")
        for report in engine.ingest(settings.raw_dir):
            if report.document:
                print(f"   {report.document.name}: {report.chunks} chunks, {report.lines} lines")

    questions = [
        "What is the monthly rent and how often does it escalate?",
        "What interest rate and compensation did the court award?",
        "Which isolation level still permits phantom reads?",
    ]
    session_id = engine.new_session("Demo session")
    for question in questions:
        print(f"\n{_RULE}\nQ: {question}")
        result = engine.ask(question, session_id=session_id, render_proof=True)
        _print_answer(result.answer)
        for image in result.proofs:
            path = engine.settings.proof_dir / f"demo_{Path(image.doc_name).stem}_p{image.page_no}.png"
            path.write_bytes(image.png)
            print(f"   proof image: {path}")

    print(f"\n{_RULE}\nchat history persisted in {engine.settings.chat_db_path}")
    print(f"session id: {session_id}   →  verirag history {session_id} --markdown")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verirag",
        description="Verifiable RAG over PDFs — answers cite document, page and line, and show the highlighted source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", help="llm provider: auto|groq|gemini|ollama|extractive")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a PDF file or directory")
    p_ingest.add_argument("path", nargs="?", help="pdf file or folder (default: data/raw)")
    p_ingest.add_argument("--force", action="store_true", help="re-index even if unchanged")
    p_ingest.add_argument("--reset", action="store_true", help="wipe the index first")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--session", help="persist this turn into a chat session id")
    p_ask.add_argument("--doc", action="append", help="restrict to doc_id (repeatable)")
    p_ask.add_argument("-k", type=int, help="number of sources to retrieve")
    p_ask.add_argument("--json", action="store_true", help="machine-readable output")
    p_ask.add_argument("--trace", action="store_true", help="show retrieval trace")
    p_ask.add_argument("--no-proof", action="store_true", help="skip highlighted page rendering")
    p_ask.add_argument("--save-proof", help="directory for proof PNGs")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive chat with persistent history")
    p_chat.add_argument("--session", help="resume an existing session id")
    p_chat.add_argument("-k", type=int)
    p_chat.set_defaults(func=cmd_chat)

    p_sessions = sub.add_parser("sessions", help="list chat sessions")
    p_sessions.set_defaults(func=cmd_sessions)

    p_history = sub.add_parser("history", help="export one session")
    p_history.add_argument("session_id")
    p_history.add_argument("--markdown", action="store_true")
    p_history.set_defaults(func=cmd_history)

    p_search = sub.add_parser("search", help="full-text search across chat history")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.set_defaults(func=cmd_search)

    p_docs = sub.add_parser("docs", help="list indexed documents")
    p_docs.set_defaults(func=cmd_docs)

    p_stats = sub.add_parser("stats", help="index and history statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_topics = sub.add_parser("topics", help="list the study topics found in a document")
    p_topics.add_argument("--doc", help="doc_id or part of the filename")
    p_topics.set_defaults(func=cmd_topics)

    p_quiz = sub.add_parser("quiz", help="generate MCQs, questions and flashcards from study material")
    p_quiz.add_argument("--doc", help="doc_id or part of the filename")
    p_quiz.add_argument("-n", type=int, default=6, help="number of MCQs (default 6)")
    p_quiz.add_argument("--short", type=int, default=3, help="number of short-answer questions")
    p_quiz.add_argument("--cards", type=int, default=4, help="number of flashcards")
    p_quiz.add_argument("--hide-answers", action="store_true", help="print the paper without the key")
    p_quiz.add_argument("--seed", type=int, help="make distractor choice reproducible")
    p_quiz.add_argument("--json", action="store_true")
    p_quiz.set_defaults(func=cmd_quiz)

    p_explain = sub.add_parser("explain", help="tutor-style explanation of a topic, with citations")
    p_explain.add_argument("topic")
    p_explain.add_argument("--doc", help="restrict to one document")
    p_explain.add_argument("--level", choices=["beginner", "exam"], default="exam")
    p_explain.add_argument("--trace", action="store_true")
    p_explain.set_defaults(func=cmd_explain)

    p_eval = sub.add_parser("eval", help="run the evaluation harness")
    p_eval.add_argument("-k", type=int, help="cutoff for recall/ndcg")
    p_eval.add_argument("--json", help="write the report to this path")
    p_eval.add_argument("--lenient", action="store_true", help="do not fail on broken gold labels")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo", help="generate corpus, index, ask, and write proof images")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
