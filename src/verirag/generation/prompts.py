"""Prompts.

The system prompt is the first line of defence against hallucination: it makes
citation mandatory, forbids outside knowledge and mandates an explicit refusal
token when the context is insufficient.  The verification layer in
``verirag.verify`` then *checks* compliance rather than trusting it — prompt
discipline plus post-hoc validation, not one or the other.
"""

from __future__ import annotations

from typing import Sequence

from ..models import Chunk, RetrievedChunk

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are VeriRAG, a document-grounded question answering assistant.

ABSOLUTE RULES
1. Answer ONLY from the numbered SOURCES supplied by the user. Never use outside
   knowledge, never guess, never fill gaps with plausible-sounding detail.
2. Every factual sentence MUST end with one or more citation markers naming the
   sources it came from, e.g. "The lock-in period is 11 months. [S2]" or
   "Both parties must sign. [S1][S4]".
3. Quote exact figures, dates, names, clause numbers and section numbers as they
   appear in the sources. Do not round, convert or reword them.
4. If the sources do not contain the answer, reply with exactly:
   {REFUSAL_TOKEN}: followed by one short sentence saying what is missing.
   Do NOT attempt a partial answer built on assumptions.
5. If sources conflict, say so explicitly and cite each conflicting source.
6. Be concise: 1-6 sentences for simple questions. Use a short bullet list only
   when the answer is genuinely enumerable, citing each bullet.
7. Never invent a source marker that was not provided.

OUTPUT
Plain text only. No preamble such as "Based on the sources". No markdown
headings. Citation markers must use the exact [S<number>] form."""


ANSWER_TEMPLATE = """SOURCES
{sources}

QUESTION
{question}

Answer using only the sources above, citing each factual sentence with [S<number>]."""


SUMMARY_SYSTEM = """You summarise a document from numbered excerpts of it.

ABSOLUTE RULES
1. Use ONLY the excerpts supplied. Never add outside knowledge and never infer
   facts that are not stated.
2. Every factual sentence MUST end with one or more [S<number>] markers naming the
   excerpts it came from.
3. Quote figures, dates, names, amounts and clause or section numbers exactly as
   they appear.
4. The excerpts are a *sample* of the document, not all of it. Do not claim
   completeness, and do not invent content for parts you were not shown.
5. Never invent a source marker that was not provided.

STRUCTURE
Open with one sentence saying what kind of document this is and what it concerns.
Then give the most important specifics as a short bullet list - parties, amounts,
dates, obligations, findings, or for study material the concepts covered. Cite
every bullet. Six to ten bullets at most; fewer if the excerpts do not support
more.

OUTPUT
Plain text. No markdown headings. No preamble such as "Here is a summary"."""


SUMMARY_USER = """DOCUMENT: {doc_name}

EXCERPTS
{sources}

Summarise this document using only the excerpts above, citing every factual
sentence with [S<number>]."""


EXPANSION_SYSTEM = """You rewrite a user's question into alternative search queries for a
document retrieval engine. Output 2 short alternatives, one per line, no numbering,
no explanation. Use synonyms and the formal vocabulary a contract, judgment or
textbook would use."""


def format_source_block(index: int, chunk: Chunk, *, max_chars: int = 1200) -> str:
    """Render one source with its full physical locator in the header."""
    text = chunk.text if len(chunk.text) <= max_chars else f"{chunk.text[:max_chars].rstrip()}…"
    section = f" | section: {chunk.section}" if chunk.section else ""
    return (
        f"[S{index}] document: {chunk.doc_name} | page: {chunk.page_no} "
        f"| lines: {chunk.line_start}-{chunk.line_end}{section}\n{text}"
    )


def build_sources_text(items: Sequence[RetrievedChunk], *, max_chars: int = 1200) -> str:
    return "\n\n".join(
        format_source_block(i, item.chunk, max_chars=max_chars) for i, item in enumerate(items, start=1)
    )


def build_answer_prompt(question: str, items: Sequence[RetrievedChunk], *, max_chars: int = 1200) -> str:
    return ANSWER_TEMPLATE.format(
        sources=build_sources_text(items, max_chars=max_chars),
        question=question.strip(),
    )


def build_history_preamble(turns: Sequence[tuple[str, str]], limit: int = 3) -> str:
    """Compact recent dialogue so follow-up questions resolve pronouns."""
    if not turns:
        return ""
    recent = turns[-limit:]
    lines = [f"Q: {q.strip()}\nA: {a.strip()[:300]}" for q, a in recent]
    return "CONVERSATION SO FAR (for pronoun resolution only; never cite it)\n" + "\n\n".join(lines) + "\n\n"
