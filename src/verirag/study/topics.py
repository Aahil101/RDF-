"""Topic discovery.

Study material already carries its own structure — headings, numbered sections,
unit titles — and the ingest stage preserved that as ``Chunk.section``. Topics are
therefore *recovered*, not invented: chunks are grouped by their section label,
which keeps every topic anchored to a real page range in the PDF.

When a document has no usable headings (common in scanned-then-OCR'd notes or
plain prose), grouping falls back to contiguous page-level buckets so the feature
still works rather than returning nothing.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..index.embedder import tokenize
from ..models import Chunk
from .models import StudyTopic

_STOP = frozenset(
    """a an and are as at be been being by for from has have had he her his i in is it its of on
    or that the their there they this to was were what when where which who will with would you
    your shall may must not no such any each other than then these those into upon under over per
    also both either neither if but so however therefore thus hence within without upon""".split()
)

_GENERIC = frozenset(
    """section clause page unit module chapter part article schedule annexure note notes example
    following above below shall will must may can also given figure table marks question questions
    answer answers explain define describe state list discuss""".split()
)

_MIN_TOPIC_WORDS = 40


def _clean_topic_name(raw: str) -> str:
    """Turn 'CLAUSE 2 - RENT, ESCALATION AND MODE OF PAYMENT' into readable text."""
    text = re.sub(r"\s+", " ", raw).strip(" -\u2013\u2014:.")
    if not text:
        return ""
    # Drop a leading enumerator ("3.5", "CLAUSE 2 -", "UNIT 3 -").
    text = re.sub(
        r"^(?:(?:SECTION|CLAUSE|UNIT|MODULE|CHAPTER|PART|ARTICLE|SCHEDULE|ANNEXURE)\s+)?"
        r"[A-Z0-9]+(?:\.\d+)*\s*[-\u2013\u2014.)]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if not text:
        text = re.sub(r"\s+", " ", raw).strip(" -:.")
    # Title-case shouted headings, leave mixed case alone.
    if text.isupper():
        text = text.title()
    return text


def extract_key_terms(text: str, limit: int = 8) -> list[str]:
    """TF-IDF-flavoured key terms: frequent, long, non-generic words."""
    tokens = [t for t in tokenize(text) if t not in _STOP and t not in _GENERIC and len(t) > 3]
    if not tokens:
        return []
    counts = Counter(tokens)
    scored = {term: count * (1.0 + math.log(len(term))) for term, count in counts.items()}
    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [term for term, _score in ranked[:limit]]


def extract_topics(chunks: list[Chunk], *, min_words: int = _MIN_TOPIC_WORDS) -> list[StudyTopic]:
    """Group *chunks* of one document into study topics."""
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda c: (c.page_no, c.line_start))
    grouped: dict[str, list[Chunk]] = {}

    has_sections = any(c.section.strip() for c in ordered)
    for chunk in ordered:
        if has_sections:
            key = chunk.section.strip() or "Introduction"
        else:
            key = f"Pages {chunk.page_no}"
        grouped.setdefault(key, []).append(chunk)

    topics: list[StudyTopic] = []
    for raw_name, members in grouped.items():
        body = " ".join(c.body_text for c in members)
        words = len(body.split())
        name = _clean_topic_name(raw_name) or raw_name
        topics.append(
            StudyTopic(
                name=name,
                doc_id=members[0].doc_id,
                doc_name=members[0].doc_name,
                chunk_ids=[c.chunk_id for c in members],
                page_start=min(c.page_no for c in members),
                page_end=max(c.page_no for c in members),
                key_terms=extract_key_terms(body),
                word_count=words,
            )
        )

    # Merge topics too thin to quiz on into the previous one, preserving order.
    merged: list[StudyTopic] = []
    for topic in sorted(topics, key=lambda t: (t.page_start, t.name)):
        if merged and topic.word_count < min_words:
            previous = merged[-1]
            previous.chunk_ids.extend(topic.chunk_ids)
            previous.page_end = max(previous.page_end, topic.page_end)
            previous.word_count += topic.word_count
            previous.key_terms = list(dict.fromkeys(previous.key_terms + topic.key_terms))[:8]
            continue
        merged.append(topic)

    return [t for t in merged if t.word_count >= 12]
