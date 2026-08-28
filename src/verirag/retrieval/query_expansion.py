"""Query understanding: normalisation and multi-query expansion.

Users ask "who has to pay the maintenance?" while the PDF says "the Lessee
shall bear all maintenance charges".  Firing a single query at a single
retriever is the most common reason RAG demos miss.  VeriRAG issues a small
family of variants and fuses their rankings:

1. the raw question,
2. a keyword-only form (stopwords and question words removed),
3. a declarative form (interrogative rewritten as a statement stem),
4. a domain-synonym expansion for legal / property / academic vocabulary.

All four are rule-based, so expansion costs nothing and never fails.  An
optional LLM paraphraser can be plugged in via ``llm_expand``.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from rapidfuzz import fuzz, process

from ..index.embedder import tokenize

_QUESTION_WORDS = frozenset(
    """what who whom whose when where why how which is are was were do does did can could shall
    should will would may might list explain describe summarise summarize tell give""".split()
)

_STOP = frozenset(
    """a an and any are as at be been by for from has have in into is it its of on or that the
    their there this to was were with""".split()
)

# Bidirectional vocabulary bridges for the three sample domains.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "landlord": ("lessor", "owner", "vendor"),
    "lessor": ("landlord", "owner"),
    "tenant": ("lessee", "occupant"),
    "lessee": ("tenant", "occupant"),
    "buyer": ("purchaser", "vendee", "transferee"),
    "purchaser": ("buyer", "transferee"),
    "seller": ("vendor", "transferor"),
    "vendor": ("seller", "transferor"),
    "rent": ("rental", "lease amount", "monthly rent"),
    "deposit": ("security deposit", "advance"),
    "penalty": ("liquidated damages", "late fee", "interest"),
    "terminate": ("termination", "cancel", "rescind", "determine"),
    "notice": ("intimation", "written notice", "notice period"),
    "verdict": ("judgment", "holding", "decree", "order"),
    "judgment": ("judgement", "verdict", "decree", "holding"),
    "damages": ("compensation", "award", "restitution"),
    "appeal": ("appellant", "appellate", "revision"),
    "plaintiff": ("appellant", "petitioner", "claimant"),
    "defendant": ("respondent", "opposite party"),
    "law": ("statute", "act", "section", "provision"),
    "definition": ("means", "defined as", "refers to"),
    "formula": ("equation", "expression"),
    "advantage": ("benefit", "pro", "merit"),
    "drawback": ("limitation", "disadvantage", "con"),
    "difference": ("versus", "compared", "distinction"),
    "example": ("e.g.", "for instance", "illustration"),
    "steps": ("procedure", "algorithm", "process"),
    "area": ("built-up area", "carpet area", "square feet"),
    "maintenance": ("upkeep", "repairs", "charges"),
    # Abbreviations and acronyms. A document usually spells the term out in full
    # exactly where it defines it, while users type the short form, so this is
    # where a keyword retriever silently misses the definitive passage.
    "bcnf": ("boyce codd normal form", "boyce-codd"),
    "1nf": ("first normal form",),
    "2nf": ("second normal form",),
    "3nf": ("third normal form",),
    "4nf": ("fourth normal form", "multivalued dependency"),
    "fd": ("functional dependency",),
    "mvd": ("multivalued dependency",),
    "acid": ("atomicity", "consistency", "isolation", "durability"),
    "wal": ("write-ahead logging", "write ahead log"),
    "2pl": ("two-phase locking", "two phase locking"),
    "rera": ("real estate regulation and development act",),
    "tds": ("tax deducted at source", "deduct tax at source"),
    "cnn": ("convolutional neural network",),
}


def normalise_query(query: str) -> str:
    """Trim, collapse whitespace, strip trailing punctuation noise."""
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    return cleaned.rstrip("?!.,;: ").strip() or cleaned


def keyword_form(query: str) -> str:
    """Content words only — helps BM25 a lot on verbose questions."""
    terms = [t for t in tokenize(query) if t not in _STOP and t not in _QUESTION_WORDS and len(t) > 1]
    return " ".join(dict.fromkeys(terms))


def declarative_form(query: str) -> str:
    """Turn a question into a statement stem the document is likelier to match."""
    text = normalise_query(query)
    lowered = text.lower()
    patterns = (
        (r"^who (?:is|are|was|were|has|have) (?:the )?(.*)$", r"\1 is"),
        (r"^who (?:must|shall|should|will|has to|have to) (.*)$", r"shall \1"),
        (r"^what (?:is|are|was|were) (?:the )?(.*)$", r"\1 is"),
        (r"^what (?:does|do) (.*?) mean$", r"\1 means"),
        (r"^when (?:is|was|does|did|will) (?:the )?(.*)$", r"\1 date"),
        (r"^where (?:is|are|was|were) (?:the )?(.*)$", r"\1 located at"),
        (r"^how (?:much|many) (?:is|are|was|were)? ?(?:the )?(.*)$", r"\1 amount"),
        (r"^why (?:did|does|was|is) (?:the )?(.*)$", r"because \1"),
        (r"^(?:can|could|may) (?:the )?(.*)$", r"\1 is permitted"),
        (r"^(?:list|explain|describe|summarise|summarize|tell me about) (?:the )?(.*)$", r"\1"),
    )
    for pattern, replacement in patterns:
        if re.match(pattern, lowered):
            return re.sub(pattern, replacement, lowered).strip()
    return ""


def synonym_form(query: str) -> str:
    """Append domain synonyms for any recognised term."""
    terms = tokenize(query)
    extra: list[str] = []
    for term in terms:
        for synonym in _SYNONYMS.get(term, ()):
            extra.append(synonym)
    if not extra:
        return ""
    return f"{normalise_query(query)} {' '.join(dict.fromkeys(extra))}"


def spelling_form(query: str, vocabulary: Sequence[str] | None = None, *, min_score: int = 82) -> str:
    """Repair out-of-vocabulary words against the indexed corpus.

    A single typo used to be fatal: "abput this document" shares no usable term
    with anything indexed, so retrieval returned nothing and the system refused a
    perfectly reasonable question. Correcting query terms against the *corpus's own
    vocabulary* — rather than a general dictionary — also fixes the more common
    case of a half-remembered technical term.

    Only words absent from the vocabulary are touched, so correctly spelled terms
    are never "corrected" into something else.
    """
    if not vocabulary:
        return ""

    known = set(vocabulary)
    tokens = tokenize(query)
    if not tokens:
        return ""

    corrected: list[str] = []
    changed = False
    for token in tokens:
        if len(token) < 4 or token in known or token in _STOP or token in _QUESTION_WORDS:
            corrected.append(token)
            continue
        match = process.extractOne(
            token, vocabulary, scorer=fuzz.ratio, score_cutoff=min_score
        )
        if match:
            corrected.append(match[0])
            changed = True
        else:
            corrected.append(token)

    return " ".join(corrected) if changed else ""


def expand_query(
    query: str,
    *,
    enabled: bool = True,
    max_variants: int = 5,
    llm_expand: Callable[[str], Sequence[str]] | None = None,
    vocabulary: Sequence[str] | None = None,
) -> list[str]:
    """Return the query plus up to ``max_variants - 1`` rule-based variants."""
    base = normalise_query(query)
    if not base:
        return []
    if not enabled:
        return [base]

    variants = [base]
    candidates = [
        spelling_form(base, vocabulary),
        keyword_form(base),
        declarative_form(base),
        synonym_form(base),
    ]
    for candidate in candidates:
        if candidate and candidate.lower() != base.lower() and candidate not in variants:
            variants.append(candidate)

    if llm_expand is not None:
        try:
            for candidate in llm_expand(base):
                cleaned = normalise_query(candidate)
                if cleaned and cleaned not in variants:
                    variants.append(cleaned)
        except Exception:  # noqa: BLE001 - expansion is best-effort only
            pass

    return variants[:max_variants]
