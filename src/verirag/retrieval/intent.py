"""Query intent.

Not every question is a lookup. "Explain this document" and "what are the key
points" are *global* requests: they are about the document as a whole, so there is
no passage that resembles them and similarity retrieval has nothing to bite on.
Routing them through the lookup path produces the worst possible outcome — a
confident refusal to answer the single most common question a new user asks.

Three intents are distinguished:

``LOOKUP``
    A specific fact. The normal retrieve → rank → cite path.
``SUMMARY``
    The whole document, or a whole topic. Needs a representative spread of the
    document rather than the top-k most similar chunks.
``TOPICS``
    "What's in here?" — answered from structure, not from retrieval at all.

Detection is rule-based, so it costs nothing and cannot fail. The rules are
deliberately conservative: a query has to look clearly global to leave the lookup
path, because misrouting a real question into a summary is also a bad outcome.
"""

from __future__ import annotations

import re
from enum import Enum

from rapidfuzz import fuzz, process


class Intent(str, Enum):
    LOOKUP = "lookup"
    SUMMARY = "summary"
    TOPICS = "topics"


# Phrases that mean "summarise" on their own, whatever else is in the query.
_SUMMARY_STRONG_RE = re.compile(
    r"\b("
    r"summari[sz]e|summari[sz]ation|summary|tl;?dr|"
    r"overview|abstract|gist|recap|synopsis|"
    r"key\s+(?:points?|takeaways?|ideas?|findings?|highlights?)|"
    r"main\s+(?:points?|ideas?|findings?|argument|takeaways?)|"
    r"in\s+(?:short|brief|a\s+nutshell)|"
    r"walk\s+me\s+through|brief\s+me|"
    r"table\s+of\s+contents"
    r")\b",
    re.IGNORECASE,
)

# A verb that asks for exposition rather than a specific fact.
_EXPOSITORY_VERB_RE = re.compile(
    r"\b(explain|describe|summari[sz]e|outline|elaborate|"
    r"tell\s+me|what(?:'s| is| are)?|whats)\b",
    re.IGNORECASE,
)

# A reference to the document as a whole rather than to something inside it.
_WHOLE_DOC_NOUN_RE = re.compile(
    r"\b(document|documents|pdf|pdfs|file|paper|notes?|deed|judgment|judgement|"
    r"case|material|contents?|everything|all\s+of\s+(?:this|it))\b",
    re.IGNORECASE,
)

# "what topics are covered", "what's in this pdf", "what can I ask"
_TOPICS_RE = re.compile(
    r"\b("
    r"(?:what|which)\s+(?:topics|sections|chapters|clauses)|"
    r"list\s+(?:the\s+)?(?:topics|sections|chapters|clauses)|"
    r"what(?:'s| is)\s+(?:in|inside|covered)\b|"
    r"what\s+(?:does|do)\s+(?:this|it|the\s+\w+)\s+cover|"
    r"what\s+(?:can|could|should)\s+i\s+ask|"
    r"sections?\s+(?:in|of)\s+(?:this|the)"
    r")\b",
    re.IGNORECASE,
)

# A bare pointer at the document with nothing else asked: "this document?"
_BARE_POINTER_RE = re.compile(
    r"^(?:this|that|the|these|it)\s*"
    r"(?:document|documents|pdf|file|paper|notes?|deed|judgment|judgement|case|material)?$",
    re.IGNORECASE,
)


def classify(query: str) -> Intent:
    """Classify *query* into a retrieval strategy.

    Deliberately built from *two independent signals* rather than one rigid
    phrase list. A single pattern like ``explain this document`` breaks on a typo
    between the words ("explain abput this document") and on an intervening noun
    ("what is this pdf about") — both of which real users type. Requiring an
    expository verb *and* a whole-document reference anywhere in the query
    survives both, while still keeping specific lookups out of the summary path.
    """
    text = (query or "").strip()
    if not text:
        return Intent.LOOKUP

    # Structure questions first: "what's in this pdf" also reads as a summary
    # request, but listing the sections is the more useful answer.
    if _TOPICS_RE.search(text):
        return Intent.TOPICS

    words = [w for w in re.sub(r"[^\w\s;]", " ", text.lower()).split() if w]

    if _SUMMARY_STRONG_RE.search(text) or _fuzzy_any(words, _STRONG_WORDS):
        return Intent.SUMMARY

    words_joined = " ".join(words)
    if words and _BARE_POINTER_RE.fullmatch(words_joined):
        return Intent.SUMMARY

    # Two independent signals, each typo-tolerant. Both must be present, which is
    # what keeps a real lookup out of the summary path.
    has_verb = bool(_EXPOSITORY_VERB_RE.search(text)) or _fuzzy_any(words, _VERB_WORDS)
    has_whole_doc = (
        bool(_WHOLE_DOC_NOUN_RE.search(text))
        or _fuzzy_any(words, _WHOLE_DOC_WORDS)
        or bool(re.search(r"\b(this|these|it)\b", text, re.IGNORECASE))
    )
    if has_verb and has_whole_doc:
        # A query carrying its own subject matter is a lookup, not an overview:
        # "explain BCNF in this document" has a subject, "explain this document"
        # does not. So any word left over after removing the intent vocabulary
        # makes it a lookup.
        #
        # The catch is typos. "explain abput this document" leaves "abput", which
        # looks like a subject but is a misspelled "about". Leftovers are therefore
        # fuzzy-matched against the intent vocabulary first — a misspelled function
        # word is still a function word.
        if not _subject_words(words):
            return Intent.SUMMARY

    return Intent.LOOKUP


def _subject_words(words: list[str]) -> list[str]:
    """Words that name a subject, ignoring intent vocabulary and its typos."""
    leftovers: list[str] = []
    for word in words:
        if word in _INTENT_VOCABULARY:
            continue
        if len(word) >= 4 and process.extractOne(
            word, _INTENT_VOCABULARY_LIST, scorer=fuzz.ratio, score_cutoff=80
        ):
            continue
        leftovers.append(word)
    return leftovers


_INTENT_VOCABULARY = frozenset(
    """a about all an and are as at be brief briefly can could describe detail details do does
    elaborate everything explain file for give i in is it its me more notes of on or outline
    paper pdf pdfs please summarise summarize tell that the these this to what whats which
    document documents deed judgment judgement case material content contents you your
    something anything much little bit short overview simple simply kindly just now here
    whole entire full complete briefly quickly again rest generally basically main key
    points point summary gist recap tldr abstract synopsis outline elaborate""".split()
)
"""Words that express *how* something is asked rather than *what* is asked.

A query whose content words are all in here is asking about the document itself;
anything left over is a subject, which means it is a lookup."""

_INTENT_VOCABULARY_LIST = sorted(_INTENT_VOCABULARY)

# Word lists for the typo-tolerant checks. Kept separate from the regexes because
# fuzzy matching works on single words, while the regexes capture phrases.
_STRONG_WORDS = [
    "summarize", "summarise", "summary", "overview", "abstract",
    "gist", "recap", "synopsis", "tldr",
]
_VERB_WORDS = [
    "explain", "describe", "summarize", "summarise", "outline", "elaborate", "overview",
]
_WHOLE_DOC_WORDS = [
    "document", "documents", "pdf", "file", "paper", "notes", "deed",
    "judgment", "judgement", "material", "contents", "everything",
]


def _fuzzy_any(words: list[str], candidates: list[str], cutoff: int = 82) -> bool:
    """True if any word is, or is nearly, one of *candidates*.

    "sumarize this documnt" must still route to a summary. Users mistype the
    trigger word as readily as anything else, and a classifier that only accepts
    perfect spelling fails precisely when the user is least careful.
    """
    for word in words:
        if len(word) < 4:
            continue
        if word in candidates:
            return True
        if process.extractOne(word, candidates, scorer=fuzz.ratio, score_cutoff=cutoff):
            return True
    return False


def topic_of_interest(query: str) -> str:
    """For a SUMMARY query, the specific topic asked about, if any.

    ``"summarise clause 3"`` should summarise clause 3, not the whole deed.
    Returns an empty string when the request is about the document as a whole.
    """
    text = (query or "").strip()
    if not text:
        return ""

    remainder = _SUMMARY_STRONG_RE.sub(" ", text)
    remainder = _EXPOSITORY_VERB_RE.sub(" ", remainder)
    remainder = re.sub(
        r"\b(this|that|these|the|a|an|of|for|in|about|me|please|document|pdf|file|"
        r"paper|notes?|deed|judgment|judgement|case|material|it|whole|entire|"
        r"and|to|on|contents?)\b",
        " ",
        remainder,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(r"[^\w\s.-]", " ", remainder)
    cleaned = " ".join(remainder.split())
    return cleaned if len(cleaned) >= 3 else ""
