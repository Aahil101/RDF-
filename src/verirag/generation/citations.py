"""Citation parsing and sentence segmentation.

The model is asked to emit ``[S3]`` markers.  This module turns that text back
into structured data and — importantly — *repairs* it:

* accepts ``[S3]``, ``[3]``, ``[S1, S2]``, ``[S1; S3]`` and ``[S2][S4]``,
* drops markers that point at sources which were never supplied
  (a hallucinated citation is itself a hallucination),
* reports which supplied sources were actually used.

Sentence splitting is abbreviation-aware because legal and academic text is
full of ``No.``, ``v.``, ``Rs.``, ``Cl.``, ``i.e.`` and section numbers that a
naive ``split('.')`` would shred, breaking per-sentence grounding checks.
"""

from __future__ import annotations

import re

from ..textnorm import fold_punctuation

MARKER_RE = re.compile(r"\[\s*(?:S|s)?\s*(\d+(?:\s*[,;/]\s*(?:S|s)?\s*\d+)*)\s*\]")
_NUM_RE = re.compile(r"\d+")

_ABBREVIATIONS = frozenset(
    """no nos vs v cl cls sec secs s ss art arts para paras fig figs eq eqs ch chs pp p pg
    rs usd inr mr mrs ms dr prof hon ltd pvt co inc jr sr st etc eg ie viz al ors anr
    jan feb mar apr jun jul aug sep sept oct nov dec approx max min avg vol ed""".split()
)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# Bullet glyphs seen in extracted PDF text, including the MIDDLE DOT that
# PyMuPDF frequently reports in place of a real bullet character.
_BULLET_CHARS = "\u2022\u00b7\u2023\u25aa\u25cf\u2043\u2219-"
_BULLET_SPLIT_RE = re.compile(rf"\s*(?:^|\s)[{re.escape(_BULLET_CHARS)}]\s+")


def parse_markers(text: str) -> list[int]:
    """Return every distinct source number cited in *text*, in first-use order."""
    found: list[int] = []
    for match in MARKER_RE.finditer(text):
        for number in _NUM_RE.findall(match.group(1)):
            value = int(number)
            if value not in found:
                found.append(value)
    return found


def strip_markers(text: str) -> str:
    """Remove citation markers — used before semantic comparison."""
    return re.sub(r"\s{2,}", " ", MARKER_RE.sub("", text)).strip()


def _ends_with_abbreviation(fragment: str) -> bool:
    tail = fragment.rstrip()
    if not tail.endswith("."):
        return True if tail.endswith((" ", "")) and not tail else False
    word = re.split(r"[\s(\[]", tail[:-1])[-1].lower().strip(".,;:\"'")
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha():  # initials: "R. Sharma"
        return True
    if re.fullmatch(r"\d+", word):  # enumerations: "1. First point"
        return True
    return False


def split_sentences(text: str, *, min_chars: int = 3) -> list[str]:
    """Abbreviation-aware sentence splitter."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    sentences: list[str] = []
    buffer = ""
    for piece in _SENTENCE_END_RE.split(cleaned):
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if _ends_with_abbreviation(buffer):
            continue
        sentences.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())

    # Bullet lists arrive as one "sentence"; treat each bullet as a claim.
    # The glyph set matters: PyMuPDF commonly reports a bullet as U+00B7
    # (MIDDLE DOT) rather than U+2022, and missing it silently glues one list
    # item onto the next.
    expanded: list[str] = []
    for sentence in sentences:
        parts = _BULLET_SPLIT_RE.split(sentence)
        expanded.extend(p.strip() for p in parts if len(p.strip()) >= min_chars)

    return [s for s in expanded if len(s) >= min_chars]


_MARKER_GROUP = r"(?:\[\s*[Ss]?\s*\d+(?:\s*[,;/]\s*[Ss]?\s*\d+)*\s*\]\s*)+"
_TRAILING_MARKERS_RE = re.compile(rf"([.!?])(\s*)({_MARKER_GROUP})")


def attach_trailing_markers(text: str) -> str:
    """Move citation markers that follow a full stop to *before* it.

    Both the prompt contract and real model behaviour produce
    ``"The rent is Rs. 48,500. [S2]"``.  Naive sentence splitting cuts after the
    full stop, so ``[S2]`` would be attributed to the *next* sentence and the
    claim it actually supports would look uncited.  Normalising to
    ``"The rent is Rs. 48,500 [S2]."`` keeps every marker with its own claim.
    """
    return _TRAILING_MARKERS_RE.sub(lambda m: f" {m.group(3).strip()}{m.group(1)}{m.group(2) or ' '}", text or "")


def sentences_with_markers(text: str) -> list[tuple[str, list[int]]]:
    """Split into sentences and attach the source numbers each one cites."""
    normalised = attach_trailing_markers(text)
    return [(sentence, parse_markers(sentence)) for sentence in split_sentences(normalised)]


def sanitize_answer(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop out-of-range markers; return cleaned text and valid numbers used.

    Also folds typographic punctuation in the *generated* text. Models routinely
    emit U+202F (narrow no-break space) and U+2011 (non-breaking hyphen), which
    render as nothing at all in a legacy console — "within\u202fninety\u202fdays"
    appears as "withinninetydays". Source quotes keep their original characters
    for fidelity to the PDF, but model prose is not a document and normalising it
    is strictly an improvement.
    """
    valid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in _NUM_RE.findall(match.group(1))]
        kept = [n for n in numbers if 1 <= n <= n_sources]
        for number in kept:
            if number not in valid:
                valid.append(number)
        return "".join(f"[S{n}]" for n in kept)

    cleaned = MARKER_RE.sub(replace, fold_punctuation(text or ""))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), valid
