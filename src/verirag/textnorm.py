"""Text normalisation used when *comparing* strings.

Language models emit typographic punctuation — a non-breaking hyphen in
``P\u2011 42``, a right single quote in ``months\u2019``, an en dash in a range —
while the PDF contains the plain ASCII forms. Any literal comparison between
generated text and source text must fold these first, otherwise:

* the evaluation harness scores a correct answer as wrong because ``"P-42"`` is
  not a substring of ``"P\u201142"``; and
* the groundedness verifier under-reports support for a claim that is in fact
  quoted verbatim.

Folding happens at *comparison* time only. Displayed quotes keep the original
characters, so the text shown beside a citation still matches the PDF exactly.
"""

from __future__ import annotations

import re
import unicodedata

# Dash and hyphen variants -> ASCII hyphen.
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\ufe58\ufe63\uff0d"
# Single-quote variants -> ASCII apostrophe.
_APOSTROPHES = "\u2018\u2019\u201a\u201b\u2032\u02bc\u02b9\u055a\uff07"
# Double-quote variants -> ASCII quote.
_QUOTES = "\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb\uff02"
# Space variants -> plain space.
_SPACES = "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"

_TRANSLATION = {
    **{ord(ch): "-" for ch in _DASHES},
    **{ord(ch): "'" for ch in _APOSTROPHES},
    **{ord(ch): '"' for ch in _QUOTES},
    **{ord(ch): " " for ch in _SPACES},
    ord("\u2026"): "...",
    ord("\u00ad"): "",  # soft hyphen
    ord("\u200b"): "",  # zero-width space
    ord("\u200c"): "",
    ord("\u200d"): "",
    ord("\ufeff"): "",
}

_WS_RE = re.compile(r"\s+")


def fold_punctuation(text: str) -> str:
    """Replace typographic punctuation with its ASCII equivalent."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).translate(_TRANSLATION)


def normalise_for_compare(text: str) -> str:
    """Fold punctuation, lowercase, and collapse whitespace."""
    return _WS_RE.sub(" ", fold_punctuation(text).lower()).strip()


def contains_phrase(haystack: str, needle: str) -> bool:
    """Punctuation-insensitive substring test."""
    if not needle:
        return False
    return normalise_for_compare(needle) in normalise_for_compare(haystack)
