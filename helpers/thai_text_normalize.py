"""Conservative normalization for Thai extraction and FTS query text.

PDF/OCR/layout extractors commonly inject arbitrary whitespace inside Thai
text.  For RAG recall, remove every whitespace run whose two neighbouring
characters are Thai (``คำ สั่ง ระบบ`` -> ``คำสั่งระบบ``), while preserving
Thai-to-Latin/number boundaries and never attempting glyph substitution.
"""
from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RUN_RE = re.compile(r"\s+")
_THAI_BASE_OR_MARK_BEFORE_SPACE_RE = re.compile(
    r"(?<=[\u0E01-\u0E2E\u0E30-\u0E33\u0E40-\u0E45\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
    r"\s+(?=[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
)
_THAI_MARK_BEFORE_BASE_AFTER_SPACE_RE = re.compile(
    r"(?<=[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E])"
    r"\s+(?=[\u0E01-\u0E2E\u0E30-\u0E33\u0E40-\u0E45])"
)
_THAI_CHAR_ON_BOTH_SIDES_SPACE_RE = re.compile(
    r"(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])"
)
# OCR sometimes emits SARA AM before a following tone mark (e.g. "นำ้").
# Thai orthography encodes the tone mark before SARA AM ("น้ำ"). This is a
# structural reorder only: no character is invented, removed, or substituted.
_THAI_SARA_AM_BEFORE_TONE_RE = re.compile(r"\u0E33([\u0E48-\u0E4B])")


def repair_thai_orthographic_order(text: str) -> str:
    """Repair deterministic Thai mark-order errors without lexical guessing."""
    return _THAI_SARA_AM_BEFORE_TONE_RE.sub(lambda match: match.group(1) + "ำ", text)


def repair_thai_combining_mark_spacing(text: str) -> str:
    """Remove internal whitespace from Thai runs for indexing and lookup."""
    while True:
        cleaned = _THAI_BASE_OR_MARK_BEFORE_SPACE_RE.sub("", text)
        cleaned = _THAI_MARK_BEFORE_BASE_AFTER_SPACE_RE.sub("", cleaned)
        cleaned = _THAI_CHAR_ON_BOTH_SIDES_SPACE_RE.sub("", cleaned)
        if cleaned == text:
            return cleaned
        text = cleaned


def normalize_extracted_text(text: str) -> str:
    """Canonicalize extraction layout while preserving ordinary word spacing."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = _WHITESPACE_RUN_RE.sub(" ", normalized).strip()
    normalized = repair_thai_combining_mark_spacing(normalized)
    return repair_thai_orthographic_order(normalized)
