"""Lazy Thai word-boundary helpers shared by Lite KB indexing and search.

Mirrors ENDEAVOR_RAG_MAX's proven PyThaiNLP ``newmm`` choice without bringing
its BM25/vector stack into the host agent. If PyThaiNLP is unavailable, callers
fall back to the previous character/trigram behavior rather than failing startup.
"""
from __future__ import annotations

from functools import lru_cache
import re


@lru_cache(maxsize=1)
def _word_tokenizer():
    try:
        from pythainlp import word_tokenize
        return word_tokenize
    except Exception:
        return None


def available() -> bool:
    return _word_tokenizer() is not None


def has_thai(text: str) -> bool:
    return any("\u0e00" <= char <= "\u0e7f" for char in text)


def word_boundary_offsets(text: str) -> list[int]:
    """Return safe end offsets for whole words/tokens in the original text.

    Thai uses PyThaiNLP/newmm when available.  Other text (and Thai when the
    optional tokenizer is unavailable) falls back to whitespace-delimited
    spans.  Offsets are character positions in ``text`` and always include
    the final length, so callers can map them onto another tokenizer without
    altering the original text.
    """
    if not text:
        return []
    tokenizer = _word_tokenizer()
    if tokenizer is not None and has_thai(text):
        tokens = tokenizer(text, engine="newmm", keep_whitespace=True)
        if "".join(tokens) == text:
            offsets: list[int] = []
            cursor = 0
            for token in tokens:
                cursor += len(token)
                if token.strip():
                    offsets.append(cursor)
            if offsets and offsets[-1] != len(text):
                offsets.append(len(text))
            return offsets or [len(text)]

    offsets = [match.end() for match in re.finditer(r"\S+", text)]
    if not offsets or offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def query_words(text: str) -> list[str]:
    """Return semantic word units for a normalized Thai/mixed query."""
    if not text:
        return []
    tokenizer = _word_tokenizer()
    if tokenizer is None or not has_thai(text):
        return [text]
    return [
        token for token in tokenizer(text, engine="newmm", keep_whitespace=False)
        if token.strip()
    ]


def bounded_pieces(text: str, chunk_size: int) -> list[str]:
    """Split at newmm word boundaries, hard-splitting only one oversized token.

    Equivalent to RAG MAX's ``_pack_words`` boundary rule but without overlap;
    Lite adds overlap when packing source units into final SQLite chunks.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    tokenizer = _word_tokenizer()
    if tokenizer is None or not has_thai(text):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    window: list[str] = []
    window_len = 0
    for token in tokenizer(text, engine="newmm", keep_whitespace=True):
        token_len = len(token)
        if token_len > chunk_size:
            if window:
                pieces.append("".join(window))
                window, window_len = [], 0
            pieces.extend(
                token[i:i + chunk_size] for i in range(0, token_len, chunk_size)
            )
            continue
        if window and window_len + token_len > chunk_size:
            pieces.append("".join(window))
            window, window_len = [], 0
        window.append(token)
        window_len += token_len
    if window:
        pieces.append("".join(window))
    return pieces or [text]


def safe_overlap_suffix(text: str, limit: int) -> str:
    """Return <=limit trailing whole newmm tokens when Thai is present."""
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    tokenizer = _word_tokenizer()
    if tokenizer is None or not has_thai(text):
        return text[-limit:]

    tokens = tokenizer(text, engine="newmm", keep_whitespace=True)
    tail: list[str] = []
    used = 0
    for token in reversed(tokens):
        size = len(token)
        if used + size > limit:
            break
        tail.insert(0, token)
        used += size
    # Match RAG MAX: if even the final whole token is larger than the
    # overlap budget, keep no overlap rather than starting the next chunk in
    # the middle of a Thai word.
    return "".join(tail)
