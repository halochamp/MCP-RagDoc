"""One deny policy for local secrets across direct read, bash, and FTS index."""
from __future__ import annotations

from pathlib import Path
import re

_SENSITIVE_DIRS = {
    ".aws", ".claude", ".codex", ".config", ".docker", ".gnupg", ".kube",
    ".npm", ".ssh", ".lite_state", "keychains", "huggingface",
}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".keystore"}
_SENSITIVE_NAMES = {
    ".env", ".git-credentials", ".netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.json", "id_ed25519", "id_rsa", "passwd",
    "password", "secrets", "secrets.json", "tokens.json",
}
_BROWSER_PARTS = {"chrome", "edge", "firefox", "safari", "brave-browser", "brave"}
_SECRET_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:hf|ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    # Assignment-style credentials need a literal-looking value.  The old
    # optional-quote expression treated ordinary source expressions such as
    # ``token = search_scope.set_request_query(user_input)`` as secrets, which
    # blocked reading agent.py itself.  Keep bare token assignments covered,
    # but require a contiguous value that cannot be a dotted/call expression;
    # quoted values remain covered even when they contain punctuation.
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|token|secret)\b"
        r"\s*[:=]\s*(?:['\"][^'\"]{12,}['\"]|[A-Za-z0-9+/=-]{12,}(?![A-Za-z0-9_./()]))",
        re.I,
    ),
    re.compile(r"\beyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{10,}\b"),
)


def is_sensitive_path(path: Path | str) -> bool:
    """Whether a path may contain authentication/session material.

    The policy intentionally favors non-disclosure over recall. It applies to
    paths, not content, so ordinary documents elsewhere on the machine remain
    readable and indexable.
    """
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except OSError:
        resolved = Path(path).expanduser()
    parts = [part.lower() for part in resolved.parts]
    name = resolved.name.lower()
    if any(part in _SENSITIVE_DIRS for part in parts):
        return True
    if any(part in _BROWSER_PARTS for part in parts):
        return True
    if "application support" in parts and any(part in _BROWSER_PARTS for part in parts):
        return True
    return (
        name in _SENSITIVE_NAMES
        or name.startswith(".env")
        or "password" in name
        or "secret" in name
        or name.endswith(tuple(_SENSITIVE_SUFFIXES))
    )


def contains_secret_content(text: str) -> bool:
    """Conservative DLP check for actual credential-shaped text.

    It is deliberately a second layer, not a claim that arbitrary secret
    values can be recognized perfectly. Path policy still catches known secret
    stores; this protects innocently named text files from common key formats.
    """
    # OCR commonly splits a token across adjacent lines/cells.  Inspect a
    # whitespace-joined view too, while retaining the original view for
    # assignment-style patterns that rely on spacing.
    compact = re.sub(r"\s+", "", text)
    return any(pattern.search(text) or pattern.search(compact) for pattern in _SECRET_CONTENT_PATTERNS)


# Longest secret-shaped match any pattern above can produce contiguously
# (assignment-style pattern: key name + separator + 12+ char value, generously
# rounded up). A chunked scan must carry at least this much trailing context
# forward, or a credential straddling a chunk boundary can slip past both
# halves without ever appearing whole in either one.
SECRET_SCAN_OVERLAP_CHARS = 256


class StreamingSecretScanner:
    """Chunk-wise ``contains_secret_content`` with boundary overlap.

    Lets a caller scan an arbitrarily large document in small pieces (a PDF
    page, a text-file line) without ever holding the whole body in memory,
    while still catching a credential that straddles two pieces.
    """

    def __init__(self) -> None:
        self._tail = ""
        self.found = False

    def feed(self, chunk: str) -> bool:
        if self.found or not chunk:
            return self.found
        window = self._tail + chunk
        if contains_secret_content(window):
            self.found = True
            return True
        self._tail = window[-SECRET_SCAN_OVERLAP_CHARS:]
        return False


def path_is_or_contains_secret(path: Path | str) -> bool:
    path = Path(path)
    if is_sensitive_path(path):
        return True
    try:
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            return False
        data = path.read_bytes()
    except OSError:
        return False
    if b"\0" in data[:4096]:
        return False
    return contains_secret_content(data.decode("utf-8", errors="replace"))
