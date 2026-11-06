"""Knowledge-base synchronisation for standalone MCP-RagDoc.

Normal path:
    rag_document/**/<document> -> streaming text/OCR units -> SQLite FTS5 chunks

Files placed in the configured source root or its visible descendants are the
source of truth. Large documents are never materialized as one full Markdown
body before indexing.
Each changed source is indexed in its own SQLite transaction.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
import base64
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time

import config
from helpers.document_extract import (
    iter_document_units,
    iter_document_units_with_locations,
    is_supported_document,
)
from helpers.sensitive_paths import (
    StreamingSecretScanner,
    is_sensitive_path,
    path_is_or_contains_secret,
)
from helpers.thai_text_normalize import normalize_extracted_text
from helpers.thai_word_break import bounded_pieces, safe_overlap_suffix
from helpers.vision_ocr import read_layout
from ocr_rewrite import rewrite_ocr_pages, vision_server_lease


_RAW_TEXT_SUFFIXES = frozenset({
    ".md", ".markdown", ".mdown", ".mkdn", ".txt", ".text", ".rtf",
    ".csv", ".tsv", ".tex", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".html", ".htm", ".xml", ".log",
})
_RAW_IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".gif", ".webp",
})
_BINARY_MARKER = b"\x00"
_TEXT_READ_CHARS = 16 * 1024
_SECRET_SCAN_CHARS = 16 * 1024
# Location metadata is persisted in every chunk. Bump this whenever the
# extractor's page/line mapping changes so an older units snapshot is rebuilt
# instead of being treated as unchanged.
_KB_PIPELINE_VERSION = 8
_OCR_ALT_CONFIDENCE_THRESHOLD = 0.85
_OCR_ALT_MAX_CANDIDATES = 3

Progress = Callable[[str, Path | None, dict[str, int | float]], None]


class _SensitiveDocument(RuntimeError):
    pass


def _pipeline_signature() -> int:
    """Version extraction semantics and the chunk policy in one integer."""
    target = int(config.KB_CHUNK_TARGET_CHARS)
    overlap = int(config.KB_CHUNK_OVERLAP_CHARS)
    return (_KB_PIPELINE_VERSION << 56) | (target << 28) | overlap


def _unique_dirs(primary: Path, name: str) -> list[Path]:
    candidates = [primary, config.KNOWLEDGE_DIR / name, config.KNOWLEDGE_DIR / name.upper()]
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        same_physical = False
        for existing in result:
            try:
                if candidate.exists() and existing.exists() and candidate.samefile(existing):
                    same_physical = True
                    break
            except OSError:
                pass
        if same_physical:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _source_dirs() -> list[Path]:
    """Return source roots; production uses the configured source root directly.

    The compatibility alias KNOWLEDGE_RAW_DIR points at the configured source
    directory. A host adapter may override it for legacy tests, but standalone
    MCP-RagDoc production always uses one configured source root.
    """
    primary = config.KNOWLEDGE_RAW_DIR
    try:
        if primary.resolve(strict=False) == config.KNOWLEDGE_DIR.resolve(strict=False):
            return [config.KNOWLEDGE_DIR]
    except OSError:
        pass
    return _unique_dirs(primary, "raw")


def _is_supported_source(path: Path) -> bool:
    suffix = path.suffix.casefold()
    return suffix in _RAW_TEXT_SUFFIXES or suffix in _RAW_IMAGE_SUFFIXES or is_supported_document(path)


def _iter_source_files() -> Iterator[tuple[str, Path]]:
    seen_paths: set[str] = set()
    knowledge_root = config.KNOWLEDGE_DIR.resolve(strict=False)
    for root in _source_dirs():
        if not root.is_dir():
            continue
        source_root = root.resolve(strict=True)
        primary_root = source_root == knowledge_root
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(dirpath)
            kept: list[str] = []
            for name in dirnames:
                child = current / name
                try:
                    if (
                        not name.startswith(".")
                        and not child.is_symlink()
                        and not is_sensitive_path(child)
                    ):
                        kept.append(name)
                except OSError:
                    continue
            # The configured source root is recursive. Hidden, symlinked, and
            # sensitive directories remain outside the document boundary.
            dirnames[:] = kept
            for name in filenames:
                if name.startswith("."):
                    continue
                path = current / name
                try:
                    if path.is_symlink() or not path.is_file() or is_sensitive_path(path) or not _is_supported_source(path):
                        continue
                    resolved = path.resolve(strict=True)
                    resolved_key = str(resolved)
                    if resolved_key in seen_paths:
                        continue
                    relative = resolved.relative_to(source_root)
                    source_key = (
                        relative.as_posix()
                        if primary_root
                        else (root.name + "/" + relative.as_posix()).replace("\\", "/")
                    )
                except (OSError, ValueError):
                    continue
                seen_paths.add(resolved_key)
                yield source_key, resolved


def _pdf_requires_ocr(path: Path) -> bool:
    """Detect an image-only/scanned PDF without running OCR twice."""
    if path.suffix.casefold() != ".pdf":
        return False
    try:
        import fitz

        with fitz.open(path) as pdf:
            return any(not page.get_text("text").strip() for page in pdf)
    except Exception:
        # The normal extraction path remains the source of truth for malformed
        # PDFs; do not start a model merely because a preflight probe failed.
        return False


# Compatibility aliases for older tests/callers. New RAG code should use the
# source-oriented names above; production no longer has a raw-directory stage.
def _raw_dirs() -> list[Path]:
    return _source_dirs()


def _is_supported_raw(path: Path) -> bool:
    return _is_supported_source(path)


def _iter_raw_files() -> Iterator[tuple[str, Path]]:
    yield from _iter_source_files()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_kind(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in _RAW_IMAGE_SUFFIXES:
        return "Image"
    if suffix == ".pdf":
        return "PDF"
    if suffix in {".doc", ".docx", ".docm", ".odt"}:
        return "Word"
    if suffix in {".xls", ".xlsx", ".xlsm", ".ods"}:
        return "Excel"
    if suffix in {".ppt", ".pptx", ".pptm", ".odp"}:
        return "PowerPoint"
    return "Text"


def _connect_kb() -> sqlite3.Connection:
    config.FILE_INDEX_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kb_documents ("
        "path TEXT PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
        "kind TEXT NOT NULL, suffix TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
        "mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
        "pipeline_version INTEGER NOT NULL)"
    )
    existing_chunk_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='kb_chunks_fts'"
    ).fetchone()
    if existing_chunk_sql:
        # sqlite_master preserves harmless case/whitespace differences from
        # CREATE VIRTUAL TABLE. Normalize both before recognizing the v3
        # content-unindexed/search_text contract so an equivalent schema is not
        # destructively dropped and rebuilt.
        existing_sql = " ".join((existing_chunk_sql[0] or "").casefold().split())
        if "content unindexed" not in existing_sql or "search_text" not in existing_sql:
            # v3 separates primary OCR text from search-only alternatives. The
            # pipeline signature forces every document to be reindexed afterward.
            conn.execute("DROP TABLE kb_chunks_fts")
            # Parent chunk identities are no longer trustworthy after an FTS
            # schema rebuild; discard only the derived semantic state.
            conn.execute("DROP TABLE IF EXISTS kb_semantic_chunks")
            conn.execute("DROP TABLE IF EXISTS kb_semantic_state")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5("
        "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, "
        "content UNINDEXED, search_text, tokenize='trigram')"
    )
    # Semantic vectors are optional acceleration/retrieval state.  Keep their
    # lifecycle in the same SQLite source of truth so a changed/deleted source
    # cannot leave stale vectors behind even when the ONNX worker is absent.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kb_semantic_chunks ("
        "path TEXT NOT NULL, parent_chunk_index INTEGER NOT NULL, child_index INTEGER NOT NULL, "
        "model_tag TEXT NOT NULL, content_hash TEXT NOT NULL, embedding BLOB NOT NULL, "
        "generation INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY(path, parent_chunk_index, child_index))"
    )
    semantic_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(kb_semantic_chunks)")
    }
    if "generation" not in semantic_columns:
        conn.execute(
            "ALTER TABLE kb_semantic_chunks ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS kb_semantic_parent_idx "
        "ON kb_semantic_chunks(path, parent_chunk_index)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kb_semantic_state ("
        "id INTEGER PRIMARY KEY CHECK(id = 1), generation INTEGER NOT NULL DEFAULT 0, "
        "model_tag TEXT NOT NULL DEFAULT '', ready INTEGER NOT NULL DEFAULT 0, "
        "vector_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO kb_semantic_state(id,generation,model_tag,ready,vector_count) "
        "VALUES(1,0,'',0,0)"
    )
    conn.commit()
    return conn


def _invalidate_semantic_state(conn: sqlite3.Connection) -> None:
    """Mark derived semantic coverage unusable after lexical source changes."""
    conn.execute(
        "INSERT INTO kb_semantic_state(id,generation,model_tag,ready,vector_count) "
        "VALUES(1,0,'',0,0) ON CONFLICT(id) DO UPDATE SET ready=0"
    )


def _line_location(first: int, last: int) -> str:
    return f"line {first}" if first == last else f"lines {first}-{last}"


def _iter_text_units(path: Path) -> Iterator[tuple[int, str, str, str]]:
    try:
        with path.open("rb") as stream:
            if _BINARY_MARKER in stream.read(4096):
                raise RuntimeError("text file contains binary data")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            block = 0
            current_line = 1
            while True:
                text = stream.read(_TEXT_READ_CHARS)
                if not text:
                    break
                block += 1
                first_line = current_line
                newline_count = text.count("\n")
                last_line = first_line + newline_count
                if newline_count and text.endswith("\n"):
                    last_line -= 1
                last_line = max(first_line, last_line)
                yield block, text, text, _line_location(first_line, last_line)
                current_line += newline_count
    except OSError as exc:
        raise RuntimeError(f"read text failed: {exc}") from exc


def _ocr_pdf_page(page, page_number: int) -> tuple[list[dict], str]:
    temporary: Path | None = None
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="kb-pdf-page-", suffix=".png", dir=config.DATA_DIR, delete=False
        ) as handle:
            temporary = Path(handle.name)
        page.get_pixmap(dpi=config.OCR_REWRITE_IMAGE_DPI).save(str(temporary))
        image_url = "data:image/png;base64," + base64.b64encode(temporary.read_bytes()).decode("ascii")
        return read_layout(str(temporary)), image_url
    except Exception as exc:
        raise RuntimeError(f"OCR PDF page {page_number} failed: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ocr_primary_and_search(item: dict) -> tuple[str, str]:
    """Keep Vision's primary text verbatim while adding bounded search variants."""
    primary = str(item.get("text") or "").strip()
    if not primary:
        return "", ""
    variants = [primary]
    confidence = item.get("confidence")
    candidate_limit = 1
    if isinstance(confidence, (int, float)):
        # Vision can report 1.0 while still offering a materially different
        # second candidate. Keep top-2 even at high confidence so short Thai
        # words (where one edit destroys the only trigram) remain searchable;
        # expand to top-3 only when Vision itself is less confident.
        candidate_limit = 3 if confidence < _OCR_ALT_CONFIDENCE_THRESHOLD else 2
    for candidate in item.get("candidates") or ():
        candidate = str(candidate).strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
        if len(variants) >= min(candidate_limit, _OCR_ALT_MAX_CANDIDATES):
            break
    return primary, " | ".join(variants)


def _merge_ocr_search(primary: str, original_search: str) -> str:
    if primary == original_search:
        return primary
    return f"{primary} | {original_search}"


def _flush_ocr_pages(
    pages: list[tuple[int, str, str, str]],
) -> Iterator[tuple[int, str, str, str]]:
    rewrites = rewrite_ocr_pages((page, primary, image_url) for page, primary, _, image_url in pages)
    for page, primary, original_search, _image_url in pages:
        final = rewrites.get(page, primary)
        yield page, final, _merge_ocr_search(final, original_search), f"page {page}"


def _iter_pdf_units(path: Path) -> Iterator[tuple[int, str, str, str]]:
    try:
        import fitz
        with fitz.open(path) as pdf:
            total = len(pdf)
            pending_ocr: list[tuple[int, str, str, str]] = []
            for index, page in enumerate(pdf, 1):
                text = page.get_text("text").strip()
                header = f"--- PDF page {index} ---"
                location = f"page {index}/{total}" if total else f"page {index}"
                if text:
                    if pending_ocr:
                        for page_number, primary, search_text, page_location in _flush_ocr_pages(pending_ocr):
                            yield page_number, f"--- PDF page {page_number} ---\n{primary}", (
                                f"--- PDF page {page_number} ---\n{search_text}"
                            ), page_location
                        pending_ocr = []
                    primary = f"{header}\n{text}"
                    yield index, primary, primary, location
                    continue
                items, image_url = _ocr_pdf_page(page, index)
                primary_lines: list[str] = []
                search_lines: list[str] = []
                for item in items:
                    primary, search_text = _ocr_primary_and_search(item)
                    if not primary:
                        continue
                    primary_lines.append(primary)
                    search_lines.append(search_text)
                if not primary_lines:
                    if pending_ocr:
                        for page_number, primary, search_text, page_location in _flush_ocr_pages(pending_ocr):
                            yield page_number, f"--- PDF page {page_number} ---\n{primary}", (
                                f"--- PDF page {page_number} ---\n{search_text}"
                            ), page_location
                        pending_ocr = []
                    yield index, header, header, location
                    continue
                pending_ocr.append((index, "\n".join(primary_lines), "\n".join(search_lines), image_url))
                if len(pending_ocr) >= config.OCR_REWRITE_BATCH_PAGES:
                    for page_number, primary, search_text, page_location in _flush_ocr_pages(pending_ocr):
                        yield page_number, f"--- PDF page {page_number} ---\n{primary}", (
                            f"--- PDF page {page_number} ---\n{search_text}"
                        ), f"page {page_number}/{total}" if total else page_location
                    pending_ocr = []
            if pending_ocr:
                for page_number, primary, search_text, page_location in _flush_ocr_pages(pending_ocr):
                    yield page_number, f"--- PDF page {page_number} ---\n{primary}", (
                        f"--- PDF page {page_number} ---\n{search_text}"
                    ), f"page {page_number}/{total}" if total else page_location
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def _iter_image_units(path: Path) -> Iterator[tuple[int, str, str]]:
    try:
        if path.stat().st_size > config.READ_IMAGE_MAX_BYTES:
            raise RuntimeError(f"image exceeds {config.READ_IMAGE_MAX_BYTES:,} bytes")
        items = read_layout(str(path))
    except Exception as exc:
        raise RuntimeError(f"image OCR failed: {exc}") from exc
    if not items:
        yield 1, "", ""
        return
    for item in items:
        primary, search_text = _ocr_primary_and_search(item)
        if primary:
            yield 1, primary, search_text


def _iter_source_units(path: Path) -> Iterator[tuple[int, str, str] | tuple[int, str, str, str]]:
    suffix = path.suffix.casefold()
    if suffix in _RAW_TEXT_SUFFIXES:
        yield from _iter_text_units(path)
        return
    if suffix in _RAW_IMAGE_SUFFIXES:
        yield from _iter_image_units(path)
        return
    if suffix == ".pdf":
        yield from _iter_pdf_units(path)
        return
    if suffix in {".docx", ".docm"}:
        for index, (location, unit) in enumerate(iter_document_units_with_locations(path), 1):
            yield index, unit, unit, location
        return
    if suffix in {".doc", ".odt"}:
        for index, unit in enumerate(iter_document_units(path), 1):
            lines = max(1, len(unit.splitlines()))
            yield index, unit, unit, _line_location(1, lines)
        return
    if is_supported_document(path):
        for index, unit in enumerate(iter_document_units(path), 1):
            yield index, unit, unit
        return
    raise RuntimeError(f"unsupported source: {path.suffix or '(none)'}")

_PAGE_LOCATION_RE = re.compile(r"^page (\d+)(?:/(\d+))?$", re.IGNORECASE)
_LINE_LOCATION_RE = re.compile(r"^(?:line|lines) (\d+)(?:-(\d+))?$", re.IGNORECASE)


def _location(
    path: Path,
    first: int,
    last: int,
    first_location: str | None = None,
    last_location: str | None = None,
) -> str:
    if first_location:
        if first_location == last_location or not last_location:
            return first_location
        first_match = _PAGE_LOCATION_RE.fullmatch(first_location.strip())
        last_match = _PAGE_LOCATION_RE.fullmatch(last_location.strip())
        if first_match and last_match:
            first_page, first_total = first_match.groups()
            last_page, last_total = last_match.groups()
            if first_total == last_total and first_page == last_page:
                return first_location
            if first_total == last_total:
                suffix = f"/{first_total}" if first_total else ""
                return f"pages {first_page}-{last_page}{suffix}"
        first_line_match = _LINE_LOCATION_RE.fullmatch(first_location.strip())
        last_line_match = _LINE_LOCATION_RE.fullmatch(last_location.strip())
        if first_line_match and last_line_match:
            first_line = int(first_line_match.group(1))
            last_line = int(last_line_match.group(2) or last_line_match.group(1))
            return _line_location(first_line, last_line)
        return f"{first_location} to {last_location}"
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return f"page {first}" if first == last else f"pages {first}-{last}"
    if suffix in _RAW_IMAGE_SUFFIXES:
        return "image"
    if suffix in _RAW_TEXT_SUFFIXES:
        return f"text block {first}" if first == last else f"text blocks {first}-{last}"
    return f"unit {first}" if first == last else f"units {first}-{last}"


def _feed_secret_scanner(scanner: StreamingSecretScanner, text: str) -> None:
    for start in range(0, len(text), _SECRET_SCAN_CHARS):
        if scanner.feed(text[start:start + _SECRET_SCAN_CHARS]):
            raise _SensitiveDocument("credential-shaped content")


def _iter_search_chunks(path: Path) -> Iterator[tuple[str, str, str]]:
    target = config.KB_CHUNK_TARGET_CHARS
    overlap = config.KB_CHUNK_OVERLAP_CHARS
    scanner = StreamingSecretScanner()
    buffer = ""
    search_buffer = ""
    first_unit: int | None = None
    last_unit: int | None = None
    first_location: str | None = None
    last_location: str | None = None
    emitted = 0
    has_new_text = False

    for source_unit in _iter_source_units(path):
        # Keep compatibility with narrow tests/extensions that still yield the
        # historical (unit_number, text) pair.
        if len(source_unit) == 2:
            unit_number, raw_unit = source_unit
            raw_search = raw_unit
            source_location = None
        elif len(source_unit) == 3:
            unit_number, raw_unit, raw_search = source_unit
            source_location = None
        else:
            unit_number, raw_unit, raw_search, source_location = source_unit
        normalized = normalize_extracted_text(raw_unit).strip()
        if not normalized:
            continue
        normalized_search = normalize_extracted_text(raw_search).strip() or normalized
        _feed_secret_scanner(scanner, normalized_search)

        # OCR alternatives must stay attached to their primary line so search
        # variants do not leak into user-visible content. Ordinary Thai units
        # are split with PyThaiNLP/newmm at whole-word boundaries, matching the
        # proven RAG MAX policy; only one pathological >target token is hard-cut.
        has_alternatives = normalized_search != normalized and len(normalized) <= target
        if has_alternatives:
            pieces = [(normalized, normalized_search)]
        else:
            pieces = [(piece, piece) for piece in bounded_pieces(normalized, target)]

        # A page-aware source must never make one persisted chunk span two
        # pages: the result needs to point the user to one exact page. Text
        # sources may span lines, so their location remains a line range.
        if (
            buffer
            and source_location
            and last_location
            and source_location != last_location
            and _PAGE_LOCATION_RE.fullmatch(source_location.strip())
            and _PAGE_LOCATION_RE.fullmatch(last_location.strip())
        ):
            assert first_unit is not None and last_unit is not None
            yield (
                _location(path, first_unit, last_unit, first_location, last_location),
                buffer,
                search_buffer,
            )
            emitted += 1
            buffer = ""
            search_buffer = ""
            first_unit = None
            last_unit = None
            first_location = None
            last_location = None
            has_new_text = False

        first_piece = True
        for primary_piece, search_piece in pieces:
            separator = "\n\n" if buffer and first_piece else ""
            if buffer and len(buffer) + len(separator) + len(primary_piece) > target:
                assert first_unit is not None and last_unit is not None
                yield (
                    _location(path, first_unit, last_unit, first_location, last_location),
                    buffer,
                    search_buffer,
                )
                emitted += 1
                tail_unit = last_unit
                tail_location = last_location
                buffer = safe_overlap_suffix(buffer, overlap) if overlap else ""
                # Search-only OCR alternatives were already searchable in the
                # emitted row. Keep overlap aligned to primary text rather than
                # trying to map a longer alternatives string back to characters.
                search_buffer = buffer
                first_unit = tail_unit if buffer else None
                last_unit = tail_unit if buffer else None
                first_location = tail_location if buffer else None
                last_location = tail_location if buffer else None
                has_new_text = False
                separator = "\n\n" if buffer and first_piece else ""
                if len(buffer) + len(separator) + len(primary_piece) > target:
                    # A valid whole word/piece fits the hard chunk size but not
                    # the retained overlap: drop overlap rather than split it.
                    buffer = ""
                    search_buffer = ""
                    first_unit = None
                    last_unit = None
                    separator = ""

            if first_unit is None:
                first_unit = unit_number
                first_location = source_location
            buffer += separator + primary_piece
            search_buffer += separator + search_piece
            last_unit = unit_number
            last_location = source_location or last_location
            has_new_text = True

            if len(buffer) == target:
                yield (
                    _location(path, first_unit, last_unit, first_location, last_location),
                    buffer,
                    search_buffer,
                )
                emitted += 1
                tail_unit = last_unit
                tail_location = last_location
                buffer = safe_overlap_suffix(buffer, overlap) if overlap else ""
                search_buffer = buffer
                first_unit = tail_unit if buffer else None
                last_unit = tail_unit if buffer else None
                first_location = tail_location if buffer else None
                last_location = tail_location if buffer else None
                has_new_text = False
            first_piece = False

    if buffer.strip() and (has_new_text or emitted == 0):
        assert first_unit is not None and last_unit is not None
        yield (
            _location(path, first_unit, last_unit, first_location, last_location),
            buffer,
            search_buffer,
        )

def _progress(
    callback: Progress | None,
    event: str,
    path: Path | None,
    stats: dict[str, int | float],
) -> None:
    if callback is None:
        return
    try:
        callback(event, path, dict(stats))
    except Exception:
        pass


def _update_progress_metrics(
    stats: dict[str, int | float], started: float, *, final: bool = False,
) -> None:
    """Attach bounded, file-based progress estimates to callback snapshots.

    Extraction/OCR time varies by document, so this is deliberately labelled
    an estimate in the UI.  The percentage is based on completed source files,
    not LLM tokens or bytes, which keeps it deterministic and cheap to report.
    """
    total = int(stats.get("total_files", 0) or 0)
    completed = int(stats.get("completed_files", 0) or 0)
    if final:
        percent = 100.0
    elif total <= 0:
        percent = 0.0
    else:
        # Reserve the last point for stale-entry cleanup and SQLite finalization.
        percent = min(99.0, round(completed / total * 99.0, 1))
    stats["progress_percent"] = percent
    elapsed = max(0.0, time.time() - started)
    if completed > 0 and total > completed and elapsed > 0:
        stats["estimated_remaining_seconds"] = round(
            elapsed * (total - completed) / completed, 1
        )
    else:
        stats["estimated_remaining_seconds"] = 0.0


class SyncCancelled(RuntimeError):
    """Raised only at safe sync checkpoints after an explicit cancel request."""


CancelCheck = Callable[[], bool]


def _check_cancel(cancel_requested: CancelCheck | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise SyncCancelled("sync cancelled")


def _index_one(
    conn: sqlite3.Connection,
    *,
    source_key: str,
    path: Path,
    raw_hash: str,
    stat,
    cancel_requested: CancelCheck | None = None,
) -> tuple[str, int]:
    stored = conn.execute(
        "SELECT sha256, pipeline_version, chunk_count FROM kb_documents WHERE path = ?",
        (str(path),),
    ).fetchone()
    pipeline_signature = _pipeline_signature()
    if stored and stored[0] == raw_hash and stored[1] == pipeline_signature:
        return "unchanged", int(stored[2])

    existed = stored is not None
    kind = _document_kind(path)
    chunk_count = 0
    with conn:
        _check_cancel(cancel_requested)
        _invalidate_semantic_state(conn)
        conn.execute("DELETE FROM kb_chunks_fts WHERE path = ?", (str(path),))
        conn.execute("DELETE FROM kb_semantic_chunks WHERE path = ?", (str(path),))
        for chunk_index, (location, content, search_text) in enumerate(_iter_search_chunks(path)):
            _check_cancel(cancel_requested)
            conn.execute(
                "INSERT INTO kb_chunks_fts(path, chunk_index, name, location, content, search_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(path), chunk_index, path.name, location, content, search_text),
            )
            chunk_count += 1
        if chunk_count == 0:
            conn.execute(
                "INSERT INTO kb_chunks_fts(path, chunk_index, name, location, content, search_text) "
                "VALUES (?, 0, ?, ?, '', '')",
                (str(path), path.name, "image" if kind == "Image" else "document"),
            )
            chunk_count = 1
        _check_cancel(cancel_requested)
        # Extraction may outlive a source edit or replacement. Publish only
        # the snapshot whose fingerprint was supplied, otherwise roll back
        # this document (including semantic invalidation) for the next sync.
        from helpers.source_paths import visible_source_file
        if not any(visible_source_file(path, root) is not None for root in _source_dirs()):
            raise RuntimeError("source changed visibility during extraction")
        final_stat = path.stat()
        if (
            final_stat.st_size != stat.st_size
            or final_stat.st_mtime_ns != stat.st_mtime_ns
            or _sha256(path) != raw_hash
        ):
            raise RuntimeError("source changed during extraction")
        conn.execute(
            "INSERT INTO kb_documents("
            "path, source_key, name, kind, suffix, size_bytes, mtime_ns, sha256, chunk_count, pipeline_version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "source_key=excluded.source_key, name=excluded.name, kind=excluded.kind, "
            "suffix=excluded.suffix, size_bytes=excluded.size_bytes, mtime_ns=excluded.mtime_ns, "
            "sha256=excluded.sha256, chunk_count=excluded.chunk_count, "
            "pipeline_version=excluded.pipeline_version",
            (
                str(path), source_key, path.name, kind, path.suffix.casefold(),
                int(stat.st_size), int(stat.st_mtime_ns), raw_hash,
                chunk_count, pipeline_signature,
            ),
        )
    return ("updated" if existed else "inserted"), chunk_count


def sync_kb(
    *,
    verbose: bool = False,
    on_progress: Progress | None = None,
    cancel_requested: CancelCheck | None = None,
) -> dict[str, int | float]:
    del verbose

    stats: dict[str, int | float] = {
        "raw_scanned": 0,
        "total_files": 0,
        "completed_files": 0,
        "progress_percent": 0.0,
        "estimated_remaining_seconds": 0.0,
        "indexed": 0,
        "converted": 0,
        "unchanged": 0,
        "removed": 0,
        "skipped": 0,
        "sync_errors": 0,
        "documents": 0,
        "chunks": 0,
        "inserted": 0,
        "updated": 0,
        "error": 0,
        "md_files": 0,
        "ocr_pdf_files": 0,
    }
    started = time.time()
    seen_paths: set[str] = set()
    _check_cancel(cancel_requested)
    conn = _connect_kb()
    vision_lease = None
    lease_entered = False
    try:
        _progress(on_progress, "prepare", None, stats)
        _check_cancel(cancel_requested)
        source_files = list(_iter_source_files())
        stats["total_files"] = len(source_files)
        ocr_pdf_count = sum(1 for _, path in source_files if _pdf_requires_ocr(path))
        stats["ocr_pdf_files"] = ocr_pdf_count
        vision_lease = vision_server_lease(required=ocr_pdf_count > 0)
        vision_lease.__enter__()
        lease_entered = True
        _update_progress_metrics(stats, started)
        _progress(on_progress, "prepared", None, stats)
        for source_key, raw_path in source_files:
            _check_cancel(cancel_requested)
            seen_paths.add(str(raw_path))
            stats["raw_scanned"] += 1
            _progress(on_progress, "file", raw_path, stats)
            result: str | None = None
            chunk_count = 0
            file_completed = False
            try:
                from helpers.source_paths import visible_source_file
                if not any(visible_source_file(raw_path, root) is not None for root in _source_dirs()):
                    raise _SensitiveDocument("source changed visibility after discovery")
                if path_is_or_contains_secret(raw_path):
                    raise _SensitiveDocument("sensitive path/content")
                raw_hash = _sha256(raw_path)
                raw_stat = raw_path.stat()
                result, chunk_count = _index_one(
                    conn,
                    source_key=source_key,
                    path=raw_path,
                    raw_hash=raw_hash,
                    stat=raw_stat,
                    cancel_requested=cancel_requested,
                )
            except SyncCancelled:
                raise
            except _SensitiveDocument:
                # A source that is now credential-shaped must fail closed.
                # Do not leave an older searchable snapshot pointing at a
                # document whose current contents are sensitive.
                existed = conn.execute(
                    "SELECT 1 FROM kb_documents WHERE path = ?", (str(raw_path),)
                ).fetchone() is not None
                with conn:
                    _invalidate_semantic_state(conn)
                    conn.execute("DELETE FROM kb_chunks_fts WHERE path = ?", (str(raw_path),))
                    conn.execute("DELETE FROM kb_semantic_chunks WHERE path = ?", (str(raw_path),))
                    conn.execute("DELETE FROM kb_documents WHERE path = ?", (str(raw_path),))
                if existed:
                    stats["removed"] += 1
                stats["sync_errors"] += 1
                stats["error"] += 1
                stats["skipped"] += 1
                file_completed = True
            except Exception:
                # Ordinary extraction/OCR failures keep the last committed
                # version: _index_one's per-document transaction rolls back.
                stats["sync_errors"] += 1
                stats["error"] += 1
                stats["skipped"] += 1
                file_completed = True
            else:
                if result == "unchanged":
                    stats["unchanged"] += 1
                else:
                    stats["indexed"] += 1
                    stats["converted"] += 1
                    stats[result] += 1
                stats["chunks"] += chunk_count
                file_completed = True
            finally:
                if file_completed:
                    stats["completed_files"] += 1
                    _update_progress_metrics(stats, started)
                    _progress(on_progress, "result", raw_path, stats)

        _check_cancel(cancel_requested)
        _progress(on_progress, "cleanup", None, stats)
        stale_paths = [
            row[0]
            for row in conn.execute("SELECT path FROM kb_documents").fetchall()
            if row[0] not in seen_paths
        ]
        for stale_path in stale_paths:
            _check_cancel(cancel_requested)
            with conn:
                _invalidate_semantic_state(conn)
                conn.execute("DELETE FROM kb_chunks_fts WHERE path = ?", (stale_path,))
                conn.execute("DELETE FROM kb_semantic_chunks WHERE path = ?", (stale_path,))
                conn.execute("DELETE FROM kb_documents WHERE path = ?", (stale_path,))
            stats["removed"] += 1
            _progress(on_progress, "cleanup", Path(stale_path), stats)

        _check_cancel(cancel_requested)
        stats["documents"] = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
        stats["chunks"] = conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0]
    finally:
        if vision_lease is not None and lease_entered:
            vision_lease.__exit__(None, None, None)
        conn.close()

    stats["seconds"] = round(time.time() - started, 2)
    _update_progress_metrics(stats, started, final=True)
    _progress(on_progress, "done", None, stats)
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(sync_kb(verbose=True), ensure_ascii=False))
