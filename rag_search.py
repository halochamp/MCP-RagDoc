"""Low-RAM hybrid retrieval core for standalone MCP-RagDoc.

Filename-first FTS5/BM25, OCR fuzzy recovery and bounded filesystem fallback
are fused with optional MiniLM semantic ranks. The heavy ONNX INT8 runtime
lives only in a short-lived subprocess; this module never imports ONNX, NumPy,
or SentencePiece. Missing semantic dependencies/models degrade to lexical
search without involving an LLM.
"""
from __future__ import annotations
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import config
import semantic_client
from helpers.sensitive_paths import is_sensitive_path, contains_secret_content, path_is_or_contains_secret
from helpers.source_paths import visible_source_file
from helpers.thai_text_normalize import normalize_extracted_text
from helpers.thai_word_break import query_words

# 2026-07-24 (user request: query robustness -- synonyms, case, spacing,
# stray characters -- pattern ported from ENDMEMEX/endeavor_db.py's
# query_terms()/fts_expression()/_QUERY_ALIASES, the same idea already
# proven for that project's own FTS5 search). Verified empirically before
# building this (not assumed): SQLite's trigram tokenizer is ALREADY
# case-insensitive by default (tested: 'REPORT'/'Report'/'report' all match
# identically), and NFC vs NFD normalization made no difference for the Thai
# test string tried -- neither needed a fix at the index level. What WAS
# genuinely missing: (1) synonym/alias expansion -- a query for "รูป" never
# found a file whose content only says "ภาพ" or "photo"; (2) stray-character
# robustness -- `q.split()` passes any punctuation/symbol run through as a
# literal FTS token, quietly contributing nothing (trigram tokenizes it) or
# even erroring; a regex-based word extractor drops that noise instead of
# feeding it to MATCH.
_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "รูป": ("ภาพ", "image", "photo", "picture"),
    "ภาพ": ("รูป", "image", "photo", "picture"),
    "เอกสาร": ("document", "ไฟล์", "doc"),
    "วิดีโอ": ("คลิป", "video", "clip"),
    "คลิป": ("วิดีโอ", "video", "clip"),
    "ใบเสร็จ": ("บิล", "receipt", "invoice"),
    "บิล": ("ใบเสร็จ", "receipt", "invoice"),
    "รายงาน": ("report",),
    "สัญญา": ("contract", "agreement"),
    "จดหมาย": ("letter", "mail"),
    "การเงิน": ("เงิน", "finance", "financial"),
    "ประวัติ": ("resume", "cv"),
}
_STOP_WORDS = {
    "หา", "ให้", "หน่อย", "ที่", "ของ", "และ", "อะไร", "บ้าง", "เรื่อง", "ไหน", "ไฟล์",
    "the", "a", "an", "of", "for", "to", "and",
}
_WORD_RE = re.compile(r"[\wก-๙]+", re.UNICODE)
_ENGLISH_SPLIT_MIN_CHARS = 5
_MAX_ENGLISH_SPLIT_VARIANTS = 24
_PROJECT_SCAN_EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".lite_state", "logs", "graphify-out",
    "build", "dist", "coverage",
}
_PROJECT_CONTENT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".markdown", ".txt", ".text", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".html", ".css",
    ".js", ".ts", ".tsx", ".jsx", ".sh", ".zsh", ".swift", ".sql",
})
_PROJECT_CONTENT_SCAN_SECONDS = 4.0
_PROJECT_CONTENT_MAX_RESULTS = 50
_PROJECT_CONTENT_DEFAULT_RESULTS = 15
_PROJECT_CONTENT_SNIPPET_CHARS = 180


def _source_marker(path: str) -> str:
    """Return the machine-readable source marker consumed by ``agent.py``.

    Web search uses ``[source: URL]`` for the same reason: the model may
    summarize a correct tool result but omit its provenance.  Keep the local
    Knowledge Base marker identical in shape while retaining the verified
    absolute filesystem path for the UI/Finder path handling.
    """
    return f"  [source: {path}]"
# Keep query normalization aligned with indexer/KB extraction: repair
# arbitrary whitespace inside Thai runs and deterministic mark-order errors
# such as นำ้ -> น้ำ. Never substitute a plausible dictionary word here.
def _query_terms(query: str) -> list[str]:
    """Normalize text, then segment Thai/mixed runs with newmm when available.

    Stray punctuation and generic filler words are dropped afterward. English
    behavior stays unchanged; PyThaiNLP is lazy/optional so a missing package
    falls back to the historical whole-run trigram query rather than failing."""
    normalized = unicodedata.normalize("NFC", query)
    normalized = re.sub(r"\s+", " ", normalized.strip().casefold())
    normalized = normalize_extracted_text(normalized)
    raw_terms = _WORD_RE.findall(normalized)
    seen: set[str] = set()
    result = []
    for raw_term in raw_terms:
        for term in query_words(raw_term):
            if term in _STOP_WORDS or term in seen:
                continue
            seen.add(term)
            result.append(term)
            if len(result) >= 12:
                return result
    return result

SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        # See tools/bash.py's matching 2026-07-23 note — reverted alongside it.
        "description": (
            "Search the configured MCP-RagDoc source files by filename, indexed keywords, "
            "OCR text, or semantic meaning when the local vector index is available. Returns "
            "absolute source paths, locations, and short matching snippets. Search never "
            "broadens outside the server's configured knowledge directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Filename, keywords, or semantic search phrase"}
            },
            "required": ["query"],
        },
    },
}


def _fts_atom(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"'


def _safe_match_query(q: str) -> str:
    """Build an FTS5 MATCH expression: each extracted query term OR'd with
    its own synonyms (if any), the per-term groups themselves OR'd together
    -- matches ANY of the user's words or their aliases, same "locator, not
    exact search" intent as before, just synonym-aware now."""
    terms = _query_terms(q)
    expressions = []
    for term in terms:
        alternatives = [_fts_atom(term), *(_fts_atom(alias) for alias in _QUERY_ALIASES.get(term, ()))]
        expressions.append(alternatives[0] if len(alternatives) == 1 else f"({' OR '.join(alternatives)})")
    return " OR ".join(expressions)


def _english_split_match_query(q: str) -> str:
    """Return bounded FTS phrases for English words split by PDF/OCR.

    The index retains ordinary English spaces, so rewriting stored text would
    corrupt valid phrases such as ``project plan``.  Instead query variants
    let ``information`` also match a stored ``infor mation``.  Only ASCII
    alphabetic words of a useful length participate and the whole query has a
    strict cap, keeping this recall path bounded.
    """
    variants: list[str] = []
    for term in _query_terms(q):
        if not (term.isascii() and term.isalpha() and len(term) >= _ENGLISH_SPLIT_MIN_CHARS):
            continue
        for position in range(1, len(term)):
            variants.append(_fts_atom(f"{term[:position]} {term[position:]}"))
            if len(variants) >= _MAX_ENGLISH_SPLIT_VARIANTS:
                return " OR ".join(variants)
    return " OR ".join(variants)


def _filename_fallback_needle(query: str) -> str:
    """Use the concise search term selected by the agent verbatim.

    Natural-language intent extraction belongs to the model/tool call, not a
    growing list of language-specific phrase rules in the fallback. This
    fallback is deliberately a literal filename scan of the tool argument.
    """
    return (query or "").strip().casefold()


def _fallback_filename_search(
    query: str, *, reason: str = "no_index", roots: list[Path] | None = None,
) -> str:
    """Fast bounded filename lookup, used two ways: (1) no index exists yet
    at all, (2) the index exists but the FTS query above matched nothing --
    a broader, code-driven retry rather than making the model notice a zero-
    result reply and manually re-issue a looser query itself (small models
    don't reliably do this, per this project's own live findings tonight)."""
    needle = _filename_fallback_needle(query)
    # Recovery is deliberately knowledge-only.  It never walks Desktop,
    # Documents, Downloads, the project tree, or another sibling project.
    # Normal runtime searches source filenames anywhere below rag_document/.
    # Explicit roots remain useful for isolated tests/developer calls.
    roots = roots or _kb_source_roots()
    deadline = time.monotonic() + 4.0
    matches: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        source_root = root.resolve(strict=False)
        configured_root = config.KNOWLEDGE_DIR.resolve(strict=False)
        recurse = source_root == configured_root or bool(
            getattr(config, "ALLOW_INDEX_ROOT_RESULTS", False)
        )
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            if time.monotonic() >= deadline or len(matches) >= 50:
                break
            current = Path(dirpath)
            dirnames[:] = [
                name for name in dirnames
                if recurse
                and not name.startswith(".")
                and not is_sensitive_path(current / name)
                and name not in {"node_modules", "__pycache__"}
            ]
            for name in filenames:
                if time.monotonic() >= deadline:
                    break
                if needle in name.casefold():
                    candidate = current / name
                    if visible_source_file(candidate, root) is not None and not path_is_or_contains_secret(candidate):
                        matches.append(str(candidate))
                        if len(matches) >= 50:
                            break
    if not matches:
        if reason == "no_index":
            return (
                "[error] ยังไม่มี Knowledge Base index และค้นใน rag_document/ ไม่พบผล — "
                "ใช้ rag_manage(action='sync') เพื่อสร้างหรืออัปเดตฐานความรู้"
            )
        return f"ไม่พบไฟล์ที่ตรงกับ '{query}' ใน Knowledge Base (ลอง rag_manage(action='sync') หากเอกสารเพิ่งเปลี่ยน)"
    suffix = "\n[หยุดหลัง 4 วินาที/50 ผล]" if time.monotonic() >= deadline or len(matches) >= 50 else ""
    header = "[knowledge filename fallback — ยังไม่มี RAG index]" if reason == "no_index" \
        else "[ขยายผลการค้นหา — ไม่พบใน index แต่พบไฟล์ชื่อใกล้เคียง]"
    formatted = []
    for match in matches:
        formatted.append(
            f"- ไฟล์: {match}\n"
            "  หน้า/ตำแหน่ง: ยังไม่ทราบ (พบจากชื่อไฟล์)\n"
            "  ตัวอย่างเนื้อหา: ยังไม่มีในดัชนี — เปิดไฟล์เพื่ออ่านเนื้อหา\n"
            f"{_source_marker(match)}"
        )
    return f"{header}\n" + "\n".join(formatted) + suffix


def _project_content_fallback(query: str) -> str:
    """Search generated Knowledge Base Markdown when FTS has no row.

    The historical function name is retained for compatibility, but it no
    longer reads project source/config files.  All fallback paths stay under
    the configured knowledge Markdown root.
    """
    terms = [term for term in _query_terms(query) if len(term) >= 2]
    if not terms:
        return ""
    compatibility_roots = bool(getattr(config, "ALLOW_INDEX_ROOT_RESULTS", False))
    root = config.KNOWLEDGE_MD_DIR if compatibility_roots else config.KNOWLEDGE_DIR
    request = query.casefold()
    exhaustive_request = bool(re.search(
        r"(?:ทั้งหมด|ครบ|ทุกไฟล์|all\s+files|every\s+file|complete|full\s+list)",
        request,
    ))
    max_results = _PROJECT_CONTENT_MAX_RESULTS if exhaustive_request else _PROJECT_CONTENT_DEFAULT_RESULTS
    deadline = time.monotonic() + _PROJECT_CONTENT_SCAN_SECONDS
    matches: list[str] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        if time.monotonic() >= deadline or len(matches) >= max_results:
            break
        current = Path(dirpath)
        # Search the configured source root recursively. Compatibility mode
        # still changes which root is searched, but not descendant visibility.
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".")
            and name not in _PROJECT_SCAN_EXCLUDED_DIRS
            and not is_sensitive_path(current / name)
        ]
        for name in filenames:
            if time.monotonic() >= deadline or len(matches) >= max_results:
                break
            path = current / name
            if path.suffix.casefold() != ".md":
                continue
            if visible_source_file(path, root) is None:
                continue
            try:
                if path.stat().st_size > 4 * 1024 * 1024:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            if contains_secret_content(text):
                continue
            normalized = unicodedata.normalize("NFC", text).casefold()
            hit = next((term for term in terms if term in normalized), None)
            if hit is None:
                continue
            path_text = str(path)
            if path_text in seen:
                continue
            seen.add(path_text)
            position = normalized.find(hit)
            start = max(0, position - 100)
            snippet = " ".join(text[start:position + len(hit) + _PROJECT_CONTENT_SNIPPET_CHARS].split())
            matches.append(
                f"- ไฟล์: {path_text}\n"
                "  หน้า/ตำแหน่ง: ไม่ระบุ (Knowledge Base Markdown)\n"
                f"  ตัวอย่างเนื้อหา: ...{snippet}...\n"
                f"{_source_marker(path_text)}"
            )
    if not matches:
        return ""
    suffix = "\n[หยุดหลังครบเวลา/จำนวนผลที่กำหนด]" if time.monotonic() >= deadline or len(matches) >= max_results else ""
    return (
        f"[ค้นจากเนื้อหาใน Knowledge Base: {root}]\n"
        f"พบ {len(matches)} ไฟล์ที่มีคำค้นจาก '{query}':\n"
        + "\n".join(matches)
        + suffix
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _kb_source_roots() -> list[Path]:
    if not bool(getattr(config, "ALLOW_INDEX_ROOT_RESULTS", False)):
        return [config.KNOWLEDGE_DIR]
    roots = [config.KNOWLEDGE_DIR, *[Path(root) for root in getattr(config, "INDEX_ROOTS", ())]]
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            result.append(root)
    return result


def _path_in_kb(path: str) -> bool:
    """Validate result scope; configured KB descendants are allowed."""
    resolved = Path(path).resolve(strict=False)
    if bool(getattr(config, "ALLOW_INDEX_ROOT_RESULTS", False)):
        return any(
            _is_relative_to(resolved, root.resolve(strict=False))
            for root in _kb_source_roots()
        )
    root = config.KNOWLEDGE_DIR.resolve(strict=False)
    return resolved != root and _is_relative_to(resolved, root) and not resolved.name.startswith(".")


def _path_in_project(path: str) -> bool:
    """Compatibility name retained for callers; RAG scope is KB-only."""
    return _path_in_kb(path)


def _is_project_artifact_path(path: str) -> bool:
    """Keep internal search/index artifacts out of user-facing results.

    The on-disk FTS database is incremental and can contain rows from an
    older index policy.  Filtering these paths at read time prevents stale
    ``logs/tool_outputs`` records from outranking a real source/config file;
    it also keeps every result focused on generated Knowledge Base Markdown.
    """
    artifact_root = (
        config.KNOWLEDGE_MD_DIR
        if bool(getattr(config, "ALLOW_INDEX_ROOT_RESULTS", False))
        else config.KNOWLEDGE_DIR
    )
    try:
        relative = Path(path).resolve(strict=False).relative_to(artifact_root.resolve())
    except ValueError:
        return False
    return any(part in _PROJECT_SCAN_EXCLUDED_DIRS for part in relative.parts)


def _allowed_result_path(path: str) -> bool:
    # The index is incremental and iCloud/other sync tools can move or remove
    # a file between index builds. Never surface a stale row: the browser's
    # clickable-path action can only open a path that still exists.
    from helpers.source_paths import visible_source_file
    if not any(visible_source_file(path, root) is not None for root in _kb_source_roots()):
        return False
    return (
        not is_sensitive_path(path)
        and _path_in_project(path)
        and not _is_project_artifact_path(path)
    )


def _kb_chunk_schema_exists(conn: sqlite3.Connection) -> bool:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name IN ('kb_documents','kb_chunks_fts')"
        )
    }
    if names != {"kb_documents", "kb_chunks_fts"}:
        return False
    columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks_fts)")}
    return {"content", "search_text"}.issubset(columns)


def _kb_metadata_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kb_documents'"
    ).fetchone() is not None


def _primary_snippet(content: str, query: str) -> str:
    """Render only primary extracted/OCR text, never search-only alternatives."""
    text = re.sub(r"\s+", " ", (content or "")).strip()
    if not text:
        return ""
    folded = text.casefold()
    normalized_query = normalize_extracted_text(query).casefold()
    exact_position = -1
    if normalized_query:
        exact_position = folded.find(normalized_query)
    if exact_position >= 0:
        center = exact_position
    else:
        positions = [
            folded.find(term.casefold()) for term in _query_terms(query)
        ]
        positions = [position for position in positions if position >= 0]
        center = min(positions) if positions else 0
    start = max(0, center - max(24, config.RAG_SNIPPET_CHARS // 3))
    snippet = text[start:start + config.RAG_SNIPPET_CHARS]
    if start:
        snippet = "..." + snippet
    if start + config.RAG_SNIPPET_CHARS < len(text):
        snippet += "..."
    return snippet


def _thai_fuzzy_terms(query: str) -> list[str]:
    return [
        term for term in _query_terms(query)
        if 4 <= len(term) <= 32 and any("\u0e00" <= char <= "\u0e7f" for char in term)
    ][:4]


def _bounded_edit_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, char_right in enumerate(right, 1):
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _fuzzy_term_match(term: str, search_text: str) -> tuple[int, int] | None:
    """Return the best bounded (distance, start) using exact trigram anchors."""
    text = (search_text or "")[:9000]
    limit = 1 if len(term) < 8 else 2
    if term in text:
        return 0, text.find(term)
    seen_starts: set[tuple[int, int]] = set()
    best: tuple[int, int] | None = None
    for query_offset in range(max(0, len(term) - 2)):
        seed = term[query_offset:query_offset + 3]
        if len(seed) < 3:
            continue
        cursor = 0
        occurrences = 0
        while occurrences < 8:
            position = text.find(seed, cursor)
            if position < 0:
                break
            occurrences += 1
            cursor = position + 1
            estimated = position - query_offset
            for shift in range(-limit, limit + 1):
                start = estimated + shift
                if start < 0:
                    continue
                for width in range(max(1, len(term) - limit), len(term) + limit + 1):
                    key = (start, width)
                    if key in seen_starts or start + width > len(text):
                        continue
                    seen_starts.add(key)
                    distance = _bounded_edit_distance(term, text[start:start + width], limit)
                    if distance <= limit and (best is None or distance < best[0]):
                        best = (distance, start)
                        # Exact matches were handled above, so distance 1 is the
                        # best remaining score and no later window can improve it.
                        if distance == 1:
                            return best
    return best


def _ocr_fuzzy_rows(conn: sqlite3.Connection, query: str) -> list[tuple]:
    terms = _thai_fuzzy_terms(query)
    if not terms:
        return []
    seeds: list[str] = []
    for term in terms:
        term_seeds = [term[index:index + 3] for index in range(max(0, len(term) - 2))]
        if term_seeds:
            # First/middle/last anchors give broad recall without a large MATCH expression.
            picks = [term_seeds[0], term_seeds[len(term_seeds) // 2], term_seeds[-1]]
            for seed in picks:
                if seed not in seeds:
                    seeds.append(seed)
                if len(seeds) >= 8:
                    break
        if len(seeds) >= 8:
            break
    if not seeds:
        return []
    # Query each trigram anchor separately so one very common Thai seed cannot
    # consume the entire candidate limit before a rarer surviving seed is seen.
    candidates: list[tuple] = []
    candidate_keys: set[tuple[str, str, str]] = set()
    for seed in seeds:
        rows = conn.execute(
            "SELECT c.path, c.name, c.location, c.content, c.search_text, "
            "d.size_bytes, d.kind FROM kb_chunks_fts AS c "
            "JOIN kb_documents AS d ON d.path = c.path "
            "WHERE d.kind IN ('PDF','Image') AND kb_chunks_fts MATCH ? "
            "ORDER BY bm25(kb_chunks_fts) LIMIT 8",
            (_fts_atom(seed),),
        ).fetchall()
        for row in rows:
            key = (row[0], row[2], row[4])
            if key in candidate_keys:
                continue
            candidate_keys.add(key)
            candidates.append(row)
            if len(candidates) >= 48:
                break
        if len(candidates) >= 48:
            break
    scored: list[tuple[int, tuple]] = []
    seen: set[tuple[str, str]] = set()
    for path, name, location, content, search_text, size_bytes, kind in candidates:
        if not _allowed_result_path(path):
            continue
        key = (path, location)
        if key in seen:
            continue
        best: int | None = None
        for term in terms:
            match = _fuzzy_term_match(term, search_text or "")
            if match is not None and (best is None or match[0] < best):
                best = match[0]
        if best is None:
            continue
        seen.add(key)
        scored.append((best, (path, name, location, content, size_bytes, kind)))
    scored.sort(key=lambda item: item[0])
    return [row for _score, row in scored[:config.RAG_MAX_RESULTS]]


def _format_chunk_rows(rows: list[tuple], query: str, *, fuzzy: bool = False) -> str:
    if fuzzy:
        lines = [f"พบ {len(rows)} ช่วงข้อความ OCR ที่ใกล้เคียงกับ '{query}':"]
    else:
        lines = [f"พบ {len(rows)} ช่วงข้อความที่ตรงกับ '{query}':"]
    for path, name, location, content, size_bytes, kind in rows:
        snippet = _primary_snippet(content or "", query)
        metadata = f"{kind or 'Document'}, {location or 'unknown location'}, {(size_bytes or 0):,} bytes"
        if fuzzy:
            metadata += ", OCR fuzzy"
        lines.append(
            f"- ไฟล์: {path}\n"
            f"  ชื่อไฟล์: {name}\n"
            f"  หน้า/ตำแหน่ง: {location or 'ไม่ระบุ'}\n"
            f"{_source_marker(path)}\n"
            f"  ตัวอย่างเนื้อหา: {snippet or '(ไม่มีข้อความตัวอย่าง)'}\n"
            f"  รายละเอียด: {metadata}"
        )
    return "\n".join(lines)


def _semantic_parent_rows(conn: sqlite3.Connection, query: str) -> list[tuple]:
    """Resolve semantic worker parent IDs back to ordinary KB result rows."""
    # Never spawn the transient ONNX worker for a partial/stale generation.
    semantic_state = semantic_client.index_status(config.FILE_INDEX_DB)
    if not semantic_state.get("ready") or int(semantic_state.get("vectors", 0) or 0) <= 0:
        return []
    rows: list[tuple] = []
    for hit in semantic_client.search(config.FILE_INDEX_DB, query, config.SEMANTIC_CANDIDATES):
        try:
            path = str(hit["path"])
            parent_index = int(hit["parent_chunk_index"])
        except (KeyError, TypeError, ValueError):
            continue
        row = conn.execute(
            "SELECT c.path, c.name, c.location, c.content, d.size_bytes, d.kind "
            "FROM kb_chunks_fts AS c JOIN kb_documents AS d ON d.path = c.path "
            "WHERE c.path = ? AND CAST(c.chunk_index AS INTEGER) = ? LIMIT 1",
            (path, parent_index),
        ).fetchone()
        if row and _allowed_result_path(row[0]):
            rows.append(row)
    return rows


def _rrf_merge(lexical_rows: list[tuple], semantic_rows: list[tuple]) -> list[tuple]:
    """Fuse incomparable lexical/semantic scores using rank-only RRF."""
    fused: dict[tuple[str, str, str], list[object]] = {}
    first_seen = 0
    for source_rows in (lexical_rows, semantic_rows):
        for rank, row in enumerate(source_rows, 1):
            key = (row[0], row[2], row[3])
            if key not in fused:
                fused[key] = [row, 0.0, first_seen]
                first_seen += 1
            fused[key][1] = float(fused[key][1]) + 1.0 / (60 + rank)
    ordered = sorted(fused.values(), key=lambda item: (-float(item[1]), int(item[2])))
    result: list[tuple] = []
    per_document: dict[str, int] = {}
    for row, _score, _order in ordered:
        path = row[0]
        if per_document.get(path, 0) >= 2:
            continue
        per_document[path] = per_document.get(path, 0) + 1
        result.append(row)
        if len(result) >= config.RAG_MAX_RESULTS:
            break
    return result


def _run_chunk_search(
    conn: sqlite3.Connection,
    *,
    query: str,
    match_query: str,
    split_match_query: str,
) -> str:
    # Filename hits are deterministic and deduplicated per source. Primary
    # content is UNINDEXED; MATCH runs over filename + search_text so OCR
    # alternatives can improve recall without being rendered back to the user.
    name_candidates = conn.execute(
        "SELECT c.path, c.name, c.location, c.content, d.size_bytes, d.kind "
        "FROM kb_chunks_fts AS c JOIN kb_documents AS d ON d.path = c.path "
        "WHERE c.name MATCH ? LIMIT 80",
        (match_query,),
    ).fetchall()
    if split_match_query:
        name_candidates.extend(
            conn.execute(
                "SELECT c.path, c.name, c.location, c.content, d.size_bytes, d.kind "
                "FROM kb_chunks_fts AS c JOIN kb_documents AS d ON d.path = c.path "
                "WHERE c.name MATCH ? LIMIT 80",
                (split_match_query,),
            ).fetchall()
        )

    def _name_rank(row):
        name = row[1]
        exact = name.casefold() == query.casefold()
        substring = query.casefold() in name.casefold()
        return (not exact, not substring, len(name))

    filename_rows = []
    seen_filename_paths: set[str] = set()
    for row in sorted(name_candidates, key=_name_rank):
        if row[0] in seen_filename_paths or not _allowed_result_path(row[0]):
            continue
        seen_filename_paths.add(row[0])
        filename_rows.append(row)
        if len(filename_rows) >= config.RAG_MAX_RESULTS:
            break

    candidate_limit = max(config.RAG_MAX_RESULTS * 6, 24)
    content_rows = conn.execute(
        "SELECT c.path, c.name, c.location, c.content, d.size_bytes, d.kind "
        "FROM kb_chunks_fts AS c JOIN kb_documents AS d ON d.path = c.path "
        "WHERE kb_chunks_fts MATCH ? ORDER BY bm25(kb_chunks_fts) LIMIT ?",
        (match_query, candidate_limit),
    ).fetchall()
    if split_match_query:
        content_rows.extend(
            conn.execute(
                "SELECT c.path, c.name, c.location, c.content, d.size_bytes, d.kind "
                "FROM kb_chunks_fts AS c JOIN kb_documents AS d ON d.path = c.path "
                "WHERE kb_chunks_fts MATCH ? ORDER BY bm25(kb_chunks_fts) LIMIT ?",
                (split_match_query, candidate_limit),
            ).fetchall()
        )

    # SQLite/BM25 keeps recall high with the OR-based MATCH expression, but a
    # long exact phrase must outrank a chunk that only happens to contain one
    # common term. The page/line location is already persisted on each row;
    # this only orders those stored rows before formatting them.
    compact_query = re.sub(
        r"\s+", "", normalize_extracted_text(query).casefold()
    )
    query_terms = _query_terms(query)

    def _content_rank(row):
        content = normalize_extracted_text(row[3] or "").casefold()
        compact_content = re.sub(r"\s+", "", content)
        exact = bool(compact_query and compact_query in compact_content)
        matched_terms = sum(
            1 for term in query_terms
            if term in content or term in compact_content
        )
        all_terms = bool(query_terms) and matched_terms == len(query_terms)
        return (not exact, not all_terms, -matched_terms)

    content_rows = [
        row for _rank, row in sorted(
            enumerate(content_rows),
            key=lambda item: (_content_rank(item[1]), item[0]),
        )
    ]

    lexical_rows = list(filename_rows)
    seen_pairs = {(row[0], row[2], row[3]) for row in lexical_rows}
    per_document = {row[0]: 1 for row in lexical_rows}
    for row in content_rows:
        if len(lexical_rows) >= candidate_limit:
            break
        if not _allowed_result_path(row[0]):
            continue
        key = (row[0], row[2], row[3])
        if key in seen_pairs or per_document.get(row[0], 0) >= 2:
            continue
        seen_pairs.add(key)
        per_document[row[0]] = per_document.get(row[0], 0) + 1
        lexical_rows.append(row)

    # A literal filename request is a locator operation, not a semantic one.
    # Preserve Lite's proven deterministic filename-first behavior and avoid a
    # ~400MB transient embedder spawn for an exact filename hit.
    if filename_rows and filename_rows[0][1].casefold() == query.casefold():
        return _format_chunk_rows(lexical_rows[:config.RAG_MAX_RESULTS], query)

    semantic_rows = _semantic_parent_rows(conn, query)
    if lexical_rows or semantic_rows:
        rows = _rrf_merge(lexical_rows, semantic_rows) if semantic_rows else lexical_rows[:config.RAG_MAX_RESULTS]
        return _format_chunk_rows(rows, query)

    fuzzy_rows = _ocr_fuzzy_rows(conn, query)
    if fuzzy_rows:
        return _format_chunk_rows(fuzzy_rows, query, fuzzy=True)

    fallback = _fallback_filename_search(
        query,
        reason="fts_zero_hits",
        roots=[root for root in _kb_source_roots() if root.name.casefold() == "raw"],
    )
    if not fallback.startswith("ไม่พบไฟล์"):
        return fallback
    return f"ไม่พบเนื้อหาที่ตรงกับ '{query}' ใน Knowledge Base (ลอง rag_manage(action='sync') หาก rag_document เพิ่งเปลี่ยน)"

def run(args: dict[str, str]) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "[error] query is required"
    if not config.FILE_INDEX_DB.exists():
        # Preserve the historical single-argument fallback contract first
        # (tests/extensions monkeypatch this helper), then broaden only to the
        # RAG/raw roots when legacy rag_document/md has no filename hit.
        filename_result = _fallback_filename_search(query)
        if not filename_result.startswith(("ไม่พบไฟล์", "[error]")):
            return filename_result
        return _fallback_filename_search(
            query,
            roots=[root for root in _kb_source_roots() if root.name.casefold() == "raw"],
        )

    match_query = _safe_match_query(query)
    if not match_query:
        return "[error] query ว่างเปล่าหลัง sanitize"
    split_match_query = _english_split_match_query(query)

    db_error_prefix = "[error] Knowledge Base index เสียหายหรืออ่านไม่ได้ — รัน rag_manage(action='sync') เพื่อสร้างดัชนีใหม่"
    try:
        conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    except sqlite3.DatabaseError as e:
        return f"{db_error_prefix}: {e}"

    # Classify the SQLite schema inside the same fail-closed DatabaseError
    # boundary as the actual MATCH queries. Corruption can surface while
    # reading sqlite_master/PRAGMA before any FTS MATCH executes.
    try:
        has_chunk_schema = _kb_chunk_schema_exists(conn)
        has_kb_metadata = _kb_metadata_exists(conn) if not has_chunk_schema else False
    except sqlite3.DatabaseError as e:
        conn.close()
        return f"{db_error_prefix}: {e}"

    if has_chunk_schema:
        try:
            return _run_chunk_search(
                conn,
                query=query,
                match_query=match_query,
                split_match_query=split_match_query,
            )
        except sqlite3.DatabaseError as e:
            return f"{db_error_prefix}: {e}"
        finally:
            conn.close()
    if has_kb_metadata:
        conn.close()
        return "[error] Knowledge Base index เป็น schema รุ่นเก่า — รัน rag_manage(action='sync') เพื่อ migrate ก่อนค้นหา"

    # Legacy /index path: one compact FTS row per file.
    if len(_query_terms(query)) == 1:
        filename_result = _fallback_filename_search(query, reason="fts_zero_hits")
        if not filename_result.startswith("ไม่พบไฟล์"):
            conn.close()
            return filename_result

    try:
        # Bug fixed 2026-07-23 (found during final architecture-switch smoke
        # testing): a single combined MATCH over (name, content), even with
        # bm25() column weighting (tried bm25(files_fts, 1.0, 10.0, 1.0) and
        # bm25(files_fts, 0.0, 1.0, 0.0) -- content-only-zeroed), silently
        # ranked the exact file "PLAN.md" itself BELOW files that merely cite
        # "[PLAN.md]", and even below unrelated files like "ENCRYPTION_PLAN.md"
        # for a name-only query. BM25 on a trigram-tokenized short "name"
        # field does not behave like word-level BM25 intuition suggests --
        # not investigated further given diminishing returns. Fix: search
        # `name MATCH` for candidates (recall via FTS5), then rank them in
        # plain Python with a simple, predictable rule (exact match > query
        # is a substring of the name > shorter name = more specific) instead
        # of trusting bm25()'s score on this field at all.
        name_candidates = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.name MATCH ? LIMIT 50",
            (match_query,),
        ).fetchall()
        split_name_candidates = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.name MATCH ? LIMIT 50",
            (split_match_query,),
        ).fetchall() if split_match_query else []

        def _name_rank(row):
            name = row[1]
            exact = name.lower() == query.lower()
            substring = query.lower() in name.lower()
            return (not exact, not substring, len(name))

        name_rows = sorted(name_candidates, key=_name_rank)
        direct_name_paths = {row[0] for row in name_rows}
        name_rows.extend(row for row in sorted(split_name_candidates, key=_name_rank) if row[0] not in direct_name_paths)
        name_rows = name_rows[: config.RAG_MAX_RESULTS]
        seen_paths = {r[0] for r in name_rows}

        remaining = config.RAG_MAX_RESULTS - len(name_rows)
        general_rows = []
        if remaining > 0:
            general_rows = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.files_fts MATCH ? ORDER BY bm25(f.files_fts) LIMIT ?",
                (match_query, config.RAG_MAX_RESULTS + len(name_rows)),
            ).fetchall()
            general_rows = [r for r in general_rows if r[0] not in seen_paths][:remaining]
            if split_match_query and len(general_rows) < remaining:
                split_general_rows = conn.execute(
                    "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
                    "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.files_fts MATCH ? ORDER BY bm25(f.files_fts) LIMIT ?",
                    (split_match_query, config.RAG_MAX_RESULTS + len(name_rows)),
                ).fetchall()
                direct_general_paths = {row[0] for row in general_rows}
                general_rows.extend(
                    row for row in split_general_rows
                    if row[0] not in seen_paths and row[0] not in direct_general_paths
                )
                general_rows = general_rows[:remaining]

        rows = [row for row in name_rows + general_rows if _allowed_result_path(row[0])]
    except sqlite3.DatabaseError as e:
        return f"{db_error_prefix}: {e}"
    finally:
        conn.close()

    # A concise one-term query is commonly a filename request (``config``,
    # ``prompts``, ``indexer.py``).  If the index has no filename candidate,
    # try the bounded filename walk before returning a content-only hit.  This
    # remains generic: the agent still chooses the search term; the fallback
    # only resolves the filesystem ambiguity.
    if not name_rows and len(_query_terms(query)) == 1:
        filename_result = _fallback_filename_search(query, reason="fts_zero_hits")
        if not filename_result.startswith("ไม่พบไฟล์"):
            return filename_result

    if not rows:
        filename_result = _fallback_filename_search(query, reason="fts_zero_hits")
        if not filename_result.startswith("ไม่พบไฟล์"):
            return filename_result
        return _project_content_fallback(query) or filename_result

    lines = [f"พบ {len(rows)} ไฟล์ที่ตรงกับ '{query}':"]
    for path, name, snippet, size_bytes, kind in rows:
        snippet = (snippet or "").replace("\n", " ").strip()
        if len(snippet) > config.RAG_SNIPPET_CHARS:
            snippet = snippet[: config.RAG_SNIPPET_CHARS] + "..."
        metadata = ""
        if kind and kind != "text excerpt":
            metadata = f" [{kind}, {(size_bytes or 0):,} bytes; content not indexed]"
        lines.append(
            f"- ไฟล์: {path}\n"
            f"  ชื่อไฟล์: {name}\n"
            "  หน้า/ตำแหน่ง: ไม่ระบุ (legacy index)\n"
            f"{_source_marker(path)}\n"
            f"  ตัวอย่างเนื้อหา: {snippet or '(ไม่มีข้อความตัวอย่าง)'}{metadata}"
        )
    return "\n".join(lines)


def rag_search(query: str) -> str:
    """Python convenience wrapper around the deterministic retrieval core."""
    return run({"query": query})
