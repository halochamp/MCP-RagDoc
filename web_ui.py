"""Loopback-only HTML UI for standalone MCP-RagDoc.

The UI is a thin browser console over the deterministic retrieval boundary. An
optional local vision model can correct scanned PDF OCR during indexing; the UI
does not expose arbitrary file reads or bypass source/private-state boundaries.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import config
import rag_service
from helpers.sensitive_paths import is_sensitive_path
from helpers.source_paths import visible_source_file

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "ui" / "rag.html"
UI_HOST = "127.0.0.1"
UI_PORT = int(os.getenv("RAGDOC_UI_PORT", "8769"))
_ALLOWED_ORIGINS = {
    f"http://127.0.0.1:{UI_PORT}",
    f"http://localhost:{UI_PORT}",
}
_SEARCH_LOCK = threading.Lock()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _same_origin(request: Request) -> None:
    if request.headers.get("origin") not in _ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="origin not allowed")


def _parse_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (raw or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in fields:
            fields[key] = value.strip()
    return fields


def _int_field(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields.get(key, "0") or 0)
    except (TypeError, ValueError):
        return 0


def _run_search(query: str, mode: str = "hybrid") -> str:
    """One patchable search boundary for the HTML UI."""
    del mode  # MCP-RagDoc has one deterministic hybrid retrieval contract.
    return str(rag_service.search(query))


def _health_payload() -> dict[str, object]:
    fields = _parse_fields(rag_service.health())
    active = rag_service.active_sync_snapshot()
    status = fields.get("status", "unknown")
    return {
        "status": status,
        "source_files": _int_field(fields, "source_files"),
        "documents": _int_field(fields, "documents"),
        "chunks": _int_field(fields, "chunks"),
        "semantic": fields.get("semantic", "unknown"),
        "semantic_vectors": _int_field(fields, "semantic_vectors"),
        "semantic_generation": _int_field(fields, "semantic_generation"),
        "sync_job": active,
        "knowledge_dir": str(config.KNOWLEDGE_DIR),
        "data_dir": str(config.DATA_DIR),
        "index_db": str(config.FILE_INDEX_DB),
        "llm_used": False,
        "ocr_rewrite_enabled": bool(config.OCR_REWRITE_ENABLED),
        "ocr_rewrite_model": config.LLM_MODEL,
        "ocr_rewrite_endpoint": config.LLM_BASE_URL,
        "retrieval": "SQLite FTS5 + BM25 + optional MiniLM + RRF",
    }


def _relative_source_path(path: str) -> str | None:
    candidate = visible_source_file(path, config.KNOWLEDGE_DIR)
    if candidate is None:
        return None
    root = config.KNOWLEDGE_DIR.expanduser().resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if candidate == root or candidate.name.startswith("."):
        return None
    if not candidate.is_file() or is_sensitive_path(candidate):
        return None
    return relative.as_posix()


def _file_rows(query: str = "", limit: int = 200) -> list[dict[str, object]]:
    db_path = Path(config.FILE_INDEX_DB)
    if not db_path.is_file():
        return []
    needle = query.strip().casefold()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT path, name, kind, size_bytes, chunk_count, mtime_ns "
                "FROM kb_documents ORDER BY path"
            ).fetchall()
    except sqlite3.Error:
        return []
    result: list[dict[str, object]] = []
    for path, name, kind, size_bytes, chunk_count, mtime_ns in rows:
        path_text = str(path)
        relative = _relative_source_path(path_text)
        if relative is None:
            continue
        if needle and needle not in path_text.casefold() and needle not in str(name).casefold():
            continue
        result.append({
            "path": path_text,
            "relative_path": relative,
            "name": str(name),
            "kind": str(kind),
            "size_bytes": int(size_bytes or 0),
            "chunks": int(chunk_count or 0),
            "mtime_ns": int(mtime_ns or 0),
        })
        if len(result) >= limit:
            break
    return result


def _openable_source_path(path: str) -> Path | None:
    return visible_source_file(path, config.KNOWLEDGE_DIR)


def _normalize_reveal_path(raw: str) -> str:
    """Remove the visible source label before validating a Finder path.

    The HTML renderer normally stores only the path, but accepting the common
    ``[source: /path]`` form keeps stale browser tabs and copied result labels
    from turning a valid source into a rejected path.
    """
    value = str(raw or "").strip()
    folded = value.casefold()
    if folded.startswith("[source:"):
        value = value[len("[source:"):].strip()
        if value.endswith("]"):
            value = value[:-1].rstrip()
    elif folded.startswith("source:"):
        value = value[len("source:"):].strip()
    return value.strip().strip("`*").rstrip("),.;:!?]").strip()


def _start_sync_payload() -> dict[str, object]:
    raw = rag_service.start_sync()
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}


@app.get("/")
def root() -> FileResponse:
    if not HTML_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI file missing")
    return FileResponse(HTML_PATH)


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        return {"ok": True, **_health_payload()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"health failed: {type(exc).__name__}") from None


@app.get("/api/files")
def files(query: str = "", limit: int = 200) -> dict[str, object]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")
    try:
        rows = _file_rows(query, limit)
        return {
            "ok": True,
            "total": len(rows),
            "files": rows,
            "truncated": len(rows) >= limit,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"file list failed: {type(exc).__name__}") from None


@app.post("/api/search")
async def search(request: Request) -> dict[str, object]:
    _same_origin(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    query = str(body.get("query") or "").strip() if isinstance(body, dict) else ""
    mode = str(body.get("mode") or "hybrid").strip() if isinstance(body, dict) else "hybrid"
    if not query or len(query) > 4_000:
        raise HTTPException(status_code=400, detail="query must be 1..4000 chars")
    if mode not in {"hybrid", "chunks", "files", "source_first"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    try:
        def execute():
            with _SEARCH_LOCK:
                return _run_search(query, mode)
        raw = await run_in_threadpool(execute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {type(exc).__name__}") from None
    return {
        "ok": True,
        "query": query,
        "mode": mode,
        "result": str(raw),
        "llm_used": False,
    }


@app.post("/api/open-file")
async def open_file(request: Request) -> dict[str, object]:
    """Reveal one indexed source file in Finder, following the local desktop behavior."""
    _same_origin(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    path = _normalize_reveal_path(body.get("path")) if isinstance(body, dict) else ""
    if not path or len(path) > 4_000:
        raise HTTPException(status_code=400, detail="path is required")
    candidate = _openable_source_path(path)
    if candidate is None:
        raise HTTPException(status_code=403, detail="file is outside the knowledge directory")
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="Finder reveal requires macOS")
    try:
        result = await run_in_threadpool(
            subprocess.run,
            ["open", "-R", str(candidate)],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise HTTPException(status_code=500, detail="could not open Finder") from None
    if result.returncode:
        raise HTTPException(status_code=500, detail="could not open Finder")
    return {"ok": True, "path": str(candidate), "relative_path": _relative_source_path(str(candidate))}


@app.post("/api/sync")
@app.post("/api/build")
def start_sync(request: Request) -> dict[str, object]:
    _same_origin(request)
    return _start_sync_payload()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    raw = rag_service.sync_status(job_id)
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, object]:
    _same_origin(request)
    raw = rag_service.cancel_sync(job_id)
    return {"ok": True, "raw": raw, "job": _parse_fields(raw)}
