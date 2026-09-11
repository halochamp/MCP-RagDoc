"""Standalone ownership/scope tests for MCP-RagDoc."""
from __future__ import annotations

import importlib
from pathlib import Path
import re
import sqlite3
import sys
import time

import pytest

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))


def _configure_instance(tmp_path, monkeypatch, *, semantic: bool = False):
    import config

    knowledge = tmp_path / "knowledge"
    state = tmp_path / "state"
    knowledge.mkdir()
    state.mkdir()
    db = state / "ragdoc.sqlite3"
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_SOURCE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", knowledge)
    monkeypatch.setattr(config, "RAW_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", knowledge)
    monkeypatch.setattr(config, "MD_DIR", knowledge)
    monkeypatch.setattr(config, "INDEX_ROOTS", [knowledge])
    monkeypatch.setattr(config, "DATA_DIR", state)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)
    monkeypatch.setattr(config, "SEMANTIC_ENABLED", semantic)
    return knowledge, state, db


def test_standalone_sources_do_not_import_host_agent():
    forbidden = (
        "from _bootstrap import PROJECT_ROOT",
        "from helpers import search_scope",
        "from helpers import turn_guard",
        "import indexer",
        "langchain_core",
        "LITE_SEMANTIC",
    )
    for path in RAG_ROOT.rglob("*.py"):
        if "test_suite" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.relative_to(RAG_ROOT)} depends on host token {needle!r}"


def test_recursive_source_scope_and_lexical_search(tmp_path, monkeypatch):
    import rag_service
    import rag_sync

    knowledge, _state, db = _configure_instance(tmp_path, monkeypatch)
    source = knowledge / "manual.txt"
    source.write_text(
        "The hydraulic pump maintains pressure for landing gear braking.",
        encoding="utf-8",
    )
    nested = knowledge / "nested"
    nested.mkdir()
    nested_source = nested / "nested.txt"
    nested_source.write_text("nested hydraulic marker", encoding="utf-8")
    (knowledge / ".hidden.txt").write_text("hidden marker", encoding="utf-8")

    scanned = list(rag_sync._iter_source_files())
    assert {(key, path.name) for key, path in scanned} == {
        ("manual.txt", "manual.txt"),
        ("nested/nested.txt", "nested.txt"),
    }

    stats = rag_service.sync_kb()
    assert stats["raw_scanned"] == 2
    assert stats["documents"] == 2
    assert stats["sync_errors"] == 0
    assert "manual.txt" in rag_service.search("hydraulic pressure")
    assert str(nested_source.resolve()) in rag_service.search("nested hydraulic")
    assert ".hidden.txt" not in rag_service.search("hidden marker")
    import rag_search
    assert rag_search._allowed_result_path(str(nested_source.resolve()))
    assert "status=healthy" in rag_service.health()

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT path FROM kb_documents").fetchall()
    assert {row[0] for row in rows} == {
        str(source.resolve()),
        str(nested_source.resolve()),
    }


def test_sync_does_not_require_source_directory_write_permission(tmp_path, monkeypatch):
    import rag_service

    knowledge, _state, _db = _configure_instance(tmp_path, monkeypatch)
    source = knowledge / "readonly.txt"
    source.write_text("read only source document", encoding="utf-8")
    source.chmod(0o400)
    knowledge.chmod(0o500)
    try:
        stats = rag_service.sync_kb()
    finally:
        knowledge.chmod(0o700)
        source.chmod(0o600)
    assert stats["documents"] == 1
    assert stats["sync_errors"] == 0


def test_missing_source_directory_is_empty_not_created(tmp_path, monkeypatch):
    import config
    import rag_service
    import rag_sync

    missing = tmp_path / "does-not-exist"
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", missing)
    monkeypatch.setattr(config, "KNOWLEDGE_SOURCE_DIR", missing)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", missing)
    monkeypatch.setattr(config, "RAW_DIR", missing)
    monkeypatch.setattr(config, "DATA_DIR", state)
    monkeypatch.setattr(config, "FILE_INDEX_DB", state / "ragdoc.sqlite3")
    monkeypatch.setattr(config, "SEMANTIC_ENABLED", False)

    assert list(rag_sync._iter_source_files()) == []
    assert not missing.exists()
    stats = rag_service.sync_kb()
    assert stats["documents"] == 0
    assert not missing.exists()


def test_fastmcp_surface_is_exactly_five_tools(monkeypatch, tmp_path):
    _configure_instance(tmp_path, monkeypatch)
    import rag_server

    manager = getattr(rag_server.mcp, "_tool_manager", None)
    if manager is None or not hasattr(manager, "list_tools"):
        pytest.skip("FastMCP tool manager API unavailable")
    tools = manager.list_tools()
    if hasattr(tools, "__await__"):
        import asyncio
        tools = asyncio.run(tools)
    assert {tool.name for tool in tools} == {
        "rag_search", "sync_kb", "sync_status", "cancel_sync", "rag_health"
    }


def _job_id(text: str) -> str:
    match = re.search(r"^job_id=(\S+)$", text, re.MULTILINE)
    assert match, text
    return match.group(1)


def _wait_terminal(rag_service, job_id: str, timeout: float = 3.0) -> str:
    deadline = time.monotonic() + timeout
    latest = ""
    while time.monotonic() < deadline:
        latest = rag_service.sync_status(job_id)
        if latest.startswith(("status=done", "status=cancelled", "status=error")):
            return latest
        time.sleep(0.02)
    raise AssertionError(f"sync job did not finish: {latest}")


def test_background_sync_returns_job_and_completes(tmp_path, monkeypatch):
    import rag_service

    knowledge, _state, _db = _configure_instance(tmp_path, monkeypatch)
    (knowledge / "a.txt").write_text("alpha hydraulic marker", encoding="utf-8")
    (knowledge / "b.txt").write_text("beta navigation marker", encoding="utf-8")

    started = rag_service.start_sync()
    assert started.startswith("status=started")
    job_id = _job_id(started)
    final = _wait_terminal(rag_service, job_id)
    assert final.startswith("status=done")
    assert "progress_percent=100.0" in final
    assert "source_scanned=2" in final
    assert "documents=2" in final


def test_second_sync_is_rejected_while_persistent_job_is_active(tmp_path, monkeypatch):
    import os
    import rag_service

    _configure_instance(tmp_path, monkeypatch)
    job_id = "sync-0123456789ab"
    status_file, _cancel_file = rag_service._job_paths(job_id)
    state = rag_service._new_job_state(job_id)
    state.update({"pid": os.getpid(), "status": "running", "phase": "file"})
    rag_service._write_job(status_file, state)

    second = rag_service.start_sync()
    assert second.startswith("status=already_running")
    assert f"job_id={job_id}" in second


def test_cancel_sync_stops_persistent_worker_at_safe_checkpoint(tmp_path, monkeypatch):
    import threading
    import rag_service

    _configure_instance(tmp_path, monkeypatch)
    entered = threading.Event()
    job_id = "sync-abcdef012345"
    status_file, cancel_file = rag_service._job_paths(job_id)
    rag_service._write_job(status_file, rag_service._new_job_state(job_id))

    def cancellable_pipeline(*, on_progress=None, cancel_requested=None, cancel_file=None):
        if on_progress:
            on_progress("file", Path("manual.txt"), {
                "progress_percent": 25.0,
                "completed_files": 1,
                "total_files": 4,
                "estimated_remaining_seconds": 1.0,
            })
        entered.set()
        while True:
            if cancel_requested and cancel_requested():
                raise rag_service.kb_sync.SyncCancelled("cancelled")
            time.sleep(0.01)

    monkeypatch.setattr(rag_service, "_sync_pipeline_locked", cancellable_pipeline)
    worker = threading.Thread(
        target=rag_service.run_persistent_job,
        args=(job_id, status_file, cancel_file),
    )
    worker.start()
    assert entered.wait(1.0)
    status = rag_service.sync_status(job_id)
    assert "phase=file" in status
    assert "progress_percent=25.0" in status
    cancelling = rag_service.cancel_sync(job_id)
    assert cancelling.startswith("status=cancelling")
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    final = rag_service.sync_status(job_id)
    assert final.startswith("status=cancelled")
    assert "cancel_requested=true" in final


def test_cancel_mid_document_rolls_back_current_file(tmp_path, monkeypatch):
    import rag_service
    import rag_sync

    knowledge, _state, db = _configure_instance(tmp_path, monkeypatch)
    source = knowledge / "manual.txt"
    source.write_text("last good searchable snapshot", encoding="utf-8")
    rag_service.sync_kb_blocking()

    source.write_text("replacement body", encoding="utf-8")
    cancelled = {"value": False}

    def chunks(_path):
        yield "line 1", "new partial text", "new partial text"
        cancelled["value"] = True
        yield "line 2", "must never commit", "must never commit"

    monkeypatch.setattr(rag_sync, "_iter_search_chunks", chunks)
    with pytest.raises(rag_sync.SyncCancelled):
        rag_sync.sync_kb(cancel_requested=lambda: cancelled["value"])

    with sqlite3.connect(db) as conn:
        content = conn.execute(
            "SELECT content FROM kb_chunks_fts WHERE path=? ORDER BY chunk_index",
            (str(source.resolve()),),
        ).fetchall()
    assert content == [("last good searchable snapshot",)]
