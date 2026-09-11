"""Loopback HTML UI tests for standalone MCP-RagDoc."""
from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

from fastapi.testclient import TestClient

RAG_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(RAG_ROOT))


def _client():
    import web_ui

    return web_ui, TestClient(web_ui.app)


def _origin(web_ui) -> dict[str, str]:
    return {"Origin": f"http://127.0.0.1:{web_ui.UI_PORT}"}


def test_root_serves_standalone_rag_console():
    web_ui, client = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert "MCP-RagDoc" in response.text
    assert "Vision OCR rewrite" in response.text
    assert "Sync KB" in response.text
    assert "Indexed files" in response.text


def test_search_is_same_origin_and_uses_deterministic_service():
    web_ui, client = _client()
    with mock.patch.object(web_ui, "_run_search", return_value="deterministic result") as invoke:
        blocked = client.post("/api/search", json={"query": "ทุน"})
        ok = client.post(
            "/api/search",
            headers=_origin(web_ui),
            json={"query": "ทุน", "mode": "hybrid"},
        )
    assert blocked.status_code == 403
    assert ok.status_code == 200
    assert ok.json()["llm_used"] is False
    assert ok.json()["result"] == "deterministic result"
    invoke.assert_called_once_with("ทุน", "hybrid")


def test_sync_status_and_cancel_use_persistent_ragdoc_jobs():
    web_ui, client = _client()
    origin = _origin(web_ui)
    started = "status=started\njob_id=sync-123456789abc\nphase=queued\nprogress_percent=0.0"
    running = "status=running\njob_id=sync-123456789abc\nphase=file\nprogress_percent=42.5"
    cancelling = "status=cancelling\njob_id=sync-123456789abc\nphase=file\nprogress_percent=42.5"
    with mock.patch.object(web_ui.rag_service, "start_sync", return_value=started), \
         mock.patch.object(web_ui.rag_service, "sync_status", return_value=running), \
         mock.patch.object(web_ui.rag_service, "cancel_sync", return_value=cancelling):
        start = client.post("/api/sync", headers=origin)
        status = client.get("/api/jobs/sync-123456789abc")
        cancel = client.post("/api/jobs/sync-123456789abc/cancel", headers=origin)
    assert start.status_code == 200
    assert start.json()["job"]["job_id"] == "sync-123456789abc"
    assert status.json()["job"]["progress_percent"] == "42.5"
    assert cancel.json()["job"]["status"] == "cancelling"


def test_files_returns_indexed_absolute_paths():
    web_ui, client = _client()
    fake_rows = [{
        "path": "/tmp/rag_document/manual.pdf",
        "relative_path": "manual.pdf",
        "name": "manual.pdf",
        "kind": "PDF",
        "size_bytes": 1234,
        "chunks": 4,
        "mtime_ns": 1,
    }]
    with mock.patch.object(web_ui, "_file_rows", return_value=fake_rows) as rows:
        response = client.get("/api/files?query=manual&limit=20")
    assert response.status_code == 200
    assert response.json()["files"][0]["path"] == "/tmp/rag_document/manual.pdf"
    rows.assert_called_once_with("manual", 20)


def test_open_file_requires_same_origin_and_uses_native_opener(tmp_path, monkeypatch):
    web_ui, client = _client()
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(web_ui.config, "KNOWLEDGE_DIR", tmp_path)
    with mock.patch.object(web_ui.subprocess, "run") as opener:
        opener.return_value.returncode = 0
        blocked = client.post("/api/open-file", json={"path": str(source)})
        opened = client.post(
            "/api/open-file",
            headers=_origin(web_ui),
            json={"path": str(source)},
        )
    assert blocked.status_code == 403
    assert opened.status_code == 200
    assert opened.json()["relative_path"] == "manual.pdf"
    opener.assert_called_once()
    assert opener.call_args.args[0][:2] == ["open", "-R"]


def test_open_file_rejects_path_outside_knowledge_root(tmp_path, monkeypatch):
    web_ui, client = _client()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"pdf")
    monkeypatch.setattr(web_ui.config, "KNOWLEDGE_DIR", knowledge)
    response = client.post(
        "/api/open-file",
        headers=_origin(web_ui),
        json={"path": str(outside)},
    )
    assert response.status_code == 403


def test_open_file_strips_source_label_from_stale_result_click(tmp_path, monkeypatch):
    web_ui, client = _client()
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(web_ui.config, "KNOWLEDGE_DIR", tmp_path)
    with mock.patch.object(web_ui.subprocess, "run") as opener:
        opener.return_value.returncode = 0
        response = client.post(
            "/api/open-file",
            headers=_origin(web_ui),
            json={"path": f"[source: {source}]"},
        )
    assert response.status_code == 200
    assert opener.call_args.args[0] == ["open", "-R", str(source.resolve())]


def test_health_declares_no_backend_llm():
    web_ui, client = _client()
    payload = {
        "status": "healthy",
        "source_files": 3,
        "documents": 3,
        "chunks": 76,
        "semantic": "ready",
        "semantic_vectors": 280,
        "semantic_generation": 4,
        "sync_job": None,
        "knowledge_dir": "/tmp/rag_document",
        "data_dir": "/tmp/state",
        "index_db": "/tmp/state/ragdoc.sqlite3",
        "llm_used": False,
        "retrieval": "SQLite FTS5 + BM25 + optional MiniLM + RRF",
    }
    with mock.patch.object(web_ui, "_health_payload", return_value=payload):
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["llm_used"] is False
