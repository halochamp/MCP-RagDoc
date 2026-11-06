"""Adversarial production-boundary oracles; all source/state is disposable."""
from pathlib import Path
import subprocess
import sys
import time
import asyncio
import threading

import pytest
from test_standalone import _configure_instance


@pytest.mark.parametrize("kind", ["hidden", "file_link", "directory_link"])
def test_read_boundaries_match_discovery(tmp_path, monkeypatch, kind):
    import rag_search
    import rag_sync
    import web_ui

    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    visible = root / "nested"
    visible.mkdir()
    target = visible / "manual.txt"
    target.write_text("ordinary hydraulic document")
    assert rag_search._allowed_result_path(str(target))
    assert web_ui._openable_source_path(str(target)) == target
    if kind == "hidden":
        hidden = root / ".hidden"
        hidden.mkdir()
        candidate = hidden / "manual.txt"
        candidate.write_text("hidden hydraulic document")
    elif kind == "file_link":
        candidate = root / "alias.txt"
        candidate.symlink_to(target)
    else:
        alias = root / "alias"
        alias.symlink_to(visible, target_is_directory=True)
        candidate = alias / "manual.txt"
    assert str(candidate) not in {str(p) for _, p in rag_sync._iter_source_files()}
    assert not rag_search._allowed_result_path(str(candidate))
    assert web_ui._openable_source_path(str(candidate)) is None


def test_semantic_partial_stdout_obeys_timeout(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("import sys,time\nsys.stdout.write('{');sys.stdout.flush();time.sleep(20)\n")
    # Outer process deadline is a safety guard, not the timeout being tested.
    code = '''
import pathlib, time, semantic_client as c, config
c.model_status=lambda: {'enabled':True,'ready':True}
c._model_root=lambda: pathlib.Path('.')
c._WORKER=pathlib.Path(%r)
config.SEMANTIC_SYNC_TIMEOUT_SECONDS=0.2
start=time.monotonic()
assert c.refresh(pathlib.Path('unused')) is None
assert time.monotonic()-start < 2
''' % str(worker)
    import os
    env = {**os.environ, "RAGDOC_DATA_DIR": str(tmp_path / "state"),
           "RAGDOC_KNOWLEDGE_DIR": str(tmp_path / "source")}
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr


def test_fallback_obeys_source_policy(tmp_path, monkeypatch):
    import rag_search
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    visible = root / "visible-marker.md"
    visible.write_text("ordinary marker content")
    hidden = root / ".hidden-marker.md"
    hidden.write_text("hidden marker content")
    link = root / "link-marker.md"
    link.symlink_to(visible)
    secret = root / "credentials-marker.md"
    secret.write_text("marker api_key = '" + "x" * 30 + "'")
    result = rag_search._fallback_filename_search("marker")
    assert str(visible) in result
    assert str(hidden) not in result
    assert str(link) not in result
    assert str(secret) not in result
    content = rag_search._project_content_fallback("marker")
    assert str(visible) in content
    assert str(hidden) not in content
    assert str(secret) not in content


def test_source_mutation_during_extraction_rolls_back(tmp_path, monkeypatch):
    import rag_sync
    import sqlite3
    root, _, db = _configure_instance(tmp_path, monkeypatch)
    source = root / "manual.txt"
    source.write_text("stable original")
    assert rag_sync.sync_kb()["documents"] == 1
    source.write_text("before extraction")
    def changing(_path):
        yield "line 1", "before extraction", "before extraction"
        source.write_text("after extraction")
    monkeypatch.setattr(rag_sync, "_iter_search_chunks", changing)
    stats = rag_sync.sync_kb()
    assert stats["sync_errors"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT content FROM kb_chunks_fts").fetchone()[0] == "stable original"


def test_ui_search_does_not_block_event_loop(tmp_path, monkeypatch):
    import web_ui
    _configure_instance(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    def blocking(*_args):
        entered.set()
        assert release.wait(2)
        return "result"
    monkeypatch.setattr(web_ui, "_run_search", blocking)
    class Request:
        headers = {"origin": next(iter(web_ui._ALLOWED_ORIGINS))}
        async def json(self):
            return {"query": "marker"}
    async def exercise():
        task = asyncio.create_task(web_ui.search(Request()))
        assert await asyncio.to_thread(entered.wait, 1)
        # If sync search blocks this loop, its barrier times out before here.
        assert entered.is_set()
        release.set()
        assert (await task)["result"] == "result"
    asyncio.run(exercise())


def test_ocr_lease_preserves_consumer_failure_after_start_timeout(monkeypatch):
    import ocr_rewrite as o
    import config
    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", True)
    monkeypatch.setattr(config, "OCR_REWRITE_AUTO_START", True)
    monkeypatch.setattr(config, "OCR_REWRITE_AUTO_STOP", True)
    monkeypatch.setattr(o, "_server_state", lambda _: "down")
    monkeypatch.setattr(o, "_wait_for_server", lambda _: False)
    actions = []
    monkeypatch.setattr(o, "_control_action", lambda _, action: actions.append(action))
    with pytest.raises(ValueError, match="consumer failure"):
        with o.vision_server_lease(required=True):
            raise ValueError("consumer failure")
    assert actions == ["server/start", "server/stop"]


def test_semantic_override_requires_model_assets(tmp_path, monkeypatch):
    import config
    import semantic_client as c
    monkeypatch.setenv("RAGDOC_SEMANTIC_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(c.importlib.util, "find_spec", lambda _: object())
    assert not c.model_status()["ready"]
    for name in (config.SEMANTIC_MODEL_ONNX, config.SEMANTIC_SENTENCEPIECE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset placeholder")
    assert c.model_status()["ready"]


def test_discovered_path_replaced_with_symlink_never_extracts(tmp_path, monkeypatch):
    import rag_sync
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    directory = root / "nested"
    directory.mkdir()
    (directory / "manual.txt").write_text("ordinary source")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manual.txt").write_text("outside source")
    extracted = []
    def chunks(path):
        extracted.append(path)
        yield "line 1", "outside source", "outside source"
    monkeypatch.setattr(rag_sync, "_iter_search_chunks", chunks)
    def progress(event, *_args):
        if event == "file":
            directory.rename(root / "moved")
            directory.symlink_to(outside, target_is_directory=True)
    assert rag_sync.sync_kb(on_progress=progress)["documents"] == 0
    assert not extracted


def test_live_child_not_abandoned_during_slow_startup(tmp_path, monkeypatch):
    import os
    import rag_service as s
    _configure_instance(tmp_path, monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(s.time, "time", lambda: clock[0])
    spawned = []
    def spawn(*_args, **_kwargs):
        spawned.append(True)
        return type("Child", (), {"pid": os.getpid()})()
    monkeypatch.setattr(s.subprocess, "Popen", spawn)
    assert s.start_sync().startswith("status=started")
    clock[0] += 6
    assert s.start_sync().startswith("status=already_running")
    assert len(spawned) == 1


def test_exited_owned_child_is_not_alive():
    import rag_service as s
    import os
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(child.pid)], capture_output=True, text=True).stdout.strip()
            if status.startswith("Z"):
                break
        assert status.startswith("Z"), status
        assert not s._process_alive(child.pid)
        assert s._process_alive(os.getpid())
    finally:
        child.wait(timeout=2)
