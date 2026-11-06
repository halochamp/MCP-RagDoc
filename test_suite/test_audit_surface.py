"""Whole-surface adversarial controls using disposable instances only."""
import asyncio
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import zipfile

import pytest
from test_standalone import _configure_instance, _job_id, _wait_terminal


@pytest.mark.parametrize("suffix,member,xml", [
    (".docx", "word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>hydraulic marker</w:t></w:r></w:p></w:body></w:document>'),
    (".pptx", "ppt/slides/slide1.xml", '<sld><sp><txBody><p><t>hydraulic marker</t></p></txBody></sp></sld>'),
    (".ods", "content.xml", '<document><p>hydraulic marker</p></document>'),
    (".odp", "content.xml", '<document><p>hydraulic marker</p></document>'),
])
def test_office_container_extraction_and_corruption(tmp_path, monkeypatch, suffix, member, xml):
    import rag_service
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    path = root / ("manual" + suffix)
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(member, xml)
    assert rag_service.sync_kb()["documents"] == 1
    assert str(path) in rag_service.search("hydraulic")
    path.write_bytes(b"corrupt package")
    assert rag_service.sync_kb()["sync_errors"] == 1
    # Extraction errors preserve the last committed snapshot by contract.
    assert str(path) in rag_service.search("hydraulic")


def test_xlsx_text_pdf_and_live_apple_vision(tmp_path, monkeypatch):
    import fitz
    from openpyxl import Workbook
    import config
    import rag_service
    import rag_sync
    from helpers import vision_ocr
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", False)
    workbook = Workbook()
    workbook.active.append(["hydraulic spreadsheet marker", 42])
    workbook.save(root / "book.xlsx")
    workbook.close()
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=180)
    page.insert_text((20, 60), "HYDRAULIC VISION MARKER", fontsize=20)
    pdf.save(root / "digital.pdf")
    pixels = page.get_pixmap(dpi=150).tobytes("png")
    pdf.close()
    (root / "ordinary.txt").write_text("hydraulic plain marker")
    if vision_ocr._ensure_binary():
        scan = fitz.open()
        page = scan.new_page(width=400, height=180)
        page.insert_image(page.rect, stream=pixels)
        scan.save(root / "scan.pdf")
        scan.close()
        units = list(rag_sync._iter_pdf_units(root / "scan.pdf"))
        assert "HYDRAULIC" in units[0][1].upper()
        assert units[0][3] == "page 1/1"
    else:
        pytest.skip("Apple Vision binary unavailable; controlled failure tests remain applicable")
    assert rag_service.sync_kb()["documents"] == 4
    result = rag_service.search("hydraulic")
    assert all(name in result for name in ("book.xlsx", "digital.pdf", "ordinary.txt", "scan.pdf"))


def test_chunk_bounds_ocr_separation_thai_and_pathological_token(tmp_path, monkeypatch):
    import rag_sync
    import config
    from helpers.thai_text_normalize import normalize_extracted_text
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 64)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 8)
    source = root / "long.txt"
    source.write_text("x" * 10000)
    rows = list(rag_sync._iter_search_chunks(source))
    assert rows and all(0 < len(row[1]) <= 64 for row in rows)
    assert all(row[1] == row[2] for row in rows)
    assert normalize_extracted_text("นำ้ คำ สั่ง 42 pump") == "น้ำคำสั่ง 42 pump"
    monkeypatch.setattr(rag_sync, "_iter_source_units", lambda _: iter([(1, "primary", "primary | alternative", "page 1/2"), (2, "next", "next", "page 2/2")]))
    rows = list(rag_sync._iter_search_chunks(source))
    assert [row[0] for row in rows] == ["page 1/2", "page 2/2"]
    assert rows[0][1] == "primary" and "alternative" in rows[0][2]


def test_sync_stale_sensitive_cleanup_and_migration(tmp_path, monkeypatch):
    import rag_sync
    import rag_service
    root, _, db = _configure_instance(tmp_path, monkeypatch)
    source = root / "manual.txt"
    source.write_text("ordinary hydraulic")
    assert rag_service.sync_kb()["documents"] == 1
    source.write_text("api_key = '" + "a" * 30 + "'")
    assert rag_service.sync_kb()["removed"] == 1
    assert str(source) not in rag_service.search("hydraulic")
    source.write_text("ordinary hydraulic")
    assert rag_service.sync_kb()["documents"] == 1
    source.unlink()
    assert str(source) not in rag_service.search("hydraulic")
    assert rag_service.sync_kb()["removed"] == 1
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE kb_chunks_fts")
        conn.execute("CREATE VIRTUAL TABLE kb_chunks_fts USING fts5(path,chunk_index,name,location,content)")
    conn = rag_sync._connect_kb()
    assert "search_text" in {r[1] for r in conn.execute("PRAGMA table_info(kb_chunks_fts)")}
    assert conn.execute("SELECT ready FROM kb_semantic_state").fetchone() == (0,)
    conn.close()


def test_jobs_abandon_prune_and_queued_cancel(tmp_path, monkeypatch):
    import rag_service as s
    _configure_instance(tmp_path, monkeypatch)
    for i in range(12):
        job = f"sync-{i:012x}"
        state = s._new_job_state(job)
        state["status"] = "done"
        s._write_job(s._job_paths(job)[0], state)
    s._prune_job_history()
    assert len(s._job_files_newest_first()) == s._JOB_HISTORY_MAX
    abandoned = s._new_job_state("sync-aaaaaaaaaaaa")
    abandoned["updated_at"] = 0
    assert s._effective_job_state(abandoned)["error"] == "worker_not_running"
    job = "sync-bbbbbbbbbbbb"
    status, cancel = s._job_paths(job)
    s._write_job(status, s._new_job_state(job))
    entered = threading.Event()
    def worker():
        entered.set()
        s.run_persistent_job(job, status, cancel)
    with s._kb_lock(exclusive=True):
        thread = threading.Thread(target=worker)
        thread.start()
        assert entered.wait(1)
        assert "cancelling" in s.cancel_sync(job)
        thread.join(2)
        assert not thread.is_alive()
    assert s.sync_status(job).startswith("status=cancelled")


def test_fresh_stdio_listing_dispatch_and_persistent_job(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    root = tmp_path / "source"
    root.mkdir()
    (root / "manual.txt").write_text("hydraulic fresh stdio marker")
    env = {**os.environ, "RAGDOC_KNOWLEDGE_DIR": str(root), "RAGDOC_DATA_DIR": str(tmp_path / "state"), "RAGDOC_DB": str(tmp_path / "state" / "ragdoc.sqlite3"), "RAGDOC_SEMANTIC": "0", "RAGDOC_OCR_REWRITE": "0"}
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve().parents[1] / "rag_server.py")], env=env)
    async def call(name, args):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert {t.name for t in listing.tools} == {"rag_search", "rag_manage", "rag_status"}
                assert listing.tools[0].inputSchema
                return await session.call_tool(name, args)
    async def exercise():
        started = await call("rag_manage", {"action": "sync"})
        assert not started.isError
        job = _job_id(started.content[0].text)
        status = await call("rag_status", {"view": "job", "job_id": job})
        assert "status=done" in status.content[0].text
        result = await call("rag_search", {"query": "hydraulic"})
        assert not result.isError and "manual.txt" in result.content[0].text
        invalid = await call("rag_manage", {"action": "delete"})
        assert invalid.isError
        legacy = await call("sync_kb", {})
        assert legacy.isError
    asyncio.run(exercise())


@pytest.mark.parametrize("corruption", ["nan", "zero", "short"])
def test_semantic_reuse_repairs_corrupt_vectors_and_exact_cosine(tmp_path, monkeypatch, corruption):
    import numpy as np
    import semantic_worker as w
    import rag_service
    root, _, db = _configure_instance(tmp_path, monkeypatch)
    (root / "manual.txt").write_text("hydraulic marker")
    rag_service.sync_kb()
    class Piece:
        id = 7
        end = 1
    class SP:
        def encode(self, *_args, **_kwargs):
            return type("Proto", (), {"pieces": [Piece()]})()
    monkeypatch.setattr(w, "_load_tokenizer", lambda _: SP())
    monkeypatch.setattr(w, "word_boundary_offsets", lambda _: [1])
    monkeypatch.setattr(w, "_load_runtime", lambda _: (np, object()))
    def encode(_np, _session, rows):
        vectors = np.zeros((len(rows), w.config.SEMANTIC_DIM), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors
    monkeypatch.setattr(w, "_encode_raw_batches", encode)
    model = tmp_path / "snapshot"
    assert w.sync(db, model)["embedded"] == 1
    assert w.sync(db, model)["reused"] == 1
    if corruption == "short":
        blob = b"bad"
    else:
        blob = np.full(w.config.SEMANTIC_DIM, np.nan if corruption == "nan" else 0, dtype="<f2").tobytes()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE kb_semantic_chunks SET embedding=?", (blob,))
    result = w.sync(db, model)
    assert result["embedded"] == 1
    assert result["reused"] == 0
    hits = w.query(db, model, "query", 8)["hits"]
    assert len(hits) == 1 and hits[0]["score"] == 1.0


@pytest.mark.parametrize("mode", ["fragmented", "no_newline", "overlong", "closed_stdout", "crash"])
def test_semantic_protocol_cleanup(tmp_path, monkeypatch, mode):
    import semantic_client as c
    import config
    import subprocess
    code = {
        "fragmented": "import os,time\nos.write(1,b'{\"type\":\"progress\",\"processed_parents\":1}\\n{\"type\":');time.sleep(.02);os.write(1,b'\"result\",\"result\":{\"status\":\"ok\",\"vectors\":1}}\\n')\n",
        "no_newline": "print('{\"type\":\"result\",\"result\":{\"status\":\"ok\",\"vectors\":1}}',end='')\n",
        "overlong": "import os,time\nos.write(1,b'x'*1100000);time.sleep(20)\n",
        "closed_stdout": "import os,time\nos.close(1);os.close(2);time.sleep(20)\n",
        "crash": "import sys\nprint('malformed');sys.exit(1)\n",
    }[mode]
    worker = tmp_path / "worker.py"
    worker.write_text(code)
    monkeypatch.setattr(c, "_WORKER", worker)
    monkeypatch.setattr(c, "_model_root", lambda: tmp_path)
    monkeypatch.setattr(c, "model_status", lambda: {"enabled": True, "ready": True})
    monkeypatch.setattr(config, "SEMANTIC_SYNC_TIMEOUT_SECONDS", .3)
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(c.subprocess, "Popen", spawn)
    progress = []
    start = time.monotonic()
    result = c.refresh(tmp_path / "unused.sqlite", on_progress=progress.append)
    assert time.monotonic() - start < 2
    if mode in {"fragmented", "no_newline"}:
        assert result == {"status": "ok", "vectors": 1}
    else:
        assert result is None
    if mode == "fragmented":
        assert len(progress) == 1 and progress[0]["processed_parents"] == 1
    assert all(child.poll() is not None and child.stdout.closed for child in children)


def test_lexical_filename_alias_punctuation_and_output_caps(tmp_path, monkeypatch):
    import rag_service
    import rag_search
    root, _, _ = _configure_instance(tmp_path, monkeypatch)
    target = root / "HYDRAULIC_REPORT.txt"
    target.write_text("ภาพ hydraulic pressure " + "filler " * 500)
    (root / "partial.txt").write_text("hydraulic filler")
    rag_service.sync_kb()
    exact = rag_service.search("hydraulic_report.txt")
    assert exact.index(str(target)) >= 0
    assert str(target) in rag_service.search("hydraulic, pressure!?")
    assert str(target) in rag_service.search("รูป")
    assert len(rag_service._cap("x" * 12000)) == 8000
    row = (str(target), target.name, "line 1", "content", 10, "Text")
    fused = rag_search._rrf_merge([row, row], [row])
    assert fused == [row]


def test_ui_malformed_inputs_and_origin_negative_controls(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import web_ui
    _configure_instance(tmp_path, monkeypatch)
    client = TestClient(web_ui.app)
    origin = {"Origin": next(iter(web_ui._ALLOWED_ORIGINS))}
    for payload in ([], {}, {"query": ""}, {"query": "x" * 4001}, {"query": "ok", "mode": "delete"}):
        assert client.post("/api/search", headers=origin, json=payload).status_code == 400
    assert client.post("/api/search", headers=origin, content="{").status_code == 400
    assert client.post("/api/sync", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/open-file", headers=origin, json={"path": str(tmp_path / "outside")}).status_code == 403
    assert client.get("/api/jobs/invalid").json()["job"]["status"] == "not_found"


@pytest.mark.parametrize("trial", range(10))
def test_simultaneous_start_status_cancel(tmp_path, monkeypatch, trial):
    from concurrent.futures import ThreadPoolExecutor
    import rag_service as s
    _configure_instance(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    def start():
        barrier.wait(timeout=2)
        return s.start_sync()
    with s._kb_lock(exclusive=True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(start) for _ in range(2)]
            results = [future.result(timeout=2) for future in futures]
        assert sum(text.startswith("status=started") for text in results) == 1
        assert sum(text.startswith("status=already_running") for text in results) == 1
        job = _job_id(results[0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(s.sync_status, job)
            cancel = pool.submit(s.cancel_sync, job)
            assert "job_id=" + job in read.result(timeout=2)
            assert "status=cancelling" in cancel.result(timeout=2)
    assert _wait_terminal(s, job).startswith("status=cancelled")


def test_pptx_uses_presentation_order_not_member_numbers(tmp_path):
    from helpers.document_extract import iter_document_units
    path = tmp_path / "reordered.pptx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("ppt/presentation.xml", '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="257" r:id="second"/><p:sldId id="256" r:id="first"/></p:sldIdLst></p:presentation>')
        package.writestr("ppt/_rels/presentation.xml.rels", '<Relationships><Relationship Id="first" Target="slides/slide1.xml"/><Relationship Id="second" Target="slides/slide2.xml"/></Relationships>')
        for number, text in ((1, "Displayed second"), (2, "Displayed first")):
            package.writestr(f"ppt/slides/slide{number}.xml", f'<sld><sp><txBody><p><t>{text}</t></p></txBody></sp></sld>')
    assert list(iter_document_units(path)) == ["--- Slide 1 ---", "Displayed first", "--- Slide 2 ---", "Displayed second"]


@pytest.mark.parametrize("target", ["../outside.xml", "slides/missing.xml"])
def test_pptx_rejects_broken_relationships(tmp_path, target):
    from helpers.document_extract import iter_document_units
    path = tmp_path / "broken.pptx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("ppt/presentation.xml", '<presentation xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sldId r:id="slide"/></presentation>')
        package.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships><Relationship Id="slide" Target="{target}"/></Relationships>')
    with pytest.raises(RuntimeError, match="relationships"):
        list(iter_document_units(path))


@pytest.mark.parametrize("suffix", [".ods", ".odp"])
def test_odf_encoded_separators_preserve_word_boundaries(tmp_path, suffix):
    from helpers.document_extract import iter_document_units
    path = tmp_path / ("spaces" + suffix)
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("content.xml", '<doc xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><text:p>hydraulic<text:s/><text:span>pump<text:tab/>pressure</text:span><text:line-break/>marker</text:p></doc>')
    assert list(iter_document_units(path)) == ["hydraulic pump\tpressure\nmarker"]
