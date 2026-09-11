"""Focused tests for standalone vision-grounded OCR rewriting."""
from __future__ import annotations

from pathlib import Path
import sys

import fitz

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))


def test_rewrite_batch_contains_real_images_and_disables_thinking(monkeypatch):
    import config
    import ocr_rewrite

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "ข้อความที่แก้แล้ว"
                    }
                }]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(ocr_rewrite.httpx, "Client", FakeClient)
    monkeypatch.setattr(config, "LLM_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit")
    result = ocr_rewrite._llm_rewrite_batch([
        (1, "ข้อควาามที่ OCR อ่านผิด", "data:image/png;base64,REAL_PAGE_PIXELS"),
    ])

    assert result == {1: "ข้อความที่แก้แล้ว"}
    payload = captured["payload"]
    assert payload["model"] == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
    assert payload["enable_thinking"] is False
    assert payload["thinking_budget"] == 0
    blocks = payload["messages"][1]["content"]
    assert any(block["type"] == "image_url" for block in blocks)
    assert blocks[-1]["image_url"]["url"] == "data:image/png;base64,REAL_PAGE_PIXELS"


def test_rewrite_pages_respects_configured_batch_size(monkeypatch):
    import config
    import ocr_rewrite

    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", True)
    monkeypatch.setattr(config, "OCR_REWRITE_BATCH_PAGES", 4)
    monkeypatch.setattr(config, "OCR_REWRITE_BATCH_CHARS", 20_000)
    calls: list[list[tuple[int, str, str]]] = []

    def fake_rewrite(batch):
        calls.append(batch)
        return {page: text for page, text, _image in batch}

    monkeypatch.setattr(ocr_rewrite, "_llm_rewrite_batch", fake_rewrite)
    rows = [
        (page, f"หน้าที่ {page} ภาษาไทย", f"data:image/png;base64,PAGE{page}")
        for page in range(1, 6)
    ]
    result = ocr_rewrite.rewrite_ocr_pages(rows)

    assert [len(call) for call in calls] == [4, 1]
    assert [page for page, _text, _image in calls[0]] == [1, 2, 3, 4]
    assert set(result) == {1, 2, 3, 4, 5}


def test_default_rewrite_is_one_page_per_request():
    import config

    assert config.OCR_REWRITE_BATCH_PAGES == 1


def test_numeric_validation_allows_llm_numeric_repairs():
    import ocr_rewrite

    assert ocr_rewrite._validate_rewrite(
        "หัวข้อ ๓๒.๔ และข้อ ๗๗.๓",
        "หัวข้อ ๓.๒.๔ และข้อ ๗.๓",
    )


def test_json_parser_accepts_fenced_json_with_wrapping_text():
    import ocr_rewrite

    parsed = ocr_rewrite._parse_json_object(
        'ผลลัพธ์:\n```json\n{"pages": [{"page": 1, "text": "แก้แล้ว"}]}\n```'
    )
    assert parsed["pages"][0]["text"] == "แก้แล้ว"




def test_rewrite_failure_falls_back_without_returning_fake_text(monkeypatch):
    import config
    import ocr_rewrite

    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", True)
    monkeypatch.setattr(ocr_rewrite, "_llm_rewrite_batch", lambda _batch: (_ for _ in ()).throw(TimeoutError()))
    rows = [(1, "ข้อความ OCR", "data:image/png;base64,PAGE1")]
    assert ocr_rewrite.rewrite_ocr_pages(rows) == {}


def test_vision_lease_starts_and_stops_only_its_own_server(monkeypatch):
    import config
    import ocr_rewrite

    events: list[str] = []

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == config.LLM_CONTROL_TIMEOUT_SECONDS

        def close(self):
            events.append("close")

    states = iter(("down",))
    monkeypatch.setattr(ocr_rewrite.httpx, "Client", FakeClient)
    monkeypatch.setattr(ocr_rewrite, "_server_state", lambda _client: next(states))
    monkeypatch.setattr(ocr_rewrite, "_control_action", lambda _client, action: events.append(action) or {})
    monkeypatch.setattr(ocr_rewrite, "_wait_for_server", lambda _client: True)
    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", True)
    monkeypatch.setattr(config, "OCR_REWRITE_AUTO_START", True)
    monkeypatch.setattr(config, "OCR_REWRITE_AUTO_STOP", True)

    with ocr_rewrite.vision_server_lease(required=True):
        assert events == ["server/start"]
    assert events == ["server/start", "server/stop", "close"]


def test_vision_lease_leaves_preexisting_server_running(monkeypatch):
    import config
    import ocr_rewrite

    events: list[str] = []

    class FakeClient:
        def __init__(self, timeout):
            pass

        def close(self):
            events.append("close")

    monkeypatch.setattr(ocr_rewrite.httpx, "Client", FakeClient)
    monkeypatch.setattr(ocr_rewrite, "_server_state", lambda _client: "ready")
    monkeypatch.setattr(ocr_rewrite, "_control_action", lambda _client, action: events.append(action) or {})
    monkeypatch.setattr(config, "OCR_REWRITE_ENABLED", True)
    monkeypatch.setattr(config, "OCR_REWRITE_AUTO_STOP", True)

    with ocr_rewrite.vision_server_lease(required=True):
        pass
    assert events == ["close"]


def test_pdf_preflight_detects_scanned_pages(tmp_path):
    import rag_sync

    native = tmp_path / "native.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "native text")
        doc.save(native)

    scanned = tmp_path / "scanned.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(scanned)

    assert rag_sync._pdf_requires_ocr(native) is False
    assert rag_sync._pdf_requires_ocr(scanned) is True


def test_sync_kb_acquires_vision_lease_for_scanned_pdf(tmp_path, monkeypatch):
    import config
    import rag_sync

    source = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(source)

    db_path = tmp_path / "ragdoc.sqlite3"
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)
    monkeypatch.setattr(rag_sync, "_iter_source_files", lambda: iter([("scan.pdf", source)]))
    monkeypatch.setattr(rag_sync, "_index_one", lambda *args, **kwargs: ("inserted", 1))
    monkeypatch.setattr(rag_sync, "_progress", lambda *args, **kwargs: None)

    events: list[str] = []

    class FakeLease:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

    monkeypatch.setattr(
        rag_sync,
        "vision_server_lease",
        lambda *, required: (events.append(f"required={required}") or FakeLease()),
    )

    stats = rag_sync.sync_kb()

    assert stats["ocr_pdf_files"] == 1
    assert events == ["required=True", "enter", "exit"]


def test_pdf_units_send_page_image_to_rewrite_before_yielding_content(tmp_path, monkeypatch):
    import rag_sync

    source = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(source)

    seen: list[tuple[int, str, str]] = []

    def fake_ocr_page(_page, page_number):
        return ([{"text": f"OCR ผิด หน้า {page_number}", "candidates": [f"OCR ผิด หน้า {page_number}"]}], f"data:image/png;base64,PAGE{page_number}")

    def fake_rewrite(rows):
        rows = list(rows)
        seen.extend(rows)
        return {1: "ข้อความแก้แล้ว หน้า 1"}

    monkeypatch.setattr(rag_sync, "_ocr_pdf_page", fake_ocr_page)
    monkeypatch.setattr(rag_sync, "rewrite_ocr_pages", fake_rewrite)
    units = list(rag_sync._iter_pdf_units(source))

    assert [page for page, _text, _image in seen] == [1, 2]
    assert all(image.startswith("data:image/") for _page, _text, image in seen)
    assert "ข้อความแก้แล้ว หน้า 1" in units[0][1]
    assert "OCR ผิด หน้า 1" in units[0][2]
    assert "OCR ผิด หน้า 2" in units[1][1]
