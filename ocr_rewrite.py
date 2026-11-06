"""Vision-grounded OCR correction for scanned PDF pages.

The module is standalone: it speaks the local OpenAI-compatible
``mlx_vlm.server`` API directly and never imports the host agent.  Each
request carries one detailed page image plus its OCR text.  If the server is
unavailable or the response fails validation, callers keep the original OCR.
"""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
import difflib
import json
import logging
import re
import time

import httpx

import config
from helpers.thai_text_normalize import repair_thai_orthographic_order

log = logging.getLogger(__name__)

_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
OCRPage = tuple[int, str, str]


def _server_state(client: httpx.Client) -> str:
    """Return ``ready``, ``wrong_model``, or ``down`` for the shared VLM."""
    try:
        response = client.get(
            config.LLM_BASE_URL.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        )
        if response.status_code != 200:
            return "down"
        body = response.json()
        ids = [
            str(item.get("id") or "")
            for item in body.get("data", [])
            if isinstance(item, dict)
        ]
        if not ids:
            return "down"
        model_name = config.LLM_MODEL.casefold()
        model_base = model_name.rsplit("/", 1)[-1]
        if any(
            model_name == item.casefold()
            or model_base in item.casefold().rsplit("/", 1)[-1]
            for item in ids
        ):
            return "ready"
        return "wrong_model"
    except Exception:
        return "down"


def _control_action(client: httpx.Client, action: str) -> dict[str, object]:
    """Ask an optional trusted loopback control plane to execute model start/stop."""
    response = client.post(
        f"{config.LLM_CONTROL_URL}/{action}",
        headers={
            "Origin": config.LLM_CONTROL_ORIGIN,
            "Content-Type": "application/json",
        },
        json={},
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or body.get("ok") is False:
        detail = body.get("error", "control request rejected") if isinstance(body, dict) else "invalid control response"
        raise RuntimeError(str(detail))
    return body


def _wait_for_server(client: httpx.Client) -> bool:
    deadline = time.monotonic() + config.LLM_SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        state = _server_state(client)
        if state == "ready":
            return True
        if state == "wrong_model":
            return False
        time.sleep(1.0)
    return False


@contextmanager
def vision_server_lease(*, required: bool):
    """Ensure a shared VLM only for an OCR build that actually needs it.

    The lease never starts a model process directly.  When explicitly enabled it delegates to an optional trusted loopback control endpoint, records whether this build started the
    server, and stops it only when this lease owns the start.  If the control
    plane is unavailable, callers keep the existing OCR fallback behavior.
    """
    started_by_us = False
    if not required or not config.OCR_REWRITE_ENABLED:
        yield
        return

    client = httpx.Client(timeout=config.LLM_CONTROL_TIMEOUT_SECONDS)
    try:
        state = _server_state(client)
        if state == "ready":
            yield
            return
        if state == "wrong_model":
            log.warning(
                "OCR vision rewrite skipped: shared port serves a different model (%s)",
                config.LLM_MODEL,
            )
            yield
            return
        if not config.OCR_REWRITE_AUTO_START:
            log.warning("OCR vision rewrite skipped: automatic VLM start is disabled")
            yield
            return
        try:
            _control_action(client, "server/start")
            started_by_us = True
            if not _wait_for_server(client):
                log.warning("OCR vision rewrite skipped: VLM did not become ready")
        except Exception as exc:
            log.warning(
                "OCR vision rewrite skipped: optional VLM control start failed: %s",
                type(exc).__name__,
            )
        # Consumer exceptions must propagate to the caller, not be mistaken
        # for a startup failure and cause this contextmanager to yield twice.
        yield
    finally:
        if started_by_us and config.OCR_REWRITE_AUTO_STOP:
            try:
                _control_action(client, "server/stop")
            except Exception as exc:
                log.warning(
                    "OCR VLM was started by this build but could not be stopped: %s",
                    type(exc).__name__,
                )
        client.close()


def _clean_text(text: str) -> str:
    lines = [
        repair_thai_orthographic_order(line.rstrip())
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    return "\n".join(lines).strip()


def _rewrite_batches(rows: Iterable[OCRPage]) -> list[list[OCRPage]]:
    batches: list[list[OCRPage]] = []
    current: list[OCRPage] = []
    current_chars = 0
    for row in rows:
        page, text, _image_url = row
        size = len(text)
        if current and (
            len(current) >= config.OCR_REWRITE_BATCH_PAGES
            or current_chars + size > config.OCR_REWRITE_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        # Keep one physical page intact even when it exceeds the preference.
        current.append((page, text, _image_url))
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _rewrite_prompt(batch: list[OCRPage]) -> str:
    payload = [{"page": page, "ocr_text": text} for page, text, _ in batch]
    return (
        "ตรวจภาพจริงของเอกสาร PDF หน้าที่แนบมา แล้วแก้ข้อความ OCR ภาษาไทย "
        "ให้ตรงกับภาพจริงที่สุด ห้ามสรุป ห้ามแปล ห้ามเพิ่มข้อมูล "
        "ห้ามเดาข้อมูลที่มองไม่เห็น และคัดลอกทุกบรรทัดที่มองเห็นให้ครบ "
        "ตัวเลขให้ยึดภาพจริง สามารถแก้ตัวเลขซ้ำ จุดคั่น หรือการจัดกลุ่มได้ "
        "เมื่อเห็นจากภาพชัดเจน แต่ห้ามเปลี่ยนตัวเลขโดยไม่มีหลักฐานจากภาพ "
        "รักษาลำดับบรรทัด หน่วย และชื่อเฉพาะตามเอกสาร "
        "ใช้ OCR text เป็นตัวช่วยค้นตำแหน่ง แต่ให้ยึด pixels ในภาพจริงเป็นหลัก "
        "ตอบเฉพาะข้อความ OCR ที่แก้แล้วของหน้านี้เป็น plain text เท่านั้น "
        "ห้ามตอบ JSON ห้ามใส่ Markdown ห้ามใส่คำอธิบาย และห้ามตัดข้อความให้สั้นลง\n\n"
        + json.dumps({"pages": payload}, ensure_ascii=False)
    )


def _validate_rewrite(source: str, rewritten: str) -> bool:
    if not rewritten.strip():
        return False
    ratio = len(rewritten) / max(1, len(source))
    if ratio < 0.55 or ratio > 1.45:
        return False
    if difflib.SequenceMatcher(None, source, rewritten, autojunk=False).ratio() < 0.72:
        return False
    # OCR frequently misreads Thai digits and section-number punctuation.
    # The image-grounded VLM is the better authority for those corrections;
    # keep only structural guards here so valid numeric repairs are not rejected.
    return True


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _parse_json_object(text: str) -> object:
    """Parse JSON even when the model wraps it in a short prose/code fence."""
    cleaned = _THINK_RE.sub("", text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        # Some VLM responses add one sentence before/after an otherwise valid
        # JSON object. Recover only a complete outer object; malformed or
        # truncated JSON still fails closed to the original OCR.
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start < 0 or end <= start:
            raise first_error
        return json.loads(cleaned[start:end])


def _response_page_text(content: str, page: int) -> str:
    """Prefer plain text for one-page calls, with compatibility for old JSON."""
    cleaned = _THINK_RE.sub("", content).strip()
    if cleaned.startswith("```"):
        fenced = re.sub(r"^```(?:text|plain|txt)?\s*", "", cleaned, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced).strip()
        if fenced.startswith("{"):
            parsed = _parse_json_object(fenced)
            pages = parsed.get("pages") if isinstance(parsed, dict) else None
            if isinstance(pages, list):
                for item in pages:
                    if isinstance(item, dict) and int(item.get("page", -1)) == page:
                        return str(item.get("text") or "").strip()
        return fenced
    if cleaned.startswith("{"):
        parsed = _parse_json_object(cleaned)
        pages = parsed.get("pages") if isinstance(parsed, dict) else None
        if isinstance(pages, list):
            for item in pages:
                if isinstance(item, dict) and int(item.get("page", -1)) == page:
                    return str(item.get("text") or "").strip()
        raise ValueError("LLM OCR rewrite JSON has no matching page")
    return cleaned


def _llm_rewrite_batch(batch: list[OCRPage]) -> dict[int, str]:
    """Send one no-think multimodal request for one detailed PDF page."""
    if len(batch) != 1:
        raise ValueError("OCR rewrite requires exactly one page per request")
    content: list[dict[str, object]] = [
        {"type": "text", "text": _rewrite_prompt(batch)}
    ]
    for page, _text, image_url in batch:
        content.append({"type": "text", "text": f"\nภาพจริง PDF หน้า {page}:"})
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative vision-grounded OCR corrector. "
                    "Read the attached page pixels. Return only the corrected "
                    "text for this one page as plain text. "
                    "Never summarize, translate, or invent content."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": config.OCR_REWRITE_MAX_TOKENS,
        "enable_thinking": False,
        "thinking_budget": 0,
        "repetition_penalty": config.LLM_REPETITION_PENALTY,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=config.OCR_REWRITE_TIMEOUT_SECONDS) as client:
        response = client.post(
            config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()

    page = batch[0][0]
    content_text = _message_text(body["choices"][0]["message"]["content"])
    return {page: _response_page_text(content_text, page)}


def rewrite_ocr_pages(rows: Iterable[OCRPage]) -> dict[int, str]:
    """Return accepted rewrites keyed by page; absent pages mean OCR fallback."""
    prepared = [
        (int(page), str(text), str(image_url))
        for page, text, image_url in rows
        if text and _THAI_RE.search(text) and image_url.startswith("data:image/")
    ]
    if not config.OCR_REWRITE_ENABLED or not prepared:
        return {}

    accepted: dict[int, str] = {}
    batches = _rewrite_batches(prepared)
    for index, batch in enumerate(batches, 1):
        try:
            rewritten = _llm_rewrite_batch(batch)
        except Exception as exc:
            log.warning(
                "OCR vision rewrite batch %s/%s failed; keeping original OCR: %s",
                index,
                len(batches),
                type(exc).__name__,
            )
            continue
        for page, source, _image_url in batch:
            candidate = _clean_text(rewritten.get(page, ""))
            if _validate_rewrite(source, candidate):
                accepted[page] = candidate
            else:
                log.warning("OCR vision rewrite page %s failed validation; keeping original OCR", page)
    return accepted
