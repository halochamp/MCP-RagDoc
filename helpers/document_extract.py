"""Lean, read-only text extraction for PDF and Office documents.

extract_text() stays the full-body, byte-identical entry point existing
tests and small-file callers use. Large-file callers (indexer.py,
tools/read_file.py) that only need a bounded prefix or a scattered sample
use the streaming primitives below (iter_document_units, pdf_page_count,
pdf_page_text) instead, so a multi-hundred-page PDF or huge XLSX never has
its whole extracted body held in memory at once (PLAN.md §9.1).
"""
from __future__ import annotations

import shutil
import posixpath
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from helpers.vision_ocr import read_layout


_DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".docm", ".odt", ".xlsx", ".xls", ".xlsm",
    ".ppt", ".pptx", ".pptm", ".ods", ".odp",
}
_SCANNED_PDF_OCR_MAX_PAGES = 30
_UNIT_SEPARATORS = {
    ".pdf": "\n\n", ".doc": "\n", ".docx": "\n", ".docm": "\n", ".odt": "\n",
    ".xlsx": "\n", ".xls": "\n", ".xlsm": "\n", ".ppt": "\n", ".pptx": "\n",
    ".pptm": "\n", ".ods": "\n", ".odp": "\n",
}
_TEXTUTIL_DOCUMENT_EXTENSIONS = {".doc", ".odt"}
_ODF_CONTENT_EXTENSIONS = {".ods", ".odp"}


def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in _DOCUMENT_EXTENSIONS


def unit_separator(suffix: str) -> str:
    """Separator extract_text() used to join this format's units."""
    return _UNIT_SEPARATORS.get(suffix.lower(), "\n")


def pdf_page_count(path: Path) -> int:
    import fitz
    with fitz.open(path) as pdf:
        return len(pdf)


def pdf_page_text(path: Path, index: int) -> str:
    """Extract one page's text on demand (0-indexed), without touching any
    other page -- lets a caller re-fetch a handful of scattered pages for
    sampling without ever extracting the whole document."""
    import fitz
    with fitz.open(path) as pdf:
        return pdf[index].get_text("text").strip()


def document_page_count(path: Path) -> int | None:
    """Return the document's stored page count when the format provides one.

    Word stores the last pagination count in ``docProps/app.xml``.  It is the
    only page-count signal available without launching a GUI renderer; the
    page-break markers in ``word/document.xml`` then locate each extracted
    paragraph within that saved pagination.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            return pdf_page_count(path)
        except (OSError, RuntimeError, ValueError):
            return None
    if suffix not in {".docx", ".docm"}:
        return None
    try:
        with zipfile.ZipFile(path) as package:
            raw = package.read("docProps/app.xml")
        root = ET.fromstring(raw)
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return None
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "Pages":
            continue
        try:
            value = int((element.text or "").strip())
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def iter_document_units_with_locations(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(location, text)`` pairs with page-aware DOCX locations.

    The historical ``iter_document_units`` API remains string-only for
    callers such as ``read_file``.  The KB/index search path uses this richer
    companion so a Word paragraph is reported as ``page 18/79`` instead of an
    opaque extraction-unit range.  Consecutive explicit and
    ``lastRenderedPageBreak`` markers are treated as one break because Word
    commonly writes both for the same boundary.
    """
    suffix = path.suffix.lower()
    if suffix in {".docx", ".docm"}:
        total = document_page_count(path)
        for page, text in _iter_docx_page_units(path):
            total_text = f"/{total}" if total else ""
            yield f"page {page}{total_text}", text
        return

    total = document_page_count(path)
    for index, unit in enumerate(iter_document_units(path), 1):
        if suffix == ".pdf":
            location = f"page {index}/{total}" if total else f"page {index}"
        else:
            location = f"unit {index}"
        yield location, unit


def _iter_docx_page_units(path: Path) -> Iterator[tuple[int, str]]:
    """Stream DOCX paragraphs while applying Word's saved page markers."""
    try:
        package = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด Word package ไม่สำเร็จ: {exc}") from exc
    with package:
        try:
            stream = package.open("word/document.xml")
        except (KeyError, OSError) as exc:
            raise RuntimeError(f"อ่าน Word document.xml ไม่สำเร็จ: {exc}") from exc
        page = 1
        with stream:
            try:
                pending_explicit_page_break = False
                for event, element in ET.iterparse(stream, events=("end",)):
                    if element.tag.rsplit("}", 1)[-1] != "p":
                        continue
                    page, segments = _docx_paragraph_page_segments(
                        element,
                        page,
                        skip_leading_rendered_break=(
                            pending_explicit_page_break
                            and _docx_starts_with_rendered_break(element)
                        ),
                    )
                    for segment_page, text in segments:
                        if text.strip():
                            yield segment_page, text.strip()
                    pending_explicit_page_break = _docx_ends_with_explicit_page_break(element)
                    if _docx_has_next_page_section(element):
                        page += 1
                    # Release the paragraph and its runs after extraction;
                    # this keeps large Word packages bounded while retaining
                    # the exact paragraph order used by the page map.
                    element.clear()
            except ET.ParseError as exc:
                raise RuntimeError(f"Word document.xml ไม่ถูกต้อง: {exc}") from exc


def _docx_paragraph_page_segments(
    paragraph: ET.Element,
    page: int,
    *,
    skip_leading_rendered_break: bool = False,
) -> tuple[int, list[tuple[int, str]]]:
    segments: list[tuple[int, str]] = []
    pieces: list[str] = []
    pending_page_break = False
    current_page = page
    page_type = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
    ignored_leading_rendered_break = False
    run_content = _docx_run_content_map(paragraph)

    for element in paragraph.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        in_run = run_content.get(id(element), False)
        is_page_break = (
            in_run
            and (
                local_name == "lastRenderedPageBreak"
                or (local_name == "br" and element.attrib.get(page_type) == "page")
            )
        )
        if is_page_break:
            if (
                skip_leading_rendered_break
                and local_name == "lastRenderedPageBreak"
                and not pieces
                and not ignored_leading_rendered_break
            ):
                # Word often writes a manual page break at the end of one
                # paragraph and repeats the same boundary as a rendered break
                # at the start of the next paragraph.  Count that boundary
                # once, even when the two markers are in separate paragraphs.
                ignored_leading_rendered_break = True
                pending_page_break = True
                continue
            if not pending_page_break:
                text = "".join(pieces).strip()
                if text:
                    segments.append((current_page, text))
                pieces = []
                current_page += 1
            pending_page_break = True
            continue
        if in_run and local_name == "t":
            pieces.append(element.text or "")
            pending_page_break = False
        elif in_run and local_name == "tab":
            pieces.append("\t")
            pending_page_break = False
        elif in_run and local_name == "br":
            pieces.append("\n")
            pending_page_break = False

    text = "".join(pieces).strip()
    if text:
        segments.append((current_page, text))
    return current_page, segments


def _docx_run_content_map(paragraph: ET.Element) -> dict[int, bool]:
    """Mark descendants that are actual run content, not paragraph styling."""
    parents: dict[int, ET.Element] = {}
    for parent in paragraph.iter():
        for child in parent:
            parents[id(child)] = parent

    result: dict[int, bool] = {}
    for element in paragraph.iter():
        cursor = element
        in_run = False
        while cursor is not paragraph:
            if cursor.tag.rsplit("}", 1)[-1] == "r":
                in_run = True
                break
            cursor = parents.get(id(cursor))
            if cursor is None:
                break
        result[id(element)] = in_run
    return result


def _docx_starts_with_rendered_break(paragraph: ET.Element) -> bool:
    """Return whether the first content marker is a rendered page break."""
    run_content = _docx_run_content_map(paragraph)
    for element in paragraph.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if not run_content.get(id(element), False):
            continue
        if local_name == "lastRenderedPageBreak":
            return True
        if local_name == "br":
            return False
        if local_name == "t" and (element.text or ""):
            return False
        # ``w:tab`` is also used in paragraph-formatting properties as a tab
        # stop.  It is content only when it appears after a run; ignoring it
        # here lets a rendered break immediately after ``w:pPr`` be detected.
    return False


def _docx_ends_with_explicit_page_break(paragraph: ET.Element) -> bool:
    """Return whether the paragraph ends with a manual page-break marker."""
    page_type = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
    run_content = _docx_run_content_map(paragraph)
    last_marker: str | None = None
    has_content_after_marker = False
    for element in paragraph.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if not run_content.get(id(element), False):
            continue
        if local_name == "lastRenderedPageBreak":
            last_marker = "rendered"
            has_content_after_marker = False
        elif local_name == "br" and element.attrib.get(page_type) == "page":
            last_marker = "explicit"
            has_content_after_marker = False
        elif local_name == "t" and (element.text or ""):
            has_content_after_marker = True
        elif local_name == "tab":
            has_content_after_marker = True
    return last_marker == "explicit" and not has_content_after_marker


def _docx_has_next_page_section(paragraph: ET.Element) -> bool:
    """Return whether a paragraph ends a section that starts on a new page."""
    for element in paragraph.iter():
        if element.tag.rsplit("}", 1)[-1] != "sectPr":
            continue
        section_type = next(
            (
                child.attrib.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                )
                for child in element
                if child.tag.rsplit("}", 1)[-1] == "type"
            ),
            None,
        )
        return section_type != "continuous"
    return False


def iter_document_units(path: Path) -> Iterator[str]:
    """Yield one formatted unit at a time -- PDF page, DOCX paragraph/table
    row, XLSX/XLS sheet header/row, or PPTX slide/shape/table row -- in the
    same order and exact text extract_text() joins with unit_separator().
    A caller only needing a bounded prefix can stop iterating early; the
    unyielded remainder of the document is never extracted.

    Scanned (image-only) PDFs are the one exception: their content isn't
    discoverable page-by-page without running OCR, so this yields the whole
    OCR'd result (already bounded to _SCANNED_PDF_OCR_MAX_PAGES) as a single
    unit, exactly matching extract_text()'s fallback.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        with fitz.open(path) as pdf:
            has_content = any(page.get_text("text").strip() for page in pdf)
        if not has_content:
            ocr_text = _ocr_scanned_pdf(path)
            if ocr_text.startswith("[error]"):
                raise RuntimeError(ocr_text)
            yield ocr_text
            return
        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf, 1):
                yield f"--- PDF page {number} ---\n{page.get_text('text').strip()}"
    elif suffix in {".docx", ".docm"}:
        yield from _iter_docx_units(path)
    elif suffix in _TEXTUTIL_DOCUMENT_EXTENSIONS:
        text = _extract_textutil_document(path)
        if text.startswith("[error]"):
            raise RuntimeError(text)
        yield text
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        yield from _iter_workbook_units(path, suffix)
    elif suffix in {".pptx", ".pptm"}:
        yield from _iter_pptx_units(path)
    elif suffix in _ODF_CONTENT_EXTENSIONS:
        yield from _iter_odf_content_units(path)
    elif suffix == ".ppt":
        text = _extract_spotlight_document(path)
        if text.startswith("[error]"):
            raise RuntimeError(text)
        yield text
    else:
        raise ValueError(f"no streaming extractor for {suffix}")


def _iter_docx_units(path: Path) -> Iterator[str]:
    """Stream WordprocessingML without building python-docx's full object graph.

    Preserve the historical output order exactly: all top-level paragraphs
    first, then tables with their row text. document.xml is parsed twice so
    memory stays bounded while matching the old paragraph/table behavior.
    """
    try:
        package = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด Word package ไม่สำเร็จ: {exc}") from exc
    with package:
        member = "word/document.xml"
        try:
            stream = package.open(member)
        except (KeyError, OSError) as exc:
            raise RuntimeError(f"อ่าน Word document.xml ไม่สำเร็จ: {exc}") from exc
        with stream:
            table_depth = 0
            try:
                for event, element in ET.iterparse(stream, events=("start", "end")):
                    local_name = element.tag.rsplit("}", 1)[-1]
                    if event == "start" and local_name == "tbl":
                        table_depth += 1
                    elif event == "end" and local_name == "p" and table_depth == 0:
                        text = _ooxml_text(element).strip()
                        if text:
                            yield text
                        element.clear()
                    elif event == "end" and local_name == "tbl":
                        table_depth -= 1
                        element.clear()
            except ET.ParseError as exc:
                raise RuntimeError(f"Word document.xml ไม่ถูกต้อง: {exc}") from exc

        try:
            stream = package.open(member)
        except (KeyError, OSError) as exc:
            raise RuntimeError(f"อ่าน Word document.xml ไม่สำเร็จ: {exc}") from exc
        with stream:
            table_number = 0
            in_table = 0
            try:
                for event, element in ET.iterparse(stream, events=("start", "end")):
                    local_name = element.tag.rsplit("}", 1)[-1]
                    if event == "start" and local_name == "tbl":
                        in_table += 1
                        if in_table == 1:
                            table_number += 1
                            yield f"--- Table {table_number} ---"
                    elif event == "end" and local_name == "tr" and in_table == 1:
                        cells = [
                            _ooxml_text(cell).replace("\n", " ").strip()
                            for cell in element
                            if cell.tag.rsplit("}", 1)[-1] == "tc"
                        ]
                        if any(cells):
                            yield " | ".join(cells)
                        element.clear()
                    elif event == "end" and local_name == "tbl":
                        in_table -= 1
                        element.clear()
            except ET.ParseError as exc:
                raise RuntimeError(f"Word document.xml ไม่ถูกต้อง: {exc}") from exc


def _iter_workbook_units(path: Path, suffix: str) -> Iterator[str]:
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                yield f"--- Sheet: {sheet.title} ---"
                for row in sheet.iter_rows(values_only=True):
                    cells = [str(value).strip() if value is not None else "" for value in row]
                    if any(cells):
                        yield " | ".join(cells)
        finally:
            workbook.close()
    else:
        import xlrd
        workbook = xlrd.open_workbook(path, on_demand=True)
        try:
            for sheet in workbook.sheets():
                yield f"--- Sheet: {sheet.name} ---"
                for row_index in range(sheet.nrows):
                    cells = [str(value).strip() if value is not None else "" for value in sheet.row_values(row_index)]
                    if any(cells):
                        yield " | ".join(cells)
        finally:
            workbook.release_resources()


def _iter_pptx_units(path: Path) -> Iterator[str]:
    """Stream visible text from OOXML slides without python-pptx.

    PowerPoint files are ZIP packages.  Parsing one shape at a time avoids
    both an undeclared runtime dependency and python-pptx's full-deck object
    graph, which is unnecessary for this read-only text path.
    """
    try:
        package = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด PowerPoint package ไม่สำเร็จ: {exc}") from exc
    with package:
        slides: list[tuple[int, str]] = []
        for name in package.namelist():
            prefix = "ppt/slides/slide"
            suffix = ".xml"
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            number_text = name[len(prefix) : -len(suffix)]
            if number_text.isdigit():
                slides.append((int(number_text), name))
        ordered = [member for _number, member in sorted(slides)]
        if "ppt/presentation.xml" in package.namelist():
            try:
                presentation = ET.fromstring(package.read("ppt/presentation.xml"))
                relationships = ET.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
                targets = {
                    item.attrib["Id"]: item.attrib["Target"]
                    for item in relationships
                    if item.tag.rsplit("}", 1)[-1] == "Relationship"
                    and item.attrib.get("TargetMode") != "External"
                }
                ordered = []
                members = set(package.namelist())
                for slide in presentation.iter():
                    if slide.tag.rsplit("}", 1)[-1] != "sldId":
                        continue
                    target = targets[slide.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                    member = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "ppt/" + target)
                    if not member.startswith("ppt/slides/") or member not in members:
                        raise ValueError("invalid presentation slide target")
                    ordered.append(member)
            except (KeyError, ET.ParseError, ValueError) as exc:
                raise RuntimeError("invalid PowerPoint presentation relationships") from exc
        for number, member in enumerate(ordered, 1):
            yield f"--- Slide {number} ---"
            try:
                stream = package.open(member)
            except (KeyError, OSError) as exc:
                raise RuntimeError(f"อ่าน slide {number} ไม่สำเร็จ: {exc}") from exc
            with stream:
                try:
                    for _event, element in ET.iterparse(stream, events=("end",)):
                        local_name = element.tag.rsplit("}", 1)[-1]
                        if local_name == "sp":
                            text_body = next(
                                (child for child in element if child.tag.rsplit("}", 1)[-1] == "txBody"),
                                None,
                            )
                            text = _ooxml_text(text_body)
                            if text:
                                yield text
                            element.clear()
                        elif local_name == "graphicFrame":
                            table = next(
                                (child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == "tbl"),
                                None,
                            )
                            if table is not None:
                                for row in (child for child in table if child.tag.rsplit("}", 1)[-1] == "tr"):
                                    cells = [
                                        _ooxml_text(cell).replace("\n", " ").strip()
                                        for cell in row
                                        if cell.tag.rsplit("}", 1)[-1] == "tc"
                                    ]
                                    if any(cells):
                                        yield " | ".join(cells)
                            element.clear()
                except ET.ParseError as exc:
                    raise RuntimeError(f"slide{number}.xml ไม่ถูกต้อง: {exc}") from exc


def _ooxml_text(container: ET.Element | None) -> str:
    if container is None:
        return ""
    paragraphs: list[str] = []
    for paragraph in (item for item in container.iter() if item.tag.rsplit("}", 1)[-1] == "p"):
        pieces: list[str] = []
        for item in paragraph.iter():
            local_name = item.tag.rsplit("}", 1)[-1]
            if local_name == "t":
                pieces.append(item.text or "")
            elif local_name == "br":
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _iter_odf_content_units(path: Path) -> Iterator[str]:
    """Yield visible paragraph/heading text from an ODS or ODP content.xml.

    OpenDocument files are ZIP packages.  Their content.xml uses text:p/text:h
    for spreadsheet cells and presentation text boxes alike, so parsing those
    elements preserves user-visible text without adding an ODF dependency.
    """
    try:
        with zipfile.ZipFile(path) as package, package.open("content.xml") as stream:
            for _event, element in ET.iterparse(stream, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] in {"p", "h"}:
                    text = _odf_text(element).strip()
                    if text:
                        yield text
                    element.clear()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด OpenDocument package ไม่สำเร็จ: {exc}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"content.xml ไม่ถูกต้อง: {exc}") from exc


def _odf_text(element: ET.Element) -> str:
    """Keep encoded separators and nested tails; do not expand untrusted space counts."""
    pieces: list[str] = []
    stack: list[ET.Element | str] = [element]
    separators = {"s": " ", "tab": "\t", "line-break": "\n"}
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            pieces.append(item)
            continue
        pieces.append(separators.get(item.tag.rsplit("}", 1)[-1], item.text or ""))
        for child in reversed(list(item)):
            stack.append(child.tail or "")
            stack.append(child)
    return "".join(pieces)


def _extract_spotlight_document(path: Path) -> str:
    """Read text imported by macOS Spotlight for a legacy PowerPoint file."""
    mdls = shutil.which("mdls")
    if not mdls:
        return "[error] อ่านไฟล์ .ppt ไม่ได้: ไม่พบ macOS mdls"
    try:
        result = subprocess.run(
            [mdls, "-raw", "-name", "kMDItemTextContent", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[error] สกัดข้อความจาก .ppt ไม่สำเร็จ: {exc}"
    text = result.stdout.strip()
    if result.returncode != 0 or not text or text == "(null)":
        return "[error] ไฟล์ .ppt ไม่มีข้อความจาก macOS Spotlight importer"
    return text


def extract_text(path: Path) -> str:
    """Return document text or an ``[error]`` string; never writes a file.

    Built on iter_document_units() so the full-body path and the streaming
    (bounded-prefix/sample) path used by indexer.py/read_file.py can never
    silently diverge -- there is exactly one extraction implementation per
    format, just two ways of consuming it.
    """
    suffix = path.suffix.lower()
    try:
        if suffix not in _DOCUMENT_EXTENSIONS:
            return f"[error] ไม่รองรับไฟล์ {suffix}"
        text = unit_separator(suffix).join(iter_document_units(path))
        return text.strip() if suffix == ".pdf" else text
    except Exception as exc:
        return f"[error] สกัดข้อความจาก {suffix} ไม่สำเร็จ: {exc}"


def _ocr_scanned_pdf(path: Path) -> str:
    """Port Agent Max's bounded PDF rasterization + Apple Vision OCR fallback."""
    try:
        import fitz
        with fitz.open(path) as pdf:
            total = len(pdf)
            page_limit = min(total, _SCANNED_PDF_OCR_MAX_PAGES)
            blocks: list[str] = []
            for index in range(page_limit):
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                        temporary = Path(handle.name)
                    pdf[index].get_pixmap(dpi=150).save(str(temporary))
                    lines = [box["text"] for box in read_layout(str(temporary)) if box.get("text")]
                    if lines:
                        blocks.append(f"--- PDF page {index + 1} (OCR) ---\n" + "\n".join(lines))
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    except Exception as exc:
        return f"[error] OCR PDF สแกนไม่สำเร็จ: {exc}"

    if not blocks:
        return f"[error] PDF สแกนนี้ไม่มีข้อความที่ OCR อ่านได้ ({total} หน้า)"
    note = (
        f"\n\n[หมายเหตุ: OCR พบข้อความ {len(blocks)}/{page_limit} หน้าที่ตรวจจากทั้งหมด {total} หน้า]"
        if total > _SCANNED_PDF_OCR_MAX_PAGES else ""
    )
    header = f"[PDF สแกน — OCR พบข้อความ {len(blocks)}/{page_limit} หน้าที่ตรวจ]"
    return header + "\n\n" + "\n\n".join(blocks) + note


def _extract_textutil_document(path: Path) -> str:
    """Extract a legacy Office text document with macOS's textutil utility."""
    suffix = path.suffix.lower()
    textutil = shutil.which("textutil")
    if not textutil:
        return f"[error] อ่านไฟล์ {suffix} ไม่ได้: ไม่พบ macOS textutil"
    try:
        result = subprocess.run(
            [textutil, "-convert", "txt", "-stdout", "-encoding", "UTF-8", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[error] สกัดข้อความจาก {suffix} ไม่สำเร็จ: {exc}"
    if result.returncode != 0:
        detail = " ".join(result.stderr.split())[:240]
        suffix = f" ({detail})" if detail else ""
        return f"[error] สกัดข้อความจาก {path.suffix.lower()} ไม่สำเร็จ{suffix}"
    text = result.stdout.strip()
    if not text:
        return f"[error] ไฟล์ {path.suffix.lower()} ไม่มีข้อความที่สกัดได้ (อาจเป็นเอกสารสแกนหรือไฟล์เสีย)"
    return text
