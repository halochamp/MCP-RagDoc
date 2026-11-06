"""Deterministic service boundary for standalone MCP-RagDoc.

Search is synchronous. MCP sync is a persistent background job so status and
cancellation survive stdio clients that launch a fresh server process for each
tool call. All job files live in the operator-selected private DATA_DIR.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid

import config
import rag_sync as kb_sync
import semantic_client
import rag_search as _rag_backend

_MAX_QUERY_CHARS = 4_000
_MAX_OUTPUT_CHARS = 8_000
_JOB_HISTORY_MAX = 8
_TERMINAL_JOB_STATES = {"done", "cancelled", "error"}
_JOB_ID_RE = re.compile(r"^sync-[0-9a-f]{12}$")
_JOBS_DIR = config.DATA_DIR / "sync_jobs"
_START_LOCK = config.DATA_DIR / "sync_start.lock"
_WORKER = Path(__file__).resolve().parent / "sync_job_worker.py"


def _cap(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    marker = f"\n...[truncated at {_MAX_OUTPUT_CHARS} chars]"
    return text[: max(0, _MAX_OUTPUT_CHARS - len(marker))] + marker


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    text = query.strip()
    if not text:
        raise ValueError("query must not be empty")
    if len(text) > _MAX_QUERY_CHARS:
        raise ValueError(f"query must be <= {_MAX_QUERY_CHARS} characters")
    return text


@contextmanager
def _kb_lock(*, exclusive: bool, cancel_requested: Callable[[], bool] | None = None):
    """Serialize syncs while permitting concurrent read searches.

    Background sync workers use non-blocking polling so cancel_sync() can stop
    a job even while it is queued behind another process holding the DB lock.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = config.DATA_DIR / "rag_mcp.lock"
    with lock_path.open("a+") as handle:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if cancel_requested is None:
            fcntl.flock(handle, mode)
        else:
            while True:
                if cancel_requested():
                    raise kb_sync.SyncCancelled("sync cancelled while waiting for lock")
                try:
                    fcntl.flock(handle, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def search(query: object) -> str:
    validated = _validate_query(query)
    with _kb_lock(exclusive=False):
        result = _rag_backend.run({"query": validated})
    if not isinstance(result, str):
        raise RuntimeError("rag backend returned non-text output")
    return _cap(result)


def _sync_pipeline_locked(
    *,
    on_progress: Callable | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    cancel_file: Path | None = None,
) -> dict[str, int | float]:
    """Run one lexical+semantic sync while the caller owns the exclusive lock."""
    semantic_expected = bool(config.SEMANTIC_ENABLED)

    def emit(event: str, path: Path | None, snapshot: dict) -> None:
        if on_progress is None or event == "done":
            return
        view = dict(snapshot)
        if semantic_expected:
            lexical_percent = float(view.get("progress_percent", 0) or 0)
            view["progress_percent"] = round(min(90.0, lexical_percent * 0.90), 1)
        on_progress(event, path, view)

    stats = kb_sync.sync_kb(
        verbose=False,
        on_progress=emit,
        cancel_requested=cancel_requested,
    )
    if cancel_requested is not None and cancel_requested():
        raise kb_sync.SyncCancelled("sync cancelled before semantic phase")

    if semantic_expected:
        if on_progress is not None:
            semantic_view = dict(stats)
            semantic_view["progress_percent"] = 90.0
            semantic_view["estimated_remaining_seconds"] = 0.0
            semantic_view["semantic_processed_parents"] = 0
            semantic_view["semantic_total_parents"] = int(stats.get("chunks", 0) or 0)
            on_progress("semantic", None, semantic_view)

        def semantic_progress(payload: dict) -> None:
            if on_progress is None:
                return
            semantic_percent = float(payload.get("progress_percent", 0) or 0)
            view = dict(stats)
            view["progress_percent"] = round(
                90.0 + min(9.0, semantic_percent * 0.09), 1
            )
            view["estimated_remaining_seconds"] = float(
                payload.get("estimated_remaining_seconds", 0) or 0
            )
            view["semantic_processed_parents"] = int(
                payload.get("processed_parents", 0) or 0
            )
            view["semantic_total_parents"] = int(
                payload.get("total_parents", 0) or 0
            )
            current = str(payload.get("current_file") or "")
            on_progress("semantic", Path(current) if current else None, view)

        semantic = semantic_client.refresh(
            config.FILE_INDEX_DB,
            cancel_file=cancel_file,
            on_progress=semantic_progress,
        )
        if semantic and semantic.get("status") == "cancelled":
            raise kb_sync.SyncCancelled("semantic sync cancelled")
    else:
        semantic = None

    if semantic and semantic.get("status") == "ok":
        stats["semantic_vectors"] = int(semantic.get("vectors", 0) or 0)
        stats["semantic_embedded"] = int(semantic.get("embedded", 0) or 0)
        stats["semantic_reused"] = int(semantic.get("reused", 0) or 0)
        stats["semantic_removed"] = int(semantic.get("removed", 0) or 0)
        stats["semantic_generation"] = int(semantic.get("generation", 0) or 0)
        stats["semantic_errors"] = 0
    else:
        stats["semantic_vectors"] = 0
        stats["semantic_embedded"] = 0
        stats["semantic_reused"] = 0
        stats["semantic_removed"] = 0
        stats["semantic_generation"] = 0
        stats["semantic_errors"] = 1 if semantic_expected else 0

    if cancel_requested is not None and cancel_requested():
        raise kb_sync.SyncCancelled("sync cancelled before completion")
    stats["progress_percent"] = 100.0
    stats["estimated_remaining_seconds"] = 0.0
    if on_progress is not None:
        on_progress("done", None, dict(stats))
    return stats


def sync_kb_blocking(
    *,
    on_progress: Callable | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, int | float]:
    """Host-facing synchronous path retained for local UI/CLI integrations."""
    with _kb_lock(exclusive=True, cancel_requested=cancel_requested):
        return _sync_pipeline_locked(
            on_progress=on_progress,
            cancel_requested=cancel_requested,
        )


# Public Python compatibility. The MCP tool itself uses start_sync().
sync_kb = sync_kb_blocking


def format_sync_result(stats: dict[str, int | float]) -> str:
    errors = int(stats.get("sync_errors", 0) or 0)
    semantic_errors = int(stats.get("semantic_errors", 0) or 0)
    status = "degraded" if errors or semantic_errors else "ok"
    return _cap(
        "\n".join([
            f"status={status}",
            f"source_scanned={int(stats.get('raw_scanned', 0) or 0)}",
            f"indexed={int(stats.get('indexed', 0) or 0)}",
            f"unchanged={int(stats.get('unchanged', 0) or 0)}",
            f"ocr_pdf_files={int(stats.get('ocr_pdf_files', 0) or 0)}",
            f"removed={int(stats.get('removed', 0) or 0)}",
            f"documents={int(stats.get('documents', 0) or 0)}",
            f"chunks={int(stats.get('chunks', 0) or 0)}",
            f"inserted={int(stats.get('inserted', 0) or 0)}",
            f"updated={int(stats.get('updated', 0) or 0)}",
            f"sync_errors={errors}",
            f"semantic_vectors={int(stats.get('semantic_vectors', 0) or 0)}",
            f"semantic_embedded={int(stats.get('semantic_embedded', 0) or 0)}",
            f"semantic_reused={int(stats.get('semantic_reused', 0) or 0)}",
            f"semantic_generation={int(stats.get('semantic_generation', 0) or 0)}",
            f"semantic_errors={semantic_errors}",
            f"seconds={float(stats.get('seconds', 0) or 0):.2f}",
        ])
    )


def _jobs_dir() -> Path:
    # Compute from config dynamically so isolated tests/host adapters may bind
    # a different DATA_DIR after this module was imported.
    return Path(config.DATA_DIR) / "sync_jobs"


def _start_lock_path() -> Path:
    return Path(config.DATA_DIR) / "sync_start.lock"


def _job_paths(job_id: str) -> tuple[Path, Path]:
    root = _jobs_dir()
    return root / f"{job_id}.json", root / f"{job_id}.cancel"


def _valid_job_id(job_id: object) -> str | None:
    if not isinstance(job_id, str):
        return None
    value = job_id.strip()
    return value if _JOB_ID_RE.fullmatch(value) else None


def _read_job(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_job(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _process_alive(pid: object) -> bool:
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    # kill(pid, 0) also succeeds for zombies. Reap our own exited workers
    # before treating the persisted PID as an active job; fresh stdio readers
    # are not their parent and simply continue to the normal existence check.
    try:
        exited, _status = os.waitpid(value, os.WNOHANG)
        if exited == value:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(value, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _effective_job_state(state: dict[str, object], cancel_file: Path | None = None) -> dict[str, object]:
    view = dict(state)
    status = str(view.get("status") or "unknown")
    if status not in _TERMINAL_JOB_STATES:
        cancel_requested = bool(cancel_file is not None and cancel_file.exists())
        view["cancel_requested"] = cancel_requested
        if cancel_requested:
            view["status"] = "cancelling"
        pid = view.get("pid")
        age = max(0.0, time.time() - float(view.get("updated_at") or 0))
        if not _process_alive(pid) and not (int(pid or 0) == 0 and age < 5.0):
            view["status"] = "error"
            view["phase"] = "error"
            view["error"] = "worker_not_running"
    return view


def _describe_activity(state: dict[str, object]) -> str:
    """One human-readable line: which stage, how far, and what it is doing now."""
    phase = str(state.get("phase") or "")
    done = int(state.get("completed_files") or 0)
    total = int(state.get("total_files") or 0)
    parents = int(state.get("semantic_processed_parents") or 0)
    parents_total = int(state.get("semantic_total_parents") or 0)
    eta = float(state.get("estimated_remaining_seconds") or 0)
    current = Path(str(state.get("current_file") or "")).name
    eta_text = f" · ETA {eta:.0f}s" if eta >= 1 else ""
    if phase in {"", "queued", "starting"}:
        text = "รอเริ่มงาน"
    elif phase in {"prepare", "prepared"}:
        text = "ช่วง 1/2 lexical: สแกนและเตรียมรายการไฟล์ต้นทาง"
    elif phase in {"file", "result"}:
        text = f"ช่วง 1/2 lexical: อ่าน/OCR/ทำดัชนีไฟล์ {done}/{total}{eta_text}"
    elif phase == "cleanup":
        text = "ช่วง 1/2 lexical: ล้างรายการไฟล์ที่ถูกลบออกจากดัชนี"
    elif phase == "semantic" and (parents_total <= 0 or parents <= 0):
        text = "ช่วง 2/2 semantic: โหลดโมเดลและสแกน chunks ที่ต้อง embed"
    elif phase == "semantic":
        text = f"ช่วง 2/2 semantic: embed {parents}/{parents_total} parents{eta_text}"
    elif phase == "done":
        text = "เสร็จสิ้น"
    else:
        text = phase
    if current and phase != "done":
        text += f" · {current}"
    return text


def _format_job(state: dict[str, object], *, prefix: str | None = None) -> str:
    status = prefix or str(state.get("status") or "unknown")
    lines = [
        f"status={status}",
        f"job_id={state.get('job_id') or ''}",
        f"phase={state.get('phase') or ''}",
        f"activity={_describe_activity(state)}",
        f"progress_percent={float(state.get('progress_percent') or 0):.1f}",
        f"completed_files={int(state.get('completed_files') or 0)}",
        f"total_files={int(state.get('total_files') or 0)}",
        f"estimated_remaining_seconds={float(state.get('estimated_remaining_seconds') or 0):.1f}",
        f"semantic_processed_parents={int(state.get('semantic_processed_parents') or 0)}",
        f"semantic_total_parents={int(state.get('semantic_total_parents') or 0)}",
        f"current_file={state.get('current_file') or ''}",
        f"cancel_requested={'true' if state.get('cancel_requested') else 'false'}",
    ]
    if status in {"started", "already_running", "queued", "running", "cancelling"}:
        job_id = state.get("job_id") or ""
        lines.append(f"next=rag_status(view=job, job_id={job_id})")
        lines.append(f"cancel=rag_manage(action=cancel, job_id={job_id})")
    if state.get("summary"):
        lines += ["result:", str(state["summary"])]
    if state.get("error"):
        lines.append(f"error={state['error']}")
    return _cap("\n".join(lines))


def _new_job_state(job_id: str) -> dict[str, object]:
    now = time.time()
    return {
        "job_id": job_id,
        "pid": 0,
        "status": "queued",
        "phase": "queued",
        "current_file": "",
        "progress_percent": 0.0,
        "completed_files": 0,
        "total_files": 0,
        "estimated_remaining_seconds": 0.0,
        "semantic_processed_parents": 0,
        "semantic_total_parents": 0,
        "cancel_requested": False,
        "summary": "",
        "error": "",
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
    }


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "RAGDOC_KNOWLEDGE_DIR": str(config.KNOWLEDGE_DIR),
        "RAGDOC_DATA_DIR": str(config.DATA_DIR),
        "RAGDOC_DB": str(config.FILE_INDEX_DB),
        "RAGDOC_SEMANTIC": "1" if config.SEMANTIC_ENABLED else "0",
        "RAGDOC_MAX_RESULTS": str(config.RAG_MAX_RESULTS),
        "RAGDOC_SNIPPET_CHARS": str(config.RAG_SNIPPET_CHARS),
        "RAGDOC_IMAGE_MAX_BYTES": str(config.READ_IMAGE_MAX_BYTES),
        "RAGDOC_CHUNK_CHARS": str(config.KB_CHUNK_TARGET_CHARS),
        "RAGDOC_CHUNK_OVERLAP_CHARS": str(config.KB_CHUNK_OVERLAP_CHARS),
        "RAGDOC_SEMANTIC_CHILD_TOKENS": str(config.SEMANTIC_CHILD_TOKENS),
        "RAGDOC_SEMANTIC_CHILD_OVERLAP_TOKENS": str(config.SEMANTIC_CHILD_OVERLAP_TOKENS),
        "RAGDOC_SEMANTIC_EMBED_BATCH": str(config.SEMANTIC_EMBED_BATCH),
        "RAGDOC_SEMANTIC_SCAN_BATCH": str(config.SEMANTIC_SCAN_BATCH),
        "RAGDOC_SEMANTIC_CANDIDATES": str(config.SEMANTIC_CANDIDATES),
        "RAGDOC_SEMANTIC_FULL_SCAN_LIMIT": str(config.SEMANTIC_FULL_SCAN_LIMIT),
        "RAGDOC_SEMANTIC_ANN_CANDIDATES": str(config.SEMANTIC_ANN_CANDIDATES),
        "RAGDOC_SEMANTIC_TIMEOUT": str(config.SEMANTIC_TIMEOUT_SECONDS),
        "RAGDOC_SEMANTIC_SYNC_TIMEOUT": str(config.SEMANTIC_SYNC_TIMEOUT_SECONDS),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _job_files_newest_first() -> list[Path]:
    root = _jobs_dir()
    if not root.is_dir():
        return []
    try:
        files = [path for path in root.glob("sync-*.json") if path.is_file()]
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _active_job() -> tuple[dict[str, object], Path] | None:
    now = time.time()
    for state_path in _job_files_newest_first():
        state = _read_job(state_path)
        if not state or state.get("status") in _TERMINAL_JOB_STATES:
            continue
        job_id = _valid_job_id(state.get("job_id"))
        if not job_id:
            continue
        _status_path, cancel_file = _job_paths(job_id)
        effective = _effective_job_state(state, cancel_file)
        if effective.get("status") != "error":
            return effective, state_path
        # A just-created pid=0 state gets a short grace period before it is
        # considered abandoned. This closes the spawn/status race safely.
        if int(state.get("pid") or 0) == 0 and now - float(state.get("updated_at") or 0) < 5.0:
            return state, state_path
    return None


def active_sync_snapshot() -> dict[str, object] | None:
    """Return the current persistent sync job for local UI integrations."""
    active = _active_job()
    return dict(active[0]) if active is not None else None


def _prune_job_history() -> None:
    files = _job_files_newest_first()
    kept = 0
    for state_path in files:
        state = _read_job(state_path)
        if not state:
            continue
        kept += 1
        if kept <= _JOB_HISTORY_MAX:
            continue
        if state.get("status") not in _TERMINAL_JOB_STATES:
            continue
        job_id = _valid_job_id(state.get("job_id"))
        try:
            state_path.unlink(missing_ok=True)
            if job_id:
                _job_paths(job_id)[1].unlink(missing_ok=True)
        except OSError:
            pass


def start_sync() -> str:
    """Spawn a persistent background sync worker and return its job id."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _jobs_dir().mkdir(parents=True, exist_ok=True)
    with _start_lock_path().open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            active = _active_job()
            if active is not None:
                return _format_job(active[0], prefix="already_running")
            _prune_job_history()
            job_id = f"sync-{uuid.uuid4().hex[:12]}"
            status_file, cancel_file = _job_paths(job_id)
            try:
                cancel_file.unlink(missing_ok=True)
            except OSError:
                pass
            state = _new_job_state(job_id)
            _write_job(status_file, state)
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(_WORKER),
                        "--job-id", job_id,
                        "--status-file", str(status_file),
                        "--cancel-file", str(cancel_file),
                    ],
                    cwd=str(_WORKER.parent),
                    env=_worker_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
                # Publish the actual child identity before releasing the
                # startup lock. Slow imports must not expire the pid=0 grace.
                state["pid"] = process.pid
                state["updated_at"] = time.time()
                _write_job(status_file, state)
            except OSError:
                state.update({
                    "status": "error",
                    "phase": "error",
                    "error": "worker_start_failed",
                    "finished_at": time.time(),
                    "updated_at": time.time(),
                })
                _write_job(status_file, state)
                return _format_job(state)
            return _format_job(state, prefix="started")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sync_status(job_id: object) -> str:
    """Read the latest persistent status for one sync job."""
    clean_id = _valid_job_id(job_id)
    if not clean_id:
        return "status=not_found"
    status_file, cancel_file = _job_paths(clean_id)
    state = _read_job(status_file)
    if state is None:
        return "status=not_found"
    return _format_job(_effective_job_state(state, cancel_file))


def cancel_sync(job_id: object) -> str:
    """Request cooperative cancellation through a private marker file."""
    clean_id = _valid_job_id(job_id)
    if not clean_id:
        return "status=not_found"
    status_file, cancel_file = _job_paths(clean_id)
    state = _read_job(status_file)
    if state is None:
        return "status=not_found"
    effective = _effective_job_state(state, cancel_file)
    if effective.get("status") in _TERMINAL_JOB_STATES:
        return _format_job(effective)
    try:
        cancel_file.parent.mkdir(parents=True, exist_ok=True)
        cancel_file.write_text("cancel\n", encoding="utf-8")
    except OSError:
        return _format_job(effective, prefix="cancel_signal_failed")
    effective["cancel_requested"] = True
    effective["status"] = "cancelling"
    return _format_job(effective, prefix="cancelling")


def run_persistent_job(job_id: str, status_file: Path, cancel_file: Path) -> int:
    """Worker-process entry point. Returns a conventional process exit code."""
    # Parent publishes the PID under this same lock; wait before reading or
    # writing progress so parent cannot overwrite a fast worker's terminal state.
    with _start_lock_path().open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            state = _read_job(status_file) or _new_job_state(job_id)
            state["pid"] = os.getpid()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

    def cancelled() -> bool:
        return cancel_file.exists()

    def update(**changes: object) -> None:
        state.update(changes)
        state["cancel_requested"] = cancelled()
        state["updated_at"] = time.time()
        _write_job(status_file, state)

    def on_progress(event: str, path: Path | None, stats: dict) -> None:
        update(
            status="cancelling" if cancelled() else "running",
            phase=event,
            current_file=str(path) if path is not None else "",
            progress_percent=float(stats.get("progress_percent", 0) or 0),
            completed_files=int(stats.get("completed_files", 0) or 0),
            total_files=int(stats.get("total_files", 0) or 0),
            ocr_pdf_files=int(stats.get("ocr_pdf_files", 0) or 0),
            estimated_remaining_seconds=float(
                stats.get("estimated_remaining_seconds", 0) or 0
            ),
            semantic_processed_parents=int(
                stats.get("semantic_processed_parents", 0) or 0
            ),
            semantic_total_parents=int(
                stats.get("semantic_total_parents", 0) or 0
            ),
        )

    update(status="queued", phase="waiting_lock")
    try:
        with _kb_lock(exclusive=True, cancel_requested=cancelled):
            if cancelled():
                raise kb_sync.SyncCancelled("sync cancelled before start")
            update(status="running", phase="prepare")
            stats = _sync_pipeline_locked(
                on_progress=on_progress,
                cancel_requested=cancelled,
                cancel_file=cancel_file,
            )
        update(
            status="done",
            phase="done",
            current_file="",
            progress_percent=100.0,
            estimated_remaining_seconds=0.0,
            summary=format_sync_result(stats),
            error="",
            finished_at=time.time(),
        )
        return 0
    except kb_sync.SyncCancelled:
        update(
            status="cancelled",
            phase="cancelled",
            current_file="",
            estimated_remaining_seconds=0.0,
            summary="",
            error="",
            finished_at=time.time(),
        )
        return 0
    except Exception as exc:
        update(
            status="error",
            phase="error",
            current_file="",
            estimated_remaining_seconds=0.0,
            summary="",
            error=type(exc).__name__,
            finished_at=time.time(),
        )
        return 1
    finally:
        try:
            cancel_file.unlink(missing_ok=True)
        except OSError:
            pass


def health() -> str:
    """Return a small deterministic health snapshot without reading document bodies."""
    with _kb_lock(exclusive=False):
        source_files = 0
        try:
            source_files = sum(1 for _source_key, _path in kb_sync._iter_source_files())
        except Exception:
            return "status=degraded\nreason=source_scan_failed"

        if not config.FILE_INDEX_DB.exists():
            return f"status=empty\nsource_files={source_files}\ndocuments=0\nchunks=0"

        db_path = config.FILE_INDEX_DB
        journal = Path(str(db_path) + "-journal")
        wal = Path(str(db_path) + "-wal")
        try:
            recovery_pending = (
                (journal.exists() and journal.stat().st_size > 512)
                or (wal.exists() and wal.stat().st_size > 32)
            )
        except OSError:
            recovery_pending = True
        if recovery_pending:
            return (
                f"status=degraded\nsource_files={source_files}\ndocuments=unknown\nchunks=unknown\n"
                "reason=sqlite_recovery_pending"
            )

        try:
            uri = f"file:{db_path}?mode=ro&immutable=1"
            with sqlite3.connect(uri, uri=True) as conn:
                names = {
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                        "AND name IN ('kb_documents','kb_chunks_fts')"
                    )
                }
                if names != {"kb_documents", "kb_chunks_fts"}:
                    return (
                        f"status=needs_sync\nsource_files={source_files}\ndocuments=0\nchunks=0\n"
                        "reason=kb_schema_missing"
                    )
                columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks_fts)")}
                if not {"content", "search_text"}.issubset(columns):
                    return (
                        f"status=needs_sync\nsource_files={source_files}\ndocuments=unknown\nchunks=unknown\n"
                        "reason=kb_schema_old"
                    )
                documents = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
                chunks = conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0]
                semantic_exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kb_semantic_chunks'"
                ).fetchone() is not None
                semantic_vectors = (
                    conn.execute("SELECT COUNT(*) FROM kb_semantic_chunks").fetchone()[0]
                    if semantic_exists else 0
                )
        except sqlite3.Error:
            return (
                f"status=degraded\nsource_files={source_files}\ndocuments=unknown\nchunks=unknown\n"
                "reason=sqlite_error"
            )

    semantic = semantic_client.model_status()
    semantic_index = semantic_client.index_status(config.FILE_INDEX_DB)
    if not semantic.get("enabled"):
        semantic_state = "disabled"
    elif not semantic.get("ready"):
        semantic_state = "unavailable"
    elif not semantic_index.get("ready"):
        semantic_state = "needs_sync"
    else:
        semantic_state = "ready"
    return (
        f"status=healthy\nsource_files={source_files}\ndocuments={documents}\nchunks={chunks}\n"
        f"semantic={semantic_state}\n"
        f"semantic_vectors={int(semantic_index.get('vectors', semantic_vectors) or 0)}\n"
        f"semantic_generation={int(semantic_index.get('generation', 0) or 0)}"
    )
