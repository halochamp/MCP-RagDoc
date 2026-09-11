"""Lightweight subprocess client for MCP-RagDoc semantic retrieval.

This module deliberately imports only the standard library. The heavy ONNX,
NumPy and SentencePiece stack lives in semantic_worker.py and therefore never
becomes resident in the long-lived MCP server process.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import selectors
import sqlite3
import subprocess
import sys
import time

import config

_WORKER = Path(__file__).resolve().parent / "semantic_worker.py"


def _model_root() -> Path | None:
    override = os.getenv("RAGDOC_SEMANTIC_MODEL_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
        return root if root.is_dir() else None
    repo_dir = "models--" + config.SEMANTIC_MODEL_REPO.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    try:
        candidates = sorted(snapshots.iterdir(), key=_mtime, reverse=True)
    except OSError:
        return None
    for root in candidates:
        if (root / config.SEMANTIC_MODEL_ONNX).is_file() and (root / config.SEMANTIC_SENTENCEPIECE).is_file():
            return root
    return None


def expected_model_tag(root: Path) -> str:
    revision = root.resolve().name
    return (
        f"{config.SEMANTIC_MODEL_REPO}:{config.SEMANTIC_POLICY_ID}:"
        f"snapshot={revision}"
    )


def model_status() -> dict[str, object]:
    root = _model_root()
    dependencies = all(
        importlib.util.find_spec(name) is not None
        for name in ("onnxruntime", "numpy", "sentencepiece")
    )
    return {
        "enabled": bool(config.SEMANTIC_ENABLED),
        "ready": bool(root and _WORKER.is_file() and dependencies),
        "dependencies": dependencies,
        "model_root": str(root) if root else "",
        "model_tag": expected_model_tag(root) if root else "",
    }


def index_status(db_path: Path) -> dict[str, object]:
    """Read semantic generation metadata without importing any ML runtime."""
    if not db_path.exists():
        return {"ready": False, "reason": "db_missing", "vectors": 0}
    model = model_status()
    if not model.get("enabled"):
        return {"ready": False, "reason": "disabled", "vectors": 0}
    if not model.get("ready"):
        return {"ready": False, "reason": "model_unavailable", "vectors": 0}
    expected = str(model.get("model_tag") or "")
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('kb_semantic_chunks','kb_semantic_state')"
                )
            }
            if names != {"kb_semantic_chunks", "kb_semantic_state"}:
                return {"ready": False, "reason": "schema_missing", "vectors": 0}
            row = conn.execute(
                "SELECT generation,model_tag,ready,vector_count FROM kb_semantic_state WHERE id=1"
            ).fetchone()
    except sqlite3.Error:
        return {"ready": False, "reason": "sqlite_error", "vectors": 0}
    if not row:
        return {"ready": False, "reason": "state_missing", "vectors": 0}
    generation, model_tag, ready, vector_count = row
    matching = bool(expected and model_tag == expected)
    return {
        "ready": bool(int(ready) == 1 and matching),
        "reason": "ok" if int(ready) == 1 and matching else "generation_not_ready",
        "generation": int(generation),
        "model_tag": model_tag,
        "expected_model_tag": expected,
        "vectors": int(vector_count or 0),
    }


def _run(
    mode: str,
    *,
    db_path: Path,
    query: str = "",
    limit: int = 0,
    cancel_file: Path | None = None,
    on_progress=None,
) -> dict[str, object] | None:
    status = model_status()
    if not status.get("enabled") or not status.get("ready"):
        return None
    root = _model_root()
    if root is None or not _WORKER.is_file():
        return None
    args = [
        sys.executable, str(_WORKER), mode,
        "--db", str(db_path),
        "--model-root", str(root),
    ]
    if query:
        args += ["--query", query]
    if limit:
        args += ["--limit", str(limit)]
    if mode == "sync" and cancel_file is not None:
        args += ["--cancel-file", str(cancel_file)]
    timeout = (
        config.SEMANTIC_SYNC_TIMEOUT_SECONDS
        if mode == "sync"
        else config.SEMANTIC_TIMEOUT_SECONDS
    )
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if mode != "sync":
        try:
            completed = subprocess.run(
                args,
                cwd=str(_WORKER.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    # Sync emits bounded JSONL progress before its final result.  selectors
    # preserves the existing timeout even if model loading pauses stdout.
    try:
        process = subprocess.Popen(
            args,
            cwd=str(_WORKER.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=environment,
        )
    except OSError:
        return None
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    result: dict[str, object] | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                return None
            events = selector.select(timeout=min(0.2, remaining))
            for _key, _mask in events:
                line = process.stdout.readline()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") == "progress":
                    if on_progress is not None:
                        try:
                            on_progress(payload)
                        except Exception:
                            pass
                elif payload.get("type") == "result" and isinstance(payload.get("result"), dict):
                    result = payload["result"]
                else:
                    result = payload
            if process.poll() is not None:
                for line in process.stdout:
                    try:
                        payload = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "result" and isinstance(payload.get("result"), dict):
                        result = payload["result"]
                break
        return result if process.returncode == 0 else None
    finally:
        selector.close()


def refresh(
    db_path: Path,
    *,
    cancel_file: Path | None = None,
    on_progress=None,
) -> dict[str, object] | None:
    return _run(
        "sync",
        db_path=db_path,
        cancel_file=cancel_file,
        on_progress=on_progress,
    )


def search(db_path: Path, query: str, limit: int | None = None) -> list[dict[str, object]]:
    payload = _run(
        "query",
        db_path=db_path,
        query=query,
        limit=limit or config.SEMANTIC_CANDIDATES,
    )
    hits = payload.get("hits") if payload else None
    return [item for item in hits if isinstance(item, dict)] if isinstance(hits, list) else []
