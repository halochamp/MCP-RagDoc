#!/usr/bin/env python3
"""Optional HNSW sidecar for MCP-RagDoc semantic child vectors.

SQLite remains the source of truth. This rebuildable sidecar accelerates only
large ready semantic generations. Missing, stale, corrupt, or unavailable ANN
state returns no candidates so callers can keep their lexical/exact fallback.

Forked from ENDMEMEX/ann_index.py and adapted to RagDoc's
(path, parent_chunk_index, child_index) semantic rows and generation contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import config

EMBED_DIM = config.SEMANTIC_DIM
SIDECAR_SCHEMA = config.SEMANTIC_ANN_SIDECAR_SCHEMA


def sidecar_dir(db_path: Path) -> Path:
    """Keep one rebuildable ANN sidecar per machine and database identity."""
    return config.semantic_ann_sidecar_dir(db_path)


def paths(db_path: Path, directory: Path | None = None) -> tuple[Path, Path]:
    root = directory or sidecar_dir(db_path)
    return root / "vectors.hnsw", root / "metadata.json"


def state_path(db_path: Path, directory: Path | None = None) -> Path:
    root = directory or sidecar_dir(db_path)
    return root / "state.json"


class AnnCancelled(RuntimeError):
    """Raised when a cooperative ANN build cancellation is requested."""


def _cancel_requested(cancel_file: Path | None) -> bool:
    return bool(cancel_file is not None and cancel_file.exists())


def _check_cancel(cancel_file: Path | None) -> None:
    if _cancel_requested(cancel_file):
        raise AnnCancelled("ANN build cancelled")


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def optional_modules():
    try:
        import hnswlib  # type: ignore
        import numpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("optional ANN requires numpy and hnswlib") from exc
    return hnswlib, numpy


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return the semantic publication identity used for freshness checks."""
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('kb_semantic_chunks','kb_semantic_state')"
        )
    }
    if names != {"kb_semantic_chunks", "kb_semantic_state"}:
        raise RuntimeError("semantic schema is unavailable")
    row = conn.execute(
        "SELECT generation,model_tag,ready,vector_count FROM kb_semantic_state WHERE id=1"
    ).fetchone()
    if row is None:
        raise RuntimeError("semantic state is unavailable")
    return {
        "generation": int(row["generation"]),
        "model_tag": str(row["model_tag"]),
        "ready": int(row["ready"]),
        "vector_count": int(row["vector_count"]),
    }


def vector_rows(conn: sqlite3.Connection, state: dict[str, Any]):
    generation = int(state["generation"])
    model_tag = str(state["model_tag"])
    for row in conn.execute(
        "SELECT path,parent_chunk_index,child_index,embedding "
        "FROM kb_semantic_chunks "
        "WHERE generation=? AND model_tag=? "
        "ORDER BY path,parent_chunk_index,child_index",
        (generation, model_tag),
    ):
        yield {
            "path": str(row["path"]),
            "parent_chunk_index": int(row["parent_chunk_index"]),
            "child_index": int(row["child_index"]),
        }, row["embedding"]


def build(
    db_path: Path,
    directory: Path | None = None,
    *,
    cancel_file: Path | None = None,
) -> dict[str, Any]:
    hnswlib, numpy = optional_modules()
    _check_cancel(cancel_file)
    target_index, target_meta = paths(db_path, directory)
    target_state = state_path(db_path, directory)
    target_index.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_read_only(db_path)
    try:
        before = snapshot(conn)
        if before["ready"] != 1:
            raise RuntimeError("semantic generation is not ready")
        if before["vector_count"] <= 0:
            raise RuntimeError("no semantic vectors are available to index")

        mapping: list[dict[str, Any]] = []
        vectors: list[tuple[float, ...]] = []
        for row_index, (item, blob) in enumerate(vector_rows(conn, before), start=1):
            if row_index == 1 or row_index % config.SEMANTIC_SCAN_BATCH == 0:
                _check_cancel(cancel_file)
            if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != EMBED_DIM * 2:
                raise RuntimeError("semantic vector blob has an invalid size")
            vector = struct.unpack(f"<{EMBED_DIM}e", blob)
            array = numpy.asarray(vector, dtype=numpy.float32)
            if not bool(numpy.isfinite(array).all()) or float(numpy.linalg.norm(array)) <= 1e-12:
                raise RuntimeError("semantic vector blob is non-finite or zero")
            mapping.append(item)
            vectors.append(vector)

        _check_cancel(cancel_file)
        after = snapshot(conn)
        if before != after:
            raise RuntimeError("semantic snapshot changed during ANN build; retry")
    finally:
        conn.close()

    if len(mapping) != int(before["vector_count"]):
        raise RuntimeError(
            "semantic vector count does not match the published generation"
        )

    _check_cancel(cancel_file)
    matrix = numpy.asarray(vectors, dtype=numpy.float32)
    index = hnswlib.Index(space="cosine", dim=EMBED_DIM)
    index.init_index(max_elements=len(mapping), ef_construction=200, M=16)
    add_batch = max(256, min(4096, config.SEMANTIC_SCAN_BATCH * 4))
    for start in range(0, len(mapping), add_batch):
        _check_cancel(cancel_file)
        end = min(len(mapping), start + add_batch)
        index.add_items(
            matrix[start:end],
            numpy.arange(start, end),
            num_threads=1,
        )
    _check_cancel(cancel_file)
    index.set_ef(min(len(mapping), max(100, config.SEMANTIC_ANN_CANDIDATES)))

    parent_count = len(
        {
            (item["path"], item["parent_chunk_index"])
            for item in mapping
        }
    )
    with tempfile.TemporaryDirectory(dir=target_index.parent) as temp_dir:
        _check_cancel(cancel_file)
        temp_root = Path(temp_dir)
        temp_index = temp_root / "vectors.hnsw"
        temp_meta = temp_root / "metadata.json"
        temp_state = temp_root / "state.json"
        index.save_index(str(temp_index))
        metadata = {
            "schema_version": SIDECAR_SCHEMA,
            "dimension": EMBED_DIM,
            "database": str(db_path.resolve()),
            "snapshot": before,
            "count": len(mapping),
            "parent_count": parent_count,
            "mapping": mapping,
        }
        temp_meta.write_text(
            json.dumps(metadata, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        building_state = {
            "schema_version": SIDECAR_SCHEMA,
            "dimension": EMBED_DIM,
            "database": str(db_path.resolve()),
            "publishing": True,
        }
        temp_state.write_text(
            json.dumps(building_state, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _check_cancel(cancel_file)
        os.replace(temp_state, target_state)

        _check_cancel(cancel_file)
        os.replace(temp_index, target_index)
        _check_cancel(cancel_file)
        os.replace(temp_meta, target_meta)
        _check_cancel(cancel_file)

        published_state = {
            "schema_version": SIDECAR_SCHEMA,
            "dimension": EMBED_DIM,
            "database": str(db_path.resolve()),
            "publishing": False,
            "snapshot": before,
            "count": len(mapping),
            "parent_count": parent_count,
            "index_file": _file_identity(target_index),
            "metadata_file": _file_identity(target_meta),
        }
        temp_state.write_text(
            json.dumps(published_state, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _check_cancel(cancel_file)
        os.replace(temp_state, target_state)

    return {
        "status": "ready",
        "available": True,
        "fresh": True,
        "index": str(target_index),
        "count": len(mapping),
        "parent_count": parent_count,
        "snapshot": before,
    }


def status(
    db_path: Path,
    directory: Path | None = None,
) -> dict[str, Any]:
    index_path, meta_path = paths(db_path, directory)
    published_path = state_path(db_path, directory)
    dependency = True
    try:
        optional_modules()
    except RuntimeError:
        dependency = False

    if (
        not index_path.is_file()
        or not meta_path.is_file()
        or not published_path.is_file()
    ):
        return {
            "status": "missing",
            "available": dependency,
            "fresh": False,
            "index": str(index_path),
        }

    try:
        published = json.loads(published_path.read_text(encoding="utf-8"))
        conn = connect_read_only(db_path)
        try:
            current = snapshot(conn)
        finally:
            conn.close()

        compatible = (
            published.get("schema_version") == SIDECAR_SCHEMA
            and published.get("dimension") == EMBED_DIM
            and published.get("database") == str(db_path.resolve())
            and published.get("publishing") is False
            and published.get("index_file") == _file_identity(index_path)
            and published.get("metadata_file") == _file_identity(meta_path)
        )
        fresh = (
            compatible
            and current.get("ready") == 1
            and published.get("snapshot") == current
            and published.get("count") == current.get("vector_count")
        )
        return {
            "status": (
                "ready"
                if dependency and fresh
                else ("stale" if compatible else "incompatible")
            ),
            "available": dependency,
            "fresh": bool(fresh),
            "index": str(index_path),
            "count": int(published.get("count", 0) or 0),
            "parent_count": int(published.get("parent_count", 0) or 0),
            "built_snapshot": published.get("snapshot"),
            "current_snapshot": current,
        }
    except (OSError, json.JSONDecodeError, sqlite3.Error, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "invalid",
            "available": dependency,
            "fresh": False,
            "index": str(index_path),
            "error": str(exc),
        }

def query(
    db_path: Path,
    vector: list[float],
    limit: int,
    directory: Path | None = None,
) -> list[dict[str, Any]]:
    """Return up to limit unique parent candidates in ANN rank order."""
    state = status(db_path, directory)
    if not state.get("available") or not state.get("fresh"):
        return []
    if len(vector) != EMBED_DIM:
        raise ValueError("invalid ANN vector dimension")

    def stable_result(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        final_state = status(db_path, directory)
        if not final_state.get("available") or not final_state.get("fresh"):
            return []
        return result

    try:
        hnswlib, numpy = optional_modules()
        index_path, meta_path = paths(db_path, directory)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        mapping = metadata["mapping"]
        if not mapping:
            return []

        parent_total = int(metadata.get("parent_count", 0) or 0)
        if parent_total <= 0:
            parent_total = len(
                {
                    (item["path"], item["parent_chunk_index"])
                    for item in mapping
                }
            )
        target_parents = min(parent_total, max(1, int(limit)))
        if target_parents <= 0:
            return []

        index = hnswlib.Index(space="cosine", dim=EMBED_DIM)
        index.load_index(str(index_path), max_elements=len(mapping))
        query_vector = numpy.asarray([vector], dtype=numpy.float32)

        candidate_count = min(
            len(mapping),
            max(100, target_parents * 4),
        )
        while True:
            index.set_ef(min(len(mapping), max(candidate_count * 2, 100)))
            labels, _distances = index.knn_query(
                query_vector,
                k=candidate_count,
                num_threads=1,
            )
            result: list[dict[str, Any]] = []
            seen: set[tuple[str, int]] = set()
            for label in labels[0]:
                item = mapping[int(label)]
                key = (str(item["path"]), int(item["parent_chunk_index"]))
                if key in seen:
                    continue
                seen.add(key)
                result.append(item)
                if len(result) >= target_parents:
                    return stable_result(result)

            if candidate_count >= len(mapping):
                return stable_result(result)
            candidate_count = min(
                len(mapping),
                max(candidate_count + 1, candidate_count * 2),
            )
    except (
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return []

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("status")
    query_parser = sub.add_parser("query")
    query_parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    try:
        if args.command == "build":
            result = build(db_path)
        elif args.command == "status":
            result = status(db_path)
        else:
            vector = json.loads(sys.stdin.read())
            result = {
                "candidates": query(
                    db_path,
                    vector,
                    max(1, min(int(args.limit), 5000)),
                )
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(
            json.dumps(
                {"status": "unavailable", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
