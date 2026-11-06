"""Short-lived ONNX INT8 semantic worker for MCP-RagDoc.

Modes:
- sync: stream bounded semantic child windows from existing FTS parents,
  persist normalized float16 vectors in SQLite, and atomically publish one
  complete semantic generation.
- query: require a ready matching generation, embed one query, use bounded
  exact cosine for small indexes or HNSW child candidates plus exact parent
  reranking for large indexes, then return top parent chunks.

The process exits after each command so ONNX/SentencePiece/NumPy RAM is never
resident in the long-lived Lite process.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

import ann_index
import config
from helpers.thai_word_break import word_boundary_offsets

def _model_tag(model_root: Path) -> str:
    """Fingerprint vectors to the actual Hugging Face snapshot revision."""
    revision = model_root.resolve().name
    return (
        f"{config.SEMANTIC_MODEL_REPO}:{config.SEMANTIC_POLICY_ID}:"
        f"snapshot={revision}"
    )


def _load_tokenizer(model_root: Path):
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(model_root / config.SEMANTIC_SENTENCEPIECE))


def _load_runtime(model_root: Path):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = config.SEMANTIC_THREADS
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model_root / config.SEMANTIC_MODEL_ONNX),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    return np, session


def _mapped_ids(sp, raw_ids: list[int]) -> list[int]:
    unk = sp.unk_id()
    return [0] + [(3 if token == unk else token + 1) for token in raw_ids] + [2]


def _safe_token_boundaries(text: str, pieces) -> list[int]:
    source_boundaries = set(word_boundary_offsets(text))
    return [
        token_index
        for token_index, piece in enumerate(pieces, 1)
        if piece.end in source_boundaries
    ]


def _window_ranges(total_tokens: int, safe_boundaries: list[int], width: int, overlap: int):
    """Yield bounded token windows, preferring whole-word starts/ends.

    A hard split happens only when one source word itself spans more than
    ``width`` SentencePiece tokens, where the model token ceiling takes
    precedence over the no-split preference.
    """
    if total_tokens <= 0:
        return
    safe = sorted({0, total_tokens, *safe_boundaries})
    start = 0
    while start < total_tokens:
        ceiling = min(total_tokens, start + width)
        if ceiling == total_tokens:
            end = total_tokens
        else:
            index = bisect_right(safe, ceiling) - 1
            end = safe[index] if index >= 0 and safe[index] > start else ceiling
        yield start, end
        if end >= total_tokens:
            break

        desired = max(start + 1, end - overlap)
        left = bisect_left(safe, desired)
        next_start = safe[left] if left < len(safe) and safe[left] < end else None
        if next_start is None:
            next_start = end if end in safe else desired
        if next_start <= start:
            next_start = end
        start = next_start


def _windows(sp, text: str, model_tag: str):
    proto = sp.encode(text, out_type="immutable_proto")
    pieces = list(proto.pieces)
    raw = [piece.id for piece in pieces]
    if not raw:
        return
    safe_boundaries = _safe_token_boundaries(text, pieces)
    for child, (start, end) in enumerate(
        _window_ranges(
            len(raw),
            safe_boundaries,
            config.SEMANTIC_CHILD_TOKENS,
            config.SEMANTIC_CHILD_OVERLAP_TOKENS,
        )
    ):
        piece = raw[start:end]
        digest = hashlib.sha256(
            (model_tag + "\0" + ",".join(map(str, piece))).encode()
        ).hexdigest()
        yield child, piece, digest


def _query_raw(sp, text: str, limit: int = 126) -> list[int]:
    """Truncate a long query at the last whole-word boundary <= limit."""
    proto = sp.encode(text, out_type="immutable_proto")
    pieces = list(proto.pieces)
    if not pieces:
        return []
    if len(pieces) <= limit:
        return [piece.id for piece in pieces]
    safe = _safe_token_boundaries(text, pieces)
    end_index = bisect_right(safe, limit) - 1
    end = safe[end_index] if end_index >= 0 and safe[end_index] > 0 else limit
    return [piece.id for piece in pieces[:end]]


def _encode_raw_batches(np, session, batches: list[list[int]]):
    names = {item.name for item in session.get_inputs()}
    width = max(len(item) + 2 for item in batches)
    input_ids = np.full((len(batches), width), 1, dtype=np.int64)
    attention = np.zeros((len(batches), width), dtype=np.int64)
    token_type = np.zeros((len(batches), width), dtype=np.int64)
    for index, raw in enumerate(batches):
        ids = _mapped_ids(_SP, raw)
        input_ids[index, :len(ids)] = ids
        attention[index, :len(ids)] = 1
    feed = {"input_ids": input_ids, "attention_mask": attention, "token_type_ids": token_type}
    feed = {key: value for key, value in feed.items() if key in names}
    output = session.run(None, feed)[0]
    if output.ndim == 3:
        mask = attention.astype(np.float32)[..., None]
        pooled = (output * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    else:
        pooled = output
    pooled = pooled.astype(np.float32)
    pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    return pooled


def _decode_vector(np, blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != config.SEMANTIC_DIM * 2:
        return None
    vector = np.frombuffer(blob, dtype="<f2").astype(np.float32, copy=False)
    if vector.size != config.SEMANTIC_DIM or not bool(np.isfinite(vector).all()):
        return None
    if float(np.linalg.norm(vector)) <= 1e-12:
        return None
    return vector


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kb_semantic_chunks ("
        "path TEXT NOT NULL, parent_chunk_index INTEGER NOT NULL, child_index INTEGER NOT NULL, "
        "model_tag TEXT NOT NULL, content_hash TEXT NOT NULL, embedding BLOB NOT NULL, "
        "generation INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY(path, parent_chunk_index, child_index))"
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_semantic_chunks)")}
    if "generation" not in columns:
        conn.execute(
            "ALTER TABLE kb_semantic_chunks ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS kb_semantic_parent_idx "
        "ON kb_semantic_chunks(path, parent_chunk_index)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kb_semantic_state ("
        "id INTEGER PRIMARY KEY CHECK(id = 1), generation INTEGER NOT NULL DEFAULT 0, "
        "model_tag TEXT NOT NULL DEFAULT '', ready INTEGER NOT NULL DEFAULT 0, "
        "vector_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO kb_semantic_state(id,generation,model_tag,ready,vector_count) "
        "VALUES(1,0,'',0,0)"
    )
    conn.commit()


def _begin_generation(conn: sqlite3.Connection, model_tag: str) -> int:
    row = conn.execute("SELECT generation FROM kb_semantic_state WHERE id=1").fetchone()
    generation = int(row[0] if row else 0) + 1
    with conn:
        conn.execute(
            "INSERT INTO kb_semantic_state(id,generation,model_tag,ready,vector_count) "
            "VALUES(1,?,?,0,0) ON CONFLICT(id) DO UPDATE SET "
            "generation=excluded.generation,model_tag=excluded.model_tag,ready=0,vector_count=0",
            (generation, model_tag),
        )
    return generation


class SemanticCancelled(RuntimeError):
    """Raised at a bounded semantic checkpoint after cancellation is requested."""


def _cancel_requested(cancel_file: Path | None) -> bool:
    return bool(cancel_file is not None and cancel_file.exists())


def _check_cancel(cancel_file: Path | None) -> None:
    if _cancel_requested(cancel_file):
        raise SemanticCancelled("semantic sync cancelled")


def _flush_embeddings(
    conn, np, session, pending, generation: int, model_tag: str,
    cancel_file: Path | None = None,
) -> int:
    if not pending:
        return 0
    _check_cancel(cancel_file)
    vectors = _encode_raw_batches(np, session, [item[4] for item in pending])
    _check_cancel(cancel_file)
    with conn:
        for item, vector in zip(pending, vectors):
            path, parent_index, child_index, digest, _raw = item
            blob = np.asarray(vector, dtype="<f2").tobytes()
            conn.execute(
                "INSERT INTO kb_semantic_chunks("
                "path,parent_chunk_index,child_index,model_tag,content_hash,embedding,generation) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(path,parent_chunk_index,child_index) DO UPDATE SET "
                "model_tag=excluded.model_tag,content_hash=excluded.content_hash,"
                "embedding=excluded.embedding,generation=excluded.generation",
                (path, parent_index, child_index, model_tag, digest, blob, generation),
            )
    count = len(pending)
    pending.clear()
    return count


def sync(
    db_path: Path,
    model_root: Path,
    *,
    cancel_file: Path | None = None,
    on_progress=None,
) -> dict[str, object]:
    """Stream one complete semantic generation with bounded Python memory."""
    _check_cancel(cancel_file)
    sp = _load_tokenizer(model_root)
    global _SP
    _SP = sp
    model_tag = _model_tag(model_root)
    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn)
    _check_cancel(cancel_file)
    generation = _begin_generation(conn, model_tag)

    pending: list[tuple[str, int, int, str, list[int]]] = []
    embedded = 0
    reused = 0
    np = session = None
    total_parents = int(
        conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0]
    )
    processed_parents = 0
    # At most ~100 progress records regardless of corpus size.
    progress_step = max(1, (total_parents + 99) // 100) if total_parents else 1
    semantic_started = time.monotonic()
    cursor = conn.execute(
        "SELECT path, CAST(chunk_index AS INTEGER), content FROM kb_chunks_fts "
        "ORDER BY path, CAST(chunk_index AS INTEGER)"
    )
    try:
        while True:
            _check_cancel(cancel_file)
            parents = cursor.fetchmany(config.SEMANTIC_SCAN_BATCH)
            if not parents:
                break
            for path, parent_index, content in parents:
                _check_cancel(cancel_file)
                if not content:
                    processed_parents += 1
                    continue
                for child_index, raw_ids, digest in _windows(sp, content, model_tag):
                    _check_cancel(cancel_file)
                    row = conn.execute(
                        "SELECT model_tag,content_hash,embedding FROM kb_semantic_chunks "
                        "WHERE path=? AND parent_chunk_index=? AND child_index=?",
                        (path, int(parent_index), child_index),
                    ).fetchone()
                    reusable = (
                        row
                        and row[0] == model_tag
                        and row[1] == digest
                    )
                    if reusable:
                        if np is None:
                            import numpy as np
                        reusable = _decode_vector(np, row[2]) is not None
                    if reusable:
                        conn.execute(
                            "UPDATE kb_semantic_chunks SET generation=? "
                            "WHERE path=? AND parent_chunk_index=? AND child_index=?",
                            (generation, path, int(parent_index), child_index),
                        )
                        reused += 1
                    else:
                        pending.append((path, int(parent_index), child_index, digest, raw_ids))
                    if len(pending) >= config.SEMANTIC_EMBED_BATCH:
                        if session is None:
                            np, session = _load_runtime(model_root)
                        embedded += _flush_embeddings(
                            conn, np, session, pending, generation, model_tag, cancel_file
                        )
                processed_parents += 1
                if on_progress is not None and (
                    processed_parents == total_parents
                    or processed_parents % progress_step == 0
                ):
                    elapsed = max(0.0, time.monotonic() - semantic_started)
                    remaining = 0.0
                    if processed_parents > 0 and total_parents > processed_parents:
                        remaining = elapsed * (total_parents - processed_parents) / processed_parents
                    on_progress({
                        "processed_parents": processed_parents,
                        "total_parents": total_parents,
                        "progress_percent": (
                            round(processed_parents / total_parents * 100.0, 1)
                            if total_parents else 100.0
                        ),
                        "estimated_remaining_seconds": round(remaining, 1),
                        "current_file": str(path),
                        "embedded": embedded,
                        "reused": reused,
                    })
            conn.commit()

        _check_cancel(cancel_file)
        if pending:
            if session is None:
                np, session = _load_runtime(model_root)
            embedded += _flush_embeddings(
                conn, np, session, pending, generation, model_tag, cancel_file
            )

        _check_cancel(cancel_file)
        with conn:
            deleted = conn.execute(
                "DELETE FROM kb_semantic_chunks WHERE generation<>? OR model_tag<>?",
                (generation, model_tag),
            ).rowcount
            total = conn.execute(
                "SELECT COUNT(*) FROM kb_semantic_chunks WHERE generation=? AND model_tag=?",
                (generation, model_tag),
            ).fetchone()[0]
            conn.execute(
                "UPDATE kb_semantic_state SET generation=?,model_tag=?,ready=1,vector_count=? WHERE id=1",
                (generation, model_tag, int(total)),
            )
        ann = {
            "status": "not_required",
            "required": int(total) > config.SEMANTIC_FULL_SCAN_LIMIT,
            "fresh": False,
        }
        if ann["required"]:
            try:
                ann = {
                    **ann_index.build(db_path, cancel_file=cancel_file),
                    "required": True,
                }
                _check_cancel(cancel_file)
            except ann_index.AnnCancelled as exc:
                raise SemanticCancelled(str(exc)) from exc
            except SemanticCancelled:
                raise
            except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
                ann = {
                    "status": "unavailable",
                    "required": True,
                    "fresh": False,
                    "error": str(exc),
                }
        return {
            "status": "ok",
            "generation": generation,
            "model_tag": model_tag,
            "embedded": embedded,
            "reused": reused,
            "removed": max(0, int(deleted)),
            "vectors": int(total),
            "ann": ann,
        }
    except SemanticCancelled:
        # Cancellation during semantic work already has ready=0 from
        # _begin_generation(). Cancellation during the optional ANN build can
        # happen after semantic publication, so restore the same fail-closed
        # generation contract without deleting reusable rows.
        with conn:
            conn.execute(
                "UPDATE kb_semantic_state SET ready=0,vector_count=0 "
                "WHERE id=1 AND generation=? AND model_tag=?",
                (generation, model_tag),
            )
        return {
            "status": "cancelled",
            "generation": generation,
            "model_tag": model_tag,
            "embedded": embedded,
            "reused": reused,
            "vectors": 0,
        }
    finally:
        conn.close()


def _push_parent(heap, limit: int, path: str, parent_index: int, score: float) -> None:
    item = (score, path, parent_index)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _score_vector(np, query_vec, blob) -> float | None:
    vector = _decode_vector(np, blob)
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return float(np.dot(query_vec, vector) / norm)


def _exact_parent_hits(
    conn: sqlite3.Connection,
    np,
    query_vec,
    model_tag: str,
    generation: int,
    limit: int,
) -> tuple[list[dict[str, object]], int]:
    cursor = conn.execute(
        "SELECT path,parent_chunk_index,embedding FROM kb_semantic_chunks "
        "WHERE model_tag=? AND generation=? "
        "ORDER BY path,parent_chunk_index,child_index",
        (model_tag, generation),
    )
    heap: list[tuple[float, str, int]] = []
    current_key = None
    current_best = -2.0
    scanned_parents = 0
    while True:
        rows = cursor.fetchmany(config.SEMANTIC_SCAN_BATCH)
        if not rows:
            break
        for path, parent_index, blob in rows:
            key = (str(path), int(parent_index))
            if current_key is not None and key != current_key:
                _push_parent(heap, limit, current_key[0], current_key[1], current_best)
                scanned_parents += 1
                current_best = -2.0
            current_key = key
            score = _score_vector(np, query_vec, blob)
            if score is not None and score > current_best:
                current_best = score
    if current_key is not None:
        _push_parent(heap, limit, current_key[0], current_key[1], current_best)
        scanned_parents += 1

    hits = [
        {"path": path, "parent_chunk_index": parent_index, "score": round(score, 7)}
        for score, path, parent_index in sorted(heap, reverse=True)
        if score > -2.0
    ]
    return hits, scanned_parents


def _ann_parent_hits(
    conn: sqlite3.Connection,
    np,
    query_vec,
    db_path: Path,
    model_tag: str,
    generation: int,
    vector_count: int,
    limit: int,
) -> tuple[list[dict[str, object]], int, str]:
    ann_state = ann_index.status(db_path)
    if not ann_state.get("available") or not ann_state.get("fresh"):
        return [], 0, str(ann_state.get("status") or "unavailable")

    parent_limit = min(
        vector_count,
        max(
            limit,
            min(config.SEMANTIC_ANN_CANDIDATES, limit * 4),
        ),
    )
    try:
        candidates = ann_index.query(
            db_path,
            [float(value) for value in query_vec],
            parent_limit,
        )
    except (OSError, RuntimeError, sqlite3.Error, ValueError):
        return [], 0, "unavailable"
    if not candidates:
        return [], 0, "unavailable"

    parent_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in candidates:
        try:
            key = (str(item["path"]), int(item["parent_chunk_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        parent_keys.append(key)

    best: dict[tuple[str, int], float] = {}
    for offset in range(0, len(parent_keys), 150):
        batch = parent_keys[offset : offset + 150]
        clauses = " OR ".join(
            "(path=? AND parent_chunk_index=?)"
            for _ in batch
        )
        params: list[object] = [model_tag, generation]
        for path, parent_index in batch:
            params.extend((path, parent_index))
        rows = conn.execute(
            "SELECT path,parent_chunk_index,embedding FROM kb_semantic_chunks "
            "WHERE model_tag=? AND generation=? AND (" + clauses + ") "
            "ORDER BY path,parent_chunk_index,child_index",
            params,
        )
        for path, parent_index, blob in rows:
            key = (str(path), int(parent_index))
            score = _score_vector(np, query_vec, blob)
            if score is None:
                continue
            if score > best.get(key, -2.0):
                best[key] = score

    heap: list[tuple[float, str, int]] = []
    for (path, parent_index), score in best.items():
        _push_parent(heap, limit, path, parent_index, score)
    hits = [
        {"path": path, "parent_chunk_index": parent_index, "score": round(score, 7)}
        for score, path, parent_index in sorted(heap, reverse=True)
        if score > -2.0
    ]
    return hits, len(parent_keys), "ready"


def query(db_path: Path, model_root: Path, text: str, limit: int) -> dict[str, object]:
    model_tag = _model_tag(model_root)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('kb_semantic_chunks','kb_semantic_state')"
            )
        }
        if names != {"kb_semantic_chunks", "kb_semantic_state"}:
            return {"status": "unavailable", "reason": "semantic_schema_missing", "hits": []}
        state = conn.execute(
            "SELECT generation,model_tag,ready,vector_count FROM kb_semantic_state WHERE id=1"
        ).fetchone()
        if not state or int(state[2]) != 1 or state[1] != model_tag:
            return {"status": "unavailable", "reason": "semantic_generation_not_ready", "hits": []}
        generation = int(state[0])
        vector_count = int(state[3])
        if vector_count <= 0:
            return {
                "status": "ok",
                "generation": generation,
                "hits": [],
                "scanned_parents": 0,
                "retrieval_mode": "exact",
            }

        sp = _load_tokenizer(model_root)
        raw = _query_raw(sp, text, 126)
        if not raw:
            return {
                "status": "ok",
                "generation": generation,
                "hits": [],
                "scanned_parents": 0,
                "retrieval_mode": (
                    "exact"
                    if vector_count <= config.SEMANTIC_FULL_SCAN_LIMIT
                    else "ann"
                ),
            }
        np, session = _load_runtime(model_root)
        global _SP
        _SP = sp
        query_vec = _encode_raw_batches(np, session, [raw])[0]

        if vector_count <= config.SEMANTIC_FULL_SCAN_LIMIT:
            hits, scanned_parents = _exact_parent_hits(
                conn,
                np,
                query_vec,
                model_tag,
                generation,
                limit,
            )
            retrieval_mode = "exact"
            ann_state = "not_required"
        else:
            hits, scanned_parents, ann_state = _ann_parent_hits(
                conn,
                np,
                query_vec,
                db_path,
                model_tag,
                generation,
                vector_count,
                limit,
            )
            retrieval_mode = "ann"

        return {
            "status": "ok",
            "generation": generation,
            "model_tag": model_tag,
            "hits": hits,
            "scanned_parents": scanned_parents,
            "retrieval_mode": retrieval_mode,
            "ann_status": ann_state,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sync", "query"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--limit", type=int, default=config.SEMANTIC_CANDIDATES)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    if args.mode == "sync":
        def emit_progress(payload) -> None:
            print(
                json.dumps({"type": "progress", **payload}, ensure_ascii=False),
                flush=True,
            )

        result = sync(
            args.db,
            args.model_root,
            cancel_file=args.cancel_file,
            on_progress=emit_progress,
        )
        print(json.dumps({"type": "result", "result": result}, ensure_ascii=False))
        return
    else:
        if not args.query.strip():
            raise SystemExit("query is required")
        result = query(args.db, args.model_root, args.query, max(1, min(args.limit, 100)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
