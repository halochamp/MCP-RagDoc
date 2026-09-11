"""Model-free regression tests for MCP-RagDoc semantic fusion."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _row(path: str, name: str, location: str, content: str):
    return (path, name, location, content, 10, "Text")


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE kb_documents(path TEXT PRIMARY KEY, size_bytes INTEGER, kind TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
        "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, "
        "content UNINDEXED, search_text, tokenize='trigram')"
    )
    conn.execute(
        "CREATE TABLE kb_semantic_chunks("
        "path TEXT, parent_chunk_index INTEGER, child_index INTEGER, "
        "model_tag TEXT, content_hash TEXT, embedding BLOB, generation INTEGER DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE kb_semantic_state("
        "id INTEGER PRIMARY KEY, generation INTEGER, model_tag TEXT, ready INTEGER, vector_count INTEGER)"
    )
    return conn


def test_rrf_merge_rewards_result_seen_by_both_retrievers():
    import rag_search

    a = _row("/a", "a", "line 1", "alpha")
    b = _row("/b", "b", "line 1", "beta")
    c = _row("/c", "c", "line 1", "gamma")
    merged = rag_search._rrf_merge([a, b], [b, c])
    assert merged[0] == b


def test_semantic_parent_rows_resolve_worker_ids(monkeypatch):
    import rag_search

    conn = _db()
    conn.execute("INSERT INTO kb_documents VALUES('/nav', 10, 'Text')")
    conn.execute(
        "INSERT INTO kb_chunks_fts VALUES('/nav', 3, 'nav.txt', 'line 1', 'navigation meaning', 'navigation meaning')"
    )
    conn.execute("INSERT INTO kb_semantic_chunks VALUES('/nav', 3, 0, 'm', 'h', X'00', 1)")
    monkeypatch.setattr(rag_search, "_allowed_result_path", lambda _path: True)
    monkeypatch.setattr(
        rag_search.semantic_client,
        "index_status",
        lambda *_args, **_kwargs: {"ready": True, "vectors": 1},
    )
    monkeypatch.setattr(
        rag_search.semantic_client,
        "search",
        lambda *_args, **_kwargs: [{"path": "/nav", "parent_chunk_index": 3, "score": 0.9}],
    )
    rows = rag_search._semantic_parent_rows(conn, "คำถาม")
    assert len(rows) == 1
    assert rows[0][1] == "nav.txt"
    assert rows[0][3] == "navigation meaning"


def test_semantic_only_hit_is_returned_when_fts_has_zero_overlap(monkeypatch):
    import rag_search

    conn = _db()
    conn.execute("INSERT INTO kb_documents VALUES('/nav', 100, 'Text')")
    conn.execute(
        "INSERT INTO kb_chunks_fts VALUES('/nav', 0, 'navigation.txt', 'line 1', "
        "'The aircraft maintains heading altitude position and route.', "
        "'The aircraft maintains heading altitude position and route.')"
    )
    conn.execute("INSERT INTO kb_semantic_chunks VALUES('/nav', 0, 0, 'm', 'h', X'00', 1)")
    monkeypatch.setattr(rag_search, "_allowed_result_path", lambda _path: True)
    monkeypatch.setattr(
        rag_search.semantic_client,
        "index_status",
        lambda *_args, **_kwargs: {"ready": True, "vectors": 1},
    )
    monkeypatch.setattr(
        rag_search.semantic_client,
        "search",
        lambda *_args, **_kwargs: [{"path": "/nav", "parent_chunk_index": 0, "score": 0.8}],
    )
    result = rag_search._run_chunk_search(
        conn,
        query="เครื่องบินรู้ทิศทางอย่างไร",
        match_query='"ไม่มีคำนี้แน่นอน"',
        split_match_query="",
    )
    assert "navigation.txt" in result
    assert "aircraft" in result.casefold()


def test_semantic_windows_keep_word_boundaries_with_bounded_overlap():
    import semantic_worker

    safe = [0, 8, 16, 24, 32, 40]
    ranges = list(semantic_worker._window_ranges(40, safe, width=20, overlap=8))
    assert ranges == [(0, 16), (8, 24), (16, 32), (24, 40)]
    assert all(start in safe and end in safe for start, end in ranges)
    assert all(end - start <= 20 for start, end in ranges)
    assert all(ranges[i][1] - ranges[i + 1][0] <= 8 for i in range(len(ranges) - 1))


def test_word_boundary_offsets_preserve_thai_newmm_tokens(monkeypatch):
    from helpers import thai_word_break

    text = "ระบบป้องกันการทำงาน"
    tokens = ["ระบบ", "ป้องกัน", "การ", "ทำงาน"]
    monkeypatch.setattr(
        thai_word_break,
        "_word_tokenizer",
        lambda: (lambda _text, engine, keep_whitespace: tokens),
    )
    offsets = thai_word_break.word_boundary_offsets(text)
    expected = []
    cursor = 0
    for token in tokens:
        cursor += len(token)
        expected.append(cursor)
    assert offsets == expected


def test_long_semantic_query_truncates_at_last_safe_word_boundary(monkeypatch):
    import semantic_worker

    class _Piece:
        def __init__(self, index: int):
            self.id = index
            self.end = index + 1

    class _Proto:
        pieces = [_Piece(index) for index in range(140)]

    class _SP:
        def encode(self, _text, out_type=None):
            assert out_type == "immutable_proto"
            return _Proto()

    monkeypatch.setattr(
        semantic_worker,
        "word_boundary_offsets",
        lambda _text: list(range(10, 141, 10)),
    )
    raw = semantic_worker._query_raw(_SP(), "x" * 140, limit=126)
    assert len(raw) == 120
    assert raw[-1] == 119


def test_model_tag_contains_snapshot_revision(tmp_path):
    import semantic_worker

    root = tmp_path / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    tag = semantic_worker._model_tag(root)
    assert "snapshot=e8f8c211226b894fcb81acc59f3b34ba3efd5f42" in tag
    assert semantic_worker.config.SEMANTIC_POLICY_ID in tag


def test_ensure_schema_migrates_old_semantic_table(tmp_path):
    import semantic_worker

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE kb_semantic_chunks("
        "path TEXT, parent_chunk_index INTEGER, child_index INTEGER, "
        "model_tag TEXT, content_hash TEXT, embedding BLOB, "
        "PRIMARY KEY(path,parent_chunk_index,child_index))"
    )
    semantic_worker._ensure_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_semantic_chunks)")}
    state = conn.execute(
        "SELECT generation,model_tag,ready,vector_count FROM kb_semantic_state WHERE id=1"
    ).fetchone()
    conn.close()
    assert "generation" in columns
    assert state == (0, "", 0, 0)


def test_failed_build_leaves_generation_not_ready(tmp_path, monkeypatch):
    import pytest
    import semantic_worker

    class _Piece:
        id = 7
        end = 1

    class _Proto:
        pieces = [_Piece()]

    class _SP:
        def encode(self, _text, out_type=None):
            assert out_type == "immutable_proto"
            return _Proto()

    db = tmp_path / "failed.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
        "path UNINDEXED,chunk_index UNINDEXED,name,location UNINDEXED,"
        "content UNINDEXED,search_text,tokenize='trigram')"
    )
    conn.execute(
        "INSERT INTO kb_chunks_fts VALUES('/a',0,'a.txt','line 1','x','x')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(semantic_worker, "_load_tokenizer", lambda _root: _SP())
    monkeypatch.setattr(semantic_worker, "word_boundary_offsets", lambda _text: [1])
    monkeypatch.setattr(
        semantic_worker,
        "_load_runtime",
        lambda _root: (_ for _ in ()).throw(RuntimeError("synthetic embed failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic embed failure"):
        semantic_worker.sync(db, tmp_path / "snapshot123")

    conn = sqlite3.connect(db)
    state = conn.execute(
        "SELECT generation,ready,vector_count FROM kb_semantic_state WHERE id=1"
    ).fetchone()
    conn.close()
    assert state == (1, 0, 0)


def test_cancelled_semantic_build_keeps_generation_not_ready(tmp_path, monkeypatch):
    import numpy as np
    import semantic_worker

    class _Piece:
        id = 7
        end = 1

    class _Proto:
        pieces = [_Piece()]

    class _SP:
        def encode(self, _text, out_type=None):
            assert out_type == "immutable_proto"
            return _Proto()

        def unk_id(self):
            return -1

    db = tmp_path / "cancelled.sqlite3"
    cancel_file = tmp_path / "cancel.flag"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
        "path UNINDEXED,chunk_index UNINDEXED,name,location UNINDEXED,"
        "content UNINDEXED,search_text,tokenize='trigram')"
    )
    conn.execute("INSERT INTO kb_chunks_fts VALUES('/a',0,'a.txt','line 1','x','x')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(semantic_worker, "_load_tokenizer", lambda _root: _SP())
    monkeypatch.setattr(semantic_worker, "word_boundary_offsets", lambda _text: [1])
    monkeypatch.setattr(semantic_worker, "_load_runtime", lambda _root: (np, object()))

    def cancel_during_encode(_np, _session, batches):
        cancel_file.write_text("cancel\n", encoding="utf-8")
        return np.zeros((len(batches), semantic_worker.config.SEMANTIC_DIM), dtype=np.float32)

    monkeypatch.setattr(semantic_worker, "_encode_raw_batches", cancel_during_encode)
    result = semantic_worker.sync(
        db,
        tmp_path / "snapshot123",
        cancel_file=cancel_file,
    )
    assert result["status"] == "cancelled"

    conn = sqlite3.connect(db)
    state = conn.execute(
        "SELECT generation,ready,vector_count FROM kb_semantic_state WHERE id=1"
    ).fetchone()
    conn.close()
    assert state == (1, 0, 0)


def test_partial_generation_refuses_query_before_loading_model(tmp_path, monkeypatch):
    import semantic_worker

    db = tmp_path / "semantic.sqlite3"
    conn = sqlite3.connect(db)
    semantic_worker._ensure_schema(conn)
    model_root = tmp_path / "snapshot123"
    model_tag = semantic_worker._model_tag(model_root)
    conn.execute(
        "UPDATE kb_semantic_state SET generation=7,model_tag=?,ready=0,vector_count=1 WHERE id=1",
        (model_tag,),
    )
    conn.execute(
        "INSERT INTO kb_semantic_chunks(path,parent_chunk_index,child_index,model_tag,content_hash,embedding,generation) "
        "VALUES('/a',0,0,?,'h',zeroblob(?),7)",
        (model_tag, semantic_worker.config.SEMANTIC_DIM * 2),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        semantic_worker,
        "_load_tokenizer",
        lambda _root: (_ for _ in ()).throw(AssertionError("model must not load")),
    )
    result = semantic_worker.query(db, model_root, "anything", 8)
    assert result["status"] == "unavailable"
    assert result["reason"] == "semantic_generation_not_ready"
    assert result["hits"] == []


def test_snapshot_mismatch_refuses_query_before_loading_model(tmp_path, monkeypatch):
    import semantic_worker

    db = tmp_path / "semantic-mismatch.sqlite3"
    conn = sqlite3.connect(db)
    semantic_worker._ensure_schema(conn)
    conn.execute(
        "UPDATE kb_semantic_state SET generation=3,model_tag='old-snapshot',ready=1,vector_count=1 WHERE id=1"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        semantic_worker,
        "_load_tokenizer",
        lambda _root: (_ for _ in ()).throw(AssertionError("model must not load")),
    )
    result = semantic_worker.query(db, tmp_path / "new-snapshot", "anything", 8)
    assert result["status"] == "unavailable"
    assert result["reason"] == "semantic_generation_not_ready"
    assert result["hits"] == []


def test_semantic_vector_decoder_rejects_corrupt_and_nonfinite_blobs():
    import numpy as np
    import semantic_worker

    assert semantic_worker._decode_vector(np, b"\x00") is None
    bad = np.zeros(semantic_worker.config.SEMANTIC_DIM, dtype="<f2")
    bad[0] = np.nan
    assert semantic_worker._decode_vector(np, bad.tobytes()) is None
    good = np.zeros(semantic_worker.config.SEMANTIC_DIM, dtype="<f2")
    good[0] = 1.0
    decoded = semantic_worker._decode_vector(np, good.tobytes())
    assert decoded is not None
    assert decoded.shape == (semantic_worker.config.SEMANTIC_DIM,)


def test_not_ready_index_bypasses_semantic_worker(monkeypatch):
    import rag_search

    conn = _db()
    monkeypatch.setattr(
        rag_search.semantic_client,
        "index_status",
        lambda *_args, **_kwargs: {"ready": False, "vectors": 99},
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("semantic worker must not run for an unready generation")

    monkeypatch.setattr(rag_search.semantic_client, "search", should_not_run)
    assert rag_search._semantic_parent_rows(conn, "semantic query") == []


def test_exact_filename_hit_bypasses_semantic_worker(monkeypatch):
    import rag_search

    conn = _db()
    conn.execute("INSERT INTO kb_documents VALUES('/target', 100, 'Text')")
    conn.execute(
        "INSERT INTO kb_chunks_fts VALUES('/target', 0, 'targetfile', 'line 1', 'known content', 'known content')"
    )
    monkeypatch.setattr(rag_search, "_allowed_result_path", lambda _path: True)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("semantic worker must not run for an exact filename hit")

    monkeypatch.setattr(rag_search.semantic_client, "search", should_not_run)
    result = rag_search._run_chunk_search(
        conn,
        query="targetfile",
        match_query='"targetfile"',
        split_match_query="",
    )
    assert "targetfile" in result
