from __future__ import annotations

import sqlite3

import numpy as np
import pytest


def _blob(*values: float) -> bytes:
    import ann_index

    vector = np.zeros(ann_index.EMBED_DIM, dtype="<f2")
    vector[: len(values)] = values
    return vector.tobytes()


def _make_db(tmp_path, *, vector_count: int = 4):
    db = tmp_path / "ragdoc.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE kb_semantic_chunks("
        "path TEXT NOT NULL,parent_chunk_index INTEGER NOT NULL,"
        "child_index INTEGER NOT NULL,model_tag TEXT NOT NULL,"
        "content_hash TEXT NOT NULL,embedding BLOB NOT NULL,"
        "generation INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE kb_semantic_state("
        "id INTEGER PRIMARY KEY,generation INTEGER NOT NULL,"
        "model_tag TEXT NOT NULL,ready INTEGER NOT NULL,"
        "vector_count INTEGER NOT NULL)"
    )
    rows = [
        ("/a", 0, 0, "model", "a0", _blob(1.0, 0.0), 7),
        ("/a", 0, 1, "model", "a1", _blob(0.9, 0.1), 7),
        ("/b", 2, 0, "model", "b0", _blob(0.0, 1.0), 7),
        ("/c", 4, 0, "model", "c0", _blob(-1.0, 0.0), 7),
    ]
    conn.executemany(
        "INSERT INTO kb_semantic_chunks VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute(
        "INSERT INTO kb_semantic_state VALUES(1,7,'model',1,?)",
        (vector_count,),
    )
    conn.commit()
    conn.close()
    return db


def test_ann_build_status_and_query(tmp_path):
    import ann_index

    db = _make_db(tmp_path)
    ann_dir = tmp_path / "ann"
    built = ann_index.build(db, ann_dir)
    assert built["status"] == "ready"
    assert built["count"] == 4

    state = ann_index.status(db, ann_dir)
    assert state["available"] is True
    assert state["fresh"] is True
    assert state["count"] == 4

    query = np.zeros(ann_index.EMBED_DIM, dtype=np.float32)
    query[0] = 1.0
    candidates = ann_index.query(db, query.tolist(), 3, ann_dir)
    assert candidates
    assert candidates[0]["path"] == "/a"
    assert candidates[0]["parent_chunk_index"] == 0


def test_ann_generation_change_marks_sidecar_stale(tmp_path):
    import ann_index

    db = _make_db(tmp_path)
    ann_dir = tmp_path / "ann"
    ann_index.build(db, ann_dir)

    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE kb_semantic_state SET generation=8 WHERE id=1"
    )
    conn.commit()
    conn.close()

    state = ann_index.status(db, ann_dir)
    assert state["fresh"] is False
    assert state["status"] == "stale"
    assert ann_index.query(
        db,
        [1.0] + [0.0] * (ann_index.EMBED_DIM - 1),
        8,
        ann_dir,
    ) == []


def test_ann_build_rejects_published_count_mismatch(tmp_path):
    import ann_index

    db = _make_db(tmp_path, vector_count=5)
    with pytest.raises(RuntimeError, match="vector count"):
        ann_index.build(db, tmp_path / "ann")


def test_ann_build_rejects_nonfinite_vector(tmp_path):
    import ann_index

    db = _make_db(tmp_path)
    bad = np.zeros(ann_index.EMBED_DIM, dtype="<f2")
    bad[0] = np.nan
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE kb_semantic_chunks SET embedding=? "
        "WHERE path='/b' AND parent_chunk_index=2 AND child_index=0",
        (bad.tobytes(),),
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="non-finite or zero"):
        ann_index.build(db, tmp_path / "ann")


def test_ann_parent_rerank_scores_all_children_of_candidate_parent(
    tmp_path,
    monkeypatch,
):
    import semantic_worker

    db = _make_db(tmp_path)
    conn = sqlite3.connect(db)

    # ANN only needs to nominate a child from each parent. Exact rerank must
    # reload every child of those parents from SQLite before parent scoring.
    monkeypatch.setattr(
        semantic_worker.ann_index,
        "status",
        lambda _db: {"status": "ready", "available": True, "fresh": True},
    )
    monkeypatch.setattr(
        semantic_worker.ann_index,
        "query",
        lambda *_args, **_kwargs: [
            {"path": "/a", "parent_chunk_index": 0, "child_index": 1},
            {"path": "/b", "parent_chunk_index": 2, "child_index": 0},
        ],
    )

    query_vec = np.zeros(semantic_worker.config.SEMANTIC_DIM, dtype=np.float32)
    query_vec[0] = 1.0
    hits, scanned, state = semantic_worker._ann_parent_hits(
        conn,
        np,
        query_vec,
        db,
        "model",
        7,
        4,
        8,
    )
    conn.close()

    assert state == "ready"
    assert scanned == 2
    assert hits[0]["path"] == "/a"
    # /a's exact best child is [1, 0], even though ANN nominated child 1.
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-6)


def _semantic_sync_db(tmp_path):
    db = tmp_path / "sync.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
        "path UNINDEXED,chunk_index UNINDEXED,name,location UNINDEXED,"
        "content UNINDEXED,search_text,tokenize='trigram')"
    )
    conn.execute(
        "INSERT INTO kb_chunks_fts VALUES('/a',0,'a.txt','line 1','alpha','alpha')"
    )
    conn.commit()
    conn.close()
    return db


def _stub_semantic_embedding(monkeypatch, semantic_worker):
    monkeypatch.setattr(semantic_worker, "_load_tokenizer", lambda _root: object())
    monkeypatch.setattr(
        semantic_worker,
        "_windows",
        lambda _sp, _text, _tag: iter([(0, [7], "digest")]),
    )
    monkeypatch.setattr(
        semantic_worker,
        "_load_runtime",
        lambda _root: (np, object()),
    )

    def encode(_np, _session, batches):
        matrix = np.zeros(
            (len(batches), semantic_worker.config.SEMANTIC_DIM),
            dtype=np.float32,
        )
        matrix[:, 0] = 1.0
        return matrix

    monkeypatch.setattr(semantic_worker, "_encode_raw_batches", encode)


def test_semantic_sync_builds_ann_above_threshold(tmp_path, monkeypatch):
    import semantic_worker

    db = _semantic_sync_db(tmp_path)
    _stub_semantic_embedding(monkeypatch, semantic_worker)
    monkeypatch.setattr(semantic_worker.config, "SEMANTIC_FULL_SCAN_LIMIT", 0)
    calls = []

    def fake_build(path, **_kwargs):
        calls.append(path)
        return {
            "status": "ready",
            "available": True,
            "fresh": True,
            "count": 1,
        }

    monkeypatch.setattr(semantic_worker.ann_index, "build", fake_build)
    result = semantic_worker.sync(db, tmp_path / "snapshot123")

    assert result["status"] == "ok"
    assert result["vectors"] == 1
    assert result["ann"]["status"] == "ready"
    assert result["ann"]["required"] is True
    assert calls == [db]


def test_semantic_sync_survives_optional_ann_build_failure(tmp_path, monkeypatch):
    import semantic_worker

    db = _semantic_sync_db(tmp_path)
    _stub_semantic_embedding(monkeypatch, semantic_worker)
    monkeypatch.setattr(semantic_worker.config, "SEMANTIC_FULL_SCAN_LIMIT", 0)
    monkeypatch.setattr(
        semantic_worker.ann_index,
        "build",
        lambda _path, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic ANN failure")),
    )

    result = semantic_worker.sync(db, tmp_path / "snapshot123")

    assert result["status"] == "ok"
    assert result["vectors"] == 1
    assert result["ann"]["status"] == "unavailable"
    assert result["ann"]["required"] is True

    conn = sqlite3.connect(db)
    state = conn.execute(
        "SELECT generation,ready,vector_count FROM kb_semantic_state WHERE id=1"
    ).fetchone()
    conn.close()
    assert state == (1, 1, 1)


def test_ann_sidecar_paths_are_database_specific(tmp_path):
    import ann_index

    first = tmp_path / "one.sqlite3"
    second = tmp_path / "two.sqlite3"
    assert ann_index.paths(first) != ann_index.paths(second)


def test_ann_query_returns_unique_parent_candidates_under_child_clustering(tmp_path):
    import ann_index

    db = tmp_path / "clustered.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE kb_semantic_chunks("
        "path TEXT,parent_chunk_index INTEGER,child_index INTEGER,"
        "model_tag TEXT,content_hash TEXT,embedding BLOB,generation INTEGER)"
    )
    conn.execute(
        "CREATE TABLE kb_semantic_state("
        "id INTEGER PRIMARY KEY,generation INTEGER,model_tag TEXT,"
        "ready INTEGER,vector_count INTEGER)"
    )
    rows = [
        ("/a", 0, child, "model", f"a{child}", _blob(1.0, 0.0), 7)
        for child in range(300)
    ]
    rows.extend([
        ("/b", 1, 0, "model", "b", _blob(0.99, 0.1), 7),
        ("/c", 2, 0, "model", "c", _blob(0.0, 1.0), 7),
    ])
    conn.executemany(
        "INSERT INTO kb_semantic_chunks VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute(
        "INSERT INTO kb_semantic_state VALUES(1,7,'model',1,?)",
        (len(rows),),
    )
    conn.commit()
    conn.close()

    ann_index.build(db)
    query = [1.0] + [0.0] * (ann_index.EMBED_DIM - 1)
    candidates = ann_index.query(db, query, 2)
    parents = {
        (item["path"], item["parent_chunk_index"])
        for item in candidates
    }
    assert len(parents) == 2
    assert ("/a", 0) in parents
    assert ("/b", 1) in parents


def test_semantic_sync_honors_cancel_requested_during_ann_build(
    tmp_path,
    monkeypatch,
):
    import semantic_worker

    db = _semantic_sync_db(tmp_path)
    cancel_file = tmp_path / "cancel.flag"
    _stub_semantic_embedding(monkeypatch, semantic_worker)
    monkeypatch.setattr(semantic_worker.config, "SEMANTIC_FULL_SCAN_LIMIT", 0)

    def fake_build(_path, **_kwargs):
        cancel_file.write_text("cancel\n", encoding="utf-8")
        return {
            "status": "ready",
            "available": True,
            "fresh": True,
            "count": 1,
        }

    monkeypatch.setattr(semantic_worker.ann_index, "build", fake_build)
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


def test_ann_build_cancels_during_vector_scan_without_publishing(tmp_path, monkeypatch):
    import ann_index

    db = tmp_path / "cancel-build.sqlite3"
    sqlite3.connect(db).close()
    cancel_file = tmp_path / "cancel.flag"
    state = {
        "generation": 7,
        "model_tag": "model",
        "ready": 1,
        "vector_count": 300,
    }
    monkeypatch.setattr(ann_index, "snapshot", lambda _conn: dict(state))

    def fake_rows(_conn, _state):
        for child in range(300):
            if child == 255:
                cancel_file.write_text("cancel\n", encoding="utf-8")
            yield {
                "path": "/a",
                "parent_chunk_index": child // 10,
                "child_index": child,
            }, _blob(1.0, child / 1000.0)

    monkeypatch.setattr(ann_index, "vector_rows", fake_rows)
    ann_dir = tmp_path / "ann"
    with pytest.raises(ann_index.AnnCancelled):
        ann_index.build(db, ann_dir, cancel_file=cancel_file)

    index_path, metadata_path = ann_index.paths(db, ann_dir)
    assert not index_path.exists()
    assert not metadata_path.exists()


def test_large_semantic_index_is_not_ready_without_required_ann(
    tmp_path,
    monkeypatch,
):
    import config
    import semantic_client

    db = tmp_path / "large.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE kb_semantic_chunks("
        "path TEXT,parent_chunk_index INTEGER,child_index INTEGER,"
        "model_tag TEXT,content_hash TEXT,embedding BLOB,generation INTEGER)"
    )
    conn.execute(
        "CREATE TABLE kb_semantic_state("
        "id INTEGER PRIMARY KEY,generation INTEGER,model_tag TEXT,"
        "ready INTEGER,vector_count INTEGER)"
    )
    conn.execute(
        "INSERT INTO kb_semantic_state VALUES(1,7,'model',1,?)",
        (config.SEMANTIC_FULL_SCAN_LIMIT + 1,),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        semantic_client,
        "model_status",
        lambda: {
            "enabled": True,
            "ready": True,
            "model_tag": "model",
        },
    )
    state = semantic_client.index_status(db)
    assert state["ready"] is False
    assert state["reason"] == "ann_missing"
    assert state["ann_required"] is True


def test_large_semantic_index_is_ready_with_fresh_ann(
    tmp_path,
    monkeypatch,
):
    import ann_index
    import config
    import semantic_client

    db = _make_db(tmp_path)
    ann_index.build(db)
    monkeypatch.setattr(config, "SEMANTIC_FULL_SCAN_LIMIT", 0)
    monkeypatch.setattr(
        semantic_client,
        "model_status",
        lambda: {
            "enabled": True,
            "ready": True,
            "model_tag": "model",
        },
    )

    state = semantic_client.index_status(db)
    assert state["ready"] is True
    assert state["reason"] == "ok"
    assert state["ann_required"] is True
    assert state["ann_status"] == "ready"
    assert state["ann_count"] == 4


def test_same_size_metadata_corruption_invalidates_ann_status(tmp_path):
    import ann_index

    db = _make_db(tmp_path)
    ann_index.build(db)
    _index_path, metadata_path = ann_index.paths(db)
    original = bytearray(metadata_path.read_bytes())
    assert original
    original[0] = ord("[") if original[0] != ord("[") else ord("{")
    metadata_path.write_bytes(original)

    state = ann_index.status(db)
    assert state["fresh"] is False
    assert state["status"] in {"stale", "invalid", "incompatible"}


def test_same_size_metadata_corruption_invalidates_semantic_client_readiness(
    tmp_path,
    monkeypatch,
):
    import ann_index
    import config
    import semantic_client

    db = _make_db(tmp_path)
    ann_index.build(db)
    monkeypatch.setattr(config, "SEMANTIC_FULL_SCAN_LIMIT", 0)
    monkeypatch.setattr(
        semantic_client,
        "model_status",
        lambda: {
            "enabled": True,
            "ready": True,
            "model_tag": "model",
        },
    )

    _index_path, metadata_path = ann_index.paths(db)
    original = bytearray(metadata_path.read_bytes())
    original[0] = ord("[") if original[0] != ord("[") else ord("{")
    metadata_path.write_bytes(original)

    state = semantic_client.index_status(db)
    assert state["ready"] is False
    assert state["reason"].startswith("ann_")


def test_ann_query_fails_closed_if_publication_changes_after_status(
    tmp_path,
    monkeypatch,
):
    import json
    import ann_index

    db = _make_db(tmp_path)
    ann_index.build(db)
    real_status = ann_index.status

    def racing_status(path, directory=None):
        result = real_status(path, directory)
        published = ann_index.state_path(path, directory)
        payload = json.loads(published.read_text(encoding="utf-8"))
        payload["publishing"] = True
        published.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(ann_index, "status", racing_status)
    query = [1.0] + [0.0] * (ann_index.EMBED_DIM - 1)
    assert ann_index.query(db, query, 2) == []
