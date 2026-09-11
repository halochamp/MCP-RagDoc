# MCP-RagDoc

MCP-RagDoc is an **agent-agnostic document RAG MCP server**. Retrieval remains
deterministic and does not depend on an agent host. For image-only PDF pages,
indexing can optionally send the rendered page image plus OCR text to the same
local OpenAI-compatible vision model, then validate the conservative rewrite
before writing SQLite. MCP-RagDoc owns its client and configuration and imports no host-agent modules.

## MCP tools

- `rag_search(query)` — filename + FTS5/BM25 + OCR fuzzy + optional MiniLM
  semantic retrieval, fused deterministically with RRF.
- `sync_kb()` — start an incremental background rebuild and return a persistent
  `job_id` immediately.
- `sync_status(job_id)` — poll phase, percentage, current file, completed/total
  files, ETA, and semantic parent progress across MCP process restarts.
- `cancel_sync(job_id)` — request cooperative cancellation at safe checkpoints;
  the current document transaction rolls back and an incomplete semantic
  generation is never published ready.
- `rag_health()` — report source/index/semantic readiness without reading full
  document bodies.

## Standalone layout

- `config.py` — generic `RAGDOC_*` configuration; no host-agent imports.
- `rag_server.py` — FastMCP stdio entry point and trusted launch arguments.
- `rag_service.py` — validation, locking, persistent sync-job/search/health service boundary.
- `sync_job_worker.py` — detached background sync worker used by the job API.
- `rag_sync.py` — streaming extraction/OCR and SQLite lexical lifecycle.
- `rag_search.py` — filename/keyword/OCR/semantic retrieval and RRF fusion.
- `semantic_client.py` — standard-library-only worker launcher.
- `semantic_worker.py` — short-lived ONNX INT8 MiniLM embed/cosine worker.
- `web_ui.py` + `ui/rag.html` — loopback Search / Sync / Progress / Health console.
- `main.py` — loopback HTML UI launcher.
- `rag_eval.py` — no-LLM Recall@1/3/8 + MRR evaluation harness.
- `helpers/` — bundled extraction, safety, Thai normalization/word-boundary,
  and Apple Vision OCR support used by the RAG pipeline.
- `requirements.txt` — Python runtime dependencies.

MCP-RagDoc is self-contained. Host-specific routing, retry, and turn guards belong in the consuming agent rather than in this backend.

## Source and state

One server process is permanently scoped to one source directory for its
lifetime. Source files are **read-only** to MCP-RagDoc. Supported files placed
in that directory or its visible nested directories are indexed; hidden,
symlinked, and sensitive paths are ignored.

Standalone defaults:

- source documents: `rag_document/`
- private state/index: `~/.mcp-ragdoc/`

For a real integration, pass explicit paths at process launch:

```sh
python ./rag_server.py \
  --knowledge-dir /absolute/path/to/rag_document \
  --data-dir /absolute/path/to/private-state
```

An explicit database can also be selected with `--db /path/index.sqlite3`.
These are operator/developer launch settings; MCP tool arguments cannot change
or broaden the source directory.

Any MCP client can launch this server with its own source and private-state paths without changing MCP-RagDoc source code.

## Sync jobs and cancellation

MCP `sync_kb()` is non-blocking. It launches a detached worker, persists job
state under the private `DATA_DIR/sync_jobs/`, and returns a `job_id`. This
persists across short-lived stdio MCP sessions, so clients that open a fresh MCP
process for each tool call can still poll or cancel the same job.

Lexical progress reports prepare/file/result/cleanup phases with file counts,
percentage, and ETA. Semantic refresh reports approximately the final 90-99%
using processed/total parent counts. Cancellation is cooperative: checks occur
before files, within chunk generation, during cleanup, and through the semantic
worker. Already committed documents remain valid, the in-progress document
transaction rolls back, and a cancelled semantic generation stays `ready=0`.

Host integrations that deliberately need the historical blocking callback API
may call the Python service function `sync_kb_blocking(...)`; that function is
not exposed as the MCP `sync_kb` tool.

## Standalone HTML UI

Run the loopback-only console:

```bash
/opt/homebrew/anaconda3/envs/mlx/bin/python3 \
  ./main.py
```

Then open `http://127.0.0.1:8769`.

The console mirrors the RAG MAX layout while using MCP-RagDoc's own service:

- deterministic hybrid search with no LLM call during retrieval;
- optional vision-grounded OCR correction before scanned-PDF text enters SQLite;
- health, document/chunk/vector counts and source/index paths;
- Sync KB with persistent progress and cancellation;
- indexed-file listing with relative path, absolute path, type, size, and chunk count;
- search output that preserves the source absolute path and page/location, so the
  original PDF or document can be opened directly.

### Optional OCR vision model

MCP-RagDoc can use an already-running local OpenAI-compatible vision server on `127.0.0.1:8090` for conservative OCR correction. The standalone public defaults do **not** start or stop a model process. If no compatible server is available, indexing safely keeps the original Apple Vision OCR text.

A host application may optionally provide a trusted loopback lifecycle endpoint compatible with `server/start` and `server/stop`. To opt in, set `RAGDOC_LLM_CONTROL_URL` and enable `RAGDOC_OCR_AUTO_START=1` (and optionally `RAGDOC_OCR_AUTO_STOP=1`). This integration is optional and MCP-RagDoc imports no host-agent modules.

State-changing browser requests require same-origin headers and the server
binds to loopback only. Set `RAGDOC_UI_PORT` or pass `--port` to choose another
local port.

## Supported documents

The sync path supports plain text/Markdown/CSV/JSON/YAML-style text, PDF,
DOC/DOCX, XLS/XLSX, PPT/PPTX, ODF documents, and common image formats. Large
sources are streamed into bounded chunks instead of materializing one full
extracted document in memory.

On macOS ARM64, the bundled `helpers/_vision_ocr.swift` source can be compiled to `helpers/vision_ocr_macos` for Apple Vision OCR
for images and scanned-PDF fallback. Legacy `.doc` uses macOS `textutil` and
legacy `.ppt` may use Spotlight `mdls` when available.

## Keyword + semantic retrieval

Lexical retrieval remains the always-available base:

1. filename-first lookup
2. SQLite FTS5 trigram + BM25
3. deterministic exact/all-term ranking
4. Thai/mixed word segmentation and aliases
5. English split-word recovery
6. bounded OCR fuzzy recovery
7. filesystem-truth/path validation

Optional semantic retrieval uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the official
ARM64 ONNX INT8 artifact, direct SentencePiece, 384-dimensional normalized
float16 vectors, streaming exact cosine, and RRF fusion. Semantic children are
word-boundary aware, capped at 112 content tokens with up to 24 tokens overlap.
The semantic process exits after each build/query so its ML runtime does not
remain resident in the MCP server.

The model is never downloaded by a search request. MCP-RagDoc auto-discovers a
compatible snapshot in the Hugging Face cache, or the operator can pass:

```sh
--semantic-model-dir /absolute/path/to/minilm-snapshot
```

Disable semantic retrieval with `--semantic off`; lexical/OCR retrieval remains
functional.

## Install

Use a Python environment available to the host MCP client:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
# macOS: build the local Apple Vision OCR helper
bash scripts/build_vision_ocr.sh
```

`onnxruntime`, `numpy`, and `sentencepiece` are required only for semantic
retrieval. `pythainlp` is optional but recommended for Thai whole-word chunking.
The OCR rewrite client uses the local OpenAI-compatible `mlx_vlm.server` at
`http://127.0.0.1:8090/v1` by default, with model
`mlx-community/Qwen3.5-2B-OptiQ-4bit`. It sends no-think multimodal requests
one page at a time, using a 300-DPI page image plus the original OCR text. If
the server is unavailable or validation fails, the original OCR is indexed
unchanged.

## Retrieval evaluation

`rag_eval.py` evaluates the real retrieval path without an LLM:

```sh
python ./rag_eval.py golden.json \
  --knowledge-dir /absolute/path/to/rag_document \
  --data-dir /absolute/path/to/state \
  --mode hybrid --details

python ./rag_eval.py golden.json \
  --knowledge-dir /absolute/path/to/rag_document \
  --data-dir /absolute/path/to/state \
  --mode lexical --details
```

The report contains Recall@1, Recall@3, Recall@8, MRR, and optional per-query
rankings. Use measured retrieval evidence before changing RRF or adding a
heavier reranker.

## Safety/lifecycle invariants

- source documents are never modified by `sync_kb` or `rag_search`;
- SQLite/state writes stay in the operator-selected private state/database;
- symlinks, hidden source files, sensitive-looking paths/content, stale source
  rows, and paths outside the configured source root fail closed;
- semantic generations publish `ready=1` only after a complete successful build;
- model snapshot/policy fingerprints prevent mixing incompatible vectors;
- missing OCR/semantic/VLM dependencies degrade explicitly, preserving the
  original OCR when rewrite is unavailable, and never broaden filesystem scope.
