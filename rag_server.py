#!/usr/bin/env python3
"""Standalone stdio MCP server for local document retrieval.

Retrieval stays deterministic; scanned PDF OCR may optionally use a separate
local vision model before SQLite indexing. One process is bound to one
operator-selected knowledge directory and one private state directory.
"""
from __future__ import annotations

import argparse
import os


def _apply_launch_configuration() -> None:
    """Apply trusted process-launch options before importing RAG modules."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--knowledge-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--db")
    parser.add_argument("--semantic", choices=("on", "off"))
    parser.add_argument("--semantic-model-dir")
    args, _unknown = parser.parse_known_args()
    if args.knowledge_dir:
        os.environ["RAGDOC_KNOWLEDGE_DIR"] = args.knowledge_dir
    if args.data_dir:
        os.environ["RAGDOC_DATA_DIR"] = args.data_dir
    if args.db:
        os.environ["RAGDOC_DB"] = args.db
    if args.semantic:
        os.environ["RAGDOC_SEMANTIC"] = "1" if args.semantic == "on" else "0"
    if args.semantic_model_dir:
        os.environ["RAGDOC_SEMANTIC_MODEL_DIR"] = args.semantic_model_dir


_apply_launch_configuration()

from mcp.server.fastmcp import FastMCP  # noqa: E402
import rag_service  # noqa: E402

mcp = FastMCP(
    "MCP-RagDoc",
    instructions=(
        "Local document RAG with deterministic retrieval. Scanned PDF OCR may "
        "use the separately configured local vision model before SQLite indexing; "
        "the retrieval path itself never calls an LLM. "
        "Search is permanently scoped to the knowledge directory selected by "
        "the server operator. Use rag_search for filename/content/semantic lookup. "
        "sync_kb starts a background refresh and returns a job id; poll sync_status "
        "for progress or call cancel_sync to request cooperative cancellation."
    ),
)


@mcp.tool()
def rag_search(query: str) -> str:
    """Search configured documents by filename, keywords, OCR text, or meaning.

    Semantic retrieval is optional; when unavailable or not ready, search
    degrades cleanly to the lexical/OCR path. Returned absolute source paths can
    be opened by whatever file-reading tool the calling agent provides.
    """
    return rag_service.search(query)


@mcp.tool()
def sync_kb() -> str:
    """Start a background refresh of the private lexical/semantic index.

    Returns immediately with a job_id. Source documents are read-only; writes
    stay inside MCP-RagDoc's private state. Poll sync_status(job_id) while the
    job runs, or call cancel_sync(job_id) to stop at a safe checkpoint.
    """
    return rag_service.start_sync()


@mcp.tool()
def sync_status(job_id: str) -> str:
    """Return phase, progress, current file and ETA for one sync job."""
    return rag_service.sync_status(job_id)


@mcp.tool()
def cancel_sync(job_id: str) -> str:
    """Request cooperative cancellation of a queued/running sync job."""
    return rag_service.cancel_sync(job_id)


@mcp.tool()
def rag_health() -> str:
    """Report source/index/semantic readiness for this MCP-RagDoc instance."""
    return rag_service.health()


if __name__ == "__main__":
    mcp.run(transport="stdio")
