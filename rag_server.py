#!/usr/bin/env python3
"""Standalone stdio MCP server for local document retrieval.

Retrieval stays deterministic; scanned PDF OCR may optionally use a separate
local vision model before SQLite indexing. One process is bound to one
operator-selected knowledge directory and one private state directory.
"""
from __future__ import annotations

import argparse
import os
from typing import Literal


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
        "Use rag_manage for sync/cancel actions and rag_status for health/job status."
    ),
)


ManageAction = Literal["sync", "cancel"]
StatusView = Literal["health", "job"]


@mcp.tool()
def rag_search(query: str) -> str:
    """Search configured documents by filename, keywords, OCR text, or meaning.

    Semantic retrieval is optional; when unavailable or not ready, search
    degrades cleanly to the lexical/OCR path. Returned absolute source paths can
    be opened by whatever file-reading tool the calling agent provides.
    """
    return rag_service.search(query)


def sync_kb() -> str:
    """Start a background refresh of the private lexical/semantic index.

    Returns immediately with a job_id. Source documents are read-only; writes
    stay inside MCP-RagDoc's private state. Public MCP callers poll through
    rag_status(view="job") and cancel through rag_manage(action="cancel").
    """
    return rag_service.start_sync()


def sync_status(job_id: str) -> str:
    """Return phase, progress, current file and ETA for one sync job."""
    return rag_service.sync_status(job_id)


def cancel_sync(job_id: str) -> str:
    """Request cooperative cancellation of a queued/running sync job."""
    return rag_service.cancel_sync(job_id)


def rag_health() -> str:
    """Report source/index/semantic readiness for this MCP-RagDoc instance."""
    return rag_service.health()


@mcp.tool()
def rag_manage(action: ManageAction, job_id: str = "") -> str:
    """Manage the persistent sync lifecycle.

    action choices:
    - sync: start an incremental background refresh and return a job id
    - cancel: request cooperative cancellation for job_id
    """
    action = str(action).strip().casefold()
    if action == "sync":
        return sync_kb()
    if action == "cancel":
        return cancel_sync(job_id)
    raise ValueError("unsupported action")


@mcp.tool()
def rag_status(view: StatusView = "health", job_id: str = "") -> str:
    """Read health or one persistent sync-job status through one tool.

    view choices:
    - health: source/index/semantic readiness
    - job: phase/progress/current file/ETA for job_id
    """
    view = str(view or "health").strip().casefold()
    if view == "health":
        return rag_health()
    if view == "job":
        return sync_status(job_id)
    raise ValueError("unsupported view")


if __name__ == "__main__":
    mcp.run(transport="stdio")
