#!/usr/bin/env python3
"""Detached background sync worker for MCP-RagDoc.

Launched only by rag_service.start_sync(). Configuration is inherited through
RAGDOC_* environment variables and all mutable state stays in DATA_DIR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import rag_service


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path, required=True)
    args = parser.parse_args()
    return rag_service.run_persistent_job(
        args.job_id,
        args.status_file,
        args.cancel_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
