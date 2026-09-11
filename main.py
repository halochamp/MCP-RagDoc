#!/usr/bin/env python3
"""Standalone MCP-RagDoc loopback HTML UI launcher."""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone MCP-RagDoc web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("RAGDOC_UI_PORT", "8769")))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("MCP-RagDoc UI is local-only; use 127.0.0.1 or localhost")
    # web_ui builds its same-origin allowlist from this environment value when
    # uvicorn imports the module. Keep --port and that policy in sync.
    os.environ["RAGDOC_UI_PORT"] = str(args.port)
    uvicorn.run("web_ui:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
