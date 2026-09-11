"""Legacy local bootstrap retained for compatibility.

MCP-RagDoc is self-contained; this module intentionally does not add any
parent project directory to sys.path.
"""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
