"""Standalone configuration for MCP-RagDoc.

MCP-RagDoc is an agent-agnostic stdio MCP server.  Operators select one
knowledge directory and one private state directory when launching the server;
no calling model can broaden those roots at runtime.
"""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


CODE_DIR = Path(__file__).resolve().parent

# Standalone defaults.  Integrators normally pass --knowledge-dir/--data-dir to
# rag_server.py; those CLI flags are converted to these environment variables
# before this module is imported.
KNOWLEDGE_DIR = Path(
    os.getenv("RAGDOC_KNOWLEDGE_DIR", str(CODE_DIR / "rag_document"))
).expanduser().resolve(strict=False)
DATA_DIR = Path(
    os.getenv("RAGDOC_DATA_DIR", str(Path.home() / ".mcp-ragdoc"))
).expanduser().resolve(strict=False)
FILE_INDEX_DB = Path(
    os.getenv("RAGDOC_DB", str(DATA_DIR / "ragdoc.sqlite3"))
).expanduser().resolve(strict=False)

# Compatibility aliases used by the mature retrieval implementation.  They all
# stay inside MCP-RagDoc and do not depend on a host-agent module.
KNOWLEDGE_SOURCE_DIR = KNOWLEDGE_DIR
KNOWLEDGE_RAW_DIR = KNOWLEDGE_DIR
RAW_DIR = KNOWLEDGE_DIR
KNOWLEDGE_MD_DIR = KNOWLEDGE_DIR
MD_DIR = KNOWLEDGE_DIR
INDEX_ROOTS = [KNOWLEDGE_DIR]
# Standalone MCP-RagDoc recursively scans supported files below KNOWLEDGE_DIR.
# Host compatibility layers may explicitly opt into additional result roots for
# a pre-existing external index; this setting does not disable root recursion.
ALLOW_INDEX_ROOT_RESULTS = _env_bool("RAGDOC_ALLOW_INDEX_ROOT_RESULTS", False)

RAG_MAX_RESULTS = _env_int("RAGDOC_MAX_RESULTS", 8, 1, 50)
RAG_SNIPPET_CHARS = _env_int("RAGDOC_SNIPPET_CHARS", 300, 50, 10_000)
READ_IMAGE_MAX_BYTES = _env_int(
    "RAGDOC_IMAGE_MAX_BYTES", 10 * 1024 * 1024, 100_000, 100 * 1024 * 1024
)

KB_CHUNK_TARGET_CHARS = _env_int("RAGDOC_CHUNK_CHARS", 3000, 1000, 100_000)
KB_CHUNK_OVERLAP_CHARS = min(
    KB_CHUNK_TARGET_CHARS // 2,
    _env_int(
        "RAGDOC_CHUNK_OVERLAP_CHARS",
        250,
        0,
        KB_CHUNK_TARGET_CHARS // 2,
    ),
)

# Optional vision-grounded OCR correction.  This is deliberately configured in
# MCP-RagDoc rather than imported from a host agent: the same local Qwen vision
# model can use a compatible loopback OpenAI-style server on port 8090, while the
# client and configuration remain independent of any host-agent Python modules.
LLM_MODEL = (
    os.getenv("RAGDOC_LLM_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit").strip()
    or "mlx-community/Qwen3.5-2B-OptiQ-4bit"
)


def _loopback_host(value: str) -> str:
    candidate = (value or "127.0.0.1").strip()
    if candidate == "localhost":
        return candidate
    try:
        if ipaddress.ip_address(candidate).is_loopback:
            return candidate
    except ValueError:
        pass
    return "127.0.0.1"


LLM_HOST = _loopback_host(os.getenv("RAGDOC_LLM_HOST", "127.0.0.1"))
LLM_PORT = _env_int("RAGDOC_LLM_PORT", 8090, 1024, 65535)


def _local_base_url(value: str, host: str, port: int) -> str:
    default = f"http://{'[' + host + ']' if ':' in host else host}:{port}/v1"
    candidate = (value or default).strip()
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold()
        is_loopback = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        if (
            parsed.scheme != "http"
            or not is_loopback
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, port)
            or parsed.path.rstrip("/") != "/v1"
        ):
            return default
    except (ValueError, UnicodeError):
        return default
    return candidate.rstrip("/")


LLM_BASE_URL = _local_base_url(
    os.getenv("RAGDOC_LLM_BASE_URL", ""), LLM_HOST, LLM_PORT
)


def _local_control_url(value: str, default: str) -> str:
    """Accept only a loopback HTTP origin for an optional lifecycle control plane."""
    candidate = (value or default).strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold()
        is_loopback = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        if (
            parsed.scheme != "http"
            or not is_loopback
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
            or parsed.port is None
        ):
            return default
    except (ValueError, UnicodeError):
        return default
    return candidate


# RagDoc does not spawn/kill mlx_vlm.server itself. An operator may opt into
# a separate trusted loopback lifecycle control plane; standalone defaults leave
# this unset and never attempt process ownership.
LLM_CONTROL_URL = _local_control_url(
    os.getenv("RAGDOC_LLM_CONTROL_URL", ""),
    "",
)
LLM_CONTROL_ORIGIN = LLM_CONTROL_URL
LLM_CONTROL_TIMEOUT_SECONDS = _env_float(
    "RAGDOC_LLM_CONTROL_TIMEOUT", 5.0, 1.0, 30.0
)
LLM_SERVER_START_TIMEOUT_SECONDS = _env_int(
    "RAGDOC_LLM_START_TIMEOUT", 180, 10, 600
)
_raw_llm_api_key = os.getenv("RAGDOC_LLM_API_KEY", "x") or "x"
LLM_API_KEY = (
    _raw_llm_api_key.strip()
    if _raw_llm_api_key.strip()
    and all(
        ord(char) >= 0x20 and ord(char) != 0x7F and not char.isspace()
        for char in _raw_llm_api_key
    )
    else "x"
)
LLM_REPETITION_PENALTY = _env_float("RAGDOC_LLM_REPETITION_PENALTY", 1.1, 1.0, 2.0)
LLM_VISION_CACHE_SIZE = _env_int("RAGDOC_LLM_VISION_CACHE_SIZE", 2, 0, 20)
LLM_PREFILL_STEP_SIZE = _env_int("RAGDOC_LLM_PREFILL_STEP_SIZE", 512, 1, 4096)

OCR_REWRITE_ENABLED = _env_bool("RAGDOC_OCR_REWRITE", True)
# Keep one page per multimodal request. A full four-page transcription asks a
# small VLM to reproduce too much dense Thai text in one response.
OCR_REWRITE_BATCH_PAGES = _env_int("RAGDOC_OCR_REWRITE_BATCH_PAGES", 1, 1, 1)
OCR_REWRITE_BATCH_CHARS = _env_int("RAGDOC_OCR_REWRITE_BATCH_CHARS", 9_000, 1_000, 20_000)
OCR_REWRITE_TIMEOUT_SECONDS = _env_int("RAGDOC_OCR_REWRITE_TIMEOUT", 90, 10, 300)
OCR_REWRITE_MAX_TOKENS = _env_int("RAGDOC_OCR_REWRITE_MAX_TOKENS", 8_192, 256, 8_192)
# The same high-resolution page image is used by Apple Vision OCR and the VLM.
OCR_REWRITE_IMAGE_DPI = _env_int("RAGDOC_OCR_REWRITE_IMAGE_DPI", 300, 150, 600)
OCR_REWRITE_AUTO_START = _env_bool("RAGDOC_OCR_AUTO_START", False)
OCR_REWRITE_AUTO_STOP = _env_bool("RAGDOC_OCR_AUTO_STOP", False)

# Optional semantic retrieval.  Heavy packages are imported only by the
# short-lived semantic_worker subprocess, never by the MCP server process.
SEMANTIC_ENABLED = _env_bool("RAGDOC_SEMANTIC", True)
SEMANTIC_MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_MODEL_ONNX = "onnx/model_qint8_arm64.onnx"
SEMANTIC_SENTENCEPIECE = "sentencepiece.bpe.model"
SEMANTIC_POLICY_ID = "onnx-int8-arm64:v3-generation-word-boundary"
SEMANTIC_DIM = 384
SEMANTIC_CHILD_TOKENS = _env_int("RAGDOC_SEMANTIC_CHILD_TOKENS", 112, 32, 126)
SEMANTIC_CHILD_OVERLAP_TOKENS = min(
    SEMANTIC_CHILD_TOKENS // 2,
    _env_int(
        "RAGDOC_SEMANTIC_CHILD_OVERLAP_TOKENS",
        24,
        0,
        SEMANTIC_CHILD_TOKENS // 2,
    ),
)
SEMANTIC_EMBED_BATCH = _env_int("RAGDOC_SEMANTIC_EMBED_BATCH", 16, 1, 64)
SEMANTIC_SCAN_BATCH = _env_int("RAGDOC_SEMANTIC_SCAN_BATCH", 256, 32, 2048)
SEMANTIC_CANDIDATES = _env_int("RAGDOC_SEMANTIC_CANDIDATES", 24, 4, 100)
SEMANTIC_TIMEOUT_SECONDS = _env_int("RAGDOC_SEMANTIC_TIMEOUT", 20, 5, 120)
SEMANTIC_SYNC_TIMEOUT_SECONDS = _env_int("RAGDOC_SEMANTIC_SYNC_TIMEOUT", 600, 30, 3600)

# Only private state is permission-tightened.  The knowledge directory may be a
# user-managed or shared folder and MCP-RagDoc must not mutate its permissions.
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    DATA_DIR.chmod(0o700)
except OSError:
    pass
