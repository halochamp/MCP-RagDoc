#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[error] Apple Vision OCR helper requires macOS." >&2
  exit 1
fi
SWIFTC="$(xcrun --find swiftc 2>/dev/null || true)"
if [[ -z "$SWIFTC" ]]; then
  echo "[error] swiftc not found. Install Xcode Command Line Tools." >&2
  exit 1
fi
"$SWIFTC" "$ROOT/helpers/_vision_ocr.swift" -o "$ROOT/helpers/vision_ocr_macos"
chmod 755 "$ROOT/helpers/vision_ocr_macos"
echo "Built: $ROOT/helpers/vision_ocr_macos"
