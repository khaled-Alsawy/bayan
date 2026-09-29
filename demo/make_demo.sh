#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# The Whisper leak gate must use the cached model, never a hub lookup that can stall.
export HF_HUB_OFFLINE=1

"$ROOT_DIR/.venv/bin/python" demo/tts.py
"$ROOT_DIR/.venv/bin/python" demo/preflight.py
(cd "$ROOT_DIR/web" && npx playwright test -c playwright.demo.config.ts)
"$ROOT_DIR/.venv/bin/python" demo/compose.py --trim-waits
"$ROOT_DIR/.venv/bin/python" demo/review.py
