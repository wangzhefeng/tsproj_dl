#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsproj_dl_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN or create the local .venv first." >&2
    exit 1
fi

"$PYTHON_BIN" -m unittest discover -s "$ROOT_DIR/tests" -p "test_*.py" -v
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/audit_runtime_utils_imports.py" --summary
