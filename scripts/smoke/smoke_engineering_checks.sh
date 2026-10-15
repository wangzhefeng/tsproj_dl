#!/usr/bin/env bash
# 工程回归与静态 utils 审计；不启动完整实验矩阵，不安装依赖。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Missing project interpreter: %s\nCreate the root .venv with uv sync first.\n' "$PYTHON_BIN" >&2
    exit 1
fi
cd "$ROOT_DIR"
unset PYTHONPATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/results/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_BIN" scripts/dev/audit_runtime_utils_imports.py --summary
