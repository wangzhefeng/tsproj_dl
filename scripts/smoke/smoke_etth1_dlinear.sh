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

export LOG_NAME="${LOG_NAME:-smoke-etth1-dlinear}"

"$PYTHON_BIN" -u "$ROOT_DIR/run_dl.py" \
    --task_name long_term_forecast \
    --des "Smoke DLinear ETTh1" \
    --is_training 1 \
    --is_testing 0 \
    --is_forecasting 0 \
    --model_id smoke_etth1_dlinear \
    --model DLinear \
    --root_path "$ROOT_DIR/dataset/ETT-small/" \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features M \
    --target OT \
    --time date \
    --results_root "$ROOT_DIR/results/smoke_core004/" \
    --freq 1h \
    --embed timeF \
    --seq_len 24 \
    --label_len 12 \
    --pred_len 24 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --n_heads 1 \
    --dropout 0.05 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 1 \
    --batch_size 8 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-4 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 0 \
    --use_multi_gpu 0
