#!/usr/bin/env bash
# 已生成 fixtures 的 CPU base 全阶段 smoke；不删除旧产物，不覆盖已有 run_id。
set -euo pipefail
cd "$(dirname "$0")/../.."
prefix="${1:?用法: bash scripts/smoke/smoke_gslaformer.sh NEW_RUN_PREFIX}"
config="scripts/smoke/gslaformer_functional.json"
python=".venv/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
env -u PYTHONPATH "$python" run.py --model GSLAformer --gsla_config "$config" --gsla_mode train --gsla_run_id "${prefix}_train"
checkpoint="results/gslaformer/${prefix}_train/checkpoints/best.pt"
env -u PYTHONPATH "$python" run.py --model GSLAformer --gsla_config "$config" --gsla_mode test --gsla_run_id "${prefix}_test" --gsla_checkpoint "$checkpoint"
for request in normal_forecast new_entity_cold new_entity_history new_parent; do
  env -u PYTHONPATH "$python" run.py --model GSLAformer --gsla_config "$config" --gsla_mode forecast --gsla_run_id "${prefix}_${request}" --gsla_checkpoint "$checkpoint" --gsla_request "results/gslaformer/fixtures/requests/${request}"
done
