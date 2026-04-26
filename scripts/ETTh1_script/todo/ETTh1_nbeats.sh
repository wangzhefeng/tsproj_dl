export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=nbeats-etth1

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsproj_dl_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

model_name=N_BEATS

"${PYTHON_BIN:-./.venv/bin/python}" -u run.py \
    --task_name long_term_forecast \
    --des 'Exp NBEATS_24_12_24' \
    --is_training 1 \
    --is_testing 0 \
    --is_forecasting 0 \
    --model_id etth1_24_12_24 \
    --model $model_name \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features MS \
    --target OT \
    --time date \
    --results_root ./results/ \
    --freq 1h \
    --embed timeF \
    --seq_len 24 \
    --label_len 12 \
    --pred_len 24 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --d_model 128 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 1 \
    --e_layers 2 \
    --d_layers 2 \
    --dropout 0.05 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 1 \
    --batch_size 32 \
    --loss MSE \
    --learning_rate 1e-4 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 0 \
    --use_multi_gpu 0
