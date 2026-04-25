export CUDA_VISIBLE_DEVICES=0

model_id=ETTh1_LSTM_todo_smoke
model_name=LSTM_todo
export LOG_NAME=$model_id

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsproj_dl_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

"${PYTHON_BIN:-./.venv/bin/python}" -u run.py \
    --task_name long_term_forecast \
    --des 'Exp LSTM_todo ETTh1 smoke' \
    --is_training 1 \
    --is_testing 1 \
    --train_step 4 \
    --valid_step 4 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id "$model_id" \
    --model "$model_name" \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features MS \
    --target OT \
    --time date \
    --freq h \
    --embed timeF \
    --seq_len 24 \
    --label_len 0 \
    --pred_len 6 \
    --step_size 1 \
    --pred_method recursive_multi_step \
    --feature_size 7 \
    --output_size 1 \
    --hidden_size 32 \
    --num_layers 2 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --checkpoints ./results/pretrained_models/ \
    --test_results ./results/test_results/ \
    --forecast_results ./results/forecast_results/ \
    --itr 1 \
    --train_epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-3 \
    --loss MSE \
    --optimizer adam \
    --activation gelu \
    --use_dtw 0 \
    --patience 3 \
    --lradj type1 \
    --scale 1 \
    --inverse 0 \
    --num_workers 0 \
    --use_gpu 0 \
    --gpu_type cpu \
    --use_multi_gpu 0 \
    --devices 0
