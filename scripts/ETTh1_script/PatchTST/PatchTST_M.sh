# export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=patchtst-etth1

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsproj_dl_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

model_name=PatchTST

# 训练、验证、测试
"${PYTHON_BIN:-./.venv/bin/python}" -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp PatchTST_M_96_48_24' \
    --is_training 1 \
    --is_testing 1 \
    --is_forecasting 1 \
    --train_step 1 \
    --valid_step 1 \
    --testing_step 1 \
    --model_id etth1_24_12_24 \
    --model $model_name \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features M \
    --target OT \
    --time date \
    --checkpoints ./results/pretrained_models/ \
    --test_results ./results/test_results/ \
    --forecast_results ./results/forecast_results/ \
    --freq h \
    --embed timeF \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 64 \
    --d_ff 128 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --n_heads 4 \
    --dropout 0.1 \
    --padding 0 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 30 \
    --batch_size 8 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-3 \
    --patience 10 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'mps' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
