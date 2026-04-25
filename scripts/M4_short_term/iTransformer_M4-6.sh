# export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=itransformer-m4-hourly

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsproj_dl_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

model_name=iTransformer

"${PYTHON_BIN:-./.venv/bin/python}" -u run_dl.py \
  --task_name short_term_forecast \
  --des 'Exp' \
  --is_training 1 \
  --is_testing 1 \
  --is_forecasting 0 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --checkpoints ./results/pretrained_models/ \
  --test_results ./results/test_results/ \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 1 \
  --dropout 0.05 \
  --num_workers 0 \
  --itr 1 \
  --train_epochs 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --activation gelu \
  --use_dtw 0 \
  --patience 7 \
  --lradj type1 \
  --use_gpu 1 \
  --gpu_type 'mps' \
  --use_multi_gpu 0 \
  --devices 0,1,2,3,4,5,6,7
