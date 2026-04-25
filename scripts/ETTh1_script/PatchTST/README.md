# PatchTST ETTh1 scripts

本目录只存放 ETTh1 数据集上的 PatchTST 主线脚本。

- `PatchTST_S.sh`、`PatchTST_MS.sh`、`PatchTST_M.sh`：研发 smoke，执行 train/test/forecast。
- `PatchTST_S_forecast.sh`、`PatchTST_MS_forecast.sh`、`PatchTST_M_forecast.sh`：生产离线推理，只执行 forecast。
- 设备通过脚本内 `--use_gpu`、`--gpu_type`、`--devices` 手动控制。
- 生产 forecast 脚本要求已存在对应 checkpoint 和 scaler artifact。
