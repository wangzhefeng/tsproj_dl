# tsproj_dl

`tsproj_dl` 是一个面向时间序列预测任务的神经网络快速测试、训练、验证与推理项目。当前仓库以 PyTorch 为主，包含 Transformer、MLP、RNN、CNN、GNN 等多类模型实现，以及配套的数据处理、实验流程和脚本模板。

项目治理、分支流转、模块状态和开发看板统一维护在 [AGENTS.md](/Users/wangzf/projects/tsproj_dl/AGENTS.md)。本文件只负责项目使用说明。

## 项目简介

- 当前推荐入口是 `run.py`。
- 当前推荐主线是长周期时间序列预测链路：
  `run.py -> exp/exp_long_term_forecasting.py -> data_provider/TFs_type -> exp/exp_basic.py`
- 目前仓库中包含可直接使用模块、在建模块和待补充模块，具体开发状态请查看根目录 `AGENTS.md`。

## 环境准备

项目根目录已经包含 `pyproject.toml`、`uv.lock` 和 `requirements.txt`。推荐使用 `uv` 初始化环境。

```bash
uv sync
source .venv/bin/activate
```

如果只使用已有虚拟环境，也可以直接：

```bash
source .venv/bin/activate
```

说明：

- Python 要求见 `pyproject.toml`，当前为 `>=3.10`
- 仓库中部分模型依赖额外第三方包，后续以 `AGENTS.md` 中的主线稳定化任务为准逐步补齐

## 快速运行

最简单的使用方式是直接运行 `scripts/` 下已经整理好的实验脚本。

示例 1：ETTh1 上运行 Transformer

```bash
bash scripts/ETTh1_script/ETTh1_transformer.sh
```

示例 2：ETTh1 上运行已经补齐并验证过的 MLP 系列脚本

```bash
bash scripts/ETTh1_script/ETTh1_tsmixer.sh
bash scripts/ETTh1_script/ETTh1_nhits.sh
bash scripts/ETTh1_script/ETTh1_nbeats.sh
```

示例 3：运行已经验证通过的 ETTh1 smoke 脚本

```bash
bash scripts/smoke/smoke_etth1_dlinear.sh
bash scripts/smoke/smoke_etth1_transformer.sh
```

示例 4：直接调用训练入口

```bash
python run.py \
    --task_name long_term_forecast \
    --des "Exp Transformer_24_12_24" \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id etth1_24_12_24 \
    --model Transformer \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features MS \
    --target OT \
    --time date \
    --freq 1h \
    --seq_len 24 \
    --label_len 12 \
    --pred_len 24 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --results_root ./results/ \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 1 \
    --e_layers 2 \
    --d_layers 1 \
    --n_heads 1 \
    --d_model 512 \
    --d_ff 2048 \
    --dropout 0.05 \
    --train_epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --loss MSE \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type mps
```

## 结果输出

- `logs/`：运行日志
- `results/pretrained_models/`：默认模型权重输出目录
- `results/test_results/`：默认测试结果与可视化输出目录
- `results/forecast_results/`：默认预测结果输出目录

说明：

- 当前统一使用 `results/` 作为结果输出根目录
- 如需自定义输出位置，可通过 `--results_root` 或单独目录参数覆盖

## 目录简介

- `run.py`：命令行训练/测试/预测入口
- `exp/`：实验流程，包括训练、验证、测试、预测主逻辑
- `data_provider/`：数据读取、切窗、标准化、时间特征与 DataLoader 构建
- `models/`：模型实现，按 `transformer`、`mlp`、`rnn`、`cnn`、`gnn`、`others` 分类
- `layers/`：模型公用层与模块
- `scripts/`：可直接执行的实验脚本，按数据集或业务场景组织
- `docs/`：研究文档、图示和资料
- `dataset/`：本地测试数据集
- `results/`：默认训练和推理结果输出目录

## 示例脚本

可以优先参考以下脚本理解当前使用方式：

- `scripts/ETTh1_script/ETTh1_transformer.sh`
- `scripts/ETTh1_script/ETTh1_dlinear.sh`
- `scripts/ETTh1_script/ETTh1_tsmixer.sh`
- `scripts/ETTh1_script/ETTh1_nhits.sh`
- `scripts/ETTh1_script/ETTh1_nbeats.sh`
- `scripts/smoke/smoke_etth1_transformer.sh`
- `scripts/smoke/smoke_etth1_dlinear.sh`
- `scripts/ETTh1_script/iTransformer/iTransformer_M.sh`
- `scripts/smoke/smoke_engineering_checks.sh`

历史风场业务脚本已归档到 [docs/archive_wind_forecast](/Users/wangzf/projects/tsproj_dl/docs/archive_wind_forecast)，不再作为当前主线可运行脚本维护。

## RNN 系列续修记录

当前 RNN 系列已完成 `pred_method` 基础语义收敛：`recursive_multi_step` 的 `train/valid` 使用单步监督，`test/forecast` 递归生成完整 horizon；`direct_multi_output` 作为现有多步直接输出实现的正式名称；RNN 脚本默认使用 `direct_multi_output`。

下次继续修复时，优先处理以下未完成项：

- 严格版 `direct_multi_step` 尚未实现。目前仍是“单模型多 horizon head”的过渡实现，未拆成每个 horizon 一个独立模型、独立 checkpoint 和独立评估流程。
- `direct_recursive_multi_step_mix` 仍是实验性策略。目前只在预测阶段支持前序预测回填，尚未形成明确的训练样本构造、损失设计和可复现实验脚本。
- `seq2seq_multi_step` 只保留接口和参数校验，训练、验证、测试、预测流程尚未实现。
- 递归预测暂未引入未来已知协变量接口。`MS` 模式默认只回填目标列，其他特征沿用窗口最后一行；需要未来外生变量时，应补充专门的数据输入与对齐逻辑。
- `recursive_multi_step` 目前复用现有多步输出模型结构，并只训练/使用第一个 horizon head；后续可以为递归策略增加真正的一步输出 head，避免无用输出参数。
- RNN 策略还缺少完整业务脚本覆盖。当前 LSTM/GRU smoke 可用于基础回归，但四类 `pred_method`、`S/MS/M` 组合还没有形成系统脚本矩阵。
- RNN 相关测试文件位于 `tests/`，但当前 `.gitignore` 忽略了 `tests/`。如果希望测试随仓库版本化，需要先调整忽略规则，再补充并提交策略级测试。

## 文档说明

- `README.md`：项目使用说明、环境准备、运行方式、目录简介
- `AGENTS.md`：项目治理规则、分支策略、主线定义、模块状态、开发看板
- `docs/`：时间序列研究资料、图示与补充文档

## 项目优化

项目治理、模块状态、里程碑和后续优化路线不再在本文件维护，请统一查看根目录 [AGENTS.md](/Users/wangzf/projects/tsproj_dl/AGENTS.md)。
