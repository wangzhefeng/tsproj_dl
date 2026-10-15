# tsproj_dl

`tsproj_dl` 是一个面向时间序列预测任务的神经网络快速测试、训练、验证与推理项目。当前仓库以 PyTorch 为主，包含 Transformer、MLP、RNN、CNN、GNN 等多类模型实现，以及配套的数据处理、实验流程和脚本模板。

项目治理、分支流转、模块状态和开发看板统一维护在 [AGENTS.md](AGENTS.md)。本文件只负责项目使用说明。

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
bash scripts/ETTh1_script/Transformer.sh
```

示例 2：ETTh1 上运行已经补齐并验证过的 MLP 系列脚本

```bash
bash scripts/ETTh1_script/TSMixer.sh
bash scripts/ETTh1_script/DLinear.sh
```

示例 3：工程检查（全仓单元回归与静态 utils 引用审计，不启动长训练）

```bash
bash scripts/smoke/smoke_engineering_checks.sh
```

示例 4：直接调用训练入口

```bash
env -u PYTHONPATH .venv/bin/python run.py \
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

<a id="gslaformer-usage"></a>

## GSLAformer：合成数据使用流程（在建）

GSLAformer 使用独立的 JSON/NPZ 数据流与 CLI，不混用普通模型的 `--is_training` 等参数。核心链路已实现，但 **AIDC 完整矩阵尚未验收，不代表生产可用**。设计与边界见 [文档目录](docs/gslaformer/v1/README.md)，当前进度与实跑证据唯一见 [实施规格 §12](docs/gslaformer/v1/implementation.md#work-packages)。

当前模型计算／checkpoint 版本为 **2**，数据 schema 仍为 **3**。旧版本权重不能继续训练或直接用于 test/forecast；请用新 run_id 重新训练，不覆盖旧实验，也不要把历史结果当作新版本结论。

以下命令均从仓库根目录执行，使用同一根 `.venv`。run_id 必须唯一，不删除旧产物来重跑；示例名称首次使用后需换新名称。

### 1. 生成数据（已有 fixtures 时跳过）

```bash
env -u PYTHONPATH .venv/bin/python scripts/dev/generate_gslaformer_fixture.py \
  --output results/gslaformer/fixtures --seed 2023 --suite all
```

生成 base、M1–M5、AIDC 标准包和无标签请求，并执行生成器自检。目标目录必须尚不存在。使用其他输出目录时，需同步修改完整配置中的 `data.package`；该路径相对配置文件解析，`results_root` 则相对仓库根目录解析。

### 2. 训练与续训

```bash
# base 完整 A/B 训练，不是 AIDC 长周期矩阵
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer \
  --gsla_config scripts/smoke/gslaformer_functional.json \
  --gsla_mode train --gsla_run_id demo_train

# 仅用于非 completed 的同一 run；配置与数据必须和原训练一致
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer \
  --gsla_config scripts/smoke/gslaformer_functional.json \
  --gsla_mode train --gsla_run_id demo_train \
  --gsla_checkpoint results/gslaformer/demo_train/checkpoints/last.pt
```

第二条是中断后的替代操作，不应在第一条已完成后紧接执行。只支持 epoch 边界恢复；中途停止的未保存 epoch 会重跑。`best_A.pt` 是阶段 A 选优产物；正式 test/forecast 使用完整训练后的 `best.pt`，不能把 A 权重当最终模型。

### 3. 测试与无标签预测

```bash
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer \
  --gsla_config scripts/smoke/gslaformer_functional.json \
  --gsla_mode test --gsla_run_id demo_test \
  --gsla_checkpoint results/gslaformer/demo_train/checkpoints/best.pt

env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer \
  --gsla_config scripts/smoke/gslaformer_functional.json \
  --gsla_mode forecast --gsla_run_id demo_cold \
  --gsla_checkpoint results/gslaformer/demo_train/checkpoints/best.pt \
  --gsla_request results/gslaformer/fixtures/requests/new_entity_cold
```

forecast 请求不含未来真值，复用 checkpoint 中的权重、scaler 和词表，不重新拟合。只加载可信的本地 checkpoint；请求的 schema、L/H、model 和 variant 须兼容。

### 4. 比较与一键流程

```bash
# 当前 base 配置的规定对照：会新建训练任务，不是只读取旧结果
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer \
  --gsla_config scripts/smoke/gslaformer_functional.json \
  --gsla_mode compare --gsla_run_id demo_compare

# 完整 base train/test + 四种请求；与上面分步命令任选一种
bash scripts/smoke/smoke_gslaformer.sh demo_smoke
```

机制任务对照命令见 [smoke 使用说明](scripts/smoke/README.md)。AIDC 的 short/day/multiday 配置在同目录，分别预测 6/24/72 小时；它们不是快速 smoke，运行或续训前须确认资源预算。本流程不自动启动 AIDC 全矩阵。

GSLAformer 输出位于 `results/gslaformer/<run_id>/`：训练保存配置、统计、阶段 checkpoint 和诊断；test 保存预测、真值与指标；forecast 保存无真值预测；compare 保存汇总和子 run 路径。`status.json` 的 completed 只代表该 run 完成，不等于整体模型验收完成。

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

- `scripts/ETTh1_script/Transformer.sh`
- `scripts/ETTh1_script/DLinear.sh`
- `scripts/ETTh1_script/TSMixer.sh`
- `scripts/smoke/smoke_gslaformer.sh`
- `scripts/ETTh1_script/iTransformer/iTransformer_M.sh`
- `scripts/smoke/smoke_engineering_checks.sh`

历史风场业务脚本不属于当前维护主线；旧说明中的 `docs/archive_wind_forecast/` 当前并不存在，不将其作为可用入口。

## RNN 系列续修记录

当前 RNN 系列已完成 `pred_method` 基础语义收敛：`recursive_multi_step` 的 `train/valid` 使用单步监督，`test/forecast` 递归生成完整 horizon；`direct_multi_output` 作为现有多步直接输出实现的正式名称；RNN 脚本默认使用 `direct_multi_output`。

下次继续修复时，优先处理以下未完成项：

- 严格版 `direct_multi_step` 尚未实现。目前仍是“单模型多 horizon head”的过渡实现，未拆成每个 horizon 一个独立模型、独立 checkpoint 和独立评估流程。
- `direct_recursive_multi_step_mix` 仍是实验性策略。目前只在预测阶段支持前序预测回填，尚未形成明确的训练样本构造、损失设计和可复现实验脚本。
- `seq2seq_multi_step` 只保留接口和参数校验，训练、验证、测试、预测流程尚未实现。
- 递归预测暂未引入未来已知协变量接口。`MS` 模式默认只回填目标列，其他特征沿用窗口最后一行；需要未来外生变量时，应补充专门的数据输入与对齐逻辑。
- `recursive_multi_step` 目前复用现有多步输出模型结构，并只训练/使用第一个 horizon head；后续可以为递归策略增加真正的一步输出 head，避免无用输出参数。
- RNN 策略还缺少完整业务脚本覆盖。当前 LSTM/GRU smoke 可用于基础回归，但四类 `pred_method`、`S/MS/M` 组合还没有形成系统脚本矩阵。
- RNN 相关测试位于 `tests/`，该目录已纳入版本控制；新增测试随对应实现审核和提交。

## 文档说明

- `README.md`：项目使用说明、环境准备、运行方式、目录简介
- `AGENTS.md`：项目治理规则、分支策略、主线定义、模块状态、开发看板
- `docs/`：时间序列研究资料、图示与补充文档

## 项目优化

项目治理、模块状态、里程碑和后续优化路线不再在本文件维护，请统一查看根目录 [AGENTS.md](AGENTS.md)。
