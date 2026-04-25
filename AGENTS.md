# AGENTS

本文件用于指导本仓库的 vibe coding、协作开发和持续跟踪项目进展。根目录 `README.md` 负责项目使用说明，本文件负责治理规则、当前主线、模块状态和开发看板。

## 1. 项目目标

当前仓库的目标是建设一个面向时间序列预测任务的神经网络快速测试、训练、验证与推理框架，并逐步将其从研究型代码仓库收敛为可持续维护的工程仓库。

## 2. 分支策略

项目采用四级分支流转：

- `dev`：日常开发与试验分支，本地默认工作分支
- `stable`：通过基本验证的稳定版本，只从 `dev` 合并进入
- `beta`：对外发布候选版本，只从 `stable` 合并进入
- `main`：最终公开成熟版本，只从 `beta` 合并进入

协作规则：

- 默认在 `dev` 分支进行开发、验证和文档更新
- 任一功能开发完成后，先更新本文件中的状态，再提交代码
- 不直接将日常开发内容从 `dev` 合并到 `main`
- 任一分支合并前，至少确认目标内容与本文件中的模块状态一致

## 3. 当前主线

当前唯一维护主线为 Transformer 长期预测链路：

`run_dl.py -> exp/exp_long_term_forecasting.py -> data_provider/TFs_type -> exp/exp_basic.py`

说明：

- `run_dl.py` 是当前推荐命令行入口
- `exp/exp_long_term_forecasting.py` 是当前推荐训练主流程
- `data_provider/TFs_type` 是当前推荐数据处理主线
- `exp/exp_basic.py` 负责维护主线模型注册与设备初始化

当前主线维护原则：

- 新模型只有在完成实现、接入 `exp/exp_basic.py`、具备可运行脚本并通过至少一次 smoke 验证后，才能视为可用主线成员
- 旧实验分支、在建分支和占位脚本不能默认视为当前主线

## 4. 目录职责

- `run_dl.py`：统一训练、测试、预测入口
- `exp/`：实验主流程
  - `exp_basic.py`：基础实验接口与模型注册
  - `exp_long_term_forecasting.py`：当前主训练主线
  - `exp_short_term_forecasting.py`：短期预测实验分支
  - `exp_forecasting_rnns.py`：旧版或分支化的深度学习实验流程
  - `exp/dl_todo/`：在建实验流程
- `data_provider/`：数据处理流程
  - `TFs_type/`：Transformer 类模型当前主线
  - `RNNs_type/`：RNN 类模型数据流
- `models/`：模型实现，按 `transformer`、`mlp`、`rnn`、`cnn`、`gnn`、`others` 分类
- `layers/`：模型公用模块与网络层
- `scripts/`：实验脚本与业务场景脚本
- `docs/`：研究资料、图示、补充文档
- `results/`：统一结果输出目录
- `utils/`：当前项目内受版本控制的运行时工具模块，后续继续按"最小必需依赖"原则收敛

## 5. 当前状态

主线链路（`run_dl.py → exp_long_term_forecasting.py → TFs_type → exp_basic.py`）已全面打通，DLinear / Transformer / TSMixer / N_HiTs / N_BEATS 五条 ETTh1 smoke 已完成，工程依赖、输出目录、字体配置均已收敛。

**资产状态：**

- 保留脚本目录：`scripts/ETTh1_script/`、`scripts/ETTm1_script/`、`scripts/Solar_script/`、`scripts/Traffic_script/`、`scripts/Weather_script/`、`scripts/dev/`、`scripts/smoke/`
- 主线 smoke 结果：`results/smoke_core004/`（ETTh1+DLinear）、`results/smoke_core005/`（ETTh1+Transformer）
- 测试覆盖：`tests/` 已覆盖工程资产、主线导入、模型前向 smoke
- 已接入主线并通过 ETTh1 smoke 的模型：`DLinear`、`Transformer`、`TSMixer`、`N_HiTs`、`N_BEATS`、`LSTMTransformer`
- 重依赖开源模型（Chronos、Moirai、TimesFM 等）已收录源码，当前不纳入本地回归测试

## 6. 模块状态看板

### 可用骨架

- 主训练骨架：`run_dl.py`、`exp/exp_long_term_forecasting.py`、`exp/exp_basic.py`
- Transformer 数据主线：`data_provider/TFs_type/*`
- 当前工程辅助：`scripts/smoke/*`、`scripts/dev/audit_runtime_utils_imports.py`、`tests/`
- 当前保留的数据集主目录：`dataset/ETT-small/`、`dataset/weather/`、`dataset/traffic/`、`dataset/illness/`、`dataset/m4/`
- 已完成 ETTh1 smoke 的模型：`DLinear`、`Transformer`、`TSMixer`、`N_HiTs`、`N_BEATS`

### 在建区

- `exp/exp_forecasting_rnns.py`、`exp/exp_short_term_forecasting.py`（待决定是否继续维护）
- `exp/dl_todo/*`
- `scripts/ETTm1_script/*`
- 开源新收录但尚未接入主入口的非 Transformer 模型

### 暂缓区

- 尚未接入主线的 `models/cnn/*`、`models/gnn/*`、多数 `models/rnn/*`
- 已迁移归档的 `docs/archive_wind_forecast/*`
- 与当前主线稳定化无直接关系的历史研究资料与扩展实验

## 7. 开发阶段计划

> 阶段 A–D 已全部完成（详见历史 git 记录）。当前进入阶段 E–F。

### 阶段 E：质量扩展

- 在工程测试之外，补主线导入测试、参数解析测试、最小数据集 smoke 测试
- 将 `scripts/dev/audit_runtime_utils_imports.py` 扩展为主线依赖审计工具，而不只是 utils 统计脚本
- 逐步建立"环境可装、入口可导入、主链可 smoke"三层验收标准

### 阶段 F：开源模型接入收敛

- 基于已同步的上游源码，优先接入非 Transformer 模型到当前主入口
- 在 `ETTh1_script/` 与 `ETTm1_script/` 中为优先模型补齐测试脚本
- 将 `_tsl` 版本作为对照实现逐步评估，而不是一次性全部激活

## 8. 进行中 / 待办

任务状态统一使用：`Todo / Doing / Blocked / Done`

| ID | Status | Owner | Targets | Acceptance |
| --- | --- | --- | --- | --- |
| EXP-001 | Doing | TBD | `exp/exp_forecasting_rnns.py`, `exp/exp_short_term_forecasting.py` | 已完成主要工具接口对齐；下一步需明确旧实验分支是继续维护还是转入暂缓区 |
| EXP-002 | Done | Codex | `exp/exp_basic.py`, `exp/exp_long_term_forecasting.py`, `data_provider/TFs_type/data_factory.py`, `tests/` | `PatchTST.sh` 已完成 ETTh1 train/valid/test smoke；相关 unittest 通过 |
| EXP-003 | Done | Codex | `data_provider/TFs_type/data_factory.py`, `tests/test_tfs_data_provider.py` | 数据集 flag 仅使用 `valid`，不再兼容 `val` |
| EXP-004 | Done | Codex | `data_provider/TFs_type/data_loader.py`, `exp/exp_long_term_forecasting.py`, `tests/` | PatchTST 特征全通道与目标单通道转换策略均有测试覆盖；train/valid/test/forecast smoke 通过 |
| EXP-005 | Done | Codex | `exp/exp_long_term_forecasting.py`, `tests/test_transformer_family.py` | MPS 训练时 valid loss 不再把 device tensor 交给 NumPy；PatchTST training-only smoke 通过 |
| EXP-006 | Done | Codex | `data_provider/TFs_type/data_loader.py`, `exp/exp_long_term_forecasting.py`, `scripts/ETTh1_script/PatchTST_*.sh` | 训练 scaler 随 checkpoint 保存，forecast 优先复用训练 scaler；PatchTST M/MS/S 脚本参数与策略一致；三类脚本 smoke 均通过 |
| EXP-007 | Done | Codex | `run_dl.py`, `data_provider/TFs_type/*`, `exp/exp_long_term_forecasting.py`, `scripts/ETTh1_script/PatchTST/PatchTST_S.sh` | PatchTST_S 的 train/valid 数据步长、验证 shuffle、标准化和学习率调度策略完成优化并通过 smoke |
| EXP-008 | Done | Codex | `exp/exp_long_term_forecasting.py`, `data_provider/TFs_type/data_loader.py`, `utils/metrics_dl.py` | test 时间轴缝合支持 testing_step，test 复用训练 scaler，MAPE/MSPE 避免 inf |
| EXP-009 | Done | Codex | `exp/exp_long_term_forecasting.py`, `data_provider/TFs_type/data_loader.py`, `scripts/ETTh1_script/PatchTST/PatchTST_S_forecast.sh` | forecast 作为生产离线推理入口：强制 checkpoint/scaler 校验，输出元数据，提供 forecast-only 脚本 |
| EXP-010 | Done | Codex | `scripts/ETTh1_script/PatchTST/` | PatchTST S/MS/M 训练测试预测脚本与 forecast-only 脚本统一归档到专用目录 |
| EXP-011 | Done | Codex | `scripts/ETTh1_script/iTransformer/` | iTransformer S/MS/M 训练测试预测脚本与 forecast-only 脚本统一归档到专用目录；脚本语法检查通过 |
| EXP-012 | Doing | Codex | `run_dl.py`, `exp/exp_short_term_forecasting.py`, `scripts/M4_short_term/`, `tests/` | M4 iTransformer 6 个短期预测脚本恢复可运行；`short_term_forecast` 走短期实验类；首个脚本完成 train/valid/test smoke |

任务更新规则：

- 新任务进入开发前，必须先登记到本表
- 任务状态变化时，优先更新本文件，再提交实现代码
- 每个任务至少写清目标文件和验收标准
- 若任务被判定为删除而非实现，也必须在本表中明确记录原因

## 9. 协作约束

- `README.md` 只维护项目使用说明，不再维护长期治理内容
- `AGENTS.md` 是项目治理主文档，任何重大结构调整、模块状态变化、开发阶段变更都需要更新本文件
- 当前阶段优先级以"主线可运行"高于"新增模型"
- 任何新模型开发必须同步登记三类状态：
  - 模型实现状态
  - 主入口接入状态
  - 验证状态
- 任何新数据流开发必须说明其归属：
  - 主线
  - 在建
  - 临时实验
- 未在 `AGENTS.md` 中登记的任务，不应直接并入 `stable`、`beta` 或 `main`
- 新的工程化辅助代码优先放在受版本控制的项目目录中，不继续引入新的外部嵌套仓库
- 当"目录存在但不可运行"时，优先做状态澄清和范围收敛，而不是继续扩散脚本与实现数量
