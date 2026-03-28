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
  - `exp_forecasting_dl.py`：旧版或分支化的深度学习实验流程
  - `exp/dl_todo/`：在建实验流程
- `data_provider/`：数据处理流程
  - `TFs_type/`：Transformer 类模型当前主线
  - `RNNs_type/`：RNN 类模型数据流
- `models/`：模型实现，按 `transformer`、`mlp`、`rnn`、`cnn`、`gnn`、`others` 分类
- `layers/`：模型公用模块与网络层
- `scripts/`：实验脚本与业务场景脚本
- `docs/`：研究资料、图示、补充文档
- `results/`：统一结果输出目录
- `utils/`：当前项目内受版本控制的运行时工具模块，后续继续按“最小必需依赖”原则收敛

## 5. 当前诊断

### 总体状态

- 项目已经完成一轮结构收敛：`results/` 统一输出目录已落地，`utils/` 已并入当前仓库，`.github/` 和 `scripts/work/` 已清理。
- 基于 `/Users/wangzf/projects/tsproj_opensource/Time-Series-Library` 的模型与层源码同步已落地：当前仓库的 `models/` 与 `layers/` 已完整收录上游源码。
- 当前仓库属于“工程骨架已收紧，DLinear / Transformer / TSMixer / N_HiTs / N_BEATS 五条 ETTh1 主线 smoke 已打通，旧脚本与旧分支边界已基本收敛”的阶段。
- `run_dl.py -> exp/exp_long_term_forecasting.py -> data_provider/TFs_type -> exp/exp_basic.py` 这条主链依然是唯一推荐维护主线。

### 已确认问题

- `import run_dl` 已在本地 `.venv` 中通过，当前 `.venv` 也已完成依赖同步；`einops` 与 `reformer-pytorch` 已可用。
- `exp/exp_long_term_forecasting.py` 的 `EarlyStopping` 调用已与 `utils/model_tools.py` 对齐，`ETTh1 + DLinear` 与 `ETTh1 + Transformer` 训练闭环都已成功产出 checkpoint 与 loss 图。
- Transformer 家族已完成第一轮收敛：`Transformer.py` 现为唯一官方主线实现，已吸收 `embed_type` 与 `RevIN` 增强能力；`Transformer_v2.py` 与 `Transformer_v3.py` 已删除，统一直接使用 `Transformer` 与 `LSTMTransformer`。
- 测试已经覆盖主线导入 smoke 与新增模型前向 smoke，训练闭环则通过 `scripts/smoke/*` 和 ETTh1 实测脚本完成验收。
- `data_provider/todo/*` 的旧 Python 数据流已删除，当前保留的数据流入口已收敛到 `TFs_type/` 与 `RNNs_type/`。
- Matplotlib 字体告警的主因已定位并修正为按环境自动回退字体；`MPLCONFIGDIR` 已下沉到 smoke 脚本和保留实验脚本，环境噪音已显著收敛。
- `scripts/ETTh1_script/` 已完成收敛：`Transformer`、`DLinear`、`TSMixer`、`N_HiTs`、`N_BEATS` 已有真实 smoke 佐证，`GRU` / `RNN` / `LSTM2LSTM` 保留脚本已按当前入口参数修正。

### 资产状态

- 保留脚本目录当前为：`scripts/ETTh1_script/`、`scripts/ETTm1_script/`、`scripts/Solar_script/`、`scripts/Traffic_script/`、`scripts/Weather_script/`、`scripts/dev/`、`scripts/smoke/`
- `scripts/work/` 已确认全部失效并删除；原 `scripts/wind_forecast/` 已迁移到 `docs/archive_wind_forecast/`，不再作为当前可运行脚本集合维护。
- 当前本地工程检查通过：`tests/` 已覆盖工程资产、主线导入与模型前向 smoke，`scripts/smoke/smoke_engineering_checks.sh` 可正常通过；`scripts/dev/audit_runtime_utils_imports.py` 当前统计为 39 个文件依赖 19 个 `utils` 模块。
- 主线 smoke 已落地：`ETTh1 + DLinear` 在 `results/smoke_core004/` 下、`ETTh1 + Transformer` 在 `results/smoke_core005/` 下都已成功生成 checkpoint 与 `loss_plot.pdf`。
- 已沉淀可复用 smoke 脚本：`scripts/smoke/smoke_etth1_dlinear.sh`、`scripts/smoke/smoke_etth1_transformer.sh`。
- `models/mlp/TSMixer.py`、`models/mlp/N_HiTs.py`、`models/mlp/N_BEATS.py` 已完成接入并通过 ETTh1 最小训练验证。
- `models/transformer/LSTMTransformer.py` 已建立为独立 hybrid 基线，并补充 ETTh1 入口脚本。
- 开源重名模型与层已按 `_tsl` 后缀并入同类目录；新增模型与层已按当前仓库分类规则分别落入 `transformer/mlp/rnn/cnn/others` 与 `layers/`。
- 重依赖开源模型如 `Chronos`、`Chronos2`、`Moirai`、`TimesFM`、`TimeMoE` 已收录源码，但当前仍不纳入本地回归测试集合。

## 6. 模块状态看板

### 可用骨架

- 主训练骨架：`run_dl.py`、`exp/exp_long_term_forecasting.py`、`exp/exp_basic.py`
- Transformer 数据主线：`data_provider/TFs_type/*`
- 当前工程辅助：`scripts/smoke/*`、`scripts/dev/audit_runtime_utils_imports.py`、`tests/test_engineering_artifacts.py`、`tests/test_run_dl_import.py`、`tests/test_model_forward_smoke.py`
- 当前保留的数据集主目录：`dataset/ETT-small/`、`dataset/weather/`、`dataset/traffic/`、`dataset/illness/`、`dataset/m4/`
- 已完成 ETTh1 smoke 的模型：`DLinear`、`Transformer`、`TSMixer`、`N_HiTs`、`N_BEATS`

### 在建区

- 开源新收录但尚未接入主入口的非 Transformer 模型
- `exp/dl_todo/*`
- `exp/exp_forecasting_dl.py`
- `exp/exp_short_term_forecasting.py`
- `scripts/ETTm1_script/*`

### 暂缓区

- 尚未接入主线的 `models/cnn/*`
- 尚未接入主线的 `models/gnn/*`
- 尚未接入主线的多数 `models/rnn/*`
- 已迁移归档的 `docs/archive_wind_forecast/*`
- 与当前主线稳定化无直接关系的历史研究资料与扩展实验

## 7. 开发阶段计划

### 阶段 A：主线解阻与可运行化

- 固化已补齐的主线运行依赖，确保新环境能稳定完成 `import run_dl`
- 将已验证通过的 `ETTh1 + DLinear` smoke 沉淀为可复用脚本或测试入口
- 将已验证通过的 `ETTh1 + Transformer` smoke 沉淀为可复用脚本或测试入口
- 验证主链的 checkpoint、日志、结果输出目录在训练闭环下行为持续正常

### 阶段 B：脚本与数据资产收敛

- 审核当前保留的 `scripts/ETTh1_script/`、`scripts/ETTm1_script/`、`scripts/Solar_script/`、`scripts/Traffic_script/`、`scripts/Weather_script/`
- 删除继续失效的脚本，或将仍有价值的脚本改到当前入口和现有数据集
- 让 README 中列出的示例脚本都满足“入口存在 + 数据存在 + 参数兼容”

### 阶段 C：数据流与旧实验分支收敛

- 保持 `TFs_type/` 与 `RNNs_type/` 作为有效数据流入口
- 清理已下线 todo 数据流遗留引用，并决定旧实验分支是否继续依赖历史接口
- 判断 `exp/exp_forecasting_dl.py` 与 `exp/exp_short_term_forecasting.py` 是否继续维护

### 阶段 D：模型补齐与接入

- 优先处理 `TSMixer`、`N_HiTs`、`N_BEATS`
- 每个模型都按“实现完成 -> 接入 `exp/exp_basic.py` -> 增加脚本 -> smoke 验证”推进
- 对未纳入主线的模型保持目录存在，但不默认承诺可运行

### 阶段 E：质量扩展

- 在工程测试之外，补主线导入测试、参数解析测试、最小数据集 smoke 测试
- 将 `scripts/dev/audit_runtime_utils_imports.py` 扩展为主线依赖审计工具，而不只是 utils 统计脚本
- 逐步建立“环境可装、入口可导入、主链可 smoke”三层验收标准

### 阶段 F：开源模型接入收敛

- 基于已同步的上游源码，优先接入非 Transformer 模型到当前主入口
- 在 `ETTh1_script/` 与 `ETTm1_script/` 中为优先模型补齐测试脚本
- 将 `_tsl` 版本作为对照实现逐步评估，而不是一次性全部激活

## 8. 进行中 / 待办

任务状态统一使用：`Todo / Doing / Blocked / Done`

| ID | Status | Owner | Targets | Acceptance |
| --- | --- | --- | --- | --- |
| GOV-001 | Done | TBD | `README.md`, `AGENTS.md` | 根目录文档职责完成拆分；`README.md` 只保留使用说明；`AGENTS.md` 维护治理规则与开发看板 |
| GOV-002 | Done | TBD | `run_dl.py`, `README.md`, `results/` | 输出目录已统一为 `results/`，CLI 与文档同步完成 |
| GOV-003 | Done | TBD | `utils/`, `scripts/dev/audit_runtime_utils_imports.py`, `.gitignore` | 主线依赖的 `utils` 已纳入当前仓库，并保留工程审计脚本 |
| GOV-004 | Done | TBD | `.github/`, `tests/*`, `scripts/smoke/*` | 已移除 GitHub Actions 依赖，保留本地工程检查链 |
| GOV-005 | Done | TBD | `scripts/work/`, `README.md`, `AGENTS.md` | 已删除全部失效 `scripts/work` 脚本并清理相关引用 |
| CORE-001 | Done | TBD | `pyproject.toml`, 主线导入链 | 本地 `.venv` 下 `import run_dl` 已通过；主线直接依赖已补充 `einops`、`reformer-pytorch`；模型导入改为按需加载 |
| CORE-002 | Done | TBD | `exp/exp_long_term_forecasting.py`, `utils/model_tools.py`, `exp/exp_forecasting_dl.py`, `exp/exp_short_term_forecasting.py` | `EarlyStopping`、学习率调度调用已与工具接口对齐，主线及保留旧分支不再存在明显参数错位 |
| CORE-003 | Done | TBD | `tests/*`, `scripts/smoke/*`, `run_dl.py` | 已新增 `tests/test_run_dl_import.py`，本地 unittest 与 smoke 检查不再只有工程资产测试 |
| CORE-004 | Done | TBD | `scripts/ETTh1_script/*`, `run_dl.py`, `exp/exp_long_term_forecasting.py`, `utils/model_memory.py` | 已以 `ETTh1 + DLinear` 跑通最小训练 smoke，并在 `results/smoke_core004/` 下生成 checkpoint 与 `loss_plot.pdf`；同时修复了通用模型的内存统计兼容性问题 |
| CORE-005 | Done | TBD | `scripts/ETTh1_script/ETTh1_transformer.sh`, `run_dl.py`, `exp/exp_long_term_forecasting.py` | 已以 `ETTh1 + Transformer` 跑通最小训练 smoke，并在 `results/smoke_core005/` 下生成 checkpoint 与 `loss_plot.pdf`；同时完成当前 `.venv` 依赖同步 |
| SCRIPT-000 | Done | TBD | `scripts/ETTh1_script/*`, `scripts/smoke/*`, `README.md` | 已将验证通过的 DLinear / Transformer 命令沉淀为 `scripts/smoke/smoke_etth1_*.sh`，并同步到 README 示例 |
| SCRIPT-001 | Done | TBD | `scripts/ETTh1_script/*`, `scripts/ETTm1_script/*`, `scripts/Solar_script/*`, `scripts/Traffic_script/*`, `scripts/Weather_script/*`, `docs/archive_wind_forecast/*` | 当前保留脚本目录已收敛；`ETTh1_gru.sh`、`ETTh1_rnn.sh`、`ETTh1_seq2seq_lstm.sh` 已按当前入口修正；旧 `wind_forecast` 脚本已整体迁移归档 |
| DATA-001 | Done | TBD | `data_provider/todo/*`, `data_provider/TFs_type/*`, `data_provider/RNNs_type/*` | 旧 todo Python 数据流已删除，当前数据入口边界已明确为 `TFs_type/` 与 `RNNs_type/` |
| EXP-001 | Doing | TBD | `exp/exp_forecasting_dl.py`, `exp/exp_short_term_forecasting.py` | 已完成主要工具接口对齐；下一步需明确旧实验分支是继续维护还是转入暂缓区 |
| MODEL-001 | Done | TBD | `models/mlp/TSMixer.py`, `exp/exp_basic.py`, `scripts/ETTh1_script/ETTh1_tsmixer.sh` | `TSMixer` 已完成实现、接入主入口、补脚本并通过 ETTh1 smoke 验证 |
| MODEL-002 | Done | TBD | `models/mlp/N_HiTs.py`, `exp/exp_basic.py`, `scripts/ETTh1_script/ETTh1_nhits.sh` | `N_HiTs` 已完成实现、接入主入口、补脚本并通过 ETTh1 smoke 验证 |
| MODEL-003 | Done | TBD | `models/mlp/N_BEATS.py`, `exp/exp_basic.py`, `scripts/ETTh1_script/ETTh1_nbeats.sh` | `N_BEATS` 已完成主入口接入与 ETTh1 最小验证 |
| ARCH-TR-001 | Done | TBD | `models/transformer/Transformer.py`, `models/transformer/LSTMTransformer.py`, `exp/exp_basic.py` | `Transformer` 已收敛为唯一官方主线；`Transformer_v2.py` 与 `Transformer_v3.py` 已删除，统一直接使用 `Transformer` 与 `LSTMTransformer` |
| OSS-001 | Done | TBD | `models/*`, `layers/*` | 已将 Time-Series-Library 的模型与层源码完整收录到当前仓库，并保持现有分类目录 |
| OSS-002 | Done | TBD | `models/*/*_tsl.py`, `layers/*_tsl.py` | 开源重名模型与层已使用 `_tsl` 后缀并入，避免覆盖当前主线实现 |
| OSS-003 | Done | TBD | `models/others/*`, `AGENTS.md` | 重依赖开源模型源码已收录，并明确标记为当前不纳入本地回归 |
| QA-001 | Done | TBD | `tests/*`, `scripts/smoke/*` | 测试已覆盖工程资产、主线导入与模型前向 smoke；DLinear / Transformer / TSMixer / N_HiTs / N_BEATS 已完成脚本级最小训练验收 |
| ENV-001 | Done | TBD | `utils/plot_results.py`, `scripts/smoke/*`, `scripts/ETTh1_script/*`, `scripts/ETTm1_script/*` | `SimHei` 已改为自动回退，`MPLCONFIGDIR` 已收敛到 smoke 与保留实验脚本，训练与测试环境噪音已显著降低 |

任务更新规则：

- 新任务进入开发前，必须先登记到本表
- 任务状态变化时，优先更新本文件，再提交实现代码
- 每个任务至少写清目标文件和验收标准
- 若任务被判定为删除而非实现，也必须在本表中明确记录原因

## 9. 协作约束

- `README.md` 只维护项目使用说明，不再维护长期治理内容
- `AGENTS.md` 是项目治理主文档，任何重大结构调整、模块状态变化、开发阶段变更都需要更新本文件
- 当前阶段优先级以“主线可运行”高于“新增模型”
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
- 当“目录存在但不可运行”时，优先做状态澄清和范围收敛，而不是继续扩散脚本与实现数量
