# GSLAformer 评审与优化计划（v2：需求追踪重审版）

> 日期：2026-10-15（Asia/Shanghai）。本文取代 v1（备份在 box 的 `/workspace/gsla/review_and_optimization_plan.v1.md`，没有入库）。
> 方式：仓库只读，没有 git 操作。所有测试、smoke、对照和复现都在仓库外的副本中运行（用户机 `/tmp/gsla_v2/repo`，已排除 .git、results、arch、__pycache__），运行完已经删除。仓库里只改写了本文件。行号以 2026-10-15 的工作区为准（新增实现仍未提交）。
> 严重度：**高**＝会让结论出错、无法复现，或让验收产物产不出来；**中**＝影响可信度、效率或覆盖真实性；**低**＝改进项。
> 每条结论都附证据，形式是 `文件:行`、测试输出或复现脚本结果（见 §5.0）。

---

## 1. 范围与已读清单

### 1.1 基线文档
- docs/gslaformer/ 下的 design.md、implementation.md(502)、input_output.md(387)、acceptance.md(660)、history.md(87)、README.md(44)，另有 arch/model-v4.0.png。
- 本轮按章节重新读过用来抽取需求的部分：
  - acceptance：§2–§6、§11、§14–§17、§19–§22；
  - implementation：§1–§4、§9–§12；
  - input_output：§6–§12；
  - design：§2 决策和 §8 评价原则。

### 1.2 代码（逐行读过）

| 类别 | 文件 |
|---|---|
| 模型 | `models/transformer/GSLAformer.py`；`layers/gslaformer/{entity_history,spatial,relations,temporal}.py` 及 README |
| 数据 | `data_provider/spatial_temporal/{dataset,scaling,batching,config,schema,aidc_mapping}.py` |
| 实验 | `exp/exp_gslaformer_forecasting.py`、`gslaformer_{support,inference,artifacts,comparison}.py` |
| 入口 | `run.py` |
| scripts/dev | 生成器（1712 行）、`fit_gslaformer_mechanisms.py`、`measure_*`、`profile_*`、`time_*` |
| scripts/smoke | `smoke_gslaformer.sh`、`compare_gslaformer.py` 及 4 份 JSON |
| 测试 | 13 个 `tests/test_gslaformer_*.py`，共 86 个用例，都读了用例正文 |

### 1.3 本轮实际执行（副本中）

| 项 | 命令 | 结果 |
|---|---|---|
| 单测 | `python -m unittest discover -s tests -p 'test_gslaformer_*.py'` | **Ran 86 tests in 33.9s, OK** |
| 夹具 | `generate_gslaformer_fixture.py --suite all` | EXIT=0，约 105 s |
| CPU smoke | `bash scripts/smoke/smoke_gslaformer.sh v2smoke` | EXIT=0，real 75.6 s |
| 机制对照 | `compare_gslaformer.py --tasks M5 M3 --run_prefix v2cmp` | EXIT=0；M5 passed；M3 passed；suite 状态为 **partial**（符合 §15） |
| 复现脚本 | r_a ~ r_f | 见 §5.0 |

**夹具窗口数（train/valid/test）**

| profile | train | valid | test |
|---|---|---|---|
| aidc short | 12976 | 960 | 960 |
| aidc day | 10768 | 240 | 240 |
| aidc multiday | 10000 | 80 | 80 |
| base | 163 | 59 | 59 |
| M1–M4（每个） | 96 | 32 | 32 |

**CPU smoke 明细**
- train → test → 4 个 forecast 请求（normal_forecast、new_entity_cold、new_entity_history、new_parent），6 个 run 的 status 全部是 completed。
- train 的 global_epoch=38，test rows=354，每个 forecast rows=6。
- test 的 metrics.json：micro_mae=0.193，micro_rmse=0.233。
- environment.json 只有 python、torch、numpy、device、resume_contract。

**M3 对照的 micro_mae（smoke 预算）**

| full | direct | no_latent_loss | total_only |
|---|---|---|---|
| 1.263 | 1.170 | 1.029 | 1.280 |

acceptance §11.2 明确"不要求 AR 优于 direct"，所以这组数字只作记录，不判为缺陷。

---

## 2. 总结

### 2.1 需求状态统计（共 48 条 REQ，见 §3）

| 已实现且有测试 | 已实现但无测试 | 部分实现 | 未实现 |
|---|---|---|---|
| **32** | **9** | **5** | **2** |

### 2.2 总体判断
工程契约非常扎实，以下几点都经过了复现确认：
- 没有数据泄漏（复现 b）；
- 训练和推理用的是同一个 rollout（复现 c）；
- 续训逐位一致；
- 86 个测试全部通过，smoke 全链路跑通。

主要风险在四个方面：可追溯性、B 阶段的代价、AIDC 合成数据里的"抄总表"捷径、指标与门禁的测试空白。

### 2.3 Top 5 问题

| # | 问题 | 严重度 | 证据 |
|---|---|---|---|
| 1 | **R1 产物不可追溯**：代码未提交，environment.json 和 checkpoint 里没有 git SHA 或源码指纹。acceptance §23（:660）要求记录"配置／数据／代码版本"。 | 高 | `exp_gslaformer_forecasting.py:166-168`；在 smoke 产物里 grep `git` 没有命中（复现 f） |
| 2 | **N1 AIDC 抄总表捷径（本轮实测确认）** | 高（v1 为中，本轮上调） | 复现 a，见下 |
| 3 | **T1/T3 B 阶段代价**：每个 B batch 都重算 teacher（相当于 H 次编码），样本也不缓存。history 记录的 A 阶段单 epoch 为 1833 s，H=72 的完整课程实际跑不完，验收矩阵（§20.1）因此难以产出。 | 高 | `exp:283`；`gslaformer_support.py:106-132`。复现 c 证明 teacher 在冻结权重下是确定的，可以缓存。 |
| 4 | **E2/REQ-24 指标没有数值单测且实现了两份**：峰值、爬坡、per-horizon 逻辑在 inference 和 comparison 中各写一份；全仓唯一的手算断言是 persistence 的 site_macro_mae。 | 中 | `gslaformer_inference.py:193-217`；`gslaformer_comparison.py:113-131` |
| 5 | **REQ-29 needs_diagnosis 门禁未实现，几处文档与产物不符**：acceptance:644 的请求路径不存在；:653 说 smoke 会生成 MPS 配置，但 smoke 并没有；implementation.md:281-282 的测试覆盖声明不实（N6）。 | 中 | 复现 e、f；`smoke_gslaformer.sh` 全文 |

Top 2（N1）的复现结果：
- site 实体的 `active_power_kw` 与 target 相关系数 **0.9992**，中位比值 1.000。
- tr0 的 `load_ratio` 与 target 相关系数 **1.0000**。
- tr0 的 `active_power_kw` 相关系数 0.9992，bd0 也是 0.9992。
- 训练站和留出站（10 号站的 dense 和 sparse）都是这样。

这意味着 AIDC 上 full、no_spatial、no_relation_all 的消融结论，无法与"只读边界电表"区分开。

---

## 3. 需求追踪矩阵

状态缩写：✅＝已实现且有测试；🟡＝已实现但无测试；🟠＝部分实现；❌＝未实现。测试文件名省略 `test_gslaformer_` 前缀。

### 3.1 数据与契约

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-01 | 历史观测和目标按 as-of 可见（io §6.2；impl §2） | `dataset.py:384`、`401`、`81-90` | package（迟到数据、不回填） | ✅ |
| REQ-02 | 已知未来只取 published_at≤origin；horizon 内拓扑变化拒绝（io §8、§12） | `dataset.py:291-309`、`333-347` | package `test_known_future_topology_change_is_rejected...` | ✅ |
| REQ-03 | scaler 只用 train，按唯一单元统计（io §6.4、§6.7） | `scaling.py:93-139` | scaling（valid 极端值）；复现 b | ✅（缺 site_time 留出站用例 E1） |
| REQ-04 | family 和站点不跨 split（io §6.5；acc §19.3） | `dataset.py:465-467`、`503-509` | package、data | ✅ |
| REQ-05 | 包完整性：SHA-256、越界、dtype、非有限值（io §6.6） | `schema.py:15-28`；`dataset.py:128-132` | package；复现 b 中篡改后触发 `SHA-256 不匹配` | ✅ |
| REQ-06 | 测点映射：单位白名单、唯一 primary、逐类型 DAG（io §3、§4） | `aidc_mapping.py:11-109` | mapping | ✅ |
| REQ-07 | 错误输入拒绝表（io §12） | dataset、schema、batching 等多处 | package、contract_repairs、integration | ✅（"MPS 不可用报错"一行未见用例） |

### 3.2 模型计算

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-08 | 属性驱动的实体 token，observed 和 applicable 分列（impl §1.1） | `GSLAformer.py`；`entity_history.py` | contract_repairs（-1 与缺失可区分） | ✅ |
| REQ-09 | present=False 时 GRU 重置（impl §1.1） | `entity_history.py:99-105` | model（GRU 重置） | ✅ |
| REQ-10 | 空间、关系、读出分离；direction_bias（impl §1.2） | `spatial.py`、`relations.py` | model、relation_vectorization | ✅（N10 断言偏弱） |
| REQ-11 | 静态条件用非线性投影（impl §1.3） | `spatial.py:86-93`（汇总是单个 Linear） | contract_repairs 锁定了现状 | 🟠 D01 |
| REQ-12 | 日历经 Linear 映射后与 token **相加**（impl §1.4） | `GSLAformer.py:280-294` 是拼接 | — | 🟠 D02 |
| REQ-13 | 因果时间编码（impl §1.4） | `temporal.py` | model（因果性） | ✅ |
| REQ-14 | 完整未来路径编码和双头（impl §1.5） | `temporal.py`、`GSLAformer.py` | model（mask=False 时计划被忽略） | ✅ |
| REQ-15 | 单一自由递推，训练与推理同路径（impl §2） | `exp:279`；`inference:126` | model（predict_step 与 rollout 一致）；复现 c | ✅ |

### 3.3 标签、损失与训练

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-16 | latent 标签和统计：只取末端 origin，q 加权（impl §3.1） | `gslaformer_support.py:106-132`、`236+` | latent_stats、training | ✅ |
| REQ-17 | reduction：horizon 内宏平均，latent 不重新归一化（impl §3.2） | support | training（手算） | ✅ |
| REQ-18 | A → Freeze → B 课程，只在 B-full 早停（impl §4.1） | `exp:260` | state_machine | ✅ |
| REQ-19 | 冻结链 no_grad；不用 AMP、KV cache、TBPTT（impl §4.2:186） | `GSLAformer.py:160-188` | training（冻结链） | ✅ |
| REQ-20 | epoch 边界逐位续训（impl §4.3） | `exp:361-389` | state_machine | ✅ |
| REQ-21 | 多站均衡采样（impl §4.4） | exp | state_machine（share_a 在 0.47–0.53） | ✅ |
| REQ-22 | 消融 variant 真正切断信息通路（acc §5） | GSLAformer variants | contract_repairs、model | ✅ |
| REQ-23 | persistence 和 seasonal_naive 基线（acc §5） | `gslaformer_comparison.py:45-66` | comparison（回退计数） | ✅（指标数值未测） |

### 3.4 指标、诊断与选优

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-24 | micro、per-horizon、峰值（最早 argmax）、爬坡指标（impl §10） | `inference:193-217`；`comparison:113-131` | 只有 site_artifacts 的 persistence MAE | 🟡 |
| REQ-25 | latent 指标：计数、加权 MSE、原始坐标 MSE、训练方差（impl §10） | `inference:217-231` | 只检查存在性 | 🟡 |
| REQ-26 | AIDC 可报告 NRMSE＝RMSE/测试均值（impl §10 末条；acc §20.2 写的是"可加"） | 无 | 在产物中 grep `nrmse` 没有命中（复现 f） | ❌（可选项，低） |
| REQ-27 | 按 site×role×**profile**×scenario 分层（acc §20.2） | `artifacts.py:132-151` 的 key 是 (site, role, scenario, horizon)，profile 只体现在顶层 pred_len | site_artifacts 只检查站点集合 | 🟠 |
| REQ-28 | AIDC 用 valid_site_macro_mse 选优，缺标签时拒绝（acc §20.2） | support、exp | training（对 batch 划分不变）、data | ✅ |
| REQ-29 | needs_diagnosis 门禁（acc §20.3，规格自身标为"候选，待确认"） | 全仓无匹配 | grep 无命中（复现 f） | ❌ |
| REQ-30 | 三种同权重诊断，ranking_eligible=False（acc §5） | `inference:233-246`；`support:402` | training（第 1 步一致） | ✅ |

### 3.5 推理输出与产物

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-31 | predictions.csv 字段和质量标记（io §10） | `inference:163-175`；`schema.py:44-75` | data（quality flags）、integration | ✅ |
| REQ-32 | metadata 精确字段：fingerprint、input_cutoff 等（io §10） | `artifacts.py:22`；`inference:185-192` | integration、validate_artifacts | ✅ |
| REQ-33 | 保留负值预测并计数，不裁剪（io §10） | `inference:175` | 未见专门断言 | 🟡 |
| REQ-34 | 原子写入、run_id 不可覆盖、状态分账（impl §11） | artifacts | integration（重复和越界的 run_id） | ✅ |
| REQ-35 | 产物记录配置、数据、**代码**版本（acc §23:660；impl §11） | 有配置和数据指纹，没有代码版本 | 复现 f | 🟠 |
| REQ-36 | checkpoint 自包含，脱离训练包也能推理（impl §11） | inference | integration | ✅ |
| REQ-37 | forecast 只用最后 origin，不带监督；同权重处理新实体请求（acc §3.1、§6） | dataset、inference | package；smoke 的 4 个请求 completed | ✅ |

### 3.6 验收流程与合成数据

| REQ | 需求（来源） | 实现位置 | 测试/证据 | 状态 |
|---|---|---|---|---|
| REQ-38 | 小集拟合阈值（acc §6） | `fit_gslaformer_mechanisms.py` | 只有脚本；N6 | 🟡 |
| REQ-39 | MPS smoke：A/B 各 1 epoch，batch=2，独立 run（acc §6、§22:653） | smoke 中没有；手工配置 `functional_mps_i5_config.json`（impl:480） | 无 | 🟠 |
| REQ-40 | CPU 全阶段 smoke（acc §6） | `smoke_gslaformer.sh` | 本轮 EXIT=0，6 个 run completed | ✅ |
| REQ-41 | M1–M5 生成和 oracle 核算（acc §8–§13） | 生成器 | 只有生成时的内部断言（N11） | 🟡 |
| REQ-42 | 必需运行矩阵（acc §14.1、§20.1） | `comparison.py:21-40` | comparison（base、M3）；本轮 M3 真实执行 | ✅ |
| REQ-43 | 只跑部分任务时 suite_result 不能是 passed（acc §15） | `compare_gslaformer.py` | 本轮实测为 partial；无单测 | 🟡 |
| REQ-44 | AIDC 三个 profile 的 L/H（acc §16） | 三份 AIDC JSON | 无一致性测试（N14） | 🟡 |
| REQ-45 | 单种子 2023 和可复现生成顺序（acc §5、§17） | `generator:1676-1677` | 无 | 🟡 |
| REQ-46 | 12 站 8/2/2 留出加时间切分（acc §19） | 配置和 dataset | data（站点角色互斥） | ✅ |
| REQ-47 | 注意力预算和节点上限（acc §21；impl §9.2） | batching | contract_repairs | ✅ |
| REQ-48 | 资源与耗时记录（acc §21；impl §10） | `exp:425-478` | 无数值断言 | 🟡 |

---

## 4. 偏差清单

### 4.1 有意变更（已在文档中登记，或属于规格决策）

| # | 内容 | 依据 |
|---|---|---|
| I-1 | 单种子 2023 | acc §5；`generator:1676-1677` |
| I-2 | 不做每站 scaler | io §9 |
| I-3 | 不用 KV cache、AMP、scheduled sampling、TBPTT | impl §4.2:186 |
| I-4 | 固定 AdamW lr=1e-3，不用调度器 | impl §4.1:173 |
| I-5 | 课程是 1 → min(3,H) → H | impl §4.1 |
| I-6 | AIDC 用 CPU、batch=2 | acc §21 |
| I-7 | 负值不裁剪 | io §10 |
| I-8 | NRMSE 只是可选诊断 | impl §10；acc §20.2 |

### 4.2 未登记偏差

| # | 内容 | 证据 | 严重度 |
|---|---|---|---|
| U-1（D01） | 静态链的汇总是单个 Linear，不是非线性投影 | `spatial.py:93` | 低 |
| U-2（D02） | 日历拼接进 token，不是相加；连 docstring 都和实现不符 | `temporal.py:3-5`；`GSLAformer.py:280-294` | 低 |
| U-3 | scenario_metrics 缺少 profile 维度 | `artifacts.py:141` | 低 |
| U-4 | needs_diagnosis 没有实现 | grep 无命中 | 中 |
| U-5 | 代码版本没有落盘 | `exp:166-168` | 高 |
| U-6 | acceptance:644 的 `--gsla_request results/gslaformer/fixtures/new_entity_cold` 路径不存在，实际是 `fixtures/requests/new_entity_cold` | 复现 e | 低 |
| U-7 | acceptance:653 说 smoke 从基准 JSON 生成 MPS 配置，但 smoke 只有 CPU 步骤 | `smoke_gslaformer.sh` 全文 | 中 |
| U-8（N6） | implementation.md:281-282 的测试覆盖声明不实（过拟合、MPS） | 测试正文 | 中 |
| U-9（N7、N8） | 断链 `model-v4.0.png`（README:27、history:74）；README:28 没有列 bot_version；layers README 缺 temporal.py | — | 低 |
| U-10 | impl §9.2（:383）的显存估算仍是"待实测"，没有回填 | — | 低 |

---

## 5. 分领域诊断

### 5.0 复现脚本与结果

脚本位于副本 `/tmp/gsla_v2/repro`，运行后已删除，核心逻辑如下。

| 编号 | 目的 | 做法 | 结果 | 结论 |
|---|---|---|---|---|
| r_a | N1、N3 捷径 | 对 aidc_00_dense、aidc_10_dense、aidc_10_sparse，逐个 实体×特征 计算与 target 的相关系数和中位比值，并找常量和完全重复的列 | 见下 | N1、N3 **确认**，留出站同样存在 |
| r_b | 泄漏反事实 | 复制 AIDC 包，对留出站（08–11）全时段、训练站 t≥864 的所有 values 和 target 做 ×7+1000 扰动，同步更新 SHA，再 `TrainingTransform.fit` | 首次未更新 SHA 时触发 `SHA-256 不匹配`；更新后 **fingerprint 相同：True**。train 窗口的 max(origin+H)=863<864 | **无泄漏**；完整性守卫有效 |
| r_c | 训练与推理一致性、teacher 可缓存 | 用 base 夹具构造模型 | 见下 | 递推一致；T1 缓存可行且不违反规格 |
| r_d | N10 断言强度 | `assertFalse(a and b is True and c)` 等价于"a、b、c 中有一个为假就通过" | 即使非法可达 allow[0,0,2]=True，只要 allow[0,1,2]=False 断言照样通过 | N10 **确认** |
| r_e | 文档路径 | ls 生成后的 fixtures | 只有 `requests/{normal_forecast,new_entity_cold,new_entity_history,new_parent}`，`fixtures/new_entity_cold` 不存在 | U-6 **确认** |
| r_f | 产物字段 | 在 smoke 和对照全部产物中 `rg -i "nrmse\|git\|needs_diagnosis"` | 0 命中 | R1、REQ-26、REQ-29 **确认** |

r_a 的结果：
- `space/site/active_power_kw`：corr 0.9992，比值 1.000。
- `distribution/tr0/load_ratio`：corr **1.0000**。
- `tr0/active_power_kw`：corr 0.9992，比值 0.96–0.97。
- `bd0/active_power_kw`：corr 0.9992，比值约 0.89。
- 常量列：room_humidity=0.5（site、bd、rm）、4 个 hvac 实体的 f1=1.0、bd1=0。
- it_power 的 `cpu_utilization` 与 `gpu_utilization` 完全相同。
- target_observed_rate=1.0。

r_c 的结果：
- rollout(K=1) 与 rollout(H)[:1] 相等：True。
- rollout(K=2) 与 rollout(H)[:2] 相等：True。
- eval 下重复运行确定：True。
- train 与 eval 模式输出最大差 0.071，来自 dropout=0.1，属于预期。
- 进入 B 阶段后，两次 `build_latent_labels` 的 z 和 q 逐位相同：True。

### 5.1 数据

**优点**
- as-of 可见性正确（REQ-01、02）。
- split 和站点隔离正确（REQ-04）。
- 完整性守卫有效（r_b）。

| # | 位置 | 问题与理由 | 严重度 | 修复 |
|---|---|---|---|---|
| D1 | `dataset.py:399`、`403-406` | 非法窗口要到 `__getitem__` 才抛错。训练会中途崩溃，而且发生在 shuffle 之后，难以复现。 | 中 | 构建 windows 时预检、剔除非法窗口，并把剔除数写进 data_manifest |
| D2 | `dataset.py:311-324`、`360` | 每个样本要逐时刻调用 `context_at`，里面 `order.index()` 是 O(N) 查找。 | 中 | 按版本边界分段缓存；order 预建成 dict；或做样本缓存（T3） |
| D4 | `dataset.py:393` | horizon 内新投运的实体不参与监督，符合设计。 | 低 | 在 latent 指标里统计这部分缺失比例 |
| D5 | `dataset.py:415` | 监督 mask 不按 target_available_at 过滤。离线评估中正确，滚动回测时需要截止时间。 | 低 | 增加注释或 `label_cutoff` 参数 |
| N12 | `aidc_mapping.py:52-109`；`spatial.py:142-156` | 每个快照都重做 DAG 校验；relation_degrees 是三重 Python 循环。 | 低 | 按快照指纹缓存；度数预先计算 |

### 5.2 合成数据与处理（含特征来源）

| # | 位置 | 问题与理由 | 严重度 | 修复 |
|---|---|---|---|---|
| **N1** | `generator:1213`、`1186-1187` | 边界电表约等于目标（r_a 实测）。AIDC 的空间、关系消融会被"抄总表"混淆。 | **高**（v1 为中） | 增加 `no_boundary_meters` 场景或 variant（mask 掉 site、bd、tr、bus 的功率和 load_ratio），并在报告中注明这条捷径 |
| N2 | `generator:223`、`544`、`1220` | 所有目标都是无噪声的确定性函数，最小可达误差为 0。 | 中 | 可选：加 0.5–1% 的标签噪声，并输出 noise_floor |
| N3 | `generator:1176-1181`、`1202`、`1217` | 常量和重复特征（r_a 实测）。 | 低 | 加入轻微变化或从 schema 删除，在 data_manifest 中列出退化特征 |
| N4 | `generator:1228`、`1364-1365` | dense 和 sparse 共用 RNG，sparse 不是 dense 的子集。 | 低 | 每个视图用独立 SeedSequence，或在 dense 的 mask 上再叠加屏蔽 |
| N5 | `generator:81-83` | AIDC 目标从不缺测（r_a：observed_rate=1.0）。 | 低 | 在部分站点加入目标缺测片段 |
| D3 | `generator:1301-1305` | 预报符合 §18.6，但是知道真实参数时的条件均值，偏乐观。 | 低 | 可选的敏感性实验 |
| P2 | `scaling.py:30-36` | std≈0 时置 1；某个声明特征完全无观测时直接报错。 | 低 | 输出退化特征清单 |
| P3 | `scaling.py:238-280`；`exp:210` | apply 原地修改并每个 epoch 重复；`_tokenize` 逐元素循环。 | 中 | 缓存 transform 后的样本；类别查表向量化 |
| P4 | `exp:91-92` | workers 固定为 0，数据准备与计算串行。 | 中 | 缓存之后再开 workers，并把 sampler 状态纳入 checkpoint |
| N14 | 三份 AIDC JSON；`generator:1407-1416` | 双源配置可能漂移（REQ-44 无测试）。 | 低 | 加等价性测试，或只维护一份基准、profile 用差异生成 |
| F1 | `temporal.py:47-60` | 日历逐个时间戳调用 datetime，并对 tensor 调 `.tolist()`（GPU 同步），是热点。 | 中 | 在数据阶段预先算好，或做向量化偏移表 |
| F2 | `spatial.py:103`、`111` | `total` 是标准化后的值求和，并与 log1p(N) 共线。 | 低 | 改为原始量纲求和后再标准化，或删除 |
| F3 | 日历 | 没有节假日和月、年相位。 | 低 | 真实数据阶段作为 known_future 通道加入 |

### 5.3 架构

主闭环与 v4.0 一致；消融的通路切断都有测试；mask 向量化与逐边 oracle 等价。

| # | 位置 | 问题与理由 | 严重度 | 修复 |
|---|---|---|---|---|
| M1 | `GSLAformer.py:363-371` | 每个 rollout 步都对整个 L 窗口重跑 TemporalEncoder，复杂度 O(H·L²)。 | 中 | **KV cache 与 impl §4.2:186 冲突，需要用户裁决（Q1）**。不违规的替代：patch 化、梯度检查点 |
| M2 | `temporal.py` | 只取末位 token，但每层都在算整条序列。 | 低 | 最后一层只算末位 query（数值等价） |
| M3 | `spatial.py:25`；`config.py:180-183` | summary 输出被丢弃；层数被校验写死为 1。 | 低 | summary 作为 readout 的额外 key；层数可配置 |
| M4 | ConditionFusion | 拼接后直接接 MLP，没有残差和 LayerNorm。 | 低 | 改为残差加 LN，或用 FiLM（属架构变更，需登记） |
| M5/N18 | `relations.py:59-93` | 逐帧逐边在 Python 中构造 mask，大部分帧的边集相同。 | 中 | 按快照指纹去重后广播 |
| M6 | `GSLAformer.py:316-331` | predict_direct 对 H 步逐步循环。 | 低 | 批量化 |
| N17 | `entity_history.py:99-105` | GRUCell 按步 Python 循环；`raw` 是死变量。 | 低 | 加 nn.GRU 快速路径；删死变量 |

### 5.4 训练

**优点**
- 状态机和续训逐位一致。
- 冻结链确定（r_c）。
- loss reduction 有手算测试。

| # | 位置 | 问题与理由 | 严重度 | 修复 |
|---|---|---|---|---|
| T1 | `exp:283`；`support:106-132` | 每个 B batch 都重算 teacher。r_c 证明它确定、可以缓存，且缓存不改变规格语义。 | **高** | Freeze 后一次性算好 (z, q, EncodedHistory) 并缓存，用表示指纹校验 |
| T3 | `exp:206-211` | 每个 batch 现场完成读包、as-of、transform、collate；A 阶段单 epoch 1833 s。 | 中 | 样本缓存，key 为数据指纹加 transform 指纹 |
| N15 | `_day.json:27` | batch=2（acc §21 的初始值）。 | 中 | 缓存后提高 batch 或做梯度累积（需改 acc §21） |
| T2 | `exp:260` | 课程从 3 步直接跳到 72 步。**这是 impl §4.1 的规格决策**。 | 中 | 几何课程，**需要用户裁决（Q2）** |
| T4 | `config.py:203-208` | 没有 warmup 和调度器（impl §4.1:173 的规格决策）。 | 中 | 增加"实验模式"开关，验收模式不变，**需要用户裁决（Q3）** |
| R2 | `exp:107-111`；`run.py:380` | 没有确定性开关；run.py 先全局 set_seed，再被 config 种子覆盖。 | 中 | 增加 deterministic 开关并写进 environment；GSLA 路由跳过全局 set_seed |
| T5 | `exp:336-344` | sensitivity 诊断只用第一个 valid batch。 | 低 | 固定抽样 N 个窗口 |
| T6 | `exp:458` | min_delta 是绝对阈值。 | 低 | 改为相对阈值 |
| N16 | `fit_gslaformer_mechanisms.py:30-33`、`110-111` | 小集拟合在训练集上选优，阈值偏宽松。 | 低 | 报告绝对 MSE 和成对通过比例，不只看 passed |

### 5.5 测试与评估

- 实际结果：86/86 通过。泄漏边界和契约的覆盖很强，多数用例使用独立 oracle。
- E1（中）：scaling 测试没有覆盖 site_time 留出站（`test_gslaformer_scaling.py:44-45` 只测了 fit(valid) 被拒绝）。r_b 证明实现正确，但缺少回归保护。
- E2（中）：指标重复实现且没有数值单测（REQ-24）。
- E3（低）：seasonal 基线没有覆盖 H 大于周期、跨时区、缺测相位。
- E5（低）：没有性能和资源回归阈值。
- E6（中）：没有端到端泄漏反事实测试（r_b 可以直接转成用例）。
- N10（低）：断言偏弱（r_d）。
- N11（中）：生成器没有单测（REQ-41）。
- N6（中）：文档声称有过拟合和 MPS 测试，实际没有。
- REQ-29 needs_diagnosis：未实现，而规格本身是"候选"，需要裁决（Q4）。
- E4（低）：`compare_seed`（`config.py:210-215`）是死配置。

### 5.6 推理

checkpoint 自包含；forecast 不带监督；本轮 smoke 的 4 类请求都完成了。

| # | 位置 | 问题 | 严重度 | 修复 |
|---|---|---|---|---|
| I1 | `inference:142-150` | test 模式下每个 batch 额外跑 teacher 和 3 种诊断。 | 低 | 加 `--diagnostics` 开关 |
| I2 | `inference:175` | 负值只打标记（符合 io §10）。 | 低 | 维持；若业务需要非负，作为显式版本变更 |
| I3 | `inference:52` | `torch.load(weights_only=False)`。 | 低 | 权重与 RNG、optimizer 分开存，推理时用 weights_only |
| N13 | `run.py:16`、`287` | ROOT 依赖 cwd；train 模式下 `--gsla_checkpoint` 兼作 resume。 | 低 | 改用 `Path(__file__)`；提供独立的 `--gsla_resume` |

### 5.7 结果保存与可复现

| # | 位置 | 问题 | 严重度 | 修复 |
|---|---|---|---|---|
| R1 | `exp:166-168` | 没有代码版本（r_f）。 | **高** | 写入 git SHA、dirty 标记、关键源文件和 uv.lock 的哈希，并让 validate_artifacts 强制检查 |
| R3 | `artifacts.py:132-151`；`comparison.py:229-235` | NRMSE 未实现；comparison.csv 只有 micro。 | **低**（v1 为中，见 §9） | 加入可选 NRMSE 和 per-site macro 列 |
| U-3 | `artifacts.py:141` | 缺少 profile 维度。 | 低 | entries 加上 profile 字段 |
| R4 | `inference:163-175` | 每行重复整份元数据。 | 低 | 拆成 requests 表 |
| R5 | `exp:474` | last.pt 内嵌 best_a 权重和全部 metrics。 | 低 | 改为路径引用 |

---

## 6. 测试缺口与建议用例

| # | 用例 | 关键断言 |
|---|---|---|
| TG-1 泄漏反事实（E6，基于 r_b） | 扰动留出站和 t≥split[1] 的数据，并重算 SHA | `assertEqual(TrainingTransform.fit(ds0).fingerprint, TrainingTransform.fit(ds1).fingerprint)`；再训练 1 个 update：`assertEqual(sha(best.pt 权重)_0, _1)` |
| TG-2 site_time scaler（E1） | 留出站注入极端值和新类别 | `assertEqual(fp0, fp1)`；`assertNotIn("new_token", state["vocab"][g][f])` |
| TG-3 指标纯函数（E2） | 构造 pred=[[1,3,2]]、true=[[1,2,4]]、mask 全真、freq=3600 | `micro_mae==1.0`；`peak_amplitude_mae==1.0`；`peak_time_mae_seconds==3600`（argmax 1 对 2）；`ramp_mae==mean(abs([2-1, -1-2]))==2.0`；mask 中间置假后 `ramp_count==0`；peak 用最早 argmax：`argmax([5,5])==0` |
| TG-4 inference 与 comparison 指标一致 | 同一组 pred/true 分别送入两份实现 | 各键 `assertAlmostEqual` |
| TG-5 rollout 前缀一致（基于 r_c） | — | `torch.allclose(rollout(b,K).y, rollout(b,H).y[:, :K])`，K∈{1, min(3,H)} |
| TG-6 teacher 确定性 | `set_training_stage("B")` 之后 | 两次 `build_latent_labels` 的 z 和 quality `torch.equal` |
| TG-7 修正 N10 | — | 拆成 `assertFalse(allow[0,0,2])`、`assertFalse(allow[0,2,0])`、`assertFalse(allow[0,1,2])` 等独立断言 |
| TG-8 生成器（N11） | `aidc_visible_plan`、`m2_target`、`m3` 递推 | 固定种子下对手算值 assertAlmostEqual；按 acc §11.1，x0=10、u=[5,0] 时 step2 = 0.8·(0.8·10+5)+0 = 10.4 |
| TG-9 捷径检测（N1） | — | 断言 `abs(corr(site.active_power, target)) < 0.95`。在 no_boundary 场景中加入；当前数据下预期失败，作为门禁 |
| TG-10 产物溯源（R1） | — | `assertIn("git_sha", environment)`，且 `validate_artifacts` 缺少该字段时 raise |
| TG-11 配置单源（N14） | — | scripts/smoke 中的三份 AIDC JSON 与生成器 full_config 去掉 4 个 profile 字段后 `assertEqual` |
| TG-12 过拟合 sanity（N6） | 1 个 batch、dropout=0、50 步 | `loss_end < 0.2*loss_start` |
| TG-13 suite_result（REQ-43） | 只运行 M3 | `suite_result["status"] == "partial"`，且其余任务为 `not_run` |
| TG-14 负值保留（REQ-33） | 构造负输出 | csv 中 y_pred<0，`negative_prediction==True`，数值未被裁剪 |
| TG-15 MPS | `skipUnless(torch.backends.mps.is_available())` | 1 epoch 的 smoke 完成，且 device=="mps"；不可用时显式请求 MPS 应 raise |

---

## 7. 路线图

### P0：可信与可产出

| # | 内容 | 验收标准 |
|---|---|---|
| P0-1 | 溯源（R1）：git SHA、dirty 标记、源码和 uv.lock 哈希；提交代码需要用户授权 | TG-10 通过；新产物都带这些字段 |
| P0-2 | teacher 和 EncodedHistory 缓存（T1） | 与不缓存时 B 阶段 loss 的差 ≤1e-6；multiday 的 B batch 耗时降低 ≥50% |
| P0-3 | 样本缓存（T3，含日历和度数） | 缓存前后样本逐位一致；A 阶段 epoch 耗时远低于 1833 s，并回填 history |
| P0-4 | 指标纯函数（E2），两处共用；可选 NRMSE、per-site macro、profile 维度 | TG-3、TG-4 通过；重复实现删除 |
| P0-5 | 泄漏和留出站测试 | TG-1、TG-2、TG-5、TG-6 入库 |
| P0-6 | 文档修正：N6、N7、N8、N9、U-6、U-7、U-10；裁决 D01、D02 | §4.2 的条目全部关闭或登记为有意保留 |
| P0-7 | needs_diagnosis：先裁决（Q4），再实现 | 有测试：构造 full 远差于 naive 时状态为 needs_diagnosis |

### P1：结论强度与效率

| # | 内容 | 验收标准 |
|---|---|---|
| P1-1 | 去捷径（N1–N5），新增 `no_boundary_meters` | TG-9 在新场景通过；AIDC 消融报告与该场景并列 |
| P1-2 | 生成器单测（N11）和配置单源（N14） | TG-8、TG-11 通过 |
| P1-3 | 确定性模式（R2） | CPU 上跑两次权重逐位一致 |
| P1-4 | 向量化：F1、M5、N12、D1、D2、P3、M2、N17 | profile 中热点占比下降；数值测试仍然通过 |
| P1-5 | 若 Q2、Q3 获批：几何课程、实验模式下的 lr 调度 | 默认行为逐位不变；有配置开关测试 |
| P1-6 | 提高 batch 或梯度累积（N15，需改 acc §21） | 在注意力预算内；记录耗时与指标 |
| P1-7 | MPS smoke 脚本化（U-7、TG-15） | smoke 有 `--mps` 步骤 |

### P2：模型能力

| # | 内容 | 验收标准 |
|---|---|---|
| P2-1 | 时间链提速：patch 化或梯度检查点；KV cache 视 Q1 而定 | 等价性测试或指标不劣化；B-full 单步耗时降到 1/5 以下 |
| P2-2 | 融合加残差和 LN（或 FiLM）、summary 接入 readout、层数可配置 | 消融报告附置信区间 |
| P2-3 | 多种子 3–5 个；实现或删除 compare_seed（E4） | 报告均值 ± 标准差 |
| P2-4 | 每站或容量归一化作为可选实验；可选标签噪声和 noise_floor（N2） | 与默认方案的 per-site NRMSE 对比 |
| P2-5 | weights_only 推理（I3）、predictions 拆表（R4）、last.pt 瘦身（R5）、诊断开关（I1）、`--gsla_resume`（N13） | 推理不再需要 `weights_only=False` |

---

## 8. 待用户裁决问题

| # | 问题 |
|---|---|
| Q1 | 是否放开 impl §4.2:186 对 KV cache 的禁令（需同时改为相对位置编码）？如果不放开，只走 patch 或梯度检查点。 |
| Q2 | 是否把课程 1 → 3 → H 改为几何课程（impl §4.1 的规格变更）？ |
| Q3 | 是否允许"实验模式"使用 warmup 和 lr 调度，验收模式保持固定 lr？ |
| Q4 | needs_diagnosis 门禁（acc §20.3"候选"）：是否确认 2× 倍数并要求实现？必须在看到 test 之前登记。 |
| Q5 | D01（静态链：改代码为 MLP，还是改规格为单 Linear）、D02（日历：相加还是拼接）以哪一边为准？ |
| Q6 | 是否授权提交当前未跟踪的实现（R1 的前提）？ |
| Q7 | AIDC 是否新增 `no_boundary_meters` 场景（会改 acc §19.2 的场景矩阵）？是否给标签加噪声？ |
| Q8 | NRMSE 是否从"可报告"升为必报？ |
| Q9 | 是否把 AIDC 的 batch=2（acc §21）改为缓存后再评估？ |

---

## 9. 与上一版的差异

**方法**
- v2 按新的 prompt 抽取了 48 条 REQ，并建立追踪矩阵。
- 实际运行了 smoke（v1 只跑了单测）和 M3/M5 对照。
- 写了 6 个复现脚本，每条可疑结论都有脚本证据。

**更正**

| 编号 | v1 结论 | v2 结论 | 依据 |
|---|---|---|---|
| R3 | NRMSE 未实现（中） | **降为低** | impl §10 和 acc §20.2 都写的是"可报告／可加"，属于可选诊断 |
| N1 | 中 | **升为高** | r_a 实测 corr 为 0.9992 和 1.0000（load_ratio），留出站和 sparse 视图同样存在 |
| N3 | cpu==gpu 是在 compute 组 | 实测是 **it_power 组**的 cpu_utilization 和 gpu_utilization；另确认 hvac 的 f1 恒为 1、bd1 恒为 0 | r_a |
| M1、T2、T4（KV cache、几何课程、lr 调度） | 直接作为 P1/P2 的优化项 | **与现行规格冲突**（impl §4.1、§4.2:186），改为待裁决（Q1–Q3） | 规格原文 |
| 峰值和爬坡指标 | — | 确认已实现且符合"最早 argmax"的定义；缺口只是没有测试 | `inference:208-216` |

**新增发现**
- U-3：scenario_metrics 缺少 profile 维度。
- U-4：needs_diagnosis 未实现。
- U-6：acceptance:644 的路径错误。
- U-7：acceptance:653 关于 MPS smoke 的说法与脚本不符。
- REQ-43：suite_result 的 partial 语义经实测正确，但没有单测。
- 复现确认：无泄漏（r_b）、rollout 前缀一致、teacher 可缓存（r_c）、N10 断言偏弱（r_d）。
- 新增测试用例 TG-1 至 TG-15，每条都写明了断言。

**保留**
- v1 中仍然成立的条目全部保留：D1–D5、P2–P4、F1–F4、M1–M6、T1–T6、E1–E6、I1–I3、R1–R5、N1–N18，以及附录 A 中被 §4.2 吸收的条目。
- v1 的 E4、P1、D3 更正维持不变。
