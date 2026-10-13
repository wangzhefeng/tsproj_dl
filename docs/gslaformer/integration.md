# GSLAformer R4：已有组件复用与项目接入

> 本文件区分**已读源码事实**与**拟议实现**。本轮没有执行已有模型训练，也没有编写 GSLAformer；源码中有类不等于该模型已通过当前回归。
>
> 主设计：[design.md](design.md)；具体接口／阶段：[implementation.md](implementation.md)；业务输入：[input_output.md](input_output.md)。

## 1. 结论：复用基础设施与成熟原语，不强塞旧张量接口

项目现有主链为 `run.py → Exp_Long_Term_Forecast → TFs_type → Exp_Basic`。GSLAformer 继续使用 `run.py` 入口与现有模型发现／设备基础，但走独立 `Exp_GSLAformer_Forecast` 和 `data_provider/spatial_temporal/`。

原因不是“新模型都要新框架”，而是语义不同：旧流程处理四个普通张量、单一输出与目标切片；GSLAformer 处理带类型关系的变长实体字典、隔离监督、两阶段冻结、latent 辅助损失及回灌。让 latent 冒充 `c_out` 列会破坏目标含义。

新数据流归属为**在建**，不是取代既有 TFs_type 主线。只有模型实现、入口接入、smoke 与产物均通过后，才更新根 AGENTS 状态。

## 2. 模型与层的源码复用判断

路径相对仓库根目录；行号为本轮检查位置，后续开发前需重新定位符号。以下“复用”均是开发决策，不是本轮已接线。

| 已核实位置 | 有帮助的机制 | 复用结论及限制 |
| --- | --- | --- |
| `layers/Transformer_EncDec.py:33`，`EncoderLayer` | attention + residual + Post-LN + FFN | 结构参考。当前封装没有独立 padding 接口；GSLAformer 优先用已有 PyTorch 的 `TransformerEncoderLayer`，不为新模型改坏共享层 |
| `layers/SelfAttention_Family.py:178`，`FullAttention` | 明确的 QK 分数与 mask | 当前 `mask_flag=False` 时不消费传入 mask，`True` 时用 `attn_mask.mask` 作布尔屏蔽；不是多类型加性 bias 的直接接口。关系层用 `nn.MultiheadAttention`，新建关系 mask／bias 逻辑 |
| `layers/Embed.py:61`，`PositionalEmbedding` | 固定正弦位置编码 | 可定向复用到系统时间轴，需验证维度、最大长度、设备、裁窗与 dropout 约束；不能加到任意实体行号 |
| `layers/Embed.py:30`，`TokenEmbedding` | 卷积数值映射 | 不直接用于需要逐时刻因果 z 的实体链：kernel=3、circular padding 会跨邻时刻混合；固定 c_in 也不适合直接展平变长实体轴 |
| `models/transformer/Transformer.py:22` | 标准 encoder-decoder 与时间 embedding | 适合作总负荷基线；不整体搬入 GSLAformer，原图无标准 cross-attention decoder，且该模型不生成回灌 latent |
| `models/transformer/iTransformer.py:25,67` | 变量作 token，变量间 attention | 借鉴“变量／对象先编码再交互”；当前输入是 `[B,L,N]` 宽表，归一化不读缺失 mask，没有类型关系和单设施聚合头，不能直接等同 AIDC 对象图 |
| `models/transformer/PatchTST.py:62,121` | patch 化与共享通道编码 | 可作总负荷或固定摘要基线。其本地预测先把通道折进 batch，主要是通道独立时间建模；不代表设备关系交互。替换实体 GRU 要另验因果前缀与缺失，首版不换 |
| `models/transformer/Crossformer.py:35` | 时间段／跨变量分轴 attention | 设计参考；可学习位置参数包含 `enc_in` 轴，绑定变量槽位，不直接满足新增设备免参数变化 |
| `models/transformer/TimeXer.py:102,174` | 主变量摘要与外生变量交互 | 可借鉴条件融合；当前 forecast 从 `x_enc` 历史外生列编码，并非自动消费完整未来任务计划。不能仅因接受 `x_mark_dec` 就声称已支持未来计划 |
| `models/transformer/TemporalFusionTransformer.py:60,254` | static／observed／known 分离、条件选择 | 输入角色划分有帮助；本地 `datatype_dict` 固定 ETTh1/ETTm1，known 来自时间标记，不是通用计划版本库。首版只借鉴分离原则，不搬整套 VSN／decoder 增加复杂度 |
| `models/rnn/GRU_todo.py:93` | 单向 GRU 与 direct horizon head | 重用 PyTorch GRU 原语／工程经验，不复用其总输出头；GSLAformer 需要每实体每步状态、present 重置和 teacher memory 延续 |
| `models/transformer/LSTMTransformer_todo.py:29` | RNN 后接内置 Transformer | 证明仓库已有组合风格；其时序 mean pooling 和固定 horizon 头不等于池化前实体编码与 latent 双头。当前实际文件是 `_todo`，不以旧 README 的无后缀名称推断路径 |
| `models/mlp/DLinear.py:15` | 趋势／季节分解及简单线性预测 | 适合作总负荷对照，检查复杂模型是否真的有价值；不能靠分解自动利用拓扑 |
| `models/mlp/TSMixer.py:4` | 时间／通道 MLP 混合 | 适合作固定宽度站点摘要基线；通道层 `Linear(enc_in,...)` 固定，不能直接在原始可变设备轴共享 |

> **核心认识：已有模型主要提供可验证的基线和结构经验。需要新写的是类型／mask／关系／读出／回灌契约，而不是重新发明 Transformer 注意力内核。**

首版必需对照以 implementation 和 synthetic_aidc 的矩阵为准；上表不意味着必须训练所有仓库模型。DLinear／PatchTST 等若另加入比较，应在相同目标／分区／可用信息下运行，并注明它们是否只看总表或固定摘要；不能把输入更多造成的收益全归给架构。

## 3. 工程基础设施：可复用到哪一层

| 现有源码事实 | 可复用部分 | 必须隔离或补足 |
| --- | --- | --- |
| `run.py:89` 根据 task/model 选实验类 | 单一 CLI 入口与显式路由 | 新模型专用 parser／run 分支；不进入旧 setting 或全零触发最终训练 |
| `exp/exp_basic.py:31` 扫描 `models/*`、懒导入 `Model`、初始化设备 | 模型发现、基类构造 | 新实验类实现 `_build_model`，递归搬运字典张量；不把 metadata `.to(device)` |
| `exp/exp_basic.py:78` 指定设备不可用可回退 CPU | 设备选择机制 | GSLA 路由先验证明确请求的设备，不可用时报错；不能把 CPU 回退称为 MPS smoke |
| `data_provider/TFs_type/data_loader.py:188,271` 读 CSV、固定列、返回四张量 | 切窗／独立 target scaler 的经验 | 新建 JSON+NPZ typed Dataset、as-of、mask、site split、变长 collate；不复制固定宽表 loader |
| `data_provider/TFs_type/data_factory.py:32` 支持 train/valid/test/pred，`drop_last=False` | 命名与 DataLoader 习惯 | GSLA 的内部 `pred` 对应外部 `forecast`，valid/test 保序，train 可按站点平衡采样 |
| `exp/exp_long_term_forecasting.py:285` decoder input + 单输出切片 | 普通模型链原样保留 | GSLA 禁止套用 `_model_forward` 的目标列切片 |
| `utils/model_tools.py:71` EarlyStopping 保存 epoch/model/optimizer/scheduler | 早停思路 | 现有保存不足以携带阶段／RNG／schema／scaler，且键为原样 `optmizer`；新类单独实现所需状态，不顺手改旧 checkpoint 格式 |
| `utils/metrics_dl.py:41` NumPy MAE/MSE/RMSE | 在明确筛选的有效一维样本上复用基础误差函数 | 新增 site/horizon reduction、计数、峰值／爬坡；不能把未掩码数组直接交给旧指标 |
| 长期实验保存训练 scaler、test/forecast 读取 artifact | 训练统计持久化原则 | 旧 test 有 scaler 缺失重拟合路径，GSLA 必须 fail-fast；不复制这种回退 |
| `run.py:63` 将产物统一至 results 根 | `results/`、日志与脚本习惯 | GSLA 按 run_id 保存专用 schema/阶段文件，不改其他模型输出目录 |

## 4. 嵌入现有框架的完整链路

### 4.1 文件职责与数据保存

拟议文件仍以 implementation §7 为准，R4 增补：

| 位置（拟议） | 负责内容 | 输入 → 输出 |
| --- | --- | --- |
| `data_provider/spatial_temporal/aidc_mapping.py` | 测点、单位、端口、关系类型映射与重复覆盖审计 | 来源表 → 标准 entity/feature/typed relation |
| `data_provider/spatial_temporal/schema.py` | schema 3、站点和目标合同、张量校验 | JSON/NPZ → 合法结构或明确错误 |
| `data_provider/spatial_temporal/dataset.py` | 起点可见性、站点／时间分区、切窗、padding | 标准包 → ForecastInput 与独立 Supervision |
| `data_provider/spatial_temporal/scaling.py` | 跨训练站点组级 scaler 与属性词表 | 训练可用观测 → 冻结可复用统计 |
| `layers/gslaformer/` | 实体历史、空间／类型关系、分组读出、因果时间与完整未来路径 | 分组对象 → 隐状态与总负荷 |
| `models/transformer/GSLAformer.py` | `Model` 及唯一 rollout 组合 | ForecastInput → ForecastResult |
| `exp/exp_gslaformer_forecasting.py` | A/B 训练、valid 选优、test、forecast | 配置／数据／模型 → 可追溯结果 |
| `exp/gslaformer_support.py` | 冻结 teacher、损失、按站／horizon 指标及 checkpoint | 模型输出／监督 → loss 与 artifact |
| `scripts/dev/generate_gslaformer_fixture.py` | base、M1–M5、AIDC 模拟及 oracle | 显式参数／seed → synthetic 标准包 |
| `scripts/smoke/gslaformer_*.json` | 小时、一天、三天的完整配置 | 不用隐藏 CLI 覆盖，保存 resolved_config |

真实业务源文件建议置于经授权的 `dataset/AIDC/<site_id>/raw/` 与 `processed/`，只记录路径与校验值，不提交敏感数据。该目录当前不创建；正式生产导入器要等源数据合同明确。模拟包仍放 `results/gslaformer/fixtures/aidc/`，模型与结果不放文档目录。

### 4.2 配置如何落地

[aidc_profile.example.json](aidc_profile.example.json) 是本轮可解析的**设计样例**，不是当前 `run.py` 已支持的运行配置。它明确六组 schema、typed relation 约束、站点切分、目标口径、时间长度、训练阶段及容量。

实现时生成三个物理配置文件，禁止通过修改同一运行目录覆盖：

- `gslaformer_aidc_hourly_short.json`：L=48,H=6；
- `gslaformer_aidc_hourly_day.json`：L=168,H=24；
- `gslaformer_aidc_hourly_multiday.json`：L=168,H=72。

每份都包含完整参数，不依赖未声明的配置继承。样例 data.package 指向其所在 docs 目录相对的模拟包，实施时复制到 scripts/smoke 后相对深度相同。运行时未知键、schema 版本、分区冲突、未来计划列变更必须报错。

### 4.3 训练、验证、测试、预测

1. **train**：验证全套合同，仅用训练站点训练期拟合统计；A 直接多步负荷监督；冻结表示链及静态链；B 课程自由递推；完整 horizon 验证选优。
2. **valid**：不更新 scaler／词表；在已声明验证站点与时段汇总 site/horizon 指标；不是每 epoch 访问最终测试站点。
3. **test**：加载同一 best checkpoint，在同站未来与完全留出的站点分别滚动回测；不改权重、统计或超参数。
4. **forecast**：加载自包含 checkpoint，核验新站 schema、目标和单位；请求只有历史／静态／关系／原点已知未来路径，无标签；一个完整 rollout 得出 H 步。
5. **保存**：原始单位预测、horizon/site 指标、质量、配置／版本／hash 和运行状态；最后检查全部必需产物才标 completed。

拟议命令模式仍为 `run.py --model GSLAformer --gsla_config ... --gsla_mode train|test|forecast`，当前不可运行，不执行这些不存在的功能。已有非 GSLAformer CLI 保持原语义。

## 5. 验证与实施工作包

| 工作包 | 在 R3 I1–I6 上增加的 R4 验收 | 成功证据 |
| --- | --- | --- |
| I1 | AIDC mapping、schema 3、typed relations、site_time split、模拟 oracle | 字段／单位／计量／版本／留出隔离测试 |
| I2 | 同组多类型、多关系同端点、空组、变长多站 batch | 有限输出、重排不变、跨样本独立、受控关系消费 |
| I3 | 小时至三天完整路径与同一起点回灌 | H=6/24/72 对齐，无未来标签泄漏 |
| I4 | 多站平衡采样与验证 site-macro，冻结与重载 | 实际训练日志、独立统计 fingerprint |
| I5 | 一套入口、不同 profile、自包含 checkpoint 和新站 forecast | CPU／MPS 分开报告、旧路由回归 |
| I6 | M1–M5 + AIDC 矩阵、负面结果、资源规模限制 | 完整 suite_result 与正式 results 产物 |

初始已有 tests 被 `.gitignore` 中 `tests/` 忽略；实施时先处理精确版本化例外再新增测试。本次不修改该规则，不以文档计划声称测试已提交。

拟议回归只覆盖受影响的参数解析、模型发现、TFs_type 主链 smoke 和新模型定向用例；不盲目 import 仓库所有重依赖大模型。模型开发仍需用户授权，不因本轮文档任务启动训练、安装依赖或提交代码。

## 6. 已知缺口与不夸大的结论

- 已有模型没有经本轮证实的“完整 AIDC 多类型输入 → 双头 latent 自回归 → 新站推理”可直接复用实现。
- 新模型仍需实测 attention masking、梯度、两阶段冻结、MPS、长 horizon 内存及结果重载。
- 生产全量节点数未知；当前稠密空间与关系 attention 的安全配置仅覆盖缩小的功能数据，不承诺万级服务器规模。
- 两层关系交互不是全层级逐级物理求解；合成效果不能替代真实站点验证。
- 本轮复用分析以源码事实为依据；未对任何候选组件作性能优劣排名。
