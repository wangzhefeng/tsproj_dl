# GSLAformer R4：统一验收协议

> 本文唯一维护验收任务、生成规则、对照、阈值、运行矩阵和证据要求。计算与训练算法见[实施规格](implementation.md)，数据格式见[输入输出契约](input_output.md)。
>
> 本文是验收要求，不是完成清单。实际进度与运行证据唯一见[实施规格 §12](implementation.md#work-packages)；已执行子集不代表整体通过。参数与门槛不是生产分布，不据合成结果启动真实数据和多 seed 业务结论。

## 1. 阅读导航与证据层次

- [整体完成定义](#completion-definition)与[通用断言](#contract-assertions)：所有路线适用的工程门禁。
- [周期 base](#base-fixture)：文件、入口、保存与重载的流程 smoke。
- [M1–M5](#mechanism-tests-2)：可证伪机制、负对照与成对小集拟合。
- [多 AIDC 场景](#synthetic-aidc-2)：组合机制、站点留出及小时至天级链路。
- [对照定义](#variants)、[阈值](#fit-thresholds)、[命令](#planned-commands)及各层运行矩阵。

三层数据的通过结论分别记账：base 不能替代机制，机制不能替代 AIDC，合成结果不能替代真实业务验证。每项记录 passed/failed/not_run；运行子集不得把整体标为 passed。

| 数据层 | 为什么需要 | 能证明／不能证明 |
| --- | --- | --- |
| 周期 base | 验证文件、训练入口、保存与重载 | 可以检查流程；日周期复制不能证明模型必要 |
| M1–M5 机制任务 | 单独检验轨迹、关系、计划、规模、缺失和版本 | 可以构造可证伪反例；不是电气／暖通仿真 |
| `aidc_scenarios` | 联合多站点、多设备、多负荷模式和小时至天级预测 | 可以检验标准合同、组合泛化与工程闭环；不等于真实 AIDC 泛化 |

<a id="completion-definition"></a>

## 2. 整体完成定义

全部达成才称“功能实现完成”：

- 数据契约、ID 无关性、可变实体数、关系路径、时间因果、未来条件路径通过测试。
- 实体历史先编码后池化；同一整段轨迹重排不变，中途交换轨迹必须被视为不同输入。
- 合成数据两阶段训练、完整 horizon 自由递推、test、无标签 forecast、模型重载成功。
- 同一个 checkpoint 在新增实体／新增父子节点请求中不改变参数，不重拟合统计量。
- CPU 验证与可用 MPS 的定向 smoke 均有实测证据；MPS 不可用时记录环境阻塞，不能冒充通过。
- 基线和最小消融按同一合成协议运行，保存结果和失败情况，不要求完整模型获胜。
- 原有模型入口定向回归通过，无全局依赖、外部服务或数据库迁移。
- AIDC 六组 schema、多类型关系、站点与时间双轴切分，按 H=6/24/72 的独立 profile 完成模拟训练／测试／无标签新站预测，资源与失败结果如实记录。

没有真实数据实验时只报告 synthetic 功能结论。文档与代码完成度分别记录，不把本规格的存在计入模型完成度。

<a id="contract-assertions"></a>

## 3. 通用契约与模型断言

| 测试 | 具体断言 |
| --- | --- |
| 数据边界 | 第一个／最后一个合法窗口、目标归属 split、scaler 不读 valid/test；as-of 迟到记录不可见 |
| 输入缺失 | NaN 被 mask 后变有限值；padding／不适用／未生效行为不同；全组缺失不产生 NaN |
| 因果性 | eval/dropout=0，修改未来 token 不改变更早时间输出 |
| 同步重排／ID 重命名 | 同步重排 values、属性、mask、关系后输出一致；只改 ID 字符串数值不变 |
| 实体轨迹保真 | 相同每帧集合、不同跨时实体对应的成对任务，可表达并拟合不同目标；逐帧置换不是合法不变性 |
| 关系方向 | 接收行／发送列与源表一致；两个方向类型不同；受控消息仅经许可边传播 |
| 关系消费 | 在固定、非退化权重和可区分属性夹具上换父级，关系层输出及最终预测变化 |
| 未来条件消费 | direct/AR 同时读取完整起点计划；只改目标步之前的计划也有合法梯度路径；mask=false 数值变化不影响结果 |
| 泄漏保护 | 更改未来 target／future_groups／晚发布计划，固定权重自由预测不变 |
| 冷启动／新实体 | 不优化、不新增参数，改变 N 后 shape 正确、结果有限、冷启动标记准确 |
| 规范 latent | encoder 标签与预测头坐标一致；方差权重仅影响 loss，不影响同权重 rollout |
| B 冻结 | 外层 train 后表示链仍 eval，参数不变；系统时间／未来路径有梯度；latent 头梯度断言遵循 [两阶段训练状态机与恢复](implementation.md#implementation-6) 的 K/H 条件 |
| 回灌梯度 | 第二步主损失能沿第一步生成 latent 与 y 回传；无意外 detach |
| loss reduction | 手算夹具验证有效单元、有效组平均及零分母；通过测试实际核算数值 |
| 课程与早停 | 短课程不会提前停止，最终 best 来自 B-full，H=1 正常 |
| 续训 | CPU workers=0，在 epoch 边界恢复与未中断对应轨迹在容差内一致 |
| 产物重载 | 同设备加载前后预测一致；所有 schema／统计校验生效 |
| 未来 mask 独立性 | 受控输入中 observed=true/标准化值=-1 与 observed=false/填零不得被同一投影抵消；masked NaN 仍应被忽略 |
| 合批资源预算 | 单样本合法、合批或 padding 后超过注意力字节预算时，在分配大张量前拒绝；不只检查节点上限 |
| 读出 head 合同 | 合法的 spatial_dim=6、spatial_heads=2、relation_heads=2 可构造并前向；读出与空间 head 一致 |
| checkpoint 溯源 | 新计算版本独立加载成功；旧版本、冻结清单错误、latent 溯源缺失或冲突必须经公开入口拒绝并写 failed |
| 运行审计 | 训练资源文件含设备、参数量、逐阶段耗时及更新数；预测 metadata 含关系指纹、输入截止、站点角色与明确口径的耗时 |
| 正式配置 | 文档和测试直接引用 scripts/smoke；校验 base/AIDC 路径解析、schema 保全、参数约束与站点角色隔离，不另维护配置副本 |
| 旧入口回归 | 普通模型路由与默认参数不受 GSLAformer 专用 parser 影响 |
| 部分观测规模 | 保持各组覆盖比例时扩大节点数，q 不因 all-observed 概率下降归零；一个无效组不关闭其他组辅助 loss |
| 有界权重 | 方差非负时连续、有限、不超过 100；near-zero 不发生跳变 |
| teacher 对齐 | 逐步延续 memory 与同起点一次顺序编码相等；标签状态不会写进生产 state |
| 历史关系切换 | 变更前后各用对应版本，新起点立即可推理，不等待完整 L 稳定窗口 |
| 计算复用 | 一次 rollout 仅一次未来路径编码与一次真实历史编码，无跨 origin 缓存 |

CPU 同输入数值一致性使用 `atol=1e-6, rtol=1e-5`；MPS／跨设备比较 `atol=1e-4, rtol=1e-3`。不要求跨设备逐位相同。受控“应变化”测试先证明输入差异到达指定模块，再以差异大于 CPU 容差断言，不依赖任意随机模型恰好敏感。

<a id="new-entity-checks"></a>

### 3.1 同权重新实体检查

- 同一权重分别预测原请求、新实体有历史请求、新实体无历史请求、新父子节点请求。
- 同一个 checkpoint 参数键与 shape 不变；冻结参数及 scaler 校验值前后不变。
- 新实体标记依据训练实体溯源集合判断，但 ID 不能进入数值推理。ID 重命名可能改变 new_entity 元数据，不得改变 y／z 数值。
- 不要求新增实体使总负荷必然增大；本架构无硬单调性，能预测不等于准确预测。
- 冷启动请求记录 `new_entity_count`、`no_history_entity_count`、`unknown_category_count`、`orphan_count`；不伪装为正常分布精度。

<a id="aidc-assertions"></a>

## 4. AIDC 外部契约补充断言

在 R3 机制测试之外，AIDC 扩展至少验证：

1. 六组 schema 映射、同组多类型／端口、合法空组、重复传感器拒绝／显式主备；
2. 多类型关系、同组边、多关系同端点偏置合成；每类单独约束和跨类型合法环；
3. 服务器—机柜—配电多层同时输入不触发重复求和；目标始终来自明确边界；
4. 多站点 batch 不串数据；身份／站点 ID 重命名不改变数值；
5. 所有留出站点不参与权重训练、scaler 或词表拟合；留出验证站点参与选模，最终留出测试站点不参与选模；同权重新站无标签推理；
6. 小时／天级 profile 的 train/valid/test/forecast、重载和逐 horizon 索引；
7. 每个 profile 的真实资源测量、参数及 scaler 不变检查、失败状态和完整结果文件。

以上是验收要求，是否已经执行以实施规格 §12 为准。

<a id="variants"></a>

## 5. 基线、消融与诊断定义

| variant | 改动 | 训练与解释 |
| --- | --- | --- |
| `persistence` | 最后有效总负荷填满 H | 无优化；无有效历史值时报错，不偷用未来 |
| `seasonal_naive` | 使用起点前最近同相位真实 target | 周期=24，缺相位则用最后有效 target，显式记录 fallback |
| `total_only` | 仅 target 历史、日历、同一已知未来协变量 | 不给静态／内部测点；时间宽度和预算一致 |
| `no_entity_history` | 将实体 GRU 替换为逐时刻共享适配输出，其他路径不变 | 重新训练，检验池化前记忆；M1 是关键反例 |
| `no_spatial` | 保留实体 GRU，移除组内／关系 attention，用 masked mean 读出 | 同一内部信息；静态关系度数清零；重新训练 |
| `no_relation_message` | 移除关系层，保留组内 attention 和静态度数 | 只检验显式消息交互，不称无关系信息 |
| `no_relation_all` | 移除关系层，所有度数／关系派生输入清零 | 检验模型可消费的全部归属信息，重新训练 |
| `direct` | 完整关系空间编码 + A 式直接 H 输出 | 最多 44 epoch（= A 20 + B-short 2 + B-medium 2 + B-full 20，与 full 的总 epoch 上限对齐，见[训练状态机](implementation.md#implementation-6-1)），patience=5；记录实际 update 与训练耗时 |
| `no_latent_loss` | 完整 AR 架构、latent_weight=0 | 从相同 best_A 初始化 B；仅多步主损失训练状态头 |
| `full` | 完整 GSLAformer | A/B 完整运行 |

训练型消融的 A/B 最大 epoch 预算相同，实际早停、参数量和逐步 forward 次数必须报告；同 epoch 不等于同 FLOPs 或时间。`no_latent_loss` 共用 full 的 best_A；direct 接收相同完整未来条件，但目标／优化路径不同，不能将差异全部归因于回灌。各架构均从自身 seed 初始化并按自己的契约重训，不加载不兼容的表示 checkpoint。

`total_only` 是独立基线，不把零动态组伪装成合法 full：A 共享完整计划条件的多步监督，B 只回灌 y，无 latent 头或标签。`no_spatial` 保留同组共享历史编码和属性，读出没有实体间 attention。比较脚本输出完整 resolved_config，不用隐藏 flag 偷偷跳过契约。

无关系消融不从真正的关系重新计算标签质量 q：`no_relation_all/no_spatial` 取 r_neighbor=1；否则即使数值输入清零，辅助权重仍能传入归属信息。no_relation_message 可以保留关系质量，含义明确。各变体的 schema 结构验证可读取关系合法性，但数值预测和 loss 权重不可偷用被消融的信息。

同权重诊断单列 `diagnostics.json`：固定最近真实 z 回灌、固定最近有效 y 回灌、teacher z 回灌；固定替代保留 generated 标记。teacher 使用未来观测，标为 oracle，不排名、不用于选模，也不保证成为严格数学上界。比较诊断误差只能提示依赖路径，不能代替重训消融。

首轮单种子 2023，结论仅限功能数据。不得声称统计显著或业务 SOTA。多 seed 和真实数据后续另行批准。

**候选补充（待确认，批准前不进入必需矩阵，不改变上述单种子决定）**：

- `flat_summary`（候选变体）：每个动态组的 masked mean／sum 与 observed coverage 作为外生通道，连同 target 历史、日历与同一已知未来协变量，由仓库已有的 TiDE 或 TimeXer 类结构做直接 H 输出；同 split、同选择指标、同 epoch 上限。用途是区分“内部信息收益”与“GSLA 结构收益”：full 与 total_only 的差只说明前者。拟只在 AIDC 一天 profile 运行；需先在 `config.py` 的 variant 清单与本节登记后才可执行。
- M1/M2 成对断言的多 seed（候选）：拟用 seed 2023/2024/2025，要求成对区分方向一致，报告均值与极差，仍不做显著性结论；预算增加约 3 倍，仅限机制任务。

<a id="fit-thresholds"></a>

## 6. 小集拟合阈值与设备 smoke

- 专用拟合测试：使用机制任务指定的无噪声机制小集，dropout=0、lr=0.003、最多 1000 update。A/B 分别报告初值和最终值；若初始标准化 MSE<=1e-4，只要求最终仍<=1e-4，否则要求最终<=1e-4 或不高于初值的 20%。不强迫已近零的 B 再按比例下降。
- 上述是预先声明的候选数值门槛，不是实测可达性声明。机制成对区分还必须通过独立任务断言，不能只拟合总体均值。首次实跑未达门槛时区分语义错误／优化限制并记录失败，不删除断言或追改成通过；预算或阈值修订需先记录依据。
- 完整 CPU smoke 使用 [配置定义与设计实例](implementation.md#configuration) 配置与所有阶段；MPS smoke 使用同一 schema，但 A=1、B-short=1、B-medium=1、B-full=1 epoch，batch_size=2，完整 H，单独 run_id。缩短预算明确标记，不用于精度比较。
- synthetic test 输出有限值和正确时间索引；forecast 数据包没有未来目标，重载同一 best 完成新实体请求。

<a id="base-fixture"></a>

## 7. 周期 base：只验流程

下面保留 R2 周期夹具，仅做 baseline/oracle、数据流程和对照代码 smoke；它的季节性 naive 理论上可零误差，**不得单独作为机制验收或 full 模型拟合有效性的证据**。生成器使用 NumPy `default_rng(2023)`，时间轴 320 个小时，从 `2020-01-01T00:00:00Z` 开始，所有值为 synthetic arbitrary_unit。

- 6 个 assets、2 个 zones、1 个 site；assets 索引 0..2 归 zone0，3..5 归 zone1；两个 zone 属于同一 site。
- asset capacity 为 `1 + i/10`，kind 为偶／奇两类；zone capacity 分别 1、2；site capacity 为 3。
- `scheduled_activity[t]=(1+sin(2*pi*t/24))/2`，它是生成器预先定义的计划，不从未来观测推得；记录 published_at 不晚于对应 origin。
- `utilization_i[t]=(1+sin(2*pi*t/24+i/3))/2`；`load_i[t]=capacity_i*(1+utilization_i[t])+0.1*scheduled_activity[t]`。
- `temperature_j[t]=20+2*sin(2*pi*t/24+j)`。
- `target[t]=sum_i(load_i[t]*(1+0.1*temperature_parent(i)[t]))+0.2*scheduled_activity[t]`；不把父级容量或其他层级功率重复相加。
- 原始 target 先由完整生成值计算，再用 RNG 为动态观测施加 5% 缺失，target 保持有效；缺失用于检验 mask，不表示实际物理过程。可另生成全缺失、坏关系、迟到记录测试变体。
- 静态数据、关系在起点已发布；observations.available_at=event_time。版本／迟到测试另用显式夹具修改，不能混成现实可用性结论。

新增实体请求分为有历史与无历史：以 origin=最后观测时刻，增加 `asset_new`（capacity=1.6、kind=even、归 zone1），一份提供同公式但新相位的历史，一份观测 mask 全 false；不把新实体加入训练统计。另生成 `zone_new + asset_new_parent` 请求验证新增父子节点。所有请求仍须提供历史 target，不能用未来 target。


base 不满足机制或多站点验收；各任务包组织和分区以以下各节为准，所有正式数据仍须标 synthetic。

<a id="mechanism-tests-2"></a>

## 8. 机制任务的数据组织与统一协议

<a id="mechanism-tests-2-1"></a>

### 8.1 任务目录和配置

生成器 `--suite all` 产出：

- `base/`：原周期 smoke；
- `mechanisms/M1/` 至 `mechanisms/M4/`：独立 schema、episode、完整 train/valid/test 配置；
- `mechanisms/M5/`：数据契约和数值边界夹具，不训练独立模型；
- `mechanisms/suite.json`：任务 ID、数据包、配置路径、预期断言、split 与 family 清单；
- `new_entity_cold/` 等请求：供 base 入口 smoke，不代替 M4。

R4 的 `--suite all` 另输出 `aidc/` 及各小时级完整 profile，按[AIDC 双轴切分](#synthetic-aidc-4-3)组织，不套用下面 M1–M5 的单 origin family 规则。

每个 family 是一组只在指定因素上不同的成对或多变体 episode；同 family 全部进入同一 split。每任务采用 48 个 train family、16 个 valid family、16 个 test family。编号 0..47/48..63/64..79，split 规则在 manifest 中写明，不能依据结果重新分配。

RNG 为 `default_rng(SeedSequence([2023, task_number, family_number]))`。模型训练种子仍为 2023。不同 family 独立采样，ID 前缀不同；family 编号、任务编号、文件路径均不输入模型。

每 episode 包含 L=24 个历史位置与 H=6 个未来标签，只有一个 origin；不跨 episode 切窗。各 family 使用相同示意时间轴，避免从时间戳推断目标差异。生产 forecast 包只含历史与原点已知条件，标签另存于测试包。

机制配置使用 `split_mode=episode_family`、`split_index=null`，collection manifest 记录 family split。配置的 groups、known_future、target_name、freq 和 L/H 必须与该任务 manifest 精确一致；生成器保存完整配置而非运行时修改 base 字段。对每个 pair 变体创建独立 episode id、共用 family_id。

<a id="mechanism-tests-2-2"></a>

### 8.2 避免伪造信息差异

- 所有目标先由生成规则计算，再施加输入缺失；填充值不能反向改变 target。
- 不把 simulator hidden state、family ID、未来真实动态值作为模型输入。
- 各任务单独拟合训练 scaler，不使用 valid/test 统计；属性词表只来自训练任务。
- M1/M2 成对版本的 target_history 必须逐元素相同，不能让模型从历史总负荷直接区分本应检验的机制。
- 变体 schema 的全部字段、关系和未来列在该任务 train/valid/test 间固定。不同任务可以有不同 schema，分别训练 checkpoint；不宣称同一模型跨任务零样本迁移。
- 先运行公式 oracle，验证成对输入“相同部分相同、不同部分确实导致不同目标”，才允许用来判断模型。

<a id="mechanism-tests-3"></a>

## 9. M1：保留同一实体的时间轨迹

<a id="mechanism-tests-3-1"></a>

### 9.1 构造

组 assets 有两个动态节点（仅 `load`），zones 为一个静态父节点；所有实体属性相同，两个 asset 归同一 zone。

基础历史：前 L-1 步恒为 `[1,3]`；origin 时甲为 `[2,4]`，乙为 `[4,2]`。逐时刻无序集合、sum、count、属性、关系与日历完全一致。每个 family 采样正尺度 a∈[0.5,2]、偏移 b∈[0,2]，把每个历史数值变成 b+a*x；同一 family 两个版本共用 a/b，a/b 不额外作为属性输入。

未来逐实体规则：

\[
x_{i,s+1}=x_{i,s}+\max(x_{i,s}-x_{i,s-1},0),\quad y_s=\sum_i x_{i,s}.
\]

没有外部未来协变量，K=0。无 mask 缺失。静态关系固定。

<a id="mechanism-tests-3-2"></a>

### 9.2 断言

- 原始基础对的下一总量分别为 8、9；生成器实际核算并保存，不手填预测。
- 将整段实体轨迹连同 ID／关系一致重排，预测不变。
- 只交换 origin 时刻数值并保持 ID 不变，代表不同轨迹；模型必须具有区分它们的参数路径。
- `no_entity_history` 在该对上的每帧不变池化表示应相同，构成结构性负对照。full 不要求任意初始化自动正确，但受控历史编码与实际小集拟合必须能够区分。

小集：train 最前 8 个 family 的成对样本，目标为全部 H。A 与 B-full 按 [小集拟合阈值与设备 smoke](acceptance.md#fit-thresholds) 训练；每对预测差与真实差同号，并且每个预测值误差小于该对相应 horizon 目标差绝对值的一半，才算不再把两种情况压成一个答案。

valid/test 的同项指标完整报告；首轮不以全体泛化胜出为交付硬门槛，但必须报告成对区分率与误差，不能只报 pooled MSE。

<a id="mechanism-tests-4"></a>

## 10. M2：归属关系必须承载可识别信息

<a id="mechanism-tests-4-1"></a>

### 10.1 构造

assets 两个动态节点，zones 两个静态父节点，属性 `gain` 分别为 1、2；两个子节点属性相同。前 L-1 步资产 load 都为 1，origin load 为 `[1,3]`。

两个版本在 origin 前使用相同的一一归属；在 origin 时，甲保持 asset0→zone0、asset1→zone1，乙交换父级。两边每个节点的入／出度相同，属性、数值及时间相同，唯一区别是端点配对。该测试也验证 R3 支持历史最后一步的关系切换。

目标采用一步滞后关系：

\[
y_s=\sum_i x_{i,s-1}\,gain_{parent(i,s-1)}.
\]

origin 前的 load 相同，因此全部 target_history 相同；未来 asset load 固定为 origin 值，未来关系固定为 origin 关系。基础下一目标甲为 7、乙为 5。family 采样正 a∈[0.5,2] 乘全部 load，以避免仅记住一个数值组合。

第一条历史 target 所需的前驱状态取 x[-1]=x[0]、parent[-1]=parent[0]，只用于生成器初值，不作为额外模型历史；两版本采用相同初值。

<a id="mechanism-tests-4-2"></a>

### 10.2 断言

- 单看节点集合、属性集合、度数、历史 target，两个版本不可区分；加入端点关系后可区分。
- 数据层必须保留历史旧边和 origin 新边，不能把新关系回填到全部历史。
- 关系层受控消息遵循接收行／发送列定义；这是局部层测试，不宣称整个模型只沿归属传播。
- full 的前 8 个 train family 小集拟合采用 M1 的成对符号／半间距标准。
- `no_relation_all` 不可消费度数、父级属性聚合、关系质量权重等旁路；在这种成对构造下应表现为无法同时恢复两个不同答案。只移除关系消息的 `no_relation_message` 独立报告。

<a id="mechanism-tests-5"></a>

## 11. M3：未来计划路径与状态积累

<a id="mechanism-tests-5-1"></a>

### 11.1 构造

assets 一个动态节点、zones 一个静态父节点，关系固定。状态即已观测 load，y=x；不是隐藏在数据文件中再偷偷输入模型的变量。

\[
x_{s+1}=0.8x_s+u_{s+1}.
\]

每个 family 的历史由独立均匀计划 u∈[0,2] 和 x 初值∈[0,10] 顺序生成。历史已过去计划不用额外输入，因为历史 x 已观测。未来 u 是起点前已公布的计划，存为 `scheduled_activity`。

成对未来路径甲全为 0，乙只在第一步加 5，其余为 0；target step2 的直接协变量相同，但状态因第一步计划不同而不同。两个版本历史、属性、关系、原点完全相同。另保留未来计划随机脉冲版本作 valid/test，计划不是日历确定性函数。

基础检查以 x_origin=10 为例：用上述递推实际计算第二步目标；不得沿用旧评估中系数为 1 的示例数字。

<a id="mechanism-tests-5-2"></a>

### 11.2 断言

- direct、A 和 AR 都收到相同 `[B,H,K]`、mask 和计划发布版本，且均有从 u[:,0] 到 step2 预测的计算路径。
- 将第一步计划改成晚于 origin 才发布，数据层必须屏蔽，不能在模拟第二步时刷新计划。
- 未知 future target 的改变不能影响 forecast；已知未来计划的改变可以影响任何合法预测步，不与历史因果 mask 测试混淆。
- 共享状态的 teacher 顺序编码与一次连续编码一致；production 没有 teacher 输入。
- full 与 direct 的前 8 个 family 小集均通过拟合标准；不要求 AR 优于 direct。若 direct 看不到前缀计划，判接口失败，而不是记录为“AR 的性能优势”。

<a id="mechanism-tests-6"></a>

## 12. M4：规模变化与新实体

<a id="mechanism-tests-6-1"></a>

### 12.1 构造

同一任务 schema 使用 assets(load)、zones(gain) 与固定属性 capacity。训练 family 的 asset 数按编号循环取 2、4、6；valid/test 取 3、5、7。父节点数量按 family 奇偶取 1 或 2，gain 按节点取 1、2；每个 asset 用 RNG 均匀选择父节点，capacity∈[0.5,2]。

\[
x_{i,0}=capacity_i,\quad x_{i,s+1}=0.8x_{i,s}+0.2capacity_i(1+u_{s+1}),
\]
\[
y_s=\sum_i gain_{parent(i)}x_{i,s}.
\]

u 为非周期的已知 Bernoulli 计划（概率 0.5），origin 后的计划已在 origin 公布。训练也包含 whole-entity history 缺失：每 family 两个版本共享同一物理数据，一份全观测，一份屏蔽一个随机 asset 的全部历史观测，属性、关系和真实 target_history 保留；缺失不改变物理目标。

新 ID 不在训练表中，新节点数不同，但属性与关系类型一致。另构造“新实体恰在 origin 生效”的请求：此前 present=false、origin 时 present=true 且无观测，其状态按生成规则的初值 capacity 定义；target_history 在 origin 真实反映激活后的总量，不能复用旧系统 target 装作一致。

<a id="mechanism-tests-6-2"></a>

### 12.2 断言

- 相同 checkpoint 不新增参数、不重拟合 scaler，就能处理不同数量、未知 ID、新父子节点、有／无历史观测。
- 静态属性、数量、归属边都存在真实消费路径；用受控参数验证不是悄悄丢弃新节点。
- 整段重排与 ID 重命名保持数值预测；new_entity 元数据允许因训练溯源集合而变化。
- cold_start、unknown_category、orphan 标记正确。
- 训练规模与测试规模分别报告，不能以 shape 成功宣称外推精度成功。测试输出误差和随数量变化趋势，但不设硬单调性，因为当前模型没有物理单调约束。
- 新实体启动的确切未来状态通常未知，本任务刻意采用可确定初值，只证明此受控条件；真实冷启动精度仍未知。

<a id="mechanism-tests-7"></a>

## 13. M5：缺失、版本、质量与数值边界

不另训练模型，从 M1–M4 及独立数值夹具执行：

- 比较全观测、部分缺失、整个组近 L 步全缺失、未生效、padding；每种语义有单独断言。
- 质量测试只调用 q helper：保持每组 observed 比例，复制单元扩大规模，q 不发生指数衰减。可测试大于模型节点上限的 mask 向量，但不得据此声称大规模模型 forward 已验证。
- 一组 q=0 不删除另一组 q>0 的辅助监督；所有 q=0 时只剩主损失；整 batch 无主目标跳过，整轮无主目标报错。
- variance 从 0 到较大值扫描，`1/(variance+0.01)` 连续、有限、不大于 100；近常量维不会突然扩大百万倍。
- 统计每个训练 episode/origin 只采最后 z 一次；不重复计入窗口内部时间点，valid/test 和未来 teacher 不参与统计。
- horizon 有效数量不同，分别核算 micro 与 macro；验证选优必须用全 split macro，不平均 batch loss。
- 父级在历史中切换，逐时刻使用对应边；发布晚于 origin 的边或计划不可见。未来已知结构变化明确拒绝，不错误声称能自动模拟变化。
- 安全坏输入：ID 重复、悬空／跨站边、违反该类型无环约束的循环、错误单位、无穷值、未知 schema、路径越界、对象 NPZ、维度超过 limits；必须明确失败，不能静默截断或重拟合。R4 另断言不同关系类型合并形成的合法环不被误拒绝。

<a id="mechanism-tests-8"></a>

## 14. 机制任务运行范围、预算与判定

<a id="mechanism-tests-8-1"></a>

### 14.1 必需运行矩阵

避免盲目执行任务×所有变体的笛卡尔积，采用有针对性的矩阵：

| 任务 | 必需路线 |
| --- | --- |
| 周期 base | persistence、seasonal_naive、full 的流程 smoke；不用于证明架构优势 |
| M1 | full、no_entity_history、direct |
| M2 | full、no_relation_message、no_relation_all |
| M3 | full、direct、no_latent_loss、total_only |
| M4 | full、direct、no_spatial、total_only；同权重新实体请求 |
| M5 | helper／接口／数值测试，不训练独立模型 |

各任务的前 8 个 family 小集先做机制拟合；之后再按实施规格的 A/B epoch 上限运行任务 train/valid/test 并报告单种子结果。机制拟合与正式任务训练使用不同 run_id，不把在小集上训练的权重当正式 test 模型。

<a id="mechanism-tests-8-2"></a>

### 14.2 候选验收阈值的边界

小集采用 [小集拟合阈值与设备 smoke](acceptance.md#fit-thresholds) 的 1000 update 预算、绝对误差或相对下降标准；M1/M2 另需成对区分。它们是实现前声明的标准；实际可达性与失败证据见实施规格 §12，不在此重复状态。

- 失败必须区分：信息路径缺失、错误标签／mask、梯度问题、表达能力限制、优化预算不足。
- 不能为交付临时删掉成对断言，或在看到 test 后调整数据生成器让模型更容易。
- 如确定是预算／优化限制，记录证据并申请修订验收，不伪称测试通过；不强制真实业务或对照胜出。
- 同权重 oracle teacher 诊断单列，永不用于选模型、早停或对照排名。

<a id="mechanism-tests-9"></a>

## 15. 机制任务产物与证据状态

每任务输出生成器版本、schema、split-family 清单、原始包校验值、oracle 检查、resolved_config、实际训练预算、每模型 per-horizon 指标和机制断言结果。M4 单列未见节点数量及冷启动情况，M5 单列坏输入预期异常。

`suite.json` 只定义任务，运行后 `suite_result.json` 记录每任务 `passed/failed/not_run`、命令、退出码、artifact 路径；只运行部分任务时总结果不能是 passed。

当前运行证据见实施规格 §12。独立公式探针与文档校验的历史边界见[审查教训](history.md#review-lessons)，原报告可从其中的 Git 入口恢复；这些不证明模型通过验收。

<a id="aidc-profiles"></a>

## 16. AIDC horizon profiles

| profile（示意） | Δ | L | H | 业务含义 |
| --- | --- | --- | --- | --- |
| `aidc_hourly_short` | 1 小时 | 48 | 6 | 历史两天，预测未来六小时 |
| `aidc_hourly_day` | 1 小时 | 168 | 24 | 历史一周，预测未来一天 |
| `aidc_hourly_multiday` | 1 小时 | 168 | 72 | 历史一周，预测未来三天 |
| `aidc_15min_day`（后续容量目标） | 15 分钟 | 672 | 96 | 历史一周，预测未来一天 |

前三个为模拟验收拟议 profile，各自完整配置、训练／验证与 checkpoint；不拿 H=6 的权重直接声称可预测 H=72。每次正式请求 H 等于 checkpoint 的训练 H；展示前几小时可在完整 H 推理后截取，不能裁剪 FuturePathEncoder 的已知条件后声称同一结果。

15 分钟 profile 展示接口如何表示业务范围，不纳入初始全链资源验收。它的 L/H 已超出 R3 限额，需要独立容量测试和授权后启用，不能只改上限宣称完成。

<a id="synthetic-aidc-2"></a>

## 17. AIDC 可复现生成顺序

拟议生成器为 `scripts/dev/generate_gslaformer_fixture.py` 的 `--suite aidc` 分支；不另起不兼容的数据格式。`--suite all` 包括 base、M1–M5 和 AIDC 任务。所有包保存 `data_kind=synthetic`、生成器版本、seed、参数、拓扑与观测层配置的 fingerprint。

顺序固定：

1. 生成站点类型、设备注册表、测点映射和 typed relations。
2. 在完整内部时间轴生成任务计划及真实执行偏差、天气过程。
3. 顺序生成服务器功耗、配电分支功率与损耗、热状态和暖通功耗。
4. 从互不重叠的用电分量形成设施总需求，再派生多级计量值。
5. 施加传感器误差、缺失、延迟、采集粒度及版本发布规则。
6. 生成标准输入包、独立标签、无标签请求包和 oracle 审计文件。
7. 固定站点／时间切分后，仅训练部分拟合统计量。生成器不预先输出用全部数据拟合的 scaler。

真值生成与观测腐化分开：缺失传感器不能改变真实设施负荷。模拟器隐藏状态和参数只在审计文件保存，不传给模型；被选作合法静态属性的额定值除外。

<a id="synthetic-aidc-3"></a>

## 18. 负荷与热惯性机制

<a id="synthetic-aidc-3-1"></a>

### 18.1 算力工作负载

每站点设置稳定任务池，包含三种混合需求：连续推理、成批训练、突发任务。计划与实际执行分开：

- 推理：基础负载 + 有界日内波动 + 有界随机扰动；不让全部站点只有同一条正弦。
- 训练：起点预先发布的任务块，可有不同开始时刻、长度、资源量；实际执行允许延迟、提前结束及资源不足。
- 突发：独立随机事件，默认不预知，不能写入 `future_known`。
- 任务放置：按照起点已知的稳定资源池部署关系分配；单次请求内不模拟未来新实体／部署图切换，以保持与首版模型能力一致。

每条任务计划保留发布时间、版本和目标时段。排队量只按当时已到达请求与已经完成的服务更新，不从最终 job log 的总时长反推过去排队情况。

天气预报也必须按 origin 生成版本：只从该 origin 及以前的天气状态与预设季节背景产生未来预测路径，真实未来天气另加入后续创新扰动。不能把未来实测天气原样或加一点噪声后伪称起点已知预报。审计文件分开保存预报误差与真实天气，模型只读该起点合法发布的版本。

<a id="synthetic-aidc-3-2"></a>

### 18.2 服务器功耗

一种首轮可执行的简化规则为：

\[
P^{server}_{i,t}=P^{idle}_i+(P^{rated}_i-P^{idle}_i)u_{i,t}^{\alpha_i},\qquad 0\le u_{i,t}\le1.
\]

其中利用率由任务到达、容量限制与执行偏差形成。参数随站点／设备类型变化；idle_ratio∈[0.15,0.45]、α∈[0.8,1.5] 为**模拟假设**。服务器额定功率示意范围 0.5–2 kW，只用于小规模功能数据，不是对 GPU 服务器实际功率的估计。

机柜、列头柜由各自已登记下游分量生成；其测点与服务器测点可同时存在。增加同口径父级测点不改变真实目标，防止生成器把重复覆盖当成新增用电。

<a id="synthetic-aidc-3-3"></a>

### 18.3 配电与多层计量

简化供电视图是有向无环网络。沿确认的分支自下而上计算；服务器／机柜上游分支有线路损耗，UPS 输出与输入之间用效率规则转换，变压器再加入自身损耗。示意效率区间 UPS 0.92–0.98、变压器 0.96–0.99，只用于模拟。它们不能被复制进生产预测作为现场事实。

双路供电场景显式分配支路份额，份额非负且对同一负载和为 1；冗余电源额定容量不等于双倍实际负荷。待机支路可有独立待机损耗。物理分配份额只用于模拟器守恒核算；模型目前只消费 typed connectivity，不声称已消费边权或完成潮流计算。

有多套暖通系统时，各系统物理耗电先按互斥设备求和，随后加在正确的上游边界，不又在房间／楼宇表处再加一次。

<a id="synthetic-aidc-3-4"></a>

### 18.4 暖通与热惯性

不使用“总负荷恒等于 IT×固定 PUE”作为所有场景。让暖通响应 IT 热源、室外环境、冷却方式和设备运行模式。一个可核算的简化系统是：

\[
T_{t+1}=T_t+\frac{\Delta_h}{C_{th}}\{Q^{IT}_t+UA(T^{out}_t-T_t)-Q^{cool}_t\},
\]

\[
Q^{cool}_t=clip(Q_0+K_p(T_t-T_{set}),0,Q_{max}),\qquad
P^{chiller}_t=Q^{cool}_t/COP_t.
\]

`C_th` 为 kWh/°C，`UA`、`K_p` 为 kW/°C，`Q` 为 kW，Δ_h 为小时，量纲一致。COP 按室外条件与部分负载率的有界函数生成，限定正值；该函数是模拟近似，不是设备性能曲线识别结果。

泵／塔示意功率采用待机功率与归一化转速的三次函数组合，分配后进入各设备实体；一次泵／二次泵启停、流量与供回水温分别记录。液冷等未定义物理模型不冒充已仿真，先留为未来扩展。

参数生成要求离散稳定性检查，例如在未饱和区间检查 `0 < Δ_h*(UA+K_p)/C_th < 1`；参数不合格时生成失败，不靠裁剪温度掩盖数值发散。热状态先有预热段，预热段不进入训练／验证／测试。极端温度是数据场景还是数值错误必须在审计中区分。

<a id="synthetic-aidc-3-5"></a>

### 18.5 总目标

\[
y_t=P^{IT}_t+P^{HVAC}_t+P^{aux}_t+P^{distribution\ loss}_t.
\]

各项必须互不重叠、属于同一设施边界，且损耗不已被其他分量重复包含。该式用于模拟器构建真值，**不是网络的硬约束，也不是把所有输入测点求和**。初始数据无储能／发电；未来净购电任务须另建边界。

生成器必须保存组成审计，证明每条下游用电仅在目标中计一次。父表减子表的残差有未计量支路时不称为纯损耗。

### 18.6 首版生成器确定性规则（generator_version=aidc-v1）

以下数值全部是人为定义的功能场景，不来自生产校准。它们在训练前固定，不因模型表现调整；随机流按 §19.1 的五个独立 SeedSequence 使用。每个采样时刻顺序固定为任务、服务器功率、冷却控制、设备功率、总需求、热状态更新；预热与正式段连续生成。

| 子过程 | 固定规则 |
| --- | --- |
| 背景利用率 | 每站相位由任务流 U(0,2π) 一次抽取；`base_t=0.30+0.12*sin(2πt/24+phase)` |
| 训练计划 | 每 12 小时一个块，起点为块起点加离散 U{0,1,2}，持续时间 U{3,4,5,6}，额外利用率 U(0.10,0.35)；计划在块起点之前 24 小时发布 |
| 执行偏差与突发 | 各任务块执行延迟 U{0,1,2}，实际长度为计划长度减 Bernoulli(0.2)，至少 1 小时；每小时 Bernoulli(0.05) 突发，幅值 U(0.1,0.3)，持续 1 小时；突发不入未来条件 |
| 单服务器利用率 | `clip(base_t+实际训练块+突发+epsilon_i,t,0,1)`，epsilon 为独立 U(-0.03,0.03)；不同任务池共享站点背景但独立块／突发；服务器按索引奇偶分到两个稳定池 |
| 任务观测 | allocated_gpu_count 为资源池容量乘当时利用率；active_jobs 为 ceil(allocated/2)；每小时到达量为 ceil(容量*(base+计划块)/2)，服务量为 floor(容量*实际利用率/2)，queue 递推取 max(0,queue+arrivals-service)，throughput 为当时 min(queue+arrivals,service)；先计算吞吐再更新 queue |
| 室外天气 | `T_out(t)=24+6*sin(2πt/24+phase_env)+e_t`，`e_(t+1)=0.85*e_t+U(-0.5,0.5)`，e 初始为 0；预报 `24+6*sin(2π(o+k)/24+phase_env)+0.85**k*e_o`，只依赖起点状态 |
| 服务器参数 | rated_kw∼U(0.5,2)，idle_ratio∼U(0.15,0.45)，alpha∼U(0.8,1.5)，按拓扑流一次抽取；功率按 §18.2，不因 mask 改变 |
| 配电参数 | UPS 效率一次取 U(0.92,0.98)，变压器效率取 U(0.96,0.99)，IT 支路线路损耗为 IT 末端功率的 1%；双路份额固定 0.5/0.5，实际 IT 用电不翻倍 |
| 热控制 | 站点服务器总额定功率为 C，`UA=0.1*C`、`Kp=0.5*C`、`Cth=8*C`、`Q0=0.6*C`、`Qmax=1.2*C`、Tset=24°C、初始 T=24°C，IT 热源 Q_IT=P_IT；单位按 §18.4 |
| COP | `clip(4.5-0.06*(T_out-20)-0.5*(1-Q_cool/Qmax)**2,2,6)`；只用于模拟器，不作为模型偷看的动态输入 |
| 泵／塔与辅助功率 | 转速比例 `r=clip(Q_cool/Qmax,0,1)`；全部泵塔总功率 `0.01*C+0.04*C*r**3`，按在册泵塔额定功率份额分配。辅助功率 `0.03*C*(1+0.1*sin(2πt/24))` |
| 流量与水温 | 每设备额定流量=额定功率*10；实际流量=额定流量*r，供水=7+0.02*(T_out-20)，回水=供水+5*r；均为简化观测方程，不宣称热力守恒求解 |
| 暖通频率与湿度 | 频率=50*r Hz，运行比例=1（待机仍在册）；室外湿度 `clip(0.55+0.1*sin(2πt/24),0,1)`，室内湿度固定 0.5；站点冷机／泵塔同类型设备按额定容量份额分配耗电 |
| 总边界 | `IT_upstream=P_IT*1.01/eta_UPS`；设施总量 `(IT_upstream+P_HVAC+P_aux)/eta_transformer`。distribution_loss 为该总量减互斥末端分量，独立记录并核对非负；房间／楼宇表只是同一真值的覆盖观测 |

站点 i 的服务器数按 `[8,12,16][i%3]`、机柜数按 `[2,3,4][i%3]`、机房数 `1+i%2`、楼宇数 `1+(i//2)%2`，每类暖通设备数 `1+i%2`；实体以索引轮转分配到上级，所有类型、端口、空间级别使用同一固定词表。奇数站使用双路供电；站点 11 额外插入一层配电支路并交换冷却配对，保持相同类型 schema。目标中只有物理末端及损耗计一次，不因加了计量节点改变目标。

每个站点生成完整计量主视图，再从同一物理过程派生场景：随机缺失率为 `[0,0.05,0.2][i%3]`；选择第一个服务器／机柜测点每 168 小时连续离线 `[6,24][i%2]` 小时；到达延迟 `[0,1,3][i%3]` 个间隔。稀疏视图屏蔽服务器动态观测但保留已知资产；不把服务器伪装为不存在。观测误差单独用噪声流生成 `measurement*(1+U(-0.01,0.01))`，目标保持未腐化真值，适用域之外不加噪声。

生成器必须为六组 schema 的每列给出观测来源或 applicable=false，不允许为了避免 scaler 报错而虚构有效列。compute 的 gpu_capacity 使用服务器数量的 8 倍；it_power 的 CPU/GPU 利用率按服务器值或已映射下游均值，online_fraction=1；distribution 的 load_ratio 为功率／额定功率；space 的功率按登记覆盖，温湿度按同站热状态。属性额定值及覆盖面积由拓扑表生成并保存，所有模拟参数随 audit 保存。场景视图共享 family_id 与 split，生成器先跑独立守恒、温度稳定、as-of 和反事实 oracle，再允许模型使用。

<a id="synthetic-aidc-4"></a>

## 19. 站点、观测差异与时间切分

<a id="synthetic-aidc-4-1"></a>

### 19.1 站点场景

预先固定 12 个模拟站点 `aidc_00` 至 `aidc_11`。使用 `SeedSequence([2023, suite_number, site_index, stream_index])` 分别生成拓扑、任务、环境、噪声和缺失流；更改缺失率不改变 underlying target。suite_number 为 AIDC 固定值 6，stream_index 顺序在 manifest 固定为 0–4。

| 站点集合 | 用途 | 场景覆盖 |
| --- | --- | --- |
| 00–07 | 共享训练站点；另保留其未来时段作同站 valid/test | 单路／双路、不同机房数量、训练／推理混合、一次／二次泵组合 |
| 08–09 | 未见验证站点，仅用于选模和配置选择 | 与训练类型相容的新数量、参数和观测稀疏度 |
| 10 | 最终未见测试站点 | 训练类型的新组合，不参与任何选模 |
| 11 | 最终结构分布外测试站点 | 更深供电链、不同冷却服务配对及规模；仍在相同类型 schema 内 |

为避免泄漏，站点 08–11 的任何时段都不拟合模型、scaler 或词表；允许在相应预测起点消费该站历史输入，这不等于用该站标签训练。00–07 与 08–09 的验证结果统一按站点等权选优；10/11 全程隔离。

参数初始网格：服务器数从 8/12/16 中选，机柜数从 2/3/4 中选，机房 1/2、楼宇 1/2、任务池 2、室外观测 1、每类暖通设备 1/2。选择满足映射约束的组合，总节点数不超过 128。服务器可按实际观测方案完全不采集，但保留已知资产时须标明有无测点，不能同时把其观测删除和声称其资产不存在。

训练覆盖所有启用类型和数值特征至少有有效观测；冷门设备不在训练出现时不能仅靠一个类别 UNK 宣称已经学会该物理机制。11 的分布外仅改变已有类型的数量与组合，不加入未经训练的新输入列。

<a id="synthetic-aidc-4-2"></a>

### 19.2 观测场景矩阵

- 全量计量：服务器至进线完整，但多级信息有重复覆盖。
- 稀疏计量：只保留机柜／UPS／楼宇等部分测点，使用真实缺测语义。
- 设备数量差异：同组对象数不同；冷却设备可以缺席某类型，组允许为空。
- 随机丢点：动态观测缺失率从 0/0.05/0.2 中固定选择。
- 连续离线：某个传感器连续 6/24 个小时缺失；与随机缺失单独记录。
- 延迟：到达时间延迟 0/1/3 个采样区间，as-of 构造屏蔽尚未到达数据。
- 计划修订：同目标时刻先后发布不同计划，晚于 origin 的版本不得进入预测。
- 历史关系改变：发生于窗口内时采用逐时刻版本；未来已知改变的请求作为明确拒绝用例。

缺测／延迟反事实使用相同物理过程的 family，并保持同站点、同时间分区，不能一个放训练一个放测试。“缺少某组”须同时做训练期暴露及测试期未见缺失两类，后者的误差只作压力结果，不要求无损。

<a id="synthetic-aidc-4-3"></a>

### 19.3 时间轴与合法窗口

主时间轴 1440 个小时，UTC，从 `2020-01-01T00:00:00Z` 起，另有 168 小时不落入正式样本的预热。正式区间索引：train `[0,864)`、valid `[864,1152)`、test `[1152,1440)`。

`split_mode=site_time`：

- 00–07 可在各自三个时间区间生成对应 train/valid/test；
- 08–09 只在 valid 时间区间产生选模窗口；
- 10–11 只在 test 时间区间产生最终测试窗口；
- 每个窗口全部未来标签必须位于该区间，历史可来自同站更早时间；必须有完整 L 个时间位置；
- 任意窗口不跨站点／episode；split 规则与站点列表写死在 manifest，不由运行结果修改。

三个小时级 profile 分别用 H=6/24/72、L=48/168/168，train stride=1，valid/test stride=H；各 H 单独训练和评价，报告各自样本数量。无标签 forecast 请求从最后观测 origin 生成，未来时间／已知计划可提供，但目标和内部未来观测从请求包剥离。

<a id="synthetic-aidc-5"></a>

## 20. AIDC 运行矩阵与判定

<a id="synthetic-aidc-5-1"></a>

### 20.1 必需最小矩阵

| 实验 | 必需路线 | 验证问题 |
| --- | --- | --- |
| AIDC 六小时 | full、direct、total_only、persistence、seasonal_naive | 多站点基础闭环与输入价值 |
| AIDC 一天 | 上述路线 + no_relation_all + no_spatial | typed 关系及内部交互是否帮助 |
| AIDC 三天 | full、direct、total_only、persistence、seasonal_naive | 完整长 horizon 回灌、资源与误差增长 |
| 新实体／新站无标签请求 | 同一对应 profile 的 full checkpoint | 不增权重、不重拟合、合法 shape 与质量标记 |
| 数据反事实 | 固定已训练 checkpoint + 独立契约测试 | 重复计量、重命名、关系方向、缺失、版本与泄漏 |

相同已知条件同时提供给 full/direct；total_only 不读内部实体，但得到相同历史总负荷、日历与站点级未来条件。naive 的信息能力不同，单独标明。M1–M5 保持原定矩阵，不扩成无目的的全排列。

<a id="synthetic-aidc-5-2"></a>

### 20.2 指标与选优

沿用每 horizon 有效计数、原始量纲 MAE/RMSE、峰值／爬坡指标。新增按 `site_id × site_role × profile × scenario` 分层输出。

AIDC 使用 `valid_site_macro_mse` 选优，其 reduction 由[实施规格](implementation.md#implementation-5-2)唯一定义，不能用各 batch 的 loss 平均替代。验收要求任一必需验证站点／horizon 均有有效标签，否则拒绝选优；与单任务 `valid_macro_mse` 明确区分。

同时报告 pooled micro、站点宏平均与各站单表；可加 NRMSE 诊断，但不能掩盖原始 kW 大误差。NRMSE 定义及不可计算条件见[指标算法](implementation.md#implementation-10-2)。

<a id="synthetic-aidc-5-3"></a>

### 20.3 完成条件与真实迁移边界

**诊断触发门槛（候选，待确认；倍数须实跑前登记，不得看到 test 后调整）**：任一 AIDC profile 上，full 的 test `micro_mae` 超过 `2 × min(persistence, seasonal_naive)`，或 per-horizon MAE 中第 H 步与第 1 步之比超过各 naive 同比值的 2 倍时，该 profile 状态记为 `needs_diagnosis`，必须附 §5 同权重诊断（固定 z、固定 y、teacher z 回灌）定位编码或回灌问题后才能记 passed。它只区分“疑似实现错误”，不是排名或优于对照的要求。

功能验收要求：生成 oracle 通过，全部必需训练／test／forecast／重载和契约断言真实执行，结果有限且时间对齐，错误输入被拒绝，产物可追溯。不要求 full 超过 direct 或 naive，但必须完整报告失败和负面结果。

未见站点上“能运行”只证明结构兼容；误差报告证明在**该生成分布**下的表现。生产迁移所需的真实口径、点表、质量核查与回测边界以[设计 §8](design.md#capability-boundary) 为准，本协议不重复列举。

<a id="synthetic-aidc-6"></a>

## 21. AIDC 资源边界与产物

AIDC profile 初始采用 CPU、batch=2、节点上限 128、历史上限 168、horizon 上限 72。关系分数张量估算仍是 `4*B*L*heads*N_total^2` 字节；该上限与真实峰值内存不是同一个数。三天不截断 rollout 梯度，实际 B-full 内存与耗时必须测量；超限报告失败，不擅自 detach、删除设备或缩短 H。

首轮是缩小的 AIDC 功能场景，**不声称可直接吞入生产中全部服务器点位**。大规模方案需另评估按电力／空间分区的局部注意力、层级读出与稀疏算子；这些不是当前已实现路径，也不以预聚合静默替代用户要求的逐点输入。

所有正式产物进 `results/gslaformer/`；各模式必需文件、AIDC 附加文件（`site_split.json`/`scenario_metrics.json`/`generation_audit.json`/`metadata.json`）与冲突规则由[产物生命周期](implementation.md#implementation-11)唯一维护，本节不复制清单。状态 not_run/failed/passed 分开记账。

<a id="planned-commands"></a>

## 22. 拟议执行命令

以下为已实现入口的调用示例；执行前须生成标准包，使用全新 run_id。命令列在此处不代表其完整验收已经通过，实际记录见实施规格 §12。

```bash
# 契约、模型、训练和集成定向测试
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py'

# 生成明确标记为 synthetic 的标准包与请求
env -u PYTHONPATH .venv/bin/python scripts/dev/generate_gslaformer_fixture.py --output results/gslaformer/fixtures --seed 2023 --suite all

# 完整功能训练；run_id 首次创建后禁止重复覆盖
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode train --gsla_run_id functional_cpu

# 最佳 B-full 模型测试
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode test --gsla_run_id functional_cpu_test --gsla_checkpoint results/gslaformer/functional_cpu/checkpoints/best.pt

# 同权重新实体冷启动 forecast，不含未来标签
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode forecast --gsla_run_id functional_cpu_new --gsla_checkpoint results/gslaformer/functional_cpu/checkpoints/best.pt --gsla_request results/gslaformer/fixtures/new_entity_cold

# 统一合成对照
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode compare --gsla_run_id functional_comparison

# 每个机制任务的完整配置由生成器保存，独立运行并汇总全部任务
env -u PYTHONPATH .venv/bin/python scripts/smoke/compare_gslaformer.py --suite_manifest results/gslaformer/fixtures/mechanisms/suite.json --run_prefix mechanism_acceptance
```

MPS 的解析配置由 smoke 脚本从基准 JSON 生成独立文件，保留差异与完整 resolved_config，不用 CLI 隐式覆盖。模型开发授权不包含安装全局依赖、提交或 push。


## 23. 执行结果与历史证据

正式数据包、训练及留出结果的当前状态统一见实施规格 §12。首轮以所有必需任务、重载和无标签 forecast 均真实完成为准；只跑部分任务总状态不能 passed。正式模型与实验产物写 results/gslaformer，各模式所需文件由实施规格定义。

历史[公式与基础算子探针](history.md#review-lessons)、[文档校验摘要](history.md#r4-validation)不构成本协议的运行证据。未来每次验收记录命令、退出码、配置／数据／代码版本、任务计数、artifact 路径、失败与偏差。修改预算或阈值必须事前说明依据并登记，不能看过 test 后改任务让模型通过。
