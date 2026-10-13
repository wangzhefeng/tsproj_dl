# GSLAformer R2：开发实施规格

> 上位设计：[design.md](design.md)。本文件落实 R2 用户决策 D1–D5。
>
> 状态：实施规格，模型、测试、配置文件和脚本均尚未实现；以下命令是后续开发目标，不是已执行证据。
>
> 当前授权：仅文档完善。实际编码需要后续授权。
>
> 首轮交付：完整模型与合成功能测试；真实业务数据、业务效果、多 seed 结论暂不启动。
>
> 数值性质：本文件的模型尺寸、训练预算和合成数据数值是工程初始配置，不是实测最优值或用户业务参数。

## 1. 范围、决定与完成标准

### 1.1 已确定的用户决定

| ID | 决定 | 对实施的硬约束 |
| --- | --- | --- |
| D1 | 先完成模型和功能测试，真实数据验收暂不启动 | 首轮用显式 synthetic 数据，不因没有业务数据阻塞功能实现 |
| D2 | 新增实体不重训也要预测 | 禁止实体 ID 参数表与固定实体轴线性层，使用共享属性编码与可变节点数 |
| D3 | 通用未来信息接口，有可靠数据再启用 | 支持零列及已声明列的数值、mask、起点版本；真实来源适配不在首轮 |
| D4 | 接受不胜过对照，必须真实报告 | 功能完成不要求胜出，不把合成结果包装成业务收益 |
| D5 | 提供归属表，让关系参与计算 | 先实体交互再池化；full 模式不允许关系表只保存不消费 |

新增实体限于同一系统、既有分组／属性／关系 schema；新实体可无动态历史，但需合法静态输入与关系，结果标注冷启动。新类型、新特征列和跨系统泛化不属于该承诺。

### 1.2 首轮完成定义

全部达成才称“功能实现完成”：

- 数据契约、ID 无关性、可变实体数、关系路径、时间因果、未来条件路径通过测试。
- 合成数据两阶段训练、完整 horizon 自由递推、test、无标签 forecast、模型重载成功。
- 同一个 checkpoint 在新增实体／新增父子节点请求中不改变参数，不重拟合统计量。
- CPU 验证与可用 MPS 的定向 smoke 均有实测证据；MPS 不可用时记录环境阻塞，不能冒充通过。
- 基线和最小消融按同一合成协议运行，保存结果和失败情况，不要求完整模型获胜。
- 原有模型入口定向回归通过，无全局依赖、外部服务或数据库迁移。

没有真实数据实验时只报告 synthetic 功能结论。文档与代码完成度分别记录，不把本规格的存在计入模型完成度。

## 2. 数据包与批次契约

### 2.1 首轮持久化格式

采用标准库 JSON + NumPy NPZ；不为配置增加 YAML 依赖，不使用允许对象反序列化的 NPZ。

一个数据包由以下文件构成，路径相对包根目录解析，拒绝越界引用：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | `schema_version=1`、`data_kind=synthetic`、时区、频率、单位、分组 schema、分区边界、文件校验值 |
| `entities.json` | 分组内唯一 ID、静态数值／类别、属性有效区间；ID 仅为字符串元数据 |
| `relations.json` | `child_group/child_id/parent_group/parent_id/type/valid_from/valid_to/available_at` |
| `observations.npz` | `timestamps`、各组 values／observed／applicable／present／available_at、target／target_observed／target_available_at |
| `known_future.npz` | 每条记录的 origin、target_time、feature index、value、observed、published_at、revision；无协变量时为空记录 |

`timestamps` 为严格递增、等间隔 UTC int64 秒；本地日历使用 manifest 的时区生成。首轮 synthetic 使用 UTC，避免把时间转换问题与模型测试混在一起。

历史动态观测首版一个单元格只存一个值及其到达时间，不自动选择历史修订；真实数据若有版本修订，必须先做 as-of 来源适配，不直接导入最终修订宽表。未来协变量支持同一起点可见的多版本记录，按 `published_at <= origin` 选最近版本，时间相同再按显式 revision 排序；完全同键的冲突记录报错。

关系边是业务归属，不是电气流向。关系 schema 声明允许的组间方向；允许多父级 DAG，但拒绝自环、重复边、循环和悬空端点。源表存子→父，计算时明确生成双向信息边。full 模式每个请求至少有一条有效边；某些孤立实体允许 self 通信，但记录 orphan 数量。

### 2.2 有效期与请求快照

- 每个请求独立解析起点可见的实体／静态属性／关系，实体数量可以不同。
- 历史窗口内同一存续实体的属性和父级不能发生版本切换；训练遇此窗口排除并记录原因，推理拒绝并要求选择有效快照窗口。
- 允许实体在历史中途出现：其生效前 `present=false`，没有观测和消息；生效后才应用该实体的边。新实体没有历史时，其历史观测 mask 全 false，不能回填虚构观测。
- 请求的静态与关系上下文在预测 horizon 内固定。若起点已知 horizon 内存在变更，首版返回不支持，不偷偷忽略。下一起点可用新快照调用同一模型，无需重训。
- latent 标签使用起点快照和同一实体集合，仅未来动态值用作监督；不能读取后来才公布的父级、新节点或静态修订。

### 2.3 标准 batch

记 `G` 为固定分组类型数，`Gd` 为动态组数，`N_g` 为当前 batch 补齐后的组内节点数，`L` 为历史，`H` 为预测长度。组顺序由 schema 固定；节点顺序每次由元数据映射处理，不进入模型位置编码。

| 字段 | dtype / shape | 说明 |
| --- | --- | --- |
| `groups[g].values` | float32 `[B,L,N_g,F_g]` | 变换后有限值；纯静态组 `F_g=0` |
| `groups[g].observed` | bool，同 values | 起点可见的真实观测 |
| `groups[g].applicable` | bool `[B,N_g,F_g]` | 不适用不是缺失，也不是零观测 |
| `groups[g].present` | bool `[B,L,N_g]` | 当时是否存续；缺测实体仍可为 true |
| `groups[g].padding` | bool `[B,N_g]` | true 表示 batch 补齐，不是真实实体 |
| `groups[g].static_numeric` | float32 `[B,N_g,S_g]` | 按组训练统计标准化 |
| `groups[g].static_numeric_mask` | bool，同 static_numeric | 缺失数值属性 |
| `groups[g].static_category` | int64 `[B,N_g,C_g]` | PAD=0、UNK=1，其他来自训练属性词表 |
| `groups[g].entity_ids` | 字符串元数据 | 不传给 embedding lookup |
| `relations` | 每样本边索引与类型 | 从该样本 ID 映射生成，随节点重排同步 |
| `target_history` | float32 `[B,L,1]` | 使用独立 target scaler |
| `target_history_observed` | bool `[B,L,1]` | 真实历史可用性 |
| `history_time` | int64 `[B,L]` | UTC 秒 |
| `future_time` | int64 `[B,H]` | 从起点按频率递增 |
| `future_known` | float32 `[B,H,K]` | 已声明通用连续协变量，`K=0` 合法 |
| `future_known_mask` | bool `[B,H,K]` | 在起点是否可用 |
| `origins` | int64 `[B]` | 最后一个历史时刻 |

类别型未来协变量先由包 schema 展开成已声明数值列；不能在 forecast 临时增加列。真实来源读取器不属于首轮，标准包与合成生成器属于首轮。

监督结构与模型输入分离：

- `Supervision.target`: float32 `[B,H,1]`；`target_mask` 同形。
- `Supervision.future_groups`: 各组 `[B,H,N_g,F_g]` 和对应 mask，仅供冻结编码器构造标签。
- `LatentLabels.z`: float32 `[B,H,Gd,d_z]`，`valid`: bool `[B,H,Gd]`。
- `ForecastInput` 不含上述三项。`forecast()` 不接受 Supervision；测试主动更改标签，预测必须不变。

### 2.4 scaler、缺失与新实体

- 连续观测和静态数值按 `(group, feature)` 在训练分区真实适用观测上统计，不按实体拟合；target 使用独立统计；协变量只使用训练 origin 可见的记录。
- 标准化均值与总体标准差存为 float64 元数据，模型输入 float32。标准差小于 `1e-6` 用 1；完全没有训练观测的已声明数值特征报错，不默默创建假统计。
- 标准化后未观测或不适用位置填 0，同时传 observed／applicable。首版不前向填充，不用未来插值。
- 新实体复用组级 scaler。未知属性类别映射 UNK，不添加新参数；保留数据质量计数，不声称已验证未知类别精度。
- 未声明列、单位变化、新分组、新关系类型报错；同 schema 新 ID 和新合法边必须接受。
- 缺测但存在的实体仍以属性和 mask 参与 attention；padding 和未生效实体不向其他实体贡献信息。

### 2.5 窗口与 split

训练／验证／测试按 target-anchored 协议：`history=[t-L+1,t]`、`labels=[t+1,t+H]`，所有标签落在同一 split；历史可来自更早 split。首轮各 split stride=1，训练形成窗口后可 shuffle。未来未知观测绝不参与 origin 可用性选择。

推理至少提供规则时间轴上的 L 个位置；实体级值可以缺失，不允许少行后偷偷改变 L。请求最后一时刻必须等于 origin。有效目标为零的监督 batch 不优化；整轮无有效主目标报错。

## 3. 模型计算规格

### 3.1 属性驱动的实体 token

每组适配器拼接：标准化动态值、observed、applicable、静态数值及 mask、各静态类别 embedding。使用 `Linear → GELU → Linear` 映射到共同 `d_s`；每个类别属性 embedding 宽度 8，词表来自训练数据。不同组适配器独立，同组跨时间／实体共享。

不使用实体 ID embedding，不加实体序号位置编码。静态属性可以参与动态表示，但不宣称纯动态解耦。可选坐标仅在 schema 有该数值特征时消费。

### 3.2 空间、关系与读出

1. 每组加入独立的可学习摘要 token，执行组内 TransformerEncoderLayer（Post-LN、GELU、FFN=`4*d_s`）。摘要 token 始终有效。
2. 提取真实实体输出，按 schema 拼接各组实体轴。关系 attention 只消费实体 token，不让组内摘要成为未声明的跨组通信边。
3. 关系层采用 `nn.MultiheadAttention` + residual/Post-LN + FFN；attention 的 query 行是接收方，key 列是发送方。self、子接收父、父接收子有效，其他为负无穷；每种业务关系的两个方向各有一个按 head 的可学习偏置。
4. padding／未生效 query 仅开放其 self 作为数值安全通路，其他有效 query 不得读取这些 key；该 query 输出随后置零，不参加池化。这避免全遮罩 softmax 的 NaN。
5. 两层关系交互使用同时更新的上一层状态，不能原位按遍历顺序更新节点。
6. 每组用独立的可学习读出 query 对关系增强实体做一次 cross-attention；补一个可学习空组 token 作为始终有效的 key。拼接 `log1p(有效实体数)`、当前动态观测覆盖率，再经 Linear 映射到 `d_z`、LayerNorm，得到规范 latent。仅动态组生成 latent 目标。

关系端点重命名或同时重排不改变语义。改变父级可能改变消息，但不承诺任何一次重连都改变任意已训练模型的输出。

### 3.3 静态条件 c

独立静态适配器只读静态数值／类别、mask、归属度数、起点有效实体集合，不读动态值。按组对属性表示取 masked mean；另拼静态数值的 masked mean、masked sum、有效数量及 `log1p(节点数)`，每组 Linear→GELU 映射至 `d_z`；拼全部组后投影至 `d_t` 得到 c。空组均值置零，数量为零。

静态链与动态链分别建模块，B 阶段一起冻结。数量／sum 保留规模信息，但不构成总功率守恒或严格单调性保证。

### 3.4 历史 token 与时间编码

动态状态保留 `[B,L,Gd,d_z]`，送时间模型时按固定组顺序展平末两维。每个 token 再拼：

- 标准化总负荷；
- 总负荷 observed、generated 两个标记；
- 每动态组 observed coverage 和 generated 标记。

得到输入后 Linear 映射至 `d_t`。位置编码使用窗口内 `0..L-1` 的固定正弦编码，滑窗重算；日历为本地日内相位与周内相位的 sin/cos 四个数，再经 Linear→`d_t` 相加。日历函数适配任意固定秒级频率，不通过拼接未来真实观测计算。

时间 Transformer 采用 Post-LN、GELU、FFN=`4*d_t`，显式上三角屏蔽未来；最后 token 输出为 h。历史测量缺失不等于时间 padding，本版本每个时间位置都有合法 token。

### 3.5 下一步条件与预测头

条件融合输入是 h、c、目标时刻四维日历、标准化已知未来数值及其 mask、16 维确定性正弦 lead 编码。拼接后 `Linear → GELU → Linear(d_t)` 形成 a。

- 总负荷头：`Linear(d_t,1)`，输出 target 标准化坐标。
- 每动态组 latent 头：`Linear(d_t,d_z)`，输出规范 latent；不加额外 LayerNorm、不在回灌前反标准化。
- 静态组无 latent 头。
- A 阶段 lead 为目标相对历史末端的步数；B 每轮向当前窗口预测下一步，lead=1。
- K=0 时没有外部协变量列，日历仍存在；K>0 但缺失时数值置零、mask=false。

阶段 A 与直接多步对照使用每个目标时刻的相同共享条件融合／负荷头，不新增 H 份互不共享参数。

## 4. 自由递推与新实体请求

### 4.1 单一 rollout 契约

`rollout` 在训练、valid、test、forecast 中复用，仅梯度上下文与监督计算由调用方决定：

1. 编码真实历史为规范 latent，读取独立静态条件；
2. 当前窗口 + 本步目标日历／未来条件预测 y、各动态组 z；
3. 保存本步预测；
4. 追加预测 z 与同一预测 y，generated=true、observed=false、coverage=0；
5. 移除最早 token，重新生成窗口位置编码，进入下一步。

每次循环使用原预测起点可见的 future_known 切片，不在模拟的下一时刻重新读取后来发布的计划。历史 token 质量标记保留；生成值不是缺失值，不能因 observed=false 被预处理器再次清零。

输出为 `ForecastResult.y [B,H,1]`、`z [B,H,Gd,d_z]`、元数据与质量标记。模型内部 y 标准化，落盘 y 逆变换为输入 manifest 的原始单位；z 保持规范坐标，注明非物理量。

### 4.2 新实体的检查

- 同一权重分别预测原请求、新实体有历史请求、新实体无历史请求、新父子节点请求。
- 同一个 checkpoint 参数键与 shape 不变；冻结参数及 scaler 校验值前后不变。
- 新实体标记依据训练实体溯源集合判断，但 ID 不能进入数值推理。ID 重命名可能改变 new_entity 元数据，不得改变 y／z 数值。
- 不要求新增实体使总负荷必然增大；本架构无硬单调性，能预测不等于准确预测。
- 冷启动请求记录 `new_entity_count`、`no_history_entity_count`、`unknown_category_count`、`orphan_count`；不伪装为正常分布精度。

## 5. 标签、损失与缺失规则

### 5.1 latent 标签与统计

加载 A 最佳模型后固定完整空间／关系／读出／LayerNorm／静态链。future_groups 在 `no_grad + eval` 下编码，只作为标签。

首轮采用保守有效性：某未来时刻所有起点在册且存续动态实体的全部适用动态特征均观测到时，非空动态组的 latent 标签才有效。原因是关系与组内 attention 可能传播其他组缺失；不把不完整的联合表示当完整状态监督。缺少观测时仍允许用有效总负荷训练，latent 项屏蔽。

统计每组每维 latent 总体标准差，样本为训练窗口中**不重复的有效历史 origin／timestamp／snapshot 状态**，不按重叠窗口重复加权。只使用训练时可见数据；步骤 B 不更新。`sigma < 1e-3` 的维度用 1 加权并记录 constant_dim，其他用实际 sigma。某组没有有效统计样本时报错。全部维度退化不冒充学到空间状态，报告诊断失败供审查，不自动换模型。

### 5.2 明确的 reduction

- 主损失：本次训练 rollout 范围内所有有效 `[B,step,1]` 标准化残差平方求平均；horizon 等权来自每个有效单元等权，同时记录各 horizon 的计数。
- 每组 latent 损失：有效样本／步／维度的 `(prediction-target)/sigma` 平方平均；再仅在本 batch 有有效标签的动态组之间等权平均。
- 无有效 latent 组：latent_loss=0，记录计数；不除以零，不复制上一 batch 损失。
- 无有效主目标：跳过该 batch 的 optimizer.step，记录原因；整 epoch 主目标计数为零时报错。
- 总损失 `load_loss + 0.1 * latent_loss`；首轮不搜索 lambda，`no_latent_loss` 对照显式设为 0。
- 梯度裁剪范数 1，NaN／Inf loss 或梯度 fail-fast，保存失败状态，不加绕过标记继续。

标签 mask 不能作为 rollout 的 future 输入。生成态没有真实 coverage，不用未来真实 mask 来构造它的输入质量标记。

## 6. 两阶段训练状态机与恢复

### 6.1 阶段与参数

| 阶段 | 可训练模块 | 输入与目标 | 选优／转换 |
| --- | --- | --- | --- |
| A | 空间／关系／静态链、时间链、条件融合、负荷头 | 真实历史，完整 H 直接负荷标签，无 latent loss | 完整 H valid 标准化负荷 MSE；最多 20 epoch，patience=5，选 best_A |
| Freeze | 无 | best_A；统计训练 latent sigma | 保存冻结链 fingerprint、统计、参数清单；不更新权重 |
| B-short | 时间链、条件融合、负荷头、latent 头 | 全自由递推 K=1，latent+负荷监督 | 2 epoch，不早停；每轮监控 H |
| B-medium | 同上 | K=min(3,H) | 2 epoch，不早停；每轮监控 H |
| B-full | 同上 | K=H | 最多 20 epoch，patience=5；只在此阶段选择最终 best |
| Test / Forecast | 无 | 最终 best；test 有标签、forecast 无标签 | 只评估，不用 test 选模型 |

H=1 时三个 B 阶段仍依序执行，前两个只作固定 warmup，最终阶段仍是唯一最终选优来源。新优化器在进入 B-short 时创建，B-medium/full 沿用；不继承 A 的动量。A/B 共同使用 AdamW，learning_rate=0.001、weight_decay=0.0001，不设学习率调度器。

所有层初始种子来自 config。latent 头在模型建立时初始化，但 A 不用于 loss、优化器排除其参数；进入 B 使用该初值，不对其他权重重新初始化。优化器参数列表显式按冻结清单构造。

### 6.2 梯度与表示稳定性

- B 冻结空间链，并在父模型 `.train()` 后仍强制 frozen_modules.eval()；测试参数不变及 dropout 无随机漂移。
- rollout K 步保留所有生成 z 与 y 回灌梯度，不 detach。验证和推理使用 no_grad。
- 不启用 AMP、scheduled sampling、KV cache 或截断反向传播。功能配置容量较小；若资源超限，报告并修订配置，而不是静默改变训练算法。
- 每轮记录主损失、分组 latent loss／计数、梯度范数、latent 标准差、历史负荷依赖诊断；不能仅凭有方差就宣称空间信息有效。

### 6.3 checkpoint 与续训

每个 epoch 完成后保存 last，选择指标改善至少 `1e-6` 才更新 best。保存 `phase/phase_epoch/global_epoch/optimizer/early_stop_counter/best_score/python_rng/numpy_rng/torch_rng/device_rng`。可用设备 RNG 若不能恢复，明确不保证同设备逐位续训；首轮 exact resume 测试在 CPU、workers=0 完成。

仅支持 epoch 边界续训，不承诺 batch 中途逐位恢复。epoch 未结束的中断重新从上一已保存 epoch 执行，并在日志记录。B 阶段恢复必须校验冻结 fingerprint、latent 统计、数据包与模型 schema 一致；禁止只加载模型权重却声称恢复完整训练。

## 7. 文件、接口与入口接入

### 7.1 拟新增文件职责

下表是后续编码目标，不代表文件已经存在。新 package 添加最小 `__init__.py` 与目录 README；无抽象空壳、外部服务或通用插件系统。

| 文件 | 主要职责／公开符号 |
| --- | --- |
| `models/transformer/GSLAformer.py` | `Model`，组合完整模型；`encode_history`、`predict_direct`、`predict_step`、`rollout` |
| `layers/gslaformer/spatial.py` | `GroupFeatureEncoder`、`GroupSpatialEncoder`、`GroupReadout`、`StaticConditionEncoder` |
| `layers/gslaformer/relations.py` | `RelationAttention`、方向 bias 与 mask 生成后的消费 |
| `layers/gslaformer/temporal.py` | `TemporalEncoder`、`ConditionFusion`、确定性时间／lead 编码 |
| `data_provider/spatial_temporal/schema.py` | `GroupSchema`、`ForecastInput`、`Supervision`、`ForecastResult`；形状／类型校验 |
| `data_provider/spatial_temporal/config.py` | JSON 解析、禁止未知键、默认值与参数约束 |
| `data_provider/spatial_temporal/dataset.py` | manifest/NPZ 加载、as-of 选择、窗口、batch collate、关系端点映射 |
| `data_provider/spatial_temporal/scaling.py` | 组级数值／target／协变量 scaler、训练词表及 artifact 校验 |
| `exp/exp_gslaformer_forecasting.py` | `Exp_GSLAformer_Forecast`，复用 Exp_Basic 设备／模型发现，训练状态机与运行分派 |
| `exp/gslaformer_support.py` | 损失 reduction、latent 标签／统计、指标、checkpoint 序列化；不复制旧实验类 |
| `scripts/dev/generate_gslaformer_fixture.py` | 确定性 synthetic 标准包与新增实体请求生成器 |
| `scripts/smoke/gslaformer_functional.json` | §8 的完整功能配置 |
| `scripts/smoke/smoke_gslaformer.sh` | synthetic 生成、train/test/forecast、重载验收 |
| `scripts/smoke/compare_gslaformer.py` | 最小基线／消融运行与统一表格，不推导虚假性能 |
| `tests/test_gslaformer_data.py` | schema、split、as-of、mask、scaler |
| `tests/test_gslaformer_model.py` | shape、因果、关系、重排、新实体、条件消费 |
| `tests/test_gslaformer_training.py` | 冻结、梯度、loss、状态机、resume、过拟合 |
| `tests/test_gslaformer_integration.py` | CLI、产物、重载、synthetic E2E 和 MPS |

模型接口语义：

- `encode_history(inputs) -> EncodedHistory`：规范 z、质量序列和静态 c；不能读 Supervision。
- `predict_step(state, next_calendar, next_known, next_mask) -> StepPrediction`：下一 y 与 z；state 不含标签。
- `rollout(inputs, horizon) -> ForecastResult`：所有预测调用唯一闭环，horizon 必须与 future_time 可用长度兼容。
- `predict_direct(inputs) -> [B,H,1]`：A／直接对照，不回灌。
- `build_latent_labels(supervision, origin_context, frozen_encoder)` 位于实验支持层，不位于生产 forecast 接口。

### 7.2 对现有入口的必要修改

已读现有 `run.py`：它的 setting 拼接依赖普通时序参数，三个运行开关全为零还会触发“最终训练”。GSLAformer 不能只插一个实验类后继续沿用这两个隐式行为。

增加专用 `--gsla_config`、`--gsla_mode`、`--gsla_run_id`、`--gsla_checkpoint`、`--gsla_request`；模式仅为 `train/test/forecast/compare`。`--gsla_checkpoint` 在 train 表示续训，test／forecast 表示权重来源。request 只用于 forecast。

为 GSLAformer 加隔离参数解析分支：预解析发现 `--model GSLAformer` 后走其专用配置与模式校验，不要求虚假的 `--enc_in/--dec_in/--root_path` 参数；其他模型保持当前 parser、required 参数和 run 逻辑。GSLAformer 禁止旧 `is_training/is_testing/is_forecasting` 混用，冲突报错；不扩大修改旧入口的范围。

`run(args)` 对 GSLAformer 先分派并返回，不进入旧 setting 和“全零即最终训练”。`_select_experiment_class` 增加明确路由；专用参数适配成 Exp_Basic 所需 `use_gpu/gpu_type/devices`。模型发现保持 `Model` 入口，不另造注册框架。

配置拥有模型、数据、训练、设备设置；CLI 只提供运行模式、配置路径、run_id、checkpoint 和请求路径，不允许散落 CLI 覆盖模型参数。相对 config 的 data.package 路径以 config 所在目录为基准；results_root 以仓库根目录为基准。生产请求路径由 CLI 指定，不修改训练 schema。

### 7.3 明确错误行为

| 情况 | 行为 |
| --- | --- |
| schema／单位／特征／关系类型不兼容 | ValueError，指出字段和期望，不尝试自动猜测 |
| 关系环、悬空端点、full 请求无边 | ValueError；no_relation 仅显式消融 |
| 新实体 ID、节点顺序变化 | 接受，按请求元数据映射，不改变权重 |
| 未来信息发布晚于 origin | as-of 屏蔽，不使用；冲突版本报错 |
| 推理输入出现标签字段 | ValueError，而非忽略后假装干净输入 |
| checkpoint 缺统计或 schema | 拒绝预测，不现场拟合 |
| 全部动态组为空／全静态配置 | 拒绝；单个空组由 empty token 处理 |
| 单 batch 全无主标签 | 跳过并计数；整 epoch 无主标签报错 |
| 显式指定 MPS 但不可用 | 报错，不把 CPU 回退记作 MPS 成功 |
| 请求含未来关系切换 | 拒绝首版不支持的语义 |

## 8. 首轮功能配置与合成任务

### 8.1 完整配置样例

以下 JSON 将在实施时落到 `scripts/smoke/gslaformer_functional.json`。当前只是文档样例，所有数值是功能测试初值。schema_version 不匹配或未知键必须拒绝。

```json
{
  "schema_version": 1,
  "experiment": {
    "model": "GSLAformer",
    "variant": "full",
    "data_kind": "synthetic",
    "seed": 2023,
    "device": "cpu",
    "results_root": "./results/gslaformer"
  },
  "data": {
    "package": "../../results/gslaformer/fixtures/base",
    "timezone": "UTC",
    "freq_seconds": 3600,
    "seq_len": 24,
    "pred_len": 6,
    "train_stride": 1,
    "valid_stride": 1,
    "test_stride": 1,
    "split_index": [0, 192, 256, 320],
    "batch_size": 8,
    "num_workers": 0,
    "target_name": "synthetic_total",
    "target_unit": "arbitrary_unit",
    "groups": [
      {"name": "assets", "dynamic": ["load", "utilization"], "static_numeric": ["capacity"], "static_category": ["kind"]},
      {"name": "zones", "dynamic": ["temperature"], "static_numeric": ["capacity"], "static_category": []},
      {"name": "sites", "dynamic": [], "static_numeric": ["capacity"], "static_category": []}
    ],
    "relation_types": [
      {"name": "asset_in_zone", "child_group": "assets", "parent_group": "zones"},
      {"name": "zone_in_site", "child_group": "zones", "parent_group": "sites"}
    ],
    "known_future": ["scheduled_activity"]
  },
  "model": {
    "spatial_dim": 32,
    "latent_dim": 32,
    "temporal_dim": 64,
    "spatial_heads": 4,
    "relation_heads": 4,
    "temporal_heads": 4,
    "spatial_layers": 1,
    "relation_layers": 2,
    "temporal_layers": 2,
    "dropout": 0.1,
    "category_embedding_dim": 8,
    "lead_dim": 16
  },
  "training": {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "clip_grad_norm": 1.0,
    "latent_weight": 0.1,
    "stage_a_max_epochs": 20,
    "stage_a_patience": 5,
    "b_short_epochs": 2,
    "b_medium_epochs": 2,
    "b_full_max_epochs": 20,
    "b_full_patience": 5,
    "min_delta": 0.000001,
    "amp": false,
    "truncate_rollout_grad": false
  },
  "evaluation": {"selection_metric": "valid_load_mse", "seasonal_period": 24, "compare_seed": 2023}
}
```

固定规范：FFN 倍数=4、Post-LN、GELU、组级 scaler、两层关系注意力、窗口正弦位置编码、严格 latent 标签有效性，不额外开放不必要开关。维度须整除 head 数；L/H 正整数；至少一个动态组；组名和特征唯一；variant 限 §10 名单。用户无需审批每个技术默认值，正式数据阶段再制定业务配置。

### 8.2 确定性合成包

生成器使用 NumPy `default_rng(2023)`，时间轴 320 个小时，从 `2020-01-01T00:00:00Z` 开始；周期函数先生成到预测末端供已知条件使用。数据全部标记 synthetic，单位为 arbitrary_unit，不冒充真实机房功率。

- 6 个 assets、2 个 zones、1 个 site；assets 索引 0..2 归 zone0，3..5 归 zone1；两个 zone 属于同一 site。
- asset capacity 为 `1 + i/10`，kind 为偶／奇两类；zone capacity 分别 1、2；site capacity 为 3。
- `scheduled_activity[t]=(1+sin(2*pi*t/24))/2`，它是生成器预先定义的计划，不从未来观测推得；记录 published_at 不晚于对应 origin。
- `utilization_i[t]=(1+sin(2*pi*t/24+i/3))/2`；`load_i[t]=capacity_i*(1+utilization_i[t])+0.1*scheduled_activity[t]`。
- `temperature_j[t]=20+2*sin(2*pi*t/24+j)`。
- `target[t]=sum_i(load_i[t]*(1+0.1*temperature_parent(i)[t]))+0.2*scheduled_activity[t]`；不把父级容量或其他层级功率重复相加。
- 原始 target 先由完整生成值计算，再用 RNG 为动态观测施加 5% 缺失，target 保持有效；缺失用于检验 mask，不表示实际物理过程。可另生成全缺失、坏关系、迟到记录测试变体。
- 静态数据、关系在起点已发布；observations.available_at=event_time。版本／迟到测试另用显式夹具修改，不能混成现实可用性结论。

新增实体请求分为有历史与无历史：以 origin=最后观测时刻，增加 `asset_new`（capacity=1.6、kind=even、归 zone1），一份提供同公式但新相位的历史，一份观测 mask 全 false；不把新实体加入训练统计。另生成 `zone_new + asset_new_parent` 请求验证新增父子节点。所有请求仍须提供历史 target，不能用未来 target。

没有要求生成器在本次文档修改中执行；真正生成后必须保存生成器版本与实际数据校验值。首轮不读取任何真实业务文件。

## 9. 验收测试、阈值与命令

### 9.1 机制测试矩阵

| 测试 | 具体断言 |
| --- | --- |
| 数据边界 | 第一个／最后一个合法窗口、目标归属 split、scaler 不读 valid/test；as-of 迟到记录不可见 |
| 输入缺失 | NaN 被 mask 后变有限值；padding／不适用／未生效行为不同；全组缺失不产生 NaN |
| 因果性 | eval/dropout=0，修改未来 token 不改变更早时间输出 |
| 同步重排／ID 重命名 | 同步重排 values、属性、mask、关系后输出一致；只改 ID 字符串数值不变 |
| 关系方向 | 接收行／发送列与源表一致；两个方向类型不同；受控消息仅经许可边传播 |
| 关系消费 | 在固定、非退化权重和可区分属性夹具上换父级，关系层输出及最终预测变化 |
| 未来条件消费 | 受控权重下只改本步已知条件，两个预测头变化；mask=false 数值变化不影响结果 |
| 泄漏保护 | 更改未来 target／future_groups／晚发布计划，固定权重自由预测不变 |
| 冷启动／新实体 | 不优化、不新增参数，改变 N 后 shape 正确、结果有限、冷启动标记准确 |
| 规范 latent | encoder 标签与预测头坐标一致；sigma 仅影响 loss，不影响同权重 rollout |
| B 冻结 | 外层 train 后冻结链仍 eval，参数与 buffers 不变；新头、时间链有梯度 |
| 回灌梯度 | 第二步主损失能沿第一步生成 latent 与 y 回传；无意外 detach |
| loss reduction | 手算夹具验证有效单元、有效组平均及零分母；通过测试实际核算数值 |
| 课程与早停 | 短课程不会提前停止，最终 best 来自 B-full，H=1 正常 |
| 续训 | CPU workers=0，在 epoch 边界恢复与未中断对应轨迹在容差内一致 |
| 产物重载 | 同设备加载前后预测一致；所有 schema／统计校验生效 |
| 旧入口回归 | 普通模型路由与默认参数不受 GSLAformer 专用 parser 影响 |

CPU 同输入数值一致性使用 `atol=1e-6, rtol=1e-5`；MPS／跨设备比较 `atol=1e-4, rtol=1e-3`。不要求跨设备逐位相同。受控“应变化”测试先证明输入差异到达指定模块，再以差异大于 CPU 容差断言，不依赖任意随机模型恰好敏感。

### 9.2 小数据拟合与 smoke

- 专用过拟合测试：取 16 个无噪声、无缺失训练窗口，dropout=0，lr=0.003，固定种子，A 最多 300 update；最终同 batch 主 MSE 应小于初始主 MSE 的 20%。再固定编码器训练 B-full，最多 300 update，主 MSE 相对 B 初始应同样下降；记录初始／最终真实值。
- 该阈值只检验优化闭环，不要求每个验证／测试集都改进。达不到时定位梯度／标签／配置原因，不移除失败断言冒充通过。
- 完整 CPU smoke 使用 §8 配置与所有阶段；MPS smoke 使用同一 schema，但 A=1、B-short=1、B-medium=1、B-full=1 epoch，batch_size=2，完整 H，单独 run_id。缩短预算明确标记，不用于精度比较。
- synthetic test 输出有限值和正确时间索引；forecast 数据包没有未来目标，重载同一 best 完成新实体请求。

### 9.3 拟议执行命令

以下脚本／配置均待实现；本轮只检查文档语法，不运行这些尚不存在的命令。

```bash
# 契约、模型、训练和集成定向测试
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py'

# 生成明确标记为 synthetic 的标准包与请求
env -u PYTHONPATH .venv/bin/python scripts/dev/generate_gslaformer_fixture.py --output results/gslaformer/fixtures --seed 2023

# 完整功能训练；run_id 首次创建后禁止重复覆盖
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode train --gsla_run_id functional_cpu

# 最佳 B-full 模型测试
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode test --gsla_run_id functional_cpu_test --gsla_checkpoint results/gslaformer/functional_cpu/checkpoints/best.pt

# 同权重新实体冷启动 forecast，不含未来标签
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode forecast --gsla_run_id functional_cpu_new --gsla_checkpoint results/gslaformer/functional_cpu/checkpoints/best.pt --gsla_request results/gslaformer/fixtures/new_entity_cold

# 统一合成对照
env -u PYTHONPATH .venv/bin/python run.py --model GSLAformer --gsla_config scripts/smoke/gslaformer_functional.json --gsla_mode compare --gsla_run_id functional_comparison
```

MPS 的解析配置由 smoke 脚本从基准 JSON 生成独立文件，保留差异与完整 resolved_config，不用 CLI 隐式覆盖。模型开发授权不包含安装全局依赖、提交或 push。

## 10. 最小对照与指标

### 10.1 对照定义

| variant | 改动 | 训练与解释 |
| --- | --- | --- |
| `persistence` | 最后有效总负荷填满 H | 无优化；无有效历史值时报错，不偷用未来 |
| `seasonal_naive` | 使用起点前最近同相位真实 target | 周期=24，缺相位则用最后有效 target，显式记录 fallback |
| `total_only` | 仅 target 历史、日历、同一已知未来协变量 | 不给静态／内部测点；时间宽度和预算一致 |
| `no_spatial` | 组内和关系 attention 均移除，实体共享 MLP + masked mean 读出 | 同一内部输入和规模特征，重新训练；测试交互机制整体价值 |
| `no_relation` | 移除关系层，保留组内 attention 与读出 | 重新训练 A/B，不等同推理时临时删边 |
| `direct` | 完整关系空间编码 + A 式直接 H 输出 | 最多 44 epoch，patience=5；记录实际 update 与训练耗时 |
| `no_latent_loss` | 完整 AR 架构、latent_weight=0 | 从相同 best_A 初始化 B；仅多步主损失训练状态头 |
| `full` | 完整 GSLAformer | A/B 完整运行 |

`total_only/no_spatial/no_relation/full` 的 A/B 最大 epoch 总预算相同，实际早停不同必须记录；no_spatial、no_relation 参数量不同，报告实际参数数量，不声称完全等容量。`no_latent_loss` 与 full 共用同一个 best_A 作为控制变量；direct 与 AR 的训练目标不同，比较是方案差异而不是单一因果证明。

`total_only` 是独立基线，不把零动态组伪装成合法 full 模型：A 使用相同条件式多步负荷监督，B 仅回灌预测总负荷，只有负荷头，没有 latent 标签或辅助损失。`no_spatial` 的实体 MLP 保留相同属性输入，静态条件仍不读动态值；组内实体不经 attention 交换信息。比较脚本生成各变体的 resolved_config，调用各自明确的模型路径，不通过修改 full 模型全局开关偷偷跳过契约。

首轮单种子 2023，结论仅限功能数据。不得声称统计显著或业务 SOTA。多 seed 和真实数据后续另行批准。

### 10.2 指标与输出粒度

- 以 origin／horizon 为评估单位，保留重叠预测；整体 MAE 为全部有效窗口单元绝对误差均值，RMSE 为同一集合平方误差均值再开方，不是各窗口 RMSE 的均值。
- 按 horizon 输出 MAE、RMSE 和有效计数。值已逆 target scaler；不按 latent 单位解释业务误差。
- 峰值幅度误差：只对整个 H 真值完整的窗口，比较预测最大值与真值最大值；时刻误差用最早 argmax 的步差绝对值乘频率。
- 爬坡 MAE：相邻两个预测步都有有效真值时，比较预测差分和真实差分；不把缺失边界拼接成爬坡。
- latent 按组报告有效计数、加权 MSE、原始规范坐标 MSE、训练表示方差；纯静态组无此指标。
- 记录训练 wall time、实际 update、参数量、预测请求耗时与设备；未来性能预算未知，不预先承诺时延。

首轮 y 测试指标好坏不决定功能验收，但异常值、时间错位、无效掩码或缺产物必须失败。

## 11. 产物、版本与运行冲突

每个 run 写 `results/gslaformer/<run_id>/`，不改其他模型目录规范：

- `resolved_config.json`、`schema.json`、`data_manifest.json`、`environment.json`；
- `scalers.json`、`category_vocab.json`、`latent_stats.npz`；
- `checkpoints/best_A.pt`、`checkpoints/last.pt`、`checkpoints/best.pt`（仅 B-full）；
- `metrics.json`、`predictions.csv`、`latents.npz`、`run.log`、`status.json`。

checkpoint 内同时保存推理所需 schema、scaler、词表、latent 统计、冻结清单和模型状态，旁车文件用于审计；校验两者一致，缺项拒绝加载。测试／预测 run 不另行训练，保存所用 checkpoint 的来源和校验值。仅加载用户认可的本地模型 artifact，不从不可信下载反序列化任意对象。

`predictions.csv` 至少含 run_id、data_kind、origin、target_time、horizon（1-based）、y_pred、y_true（forecast 空）、target_valid、unit、new_entity_count、no_history_entity_count。latent 单独 NPZ，保留动态组顺序，禁止把其维度伪装成实际物理特征。

运行状态为 `running/completed/failed/interrupted`，与项目任务 Todo/Doing/Done 不混用。写临时文件后原子替换；全部必需产物可读取且校验通过后才能 completed。

已存在 run_id 一律拒绝新训练；使用明确 checkpoint 续训时，必须处于非 completed 状态且 config/schema/数据一致。test／forecast 必须新 run_id，不能覆盖训练结果。变更节点快照属于新请求，不改变权重 schema；数据 fingerprint 记录新值而不触发“实体必须等于训练清单”的错误限制。

## 12. 开发工作包与进度证据

以下为后续实现顺序，不是本次已执行计划。每个工作包产出可独立检查的功能或测试，不先建立不工作的空壳；完整 GSLAformer 仍以 §1.2 整体验收。

| ID | 工作包／主要文件 | 前置 | 本包验收 |
| --- | --- | --- | --- |
| I1 | schema/config/scaling、合成生成器、数据测试 | 实施授权与 AGENTS 登记 | 标准包生成、JSON/NPZ 契约、split、as-of、scaler、新 ID 映射测试 |
| I2 | spatial/relations、模型入口、模型测试 | I1 | 编码 shape、重排、关系 mask／受控消息、空组、新实体免参数变化 |
| I3 | temporal/fusion/heads/rollout | I2 | 双头一步索引、目标时刻条件、因果、无未来标签输入、多步梯度 |
| I4 | 实验类／支持层、阶段 A/B、训练测试 | I3 | 预训练迁移、冻结、课程、loss reduction、小数据拟合、CPU 恢复 |
| I5 | run.py 隔离路由、smoke、产物、集成测试 | I4 | train/test/forecast/重载/新实体、CPU/MPS、旧入口回归 |
| I6 | 对照脚本、指标、使用说明与治理收口 | I5 | 合成对照真实运行、报告能力边界、无缺失产物、状态更新 |

涉及新增／修改超过八个文件；仅改本功能所需位置。`.gitignore` 对新增测试加精准例外，既有 tests 内容不删除。代码开发前先检查规则与分支漂移，保持当前分支，不自动 commit/push。

每个包的记录必须包含实际命令、退出码、测试计数、产物路径和偏差；任何预算调整／失败原因追加到设计 §12.1，不回改成“原本就如此”。

当前状态：I1–I6 均未实施。用户已确定功能范围，剩余业务目标、真实数据路径、生产预算属于后续业务验证的输入，不是继续追问首轮功能参数的理由。
