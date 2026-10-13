# GSLAformer R4：开发实施规格

> 上位设计：[design.md](design.md)。本文件保留 D1–D5 与 R3 核心计算，落实 R4 AIDC 输入合同和跨站点模拟验收。
>
> 状态：实施规格，模型、测试、配置文件和脚本均尚未实现；以下命令是后续开发目标，不是已执行证据。
>
> 当前授权：仅文档完善。实际编码需要后续授权。
>
> 首轮交付：完整模型与合成功能测试；真实业务数据、业务效果、多 seed 结论暂不启动。
>
> 数值性质：本文件的模型尺寸、训练预算和合成数据数值是工程初始配置，不是实测最优值或用户业务参数。

机制任务见 [mechanism_tests.md](mechanism_tests.md)，AIDC 业务字段见 [input_output.md](input_output.md)，多站点模拟协议见 [synthetic_aidc.md](synthetic_aidc.md)，组件与框架映射见 [integration.md](integration.md)。当前数据／模型 schema_version 为 3；R2/R3 只有设计，不存在模型产物迁移。[review_r3.md](review_r3.md)只保留历史审查证据，不覆盖本版新增合同。

## 1. 范围、决定与完成标准

### 1.1 已确定的用户决定

| ID | 决定 | 对实施的硬约束 |
| --- | --- | --- |
| D1 | 先完成模型和功能测试，真实数据验收暂不启动 | 首轮用显式 synthetic 数据，不因没有业务数据阻塞功能实现 |
| D2 | 新增实体不重训也要预测 | 禁止实体 ID 参数表与固定实体轴线性层，使用共享属性编码与可变节点数 |
| D3 | 通用未来信息接口，有可靠数据再启用 | 支持零列及已声明列的数值、mask、起点版本；真实来源适配不在首轮 |
| D4 | 接受不胜过对照，必须真实报告 | 功能完成不要求胜出，不把合成结果包装成业务收益 |
| D5 | 提供归属表，让关系参与计算 | 先实体交互再池化；full 模式不允许关系表只保存不消费 |

新增实体限于既有分组／属性／关系 schema；新实体可无动态历史，但需合法静态输入与关系，结果标注冷启动。R4 增加多 AIDC 共享训练和同 schema 未见站点冻结推理，必须独立报告留出误差；不承诺新类型、新特征列或真实跨系统零样本精度。site_id 与 entity_id 都不进入可学习身份表。

### 1.2 首轮完成定义

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

## 2. 数据包与批次契约

### 2.1 首轮持久化格式

采用标准库 JSON + NumPy NPZ；不为配置增加 YAML 依赖，不使用允许对象反序列化的 NPZ。

一个数据包由以下文件构成，路径相对包根目录解析，拒绝越界引用：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | `schema_version=3`、`data_kind=synthetic`、episode 清单、时区、频率、单位、分组 schema、分区边界、文件校验值 |
| `entities.json` | 分组内唯一 ID、数值／类别属性、有效区间、available_at、revision；ID 仅为字符串元数据 |
| `relations.json` | `src_group/src_id/dst_group/dst_id/type/valid_from/valid_to/available_at/revision`；业务方向由类型规定 |
| `site.json` | site_id、目标计量合同、站点角色；机制任务使用 synthetic_site，仍不作数值 ID 输入 |
| `measurements.json` | 测点到实体／特征／端口的显式映射、来源单位及转换、冗余策略；纯机制夹具也声明直接生成的映射 |
| `observations.npz` | `timestamps`、各组 values／observed／applicable／present／available_at、target／target_observed／target_available_at |
| `known_future.npz` | 每条记录的 origin、target_time、feature index、value、observed、published_at、revision；无协变量时为空记录 |

manifest 的 `layout` 必须是 `single_episode` 或 `episode_collection`。前者包含上表文件；后者包含 `episodes:[{id,family_id,site_id,split,path}]`，path 指向完整 single_episode 子包。episode_family 模式的 split 限 train/valid/test，同一 family_id 不得跨 split。site_time 模式的 split 为 null，另附 topology_family_id、time_split 与 site_split，按 synthetic_aidc 的双轴协议生成窗口，不能给整个站点套一个时间分区。collection 根不伪造合并 observations，不把不同实体轴强拼成一个时间序列。

`timestamps` 为严格递增、等间隔 UTC int64 秒；本地日历使用 manifest 的时区生成。首轮 synthetic 使用 UTC，避免把时间转换问题与模型测试混在一起。

历史动态观测首版一个单元格只存一个值及其到达时间，不自动选择历史修订；真实数据若有版本修订，必须先做 as-of 来源适配，不直接导入最终修订宽表。未来协变量支持同一起点可见的多版本记录，按 `published_at <= origin` 选最近版本，时间相同再按显式 revision 排序；完全同键的冲突记录报错。

关系 schema 声明 `name/src_groups/dst_groups/acyclic/reverse_message`，允许同组边、多来源和多目标，不把任意类型都解释成父子。业务方向与逐类型合法性见 input_output §3；acyclic=true 的类型单独拒绝循环，不要求所有类型合并后仍为 DAG。拒绝源记录自环、同类型重复边、跨站边和悬空端点；模型再生成 self 与显式反向通信，不把双向消息当物理反向供电。full 每个请求至少一条有效边；孤立实体允许 self，但记录 orphan 数量。AIDC 的 contains 还需按实体 space_level 验证方向；supplies 按端口／设备类型验证，组级白名单不是全部约束。

### 2.2 有效期、历史版本与请求快照

- 每个请求独立解析起点可见的实体／静态属性／关系，实体数量可以不同。
- 历史每个时刻 s 使用 `valid_from <= s < valid_to` 且 `available_at <= origin` 的版本，重叠有效期按可见 revision 选择，冲突报错。属性和父级可在历史内切换；绝不把 origin 的新父级覆盖到全部历史。
- 允许实体在历史中途出现：其生效前 `present=false`，没有观测和消息；生效后才应用该实体的边。新实体没有历史时，其历史观测 mask 全 false，不能回填虚构观测。
- 未来的属性、成员和关系固定为 origin 上下文；若起点已知 horizon 内有变化，拒绝该请求。历史内变化不触发此限制，新起点立即可用最新已知版本，无需重训。
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
| `groups[g].static_numeric_history` | float32 `[B,L,N_g,S_g]` | 历史有效版本，按组标准化；纯常量可在加载时广播 |
| `groups[g].static_numeric_mask` | bool，同 static_numeric_history | 缺失数值属性 |
| `groups[g].static_category_history` | int64 `[B,L,N_g,C_g]` | 历史属性类别，PAD=0、UNK=1 |
| `groups[g].entity_ids` | 字符串元数据 | 不传给 embedding lookup |
| `relations_history` | B×L 个边索引／类型集合 | 历史逐时刻版本；统一历史实体轴，不能独立逐帧重编号 |
| `relations_origin` | B 个边集合 | 历史最后时刻的边，用作固定未来上下文 |
| `target_history` | float32 `[B,L,1]` | 使用独立 target scaler |
| `target_history_observed` | bool `[B,L,1]` | 真实历史可用性 |
| `history_time` | int64 `[B,L]` | UTC 秒 |
| `future_time` | int64 `[B,H]` | 从起点按频率递增 |
| `future_known` | float32 `[B,H,K]` | 已声明通用连续协变量，`K=0` 合法 |
| `future_known_mask` | bool `[B,H,K]` | 在起点是否可用 |
| `origins` | int64 `[B]` | 最后一个历史时刻 |
| `site_ids/target_contract` | B 份元数据 | 站点隔离、目标边界、原始单位和审计；不作学习 ID |

类别型未来协变量先由包 schema 展开成已声明数值列；不能在 forecast 临时增加列。真实来源读取器不属于首轮，标准包与合成生成器属于首轮。

监督结构与模型输入分离：

- `Supervision.target`: float32 `[B,H,1]`；`target_mask` 同形。
- `Supervision.future_groups`: 各组 `[B,H,N_g,F_g]` 和对应 mask，仅供冻结编码器构造标签。
- `LatentLabels.z`: float32 `[B,H,Gd,d_z]`，`quality`: float32 `[B,H,Gd]` 在 `[0,1]`；由 quality>0 导出 valid。
- `ForecastInput` 不含上述三项。`forecast()` 不接受 Supervision；测试主动更改标签，预测必须不变。

历史属性、present 和边采用统一实体轴；origin 静态条件取最后历史位置的属性及 present。未来 teacher 使用 origin 在册集合与固定上下文，不能引入起点不存在的节点。未来标签的 observed 表示离线标签快照的可观测性，不要求 available_at<=origin，否则真正未来标签都会被屏蔽；该标签快照只用于监督，不参加推理输入选择。

### 2.4 scaler、缺失与新实体

- 连续观测和静态数值按 `(group, feature)` 在训练分区真实适用观测上统计，不按实体拟合；target 使用独立统计；协变量只使用训练 origin 可见的记录。AIDC 还必须限于训练站点训练时段，留出验证／测试站点的全部数据不能参与拟合。
- 标准化均值与总体标准差存为 float64 元数据，模型输入 float32。标准差小于 `1e-6` 用 1；完全没有训练观测的已声明数值特征报错，不默默创建假统计。
- 标准化后未观测或不适用位置填 0，同时传 observed／applicable。首版不前向填充，不用未来插值。
- 新实体复用组级 scaler。未知属性类别映射 UNK，不添加新参数；保留数据质量计数，不声称已验证未知类别精度。
- 属性变更记录去重后按训练可见版本统计，不按广播到每个时间格的次数重复拟合；episode 不同源数据分别记录键，归一化总体统计只使用训练 split。
- 未声明列、单位变化、新分组、新关系类型报错；同 schema 新 ID 和新合法边必须接受。
- 缺测但存在的实体仍以属性和 mask 参与 attention；padding 和未生效实体不向其他实体贡献信息。

### 2.5 窗口与 split

训练／验证／测试按 target-anchored 协议：`history=[t-L+1,t]`、`labels=[t+1,t+H]`，所有标签落在同一 split；历史可来自更早 split。首轮各 split stride=1，训练形成窗口后可 shuffle。未来未知观测绝不参与 origin 可用性选择。

该规则对应配置 `split_mode=target_index`，需要严格递增 split_index。机制 collection 用 `split_mode=episode_family`、`split_index=null`，从 manifest 取 family split，并只选各长度 L+H episode 的 origin=L-1；拒绝同时提供 target_index 边界。不要把 base 的 320 行边界复制到 30 行的机制 episode。

AIDC collection 使用第三种 `split_mode=site_time`，有 split_index、site_split 和每站独立时间轴。00–07 的训练／未来验证／未来测试与 08–09 未见验证、10–11 未见测试按 synthetic_aidc §4.3 分配；样本不跨站点。训练采用先均匀选站点再选 origin，epoch 的总抽样数等于所有训练站点合法窗口数（有放回），随机状态进入 checkpoint；valid/test 不抽样、不 shuffle，使用 profile 明确 stride。各 profile 的完整配置与样本数随产物保存。

推理至少提供规则时间轴上的 L 个位置；实体级值可以缺失，不允许少行后偷偷改变 L。请求最后一时刻必须等于 origin。有效目标为零的监督 batch 不优化；整轮无有效主目标报错。

## 3. 模型计算规格

### 3.1 属性驱动的实体 token

每组适配器拼接：标准化动态值、observed、applicable、静态数值及 mask、各静态类别 embedding。使用 `Linear → GELU → Linear` 映射到共同 `d_s`；每个类别属性 embedding 宽度 8，词表来自训练数据。不同组适配器独立，同组跨时间／实体共享。

不使用实体 ID embedding，不加实体序号位置编码。静态属性可以参与动态表示，但不宣称纯动态解耦。可选坐标仅在 schema 有该数值特征时消费。

**池化前实体时序编码是必需路径。** 动态组使用单层单向 `nn.GRUCell(d_s,d_s)`，同组实体共享参数。按统一 ID 轴逐时刻输入适配后的 token，产生 `[B,L,N_g,d_s]`：

- 请求最早位置 hidden=0；present=false 或 padding=true 的状态置零，重新出现从零开始。
- 缺测但 present=true：用填零数值及 mask 输入 GRU，不将状态清零，也不伪造观测。
- 父级／属性变化不重置同一实体的 hidden；空间交互再使用该时刻的边和属性。
- 纯静态组无 GRU，直接用当时属性 token。实体 GRU 不先进行跨实体平均，避免轨迹对应关系丢失。
- GRU 采用单层、无内置 dropout，空间层 dropout 仍由配置控制。每个请求重新初始化，批次之间不传 hidden。

合法不变性是“同一置换作用于该实体整段历史、属性、mask 和关系”。逐帧不同置换会改变实体轨迹，不应断言输出不变。`EncodedHistory.entity_memory` 仅供 teacher 延续编码，生产 rollout 不使用预测 z 反解该 memory。

### 3.2 空间、关系与读出

1. 对每时刻的实体 GRU 输出（静态组为属性 token）加入组内摘要 token，执行组内 TransformerEncoderLayer（Post-LN、GELU、FFN=`4*d_s`）。摘要 token 始终有效。
2. 提取真实实体输出，按 schema 拼接各组实体轴。关系 attention 只消费实体 token，不让组内摘要成为未声明的跨组通信边。
3. 关系层采用 `nn.MultiheadAttention` + residual/Post-LN + FFN；query 行是接收方，key 列是发送方。self、注册的源→目标及 reverse_message 允许的反向通信有效，其他为负无穷；每种业务关系的两个方向各有按 head 的可学习偏置。同端点多类型偏置相加；未连接位置不能因零偏置被误开放。偏置／mask 按 batch、时刻、head 对齐，禁止不同站点间建立边。
4. padding／未生效 query 仅开放其 self 作为数值安全通路，其他有效 query 不得读取这些 key；该 query 输出随后置零，不参加池化。这避免全遮罩 softmax 的 NaN。
5. 两层关系交互使用同时更新的上一层状态，不能原位按遍历顺序更新节点。
6. 每组用独立的可学习读出 query 对关系增强实体做一次 cross-attention；补一个可学习空组 token 作为始终有效的 key。拼接 `log1p(有效实体数)`、当前动态观测覆盖率，再经 Linear 映射到 `d_z`、LayerNorm，得到规范 latent。仅动态组生成 latent 目标。

关系端点重命名或同时重排不改变语义。改变父级可能改变消息，但不承诺任何一次重连都改变任意已训练模型的输出。

允许边约束是关系层局部契约，不是整个模型的隔离保证：组内全连接注意力已经混合了同组信息。关系层禁止 key_padding_mask 再次屏蔽无效 query 的安全 self，必须用统一的加性 mask 处理该情形；无效输出归零后不参与任何有效节点计算。

### 3.3 静态条件 c

独立静态适配器只读静态数值／类别、mask、按类型与方向分别统计的关系度数、起点有效实体集合，不读动态值。按组对属性表示取 masked mean；另拼静态数值的 masked mean、masked sum、有效数量及 `log1p(节点数)`，每组 Linear→GELU 映射至 `d_z`；拼全部组后投影至 `d_t` 得到 c。空组均值置零，数量为零。

静态链与动态链分别建模块，B 阶段一起冻结。数量／sum 保留规模信息，但不构成总功率守恒或严格单调性保证。

### 3.4 历史 token 与时间编码

动态状态保留 `[B,L,Gd,d_z]`，送时间模型时按固定组顺序展平末两维。每个 token 再拼：

- 标准化总负荷；
- 总负荷 observed、generated 两个标记；
- 每动态组 observed coverage 和 generated 标记。

得到输入后 Linear 映射至 `d_t`。位置编码使用窗口内 `0..L-1` 的固定正弦编码，滑窗重算；日历为本地日内相位与周内相位的 sin/cos 四个数，再经 Linear→`d_t` 相加。日历函数适配任意固定秒级频率，不通过拼接未来真实观测计算。

时间 Transformer 采用 Post-LN、GELU、FFN=`4*d_t`，显式上三角屏蔽未来；最后 token 输出为 h。历史测量缺失不等于时间 padding，本版本每个时间位置都有合法 token。

### 3.5 完整未来路径与预测头

`FuturePathEncoder` 读取原 origin 可见的整段 `[B,H,calendar+known+mask]`，线性映射到 d_t，加相对原起点的正弦位置，再用一层双向 TransformerEncoderLayer（heads=temporal_heads、FFN=4*d_t、Post-LN、GELU）生成 p `[B,H,d_t]`。即使某步所有协变量缺失，日历和 mask 仍使该 token 有效。K=0 时仅编码日历。

A/direct 的第 k 步和 AR 的第 k 步都使用 p[:,k]，并能通过该表示访问整段起点已知计划。这里双向 attention 不违反预测因果性；因果限制针对未来未知观测与后来发布的条件，而非已知计划的时间索引。

条件融合输入为历史或回灌 hidden h、静态 c、p[:,k]、相对**原预测起点**的 16 维正弦 lead(k)。拼接后 `Linear → GELU → Linear(d_t)` 形成 a。

- 总负荷头：`Linear(d_t,1)`，输出 target 标准化坐标。
- 每动态组 latent 头：`Linear(d_t,d_z)`，输出规范 latent；不加额外 LayerNorm、不在回灌前反标准化。
- 静态组无 latent 头。
- A/direct/AR 的 lead 均相对原 origin，取 1..H；AR 只改变状态窗口，不重置计划原点。
- K=0 时没有外部协变量列，日历仍存在；K>0 但缺失时数值置零、mask=false。

阶段 A 与直接多步对照使用每个目标时刻的相同共享条件融合／负荷头，不新增 H 份互不共享参数。

路径编码在一次请求中只算一次；训练时缓存的是当前计算图 tensor，不 detach；不同 origin 的计划版本不同，禁止跨请求复用。为了公平，训练 K 步课程仍编码完整 H 条件，但仅对前 K 步计算 loss；这只是使用原本已知的计划，不引入 H 步真实标签。

## 4. 自由递推与新实体请求

### 4.1 单一 rollout 契约

`rollout` 在训练、valid、test、forecast 中复用，仅梯度上下文与监督计算由调用方决定：

1. 按实体编码真实历史并完成空间／关系读出，得到规范 latent、末端实体 memory、独立静态条件；同时编码原 origin 的完整已知未来路径；
2. 当前系统状态窗口 + 对应 p[:,k] 与原起点 lead(k) 预测 y、各动态组 z；
3. 保存本步预测；
4. 追加预测 z 与同一预测 y，generated=true、observed=false、coverage=0；
5. 移除最早 token，重新生成窗口位置编码，进入下一步。

每次循环使用原预测起点可见的 future_known 切片，不在模拟的下一时刻重新读取后来发布的计划。历史 token 质量标记保留；生成值不是缺失值，不能因 observed=false 被预处理器再次清零。

历史 z 只计算一次，窗口移位后不重新运行实体编码器。生产回灌没有实体级生成值，不更新实体 memory；这份 memory 只在训练标签构造的隔离 teacher 路径中使用。两条路径不能互相写状态。

输出为 `ForecastResult.y [B,H,1]`、`z [B,H,Gd,d_z]`、元数据与质量标记。模型内部 y 标准化，落盘 y 逆变换为输入 manifest 的原始单位；z 保持规范坐标，注明非物理量。

### 4.2 新实体的检查

- 同一权重分别预测原请求、新实体有历史请求、新实体无历史请求、新父子节点请求。
- 同一个 checkpoint 参数键与 shape 不变；冻结参数及 scaler 校验值前后不变。
- 新实体标记依据训练实体溯源集合判断，但 ID 不能进入数值推理。ID 重命名可能改变 new_entity 元数据，不得改变 y／z 数值。
- 不要求新增实体使总负荷必然增大；本架构无硬单调性，能预测不等于准确预测。
- 冷启动请求记录 `new_entity_count`、`no_history_entity_count`、`unknown_category_count`、`orphan_count`；不伪装为正常分布精度。

## 5. 标签、损失与缺失规则

### 5.1 latent 标签与统计

加载 best_A 后冻结适配器、实体 GRU、空间／关系／读出／LayerNorm／静态链。在 `no_grad + eval` 下先编码与模型相同的历史，复制末端实体 memory，再按 1..H 消费 future_groups 的数值与 mask。未来属性／present／关系保持 origin 上下文，每一步生成 teacher z；不能每步从零重建 GRU，也不能读取 future target 生成 z。

标签是**带缺失条件的可用信息表示**，不是完整物理状态。允许部分观测，质量只软加权，不设全系统 all-observed 门槛：

1. 对动态组 g 的时刻 s，r[g,s] 为最近 L 步内所有 present、非 padding、applicable 动态单元的 observed 比例；窗口起点早于本请求历史时截断到已编码起点。分母为零时 r=0。
2. V[g,s] 为从 g 的有效实体沿至多两条已启用类型通信边可达的其他动态组集合（按组去重）。r_neighbor 为这些组 r 的算术均值；集合为空取 1。
3. `q[g,s]=r[g,s]*(0.5+0.5*r_neighbor)`；组在 s 为空时 q=0。r 与 q 只由真实 mask／图计算，禁止用可学习 attention 权重提高自己的监督权重。
4. 该 q 是有界质量启发式，不是统计置信度，不覆盖全部组内混合造成的依赖；部分缺失 teacher 仍是噪声目标。记录 q 分布、有效率和缺失率敏感性，不能宣称缺失已完全解决。
5. 未来 q 与 mask 仅进入 loss／诊断，生产生成 token 仍用 generated=true、coverage=0，不泄漏未来观测质量。

latent 统计只取训练 split 的每个唯一 `(episode, origin)` **最后一个历史 z** 一次，并存该请求数据／版本 fingerprint；不收集重叠窗口中的其他 z，也不收集未来 teacher z。重复键内容不一致时报错。各维以 origin 的 q 作为权重计算均值和总体方差；该组至少有两个 q>0 的训练 origin，否则无法估计而报错。B 不重拟合。

辅助维度权重固定为 `w[g,j]=1/(variance[g,j]+0.01)`，连续且不超过 100。近常量维仍有限加权，不突然把权重改为 1。单个组全部维度方差低于 `1e-8` 只告警并保留常量目标；全部动态组均如此才暂停 full 训练、报告 representation_degenerate，交由机制任务定位，不静默改模型。权重只用于 loss，不改预测值与回灌坐标。

### 5.2 明确的 reduction

- 主损失：每个 horizon 先在有效 batch 样本中计算 MSE，再对有有效标签的 horizon 等权平均。没有标签的 horizon 不以 0 混入平均，记录缺项。
- latent：先对维度平均 `w*(prediction-target)^2`；每个组／horizon 的 batch 项为 `sum(q*error)/count(q>0)`，再对该组有效 horizon 等权、对有效动态组等权。分母不是 sum(q)，使低质量标签确实减弱辅助项，而不是被归一化抵消。
- 无有效 latent 组：latent_loss=0，记录计数；不除以零，不复制上一 batch 损失。
- 无有效主目标：跳过该 batch 的 optimizer.step，记录原因；整 epoch 主目标计数为零时报错。
- 总损失 `load_loss + 0.1 * latent_loss`；首轮不搜索 lambda，`no_latent_loss` 对照显式设为 0。
- 梯度裁剪范数 1，NaN／Inf loss 或梯度 fail-fast，保存失败状态，不加绕过标记继续。

标签 mask 不能作为 rollout 的 future 输入。生成态没有真实 coverage，不用未来真实 mask 来构造它的输入质量标记。

验证选优指标为标准化的 horizon-macro MSE：先在整个 split 累积各 horizon 残差和计数，再求各 horizon 均值的平均，不平均 batch loss。最终 H 任一 horizon 无有效 valid 标签时拒绝选优；test 可报告缺项，不能伪称完整 H 指标。合成验收的 valid/test 每步都有标签。

AIDC profile 使用 `valid_site_macro_mse`：先在每站内按上述方法算 horizon-macro，再对所有验证站点等权。A/B 都使用该选择指标，表 §6 中的 valid 指标随 profile 解析。训练 batch 仍按 §5 的 loss reduction，站点平衡通过 sampler 实现，不误称 batch 平均就是 split 选优指标。

## 6. 两阶段训练状态机与恢复

### 6.1 阶段与参数

| 阶段 | 可训练模块 | 输入与目标 | 选优／转换 |
| --- | --- | --- | --- |
| A | 实体／空间／关系／静态链、系统时间链、未来路径编码、融合、负荷头 | 真实历史和完整已知条件，H 直接负荷标签，无 latent loss | 完整 H valid horizon-macro MSE；最多 20 epoch，patience=5，选 best_A |
| Freeze | 无 | best_A；统计训练 latent 方差和有界权重 | 保存冻结链 fingerprint、统计、参数清单；不更新权重 |
| B-short | 系统时间链、未来路径编码、融合、负荷头、latent 头 | 全自由递推 K=1，latent+负荷监督 | 2 epoch，不早停；每轮监控 H |
| B-medium | 同上 | K=min(3,H) | 2 epoch，不早停；每轮监控 H |
| B-full | 同上 | K=H | 最多 20 epoch，patience=5；只在此阶段选择最终 best |
| Test / Forecast | 无 | 最终 best；test 有标签、forecast 无标签 | 只评估，不用 test 选模型 |

H=1 时三个 B 阶段仍依序执行，前两个只作固定 warmup，最终阶段仍是唯一最终选优来源。新优化器在进入 B-short 时创建，B-medium/full 沿用；不继承 A 的动量。A/B 共同使用 AdamW，learning_rate=0.001、weight_decay=0.0001，不设学习率调度器。

`no_latent_loss` 且 K=1 时 latent 头没有主损失梯度，这是图结构必然结果，不能伪造梯度；K>=2 才验证经回灌的梯度。H=1 的该对照记录 latent 未训练／不适用，不计入 latent 能力结论。

所有层初始种子来自 config。latent 头在模型建立时初始化，但 A 不用于 loss、优化器排除其参数；进入 B 使用该初值，不对其他权重重新初始化。优化器参数列表显式按冻结清单构造。

### 6.2 梯度与表示稳定性

- B 冻结实体／空间／关系／读出／静态链，并在父模型 `.train()` 后仍强制 frozen_modules.eval()；未来路径编码器属于可训练预测链，不误冻结。
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
| `layers/gslaformer/entity_history.py` | `EntityHistoryEncoder`，共享 GRU、ID 对齐后的逐实体状态和 present 重置 |
| `layers/gslaformer/relations.py` | `RelationAttention`、方向 bias 与 mask 生成后的消费 |
| `layers/gslaformer/temporal.py` | `TemporalEncoder`、`FuturePathEncoder`、`ConditionFusion`、确定性时间／lead 编码 |
| `data_provider/spatial_temporal/schema.py` | `GroupSchema`、`ForecastInput`、`Supervision`、`ForecastResult`；schema 3、形状／站点／目标／类型校验 |
| `data_provider/spatial_temporal/aidc_mapping.py` | 设备／测点／端口／单位映射及覆盖审计；不猜测现场接线 |
| `data_provider/spatial_temporal/config.py` | JSON 解析、禁止未知键、默认值与参数约束 |
| `data_provider/spatial_temporal/dataset.py` | manifest/NPZ 加载、as-of 选择、窗口、batch collate、关系端点映射 |
| `data_provider/spatial_temporal/scaling.py` | 组级数值／target／协变量 scaler、训练词表及 artifact 校验 |
| `exp/exp_gslaformer_forecasting.py` | `Exp_GSLAformer_Forecast`，复用 Exp_Basic 设备／模型发现，训练状态机与运行分派 |
| `exp/gslaformer_support.py` | 损失 reduction、latent 标签／统计、指标、checkpoint 序列化；不复制旧实验类 |
| `scripts/dev/generate_gslaformer_fixture.py` | base、M1–M5、AIDC synthetic 标准包、oracle 与新增实体／新站请求 |
| `scripts/smoke/gslaformer_functional.json` | §8 的完整功能配置 |
| `scripts/smoke/smoke_gslaformer.sh` | synthetic 生成、train/test/forecast、重载验收 |
| `scripts/smoke/compare_gslaformer.py` | 最小基线／消融运行与统一表格，不推导虚假性能 |
| `tests/test_gslaformer_data.py` | schema、split、as-of、mask、scaler |
| `tests/test_gslaformer_model.py` | shape、因果、关系、重排、新实体、条件消费 |
| `tests/test_gslaformer_training.py` | 冻结、梯度、loss、状态机、resume、过拟合 |
| `tests/test_gslaformer_integration.py` | CLI、产物、重载、synthetic E2E 和 MPS |

模型接口语义：

- `encode_history(inputs) -> EncodedHistory`：规范 z、质量序列、静态 c、按组末端实体 memory；不能读 Supervision。
- `encode_future_path(inputs) -> [B,H,d_t]`：整个请求条件一次编码，保留当前训练图。
- `predict_step(state, path_context_k, origin_lead_k) -> StepPrediction`：下一 y 与 z；state 不含标签或 teacher memory。
- `rollout(inputs, horizon) -> ForecastResult`：所有预测调用唯一闭环，正式请求 horizon 等于 checkpoint 配置 H，future_time／known 同长；若只展示前几步，在完整 H 计算后截取，不改变 FuturePathEncoder 的信息集。课程 K 只限制训练 rollout 的步数，不裁剪原始 H 条件。
- `predict_direct(inputs) -> [B,H,1]`：A／直接对照，不回灌。
- `build_latent_labels(supervision, origin_context, entity_memory_copy, frozen_encoder)` 位于实验支持层，返回 z、quality，不位于生产 forecast 接口。

### 7.2 对现有入口的必要修改

已读现有 `run.py`：它的 setting 拼接依赖普通时序参数，三个运行开关全为零还会触发“最终训练”。GSLAformer 不能只插一个实验类后继续沿用这两个隐式行为。

增加专用 `--gsla_config`、`--gsla_mode`、`--gsla_run_id`、`--gsla_checkpoint`、`--gsla_request`；模式仅为 `train/test/forecast/compare`。`--gsla_checkpoint` 在 train 表示续训，test／forecast 表示权重来源。request 只用于 forecast。

为 GSLAformer 加隔离参数解析分支：预解析发现 `--model GSLAformer` 后走其专用配置与模式校验，不要求虚假的 `--enc_in/--dec_in/--root_path` 参数；其他模型保持当前 parser、required 参数和 run 逻辑。GSLAformer 禁止旧 `is_training/is_testing/is_forecasting` 混用，冲突报错；不扩大修改旧入口的范围。

`run(args)` 对 GSLAformer 先分派并返回，不进入旧 setting 和“全零即最终训练”。`_select_experiment_class` 增加明确路由；专用参数适配成 Exp_Basic 所需 `use_gpu/gpu_type/devices/use_multi_gpu`。明确请求 MPS/CUDA 时先检查可用性，禁止继承基类的静默 CPU 回退。模型发现保持 `Model` 入口，不另造注册框架。

配置拥有模型、数据、训练、设备设置；CLI 只提供运行模式、配置路径、run_id、checkpoint 和请求路径，不允许散落 CLI 覆盖模型参数。相对 config 的 data.package 路径以 config 所在目录为基准；results_root 以仓库根目录为基准。生产请求路径由 CLI 指定，不修改训练 schema。

### 7.3 明确错误行为

| 情况 | 行为 |
| --- | --- |
| schema／单位／特征／关系类型不兼容 | ValueError，指出字段和期望，不尝试自动猜测 |
| 违反类型无环规则、悬空／跨站端点、full 起点无边 | ValueError；跨类型合并形成的合法环不拒绝，无关系仅显式消融／基线 |
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

以下 JSON 将在实施时落到 `scripts/smoke/gslaformer_functional.json`，是已转换至 schema 3 的通用周期 smoke，不是 AIDC 业务配置。六组 AIDC 完整设计样例见 [aidc_profile.example.json](aidc_profile.example.json)，其新增配置键含义见 input_output、integration 和 synthetic_aidc；两种 profile 不共用权重。所有数值是功能初值，schema_version 不匹配或未知键必须拒绝。

```json
{
  "schema_version": 3,
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
    "split_mode": "target_index",
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
      {"name": "asset_in_zone", "src_groups": ["assets"], "dst_groups": ["zones"], "acyclic": true, "reverse_message": true},
      {"name": "zone_in_site", "src_groups": ["zones"], "dst_groups": ["sites"], "acyclic": true, "reverse_message": true}
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
    "entity_history_layers": 1,
    "relation_layers": 2,
    "temporal_layers": 2,
    "future_path_layers": 1,
    "dropout": 0.1,
    "category_embedding_dim": 8,
    "lead_dim": 16
  },
  "limits": {
    "max_total_nodes": 128,
    "max_history_steps": 96,
    "max_horizon_steps": 24,
    "max_explicit_attention_bytes": 268435456
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
  "evaluation": {"selection_metric": "valid_macro_mse", "seasonal_period": 24, "compare_seed": 2023}
}
```

固定规范：FFN 倍数=4、Post-LN、GELU、组级 scaler、单层实体 GRU、两层关系注意力、单层未来路径编码、窗口正弦编码、§5 质量加权与有界方差权重。layers 字段只允许上述首版值，不默默引入另一种架构。维度整除 head 数；L/H 正整数；至少一个动态组；组／特征唯一；variant 限 §10。

limits 是参考实现的安全边界，不是已测性能承诺：包括 batch padding 后的真实节点轴，不通过截断节点满足限制。分配前按 `4*B*L*relation_heads*N_total^2` 估算单个显式 FP32 分数张量，再检查各组空间层及时间层；任一超过字节上限时报错，显示估算。它不等于峰值内存估算，forward/backward 仍须实测；OOM 不能捕获后静默降级。上调 limit 需独立容量验证，首轮不宣称无限扩展。

以上 L≤96/H≤24 仅为周期 smoke 配置。AIDC 拟议 profile 明确采用 L≤168/H≤72、batch=2，支持小时至三天的验收目标；这些更大限额仍需实跑，不能把配置存在当性能通过。15 分钟／万级服务器属于另行容量验证范围，不截断输入或隐式预聚合来满足限额。

### 8.2 周期 smoke 与机制任务的分工

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

首轮必须另外生成 [机制任务 M1–M5](mechanism_tests.md)，分别覆盖轨迹、关系、状态／计划、规模／新实体和缺失／版本安全。每个任务的 manifest 显式定义 schema，不将测试任务硬塞入不相符的 base schema。生成器产出对应完整 JSON 配置，不隐式覆盖根配置；周期 base 不满足机制任务验收。

R4 还必须生成 [多 AIDC 场景](synthetic_aidc.md)，分别覆盖多级计量、暖通热惯性、任务执行偏差、点位稀疏与站点留出。M1–M5 的数学定义保留，仅包结构升至 schema 3；不得让 AIDC 周期数据替代成对机制任务。

episode 可有独立实体轴与数据文件，manifest 的 episodes 数组逐项指向同格式的包及 split；批次 collate 才做 padding，窗口不得跨 episode。同一机制任务成对样本视为不可拆的 episode-family，整族划分 train/valid/test，避免把成对变体拆到不同分区。首轮不读取真实业务文件。

## 9. 验收测试、阈值与命令

### 9.1 机制测试矩阵

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
| B 冻结 | 外层 train 后表示链仍 eval，参数不变；系统时间／未来路径有梯度；latent 头梯度断言遵循 §6 的 K/H 条件 |
| 回灌梯度 | 第二步主损失能沿第一步生成 latent 与 y 回传；无意外 detach |
| loss reduction | 手算夹具验证有效单元、有效组平均及零分母；通过测试实际核算数值 |
| 课程与早停 | 短课程不会提前停止，最终 best 来自 B-full，H=1 正常 |
| 续训 | CPU workers=0，在 epoch 边界恢复与未中断对应轨迹在容差内一致 |
| 产物重载 | 同设备加载前后预测一致；所有 schema／统计校验生效 |
| 旧入口回归 | 普通模型路由与默认参数不受 GSLAformer 专用 parser 影响 |
| 部分观测规模 | 保持各组覆盖比例时扩大节点数，q 不因 all-observed 概率下降归零；一个无效组不关闭其他组辅助 loss |
| 有界权重 | 方差非负时连续、有限、不超过 100；near-zero 不发生跳变 |
| teacher 对齐 | 逐步延续 memory 与同起点一次顺序编码相等；标签状态不会写进生产 state |
| 历史关系切换 | 变更前后各用对应版本，新起点立即可推理，不等待完整 L 稳定窗口 |
| 计算复用 | 一次 rollout 仅一次未来路径编码与一次真实历史编码，无跨 origin 缓存 |

CPU 同输入数值一致性使用 `atol=1e-6, rtol=1e-5`；MPS／跨设备比较 `atol=1e-4, rtol=1e-3`。不要求跨设备逐位相同。受控“应变化”测试先证明输入差异到达指定模块，再以差异大于 CPU 容差断言，不依赖任意随机模型恰好敏感。

### 9.2 小数据拟合与 smoke

- 专用拟合测试：使用 mechanism_tests.md 指定的无噪声机制小集，dropout=0、lr=0.003、最多 1000 update。A/B 分别报告初值和最终值；若初始标准化 MSE<=1e-4，只要求最终仍<=1e-4，否则要求最终<=1e-4 或不高于初值的 20%。不强迫已近零的 B 再按比例下降。
- 上述是预先声明的候选数值门槛，不是实测可达性声明。机制成对区分还必须通过独立任务断言，不能只拟合总体均值。首次实跑未达门槛时区分语义错误／优化限制并记录失败，不删除断言或追改成通过；预算或阈值修订需先记录依据。
- 完整 CPU smoke 使用 §8 配置与所有阶段；MPS smoke 使用同一 schema，但 A=1、B-short=1、B-medium=1、B-full=1 epoch，batch_size=2，完整 H，单独 run_id。缩短预算明确标记，不用于精度比较。
- synthetic test 输出有限值和正确时间索引；forecast 数据包没有未来目标，重载同一 best 完成新实体请求。

### 9.3 拟议执行命令

以下脚本／配置均待实现；本轮只检查文档语法，不运行这些尚不存在的命令。

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

## 10. 最小对照与指标

### 10.1 对照定义

| variant | 改动 | 训练与解释 |
| --- | --- | --- |
| `persistence` | 最后有效总负荷填满 H | 无优化；无有效历史值时报错，不偷用未来 |
| `seasonal_naive` | 使用起点前最近同相位真实 target | 周期=24，缺相位则用最后有效 target，显式记录 fallback |
| `total_only` | 仅 target 历史、日历、同一已知未来协变量 | 不给静态／内部测点；时间宽度和预算一致 |
| `no_entity_history` | 将实体 GRU 替换为逐时刻共享适配输出，其他路径不变 | 重新训练，检验池化前记忆；M1 是关键反例 |
| `no_spatial` | 保留实体 GRU，移除组内／关系 attention，用 masked mean 读出 | 同一内部信息；静态关系度数清零；重新训练 |
| `no_relation_message` | 移除关系层，保留组内 attention 和静态度数 | 只检验显式消息交互，不称无关系信息 |
| `no_relation_all` | 移除关系层，所有度数／关系派生输入清零 | 检验模型可消费的全部归属信息，重新训练 |
| `direct` | 完整关系空间编码 + A 式直接 H 输出 | 最多 44 epoch，patience=5；记录实际 update 与训练耗时 |
| `no_latent_loss` | 完整 AR 架构、latent_weight=0 | 从相同 best_A 初始化 B；仅多步主损失训练状态头 |
| `full` | 完整 GSLAformer | A/B 完整运行 |

训练型消融的 A/B 最大 epoch 预算相同，实际早停、参数量和逐步 forward 次数必须报告；同 epoch 不等于同 FLOPs 或时间。`no_latent_loss` 共用 full 的 best_A；direct 接收相同完整未来条件，但目标／优化路径不同，不能将差异全部归因于回灌。各架构均从自身 seed 初始化并按自己的契约重训，不加载不兼容的表示 checkpoint。

`total_only` 是独立基线，不把零动态组伪装成合法 full：A 共享完整计划条件的多步监督，B 只回灌 y，无 latent 头或标签。`no_spatial` 保留同组共享历史编码和属性，读出没有实体间 attention。比较脚本输出完整 resolved_config，不用隐藏 flag 偷偷跳过契约。

无关系消融不从真正的关系重新计算标签质量 q：`no_relation_all/no_spatial` 取 r_neighbor=1；否则即使数值输入清零，辅助权重仍能传入归属信息。no_relation_message 可以保留关系质量，含义明确。各变体的 schema 结构验证可读取关系合法性，但数值预测和 loss 权重不可偷用被消融的信息。

同权重诊断单列 `diagnostics.json`：固定最近真实 z 回灌、固定最近有效 y 回灌、teacher z 回灌；固定替代保留 generated 标记。teacher 使用未来观测，标为 oracle，不排名、不用于选模，也不保证成为严格数学上界。比较诊断误差只能提示依赖路径，不能代替重训消融。

首轮单种子 2023，结论仅限功能数据。不得声称统计显著或业务 SOTA。多 seed 和真实数据后续另行批准。

### 10.2 指标与输出粒度

- 以 origin／horizon 为评估单位，保留重叠预测；整体 MAE 为全部有效窗口单元绝对误差均值，RMSE 为同一集合平方误差均值再开方，不是各窗口 RMSE 的均值。
- 上项命名 `micro_mae/micro_rmse`；选优另用 §5 的 `macro_mse`。对外同时报告 per-horizon 与计数，不能把 micro 称为 horizon 等权；batch loss 的平均值不是 split 指标。
- 按 horizon 输出 MAE、RMSE 和有效计数。值已逆 target scaler；不按 latent 单位解释业务误差。
- 峰值幅度误差：只对整个 H 真值完整的窗口，比较预测最大值与真值最大值；时刻误差用最早 argmax 的步差绝对值乘频率。
- 爬坡 MAE：相邻两个预测步都有有效真值时，比较预测差分和真实差分；不把缺失边界拼接成爬坡。
- latent 按组报告有效计数、加权 MSE、原始规范坐标 MSE、训练表示方差；纯静态组无此指标。
- 记录训练 wall time、实际 update、参数量、预测请求耗时与设备；未来性能预算未知，不预先承诺时延。

首轮 y 测试指标好坏不决定功能验收，但异常值、时间错位、无效掩码或缺产物必须失败。

## 11. 产物、版本与运行冲突

每个 run 写 `results/gslaformer/<run_id>/`，不改其他模型目录。所有模式必需 `resolved_config.json/schema.json/data_manifest.json/environment.json/run.log/status.json`；其他按模式／变体区分：

| 模式／变体 | 额外必需产物 | 不应要求的产物 |
| --- | --- | --- |
| full 与带 latent 的 AR 训练 | scalers/vocab、latent_stats、best_A、last、B-full 的 best、training_metrics | 尚未执行 test 时不要求测试 predictions |
| total_only 训练 | scalers、best_A、last、B-full 的 best、training_metrics | latent_stats、latents、实体词表 |
| direct 训练 | scalers/vocab、last、direct-valid 选出的 best、training_metrics | B 阶段、latent_stats、best_A |
| persistence／seasonal_naive | predictions、metrics、baseline 参数和 fallback 计数 | 任何模型 checkpoint 或 latent 文件 |
| test | checkpoint 来源与校验值、predictions、metrics；带 latent 的 AR 模型另存 latents | 不重新训练，不自产 best_A |
| forecast | checkpoint 来源与校验值、无真值 predictions；带 latent 的 AR 模型另存 latents | 无 metrics 强制项，不复制训练 checkpoint |
| compare | comparison.json/csv、各子 run 路径与最终状态 | 自己不伪造模型权重 |

上述简写文件类型：JSON 配置／指标／统计、CSV predictions，latent_stats/latents 为 NPZ，checkpoint 为 PT；scalers.json、category_vocab.json、training_metrics.json、checkpoints/best_A.pt、checkpoints/last.pt、checkpoints/best.pt 为精确名称。AR 的 best 仅由 B-full 产生，direct 的 best 由 direct 验证产生，artifact 中存 variant/phase 防混用。

checkpoint 按 variant 保存所需 schema、scaler、词表、latent 统计、冻结清单和模型状态；训练目录的旁车用于审计，存在时校验一致。test/forecast 可只用完整自包含 checkpoint，不强求来源目录旁车仍存在。缺该 variant 的必需内容才拒绝加载。仅加载用户认可的本地模型 artifact，不从不可信来源反序列化任意对象。

`predictions.csv` 至少含 run_id、data_kind、origin、target_time、horizon（1-based）、y_pred、y_true（forecast 空）、target_valid、unit、new_entity_count、no_history_entity_count。latent 单独 NPZ，保留动态组顺序，禁止把其维度伪装成实际物理特征。

AIDC 另要求 site_id、target_id、boundary_id、freq_seconds、新站／质量／负值标记，以及 site_split.json、scenario_metrics.json、generation_audit.json 和 input_output §7 的 metadata。输入源单位先在 mapping 转为 profile 标准单位，AIDC y 的“原始量纲”统一指标准 kW，不恢复成各采集器混杂的 W/MW。

运行状态为 `running/completed/failed/interrupted`，与项目任务 Todo/Doing/Done 不混用。写临时文件后原子替换；全部必需产物可读取且校验通过后才能 completed。

已存在 run_id 一律拒绝新训练；使用明确 checkpoint 续训时，必须处于非 completed 状态且 config/schema/数据一致。test／forecast 必须新 run_id，不能覆盖训练结果。变更节点快照属于新请求，不改变权重 schema；数据 fingerprint 记录新值而不触发“实体必须等于训练清单”的错误限制。

run_id 和 episode_id 必须是单一安全名称（字母、数字、下划线、短横线），拒绝路径分隔符与 `..`；包路径解析后必须留在包根目录，符号链接不能越界。NPZ 以 allow_pickle=False 加载，并在大张量分配前检查 manifest 维度／资源限制。标签列默认不写应用日志，不为审查上传业务数据。

## 12. 开发工作包与进度证据

以下为后续实现顺序，不是本次已执行计划。每个工作包产出可独立检查的功能或测试，不先建立不工作的空壳；完整 GSLAformer 仍以 §1.2 整体验收。

| ID | 工作包／主要文件 | 前置 | 本包验收 |
| --- | --- | --- | --- |
| I1 | schema/config/scaling、合成生成器、数据测试 | 实施授权与 AGENTS 登记 | 标准包生成、JSON/NPZ 契约、split、as-of、scaler、新 ID 映射测试 |
| I2 | entity_history/spatial/relations、模型入口、模型测试 | I1 | 轨迹保真、编码 shape、关系 mask、空组、新实体免参数变化 |
| I3 | temporal/future_path/fusion/heads/rollout | I2 | 完整条件路径公平性、双头索引、teacher memory 隔离、因果／梯度 |
| I4 | 实验类／支持层、阶段 A/B、训练测试 | I3 | 预训练迁移、冻结、课程、loss reduction、小数据拟合、CPU 恢复 |
| I5 | run.py 隔离路由、smoke、产物、集成测试 | I4 | train/test/forecast/重载/新实体、CPU/MPS、旧入口回归 |
| I6 | 机制任务、对照脚本、指标、说明与治理收口 | I5 | M1–M5 与对照真实运行、风险报告、mode 产物矩阵、状态更新 |

涉及新增／修改超过八个文件；仅改本功能所需位置。`.gitignore` 对新增测试加精准例外，既有 tests 内容不删除。代码开发前先检查规则与分支漂移，保持当前分支，不自动 commit/push。

每个包的记录必须包含实际命令、退出码、测试计数、产物路径和偏差；任何预算调整／失败原因追加到设计 §12.1，不回改成“原本就如此”。

当前状态：I1–I6 均未实施。用户已确定功能范围，剩余业务目标、真实数据路径、生产预算属于后续业务验证的输入，不是继续追问首轮功能参数的理由。

R4 将 input_output、synthetic_aidc、integration 的补项并入 I1–I6 的完成定义；只完成旧三组 smoke 或 M1–M5 不能称完整 AIDC 功能交付。本轮 DOC-003 只登记文档产物。
