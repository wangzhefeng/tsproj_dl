# GSLAformer R4：开发实施规格

> 本文唯一维护模型内部计算、损失与指标算法、训练恢复、源码复用、CLI 接入、产物生命周期和工作包。业务数据／张量见[输入输出契约](input_output.md)，运行矩阵和通过标准见[验收协议](acceptance.md)。
>
> 状态：GSLA-001 核心实现已接通，正在工程与完整模拟验收收口；最新证据见 §12，历史增量不代表当前缺项。数值是工程初值，不是用户业务值或实测最优值。根 AGENTS.md 管项目状态；工作包完成必须附实际证据。

<a id="implementation-3"></a>

## 1. 模型计算规格

<a id="implementation-3-1"></a>

### 1.1 属性驱动的实体 token

每组适配器拼接：标准化动态值、observed、applicable、静态数值及 mask、各静态类别 embedding。使用 `Linear → GELU → Linear` 映射到共同 `d_s`；每个类别属性 embedding 宽度 8，词表来自训练数据。不同组适配器独立，同组跨时间／实体共享。

不使用实体 ID embedding，不加实体序号位置编码。静态属性可以参与动态表示，但不宣称纯动态解耦。可选坐标仅在 schema 有该数值特征时消费。

**池化前实体时序编码是必需路径。** 动态组使用单层单向 `nn.GRUCell(d_s,d_s)`，同组实体共享参数。按统一 ID 轴逐时刻输入适配后的 token，产生 `[B,L,N_g,d_s]`：

- 请求最早位置 hidden=0；present=false 或 padding=true 的状态置零，重新出现从零开始。
- 缺测但 present=true：用填零数值及 mask 输入 GRU，不将状态清零，也不伪造观测。
- 父级／属性变化不重置同一实体的 hidden；空间交互再使用该时刻的边和属性。
- 纯静态组无 GRU，直接用当时属性 token。实体 GRU 不先进行跨实体平均，避免轨迹对应关系丢失。
- GRU 采用单层、无内置 dropout，空间层 dropout 仍由配置控制。每个请求重新初始化，批次之间不传 hidden。

合法不变性是“同一置换作用于该实体整段历史、属性、mask 和关系”。逐帧不同置换会改变实体轨迹，不应断言输出不变。`EncodedHistory.entity_memory` 仅供 teacher 延续编码，生产 rollout 不使用预测 z 反解该 memory。

实体时间编码完成后，才合并 batch 与时间维进入逐时刻空间计算：

\[
[B,L,N_g,d_s]\rightarrow[B L,N_g,d_s]
\]

<a id="implementation-3-2"></a>

### 1.2 空间、关系与读出

1. 对每时刻的实体 GRU 输出（静态组为属性 token）加入组内摘要 token，执行组内 TransformerEncoderLayer（Post-LN、GELU、FFN=`4*d_s`）。摘要 token 始终有效。
2. 提取真实实体输出，按 schema 拼接各组实体轴。关系 attention 只消费实体 token，不让组内摘要成为未声明的跨组通信边。
3. 关系层采用 `nn.MultiheadAttention` + residual/Post-LN + FFN；query 行是接收方，key 列是发送方。self、注册的源→目标及 reverse_message 允许的反向通信有效，其他为负无穷；每种业务关系的两个方向各有按 head 的可学习偏置。同端点多类型偏置相加；未连接位置不能因零偏置被误开放。偏置／mask 按 batch、时刻、head 对齐，禁止不同站点间建立边。
4. padding／未生效 query 仅开放其 self 作为数值安全通路，其他有效 query 不得读取这些 key；该 query 输出随后置零，不参加池化。这避免全遮罩 softmax 的 NaN。
5. 两层关系交互使用同时更新的上一层状态，不能原位按遍历顺序更新节点。
6. 每组用独立的可学习读出 query 对关系增强实体做一次 cross-attention（head 数沿用 spatial_heads）；补一个可学习空组 token 作为始终有效的 key。拼接 `log1p(有效实体数)`、当前动态观测覆盖率，再经 Linear 映射到 `d_z`、LayerNorm，得到规范 latent。仅动态组生成 latent 目标。

关系端点重命名或同时重排不改变语义。改变父级可能改变消息，但不承诺任何一次重连都改变任意已训练模型的输出。

允许边约束是关系层局部契约，不是整个模型的隔离保证：组内全连接注意力已经混合了同组信息。关系层禁止 key_padding_mask 再次屏蔽无效 query 的安全 self，必须用统一的加性 mask 处理该情形；无效输出归零后不参与任何有效节点计算。

<a id="implementation-3-3"></a>

### 1.3 静态条件 c

独立静态链只读 origin 有效实体的静态数值／类别、mask、按类型与方向的平均关系度数，不读动态值。先按组聚合：各数值属性的有效 masked mean、sum、count，类别 embedding 的有效均值，`log1p(有效节点数)` 及平均入／出度；再由每组 Linear→GELU 投影至 `d_z`，拼全部组后单 Linear 投影至 `d_t` 得到 c（GSLA-003 批准保留当前实现，不增加跨组 MLP）。不存在额外的“逐实体非线性适配再池化”；动态实体适配器与静态链参数独立。空组统计为零，度数分母排除 padding 和 origin 未生效实体。

静态链与动态链分别建模块，B 阶段一起冻结。数量／sum 保留规模信息，但不构成总功率守恒或严格单调性保证。

<a id="implementation-3-4"></a>

### 1.4 历史 token 与时间编码

动态状态保留 `[B,L,Gd,d_z]`，送时间模型时按固定组顺序展平末两维。每个 token 再拼：

- 标准化总负荷；
- 总负荷 observed、generated 两个标记；
- 每动态组 observed coverage 和 generated 标记。

日历为本地日内相位与周内相位的 sin/cos 四个数，与上述输入拼接后经联合 Linear 映射至 `d_t`，再加窗口内 `0..L-1` 的固定正弦编码，滑窗重算。日历函数适配任意固定秒级频率，不通过拼接未来真实观测计算。GSLA-003 保留此计算；其仿射函数族等价于输入与日历分别投影后相加，不承诺不同参数化的重新训练轨迹相同。

时间 Transformer 采用 Post-LN、GELU、FFN=`4*d_t`，显式上三角屏蔽未来；最后 token 输出为 h。历史测量缺失不等于时间 padding，本版本每个时间位置都有合法 token。

<a id="implementation-3-5"></a>

### 1.5 完整未来路径与预测头

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


AIDC 类型关系的业务方向和合法性以[输入输出契约](input_output.md#relation-contract)为准；q 的两跳集合从全部启用类型通信图计算。

<a id="implementation-4-1"></a>

## 2. 单一自由递推契约

`rollout` 在训练、valid、test、forecast 中复用，仅梯度上下文与监督计算由调用方决定：

1. 按实体编码真实历史并完成空间／关系读出，得到规范 latent、末端实体 memory、独立静态条件；同时编码原 origin 的完整已知未来路径；
2. 当前系统状态窗口 + 对应 p[:,k] 与原起点 lead(k) 预测 y、各动态组 z；
3. 保存本步预测；
4. 追加预测 z 与同一预测 y，generated=true、observed=false、coverage=0；
5. 移除最早 token，重新生成窗口位置编码，进入下一步。

每次循环使用原预测起点可见的 future_known 切片，不在模拟的下一时刻重新读取后来发布的计划。历史 token 质量标记保留；生成值不是缺失值，不能因 observed=false 被预处理器再次清零。

历史 z 只计算一次，窗口移位后不重新运行实体编码器。生产回灌没有实体级生成值，不更新实体 memory；这份 memory 只在训练标签构造的隔离 teacher 路径中使用。两条路径不能互相写状态。

模型输出为 `ForecastResult.y [B,H,1]`、`z [B,H,Gd,d_z]`、generated 标记及元数据；业务质量标记在推理出口组装。模型内部 y 标准化，落盘 y 逆变换为输入 manifest 的原始单位；z 保持规范坐标，注明非物理量。

<a id="implementation-5"></a>

## 3. 标签、损失与缺失规则

<a id="implementation-5-1"></a>

### 3.1 latent 标签与统计

加载 best_A 后冻结适配器、实体 GRU、空间／关系／读出／LayerNorm／静态链。在 `no_grad + eval` 下先编码与模型相同的历史，复制末端实体 memory，再按 1..H 消费 future_groups 的数值与 mask。未来属性／present／关系保持 origin 上下文，每一步生成 teacher z；不能每步从零重建 GRU，也不能读取 future target 生成 z。

标签是**带缺失条件的可用信息表示**，不是完整物理状态。允许部分观测，质量只软加权，不设全系统 all-observed 门槛：

1. 对动态组 g 的时刻 s，r[g,s] 为最近 L 步内所有 present、非 padding、applicable 动态单元的 observed 比例；窗口起点早于本请求历史时截断到已编码起点。分母为零时 r=0。
2. V[g,s] 为从 g 的有效实体沿通信图至多两条边可达的其他动态组集合（按组去重，不含 g 自身）。通信图由全部启用关系类型的有向边（src→dst）构成，`reverse_message=true` 的类型另加入反向边；self 边不参与。r_neighbor 为这些组 r 的算术均值；集合为空取 1。
3. `q[g,s]=r[g,s]*(0.5+0.5*r_neighbor)`；组在 s 为空时 q=0。r 与 q 只由真实 mask／图计算，禁止用可学习 attention 权重提高自己的监督权重。
4. 该 q 是有界质量启发式，不是统计置信度，不覆盖全部组内混合造成的依赖；部分缺失 teacher 仍是噪声目标。记录 q 分布、有效率和缺失率敏感性，不能宣称缺失已完全解决。
5. 未来 q 与 mask 仅进入 loss／诊断，生产生成 token 仍用 generated=true、coverage=0，不泄漏未来观测质量。

latent 统计只取训练 split 的每个唯一 `(episode, origin)` **最后一个历史 z** 一次，并存该请求数据／版本 fingerprint；不收集重叠窗口中的其他 z，也不收集未来 teacher z。重复键内容不一致时报错。各维以 origin 的 q 作为权重计算均值和总体方差；该组至少有两个 q>0 的训练 origin，否则无法估计而报错。B 不重拟合。

`latent_stats.npz` 布局：每动态组三个键 `{group}__mean [d_z] float64`、`{group}__variance [d_z] float64`、`{group}__count`（q>0 的 `(episode, origin)` 数，int64 标量）。辅助维度权重 w 在加载时按上式 `1/(variance+0.01)` 重算并截断到 100，不单独落盘；NPZ 以 allow_pickle=False 读写。

统计溯源旁车 `latent_stats_manifest.json` 保存 version=1、动态组顺序、latent_dim、schema/data/representation 三个指纹、唯一 origin 的 episode/origin/request_fingerprint/content_fingerprint 清单、NPZ 字节 SHA-256、统计值规范指纹 statistics_fingerprint 及清单规范指纹。NPZ 与旁车各自临时写入后原子替换；旁车最后写入作为完整标志，加载缺任一文件即失败，不覆盖既有统计。checkpoint 集成时须携带相同溯源信息；本旁车不是额外拟合来源。

辅助维度权重固定为 `w[g,j]=1/(variance[g,j]+0.01)`，连续且不超过 100。近常量维仍有限加权，不突然把权重改为 1。单个组全部维度方差低于 `1e-8` 只告警并保留常量目标；全部动态组均如此才暂停 full 训练、报告 representation_degenerate，交由机制任务定位，不静默改模型。权重只用于 loss，不改预测值与回灌坐标。

<a id="implementation-5-2"></a>

### 3.2 明确的 reduction

- 主损失：每个 horizon 先在有效 batch 样本中计算 MSE，再对有有效标签的 horizon 等权平均。没有标签的 horizon 不以 0 混入平均，记录缺项。
- latent：先对维度平均 `w*(prediction-target)^2`；每个组／horizon 的 batch 项为 `sum(q*error)/count(q>0)`，再对该组有效 horizon 等权、对有效动态组等权。分母不是 sum(q)，使低质量标签确实减弱辅助项，而不是被归一化抵消。
- 无有效 latent 组：latent_loss=0，记录计数；不除以零，不复制上一 batch 损失。
- 无有效主目标：跳过该 batch 的 optimizer.step，记录原因；整 epoch 主目标计数为零时报错。
- 总损失 `load_loss + 0.1 * latent_loss`；首轮不搜索 lambda，`no_latent_loss` 对照显式设为 0。
- 梯度裁剪范数 1，NaN／Inf loss 或梯度 fail-fast，保存失败状态，不加绕过标记继续。

标签 mask 不能作为 rollout 的 future 输入。生成态没有真实 coverage，不用未来真实 mask 来构造它的输入质量标记。

验证选优指标为标准化的 horizon-macro MSE：先在整个 split 累积各 horizon 残差和计数，再求各 horizon 均值的平均，不平均 batch loss。最终 H 任一 horizon 无有效 valid 标签时拒绝选优；test 可报告缺项，不能伪称完整 H 指标。合成验收的 valid/test 每步都有标签。

AIDC profile 使用 `valid_site_macro_mse`：先在每站内按上述方法算 horizon-macro，再对所有验证站点等权。A/B 都使用该选择指标，表 [两阶段训练状态机与恢复](implementation.md#implementation-6) 中的 valid 指标随 profile 解析。训练 batch 仍按 [标签、损失与缺失规则](implementation.md#implementation-5) 的 loss reduction，站点平衡通过 sampler 实现，不误称 batch 平均就是 split 选优指标。

<a id="implementation-6"></a>

## 4. 两阶段训练状态机与恢复

<a id="implementation-6-1"></a>

### 4.1 阶段与参数

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

<a id="implementation-6-2"></a>

### 4.2 梯度与表示稳定性

- B 冻结实体／空间／关系／读出／静态链，并在父模型 `.train()` 后仍强制 frozen_modules.eval()；未来路径编码器属于可训练预测链，不误冻结。
- rollout K 步保留所有生成 z 与 y 回灌梯度，不 detach。验证和推理使用 no_grad。
- 冻结链的梯度上下文（已确认，2026-10-14）：B 阶段训练前向中，冻结的实体／空间／关系／读出／静态链在 `torch.no_grad()` 下产出历史 z 与 c，再进入可训练时间链；`.eval()` 只关闭 dropout，不阻止建图。在 `tests/test_gslaformer_training.py` 断言冻结链输出无 `grad_fn`、优化器参数清单不含冻结参数。该规则只减少无效激活，不改变 §3.1 teacher 路径与回灌梯度不 detach 的既定规则。
- 不启用 AMP、scheduled sampling、KV cache 或截断反向传播。功能配置容量较小；若资源超限，报告并修订配置，而不是静默改变训练算法。
- 每轮记录主损失、分组 latent loss／计数、梯度范数、latent 标准差、历史负荷依赖诊断；不能仅凭有方差就宣称空间信息有效。

<a id="implementation-6-3"></a>

### 4.3 checkpoint 与续训

每个 epoch 完成后保存 last，选择指标改善至少 `1e-6` 才更新 best。保存 `phase/phase_epoch/global_epoch/optimizer/early_stop_counter/best_score/python_rng/numpy_rng/torch_rng/device_rng`。可用设备 RNG 若不能恢复，明确不保证同设备逐位续训；首轮 exact resume 测试在 CPU、workers=0 完成。

仅支持 epoch 边界续训，不承诺 batch 中途逐位恢复。epoch 未结束的中断重新从上一已保存 epoch 执行，并在日志记录。B 阶段恢复必须校验冻结 fingerprint、latent 统计、数据包与模型 schema 一致；禁止只加载模型权重却声称恢复完整训练。

### 4.4 多站点训练采样

AIDC 先均匀选训练站点再选该站训练 origin，epoch 总抽样数等于所有训练站点合法窗口数（有放回），随机状态进入 checkpoint。valid/test 不抽样、不 shuffle，stride 由验收 profile 明确。只允许训练站点训练时间拟合 scaler／词表；输入的分区合同见输入输出，具体站点列表见验收。

<a id="integration-2"></a>

## 5. 既有模型与层的复用依据

路径相对仓库根目录；行号为 DOC-003 检查位置，后续开发前需重新定位符号。以下“复用”均是开发决策，不是已完成接线。

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

首版必需对照以[验收协议](acceptance.md)的机制与 AIDC 矩阵为准；上表不意味着必须训练所有仓库模型。DLinear／PatchTST 等若另加入比较，应在相同目标／分区／可用信息下运行，并注明它们是否只看总表或固定摘要；不能把输入更多造成的收益全归给架构。

<a id="integration-3"></a>

## 6. 基础设施复用与隔离

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


上述源码位置是 DOC-003 时的核查记录，不是 DOC-005 重新运行组件的证据；编码前须重新定位符号和行为。不照搬固定列 loader、不复制旧实验类，也不创建脱离项目的训练工程。真实源文件目录待业务授权后确定，当前仅生成 synthetic 标准包。

<a id="implementation-7-1"></a>

## 7. 实现文件与内部 API

下表定义实现文件职责，存在性与完成状态由 §12 的实际证据确认。新 package 添加最小 `__init__.py` 与目录 README；无抽象空壳、外部服务或通用插件系统。

| 文件 | 主要职责／公开符号 |
| --- | --- |
| `models/transformer/GSLAformer.py` | `Model`，组合完整模型；`encode_history`、`predict_direct`、`predict_step`、`rollout` |
| `layers/gslaformer/spatial.py` | `GroupSpatialEncoder`、`GroupReadout`、`GroupMeanReadout`、`StaticConditionEncoder` |
| `layers/gslaformer/entity_history.py` | `GroupAdapter`、`EntityHistoryEncoder`，共享 GRU、ID 对齐后的逐实体状态和 present 重置 |
| `layers/gslaformer/relations.py` | `RelationAttention`、方向 bias 与 mask 生成后的消费 |
| `layers/gslaformer/temporal.py` | `TemporalEncoder`、`FuturePathEncoder`、`ConditionFusion`、确定性时间／lead 编码 |
| `data_provider/spatial_temporal/schema.py` | schema 3、形状／站点／目标／类型校验；schema 为 JSON 字典，不声明不存在的 GroupSchema 类 |
| `data_provider/spatial_temporal/aidc_mapping.py` | 设备／测点／端口／单位映射及覆盖审计；不猜测现场接线 |
| `data_provider/spatial_temporal/config.py` | JSON 解析、显式键集合、禁止未知键与参数约束 |
| `data_provider/spatial_temporal/dataset.py` | manifest/NPZ 加载、as-of 选择、窗口、关系端点映射 |
| `data_provider/spatial_temporal/batching.py` | `Batch`、`BatchInputs`、`BatchSupervision`、`collate_samples`：变长 padding、关系重编号与分配预算；规则见 [batch 组装](input_output.md#batch-contract) |
| `data_provider/spatial_temporal/scaling.py` | 组级数值／target／协变量 scaler、训练词表及 artifact 校验 |
| `exp/exp_gslaformer_forecasting.py` | `Exp_GSLAformer_Forecast`，复用 Exp_Basic 设备／模型发现，训练状态机与运行分派 |
| `exp/gslaformer_inference.py` | 自包含 checkpoint 的 test/forecast 加载与产物导出；不读取训练包或重拟合统计 |
| `exp/gslaformer_comparison.py` | 规定任务矩阵调度、朴素基线、共享 best_A、比较与成对区分报告 |
| `exp/gslaformer_support.py` | 损失 reduction、latent 标签／统计与诊断；checkpoint 序列化由实验类负责 |
| `exp/gslaformer_artifacts.py` | 模式必需产物门禁、scaler 序列化与站点分层指标；不参与训练选模 |
| `scripts/dev/generate_gslaformer_fixture.py` | base、M1–M5、AIDC synthetic 标准包、oracle 与新增实体／新站请求 |
| `scripts/smoke/gslaformer_functional.json` | [配置定义与设计实例](implementation.md#configuration) 的完整功能配置 |
| `scripts/smoke/smoke_gslaformer.sh` | 对已生成的 synthetic 包执行 train/test/forecast、重载验收；不自行生成数据 |
| `scripts/smoke/compare_gslaformer.py` | 最小基线／消融运行与统一表格，不推导虚假性能 |
| `tests/test_gslaformer_data.py` | schema、split、as-of、mask、scaler |
| `tests/test_gslaformer_package.py` | 标准包真实落盘边界：时间、来源、版本与监督隔离，不 mock loader |
| `tests/test_gslaformer_mapping.py` | 测点单位与业务类型图的来源边界 |
| `tests/test_gslaformer_scaling.py` | 组级 scaler／词表与 artifact 校验；完成证据见 §12 |
| `tests/test_gslaformer_model.py` | shape、因果、关系、重排、新实体、条件消费 |
| `tests/test_gslaformer_training.py` | 冻结、梯度、loss；状态机与 resume 由 state_machine 测试负责，小集拟合另走专用脚本 |
| `tests/test_gslaformer_integration.py` | CLI、产物、重载、synthetic E2E；真实 MPS 走独立设备 smoke，不冒充普通 CPU 单测覆盖 |

文件现状与运行证据统一由 [开发工作包与进度证据](implementation.md#work-packages) 维护，不另设容易过期的存在性清单。

模型接口语义：

- `encode_history(batch) -> EncodedHistory`：仅读 `batch.inputs`，输出规范 z、覆盖率、静态 c、按组末端 memory；不读取 `batch.supervision`。
- `encode_future_path(batch) -> [B,H,d_t]`：整个请求条件一次编码，保留当前训练图。
- `predict_step(state, path_context_k, origin_lead_k) -> StepPrediction`：下一 y 与 z；state 不含标签或 teacher memory。
- `rollout(batch, horizon) -> ForecastResult`：B 训练／验证与最终 AR 推理的唯一闭环，正式请求 H 等于 checkpoint 配置 H，future_time／known 同长；课程 K 只限制回灌步数，不裁剪原始 H 条件。
- `predict_direct(batch) -> [B,H,1]`：A／direct 的训练和验证，不回灌。
- `build_latent_labels(model, inputs, supervision, encoded_history=None)` 位于实验支持层；分别传入 BatchInputs 与 BatchSupervision 并复制 entity_memory，返回 z、quality，不在生产 forecast 路径中调用。

边界对象字段（首版固定，变更须先改本节）：

- `EncodedHistory`：`z [B,L,Gd,d_z]`、`observed_coverage [B,L,Gd]`、`generated [B,L]`（真实历史全 false）、`c [B,d_t]`、`entity_memory {group: [B,N_g,d_s]}`、`entity_ids`。
- `RolloutState`：`z/y/y_observed/y_generated/coverage/group_generated/calendar/condition/entity_ids`；不含标签和 teacher memory。condition 为静态 c，calendar 已编码，原点 lead 由 predict_step 参数传入；窗口移位只移除最早 token 并重算位置编码。
- `StepPrediction`：`y [B,1]`（target 标准化坐标）、`z [B,Gd,d_z]`（规范坐标）。

<a id="implementation-7-2"></a>

## 8. 现有入口的隔离接入

已读现有 `run.py`：它的 setting 拼接依赖普通时序参数，三个运行开关全为零还会触发“最终训练”。GSLAformer 不能只插一个实验类后继续沿用这两个隐式行为。

增加专用 `--gsla_config`、`--gsla_mode`、`--gsla_run_id`、`--gsla_checkpoint`、`--gsla_request`；模式仅为 `train/test/forecast/compare`。`--gsla_checkpoint` 在 train 表示续训，test／forecast 表示权重来源。request 只用于 forecast。

为 GSLAformer 加隔离参数解析分支：预解析发现 `--model GSLAformer` 后走其专用配置与模式校验，不要求虚假的 `--enc_in/--dec_in/--root_path` 参数；其他模型保持当前 parser、required 参数和 run 逻辑。GSLAformer 禁止旧 `is_training/is_testing/is_forecasting` 混用，冲突报错；不扩大修改旧入口的范围。

`run(args)` 对 GSLAformer 先分派并返回，不进入旧 setting 和“全零即最终训练”。`_select_experiment_class` 增加明确路由；专用参数适配成 Exp_Basic 所需 `use_gpu/gpu_type/devices/use_multi_gpu`。明确请求 MPS/CUDA 时先检查可用性，禁止继承基类的静默 CPU 回退。模型发现保持 `Model` 入口，不另造注册框架。

配置拥有模型、数据、训练、设备设置；CLI 只提供运行模式、配置路径、run_id、checkpoint 和请求路径，不允许散落 CLI 覆盖模型参数。相对 config 的 data.package 路径以 config 所在目录为基准；results_root 以仓库根目录为基准。生产请求路径由 CLI 指定，不修改训练 schema。


<a id="configuration"></a>

## 9. 配置定义与设计实例

[周期 base 配置](../../../scripts/smoke/gslaformer_functional.json)与[AIDC 一天配置](../../../scripts/smoke/gslaformer_aidc_hourly_day.json)唯一维护在 scripts/smoke，文档和测试直接使用，不保留镜像副本。生成标准包后均可解析加载；base 与 AIDC 是不同 schema，不共用权重。

| 键组 | 含义与约束的权威位置 |
| --- | --- |
| schema_version | 输入输出契约的数据／模型版本，不与绘图格式版本混用 |
| experiment | model 固定 GSLAformer；variant 见验收对照表；profile 标识独立任务，data_kind 为 synthetic；seed/device/results_root 明确保存 |
| data.package / timezone / freq / target / groups / relation_types / known_future / feature_units | 输入输出契约的来源、单位、对象与外部数据 schema；relative package 按实际配置文件目录解析 |
| data.seq_len / pred_len / train_stride / valid_stride / test_stride / split_mode / split_index / site_split | 输入输出定义切窗与分区语义，验收协议唯一维护各任务的具体值及站点角色；字段集合以 `config.py` 校验为准 |
| data.sampler / batch_size / num_workers | 本文的训练采样、设备与恢复约束；参数实例见完整配置 |
| model | 本文的实体／空间／关系／时间／未来路径计算；不得用字段偷偷切换另一架构 |
| limits | 本节安全分配规则；实例限额不是峰值内存或性能承诺 |
| training | 本文的优化器、损失、课程、早停与梯度规则 |
| evaluation | 本文定义指标算法，验收定义选择指标、报告粒度、季节周期与比较种子 |

固定规范：FFN 倍数=4、Post-LN、GELU、组级 scaler、单层实体 GRU、两层关系注意力、单层未来路径编码、窗口正弦编码、[标签、损失与缺失规则](implementation.md#implementation-5) 质量加权与有界方差权重。layers 字段只允许上述首版值，不默默引入另一种架构。维度整除 head 数；L/H 正整数；至少一个动态组；组／特征唯一；variant 限 [基线、消融与诊断定义](acceptance.md#variants)。

limits 是参考实现的安全边界，不是已测性能承诺：包括 batch padding 后的真实节点轴，不通过截断节点满足限制。分配前按 `4*B*L*relation_heads*N_total^2` 估算单个显式 FP32 分数张量，再检查各组空间层及时间层；任一超过字节上限时报错，显示估算。它不等于峰值内存估算，forward/backward 仍须实测；OOM 不能捕获后静默降级。上调 limit 需独立容量验证，首轮不宣称无限扩展。

周期样例的 L≤96/H≤24 仅为周期 smoke 配置。AIDC 拟议 profile 明确采用 L≤168/H≤72、batch=2，支持小时至三天的验收目标；这些更大限额仍需实跑，不能把配置存在当性能通过。15 分钟／万级服务器属于另行容量验证范围，不截断输入或隐式预聚合来满足限额。


正式配置为 `scripts/smoke/gslaformer_functional.json`，以及同目录的 `gslaformer_aidc_hourly_short.json`、`gslaformer_aidc_hourly_day.json`、`gslaformer_aidc_hourly_multiday.json`。不依赖隐式继承或 CLI 模型参数覆盖；准确 L/H、stride 与运行范围见验收。相对 data.package 按配置所在目录解析，复制到其他目录时须显式调整路径。

Python 使用项目根 .venv；核心基于已有 PyTorch，无新账号或外部服务。若确需依赖，按项目规则以 uv add 管理并说明必要性。新目录添加职责 README，不建抽象空壳。

<a id="hyperparameters"></a>

### 9.1 首版超参取值表

`config.py` 要求所有键显式给出、不提供默认值；下表汇总两份设计样例的现值，便于审阅，不新增任何决策。“已定”指本文其他节或 `config.py` 已锁定；“候选”指仅出现在样例中、尚无独立依据，修改前先登记。

| 键 | 周期 base 样例 | AIDC 样例 | 状态 | 依据／锁定位置 |
| --- | --- | --- | --- | --- |
| model.spatial_dim（d_s） | 32 | 32 | 候选 | 仅样例；需整除 spatial_heads／relation_heads（`config.py`） |
| model.latent_dim（d_z） | 32 | 32 | 候选 | 仅样例；Gd=6 时时间 token 的 z 部分为 192 维 |
| model.temporal_dim（d_t） | 64 | 64 | 候选 | 仅样例；需整除 temporal_heads |
| model.spatial_heads / relation_heads / temporal_heads | 4 / 4 / 4 | 4 / 4 / 4 | 候选 | 仅样例 |
| model.spatial_layers / entity_history_layers / relation_layers / temporal_layers / future_path_layers | 1 / 1 / 2 / 2 / 1 | 1 / 1 / 2 / 2 / 1 | 已定（temporal_layers 除外为候选） | §1 与 §9 固定规范：单层 GRU、两层关系、单层未来路径 |
| model.dropout | 0.1 | 0.1 | 候选 | 小集拟合另用 0（[验收 §6](acceptance.md#fit-thresholds)） |
| model.category_embedding_dim / lead_dim | 8 / 16 | 8 / 16 | 已定 | §1.1 宽度 8、§1.5 16 维 lead |
| training.optimizer / learning_rate / weight_decay | AdamW / 0.001 / 0.0001 | 同左 | 已定 | §4.1 |
| training.clip_grad_norm | 1.0 | 1.0 | 已定 | §3.2；`config.py` 拒绝其他值 |
| training.latent_weight（λ） | 0.1 | 0.1 | 已定 | §3.2，首轮不搜索；`no_latent_loss` 为 0 |
| training.stage_a / b_short / b_medium / b_full epochs | 20(p=5) / 2 / 2 / 20(p=5) | 同左 | 已定 | §4.1 |
| training.min_delta / amp / truncate_rollout_grad | 1e-6 / false / false | 同左 | 已定 | §4.2–4.3 |
| data.batch_size | 8 | 2 | 候选 | AIDC batch=2 见 §9 与验收 §21 |
| limits.max_total_nodes / history / horizon | 128 / 96 / 24 | 128 / 168 / 72 | 已定（作为安全上限） | §9；不是性能承诺 |
| limits.max_explicit_attention_bytes | 268435456（256 MiB） | 同左 | 候选 | 仅样例；见 §9.2 估算 |

<a id="complexity"></a>

### 9.2 复杂度与显存估算（候选，待实测）

以下为按本文计算规格推导的量级估算，不是实测，也不替代 §9 的“forward/backward 仍须实测”。记 N 为 batch padding 后的真实节点总数、h 为 relation_heads。

| 组件 | 计算量主项 | 需保留激活的主项（训练） |
| --- | --- | --- |
| 实体 GRU（§1.1） | O(B·L·N·d_s²)，L 步串行 | O(B·L·N·d_s) |
| 组内空间 attention（§1.2） | O(B·L·Σ_g N_g²·d_s) | 每层分数 O(B·L·heads·Σ_g N_g²) |
| 类型关系 attention（§1.2，两层） | O(B·L·N²·d_s) | 每层分数与加性 mask 各约 4·B·L·h·N² 字节 |
| 时间 Transformer（§1.4） | 每次 O(L²·d_t+L·d_t²) | A：一次；B-full：H 次完整窗口前向（无 KV cache），图全部保留 |
| 未来路径编码（§1.5） | O(H²·d_t) | 一次 |

按 §9 单张分数张量公式 `4*B*L*h*N^2`（h=4、N=128 上限）：周期 base（B=8、L=24）约 48 MiB；AIDC 六小时（B=2、L=48）约 24 MiB；AIDC 一天／三天（B=2、L=168）约 84 MiB，均低于样例 256 MiB 上限。判断（待实测）：A 阶段两层关系注意力需同时保留分数、softmax 输出、dropout mask 与加性 bias，仅此部分在 L=168 时可能达到数百 MiB；B-full 的主要增量来自 H=72 次时间链前向的保留图，若冻结链按 §4.2 既定规则在 no_grad 下运行，则不含实体／关系激活。I4/I5 实跑时按 profile 与阶段记录峰值内存（CPU RSS／MPS 当前分配）与 wall time，回填本表实测列后再讨论是否调整 limits。

<a id="implementation-10-2"></a>

## 10. 指标计算与输出粒度

- 以 origin／horizon 为评估单位，保留重叠预测；整体 MAE 为全部有效窗口单元绝对误差均值，RMSE 为同一集合平方误差均值再开方，不是各窗口 RMSE 的均值。
- 上项命名 `micro_mae/micro_rmse`；选优另用 [标签、损失与缺失规则](implementation.md#implementation-5) 的 `macro_mse`。对外同时报告 per-horizon 与计数，不能把 micro 称为 horizon 等权；batch loss 的平均值不是 split 指标。
- 按 horizon 输出 MAE、RMSE 和有效计数。值已逆 target scaler；不按 latent 单位解释业务误差。
- 峰值幅度误差：只对整个 H 真值完整的窗口，比较预测最大值与真值最大值；时刻误差用最早 argmax 的步差绝对值乘频率。
- 爬坡 MAE：相邻两个预测步都有有效真值时，比较预测差分和真实差分；不把缺失边界拼接成爬坡。
- latent 按组报告有效计数、加权 MSE、原始规范坐标 MSE、训练表示方差；纯静态组无此指标。
- 记录训练 wall time、实际 update、参数量、预测请求耗时与设备；未来性能预算未知，不预先承诺时延。
- AIDC 可报告 `NRMSE=RMSE/该站测试真值均值` 的诊断值；均值非正／零时不可计算。测试均值只用于最终指标分母，不反馈 scaler、模型或选模，也不替代原始 kW 误差。

首轮 y 测试指标好坏不决定功能验收，但异常值、时间错位、无效掩码或缺产物必须失败。

<a id="implementation-11"></a>

## 11. 产物生命周期、版本与运行冲突

数据 schema=3 与模型计算版本分离。当前 checkpoint `version=2 / computation_version=2`；未来路径独立数值/mask 投影和读出头数修正改变了模型语义，旧版本明确拒绝 train resume/test/forecast/共享 A 加载，不自动迁移。历史产物不覆盖，也不为新实现背书；重新训练须新 run_id，完整 AIDC 仍受预算门禁约束。

所有训练另存 `resource_metrics.json`：设备、参数量、每次初始化耗时以及各 epoch 的 phase/global_epoch/updates/train_seconds/valid_seconds/save_seconds；phase_transitions 另记录阶段切换（包含 Freeze 统计）的耗时。它不参加 exact resume 数值比较；加速器计时先同步，不把异步提交耗时冒充计算耗时。参数量统计模型所有参数（含冻结参数）。test/forecast metadata 记录 forward+逆标准化的 batch 耗时（排除 loader 与 teacher 诊断），不假装是各请求独立服务延迟；精确字段见输出合同。

checkpoint 的 `latent_stats_manifest` 完整复制统计旁车，包含 origin 清单、NPZ 校验和与统计值规范指纹；`frozen_parameters` 为 B 冻结参数名排序清单，A/direct 为空。加载验证计算版本、清单与实际阶段、溯源指纹及统计值绑定；来源旁车存在时要求一致，缺失时允许可信自包含 checkpoint 独立推理。

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

predictions、latent 与 metadata 的字段及物理含义唯一见[输出合同](input_output.md#output-contract)，不在本节复制字段表。`scalers.json`／`category_vocab.json` 的字段级格式唯一见[落盘格式](input_output.md#scaling-artifacts)；`latent_stats.npz` 布局唯一见 [latent 标签与统计](implementation.md#implementation-5-1)。

AIDC 额外保存 `site_split.json`、`scenario_metrics.json`、`generation_audit.json` 与 `metadata.json`；具体字段、单位和质量语义只由输出合同定义。

运行状态为 `running/completed/failed/interrupted`，与项目任务 Todo/Doing/Done 不混用。写临时文件后原子替换；全部必需产物可读取且校验通过后才能 completed。

已存在 run_id 一律拒绝新训练；使用明确 checkpoint 续训时，必须处于非 completed 状态且 config/schema/数据一致。test／forecast 必须新 run_id，不能覆盖训练结果。变更节点快照属于新请求，不改变权重 schema；数据 fingerprint 记录新值而不触发“实体必须等于训练清单”的错误限制。

run_id 和 episode_id 必须是单一安全名称（字母、数字、下划线、短横线），拒绝路径分隔符与 `..`；包路径解析后必须留在包根目录，符号链接不能越界。NPZ 以 allow_pickle=False 加载，并在大张量分配前检查 manifest 维度／资源限制。标签列默认不写应用日志，不为审查上传业务数据。


<a id="work-packages"></a>

## 12. 开发工作包与进度证据

### GSLA-004 C2 机制验收（Done：用户接受结果并保留预算偏差）

用户授权明确任务/预算/停止条件并执行 C2，不包含 AIDC。输出根 `results/gslaformer/c2_20261016_110225/`，新生成 base 与 mechanisms 包，保留源码身份、数据校验与实际命令；不复用历史计算版本权重。

> 范围说明：本节开头预声明的 direct≤240 是当时的预算估算错误，已有正式入口及 acceptance §5 的合同是 44 epoch；该行保留原记录，不把后续结果追改成事前授权。最终实跑偏差与裁决见本节末尾。

- 前置门禁：当前 GSLA 回归、生成器 oracle、M5 坏输入；失败不启动训练。
- 小集：M1/full、M2/full、M3/full/direct、M4/full/direct，前八个训练 family；full 的 A/B 各最多 1000 update，direct 仅 A。CPU 单线程、seed=2023、dropout=0、lr=0.003，沿用 §6 拟合阈值，M1/M2 加严格成对方向/半间距断言。总上限 10000 update，单路线墙钟上限 20 分钟，合计墙钟上限 60 分钟；达到通过标准立即停止该阶段。A 未通过不进入 B；任一路线失败或超时，不启动正式矩阵、不自动加预算。
- 正式矩阵：M1 full/no_entity_history/direct；M2 full/no_relation_message/no_relation_all；M3 full/direct/no_latent_loss/total_only；M4 full/direct/no_spatial/total_only。各自独立 train/valid/test，no_latent_loss 复用同任务 full 的 best_A，不重训 A。保留正式 AdamW、lr=0.001、dropout=0.1、batch=8；A≤20、B-short=2、B-medium=2、B-full≤20 epoch，A/B-full patience=5、min_delta=1e-6。按每任务 train=96 窗口计算，每 epoch 12 update，full 类单路线≤528、direct≤240、no_latent_loss≤288，矩阵总上限 6288 update。每任务墙钟≤30 分钟，四任务合计≤120 分钟；在验证集上选优，test 不参与调参。
- M4：正式 full 最终权重重载 origin_activation 请求，校验 checkpoint 不变、无未来标签及新实体元数据；单列不同节点数误差，不设未声明的单调门槛。M5 不训练独立模型。
- 硬停止：NaN/Inf、坏 checkpoint/来源、数据泄漏或标签/接口不一致、OOM、未通过小集门槛、墙钟/update/epoch 上限。保留结果并标 failed/interrupted/not_run，不启动后续任务；到限不视为 passed。完整矩阵执行成功与机制门槛通过分开报告，不要求 full 胜过 direct/naive，不自动修改模型/生成分布/阈值。整轮含前置验证与审计墙钟上限 4 小时，串行执行，不争抢 CPU。

本轮实跑结果与停止回执：

- 首次 `c2_20261016_110225` 在生成 base 后失败：`write_suite` 无条件读取未生成的 M1–M4 并声明 M5，且新目录缺 mechanisms 父目录。修复为仅枚举实际存在的完整包、创建汇总目录；存在但不完整的包仍报错，不吞掉损坏。新增 fresh-directory CLI 回归通过。未改物理公式或训练数据分布，原失败日志保留。
- 修复后新目录 `results/gslaformer/c2_20261016_110225_r2/`：`env -u PYTHONPATH OMP_NUM_THREADS=1 TMPDIR=/Users/wangzf/.hermes/profiles/coder/cache/scratch .venv/bin/python results/gslaformer/c2_20261016_110225_r2/execute_c2.py`。前置 GSLA **103 项测试通过**（`regression.log`），新 base/M1–M4 包生成校验通过；M5 坏输入包通过（同级 `c2_20261016_110225_r2_M5/suite_result.json`），其 suite 保持 partial，不代表全部机制通过。
- M1/full 小集 A 达到 **1000 update** 上限，最终返回失败，执行器退出码 1。初始标准化 MSE=3.905261278152466；保留的最小训练 MSE 权重为 0.06521259248256683，小于预声明阈值 0.7810522556304932，但严格成对门槛失败。注意该权重不是最后一次更新：最后一次的 MSE=0.14353029429912567，不能混写为 best 值。
- 不增训练的 `diagnose_m1.py` 重载 `fit_M1_full/M1_full/fit_A.pt`：相同 eval/no_grad 下 MSE 与报告逐位一致；独立按 family 分组确认原 reshape 顺序无误、历史 target 相同。仅 **3/8 个 family** 满足所有 horizon 的方向与半间距断言；误差单元通过率约 0.729167，不等于 family 通过率。结果 `failure_diagnosis.json` 保留逐对预测/真值/方向/误差、初始与训练后的 latent 差异及 GRU 梯度范数。
- 历史表示存在成对差异、GRU 梯度有限且非零，不能简单归因于完全没有历史路径或完全断梯度；也不能仅凭此排除表达能力限制或认定只是预算不足。当前结论严格为“固定配置、固定种子、既定预算下，M1 机制拟合未过”。未增加 update、换 seed、调 lr、改模型或放宽阈值。
- `execution_state.json` 已记录 stopped：M1 的 B 未执行，M2/M3/M4 小集和全部正式矩阵为 not_run。任务保持 Blocked，C2 不标 Done/passed。后续需单独裁决 M1 根因实验及可能的优化预算调整，不自动扩大训练。拟合脚本新增按任务运行、失败即停及源码/计算版本记录；本轮所有更改未提交、未推送。
- 后续用户选择“先定位并修复 M1，涉及模型语义或训练预算变更先给方案”。不增加更新的 `trace_m1.py` 已运行，逐层 hook 对比初始化与 best 权重，并对两个 assets 做置换对齐，排除实体编号顺序造成的假差异。f001/f003/f004/f005 在 GRU 输出仍有可辨差异，但经组内注意力、关系层和读出后大幅减弱；例如 f001 相对激活 RMS 的成对差异依次约 0.264693 → 0.035385 → 0.000939 → 0.0000249。f000 的预测方向全部正确，但两个数值单元未满足严格半间距；不能把所有失败归成同一种情况。
- 同权重只将成对 latent 替换为各对均值，其余输入保持不变，最大预测变化 0.7623529434204102，预测成对差降为 0。这证明当前预测实际使用 latent 差异，并非绕过历史链；不证明所学表示已足够。逐层 trace 与反事实保存在 `layer_trace.json`。现有证据定位到“部分 family 的已学空间表示区分能力退化”，尚未发现可直接修复的确定实现错误；不能据此断言学习率是根因。未新增训练或更改模型，优化参数对照需用户确认后执行。
- 用户随后批准两组 M1/A 优化诊断：lr=0.001、0.0003，各至多 1000 update/10 分钟，串行；同 seed=2023、同八个 family、dropout=0、原严格阈值，不进入 B、不改正式配置。输出 `results/gslaformer/m1_lr_20261016_112826/`；每组校验初始 MSE 与原 lr=0.003 运行完全一致，记录逐对门槛、GRU/latent 差异及梯度，并重载选中权重。两组跑完后再裁决是否调整机制小集优化设置，C2 正式矩阵仍不自动解锁。
- 两组诊断已完成，`run_study.py` 退出码 0：lr=0.001 在 **271 update** 首次通过，标准化 MSE=0.003584706224501133；lr=0.0003 在 **729 update** 首次通过，MSE=0.004328960087150335。二者均 8/8 family、全部 horizon 通过严格方向与半间距断言，初始 MSE 均为 3.905261278152466；数据/transform 指纹相同、选中 A 权重重载预测逐位一致。`summary.json` 记录完整运行结果，`independent_validation.json` 从保存的预测/真值独立复算门槛通过；最小半间距余量分别为 0.00031819939613342285、0.00048300623893737793，不据此宣称大安全余量或跨种子稳定。
- 此结果证明当前模型结构在该固定小集与种子下能够拟合 M1，降低学习率足以消除本轮 A 阶段的失败，支持“优化设置敏感”而非“结构必然不可区分”的解释；不能证明 lr=0.003 普遍失败或新设置保证泛化。未调整正式配置或原验收阈值，未运行 B 或剩余 C2。建议只将 M1 小集 A 的学习率改为 0.001，沿用本次通过权重继续原设置的 B；该协议调整待用户裁决。
- 用户已批准上述恢复方案。小集脚本仅调整 M1/A 学习率，并增加受限的本机 A 权重复用入口：核验计算版本、配置、数据/transform/family 身份，重新评估原门槛通过后才进入 B，来源路径和 SHA256 随回执保存。恢复执行根为 `results/gslaformer/c2_resume_20261016_113510/`；复用原 fixtures/M5，先跑 GSLA 回归，再运行 M1/B、其余小集和正式矩阵，维持原更新与墙钟预算（原失败运行时间计入累计预算）。任一步失败即停。根 AGENTS.md 的 Doing 状态更新审批超时，未绕过审批；该行仍保留上次 Blocked 状态，实际执行以本段为准。
- 恢复运行已退出 0：M1/A 复用 271 update 的通过权重，本轮 A 新增 update=0；M1/B 在 388 update 通过，MSE=0.004668897483497858，严格成对门槛全部通过。其余小集 M2/full A/B=50/15、M3/full=6/13、M3/direct=6、M4/full=9/14、M4/direct=9 update，全部通过；本轮小集新增合计 510 update。六条小集最终权重均重载复算，MSE 与通过回执完全一致。
- M1–M4 正式矩阵 14 条路线均完成 train/test，共 4236 update；每个任务的单独 suite 正确保持 partial，由本轮 `delivery_verification.json` 聚合验收，不改写原 suite。14 个最终 checkpoint 经公开 test 入口重载，逐行对齐的预测最大差均为 0；M3 full/no_latent_loss 的 best_A 完整加载内容相同，来源指针匹配（torch.save 再序列化导致文件 SHA256 不同，不以字节相同替代内容核验）。M4 origin_activation 无标签请求成功，new_entity_count=1、no_history_entity_count=1，重载预测差为 0，来源 checkpoint 哈希不变；M5 复核原七个坏输入断言通过。M4 各变体按实际 3/5/7 个 asset 分组的 test RMSE 见交付 JSON，不宣称规模外推准确。
- 留出结果与小集门禁分开：M1/full 严格成对通过 4/16 family，M2/full 为 14/16；M1/no_entity_history 与 M2/no_relation_all 均为 0/16。M1–M4 的 full test RMSE 分别为 1.650672、0.386737、1.516823、1.933235；对应 M1/M3/M4 direct 为 1.259416、1.377667、1.522260。full 未普遍胜出，不能将运行完成解释为优于 direct 或泛化全部通过。
- **预算偏差，待用户裁决，C2 暂不标 Done**：前置预算表误把正式 direct 按仅 A=20 epoch/240 update 计算，而既有正式入口及 acceptance §5 的 direct 合同是 A+B-short+B-medium+B-full 合计 44 epoch。恢复运行前未识别该冲突；M1/direct 实际 28 epoch/336 update，超过此前声明的单路 240 update 上限。其他路线未超各自声明上限，实际总量 4236 未超原总量 6288，但总量未超不能消除单路偏差。全部训练已结束；保留原预算与原结果，不补跑、不追改成事前批准。交付状态为 `technical_checks_passed_budget_review_required`，建议接受现有结果并登记估算错误，是否认可由用户决定。
- 验证：`env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python results/gslaformer/c2_resume_20261016_113510/execute_c2.py` 退出 0，前置 GSLA 105 tests/39.833s；`verify_delivery.py` 最终运行退出 0，日志 `verification_complete.log`。初次审计将 direct 最终 phase 误认作 A、将 torch.save 副本误要求字节相同，两个审计脚本断言修正后通过，失败日志保留，不涉及模型改动。全仓 `env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests` 退出 0，200 tests/41.121s（`full_tests.log`）。未启动 C3/AIDC、未改正式配置、未提交推送。

- **最终裁决与收口**：用户明确选择“接受现有结果，保留预算偏差记录，收口 C2”。据此 GSLA-004 更新为 Done；上面的失败、待裁决及审批超时描述均保留为历史过程，最终状态以本条为准。原 `delivery_verification.json` 的 `technical_checks_passed_budget_review_required` 是裁决前技术审计回执，保持不变；单独新增同目录 `acceptance_decision.json` 绑定该回执与用户决定。接受本轮结果不等于追认超限为事前授权，不等于 full 普遍优于 direct 或留出集全部通过；GSLA-001 整体仍 Doing，C3/AIDC 不在本次授权范围。本次仅文档与回执收口，复核已有六条小集、十四条正式路线及重载结果，不新增训练。

### GSLA-003 工程可信度与等价降本（Done：仅 A+B+C1）

用户批准 A+B+C1，不包含 C2/C3 机制长训练或 AIDC 完整矩阵。沿用当前分支、计算版本 2，保留静态汇总单 Linear、日历拼接联合 Linear；D01/D02 按当前实现收口，历史审计记录保留当时语境。分批完成溯源、profile/种子、公共指标、行为回归、teacher 按 K 与冻结历史复用、有界预处理缓存、性能对照、CPU/MPS 短流程；不提交推送、不覆盖旧结果。

新增工程合同：新产物必须有带规范指纹的源码清单与环境来源；历史 checkpoint 缺源码身份时明确标为 legacy_missing，不伪造。训练来源与续训/推理代码来源分开保存，源码不同不等于计算不兼容。profile 显式输出，compare_seed 必须等于 experiment.seed。缓存仅为运行内优化，键绑定数据/配置/transform/窗口身份；预算以字节记录，0 禁用，淘汰不得改变语义；不缓存可训练计算图、跨 run teacher 或持久化 entity memory。teacher 可只生成当前监督 K 步，已知未来路径仍完整 H；生产 rollout 与 teacher 可复用同 batch 的冻结历史，memory 仍复制隔离。

本轮证据根：`results/gslaformer/gsla003_evidence/`。

- **A1–A4**：D01/D02 规格对齐，源码身份、profile/种子合同、共用预测/latent 指标已接通。新增独立手算指标、负预测不裁剪、留出站/非训练时间扰动不改变 scaler/vocab/一次更新、计划版本可见性、M2/M3 生成公式、正式配置与生成器一致、部分 suite 保持 partial、MPS 不可用失败、旧 checkpoint 来源缺失与新坏溯源拒绝等回归。生成器 profile 与正式配置名称的不一致由新增测试暴露并修正，仅改身份字段，不改数据公式。
- **B1/B2**：teacher 仅构造监督所需 K，复用同 batch 冻结历史，未来条件仍编码完整 H；预测/q/梯度/AdamW 更新等价回归通过。变换后样本使用数据/配置/transform/split/index 绑定的运行内 LRU；`preprocess_cache_bytes=0` 禁用，正式配置载荷预算 64 MiB。pickle 只隔离进程内自产样本，不加载外部缓存。预算不包含 Python 索引和活跃 batch，不是 RSS 上限。单请求按版本有效期复用 context，关系整数拓扑局部复用；CPU batching 预计算日历，时区不一致保留 fallback，不跨请求缓存可训练图或 teacher。
- **全仓回归**：`env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests`，**197 项通过，51.407 s，退出码 0**；含 CPU exact resume、共享 best_A、独立 checkpoint、比较与失败路径。日志 `full_tests_final.log`；此前失败日志保留。
- **CPU**：`bash scripts/smoke/smoke_gslaformer.sh gsla003_cpu`，退出码 0；base 训练结束于 B-full/global_epoch=38。test 354 行，四种无标签 forecast 各 6 行；test micro_mae=0.19278680974063656，仅为本次 smoke 记录。
- **MPS**：`env -u PYTHONPATH .venv/bin/python scripts/smoke/smoke_gslaformer_mps.py --prefix gsla003`，退出码 0；独立配置 `results/gslaformer/gsla003_mps_config.json` 明确 device=mps、batch=2、四阶段各一 epoch，最终 global_epoch=4。test 354 行，四种 forecast 各 6 行；此预算不与 CPU 比精度。
- **产物重载**：CPU 增补同 checkpoint 的 `gsla003_cpu_reload`，MPS 脚本生成 `gsla003_mps_reload`。`env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python results/gslaformer/gsla003_evidence/verify_delivery.py` 退出码 0；14 个 run 均 completed，最终 checkpoint 均计算版本 2/B-full；读回验证来源、产物合同、行数及 forecast 无未来标签，CPU/MPS 正常请求重复重载最大预测差均 **0**。回执 `delivery_verification.json`。

**B3 有界同配置对照**：基线为实施前 `7ae9e68` 导出源码，不是旧计算版本的性能数字。`paired_probe.py` 分别在独立进程运行，CPU 单线程、同一 multiday 包、L=168/H=72/batch=2、相同种子/初始化；每阶段固定首 batch，1 次预热 + 3 次计量更新，共五组进程对。B-short K=1、B-medium K=3、B-full K=72；使用相同固定 latent 权重隔离计算复用，**不是完整课程、拟合方差验收或完整 epoch**。原始报告在 `results/gslaformer/gsla003_perf_{reference,optimized}_{1..5}/paired_report.json`，脚本与日志保留于证据根。

| 阶段 | 基线中位数 s/update | 当前中位数 s/update | 本次中位数降幅 |
| --- | ---: | ---: | ---: |
| A | 0.286510 | 0.264937 | 7.53% |
| B-short | 0.407153 | 0.073086 | 82.05% |
| B-medium | 0.429066 | 0.094471 | 77.98% |
| B-full | 0.948854 | 0.862606 | 9.09% |

所有配对更新（含预热）的 loss 和完整模型参数哈希逐位一致。每侧每阶段计量 15 次，范围与 RSS 原值见 `delivery_verification.json`。B-full 基线区间 0.863709–1.131986 s，当前 0.780570–1.175950 s，存在重叠，不能保证稳定加速。当前 B-full 分段中位数：取样/变换/collate 0.002285 s、forward+teacher 0.684665 s、backward+optimizer 0.154841 s；分段中位数不要求相加等于总中位数。剩余成本主要在模型计算，未进一步拆解，不把 teacher 复用描述为全链瓶颈的解决。固定 batch 的热缓存收益不代表站点随机采样命中率；RSS 有波动，不宣称内存下降。

自审按规范与规格两轴在同一上下文完成，不声称独立多 Agent 审核。未改模型函数族、正式训练课程/优化器/学习率或生成分布；未新增依赖、访问凭据、修改 CI、提交或推送。C2/C3 与完整机制/AIDC 矩阵仍需另批预算，GSLA-001 不因此标 Done。首次收尾时根 AGENTS.md 的 Doing→Done 审批超时，未绕道写入；随后用户明确要求重新发起审批，本次获准同步 GSLA-003 为 Done。

### DOC-009 当前架构图与原始需求对照（Done：审计交付，不代表模型完整验收）

目录迁移追加（Done）：按用户要求将本轮图稿、报告与证据收进 `arch/gpt-version/`，四个配套脚本收进 `arch/gpt-version/scripts/`。迁移清单完整、旧位置无残留，三组 SVG／PNG／PDF 字节一致，`bot_version/` 清单及哈希不变；从仓库外独立子进程加载四个脚本并核对根路径通过，实际执行新路径的 render／verify 均退出 0，新地址报告经隔离浏览器加载与截图核验通过。仅修路径与引用，不改图形语义，不重跑模型。根 AGENTS 状态写入审批超时，未绕道修改；迁移证据在本节记账。

以 model-v4.0.png 为原始需求、当前未提交工作区为实现事实，交付 [GPT 版离线报告](../arch/gpt-version/index.html)、当前整体架构／机制展开／需求对照三组 SVG、PNG、矢量 PDF，以及 [逐项映射](../arch/gpt-version/comparison.md)。保留原始主闭环，分别标出新增、替代与收窄；本目录是审计快照，不恢复已退役的 architecture/，不替代现行四份规格。

验证命令与实际结果（以下命令已同步到迁移后的等价入口；模型运行和浏览器导出结果沿用原审计证据，目录迁移本身不重跑模型）：

- `env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/probe.py`：退出码 0；复用正式 AIDC day profile 和既有 synthetic 包，L=168、H=24，有界随机初始化前向通过。实际捕获并逐值核对第二步输入消费第一步生成的 y/z、生成态标记，验证 direct/rollout 首步一致和 B 阶段表示冻结；内嵌运行模型、契约修复、训练支持三组单测，**45 项通过，0 失败／错误／跳过**。机器记录见 [runtime.json](../arch/gpt-version/evidence/runtime.json) 与 [定向日志](../arch/gpt-version/evidence/targeted-tests.log)。
- `env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/render.py`：退出码 0；15 项原图映射、2 项现行规格差异，源码哈希与行号范围检查通过。
- `env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/export.py`：退出码 0；隔离 Chrome profile 导出并核查三组图，文本无越界、连线不穿节点，PDF 各一页；桌面／窄屏无页面横溢，禁用页面 JS 后内容仍完整。回执见 [visual-check.json](../arch/gpt-version/evidence/visual-check.json)，图形及页面截图另经逐图视觉核对。

- `env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/verify.py`：退出码 0；上级规格与本次报告的本地链接／锚点、内嵌原图字节、SVG 一致性、PNG／PDF 完整性、源码哈希及 `git diff --check` 通过，交付清单见 [delivery-check.json](../arch/gpt-version/evidence/delivery-check.json)。不将并行会话图稿子目录列入本次交付。

审计偏差 **D01**：§1.3 写拼接组静态投影后使用 MLP，代码实际为单 Linear；**D02**：§1.4 写日历独立投影相加，代码为日历拼接后联合 Linear。两者按实际源码绘图，差异与处理代价见报告；本次不改写上述规格或擅改模型，后续需裁决。没有新增依赖、运行长训练或修改模型／配置，未提交推送。

### DOC-010 历史文档精简（Done）

按用户确认将 archive 四份文档提炼为 history.md：保留设计取舍、计算版本兼容边界、失败教训、负面对照结果、资源授权边界及证据/Git 恢复入口；删除重复流水账与 archive 目录。未提交增量的必要信息已提炼，明确不承诺舍弃全文可从 Git 恢复。同步根 AGENTS、导航、设计、实施与验收引用，不触及并行开发的 arch/。

本轮仅文档验证：根 `.venv` 执行 `env -u PYTHONPATH .venv/bin/python /Users/wangzf/.hermes/profiles/coder/cache/scratch/check_doc010.py`，退出码 0；本地链接/锚点、围栏、原图哈希通过，历史索引的 10 个结果路径与 6 个 Git 文件恢复入口均存在，旧 archive 无运行代码引用且目录已删除。脚本为本轮只读核验工具，不是长期模型测试入口。`git diff --check` 通过；没有重跑训练/模型测试，没有修改运行代码、配置或实验结果，未提交推送。

### DOC-008 配置单源化（Done）

用户确认清理 examples：删除两份 JSON 副本及空目录，文档直接链接 scripts/smoke 的正式配置；数据合同、标准包与业务映射测试同步迁移，base/AIDC 路径解析、schema 保全、站点角色隔离和数值业务断言保留。只移除已无对象的“文档副本等于正式配置”测试，不改模型或正式配置。下方 GSLA-002 的样例同步记录保留历史语境，现不再维护镜像。

验证：`env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py'`，86 项通过、退出码 0。日志 `results/gslaformer/doc008_evidence/gsla_doc008_tests.log`。11 份 Markdown 的 103 处本地链接/锚点及代码围栏检查通过；运行代码与测试无旧样例引用，目录已不存在；`git diff --check` 通过。未启动训练，未提交或推送。

### GSLA-002 规格对齐（Done：本轮工程范围，不代表整体模型验收）

本轮按审计差异执行：未来条件改为 calendar/known/mask 拼接投影；组读出沿用 spatial_heads；在 collate 分配前按实际 batch 和补齐节点检查注意力预算；checkpoint 携带统计溯源与冻结参数清单；补训练/预测资源与关系/站点角色审计。数据 schema 保持 3，模型计算版本显式升级，旧 checkpoint 不自动迁移或继续训练；旧实验保留历史语境，不为新实现背书。根 AGENTS 的首次写入审批超时；用户随后明确批准，本轮已正式登记，不绕道改写。

静态条件的本轮决定：保留已实现的“有效属性统计、类别 embedding 均值、关系度数汇总 → 组投影”路径，明确它不是逐实体非线性适配后池化；不新增未明确的静态网络。修复度数分母排除未生效实体。未来预测输入仍通过 Batch.inputs 隔离，Batch.supervision 仅由实验支持层读取；文档以实际边界类型命名，不宣称存在另一套未实现 API。

本轮验证（2026-10-15，计算版本 2）：

- 修复前新增未来 mask、合批预算、读出 head 用例真实失败；溯源/资源/metadata 断言亦先失败再修复。红绿日志统一归档在 `results/gslaformer/contract_v2_evidence_20261015/`。
- `env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`：182 项通过、退出码 0；包含 CPU exact resume、独立 checkpoint 推理、缺失溯源/旧版本/冻结清单错误的公开入口拒绝，以及真实 compare/基线行为。不是完整 AIDC 矩阵。
- `bash scripts/smoke/smoke_gslaformer.sh contract_v2_final_20261015`：退出码 0。CPU base 完整 A/B train、test、normal/new_entity_cold/new_entity_history/new_parent 六个 run 均 completed；最终 best 为版本 2 的 B-full。test 354 行，各无标签请求 6 行，未写未来真值。
- 使用 `results/gslaformer/functional_mps_i5_config.json`，经 `run.py --model GSLAformer --gsla_mode train/test/forecast` 执行 `contract_v2_mps_20261015`、同前缀 `_test` 和 `_forecast`；全部退出码 0、completed，禁用 MPS fallback。四阶段各 1 epoch，是设备短预算验证，不用于与 CPU 正式预算比较精度。
- AIDC 文档样例现能加载既有标准包的 valid 分区（240 窗口）；两份文档样例与正式配置的解析一致性由测试锁定，不重新生成数据或启动长训练。
- 校验 11 份 Markdown、103 处本地链接/锚点与代码围栏，保留原图哈希一致，退役文件无残留有效引用；`git diff --check` 通过。只检查剩余资产，不把已退役图示算作当前门禁。
- 自审区分项目规范与规格语义：未改依赖、密钥、CI 或旧模型链；未降低 L/H/梯度长度；历史图示与回执只按用户确认清理。所有修改未提交、未推送。

### 当前工作包与剩余边界

| ID | 状态 | 当前事实 | 尚未收口 |
| --- | --- | --- | --- |
| I1 | Done | schema 3、标准包、scaler、生成器与站点切分已实现 | 真实来源适配不在本轮范围 |
| I2 / I3 | Doing | 空间/关系/静态/未来条件与回灌已实现；本轮修正计算契约 | 新计算版本的完整机制验收与跨站场景，不能继承旧版本实验结论 |
| I4 / I5 | Doing | A/B、CPU exact resume、独立 checkpoint、CLI 与产物门禁已接通 | AIDC 长 horizon 全课程、完整资源预算与新站预测 |
| I6 | Doing | 比较与基线入口存在，工程回归单独验收 | AIDC 三 profile 全矩阵与最终报告，预算未授权 |

旧实现曾完成 CPU/MPS 短流程、机制对照、小集拟合以及单个完整 CPU A epoch；这些仍是历史事实，但不是计算版本 2 的训练或性能结论。不得自动续用旧 best_A/best.pt，也不得因本次短流程通过而启动长矩阵。

### 证据入口与资料清理边界

- 当前回归／短流程的命令、状态及证据由本节统一记录，不在其他 README 复制计数。
- [历史证据索引](history.md#experiment-evidence)：旧实验及资源测量的定位与边界，不将旧结果当新版本通过。
- [关键决策与失败教训](history.md#design-decisions)：保留语义决定与偏差，不再保存重复流水账；不以更新文档替代修复错误。
- 用户已批准合并并删除 reorganization.md、validation_r4.md、validation_checks.json，摘要与原文恢复入口见 [历史文档摘要](history.md#doc-reorganization)；随后用户明确确认 architecture/ 外部删除，清理相关引用，仅保留 model-v4.0.png 原图。全部实验结果/权重不删除。
- 根 AGENTS.md 已在后续明确授权下登记 GSLA-002 并更新历史证据链接。
