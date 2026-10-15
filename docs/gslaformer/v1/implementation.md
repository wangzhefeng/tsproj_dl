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

独立静态链只读 origin 有效实体的静态数值／类别、mask、按类型与方向的平均关系度数，不读动态值。先按组聚合：各数值属性的有效 masked mean、sum、count，类别 embedding 的有效均值，`log1p(有效节点数)` 及平均入／出度；再由每组 Linear→GELU 投影至 `d_z`，拼全部组后 Linear→GELU→Linear 投影至 `d_t` 得到 c。不存在额外的“逐实体非线性适配再池化”；动态实体适配器与静态链参数独立。空组统计为零，度数分母排除 padding 和 origin 未生效实体。

静态链与动态链分别建模块，B 阶段一起冻结。数量／sum 保留规模信息，但不构成总功率守恒或严格单调性保证。

<a id="implementation-3-4"></a>

### 1.4 历史 token 与时间编码

动态状态保留 `[B,L,Gd,d_z]`，送时间模型时按固定组顺序展平末两维。每个 token 再拼：

- 标准化总负荷；
- 总负荷 observed、generated 两个标记；
- 每动态组 observed coverage 和 generated 标记。

得到输入后 Linear 映射至 `d_t`。位置编码使用窗口内 `0..L-1` 的固定正弦编码，滑窗重算；日历为本地日内相位与周内相位的 sin/cos 四个数，再经 Linear→`d_t` 相加。日历函数适配任意固定秒级频率，不通过拼接未来真实观测计算。

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
| `tests/test_gslaformer_training.py` | 冻结、梯度、loss、状态机、resume、过拟合 |
| `tests/test_gslaformer_integration.py` | CLI、产物、重载、synthetic E2E 和 MPS |

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
