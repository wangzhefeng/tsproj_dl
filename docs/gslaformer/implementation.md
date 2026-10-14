# GSLAformer R4：开发实施规格

> 本文唯一维护模型内部计算、损失与指标算法、训练恢复、源码复用、CLI 接入、产物生命周期和工作包。业务数据／张量见[输入输出契约](input_output.md)，运行矩阵和通过标准见[验收协议](acceptance.md)。
>
> 状态：用户已授权 GSLA-001 进入实施，模型和脚本尚待完成，拟议命令不是运行证据。数值是工程初值，不是用户业务值或实测最优值。根 AGENTS.md 管项目状态；工作包完成必须附实际证据。

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
6. 每组用独立的可学习读出 query 对关系增强实体做一次 cross-attention；补一个可学习空组 token 作为始终有效的 key。拼接 `log1p(有效实体数)`、当前动态观测覆盖率，再经 Linear 映射到 `d_z`、LayerNorm，得到规范 latent。仅动态组生成 latent 目标。

关系端点重命名或同时重排不改变语义。改变父级可能改变消息，但不承诺任何一次重连都改变任意已训练模型的输出。

允许边约束是关系层局部契约，不是整个模型的隔离保证：组内全连接注意力已经混合了同组信息。关系层禁止 key_padding_mask 再次屏蔽无效 query 的安全 self，必须用统一的加性 mask 处理该情形；无效输出归零后不参与任何有效节点计算。

<a id="implementation-3-3"></a>

### 1.3 静态条件 c

独立静态适配器只读静态数值／类别、mask、按类型与方向分别统计的关系度数、起点有效实体集合，不读动态值。按组对属性表示取 masked mean；另拼静态数值的 masked mean、masked sum、有效数量及 `log1p(节点数)`，每组 Linear→GELU 映射至 `d_z`；拼全部组后投影至 `d_t` 得到 c。空组均值置零，数量为零。

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


AIDC 类型关系的业务方向和合法性以[输入输出契约](input_output.md#input-output-3-1)为准；q 的两跳集合从全部启用类型通信图计算。

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

输出为 `ForecastResult.y [B,H,1]`、`z [B,H,Gd,d_z]`、元数据与质量标记。模型内部 y 标准化，落盘 y 逆变换为输入 manifest 的原始单位；z 保持规范坐标，注明非物理量。

<a id="implementation-5"></a>

## 3. 标签、损失与缺失规则

<a id="implementation-5-1"></a>

### 3.1 latent 标签与统计

加载 best_A 后冻结适配器、实体 GRU、空间／关系／读出／LayerNorm／静态链。在 `no_grad + eval` 下先编码与模型相同的历史，复制末端实体 memory，再按 1..H 消费 future_groups 的数值与 mask。未来属性／present／关系保持 origin 上下文，每一步生成 teacher z；不能每步从零重建 GRU，也不能读取 future target 生成 z。

标签是**带缺失条件的可用信息表示**，不是完整物理状态。允许部分观测，质量只软加权，不设全系统 all-observed 门槛：

1. 对动态组 g 的时刻 s，r[g,s] 为最近 L 步内所有 present、非 padding、applicable 动态单元的 observed 比例；窗口起点早于本请求历史时截断到已编码起点。分母为零时 r=0。
2. V[g,s] 为从 g 的有效实体沿至多两条已启用类型通信边可达的其他动态组集合（按组去重）。r_neighbor 为这些组 r 的算术均值；集合为空取 1。
3. `q[g,s]=r[g,s]*(0.5+0.5*r_neighbor)`；组在 s 为空时 q=0。r 与 q 只由真实 mask／图计算，禁止用可学习 attention 权重提高自己的监督权重。
4. 该 q 是有界质量启发式，不是统计置信度，不覆盖全部组内混合造成的依赖；部分缺失 teacher 仍是噪声目标。记录 q 分布、有效率和缺失率敏感性，不能宣称缺失已完全解决。
5. 未来 q 与 mask 仅进入 loss／诊断，生产生成 token 仍用 generated=true、coverage=0，不泄漏未来观测质量。

latent 统计只取训练 split 的每个唯一 `(episode, origin)` **最后一个历史 z** 一次，并存该请求数据／版本 fingerprint；不收集重叠窗口中的其他 z，也不收集未来 teacher z。重复键内容不一致时报错。各维以 origin 的 q 作为权重计算均值和总体方差；该组至少有两个 q>0 的训练 origin，否则无法估计而报错。B 不重拟合。

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

## 7. 拟议文件与内部 API

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
| `scripts/smoke/gslaformer_functional.json` | [配置定义与设计实例](implementation.md#configuration) 的完整功能配置 |
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

<a id="implementation-7-2"></a>

## 8. 现有入口的隔离接入

已读现有 `run.py`：它的 setting 拼接依赖普通时序参数，三个运行开关全为零还会触发“最终训练”。GSLAformer 不能只插一个实验类后继续沿用这两个隐式行为。

增加专用 `--gsla_config`、`--gsla_mode`、`--gsla_run_id`、`--gsla_checkpoint`、`--gsla_request`；模式仅为 `train/test/forecast/compare`。`--gsla_checkpoint` 在 train 表示续训，test／forecast 表示权重来源。request 只用于 forecast。

为 GSLAformer 加隔离参数解析分支：预解析发现 `--model GSLAformer` 后走其专用配置与模式校验，不要求虚假的 `--enc_in/--dec_in/--root_path` 参数；其他模型保持当前 parser、required 参数和 run 逻辑。GSLAformer 禁止旧 `is_training/is_testing/is_forecasting` 混用，冲突报错；不扩大修改旧入口的范围。

`run(args)` 对 GSLAformer 先分派并返回，不进入旧 setting 和“全零即最终训练”。`_select_experiment_class` 增加明确路由；专用参数适配成 Exp_Basic 所需 `use_gpu/gpu_type/devices/use_multi_gpu`。明确请求 MPS/CUDA 时先检查可用性，禁止继承基类的静默 CPU 回退。模型发现保持 `Model` 入口，不另造注册框架。

配置拥有模型、数据、训练、设备设置；CLI 只提供运行模式、配置路径、run_id、checkpoint 和请求路径，不允许散落 CLI 覆盖模型参数。相对 config 的 data.package 路径以 config 所在目录为基准；results_root 以仓库根目录为基准。生产请求路径由 CLI 指定，不修改训练 schema。


<a id="configuration"></a>

## 9. 配置定义与设计实例

[周期 base 样例](examples/functional_profile.example.json)与[AIDC 一天样例](examples/aidc_profile.example.json)均为 schema 3 的完整实例，不是当前可执行配置，二者不共用权重。

| 键组 | 含义与约束的权威位置 |
| --- | --- |
| schema_version | 输入输出契约的数据／模型版本，不与绘图格式版本混用 |
| experiment | model 固定 GSLAformer；variant 见验收对照表；profile 标识独立任务，data_kind 为 synthetic；seed/device/results_root 明确保存 |
| data.package / timezone / freq / target / groups / relation_types / known_future / feature_units | 输入输出契约的来源、单位、对象与外部数据 schema；relative package 按实际配置文件目录解析 |
| data.seq_len / pred_len / stride / split_mode / split_index / site_split | 输入输出定义切窗与分区语义，验收协议唯一维护各任务的具体值及站点角色 |
| data.sampler / batch_size / num_workers | 本文的训练采样、设备与恢复约束；参数实例见完整配置 |
| model | 本文的实体／空间／关系／时间／未来路径计算；不得用字段偷偷切换另一架构 |
| limits | 本节安全分配规则；实例限额不是峰值内存或性能承诺 |
| training | 本文的优化器、损失、课程、早停与梯度规则 |
| evaluation | 本文定义指标算法，验收定义选择指标、报告粒度、季节周期与比较种子 |

固定规范：FFN 倍数=4、Post-LN、GELU、组级 scaler、单层实体 GRU、两层关系注意力、单层未来路径编码、窗口正弦编码、[标签、损失与缺失规则](implementation.md#implementation-5) 质量加权与有界方差权重。layers 字段只允许上述首版值，不默默引入另一种架构。维度整除 head 数；L/H 正整数；至少一个动态组；组／特征唯一；variant 限 [基线、消融与诊断定义](acceptance.md#implementation-10-1)。

limits 是参考实现的安全边界，不是已测性能承诺：包括 batch padding 后的真实节点轴，不通过截断节点满足限制。分配前按 `4*B*L*relation_heads*N_total^2` 估算单个显式 FP32 分数张量，再检查各组空间层及时间层；任一超过字节上限时报错，显示估算。它不等于峰值内存估算，forward/backward 仍须实测；OOM 不能捕获后静默降级。上调 limit 需独立容量验证，首轮不宣称无限扩展。

周期样例的 L≤96/H≤24 仅为周期 smoke 配置。AIDC 拟议 profile 明确采用 L≤168/H≤72、batch=2，支持小时至三天的验收目标；这些更大限额仍需实跑，不能把配置存在当性能通过。15 分钟／万级服务器属于另行容量验证范围，不截断输入或隐式预聚合来满足限额。


实施时生成 `scripts/smoke/gslaformer_functional.json`，以及同目录的 `gslaformer_aidc_hourly_short.json`、`gslaformer_aidc_hourly_day.json`、`gslaformer_aidc_hourly_multiday.json` 三份完整物理配置。不依赖隐式继承或 CLI 模型参数覆盖；准确 L/H、stride 与运行范围见验收。examples/ 比目标 scripts/smoke/ 多一层，相对 data.package 必须重新解析并改写；复制 JSON 后不改路径会指错目录。

Python 使用项目根 .venv；核心基于已有 PyTorch，无新账号或外部服务。若确需依赖，按项目规则以 uv add 管理并说明必要性。新目录添加职责 README，不建抽象空壳。

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

predictions、latent 与 metadata 的字段及物理含义唯一见[输出合同](input_output.md#input-output-7-2)，不在本节复制字段表。

AIDC 额外保存 `site_split.json`、`scenario_metrics.json`、`generation_audit.json` 与 `metadata.json`；具体字段、单位和质量语义只由输出合同定义。

运行状态为 `running/completed/failed/interrupted`，与项目任务 Todo/Doing/Done 不混用。写临时文件后原子替换；全部必需产物可读取且校验通过后才能 completed。

已存在 run_id 一律拒绝新训练；使用明确 checkpoint 续训时，必须处于非 completed 状态且 config/schema/数据一致。test／forecast 必须新 run_id，不能覆盖训练结果。变更节点快照属于新请求，不改变权重 schema；数据 fingerprint 记录新值而不触发“实体必须等于训练清单”的错误限制。

run_id 和 episode_id 必须是单一安全名称（字母、数字、下划线、短横线），拒绝路径分隔符与 `..`；包路径解析后必须留在包根目录，符号链接不能越界。NPZ 以 allow_pickle=False 加载，并在大张量分配前检查 manifest 维度／资源限制。标签列默认不写应用日志，不为审查上传业务数据。


<a id="work-packages"></a>

## 12. 开发工作包与进度证据

以下将原 I1–I6 与 R4 补项直接合为一张执行表，不再维护另一份增补清单。GSLA-001 已开工；完成任一工作包不等于整体交付。

| ID | 状态 | 工作包 | 前置 | 验收证据 |
| --- | --- | --- | --- | --- |
| I1 | Doing | schema/config/scaling、AIDC mapping、标准包与 base/M1–M5/AIDC 生成器 | 编码授权、AGENTS 登记 | JSON/NPZ、单位／计量、typed relations、split/as-of/scaler、站点隔离、oracle 与数据测试 |
| I2 | Todo | 实体历史、组内空间、类型关系、读出与模型入口 | I1 | 轨迹、shape、mask、空组、同步重排、新实体免参数变化、同组多类型／同端点多关系、跨样本独立 |
| I3 | Todo | temporal/future_path/fusion/heads/rollout | I2 | 完整条件公平性、teacher memory 隔离、双头索引、因果与回灌梯度、H=6/24/72 对齐 |
| I4 | Todo | 实验支持、A/B 训练与恢复、指标 | I3 | 冻结、课程、reduction、小集拟合、CPU exact resume、站点平衡采样与 site-macro、统计 fingerprint |
| I5 | Todo | run.py 隔离路由、profile、smoke、模式产物 | I4 | train/test/无标签新实体及新站 forecast、自包含 checkpoint 重载、CPU/MPS 分报、旧路由定向回归 |
| I6 | Todo | 统一验收矩阵、比较脚本、完整报告与文档收口 | I5 | base、M1–M5、三个 AIDC horizon、全部必需路线及 suite_result；负面结果、容量、产物和根治理状态 |

每个包记录实际命令、退出码、测试计数、产物路径和偏差；没有证据不能标 Done。训练 run 的 running/completed 等状态与项目任务状态分开。预算、阈值或算法变更先留依据与批准记录，再实施，不追改历史成“原本如此”。偏差追加到[设计演进记录](archive/design_history.md)。

本次已完成的窄切片：input_output §6.6 固定标准包与指纹、§10 固定质量标记；acceptance §18.6 固定 aidc-v1 模拟假设；schema.py 已实现严格 JSON、规范 JSON 指纹和单请求质量计算。先运行定向测试观察到缺模块的预期红灯，再实现并运行 `env -u PYTHONPATH TMPDIR=/Users/wangzf/.hermes/profiles/coder/cache/scratch .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_data.py' -v`：3 项通过、退出码 0。它们不证明标准包加载、模型或 I1 全部完成；tests 仍按用户决定被忽略，不擅改 .gitignore。

实施前重新检查分支、目录规则和源码漂移。新增 tests 的版本化例外须精准处理，不删除已有测试；不盲目 import 全仓重依赖模型。普通 parser／模型发现／TFs_type 路由只做受影响的定向回归。根 README 在功能实际完成后补运行说明，不在文档阶段宣称可运行。
