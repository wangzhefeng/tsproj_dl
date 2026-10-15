# GSLAformer：当前架构与原始 v4.0 需求对照

> 审计时间：2026-10-15T17:02:13.517739+08:00。分支 `gslaformer`，基准提交 `5a50d02a0808b64c29b59eaa627c5c648895244e`，**包含未提交工作区实现**；精确源码快照见 [运行证据](evidence/runtime.json)。

## 1. 结论与边界

原始闭环（分组空间编码→空间横拼／时间纵拼→因果时间模型→负荷／下一状态双头→回灌）已实现。当前不是原图逐字复刻：增加实体 GRU、类型关系、查询池化、总表历史、静态条件、已知未来路径与 A/B 训练；独立 Space PE 被属性／关系编码替代。

只把原图可辨认的节点与箭头当作原始要求；图未规定的池化、损失、实体身份等记为后续具体化或扩展，不倒推为原图缺陷。当前图以工作区 full 变体和 AIDC day profile 为主，不代表模型已完成全量验收。

“任何时序结构”尚未成为可配置后端，空间 Nx 首版固定为一层。不能以结构核验、随机初始化前向或定向测试通过推断完整模拟验收、真实精度或相对对照收益。

## 2. 图形

- [离线完整报告](index.html)
- [当前架构 SVG](current-architecture.svg) / [PNG](current-architecture.png) / [PDF](current-architecture.pdf)
- [机制展开 SVG](mechanisms.svg) / [PNG](mechanisms.png) / [PDF](mechanisms.pdf)
- [原始与当前对照 SVG](comparison.svg) / [PNG](comparison.png) / [PDF](comparison.pdf)

## 3. 逐项需求映射

### C01 输入分组｜演进

- **原图**：用户层、机房层、楼宇层各有独立空间分支，图中省略号允许更多分支。
- **实现**：AIDC profile 为算力、IT 电力、配电、暖通、空间、环境六组；机房与楼宇通过 space 的类别属性区分，用户／业务池映射到 compute，而非保留原来三个固定分支。
- **判断**：保留按组建模思想，业务分类已变化。不能将当前图标成对原始三个分支的逐字复刻；模型一般性分组由 schema 决定。
- **证据**：[scripts/smoke/gslaformer_aidc_hourly_day.json:31-123](../../../../scripts/smoke/gslaformer_aidc_hourly_day.json#L31)；[docs/gslaformer/v1/input_output.md:56-71](../../v1/input_output.md#L56)

### C02 空间位置编码｜替代

- **原图**：每组进入空间注意力前显式加 Space Positional Encoding；图未说明编码基于行号、身份、坐标还是拓扑。
- **实现**：不加实体序号位置编码，也不使用实体 ID embedding。静态数值／类别与观测 mask 进入组适配器，拓扑通过类型关系注意力进入；只有 schema 声明的坐标特征才可能作为属性消费。
- **判断**：不是漏画：原图独立加法式 Space PE 没有原样实现；R4 有意选择属性与关系归纳编码，以支持可变实体及同步重排不变性。不能说空间信息全部删除，也不能声称已有几何相对位置编码。
- **证据**：[layers/gslaformer/entity_history.py:19-43](../../../../layers/gslaformer/entity_history.py#L19)；[docs/gslaformer/v1/design.md:29-35](../../v1/design.md#L29)；[docs/gslaformer/v1/implementation.md:13-27](../../v1/implementation.md#L13)

### C03 池化前实体历史｜新增

- **原图**：图展示逐时刻的空间分支，再沿时间堆叠；没有独立实体循环单元。
- **实现**：每个动态组先用属性适配器，再用同组实体共享的单层单向 GRUCell，生成逐实体因果历史表示；纯静态组无 GRU。
- **判断**：当前模型不只是纯空间 Transformer 接时间 Transformer，而是实体 GRU→空间交互→系统时间建模，避免汇总前丢失实体轨迹。
- **证据**：[layers/gslaformer/entity_history.py:46-106](../../../../layers/gslaformer/entity_history.py#L46)；[models/transformer/GSLAformer.py:98-100](../../../../models/transformer/GSLAformer.py#L98)

### C04 组内空间 Transformer｜保留并收窄

- **原图**：各组 Multi-Head Attention→Add & Norm→Feed Forward→Add & Norm，以 Nx 标重复。
- **实现**：各组一层 Post-LN TransformerEncoderLayer，GELU、FFN 宽度 4d_s；加入可学习摘要 token 与实体有效性 mask。配置校验将 spatial_layers 固定为 1。
- **判断**：注意力／残差／归一化骨架保留，但不是任意 Nx 堆叠。摘要 token 在该层参与注意力，输出后被剥离，不直接当作最终组向量。
- **证据**：[layers/gslaformer/spatial.py:6-25](../../../../layers/gslaformer/spatial.py#L6)；[data_provider/spatial_temporal/config.py:170-183](../../../../data_provider/spatial_temporal/config.py#L170)

### C05 实体级跨组关系｜新增

- **原图**：各组向量在读出后横向拼接；没有显式画出归属边或跨组实体级消息。
- **实现**：组内输出先拼实体轴，再经过两层带类型／方向偏置的关系注意力，按源→接收方方向和显式反向通信开边，随后切回各组。
- **判断**：当前不是仅后期拼接：在组读出前已有跨组信息交互。关系层的图 mask 不代表整个网络只沿物理边传播，因为组内注意力已经混合同组实体。
- **证据**：[layers/gslaformer/relations.py:17-105](../../../../layers/gslaformer/relations.py#L17)；[models/transformer/GSLAformer.py:223-247](../../../../models/transformer/GSLAformer.py#L223)；[docs/gslaformer/v1/design.md:45-55](../../v1/design.md#L45)

### C06 固定宽度组读出｜具体化

- **原图**：每个空间分支输出一个向量，但没有指定从多实体到向量的聚合算子。
- **实现**：每动态组可学习 query 对关系增强实体及空组 token 做 cross-attention；拼 log1p(有效实体数) 与观测覆盖率，经 Linear→LayerNorm 得到 d_z 维规范状态。
- **判断**：不是简单展平，也不是直接取空间摘要 token。实体数可变，但组向量宽度固定；纯静态组无 latent 头。
- **证据**：[layers/gslaformer/spatial.py:28-52](../../../../layers/gslaformer/spatial.py#L28)；[models/transformer/GSLAformer.py:248-261](../../../../models/transformer/GSLAformer.py#L248)

### C07 空间横拼／时间纵拼｜保留

- **原图**：空间向量横向拼接，按时间顺序纵向堆叠成 time series。
- **实现**：读出按 schema 动态组顺序堆成 z[B,L,Gd,d_z]，进入时间链时仅展平 Gd 与 d_z，保留 L 时间轴。
- **判断**：原图的组织原则保留；AIDC 六动态组每组 32 维，对应每时刻 192 维规范状态。
- **证据**：[models/transformer/GSLAformer.py:248-261](../../../../models/transformer/GSLAformer.py#L248)；[models/transformer/GSLAformer.py:280-294](../../../../models/transformer/GSLAformer.py#L280)；[scripts/smoke/gslaformer_aidc_hourly_day.json:259-274](../../../../scripts/smoke/gslaformer_aidc_hourly_day.json#L259)

### C08 总表历史与来源标记｜新增

- **原图**：时间输入只明确画出拼接后的空间向量，未单独画总负荷历史旁路与质量标记。
- **实现**：时间 token 还拼入总负荷 y、总表 observed/generated、每组 coverage/generated 及四维日历；AIDC profile 输入宽度为 211。
- **判断**：时间模型不必只从内部实体表示重建总负荷。该旁路有实用价值，也可能降低对空间分支的依赖，结构存在不证明空间分支带来收益。
- **证据**：[models/transformer/GSLAformer.py:280-294](../../../../models/transformer/GSLAformer.py#L280)；[layers/gslaformer/temporal.py:68-93](../../../../layers/gslaformer/temporal.py#L68)；[docs/gslaformer/v1/design.md:63-77](../../v1/design.md#L63)

### C09 因果时间模块与可替换性｜保留并收窄

- **原图**：time Positional Encoding＋Masked Multi-Head Attention 的 Nx 堆叠，并备注“任何时序结构都可适用”。
- **实现**：固定正弦窗口位置编码＋显式因果 Post-LN Transformer；day profile 为两层，取窗口最后 hidden。没有时间后端选择器，未实现任意 RNN/TCN/其他结构的配置替换。
- **判断**：原图画出的因果 Transformer 已有；“任意时序结构”当前只能视为设计可替换思想，不能宣称已经工程化支持。时间层数可配置，空间层数首版固定。
- **证据**：[layers/gslaformer/temporal.py:68-93](../../../../layers/gslaformer/temporal.py#L68)；[models/transformer/GSLAformer.py:118-120](../../../../models/transformer/GSLAformer.py#L118)；[data_provider/spatial_temporal/config.py:170-183](../../../../data_provider/spatial_temporal/config.py#L170)

### C10 总负荷／下一状态双头｜保留并具体化

- **原图**：时间模块输出分别经过 Linear 生成总负荷、multi Linear 生成 next embedding。
- **实现**：条件融合后共享同一隐向量 a，总负荷 Linear(d_t,1)，每动态组独立 Linear(d_t,d_z)。预测 z 位于组读出的规范坐标，不是时间 hidden，不加额外 LayerNorm。
- **判断**：双头主旨保留；状态头按组拆分且与负荷头预测同一下一时刻。原图未定义 next embedding 的精确坐标，当前已经明确。
- **证据**：[models/transformer/GSLAformer.py:125-131](../../../../models/transformer/GSLAformer.py#L125)；[models/transformer/GSLAformer.py:306-314](../../../../models/transformer/GSLAformer.py#L306)；[docs/gslaformer/v1/implementation.md:80-88](../../v1/implementation.md#L80)

### C11 回灌闭环与窗口｜保留并扩展

- **原图**：next embedding 箭头返回 time series；没有画总负荷回灌，也未规定无限增长还是滑动窗口。
- **实现**：每步同时追加预测 z 和同一个预测 y，窗口保持 L，生成态 observed=false、generated=true、coverage=0；历史实体编码与未来路径各算一次，之后每步重算时间窗口。
- **判断**：不是把未来真实实体输入空间模块，也不是逐步重跑 GRU；预测状态直接进入系统时间链。当前没有 KV cache；完整递推梯度保留。
- **证据**：[models/transformer/GSLAformer.py:340-386](../../../../models/transformer/GSLAformer.py#L340)；[docs/gslaformer/v1/implementation.md:97-111](../../v1/implementation.md#L97)；[tests/test_gslaformer_model.py:410-457](../../../../tests/test_gslaformer_model.py#L410)

### C12 独立静态条件｜新增

- **原图**：静态属性隐含在输入表，未画一条独立静态条件到预测头的通路。
- **实现**：起点有效实体的静态数值统计、类别均值、关系入出度及规模统计，先逐组 Linear→GELU，再拼接经单层 Linear 得到 c[B,d_t]。
- **判断**：c 不读取动态观测，也不是从最后一个动态 hidden 冒充静态表示。该独立条件在当前主图中必须出现。
- **证据**：[layers/gslaformer/spatial.py:71-157](../../../../layers/gslaformer/spatial.py#L71)；[models/transformer/GSLAformer.py:262-266](../../../../models/transformer/GSLAformer.py#L262)

### C13 已知未来路径与 lead｜新增

- **原图**：没有已知未来计划／天气路径或相对预测步长条件。
- **实现**：原起点已知的完整 H 步日历＋协变量数值＋mask 经过单层双向 Transformer；每步用 p_k，与时间 hidden、静态 c、16 维 lead(k) 拼接，经 MLP 融合。K=0 时仍有未来日历。
- **判断**：主时间链因果，未来已知路径链双向；二者不可混画为同一种 causal attention。只允许原起点可得信息，不是读未来实测。
- **证据**：[layers/gslaformer/temporal.py:96-131](../../../../layers/gslaformer/temporal.py#L96)；[models/transformer/GSLAformer.py:296-304](../../../../models/transformer/GSLAformer.py#L296)；[docs/gslaformer/v1/design.md:79-99](../../v1/design.md#L79)

### C14 A／B 训练与 teacher｜新增

- **原图**：原图只有结构，没有损失、冻结、teacher 或训练阶段。
- **实现**：A 用直接多步负荷监督学习表示；选择 best_A 冻结实体／空间／关系／读出／静态链。B 按 1→min(3,H)→H 自由递推，主负荷损失加质量／方差加权 latent 辅助项。teacher 仅在监督路径延续历史末端实体 memory。
- **判断**：这是实现闭环所需的后续训练设计，不应判为原图未满足。direct 是 A 的计算方式和单独对照，不是 full 模型已经取消回灌。
- **证据**：[models/transformer/GSLAformer.py:156-183](../../../../models/transformer/GSLAformer.py#L156)；[exp/exp_gslaformer_forecasting.py:237-288](../../../../exp/exp_gslaformer_forecasting.py#L237)；[exp/gslaformer_support.py:76-179](../../../../exp/gslaformer_support.py#L76)

### C15 总负荷输出语义｜保留

- **原图**：输出标为总负荷，没有设备级输出或功率守恒约束。
- **实现**：每站点单目标 y[B,H,1]；模型内 y 标准化，出口逆变换到 kW。latent 是辅助状态，不是各组功率；多层测点不相加成设施总负荷。
- **判断**：与原图总负荷目标一致。不能将 latent 六组误画为六条业务负荷输出，也不能由输入多层功率推断模型具备守恒约束。
- **证据**：[models/transformer/GSLAformer.py:45-52](../../../../models/transformer/GSLAformer.py#L45)；[docs/gslaformer/v1/input_output.md:9-33](../../v1/input_output.md#L9)；[docs/gslaformer/v1/implementation.md:109-111](../../v1/implementation.md#L109)

## 4. 现行规格与代码差异（待裁决，不擅改）

### D01 静态链最终投影层数

**实质结构不一致，待裁决**。

- 规格：实施规格 §1.3 写为：组输出拼接后 Linear→GELU→Linear 到 d_t。
- 代码：StaticConditionEncoder.project 实际只有 Linear(G*d_z,d_t)；非线性只存在于每组投影中。
- 判断：少了规格描述的跨组非线性融合层。不能认为两者一般等价，也不能仅为文档对齐擅自增加层。本次架构图按实际单层 Linear 绘制；后续需决定接受现实现并修规格，还是补 MLP 并升级计算版本、重训验证。
- 证据：[docs/gslaformer/v1/implementation.md:52-56](../../v1/implementation.md#L52)；[layers/gslaformer/spatial.py:86-93](../../../../layers/gslaformer/spatial.py#L86)；[layers/gslaformer/spatial.py:116-126](../../../../layers/gslaformer/spatial.py#L116)

### D02 历史日历特征注入方式

**结构表述不一致，通常可做等价表述对齐**。

- 规格：实施规格 §1.4 与 temporal.py 顶部说明写为：日历四维经独立 Linear 投影后与其他 token 投影相加。
- 代码：_window_tokens 把四维日历直接拼在末尾，TemporalEncoder.project 对所有 211 维联合做一次 Linear，再加正弦位置编码；没有单独日历投影模块。
- 判断：线性层按输入列拆块后，在合并 bias 的条件下可写成独立投影相加，表达能力并不因此必然降低。但当前计算图确实只有一次联合投影，图和模块清单不能虚构第二个 Linear；本次只登记，不改规范或模型。
- 证据：[docs/gslaformer/v1/implementation.md:60-70](../../v1/implementation.md#L60)；[layers/gslaformer/temporal.py:3-5](../../../../layers/gslaformer/temporal.py#L3)；[layers/gslaformer/temporal.py:74-86](../../../../layers/gslaformer/temporal.py#L74)；[models/transformer/GSLAformer.py:280-294](../../../../models/transformer/GSLAformer.py#L280)

## 5. 真实运行证据

使用既有 AIDC synthetic 包，正式 day profile 的 L=168、H=24、六组与模型宽度不变；取一个实际训练样本，临时统计只拟合前两个训练窗口，模型随机初始化。只验证结构，不产出训练模型或业务预测结果。

- 历史状态：`[1, 168, 6, 32]`；静态条件：`[1, 64]`。
- 时间输入宽度：`211`；输出 y：`[1, 24, 1]`；输出 z：`[1, 24, 6, 32]`。
- 实际捕获第二步窗口末 token，逐值验证它等于第一步生成的 z 与 y，并核对 observed/generated/coverage。
- 验证历史编码与未来路径各调用一次，时间链调用 H 次，direct 与 rollout 首步相等。
- 定向模型／契约／训练支持测试真实执行，命令与计数唯一记账于 [实施规格 §12](../../v1/implementation.md#work-packages)；机器结果与日志见 [runtime.json](evidence/runtime.json)、[targeted-tests.log](evidence/targeted-tests.log)。
- 图形浏览器渲染与文本边界检查见 [visual-check.json](evidence/visual-check.json)。

## 6. 后续判断

- 若以原图主旨为标准：主干与双头回灌已经成立，新增部分是后续 R4 设计扩展。
- 若要求原图逐字一致：需重新裁定 Space PE、空间重复层数、任意时间后端，以及保留哪些新增路径，不能默认为删减授权。
- 优先裁定 D01：接受当前单 Linear 并改文字规格，或补跨组 MLP 并按计算版本升级与重训处理。D02 可优先按等价线性分解澄清文档。
- 本次不修改模型、不改配置／依赖、不启动长训练、不恢复旧 architecture、不提交或推送。
