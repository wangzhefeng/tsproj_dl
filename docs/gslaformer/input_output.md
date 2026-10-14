# GSLAformer R4：业务对象与输入输出契约

> 本文唯一维护业务对象、数据包、外部张量、时间／单位／可用性和输出字段。网络内部计算见[实施规格](implementation.md)，模拟场景与验收见[验收协议](acceptance.md)。
>
> 状态：待实现。设备目录是候选标准，不是已经核实的生产点表；schema_version=3，无旧模型产物迁移。真实来源适配仍需后续授权。

<a id="input-output-1"></a>

## 1. 要预测什么

<a id="input-output-1-1"></a>

### 1.1 一个样本对应一个 AIDC、一个起点、一个总负荷口径

目标不是“预测所有设备”，而是用不同数量的内部对象及其历史，预测一个 AIDC 在未来各时间区间的总有功功率。一个 batch 可包含多个 AIDC 的独立请求；**B 是样本轴，不是跨站点通信轴**。站点之间共享模型参数，不建立跨 AIDC attention 边。

主输出为 `y_pred[B,H,1]`。设备／测点数量变化的是输入的 `N_g`，不是输出的最后一维。内部 `z_pred[B,H,Gd,d_z]` 是辅助隐状态，不是服务器或机房功率，不能逆变换成设备功率。

<a id="input-output-1-2"></a>

### 1.2 总负荷与市电净购电量不能混称

| 口径 | 为什么需要明确 | 简单示例 |
| --- | --- | --- |
| `facility_demand` | IT、暖通、辅助用电及边界内配电损耗的用电需求 | 没有自发电／储能时，完整进线表可能近似代表它 |
| `grid_net_import` | 含自发电、储能、外送影响的电网交换功率 | 同样 IT 负荷，储能放电时市电功率会降低 |
| IT 电力 | 只是总需求的一部分 | UPS 输出可能主要覆盖 IT，未覆盖独立供电冷机 |

本轮 AIDC 模拟主任务选 **`facility_demand`、kW、区间平均有功功率**，无自发电和储能；因此模拟进线与设施总需求一致。真实站点不能据此默认二者相等。若以后选择 `grid_net_import`，必须提供相容的标签与储能／发电观测及起点已知计划，形成独立 `target_contract` 和 checkpoint，不静默换目标。

`target_contract` 必填：`target_id`、`quantity`、`boundary_id`、`unit`、`sign_convention`、`interval_label`、`aggregation`、`meter_ids`、`coverage_description`。多个进线表相加前先证明互不重叠、方向相容、时刻一致；同一表的冗余采集通道不可重复计入。

> **核心认识：多层功率可以共同作为解释信息，但不能把服务器、机柜、UPS、变压器、楼宇和市电功率全部相加得到总负荷。**

<a id="input-output-2"></a>

## 2. 从原图的三组扩展为业务对象目录

<a id="input-output-2-1"></a>

### 2.1 电力层级是观察线索，不是已核实的现场接线图

用户给出的观察层级为：服务器功耗 → 机柜／其他测点 → 列头柜／其他测点 → 配电室 USP 输出 → 配电室变压器 → 机房 → 楼宇 → 市电。

保留原始术语“USP”。本文将“不间断电源”候选标准类型记为 `ups`，但**不自动把现场名 USP 当成已确认的 UPS**；实际映射须由点表、设备资料确认。类似地，用户原文“冷冻会二次泵”暂按“冷冻水二次泵”候选类型说明，源名称原样保存在注册表中，不能靠拼写猜测完成正式映射。

上述链路混合了电气设备、测量位置和空间区域：

- 机房／楼宇是空间对象，不一定串联在 UPS 与变压器之间。
- 电气供电方向通常从上游供电设备到下游负载；原描述是由细到粗的观测顺序，二者不能混用。
- 双路供电、跨房间支路、冗余 UPS、母联切换、多变压器服务同一楼宇，都需要现场关系表。
- 缺少某级测点时允许跳级关系，但关系必须来自资料确认，不能自动补造中间设备或观测。

<a id="input-output-2-2"></a>

### 2.2 固定组类型、可变实体数

R4 的 AIDC 标准 profile 固定六组：`compute`、`it_power`、`distribution`、`hvac`、`space`、`environment`。类型由类别属性细分，而不是每新增一台泵就增加一个组。与 R3 `assets/zones/sites` 机制夹具是不同 schema，不共用 checkpoint。

| 分组 | 实体粒度与候选类型 | 动态特征示例 | 静态属性示例 |
| --- | --- | --- | --- |
| `compute` | 稳定任务池／集群／租户业务池；不是每条瞬时作业都建节点 | 活跃作业数、排队量、已申请 GPU 数、吞吐 | 任务类型、资源容量、汇总粒度 |
| `it_power` | 服务器、机柜、列头柜或已确认的等价计量对象 | 有功功率、CPU/GPU 利用率、开机比例 | 对象类型、额定功率、硬件类别、所代表服务器数量 |
| `distribution` | UPS 输入／输出口、变压器低压侧、配电支路等计量边界 | 有功功率、负载率、效率观测（若有） | 电力设备类型、端口、额定容量、计量覆盖类型 |
| `hvac` | 冷水机组、冷却塔、冷却泵、冷冻水一次泵／二次泵；有资料时纳入末端／液冷设备 | 功率、启停、频率、流量、供回水温度 | 设备类型、额定功率／流量、设计制冷量、冷却方式 |
| `space` | 机房、楼宇、园区边界等空间／聚合对象 | 对应总表功率、室温／湿度 | 空间层级、面积、IT 设计容量 |
| `environment` | 室外环境观测站或代表性环境区域 | 干球温度、湿度等实测值 | 观测站类型、区域属性 |

这些是目录，不要求每个站点每项都有值。机柜不具备 GPU 利用率时 `applicable=false`；服务器有 GPU 但利用率采集离线时 `applicable=true, observed=false`。不创建成千上万个“根本不存在”的虚拟服务器来补到共同数量。

配电室 UPS 输入／输出属于同一资产的不同计量边界，不能把两路数值互相覆盖。允许表示成两个有 `asset_id` 关联的计量实体，或在明确的固定 schema 中建不同端口特征；AIDC profile 采用前者，以统一 `active_power_kw` 语义。

<a id="input-output-2-3"></a>

### 2.3 资产、点位、特征不是同一个概念

| 对象 | 为什么需要 | 简单示例（示意） |
| --- | --- | --- |
| asset | 记录物理设备身份 | 一台服务器、一台 UPS |
| entity | 定义模型的对象轴及计量边界 | UPS-A 的输出侧计量实体 |
| measurement | 绑定来源点位、单位和特征 | `P_103` → 输出侧 `active_power_kw` |
| feature | 跨站点保持可比较含义 | kW 功率与百分比利用率分别为不同特征 |

同一设备的功率和温度一般是同一实体的不同特征；设备及其独立电表是否合并由 mapping 声明。冗余传感器按主备规则选择或明确融合，不能因两个点名不同就当成两份独立负荷。对混合粒度的服务器／机柜观测保留 `entity_type` 与 `represented_count`，不把机柜伪装成一台大服务器。

<a id="relation-contract"></a>

## 3. 类型关系的业务合同

R4 将 `relations.json` 的通用端点改为 `src_group/src_id/dst_group/dst_id/type`，另带 `valid_from/valid_to/available_at/revision`。`src/dst` 的业务方向由下表固定，反向信息边由模型显式构造，不反写源表。

| 类型 | 源 → 目标 | 合法性与用途 |
| --- | --- | --- |
| `contains` | 较大空间 → 子空间／设备 | 子图无环；在一个空间层级按声明限制父级数量；提供位置归属 |
| `supplies` | 上游电力计量实体 → 下游计量实体／设备 | 首版采用已确认的有向无环供电视图，允许多路父级；带现场端口说明 |
| `meter_covers` | 汇总计量实体 → 被覆盖实体 | 子图无环，可多父级；表示覆盖，不保证覆盖完整、也不保证可加 |
| `cools` | 暖通设备／冷却系统代表实体 → 服务设备／空间 | 多对多；不等同于空间包含或电气供电 |
| `runs_on` | 稳定任务池 → 已确认部署的服务器／机柜／空间 | 多对多；只按实际可得粒度建立，未知部署不猜测 |
| `associated_env` | 环境观测实体 → 代表空间／暖通实体 | 显式关联，不能用近邻点名自动认定 |

这些关系使图示中的关系分支真实可计算；管网循环与有环电气网络求解不入当前 attention 契约（非承诺的统一清单见[设计 §8](design.md#capability-boundary)）。若需要物理有环电气网络，先作显式适配／扩展。

每种关系的允许端点类型、是否无环与 `reverse_message` 写入 schema。首版允许多来源／多目标，contains 仍按空间层级检查同层归属唯一性；不把所有关系都命名成父子。拒绝跨站点边、自环源记录、同类型重复边、悬空端点和违反该类型约束的循环；**不要求所有类型合并后仍是 DAG**。供电与冷却等不同类型叠加形成环可以合法。


关系在实体读出前参与计算；query/key 方向、偏置叠加、两层消息与质量旁路的精确定义见[关系计算](implementation.md#implementation-3-2)。它不是电气／热力求解器，也不保证全网络仅沿关系边通信。

<a id="input-output-4-1"></a>

## 4. 来源表与映射

生产来源暂未整理，以下是接入合同，不是现有文件。

| 表／文件职责 | 最少字段 | 为什么需要 |
| --- | --- | --- |
| 站点／目标登记 | site_id、target_contract、timezone、freq_seconds | 同一单位下仍可能存在不同计量边界 |
| 实体／资产注册 | site_id、entity_id、asset_id、group、type、属性及有效期 | 同名设备在不同 AIDC 不应撞 ID |
| 测点映射 | measurement_id、source_name、entity_id、feature、unit、转换、主备策略 | 不从列名猜设备和物理量 |
| 历史观测 | measurement_id、event_time、available_at、value、quality | 测量时刻不等于预测时已收到 |
| 类型关系 | 两端点、type、有效期、available_at、revision | 支持变更且不覆盖旧历史 |
| 任务历史 | 稳定任务池、事件、汇总窗口、available_at | 用因果汇总避免最终任务信息泄漏 |
| 未来条件版本 | origin、target_time、feature、value、published_at、revision | 回测使用起点可见的计划／预报 |

保留原始来源，不用清洗结果覆盖原始文件。标准化值只是模型输入表示，正式结果必须保留物理单位和输入版本。

<a id="input-output-4-3"></a>

## 5. 时间对齐与单位

统一时间索引为 UTC 秒，本地时区只用于日历。AIDC 示例采用区间右端标记：时刻 `t` 的功率代表 `(t-Δ,t]` 的平均 kW；预测 `o+kΔ` 对应 `(o+(k-1)Δ,o+kΔ]`。origin 是最后一个历史区间末端，实际读取还要求 available_at 不晚于 origin。

- 功率：规则等间隔采样可取均值；不规则采样用明确覆盖时长的时间加权平均。不能把 kW 简单求和当成 kWh。
- 累计电量：差分为区间电量后除以区间小时数；计量复位、负跳变及间隙须标记，不跨缺口强行平均。
- 状态／启停：按 schema 声明区间末值或运行时长比例，不能混用。
- 任务数：声明是存量（并发量）、新增计数还是完成吞吐；三者聚合方法不同。
- 室温／流量：按各特征的时序语义对齐；温度缺失不能由未来插值填上。
- 标签缺失：mask 屏蔽，不为训练方便生成假总表真值。

<a id="standard-package"></a>

## 6. 标准包、请求快照与 batch

<a id="package-format"></a>

### 6.1 首轮持久化格式

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

manifest 的 `layout` 必须是 `single_episode` 或 `episode_collection`。前者包含上表文件；后者包含 `episodes:[{id,family_id,site_id,split,path}]`，path 指向完整 single_episode 子包。episode_family 模式的 split 限 train/valid/test，同一 family_id 不得跨 split。site_time 模式的 split 为 null，另附 topology_family_id、time_split 与 site_split，按[验收协议的 AIDC 双轴切分](acceptance.md#synthetic-aidc-4-3)生成窗口，不能给整个站点套一个时间分区。collection 根不伪造合并 observations，不把不同实体轴强拼成一个时间序列。

`timestamps` 为严格递增、等间隔 UTC int64 秒；本地日历使用 manifest 的时区生成。首轮 synthetic 使用 UTC，避免把时间转换问题与模型测试混在一起。

历史动态观测首版一个单元格只存一个值及其到达时间，不自动选择历史修订；真实数据若有版本修订，必须先做 as-of 来源适配，不直接导入最终修订宽表。未来协变量支持同一起点可见的多版本记录，按 `published_at <= origin` 选最近版本，时间相同再按显式 revision 排序；完全同键的冲突记录报错。

关系 schema 声明 `name/src_groups/dst_groups/acyclic/reverse_message`，允许同组边、多来源和多目标，不把任意类型都解释成父子。业务方向与逐类型合法性见 [类型关系的业务合同](input_output.md#relation-contract)；acyclic=true 的类型单独拒绝循环，不要求所有类型合并后仍为 DAG。拒绝源记录自环、同类型重复边、跨站边和悬空端点；模型再生成 self 与显式反向通信，不把双向消息当物理反向供电。full 每个请求至少一条有效边；孤立实体允许 self，但记录 orphan 数量。AIDC 的 contains 还需按实体 space_level 验证方向；supplies 按端口／设备类型验证，组级白名单不是全部约束。

<a id="request-snapshot"></a>

### 6.2 有效期、历史版本与请求快照

- 每个请求独立解析起点可见的实体／静态属性／关系，实体数量可以不同。
- 历史每个时刻 s 使用 `valid_from <= s < valid_to` 且 `available_at <= origin` 的版本，重叠有效期按可见 revision 选择，冲突报错。属性和父级可在历史内切换；绝不把 origin 的新父级覆盖到全部历史。
- 允许实体在历史中途出现：其生效前 `present=false`，没有观测和消息；生效后才应用该实体的边。新实体没有历史时，其历史观测 mask 全 false，不能回填虚构观测。
- 未来的属性、成员和关系固定为 origin 上下文；若起点已知 horizon 内有变化，拒绝该请求。历史内变化不触发此限制，新起点立即可用最新已知版本，无需重训。
- latent 标签使用起点快照和同一实体集合，仅未来动态值用作监督；不能读取后来才公布的父级、新节点或静态修订。

<a id="batch-contract"></a>

### 6.3 标准 batch

记 `G` 为固定分组类型数，`Gd` 为动态组数，`N_g` 为当前 batch 补齐后的组内节点数，`L` 为历史，`H` 为预测长度。组顺序由 schema 固定；节点顺序每次由元数据映射处理，不进入模型位置编码。

\[
X^{(g)}\in\mathbb R^{B\times L\times N_g\times F_g}
\]

B 是独立请求的 batch，F_g 是固定 schema 的组内动态特征数；该张量对应下表 groups[g].values。

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

**batch 组装（collate）规则**：不同样本的实体集合与数量不同，collate 将一个 batch 内每组 `N_g` 补齐到该组在本 batch 的最大实体数，不产生跨样本共享的实体轴：

- 补齐行 `padding=true`，`values/static_numeric_history` 填 0，`observed/present/applicable/static_numeric_mask` 全 false，`static_category_history` 填 PAD(0)；补齐行不向其他实体贡献信息（屏蔽规则见[关系计算](implementation.md#implementation-3-2)）。
- `relations_history` 与 `relations_origin` 保持每样本独立的边列表，端点为**样本内**实体索引；不跨样本拼接成统一图，不引入全局节点编号。
- `entity_ids/site_ids/target_contract` 逐样本作为元数据列表传递，不进入张量。
- 同组内实体顺序在单请求内固定（与关系端点索引一致），batch 维不要求不同样本顺序对齐。

<a id="scaling-missing"></a>

### 6.4 scaler、缺失与新实体

- 连续观测和静态数值按 `(group, feature)` 在训练分区真实适用观测上统计，不按实体拟合；target 使用独立统计；协变量只使用训练 origin 可见的记录。AIDC 还必须限于训练站点训练时段，留出验证／测试站点的全部数据不能参与拟合。
- 标准化均值与总体标准差存为 float64 元数据，模型输入 float32。标准差小于 `1e-6` 用 1；完全没有训练观测的已声明数值特征报错，不默默创建假统计。
- 标准化后未观测或不适用位置填 0，同时传 observed／applicable。首版不前向填充，不用未来插值。
- 新实体复用组级 scaler。未知属性类别映射 UNK，不添加新参数；保留数据质量计数，不声称已验证未知类别精度。
- 属性变更记录去重后按训练可见版本统计，不按广播到每个时间格的次数重复拟合；episode 不同源数据分别记录键，归一化总体统计只使用训练 split。
- 未声明列、单位变化、新分组、新关系类型报错；同 schema 新 ID 和新合法边必须接受。
- 缺测但存在的实体仍以属性和 mask 参与 attention；padding 和未生效实体不向其他实体贡献信息。

<a id="window-split"></a>

### 6.5 窗口与 split

训练／验证／测试按 target-anchored 协议：`history=[t-L+1,t]`、`labels=[t+1,t+H]`，所有标签落在同一 split；历史可来自更早 split。周期 base 各 split stride=1，AIDC 的 stride 由验收 profile 明确，不能套用 base 值。训练形成窗口后可 shuffle；未来未知观测绝不参与 origin 可用性选择。

该规则对应配置 `split_mode=target_index`，需要严格递增 split_index。机制 collection 用 `split_mode=episode_family`、`split_index=null`，从 manifest 取 family split，并只选各长度 L+H episode 的 origin=L-1；拒绝同时提供 target_index 边界。不要把 base 的 320 行边界复制到 30 行的机制 episode。

AIDC collection 使用第三种 `split_mode=site_time`，有 split_index、site_split 和每站独立时间轴。00–07 的训练／未来验证／未来测试与 08–09 未见验证、10–11 未见测试按 [时间轴与合法窗口](acceptance.md#synthetic-aidc-4-3) 分配；样本不跨站点。站点平衡采样算法见[训练状态机](implementation.md#implementation-6)；valid/test 保序且按 profile 的 stride 取窗，各 profile 的完整配置与样本数随产物保存。

推理至少提供规则时间轴上的 L 个位置；实体级值可以缺失，不允许少行后偷偷改变 L。请求最后一时刻必须等于 origin。有效目标为零的监督 batch 不优化；整轮无有效主目标报错。

AIDC profile 固定 G=6，六组都声明动态列，因此 Gd=6；合法空组仍占固定 latent 槽位且质量为零。`(site_id, group, entity_id)` 为唯一身份，site_id 与 entity_id 都不进入可学习查表。

站点级属性通过 space 组中唯一 aidc_boundary 实体进入独立静态链。该实体可承载总表特征，但 target_history 仍是单列目标历史；重复提供解释信息不等于重复计量。类型变化若改变 applicable，须分新实体或拒绝跨变更窗口，不在请求中改变特征含义。

生成器热状态、真实未来天气、最终任务执行量和实际完成时间不能混进 ForecastInput。无标签请求不含未来目标、目标 mask 或内部未来观测；监督包与请求包隔离。

### 6.6 落盘编码与版本指纹（实施补充）

以下固定首版 JSON/NPZ 的精确编码，不依赖 Python 对象序列化。文件统一 UTF-8；JSON 拒绝重复键、NaN 和 Infinity；所有时间为 UTC int64 秒。

- `entities.json` 为 `{"groups": {group: [record, ...]}}`。record 必含 `id/valid_from/valid_to/available_at/revision/numeric/category`；numeric/category 为按 schema 特征名索引的对象。缺失数值为 null，缺失类别为 null（映射 UNK）；valid_to=null 表示无上界。同 ID 可以有多个版本，但 `(id,valid_from,revision)` 不得重复。数组轴与记录顺序无关。
- `relations.json` 为 `{"edges": [record, ...]}`。record 必含原类型关系合同的端点、type、有效期、available_at、revision；同逻辑边的历史版本允许，起点选择后不得出现同类型重复边。
- `site.json` 为 `{"site_id": string, "target_contract": object}`，目标合同遵守 §1.2。站点角色来自 collection 切分表，不进入学习输入。
- `measurements.json` 为 `{"mappings": [record, ...]}`。record 含 `measurement_id/source_name/group/entity_id/feature/source_unit/scale/offset/role`；转换为 `value*scale+offset`，role 为 primary 或 backup，同实体特征至多一个 primary。标准 observations 已完成转换，不重复乘系数；冗余源只用于来源审计。
- `manifest.json` 的单 episode 必含 `schema_version/data_kind/layout/schema/entity_order/files`。schema 包含 `timezone/freq_seconds/groups/relation_types/known_future/target_name/target_unit`，AIDC 另包含 feature_units、target_quantity、interval_label、target_aggregation。`entity_order[group]` 是唯一 ID 列表，精确绑定所有组数组的 N 轴；files 将固定文件名映射到该文件原始字节 SHA-256。
- collection manifest 使用相同 schema，episodes 每项含 `id/family_id/site_id/split/path`；site_time 另有 topology_family_id，根保存 `split_mode/split_index/site_split`。子包路径必须留在 collection 根内；不允许重复 episode ID。每个子包仍独立验证文件 hash。
- `observations.npz` 固定公共键为 `timestamps/target/target_observed/target_available_at`，shape 分别为 `[T]`、`[T,1]`、`[T,1]`、`[T,1]`。每组以 `group + "__" + field` 命名：values/observed/available_at 为 `[T,N,F]`，applicable 为 `[N,F]`，present 为 `[T,N]`。values/target 用 float32，mask 用 bool，到达时间用 int64。组名只允许字母、数字、下划线且禁止双下划线，避免键歧义。零特征／零实体轴保留形状，不存对象 dtype。
- `known_future.npz` 是列式记录，固定键 `origin/target_time/feature_index/value/observed/published_at/revision`，均为 `[R]`；时间、index、revision 为 int64，value 为 float32，observed 为 bool。记录 origin 是请求的原点；先筛 origin 相等，再按 `(target_time,feature_index)` 取 published_at 不晚于原点的最新版本；同发布时间以 revision 最大者优先。K=0 时仍保存这些零长度数组。
- 无标签请求使用与单 episode 相同的历史文件结构，timestamps 最后值为请求原点且至少 L 行，manifest 另记 `purpose="forecast"`。target 只包含历史总表，不是未来标签；future_target/future_groups 等字段拒绝。未来条件由 known_future 提供，不在 observations 中附加未来行。

NPZ 数值在 observed=true 处必须有限；未观测位置允许 NaN，但进入模型前按 mask 置零。available_at 仍逐单元筛选，不能只验证最后一个时间戳。manifest 声明的实体顺序与注册 ID 集必须一致；历史结束后失效实体可以在轴上，但 present 和边必须按有效时刻屏蔽。

指纹使用 SHA-256：JSON 先 `sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False` 转为 UTF-8。schema 指纹只包含上述 schema 对象；不包含实体 ID、数量、关系快照、站点切分或绝对路径。数据指纹包含 schema 指纹、entity_order、所有文件字节 hash 及 collection 切分／子包指纹。模型兼容性另验证 L/H、model 配置、variant、schema 和冻结 scaler／词表；不同节点快照改变数据指纹，但不据此拒绝同 schema 新实体预测。指纹算法版本固定为 `sha256-canonical-json-v1`。

<a id="scaling-artifacts"></a>

### 6.7 scaler 与词表落盘格式

统计语义由 [scaler、缺失与新实体](input_output.md#scaling-missing) 定义；本节唯一固定训练产物的字段级格式，checkpoint 校验与 test/forecast 重载只认本格式，不现场重拟合。

- `scalers.json`：`{"version": 1, "schema_fingerprint": hex, "groups": {group: {"dynamic": {feature: {"mean": number, "std": number}}, "static_numeric": {feature: {"mean", "std"}}}}, "target": {"mean", "std"}, "known_future": {feature: {"mean", "std"}}}`。均值／总体标准差以 JSON number 保存 float64 值；标准差小于 `1e-6` 存 1。完全没有训练观测的已声明数值特征**不写入**该文件；加载时发现声明特征缺统计即报错，不默默补假统计。`known_future` 只含 K>0 时已声明协变量。
- `category_vocab.json`：`{"version": 1, "groups": {group: {feature: [token, ...]}}}`。索引 0 固定为 PAD、1 固定为 UNK；其余 token 取自训练分区可见版本，按字典序排列以保证可复现。推理期未见类别映射 UNK，不追加词表、不新增参数。
- 两文件均按本节规范 JSON 规则编码（拒绝重复键、NaN、Infinity），其规范指纹随 checkpoint 保存并在 test/forecast 校验一致。

<a id="missing-semantics"></a>

## 7. 缺失与设备存续语义

| 情况 | 表达 | 为什么不能都填零 |
| --- | --- | --- |
| 设备不存在 | 该实体不在轴中，组可以 N=0 | 零功率设备仍有数量和属性影响 |
| 设备已存在，尚未投运 | present=false | 不应参与此时消息和数量统计 |
| 运行设备数据离线 | present=true，observed=false | 保留属性／身份／因果历史记忆 |
| 某特征不适用 | applicable=false | 变压器没有 GPU 利用率不是传感器故障 |
| 值确实为零 | observed=true，value=0 | 停机是真实业务状态 |
| batch 补齐 | padding=true | 不应冒充未知实体 |

全缺失组使用空组／质量机制得到有限输出，但若站点没有任何历史总表且没有可用内部动态观测，AIDC full 请求直接拒绝；不以纯静态猜测伪装成正常预测。有部分数据的低质量预测须标注 `degraded_input`，不保证精度。

<a id="future-plan-availability"></a>

## 8. 任务与未来计划的可用边界

历史任务指标优先按稳定资源池汇总，避免大量短生命作业导致未来成员变化与 R3 固定未来拓扑冲突。任务开始／结束是计数和利用率变化，不是每个 horizon 更换实体轴。

未来条件首版仍是固定宽度的**站点级**路径，例如已申请 GPU 数、计划训练任务数、站点天气预报。存在设备级计划时，可按固定类型汇总成有明确定义的列，保留映射与统计口径；它丢失具体部署信息，不能宣称已完整消费逐设备未来调度。未来可变任务—服务器部署图不属于首版承诺。

已发布计划是条件，不保证必定执行；模拟器和评估需覆盖计划偏离。不存在可靠计划时 mask=false，不能使用未来实际任务量填补。可新增的协变量必须在训练 schema 中预声明并有训练证据，forecast 不临时加列。

<a id="site-data-rules"></a>

## 9. 跨站点数据规则与 schema 兼容性

能力分层（数据通用 → 结构通用 → 参数共享 → 未见站点 → 真实迁移）及“不应推出的结论”统一见[设计 §8](design.md#capability-boundary)，本节只维护数据侧规则。

训练统计量按组／特征跨训练站点真实观测拟合，目标 scaler 也只用训练站点训练时间；不用每个新站测试数据重新估均值。原始 kW 目标的 MSE 仍可能偏向大负荷站点，这是已知边界；同时报告各站点误差和归一化误差，不用 pooled 数字掩盖小站失败。

站点规模通过额定容量、represented_count、有效实体数量及静态统计被模型消费。跨层额定容量的 sum 只是分组描述量，不解释成设施总额定功率。首版不引入额外按站 scaler，不靠归一化偷偷抹去规模差异。

联合训练、同站新实体、留出新站测试分别记账。新类型／新列／不同目标口径／不同采样间隔意味着新 schema 或独立 profile，需要重新训练／验证。改变设备数量和合法关系快照不改变 schema fingerprint。

<a id="output-contract"></a>

## 10. 业务输出、诊断与审计字段

`predictions.csv` 每行保留：

- `run_id/model_version/data_kind/site_id/target_id/boundary_id`；
- `origin/target_time/horizon/freq_seconds/unit`；
- `y_pred`、test 时的 `y_true/target_valid`（forecast 真值为空）；
- `new_site/new_entity_count/no_history_entity_count/unknown_category_count`；
- `input_coverage/degraded_input/negative_prediction/plan_available`。

`latents.npz` 保存隐状态与组顺序，只用于研究诊断。`metadata.json` 保存 schema／数据／关系／checkpoint fingerprint、目标合同、输入截至时刻与版本、站点留出角色。

总负荷头沿用线性输出，可能产生负值；保留原始预测并记录负值计数，不在评分前偷偷裁成零。若业务要非负输出，需将约束作为训练／输出规则的显式版本变化；当前不宣称物理约束已实现。

`predictions.csv` 的 target_valid 对 test 表示真值有效性；forecast 的 y_true 为空且不将缺少真值当成预测失败。隐状态文件保留 `[B,H,Gd,d_z]` 对应的组顺序，不能伪装为设备物理量。

质量字段按请求计算，在各 horizon 行重复保存，不当作概率置信区间：

- `input_coverage`：历史 L 步中，所有动态组 present、非 padding、applicable 单元的起点可见 observed 数／这些适用单元总数；分母零取 0。采用单元加权，不按组宏平均，不包含历史总表。
- `target_history_coverage`：L 个历史目标位置中起点可见有效值的比例，单独写 metadata，不与内部覆盖率混算。
- `plan_available`：K>0 且至少一个 `[H,K]` 条件单元在原点可用。K=0 或全部缺失均 false；日历不算外部计划。
- `plan_coverage`：K>0 时可用条件单元数／(H*K)，K=0 时为 null；写 metadata。部分计划可用不会被伪装成完整路径。
- `degraded_input`：内部适用单元不完整，或 target_history_coverage<1，或存在无历史实体，或 unknown_category_count>0，或 orphan_count>0，或 K>0 且 plan_coverage<1，则为 true。无内部适用单元时也为 true；若同时无总表历史则按 §7 拒绝。该标记仅表示输入合同不完整，不代表低于某个经验精度阈值。
- `new_site`：site_id 不在 checkpoint 的训练站点溯源集合中；ID 重命名只改变审计标记，不改变数值推理。`negative_prediction` 按各 horizon 逆变换后的 y_pred<0 独立判断。

输出 y 逆 target scaler 回到标准包的单位；AIDC mapping 已将源端 W/MW 转为 kW，因此 AIDC 的原始量纲是标准 kW，不恢复成混杂源单位。是否要求该模式生成 metrics、checkpoint 或 latents，以及文件名、原子写入和冲突规则，由[产物生命周期](implementation.md#implementation-11)维护。

H 必须等于 checkpoint 配置的训练 H；若只展示前几步，应完整推理后截取，不能裁剪 FuturePathEncoder 的条件改变信息集。各模拟 profile 的 L/H 与容量范围由[验收协议](acceptance.md#aidc-profiles)维护。

<a id="request-example"></a>

## 11. 请求示例

站点 A 有服务器、机柜、UPS、两类泵及机房，站点 B 只有机柜级功率、另一套暖通设备和楼宇总表。二者按同一标准 schema 映射：服务器缺席不是伪造零服务器，缺失特征带 mask，实际关系各自提供。以同一 `aidc_hourly_day` 模型分别读取各站点历史与起点已知条件，输出各自 `[1,24,1]` 的 kW 曲线。

组内设备顺序可以一致重排；整个历史的 ID、属性、mask、关系必须同步重排。站点 A 与 B 的数据可以组成 B=2 的 batch，但不能把它们的设备连成一个“超大 AIDC”。


<a id="error-contract"></a>

## 12. 错误输入与拒绝行为

> 本节是外部可见错误行为的唯一权威表；前文各节只定义正常语义，遇到冲突以本节为准并显式修正。

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
