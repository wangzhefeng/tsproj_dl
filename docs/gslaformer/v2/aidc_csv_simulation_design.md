# GSLAformer：三楼 AIDC 全 CSV 模拟数据设计

> 文档状态：Draft／待评审；创建日期：2026-10-16（Asia/Shanghai）。用户决定暂不运行测试，下一步重新构造接近生产的模拟数据；本方案是候选数据设计，不是已完成的数据实现或验收。
> 本轮交付：仅设计方案；不生成 CSV、不修改模型、不运行训练。模拟数据的后续保存位置固定为 `dataset/aidc_gslaformer/your_version/`。
> 任务：DOC-012，已登记到根 AGENTS.md，状态 Done 仅表示设计文档交付完成；方案仍待评审，不表示候选假设已批准，也不表示模拟器或 CSV 加载器已经实现。

## 1. 定位与已确认决策

本方案把用户描述的真实设施结构转为可核算、可复现的模拟实验，不复制旧模拟站点，不把未知接线猜成现场事实。

| 项目 | 已确认结论 |
| --- | --- |
| 本次范围 | 先完成目录、表头、生成逻辑和验收设计；评审后再生成数据 |
| 预测目标 | A1+A2+A3 合计设施负荷，包含 IT、暖通、辅助用电及边界内配电损耗，排除储能 |
| 楼栋用途 | 三栋楼，每栋五层；一层暖通，二至五层为 IT 机房 |
| 供电结构 | 模组有 A/B 两路供电；部分负载经 UPS、部分不经 UPS；设备基本双路接入 |
| 算力观测 | 训练／推理两类；有 CPU、内存、GPU 使用率、功耗、温度、使用量等指标；可提供服务器至楼栋多层统计 |
| 模拟时间 | 15 分钟间隔，连续 90 天；预测未来 6／24／72 小时 |
| 尚未确认 | 独立变压器／UPS 数量、双路负载分担方式、暖通 UPS 覆盖、塔泵共用、冷却服务关系、下级设备数量及任务混部 |

**未确认项只形成候选模拟假设，不能解释为用户已同意，更不能作为现场资产清单。** 三栋楼属于同一站点，空间信息在站点内交互；本轮不宣称跨 AIDC 泛化。

与既有文档的关系：

- [v1 输入输出契约](../v1/input_output.md)继续定义现有模型合同；本方案提出新的 CSV 来源合同，不静默改写 v1。
- [v1 模拟验收协议](../v1/acceptance.md)中的旧站点、小时级 profile 和旧实验结论，不自动适用于本数据。
- [v2 评审与优化计划](review_and_optimization_plan.md)是历史评审依据。本方案吸收其中重复总表代理、指标同值和机制不可识别的风险，不把该评审中的旧问题状态当作当前实现状态。
- 文档目录 v2、模型计算版本、旧 schema_version=3、新 CSV 数据合同版本是不同概念。建议新来源合同标记 `aidc_csv_v1`，不伪装为已兼容旧包。

## 2. 对 my_version 的核对与补充方向

已只读核对 `dataset/aidc_gslaformer/my_version/`：68 个 CSV 全部为空，无表头；`your_version/` 为空。

| 现有组织 | 保留的业务含义 | 新方案补充 |
| --- | --- | --- |
| `computility/` 按楼分训练／推理 | 两类工作负载 | 统计对象、任务口径、容量分母、部署关系与聚合血缘 |
| `hvac/<楼>/module_*/` 五类设备 | 每模组三组冷机及辅助设备 | 模组身份、冷机组身份、设备 ID、共用关系、冷却服务范围 |
| `it_load/<楼>/module_*/` | 服务器、机柜、列头柜、UPS、变压器 | 增加楼层、机房、A/B 路及输入／输出端口，避免不同楼层模组重名 |
| `target.csv` | 一个总负荷目标 | 明确三楼边界、单位、储能排除和核算来源 |
| `weather/weather.csv` | 环境影响 | 区分历史实测与历史发布的未来天气预报 |

不修改、搬迁或填充 `my_version/`。新结构按“注册表 + 观测表 + 关系表”组织，楼层等身份进入显式字段和外键，而非只能靠目录名推断。用户可以按楼／日期筛选获得原来熟悉的视图。

## 3. 设施模型与候选假设

### 3.1 空间层级与身份

空间主链为：园区 → 楼栋 → 楼层 → 机房；电路模组是供电组织，不等于房间，也不等于一台变压器。

- 一层：每栋两个暖通模组，每模组三组冷机与辅助设备。
- 二至五层：候选固定每层四个机房、两个 IT 模组，每模组负责两个机房；“基本四个”在本次模拟中作规则化处理，不表示实际每层都已盘点。
- 暖通模组与 IT 模组编号即使相同，也不是同一对象。示意 ID：`A1_F01_HVAC_M01`、`A1_F02_IT_M01`。
- 设备 ID、测点 ID、模型实体 ID分别维护。ID只用于对齐，不作为可学习编号；容量、类型、空间层级等才是特征。
- 例如 `A1_F02_R01_RK01_S01` 是服务器；其 `port_A`、`port_B` 是该设备的供电端口，不是两台服务器。

### 3.2 候选小规模实例（全部待评审）

为保留所有楼层和模组而控制服务器数量，提出以下实例。规模参数可在生成前修改，任何改变均生成新的数据身份；不自动把该规模视为模型资源预算已通过。

| 对象 | 候选规则 | 推导数量 |
| --- | --- | ---: |
| 楼栋／楼层 | 三栋，每栋五层 | 3／15 |
| IT 机房 | 每栋四个 IT 楼层，每层四间 | 48 |
| 暖通／IT 模组 | 每层两模组 | 6／24 |
| 变压器 | 每模组每路一台 | 60 |
| IT UPS | 每 IT 模组每路一台 | 48 |
| 列头柜 | 每机房一对独立 A/B 列头柜 | 96 |
| 机柜 | 每机房两个物理机柜，双路接入 | 96 |
| 服务器 | 每机柜两台物理服务器，双路接入 | 192 |
| 制冷组 | 每暖通模组三组 | 18 |
| 暖通大设备 | 每组各一台冷机、塔、冷却水泵、冷冻水一次泵、冷冻水二次泵 | 90 |

以上是物理资产数，不是模型 token 数。UPS 输入／输出、双路机柜端口和任务池等会增加计量实体；生成器必须另报最终实体与边数，不能拿资产数代替注意力规模。

### 3.3 电路与冷却候选方案

| 决策 ID | 候选方案 | 真实边界与代价 |
| --- | --- | --- |
| D01 | 每模组 A/B 独立变压器，IT 每路独立 UPS；列头柜保留 A/B 两实体 | 未核实实际数量；批准后才展开清单 |
| D02 | 双路正常同时供电，按设备／时刻的份额分担，不固定所有设备 50:50 | 两路功率之和等于该设备输入功率，不能各复制一份总功率；模拟端口分担不等于模拟电网潮流 |
| D03 | 候选暖通大设备不经 UPS，IT 服务器经 UPS；照明等辅助负载走明确非 UPS 支路 | 只是模拟简化，现场可能不同；保留支路类型以便以后调整 |
| D04 | 每组三类泵及塔独立配置，不共享；冷却水／冷冻水循环由热力生成器表达 | 若采用共用泵，则改为独立共享资产，禁止给每台冷机复制同一泵功率 |
| D05 | 暖通 M01 服务本楼各 IT 层 R01/R02，M02 服务 R03/R04；三冷机共同服务模组覆盖区 | 不代表已确认管路；通过独立 `cools` 服务份额表表达，不从同名模组自动推断 |
| D06 | 服务器允许训练／推理混部，资源总占用受容量限制；初始任务占比不设现场结论 | 需记录分配和归因方法；聚合温度不能被当成任务可加量 |
| D07 | 使用候选小规模实例、固定拓扑、无切换故障的主时间序列 | 实体增减／供电切换另设契约测试，不偷偷加入主序列突破现有未来拓扑限制 |

D01–D07 当前均为**提案，未获批准**。用户回答“不确定”仅表示现场信息未知，不是同意候选值。

## 4. 总负荷边界与避免重复计量

### 4.1 唯一目标

目标 `facility_demand_kw` 为区间平均有功功率，消费方向为正，覆盖三栋楼已列出的用电与边界内损耗，不包含边界外园区公共用电。无资料的公共负荷不默认为零并混入现场结论，而是明确不在本模拟范围。

每时刻的无噪声核算目标：

$$
P_{target}=\sum_{b\in\{A1,A2,A3\}}(P_{IT,b}+P_{HVAC,b}+P_{aux,b}+P_{loss,b}).
$$

- `P_IT` 是服务器及独立列明的网络／其他 IT 终端消耗，服务器 PSU 损耗若已含在输入功率内不得再计。
- `P_HVAC` 是各唯一暖通资产输入电功率，只记一次。
- `P_aux` 是明确列入边界的非 IT／非暖通辅助支路，如照明与控制负载；不与设备自耗重复。
- `P_loss` 只包含边界内配电线损、UPS 转换损耗、变压器损耗；不包含储能支路损耗。
- 机柜、列头柜、UPS、变压器和楼栋表均为不同覆盖层级的观测；不是上述公式中额外相加的负荷。

### 4.2 A/B 与储能

终端设备输入功率先产生一次，再分到 A/B：`P_A = share_A × P_device`，`P_B = (1-share_A) × P_device`。份额范围 `[0,1]`，由端口配置及运行状态决定；上游各路再逐级加独立损耗，不给两路各分配完整设备需求。

主模拟不生成储能状态或调度策略，不把原始市电净输入冒充标签。将来接真实数据时：若只能获得含储能的市电表，需要明确储能功率方向、接入点、支路损耗和边界后重建目标；不能一概用“市电减储能功率”。本设计的目标是直接从排除储能后的设施负荷树生成。

### 4.3 核算数据与模型数据隔离

无噪声设备真值、负荷分解、端口份额、内部热状态和守恒残差保存到 `audit/`；模型只能通过输入清单白名单读取观测与合法未来条件。`audit/` 不得作为模型特征来源。

目标历史是合法输入，但未来目标不是；真实边界总表与历史目标高度相关本身不是泄漏。需要防止的是：把目标复制成多个实体特征、把未来目标塞入计划、或用全边界代理掩盖空间机制没有贡献。

## 5. 全 CSV 目录与文件合同

### 5.1 持久化约定

以下是待实现目录，当前不创建这些文件。`your_version/` 下数据、元数据、配置、审计均为 CSV，不放 JSON、NPZ 或把 JSON 字符串藏进 CSV 单元格。模型训练产生的 checkpoint 不属于此目录及此格式约束。

```text
 dataset/aidc_gslaformer/your_version/
   manifest.csv
   metadata/
     sites.csv
     target_contracts.csv
     target_meters.csv
     assets.csv
     entities.csv
     entity_attributes.csv
     feature_catalog.csv
     applicability.csv
     measurements.csv
     relations.csv
     service_shares.csv
     aggregation_members.csv
   config/
     parameters.csv
     profiles.csv
     splits.csv
     input_views.csv
   observations/
     it_power/<building_id>/<date>.csv
     distribution/<building_id>/<date>.csv
     hvac/<building_id>/<date>.csv
     space/<building_id>/<date>.csv
     compute_training/<building_id>/<date>.csv
     compute_inference/<building_id>/<date>.csv
     environment/<date>.csv
   targets/<date>.csv
   known_future/weather/<issue_date>.csv
   known_future/workload/<issue_date>.csv
   audit/
     decisions.csv
     generation_summary.csv
     physical_balance/<date>.csv
     latent_truth/<date>.csv
     checks.csv
```

`<date>` 是 UTC 事件日期，`<issue_date>` 是 UTC 发布日期。文件分区不改变表的主键；先查 manifest 再读取，不通过 glob 猜测版本。日期与起止时间在元数据里记录。

### 5.2 通用编码、时间、空值与版本

- UTF-8、英文列名、逗号分隔，标准 CSV 引号转义；每文件一行表头。布尔用 `0/1`，空值为空字段；不用 `NaN`、`inf`、`-999`。
- UTC 时间使用 ISO 8601，例如 `2026-01-01T00:15:00Z`；现场日历时区候选为 `Asia/Shanghai`。示例年份／起点待参数确认，不声称对应真实天气。
- `event_time=t` 表示 `(t-15min,t]` 区间；`origin` 是最后历史区间末端。窗口读取同时满足事件时间和到达时间约束。
- 元数据版本区间 `[valid_from, valid_to)`；空 valid_to 表示持续有效。版本必须包含 `available_at`、整数 `revision`。
- 动态观测保留迟到／修订记录，选择 `available_at <= origin` 的最新合法版本；相同主键和 revision 冲突时报错，不采取“最后一行覆盖”。这是新来源适配需求，旧加载器尚不能直接支持完整历史修订。
- 不适用由 applicability 声明；设备不存在由有效期表达；缺测写 observed=0、value 为空；真实零值写 observed=1、value=0。三者不可互换。

### 5.3 注册表精确表头

下表字段顺序即建议表头。`site_id` 在本数据中始终只有一个园区值；楼栋另用 `building_id`。

| 文件 | 表头／一行含义 |
| --- | --- |
| `manifest.csv` | `dataset_id,contract_version,generator_version,seed,relative_path,table_name,row_count,sha256`；一行一个数据文件，清单不含自身哈希 |
| `sites.csv` | `site_id,site_name,timezone,freq_seconds,data_kind`；data_kind 固定 synthetic |
| `target_contracts.csv` | `site_id,target_id,quantity,boundary_id,unit,sign_convention,interval_label,aggregation,coverage_description,storage_excluded` |
| `target_meters.csv` | `site_id,target_id,measurement_id,coefficient,valid_from,valid_to,available_at,revision`；非重叠目标计量项及符号，模拟用专用虚拟总表，不与输入总表混用 |
| `assets.csv` | `site_id,asset_id,asset_type,building_id,floor_id,room_id,module_id,cooling_train_id,valid_from,valid_to,available_at,revision`；一物理资产一身份 |
| `entities.csv` | `site_id,entity_id,asset_id,group_name,entity_type,port,route,space_level,valid_from,valid_to,available_at,revision`；空间／任务池允许 asset_id 为空 |
| `entity_attributes.csv` | `site_id,entity_id,attribute_name,numeric_value,category_value,unit,valid_from,valid_to,available_at,revision`；数值／类别恰填一种 |
| `feature_catalog.csv` | `feature_name,group_name,value_type,unit,temporal_aggregation,spatial_aggregation,denominator_feature,scope,description`；一行一个分组特征合同 |
| `applicability.csv` | `site_id,entity_id,feature_name,applicable,valid_from,valid_to,available_at,revision` |
| `measurements.csv` | `site_id,measurement_id,source_name,entity_id,feature_name,source_unit,scale,offset,role,valid_from,valid_to,available_at,revision`；role 为 primary/backup |
| `relations.csv` | `site_id,relation_id,src_entity_id,dst_entity_id,relation_type,valid_from,valid_to,available_at,revision` |
| `service_shares.csv` | `site_id,relation_id,share_kind,share,valid_from,valid_to,available_at,revision`；冷却责任／任务部署份额，不混入普通边权语义 |
| `aggregation_members.csv` | `site_id,aggregate_entity_id,member_entity_id,feature_name,aggregation,weight_feature,valid_from,valid_to,available_at,revision`；一行一个聚合成员，电力、任务聚合分别定义 |

metadata 的自然键是业务 ID 加 revision；属性／适用性／聚合成员的键还包含属性名、特征名或成员 ID。所有引用必须在有效区间内有合法端点；主测点唯一规则在“实体 × 特征 × 可见有效快照”上检查。

### 5.4 观测、标签与未来条件精确表头

所有 observations 分区统一长表：

```csv
site_id,measurement_id,event_time,available_at,revision,value,observed,quality,coverage_fraction
```

- 一行是一条测点记录；主键 `(site_id,measurement_id,event_time,revision)`。
- `quality` 枚举建议 `good/missing/delayed/outlier/revised`；迟到不等于值无效，但起点未到达时不可读。异常值是否屏蔽由特征合同规定，不能看测试标签后决定。
- `coverage_fraction` 表示区间有效覆盖比例；不足门槛时拒绝作为完整区间观测，门槛写参数表。
- 同一物理量不在多个业务文件重复写权威记录；任务归因功率与设备总功率使用不同特征及测点 ID。

targets 表：

```csv
site_id,target_id,event_time,available_at,revision,value,observed,quality
```

known_future 表：

```csv
site_id,condition_id,scope_entity_id,target_time,published_at,revision,feature_name,value,observed
```

未来条件原始版本表无需给每个 origin 复制一份；加载时对每个预测起点执行 `published_at <= origin` 选择。`scope_entity_id` 为空表示园区级，非空表示局部条件；当前模型只接受 `[B,H,K]`，局部条件进入模型前必须显式做园区聚合或另行扩展，不能把不同作用域混在同一 feature 索引中。

### 5.5 配置与审计表头

| 文件 | 表头 |
| --- | --- |
| `parameters.csv` | `parameter_name,scope_id,value,value_type,unit,status,source,description` |
| `profiles.csv` | `profile_id,freq_seconds,history_steps,horizon_steps,train_stride,eval_stride,target_id,input_view` |
| `splits.csv` | `split_id,site_id,target_start,target_end,role`；目标区间右端时刻使用左闭右开范围 |
| `input_views.csv` | `view_id,entity_id,feature_name,enabled,reason`；穷举允许的实体特征，默认拒绝未列项 |
| `decisions.csv` | `decision_id,status,proposal,confirmed_value,source` |
| `generation_summary.csv` | `dataset_id,item,value,unit` |
| `physical_balance/<date>.csv` | `site_id,event_time,boundary_id,it_kw,hvac_kw,aux_kw,loss_kw,input_kw,residual_kw` |
| `latent_truth/<date>.csv` | `site_id,entity_id,event_time,variable_name,value,unit`；生成器真值，不是模型 latent 标签 |
| `checks.csv` | `check_id,scope,status,actual,expected,detail` |

参数表必须完整展开最终值，不能把“默认值”藏在生成器里。涉及物理数值的功率上下限、容量、COP、热容、损耗、天气幅度、任务随机过程及噪声强度目前未标定，实施前须形成独立参数评审；不得把未填写项按零值运行。

## 6. 算力与多层统计的统一语义

### 6.1 设备指标与任务指标分开

设备总功率、GPU 总功耗和训练任务归因功率可能重叠。设备总功率用于终端负荷核算；组件功率和任务归因功率用于解释，不再相加到总目标。

- 设备观测覆盖服务器／机柜／列头柜／机房／楼栋；聚合来源通过 aggregation_members 指定。
- `compute_training`、`compute_inference` 的稳定对象是“统计范围 × 任务类型”的任务池，不是每个瞬时作业一个节点。
- 任务到服务器的 `runs_on` 只使用生成器实际安排的部署；父层聚合不反向捏造叶子分布。
- CPU 与 GPU 不能共用一条利用率序列；内存占用、利用率、功率、温度也不是彼此改名。
- 物理设备温度是设备状态。任务分类温度如需提供，标为“该任务涉及设备温度的加权统计”，不是两类任务各有一个独立物理温度。

### 6.2 候选字段字典

| 对象 | 动态特征建议 | 单位／聚合要点 |
| --- | --- | --- |
| 服务器 | `active_power_kw,cpu_utilization,gpu_utilization,memory_utilization,cpu_power_kw,gpu_power_kw,memory_power_kw,cpu_temperature_c,gpu_temperature_c,memory_temperature_c,cpu_used_cores,gpu_used_count,memory_used_gib,online_fraction` | 功率为输入／组件观测；利用率 fraction；温度 degC；容量为对应分母 |
| 训练任务池 | `active_jobs,queued_jobs,cpu_used_cores,gpu_used_count,memory_used_gib,attributed_power_kw,completed_work_units` | 并发量、队列量为存量；完成工作量为区间流量，工作量单位在字典固定 |
| 推理任务池 | `active_jobs,queued_requests,request_count,cpu_used_cores,gpu_used_count,memory_used_gib,attributed_power_kw` | request_count 为区间新增请求数；不能和队列长度混加 |
| 机柜／列头柜／楼宇等 | `active_power_kw`，以及实际可定义的利用率／温度统计 | 不强迫没有 GPU 的配电设备“具备 GPU 利用率”；任务统计挂在该范围任务池 |
| 变压器／UPS 端口 | `active_power_kw,load_ratio` | 负载率基于本设备容量，不直接由园区目标除常数生成 |
| 暖通 | `active_power_kw,running_fraction,frequency_hz,flow_m3h,supply_temperature_c,return_temperature_c` | 按设备类型声明适用性；冷却塔与各类泵不强行填相同温度和频率 |
| 机房环境 | `room_temperature_c,room_humidity_fraction` | 来自热状态与环境机制；空间总表仅在允许视图启用 |
| 室外环境 | `outdoor_temperature_c,relative_humidity_fraction` | 观测与预报分别保存 |

所有字段是新 schema 候选，不是当前旧 profile 已接受的列名清单；字段纳入前要求训练段存在有效观测。

### 6.3 聚合规则

1. 可加量：对不重叠叶子集合求和。任务计数须按任务 ID 去重；跨多个服务器的同一训练任务不能每台各算一份活跃作业。
2. 利用率：先汇总已用资源与总容量，再相除；不能把利用率相加，也不能在异构硬件上无权平均。
3. 温度：明确容量／数量加权平均或最大值，各自独立特征；禁止“温度求和”。
4. 功率：设备输入为唯一核算入口；父表可以包含自身损耗，与子表差值要有来源。A/B 列头柜对子机柜只覆盖各自端口份额。
5. 训练／推理资源占用相加不得超过设备容量。任务归因功率之和不得超过可归因设备总功率，空闲和系统开销单列残差。
6. 父层统计同时保存覆盖率；下级缺测时不能把部分求和标为完整观测。完整父级物理表和缺测子级统计允许不同，但必须标明 measured/derived 来源。
7. 模型消融同时移除被消融变量的确定性聚合代理。例如移除服务器功率却保留其完全等价机柜汇总，不足以证明服务器信息不重要。

## 7. 模拟生成逻辑：先产生行为，再产生观测

### 7.1 可复现生成顺序

1. 验证已批准参数和 D01–D07；构建空间、资产、端口和关系，生成稳定排序与 ID。
2. 分离拓扑、天气、任务到达、部署、设备差异、测量噪声、缺失等随机流。输入视图变化不得重新抽取物理真值。
3. 生成室外温湿度：日周期、缓慢变化及相关扰动；90 天只代表设定季节窗口，不伪装全年季节覆盖。
4. 生成训练／推理任务到达、持续、排队和资源分配。训练具有持续块状占用，推理具有请求波动；共享外生驱动但保留独立变化，避免两类任务完全相同。
5. 由资源占用和硬件容量生成服务器组件功率及总输入功率。容量限制在调度时处理，不能事后大量裁剪掩盖不合理任务。
6. 根据房间 IT 热负荷、环境和热惯性生成温度；由既定控制规则安排三组冷机及塔泵出力，再生成暖通用电。
7. 将终端电功率分到 A/B 路，逐级计算配电、UPS、变压器损耗及辅助负载；生成各层物理电表与三楼总目标。
8. 对真值应用测量噪声、丢失、延迟、修订，形成观测表；按发布时间形成计划／天气预报版本。
9. 生成聚合血缘、输入视图、时间切分、核算与审计清单；所有引用和物理检查通过后才形成完整 manifest。

### 7.2 电热机制最低要求

服务器输入功率可采用空闲项、CPU/GPU/内存占用项、交互项和电源转换项。硬件参数明确写入参数表；非负、容量合理，但不声称复现特定服务器型号。

房间热状态采用简化能量平衡，例如：

$$
C_r\frac{T_{r,t+1}-T_{r,t}}{\Delta t}=Q_{IT,r,t}+UA_r(T_{out,t}-T_{r,t})-Q_{cool,r,t}.
$$

`C_r` 使用 kWh/℃、`Δt` 使用小时，其余热流为 kW。内部求解步长可小于 15 分钟，必须验证稳定性后输出区间均值；不能用事后裁温度让发散模型“看起来正常”。初始热状态和预热期写参数，预热输出不混入正式 90 天。

冷机电功率由制冷量／COP 与明确待机项决定；COP 随室外条件和部分负荷变化。塔泵功率与运行台数、流量／频率相关，频率立方关系只能作为明示的简化近似。未满足冷量需求应进入室温变化／未满足负荷审计，不能凭空增加制冷量。

配电逐级满足输入等于下游需求加损耗。变压器使用空载和负载损耗项，UPS 使用负载相关损耗；不能给任意大负荷配任意小容量。各级量纲一致且能追溯到终端，不把总目标按比例复制到所有上游实体。

这是用于预测机制试验的简化生成器，不是电气潮流、实际水力网络或设备控制器数字孪生。

### 7.3 未来计划与未知未来

- 天气预报从发布时可见状态产生、带预测误差和版本；不得直接把未来实测天气无噪声复制进去。
- 计划描述预测时已经安排的资源需求；实际任务还受取消、延期、临时请求影响。实现结果不能反向修改历史发布版本。
- 无可靠未来计划时可启用 `no_future_conditions` 视图；日期／星期等确定性日历仍合法。
- 未来输入只允许声明白名单；任务完成后的实际时长、最终吞吐、未来设备总功率与内部热状态不得作为已知未来。

## 8. 输入视图与防捷径实验

各视图共享相同物理真值、目标、时间切分和噪声来源，改变的是可观察信息，不生成互相不一致的新世界。

| 视图 | 保留信息 | 用途 |
| --- | --- | --- |
| `operational_full` | 合法历史总负荷、可得多层电表、算力、暖通、环境 | 检查贴近实际可观测条件的预测能力 |
| `no_boundary_meters` | 保留合法历史目标与细粒度对象；去除园区／楼栋完整边界功率及其确定性代理 | 检查额外空间观测是否有价值；不称为完全无总表历史 |
| `sparse_observations` | 同一对象体系，加入预声明缺测、通信延迟与覆盖不全 | 检查可用性和质量处理，不隐藏失败窗口 |
| `no_future_conditions` | 历史观测不变，未来计划／预报不输入 | 判断已知未来信息贡献 |

基线至少包括 persistence、seasonal naive、total_only、direct；关系和实体历史消融使用相同窗口与共同标签 mask。季节基线按 15 分钟合同使用日／周周期，历史不足必须报告回退或不合格窗口。

相关系数仅用于审计，不用“所有输入与目标相关必须低于某阈值”否定真实物理关系。重点检测逐值复制、仿射复制、全边界覆盖与特征血缘；高相关代理应披露，并在适当输入视图中移除。

关系作用另用受控成对样例检验：保持可观察历史总量相近、改变部署／服务关系并重新运行物理过程，核对未来差异能否由起点可见关系解释。不能只打乱关系而沿用不相容标签，也不要求 full 必然胜出。

## 9. 时间、切分与数据体量

### 9.1 已确认时间尺度与候选窗口

90 天 × 每天 96 个区间，共 **8640 个时间点**。三个 profile 共用同一原始数据，不生成三份复制时序：

| profile | 未来时长 | 预测步数 | 候选历史长度 |
| --- | ---: | ---: | ---: |
| `aidc_csv_short` | 6 小时 | 24 | 48 小时／192 步 |
| `aidc_csv_day` | 24 小时 | 96 | 48 小时／192 步 |
| `aidc_csv_multiday` | 72 小时 | 288 | 48 小时／192 步 |

48 小时历史是建议而非已批准参数；周期基线若使用周周期，须单独声明更长历史或不能计算，不能假称同输入条件。日期起点、参数种子、训练／评估 stride 也须在配置中显式确定。

候选时间切分为前 54 天训练、中 18 天验证、后 18 天测试，即 5184／1728／1728 个目标时间点。同一 origin 的完整未来区间必须落入同一 split；验证／测试历史可以读取此前已经可见的历史，不能读取未来。归一化、缺失处理统计和选特征只用训练段。

仅一个园区、90 天数据：可验证同园区时间外推，不能证明跨站点或跨全年季节泛化。三栋楼不随机拆为训练／测试“独立站点”。

### 9.2 体量与资源

按候选 192 台服务器：每个服务器每时间点一条宽记录将有 **1,658,880 行**；本方案采用长表，若 14 个指标均适用，则服务器设备观测有 **23,224,320 行**，尚未包含任务、配电、暖通、其他聚合层、端口和修订版本。

这只是配置推导，不是实测文件大小或运行耗时。采用按业务／楼／日分区 CSV、流式生成和分块校验；不得把所有文件一次性装入内存。字段数、空值率、ID长度影响磁盘占用，实施前先用小样本实测估算并取得完整生成预算。

保存服务器级数据不等于一次把所有实体送入模型。实施时需核算完整节点轴、组内注意力及关系注意力的 B×L×N² 成本，尤其 H=288 的递推；若需聚合输入，必须另设显式输入视图和对照，不能偷偷删服务器或降低已确认的时间分辨率。

## 10. 与当前 GSLAformer 的接入差距

| 项目 | 已核实当前事实 | 本方案要求 |
| --- | --- | --- |
| 持久化 | 现加载器读取 JSON+NPZ，synthetic 门禁 | 新增 CSV 来源适配及校验；不能改扩展名冒充可用 |
| 时间分辨率 | 配置支持 freq_seconds，但现有 AIDC 样例是小时级 | 独立 900 秒 profile，检查日历、步长、horizon、周期基线和资源 |
| 空间层级 | `SPACE_LEVELS` 只有 aidc_boundary/building/room | 新增楼层的正式层级合同和测试；电路模组不硬塞成 room |
| 双路关系 | supplies 支持多父级，UPS 端口有方向约束 | 保留资产／端口身份、覆盖份额及核算；现 attention 不求解电力流 |
| 多层算力 | 现有字段集不是本方案完整指标集合 | 新 feature schema、聚合合同、容量分母和任务归因校验 |
| 历史修订 | 旧 NPZ 单元只保留单值和到达时间 | CSV 来源保留版本，在每个 origin 做 as-of 选择，不只导入最终版 |
| 未来条件 | `[B,H,K]`，不是每实体一组未来条件 | 局部计划做显式聚合；若要实体级未来计划，另立模型扩展决策 |
| 拓扑变化 | 历史内允许变化；起点已知 horizon 内变化被拒绝 | 主序列固定拓扑；变化案例独立，不承诺现模型消费全部故障模拟 |

建议实现 CSV→现有内存 batch 的独立适配层，复用缩放、掩码与模型调用，不在 `your_version/` 内落地 JSON/NPZ 中间包。是否允许目录外短期缓存由后续实施方案决定，不把隐式缓存作为唯一真源。

本方案不授权修改模型、旧配置、旧生成器或既有验收门槛。CSV source contract 的版本升级与模型计算版本是否变化，应在接入实施时分别判定。

## 11. 验收规则与交付阶段

### 11.1 数据验收清单

| 编号 | 验收内容 | 明确通过条件 |
| --- | --- | --- |
| V01 | 文件与表头 | manifest 枚举全部 CSV；表头、类型、行数、SHA-256 相符，无越界路径；非法空值／非有限值拒绝 |
| V02 | 身份与拓扑 | 资产数符合批准参数；有效快照外键完整；同空间层级父级唯一；supplies 单独无环；A/B 不重复设备身份 |
| V03 | 目标边界 | 三楼目标核算一致；储能与边界外负荷不在树中；上下级电表不重复求和 |
| V04 | 电力／热量 | 无噪声每边界输入=需求+损耗；端口份额守恒；制冷量、未满足需求及热状态一致；容差先写参数，不看结果后调整 |
| V05 | 任务／聚合 | 容量不超限、任务去重、利用率容量加权、温度不求和、任务功率不重复计入设备功率 |
| V06 | 可用性反例 | 注入迟到、未来修订、同键冲突；起点不可见数据不进入输入，冲突拒绝，不适用／缺失／真实零可区分 |
| V07 | 无未来泄漏 | 修改未来标签、未来实测天气、未发布计划，不改变该 origin 输入；修改测试期数据不改变训练统计 |
| V08 | 视图一致性 | 各视图物理真值和目标一致；no_boundary 同时移除边界代理；audit 文件读取被输入白名单拒绝 |
| V09 | 时间与切分 | 8640 个等间隔时刻；24/96/288 步对齐；完整未来标签不跨 split；所有不可用窗口计数可追溯 |
| V10 | 可复现 | 同版本／参数／随机种子产生同一规范 CSV 内容；改变视图不改变物理真值；排序和数值序列化规则固定 |
| V11 | CSV 接入 | 小样本真实加载、batch、前向和无标签请求通过；非法单元和楼层关系有拒绝测试；不能仅验证文件存在 |
| V12 | 模型实验 | 另获预算后运行基线与消融、test、最终权重重载及 forecast；记录负面结果，不把数据验收通过当模型有效 |

带测量噪声的各级读数不要求严格零残差；应按预声明误差模型报告残差分布。严格守恒检查针对独立生成真值，不能通过修改标签迎合电表。

### 11.2 实施分期与停止边界

| 阶段 | 产出 | 退出条件 |
| --- | --- | --- |
| P0 设计评审（本轮） | 本文；已确认项与待决项分开 | 目录、表头、生成与验收可审阅；未生成数据 |
| P1 参数与结构冻结 | 已批准拓扑、容量／物理参数、数据规模、时间与视图 | D01–D07 有决议；结构计数和总负荷边界一致 |
| P2 CSV 生成器和小样本 | 独立生成／校验工具及少量真实生成样例 | V01–V10；先报磁盘与内存估算，再申请完整生成 |
| P3 正式模拟数据 | your_version 全 CSV，完整 manifest 和审计 | 完整 90 天通过数据验收；失败保留报告，不覆盖旧完整数据 |
| P4 模型适配 | CSV 内存 batch 路径及定向回归 | V11；旧链路保持可用，独立记录新合同身份 |
| P5 实验 | 三 horizon 的基线与消融及推理产物 | 单独批准训练预算后执行 V12；不因文件生成完自动开训 |

## 12. 待评审清单与本轮完成界限

已确认：仅设计、三楼合计设施需求且排除储能、15 分钟、90 天、6/24/72 小时。

待评审：

1. D01–D07 的接线、冷却分配、设备规模与混部假设；现场未知时可批准“仅用于模拟”，不必等待真实单线图，但批准须明确。
2. 热力、电力、容量和任务参数；噪声、缺失、延迟和修订的幅度及比例；所有参数先声明再生成。
3. 日历起点、种子、48 小时历史、54/18/18 天切分及 stride。
4. 全 CSV 长表及分区方案；完整数据规模与资源预算。
5. 输入视图具体允许的实体特征和局部未来计划的聚合规则。
6. CSV 加载器、楼层类型及历史修订的后续实施授权。

本轮仅完成可评审方案。没有创建 your_version 数据、没有改动 my_version、没有更改旧模型或训练配置；文档验收不等于数据生成、加载验证或模型验收。后续工程实跑证据仍按仓库约定记入 [实施规格 §12](../v1/implementation.md#work-packages)，不在多个文档复制测试计数。
