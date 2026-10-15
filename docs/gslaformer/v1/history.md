# GSLAformer：关键决策与历史证据

> 本文由原 archive 四份文档提炼，不维护开发流水账或当前测试计数。现行规则见[设计](design.md)、[输入输出](input_output.md)、[实施规格](implementation.md)与[验收协议](acceptance.md)；最新进度唯一见[实施规格 §12](implementation.md#work-packages)。历史通过不等于当前版本通过。

<a id="design-decisions"></a>

## 1. 为什么形成当前设计

| 阶段 | 关键决定 | 原因与代价 |
| --- | --- | --- |
| 初稿 → R2 | 统一命名 GSLAformer；先做合成功能验收，真实数据另行授权；不要求强制胜出 | 将工程可运行与业务采用分开，保留负面结果 |
| R2：新实体 | 同 schema 新实体免重训；不用实体 ID embedding 或按实体 scaler | 用属性与组级统计泛化，ID 只用于对齐；冷启动能运行不等于精度可靠 |
| R2：关系与条件 | 显式关系参与实体交互，支持有发布时间的通用未来条件 | 不能只存归属表却不消费；无可靠未来信息时允许缺失 |
| R2：训练稳定性 | 多步负荷预训练 A，冻结表示链后进入 B；方差仅加权 loss | 固定回灌坐标，但冻结表示不保证是充分预测状态；只在最终课程早停 |
| R3：轨迹 | 池化前增加同组共享的因果实体 GRU | 逐帧集合池化会丢跨时对应；增加实体 memory 与 teacher 延续管理 |
| R3：缺失 | 采用分组软质量 q 与连续有界方差权重 | 全系统完整观测门槛随规模失效；q 是启发式而非置信概率 |
| R3：公平对照 | A/direct/AR 使用相同完整起点计划，lead 相对原 origin；历史关系逐时刻 as-of | 避免对照少拿信息，也避免换父级后等待完整稳定窗口；禁止刷新后来发布的计划 |
| R3：验收 | 周期 smoke 与 M1–M5 分开；关系消息消融与全关系信息消融分开 | 简单季节复制、关系度数旁路或 shape 正确都不能证明机制有效 |
| R3：统计 | horizon-macro 选优；每个训练 origin 只收末端 latent；按模式/变体区分产物 | 不平均 batch 指标冒充 split 指标，不因重叠窗口重复加权，不误要求 naive/direct 生成 latent |
| R4：跨站 | 数据 schema=3，六组对象、多类型关系、站点与时间双轴切分 | 资产/测点/特征与计量边界分开，避免多层功率重复相加；逐类型合法性不等于所有边合并为 DAG |
| R4：容量与接入 | 小时至三天独立 profile；独立数据/实验合同接入 run.py | 不把短 horizon 通过当长周期能力；不复用旧主线的固定宽表和统计回退 |

留出验证站可以参与选模，最终测试站不能参与；所有留出站都不参与权重、scaler 或词表拟合。组级 mean/sum/count 不是守恒约束，也不保证节点数量与输出单调。未来结构变化、真实新站零样本精度不因文档或合成结果自动获得保证。

<a id="review-lessons"></a>

## 2. 审查与实现中的关键教训

- **反例优先于形状断言。** R3 的轨迹、端点配对、计划前缀反例说明“输入集合一样”仍可对应不同未来。基础 GRU/公式探针只证明表达可能性，不证明完整模型学会；原 R3 为单上下文审查，不是独立多代理验收。
- **修正过强结论。** 早期 M1 用例实际是整段置换，不能代替单时刻轨迹交换；参数有梯度不能证明回灌路径有梯度。早期 I2/I3 整体 Done 推断已撤回，以行为和完整验收证据为准。
- **方向、单位和遮罩需要数值检查。** 曾修正消息收发方向、按实体而非特征单元计算 coverage、分钟日历换算、NaN 乘零仍污染编码、静态均值分母以及 future_groups 将节点轴误取 H 等问题。
- **消融必须清除相应信息通路。** no_spatial 去掉查询注意力；total_only 不消费内部实体/关系；no_relation_all 还清除静态度数和质量权重旁路。K=1 且无 latent loss 时不伪造 latent 梯度，也不通过零损失触发 AdamW 衰减。
- **训练与推理必须比较。** 关系层曾在 dropout=0 下出现 train 与 eval+no_grad 不一致，改用显式 MHA+Post-LN+FFN 后重跑；旧失败权重未迁移或重新背书。
- **性能不能靠改语义获得。** 逐边稠密 bias 写入导致大量 CopySlices，改成索引收集/index_add；输出与梯度对齐，不缩短 L/H 或截断反传。有限 batch 外推不是完整 epoch；MPS 探针未显示加速，不能假设换设备就更快。
- **诊断不参与排名。** teacher_z 使用未来观测，仅作诊断，不作为数学上界或选模依据；latent 方差不等于有效空间信息，缺失质量与闭环漂移仍需实验。

<a id="compatibility"></a>

## 3. 计算版本与兼容边界

GSLA-002 将计算/checkpoint 升级为 **2**，数据 schema 仍为 **3**：未来数值和 mask 独立投影，读出 head 沿用 spatial_heads，collate 按真实 batch/padding 检查注意力预算；checkpoint 补统计溯源与冻结清单，运行补资源和请求审计。

旧权重明确拒绝续训、test/forecast 和共享 A 加载，不自动迁移；重新训练使用新 run_id。旧小集拟合、精度和吞吐量不能为计算版本 2 背书。静态链明确为属性统计/类别均值/方向度数聚合后投影，不宣称逐实体非线性适配后池化。

AIDC 正式 profile 的 supplies、meter_covers、runs_on 不开启反向消息，其余类型开启；这不是所有业务图的普遍规则。生成器中的重叠任务计划按可见块求和再平均（缺块池计零），不是逐设备计划。配置现唯一维护在 scripts/smoke，旧 examples 镜像已清退。

<a id="experiment-evidence"></a>

## 4. 重要实验索引（历史，不重跑或覆盖）

以下路径相对仓库根目录，均是原实验产物；这里只保留意义与边界，不复制历次测试计数。

| 证据 | 路径 | 可以说明什么 |
| --- | --- | --- |
| 旧版小集拟合 | `results/gslaformer/small_fit_i4_20261014_r2/suite_result.json` | 机制小集拟合曾通过，不是留出泛化；首次失败另保留在 small_fit_i4_20261014 |
| 旧版机制矩阵 | `results/gslaformer/closure_mechanisms_20261015/suite_result.json` | 执行状态 partial，AIDC not_run；不能当整体验收 |
| 旧版扩展比较 | `results/gslaformer/mechanisms_i6_20261014/extended_artifact_index.json` | 历史 full 的 M1/M2 成对完全区分比例为 0.25/0.875；M3 total_only、M1/M4 direct 曾有更低 MAE，不宣传 full 全面优胜 |
| 旧版性能定位 | `results/gslaformer/profile_multiday_after20_20261014/profile.json` | 优化后的有限 A 更新测量，不代表完整 B-full 训练 |
| 旧版 CPU/MPS 资源探针 | `results/gslaformer/resource_multiday_20261015/resource_report.json`；`results/gslaformer/resource_multiday_mps_20261015/resource_report.json` | 有界 batch 与验证计时，不是正式模型；外推的长训练预算非实测总耗时 |
| 旧版完整 CPU A 单轮 | `results/gslaformer/aidc_multiday_cpu_epoch_20261015_112615/epoch_timing.json` | 原记录总耗时 1833.803 秒、峰值 RSS 1.183 GiB；仅 A，未进入 Freeze/B。计时 completed 与完整训练 interrupted 不矛盾 |
| 旧版工程修复 | `results/gslaformer/closure_evidence_20261015/verification.json`；`results/engineering/closure_24_20261015/verification.json` | 诊断/产物门禁及既有工程问题修复；旧失败日志不是当前失败清单 |
| 计算版本 2 收口 | `results/gslaformer/contract_v2_evidence_20261015/gsla002_verification.json` | 新版 CPU/MPS 流程、权重版本与源码指纹；不是 AIDC 完整矩阵 |

长矩阵仍需资源预算确认。一次 CPU A epoch 授权不等于全课程授权；历史中断保留，不自动重启。当前待办由实施规格 §12 管理，本表不另建状态看板。

<a id="r4-validation"></a>
<a id="doc-reorganization"></a>
<a id="retired-architecture"></a>

## 5. 文档退役与历史恢复

DOC-003/R4 的文档、公式和绘图检查发生在模型实现之前，不是训练验收。DOC-005 将接入/源码复用并入 implementation，机制/AIDC 验收并入 acceptance，数据合同集中于 input_output；旧数字节号不能按当前同号章节解释。

用户先确认合并三份旧文档回执，随后确认退役 architecture/，保留 [model-v4.0.png](../arch/model-v4.0.png) 原图。其 SHA-256 为 `4817a7169a6a8492ed34884b9037947f00e178523ecbe52bdab92b850718cc49`。退役范围不包含另行开发的 arch/。DOC-008 清退 examples 副本；本次 DOC-010 将 archive 四文档提炼为本文，不再保存重复流水账。

已提交的旧设计、R3 审查、文档回执和 R4 图示可从固定基线读取（从仓库根目录执行）：

```bash
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/archive/design_history.md
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/archive/review_r3.md
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/archive/reorganization.md
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/archive/validation_r4.md
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/archive/validation_checks.json
git show 5a50d02a0808b64c29b59eaa627c5c648895244e:docs/gslaformer/architecture/migration.json
```

更早的文档整理前基线为 `f30e5f01f3117486f719478d906bf196cba1ccc5`。**恢复范围仅限提交中的内容**：本轮删除的 implementation_history.md、documentation_checks.md 及 design_history 的后续未提交增量不在上述基线，必要信息已提炼到本文；舍弃的会话流水不能宣称可从 Git 完整恢复。此前未提交的实现图同样不作恢复承诺。
