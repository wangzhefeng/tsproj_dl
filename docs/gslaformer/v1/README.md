# GSLAformer 文档目录

**GSLAformer**：Grouped Spatial Latent Autoregressive Transformer，分组空间隐状态自回归 Transformer。项目工作名，不宣称公开名称唯一、原创性或已经发表。

## 1. 当前状态

模型设计版本为 **R4**，数据 schema 为 **3**，模型计算／checkpoint 版本为 **2**（不兼容旧权重）。数据、模型、两阶段训练及 train/test/forecast/compare 已实现，处于工程与模拟验收收口阶段，尚非完整交付。实际命令、测试与产物唯一见[实施规格 §12](implementation.md#work-packages)，根 AGENTS 的 GSLA-001 跟踪治理状态。tests/ 已纳入版本控制；当前新增实现与测试仍保持未提交。

首轮目标是跨 AIDC 总负荷的合成功能闭环：同 schema 可变实体和未见站点冻结推理、类型关系参与计算、起点已知未来条件与双头回灌。真实业务验证不启动，不要求优于对照，不承诺冷启动或生产零样本精度。

## 2. 现行规格：每个主题一个权威位置

| 文档 | 唯一职责 | 不负责 |
| --- | --- | --- |
| [模型设计](design.md) | 目标、原理、关键决策、取舍与能力边界 | 字段明细、代码文件清单、验收运行矩阵 |
| [输入输出契约](input_output.md) | 业务对象、计量、标准包、张量、可用性、外部输出 | 网络内部计算、训练状态机 |
| [开发实施规格](implementation.md) | 计算、损失、训练恢复、组件复用、入口与产物生命周期、工作包 | 重复定义数据字段和验收任务 |
| [统一验收协议](acceptance.md) | 契约测试、base、M1–M5、多站点模拟、对照、阈值及完成证据 | 历史通过结论或生产精度保证 |

阅读顺序：理解业务先读输入输出与设计；编码读实施规格及其引用的契约；验收统一从 acceptance 进入。

实际使用见[根 README 的生成／训练／续训／测试／预测／比较命令](../../../README.md#gslaformer-usage)。这里只维护规格导航，不复制操作手册和进度计数。

## 3. 设计样例与图示

- [AIDC 一天配置](../../../scripts/smoke/gslaformer_aidc_hourly_day.json)、[周期 base 配置](../../../scripts/smoke/gslaformer_functional.json)：直接引用正式配置；键的解释见[配置约定](implementation.md#configuration)。
- [原始 v4.0 图](../arch/model-v4.0.png)：保留作为设计起点，不代表当前完整实现。旧 `architecture/` 及其绘图回执已按用户确认退役。
- [当前架构审计](../arch/gpt-version/README.md) / [离线图文报告](../arch/gpt-version/index.html) / [逐项差异](../arch/gpt-version/comparison.md)：对照原图与当前工作区源码，含整体、机制和对比三组 SVG／PNG／PDF。产物在 `arch/gpt-version/`，配套脚本在其 `scripts/`；这是实现快照，不替代四份规格，现行文字与代码的两处差异单独登记，尚未据此修改模型。

配置唯一维护在 scripts/smoke，文档和测试直接引用，不再维护 examples 副本；相对数据路径按正式配置所在目录解析。退役图示的历史恢复入口见[文档历史摘要](history.md#retired-architecture)，不再将旧截图或回执作为当前目录的验证门禁。

## 4. 历史与整理证据

- [关键决策与历史证据](history.md)：设计取舍、版本兼容边界、失败教训、实验索引及旧资料恢复入口。

历史统一精简保存在 history.md，不再维护 archive 目录或开发流水账。历史证据不随现行规格更新而重新背书；当前状态只见实施规格 §12。

## 5. 维护约定

- 根 `AGENTS.md` 管项目任务与模型／入口／验证状态；实施规格只维护本模型工作包，不另建第二套项目看板。
- 规则变更先修改该主题的权威文档，再更新必要摘要、样例和测试；发现冲突必须显式修正并登记，不静默选一份。
- 当前有效决策写入设计；历史决策与偏差追加到演进记录，不改写过去的执行结论。
- 工作包只有附实际命令、结果和产物后才可 Done；拟议文件、接口和命令不表示已经存在。
- 正式模型产物进入 `results/`，本目录不存生产数据、权重或预测结果。
