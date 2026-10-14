# DOC-005：文档职责整理记录

> 状态：文档整理完成，结构与迁移保全检查通过。仅文档重组，不是模型实现、训练、机制测试或业务验证。模型设计仍为 R4，数据／模型 schema 仍为 3。
>
> 本记录说明旧章节去了哪里、哪些内容被摘要或去重，以及本次实际检查的范围。现行规范从[目录](../README.md)进入；旧报告中的数字节号属于旧结构，不按新文档同号章节解释。

## 1. 权威与目录约定

- 设计：原理、范围、决策与取舍。
- 输入输出：业务对象、计量、标准包、外部张量与输出字段。
- 实施：内部计算、训练恢复、接入、产物生命周期及 I1–I6。
- 验收：通用断言、base、M1–M5、AIDC 场景、对照、阈值与运行证据。
- examples：具体配置实例，不是配置 schema 或已可运行功能。
- archive：历史决策、审查及实际检查回执，不承担现行规则。

本次合并后撤下 integration.md、mechanism_tests.md、synthetic_aidc.md 的独立入口；报告和配置移动到对应子目录。不是删除它们的独有规格或历史结论。架构图源、HTML、PNG 与绘图回执不改；architecture 下两份 Markdown 只同步导航。

## 2. 原章节去向

下表按整理前标题编号列出；目标优先使用稳定锚点，不依赖重排后的章节数字。归并表示只保留一份完整规则，其他位置改摘要／引用，而非删掉要求。

| 原文件／章节 | 现行去向与处置 |
| --- | --- |
| README 全文 | [目录](../README.md)：按现行规格、样例、历史分区；去掉会话式授权叙述 |
| design §1 | [设计](../design.md)：目标、非目标和完整交付边界 |
| design §2 | 设计的实体、空间、关系与读出原理；图仍只引用现有 architecture |
| design §3 | [输入输出](../input_output.md)：与原 implementation §2 合并，设计只保留语义摘要 |
| design §4 | [设计](../design.md)解释原理／代价；[实施计算](../implementation.md#implementation-3)保留精确算子与数值规则 |
| design §5–6 | 设计解释两阶段与回灌；[损失](../implementation.md#implementation-5)、[训练](../implementation.md#implementation-6)、[rollout](../implementation.md#implementation-4-1)唯一维护执行规则 |
| design §7.1 | 当时仓库观察保留在[演进记录](design_history.md)，不宣称本次重新核查源码 |
| design §7.2–7.4 | [内部 API](../implementation.md#implementation-7-1)、[隔离接入](../implementation.md#implementation-7-2)与复用分析合并 |
| design §8 | [验收](../acceptance.md)：数据、模型、完整流程、消融与完成标准 |
| design §9 | [产物生命周期](../implementation.md#implementation-11)、配置／依赖；设计保留回退与能力限制 |
| design §10–11 | 设计中的当前有效决策、业务前提和后续真实数据输入 |
| design §12 进度 | [唯一工作包](../implementation.md#work-packages)；根 AGENTS 仍管项目状态 |
| design §12.1 | [演进记录](design_history.md)：保留历史表、D/E/R3/R4 修订和过程事实 |
| design §13–14 | 设计保留全部参考来源，结论收束到功能／业务成功的区别 |
| implementation §1 | 设计集中 D1–D5；[整体验收](../acceptance.md#completion-definition)集中完成条件 |
| implementation §2 | [标准包与 batch](../input_output.md#standard-package)，含 scaler、版本、监督隔离与 split |
| implementation §3 | [模型计算](../implementation.md#implementation-3)，原精确规则保留 |
| implementation §4.1 / §4.2 | [rollout](../implementation.md#implementation-4-1)／[新实体断言](../acceptance.md#new-entity-checks) |
| implementation §5–6 | [损失与统计](../implementation.md#implementation-5)／[训练恢复](../implementation.md#implementation-6) |
| implementation §7.1–7.2 | 实施规格唯一文件/API/CLI 定义，合入 integration 的补项 |
| implementation §7.3 | [拒绝行为](../input_output.md#error-contract) |
| implementation §8.1 | [配置约定](../implementation.md#configuration)与[外置 base 样例](../examples/functional_profile.example.json) |
| implementation §8.2 | [周期 base](../acceptance.md#base-fixture)：公式、数据与新实体请求规则保留 |
| implementation §9 | [通用断言](../acceptance.md#contract-assertions)、[阈值](../acceptance.md#fit-thresholds)、[拟议命令](../acceptance.md#planned-commands) |
| implementation §10.1 / §10.2 | [对照定义](../acceptance.md#variants)／[指标算法](../implementation.md#implementation-10-2) |
| implementation §11–12 | 产物生命周期和统一工作包；输出字段交给输入输出契约 |
| input_output §1–2 | 输入输出保留目标口径、六组目录与资产／实体／测点／特征区别 |
| input_output §3.1 / §3.2 | [类型业务合同](../input_output.md#relation-contract)／[关系计算](../implementation.md#implementation-3-2)；两跳、反向通信与非物理求解边界保留 |
| input_output §4 | [来源映射](../input_output.md#input-output-4-1)、[时间单位](../input_output.md#input-output-4-3)、标准包；同主题只保留一个明细定义 |
| input_output §5 | 标准 batch 增补 AIDC G/Gd、边界实体、类型变化和未来隐藏量限制；[缺失](../input_output.md#missing-semantics)、[任务计划](../input_output.md#future-plan-availability)独立解释 |
| input_output §6 | [兼容性](../input_output.md#site-data-rules)；采样算法移到实施训练章节 |
| input_output §7.1 | [AIDC profiles](../acceptance.md#aidc-profiles)，15 分钟仍只是后续容量目标 |
| input_output §7.2–7.3 | [输出合同](../input_output.md#output-contract)与请求示例；schema/单位/质量语义不弱化 |
| input_output §8 | [AIDC 断言](../acceptance.md#aidc-assertions)，澄清留出验证与留出测试的不同角色 |
| integration §1 | 设计工程归属和实施隔离接入，不新增另一训练框架 |
| integration §2–3 | [组件复用](../implementation.md#integration-2)、[基础设施](../implementation.md#integration-3)；保留源码事实与限制，明确是 DOC-003 核查位置 |
| integration §4 | 文件/API/配置/CLI/产物职责合入实施；真实源目录只是建议，不创建或承诺生产导入器 |
| integration §5–6 | R4 补项直接进 I1–I6；已知风险留设计，实施前置留实施规格 |
| mechanism_tests §1 | 验收入口的证据分层，不把公式探针当模型测试 |
| mechanism_tests §2–7 | [M1–M5](../acceptance.md#mechanism-tests-2)：全部数学任务、family split、成对断言、负对照保留 |
| mechanism_tests §8–9 | [机制矩阵](../acceptance.md#mechanism-tests-8)、阈值引用及 suite_result 证据规则 |
| synthetic_aidc §1 | 验收入口的 base／机制／AIDC 三层边界 |
| synthetic_aidc §2–4 | [生成顺序](../acceptance.md#synthetic-aidc-2)、服务器／配电／热惯性公式、站点与观测场景、site_time 切分 |
| synthetic_aidc §5–6 | [AIDC 矩阵](../acceptance.md#synthetic-aidc-5)、站点指标、资源和产物要求 |
| synthetic_aidc §7 | 验收结尾统一说明全部待执行，不伪造通过结论 |
| aidc_profile.example.json | [examples 样例](../examples/aidc_profile.example.json)：仅 data.package 随目录深度调整，其他值不变 |
| review_r3.md / validation_r4.md | 移入本目录，只调整相对导航链接，不改历史审查范围、结果、公式命令和数字 |
| validation_checks.json | 移入本目录，字节不变；pass 仍仅代表当时静态检查 |

## 3. 显式澄清与保留边界

1. **留出站点**：原 input_output §8 将其统称为“不参与选模”，与既有 AIDC 协议不符。本次按原 site_split 的角色澄清：08–09 是留出验证站点，参与选模；10–11 是最终测试站点，不参与选模。所有留出站点都不参与权重训练、scaler 或词表拟合。没有改变站点列表、时间切分或选择指标。
2. **配置移动**：examples/ 比原位置深一层；data.package 调整到 ../../../results/gslaformer/fixtures/...。模型参数、阈值、组与关系 schema 不变。未来生成到 scripts/smoke 时需再次按实际目录解析，不能原样复制路径。
3. **历史引用**：旧报告的 design／implementation 数字节号不对应新编号；需用本表或整理前 Git 快照复核。只更新可点击导航，不把原文证据改写为本次新验证。
4. **模型状态**：未实现、未训练，完整功能与业务效果仍分别判断。旧 R3 审查、R4 文档通过和本次重组通过都不等于模型验收。
5. **执行限制**：没有运行文档中的拟议模型命令，没有改依赖、模型、数据管道或 .gitignore，没有提交或推送。
6. **切窗口径**：原通用描述中的 stride=1 明确限定为周期 base，AIDC 仍按原协议 train=1、valid/test=H；没有改动已有配置值。模型公式原样迁回各自主题，补充 G 仅枚举动态组、质量标记不等于未来 teacher 质量的符号说明，避免精简后误读。

## 4. 本次验证记录

### 4.1 基线与可恢复性

整理前基线为 `f30e5f01f3117486f719478d906bf196cba1ccc5`。逐文件以 `git show <基线>:docs/gslaformer/<原文件>` 回读并比较，原根目录 11 份文件均与本轮只读备份字节一致。历史完整文本可从该提交恢复，不依赖会自动清理的临时目录；本轮没有改写 Git 历史。

### 4.2 已实际执行

在仓库根目录执行，最终退出码均为 0：

```bash
env -u PYTHONPATH .venv/bin/python /Users/wangzf/.hermes/profiles/coder/cache/scratch/gslaformer-doc005/check_docs.py
git diff --check -- AGENTS.md docs/gslaformer/
```

check_docs.py 为本轮临时只读核验工具，不是新增模型测试或长期仓库门禁；它读取整理前备份／固定 Git 基线与当前文档。以下列出真实检查范围，不能用脚本文件名替代模型证据：

| 检查 | 结果 |
| --- | --- |
| Markdown 文件、代码／数学围栏、现行章节编号、显式锚点与行末空白 | 11 份 Markdown 通过，包含未跟踪的新文件 |
| 本地文件链接与章节锚点 | 144 处通过；目录内历史导航也检查，历史命令中的旧绝对路径不伪装成当前有效路径 |
| 完整配置的 JSON、shape/head 整除、L/H 限额、相对包路径 | 两份样例通过；除路径深度外，AIDC 样例与原 JSON、base 样例与原内嵌 JSON 的全部值相等 |
| 原设计公式 | 9 个数学块完整保留在设计／输入输出／实施的相应职责中 |
| M1–M5 与 AIDC 公式 | 9 个数学块完整保留；只是迁移一致性检查，没有运行生成器或训练 |
| 核心协议段落 | 原 implementation §3/5/6 的 36/36、mechanism_tests §3–7 的 28/28、synthetic_aidc §2–4 的 35/35 保留；仅规范化导航和空白后比较 |
| 验收矩阵、源码复用表与拟议命令 | 原运行矩阵及组件复用明细保留，完整 bash 命令块未改；没有执行该命令块 |
| R3／R4 历史报告 | 除链接目标调整外正文一致；历史数字、结论、限制和命令不变 |
| 历史 validation_checks.json | 迁移前后字节一致，pass 仍是历史静态检查结果 |
| 架构图及绘图证据 | 除两份仅改导航的 Markdown 外，19 份资产 SHA-256 全部一致 |
| 既有 .gitignore | 与本轮起点 SHA-256 相同，本轮不修改已有工作 |
| 定向 diff 检查 | 通过；未用全仓其他工作的状态冒充本轮结果 |

初检发现归档导航错误和过度精简的公式表达，已按原文恢复／修正后重验；未将失败检查记为通过。其余摘要与去重按 §2 的章节映射人工复核，机器段落匹配不冒充全文语义证明。

### 4.3 收口自审

- **规范维度**：已先登记 DOC-005，沿用 dev；只改本目录和根 AGENTS 的文档任务／证据导航，不改模型、依赖和其他工作。
- **需求维度**：完成按主题唯一权威、接入合并、统一验收、样例外置、历史分层与章节去向记录；保留 D1–D5、R4 路线、公式、反例、阈值、运行矩阵和证据边界。
- 本轮为同一上下文的自审，不称独立代理审查。未运行模型 unittest、smoke、浏览器重绘或训练；图示仅做字节保全检查。
