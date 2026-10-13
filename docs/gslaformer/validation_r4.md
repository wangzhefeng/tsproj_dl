# GSLAformer R4：文档与图示验证记录

> 范围：DOC-003 的 AIDC 文档扩展与重绘图示。不是 GSLAformer 代码测试、训练结果、独立四代理审查或生产验收。
>
> 当前记录：文档结构／示例检查与图示检查均已通过，最终范围与未验证事项见 §5。

> 归档补记：DOC-004 将全部图示资产移入 [architecture/](architecture/README.md)。以下命令、计数、哈希及回执中的绝对路径保留为首次 R4 检查的历史事实；当前路径的绘图回执见 [迁移后验证](architecture/diagram_evidence/relocation/architecture.finalize-summary.json)。`validation_checks.json` 同为迁移前历史记录，不冒称本次重新执行。

## 1. 变更范围与历史衔接

保留接手时的 R3 文档／机制任务；补充六组业务对象、类型关系、结构化输入输出、多 AIDC 模拟协议、源码复用、框架接入与 schema 3 配置样例。design 和 implementation 同步升级 R4，D1–D5、R3 修订历史仍保留。

R3 审查 §6.6 提到的范围差异在本轮明确处理：跨 AIDC 是共享训练和独立留出推理目标，不是零样本精度保证；关系从归属 DAG 扩展为逐类型约束；图示与六组／旁路／回灌合同一致。这里是文档一致性处理，不改写 R3 审查的原始结论或数值证据。

未修改模型、数据处理、实验类、run.py、项目依赖或原 v4.0 图片；未启动训练；没有 commit/push。中断后存在其他工作的 `.gitignore` 改动，本轮不修改它。

最终复验观察到 HEAD 从初始基线变为 `1e87343`（R3 文档提交），原图从 `docs/model_arch/v4.0.png` 移到 `docs/gslaformer/v4.0.png`。上述提交和移动不是本轮执行；读取原 Git blob 与当前位置比较，SHA-256 均为 `4817a7169a6a8492ed34884b9037947f00e178523ecbe52bdab92b850718cc49`。本轮只同步三处本地链接，不恢复或重复移动图片。

## 2. 文档结构与设计样例检查

使用项目根 `.venv/bin/python` 执行临时文档检查脚本，内容包括：

- Markdown 编号连续、子章节归属、代码／数学围栏、行末空白；
- 本地文件链接及章节锚点；
- implementation 内 JSON 与独立 AIDC JSON 解析，数据 schema 3 一致；
- 六组、特征单位、关系端点组、维度整除、历史／horizon 限额；
- 训练／验证／测试站点分离，配置相对数据路径位于 results；
- 组件复用文档中的实际源文件与引用行号存在；语义判断基于本轮逐个读取源码；
- 架构图关键边、HTML 与图源 SHA-256 对应回执；
- `git diff --check -- AGENTS.md docs/gslaformer/`。

执行形式（临时核验工具，不是项目模型测试）：

```bash
env -u PYTHONPATH .venv/bin/python /Users/wangzf/.hermes/profiles/coder/cache/scratch/validate_gslaformer_r4.py
git diff --check -- AGENTS.md docs/gslaformer/
```

首次脚本检查仅报告尚未创建的本文件两处链接缺失，其他已执行检查未报错；该次总体返回非零，未记成通过。本报告写入后重新执行，不忽略缺失链接。

### 2.1 独立枚举的窗口与容量核算

下列是设计参数的数学核算，不是 Dataset 实现输出或模型结果。右端标签起点索引 s 满足 `s≥L`、`s≥split_start`、`s+H≤split_end`。训练 stride=1，valid/test stride=H，按每分区第一个合法 s 起步。

| profile | 每训练站 train 窗口 | 每适用站 valid 窗口 | 每适用站 test 窗口 |
| --- | --- | --- | --- |
| L=48,H=6 | 811 | 48 | 48 |
| L=168,H=24 | 673 | 12 | 12 |
| L=168,H=72 | 625 | 4 | 4 |

站点集合为 8 个训练、2 个留出验证、2 个留出测试，无交集。这里 valid/test 每站计数不表示所有站点都参与每个阶段，适用角色以 synthetic_aidc 为准。

`B=2,L=168,heads=4,N_total=128` 的单个 FP32 关系分数张量估算为 88,080,384 字节，即 84 MiB，低于声明的 256 MiB 单张量限额；**不是峰值内存测量**。三天 H=72、十五分钟 H=96 对应一天等时间换算已用 Python 核算；后者不是本轮模型验收范围。

## 3. 图示的真实工具证据

Archify 3.0.1，执行：

```bash
node /Users/wangzf/.hermes/profiles/coder/skills/creative/archify/bin/archify.mjs finalize architecture docs/gslaformer/architecture.json docs/gslaformer/architecture.html --quality showcase --out-dir docs/gslaformer/diagram_evidence --json
node /Users/wangzf/.hermes/profiles/coder/skills/creative/archify/bin/archify.mjs visual-check docs/gslaformer/architecture.html --out-dir docs/gslaformer/diagram_evidence/visual --summary --require-provenance
```

两条命令退出码均为 0：

- validate、deliver、strict check、browser-check 全部 pass；
- showcase **9/9**，0 errors、0 warnings；
- properCrossings=0、resolvedCrossovers=0、ambiguousCorridors=0；
- 1440×900、1600×1000、1920×1080、2048×1320 的浏览器检查通过，无横向溢出；前两个视口按工具明确允许的可读纵向滚动，不冒称全图均在首屏；
- 明／暗主题与 READ/Still 状态通过，截图采集完成；
- `state-feedback` 有三折路由提示，属于可读性建议不是失败；实际视觉检查确认回灌在独立空白通道，不穿盒、不交叉，因此不为了减少折点破坏布局。

使用图像读取工具实际查看了 2048×1320 浅色截图、1440×900 深色截图及最终 PNG：中文完整，主链、总表旁路、条件旁路和紫色虚线反馈环可辨。可记录 `visual_review: passed`；这一结论来自已看图，不来自工具自动宣称。

尝试用个人 Chrome profile 打开导出页面时，browser_exec 被用户 Chrome 的 SQLite 写锁阻止；没有关闭用户浏览器或更改配置。最终 PNG 采用上述独立 Chrome 检查产出的浅色截图，按 `(480,158,1575,1165)` 截取为 1095×1007，保留全部图形内容。它不是原生 SVG 导出。

| 产物 | SHA-256 |
| --- | --- |
| architecture.json | `a5c7315a45544a0623964b5fe39dc75d95688beebf54d2147848656be64e2d5d` |
| architecture.html | `ddcbdb40b22a126a78249132bbc2f061dc48a643222e821e5d260d579b9c89cf` |
| architecture.png | `27a9f8b6251f84d1b5ebdfb944b1496eb58a71bce17e24f3795a58ed7deaedd4` |

迁移前完整证据（仅链接随文件位置更新）：[finalize 回执](architecture/diagram_evidence/architecture.finalize-summary.json)、[浏览器回执](architecture/diagram_evidence/architecture.browser-check.json)、[截图联系表](architecture/diagram_evidence/visual/architecture.visual-check.html)。

## 4. 保留风险

1. 模拟规则尚未校准，不能确保真实跨 AIDC 泛化；生产数据边界与可用点位仍待核实。
2. 两层 typed attention 不保证全层级信息充分；dense attention 不保证生产全量服务器容量。
3. H=6/24/72 是待跑的独立模型验收 profile，未实测长递推训练／内存／精度。
4. 未来属性／成员／关系切换仍明确不支持；设备级未来计划目前仅有站点级声明汇总接口。
5. 全仓 `git diff --check` 返回非零：已有 `.gitignore:165–166` 的 CRLF 行末空白告警。本轮范围的定向检查通过，不把全仓结果记成通过，也不顺手改其他工作的文件。
6. R4 没有新的独立对抗四审查或安全扫描；文档检查不是 SAST，也不是模型单元测试。

## 5. 最终状态

最终复验通过，命令退出码 0：10 份 Markdown、65 处本地链接／锚点、内嵌与独立 JSON 的指定合同断言、23 处源码位置引用、窗口／容量核算和本轮定向 diff 检查无错误。机器结果保存在 `validation_checks.json`；图源和 HTML 的校验值与实际自动回执一致。

DOC-003 可标记为文档任务 Done。模型实现、正式模拟数据集与全部训练／测试／forecast 仍未开始；这不是模型或业务效果完成声明。全仓 `.gitignore` 空白告警保留，未纳入本轮修改。

提交前补记：首次暂存全部新增资产后，`git diff --cached --check` 检出生成文件 `architecture/architecture.html` 的 15 处空白行含空格；此前未暂存检查不覆盖这些未跟踪内容，不能据此声称完整提交零空白告警。逐行核实均为空白行，HTML 的 SHA-256 与已验证交付一致，因此保留生成字节及回执，不手工清理 HTML。排除该生成文件后，其余暂存内容的空白检查通过；本次提交复验的 80 处链接、JSON、图片／HTML 哈希检查通过，未运行模型测试。
