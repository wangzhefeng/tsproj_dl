# GSLAformer 实现图修订说明

## 范围与边界

- 本次依照用户确认的五项建议修订图稿；不更改任何已有文件。
- 全部新增交付位于 `docs/gslaformer/arch/bot_version/revised_impl/`。
- 不修改模型、配置、原始图或根治理文档；不运行模型测试、训练、资源探针，不提交或推送。
- 本次任务状态与验收在本目录记账，避免违反用户“不修改当前已有文件”的限制。
- 以当前源码为依据，源文件 SHA256 与 Git HEAD 记录于 `manifest.json`。本图是 full 的通用实现概览，direct 单独展示；不是全部消融配置的计算图。

## 图册

| 编号 | 内容 | 可编辑源 | 矢量图 | 预览 | PDF |
| --- | --- | --- | --- | --- | --- |
| 01 | 实体历史与空间表示 | [draw.io](gslaformer_impl_revised_01.drawio) | [SVG](gslaformer_impl_revised_01.svg) | [PNG](gslaformer_impl_revised_01.png) | [PDF](gslaformer_impl_revised_01.pdf) |
| 02 | direct 与自由递推 | [draw.io](gslaformer_impl_revised_02.drawio) | [SVG](gslaformer_impl_revised_02.svg) | [PNG](gslaformer_impl_revised_02.png) | [PDF](gslaformer_impl_revised_02.pdf) |
| 03 | 两阶段训练与监督隔离 | [draw.io](gslaformer_impl_revised_03.drawio) | [SVG](gslaformer_impl_revised_03.svg) | [PNG](gslaformer_impl_revised_03.png) | [PDF](gslaformer_impl_revised_03.pdf) |

[离线图册](index.html) 内嵌三张 SVG，无外部资源依赖。单图 HTML 用于浏览器打印导出。

## 五项修改与源码依据

### 1. 通用结构与六组配置分开

**改动：** 用 G 表示全部组，用 Gd 表示动态组；动态组通过同组共享 GRU，纯静态组绕过 GRU。仅动态组生成 latent。明确全窗口与单时刻形状，关系层数来自配置，不固定写成 ×2。

**保留：** AIDC 六动态组仍是当前配置实例，不否认旧图在该实例下的组数。

**依据：** `models/transformer/GSLAformer.py:89–113`；`layers/gslaformer/entity_history.py:49–84`。

### 2. 空间链与静态链的关键边界补齐

**改动：** summary token 的输出不向后传；关系交互仍操作实体。独立查询读出以可学习 query 为 Q，以关系增强实体和常驻空组 token 为 K/V；池化结果与数量、coverage 是拼接而非相加。

**新增：** 静态链显式接入预测起点的静态证据和关系度数，画出逐组 Linear→GELU、跨组拼接后的单 Linear。动态关系来自各历史时刻，静态度数来自 origin。

**依据：** `layers/gslaformer/spatial.py:17–52,71–154`；`models/transformer/GSLAformer.py:233–260`。

### 3. 修正日历与位置表达

**改动：** 删除旧图所表达的额外“日历 Linear 相加”路径。日历四维属于 token 拼接输入，全部特征联合 Linear 后仅加窗口内正弦位置。

**保留：** 起点已知完整未来路径采用双向注意力；lead 相对原起点计算，不随滚动窗口重置。

**依据：** `models/transformer/GSLAformer.py:280–309`；`layers/gslaformer/temporal.py:74–86,96–118`。

### 4. direct 与内部预测回灌分开

**改动：** direct 仅运行一次历史时间编码，各步使用自身未来条件和 lead，仅调用负荷头，无 latent 头和回灌。

**改动：** 自由递推的预测负荷和 latent 都追加到内部状态序列，下一步取新的 L 长窗口，不指向原始真实历史输入框。显式列出生成位置的 observed/generated/coverage/group_generated。

**说明：** 图中“丢弃最旧位置”指下一步窗口不再读取该位置。代码实际预分配 L+K 序列并滑动切片，不是在内存中逐步删除历史元素。模型负荷输出为标准化坐标，对外结果另行反标准化。

**依据：** `models/transformer/GSLAformer.py:321–394`；`exp/gslaformer_inference.py:125–139`。

### 5. 单独展示两阶段训练及 teacher 隔离

**新增：** A 仅负荷监督；加载 best_A 后冻结完整表示链（含静态链），建立 latent 训练统计；B 保留时间链、未来路径、融合和两头的可训练性。

**新增：** teacher 使用未来实体监督、历史末端 memory 副本和冻结表示链，在 no_grad 下生成 latent 标签；标签只流向辅助损失，没有任何流向生产状态的箭头。显式区分负荷主损失与 latent 辅助损失。

**保留：** B 的 1→min(3,H)→H 课程；缩短监督递推长度不截掉完整 H 的起点已知未来条件，也不截断回灌梯度。

**依据：** `models/transformer/GSLAformer.py:156–182`；`exp/exp_gslaformer_forecasting.py:248–299`；`exp/gslaformer_support.py:106–182`。

## 表达约定与已知边界

- 方框颜色表示模块职责，不表示审查结论；图例在每页底部。
- 箭头不带文字，输入输出及维度写进方框。虚线在图 02 表示状态回灌，在图 03 表示预测 latent 通向辅助损失；没有 teacher 到生产路径的虚线。
- 图 01 用一个逐组共享结构表示重复的 G 组，不枚举所有 AIDC 设备类型。纯静态组与动态组是按组类型选择的路径，不是将同一实体同时走两条路径。
- 图 02 顶部条带表示两条预测路径共用的准备规则，不是额外独立运行的编码分支；各路径执行时均各自准备一次。图 01 的 z、c 分别对应图 02 的历史状态和固定条件。
- 图 03 顶部是阶段顺序；下方是各阶段计算及监督路径。teacher 框概括冻结的实体、空间、关系、读出完整链，不只是一层 GRU。
- 为保持可读性，未展开每层多头注意力、残差、LayerNorm、FFN 和全部消融分支。图中“空间注意力”代表完整 TransformerEncoderLayer，不是仅 MHA。
- 现有图是源码语义重绘，不是 torchview 运行追踪，不提供新的模型可用性或精度结论。

## 生成与验证

从仓库根目录运行：

```bash
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/bot_version/revised_impl/generate.py
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/bot_version/revised_impl/render.py
```

生成器只写本修订目录。`manifest.json` 记录旧图资产及所读源码的哈希；`layout.json` 保留节点、连线和画布尺寸；`validation.json` 保存真实浏览器渲染检查结果。

本机没有 draw.io 桌面程序；浏览器工具又因现有 Chrome 用户配置写锁无法启动，未关闭用户浏览器、未修改配置。导出改用已安装 Chrome 的独立临时 profile，仅结束本脚本创建的 headless 子进程，无需安装任何依赖。PNG/PDF 来自同源布局生成的 SVG，不是 draw.io 桌面原生导出；draw.io 仅验证 XML、节点及连接结构，未声称经过桌面应用渲染。

已完成：

- 三张 SVG / draw.io 生成与 XML 解析；三张 PNG / PDF 真实导出。
- 浏览器实际字体边界检查：无越框文本。
- 正交连线几何检查：无连线穿过其他节点。
- 三张 PNG 逐张目检：未发现文字截断、箭头交叉或错误回灌指向。
- 原目录已有资产 SHA256 不变。

未执行：模型 forward、模型单测、训练或原实验重跑。
