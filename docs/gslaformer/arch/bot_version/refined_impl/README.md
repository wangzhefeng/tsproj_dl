# GSLAformer 原图精修版

状态：Done（图稿交付；不是模型验收）。

## 范围

直接读取并复制上级目录的 `gslaformer_impl.drawio`，在副本上修正文字、局部尺寸与连线。不是将三图修订版缩成一张。原图与 `revised_impl/` 保持不变；不修改根治理文档，本目录单独记录本次图稿任务，以遵守用户“不修改已有文件”的要求。

## 文件

- [可编辑 draw.io](gslaformer_impl_refined.drawio)
- [矢量 SVG](gslaformer_impl_refined.svg)
- [高清 PNG](gslaformer_impl_refined.png)
- [单页 PDF](gslaformer_impl_refined.pdf)
- [离线 HTML](gslaformer_impl_refined.html)
- `generate_refined.py`：原图副本修改及同源 SVG 生成。
- `render_refined.py`：隔离 Chrome 导出和边界检查。
- `preserved_files.json`：原图与既有三图版全部文件的哈希基线。
- `layout.json`、`validation.json`：几何结构和真实验证结果。

## 本次修改

### 用户新增要求

1. 左右两个分区外框均为灰色虚线；内部模块维持原配色，不把所有模块都变成虚线。
2. 按用户后续纠正，保留代码英文模块名与中文名，不以中文替换英文。例如 `GroupAdapter · 实体适配`、`GroupReadout · 查询读出`、`TemporalEncoder · 因果时间编码`、`ConditionFusion · 条件融合`；保留 `load_head`、`latent_heads` 等实际成员名，并标注中文职责。
3. 逻辑画布扩大为 3080×2800，节点和间距相对首次精修稿放大 2 倍，字体放大 1.8 倍，不只是提高导出分辨率。补充 `EntityHistoryEncoder` 的适配器/GRU 范围、静态统计项目、关系头偏置及未来路径编码细节。

### 已确认的五项语义修正

- 标题明确为 AIDC 六动态组实例，保留原六组画法；单时刻与全窗口形状分别标注。纯静态组边界放入说明框，不扩展当前实例。
- 补齐查询读出的 Q/K/V、常驻空组 token 与统计量拼接；注明 summary 输出丢弃。静态链补起点输入、逐组投影和拼接后的单 Linear。
- 日历属于 token 特征，全部特征联合投影后只加窗口位置，不再绘制额外日历投影。
- 预测 z 与 y 合流后回到内部滚动状态，而非原始真实负荷输入；明确生成位置的四类标记。direct 在独立说明框中明确只用负荷头、不回灌。
- 底部增加 A→best_A/冻结→B 流程；teacher 只向 B 的辅助损失提供标签，没有进入生产预测状态的箭头。

## 源码对应

- `models/transformer/GSLAformer.py:89–113,156–182,280–394`：动态组、冻结、token、direct、回灌。
- `layers/gslaformer/entity_history.py:19–106`：适配、共享 GRU 和静态组边界。
- `layers/gslaformer/spatial.py:17–52,71–154`：summary、查询读出与静态链。
- `layers/gslaformer/temporal.py:74–86,96–131`：时间投影、完整已知未来路径、融合。
- `exp/gslaformer_support.py:106–182`：teacher 隔离及监督损失。
- `exp/exp_gslaformer_forecasting.py:248–299`：best_A、冻结、课程与损失组合。

## 验证结果与边界

- 原图节点和连线 ID 全部保留；原有 155 个图形节点中，127 个节点除整体 2 倍缩放外保留原几何布局，其余为文字容纳、分区及局部修正而调整。
- 12 项关键代码模块的英文名与中文说明自动核验通过。
- 原图与既有修订版共 33 个文件 SHA256 不变。
- 左右两个外框的 `dashed=1` 已核验。
- 浏览器实际字体测量无文字越框；正交路径检查无连线穿过非容器节点。
- PNG 为 6160×5600；SVG/draw.io 逻辑画布为 3080×2800；PDF 为单页；最终图已目检。
- 因本机无 draw.io 桌面程序，PNG/PDF 使用同一布局的 SVG 通过隔离 Chrome 导出。draw.io 已校验 XML 与连接结构，未声称完成 draw.io 桌面原生渲染。
- 不运行模型 forward、测试、训练或资源探针；不改算法；未提交推送。
- 底部训练条带概括流程，非完整反向传播图；重复组、省略的消融分支和全部数据预处理不在本图展开。图内“窗口保留最近 L 个位置”对应代码预分配序列上的滑动切片，不表示物理删除历史缓存。

## 复现

从仓库根目录执行，依赖现有根 `.venv` 和本机 Chrome，不安装新依赖：

```bash
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/bot_version/refined_impl/generate_refined.py
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/bot_version/refined_impl/render_refined.py
```

脚本只写当前新增目录，浏览器隔离 profile 位于 Hermes scratch；不使用或关闭用户已有 Chrome 会话。
