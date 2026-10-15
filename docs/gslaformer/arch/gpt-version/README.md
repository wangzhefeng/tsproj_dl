# GSLAformer 当前架构与原始需求对照

本目录是对 `../model-v4.0.png` 与当前工作区实现的**只读审计快照**，不是新的模型设计权威；现行设计、输入输出、实施及验收规则仍由v1/ 下四份规格维护。不恢复已退役的 `architecture/`。

## 目录约定

- `index.html`：离线可读的架构与差异报告，内嵌原始需求图及当前矢量图。
- `current-architecture.svg/.png/.pdf`：当前 full 模型整体架构。
- `mechanisms.svg/.png/.pdf`：空间／关系／时间模块及双头回灌的局部展开。
- `comparison.svg/.png/.pdf`：原始需求与当前实现的简化对照。
- `comparison.md`、`comparison.json`：逐项差异、证据定位及审计边界。
- `scripts/probe.py`：有界运行探针，复用现有测试与数据包；不训练业务模型、不保存权重、不修改配置。
- `scripts/render.py`：从已核实的对照内容和运行证据生成 SVG／HTML／Markdown；不是自动解析任意模型的绘图器。
- `scripts/export.py`：使用独立 scratch profile 启动本机 Chrome，导出 PNG／矢量 PDF 并检查文本边界、穿框连线和页面排版；不读个人浏览器登录状态。
- `scripts/verify.py`：核对链接、源码哈希、内嵌原图与 SVG 一致性及文件完整性，生成本次交付清单；不把其他会话的图稿子目录纳入本次产物。
- `evidence/`：源文件 SHA-256、运行形状、断言、测试日志与浏览器验证。项目测试计数只在v1/implementation.md §12 记账，此处保存机器回执，不复制进度表。

## 复现

从仓库根目录执行，使用根 `.venv`，不安装绘图依赖：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/probe.py
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/render.py
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/export.py
env -u PYTHONPATH .venv/bin/python docs/gslaformer/arch/gpt-version/scripts/verify.py
```

探针使用既有 AIDC synthetic 标准包，保留正式 day profile 的 L/H 与模型结构；仅以选定训练窗口拟合临时变换并进行有界前向，用于形状核验，不能视为正式训练或精度验证。导出使用本机已安装 Chrome 和当前 `.venv` 中已有的 `websocket-client`，不新增依赖；其他环境需核对这两个可选导出前提。查看自包含 HTML 不依赖它们，也不需要浏览器控制服务。

修改模型后，先重审 `comparison.json` 中的语义和源码定位，再运行探针及绘图脚本；仅重新生成图片不能证明架构仍准确。源哈希变化时旧报告是历史快照，不自动升级为新实现证据。

文档分版后，四份规格位于 `../../v1/`，评审计划位于 `../../v2/`，本目录仍在两者之外的 arch/。迁移只更新引用与路径，不重画图、不重跑旧实验；evidence/ 的回执保留原始字节，其中交付哈希描述迁移前文本，不冒称是迁移后的新回执。
