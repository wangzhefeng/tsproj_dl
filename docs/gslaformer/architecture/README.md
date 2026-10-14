# GSLAformer 架构图资料

本目录集中保存模型架构图、原始参考图、图源、说明和绘图验证证据；不存模型训练结果。模型设计和实施合同位于上一级目录，本次整理不改变其语义。

## 阅读入口

- [架构图与数据流说明](architecture.md)
- [交互架构图](architecture.html)
- [静态架构图](architecture.png)
- [原始 v4.0 架构图](v4.0.png)
- [可编辑图源](architecture.json)
- [返回模型文档目录](../README.md)

## 文件与证据约定

- `architecture.json` 是图源，`meta.output` 从仓库根目录解析；修改图源后重新生成 HTML，不手改生成文件。
- `architecture.delivery.json` 是当前路径的交付回执，生成工具维护。
- `diagram_evidence/` 保留迁移前的回执和截图，文件内容不篡改；其中绝对路径为当时运行位置，不代表当前路径有效。
- `diagram_evidence/relocation/` 保存本次新路径的自动验证回执。
- 原始 `v4.0.png` 与静态 `architecture.png` 只移动，不重画、不改像素。
- [R4 验证报告](../archive/validation_r4.md)与[机器回执](../archive/validation_checks.json)是整个 R4 文档任务的历史证据，保存在上级 archive/，不混入纯绘图资产。

## 新路径验证命令

在仓库根目录，使用已安装 Archify 的绝对 CLI 路径替换下方 `<archify-cli>`：

```bash
node <archify-cli> finalize architecture docs/gslaformer/architecture/architecture.json docs/gslaformer/architecture/architecture.html --quality showcase --out-dir docs/gslaformer/architecture/diagram_evidence/relocation --json
```

当前归档记录与完整性核验见 `migration.json`；它只记录资料迁移，不是模型验收。
