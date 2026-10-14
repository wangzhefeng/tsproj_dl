# GSLAformer 标准数据管道（在建）

本包归属在建数据流，不替代 TFs_type。契约见 docs/gslaformer/input_output.md。

- schema.py：严格 JSON、规范内容指纹与输入质量计算。
- config.py：JSON 配置与合法值检查，不允许隐式 CLI 参数覆盖。
- dataset.py：校验标准包、as-of 版本、时间窗口及输入／监督隔离。
- aidc_mapping.py：单位／测点／端口映射与类型关系合法性。

文件与符号只在实现并经测试后视为可用；正式产物进入 results/gslaformer。

## 在建边界与验证

- 尚未实现 `scaling.py`（训练统计与词表）、`batching.py`（变长 batch）及模拟生成器；I1 未完成。
- 未接入模型、训练流程或主入口；本目录的标准包无标签读取测试不等于模型 forecast 验收。
- 本地执行以下三个测试模块，共 16 项通过；覆盖配置、输入质量、标准包读取、时间可见性、监督隔离及测点／关系校验。

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_data.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_package.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_mapping.py' -v
```

整组检查命令：

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py' -v
```

当前整组结果为 **16 项通过、1 项测试模块导入错误**：`test_gslaformer_scaling.py` 因缺少 `data_provider.spatial_temporal.scaling` 无法导入；它还依赖未实现的 `batching`。保留该失败测试，不以缩小检查范围宣称整组通过。用户已确认按此在建状态提交推送。

`tests/` 仍被现有 `.gitignore` 忽略，以上测试仅在本地保留、未随代码提交；远端检出不包含这些验证文件。未运行完整模拟验收、CPU/MPS 模型验证或旧入口回归。
