# GSLAformer 标准数据管道（在建）

本包归属在建数据流，不替代 TFs_type。契约见 docs/gslaformer/input_output.md。

- schema.py：严格 JSON、规范内容指纹与输入质量计算。
- config.py：JSON 配置与合法值检查，不允许隐式 CLI 参数覆盖。
- dataset.py：校验标准包、as-of 版本、时间窗口及输入／监督隔离。
- aidc_mapping.py：单位／测点／端口映射与类型关系合法性。

文件与符号只在实现并经测试后视为可用；正式产物进入 results/gslaformer。

## 在建边界与验证

在建状态、测试计数与证据唯一记账于 [实施规格工作包](../../docs/gslaformer/implementation.md#work-packages)，本 README 只保留运行命令，不单独记录通过数。

- 尚未实现 `scaling.py`（训练统计与词表）、`batching.py`（变长 batch，规则见 [batch 组装](../../docs/gslaformer/input_output.md#batch-contract)）及模拟生成器；I1 未完成。
- 未接入模型、训练流程或主入口；本目录的标准包无标签读取测试不等于模型 forecast 验收。

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_data.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_package.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_mapping.py' -v
```

整组检查命令：

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py' -v
```

`tests/` 已纳入版本控制（含预期红灯的 `test_gslaformer_scaling.py`），远端检出可直接复现上述验证。未运行完整模拟验收、CPU/MPS 模型验证或旧入口回归。
