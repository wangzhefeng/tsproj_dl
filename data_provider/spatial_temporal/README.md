# GSLAformer 标准数据管道（在建）

本包归属在建数据流，不替代 TFs_type。契约见 docs/gslaformer/v1/input_output.md。

- schema.py：严格 JSON、规范内容指纹与输入质量计算。
- config.py：JSON 配置与合法值检查，不允许隐式 CLI 参数覆盖。
- dataset.py：校验标准包、as-of 版本、时间窗口及输入／监督隔离。
- aidc_mapping.py：单位／测点／端口映射与类型关系合法性。
- scaling.py：组级训练统计、类别词表、标准化与 artifact 校验；fit 只接受 train split。
- batching.py：变长实体轴 padding、逐样本关系边表与元数据组装，监督视图独立。

文件与符号只在实现并经测试后视为可用；正式产物进入 results/gslaformer。

## 在建边界与验证

在建状态、测试计数与证据唯一记账于 [实施规格工作包](../../docs/gslaformer/v1/implementation.md#work-packages)，本 README 只保留运行命令，不单独记录通过数。

- base／M1–M5／AIDC 生成器已有实现，见 `scripts/dev/generate_gslaformer_fixture.py`。
- GSLAformer 已接入独立 train/test/forecast CLI，部分合成 smoke 已运行；完整模拟与对照矩阵仍未完成。本目录的读取测试不替代模型验收。

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_data.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_package.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_mapping.py' -v
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_scaling.py' -v
```

整组检查命令：

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p 'test_gslaformer_*.py' -v
```

`tests/` 已纳入版本控制；本会话新增实现尚未提交。当前 CPU/MPS smoke、旧入口回归与完整矩阵的具体边界以实施规格 §12 为准。
