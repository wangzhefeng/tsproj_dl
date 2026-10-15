# 开发审计与 GSLAformer 验证脚本

本目录存放受版本控制的离线生成与验证工具，不包含生产入口。

- `audit_runtime_utils_imports.py`：AST 静态盘点 `run.py/exp/data_provider/models/layers` 中的绝对 utils 导入，不执行模型源码。`--summary` 输出汇总，默认另列文件／行号。它不覆盖动态 import，也不是完整依赖可安装性检查。

```bash
env -u PYTHONPATH .venv/bin/python scripts/dev/audit_runtime_utils_imports.py --summary
```

- `generate_gslaformer_fixture.py`：合成标准包生成器；使用新输出目录，禁止先删除旧结果来重跑。
- `fit_gslaformer_mechanisms.py`：M1/M2/M3 前八个训练 family 的小集拟合；输出明确标记 `small_fit_only`，不能用于正式 test。
- `measure_gslaformer_resources.py`：真实 profile 的有界 A/B 各阶段更新和完整 valid 计时；使用新输出目录，固定 L/H/设备数量，不生成正式训练权重。B 使用有限 A 更新后的表示与采样统计，仅测成本，不冒充 best_A 或完整 Freeze。报告的 epoch 耗时是外推，CPU RSS 为进程累计高水位。

```bash
env -u PYTHONPATH .venv/bin/python scripts/dev/fit_gslaformer_mechanisms.py \
  --fixtures results/gslaformer/fixtures --output results/gslaformer/NEW_SMALL_FIT_RUN
```

阶段状态和验证证据唯一记于 `docs/gslaformer/v1/implementation.md` §12；测试门槛遵循 acceptance.md。
