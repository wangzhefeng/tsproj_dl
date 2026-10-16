# 工程检查与 GSLAformer smoke

`bash scripts/smoke/smoke_engineering_checks.sh` 可从任意工作目录通过脚本路径启动：自动定位仓库、固定使用根 `.venv/bin/python`、清除 PYTHONPATH，运行全仓 unittest 后执行静态 utils 导入审计。任一步失败即非零退出，不安装依赖、不启动完整实验矩阵。

## GSLAformer

四份 JSON 是独立完整配置，不做隐式继承；数据包路径相对配置文件解析。
本目录新增 GSLAformer 链路仍在验收，不代表全部对照矩阵已完成。

已生成 `results/gslaformer/fixtures` 后：

```bash
bash scripts/smoke/smoke_gslaformer.sh NEW_RUN_PREFIX
```

脚本依次运行 CPU base 的完整阶段训练、test 和四个无标签请求。每个 run_id 必须是新名称；禁止覆盖旧结果。

MPS 独立短流程（明确请求设备，不可用即失败）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/smoke/smoke_gslaformer_mps.py --prefix NEW_RUN_PREFIX
```

从正式 base 配置生成 results 下独立 JSON，仅改为 MPS、batch=2、A/B-short/B-medium/B-full 各 1 epoch；执行 test、四种无标签请求及同权重重复重载。此预算只用于工程 smoke，不代替机制拟合或 AIDC 验收。

正式配置显式标记 `experiment.profile`；`evaluation.compare_seed` 必须与 `experiment.seed` 相同。`experiment.preprocess_cache_bytes` 是训练进程共享的样本缓存序列化载荷预算，0 禁用；正式配置为 64 MiB，不是进程总 RSS 上限，实际命中/淘汰与载荷大小见 `preprocess_cache.json`。缓存不落盘、不修改 checkpoint 计算版本，不缓存梯度图。新产物的 `environment.json` 与 checkpoint 记录训练和当前执行源码身份；旧计算版本 2 checkpoint 缺历史身份时明确为 `legacy_missing`。
`run.py --model GSLAformer` 支持 `train/test/forecast/compare`。compare 按当前任务的规定矩阵运行，`no_latent_loss` 自动引用同次 full 的 best_A；不暗中重训独立 A。

机制比较（AIDC 不在此次选择范围，suite 总状态为 partial）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/smoke/compare_gslaformer.py \
  --suite_manifest results/gslaformer/fixtures/mechanisms/suite.json \
  --run_prefix NEW_MECHANISM_COMPARISON --tasks M1 M2 M3 M4 M5
```

模型执行成功、坏输入被拒绝与完整机制／业务验收不是同一结论；报告标明执行范围。完整 AIDC 训练须先确认资源预算。

小集拟合脚本在 `scripts/dev/`；实施进度与证据见 `docs/gslaformer/v1/implementation.md` §12。
