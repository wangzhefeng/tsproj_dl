# GSLAformer 模型层（在建）

实体历史编码、组内空间交互、类型关系注意力、读出与静态条件链。计算规格唯一见
[docs/gslaformer/v1/implementation.md §1](../../docs/gslaformer/v1/implementation.md#implementation-3-1)；
接口边界对象字段见同文 §7。模型入口 `models/transformer/GSLAformer.py`。

- entity_history.py：属性驱动实体 token 与共享 GRU 历史编码；不使用实体 ID 查表。
- relations.py：类型关系注意力，方向 bias 与加性 mask 的构建和消费。
- spatial.py：组内摘要注意力、分组读出与独立静态条件链。

本目录不属于旧 TFs_type 主线；正式产物进入 results/gslaformer。
