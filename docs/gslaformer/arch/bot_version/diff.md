# GSLAformer：当前实现 vs v4.0 原图（model-v4.0.png）

对照对象：`models/transformer/GSLAformer.py`、`layers/gslaformer/*`；图见 `gslaformer_impl.*`（当前实现）与 `gslaformer_diff.*`（按差异着色：绿=一致，橙=改动，蓝=新增，红虚线=缺失）。

## 一致
| 原图 | 实现 |
|---|---|
| 每组一个空间 Transformer（MHA + Add&Norm + FFN） | `spatial.GroupSpatialEncoder`（nn.TransformerEncoderLayer） |
| 空间向量横向拼接 | `Model._encode_history` 中 `torch.stack(latents, dim=2)` → z |
| 时间顺序纵向拼接 + time PE + Masked MHA ×N | `temporal.TemporalEncoder`（Post-LN 因果 Transformer，上三角 mask） |
| multi Linear → next embedding | `Model.latent_heads`（每动态组 Linear(d_t,d_z)） |
| Linear → 总负荷 | `Model.load_head` |
| next embedding 回灌时间序列 | `Model.rollout`：ẑ、ŷ 写回窗口并右移 H 步 |

## 改动
| 原图 | 实现 | 位置 |
|---|---|---|
| 用户层/机房层/楼宇层 3 组 | AIDC 固定 6 组 compute/it_power/distribution/hvac/space/environment | `docs/gslaformer/v1/input_output.md`、groups_spec |
| 每帧实体表直接进空间编码 | 先 `GroupAdapter` 映射，再逐实体 `GRUCell` 编码因果历史 | `entity_history.EntityHistoryEncoder` |
| 空间 Transformer N× | 单层 + 可学习 summary token，mask 无效实体 | `GroupSpatialEncoder` |
| 组输出向量（未定义池化） | 查询池化 + 空组 token + log(1+N) + coverage → Linear → LN | `spatial.GroupReadout`（消融 `GroupMeanReadout`） |
| 时间 token 仅空间向量 | z ‖ 总负荷 y ‖ observed/generated ‖ coverage/group_generated ‖ 日历 | `Model._window_tokens` |
| time PE | 窗口内正弦位置 + 日历相位 Linear | `temporal.py` |
| “任何时序结构都可适用” | 固定为因果 Transformer | `TemporalEncoder` |
| 两头直接接 Transformer 输出 | 先经 `ConditionFusion` 再接两头 | `temporal.ConditionFusion` |

## 缺失（原图有、实现无）
- **Space Positional Encoding**：有意移除，不用实体 ID/序号编码，保证实体排列不变、新增实体不改变参数（design.md §3、D2）。

## 新增（原图无）
- `relations.RelationAttention`：2 层类型关系注意力，加性 mask + 关系类型×方向×head bias（D5）。
- `spatial.StaticConditionEncoder` + `relation_degrees`：独立静态条件 c。
- `temporal.FuturePathEncoder`：已知未来计划/日历双向编码 p_k；`lead_encoding` lead(k)。
- 总负荷历史显式旁路进入时间 token。
- `Model.predict_direct`：direct 多步对照；`variant` 消融（no_spatial、no_relation_message、no_entity_history、total_only 等）。
- 训练：`set_training_stage` A/B 两阶段（B 冻结表示链、启用 latent 辅助损失）。

> 结论：主干（空间编码 → 横向拼接 → 因果时间 Transformer → 双头 → 回灌）与原图一致；design.md §4 已声明原图只是起点，扩展以 implementation.md 为准。
