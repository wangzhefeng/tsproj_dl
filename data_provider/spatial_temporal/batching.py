"""变长实体轴的 batch 组装：组内 padding、逐样本边表与元数据（input_output.md §6.3）。

规则要点：
- 每组 N_g 补齐到该组在本 batch 的最大实体数，不产生跨样本共享实体轴；
- 补齐行 padding=true，数值填 0、全部 mask 置 false、类别填 PAD(0)；
- 关系边表保持每样本独立，端点重编号到 batch 统一拼接轴（按 schema 组顺序的
  N_g_max 累积偏移），不跨样本拼接成统一图；
- entity_ids/site_ids/target_contract 逐样本作为元数据传递，不进入张量；
- 监督对象与输入视图分离：修改 supervision 不得写回 inputs。
"""
import numpy as np
import torch
from .dataset import check_attention_budget


# ------------------------------------------------------------------------------
# batch 容器
# ------------------------------------------------------------------------------

class BatchInputs:
    """模型输入视图；不含任何监督字段。"""

    def __init__(self, groups, relations_history, relations_origin, target_history,
                 target_history_observed, history_time, future_time, future_known,
                 future_known_mask, origins, site_ids, target_contracts, metadata):
        self.groups = groups
        self.relations_history = relations_history
        self.relations_origin = relations_origin
        self.target_history = target_history
        self.target_history_observed = target_history_observed
        self.history_time = history_time
        self.future_time = future_time
        self.future_known = future_known
        self.future_known_mask = future_known_mask
        self.origins = origins
        self.site_ids = site_ids
        self.target_contracts = target_contracts
        self.metadata = metadata


class BatchSupervision:
    """独立标签视图；与 inputs 无共享存储。"""

    def __init__(self, target, target_mask, future_groups):
        self.target = target
        self.target_mask = target_mask
        self.future_groups = future_groups


class Batch:
    """collate 输出；to(device) 原地迁移张量并返回自身。"""

    def __init__(self, inputs, supervision):
        self.inputs = inputs
        self.supervision = supervision

    def to(self, device):
        def move(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        self.inputs.groups = {name: move(tensors) for name, tensors in self.inputs.groups.items()}
        for field in ("target_history", "target_history_observed", "history_time", "future_time",
                      "future_known", "future_known_mask", "origins"):
            setattr(self.inputs, field, move(getattr(self.inputs, field)))
        if self.supervision is not None:
            self.supervision.target = move(self.supervision.target)
            self.supervision.target_mask = move(self.supervision.target_mask)
            self.supervision.future_groups = {
                name: move(tensors) for name, tensors in self.supervision.future_groups.items()}
        return self


# ------------------------------------------------------------------------------
# 组装
# ------------------------------------------------------------------------------

def collate_samples(samples, config):
    """把 apply 后的样本列表组装成单 batch；样本间实体数可不同。"""
    if not samples:
        raise ValueError("collate 需要至少一个样本")
    data = config["data"]
    length, horizon = data["seq_len"], data["pred_len"]
    groups_spec = data["groups"]
    group_names = [group["name"] for group in groups_spec]
    for sample in samples:
        groups_block = sample["inputs"]["groups"]
        for name in group_names:
            values = groups_block[name]["values"]
            if values.shape[0] != length:
                raise ValueError(f"collate: {name}.values 历史长度与 config.seq_len 不一致")
        if sample["inputs"]["future_time"].shape[0] != horizon:
            raise ValueError("collate: future_time 长度与 config.pred_len 不一致")
    batch = len(samples)
    supervised = [sample["supervision"] is not None for sample in samples]
    if any(supervised) and not all(supervised):
        raise ValueError("同 batch 混用监督与无标签样本")
    known = len(data["known_future"])
    # ------------------------------------------------------- 组内张量 ----
    sizes = {name: max(len(sample["inputs"]["groups"][name]["entity_ids"]) for sample in samples)
             for name in group_names}
    offsets, total = {}, 0
    for name in group_names:
        offsets[name] = total
        total += sizes[name]
    if total > config["limits"]["max_total_nodes"]:
        raise ValueError(f"padding 后节点总数 {total} 超过 max_total_nodes")
    check_attention_budget(config, list(sizes.values()), batch_size=batch)
    type_index = {relation["name"]: index for index, relation in enumerate(data["relation_types"])}
    group_tensors = {}
    for spec in groups_spec:
        name = spec["name"]
        count = sizes[name]
        features = len(spec["dynamic"])
        numeric = len(spec["static_numeric"])
        categories = len(spec["static_category"])
        values = np.zeros((batch, length, count, features), np.float32)
        observed = np.zeros((batch, length, count, features), bool)
        present = np.zeros((batch, length, count), bool)
        padding = np.zeros((batch, count), bool)
        applicable = np.zeros((batch, count, features), bool)
        static_numeric = np.zeros((batch, length, count, numeric), np.float32)
        static_mask = np.zeros((batch, length, count, numeric), bool)
        static_category = np.zeros((batch, length, count, categories), np.int64)  # 补齐行 PAD=0
        entity_ids = []
        for index, sample in enumerate(samples):
            source = sample["inputs"]["groups"][name]
            nodes = len(source["entity_ids"])
            if nodes == 0:
                entity_ids.append([])
                continue
            entity_ids.append(list(source["entity_ids"]))
            values[index, :, :nodes] = source["values"]
            observed[index, :, :nodes] = source["observed"]
            present[index, :, :nodes] = source["present"]
            applicable[index, :nodes] = source["applicable"]
            static_numeric[index, :, :nodes] = source["static_numeric_history"]
            static_mask[index, :, :nodes] = source["static_numeric_mask"]
            static_category[index, :, :nodes] = source["static_category_history"]
            padding[index, nodes:] = True
        group_tensors[name] = {
            "values": torch.from_numpy(values),
            "observed": torch.from_numpy(observed),
            "present": torch.from_numpy(present),
            "padding": torch.from_numpy(padding),
            "applicable": torch.from_numpy(applicable),
            "static_numeric_history": torch.from_numpy(static_numeric),
            "static_numeric_mask": torch.from_numpy(static_mask),
            "static_category_history": torch.from_numpy(static_category),
            "entity_ids": entity_ids,
        }
    # --------------------------------------------------------- 关系边 ----
    def reindex(edges):
        mapped = []
        for edge in edges:
            src_group, src_node, dst_group, dst_node, kind = edge
            mapped.append((offsets[src_group] + src_node, offsets[dst_group] + dst_node, type_index[kind]))
        return mapped

    relations_history = [[reindex(step) for step in sample["inputs"]["relations_history"]]
                         for sample in samples]
    relations_origin = [reindex(sample["inputs"]["relations_origin"]) for sample in samples]
    # ------------------------------------------------------- 标量视图 ----
    target_history = np.stack([sample["inputs"]["target_history"] for sample in samples])
    target_observed = np.stack([sample["inputs"]["target_history_observed"] for sample in samples])
    history_time = np.stack([sample["inputs"]["history_time"] for sample in samples])
    future_time = np.stack([sample["inputs"]["future_time"] for sample in samples])
    future_known = np.stack([sample["inputs"]["future_known"] for sample in samples])
    future_mask = np.stack([sample["inputs"]["future_known_mask"] for sample in samples])
    origins = np.array([sample["inputs"]["origin"] for sample in samples], np.int64)
    inputs = BatchInputs(
        groups=group_tensors,
        relations_history=relations_history,
        relations_origin=relations_origin,
        target_history=torch.from_numpy(target_history),
        target_history_observed=torch.from_numpy(target_observed),
        history_time=torch.from_numpy(history_time),
        future_time=torch.from_numpy(future_time),
        future_known=torch.from_numpy(future_known),
        future_known_mask=torch.from_numpy(future_mask),
        origins=torch.from_numpy(origins),
        site_ids=[sample["inputs"]["site_id"] for sample in samples],
        target_contracts=[sample["inputs"]["target_contract"] for sample in samples],
        metadata=[sample["metadata"] for sample in samples],
    )
    supervision = None
    if all(supervised):
        target = np.stack([sample["supervision"]["target"] for sample in samples])
        target_mask = np.stack([sample["supervision"]["target_mask"] for sample in samples])
        future_groups = {}
        for spec in groups_spec:
            name = spec["name"]
            count = sizes[name]
            features = len(spec["dynamic"])
            values = np.zeros((batch, horizon, count, features), np.float32)
            observed = np.zeros((batch, horizon, count, features), bool)
            for index, sample in enumerate(samples):
                source = sample["supervision"]["future_groups"].get(name)
                if source is None or not features:
                    continue
                nodes = len(sample["inputs"]["groups"][name]["entity_ids"])
                if source["values"].shape != (horizon, nodes, features) or source["observed"].shape != source["values"].shape:
                    raise ValueError(f"collate: {name} future_groups 轴与 H/实体/特征不一致")
                values[index, :, :nodes] = source["values"][:, :nodes]
                observed[index, :, :nodes] = source["observed"][:, :nodes]
            future_groups[name] = {"values": torch.from_numpy(values), "observed": torch.from_numpy(observed)}
        supervision = BatchSupervision(
            target=torch.from_numpy(target),
            target_mask=torch.from_numpy(target_mask),
            future_groups=future_groups,
        )
    return Batch(inputs, supervision)
