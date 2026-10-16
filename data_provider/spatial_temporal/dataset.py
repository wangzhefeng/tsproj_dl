"""标准包读取与起点快照。原始快照先保留类别文本，scaling/collate 后才送模型。"""
from collections import defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from .config import keys, names, integer, safe_name, schema_from_config, validate_config, validate_schema
from .schema import canonical_hash, read_json


FILES = {"entities.json", "relations.json", "site.json", "measurements.json", "observations.npz", "known_future.npz"}


def contained_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("包文件必须是相对路径")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"包路径越界: {relative}")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_headers(path):
    """分配数组前读取 NPY header；禁止对象／压缩炸弹的伪造 shape。"""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy") or "/" in info.filename or info.filename[:-4] in headers:
                raise ValueError("NPZ: 非法或重复成员")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("NPZ: 不支持的 NPY 版本")
                if dtype.hasobject or any(type(s) is not int or s < 0 for s in shape):
                    raise ValueError("NPZ: 禁止对象 dtype 或非法维度")
                count = 1
                for size in shape:
                    count *= size
                if count * dtype.itemsize != info.file_size - stream.tell():
                    raise ValueError("NPZ: header shape 与文件字节数不符")
                headers[info.filename[:-4]] = (shape, dtype)
    return headers


def load_arrays(path, expected):
    headers = npz_headers(path)
    keys(headers, expected, name=Path(path).name)
    for key, (shape, dtype) in expected.items():
        if headers[key] != (shape, np.dtype(dtype)):
            raise ValueError(f"{Path(path).name}.{key}: 期望 {shape}/{dtype}, 实际 {headers[key]}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in expected}


def version_fields(record, label):
    for field in ("valid_from", "available_at", "revision"):
        integer(record[field], f"{label}.{field}", 0 if field == "revision" else -(2 ** 63))
    end = record["valid_to"]
    if end is not None:
        integer(end, f"{label}.valid_to", -(2 ** 63))
        if end <= record["valid_from"]:
            raise ValueError(f"{label}: 空有效期")


def choose_version(records, time, origin):
    candidates = [r for r in records if r["available_at"] <= origin and r["valid_from"] <= time
                  and (r["valid_to"] is None or time < r["valid_to"])]
    if not candidates:
        return None
    revision = max(r["revision"] for r in candidates)
    selected = [r for r in candidates if r["revision"] == revision]
    if len(selected) != 1:
        raise ValueError("可见版本冲突: 同一有效时刻/revision 存在多个记录")
    return selected[0]


def check_attention_budget(config, node_counts, *, batch_size=1):
    """只估显式 FP32 score 张量，不冒充真实峰值内存。"""
    total = sum(node_counts)
    limits, data, model = config["limits"], config["data"], config["model"]
    if total > limits["max_total_nodes"]:
        raise ValueError(f"节点数 {total} 超过 max_total_nodes")
    length, horizon = data["seq_len"], data["pred_len"]
    estimates = {"relation": 4 * batch_size * length * model["relation_heads"] * total ** 2,
                 "temporal": 4 * batch_size * model["temporal_heads"] * length ** 2,
                 "future": 4 * batch_size * model["temporal_heads"] * horizon ** 2}
    for index, size in enumerate(node_counts):
        estimates[f"spatial_{index}"] = 4 * batch_size * length * model["spatial_heads"] * (size + 1) ** 2
    if any(value > limits["max_explicit_attention_bytes"] for value in estimates.values()):
        raise ValueError(f"attention 分配超限（单 score 张量估算字节）: {estimates}")
    return estimates


class EpisodePackage:
    """校验完整 single_episode；实体 ID 仅用于对齐，不是学习特征。"""

    def __init__(self, path, config):
        self.path = Path(path).resolve()
        self.config = validate_config(deepcopy(config))
        self.manifest = read_json(self.path / "manifest.json")
        manifest = self.manifest
        keys(manifest, {"schema_version", "data_kind", "layout", "schema", "entity_order", "files"},
             {"purpose", "generator", "scenario", "family_id"}, name="single_episode manifest")
        if manifest["schema_version"] != 3 or manifest["data_kind"] != "synthetic" or manifest["layout"] != "single_episode":
            raise ValueError("single_episode: schema_version/data_kind/layout 不兼容")
        if manifest.get("purpose", "supervised") not in {"supervised", "forecast"}:
            raise ValueError("purpose: 未知用途")
        self.schema = validate_schema(manifest["schema"])
        if self.schema != schema_from_config(config):
            raise ValueError("数据包 schema 与配置不兼容")
        self.schema_fingerprint = canonical_hash(self.schema)
        keys(manifest["files"], FILES, name="manifest.files")
        for filename, expected in manifest["files"].items():
            path = contained_path(self.path, filename)
            if not isinstance(expected, str) or file_hash(path) != expected:
                raise ValueError(f"SHA-256 不匹配: {filename}")
        self.order = manifest["entity_order"]
        keys(self.order, [g["name"] for g in self.schema["groups"]], name="entity_order")
        for group, ids in self.order.items():
            names(ids, f"entity_order.{group}")
        check_attention_budget(config, [len(ids) for ids in self.order.values()])
        self.entities = read_json(contained_path(self.path, "entities.json"))
        self.relations = read_json(contained_path(self.path, "relations.json"))
        self.site = read_json(contained_path(self.path, "site.json"))
        self.measurements = read_json(contained_path(self.path, "measurements.json"))
        self._validate_metadata()
        self.order_index = {g: {eid: i for i, eid in enumerate(ids)} for g, ids in self.order.items()}
        records = [r for group in self.entity_versions.values() for versions in group.values() for r in versions]
        records += [r for versions in self.edge_versions.values() for r in versions]
        self.context_boundaries = sorted({r[k] for r in records for k in ("valid_from", "valid_to") if r[k] is not None})
        self.observations = self._load_observations()
        self.times = self.observations["timestamps"]
        self.known = self._load_known()
        self.fingerprint = canonical_hash({"schema": self.schema_fingerprint, "entity_order": self.order,
                                           "files": manifest["files"], "purpose": manifest.get("purpose", "supervised")})

    def _validate_metadata(self):
        keys(self.site, {"site_id", "target_contract"}, name="site")
        safe_name(self.site["site_id"], "site_id")
        contract = self.site["target_contract"]
        fields = {"target_id", "quantity", "boundary_id", "unit", "sign_convention", "interval_label",
                  "aggregation", "meter_ids", "coverage_description"}
        keys(contract, fields, name="target_contract")
        for field in fields - {"meter_ids"}:
            if not isinstance(contract[field], str) or not contract[field]:
                raise ValueError(f"target_contract.{field}: 非空字符串必填")
        names(contract["meter_ids"], "meter_ids", nonempty=True)
        if contract["unit"] != self.schema["target_unit"]:
            raise ValueError("目标单位不一致")
        for schema_key, contract_key in (("target_quantity", "quantity"), ("interval_label", "interval_label"),
                                         ("target_aggregation", "aggregation")):
            if schema_key in self.schema and self.schema[schema_key] != contract[contract_key]:
                raise ValueError(f"target_contract.{contract_key}: 与 schema 不相容")
        keys(self.entities, {"groups"}, name="entities")
        keys(self.entities["groups"], self.order, name="entities.groups")
        self.entity_versions = {}
        for group in self.schema["groups"]:
            name = group["name"]
            records = self.entities["groups"][name]
            if not isinstance(records, list):
                raise ValueError("entities: 需要版本数组")
            versions = defaultdict(list)
            seen = set()
            for record in records:
                keys(record, {"id", "valid_from", "valid_to", "available_at", "revision", "numeric", "category"},
                     {"asset_id"}, name="entity")
                if record["id"] not in self.order[name]:
                    raise ValueError("entity_order 与注册 ID 不一致")
                version_fields(record, "entity")
                key = (record["id"], record["valid_from"], record["revision"])
                if key in seen:
                    raise ValueError("重复实体版本键")
                seen.add(key)
                keys(record["numeric"], group["static_numeric"], name="numeric")
                keys(record["category"], group["static_category"], name="category")
                for value in record["numeric"].values():
                    if value is not None and (type(value) not in (int, float) or not np.isfinite(value)):
                        raise ValueError("numeric: 非有限数值")
                for value in record["category"].values():
                    if value is not None and (not isinstance(value, str) or not value):
                        raise ValueError("category: 需要字符串或 null")
                versions[record["id"]].append(record)
            if set(versions) != set(self.order[name]):
                raise ValueError("entity_order 与注册 ID 集不一致")
            self.entity_versions[name] = dict(versions)
        keys(self.relations, {"edges"}, name="relations")
        if not isinstance(self.relations["edges"], list):
            raise ValueError("relations.edges: 需要数组")
        self.relation_types = {r["name"]: r for r in self.schema["relation_types"]}
        self.edge_versions = defaultdict(list)
        seen = set()
        for record in self.relations["edges"]:
            fields = {"src_group", "src_id", "dst_group", "dst_id", "type", "valid_from", "valid_to", "available_at", "revision"}
            keys(record, fields, name="edge")
            version_fields(record, "edge")
            kind = self.relation_types.get(record["type"])
            if kind is None:
                raise ValueError("未声明关系类型")
            for side in ("src", "dst"):
                group, entity = record[f"{side}_group"], record[f"{side}_id"]
                if group not in kind[f"{side}_groups"] or entity not in self.order.get(group, []):
                    raise ValueError("关系悬空／跨站／端点类型不合法")
            endpoints = (record["src_group"], record["src_id"], record["dst_group"], record["dst_id"])
            if endpoints[:2] == endpoints[2:]:
                raise ValueError("源记录不允许 self 边")
            logical = endpoints + (record["type"],)
            version = logical + (record["valid_from"], record["revision"])
            if version in seen:
                raise ValueError("重复关系版本键")
            seen.add(version)
            self.edge_versions[logical].append(record)
        # 计量校验在来源边界运行；标准 observations 已转换，不能重复换算。
        from .aidc_mapping import validate_mappings
        self.mapping_audit = validate_mappings(self.measurements, self.schema, self.order)

    def _load_observations(self):
        path = contained_path(self.path, "observations.npz")
        headers = npz_headers(path)
        time_header = headers.get("timestamps")
        if time_header is None or len(time_header[0]) != 1 or time_header[1] != np.dtype("int64"):
            raise ValueError("timestamps: 需要 [T] int64")
        length = time_header[0][0]
        if length < self.config["data"]["seq_len"]:
            raise ValueError("不足 L 个规则时间位置")
        expected = {"timestamps": ((length,), "int64"), "target": ((length, 1), "float32"),
                    "target_observed": ((length, 1), "bool"), "target_available_at": ((length, 1), "int64")}
        for group in self.schema["groups"]:
            name, features = group["name"], len(group["dynamic"])
            nodes = len(self.order[name])
            expected.update({f"{name}__values": ((length, nodes, features), "float32"),
                             f"{name}__observed": ((length, nodes, features), "bool"),
                             f"{name}__available_at": ((length, nodes, features), "int64"),
                             f"{name}__applicable": ((nodes, features), "bool"),
                             f"{name}__present": ((length, nodes), "bool")})
        arrays = load_arrays(path, expected)
        if np.any(np.diff(arrays["timestamps"]) != self.schema["freq_seconds"]):
            raise ValueError("timestamps: 非严格规则时间轴")
        self._finite_observed(arrays["target"], arrays["target_observed"], "target")
        for group in self.schema["groups"]:
            name = group["name"]
            self._finite_observed(arrays[f"{name}__values"], arrays[f"{name}__observed"], name)
            eligible = arrays[f"{name}__present"][..., None] & arrays[f"{name}__applicable"][None]
            if np.any(arrays[f"{name}__observed"] & ~eligible):
                raise ValueError(f"{name}: observed 不能位于不适用／未生效单元")
        return arrays

    @staticmethod
    def _finite_observed(values, mask, name):
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{name}: 有效观测必须有限")

    def _load_known(self):
        path = contained_path(self.path, "known_future.npz")
        headers = npz_headers(path)
        header = headers.get("origin")
        if header is None or len(header[0]) != 1:
            raise ValueError("known_future.origin: 需要 [R]")
        shape = header[0]
        expected = {key: (shape, "int64") for key in ("origin", "target_time", "feature_index", "published_at", "revision")}
        expected.update(value=(shape, "float32"), observed=(shape, "bool"))
        arrays = load_arrays(path, expected)
        count = len(self.schema["known_future"])
        if np.any((arrays["feature_index"] < 0) | (arrays["feature_index"] >= count)) or np.any(arrays["revision"] < 0):
            raise ValueError("known_future: 非法 feature_index/revision")
        self._finite_observed(arrays["value"], arrays["observed"], "known_future")
        delta = arrays["target_time"] - arrays["origin"]
        if np.any(delta <= 0) or np.any(delta % self.schema["freq_seconds"]):
            raise ValueError("known_future: target_time 必须位于原点后的规则网格")
        self.known_by_origin = defaultdict(list)
        seen = set()
        for index in range(shape[0]):
            key = tuple(int(arrays[k][index]) for k in ("origin", "target_time", "feature_index", "published_at", "revision"))
            if key in seen:
                raise ValueError("known_future: 重复／冲突版本键")
            seen.add(key)
            self.known_by_origin[key[0]].append(index)
        return arrays

    def select_known(self, origin, future_time):
        values = np.zeros((len(future_time), len(self.schema["known_future"])), np.float32)
        mask = np.zeros_like(values, dtype=bool)
        time_map = {int(t): i for i, t in enumerate(future_time)}
        selected = {}
        for i in self.known_by_origin.get(int(origin), []):
            a = self.known
            target = int(a["target_time"][i])
            if a["published_at"][i] > origin or target not in time_map:
                continue
            key = (time_map[target], int(a["feature_index"][i]))
            rank = (int(a["published_at"][i]), int(a["revision"][i]))
            if key not in selected or rank > selected[key][0]:
                selected[key] = (rank, i)
        for key, (_, i) in selected.items():
            mask[key] = self.known["observed"][i]
            if mask[key]:
                values[key] = self.known["value"][i]
        return values, mask

    def context_at(self, time, origin):
        entities = {group: {entity: choose_version(records, time, origin) for entity, records in versions.items()}
                    for group, versions in self.entity_versions.items()}
        edges = []
        for records in self.edge_versions.values():
            record = choose_version(records, time, origin)
            if record is None:
                continue
            src, dst = (record["src_group"], record["src_id"]), (record["dst_group"], record["dst_id"])
            if entities[src[0]][src[1]] is not None and entities[dst[0]][dst[1]] is not None:
                edges.append((src[0], self.order_index[src[0]][src[1]], dst[0], self.order_index[dst[0]][dst[1]], record["type"]))
        from .aidc_mapping import validate_relation_snapshot
        validate_relation_snapshot(edges, entities, self.order, self.relation_types)
        return entities, edges

    @staticmethod
    def _context_semantics(context):
        entities, edges = context
        values = {g: {eid: None if r is None else (r["numeric"], r["category"]) for eid, r in records.items()}
                  for g, records in entities.items()}
        return canonical_hash({"entities": values, "edges": sorted(edges)})

    def _assert_fixed_future(self, origin, end, context):
        # 检查所有起点已知变更边界，包括非采样网格上的事件。
        boundaries = set()
        records = [r for group in self.entity_versions.values() for versions in group.values() for r in versions]
        records += [r for versions in self.edge_versions.values() for r in versions]
        for record in records:
            if record["available_at"] <= origin:
                for key in ("valid_from", "valid_to"):
                    t = record[key]
                    if t is not None and origin < t <= end:
                        boundaries.add(t)
        fingerprint = self._context_semantics(context)
        for time in sorted(boundaries):
            if self._context_semantics(self.context_at(time, origin)) != fingerprint:
                raise ValueError("未来已知实体／属性／关系变化不受首版支持")

    def sample(self, origin_index, *, supervised=True):
        data = self.config["data"]
        length, horizon = data["seq_len"], data["pred_len"]
        integer(origin_index, "origin_index", 0)
        if self.manifest.get("purpose") == "forecast" and (supervised or origin_index != len(self.times) - 1):
            raise ValueError("forecast 包只能读取最后 origin 的无标签请求")
        if origin_index < length - 1 or origin_index >= len(self.times) or (supervised and origin_index + horizon >= len(self.times)):
            raise ValueError("请求窗口越界")
        indices = np.arange(origin_index - length + 1, origin_index + 1)
        origin = int(self.times[origin_index])
        future_time = origin + np.arange(1, horizon + 1, dtype=np.int64) * self.schema["freq_seconds"]
        # 仅在一个请求内复用同版本区间；origin 固定，不跨可见性边界共享。
        from bisect import bisect_right
        context_cache, contexts = {}, []
        for time in self.times[indices]:
            key = (origin, bisect_right(self.context_boundaries, int(time)))
            if key not in context_cache:
                context_cache[key] = self.context_at(int(time), origin)
            contexts.append(context_cache[key])
        self._assert_fixed_future(origin, int(future_time[-1]), contexts[-1])
        arrays, groups, future_groups = self.observations, {}, {}
        for group in self.schema["groups"]:
            name, ids = group["name"], self.order[group["name"]]
            static = np.zeros((length, len(ids), len(group["static_numeric"])), np.float32)
            static_mask = np.zeros_like(static, dtype=bool)
            category = np.empty((length, len(ids), len(group["static_category"])), dtype=object)
            category.fill(None)
            active = np.zeros((length, len(ids)), bool)
            for t, (entities, _) in enumerate(contexts):
                for n, entity in enumerate(ids):
                    record = entities[name][entity]
                    if record is None:
                        continue
                    active[t, n] = True
                    for j, feature in enumerate(group["static_numeric"]):
                        value = record["numeric"][feature]
                        if value is not None:
                            static[t, n, j], static_mask[t, n, j] = value, True
                    for j, feature in enumerate(group["static_category"]):
                        category[t, n, j] = record["category"][feature]
            present = arrays[f"{name}__present"][indices] & active
            applicable = arrays[f"{name}__applicable"]
            observed = (arrays[f"{name}__observed"][indices] & (arrays[f"{name}__available_at"][indices] <= origin)
                        & present[..., None] & applicable[None])
            groups[name] = {"values": np.where(observed, arrays[f"{name}__values"][indices], 0),
                            "observed": observed, "present": present, "applicable": applicable.copy(),
                            "padding": np.zeros(len(ids), bool), "static_numeric_history": static,
                            "static_numeric_mask": static_mask, "static_category_history": category,
                            "entity_ids": list(ids)}
            if supervised:
                future = slice(origin_index + 1, origin_index + horizon + 1)
                mask = (arrays[f"{name}__observed"][future] & present[-1, :, None] & applicable[None])
                future_groups[name] = {"values": np.where(mask, arrays[f"{name}__values"][future], 0), "observed": mask}
        # 包中 present 可表达投运状态，消息必须与最终有效集合一致。
        relations = []
        for t, (_, edges) in enumerate(contexts):
            relations.append([e for e in edges if groups[e[0]]["present"][t, e[1]] and groups[e[2]]["present"][t, e[3]]])
        if self.config["experiment"]["variant"] == "full" and not relations[-1]:
            raise ValueError("full 起点至少需要一条有效关系")
        target_mask = arrays["target_observed"][indices] & (arrays["target_available_at"][indices] <= origin)
        known, known_mask = self.select_known(origin, future_time)
        if not target_mask.any() and not any(g["observed"].any() for g in groups.values()):
            raise ValueError("没有历史总表或内部有效动态观测")
        if not any(g["present"].any() and g["values"].shape[-1] for g in groups.values()):
            raise ValueError("全部动态组为空")
        inputs = {"groups": groups, "relations_history": relations, "relations_origin": relations[-1],
                  "target_history": np.where(target_mask, arrays["target"][indices], 0),
                  "target_history_observed": target_mask, "history_time": self.times[indices].copy(),
                  "future_time": future_time, "future_known": known, "future_known_mask": known_mask,
                  "origin": origin, "site_id": self.site["site_id"], "target_contract": deepcopy(self.site["target_contract"])}
        supervision = None
        if supervised:
            future = slice(origin_index + 1, origin_index + horizon + 1)
            mask = arrays["target_observed"][future].copy()
            supervision = {"target": np.where(mask, arrays["target"][future], 0), "target_mask": mask, "future_groups": future_groups}
        return {"inputs": inputs, "supervision": supervision,
                "metadata": {"data_fingerprint": self.fingerprint, "schema_fingerprint": self.schema_fingerprint,
                             "episode_path": str(self.path), "origin_index": origin_index,
                             "scenario": self.manifest.get("scenario", "default")}}


class PackageDataset:
    """按固定分区构建 (episode, origin)；不跨 episode/站点切窗。"""

    def __init__(self, config, split):
        self.config = validate_config(deepcopy(config))
        if split not in {"train", "valid", "test", "forecast"}:
            raise ValueError("split: 仅 train/valid/test/forecast")
        self.split = split
        root = Path(config["data"]["package"]).resolve()
        manifest = read_json(root / "manifest.json")
        self.manifest = manifest
        self.episodes, self.entries, self.windows = [], [], []
        data = config["data"]
        mode = data["split_mode"]
        if manifest.get("layout") == "single_episode":
            if mode != "target_index" and split != "forecast":
                raise ValueError("该 split_mode 需要 collection")
            self.episodes = [EpisodePackage(root, config)]
            self.entries = [{"id": "single", "site_id": self.episodes[0].site["site_id"], "split": None}]
        else:
            keys(manifest, {"schema_version", "data_kind", "layout", "schema", "episodes", "split_mode", "split_index"},
                 {"site_split", "generator"}, name="collection manifest")
            if manifest["schema_version"] != 3 or manifest["data_kind"] != "synthetic" or manifest["layout"] != "episode_collection":
                raise ValueError("collection schema_version/data_kind/layout 不兼容")
            if manifest["schema"] != schema_from_config(config) or manifest["split_mode"] != mode or manifest["split_index"] != data["split_index"]:
                raise ValueError("collection schema/split 与 config 不符")
            if mode == "site_time" and manifest.get("site_split") != data["site_split"]:
                raise ValueError("collection site_split 与 config 不符")
            families, ids, paths = {}, set(), set()
            for entry in manifest["episodes"]:
                keys(entry, {"id", "family_id", "site_id", "split", "path"}, {"topology_family_id"}, name="episode entry")
                safe_name(entry["id"], "episode_id")
                safe_name(entry["family_id"], "family_id")
                safe_name(entry["site_id"], "site_id")
                path = contained_path(root, entry["path"])
                if entry["id"] in ids or path in paths:
                    raise ValueError("重复 episode ID/path")
                ids.add(entry["id"])
                paths.add(path)
                if mode == "episode_family":
                    if entry["split"] not in {"train", "valid", "test"}:
                        raise ValueError("episode_family: 非法 split")
                    previous = families.setdefault(entry["family_id"], entry["split"])
                    if previous != entry["split"]:
                        raise ValueError("family 跨 split 泄漏")
                elif mode == "site_time":
                    if entry["split"] is not None or "topology_family_id" not in entry:
                        raise ValueError("site_time: split=null 并声明 topology_family_id")
                    sites = {site for role in data["site_split"].values() for site in role}
                    if entry["site_id"] not in sites:
                        raise ValueError("site_time: 未声明站点")
                    previous = families.setdefault(entry["family_id"], entry["site_id"])
                    if previous != entry["site_id"]:
                        raise ValueError("family 跨站点")
                else:
                    raise ValueError("collection 不支持 target_index")
            # 先校验整个清单的隔离约束，再加载子包。
            for entry in manifest["episodes"]:
                episode = EpisodePackage(contained_path(root, entry["path"]), config)
                if episode.site["site_id"] != entry["site_id"]:
                    raise ValueError("collection 与 episode 站点不一致")
                self.episodes.append(episode)
                self.entries.append(deepcopy(entry))
        for index, (episode, entry) in enumerate(zip(self.episodes, self.entries)):
            count, length, horizon = len(episode.times), data["seq_len"], data["pred_len"]
            if split == "forecast":
                if episode.manifest.get("purpose") != "forecast":
                    raise ValueError("forecast 必须提供无标签 request 包")
                origins = [count - 1]
            elif episode.manifest.get("purpose") == "forecast":
                raise ValueError("forecast request 不能作为训练／评估数据")
            elif mode == "episode_family":
                if count != length + horizon:
                    raise ValueError("episode_family: 每包长度必须恰为 L+H")
                origins = [length - 1] if entry["split"] == split else []
            else:
                boundaries = data["split_index"]
                if boundaries[-1] != count:
                    raise ValueError("split_index 末端与 episode 长度不一致")
                role_ok = True
                if mode == "site_time":
                    site = entry["site_id"]
                    roles = data["site_split"]
                    role_ok = site in roles["train"] or (split == "valid" and site in roles["heldout_valid"]) or (split == "test" and site in roles["heldout_test"])
                i = {"train": 0, "valid": 1, "test": 2}[split]
                start, end = boundaries[i:i + 2]
                origins = range(max(length - 1, start - 1), end - horizon, data[f"{split}_stride"]) if role_ok else []
            self.windows.extend((index, origin) for origin in origins)
        if not self.windows:
            raise ValueError(f"{split}: 没有合法窗口")
        self.fingerprint = canonical_hash({"manifest": manifest, "episodes": [e.fingerprint for e in self.episodes]})

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, origin = self.windows[index]
        sample = self.episodes[episode].sample(origin, supervised=self.split != "forecast")
        sample["metadata"].update(episode_id=self.entries[episode]["id"], split=self.split)
        return sample
