"""GSLAformer schema 3 配置：显式参数、严格类型与分区合同。"""
from copy import deepcopy
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .schema import read_json


VARIANTS = {
    "full", "direct", "total_only", "no_entity_history", "no_spatial",
    "no_relation_message", "no_relation_all", "no_latent_loss",
    "persistence", "seasonal_naive",
}
SCHEMA_REQUIRED = {
    "timezone", "freq_seconds", "groups", "relation_types", "known_future",
    "target_name", "target_unit",
}
SCHEMA_OPTIONAL = {"feature_units", "target_quantity", "interval_label", "target_aggregation"}


def keys(value, required, optional=(), *, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}: 需要 JSON object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: 需要 >= {minimum} 的整数")
    return value


def number(value, name, minimum=0, maximum=None):
    if (type(value) not in (float, int) or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name}: 非法有限数值 {value!r}")
    return value


def names(value, name, *, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name}: 需要不重复的名称列表")
    return value


def safe_name(value, name="name"):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError(f"{name}: 非法安全名称 {value!r}")
    return value


def validate_schema(schema):
    keys(schema, SCHEMA_REQUIRED, SCHEMA_OPTIONAL, name="schema")
    integer(schema["freq_seconds"], "freq_seconds")
    try:
        ZoneInfo(schema["timezone"])
    except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("timezone: 无效时区") from exc
    for name in ("target_name", "target_unit"):
        if not isinstance(schema[name], str) or not schema[name]:
            raise ValueError(f"{name}: 需要非空字符串")
    groups = schema["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups: 至少需要一个动态组")
    group_names = []
    for group in groups:
        keys(group, {"name", "dynamic", "static_numeric", "static_category"}, name="group")
        name = group["name"]
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_]+", name) is None or "__" in name:
            raise ValueError("group.name: 不允许双下划线或非字母数字下划线")
        group_names.append(name)
        all_features = []
        for field in ("dynamic", "static_numeric", "static_category"):
            all_features.extend(names(group[field], f"{name}.{field}"))
        if len(set(all_features)) != len(all_features):
            raise ValueError(f"{name}: 跨角色重复特征")
    names(group_names, "groups", nonempty=True)
    if not any(group["dynamic"] for group in groups):
        raise ValueError("groups: 至少需要一个动态组")
    names(schema["known_future"], "known_future")
    relations = schema["relation_types"]
    if not isinstance(relations, list):
        raise ValueError("relation_types: 需要数组")
    relation_names = []
    for relation in relations:
        keys(relation, {"name", "src_groups", "dst_groups", "acyclic", "reverse_message"}, name="relation_type")
        relation_names.append(safe_name(relation["name"], "relation_type.name"))
        for field in ("src_groups", "dst_groups"):
            endpoints = names(relation[field], f"relation_type.{field}", nonempty=True)
            if not set(endpoints) <= set(group_names):
                raise ValueError(f"relation_type.{field}: 未声明组")
        if any(type(relation[k]) is not bool for k in ("acyclic", "reverse_message")):
            raise ValueError("relation_type: acyclic/reverse_message 必须是 bool")
    names(relation_names, "relation_types")
    if "target_quantity" in schema:
        if schema["target_quantity"] not in {"facility_demand", "grid_net_import"}:
            raise ValueError("target_quantity: 不支持的目标")
        if schema.get("interval_label") != "right" or schema.get("target_aggregation") != "time_weighted_mean":
            raise ValueError("AIDC 需要 right/time_weighted_mean 计量合同")
        units = schema.get("feature_units", {})
        features = set(schema["known_future"])
        for group in groups:
            features.update(group["dynamic"] + group["static_numeric"])
        keys(units, features, name="feature_units")
        if any(not isinstance(u, str) or not u for u in units.values()):
            raise ValueError("feature_units: 需要非空单位")
    return schema


def schema_from_config(config):
    return {key: deepcopy(value) for key, value in config["data"].items()
            if key in SCHEMA_REQUIRED | SCHEMA_OPTIONAL}


def validate_config(config):
    keys(config, {"schema_version", "experiment", "data", "model", "limits", "training", "evaluation"}, name="config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 3:
        raise ValueError("schema_version: 仅支持 3")
    experiment = config["experiment"]
    keys(experiment, {"model", "variant", "data_kind", "seed", "device", "results_root"}, {"profile"}, name="experiment")
    if experiment["model"] != "GSLAformer" or experiment["variant"] not in VARIANTS:
        raise ValueError("experiment: 不支持的 model/variant")
    if experiment["data_kind"] != "synthetic" or experiment["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("experiment: 本轮仅支持 synthetic 与 cpu/mps/cuda")
    integer(experiment["seed"], "seed", 0)
    if "profile" in experiment:
        safe_name(experiment["profile"], "profile")
    data = config["data"]
    keys(data, SCHEMA_REQUIRED | {"package", "seq_len", "pred_len", "train_stride", "valid_stride", "test_stride",
                                 "split_mode", "split_index", "batch_size", "num_workers"},
         SCHEMA_OPTIONAL | {"sampler", "site_split"}, name="data")
    for key in ("package",):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key}: 需要路径")
    if not isinstance(experiment["results_root"], str) or not experiment["results_root"]:
        raise ValueError("results_root: 需要路径")
    validate_schema(schema_from_config(config))
    for key in ("seq_len", "pred_len", "train_stride", "valid_stride", "test_stride", "batch_size"):
        integer(data[key], f"data.{key}")
    integer(data["num_workers"], "num_workers", 0)
    mode = data["split_mode"]
    if mode not in {"target_index", "episode_family", "site_time"}:
        raise ValueError("split_mode: 未知分区协议")
    bounds = data["split_index"]
    if mode == "episode_family":
        if bounds is not None:
            raise ValueError("episode_family 的 split_index 必须是 null")
    elif (not isinstance(bounds, list) or len(bounds) != 4
          or any(type(v) is not int for v in bounds) or bounds[0] != 0
          or any(a >= b for a, b in zip(bounds, bounds[1:]))):
        raise ValueError("split_index: 需要从 0 开始的四个递增整数")
    if mode == "site_time":
        roles = data.get("site_split")
        keys(roles, {"train", "heldout_valid", "heldout_test"}, name="site_split")
        sites = []
        for role, values in roles.items():
            sites.extend(names(values, f"site_split.{role}", nonempty=True))
        names(sites, "site_split")
        if data.get("sampler") != "uniform_site_then_origin":
            raise ValueError("site_time: 需要 uniform_site_then_origin")
    elif "site_split" in data or data.get("sampler", "shuffle") != "shuffle":
        raise ValueError("非 site_time 不接受 site_split/站点采样")
    model = config["model"]
    model_ints = {"spatial_dim", "latent_dim", "temporal_dim", "spatial_heads", "relation_heads", "temporal_heads",
                  "spatial_layers", "entity_history_layers", "relation_layers", "temporal_layers",
                  "future_path_layers", "category_embedding_dim", "lead_dim"}
    keys(model, model_ints | {"dropout"}, name="model")
    for key in model_ints:
        integer(model[key], f"model.{key}")
    for dim, head in (("spatial_dim", "spatial_heads"), ("spatial_dim", "relation_heads"), ("temporal_dim", "temporal_heads")):
        if model[dim] % model[head]:
            raise ValueError(f"model.{dim}: 不能被 {head} 整除")
    for key, value in {"spatial_layers": 1, "entity_history_layers": 1, "relation_layers": 2,
                       "future_path_layers": 1, "category_embedding_dim": 8, "lead_dim": 16}.items():
        if model[key] != value:
            raise ValueError(f"model.{key}: 首版固定 {value}")
    number(model["dropout"], "dropout", 0, 1)
    if model["dropout"] == 1:
        raise ValueError("dropout 必须小于 1")
    limits = config["limits"]
    keys(limits, {"max_total_nodes", "max_history_steps", "max_horizon_steps", "max_explicit_attention_bytes"}, name="limits")
    for key, value in limits.items():
        integer(value, f"limits.{key}")
    if data["seq_len"] > limits["max_history_steps"] or data["pred_len"] > limits["max_horizon_steps"]:
        raise ValueError("L/H 超出 limits")
    training = config["training"]
    epochs = {"stage_a_max_epochs", "stage_a_patience", "b_short_epochs", "b_medium_epochs", "b_full_max_epochs", "b_full_patience"}
    keys(training, epochs | {"optimizer", "learning_rate", "weight_decay", "clip_grad_norm", "latent_weight",
                             "min_delta", "amp", "truncate_rollout_grad"}, name="training")
    for key in epochs:
        integer(training[key], f"training.{key}")
    for key in ("learning_rate", "weight_decay", "clip_grad_norm", "latent_weight", "min_delta"):
        number(training[key], f"training.{key}")
    if training["learning_rate"] == 0:
        raise ValueError("learning_rate 必须为正")
    if (training["optimizer"] != "AdamW" or training["amp"] is not False
            or training["truncate_rollout_grad"] is not False or training["clip_grad_norm"] != 1):
        raise ValueError("training: AdamW、clip=1、禁用 AMP 与截断梯度")
    weight = 0 if experiment["variant"] == "no_latent_loss" else 0.1
    if training["latent_weight"] != weight:
        raise ValueError(f"latent_weight: {experiment['variant']} 要求 {weight}")
    evaluation = config["evaluation"]
    keys(evaluation, {"selection_metric", "seasonal_period", "compare_seed"}, {"report_by"}, name="evaluation")
    expected = "valid_site_macro_mse" if mode == "site_time" else "valid_macro_mse"
    if evaluation["selection_metric"] != expected:
        raise ValueError(f"selection_metric: 需要 {expected}")
    integer(evaluation["seasonal_period"], "seasonal_period")
    integer(evaluation["compare_seed"], "compare_seed", 0)
    if "report_by" in evaluation:
        names(evaluation["report_by"], "report_by")
        if not set(evaluation["report_by"]) <= {"site_id", "site_role", "horizon", "scenario"}:
            raise ValueError("report_by: 未知分层字段")
    return config


def load_config(path):
    path = Path(path).resolve()
    config = validate_config(read_json(path))
    config["data"]["package"] = str((path.parent / config["data"]["package"]).resolve())
    root = Path(__file__).resolve().parents[2]
    config["experiment"]["results_root"] = str((root / config["experiment"]["results_root"]).resolve())
    return config
