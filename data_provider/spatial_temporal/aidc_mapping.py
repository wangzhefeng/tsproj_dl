"""显式测点注册、单位与类型关系校验；不猜测源名称，不求和跨层计量。"""
from collections import defaultdict
import math

from .config import keys


SPACE_LEVELS = {"aidc_boundary": 0, "building": 1, "room": 2}


def validate_mappings(document, schema, entity_order):
    keys(document, {"mappings"}, name="measurements")
    if not isinstance(document["mappings"], list):
        raise ValueError("mappings: 需要数组")
    groups = {group["name"]: group for group in schema["groups"]}
    ids, sources, mapped = set(), set(), defaultdict(list)
    for record in document["mappings"]:
        fields = {"measurement_id", "source_name", "group", "entity_id", "feature", "source_unit", "scale", "offset", "role"}
        keys(record, fields, name="measurement")
        for field in fields - {"scale", "offset"}:
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"measurement.{field}: 需要非空字符串")
        if record["measurement_id"] in ids or record["source_name"] in sources:
            raise ValueError("重复 measurement_id/source_name")
        ids.add(record["measurement_id"])
        sources.add(record["source_name"])
        group, feature = record["group"], record["feature"]
        if (group not in groups or record["entity_id"] not in entity_order[group]
                or feature not in groups[group]["dynamic"]):
            raise ValueError("measurement: 未声明实体／动态特征")
        for field in ("scale", "offset"):
            value = record[field]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"measurement.{field}: 需要有限数值")
        unit = schema.get("feature_units", {}).get(feature, schema["target_unit"])
        source = record["source_unit"]
        conversions = {("W", "kW"): (0.001, 0), ("MW", "kW"): (1000, 0), ("percent", "fraction"): (0.01, 0)}
        expected = (1, 0) if source == unit else conversions.get((source, unit))
        if expected is None or (record["scale"], record["offset"]) != expected:
            raise ValueError(f"measurement: 未支持或不一致的单位换算 {source} -> {unit}")
        if record["role"] not in {"primary", "backup"}:
            raise ValueError("measurement.role: 需要 primary/backup")
        mapped[(group, record["entity_id"], feature)].append(record["role"])
    for roles in mapped.values():
        if roles.count("primary") != 1:
            raise ValueError("每个已映射实体特征必须恰有一个 primary；backup 不能独立冒充主测点")
    return {"primary_count": sum(r.count("primary") for r in mapped.values()),
            "backup_count": sum(r.count("backup") for r in mapped.values()),
            "mapped_entity_features": len(mapped)}


def validate_relation_snapshot(edges, entities, entity_order, relation_types):
    """有效快照逐类型 DAG；contains 另检空间层级和同层父级唯一性。"""
    graphs = defaultdict(lambda: defaultdict(set))
    seen, contains_parents = set(), defaultdict(set)
    for sg, si, dg, di, kind in edges:
        edge = (sg, si, dg, di, kind)
        if edge in seen:
            raise ValueError("快照存在同类型重复边")
        seen.add(edge)
        src, dst = (sg, si), (dg, di)
        if src == dst:
            raise ValueError("源记录禁止 self")
        source = entities[sg][entity_order[sg][si]]
        target = entities[dg][entity_order[dg][di]]
        if source is None or target is None:
            raise ValueError("关系端点尚未生效")
        if kind == "contains":
            level = source["category"].get("space_level")
            if level not in SPACE_LEVELS:
                raise ValueError("contains 源需要已知 space_level")
            target_level = target["category"].get("space_level")
            if dg == "space" and (target_level not in SPACE_LEVELS or SPACE_LEVELS[level] >= SPACE_LEVELS[target_level]):
                raise ValueError("contains 方向必须由较大空间指向较小空间")
            parents = contains_parents[(dst, level)]
            parents.add(src)
            if len(parents) > 1:
                raise ValueError("contains 同空间层级父级必须唯一")
        if kind == "supplies":
            source_type = source["category"].get("entity_type")
            target_type = target["category"].get("entity_type")
            source_port = source["category"].get("port")
            target_port = target["category"].get("port")
            if source_type == "server":
                raise ValueError("supplies: 服务器不能是上游供电源")
            if source_type == "ups" and source_port == "input" and not (target_type == "ups" and target_port == "output"):
                raise ValueError("supplies: UPS input 必须先到 output")
            if source_type == target_type == "ups" and source_port == "output" and target_port == "input":
                raise ValueError("supplies: UPS 端口方向错误")
        graphs[kind][src].add(dst)
    for kind, graph in graphs.items():
        if not relation_types[kind]["acyclic"]:
            continue
        nodes = set(graph) | {n for neighbors in graph.values() for n in neighbors}
        indegree = {node: 0 for node in nodes}
        for neighbors in graph.values():
            for node in neighbors:
                indegree[node] += 1
        ready = [node for node in nodes if indegree[node] == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for neighbor in graph.get(node, ()):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    ready.append(neighbor)
        if visited != len(nodes):
            raise ValueError(f"{kind}: 违反逐类型无环约束，发现循环")
