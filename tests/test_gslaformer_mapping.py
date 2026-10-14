"""测点单位与业务类型图的来源边界。"""
import copy
from pathlib import Path
import unittest

from data_provider.spatial_temporal.aidc_mapping import validate_mappings, validate_relation_snapshot
from data_provider.spatial_temporal.schema import read_json
from data_provider.spatial_temporal.config import schema_from_config


class MappingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "docs/gslaformer/examples/aidc_profile.example.json"
        self.schema = schema_from_config(read_json(path))
        self.order = {g["name"]: [] for g in self.schema["groups"]}
        self.order["it_power"] = ["server"]
        self.mapping = {"measurement_id": "p1", "source_name": "raw_power", "group": "it_power", "entity_id": "server",
                        "feature": "active_power_kw", "source_unit": "W", "scale": 0.001, "offset": 0, "role": "primary"}

    def test_explicit_unit_conversion_and_primary_backup_audit(self):
        backup = dict(self.mapping, measurement_id="p2", source_name="backup_power", role="backup")
        result = validate_mappings({"mappings": [self.mapping, backup]}, self.schema, self.order)
        self.assertEqual(result["primary_count"], 1)
        self.assertEqual(result["backup_count"], 1)
        self.assertEqual(result["mapped_entity_features"], 1)
        for change in ({"scale": 1}, {"source_unit": "kWh"}, {"entity_id": "unknown"}, {"feature": "typo"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_mappings({"mappings": [dict(self.mapping, **change)]}, self.schema, self.order)
        with self.assertRaisesRegex(ValueError, "primary"):
            validate_mappings({"mappings": [self.mapping, dict(backup, role="primary")]}, self.schema, self.order)
        with self.assertRaisesRegex(ValueError, "primary"):
            validate_mappings({"mappings": [backup]}, self.schema, self.order)

    def test_each_relation_type_dag_is_independent(self):
        order = {"x": ["a", "b"]}
        entities = {"x": {x: {"numeric": {}, "category": {}} for x in order["x"]}}
        kinds = {name: {"acyclic": True} for name in ("one", "two")}
        edges = [("x", 0, "x", 1, "one"), ("x", 1, "x", 0, "two")]
        validate_relation_snapshot(edges, entities, order, kinds)
        with self.assertRaisesRegex(ValueError, "循环"):
            validate_relation_snapshot([edges[0], ("x", 1, "x", 0, "one")], entities, order, kinds)

    def test_contains_direction_and_same_level_parent_uniqueness(self):
        order = {"space": ["campus", "building1", "building2", "room"]}
        levels = ["aidc_boundary", "building", "building", "room"]
        entities = {"space": {name: {"category": {"space_level": level}, "numeric": {}}
                              for name, level in zip(order["space"], levels)}}
        kinds = {"contains": {"acyclic": True}}
        valid = [("space", 0, "space", 1, "contains"), ("space", 1, "space", 3, "contains")]
        validate_relation_snapshot(valid, entities, order, kinds)
        for invalid in ([ ("space", 3, "space", 1, "contains") ], valid + [("space", 2, "space", 3, "contains")]):
            with self.assertRaises(ValueError):
                validate_relation_snapshot(invalid, entities, order, kinds)


if __name__ == "__main__":
    unittest.main()
