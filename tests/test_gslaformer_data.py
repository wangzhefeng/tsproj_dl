"""GSLAformer 公共数据合同；期望值来自独立小夹具。"""
import json
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_provider.spatial_temporal.schema import canonical_hash, read_json, quality_flags
from data_provider.spatial_temporal.config import load_config, validate_config, schema_from_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / "scripts/smoke/gslaformer_functional.json"
        self.config = read_json(self.path)

    def test_profiles_resolve_paths_and_preserve_schema(self):
        config = load_config(self.path)
        self.assertEqual(Path(config["data"]["package"]), self.path.parents[2] / "results/gslaformer/fixtures/base")
        schema = schema_from_config(config)
        self.assertEqual(schema["groups"], self.config["data"]["groups"])
        self.assertNotIn("package", schema)
        self.assertNotIn("split_index", schema)
        aidc = load_config(self.path.with_name("gslaformer_aidc_hourly_day.json"))
        self.assertEqual(Path(aidc["data"]["package"]), self.path.parents[2] / "results/gslaformer/fixtures/aidc")
        self.assertEqual(aidc["data"]["seq_len"], 168)
        self.assertEqual(aidc["evaluation"]["selection_metric"], "valid_site_macro_mse")

    def test_invalid_configs_fail_before_allocation_or_training(self):
        cases = [
            ("unknown field", lambda c: c["model"].update(typo=1)),
            ("bool dimension", lambda c: c["model"].update(spatial_dim=True)),
            ("fractional dimension", lambda c: c["model"].update(spatial_dim=32.5)),
            ("heads", lambda c: c["model"].update(relation_heads=3)),
            ("history limit", lambda c: c["data"].update(seq_len=97)),
            ("entity layers", lambda c: c["model"].update(entity_history_layers=2)),
            ("truncated gradients", lambda c: c["training"].update(truncate_rollout_grad=True)),
            ("AMP", lambda c: c["training"].update(amp=True)),
            ("latent loss", lambda c: c["training"].update(latent_weight=0)),
            ("bad split", lambda c: c["data"].update(split_index=[0, 192, 192, 320])),
            ("episode bounds", lambda c: c["data"].update(split_mode="episode_family")),
            ("duplicate group", lambda c: c["data"]["groups"].append(copy.deepcopy(c["data"]["groups"][0]))),
            ("NPZ ambiguity", lambda c: c["data"]["groups"][0].update(name="bad__group")),
            ("relation endpoint", lambda c: c["data"]["relation_types"][0].update(src_groups=["absent"])),
            ("all static", lambda c: [g.update(dynamic=[]) for g in c["data"]["groups"]]),
        ]
        for name, change in cases:
            config = copy.deepcopy(self.config)
            change(config)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_config(config)

    def test_site_roles_must_be_disjoint_and_select_site_macro(self):
        config = read_json(self.path.with_name("gslaformer_aidc_hourly_day.json"))
        config["data"]["site_split"]["heldout_test"].append("aidc_00")
        with self.assertRaisesRegex(ValueError, "site_split"):
            validate_config(config)
        config = read_json(self.path.with_name("gslaformer_aidc_hourly_day.json"))
        config["evaluation"]["selection_metric"] = "valid_macro_mse"
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            validate_config(config)


class SchemaTests(unittest.TestCase):
    def test_json_keys_are_order_independent_but_values_are_not(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))
        with self.assertRaises(ValueError):
            canonical_hash({"a": float("nan")})

    def test_duplicate_json_key_and_nonfinite_number_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}'):
                p.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_json(p)

    def test_quality_uses_applicable_cells_not_padding_or_group_mean(self):
        groups = {
            "a": {"observed": np.array([[[True, False], [True, True]]]),
                  "applicable": np.array([[True, True], [True, True]]),
                  "present": np.array([[True, False]]), "padding": np.array([False, True])},
            "b": {"observed": np.array([[[True]]]), "applicable": np.array([[True]]),
                  "present": np.array([[True]]), "padding": np.array([False])},
        }
        q = quality_flags(groups, np.array([[True]]), np.array([[True, False]]))
        self.assertAlmostEqual(q["input_coverage"], 2 / 3)
        self.assertEqual(q["target_history_coverage"], 1)
        self.assertEqual(q["plan_coverage"], 0.5)
        self.assertTrue(q["plan_available"])
        self.assertTrue(q["degraded_input"])
        q = quality_flags({"b": groups["b"]}, np.array([[True]]), np.empty((2, 0), dtype=bool))
        self.assertFalse(q["plan_available"])
        self.assertIsNone(q["plan_coverage"])
        self.assertFalse(q["degraded_input"])


if __name__ == "__main__":
    unittest.main()
