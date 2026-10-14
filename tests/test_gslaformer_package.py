"""标准包真实落盘边界：时间、来源、版本与监督隔离，不 mock loader。"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_provider.spatial_temporal.config import schema_from_config
from data_provider.spatial_temporal.schema import read_json
from data_provider.spatial_temporal.dataset import EpisodePackage, PackageDataset


ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = read_json(ROOT / "docs/gslaformer/examples/functional_profile.example.json")
    data = config["data"]
    data.update(seq_len=3, pred_len=2, split_index=[0, 6, 9, 12], batch_size=2)
    data["groups"] = [
        {"name": "assets", "dynamic": ["load"], "static_numeric": ["capacity"], "static_category": ["kind"]},
        {"name": "zones", "dynamic": [], "static_numeric": ["capacity"], "static_category": []},
    ]
    data["relation_types"] = [{"name": "belongs", "src_groups": ["assets"], "dst_groups": ["zones"],
                               "acyclic": True, "reverse_message": True}]
    return config


def write_package(path, config=None, *, forecast=False):
    """固定解析夹具；生产生成器不能作为这个 oracle 的来源。"""
    config = small_config() if config is None else config
    path.mkdir(parents=True)
    times = np.arange(12, dtype=np.int64) * 3600
    n = 6 if forecast else 12
    times = times[:n]
    arrays = {"timestamps": times, "target": np.arange(n, dtype=np.float32)[:, None] * 10,
              "target_observed": np.ones((n, 1), bool), "target_available_at": times[:, None].copy()}
    entities = {"groups": {}}
    for group, ids in (("assets", ["a", "b"]), ("zones", ["z0", "z1"])):
        features = 1 if group == "assets" else 0
        arrays[group + "__values"] = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None, None], (n, 2, features)).copy()
        arrays[group + "__observed"] = np.ones((n, 2, features), bool)
        arrays[group + "__applicable"] = np.ones((2, features), bool)
        arrays[group + "__present"] = np.ones((n, 2), bool)
        arrays[group + "__available_at"] = np.broadcast_to(times[:, None, None], (n, 2, features)).copy()
        entities["groups"][group] = [
            {"id": name, "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0,
             "numeric": {"capacity": 2 + i}, "category": {"kind": "even" if i == 0 else "odd"} if features else {}}
            for i, name in enumerate(ids)]
    edges = [{"src_group": "assets", "src_id": a, "dst_group": "zones", "dst_id": z,
              "type": "belongs", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
             for a, z in (("a", "z0"), ("b", "z1"))]
    target = {"target_id": "main", "quantity": "synthetic_total", "boundary_id": "whole",
              "unit": "arbitrary_unit", "sign_convention": "positive_consumption", "interval_label": "right",
              "aggregation": "time_weighted_mean", "meter_ids": ["main"], "coverage_description": "synthetic"}
    mappings = [{"measurement_id": name, "source_name": name, "group": "assets", "entity_id": name,
                 "feature": "load", "source_unit": "arbitrary_unit", "scale": 1.0, "offset": 0.0, "role": "primary"}
                for name in ("a", "b")]
    for filename, obj in {"entities.json": entities, "relations.json": {"edges": edges},
                          "site.json": {"site_id": "synthetic_site", "target_contract": target},
                          "measurements.json": {"mappings": mappings}}.items():
        (path / filename).write_text(json.dumps(obj))
    np.savez(path / "observations.npz", **arrays)
    future = {key: np.empty(0, dtype=np.int64) for key in ("origin", "target_time", "feature_index", "published_at", "revision")}
    future.update(value=np.empty(0, dtype=np.float32), observed=np.empty(0, dtype=bool))
    np.savez(path / "known_future.npz", **future)
    manifest = {"schema_version": 3, "data_kind": "synthetic", "layout": "single_episode",
                "schema": schema_from_config(config), "entity_order": {"assets": ["a", "b"], "zones": ["z0", "z1"]}, "files": {}}
    if forecast:
        manifest["purpose"] = "forecast"
    (path / "manifest.json").write_text(json.dumps(manifest))
    rehash(path)
    return config


def rehash(path):
    manifest = read_json(path / "manifest.json")
    manifest["files"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in
                         ("entities.json", "relations.json", "site.json", "measurements.json", "observations.npz", "known_future.npz")}
    (path / "manifest.json").write_text(json.dumps(manifest))


def edit_arrays(path, filename, edit):
    with np.load(path / filename, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    edit(arrays)
    np.savez(path / filename, **arrays)
    rehash(path)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "episode"
        self.config = write_package(self.path)
        self.config["data"]["package"] = str(self.path)

    def test_target_anchored_windows_and_supervision_boundary(self):
        dataset = PackageDataset(self.config, "train")
        self.assertEqual([w[1] for w in dataset.windows], [2, 3])
        self.assertEqual([w[1] for w in PackageDataset(self.config, "valid").windows], [5, 6])
        self.assertEqual([w[1] for w in PackageDataset(self.config, "test").windows], [8, 9])
        sample = dataset[0]
        np.testing.assert_array_equal(sample["inputs"]["history_time"], [0, 3600, 7200])
        np.testing.assert_array_equal(sample["supervision"]["target"].ravel(), [30, 40])
        self.assertNotIn("target", sample["inputs"])
        self.assertNotIn("future_groups", sample["inputs"])
        self.assertEqual(sample["inputs"]["groups"]["zones"]["values"].shape, (3, 2, 0))

    def test_delayed_history_and_plan_revisions_use_original_origin(self):
        def corrupt(a):
            a["assets__available_at"][2, 0, 0] = 10800
            a["assets__values"][1, 1, 0] = np.nan
            a["assets__observed"][1, 1, 0] = False
            a["target_available_at"][2] = 10800
        edit_arrays(self.path, "observations.npz", corrupt)
        future = {"origin": [7200] * 4, "target_time": [10800] * 4, "feature_index": [0] * 4,
                  "published_at": [0, 3600, 3600, 10800], "revision": [0, 0, 1, 2],
                  "value": [1, 2, 3, 99], "observed": [True] * 4}
        np.savez(self.path / "known_future.npz", **{k: np.array(v, dtype=bool if k == "observed" else np.float32 if k == "value" else np.int64) for k, v in future.items()})
        rehash(self.path)
        sample = EpisodePackage(self.path, self.config).sample(2)
        inputs = sample["inputs"]
        self.assertEqual(inputs["groups"]["assets"]["values"][2, 0, 0], 0)
        self.assertFalse(inputs["groups"]["assets"]["observed"][2, 0, 0])
        self.assertTrue(np.isfinite(inputs["groups"]["assets"]["values"]).all())
        self.assertFalse(inputs["target_history_observed"][-1, 0])
        np.testing.assert_array_equal(inputs["future_known"].ravel(), [3, 0])
        np.testing.assert_array_equal(inputs["future_known_mask"].ravel(), [True, False])
        self.assertTrue(sample["supervision"]["target_mask"].all())

    def test_historical_attributes_and_edges_are_not_backfilled(self):
        entities = read_json(self.path / "entities.json")
        newer = copy.deepcopy(entities["groups"]["assets"][0])
        newer.update(valid_from=7200, available_at=7200, revision=1, numeric={"capacity": 9})
        entities["groups"]["assets"].append(newer)
        (self.path / "entities.json").write_text(json.dumps(entities))
        rel = read_json(self.path / "relations.json")
        rel["edges"][0]["valid_to"] = 7200
        newer_edge = copy.deepcopy(rel["edges"][0])
        newer_edge.update(dst_id="z1", valid_from=7200, valid_to=None, available_at=7200, revision=1)
        rel["edges"].append(newer_edge)
        (self.path / "relations.json").write_text(json.dumps(rel))
        rehash(self.path)
        sample = EpisodePackage(self.path, self.config).sample(2)["inputs"]
        np.testing.assert_array_equal(sample["groups"]["assets"]["static_numeric_history"][:, 0, 0], [2, 2, 9])
        self.assertIn(("assets", 0, "zones", 0, "belongs"), sample["relations_history"][0])
        self.assertIn(("assets", 0, "zones", 1, "belongs"), sample["relations_origin"])

    def test_known_future_topology_change_is_rejected_but_late_release_is_not_used(self):
        entities = read_json(self.path / "entities.json")
        record = copy.deepcopy(entities["groups"]["assets"][0])
        record.update(valid_from=10800, available_at=7200, revision=1, numeric={"capacity": 9})
        entities["groups"]["assets"].append(record)
        (self.path / "entities.json").write_text(json.dumps(entities))
        rehash(self.path)
        with self.assertRaisesRegex(ValueError, "未来.*变化"):
            EpisodePackage(self.path, self.config).sample(2)
        record["available_at"] = 10800
        (self.path / "entities.json").write_text(json.dumps(entities))
        rehash(self.path)
        sample = EpisodePackage(self.path, self.config).sample(2)
        self.assertEqual(sample["inputs"]["groups"]["assets"]["static_numeric_history"][-1, 0, 0], 2)

    def test_forecast_has_no_supervision_or_future_observations(self):
        request = self.path.parent / "request"
        write_package(request, forecast=True)
        package = EpisodePackage(request, self.config)
        sample = package.sample(5, supervised=False)
        self.assertIsNone(sample["supervision"])
        np.testing.assert_array_equal(sample["inputs"]["future_time"], [21600, 25200])
        with self.assertRaisesRegex(ValueError, "forecast"):
            package.sample(3, supervised=True)
        edit_arrays(request, "observations.npz", lambda a: a.update(future_target=np.ones((2, 1), np.float32)))
        with self.assertRaisesRegex(ValueError, "unknown"):
            EpisodePackage(request, self.config)

    def test_hash_path_dtype_axis_and_dimension_guards(self):
        (self.path / "site.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            EpisodePackage(self.path, self.config)
        for kind in ("path", "dtype", "axis", "time", "size", "observed_nan"):
            with self.subTest(kind=kind):
                path = self.path.parent / kind
                write_package(path)
                config = copy.deepcopy(self.config)
                if kind == "path":
                    manifest = read_json(path / "manifest.json")
                    manifest["files"]["../escape"] = "0" * 64
                    (path / "manifest.json").write_text(json.dumps(manifest))
                elif kind == "dtype":
                    edit_arrays(path, "observations.npz", lambda a: a.update(target=np.ones((12, 1), dtype=object)))
                elif kind == "axis":
                    manifest = read_json(path / "manifest.json")
                    manifest["entity_order"]["assets"] = ["a", "a"]
                    (path / "manifest.json").write_text(json.dumps(manifest))
                elif kind == "time":
                    edit_arrays(path, "observations.npz", lambda a: a["timestamps"].__setitem__(3, 7200))
                elif kind == "size":
                    config["limits"]["max_total_nodes"] = 3
                else:
                    edit_arrays(path, "observations.npz", lambda a: a["target"].__setitem__(1, np.nan))
                with self.assertRaises(ValueError):
                    EpisodePackage(path, config)

    def test_collection_rejects_family_split_leak_and_isolates_sites(self):
        root = self.path.parent / "collection"
        root.mkdir()
        config = copy.deepcopy(self.config)
        config["data"].update(package=str(root), split_mode="episode_family", split_index=None)
        entries = []
        for index, split in enumerate(("train", "test")):
            write_package(root / f"e{index}")
            entries.append({"id": f"e{index}", "family_id": "pair", "site_id": "synthetic_site", "split": split, "path": f"e{index}"})
        manifest = {"schema_version": 3, "data_kind": "synthetic", "layout": "episode_collection",
                    "schema": schema_from_config(config), "split_mode": "episode_family", "split_index": None, "episodes": entries}
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "family"):
            PackageDataset(config, "train")


if __name__ == "__main__":
    unittest.main()
