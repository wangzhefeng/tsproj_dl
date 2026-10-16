"""缓存只减少重复工作，不能改变样本、可见性、时区或调用方隔离。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from data_provider.spatial_temporal.cache import SampleCache
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from data_provider.spatial_temporal.batching import collate_samples
from layers.gslaformer.temporal import calendar_features
from test_gslaformer_package import small_config, write_package


class CacheTests(unittest.TestCase):
    def test_cache_isolation_eviction_identity_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config()
            config['data'].update(package=directory + '/data', known_future=[])
            write_package(Path(config['data']['package']), config)
            dataset = PackageDataset(config, 'train')
            transform = TrainingTransform.fit(dataset)
            cache = SampleCache(1000000)
            expected = transform.apply(dataset[0])
            actual = cache.get(dataset, transform, 0)
            actual['inputs']['target_history'][:] = 999
            cached = cache.get(dataset, transform, 0)
            for key in ('target_history', 'target_history_observed', 'future_known', 'future_known_mask'):
                np.testing.assert_array_equal(cached['inputs'][key], expected['inputs'][key])
            self.assertEqual(cache.hits, 1)
            original_bytes = cache.bytes
            cache.budget_bytes = original_bytes
            cache.get(dataset, transform, 1)
            self.assertGreater(cache.evictions, 0)
            self.assertLessEqual(cache.bytes, cache.budget_bytes)
            cache.get(dataset, transform, 0)
            np.testing.assert_array_equal(cache.get(dataset, transform, 0)['inputs']['target_history'], expected['inputs']['target_history'])
            old_misses = cache.misses
            dataset.config['experiment']['variant'] = 'direct'
            cache.get(dataset, transform, 0)
            self.assertEqual(cache.misses, old_misses + 1)
            changed = deepcopy(transform)
            changed.state['target']['mean'][0] += 1
            changed = TrainingTransform(changed.state)
            transformed = cache.get(dataset, changed, 0)
            np.testing.assert_array_equal(transformed['inputs']['target_history'], changed.apply(dataset[0])['inputs']['target_history'])
            disabled = SampleCache(0)
            disabled.get(dataset, transform, 0)
            disabled.get(dataset, transform, 0)
            self.assertEqual((disabled.hits, disabled.bytes), (0, 0))

    def test_calendar_matches_independent_dst_clock(self):
        # 2024-03-10 纽约 DST 跳过 02:00，本地 01:00 后为 03:00。
        from datetime import datetime, timezone
        times = torch.tensor([[int(datetime(2024, 3, 10, hour, tzinfo=timezone.utc).timestamp()) for hour in (6, 7)]])
        expected = []
        for hour in (1, 3):
            day = hour / 24
            week = (6 + day) / 7
            expected.append([np.sin(2*np.pi*day), np.cos(2*np.pi*day), np.sin(2*np.pi*week), np.cos(2*np.pi*week)])
        torch.testing.assert_close(calendar_features(times, 'America/New_York'), torch.tensor([expected], dtype=torch.float32), atol=0, rtol=0)
