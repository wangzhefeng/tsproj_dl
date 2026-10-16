"""生成公式、计划发布时间及正式 profile 配置的独立合同。"""

import json
from pathlib import Path
import unittest
import tempfile
import subprocess
import sys
import numpy as np
from scripts.dev import generate_gslaformer_fixture as gen

ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_base_only_cli_in_fresh_directory_has_no_phantom_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fixtures'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/dev/generate_gslaformer_fixture.py'),
                                     '--suite', 'base', '--output', str(output)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            suite = json.loads((output / 'mechanisms/suite.json').read_text())
            self.assertEqual(suite['tasks'], [])
            self.assertEqual(len(suite['requests']), 4)
            for path in suite['requests'].values():
                self.assertTrue((output / path / 'manifest.json').is_file())
            # 存在但不完整的包必须报错，不能误当成未选择的任务。
            (output / 'mechanisms/M1').mkdir()
            with self.assertRaises(FileNotFoundError):
                gen.write_suite(output)

    def test_m2_parent_swap_changes_future_but_not_history(self):
        loads = np.tile([1., 3.], (gen.MECH_LENGTH, 1))
        a, b = gen.m2_target(loads, False), gen.m2_target(loads, True)
        np.testing.assert_array_equal(a[:gen.MECH_LENGTH], b[:gen.MECH_LENGTH])
        np.testing.assert_array_equal(a[gen.MECH_LENGTH:], np.full(gen.MECH_HORIZON, 7.))
        np.testing.assert_array_equal(b[gen.MECH_LENGTH:], np.full(gen.MECH_HORIZON, 5.))

    def test_m3_formula_and_future_innovation_cannot_change_past(self):
        np.testing.assert_allclose(gen.m3_series(10., [5., 0.]), [10., 13., 10.4], atol=1e-12)
        a = gen.m3_series(10., [5., 0., 7.])
        b = gen.m3_series(10., [5., 0., 999.])
        np.testing.assert_array_equal(a[:3], b[:3])
        self.assertNotEqual(a[3], b[3])

    def test_plan_revision_not_visible_before_publication(self):
        base = gen.AIDC_PREWARM
        block = {'start': base + 5, 'duration': 5, 'pub0': base + 1, 'extra': .4,
                 'revised': True, 'pub1': base + 3, 'extra1': .8}
        blocks = {'inference': [], 'training': [block]}
        value, published, revision, observed = gen.aidc_visible_plan(blocks, np.array([0, 2, 3]))
        # 对齐同一个 target=5，各 origin 下对应索引分别为 4/2/1。
        self.assertFalse(observed[0, 4])
        self.assertEqual(value[0, 4], 0.)
        self.assertEqual(value[1, 2], .2)
        self.assertEqual(value[2, 1], .4)
        self.assertFalse(revision[1, 2])
        self.assertTrue(revision[2, 1])
        self.assertEqual(published[1, 2], base + 1)
        self.assertEqual(published[2, 1], base + 3)

    def test_formal_aidc_profiles_equal_generator_semantics(self):
        for profile in gen.AIDC_PROFILES:
            generated = gen.aidc_profile_config(profile)
            formal = json.loads((ROOT / f'scripts/smoke/gslaformer_{profile}.json').read_text())

            generated['data']['package'] = formal['data']['package']
            self.assertEqual(generated, formal)
