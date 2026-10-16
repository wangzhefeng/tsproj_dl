"""独立小集验收：总体 MSE 不可掩盖成对失败，复用权重限定任务。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.dev.fit_gslaformer_mechanisms import measure


ROOT = Path(__file__).resolve().parents[1]


class SmallFitAcceptanceTest(unittest.TestCase):
    def test_pair_gate_rejects_mean_and_reversed_predictions(self):
        target = torch.tensor([-1.0, 1.0] * 8).reshape(16, 1, 1).expand(-1, 2, -1)
        for task in ('M1', 'M2'):
            for prediction in (torch.zeros_like(target), -target):
                with self.subTest(task=task, prediction=prediction[0, 0, 0].item()):
                    result = measure(prediction, target, task, initial=100.0)
                    self.assertTrue(result['loss_passed'])
                    self.assertFalse(result['pair_passed'])
                    self.assertFalse(result['passed'])
            self.assertTrue(measure(target.clone(), target, task, initial=100.0)['passed'])

    def test_resume_cli_rejects_other_tasks_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/dev/fit_gslaformer_mechanisms.py'),
                 '--fixtures', str(root), '--output', str(output), '--tasks', 'M2_full',
                 '--stage_a_checkpoint', str(root / 'unused.pt')],
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('仅接受 --tasks M1_full', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
