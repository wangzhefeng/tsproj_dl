"""公共指标的独立数值合同；与生产实现无关的可解析序列。"""
import unittest
import numpy as np
from exp.gslaformer_metrics import forecast_metrics, latent_metrics
from exp.gslaformer_artifacts import site_metrics
from test_gslaformer_package import small_config


class MetricsTests(unittest.TestCase):
    def test_forecast_exact_mask_and_horizon(self):
        p = np.array([[[1.], [3.], [2.]]])
        y = np.array([[[1.], [2.], [4.]]])
        mask = np.ones_like(p, bool)
        result = forecast_metrics(p, y, mask, 3600)
        self.assertEqual(result['micro_mae'], 1.)
        self.assertAlmostEqual(result['micro_rmse'], np.sqrt(5 / 3))
        self.assertEqual(result['peak_amplitude_mae'], 1.)
        self.assertEqual(result['peak_time_mae_seconds'], 3600.)
        self.assertEqual(result['ramp_mae'], 2.)
        self.assertEqual([x['mae'] for x in result['per_horizon']], [0., 1., 2.])
        mask[:, 1] = False
        result = forecast_metrics(p, y, mask, 3600)
        self.assertEqual(result['ramp_count'], 0)
        self.assertIsNone(result['ramp_mae'])
        self.assertIsNone(result['peak_amplitude_mae'])
        self.assertIsNone(result['per_horizon'][1]['mae'])
        self.assertEqual(result['micro_mae'], 1.)
        with self.assertRaisesRegex(ValueError, '有效目标'):
            forecast_metrics(p, y, np.zeros_like(mask), 3600)

    def test_ties_and_single_step(self):
        result = forecast_metrics(np.array([[[5.], [5.]]]), np.array([[[5.], [4.]]]), np.ones((1, 2, 1), bool), 3600)
        self.assertEqual(result['peak_time_mae_seconds'], 0.)
        result = forecast_metrics(np.ones((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1), bool), 3600)
        self.assertEqual(result['ramp_count'], 0)
        self.assertIsNone(result['ramp_mae'])

    def test_latent_quality_is_not_renormalized(self):
        errors = np.array([[[[2., 4.]]], [[[6., 8.]]]])
        quality = np.array([[[0.5]], [[0.]]])
        result = latent_metrics(errors, quality, ['g'], {'g__variance': np.array([.99, 1.99])})['groups']['g']
        self.assertEqual(result['valid_count'], 1)
        self.assertEqual(result['weighted_horizon_macro_mse'], 1.)
        self.assertEqual(result['raw_coordinate_mse'], 3.)
        self.assertEqual(result['quality_mean'], .25)
        empty = latent_metrics(errors, quality * 0, ['g'], {'g__variance': np.array([.99, 1.99])})['groups']['g']
        self.assertIsNone(empty['weighted_horizon_macro_mse'])

    def test_site_macro_not_sample_weighted(self):
        config = small_config()
        config['data']['site_split'] = {'train': ['a'], 'heldout_test': ['b']}
        rows = [{'site_id': site, 'scenario': 'dense', 'horizon': 1, 'target_valid': True,
                 'y_pred': error, 'y_true': 0.} for site, error in [('a', 1.), ('a', 1.), ('b', 3.)]]
        result = site_metrics(rows, config)
        self.assertEqual(result['site_macro_mae'], 2.)
        self.assertTrue(all(e['profile'] == 'unit_fixture' for e in result['entries']))
