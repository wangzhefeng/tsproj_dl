#!/usr/bin/env python
"""完整配置的一个 A 阶段 epoch 实测；不把结果外推成 B-full 实测。"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from data_provider.spatial_temporal.config import load_config, safe_name
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_id', required=True)
    args = parser.parse_args()
    safe_name(args.run_id, 'run_id')
    config = load_config(args.config)
    root = Path(config['experiment']['results_root']) / args.run_id
    torch.set_num_threads(1)
    started = time.perf_counter()
    experiment = Exp_GSLAformer_Forecast(config, root)
    initialized = time.perf_counter()
    rows = experiment.train(max_epochs_this_call=1)
    report = {'scope': 'A stage: one complete train epoch + full horizon valid + checkpoint',
              'not_measured': ['B-full epoch', 'full comparison matrix'],
              'setup_seconds': initialized - started, 'epoch_seconds': time.perf_counter() - initialized,
              'maxrss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'maxrss_unit': 'bytes' if sys.platform == 'darwin' else 'KiB',
              'training_windows': len(experiment.train_data), 'validation_windows': len(experiment.valid_data),
              'phase': rows[-1]['phase'], 'updates': rows[-1]['updates']}
    (root / 'timing.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
