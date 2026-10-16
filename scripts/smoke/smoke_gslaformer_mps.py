#!/usr/bin/env python
"""独立 MPS 短流程：A/B 各阶段一 epoch；不改正式配置，不覆盖旧 run。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from data_provider.spatial_temporal.config import load_config, safe_name, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    safe_name(args.prefix, 'prefix')
    if not torch.backends.mps.is_available():
        raise ValueError('明确请求 MPS smoke，但设备不可用')
    config = load_config(ROOT / 'scripts/smoke/gslaformer_functional.json')
    config['experiment']['device'] = 'mps'
    config['data']['batch_size'] = 2
    config['training'].update(stage_a_max_epochs=1, b_short_epochs=1, b_medium_epochs=1, b_full_max_epochs=1)
    validate_config(config)
    results = Path(config['experiment']['results_root'])
    results.mkdir(parents=True, exist_ok=True)
    path = results / f'{args.prefix}_mps_config.json'
    with path.open('x') as stream:
        json.dump(config, stream, indent=2)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    def run(mode, suffix, *extra):
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'run.py'), '--model', 'GSLAformer',
                        '--gsla_config', str(path), '--gsla_mode', mode,
                        '--gsla_run_id', f'{args.prefix}_mps_{suffix}', *map(str, extra)], cwd=ROOT, env=env, check=True)
    run('train', 'train')
    checkpoint = results / f'{args.prefix}_mps_train/checkpoints/best.pt'
    run('test', 'test', '--gsla_checkpoint', checkpoint)
    for request in ('normal_forecast', 'new_entity_cold', 'new_entity_history', 'new_parent'):
        run('forecast', request, '--gsla_checkpoint', checkpoint,
            '--gsla_request', results / 'fixtures/requests' / request)
    run('forecast', 'reload', '--gsla_checkpoint', checkpoint,
        '--gsla_request', results / 'fixtures/requests/normal_forecast')


if __name__ == '__main__':
    main()
