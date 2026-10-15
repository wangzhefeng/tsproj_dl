#!/usr/bin/env python
"""真实 AIDC 单 batch 性能定位；不启动 epoch、不降低 L/H 或删除实体。"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from data_provider.spatial_temporal.config import load_config
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast
from exp.gslaformer_support import masked_load_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    root = Path(args.output)
    start = time.perf_counter()
    experiment = Exp_GSLAformer_Forecast(config, root)
    setup = time.perf_counter() - start
    order = experiment.site_sampler.draw(experiment.generator)
    started = time.perf_counter()
    batch = next(experiment._batches(experiment.train_data, order))
    batch_seconds = time.perf_counter() - started
    profiler = cProfile.Profile()
    profiler.enable()
    started = time.perf_counter()
    experiment.model.train()
    y = experiment.model.predict_direct(batch)
    loss, _ = masked_load_loss(y, batch.supervision.target, batch.supervision.target_mask)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - started
    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_([p for p in experiment.model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
    experiment.optimizer.step()
    optimizer = time.perf_counter() - started
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(25)
    (root / 'profile.txt').write_text(stream.getvalue())
    report = {'scope': 'single A training batch; not epoch benchmark', 'setup_seconds': setup,
              'data_seconds': batch_seconds, 'forward_seconds': forward, 'backward_seconds': backward,
              'optimizer_seconds': optimizer, 'loss': loss.item(), 'sites': batch.inputs.site_ids,
              'L': config['data']['seq_len'], 'H': config['data']['pred_len'],
              'nodes_by_group': {k: v['values'].shape[2] for k, v in batch.inputs.groups.items()}}
    (root / 'profile.json').write_text(json.dumps(report, indent=2))
    experiment._status('interrupted', 'profiling_only: one batch, no epoch checkpoint')
    print(json.dumps(report, indent=2))
    print(stream.getvalue())


if __name__ == '__main__':
    main()
